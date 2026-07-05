//! OpenGL (glow) rendering for the composited image: up to `MAX_CHANNELS`
//! single-channel raw-sample textures, a LUT texture (one row per channel), and
//! a fragment shader that does per-channel window/level → LUT → additive blend,
//! with a minification box filter for clean zoom-out. Each channel is either an
//! integer texture (R16Uint — 8/16/32-bit-int sources, window/level in 0..65535
//! units) or a float texture (R32F — 32-bit float sources, window/level in the
//! data's own units, so the per-frame CPU rescale is avoided). Uploads happen
//! from `app.rs` via `UploadCtx` (the live GL context from `Frame::gl`); `paint`
//! is invoked from the egui_glow callback built by `paint_callback`.

use super::{ChannelKind, ChannelUniform, VolumeInterp, VolumeKind, VolumeParams, MAX_CHANNELS};
use eframe::egui_glow;
use eframe::glow::{self, HasContext as _};
use std::sync::{Arc, Mutex};

const LUT_WIDTH: i32 = 256;

/// The `eframe::Renderer` this backend needs requested in `NativeOptions`.
pub const RENDERER: eframe::Renderer = eframe::Renderer::Glow;

/// Short human-readable backend name, shown in the UI.
pub const BACKEND: &str = "glow";

/// Shared handle to the GL render resources. `Arc<Mutex>` because the egui_glow
/// paint callback (which draws) must be `Send + Sync + 'static`; uploads happen
/// in `app::sync_gpu`, so the lock is uncontended (both on the UI thread).
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// Build the render resources from eframe's creation context (its glow
/// context). Called once at startup.
pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
    let gl = cc
        .gl
        .as_ref()
        .expect("FastTIFF requires the glow backend (NativeOptions::renderer = Glow)");
    Arc::new(Mutex::new(ImageRenderResources::new(gl)))
}

/// Per-frame upload handle: the live GL context, pulled from `eframe::Frame`.
/// `None` before the backend is up (shouldn't happen after init).
pub struct UploadCtx<'a> {
    gl: &'a glow::Context,
}

pub fn upload_ctx(frame: &eframe::Frame) -> Option<UploadCtx<'_>> {
    frame.gl().map(|gl| UploadCtx { gl })
}

/// The egui paint callback that draws the current image into `rect`. Captures a
/// clone of the shared resources and locks them at paint time.
pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    let res = render.clone();
    let callback = egui_glow::CallbackFn::new(move |_info, painter| {
        if let Ok(r) = res.lock() {
            r.paint(painter.gl());
        }
    });
    egui::Shape::Callback(egui::PaintCallback {
        rect,
        callback: Arc::new(callback),
    })
}

/// The egui paint callback that ray-marches the 3D volume into `rect`.
pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    let res = render.clone();
    let callback = egui_glow::CallbackFn::new(move |_info, painter| {
        if let Ok(r) = res.lock() {
            r.paint_volume(painter.gl());
        }
    });
    egui::Shape::Callback(egui::PaintCallback {
        rect,
        callback: Arc::new(callback),
    })
}

/// egui_glow gives us a desktop GL 3.x context, for which it uses GLSL `#version
/// 140`. We match that so usampler2D / texelFetch / gl_VertexID are available.
const VERTEX_SRC: &str = r#"#version 140
out vec2 v_uv;
void main() {
    const vec2 POS[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0));
    const vec2 UV[6] = vec2[6](
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
    gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
    v_uv = UV[gl_VertexID];
}
"#;

// Each channel has both an integer sampler (`chN_tex`, R16Uint) and a float
// sampler (`chN_ftex`, R32F); `ch_is_float[N]` selects which one carries this
// channel's data (the other is a 1x1 dummy). The window/level math is identical
// for both — only the sampled value's source and units differ.
const FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_uv;
out vec4 frag_color;

uniform usampler2D ch0_tex;
uniform usampler2D ch1_tex;
uniform usampler2D ch2_tex;
uniform usampler2D ch3_tex;
uniform usampler2D ch4_tex;
uniform usampler2D ch5_tex;
uniform sampler2D ch0_ftex;
uniform sampler2D ch1_ftex;
uniform sampler2D ch2_ftex;
uniform sampler2D ch3_ftex;
uniform sampler2D ch4_ftex;
uniform sampler2D ch5_ftex;
uniform sampler2D lut_tex;
uniform vec2 ch_min_max[6];
uniform float ch_enabled[6];
uniform float ch_is_float[6];
uniform int num_channels;
uniform vec2 uv_offset;
uniform vec2 uv_scale;

float load_texel(usampler2D tex, vec2 uv, vec2 dims) {
    ivec2 c = clamp(ivec2(uv * dims), ivec2(0), ivec2(dims) - ivec2(1));
    return float(texelFetch(tex, c, 0).r);
}
float load_texel_f(sampler2D tex, vec2 uv, vec2 dims) {
    ivec2 c = clamp(ivec2(uv * dims), ivec2(0), ivec2(dims) - ivec2(1));
    return texelFetch(tex, c, 0).r;
}

// Single crisp nearest read at 1:1 / zoom-in; an NxN box filter when minifying
// (zoom-out), to kill nearest-neighbor shimmer. N is capped so cost is bounded.
// Two near-identical copies, one per sampler type (GLSL can't pass a sampler of
// runtime-chosen type).
float sample_int(usampler2D tex, vec2 uv, vec2 footprint) {
    vec2 dims = vec2(textureSize(tex, 0));
    vec2 texels = footprint * dims;
    int n = int(clamp(ceil(max(texels.x, texels.y)), 1.0, 4.0));
    if (n <= 1) {
        return load_texel(tex, uv, dims);
    }
    float fn = float(n);
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec2 o = (vec2(float(i), float(j)) + 0.5) / fn - 0.5;
            sum += load_texel(tex, uv + o * footprint, dims);
        }
    }
    return sum / (fn * fn);
}
float sample_flt(sampler2D tex, vec2 uv, vec2 footprint) {
    vec2 dims = vec2(textureSize(tex, 0));
    vec2 texels = footprint * dims;
    int n = int(clamp(ceil(max(texels.x, texels.y)), 1.0, 4.0));
    if (n <= 1) {
        return load_texel_f(tex, uv, dims);
    }
    float fn = float(n);
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec2 o = (vec2(float(i), float(j)) + 0.5) / fn - 0.5;
            sum += load_texel_f(tex, uv + o * footprint, dims);
        }
    }
    return sum / (fn * fn);
}

vec3 apply_channel(float value, int idx, vec2 mm, float en) {
    float span = max(mm.y - mm.x, 1.0);
    float t = clamp((value - mm.x) / span, 0.0, 1.0);
    vec3 color = texture(lut_tex, vec2(t, (float(idx) + 0.5) / 6.0)).rgb;
    return color * en;
}

void main() {
    vec2 uv = uv_offset + v_uv * uv_scale;
    vec2 fp = fwidth(uv);
    vec3 color = vec3(0.0);
    color += apply_channel(ch_is_float[0] > 0.5 ? sample_flt(ch0_ftex, uv, fp) : sample_int(ch0_tex, uv, fp), 0, ch_min_max[0], ch_enabled[0]);
    if (num_channels > 1) color += apply_channel(ch_is_float[1] > 0.5 ? sample_flt(ch1_ftex, uv, fp) : sample_int(ch1_tex, uv, fp), 1, ch_min_max[1], ch_enabled[1]);
    if (num_channels > 2) color += apply_channel(ch_is_float[2] > 0.5 ? sample_flt(ch2_ftex, uv, fp) : sample_int(ch2_tex, uv, fp), 2, ch_min_max[2], ch_enabled[2]);
    if (num_channels > 3) color += apply_channel(ch_is_float[3] > 0.5 ? sample_flt(ch3_ftex, uv, fp) : sample_int(ch3_tex, uv, fp), 3, ch_min_max[3], ch_enabled[3]);
    if (num_channels > 4) color += apply_channel(ch_is_float[4] > 0.5 ? sample_flt(ch4_ftex, uv, fp) : sample_int(ch4_tex, uv, fp), 4, ch_min_max[4], ch_enabled[4]);
    if (num_channels > 5) color += apply_channel(ch_is_float[5] > 0.5 ? sample_flt(ch5_ftex, uv, fp) : sample_int(ch5_tex, uv, fp), 5, ch_min_max[5], ch_enabled[5]);
    frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
"#;

// --- 3D volume ray-march (MIP) ---------------------------------------------
// A fullscreen pass that, per pixel, builds a camera ray and marches it
// through a 3D texture, tracking the maximum sample (maximum-intensity
// projection — order-independent, so no blending/sorting and no early-out
// needed). The window/level + LUT are applied once at the end, so scrubbing
// the contrast never rebuilds the volume. The camera arrives as an explicit
// basis (eye/forward/right/up + fov), avoiding any matrix inverse in-shader.

const VOL_VERTEX_SRC: &str = r#"#version 140
out vec2 v_ndc;
void main() {
    const vec2 P[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0));
    gl_Position = vec4(P[gl_VertexID], 0.0, 1.0);
    v_ndc = P[gl_VertexID];
}
"#;

const VOL_FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_ndc;
out vec4 frag_color;

uniform sampler3D vol_tex;   // R8/R16 unorm (integer sources)
uniform sampler3D vol_ftex;  // R32F (float sources)
uniform sampler2D lut_tex;
uniform vec3 cam_eye;
uniform vec3 cam_forward;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform float tan_half_fov;
uniform float aspect;
uniform vec3 box_he;       // volume box half-extents (scaled dims)
uniform vec2 window;       // min, max in sampled-texture units
uniform int u_steps;
uniform int u_lut_row;
uniform float u_is_float;

float sample_vol(vec3 tc) {
    return (u_is_float > 0.5) ? texture(vol_ftex, tc).r : texture(vol_tex, tc).r;
}

void main() {
    vec3 rd = normalize(cam_forward
        + v_ndc.x * aspect * tan_half_fov * cam_right
        + v_ndc.y * tan_half_fov * cam_up);
    vec3 ro = cam_eye;

    // Slab intersection with the axis-aligned box [-box_he, box_he].
    vec3 inv = 1.0 / rd;
    vec3 ta = (-box_he - ro) * inv;
    vec3 tb = (box_he - ro) * inv;
    vec3 tmin = min(ta, tb);
    vec3 tmax = max(ta, tb);
    float t0 = max(max(tmin.x, tmin.y), tmin.z);
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    if (t1 < max(t0, 0.0)) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    t0 = max(t0, 0.0);

    int steps = clamp(u_steps, 1, 512);
    float dt = (t1 - t0) / float(steps);
    vec3 p = ro + rd * (t0 + dt * 0.5);
    vec3 dp = rd * dt;
    vec3 to_uvw = 0.5 / box_he;   // world point -> [0,1] texcoord
    float maxv = 0.0;
    for (int i = 0; i < 512; i++) {
        if (i >= steps) break;
        maxv = max(maxv, sample_vol((p + box_he) * to_uvw));
        p += dp;
    }

    float t = clamp((maxv - window.x) / max(window.y - window.x, 1e-6), 0.0, 1.0);
    vec3 col = texture(lut_tex, vec2(t, (float(u_lut_row) + 0.5) / 6.0)).rgb;
    frag_color = vec4(col, 1.0);
}
"#;

/// Per-channel texture-allocation kind, tracked so `ensure_size` only
/// reallocates when something actually changes.
const KIND_UNUSED: u8 = 0; // channel not present: both textures are 1x1 dummies
const KIND_INT8: u8 = 1; // integer channel: R8Uint full-size, R32F dummy
const KIND_INT16: u8 = 2; // integer channel: R16Uint full-size, R32F dummy
const KIND_FLOAT: u8 = 3; // float channel: R32F full-size, int texture dummy

pub struct ImageRenderResources {
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    channel_textures: [glow::NativeTexture; MAX_CHANNELS], // R16Uint (integer channels)
    channel_ftextures: [glow::NativeTexture; MAX_CHANNELS], // R32F (float channels)
    lut_texture: glow::NativeTexture,
    current_size: (u32, u32),
    current_kinds: [u8; MAX_CHANNELS],

    u_ch_tex: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
    u_ch_ftex: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
    u_lut: Option<glow::NativeUniformLocation>,
    u_min_max: Option<glow::NativeUniformLocation>,
    u_enabled: Option<glow::NativeUniformLocation>,
    u_is_float: Option<glow::NativeUniformLocation>,
    u_num_channels: Option<glow::NativeUniformLocation>,
    u_uv_offset: Option<glow::NativeUniformLocation>,
    u_uv_scale: Option<glow::NativeUniformLocation>,

    // Current draw params, set from app.rs and applied in `paint`.
    min_max: [f32; MAX_CHANNELS * 2],
    enabled: [f32; MAX_CHANNELS],
    is_float: [f32; MAX_CHANNELS],
    num_channels: i32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],

    /// The 3D ray-march pipeline + volume texture (see `paint_volume`).
    volume: VolumeGl,
}

/// The 3D volume pipeline: its own program, an R8/R16 unorm texture and an
/// R32F texture (one carries the data, the other is a 1x1 dummy — mirroring the
/// 2D int/float split), and the marshalled draw params.
struct VolumeGl {
    program: glow::NativeProgram,
    tex_unorm: glow::NativeTexture,
    tex_float: glow::NativeTexture,
    dims: (u32, u32, u32),
    kind: Option<VolumeKind>,
    interp: i32, // GL_NEAREST / GL_LINEAR
    u_vol: Option<glow::NativeUniformLocation>,
    u_volf: Option<glow::NativeUniformLocation>,
    u_lut: Option<glow::NativeUniformLocation>,
    u_eye: Option<glow::NativeUniformLocation>,
    u_forward: Option<glow::NativeUniformLocation>,
    u_right: Option<glow::NativeUniformLocation>,
    u_up: Option<glow::NativeUniformLocation>,
    u_tan: Option<glow::NativeUniformLocation>,
    u_aspect: Option<glow::NativeUniformLocation>,
    u_box_he: Option<glow::NativeUniformLocation>,
    u_window: Option<glow::NativeUniformLocation>,
    u_steps: Option<glow::NativeUniformLocation>,
    u_lut_row: Option<glow::NativeUniformLocation>,
    u_is_float: Option<glow::NativeUniformLocation>,
    params: Option<VolumeParams>,
}

impl VolumeGl {
    unsafe fn new(gl: &glow::Context) -> Self {
        let program = link_program(gl, VOL_VERTEX_SRC, VOL_FRAGMENT_SRC);
        VolumeGl {
            program,
            tex_unorm: create_volume_texture(gl, false),
            tex_float: create_volume_texture(gl, true),
            dims: (1, 1, 1),
            kind: None,
            interp: glow::NEAREST as i32,
            u_vol: gl.get_uniform_location(program, "vol_tex"),
            u_volf: gl.get_uniform_location(program, "vol_ftex"),
            u_lut: gl.get_uniform_location(program, "lut_tex"),
            u_eye: gl.get_uniform_location(program, "cam_eye"),
            u_forward: gl.get_uniform_location(program, "cam_forward"),
            u_right: gl.get_uniform_location(program, "cam_right"),
            u_up: gl.get_uniform_location(program, "cam_up"),
            u_tan: gl.get_uniform_location(program, "tan_half_fov"),
            u_aspect: gl.get_uniform_location(program, "aspect"),
            u_box_he: gl.get_uniform_location(program, "box_he"),
            u_window: gl.get_uniform_location(program, "window"),
            u_steps: gl.get_uniform_location(program, "u_steps"),
            u_lut_row: gl.get_uniform_location(program, "u_lut_row"),
            u_is_float: gl.get_uniform_location(program, "u_is_float"),
            params: None,
        }
    }
}

impl ImageRenderResources {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link_program(gl, VERTEX_SRC, FRAGMENT_SRC);
            let vao = gl.create_vertex_array().expect("create VAO");
            let channel_textures = std::array::from_fn(|_| create_int_texture(gl, 1, 1));
            let channel_ftextures = std::array::from_fn(|_| create_float_texture(gl, 1, 1));
            let lut_texture = create_lut_texture(gl);

            let u_ch_tex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_tex")));
            let u_ch_ftex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_ftex")));

            Self {
                program,
                vao,
                channel_textures,
                channel_ftextures,
                lut_texture,
                current_size: (1, 1),
                current_kinds: [KIND_UNUSED; MAX_CHANNELS],
                u_ch_tex,
                u_ch_ftex,
                u_lut: gl.get_uniform_location(program, "lut_tex"),
                u_min_max: gl.get_uniform_location(program, "ch_min_max"),
                u_enabled: gl.get_uniform_location(program, "ch_enabled"),
                u_is_float: gl.get_uniform_location(program, "ch_is_float"),
                u_num_channels: gl.get_uniform_location(program, "num_channels"),
                u_uv_offset: gl.get_uniform_location(program, "uv_offset"),
                u_uv_scale: gl.get_uniform_location(program, "uv_scale"),
                min_max: [0.0; MAX_CHANNELS * 2],
                enabled: [0.0; MAX_CHANNELS],
                is_float: [0.0; MAX_CHANNELS],
                num_channels: 0,
                uv_offset: [0.0, 0.0],
                uv_scale: [1.0, 1.0],
                volume: VolumeGl::new(gl),
            }
        }
    }

    /// Largest per-axis 3D-texture dimension the driver supports; the app uses
    /// it (with a memory cap) to decide whether the volume must be subsampled.
    pub fn max_3d_texture_size(&self, ctx: &UploadCtx) -> u32 {
        unsafe { ctx.gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE).max(0) as u32 }
    }

    /// (Re)upload the whole volume: `bytes` is `w*h*d` samples, z-major, in the
    /// units of `kind` (U8/U16 unorm, F32). Sized once per stack — window/level,
    /// LUT, camera and interpolation all change without touching this.
    pub fn upload_volume(&mut self, ctx: &UploadCtx, w: u32, h: u32, d: u32, kind: VolumeKind, bytes: &[u8]) {
        let gl = ctx.gl;
        let v = &mut self.volume;
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            let (data_tex, dummy_tex) = if kind == VolumeKind::F32 {
                (v.tex_float, v.tex_unorm)
            } else {
                (v.tex_unorm, v.tex_float)
            };
            gl.bind_texture(glow::TEXTURE_3D, Some(data_tex));
            set_volume_filter(gl, v.interp, kind);
            let (internal, fmt, ty) = match kind {
                VolumeKind::U8 => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
                VolumeKind::U16 => (glow::R16, glow::RED, glow::UNSIGNED_SHORT),
                VolumeKind::F32 => (glow::R32F, glow::RED, glow::FLOAT),
            };
            gl.tex_image_3d(
                glow::TEXTURE_3D,
                0,
                internal as i32,
                w as i32,
                h as i32,
                d as i32,
                0,
                fmt,
                ty,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
            // Shrink the unused texture back to a 1x1x1 dummy to free its VRAM.
            gl.bind_texture(glow::TEXTURE_3D, Some(dummy_tex));
            let dummy_kind = if kind == VolumeKind::F32 { VolumeKind::U8 } else { VolumeKind::F32 };
            alloc_volume_dummy(gl, dummy_kind);
        }
        v.dims = (w, h, d);
        v.kind = Some(kind);
    }

    /// Switch the volume texture's sampling filter (crisp voxels vs trilinear)
    /// without re-uploading.
    pub fn set_volume_interp(&mut self, ctx: &UploadCtx, interp: VolumeInterp) {
        let gl_interp = match interp {
            VolumeInterp::Nearest => glow::NEAREST as i32,
            VolumeInterp::Linear => glow::LINEAR as i32,
        };
        self.volume.interp = gl_interp;
        if let Some(kind) = self.volume.kind {
            let tex = if kind == VolumeKind::F32 { self.volume.tex_float } else { self.volume.tex_unorm };
            unsafe {
                ctx.gl.bind_texture(glow::TEXTURE_3D, Some(tex));
                set_volume_filter(ctx.gl, gl_interp, kind);
            }
        }
    }

    /// Stash the ray-march params (camera, window, steps); applied in `paint_volume`.
    pub fn set_volume_params(&mut self, params: VolumeParams) {
        self.volume.params = Some(params);
    }

    /// Ray-march the current volume. The egui_glow callback has already set the
    /// viewport/scissor to the canvas rect.
    pub fn paint_volume(&self, gl: &glow::Context) {
        let v = &self.volume;
        let Some(p) = v.params else { return };
        unsafe {
            gl.use_program(Some(v.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_unorm));
            gl.uniform_1_i32(v.u_vol.as_ref(), 0);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_float));
            gl.uniform_1_i32(v.u_volf.as_ref(), 1);
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(v.u_lut.as_ref(), 2);

            gl.uniform_3_f32(v.u_eye.as_ref(), p.eye[0], p.eye[1], p.eye[2]);
            gl.uniform_3_f32(v.u_forward.as_ref(), p.forward[0], p.forward[1], p.forward[2]);
            gl.uniform_3_f32(v.u_right.as_ref(), p.right[0], p.right[1], p.right[2]);
            gl.uniform_3_f32(v.u_up.as_ref(), p.up[0], p.up[1], p.up[2]);
            gl.uniform_1_f32(v.u_tan.as_ref(), p.tan_half_fov);
            gl.uniform_1_f32(v.u_aspect.as_ref(), p.aspect);
            gl.uniform_3_f32(v.u_box_he.as_ref(), p.box_he[0], p.box_he[1], p.box_he[2]);
            gl.uniform_2_f32(v.u_window.as_ref(), p.window[0], p.window[1]);
            gl.uniform_1_i32(v.u_steps.as_ref(), p.steps);
            gl.uniform_1_i32(v.u_lut_row.as_ref(), p.lut_row as i32);
            gl.uniform_1_f32(v.u_is_float.as_ref(), if p.is_float { 1.0 } else { 0.0 });

            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }

    /// (Re)allocate channel textures for the current frame size and per-channel
    /// `kind`. An `Int8`/`Int16` channel gets a full-size R8Uint/R16Uint integer
    /// texture (float texture a 1x1 dummy); a `Float` channel gets a full-size
    /// R32F float texture (integer texture a 1x1 dummy); channels past
    /// `kinds.len()` are unused (both 1x1). No-op when nothing changed.
    pub fn ensure_size(&mut self, ctx: &UploadCtx, width: u32, height: u32, kinds: &[ChannelKind]) {
        let mut want = [KIND_UNUSED; MAX_CHANNELS];
        for (c, slot) in want.iter_mut().enumerate() {
            *slot = match kinds.get(c) {
                None => KIND_UNUSED,
                Some(ChannelKind::Int8) => KIND_INT8,
                Some(ChannelKind::Int16) => KIND_INT16,
                Some(ChannelKind::Float) => KIND_FLOAT,
            };
        }
        if self.current_size == (width, height) && self.current_kinds == want {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            for c in 0..MAX_CHANNELS {
                // Integer texture: R8Uint or R16Uint at full size for an integer
                // channel, else a 1x1 R16Uint dummy.
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[c]));
                match want[c] {
                    KIND_INT8 => alloc_int(gl, width, height, true),
                    KIND_INT16 => alloc_int(gl, width, height, false),
                    _ => alloc_int(gl, 1, 1, false),
                }
                // Float texture: full size for a float channel, else 1x1 dummy.
                let (fw, fh) = if want[c] == KIND_FLOAT { (width, height) } else { (1, 1) };
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_ftextures[c]));
                alloc_float(gl, fw, fh);
            }
        }
        self.current_size = (width, height);
        self.current_kinds = want;
    }

    /// Upload one integer channel's raw 16-bit samples (R16Uint texture).
    pub fn upload_channel(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[channel]));
            // Our rows are tightly packed (`width * 2` bytes for R16UI). Force
            // the unpack alignment to 1 so an odd-width image — whose row length
            // isn't a multiple of GL's default alignment of 4 — isn't read with
            // phantom end-of-row padding, which shears the image diagonally. We
            // set it on every upload because the GL context is shared with
            // egui_glow, which changes this global state for its own textures.
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::RED_INTEGER,
                glow::UNSIGNED_SHORT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(samples))),
            );
        }
    }

    /// Upload one integer channel's raw 8-bit samples (R8Uint texture). Skips
    /// the CPU `0..255 -> 0..65535` widening; the window/level is scaled to
    /// 0..255 units on the app side instead.
    pub fn upload_channel_u8(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[u8]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[channel]));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::RED_INTEGER,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(samples)),
            );
        }
    }

    /// Upload one float channel's raw 32-bit float samples (R32F texture).
    pub fn upload_channel_f32(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[f32]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_ftextures[channel]));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::RED,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(samples))),
            );
        }
    }

    /// Upload one channel's LUT (256 RGB entries) into row `channel`.
    pub fn upload_lut(&self, ctx: &UploadCtx, channel: usize, lut: &[[u8; 3]; 256]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        let mut rgba = [0u8; (LUT_WIDTH * 4) as usize];
        for (i, px) in lut.iter().enumerate() {
            rgba[i * 4] = px[0];
            rgba[i * 4 + 1] = px[1];
            rgba[i * 4 + 2] = px[2];
            rgba[i * 4 + 3] = 255;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            // Match `upload_channel`: keep alignment at 1 so the shared context's
            // unpack state can't misread our tightly-packed rows. (The LUT's
            // 256*4-byte rows are already 4-aligned, but stay explicit.)
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                channel as i32,
                LUT_WIDTH,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&rgba)),
            );
        }
    }

    /// Stash the per-channel window/level + enabled + is-float flags, channel
    /// count, and the visible UV sub-rect (pan/zoom); applied as uniforms in
    /// `paint`. The glow backend uploads no per-frame uniform buffer, so `ctx`
    /// is unused — the signature matches the wgpu backend, which does.
    pub fn set_params(
        &mut self,
        _ctx: &UploadCtx,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) {
        self.min_max = [0.0; MAX_CHANNELS * 2];
        self.enabled = [0.0; MAX_CHANNELS];
        self.is_float = [0.0; MAX_CHANNELS];
        for (i, c) in channels.iter().take(MAX_CHANNELS).enumerate() {
            self.min_max[i * 2] = c.min;
            self.min_max[i * 2 + 1] = c.max;
            self.enabled[i] = if c.enabled { 1.0 } else { 0.0 };
            self.is_float[i] = if c.is_float { 1.0 } else { 0.0 };
        }
        self.num_channels = num_channels as i32;
        self.uv_offset = uv_offset;
        self.uv_scale = uv_scale;
    }

    /// Draw the composited image. The egui_glow callback sets the viewport and
    /// scissor to the image's on-screen rect for us.
    pub fn paint(&self, gl: &glow::Context) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            // Integer channel textures → units 0..MAX_CHANNELS.
            for c in 0..MAX_CHANNELS {
                gl.active_texture(glow::TEXTURE0 + c as u32);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[c]));
                gl.uniform_1_i32(self.u_ch_tex[c].as_ref(), c as i32);
            }
            // LUT → unit MAX_CHANNELS.
            let lut_unit = MAX_CHANNELS as u32;
            gl.active_texture(glow::TEXTURE0 + lut_unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(self.u_lut.as_ref(), lut_unit as i32);
            // Float channel textures → units MAX_CHANNELS+1 .. 2*MAX_CHANNELS+1.
            for c in 0..MAX_CHANNELS {
                let unit = MAX_CHANNELS as u32 + 1 + c as u32;
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_ftextures[c]));
                gl.uniform_1_i32(self.u_ch_ftex[c].as_ref(), unit as i32);
            }

            gl.uniform_2_f32_slice(self.u_min_max.as_ref(), &self.min_max);
            gl.uniform_1_f32_slice(self.u_enabled.as_ref(), &self.enabled);
            gl.uniform_1_f32_slice(self.u_is_float.as_ref(), &self.is_float);
            gl.uniform_1_i32(self.u_num_channels.as_ref(), self.num_channels);
            gl.uniform_2_f32(self.u_uv_offset.as_ref(), self.uv_offset[0], self.uv_offset[1]);
            gl.uniform_2_f32(self.u_uv_scale.as_ref(), self.uv_scale[0], self.uv_scale[1]);

            // The image is opaque (alpha 1); draw straight, no blending.
            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }
}

/// Allocate storage for the currently-bound integer texture (no data) as either
/// R8Uint (`eight_bit`) or R16Uint. Both are read through the same `usampler2D`
/// in the shader; `texelFetch` returns 0..255 or 0..65535 accordingly.
unsafe fn alloc_int(gl: &glow::Context, width: u32, height: u32, eight_bit: bool) {
    let (internal, ty) = if eight_bit {
        (glow::R8UI, glow::UNSIGNED_BYTE)
    } else {
        (glow::R16UI, glow::UNSIGNED_SHORT)
    };
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        internal as i32,
        width as i32,
        height as i32,
        0,
        glow::RED_INTEGER,
        ty,
        glow::PixelUnpackData::Slice(None),
    );
}

/// Allocate storage for the currently-bound texture as R32F (no data).
unsafe fn alloc_float(gl: &glow::Context, width: u32, height: u32) {
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        glow::R32F as i32,
        width as i32,
        height as i32,
        0,
        glow::RED,
        glow::FLOAT,
        glow::PixelUnpackData::Slice(None),
    );
}

unsafe fn create_int_texture(gl: &glow::Context, width: u32, height: u32) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create channel texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    // Integer textures must use NEAREST filtering (we read via texelFetch).
    set_nearest_clamp(gl);
    alloc_int(gl, width, height, false);
    tex
}

unsafe fn create_float_texture(gl: &glow::Context, width: u32, height: u32) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create float channel texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    // NEAREST is required for R32F sampling on older GL (and we read via
    // texelFetch anyway, which ignores the filter).
    set_nearest_clamp(gl);
    alloc_float(gl, width, height);
    tex
}

unsafe fn set_nearest_clamp(gl: &glow::Context) {
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
}

/// Allocate a 1x1x1 dummy for the currently-bound 3D texture in `kind`'s format
/// (the volume's unused int/float sibling always shrinks to this).
unsafe fn alloc_volume_dummy(gl: &glow::Context, kind: VolumeKind) {
    let (internal, fmt, ty) = match kind {
        VolumeKind::U8 => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
        VolumeKind::U16 => (glow::R16, glow::RED, glow::UNSIGNED_SHORT),
        VolumeKind::F32 => (glow::R32F, glow::RED, glow::FLOAT),
    };
    gl.tex_image_3d(glow::TEXTURE_3D, 0, internal as i32, 1, 1, 1, 0, fmt, ty, glow::PixelUnpackData::Slice(None));
}

/// Create a 3D texture (clamped, NEAREST) with a 1x1x1 dummy allocation.
/// `is_float` picks R32F, else R8 (re-specified to R8/R16 at upload time).
unsafe fn create_volume_texture(gl: &glow::Context, is_float: bool) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create volume texture");
    gl.bind_texture(glow::TEXTURE_3D, Some(tex));
    let kind = if is_float { VolumeKind::F32 } else { VolumeKind::U8 };
    set_volume_filter(gl, glow::NEAREST as i32, kind);
    alloc_volume_dummy(gl, kind);
    tex
}

/// Set the currently-bound 3D texture's filter (`gl_interp` = NEAREST/LINEAR)
/// and clamp all three axes. `kind` is unused today but kept so a future
/// nearest-only fallback for unfilterable float formats stays a one-line change.
unsafe fn set_volume_filter(gl: &glow::Context, gl_interp: i32, _kind: VolumeKind) {
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MIN_FILTER, gl_interp);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MAG_FILTER, gl_interp);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_R, glow::CLAMP_TO_EDGE as i32);
}

unsafe fn create_lut_texture(gl: &glow::Context) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create LUT texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        glow::RGBA8 as i32,
        LUT_WIDTH,
        MAX_CHANNELS as i32,
        0,
        glow::RGBA,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(None),
    );
    tex
}

unsafe fn link_program(gl: &glow::Context, vs_src: &str, fs_src: &str) -> glow::NativeProgram {
    let vs = compile_shader(gl, glow::VERTEX_SHADER, vs_src);
    let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fs_src);
    let program = gl.create_program().expect("create program");
    gl.attach_shader(program, vs);
    gl.attach_shader(program, fs);
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        panic!("FastTIFF shader link failed: {}", gl.get_program_info_log(program));
    }
    gl.detach_shader(program, vs);
    gl.detach_shader(program, fs);
    gl.delete_shader(vs);
    gl.delete_shader(fs);
    program
}

unsafe fn compile_shader(gl: &glow::Context, kind: u32, src: &str) -> glow::NativeShader {
    let shader = gl.create_shader(kind).expect("create shader");
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        let which = if kind == glow::VERTEX_SHADER { "vertex" } else { "fragment" };
        panic!("FastTIFF {which} shader compile failed: {}", gl.get_shader_info_log(shader));
    }
    shader
}

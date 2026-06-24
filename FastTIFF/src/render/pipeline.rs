//! OpenGL (glow) rendering for the composited image. Mirrors the previous wgpu
//! pipeline: up to `MAX_CHANNELS` single-channel raw-sample textures (R16UI),
//! a LUT texture (one row per channel), and a fragment shader that does
//! per-channel window/level → LUT → additive blend, with a minification box
//! filter for clean zoom-out. Uploads happen from `app.rs` (it holds the glow
//! context via `Frame::gl`); `paint` is invoked from the egui_glow callback.

use eframe::glow::{self, HasContext as _};

pub const MAX_CHANNELS: usize = 6;
const LUT_WIDTH: i32 = 256;

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

const FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_uv;
out vec4 frag_color;

uniform usampler2D ch0_tex;
uniform usampler2D ch1_tex;
uniform usampler2D ch2_tex;
uniform usampler2D ch3_tex;
uniform usampler2D ch4_tex;
uniform usampler2D ch5_tex;
uniform sampler2D lut_tex;
uniform vec2 ch_min_max[6];
uniform float ch_enabled[6];
uniform int num_channels;
uniform vec2 uv_offset;
uniform vec2 uv_scale;

float load_texel(usampler2D tex, vec2 uv, vec2 dims) {
    ivec2 c = clamp(ivec2(uv * dims), ivec2(0), ivec2(dims) - ivec2(1));
    return float(texelFetch(tex, c, 0).r);
}

// Single crisp nearest read at 1:1 / zoom-in; an NxN box filter when minifying
// (zoom-out), to kill nearest-neighbor shimmer. N is capped so cost is bounded.
float sample_channel(usampler2D tex, vec2 uv, vec2 footprint) {
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

vec3 apply_channel(float value, int idx, vec2 mm, float en) {
    float span = max(mm.y - mm.x, 1.0);
    float t = clamp((value - mm.x) / span, 0.0, 1.0);
    vec3 color = texture(lut_tex, vec2(t, (float(idx) + 0.5) / 6.0)).rgb;
    return color * en;
}

void main() {
    vec2 uv = uv_offset + v_uv * uv_scale;
    vec2 footprint = fwidth(uv);
    vec3 color = vec3(0.0);
    color += apply_channel(sample_channel(ch0_tex, uv, footprint), 0, ch_min_max[0], ch_enabled[0]);
    if (num_channels > 1) color += apply_channel(sample_channel(ch1_tex, uv, footprint), 1, ch_min_max[1], ch_enabled[1]);
    if (num_channels > 2) color += apply_channel(sample_channel(ch2_tex, uv, footprint), 2, ch_min_max[2], ch_enabled[2]);
    if (num_channels > 3) color += apply_channel(sample_channel(ch3_tex, uv, footprint), 3, ch_min_max[3], ch_enabled[3]);
    if (num_channels > 4) color += apply_channel(sample_channel(ch4_tex, uv, footprint), 4, ch_min_max[4], ch_enabled[4]);
    if (num_channels > 5) color += apply_channel(sample_channel(ch5_tex, uv, footprint), 5, ch_min_max[5], ch_enabled[5]);
    frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}
"#;

#[derive(Clone, Copy)]
pub struct ChannelUniform {
    pub min: f32,
    pub max: f32,
    pub enabled: bool,
}

pub struct ImageRenderResources {
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    channel_textures: [glow::NativeTexture; MAX_CHANNELS],
    lut_texture: glow::NativeTexture,
    current_size: (u32, u32),

    u_ch_tex: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
    u_lut: Option<glow::NativeUniformLocation>,
    u_min_max: Option<glow::NativeUniformLocation>,
    u_enabled: Option<glow::NativeUniformLocation>,
    u_num_channels: Option<glow::NativeUniformLocation>,
    u_uv_offset: Option<glow::NativeUniformLocation>,
    u_uv_scale: Option<glow::NativeUniformLocation>,

    // Current draw params, set from app.rs and applied in `paint`.
    min_max: [f32; MAX_CHANNELS * 2],
    enabled: [f32; MAX_CHANNELS],
    num_channels: i32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
}

impl ImageRenderResources {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link_program(gl, VERTEX_SRC, FRAGMENT_SRC);
            let vao = gl.create_vertex_array().expect("create VAO");
            let channel_textures = std::array::from_fn(|_| create_channel_texture(gl, 1, 1));
            let lut_texture = create_lut_texture(gl);

            let u_ch_tex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_tex")));

            Self {
                program,
                vao,
                channel_textures,
                lut_texture,
                current_size: (1, 1),
                u_ch_tex,
                u_lut: gl.get_uniform_location(program, "lut_tex"),
                u_min_max: gl.get_uniform_location(program, "ch_min_max"),
                u_enabled: gl.get_uniform_location(program, "ch_enabled"),
                u_num_channels: gl.get_uniform_location(program, "num_channels"),
                u_uv_offset: gl.get_uniform_location(program, "uv_offset"),
                u_uv_scale: gl.get_uniform_location(program, "uv_scale"),
                min_max: [0.0; MAX_CHANNELS * 2],
                enabled: [0.0; MAX_CHANNELS],
                num_channels: 0,
                uv_offset: [0.0, 0.0],
                uv_scale: [1.0, 1.0],
            }
        }
    }

    /// Reallocate the channel textures when the frame size changes.
    pub fn ensure_size(&mut self, gl: &glow::Context, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
        unsafe {
            for &tex in &self.channel_textures {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::R16UI as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RED_INTEGER,
                    glow::UNSIGNED_SHORT,
                    glow::PixelUnpackData::Slice(None),
                );
            }
        }
        self.current_size = (width, height);
    }

    /// Upload one channel's raw 16-bit samples.
    pub fn upload_channel(&self, gl: &glow::Context, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[channel]));
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

    /// Upload one channel's LUT (256 RGB entries) into row `channel`.
    pub fn upload_lut(&self, gl: &glow::Context, channel: usize, lut: &[[u8; 3]; 256]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let mut rgba = [0u8; (LUT_WIDTH * 4) as usize];
        for (i, px) in lut.iter().enumerate() {
            rgba[i * 4] = px[0];
            rgba[i * 4 + 1] = px[1];
            rgba[i * 4 + 2] = px[2];
            rgba[i * 4 + 3] = 255;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
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

    /// Stash the per-channel window/level + enabled flags, channel count, and
    /// the visible UV sub-rect (pan/zoom); applied as uniforms in `paint`.
    pub fn set_params(
        &mut self,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) {
        self.min_max = [0.0; MAX_CHANNELS * 2];
        self.enabled = [0.0; MAX_CHANNELS];
        for (i, c) in channels.iter().take(MAX_CHANNELS).enumerate() {
            self.min_max[i * 2] = c.min;
            self.min_max[i * 2 + 1] = c.max;
            self.enabled[i] = if c.enabled { 1.0 } else { 0.0 };
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
            for c in 0..MAX_CHANNELS {
                gl.active_texture(glow::TEXTURE0 + c as u32);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[c]));
                gl.uniform_1_i32(self.u_ch_tex[c].as_ref(), c as i32);
            }
            gl.active_texture(glow::TEXTURE0 + MAX_CHANNELS as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(self.u_lut.as_ref(), MAX_CHANNELS as i32);

            gl.uniform_2_f32_slice(self.u_min_max.as_ref(), &self.min_max);
            gl.uniform_1_f32_slice(self.u_enabled.as_ref(), &self.enabled);
            gl.uniform_1_i32(self.u_num_channels.as_ref(), self.num_channels);
            gl.uniform_2_f32(self.u_uv_offset.as_ref(), self.uv_offset[0], self.uv_offset[1]);
            gl.uniform_2_f32(self.u_uv_scale.as_ref(), self.uv_scale[0], self.uv_scale[1]);

            // The image is opaque (alpha 1); draw straight, no blending.
            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }
}

unsafe fn create_channel_texture(gl: &glow::Context, width: u32, height: u32) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create channel texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    // Integer textures must use NEAREST filtering (we read via texelFetch).
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        glow::R16UI as i32,
        width as i32,
        height as i32,
        0,
        glow::RED_INTEGER,
        glow::UNSIGNED_SHORT,
        glow::PixelUnpackData::Slice(None),
    );
    tex
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

//! OpenGL (glow) rendering for the composited image: up to `MAX_CHANNELS`
//! single-channel raw-sample textures, a LUT texture (one row per channel), and
//! a fragment shader that does per-channel window/level → LUT → additive blend,
//! with a minification box filter for clean zoom-out. Each channel is either an
//! integer texture (R16Uint — 8/16/32-bit-int sources, window/level in 0..65535
//! units) or a float texture (R32F — 32-bit float sources, window/level in the
//! data's own units, so the per-frame CPU rescale is avoided).
//!
//! # Host responsibilities
//!
//! The host creates the GL context and hands over an `Arc` of it at
//! construction; uploads then need no per-call context. `paint` still takes a
//! `&glow::Context` explicitly, because GL is stateful and the host's drawing
//! layer may hand back a context it has already configured (viewport, scissor,
//! bound framebuffer) for the target rect — that setup is the host's job, and
//! this backend deliberately does not second-guess it.

use crate::pack::{self, BrickMaxima, PackedVolume, TexelFormat, Visibility, VolumeLayout};
use crate::{
    capture_tiles, tile_params, ChannelKind, ChannelUniform, Lut, VolumeImage, VolumeInterp,
    VolumeKind, VolumeParams, VolumeUploadError, CAPTURE_TILE, MAX_CHANNELS,
};
use glow::HasContext as _;
use std::sync::Arc;

const LUT_WIDTH: i32 = 256;

/// Short human-readable backend name, for a host that wants to show it.
pub const BACKEND: &str = "glow";

/// GPU bytes per volume sample for `kind` on this backend — glow stores each
/// kind natively (R8/R16/R32F), so it matches the CPU-side size exactly. A
/// volume builder should budget on the larger of CPU/GPU footprint.
pub fn volume_gpu_bps(kind: VolumeKind) -> usize {
    match kind {
        VolumeKind::U8 => 1,
        VolumeKind::U16 => 2,
        VolumeKind::F32 => 4,
    }
}

/// Targets a desktop GL 3.x context with GLSL `#version 140` — what egui_glow
/// and most native GL hosts provide — so usampler2D / texelFetch / gl_VertexID
/// are available.
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

// --- 3D volume ray-march ---------------------------------------------------
// A fullscreen pass that, per pixel, builds a camera ray and marches it through
// the packed 3D textures. Three compositing modes (`u_mode`): maximum-intensity
// projection (order-independent), emission-absorption alpha compositing (the
// ImageJ 3D Viewer's "Volume" look), and an opaque gradient-shaded isosurface.
// The camera arrives as an explicit basis (eye/forward/right/up + fov), avoiding
// any matrix inverse in-shader.
//
// Sample spacing is derived from the actual voxel size (via `textureSize`),
// half a voxel along every axis the ray crosses — this, plus a per-pixel jitter
// of the ray start, is what kills the slice-aligned banding that showed up when
// looking down an axis. Keeping every sample strictly inside the box (jitter <
// one step) plus a zero texture border avoids the edge artifacts.
//
// The same accelerations as the WGSL twin (`volume.wgsl`, whose header says
// more): channels packed four to a texture, empty bricks skipped on the sample
// lattice, a MIP stopping once every channel is saturated, and a MIP reading a
// coarser (max-reduced) mip level where a voxel is smaller than a pixel.

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

uniform sampler3D vol_a;     // channels 0..3, packed into the components
uniform sampler3D vol_b;     // channels 4, 5
uniform sampler3D occ;       // one texel per brick: may a ray need to sample there
uniform sampler2D lut_tex;
uniform vec3 cam_eye;
uniform vec3 cam_forward;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform float tan_half_fov;
uniform float aspect;
uniform vec3 box_he;         // volume box half-extents (scaled dims)
uniform vec2 ch_window[6];   // per-channel min, max in sampled-texture units
uniform float ch_enabled[6];
// Where to sample each channel's LUT for the isosurface albedo. Not always
// the top entry: a LUT that peaks early and ends dark (a contrast-stretched
// palette) would otherwise paint a black surface.
uniform float ch_albedo[6];
uniform int num_channels;
uniform int u_mode;          // 0 = MIP, 1 = alpha DVR, 2 = isosurface
uniform float u_density;     // alpha-DVR opacity scale (higher = more solid)
uniform float u_iso;         // isosurface threshold in windowed units (0..1)
uniform float u_brightness;  // gain on the composited colour (MIP and alpha DVR)
uniform int u_interp;        // 0 = point/linear (GL filter), 1 = in-shader tricubic
uniform int u_max_steps;     // steps per ray at most (<= 0: MAX_STEPS)
// (size of one pixel at unit distance, highest mip level a MIP may read,
//  brick size in voxels, skip empty bricks: 1 or 0)
uniform vec4 u_accel;

// ---------------------------------------------------------------------------
// Isosurface appearance (u_mode == 2). Both are compile-time constants: they
// are a fixed look, not per-frame state, so they cost no uniform slot and stay
// in step with the wgpu backend's copy of the same two values (volume.wgsl).
//
//   SURFACE_GRAY     Albedo the surface is scaled to. 1.0 is a blown-out white
//                    for a plain grayscale stack; lower reads as softer gray.
//                    (Blender's classic default base color is 0.8.)
//   SURFACE_CONTRAST Split between ambient and camera-headlight diffuse:
//                    shade = (1 - CONTRAST) + CONTRAST * diff. 0.0 is flat and
//                    unshaded, 1.0 falls all the way to black at silhouettes.
// ---------------------------------------------------------------------------
const float SURFACE_GRAY = 0.6;
const float SURFACE_CONTRAST = 0.85;


// Upper bound on ray-march samples. High enough that the largest volume the
// memory budget admits (~1600-voxel diagonal) still gets half-voxel sampling;
// the loops run `n` iterations (n <= MAX_STEPS), so ordinary volumes never pay
// for the headroom.
const int MAX_STEPS = 4096;

// NOTE: every texture read in this shader uses textureLod (explicit LOD) or
// texelFetch. The march loop's trip count and the early return are
// per-fragment, i.e. non-uniform control flow, where implicit-LOD texture() is
// undefined in GLSL.

// Fast tricubic B-spline reconstruction at mip level `lvl`: 8 hardware-linear
// taps (Sigg & Hadwiger), every component at once. Used when u_interp == 1 (the
// GL filter is LINEAR then, so each tap is a bilinear-in-3D fetch).
vec4 sampleCubic(sampler3D tex, vec3 coord, int lvl) {
    vec3 n = vec3(textureSize(tex, lvl));
    float l = float(lvl);
    vec3 cg = coord * n - 0.5;
    vec3 idx = floor(cg);
    vec3 f = cg - idx;
    vec3 f2 = f * f;
    vec3 f3 = f2 * f;
    vec3 w0 = (1.0 / 6.0) * (-f3 + 3.0 * f2 - 3.0 * f + 1.0);
    vec3 w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + 4.0);
    vec3 w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0);
    vec3 w3 = (1.0 / 6.0) * f3;
    vec3 g0 = w0 + w1;
    vec3 g1 = w2 + w3;
    vec3 h0 = (w1 / g0 - 0.5 + idx) / n;
    vec3 h1 = (w3 / g1 + 1.5 + idx) / n;
    vec4 s000 = textureLod(tex, vec3(h0.x, h0.y, h0.z), l);
    vec4 s100 = textureLod(tex, vec3(h1.x, h0.y, h0.z), l);
    vec4 s010 = textureLod(tex, vec3(h0.x, h1.y, h0.z), l);
    vec4 s110 = textureLod(tex, vec3(h1.x, h1.y, h0.z), l);
    vec4 s001 = textureLod(tex, vec3(h0.x, h0.y, h1.z), l);
    vec4 s101 = textureLod(tex, vec3(h1.x, h0.y, h1.z), l);
    vec4 s011 = textureLod(tex, vec3(h0.x, h1.y, h1.z), l);
    vec4 s111 = textureLod(tex, vec3(h1.x, h1.y, h1.z), l);
    vec4 x00 = mix(s100, s000, g0.x);
    vec4 x10 = mix(s110, s010, g0.x);
    vec4 x01 = mix(s101, s001, g0.x);
    vec4 x11 = mix(s111, s011, g0.x);
    vec4 y0 = mix(x10, x00, g0.y);
    vec4 y1 = mix(x11, x01, g0.y);
    return mix(y1, y0, g0.z);
}

// Every component at a point: GL-filtered fetch (nearest/linear) or the
// tricubic above.
vec4 smp(sampler3D tex, vec3 c, int lvl) {
    if (u_interp == 1) {
        return sampleCubic(tex, c, lvl);
    }
    return textureLod(tex, c, float(lvl));
}

// The six channels at a point: one fetch of each packed texture that holds an
// enabled channel. Unfetched components read as 0.
void fetch(vec3 tc, int lvl, bool need_a, bool need_b, out vec4 a, out vec4 b) {
    a = vec4(0.0);
    b = vec4(0.0);
    if (need_a) { a = smp(vol_a, tc, lvl); }
    if (need_b) { b = smp(vol_b, tc, lvl); }
}

float norm_ch(float v, int idx) {
    vec2 mm = ch_window[idx];
    return clamp((v - mm.x) / max(mm.y - mm.x, 1e-6), 0.0, 1.0);
}
vec3 lut_col(float t, int idx) {
    return textureLod(lut_tex, vec2(t, (float(idx) + 0.5) / 6.0), 0.0).rgb;
}
// Whether channel `idx` is present and switched on.
bool on(int idx) {
    return idx < num_channels && ch_enabled[idx] > 0.5;
}

// World point -> volume texcoord, with Y and Z flipped so the volume matches the
// 2D movie: image row 0 stays at the top (Y), and movie frame 0 is the slice
// nearest the default camera (Z). X already matches.
vec3 vol_coord(vec3 p) {
    vec3 tc = (p + box_he) * (0.5 / box_he);
    return clamp(vec3(tc.x, 1.0 - tc.y, 1.0 - tc.z), 0.0, 1.0);
}

// Whether the brick at `tc` may contribute. `dims0` is the level-0 size.
bool occupied(vec3 tc, vec3 dims0) {
    vec3 grid = vec3(textureSize(occ, 0));
    vec3 b = min(floor(tc * dims0 / u_accel.z), grid - 1.0);
    return texelFetch(occ, ivec3(b), 0).r > 0.5;
}

// How many steps of `dt` take the ray from `tc` (moving `dtc` in texture
// coordinates per unit of t) to the first sample past the brick it is in: at
// least one, so the march always advances, and at most `n` + 1. See the WGSL
// twin: GLSL leaves int() of an infinity undefined, and drivers wrap it.
int steps_past_brick(vec3 tc, vec3 dtc, vec3 dims0, float dt, int n) {
    float bs = u_accel.z;
    vec3 b = floor(tc * dims0 / bs);
    vec3 lo = b * bs / dims0;
    vec3 hi = min((b + 1.0) * bs, dims0) / dims0;
    float t = 1e30;
    if (dtc.x > 1e-9) t = min(t, (hi.x - tc.x) / dtc.x);
    if (dtc.x < -1e-9) t = min(t, (lo.x - tc.x) / dtc.x);
    if (dtc.y > 1e-9) t = min(t, (hi.y - tc.y) / dtc.y);
    if (dtc.y < -1e-9) t = min(t, (lo.y - tc.y) / dtc.y);
    if (dtc.z > 1e-9) t = min(t, (hi.z - tc.z) / dtc.z);
    if (dtc.z < -1e-9) t = min(t, (lo.z - tc.z) / dtc.z);
    return int(min(floor(max(t, 0.0) / dt), float(n))) + 1;
}

// Composite scalar field for the isosurface: the max windowed value across every
// enabled channel, plus (via `col`) the *full-intensity* LUT color of whichever
// channel is the maximum there. Using the LUT's top color (not the color at the
// crossing value) as the surface albedo keeps the surface equally bright at every
// threshold — the iso level chooses only where the surface is, the lighting term
// does the shading. Returns the field value in [0,1].
float field_col(vec3 tc, bool need_a, bool need_b, out vec3 col) {
    vec4 a;
    vec4 b;
    fetch(tc, 0, need_a, need_b, a, b);
    float f = 0.0;
    col = vec3(0.0);
    if (on(0)) { float t = norm_ch(a.x, 0); if (t > f) { f = t; col = lut_col(ch_albedo[0], 0); } }
    if (on(1)) { float t = norm_ch(a.y, 1); if (t > f) { f = t; col = lut_col(ch_albedo[1], 1); } }
    if (on(2)) { float t = norm_ch(a.z, 2); if (t > f) { f = t; col = lut_col(ch_albedo[2], 2); } }
    if (on(3)) { float t = norm_ch(a.w, 3); if (t > f) { f = t; col = lut_col(ch_albedo[3], 3); } }
    if (on(4)) { float t = norm_ch(b.x, 4); if (t > f) { f = t; col = lut_col(ch_albedo[4], 4); } }
    if (on(5)) { float t = norm_ch(b.y, 5); if (t > f) { f = t; col = lut_col(ch_albedo[5], 5); } }
    return f;
}

float field_scalar(vec3 tc, bool need_a, bool need_b) {
    vec3 ignore;
    return field_col(tc, need_a, need_b, ignore);
}

// World-space gradient of the field (central differences, one voxel apart). It
// points toward increasing intensity; used (normalized, two-sided) as the
// surface normal for shading.
vec3 field_grad(vec3 wp, vec3 voxel, bool need_a, bool need_b) {
    return vec3(
        field_scalar(vol_coord(wp + vec3(voxel.x, 0.0, 0.0)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(voxel.x, 0.0, 0.0)), need_a, need_b),
        field_scalar(vol_coord(wp + vec3(0.0, voxel.y, 0.0)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(0.0, voxel.y, 0.0)), need_a, need_b),
        field_scalar(vol_coord(wp + vec3(0.0, 0.0, voxel.z)), need_a, need_b) - field_scalar(vol_coord(wp - vec3(0.0, 0.0, voxel.z)), need_a, need_b)
    );
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

    // Which packed textures hold a channel that is on: the other is not read.
    bool need_a = on(0) || on(1) || on(2) || on(3);
    bool need_b = on(4) || on(5);

    vec3 dims0 = vec3(textureSize(vol_a, 0));
    vec3 voxel0 = 2.0 * box_he / max(dims0, vec3(1.0));
    vec3 ard = max(abs(rd), vec3(1e-6));

    // The mip level this ray reads (MIP only); see the WGSL twin.
    int lvl = 0;
    if (u_mode == 0 && u_accel.y > 0.5) {
        float foot = t0 * u_accel.x;
        vec3 lat = voxel0 * sqrt(max(vec3(1.0) - rd * rd, vec3(0.0)));
        float lateral = max(lat.x, max(lat.y, lat.z));
        if (lateral > 0.0 && foot > lateral) {
            lvl = int(clamp(floor(log2(foot / lateral)), 0.0, u_accel.y));
        }
    }
    vec3 voxel = 2.0 * box_he / max(vec3(textureSize(vol_a, lvl)), vec3(1.0));

    // Half a voxel along every axis the ray crosses, measured along the ray,
    // then capped by the host's per-frame budget. See the WGSL twin.
    vec3 along = voxel / ard;
    float base_step = max(0.5 * min(along.x, min(along.y, along.z)), 1e-4);
    float span = t1 - t0;
    int cap = (u_max_steps > 0) ? min(u_max_steps, MAX_STEPS) : MAX_STEPS;
    int n = clamp(int(span / base_step) + 1, 1, cap);
    float dt = span / float(n);

    // Jitter the start within one step: decorrelates slice-aligned banding and
    // keeps all n samples strictly inside [t0, t1].
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 p0 = ro + rd * (t0 + jitter * dt);
    vec3 dp = rd * dt;
    // Sample i is at p0 + i * dp, worked out afresh each step, so a ray that
    // skips lands exactly where one that walked would.
    // Not for a ray of no length, which has no step to count bricks in.
    bool skip = u_accel.w > 0.5 && dt > 0.0;
    vec3 dtc = vec3(rd.x, -rd.y, -rd.z) * (0.5 / box_he);

    if (u_mode == 1) {
        // Alpha DVR (ImageJ 3D Viewer "Volume"): emission-absorption, front-to-back.
        // Disabled channels are skipped entirely (uniform branch, no sampling cost).
        vec3 col = vec3(0.0);
        float acc = 0.0;
        int i = 0;
        while (i < n) {
            vec3 tc = vol_coord(p0 + dp * float(i));
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                continue;
            }
            vec4 a;
            vec4 b;
            fetch(tc, 0, need_a, need_b, a, b);
            vec3 emit = vec3(0.0);
            float wsum = 0.0;
            if (on(0)) { float t = norm_ch(a.x, 0); emit += t * lut_col(t, 0); wsum += t; }
            if (on(1)) { float t = norm_ch(a.y, 1); emit += t * lut_col(t, 1); wsum += t; }
            if (on(2)) { float t = norm_ch(a.z, 2); emit += t * lut_col(t, 2); wsum += t; }
            if (on(3)) { float t = norm_ch(a.w, 3); emit += t * lut_col(t, 3); wsum += t; }
            if (on(4)) { float t = norm_ch(b.x, 4); emit += t * lut_col(t, 4); wsum += t; }
            if (on(5)) { float t = norm_ch(b.y, 5); emit += t * lut_col(t, 5); wsum += t; }
            float al = 1.0 - exp(-u_density * wsum * dt);
            vec3 albedo = (wsum > 1e-6) ? emit / wsum : vec3(0.0);
            col += (1.0 - acc) * al * albedo;
            acc += (1.0 - acc) * al;
            if (acc > 0.995) break;
            i += 1;
        }
        frag_color = vec4(clamp(col * u_brightness, vec3(0.0), vec3(1.0)), 1.0);
    } else if (u_mode == 2) {
        // Isosurface: walk the ray until the composite field first reaches u_iso,
        // refine the crossing between the two bracketing samples, then shade the
        // hit from the field gradient with a camera headlight (two-sided, so both
        // faces are lit and silhouettes fall off to the ambient term).
        float prev_f = 0.0;
        bool skipped = false;
        vec3 hit_col = vec3(0.0);
        bool hit = false;
        int i = 0;
        while (i < n) {
            vec3 p = p0 + dp * float(i);
            vec3 tc = vol_coord(p);
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                skipped = true;
                continue;
            }
            vec3 fcol;
            float f = field_col(tc, need_a, need_b, fcol);
            if (f >= u_iso) {
                // Refined against the sample before, read now if a skip
                // jumped over it; see the WGSL twin.
                float before = skipped ? field_scalar(vol_coord(p - dp), need_a, need_b) : prev_f;
                float frac = clamp((u_iso - before) / max(f - before, 1e-6), 0.0, 1.0);
                vec3 hp = p - dp * (1.0 - frac);
                vec3 g = field_grad(hp, voxel0, need_a, need_b);
                vec3 nrm = (length(g) > 1e-6) ? normalize(g) : vec3(0.0, 0.0, 1.0);
                float diff = abs(dot(nrm, -rd));
                float shade = (1.0 - SURFACE_CONTRAST) + SURFACE_CONTRAST * diff;
                hit_col = clamp(fcol * SURFACE_GRAY * shade, vec3(0.0), vec3(1.0));
                hit = true;
                break;
            }
            prev_f = f;
            skipped = false;
            i += 1;
        }
        frag_color = vec4(hit ? hit_col : vec3(0.0), 1.0);
    } else {
        // MIP: per-channel maximum of the *windowed* value (norm_ch is monotonic,
        // so this equals windowing the raw max — but stays correct for float data
        // with negative values, where a raw-space max seeded at 0 would be wrong),
        // then color + sum. Disabled channels are skipped entirely.
        float m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0;
        int i = 0;
        while (i < n) {
            vec3 tc = vol_coord(p0 + dp * float(i));
            if (skip && !occupied(tc, dims0)) {
                i += steps_past_brick(tc, dtc, dims0, dt, n);
                continue;
            }
            vec4 a;
            vec4 b;
            fetch(tc, lvl, need_a, need_b, a, b);
            if (on(0)) m0 = max(m0, norm_ch(a.x, 0));
            if (on(1)) m1 = max(m1, norm_ch(a.y, 1));
            if (on(2)) m2 = max(m2, norm_ch(a.z, 2));
            if (on(3)) m3 = max(m3, norm_ch(a.w, 3));
            if (on(4)) m4 = max(m4, norm_ch(b.x, 4));
            if (on(5)) m5 = max(m5, norm_ch(b.y, 5));
            // Every channel at the top of its window: nothing further can raise it.
            if ((!on(0) || m0 >= 1.0) && (!on(1) || m1 >= 1.0) && (!on(2) || m2 >= 1.0)
                && (!on(3) || m3 >= 1.0) && (!on(4) || m4 >= 1.0) && (!on(5) || m5 >= 1.0)) {
                break;
            }
            i += 1;
        }
        vec3 color = vec3(0.0);
        if (on(0)) color += lut_col(m0, 0);
        if (on(1)) color += lut_col(m1, 1);
        if (on(2)) color += lut_col(m2, 2);
        if (on(3)) color += lut_col(m3, 3);
        if (on(4)) color += lut_col(m4, 4);
        if (on(5)) color += lut_col(m5, 5);
        frag_color = vec4(clamp(color * u_brightness, vec3(0.0), vec3(1.0)), 1.0);
    }
}
"#;

// Scale a reduced-size render of the 3D view up to the view's own size: one
// full-screen triangle sampling the render bilinearly (the WGSL twin is
// `blit.wgsl`).
const BLIT_VERTEX_SRC: &str = r#"#version 140
out vec2 v_uv;
void main() {
    vec2 c = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0);
    v_uv = c;
}
"#;

const BLIT_FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D src;
void main() {
    frag_color = textureLod(src, v_uv, 0.0);
}
"#;

/// Per-channel texture-allocation kind, tracked so `ensure_size` only
/// reallocates when something actually changes.
const KIND_UNUSED: u8 = 0; // channel not present: both textures are 1x1 dummies
const KIND_INT8: u8 = 1; // integer channel: R8Uint full-size, R32F dummy
const KIND_INT16: u8 = 2; // integer channel: R16Uint full-size, R32F dummy
const KIND_FLOAT: u8 = 3; // float channel: R32F full-size, int texture dummy

pub struct ImageRenderResources {
    /// Our handle on the host's GL context, so uploads need no per-call
    /// context argument. `paint` still takes one explicitly — see the module
    /// docs on why.
    gl: Arc<glow::Context>,
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
    /// The picture `capture_volume` rendered, until `take_volume_capture`
    /// collects it.
    capture: Option<Result<VolumeImage, String>>,
}

/// The 3D volume pipeline: its own program, the packed volume textures (see
/// [`crate::pack`]), the brick occupancy texture, and the marshalled draw
/// params — plus the reduced-size render target the view uses while it moves.
struct VolumeGl {
    program: glow::NativeProgram,
    /// Channels 0..3 and channels 4..5 (1x1x1 dummies until a volume is
    /// uploaded, and the second whenever there are at most four channels).
    tex: [glow::NativeTexture; 2],
    occupancy: glow::NativeTexture,
    /// What each texture is allocated as: size, internal format, levels.
    tex_alloc: [((u32, u32, u32), u32, u32); 2],
    occupancy_dims: (u32, u32, u32),
    layout: Option<VolumeLayout>,
    bricks: Option<BrickMaxima>,
    occupancy_for: Option<Visibility>,
    interp: i32,      // GL_NEAREST / GL_LINEAR
    interp_mode: i32, // shader `u_interp`: 0 = point/linear, 1 = tricubic
    accel_skip: bool,
    accel_lod: bool,
    downscale: u32,
    u_vol_a: Option<glow::NativeUniformLocation>,
    u_vol_b: Option<glow::NativeUniformLocation>,
    u_occ: Option<glow::NativeUniformLocation>,
    u_lut: Option<glow::NativeUniformLocation>,
    u_eye: Option<glow::NativeUniformLocation>,
    u_forward: Option<glow::NativeUniformLocation>,
    u_right: Option<glow::NativeUniformLocation>,
    u_up: Option<glow::NativeUniformLocation>,
    u_tan: Option<glow::NativeUniformLocation>,
    u_aspect: Option<glow::NativeUniformLocation>,
    u_box_he: Option<glow::NativeUniformLocation>,
    u_window: Option<glow::NativeUniformLocation>,
    u_enabled: Option<glow::NativeUniformLocation>,
    u_albedo: Option<glow::NativeUniformLocation>,
    u_num_channels: Option<glow::NativeUniformLocation>,
    u_mode: Option<glow::NativeUniformLocation>,
    u_density: Option<glow::NativeUniformLocation>,
    u_iso: Option<glow::NativeUniformLocation>,
    u_brightness: Option<glow::NativeUniformLocation>,
    u_max_steps: Option<glow::NativeUniformLocation>,
    u_interp: Option<glow::NativeUniformLocation>,
    u_accel: Option<glow::NativeUniformLocation>,
    params: Option<VolumeParams>,
    blit_program: glow::NativeProgram,
    u_blit_src: Option<glow::NativeUniformLocation>,
    /// The reduced-size render target: framebuffer, colour texture, size.
    lowres: Option<(glow::NativeFramebuffer, glow::NativeTexture, (u32, u32))>,
}

impl VolumeGl {
    unsafe fn new(gl: &glow::Context) -> Self {
        let program = link_program(gl, VOL_VERTEX_SRC, VOL_FRAGMENT_SRC);
        let blit_program = link_program(gl, BLIT_VERTEX_SRC, BLIT_FRAGMENT_SRC);
        let placeholder = || {
            let tex = gl.create_texture().expect("create volume texture");
            gl.bind_texture(glow::TEXTURE_3D, Some(tex));
            set_volume_filter(gl, glow::NEAREST as i32, 1);
            alloc_volume_dummy(gl);
            tex
        };
        let u = |name: &str| gl.get_uniform_location(program, name);
        VolumeGl {
            program,
            tex: [placeholder(), placeholder()],
            occupancy: placeholder(),
            tex_alloc: [((1, 1, 1), glow::R8, 1); 2],
            occupancy_dims: (1, 1, 1),
            layout: None,
            bricks: None,
            occupancy_for: None,
            interp: glow::NEAREST as i32,
            interp_mode: 0,
            accel_skip: true,
            accel_lod: true,
            downscale: 1,
            u_vol_a: u("vol_a"),
            u_vol_b: u("vol_b"),
            u_occ: u("occ"),
            u_lut: u("lut_tex"),
            u_eye: u("cam_eye"),
            u_forward: u("cam_forward"),
            u_right: u("cam_right"),
            u_up: u("cam_up"),
            u_tan: u("tan_half_fov"),
            u_aspect: u("aspect"),
            u_box_he: u("box_he"),
            u_window: u("ch_window"),
            u_enabled: u("ch_enabled"),
            u_albedo: u("ch_albedo"),
            u_num_channels: u("num_channels"),
            u_mode: u("u_mode"),
            u_density: u("u_density"),
            u_iso: u("u_iso"),
            u_brightness: u("u_brightness"),
            u_max_steps: u("u_max_steps"),
            u_interp: u("u_interp"),
            u_accel: u("u_accel"),
            params: None,
            u_blit_src: gl.get_uniform_location(blit_program, "src"),
            blit_program,
            lowres: None,
        }
    }
}

impl ImageRenderResources {
    /// Build every program and texture on `gl`. The `Arc` is kept so later
    /// uploads don't need the context passed back in.
    pub fn new(gl: Arc<glow::Context>) -> Self {
        let ctx = Arc::clone(&gl);
        let gl: &glow::Context = &ctx;
        unsafe {
            let program = link_program(gl, VERTEX_SRC, FRAGMENT_SRC);
            let vao = gl.create_vertex_array().expect("create VAO");
            let channel_textures = std::array::from_fn(|_| create_int_texture(gl, 1, 1));
            let channel_ftextures = std::array::from_fn(|_| create_float_texture(gl, 1, 1));
            let lut_texture = create_lut_texture(gl);

            let u_ch_tex =
                std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_tex")));
            let u_ch_ftex =
                std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_ftex")));

            Self {
                gl: Arc::clone(&ctx),
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
                capture: None,
            }
        }
    }

    /// Largest per-axis 3D-texture dimension the driver supports; the app uses
    /// it (with a memory cap) to decide whether the volume must be subsampled.
    pub fn max_3d_texture_size(&self) -> u32 {
        unsafe { self.gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE).max(0) as u32 }
    }

    /// GPU memory a new volume could take, in bytes: what the driver says is
    /// free (`GL_NVX_gpu_memory_info` on NVIDIA, `GL_ATI_meminfo` on AMD), plus
    /// what the current volume holds (a new one replaces it). `None` on a
    /// driver with neither. Both count the whole device, other programs
    /// included, so they are the cautious figure. A builder should take a
    /// share of this, not all of it.
    ///
    /// Also `None` for a GPU without memory of its own worth the name. Mesa
    /// answers both extensions with dedicated memory only, which on an
    /// integrated GPU is a small carve-out of system RAM (or nothing at all,
    /// on Intel's), while textures there live in system RAM anyway: going by
    /// it would cut a volume to a fraction for no gain.
    pub fn volume_memory_available(&self) -> Option<u64> {
        /// `GL_NVX_gpu_memory_info`: the dedicated memory, and the free part
        /// of it, KiB.
        const DEDICATED_VIDMEM_NVX: u32 = 0x9047;
        const CURRENT_AVAILABLE_VIDMEM_NVX: u32 = 0x9049;
        /// `GL_ATI_meminfo`: four KiB figures, the first the free pool.
        const TEXTURE_FREE_MEMORY_ATI: u32 = 0x87FC;
        /// Less dedicated memory than this is taken for an integrated GPU's.
        const DEDICATED_KIB_MIN: i32 = 1 << 20;
        let gl: &glow::Context = &self.gl;
        let extensions = gl.supported_extensions();
        let free_kib = if extensions.contains("GL_NVX_gpu_memory_info") {
            if unsafe { gl.get_parameter_i32(DEDICATED_VIDMEM_NVX) } < DEDICATED_KIB_MIN {
                return None;
            }
            unsafe { gl.get_parameter_i32(CURRENT_AVAILABLE_VIDMEM_NVX) }
        } else if extensions.contains("GL_ATI_meminfo") {
            // glGetIntegerv writes all four: the slice must hold them.
            let mut v = [0i32; 4];
            unsafe { gl.get_parameter_i32_slice(TEXTURE_FREE_MEMORY_ATI, &mut v) };
            v[0]
        } else {
            return None;
        };
        if free_kib <= 0 {
            return None;
        }
        let current = self
            .volume
            .layout
            .as_ref()
            .map_or(0, |l| l.total_bytes() as u64);
        Some(free_kib as u64 * 1024 + current)
    }

    /// An upload that returned `Ok` and was refused after all. Never, here:
    /// OpenGL says at once (see the wgpu backend, where WebGPU does not).
    pub fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)> {
        None
    }

    /// Whether the volume is waiting to hear there was room for it. Never,
    /// here (see `take_late_upload_error`).
    pub fn volume_pending(&mut self) -> bool {
        false
    }

    /// How this context stores a volume whose channels are `kind`: what a
    /// builder should pack it as (see [`crate::pack`]). Every kind keeps its
    /// own width: 8-bit and 16-bit as normalized integers, float as `f32`.
    pub fn volume_texel_format(&self, kind: VolumeKind) -> TexelFormat {
        match kind {
            VolumeKind::U8 => TexelFormat::Unorm8,
            VolumeKind::U16 => TexelFormat::Unorm16,
            VolumeKind::F32 => TexelFormat::Float32,
        }
    }

    /// Upload a volume packed by [`crate::pack`], every level of every texture,
    /// and the brick maxima the renderer skips empty space with. When a
    /// texture's size and format are unchanged (a 4D timepoint step), its
    /// storage is refilled with `tex_sub_image_3d` instead of reallocated.
    ///
    /// Fails with [`VolumeUploadError::OutOfMemory`] when the driver had no
    /// room, leaving 1x1x1 placeholders and no volume: build a smaller one and
    /// try again.
    pub fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError> {
        // The driver reads as many bytes as each level's size says, from
        // wherever the slice starts: one too short is read past.
        volume.check().map_err(VolumeUploadError::Unsupported)?;
        let layout = &volume.layout;
        if layout.format == TexelFormat::Float16 {
            return Err(VolumeUploadError::Unsupported(
                "this context stores float volumes as f32; pack them as Float32".into(),
            ));
        }
        let max = self.max_3d_texture_size();
        let (w, h, d) = layout.dims;
        if w > max || h > max || d > max {
            return Err(VolumeUploadError::Unsupported(format!(
                "a {w} x {h} x {d} volume is beyond this GPU's 3D texture limit of {max}"
            )));
        }
        let gl: &glow::Context = &self.gl;
        let v = &mut self.volume;
        let mut out_of_memory = false;
        unsafe {
            drain_errors(gl);
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            for t in 0..2 {
                let Some(tex) = layout.textures.get(t) else {
                    if v.tex_alloc[t] != ((1, 1, 1), glow::R8, 1) {
                        v.tex[t] = renew_texture(gl, v.tex[t]);
                        set_volume_filter(gl, v.interp, 1);
                        alloc_volume_dummy(gl);
                        v.tex_alloc[t] = ((1, 1, 1), glow::R8, 1);
                    }
                    continue;
                };
                let (internal, fmt, ty) = gl_format(layout.format, tex.components);
                let want = (layout.dims, internal, layout.levels);
                let refill = v.tex_alloc[t] == want;
                if refill {
                    gl.bind_texture(glow::TEXTURE_3D, Some(v.tex[t]));
                } else {
                    v.tex[t] = renew_texture(gl, v.tex[t]);
                    set_volume_filter(gl, v.interp, layout.levels);
                }
                for (level, data) in volume.levels[t].iter().enumerate() {
                    let (lw, lh, ld) = layout.level_dims(level as u32);
                    if refill {
                        gl.tex_sub_image_3d(
                            glow::TEXTURE_3D,
                            level as i32,
                            0,
                            0,
                            0,
                            lw as i32,
                            lh as i32,
                            ld as i32,
                            fmt,
                            ty,
                            glow::PixelUnpackData::Slice(Some(data)),
                        );
                    } else {
                        gl.tex_image_3d(
                            glow::TEXTURE_3D,
                            level as i32,
                            internal as i32,
                            lw as i32,
                            lh as i32,
                            ld as i32,
                            0,
                            fmt,
                            ty,
                            glow::PixelUnpackData::Slice(Some(data)),
                        );
                    }
                    if gl.get_error() == glow::OUT_OF_MEMORY {
                        out_of_memory = true;
                        break;
                    }
                }
                if out_of_memory {
                    break;
                }
                v.tex_alloc[t] = want;
            }
            if out_of_memory {
                // Nothing half-made stays bound: placeholders, and no volume.
                for t in 0..2 {
                    v.tex[t] = renew_texture(gl, v.tex[t]);
                    set_volume_filter(gl, v.interp, 1);
                    alloc_volume_dummy(gl);
                    v.tex_alloc[t] = ((1, 1, 1), glow::R8, 1);
                }
                drain_errors(gl);
                v.layout = None;
                v.bricks = None;
                v.occupancy_for = None;
                return Err(VolumeUploadError::OutOfMemory(format!(
                    "a {w} x {h} x {d} volume of {} MB did not fit",
                    layout.total_bytes() >> 20
                )));
            }
        }
        v.layout = Some(layout.clone());
        v.bricks = Some(volume.bricks.clone());
        v.occupancy_for = None;
        self.refresh_occupancy();
        Ok(())
    }

    /// (Re)upload the whole volume, one entry per channel: each `bytes` is
    /// `w*h*d` samples, z-major, in the units of its `kind` (U8/U16 unorm, F32).
    /// Packs it here, on the calling thread — a builder with a thread of its own
    /// should pack it there instead (see [`crate::pack`]) and call
    /// [`upload_packed_volume`].
    ///
    /// [`upload_packed_volume`]: Self::upload_packed_volume
    pub fn upload_volumes(
        &mut self,
        w: u32,
        h: u32,
        d: u32,
        channels: &[(VolumeKind, Vec<u8>)],
    ) -> Result<(), VolumeUploadError> {
        let kinds: Vec<VolumeKind> = channels.iter().map(|(k, _)| *k).collect();
        let format = self.volume_texel_format(pack::common_kind(&kinds));
        let borrowed: Vec<(VolumeKind, &[u8])> =
            channels.iter().map(|(k, b)| (*k, b.as_slice())).collect();
        let packed = PackedVolume::from_channels((w, h, d), &borrowed, format);
        self.upload_packed_volume(&packed)
    }

    /// Work out which bricks can be seen under the current params, if that
    /// changed, and upload it.
    fn refresh_occupancy(&mut self) {
        let v = &mut self.volume;
        let (Some(bricks), Some(params)) = (&v.bricks, &v.params) else {
            return;
        };
        let vis = Visibility::of(params);
        if v.occupancy_for == Some(vis) {
            return;
        }
        let occ = pack::occupancy(bricks, &vis);
        let dims = bricks.dims;
        let gl: &glow::Context = &self.gl;
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.bind_texture(glow::TEXTURE_3D, Some(v.occupancy));
            if v.occupancy_dims != dims {
                set_volume_filter(gl, glow::NEAREST as i32, 1);
                gl.tex_image_3d(
                    glow::TEXTURE_3D,
                    0,
                    glow::R8 as i32,
                    dims.0 as i32,
                    dims.1 as i32,
                    dims.2 as i32,
                    0,
                    glow::RED,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&occ)),
                );
                v.occupancy_dims = dims;
            } else {
                gl.tex_sub_image_3d(
                    glow::TEXTURE_3D,
                    0,
                    0,
                    0,
                    0,
                    dims.0 as i32,
                    dims.1 as i32,
                    dims.2 as i32,
                    glow::RED,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&occ)),
                );
            }
        }
        v.occupancy_for = Some(vis);
    }

    /// Switch every present volume texture's sampling filter without
    /// re-uploading. Cubic uses the GL linear filter (its 8-tap reconstruction
    /// is in-shader). No-op when nothing changed (this is called every 3D frame).
    pub fn set_volume_interp(&mut self, interp: VolumeInterp) {
        let gl_interp = match interp {
            VolumeInterp::Nearest => glow::NEAREST as i32,
            VolumeInterp::Linear | VolumeInterp::Cubic => glow::LINEAR as i32,
        };
        let mode = interp.shader_mode();
        if self.volume.interp == gl_interp && self.volume.interp_mode == mode {
            return;
        }
        self.volume.interp = gl_interp;
        self.volume.interp_mode = mode;
        for t in 0..2 {
            let levels = self.volume.tex_alloc[t].2;
            unsafe {
                self.gl
                    .bind_texture(glow::TEXTURE_3D, Some(self.volume.tex[t]));
                set_volume_filter(&self.gl, gl_interp, levels);
            }
        }
    }

    /// Stash the ray-march params (camera, per-channel window, steps); applied in
    /// `paint_volume`.
    pub fn set_volume_params(&mut self, params: VolumeParams) {
        self.volume.params = Some(params);
        self.refresh_occupancy();
    }

    /// Switch the two accelerations on or off: skipping parts of the volume
    /// that cannot be seen (exact: the picture is the same either way) and, in
    /// the maximum projection, reading coarser mip levels where a voxel is
    /// smaller than a pixel. Both are on unless switched off; off is for
    /// comparing, measuring, or ruling them out.
    pub fn set_volume_acceleration(&mut self, skip_empty: bool, mip_levels: bool) {
        self.volume.accel_skip = skip_empty;
        self.volume.accel_lod = mip_levels;
    }

    /// How many times smaller per axis to render the 3D view than it is
    /// painted: 1 renders every pixel, and more renders a reduced picture
    /// `paint_volume` scales up — for a view that is moving, where a sharp
    /// frame is wasted (see [`crate::MOVING_DOWNSCALE`]). The size itself is
    /// the viewport the host has set when `paint_volume` runs; the two numbers
    /// before `downscale` are taken for the wgpu backend's sake and unused.
    pub fn set_volume_target(&mut self, _width: u32, _height: u32, downscale: u32) {
        self.volume.downscale = downscale.max(1);
    }

    /// Draw the 3D view into the viewport the host has set (egui_glow sets it
    /// to the callback's rect): the ray-march itself, or — when
    /// [`set_volume_target`] asked for a reduced render — a picture that many
    /// times smaller per axis, rendered into a framebuffer of its own and then
    /// scaled up. The host's framebuffer, viewport and scissor are as they
    /// were afterwards.
    ///
    /// [`set_volume_target`]: Self::set_volume_target
    pub fn paint_volume(&mut self, gl: &glow::Context) {
        let Some(p) = self.volume.params else { return };
        let mut viewport = [0i32; 4];
        unsafe { gl.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport) };
        let [vx, vy, vw, vh] = viewport;
        let f = self.volume.downscale;
        if f <= 1 {
            self.draw_raymarch(gl, &p, vh.max(1) as u32);
            return;
        }
        let size = (
            (vw.max(1) as u32).div_ceil(f),
            (vh.max(1) as u32).div_ceil(f),
        );
        unsafe {
            let Some((fb, tex)) = self.ensure_lowres(gl, size) else {
                // No framebuffer to render into: draw at full size instead.
                self.draw_raymarch(gl, &p, vh.max(1) as u32);
                return;
            };
            let previous_fb = gl.get_parameter_framebuffer(glow::FRAMEBUFFER_BINDING);
            let scissor = gl.is_enabled(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fb));
            gl.disable(glow::SCISSOR_TEST);
            gl.viewport(0, 0, size.0 as i32, size.1 as i32);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            self.draw_raymarch(gl, &p, size.1);
            gl.bind_framebuffer(glow::FRAMEBUFFER, previous_fb);
            gl.viewport(vx, vy, vw, vh);
            if scissor {
                gl.enable(glow::SCISSOR_TEST);
            }
            // Scale it up over the host's viewport.
            let v = &self.volume;
            gl.use_program(Some(v.blit_program));
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.uniform_1_i32(v.u_blit_src.as_ref(), 0);
            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            // Not left bound to a unit while it is the framebuffer's
            // attachment next frame: that is a feedback loop to some drivers.
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    /// The reduced-size render target, `size` pixels, made or remade to fit.
    /// `None` when the driver will not make one.
    unsafe fn ensure_lowres(
        &mut self,
        gl: &glow::Context,
        size: (u32, u32),
    ) -> Option<(glow::NativeFramebuffer, glow::NativeTexture)> {
        if let Some((fb, tex, have)) = self.volume.lowres {
            if have == size {
                return Some((fb, tex));
            }
            gl.delete_framebuffer(fb);
            gl.delete_texture(tex);
            self.volume.lowres = None;
        }
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        for (pname, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
            (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, pname, value as i32);
        }
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            size.0 as i32,
            size.1 as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        let previous_fb = gl.get_parameter_framebuffer(glow::FRAMEBUFFER_BINDING);
        let fb = match gl.create_framebuffer() {
            Ok(fb) => fb,
            Err(_) => {
                gl.delete_texture(tex);
                return None;
            }
        };
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(tex),
            0,
        );
        let complete = gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE;
        gl.bind_framebuffer(glow::FRAMEBUFFER, previous_fb);
        if !complete {
            gl.delete_framebuffer(fb);
            gl.delete_texture(tex);
            return None;
        }
        self.volume.lowres = Some((fb, tex, size));
        Some((fb, tex))
    }

    /// Ray-march the volume with params `p` into the bound framebuffer's
    /// viewport, which is `height` pixels tall.
    fn draw_raymarch(&self, gl: &glow::Context, p: &VolumeParams, height: u32) {
        let v = &self.volume;
        let levels = v.layout.as_ref().map_or(1, |l| l.levels);
        let accel = [
            2.0 * p.tan_half_fov / height.max(1) as f32,
            if v.accel_lod {
                (levels - 1) as f32
            } else {
                0.0
            },
            pack::BRICK as f32,
            // Only where the occupancy is `p`'s own; see the wgpu backend.
            if v.accel_skip && v.occupancy_for == Some(Visibility::of(p)) {
                1.0
            } else {
                0.0
            },
        ];
        unsafe {
            gl.use_program(Some(v.program));
            gl.bind_vertex_array(Some(self.vao));
            // Packed volume → units 0 and 1, occupancy → 2, LUT → 3.
            for (unit, (tex, loc)) in [
                (v.tex[0], &v.u_vol_a),
                (v.tex[1], &v.u_vol_b),
                (v.occupancy, &v.u_occ),
            ]
            .into_iter()
            .enumerate()
            {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(glow::TEXTURE_3D, Some(tex));
                gl.uniform_1_i32(loc.as_ref(), unit as i32);
            }
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(v.u_lut.as_ref(), 3);

            gl.uniform_3_f32(v.u_eye.as_ref(), p.eye[0], p.eye[1], p.eye[2]);
            gl.uniform_3_f32(
                v.u_forward.as_ref(),
                p.forward[0],
                p.forward[1],
                p.forward[2],
            );
            gl.uniform_3_f32(v.u_right.as_ref(), p.right[0], p.right[1], p.right[2]);
            gl.uniform_3_f32(v.u_up.as_ref(), p.up[0], p.up[1], p.up[2]);
            gl.uniform_1_f32(v.u_tan.as_ref(), p.tan_half_fov);
            gl.uniform_1_f32(v.u_aspect.as_ref(), p.aspect);
            gl.uniform_3_f32(v.u_box_he.as_ref(), p.box_he[0], p.box_he[1], p.box_he[2]);
            gl.uniform_2_f32_slice(v.u_window.as_ref(), &p.windows);
            gl.uniform_1_f32_slice(v.u_enabled.as_ref(), &p.enabled);
            gl.uniform_1_f32_slice(v.u_albedo.as_ref(), &p.albedo_t);
            gl.uniform_1_i32(v.u_num_channels.as_ref(), p.num_channels);
            gl.uniform_1_i32(v.u_mode.as_ref(), p.render_mode);
            gl.uniform_1_f32(v.u_density.as_ref(), p.density);
            gl.uniform_1_f32(v.u_iso.as_ref(), p.iso);
            gl.uniform_1_f32(v.u_brightness.as_ref(), p.brightness);
            gl.uniform_1_i32(v.u_max_steps.as_ref(), p.max_steps);
            gl.uniform_1_i32(v.u_interp.as_ref(), v.interp_mode);
            gl.uniform_4_f32(v.u_accel.as_ref(), accel[0], accel[1], accel[2], accel[3]);

            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
    }

    /// Render the volume off screen at `width` x `height` pixels, with the
    /// params last given to `set_volume_params`; collect the picture with
    /// [`take_volume_capture`].
    ///
    /// The ray-march has no resolution of its own (every pixel is a ray), so
    /// this is the view at any size: sharper than the window when larger, and
    /// with nothing the host draws over the view in it. The aspect ratio is
    /// taken from the size, not from the params. GL reads back synchronously,
    /// so the picture is ready at once; the two calls match the wgpu backend,
    /// whose readback is not. The host's framebuffer, viewport and scissor are
    /// put back as they were.
    ///
    /// Rendered in tiles of [`CAPTURE_TILE`] pixels, each read back before the
    /// next is drawn, so that no single job runs long enough for the system to
    /// reset the GPU, and GPU memory holds one tile rather than the picture.
    ///
    /// [`take_volume_capture`]: Self::take_volume_capture
    pub fn capture_volume(&mut self, width: u32, height: u32) -> Result<(), String> {
        let Some(params) = self.volume.params else {
            return Err("there is no volume to render".into());
        };
        let max = self.max_2d_texture_size();
        if width == 0 || height == 0 || width > max || height > max {
            return Err(format!(
                "a {width} x {height} picture is beyond what this GPU can render (up to {max} a side)"
            ));
        }
        if self.capture.is_some() {
            return Err("the last capture has not been taken yet".into());
        }
        let result = unsafe { self.render_offscreen(&params, width, height) };
        self.capture = Some(result);
        Ok(())
    }

    /// The picture [`capture_volume`] rendered, or `None` when none was asked
    /// for since the last one was taken.
    ///
    /// [`capture_volume`]: Self::capture_volume
    pub fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>> {
        self.capture.take()
    }

    unsafe fn render_offscreen(
        &self,
        params: &VolumeParams,
        width: u32,
        height: u32,
    ) -> Result<VolumeImage, String> {
        let gl = Arc::clone(&self.gl);
        let previous_fb = gl.get_parameter_framebuffer(glow::FRAMEBUFFER_BINDING);
        let mut previous_viewport = [0i32; 4];
        gl.get_parameter_i32_slice(glow::VIEWPORT, &mut previous_viewport);
        let scissor = gl.is_enabled(glow::SCISSOR_TEST);

        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            CAPTURE_TILE as i32,
            CAPTURE_TILE as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        let fb = match gl.create_framebuffer() {
            Ok(fb) => fb,
            Err(why) => {
                gl.delete_texture(texture);
                return Err(why);
            }
        };
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        let result = if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE
        {
            Err("cannot render the volume off screen".to_string())
        } else {
            gl.disable(glow::SCISSOR_TEST);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
            let whole = VolumeParams {
                aspect: width as f32 / height as f32,
                ..*params
            };
            let stride = width as usize * 4;
            let mut rgba = vec![0u8; stride * height as usize];
            let mut pixels = vec![0u8; (CAPTURE_TILE * CAPTURE_TILE * 4) as usize];
            for tile in capture_tiles(width, height) {
                let (x, y, w, h) = tile;
                gl.viewport(0, 0, w as i32, h as i32);
                gl.clear_color(0.0, 0.0, 0.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
                // Always the ray-march itself, at full size, whatever the
                // view on screen is doing.
                self.draw_raymarch(&gl, &tile_params(&whole, (width, height), tile), h);
                let tile_bytes = &mut pixels[..(w * h * 4) as usize];
                // Waits for the tile: the next is not drawn until this one is
                // done, which is what keeps each job short.
                gl.read_pixels(
                    0,
                    0,
                    w as i32,
                    h as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(tile_bytes)),
                );
                // GL's rows run bottom to top.
                for (r, row) in pixels[..(w * h * 4) as usize]
                    .chunks_exact(w as usize * 4)
                    .rev()
                    .enumerate()
                {
                    let at = (y as usize + r) * stride + x as usize * 4;
                    let out = &mut rgba[at..at + w as usize * 4];
                    for (o, px) in out.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                        o.copy_from_slice(&[px[0], px[1], px[2], 255]);
                    }
                }
            }
            Ok(VolumeImage {
                width,
                height,
                rgba,
            })
        };

        gl.bind_framebuffer(glow::FRAMEBUFFER, previous_fb);
        gl.delete_framebuffer(fb);
        gl.delete_texture(texture);
        let [x, y, w, h] = previous_viewport;
        gl.viewport(x, y, w, h);
        if scissor {
            gl.enable(glow::SCISSOR_TEST);
        }
        result
    }

    /// (Re)allocate channel textures for the current frame size and per-channel
    /// `kind`. An `Int8`/`Int16` channel gets a full-size R8Uint/R16Uint integer
    /// texture (float texture a 1x1 dummy); a `Float` channel gets a full-size
    /// R32F float texture (integer texture a 1x1 dummy); channels past
    /// `kinds.len()` are unused (both 1x1). No-op when nothing changed.
    /// Largest per-axis 2D-texture dimension the driver supports. Frames beyond
    /// it cannot be uploaded, so the app subsamples to fit; see
    /// `fast_tiff_viewer::sync::fit_stride`.
    pub fn max_2d_texture_size(&self) -> u32 {
        unsafe { self.gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE).max(0) as u32 }
    }

    pub fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind]) {
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
        let gl: &glow::Context = &self.gl;
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
                let (fw, fh) = if want[c] == KIND_FLOAT {
                    (width, height)
                } else {
                    (1, 1)
                };
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_ftextures[c]));
                alloc_float(gl, fw, fh);
            }
        }
        self.current_size = (width, height);
        self.current_kinds = want;
    }

    /// Upload one integer channel's raw 16-bit samples (R16Uint texture).
    pub fn upload_channel_u16(&self, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl: &glow::Context = &self.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[channel]));
            // Our rows are tightly packed (`width * 2` bytes for R16UI). Force
            // the unpack alignment to 1 so an odd-width image — whose row length
            // isn't a multiple of GL's default alignment of 4 — isn't read with
            // phantom end-of-row padding, which shears the image diagonally. We
            // set it on every upload because the GL context is shared with the
            // host's own drawing layer, which changes this global state for its
            // own textures.
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
    pub fn upload_channel_u8(&self, channel: usize, width: u32, height: u32, samples: &[u8]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl: &glow::Context = &self.gl;
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
    pub fn upload_channel_f32(&self, channel: usize, width: u32, height: u32, samples: &[f32]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl: &glow::Context = &self.gl;
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
    pub fn upload_lut(&self, channel: usize, lut: &Lut) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl: &glow::Context = &self.gl;
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
    /// `paint`. Nothing reaches the GPU here — the glow backend has no per-frame
    /// uniform buffer, so these are stashed and set at draw time.
    pub fn set_params(
        &mut self,
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

    /// Draw the composited image. Assumes the host has already set the viewport
    /// and scissor to the image's on-screen rect.
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
            gl.uniform_2_f32(
                self.u_uv_offset.as_ref(),
                self.uv_offset[0],
                self.uv_offset[1],
            );
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
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        glow::NEAREST as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MAG_FILTER,
        glow::NEAREST as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_S,
        glow::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_T,
        glow::CLAMP_TO_EDGE as i32,
    );
}

/// Allocate a 1x1x1 single-level R8 placeholder for the currently-bound 3D
/// texture.
/// Delete `old` and make a new texture in its place, bound to `TEXTURE_3D`
/// with nothing in it.
///
/// For a volume texture whose storage changes. Redefining a texture's levels
/// in place frees only the levels redefined: a 1x1x1 placeholder written over
/// level 0 leaves the old levels 1 and up defined, and some drivers (Mesa's,
/// for one) keep the whole old allocation alive for them. A new object has
/// none. The draw binds the textures afresh each time, so nothing else holds
/// the old name.
unsafe fn renew_texture(gl: &glow::Context, old: glow::NativeTexture) -> glow::NativeTexture {
    gl.delete_texture(old);
    let tex = gl.create_texture().expect("create volume texture");
    gl.bind_texture(glow::TEXTURE_3D, Some(tex));
    tex
}

unsafe fn alloc_volume_dummy(gl: &glow::Context) {
    gl.tex_image_3d(
        glow::TEXTURE_3D,
        0,
        glow::R8 as i32,
        1,
        1,
        1,
        0,
        glow::RED,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(Some(&[0])),
    );
}

/// The `(internal format, format, type)` of a packed texture of `components`
/// components stored as `format`. One or two components keep their own; three
/// or four are RGBA.
fn gl_format(format: TexelFormat, components: usize) -> (u32, u32, u32) {
    let fmt = match components {
        1 => glow::RED,
        2 => glow::RG,
        _ => glow::RGBA,
    };
    let (internal, ty) = match (format, components) {
        (TexelFormat::Unorm8, 1) => (glow::R8, glow::UNSIGNED_BYTE),
        (TexelFormat::Unorm8, 2) => (glow::RG8, glow::UNSIGNED_BYTE),
        (TexelFormat::Unorm8, _) => (glow::RGBA8, glow::UNSIGNED_BYTE),
        (TexelFormat::Unorm16, 1) => (glow::R16, glow::UNSIGNED_SHORT),
        (TexelFormat::Unorm16, 2) => (glow::RG16, glow::UNSIGNED_SHORT),
        (TexelFormat::Unorm16, _) => (glow::RGBA16, glow::UNSIGNED_SHORT),
        (TexelFormat::Float16, 1) => (glow::R16F, glow::HALF_FLOAT),
        (TexelFormat::Float16, 2) => (glow::RG16F, glow::HALF_FLOAT),
        (TexelFormat::Float16, _) => (glow::RGBA16F, glow::HALF_FLOAT),
        (TexelFormat::Float32, 1) => (glow::R32F, glow::FLOAT),
        (TexelFormat::Float32, 2) => (glow::RG32F, glow::FLOAT),
        (TexelFormat::Float32, _) => (glow::RGBA32F, glow::FLOAT),
    };
    (internal, fmt, ty)
}

/// Clear the GL error flags, so the next `get_error` speaks only for what
/// follows. Bounded: a lost context can report errors for ever.
unsafe fn drain_errors(gl: &glow::Context) {
    for _ in 0..32 {
        if gl.get_error() == glow::NO_ERROR {
            break;
        }
    }
}

/// Set the currently-bound 3D texture's filter (`gl_interp` = NEAREST/LINEAR,
/// choosing within one of its `levels` mip levels — the shader picks the level
/// itself) and a **zero border** on all three axes. `CLAMP_TO_BORDER` (border =
/// 0) rather than `CLAMP_TO_EDGE` means samples at/just outside a face fade to 0
/// instead of smearing the boundary voxel — no bright edge shell on MIP, and a
/// clean silhouette on the alpha DVR.
unsafe fn set_volume_filter(gl: &glow::Context, gl_interp: i32, levels: u32) {
    let min_filter = match (levels > 1, gl_interp == glow::NEAREST as i32) {
        (false, _) => gl_interp,
        (true, true) => glow::NEAREST_MIPMAP_NEAREST as i32,
        (true, false) => glow::LINEAR_MIPMAP_NEAREST as i32,
    };
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MIN_FILTER, min_filter);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MAG_FILTER, gl_interp);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_BASE_LEVEL, 0);
    gl.tex_parameter_i32(
        glow::TEXTURE_3D,
        glow::TEXTURE_MAX_LEVEL,
        levels.max(1) as i32 - 1,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_3D,
        glow::TEXTURE_WRAP_S,
        glow::CLAMP_TO_BORDER as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_3D,
        glow::TEXTURE_WRAP_T,
        glow::CLAMP_TO_BORDER as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_3D,
        glow::TEXTURE_WRAP_R,
        glow::CLAMP_TO_BORDER as i32,
    );
    gl.tex_parameter_f32_slice(
        glow::TEXTURE_3D,
        glow::TEXTURE_BORDER_COLOR,
        &[0.0, 0.0, 0.0, 0.0],
    );
}

unsafe fn create_lut_texture(gl: &glow::Context) -> glow::NativeTexture {
    let tex = gl.create_texture().expect("create LUT texture");
    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MIN_FILTER,
        glow::LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_MAG_FILTER,
        glow::LINEAR as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_S,
        glow::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameter_i32(
        glow::TEXTURE_2D,
        glow::TEXTURE_WRAP_T,
        glow::CLAMP_TO_EDGE as i32,
    );
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
        panic!(
            "FastTIFF shader link failed: {}",
            gl.get_program_info_log(program)
        );
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
        let which = if kind == glow::VERTEX_SHADER {
            "vertex"
        } else {
            "fragment"
        };
        panic!(
            "FastTIFF {which} shader compile failed: {}",
            gl.get_shader_info_log(shader)
        );
    }
    shader
}

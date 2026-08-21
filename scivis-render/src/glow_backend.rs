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

use crate::{ChannelKind, ChannelUniform, Lut, VolumeInterp, VolumeKind, VolumeParams, MAX_CHANNELS};
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
// the 3D texture(s). Three compositing modes (`u_mode`): maximum-intensity
// projection (order-independent), emission-absorption alpha compositing (the
// ImageJ 3D Viewer's "Volume" look), and an opaque gradient-shaded isosurface.
// The camera arrives as an explicit basis (eye/forward/right/up + fov), avoiding
// any matrix inverse in-shader.
//
// Sample spacing is derived from the actual voxel size (via `textureSize`) so
// it's the same regardless of the view direction — this, plus a per-pixel
// jitter of the ray start, is what kills the slice-aligned banding that showed
// up when looking down an axis. Keeping every sample strictly inside the box
// (jitter < one step) plus a zero texture border avoids the edge artifacts.

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

// One MIP pass per channel, summed — the 3D analog of the 2D compositor. Each
// channel has an unorm sampler (`volN`, integer sources) and a float sampler
// (`volfN`, R32F); `ch_is_float[N]` selects which carries the data (the other is
// a 1x1x1 dummy). `num_channels` is a uniform, so the per-channel `if` guards are
// uniform control flow (texture sampling inside them is well-defined).
const VOL_FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_ndc;
out vec4 frag_color;

uniform sampler3D vol0;
uniform sampler3D vol1;
uniform sampler3D vol2;
uniform sampler3D vol3;
uniform sampler3D vol4;
uniform sampler3D vol5;
uniform sampler3D volf0;
uniform sampler3D volf1;
uniform sampler3D volf2;
uniform sampler3D volf3;
uniform sampler3D volf4;
uniform sampler3D volf5;
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
uniform float ch_is_float[6];
// Where to sample each channel's LUT for the isosurface albedo. Not always
// the top entry: a LUT that peaks early and ends dark (a contrast-stretched
// palette) would otherwise paint a black surface.
uniform float ch_albedo[6];
uniform int num_channels;
uniform int u_mode;          // 0 = MIP, 1 = alpha DVR, 2 = isosurface
uniform float u_density;     // alpha-DVR opacity scale (higher = more solid)
uniform float u_iso;         // isosurface threshold in windowed units (0..1)
uniform int u_interp;        // 0 = point/linear (GL filter), 1 = in-shader tricubic

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

// NOTE: every texture read in this shader uses textureLod (explicit LOD 0).
// The march loop's trip count and the early return are per-fragment, i.e.
// non-uniform control flow, where implicit-LOD texture() is undefined in GLSL
// — the textures are single-level so most drivers cope, but explicit LOD makes
// it well-defined everywhere (and skips the LOD calculation).

// Fast tricubic B-spline reconstruction: 8 hardware-linear taps (Sigg &
// Hadwiger). Smoother than trilinear; used when u_interp == 1 (the GL filter is
// LINEAR then, so each tap is a bilinear-in-3D fetch).
float sampleCubic(sampler3D tex, vec3 coord) {
    vec3 n = vec3(textureSize(tex, 0));
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
    float s000 = textureLod(tex, vec3(h0.x, h0.y, h0.z), 0.0).r;
    float s100 = textureLod(tex, vec3(h1.x, h0.y, h0.z), 0.0).r;
    float s010 = textureLod(tex, vec3(h0.x, h1.y, h0.z), 0.0).r;
    float s110 = textureLod(tex, vec3(h1.x, h1.y, h0.z), 0.0).r;
    float s001 = textureLod(tex, vec3(h0.x, h0.y, h1.z), 0.0).r;
    float s101 = textureLod(tex, vec3(h1.x, h0.y, h1.z), 0.0).r;
    float s011 = textureLod(tex, vec3(h0.x, h1.y, h1.z), 0.0).r;
    float s111 = textureLod(tex, vec3(h1.x, h1.y, h1.z), 0.0).r;
    float x00 = mix(s100, s000, g0.x);
    float x10 = mix(s110, s010, g0.x);
    float x01 = mix(s101, s001, g0.x);
    float x11 = mix(s111, s011, g0.x);
    float y0 = mix(x10, x00, g0.y);
    float y1 = mix(x11, x01, g0.y);
    return mix(y1, y0, g0.z);
}

// One texel value: GL-filtered fetch (nearest/linear) or the tricubic above.
float smp(sampler3D tex, vec3 c) {
    return (u_interp == 1) ? sampleCubic(tex, c) : textureLod(tex, c, 0.0).r;
}

// Raw sampled value for each channel (its unorm or float texture).
float raw0(vec3 tc) { return ch_is_float[0] > 0.5 ? smp(volf0, tc) : smp(vol0, tc); }
float raw1(vec3 tc) { return ch_is_float[1] > 0.5 ? smp(volf1, tc) : smp(vol1, tc); }
float raw2(vec3 tc) { return ch_is_float[2] > 0.5 ? smp(volf2, tc) : smp(vol2, tc); }
float raw3(vec3 tc) { return ch_is_float[3] > 0.5 ? smp(volf3, tc) : smp(vol3, tc); }
float raw4(vec3 tc) { return ch_is_float[4] > 0.5 ? smp(volf4, tc) : smp(vol4, tc); }
float raw5(vec3 tc) { return ch_is_float[5] > 0.5 ? smp(volf5, tc) : smp(vol5, tc); }

float norm_ch(float v, int idx) {
    vec2 mm = ch_window[idx];
    return clamp((v - mm.x) / max(mm.y - mm.x, 1e-6), 0.0, 1.0);
}
vec3 lut_col(float t, int idx) {
    return textureLod(lut_tex, vec2(t, (float(idx) + 0.5) / 6.0), 0.0).rgb;
}

// World point -> volume texcoord, with Y and Z flipped so the volume matches the
// 2D movie: image row 0 stays at the top (Y), and movie frame 0 is the slice
// nearest the default camera (Z). X already matches.
vec3 vol_coord(vec3 p) {
    vec3 tc = (p + box_he) * (0.5 / box_he);
    return clamp(vec3(tc.x, 1.0 - tc.y, 1.0 - tc.z), 0.0, 1.0);
}

// Composite scalar field for the isosurface: the max windowed value across every
// enabled channel, plus (via `col`) the *full-intensity* LUT color of whichever
// channel is the maximum there. Using the LUT's top color (not the color at the
// crossing value) as the surface albedo keeps the surface equally bright at every
// threshold — the iso level chooses only where the surface is, the lighting term
// does the shading. Returns the field value in [0,1].
float field_col(vec3 tc, out vec3 col) {
    float f = 0.0;
    col = vec3(0.0);
    if (ch_enabled[0] > 0.5)                     { float t = norm_ch(raw0(tc), 0); if (t > f) { f = t; col = lut_col(ch_albedo[0], 0); } }
    if (num_channels > 1 && ch_enabled[1] > 0.5) { float t = norm_ch(raw1(tc), 1); if (t > f) { f = t; col = lut_col(ch_albedo[1], 1); } }
    if (num_channels > 2 && ch_enabled[2] > 0.5) { float t = norm_ch(raw2(tc), 2); if (t > f) { f = t; col = lut_col(ch_albedo[2], 2); } }
    if (num_channels > 3 && ch_enabled[3] > 0.5) { float t = norm_ch(raw3(tc), 3); if (t > f) { f = t; col = lut_col(ch_albedo[3], 3); } }
    if (num_channels > 4 && ch_enabled[4] > 0.5) { float t = norm_ch(raw4(tc), 4); if (t > f) { f = t; col = lut_col(ch_albedo[4], 4); } }
    if (num_channels > 5 && ch_enabled[5] > 0.5) { float t = norm_ch(raw5(tc), 5); if (t > f) { f = t; col = lut_col(ch_albedo[5], 5); } }
    return f;
}

float field_scalar(vec3 tc) {
    vec3 ignore;
    return field_col(tc, ignore);
}

// World-space gradient of the field (central differences, one voxel apart). It
// points toward increasing intensity; used (normalized, two-sided) as the
// surface normal for shading.
vec3 field_grad(vec3 wp, vec3 voxel) {
    return vec3(
        field_scalar(vol_coord(wp + vec3(voxel.x, 0.0, 0.0))) - field_scalar(vol_coord(wp - vec3(voxel.x, 0.0, 0.0))),
        field_scalar(vol_coord(wp + vec3(0.0, voxel.y, 0.0))) - field_scalar(vol_coord(wp - vec3(0.0, voxel.y, 0.0))),
        field_scalar(vol_coord(wp + vec3(0.0, 0.0, voxel.z))) - field_scalar(vol_coord(wp - vec3(0.0, 0.0, voxel.z)))
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

    // Even sample spacing from the actual voxel size (view-independent).
    ivec3 tdim = (ch_is_float[0] > 0.5) ? textureSize(volf0, 0) : textureSize(vol0, 0);
    vec3 voxel = 2.0 * box_he / max(vec3(tdim), vec3(1.0));
    float base_step = max(min(voxel.x, min(voxel.y, voxel.z)) * 0.5, 1e-4);
    float span = t1 - t0;
    int n = clamp(int(span / base_step) + 1, 1, MAX_STEPS);
    float dt = span / float(n);

    // Jitter the start within one step: decorrelates slice-aligned banding and
    // keeps all n samples strictly inside [t0, t1].
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 dp = rd * dt;
    vec3 p = ro + rd * (t0 + jitter * dt);

    if (u_mode == 1) {
        // Alpha DVR (ImageJ 3D Viewer "Volume"): emission-absorption, front-to-back.
        // Disabled channels are skipped entirely (uniform branch, no sampling cost).
        vec3 col = vec3(0.0);
        float acc = 0.0;
        for (int i = 0; i < n; i++) {
            vec3 tc = vol_coord(p);
            vec3 emit = vec3(0.0);
            float wsum = 0.0;
            if (ch_enabled[0] > 0.5)                       { float t = norm_ch(raw0(tc), 0); emit += t * lut_col(t, 0); wsum += t; }
            if (num_channels > 1 && ch_enabled[1] > 0.5) { float t = norm_ch(raw1(tc), 1); emit += t * lut_col(t, 1); wsum += t; }
            if (num_channels > 2 && ch_enabled[2] > 0.5) { float t = norm_ch(raw2(tc), 2); emit += t * lut_col(t, 2); wsum += t; }
            if (num_channels > 3 && ch_enabled[3] > 0.5) { float t = norm_ch(raw3(tc), 3); emit += t * lut_col(t, 3); wsum += t; }
            if (num_channels > 4 && ch_enabled[4] > 0.5) { float t = norm_ch(raw4(tc), 4); emit += t * lut_col(t, 4); wsum += t; }
            if (num_channels > 5 && ch_enabled[5] > 0.5) { float t = norm_ch(raw5(tc), 5); emit += t * lut_col(t, 5); wsum += t; }
            float a = 1.0 - exp(-u_density * wsum * dt);
            vec3 albedo = (wsum > 1e-6) ? emit / wsum : vec3(0.0);
            col += (1.0 - acc) * a * albedo;
            acc += (1.0 - acc) * a;
            if (acc > 0.995) break;
            p += dp;
        }
        frag_color = vec4(clamp(col, vec3(0.0), vec3(1.0)), 1.0);
    } else if (u_mode == 2) {
        // Isosurface: walk the ray until the composite field first reaches u_iso,
        // refine the crossing between the two bracketing samples, then shade the
        // hit from the field gradient with a camera headlight (two-sided, so both
        // faces are lit and silhouettes fall off to the ambient term).
        float prev_f = 0.0;
        vec3 hit_col = vec3(0.0);
        bool hit = false;
        for (int i = 0; i < n; i++) {
            vec3 tc = vol_coord(p);
            vec3 fcol;
            float f = field_col(tc, fcol);
            if (f >= u_iso) {
                float frac = clamp((u_iso - prev_f) / max(f - prev_f, 1e-6), 0.0, 1.0);
                vec3 hp = p - dp * (1.0 - frac);
                vec3 g = field_grad(hp, voxel);
                vec3 nrm = (length(g) > 1e-6) ? normalize(g) : vec3(0.0, 0.0, 1.0);
                float diff = abs(dot(nrm, -rd));
                float shade = (1.0 - SURFACE_CONTRAST) + SURFACE_CONTRAST * diff;
                hit_col = clamp(fcol * SURFACE_GRAY * shade, vec3(0.0), vec3(1.0));
                hit = true;
                break;
            }
            prev_f = f;
            p += dp;
        }
        frag_color = vec4(hit ? hit_col : vec3(0.0), 1.0);
    } else {
        // MIP: per-channel maximum of the *windowed* value (norm_ch is monotonic,
        // so this equals windowing the raw max — but stays correct for float data
        // with negative values, where a raw-space max seeded at 0 would be wrong),
        // then color + sum. Disabled channels are skipped entirely.
        float m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0;
        for (int i = 0; i < n; i++) {
            vec3 tc = vol_coord(p);
            if (ch_enabled[0] > 0.5) m0 = max(m0, norm_ch(raw0(tc), 0));
            if (num_channels > 1 && ch_enabled[1] > 0.5) m1 = max(m1, norm_ch(raw1(tc), 1));
            if (num_channels > 2 && ch_enabled[2] > 0.5) m2 = max(m2, norm_ch(raw2(tc), 2));
            if (num_channels > 3 && ch_enabled[3] > 0.5) m3 = max(m3, norm_ch(raw3(tc), 3));
            if (num_channels > 4 && ch_enabled[4] > 0.5) m4 = max(m4, norm_ch(raw4(tc), 4));
            if (num_channels > 5 && ch_enabled[5] > 0.5) m5 = max(m5, norm_ch(raw5(tc), 5));
            p += dp;
        }
        vec3 color = ch_enabled[0] * lut_col(m0, 0);
        if (num_channels > 1) color += ch_enabled[1] * lut_col(m1, 1);
        if (num_channels > 2) color += ch_enabled[2] * lut_col(m2, 2);
        if (num_channels > 3) color += ch_enabled[3] * lut_col(m3, 3);
        if (num_channels > 4) color += ch_enabled[4] * lut_col(m4, 4);
        if (num_channels > 5) color += ch_enabled[5] * lut_col(m5, 5);
        frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
    }
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
}

/// The 3D volume pipeline: its own program, per-channel unorm (R8/R16) and R32F
/// 3D textures (one of each pair carries the data, the other is a 1x1x1 dummy —
/// mirroring the 2D int/float split), and the marshalled draw params.
struct VolumeGl {
    program: glow::NativeProgram,
    tex_unorm: [glow::NativeTexture; MAX_CHANNELS],
    tex_float: [glow::NativeTexture; MAX_CHANNELS],
    dims: (u32, u32, u32),
    /// Per-channel kind (`None` = unused: both textures are 1x1x1 dummies).
    kinds: [Option<VolumeKind>; MAX_CHANNELS],
    interp: i32,      // GL_NEAREST / GL_LINEAR
    interp_mode: i32, // shader `u_interp`: 0 = point/linear, 1 = tricubic
    u_vol: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
    u_volf: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
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
    u_is_float: Option<glow::NativeUniformLocation>,
    u_albedo: Option<glow::NativeUniformLocation>,
    u_num_channels: Option<glow::NativeUniformLocation>,
    u_mode: Option<glow::NativeUniformLocation>,
    u_density: Option<glow::NativeUniformLocation>,
    u_iso: Option<glow::NativeUniformLocation>,
    u_interp: Option<glow::NativeUniformLocation>,
    params: Option<VolumeParams>,
}

impl VolumeGl {
    unsafe fn new(gl: &glow::Context) -> Self {
        let program = link_program(gl, VOL_VERTEX_SRC, VOL_FRAGMENT_SRC);
        VolumeGl {
            program,
            tex_unorm: std::array::from_fn(|_| create_volume_texture(gl, false)),
            tex_float: std::array::from_fn(|_| create_volume_texture(gl, true)),
            dims: (1, 1, 1),
            kinds: [None; MAX_CHANNELS],
            interp: glow::NEAREST as i32,
            interp_mode: 0,
            u_vol: std::array::from_fn(|c| gl.get_uniform_location(program, &format!("vol{c}"))),
            u_volf: std::array::from_fn(|c| gl.get_uniform_location(program, &format!("volf{c}"))),
            u_lut: gl.get_uniform_location(program, "lut_tex"),
            u_eye: gl.get_uniform_location(program, "cam_eye"),
            u_forward: gl.get_uniform_location(program, "cam_forward"),
            u_right: gl.get_uniform_location(program, "cam_right"),
            u_up: gl.get_uniform_location(program, "cam_up"),
            u_tan: gl.get_uniform_location(program, "tan_half_fov"),
            u_aspect: gl.get_uniform_location(program, "aspect"),
            u_box_he: gl.get_uniform_location(program, "box_he"),
            u_window: gl.get_uniform_location(program, "ch_window"),
            u_enabled: gl.get_uniform_location(program, "ch_enabled"),
            u_is_float: gl.get_uniform_location(program, "ch_is_float"),
            u_albedo: gl.get_uniform_location(program, "ch_albedo"),
            u_num_channels: gl.get_uniform_location(program, "num_channels"),
            u_mode: gl.get_uniform_location(program, "u_mode"),
            u_density: gl.get_uniform_location(program, "u_density"),
            u_iso: gl.get_uniform_location(program, "u_iso"),
            u_interp: gl.get_uniform_location(program, "u_interp"),
            params: None,
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

            let u_ch_tex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_tex")));
            let u_ch_ftex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_ftex")));

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
            }
        }
    }

    /// Largest per-axis 3D-texture dimension the driver supports; the app uses
    /// it (with a memory cap) to decide whether the volume must be subsampled.
    pub fn max_3d_texture_size(&self) -> u32 {
        unsafe { self.gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE).max(0) as u32 }
    }

    /// (Re)upload the whole volume, one entry per channel: each `bytes` is
    /// `w*h*d` samples, z-major, in the units of its `kind` (U8/U16 unorm, F32).
    /// Channels past `channels.len()` are shrunk to 1x1x1 dummies. When a
    /// channel's dims + kind are unchanged (a 4D timepoint step), the existing
    /// storage is refilled with `tex_sub_image_3d` instead of reallocated.
    pub fn upload_volumes(&mut self, w: u32, h: u32, d: u32, channels: &[(VolumeKind, Vec<u8>)]) {
        let gl: &glow::Context = &self.gl;
        let v = &mut self.volume;
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            for c in 0..MAX_CHANNELS {
                match channels.get(c) {
                    Some((kind, bytes)) => {
                        let (data_tex, dummy_tex) = if *kind == VolumeKind::F32 {
                            (v.tex_float[c], v.tex_unorm[c])
                        } else {
                            (v.tex_unorm[c], v.tex_float[c])
                        };
                        let (internal, fmt, ty) = match kind {
                            VolumeKind::U8 => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
                            VolumeKind::U16 => (glow::R16, glow::RED, glow::UNSIGNED_SHORT),
                            VolumeKind::F32 => (glow::R32F, glow::RED, glow::FLOAT),
                        };
                        gl.bind_texture(glow::TEXTURE_3D, Some(data_tex));
                        if v.kinds[c] == Some(*kind) && v.dims == (w, h, d) {
                            // Same shape: refill in place (no driver realloc).
                            gl.tex_sub_image_3d(
                                glow::TEXTURE_3D,
                                0,
                                0,
                                0,
                                0,
                                w as i32,
                                h as i32,
                                d as i32,
                                fmt,
                                ty,
                                glow::PixelUnpackData::Slice(Some(bytes)),
                            );
                        } else {
                            set_volume_filter(gl, v.interp, *kind);
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
                            // Shrink the unused sibling texture to a 1x1x1 dummy.
                            gl.bind_texture(glow::TEXTURE_3D, Some(dummy_tex));
                            let dummy_kind = if *kind == VolumeKind::F32 { VolumeKind::U8 } else { VolumeKind::F32 };
                            alloc_volume_dummy(gl, dummy_kind);
                            v.kinds[c] = Some(*kind);
                        }
                    }
                    None => {
                        if v.kinds[c].is_some() {
                            // Absent channel: both textures back to 1x1x1 dummies.
                            gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_unorm[c]));
                            alloc_volume_dummy(gl, VolumeKind::U8);
                            gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_float[c]));
                            alloc_volume_dummy(gl, VolumeKind::F32);
                            v.kinds[c] = None;
                        }
                    }
                }
            }
        }
        v.dims = (w, h, d);
    }

    /// Switch every present volume channel's sampling filter without re-uploading.
    /// Cubic uses the GL linear filter (its 8-tap reconstruction is in-shader).
    /// No-op when nothing changed (this is called every 3D frame).
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
        for c in 0..MAX_CHANNELS {
            if let Some(kind) = self.volume.kinds[c] {
                let tex = if kind == VolumeKind::F32 { self.volume.tex_float[c] } else { self.volume.tex_unorm[c] };
                unsafe {
                    self.gl.bind_texture(glow::TEXTURE_3D, Some(tex));
                    set_volume_filter(&self.gl, gl_interp, kind);
                }
            }
        }
    }

    /// Stash the ray-march params (camera, per-channel window, steps); applied in
    /// `paint_volume`.
    pub fn set_volume_params(&mut self, params: VolumeParams) {
        self.volume.params = Some(params);
    }

    /// Ray-march the current volume, compositing every channel. Assumes the
    /// host has already set the viewport/scissor to the target rect.
    pub fn paint_volume(&self, gl: &glow::Context) {
        let v = &self.volume;
        let Some(p) = v.params else { return };
        unsafe {
            gl.use_program(Some(v.program));
            gl.bind_vertex_array(Some(self.vao));
            // Unorm channel textures → units 0..MAX_CHANNELS.
            for c in 0..MAX_CHANNELS {
                gl.active_texture(glow::TEXTURE0 + c as u32);
                gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_unorm[c]));
                gl.uniform_1_i32(v.u_vol[c].as_ref(), c as i32);
            }
            // LUT → unit MAX_CHANNELS.
            let lut_unit = MAX_CHANNELS as u32;
            gl.active_texture(glow::TEXTURE0 + lut_unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(v.u_lut.as_ref(), lut_unit as i32);
            // Float channel textures → units MAX_CHANNELS+1 .. 2*MAX_CHANNELS+1.
            for c in 0..MAX_CHANNELS {
                let unit = MAX_CHANNELS as u32 + 1 + c as u32;
                gl.active_texture(glow::TEXTURE0 + unit);
                gl.bind_texture(glow::TEXTURE_3D, Some(v.tex_float[c]));
                gl.uniform_1_i32(v.u_volf[c].as_ref(), unit as i32);
            }

            gl.uniform_3_f32(v.u_eye.as_ref(), p.eye[0], p.eye[1], p.eye[2]);
            gl.uniform_3_f32(v.u_forward.as_ref(), p.forward[0], p.forward[1], p.forward[2]);
            gl.uniform_3_f32(v.u_right.as_ref(), p.right[0], p.right[1], p.right[2]);
            gl.uniform_3_f32(v.u_up.as_ref(), p.up[0], p.up[1], p.up[2]);
            gl.uniform_1_f32(v.u_tan.as_ref(), p.tan_half_fov);
            gl.uniform_1_f32(v.u_aspect.as_ref(), p.aspect);
            gl.uniform_3_f32(v.u_box_he.as_ref(), p.box_he[0], p.box_he[1], p.box_he[2]);
            gl.uniform_2_f32_slice(v.u_window.as_ref(), &p.windows);
            gl.uniform_1_f32_slice(v.u_enabled.as_ref(), &p.enabled);
            gl.uniform_1_f32_slice(v.u_is_float.as_ref(), &p.is_float);
            gl.uniform_1_f32_slice(v.u_albedo.as_ref(), &p.albedo_t);
            gl.uniform_1_i32(v.u_num_channels.as_ref(), p.num_channels);
            gl.uniform_1_i32(v.u_mode.as_ref(), p.render_mode);
            gl.uniform_1_f32(v.u_density.as_ref(), p.density);
            gl.uniform_1_f32(v.u_iso.as_ref(), p.iso);
            gl.uniform_1_i32(v.u_interp.as_ref(), v.interp_mode);

            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
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
                let (fw, fh) = if want[c] == KIND_FLOAT { (width, height) } else { (1, 1) };
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
/// and a **zero border** on all three axes. `CLAMP_TO_BORDER` (border = 0)
/// rather than `CLAMP_TO_EDGE` means samples at/just outside a face fade to 0
/// instead of smearing the boundary voxel — no bright edge shell on MIP, and a
/// clean silhouette on the alpha DVR. `kind` is unused today but kept so a
/// future nearest-only fallback for unfilterable float formats stays a one-line
/// change.
unsafe fn set_volume_filter(gl: &glow::Context, gl_interp: i32, _kind: VolumeKind) {
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MIN_FILTER, gl_interp);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_MAG_FILTER, gl_interp);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_BORDER as i32);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_BORDER as i32);
    gl.tex_parameter_i32(glow::TEXTURE_3D, glow::TEXTURE_WRAP_R, glow::CLAMP_TO_BORDER as i32);
    gl.tex_parameter_f32_slice(glow::TEXTURE_3D, glow::TEXTURE_BORDER_COLOR, &[0.0, 0.0, 0.0, 0.0]);
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

// composite.frag.glsl - the 2D composite (glow backend): per-channel
// window/level -> LUT -> additive blend, with a minification box filter for
// clean zoom-out. The twin of `composite.wgsl`.
//
// Not a complete shader on its own: the `#version` line and the constants in
// capitals are written in ahead of it by `composite_glsl` in
// `scivis-render/src/shaders.rs`, from the Rust values the WGSL copy is given
// too.
//
// Each channel has both an integer sampler (`ch_tex[c]`, R8Uint/R16Uint) and a
// float sampler (`ch_ftex[c]`, R32F); `ch_params[c].w` selects which one
// carries this channel's data (the other is a 1x1 dummy). The window/level
// math is identical for both - only the sampled value's source and units
// differ. GLSL 1.40 indexes an array of samplers only with a constant, which
// is why the per-channel lines in `main` are unrolled.
//
// `ch_params[c]` is one channel of the uniform block both backends are
// handed, `uniforms::CompositeParams` in `scivis-render/src/uniforms.rs`
// (`ChannelParams` in `composite.wgsl`): the window's min and max, on (1.0)
// or off (0.0), and float (1.0) or integer (0.0). One vec4 a channel, so the
// glow backend sets every channel in one call, from the bytes the wgpu
// backend writes to its uniform buffer.
in vec2 v_uv;
out vec4 frag_color;

uniform usampler2D ch_tex[MAX_CHANNELS];
uniform sampler2D ch_ftex[MAX_CHANNELS];
uniform sampler2D lut_tex;   // one row per channel
uniform vec4 ch_params[MAX_CHANNELS];
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
    int n = int(clamp(ceil(max(texels.x, texels.y)), 1.0, MAX_MINIFY_TAPS));
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
    int n = int(clamp(ceil(max(texels.x, texels.y)), 1.0, MAX_MINIFY_TAPS));
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

vec3 apply_channel(float value, int idx, vec4 p) {
    float span = max(p.y - p.x, 1.0);
    float t = clamp((value - p.x) / span, 0.0, 1.0);
    vec3 color = texture(lut_tex, vec2(t, (float(idx) + 0.5) / float(MAX_CHANNELS))).rgb;
    return color * p.z;
}

void main() {
    vec2 uv = uv_offset + v_uv * uv_scale;
    vec2 fp = fwidth(uv);
    vec3 color = vec3(0.0);
    color += apply_channel(ch_params[0].w > 0.5 ? sample_flt(ch_ftex[0], uv, fp) : sample_int(ch_tex[0], uv, fp), 0, ch_params[0]);
    if (num_channels > 1) color += apply_channel(ch_params[1].w > 0.5 ? sample_flt(ch_ftex[1], uv, fp) : sample_int(ch_tex[1], uv, fp), 1, ch_params[1]);
    if (num_channels > 2) color += apply_channel(ch_params[2].w > 0.5 ? sample_flt(ch_ftex[2], uv, fp) : sample_int(ch_tex[2], uv, fp), 2, ch_params[2]);
    if (num_channels > 3) color += apply_channel(ch_params[3].w > 0.5 ? sample_flt(ch_ftex[3], uv, fp) : sample_int(ch_tex[3], uv, fp), 3, ch_params[3]);
    if (num_channels > 4) color += apply_channel(ch_params[4].w > 0.5 ? sample_flt(ch_ftex[4], uv, fp) : sample_int(ch_tex[4], uv, fp), 4, ch_params[4]);
    if (num_channels > 5) color += apply_channel(ch_params[5].w > 0.5 ? sample_flt(ch_ftex[5], uv, fp) : sample_int(ch_tex[5], uv, fp), 5, ch_params[5]);
    frag_color = vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0);
}

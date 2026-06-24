// composite.wgsl
//
// Renders up to MAX_CHANNELS planes of raw 16-bit sample data as a single
// composited image, reproducing ImageJ's display pipeline: per-channel
// window/level (contrast min/max) -> per-channel LUT lookup -> additive
// blend (clamped), matching ImageJ's "Composite" channel display mode.
//
// All per-pixel work happens here, on the GPU. The CPU side only uploads
// raw sample values — no contrast/color math happens on the CPU at all,
// which is what makes scrubbing through huge stacks fast: changing frames
// is just a texture upload, not a CPU-side image processing pass.

struct ChannelParams {
    min_max: vec2<f32>,
    enabled: f32,
    _pad: f32,
};

struct Params {
    channels: array<ChannelParams, 6>,
    // Maps the quad's 0..1 UV onto the visible sub-rectangle of the image:
    // sampled_uv = uv_offset + uv * uv_scale. With (0,0)/(1,1) the whole image
    // fills the quad; smaller values show a zoomed/panned sub-region.
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
    num_channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var ch0_tex: texture_2d<u32>;
@group(0) @binding(2) var ch1_tex: texture_2d<u32>;
@group(0) @binding(3) var ch2_tex: texture_2d<u32>;
@group(0) @binding(4) var ch3_tex: texture_2d<u32>;
@group(0) @binding(5) var ch4_tex: texture_2d<u32>;
@group(0) @binding(6) var ch5_tex: texture_2d<u32>;
@group(0) @binding(7) var lut_tex: texture_2d_array<f32>;
@group(0) @binding(8) var lut_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    // Two triangles covering clip space [-1,1]^2; the egui_wgpu callback
    // already scopes the viewport to the image's allocated rect (which we
    // size to match the image's aspect ratio on the CPU side), so a plain
    // fullscreen quad here is all that's needed.
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    );

    var out: VertexOut;
    out.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

fn load_texel(tex: texture_2d<u32>, uv: vec2<f32>, dims: vec2<f32>) -> f32 {
    let coord = clamp(vec2<i32>(uv * dims), vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
    return f32(textureLoad(tex, coord, 0).r);
}

// Returns the (averaged) sample value at `uv`. `footprint` is the UV-space size
// of one output pixel (from screen-space derivatives). When magnifying or at
// 1:1 the footprint is ≤ 1 texel, so it's a single crisp nearest read. When
// minifying (zoomed out), it box-filters an NxN grid spread across the texel
// footprint to anti-alias — the shimmer that plain nearest sampling produces on
// shrunk images. N is capped so the per-pixel cost stays bounded.
fn sample_channel(tex: texture_2d<u32>, uv: vec2<f32>, footprint: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(tex));
    let texels = footprint * dims; // source texels covered by this output pixel
    let n = i32(clamp(ceil(max(texels.x, texels.y)), 1.0, 4.0));
    if (n <= 1) {
        return load_texel(tex, uv, dims);
    }
    let fn_ = f32(n);
    var sum = 0.0;
    for (var i = 0; i < n; i = i + 1) {
        for (var j = 0; j < n; j = j + 1) {
            // Stratified offsets in [-0.5, 0.5] across the footprint.
            let o = (vec2<f32>(f32(i), f32(j)) + 0.5) / fn_ - 0.5;
            sum = sum + load_texel(tex, uv + o * footprint, dims);
        }
    }
    return sum / (fn_ * fn_);
}

fn apply_channel(value: f32, channel_index: i32, cp: ChannelParams) -> vec3<f32> {
    let span = max(cp.min_max.y - cp.min_max.x, 1.0);
    let t = clamp((value - cp.min_max.x) / span, 0.0, 1.0);
    let color = textureSample(lut_tex, lut_sampler, vec2<f32>(t, 0.5), channel_index).rgb;
    return color * cp.enabled;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);

    // Remap the quad UV onto the visible sub-region of the image (pan/zoom).
    let uv = params.uv_offset + in.uv * params.uv_scale;
    // UV-space size of one output pixel — drives the minification box filter.
    // Computed in uniform control flow (derivatives must not be in a branch).
    let footprint = fwidth(uv);

    color += apply_channel(sample_channel(ch0_tex, uv, footprint), 0, params.channels[0]);
    if (params.num_channels > 1u) {
        color += apply_channel(sample_channel(ch1_tex, uv, footprint), 1, params.channels[1]);
    }
    if (params.num_channels > 2u) {
        color += apply_channel(sample_channel(ch2_tex, uv, footprint), 2, params.channels[2]);
    }
    if (params.num_channels > 3u) {
        color += apply_channel(sample_channel(ch3_tex, uv, footprint), 3, params.channels[3]);
    }
    if (params.num_channels > 4u) {
        color += apply_channel(sample_channel(ch4_tex, uv, footprint), 4, params.channels[4]);
    }
    if (params.num_channels > 5u) {
        color += apply_channel(sample_channel(ch5_tex, uv, footprint), 5, params.channels[5]);
    }

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

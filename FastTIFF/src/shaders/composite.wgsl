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
    channels: array<ChannelParams, 4>,
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
@group(0) @binding(5) var lut_tex: texture_2d_array<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

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

fn sample_channel(tex: texture_2d<u32>, uv: vec2<f32>) -> u32 {
    let dims = textureDimensions(tex);
    let coord = vec2<i32>(uv * vec2<f32>(dims));
    let clamped = clamp(coord, vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
    return textureLoad(tex, clamped, 0).r;
}

fn apply_channel(raw: u32, channel_index: i32, cp: ChannelParams) -> vec3<f32> {
    let value = f32(raw);
    let span = max(cp.min_max.y - cp.min_max.x, 1.0);
    let t = clamp((value - cp.min_max.x) / span, 0.0, 1.0);
    let color = textureSample(lut_tex, lut_sampler, vec2<f32>(t, 0.5), channel_index).rgb;
    return color * cp.enabled;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var color = vec3<f32>(0.0, 0.0, 0.0);

    color += apply_channel(sample_channel(ch0_tex, in.uv), 0, params.channels[0]);
    if (params.num_channels > 1u) {
        color += apply_channel(sample_channel(ch1_tex, in.uv), 1, params.channels[1]);
    }
    if (params.num_channels > 2u) {
        color += apply_channel(sample_channel(ch2_tex, in.uv), 2, params.channels[2]);
    }
    if (params.num_channels > 3u) {
        color += apply_channel(sample_channel(ch3_tex, in.uv), 3, params.channels[3]);
    }

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

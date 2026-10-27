// blit.wgsl — scale a reduced-size render of the 3D view up to the view's own
// size (wgpu backend): one full-screen triangle sampling the render bilinearly.
//
// The 3D view renders at a quarter of its size per axis while it moves (a
// sixteenth of the rays), into a texture of its own; this draws that texture
// over the view's rect. Texture row 0 is the top, as the ray-march renders it.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // (0, 0), (2, 0), (0, 2): a triangle covering the whole target.
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}

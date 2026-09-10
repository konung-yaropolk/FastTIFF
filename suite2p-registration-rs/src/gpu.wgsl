// Radix-2 Stockham FFT and the phase-correlation multiply, on the device.
//
// Ported alongside suite2p's registration (GPL-3); see the crate root.
//
// Stockham rather than Cooley-Tukey: it writes to a separate output buffer each
// stage, so there is no bit-reversal permutation and no in-place hazard between
// workgroups. The cost is two buffers ping-ponged, which is the right trade on
// a device where a barrier across the whole dispatch is not available.

struct Params {
    // Length of the transform along the axis being processed.
    n: u32,
    // How many independent transforms are in flight (rows, or columns).
    batch: u32,
    // Half-size of the current butterfly stage.
    span: u32,
    // 0 forward, 1 inverse.
    inverse: u32,
    // Stride between consecutive elements of one transform, in complex numbers.
    stride: u32,
    // Stride between one transform and the next.
    batch_stride: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> p: Params;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// One Stockham butterfly stage.
@compute @workgroup_size(64)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.n / 2u;
    let total = half * p.batch;
    let i = gid.x;
    if (i >= total) { return; }

    let b = i / half;            // which transform
    let k = i % half;            // which butterfly
    let span = p.span;
    let j = k % span;            // position within the sub-transform
    let base = (k / span) * span;

    // Twiddle: exp(-2*pi*i*j/(2*span)), conjugated for the inverse.
    let sign = select(-1.0, 1.0, p.inverse == 1u);
    let ang = sign * 6.283185307179586 * f32(j) / f32(2u * span);
    let w = vec2<f32>(cos(ang), sin(ang));

    let in0 = b * p.batch_stride + (base + j) * p.stride;
    let in1 = b * p.batch_stride + (base + j + half) * p.stride;
    let a0 = src[in0];
    let a1 = cmul(src[in1], w);

    let o0 = b * p.batch_stride + (2u * base + j) * p.stride;
    let o1 = b * p.batch_stride + (2u * base + j + span) * p.stride;
    dst[o0] = a0 + a1;
    dst[o1] = a0 - a1;
}

struct CorrParams {
    len: u32,
    // 1 to divide by len (the inverse's normalisation).
    scale: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> reference: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> cp: CorrParams;

// Whiten this frame's spectrum and multiply by the reference's conjugate —
// the step that makes it phase correlation rather than plain correlation.
@compute @workgroup_size(64)
fn phase_multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cp.len) { return; }
    let v = data[i];
    let mag = length(v);
    let whitened = v / (1e-5 + mag);
    data[i] = cmul(whitened, reference[i]);
}

// Scale after the inverse transform.
@compute @workgroup_size(64)
fn normalise(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cp.len) { return; }
    data[i] = data[i] / f32(cp.len);
}

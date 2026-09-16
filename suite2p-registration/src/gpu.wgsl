// Phase correlation of a batch of frames, on the device.
//
// Ported alongside suite2p's registration (GPL-3); see the crate root.
//
// Four kernels, run in this order over every frame of a batch at once:
//
//   expand   raw samples -> clipped, tapered, offset complex plane
//   fft_line one whole axis of a radix-2 Stockham FFT, forward or inverse
//   phase    whiten, and multiply by the reference's prepared spectrum
//   crop     the (2*lcorr+1)^2 window around no-shift, scaled by 1/N
//
// `fft_line` runs twice forward and twice inverse (rows, then columns), so a
// batch is seven dispatches, one submission, and a read of the windows alone.
//
// `LONGEST_LINE` is substituted by the host before compiling: the longer side
// of the frame, which sizes the workgroup scratch below.
const LONGEST_LINE: u32 = __LONGEST_LINE__u;

// ---- inputs shared by the kernels -------------------------------------------

// The reference, one entry per pixel: its whitened, smoothed, conjugated
// spectrum, and the taper a frame is multiplied by and offset by.
struct RefPixel {
    cf: vec2<f32>,
    mul: f32,
    off: f32,
};

// The complex planes of every frame in the batch, frame after frame.
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> refs: array<RefPixel>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---- expand -------------------------------------------------------------------

struct Expand {
    pixels: u32,
    // 1 to clip to [lo, hi] first: suite2p's `norm_frames`.
    clip: u32,
    lo: f32,
    hi: f32,
};

@group(0) @binding(2) var<storage, read> raw: array<f32>;
@group(0) @binding(3) var<uniform> ex: Expand;

// x is the pixel, y the frame.
@compute @workgroup_size(256, 1, 1)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ex.pixels) { return; }
    let at = gid.y * ex.pixels + i;
    var v = raw[at];
    if (ex.clip == 1u) {
        v = min(max(v, ex.lo), ex.hi);
    }
    let r = refs[i];
    data[at] = vec2<f32>(v * r.mul + r.off, 0.0);
}

// ---- fft_line -----------------------------------------------------------------

struct Line {
    // Length of one line: lx for rows, ly for columns.
    n: u32,
    // How many lines a frame has along this axis.
    lines: u32,
    // Distance between consecutive samples of a line, and between lines.
    stride: u32,
    line_stride: u32,
    // 1.0 forward, -1.0 inverse: the sign of each twiddle's imaginary part.
    conjugate: f32,
    pixels: u32,
    // Workgroup x is line `(x + first) % all`. The last pass only transforms
    // the lines the crop will read, which start before zero and wrap round.
    first: u32,
    all: u32,
};

@group(0) @binding(4) var<uniform> ln: Line;
// exp(-i*pi*j/s) for every stage half-size s and j < s, stage s starting at
// index s - 1. Computed once on the host in double precision.
@group(0) @binding(5) var<storage, read> twiddles: array<vec2<f32>>;

// A workgroup is one line, and its invocations are the lanes that share it —
// as many as a stage has butterflies, up to 256. `PER` is how many butterflies,
// and twice as many samples, each lane owns: one at the common sizes, two for a
// 1024-sample line.
const PER: u32 = __PER__u;

// Two line-lengths of working space, shared by every lane of the workgroup.
//
// This is the whole point of the kernel. Stockham writes every stage to fresh
// storage, and doing that in the device's main memory — as the per-stage
// dispatches this replaced did — streamed the entire batch through it nine
// times per axis. Here the lanes read a line into this scratch together, run
// every stage's butterflies side by side within it, and write it back
// together: one read and one write of the batch per axis.
var<workgroup> scratch: array<vec2<f32>, __TWICE_LONGEST_LINE__>;

// The lanes must stay in step at every barrier, so nothing a lane does may
// change how many times it reaches one. Every loop below runs a fixed number of
// times, the same for all lanes, and work that falls outside the line is
// skipped with an `if` rather than by a shorter loop — which is also what the
// language's uniformity rules require of code around a barrier.
//
// workgroup_id.x is the line, workgroup_id.y the frame.
@compute @workgroup_size(__LANES__, 1, 1)
fn fft_line(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let n = ln.n;
    let half = n / 2u;
    // A workgroup past the last line still runs, and still meets every
    // barrier; it just touches nothing.
    let live = wid.x < ln.lines;
    let line = (wid.x + ln.first) % ln.all;
    let origin = wid.y * ln.pixels + line * ln.line_stride;

    let first = lane * 2u * PER;
    for (var o = 0u; o < 2u * PER; o++) {
        let i = first + o;
        if (live && i < n) {
            scratch[i] = data[origin + i * ln.stride];
        }
    }
    workgroupBarrier();

    var src = 0u;
    var span = 1u;
    loop {
        if (span >= n) { break; }
        let dst = LONGEST_LINE - src;
        let mask = span - 1u;
        for (var o = 0u; o < PER; o++) {
            let k = lane * PER + o;
            if (k < half) {
                let j = k & mask;
                let base = k - j;
                let t = twiddles[span - 1u + j];
                let w = vec2<f32>(t.x, t.y * ln.conjugate);
                let a0 = scratch[src + k];
                let a1 = cmul(scratch[src + k + half], w);
                scratch[dst + 2u * base + j] = a0 + a1;
                scratch[dst + 2u * base + j + span] = a0 - a1;
            }
        }
        workgroupBarrier();
        src = dst;
        span = span * 2u;
    }

    for (var o = 0u; o < 2u * PER; o++) {
        let i = first + o;
        if (live && i < n) {
            data[origin + i * ln.stride] = scratch[src + i];
        }
    }
}

// ---- phase --------------------------------------------------------------------

struct Phase {
    pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(6) var<uniform> ph: Phase;

// Whiten this frame's spectrum and multiply by the reference's — the step that
// makes it phase correlation rather than plain correlation. x pixel, y frame.
@compute @workgroup_size(256, 1, 1)
fn phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ph.pixels) { return; }
    let at = gid.y * ph.pixels + i;
    let v = data[at];
    data[at] = cmul(v / (1e-5 + length(v)), refs[i].cf);
}

// ---- crop ---------------------------------------------------------------------

struct Crop {
    ly: u32,
    lx: u32,
    lcorr: u32,
    width: u32,
    pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(7) var<uniform> cr: Crop;
@group(0) @binding(8) var<storage, read_write> windows: array<f32>;

// The wrapped corners nearest the origin, laid out so index `lcorr` is no-shift
// in each axis, with the inverse transform's 1/N applied. x is the position in
// the window, y the frame.
@compute @workgroup_size(256, 1, 1)
fn crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let area = cr.width * cr.width;
    let i = gid.x;
    if (i >= area) { return; }
    let y = (i / cr.width + cr.ly - cr.lcorr) % cr.ly;
    let x = (i % cr.width + cr.lx - cr.lcorr) % cr.lx;
    let scale = 1.0 / f32(cr.pixels);
    windows[gid.y * area + i] = data[gid.y * cr.pixels + y * cr.lx + x].x * scale;
}

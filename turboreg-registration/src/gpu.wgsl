// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

// TurboReg's criterion, for many frames in one dispatch.
//
// One workgroup scores one share of one lane's pixels: `workgroup_id.y` picks
// the lane (a frame of the batch), `workgroup_id.x` the share. Each invocation
// walks the lane's pixels with a stride of the whole lane's invocations,
// keeping its own running sums, and the workgroup then reduces those value by
// value through workgroup memory. One row of K partial sums per workgroup goes
// back to the host, which adds the rows in workgroup order in f64.
//
// Which pixel each invocation takes, and the shape of the reduction tree, are
// fixed by the image and the constants below — never by scheduling — so the
// result is the same on every run on the same device. It is f32 arithmetic
// rather than the CPU's f64, so it agrees with the CPU path to a tolerance, not
// bit for bit; see gpu.rs.
//
// __N__, __H__ and __K__ are substituted before compilation: the parameter
// count, the size of the curvature's upper triangle, and the values reduced
// per workgroup. Constants rather than runtime values so the per-invocation
// sums are fixed-size arrays the compiler can keep in registers.

const N: u32 = __N__u;
const H: u32 = __H__u;
const K: u32 = __K__u;
const WG: u32 = 256u;

struct Info {
    w: u32,
    h: u32,
    pixels: u32,
    groups: u32,
}

// One frame's candidate: where its coefficients are, whether derivatives are
// wanted, the transformation, and its eight derivative matrices.
struct Lane {
    slot: u32,
    derivatives: u32,
    pad0: u32,
    pad1: u32,
    m: array<f32, 8>,
    dm: array<f32, 64>,
}

@group(0) @binding(0) var<uniform> info: Info;
// Four planes per slot: the reference's samples, then the moving image's
// spline coefficients and those of its x and y gradients. The reference is per
// slot because each lane may be aligned to a different one — under the
// previous-frame reference, lane t is aligned to frame t - 1.
@group(0) @binding(1) var<storage, read> planes: array<f32>;
@group(0) @binding(2) var<storage, read> lanes: array<Lane>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> scratch: array<f32, 256>;

// The cubic B-spline weights for a fractional offset, in descending position:
// .x belongs to floor(x) + 2 and .w to floor(x) - 1. The same formula as
// `spline::weights`, with the second weight taken as the remainder so the four
// sum to one.
fn weights(f: f32) -> vec4<f32> {
    let s = 1.0 - f;
    let w3 = s * s * s / 6.0;
    let sq = f * f;
    let w2 = 2.0 / 3.0 - 0.5 * sq * (2.0 - f);
    let w0 = sq * f / 6.0;
    let w1 = 1.0 - w0 - w2 - w3;
    return vec4<f32>(w0, w1, w2, w3);
}

// An index reflected back inside 0..n: half-sample symmetric, period 2n — the
// boundary the prefilter assumed. `spline::indexes`, one tap at a time.
fn mirror(p: i32, n: i32) -> u32 {
    let twice = 2 * n;
    var q = p;
    if (q < 0) {
        q = -1 - q;
    }
    if (q >= twice) {
        q = q - twice * (q / twice);
    }
    if (q >= n) {
        q = twice - 1 - q;
    }
    return u32(q);
}

// The spline whose coefficients start at `base`, evaluated at (x, y).
fn interp(base: u32, x: f32, y: f32) -> f32 {
    let w = i32(info.w);
    let h = i32(info.h);
    // The index and the weights' offset split each coordinate at the same
    // floor, as `spline::indexes` and `spline::frac` do. (Truncation would
    // disagree with the floor at negative integers, reading one tap short.)
    let fx = floor(x);
    let fy = floor(y);
    let wx = weights(x - fx);
    let wy = weights(y - fy);
    let px = i32(fx) + 2;
    let py = i32(fy) + 2;
    let x0 = mirror(px, w);
    let x1 = mirror(px - 1, w);
    let x2 = mirror(px - 2, w);
    let x3 = mirror(px - 3, w);
    var sum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        let row = base + mirror(py - j, h) * info.w;
        let across = wx.x * planes[row + x0] + wx.y * planes[row + x1]
            + wx.z * planes[row + x2] + wx.w * planes[row + x3];
        sum = sum + wy[j] * across;
    }
    return sum;
}

@compute @workgroup_size(256)
fn evaluate(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let li = group.y;
    let derivs = lanes[li].derivatives != 0u;
    let m = lanes[li].m;
    // The slot's reference, then its image, x-gradient and y-gradient planes.
    let base = lanes[li].slot * 4u * info.pixels;
    let image = base + info.pixels;

    var sum_sq = 0.0;
    var grad: array<f32, N>;
    var hess: array<f32, H>;

    let stride = info.groups * WG;
    for (var idx = group.x * WG + lid; idx < info.pixels; idx = idx + stride) {
        let x = f32(idx % info.w);
        let y = f32(idx / info.w);
        let qx = m[0] + m[1] * x + m[2] * y + m[3] * x * y;
        let qy = m[4] + m[5] * x + m[6] * y + m[7] * x * y;
        let diff = planes[base + idx] - interp(image, qx, qy);
        sum_sq = sum_sq + diff * diff;
        if (derivs) {
            let gx = interp(image + info.pixels, qx, qy);
            let gy = interp(image + 2u * info.pixels, qx, qy);
            // d(residual)/d(p_k) through the sampling position; the minus is
            // the residual's own sign, as on the CPU.
            var dk: array<f32, N>;
            for (var k = 0u; k < N; k = k + 1u) {
                let o = k * 8u;
                let vx = lanes[li].dm[o] + lanes[li].dm[o + 1u] * x
                    + lanes[li].dm[o + 2u] * y + lanes[li].dm[o + 3u] * x * y;
                let vy = lanes[li].dm[o + 4u] + lanes[li].dm[o + 5u] * x
                    + lanes[li].dm[o + 6u] * y + lanes[li].dm[o + 7u] * x * y;
                dk[k] = -(gx * vx + gy * vy);
            }
            var t = 0u;
            for (var a = 0u; a < N; a = a + 1u) {
                grad[a] = grad[a] - diff * dk[a];
                for (var b = a; b < N; b = b + 1u) {
                    hess[t] = hess[t] + dk[a] * dk[b];
                    t = t + 1u;
                }
            }
        }
    }

    // Reduce each of the K values across the workgroup, one at a time through
    // a 256-float scratch, so the workgroup memory needed is 1 KiB whatever the
    // model — well inside what every WebGPU device must provide.
    let out = (li * info.groups + group.x) * K;
    for (var v = 0u; v < K; v = v + 1u) {
        var mine = 0.0;
        if (v == 0u) {
            mine = sum_sq;
        } else if (v <= N) {
            mine = grad[v - 1u];
        } else {
            mine = hess[v - 1u - N];
        }
        scratch[lid] = mine;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                scratch[lid] = scratch[lid] + scratch[lid + s];
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            partials[out + v] = scratch[0];
        }
        workgroupBarrier();
    }
}

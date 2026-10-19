// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.
//
// The algorithm is Thevenaz, Ruttimann & Unser's, as published in 1998 and
// shipped by EPFL as TurboReg/StackReg (GPL-3-or-later since April 2023). See
// the crate manifest for the full licensing note and the citation.

//! Cubic B-splines: turning samples into coefficients, and reading them back at
//! fractional positions.
//!
//! Everything the registration does to pixels goes through here. The image is
//! never treated as a grid of values to be averaged; it is treated as a
//! continuous function represented by B-spline coefficients, and "the value at
//! (x, y)" means that function evaluated there. That is what makes the
//! registration subpixel at all, and it is why a prefiltering step exists:
//! the coefficients of the spline that *interpolates* the samples are not the
//! samples.
//!
//! # Why the storage is `f32` when the arithmetic is `f64`
//!
//! Each separable pass computes a line in `f64` and stores the result as `f32`,
//! which looks like a needless loss and is not. The reference implementation
//! does exactly this — its 2-D buffers are `float[]` and only the 1-D line
//! buffer is `double` — and the quantization is *load-bearing*: the optimizer
//! below branches on whether a step improved the criterion, so a difference in
//! the last bits of a coefficient does not stay in the last bits of the answer,
//! it changes which steps are accepted and therefore where the run lands.
//! Keeping `f64` throughout would be more accurate and would not agree with
//! StackReg, and agreeing is the point of a port.
//!
//! It also halves the memory of every pyramid level, which for a viewer holding
//! a stack as well is not nothing.

/// The single pole of the cubic (degree 3) B-spline prefilter.
///
/// `sqrt(3) - 2`, written out rather than computed so the constant cannot drift
/// with a change of `const fn` support.
pub const POLES_3: [f64; 1] = [-0.267_949_192_431_122_7];

/// The three poles of the degree-7 prefilter.
///
/// Used only inside the pyramid's least-squares reduction, never for sampling.
pub const POLES_7: [f64; 3] = [
    -0.535_280_430_796_438_2,
    -0.122_554_615_192_326_69,
    -0.009_148_694_809_608_277,
];

/// Turn samples into interpolation coefficients, in place, along one line.
///
/// The recursive filter of Unser et al.: scale by the gain, then run a causal
/// and an anti-causal first-order recursion per pole. `line` is both input and
/// output.
///
/// The boundary model is **mirror off bounds** — half-sample symmetric, period
/// `2N`, `c[-1-k] == c[k]`. Not the whole-sample mirror that the same family of
/// filters is more often written with: the two differ by half a pixel at the
/// edge, which is a visible seam on a registration that moves the image toward
/// one.
pub fn samples_to_coefficients_1d(line: &mut [f64], poles: &[f64]) {
    let n = line.len();
    // A line of one sample is its own coefficient; the recursions below would
    // index past it.
    if n < 2 {
        return;
    }
    // Overall gain, applied once before the recursions.
    let mut gain = 1.0f64;
    for &z in poles {
        gain *= (1.0 - z) * (1.0 - 1.0 / z);
    }
    for v in line.iter_mut() {
        *v *= gain;
    }
    for &z in poles {
        // Causal.
        line[0] = initial_causal(line, z);
        for i in 1..n {
            line[i] += z * line[i - 1];
        }
        // Anti-causal.
        line[n - 1] = initial_anticausal(line, z);
        for i in (0..n - 1).rev() {
            line[i] = z * (line[i + 1] - line[i]);
        }
    }
}

/// The causal recursion's starting value under the mirror-off-bounds boundary.
///
/// The closed form of the infinite sum the periodic extension implies, summed
/// in full rather than truncated at a tolerance — the reference uses a
/// tolerance of zero, so there is no horizon to cut at.
fn initial_causal(c: &[f64], z: f64) -> f64 {
    let n = c.len();
    let zn = z.powi(n as i32);
    let mut sum = (1.0 + z) * (c[0] + zn * c[n - 1]);
    let mut z1 = z;
    // `z^(2n)`, which the loop then walks back down.
    let mut zk = zn * zn;
    for &v in c.iter().take(n.saturating_sub(1)).skip(1) {
        z1 *= z;
        zk /= z;
        sum += (z1 + zk) * v;
    }
    sum / (1.0 - zn * zn)
}

/// The anti-causal recursion's starting value, same boundary model.
fn initial_anticausal(c: &[f64], z: f64) -> f64 {
    z * c[c.len() - 1] / (z - 1.0)
}

/// Prefilter a whole image into coefficients: every row, then every column.
///
/// Separable, which is exact rather than an approximation — the 2-D cubic
/// B-spline is the product of two 1-D ones.
///
/// Stored back as `f32` after each pass, for the reason in the module header.
///
/// On one thread; [`prefilter_2d_with`] is the same arithmetic on every core.
pub fn prefilter_2d(img: &mut [f32], w: usize, h: usize, poles: &[f64]) {
    prefilter_2d_with(img, w, h, poles, false);
}

/// [`prefilter_2d`], optionally on every core.
///
/// Every row is independent of every other, and so is every column, so each
/// pass parallelises with nothing to combine. The column pass goes through a
/// transposed copy: a column of a row-major image is strided, and handing
/// threads disjoint *columns* to write would mean handing them interleaved
/// slices of one buffer. Transposed, each column is a contiguous row and the
/// pass is the row pass again.
///
/// The arithmetic is unchanged by any of this — each line is read from the
/// `f32` store into an `f64` line, filtered, and stored back as `f32`, exactly
/// as before — so the result is bit-identical with `parallel` on or off.
pub fn prefilter_2d_with(img: &mut [f32], w: usize, h: usize, poles: &[f64], parallel: bool) {
    if w == 0 || h == 0 {
        return;
    }
    let filter_line = |row: &mut [f32]| {
        let mut line: Vec<f64> = row.iter().map(|&v| v as f64).collect();
        samples_to_coefficients_1d(&mut line, poles);
        for (o, &v) in row.iter_mut().zip(line.iter()) {
            *o = v as f32;
        }
    };
    // Rows.
    crate::par::for_each_chunk_mut(img, w, parallel, |_, row| filter_line(row));
    // Columns, as the rows of a transposed copy.
    let mut t = vec![0.0f32; w * h];
    {
        let src: &[f32] = img;
        crate::par::for_each_chunk_mut(&mut t, h, parallel, |x, col| {
            for (y, o) in col.iter_mut().enumerate() {
                *o = src[y * w + x];
            }
            filter_line(col);
        });
    }
    crate::par::for_each_chunk_mut(img, w, parallel, |y, row| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = t[x * h + y];
        }
    });
}

/// The four cubic B-spline weights for a fractional offset in `0..1`.
///
/// Returned in the reference's own order, which is **descending** in position:
/// `w[0]` belongs to the sample at `floor(x) + 2` and `w[3]` to the one at
/// `floor(x) - 1`. Pairing them the other way round is a plausible-looking
/// mistake that shifts every interpolated value by three pixels.
///
/// `w[1]` is computed as the remainder rather than from its own polynomial, so
/// the four weights sum to exactly 1 in floating point and a flat image
/// interpolates to its own value everywhere.
pub fn weights(frac: f64) -> [f64; 4] {
    let s = 1.0 - frac;
    let w3 = s * s * s / 6.0;
    let sq = frac * frac;
    let w2 = 2.0 / 3.0 - 0.5 * sq * (2.0 - frac);
    let w0 = sq * frac / 6.0;
    let w1 = 1.0 - w0 - w2 - w3;
    [w0, w1, w2, w3]
}

/// The four sample indices a position reads, mirrored back inside `0..n`.
///
/// Same descending order as [`weights`]. The mirror is the one the prefilter
/// assumed: period `2n`, reflecting without repeating the edge sample.
pub fn indexes(x: f64, n: usize) -> [usize; 4] {
    debug_assert!(n > 0, "an axis of no samples has no indices");
    let twice = 2 * n as i64;
    // `floor(x) + 2`, from the true floor — the same floor [`frac`] takes the
    // weights' offset from, which is the whole requirement: the index and the
    // offset must split `x` the same way.
    //
    // This used to be `trunc(x) + 2`, or `trunc(x) + 1` below zero, which is
    // the reference's spelling of the same thing. It is only the same thing
    // where `x` is not a negative integer. At one, truncation *is* the floor, so
    // the `+ 1` lands a tap short while the weights still come from the floor:
    // at x = -44.0 the spline was read at -45. The reference does not have that
    // problem only because it also takes the offset from the truncation; this
    // code took it from the floor and kept the truncated index. Every integer
    // shift toward the left or top edge read the wrong pixel for the samples it
    // sent past that edge.
    let mut p = x.floor() as i64 + 2;
    let mut out = [0usize; 4];
    for slot in out.iter_mut() {
        let mut q = if p < 0 { -1 - p } else { p };
        if twice <= q {
            q -= twice * (q / twice);
        }
        *slot = if n as i64 <= q {
            (twice - 1 - q) as usize
        } else {
            q as usize
        };
        p -= 1;
    }
    out
}

/// The fractional part of a coordinate, floored rather than truncated.
///
/// `-0.25` has a fractional part of `0.75` here, not `-0.25`: the weights are
/// defined on `0..1` and a negative offset would evaluate the spline off its
/// own support.
pub fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// Evaluate the spline at `(x, y)` from its coefficients.
///
/// `coeff` is `w * h`, row-major, as [`prefilter_2d`] leaves it.
pub fn interpolate(coeff: &[f32], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let xw = weights(frac(x));
    let yw = weights(frac(y));
    let xi = indexes(x, w);
    let yi = indexes(y, h);
    let mut sum = 0.0;
    for (j, &wy) in yw.iter().enumerate() {
        let row = yi[j] * w;
        let mut acc = 0.0;
        for (i, &wx) in xw.iter().enumerate() {
            acc += wx * coeff[row + xi[i]] as f64;
        }
        sum += wy * acc;
    }
    sum
}

#[cfg(test)]
#[path = "spline_tests.rs"]
mod tests;

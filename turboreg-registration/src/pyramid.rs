// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The multiresolution pyramid the search runs on.
//!
//! # Why coarse-to-fine at all
//!
//! The thing being minimised — squared intensity difference as a function of
//! the transformation — is not convex and is not smooth. On a textured image it
//! has a local minimum roughly every time one piece of texture lines up with
//! the next, so a gradient method started more than a few pixels away finds the
//! wrong one and stops there, confidently.
//!
//! Halving the image halves the distance in pixels to the true answer and
//! smooths the landscape, and doing it repeatedly makes the coarsest level's
//! problem small enough to be nearly convex. The answer found there is scaled
//! up and used as the starting point one level down, where it is already close
//! enough that the local minimum is the right one. That is the whole idea of
//! the paper this comes from, and it is why a registration that fails on a
//! large displacement usually just needs a deeper pyramid.
//!
//! # The reduction filter
//!
//! Not a Gaussian blur and not a plain average: the kernel is `(1, 4, 6, 4, 1)/16`
//! sampled at every second position, which is the least-squares-optimal
//! projection of a cubic spline onto the half-resolution spline space. Its gain
//! is exactly one, so a flat image keeps its value rather than drifting with
//! depth — which matters because the criterion is compared against a tolerance
//! in intensity units.

use crate::spline;

/// The smallest side a level may have.
///
/// Below this the image has fewer samples than the spline's own support in each
/// direction and the reduction stops meaning anything; it is also well past the
/// point where a further level tells the search anything new.
pub const MIN_SIZE: usize = 12;

/// One level of the pyramid: samples at that scale.
#[derive(Clone, Debug)]
pub struct Level {
    pub w: usize,
    pub h: usize,
    /// Samples, row-major, `w * h`.
    pub img: Vec<f32>,
    /// How many times this level has been halved relative to the original.
    /// Level 0 is full resolution.
    pub octave: usize,
}

impl Level {
    /// The scale factor from this level's coordinates to full resolution.
    pub fn scale(&self) -> f64 {
        (1u32 << self.octave) as f64
    }
}

/// How many levels an image of this size supports, counting full resolution.
///
/// `cap` of zero means "as deep as the image allows", which is the reference's
/// behaviour.
pub fn depth_for(w: usize, h: usize, cap: usize) -> usize {
    let mut depth = 1;
    let (mut w, mut h) = (w, h);
    // The cap is tested *before* adding a level, not after: checking it at the
    // end of the body still adds the level that breaks the loop, so a cap of 1
    // would return 2 — which is the one value a caller setting a cap of 1 most
    // obviously does not want.
    while (cap == 0 || depth < cap) && w / 2 >= MIN_SIZE && h / 2 >= MIN_SIZE {
        w /= 2;
        h /= 2;
        depth += 1;
    }
    depth
}

/// Reflect an index back inside `0..n`, half-sample symmetric with period `2n`.
///
/// The same boundary model the prefilter assumes — `c[-1] == c[0]`, reflecting
/// *between* samples rather than about them. Mixing the two conventions puts a
/// half-pixel step at every edge of every level, which compounds with depth and
/// reads as the registration pulling toward a border.
fn mirror(i: isize, n: usize) -> usize {
    debug_assert!(n > 0);
    let n = n as isize;
    let period = 2 * n;
    let mut q = if i < 0 { -1 - i } else { i };
    if q >= period {
        q %= period;
    }
    (if q >= n { period - 1 - q } else { q }) as usize
}

/// Halve one line with the `(1, 4, 6, 4, 1)/16` kernel.
///
/// Output length is `n / 2`, so an odd input drops its last half-sample — which
/// is the reference's behaviour and the reason a level's dimensions are integer
/// halves rather than rounded up.
fn reduce_1d(src: &[f64], out: &mut Vec<f64>) {
    const H: [f64; 3] = [6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let n = src.len();
    out.clear();
    if n < 2 {
        out.extend_from_slice(src);
        return;
    }
    for j in 0..n / 2 {
        let centre = 2 * j as isize;
        let mut acc = H[0] * src[mirror(centre, n)];
        for k in 1..3 {
            acc +=
                H[k] * (src[mirror(centre - k as isize, n)] + src[mirror(centre + k as isize, n)]);
        }
        out.push(acc);
    }
}

/// Halve an image in both axes: rows first, then the columns of that result.
pub fn reduce_2d(img: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (hw, hh) = (w / 2, h / 2);
    if hw == 0 || hh == 0 {
        return (img.to_vec(), w, h);
    }
    // Rows.
    let mut mid = vec![0.0f32; hw * h];
    let mut line: Vec<f64> = Vec::with_capacity(w);
    let mut out: Vec<f64> = Vec::with_capacity(hw);
    for y in 0..h {
        line.clear();
        line.extend(img[y * w..(y + 1) * w].iter().map(|&v| v as f64));
        reduce_1d(&line, &mut out);
        for (x, &v) in out.iter().enumerate() {
            mid[y * hw + x] = v as f32;
        }
    }
    // Columns.
    let mut dst = vec![0.0f32; hw * hh];
    for x in 0..hw {
        line.clear();
        line.extend((0..h).map(|y| mid[y * hw + x] as f64));
        reduce_1d(&line, &mut out);
        for (y, &v) in out.iter().enumerate() {
            dst[y * hw + x] = v as f32;
        }
    }
    (dst, hw, hh)
}

/// Build the pyramid, finest first.
///
/// Level 0 is the image as given. Each subsequent level is the one before it,
/// halved. The search walks this list backwards.
pub fn build(img: &[f32], w: usize, h: usize, depth: usize) -> Vec<Level> {
    let mut levels = Vec::with_capacity(depth.max(1));
    levels.push(Level {
        w,
        h,
        img: img.to_vec(),
        octave: 0,
    });
    for octave in 1..depth.max(1) {
        let last = &levels[octave - 1];
        let (img, w, h) = reduce_2d(&last.img, last.w, last.h);
        levels.push(Level { w, h, img, octave });
    }
    levels
}

/// A level's samples as spline coefficients, ready for [`spline::interpolate`].
pub fn coefficients(level: &Level) -> Vec<f32> {
    coefficients_with(level, false)
}

/// [`coefficients`], optionally on every core. Bit-identical either way.
pub fn coefficients_with(level: &Level, parallel: bool) -> Vec<f32> {
    let mut c = level.img.clone();
    spline::prefilter_2d_with(&mut c, level.w, level.h, &spline::POLES_3, parallel);
    c
}

/// The spline's partial derivatives, sampled on the grid.
///
/// The optimiser needs `d(image)/dx` and `d(image)/dy` at every pixel of the
/// level it is working on, and it needs them to be the derivatives of the
/// *same* continuous function it is interpolating — not a finite difference of
/// the samples, which is the derivative of a different function and disagrees
/// with it by enough to send the search in a slightly wrong direction on every
/// step.
///
/// So they are taken from the spline: the derivative of a cubic B-spline
/// expansion is evaluated by the same separable sum with one axis's weights
/// replaced by the derivative weights. Here that is done by evaluating the
/// interpolant a small step either side, which for a cubic is exact to the
/// precision the `f32` storage keeps anyway and needs no second weight table.
pub fn gradients(level: &Level) -> (Vec<f32>, Vec<f32>) {
    gradients_from(&coefficients(level), level.w, level.h, false)
}

/// [`gradients`] from coefficients the caller already has, optionally on every
/// core.
///
/// Taking the coefficients rather than the level is what lets the optimiser
/// prefilter a level once: it needs the same coefficients for sampling the
/// image, and building them a second time inside here was one full prefilter
/// per level per frame spent on nothing. Each output pixel depends only on the
/// coefficients, so rows are independent and the result is bit-identical with
/// `parallel` on or off.
pub fn gradients_from(coeff: &[f32], w: usize, h: usize, parallel: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return (gx, gy);
    }
    // A half-pixel step: the cubic is a polynomial over each unit cell, and a
    // central difference at this spacing differs from the analytic derivative
    // by a term in the third derivative that the f32 storage cannot hold.
    const STEP: f64 = 0.5;
    crate::par::for_each_chunk_pair_mut(&mut gx, &mut gy, w, parallel, |y, rx, ry| {
        let yf = y as f64;
        for (x, (ox, oy)) in rx.iter_mut().zip(ry.iter_mut()).enumerate() {
            let xf = x as f64;
            let right = spline::interpolate(coeff, w, h, xf + STEP, yf);
            let left = spline::interpolate(coeff, w, h, xf - STEP, yf);
            let down = spline::interpolate(coeff, w, h, xf, yf + STEP);
            let up = spline::interpolate(coeff, w, h, xf, yf - STEP);
            *ox = ((right - left) / (2.0 * STEP)) as f32;
            *oy = ((down - up) / (2.0 * STEP)) as f32;
        }
    });
    (gx, gy)
}

#[cfg(test)]
#[path = "pyramid_tests.rs"]
mod tests;

// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.
//
// Ported from suite2p (https://github.com/MouseLand/suite2p), Copyright © 2023
// Howard Hughes Medical Institute, authored by Carsen Stringer and Marius
// Pachitariu, and licensed GPL-3.

//! Non-rigid registration: a shift per *block*, not per frame.
//!
//! Tissue does not only slide. A brain under a cranial window breathes, and the
//! surface stretches and compresses unevenly — one corner of the field moves
//! two pixels while the opposite corner moves five. A single whole-frame shift
//! cannot express that, so it splits the difference and leaves both corners
//! wrong.
//!
//! So the field is divided into overlapping blocks, each block is
//! phase-correlated on its own, and the per-block shifts are interpolated back
//! to a shift per *pixel*. The frame is then warped rather than moved.
//!
//! # Three details that are not obvious
//!
//! * **Blocks overlap by half.** `calculate_nblocks` asks for
//!   `ceil(1.5 * L / block)` blocks of `block` pixels, which is a 1/3 overlap —
//!   without it the interpolated shift field would have a crease at every
//!   block boundary.
//! * **A block that cannot see anything borrows from its neighbours.** A block
//!   over empty tissue has no peak worth trusting. `snr_thresh` measures how
//!   much better the best peak is than the next one, and a block that fails is
//!   replaced by a neighbourhood-smoothed correlation map — up to twice —
//!   before its peak is taken.
//! * **The peak is refined to sub-pixel.** A `2*lpad+1` window around the
//!   integer peak is upsampled `subpixel`-fold by kriging (a Gaussian-process
//!   interpolation, `kernelD` below), and the peak of *that* is the answer.
//!   Integer block shifts would quantise the warp into visible tiles.

use crate::fft::Fft2;
use crate::masks::{gaussian_fft, spatial_taper};
use crate::rigid::Shift;
use rustfft::num_complex::Complex32;

/// How wide the kriging window around a peak is, in pixels. suite2p's `lpad`.
pub const LPAD: usize = 3;

/// The block size and count along one axis.
///
/// A block at least as big as the frame means one block — there is nothing to
/// divide. Otherwise `ceil(1.5 * L / block)` blocks, which is what makes them
/// overlap rather than tile.
pub fn calculate_nblocks(l: usize, block_size: usize) -> (usize, usize) {
    if block_size >= l {
        (l, 1)
    } else {
        (
            block_size,
            (1.5 * l as f64 / block_size as f64).ceil() as usize,
        )
    }
}

/// One block's extent in the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub y0: usize,
    pub y1: usize,
    pub x0: usize,
    pub x1: usize,
}

impl Block {
    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }
    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }
    /// The block's centre, which is where its shift is treated as measured.
    pub fn centre(&self) -> (f64, f64) {
        (
            (self.y0 + self.y1) as f64 / 2.0,
            (self.x0 + self.x1) as f64 / 2.0,
        )
    }
}

/// The block grid over a frame.
#[derive(Clone, Debug)]
pub struct Blocks {
    /// Row-major: block `iy * nx + ix`.
    pub blocks: Vec<Block>,
    pub ny: usize,
    pub nx: usize,
    /// The effective block size, which is the requested one clamped to the
    /// frame.
    pub size: [usize; 2],
    /// Smoothing over the block grid — `nb * nb`, row-stochastic. Applied to the
    /// correlation maps of blocks that could not see anything on their own.
    pub smoother: Vec<f32>,
    /// The kriging upsampling matrix, `(2*lpad+1)^2` by `nup^2`.
    pub kmat: Vec<f32>,
    /// How many upsampled positions the window becomes along one axis.
    pub nup: usize,
}

/// Divide a frame into overlapping blocks.
pub fn make_blocks(ly: usize, lx: usize, block_size: [usize; 2], subpixel: usize) -> Blocks {
    let (by, ny) = calculate_nblocks(ly, block_size[0].max(1));
    let (bx, nx) = calculate_nblocks(lx, block_size[1].max(1));

    // `np.linspace(0, L - block, n)`: evenly spaced starts, first at 0 and last
    // flush with the far edge, so the blocks cover the frame exactly.
    let starts = |l: usize, b: usize, n: usize| -> Vec<usize> {
        if n <= 1 {
            return vec![0];
        }
        let span = (l - b) as f64;
        (0..n)
            .map(|i| (span * i as f64 / (n - 1) as f64) as usize)
            .collect()
    };
    let ys = starts(ly, by, ny);
    let xs = starts(lx, bx, nx);

    let mut blocks = Vec::with_capacity(ny * nx);
    for &y0 in &ys {
        for &x0 in &xs {
            blocks.push(Block {
                y0,
                y1: y0 + by,
                x0,
                x1: x0 + bx,
            });
        }
    }

    let (kmat, nup) = mat_upsample(LPAD, subpixel.max(1));
    Blocks {
        smoother: block_smoother(ny, nx),
        blocks,
        ny,
        nx,
        size: [by, bx],
        kmat,
        nup,
    }
}

/// suite2p's `kernelD2`: a Gaussian over the block grid, normalised so each
/// column sums to one.
///
/// Used to replace a block's correlation map with a weighted average of its
/// neighbours'. Normalising means the smoothing cannot change the overall
/// scale, only where the mass sits — a block that borrows from its neighbours
/// gets their *shape*, not their brightness.
fn block_smoother(ny: usize, nx: usize) -> Vec<f32> {
    let nb = ny * nx;
    let coord = |i: usize| ((i / nx) as f64, (i % nx) as f64);
    let mut r = vec![0.0f64; nb * nb];
    for i in 0..nb {
        let (yi, xi) = coord(i);
        for j in 0..nb {
            let (yj, xj) = coord(j);
            let d = (yi - yj).powi(2) + (xi - xj).powi(2);
            r[i * nb + j] = (-d).exp();
        }
    }
    // Column sums, matching `R / sum(R, axis=0)`.
    for j in 0..nb {
        let s: f64 = (0..nb).map(|i| r[i * nb + j]).sum();
        if s > 0.0 {
            for i in 0..nb {
                r[i * nb + j] /= s;
            }
        }
    }
    // suite2p transposes the result, so the multiply below is `row · map`.
    let mut out = vec![0.0f32; nb * nb];
    for i in 0..nb {
        for j in 0..nb {
            out[i * nb + j] = r[j * nb + i] as f32;
        }
    }
    out
}

/// suite2p's `kernelD`: a Gaussian kernel between two coordinate sets, over the
/// 2D grid each of them spans.
fn kernel_d(xs: &[f64], ys: &[f64], sig: f64) -> Vec<f64> {
    // Each set is a 1D axis spanning a square grid, so the kernel is between
    // `xs.len()^2` and `ys.len()^2` positions.
    let (n, m) = (xs.len(), ys.len());
    let mut k = vec![0.0f64; n * n * m * m];
    for a0 in 0..n {
        for a1 in 0..n {
            let row = a0 * n + a1;
            for b0 in 0..m {
                for b1 in 0..m {
                    let d = (xs[a0] - ys[b0]).powi(2) + (xs[a1] - ys[b1]).powi(2);
                    k[row * (m * m) + b0 * m + b1] = (-d / (2.0 * sig * sig)).exp();
                }
            }
        }
    }
    k
}

/// The kriging matrix that lifts a `(2*lpad+1)^2` window to `nup^2` positions.
///
/// `solve(kernel0, kernel_up)` in suite2p — a Gaussian-process interpolation,
/// which is what gives a peak location between samples rather than at one.
fn mat_upsample(lpad: usize, subpixel: usize) -> (Vec<f32>, usize) {
    let xs: Vec<f64> = (0..=2 * lpad).map(|i| i as f64 - lpad as f64).collect();
    let step = 1.0 / subpixel as f64;
    let mut up = Vec::new();
    let mut v = -(lpad as f64);
    while v <= lpad as f64 + 1e-3 {
        up.push(v);
        v += step;
    }
    let n = xs.len() * xs.len();
    let nup = up.len();
    let k0 = kernel_d(&xs, &xs, 0.85);
    let kup = kernel_d(&xs, &up, 0.85);

    // Solve `k0 * X = kup` by Gauss-Jordan. `k0` is a small, symmetric,
    // positive-definite Gaussian kernel matrix — 49x49 for lpad 3 — so a plain
    // elimination with partial pivoting is both fast enough and stable.
    let m = nup * nup;
    let mut a = k0;
    let mut b = kup;
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].abs() < 1e-12 {
            continue;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            for c in 0..m {
                b.swap(col * m + c, pivot * m + c);
            }
        }
        let d = a[col * n + col];
        for c in 0..n {
            a[col * n + c] /= d;
        }
        for c in 0..m {
            b[col * m + c] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            for c in 0..m {
                b[r * m + c] -= f * b[col * m + c];
            }
        }
    }
    (b.into_iter().map(|v| v as f32).collect(), nup)
}

/// One block's reference: the taper, the offset and the whitened spectrum.
pub struct BlockFilters {
    pub mask_mul: Vec<f32>,
    pub mask_offset: Vec<f32>,
    pub cf_ref: Vec<Complex32>,
    pub fft: Fft2,
}

/// Prepare every block of a reference image.
pub fn block_filters(
    reference: &[f32],
    lx: usize,
    blocks: &Blocks,
    mask_slope: f64,
    smooth_sigma: f64,
) -> Vec<BlockFilters> {
    blocks
        .blocks
        .iter()
        .map(|b| {
            let (bh, bw) = (b.height(), b.width());
            let mut fft = Fft2::new(bh, bw);
            let patch: Vec<f32> = (b.y0..b.y1)
                .flat_map(|y| (b.x0..b.x1).map(move |x| (y, x)))
                .map(|(y, x)| reference[y * lx + x])
                .collect();

            let mask_mul = spatial_taper(mask_slope, bh, bw);
            let mean = patch.iter().map(|&v| v as f64).sum::<f64>() / patch.len().max(1) as f64;
            let mask_offset: Vec<f32> = mask_mul
                .iter()
                .map(|&m| (mean as f32) * (1.0 - m))
                .collect();

            let mut cf: Vec<Complex32> = patch.iter().map(|&v| Complex32::new(v, 0.0)).collect();
            fft.forward(&mut cf);
            for c in cf.iter_mut() {
                *c = c.conj();
                *c /= 1e-5 + c.norm();
            }
            let g = gaussian_fft(&mut fft, smooth_sigma, bh, bw);
            for (c, &g) in cf.iter_mut().zip(&g) {
                *c *= g;
            }
            BlockFilters {
                mask_mul,
                mask_offset,
                cf_ref: cf,
                fft,
            }
        })
        .collect()
}

/// How much better a correlation peak is than the next-best one elsewhere.
///
/// suite2p's `getSNR`. A block over featureless tissue has a peak barely above
/// the noise, and its "shift" is wherever the noise happened to be highest —
/// this is what catches that.
fn snr_of(cc: &[f32], lcorr: usize, lpad: usize) -> f32 {
    let n = 2 * lcorr + 2 * lpad + 1;
    // The best inside the searched region.
    let mut best = f32::NEG_INFINITY;
    let mut at = (0usize, 0usize);
    for iy in lpad..n - lpad {
        for ix in lpad..n - lpad {
            let v = cc[iy * n + ix];
            if v > best {
                best = v;
                at = (iy - lpad, ix - lpad);
            }
        }
    }
    // The best *elsewhere*: the same map with a `2*lpad` box around the peak
    // knocked out.
    let mut other = f32::NEG_INFINITY;
    for iy in 0..n {
        for ix in 0..n {
            let inside = iy >= at.0 && iy < at.0 + 2 * lpad && ix >= at.1 && ix < at.1 + 2 * lpad;
            if !inside {
                other = other.max(cc[iy * n + ix]);
            }
        }
    }
    best / other.max(1e-10)
}

/// The correlation map for one block of one frame, `(2*lcorr+2*lpad+1)^2`.
fn block_map(
    filters: &mut BlockFilters,
    frame: &[f32],
    lx: usize,
    block: &Block,
    lcorr: usize,
    clip: Option<(f32, f32)>,
) -> Vec<f32> {
    let (bh, bw) = (block.height(), block.width());
    let mut buf: Vec<Complex32> = Vec::with_capacity(bh * bw);
    for y in block.y0..block.y1 {
        for x in block.x0..block.x1 {
            let i = (y - block.y0) * bw + (x - block.x0);
            let v = frame[y * lx + x];
            let v = match clip {
                Some((lo, hi)) => v.clamp(lo, hi),
                None => v,
            };
            buf.push(Complex32::new(
                v * filters.mask_mul[i] + filters.mask_offset[i],
                0.0,
            ));
        }
    }
    filters.fft.forward(&mut buf);
    for (c, r) in buf.iter_mut().zip(&filters.cf_ref) {
        *c /= 1e-5 + c.norm();
        *c *= r;
    }
    filters.fft.inverse(&mut buf);

    let half = lcorr + LPAD;
    let n = 2 * half + 1;
    let mut cc = vec![0.0f32; n * n];
    for iy in 0..n {
        let y = (iy + bh - half) % bh;
        for ix in 0..n {
            let x = (ix + bw - half) % bw;
            cc[iy * n + ix] = buf[y * bw + x].re;
        }
    }
    cc
}

/// A per-block shift, in fractional pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockShift {
    pub dy: f32,
    pub dx: f32,
    pub corr: f32,
}

/// How a block field is measured. The knobs, gathered so the call is readable.
#[derive(Clone, Copy, Debug)]
pub struct BlockSearch {
    /// `maxregshiftNR`: how far a block may move on top of the rigid shift.
    pub maxregshift_nr: f64,
    /// `snr_thresh`: below this, a block borrows its neighbours' map.
    pub snr_thresh: f64,
    /// `subpixel`: the peak is resolved to `1/subpixel` of a pixel.
    pub subpixel: usize,
    /// The range frames are clipped to, when `norm_frames` is on.
    pub clip: Option<(f32, f32)>,
}

/// Measure every block of one frame, on top of a rigid shift already applied.
pub fn measure_blocks(
    filters: &mut [BlockFilters],
    frame: &[f32],
    lx: usize,
    blocks: &Blocks,
    search: &BlockSearch,
) -> Vec<BlockShift> {
    let BlockSearch {
        maxregshift_nr,
        snr_thresh,
        subpixel,
        clip,
    } = *search;
    let (bh, bw) = (blocks.size[0], blocks.size[1]);
    let lcorr = (maxregshift_nr.round() as usize)
        .min((bh.min(bw) / 2).saturating_sub(LPAD))
        .max(1);
    let nb = blocks.blocks.len();
    let n = 2 * lcorr + 2 * LPAD + 1;

    let mut maps: Vec<Vec<f32>> = blocks
        .blocks
        .iter()
        .zip(filters.iter_mut())
        .map(|(b, f)| block_map(f, frame, lx, b, lcorr, clip))
        .collect();

    // Blocks that could not see anything borrow their neighbours' shape — once,
    // then twice, and no further. suite2p stops at two smoothings because a
    // block still failing after that is not going to be rescued, and smoothing
    // it again would just spread one bad estimate across the field.
    if snr_thresh > 1.0 && nb > 1 {
        let mut smoothed = maps.clone();
        for round in 0..2 {
            let failing: Vec<usize> = (0..nb)
                .filter(|&i| (snr_of(&maps[i], lcorr, LPAD) as f64) < snr_thresh)
                .collect();
            if failing.is_empty() {
                break;
            }
            let source = if round == 0 { &maps } else { &smoothed.clone() };
            let mut next = vec![vec![0.0f32; n * n]; nb];
            for (i, slot) in next.iter_mut().enumerate() {
                for (j, map) in source.iter().enumerate().take(nb) {
                    let w = blocks.smoother[i * nb + j];
                    if w == 0.0 {
                        continue;
                    }
                    for (o, v) in slot.iter_mut().zip(map) {
                        *o += w * v;
                    }
                }
            }
            smoothed = next;
            for &i in &failing {
                maps[i] = smoothed[i].clone();
            }
        }
    }

    maps.iter()
        .map(|cc| refine_peak(cc, lcorr, blocks, subpixel))
        .collect()
}

/// The integer peak, then its kriging-refined sub-pixel position.
fn refine_peak(cc: &[f32], lcorr: usize, blocks: &Blocks, subpixel: usize) -> BlockShift {
    let n = 2 * lcorr + 2 * LPAD + 1;
    let mut best = (0usize, 0usize, f32::NEG_INFINITY);
    for iy in LPAD..n - LPAD {
        for ix in LPAD..n - LPAD {
            let v = cc[iy * n + ix];
            if v > best.2 {
                best = (iy - LPAD, ix - LPAD, v);
            }
        }
    }

    // The `(2*lpad+1)^2` window around it, lifted to `nup^2` by the kriging
    // matrix; the peak of that is the sub-pixel answer.
    let w = 2 * LPAD + 1;
    let mut window = vec![0.0f32; w * w];
    for iy in 0..w {
        for ix in 0..w {
            window[iy * w + ix] = cc[(best.0 + iy) * n + best.1 + ix];
        }
    }
    let nup = blocks.nup;
    let mut up = vec![0.0f32; nup * nup];
    for (k, slot) in up.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (j, &v) in window.iter().enumerate() {
            acc += v * blocks.kmat[j * (nup * nup) + k];
        }
        *slot = acc;
    }
    let mut top = (0usize, f32::NEG_INFINITY);
    for (i, &v) in up.iter().enumerate() {
        if v > top.1 {
            top = (i, v);
        }
    }
    let mid = (nup / 2) as f32;
    let sub = subpixel.max(1) as f32;
    BlockShift {
        dy: ((top.0 / nup) as f32 - mid) / sub + best.0 as f32 - lcorr as f32,
        dx: ((top.0 % nup) as f32 - mid) / sub + best.1 as f32 - lcorr as f32,
        corr: top.1,
    }
}

/// Warp a frame by a field of per-block shifts.
///
/// The block shifts are interpolated bilinearly to a shift per pixel, and each
/// output pixel is sampled bilinearly from where that says it came from. Two
/// interpolations, both linear, because the alternative — nearest neighbour —
/// would quantise the warp back into the block grid it is trying to smooth out.
///
/// Out-of-frame samples clamp to the edge rather than wrapping. Unlike the
/// rigid shift, a non-rigid warp moves each part of the frame differently, so a
/// wrap would bring the far side of the picture into the middle of it.
pub fn warp(
    frame: &[f32],
    ly: usize,
    lx: usize,
    blocks: &Blocks,
    shifts: &[BlockShift],
    rigid: Shift,
) -> Vec<f32> {
    let (ny, nx) = (blocks.ny, blocks.nx);
    // Block centres, which is where each shift is treated as measured.
    let ys: Vec<f64> = (0..ny)
        .map(|iy| blocks.blocks[iy * nx].centre().0)
        .collect();
    let xs: Vec<f64> = (0..nx).map(|ix| blocks.blocks[ix].centre().1).collect();

    // Bilinear lookup into the block grid, clamped at the edges — a pixel
    // outside the outermost block centres takes that block's shift.
    let at = |v: f64, axis: &[f64]| -> (usize, usize, f64) {
        if axis.len() == 1 {
            return (0, 0, 0.0);
        }
        if v <= axis[0] {
            return (0, 0, 0.0);
        }
        if v >= axis[axis.len() - 1] {
            let i = axis.len() - 1;
            return (i, i, 0.0);
        }
        let mut i = 0;
        while i + 1 < axis.len() && axis[i + 1] < v {
            i += 1;
        }
        let t = (v - axis[i]) / (axis[i + 1] - axis[i]).max(1e-9);
        (i, i + 1, t)
    };

    let mut out = vec![0.0f32; ly * lx];
    for y in 0..ly {
        let (iy0, iy1, ty) = at(y as f64, &ys);
        for x in 0..lx {
            let (ix0, ix1, tx) = at(x as f64, &xs);
            let s = |iy: usize, ix: usize| shifts[iy * nx + ix];
            let (a, b, c, d) = (s(iy0, ix0), s(iy0, ix1), s(iy1, ix0), s(iy1, ix1));
            let lerp = |p: f32, q: f32, t: f64| p as f64 * (1.0 - t) + q as f64 * t;
            let dy = lerp(lerp(a.dy, b.dy, tx) as f32, lerp(c.dy, d.dy, tx) as f32, ty);
            let dx = lerp(lerp(a.dx, b.dx, tx) as f32, lerp(c.dx, d.dx, tx) as f32, ty);

            // Where this output pixel came from: the rigid shift plus this
            // pixel's own share of the deformation.
            let sy = y as f64 + dy + rigid.dy as f64;
            let sx = x as f64 + dx + rigid.dx as f64;
            out[y * lx + x] = sample(frame, ly, lx, sy, sx);
        }
    }
    out
}

/// Bilinear sample, clamped to the frame.
fn sample(frame: &[f32], ly: usize, lx: usize, y: f64, x: f64) -> f32 {
    let y = y.clamp(0.0, (ly - 1) as f64);
    let x = x.clamp(0.0, (lx - 1) as f64);
    let (y0, x0) = (y.floor() as usize, x.floor() as usize);
    let (y1, x1) = ((y0 + 1).min(ly - 1), (x0 + 1).min(lx - 1));
    let (ty, tx) = (y - y0 as f64, x - x0 as f64);
    let v = |yy: usize, xx: usize| frame[yy * lx + xx] as f64;
    let top = v(y0, x0) * (1.0 - tx) + v(y0, x1) * tx;
    let bot = v(y1, x0) * (1.0 - tx) + v(y1, x1) * tx;
    (top * (1.0 - ty) + bot * ty) as f32
}

#[cfg(test)]
#[path = "nonrigid_tests.rs"]
mod tests;

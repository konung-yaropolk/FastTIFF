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

//! Two-dimensional FFT over a row-major plane.
//!
//! Everything the registration needs is a 2D transform of a real plane and its
//! inverse, so this wraps `rustfft`'s 1D engine in the usual row-then-column
//! form and caches the plans. A plan is expensive to build and cheap to reuse,
//! and a timelapse transforms the same size thousands of times.
//!
//! # Convention
//!
//! Unnormalised forward, `1/N` on the inverse — NumPy's convention, which is
//! what suite2p is written against. Splitting the scale differently would
//! change nothing about where the correlation peak *is*, but it would change
//! `cmax`, and `cmax` is what picks the frames that build the reference.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

/// Cached forward and inverse plans for one plane size.
pub struct Fft2 {
    pub ly: usize,
    pub lx: usize,
    fwd_row: Arc<dyn Fft<f32>>,
    fwd_col: Arc<dyn Fft<f32>>,
    inv_row: Arc<dyn Fft<f32>>,
    inv_col: Arc<dyn Fft<f32>>,
    /// Scratch for the column pass, so a transform allocates nothing.
    col: Vec<Complex32>,
}

impl Fft2 {
    pub fn new(ly: usize, lx: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Fft2 {
            ly,
            lx,
            fwd_row: planner.plan_fft_forward(lx),
            fwd_col: planner.plan_fft_forward(ly),
            inv_row: planner.plan_fft_inverse(lx),
            inv_col: planner.plan_fft_inverse(ly),
            col: vec![Complex32::new(0.0, 0.0); ly],
        }
    }

    pub fn len(&self) -> usize {
        self.ly * self.lx
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forward transform, in place. `data` is row-major `ly * lx`.
    pub fn forward(&mut self, data: &mut [Complex32]) {
        debug_assert_eq!(data.len(), self.len());
        for row in data.chunks_exact_mut(self.lx) {
            self.fwd_row.process(row);
        }
        self.columns(data, true);
    }

    /// Inverse transform, in place, scaled by `1/(ly*lx)`.
    pub fn inverse(&mut self, data: &mut [Complex32]) {
        debug_assert_eq!(data.len(), self.len());
        for row in data.chunks_exact_mut(self.lx) {
            self.inv_row.process(row);
        }
        self.columns(data, false);
        let scale = 1.0 / self.len() as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
    }

    /// The column pass: gather a column into scratch, transform, scatter back.
    ///
    /// A strided transform rather than a transpose. Two transposes of a
    /// megapixel plane per frame is real time on a timelapse, and the gather is
    /// the same memory traffic without the second copy.
    fn columns(&mut self, data: &mut [Complex32], forward: bool) {
        let lx = self.lx;
        for x in 0..lx {
            for (y, slot) in self.col.iter_mut().enumerate() {
                *slot = data[y * lx + x];
            }
            if forward {
                self.fwd_col.process(&mut self.col);
            } else {
                self.inv_col.process(&mut self.col);
            }
            for (y, slot) in self.col.iter().enumerate() {
                data[y * lx + x] = *slot;
            }
        }
    }
}

/// Move the zero-frequency component from the corner to the centre.
///
/// `numpy.fft.fftshift`, for even and odd sizes alike: the split is at
/// `n - n/2`, which is `n/2` when `n` is even and `(n+1)/2` when it is odd.
pub fn fftshift(data: &[f32], ly: usize, lx: usize) -> Vec<f32> {
    let (sy, sx) = (ly - ly / 2, lx - lx / 2);
    let mut out = vec![0.0; ly * lx];
    for y in 0..ly {
        for x in 0..lx {
            out[((y + sy) % ly) * lx + (x + sx) % lx] = data[y * lx + x];
        }
    }
    out
}

/// The inverse of [`fftshift`] — centre back to corner.
pub fn ifftshift(data: &[f32], ly: usize, lx: usize) -> Vec<f32> {
    let (sy, sx) = (ly / 2, lx / 2);
    let mut out = vec![0.0; ly * lx];
    for y in 0..ly {
        for x in 0..lx {
            out[((y + sy) % ly) * lx + (x + sx) % lx] = data[y * lx + x];
        }
    }
    out
}

#[cfg(test)]
#[path = "fft_tests.rs"]
mod tests;

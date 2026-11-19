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

/// Columns gathered and transformed together in the column pass. Eight
/// `Complex32` are one 64-byte cache line, so each row of the plane is read a
/// line at a time instead of a sample at a time.
const COLUMN_TILE: usize = 8;

/// Forward and inverse plans for one plane size: the expensive, shareable part
/// of a transform.
///
/// Plans hold no state a transform writes to, and they are `Send + Sync`, so
/// one set serves every worker of a run. Building them per worker — which
/// bundling them with the scratch below forced — replanned a Bluestein size
/// (1024 x 768, say) once per chunk, and made the multi-threaded path hand out
/// work in chunks of at least eight frames, which on a short batch of large
/// frames left most cores idle.
#[derive(Clone)]
pub struct FftPlans2 {
    /// Rows in the plane these plans are for.
    pub ly: usize,
    /// Columns in it.
    pub lx: usize,
    fwd_row: Arc<dyn Fft<f32>>,
    fwd_col: Arc<dyn Fft<f32>>,
    inv_row: Arc<dyn Fft<f32>>,
    inv_col: Arc<dyn Fft<f32>>,
}

impl FftPlans2 {
    /// Plan both transforms for an `ly * lx` plane. Expensive to build and
    /// cheap to reuse — a timelapse transforms the same size thousands of
    /// times, so one set is kept for the run.
    pub fn new(ly: usize, lx: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        FftPlans2 {
            ly,
            lx,
            fwd_row: planner.plan_fft_forward(lx),
            fwd_col: planner.plan_fft_forward(ly),
            inv_row: planner.plan_fft_inverse(lx),
            inv_col: planner.plan_fft_inverse(ly),
        }
    }
}

/// Plans plus the scratch one worker transforms with — what a worker keeps.
pub struct Fft2 {
    /// Rows in the plane these plans are for.
    pub ly: usize,
    /// Columns in it.
    pub lx: usize,
    plans: FftPlans2,
    /// Up to [`COLUMN_TILE`] columns, gathered one after another.
    col: Vec<Complex32>,
    /// What the plans need to run in place. Held rather than allocated per
    /// call: `Fft::process` allocates its scratch every time it is called,
    /// which in a row-by-row loop was one allocation per row of every frame.
    scratch: Vec<Complex32>,
}

impl Fft2 {
    /// Plan both transforms for an `ly * lx` plane, with scratch to run them.
    /// See [`FftPlans2::new`]; workers sharing a run should share its plans
    /// through [`with_plans`](Self::with_plans) instead.
    pub fn new(ly: usize, lx: usize) -> Self {
        Self::with_plans(&FftPlans2::new(ly, lx))
    }

    /// A transform over shared `plans`, with scratch of its own. Cheap: the
    /// plans are reference-counted, and the scratch is a column tile and what
    /// the plans ask for.
    pub fn with_plans(plans: &FftPlans2) -> Self {
        let scratch = [
            &plans.fwd_row,
            &plans.fwd_col,
            &plans.inv_row,
            &plans.inv_col,
        ]
        .iter()
        .map(|p| p.get_inplace_scratch_len())
        .max()
        .unwrap_or(0);
        let zero = Complex32::new(0.0, 0.0);
        Fft2 {
            ly: plans.ly,
            lx: plans.lx,
            plans: plans.clone(),
            col: vec![zero; COLUMN_TILE * plans.ly],
            scratch: vec![zero; scratch],
        }
    }

    /// The plans this transform runs, to share with another worker.
    pub fn plans(&self) -> &FftPlans2 {
        &self.plans
    }

    /// Samples in a plane: `ly * lx`.
    pub fn len(&self) -> usize {
        self.ly * self.lx
    }

    /// Whether the plane has no samples at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forward transform, in place. `data` is row-major `ly * lx`.
    pub fn forward(&mut self, data: &mut [Complex32]) {
        debug_assert_eq!(data.len(), self.len());
        // Every row in one call: the plan transforms each `lx` run of the
        // buffer in turn.
        if !data.is_empty() {
            self.plans
                .fwd_row
                .process_with_scratch(data, &mut self.scratch);
        }
        self.columns(data, true);
    }

    /// Inverse transform, in place, scaled by `1/(ly*lx)`.
    pub fn inverse(&mut self, data: &mut [Complex32]) {
        debug_assert_eq!(data.len(), self.len());
        if !data.is_empty() {
            self.plans
                .inv_row
                .process_with_scratch(data, &mut self.scratch);
        }
        self.columns(data, false);
        let scale = 1.0 / self.len() as f32;
        for v in data.iter_mut() {
            *v *= scale;
        }
    }

    /// The column pass: gather a tile of columns into scratch, transform
    /// them in one call, scatter them back.
    ///
    /// A strided transform rather than a transpose. Two transposes of a
    /// megapixel plane per frame is real time on a timelapse, and the gather is
    /// the same memory traffic without the second copy. A tile of
    /// [`COLUMN_TILE`] columns at a time reads each row a cache line at a time,
    /// where one column at a time read a single sample from every line.
    fn columns(&mut self, data: &mut [Complex32], forward: bool) {
        let (ly, lx) = (self.ly, self.lx);
        if ly == 0 || lx == 0 {
            return;
        }
        let plan = if forward {
            &self.plans.fwd_col
        } else {
            &self.plans.inv_col
        };
        for x0 in (0..lx).step_by(COLUMN_TILE) {
            let tile = COLUMN_TILE.min(lx - x0);
            let col = &mut self.col[..tile * ly];
            for y in 0..ly {
                let row = &data[y * lx + x0..y * lx + x0 + tile];
                for (c, &v) in row.iter().enumerate() {
                    col[c * ly + y] = v;
                }
            }
            plan.process_with_scratch(col, &mut self.scratch);
            for y in 0..ly {
                let row = &mut data[y * lx + x0..y * lx + x0 + tile];
                for (c, v) in row.iter_mut().enumerate() {
                    *v = col[c * ly + y];
                }
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

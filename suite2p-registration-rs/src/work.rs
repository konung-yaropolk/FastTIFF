// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! How long the stages of a registration take, relative to each other.
//!
//! For one purpose only: dividing a progress bar so that it moves at the speed
//! the work does. None of this affects a result.
//!
//! A bar split by stage rather than by cost is wrong in the most visible way.
//! The reference pick is a single step, but it correlates every sampled frame
//! against every other and is a third of a run; the pass over the recording is
//! thousands of steps and, rigid, a tenth of one. Splitting the bar by stage
//! made it freeze for seconds and then race.
//!
//! The unit is one whole-frame dot product in [`crate::pick_initial_reference`].
//! Every figure was measured on 512 x 512 frames with the multi-thread backend
//! on sixteen threads (`examples/profile.rs`, and the plugin run end to end),
//! then divided by the dot product's 48 microseconds. They are ratios, so a
//! faster or slower machine moves them all together; frame size shifts them a
//! little, since an FFT grows slightly faster than a dot product does.

/// One frame's dot product with another, in the reference pick.
pub const PAIR: f64 = 1.0;
/// Reading one plane off the file and converting it to `f32`.
pub const READ: f64 = 4.0;
/// Phase-correlating one frame against the reference and finding its peak.
pub const CORRELATION: f64 = 24.0;
/// Moving one plane by a whole-pixel shift, and storing it.
pub const SHIFT: f64 = 12.0;
/// Measuring one frame's deformation, block by block.
pub const BLOCKS: f64 = 98.0;
/// Warping one plane by a deformation field, and storing it.
pub const WARP: f64 = 47.0;

/// The reference pick for a sample of `n` frames: every pair, once.
pub fn pick(n: usize) -> f64 {
    let n = n as f64;
    n * (n + 1.0) / 2.0 * PAIR
}

/// The whole of [`crate::compute_reference`] for a sample of `n` frames: the
/// pick, then `iterations` passes that correlate and shift every frame.
pub fn reference(n: usize, iterations: usize) -> f64 {
    pick(n) + iterations.max(1) as f64 * n as f64 * (CORRELATION + SHIFT)
}

#[cfg(test)]
#[path = "work_tests.rs"]
mod tests;

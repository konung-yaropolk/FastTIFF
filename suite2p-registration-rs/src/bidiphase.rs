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

//! Bidirectional scan-phase correction.
//!
//! A resonant scanner sweeps left-to-right on one line and right-to-left on the
//! next. If the two sweeps are not perfectly aligned in time, alternate lines
//! land a pixel or two apart and the picture combs — every edge splits into a
//! zigzag. It is not motion, it is not noise, and no amount of frame
//! registration fixes it, because it is *within* each frame.
//!
//! The offset is found by phase-correlating the odd lines against the even ones
//! along x, and undone by sliding the odd lines back.

use rustfft::num_complex::Complex32;

/// The bidirectional offset, in pixels, over a sample of frames.
///
/// Positive means the odd lines sit to the *left* of the even ones and need
/// moving right. Searched over ±10 pixels, as suite2p does — a scanner further
/// out than that is misconfigured rather than drifting.
pub fn compute(frames: &[Vec<f32>], ly: usize, lx: usize) -> i32 {
    if frames.is_empty() || ly < 2 || lx == 0 {
        return 0;
    }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(lx);
    let inv = planner.plan_fft_inverse(lx);

    // Accumulated over every line pair of every frame: one line pair is far too
    // noisy to read an offset off.
    let mut acc = vec![0.0f64; lx];
    let mut pairs = 0usize;
    let mut odd = vec![Complex32::new(0.0, 0.0); lx];
    let mut even = vec![Complex32::new(0.0, 0.0); lx];

    for frame in frames {
        // Odd lines are 1, 3, 5...; the even line paired with each is the one
        // above it. `d2` is truncated to the odd count in suite2p, which is the
        // same as pairing (1,0), (3,2), ... and stopping when either runs out.
        let n_pairs = (ly - 1).div_ceil(2);
        for p in 0..n_pairs {
            let (yo, ye) = (2 * p + 1, 2 * p);
            if yo >= ly {
                break;
            }
            for x in 0..lx {
                odd[x] = Complex32::new(frame[yo * lx + x], 0.0);
                even[x] = Complex32::new(frame[ye * lx + x], 0.0);
            }
            fwd.process(&mut odd);
            fwd.process(&mut even);
            for (o, e) in odd.iter_mut().zip(even.iter()) {
                // Whiten both, conjugate the even one: phase correlation along
                // a single line.
                *o /= 1e-5 + o.norm();
                *o *= e.conj() / (1e-5 + e.norm());
            }
            inv.process(&mut odd);
            let scale = 1.0 / lx as f32;
            for (a, o) in acc.iter_mut().zip(odd.iter()) {
                *a += (o.re * scale) as f64;
            }
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0;
    }
    for v in acc.iter_mut() {
        *v /= pairs as f64;
    }

    // Centre the correlation, then take the peak within ±10 of no offset.
    let shift = lx - lx / 2;
    let centred: Vec<f64> = (0..lx).map(|x| acc[(x + lx - shift) % lx]).collect();
    let mid = lx / 2;
    let lo = mid.saturating_sub(10);
    let hi = (mid + 11).min(lx);
    let mut best = (lo, f64::NEG_INFINITY);
    for (i, &v) in centred.iter().enumerate().take(hi).skip(lo) {
        if v > best.1 {
            best = (i, v);
        }
    }
    -((best.0 as i32) - mid as i32)
}

/// Slide the odd lines of a frame by `offset`, in place.
///
/// The vacated columns keep what was already there, which is what suite2p's
/// slice assignment does — it copies a window and leaves the rest alone rather
/// than blanking it.
pub fn shift(frame: &mut [f32], ly: usize, lx: usize, offset: i32) {
    if offset == 0 || lx == 0 {
        return;
    }
    let n = offset.unsigned_abs() as usize;
    if n >= lx {
        return;
    }
    let mut y = 1;
    while y < ly {
        let row = &mut frame[y * lx..(y + 1) * lx];
        if offset > 0 {
            // row[n..] = row[..lx-n]
            row.copy_within(0..lx - n, n);
        } else {
            // row[..lx-n] = row[n..]
            row.copy_within(n..lx, 0);
        }
        y += 2;
    }
}

#[cfg(test)]
#[path = "bidiphase_tests.rs"]
mod tests;

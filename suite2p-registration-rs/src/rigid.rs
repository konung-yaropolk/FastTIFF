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

//! Rigid registration: one whole-frame shift per frame, by phase correlation.

use crate::fft::Fft2;
use crate::masks::RefFilters;
use rustfft::num_complex::Complex32;

/// What a frame's correlation against the reference found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shift {
    /// Rows to move the frame by to bring it onto the reference.
    pub dy: i32,
    /// Columns, likewise.
    pub dx: i32,
    /// The correlation peak. Used to rank frames when building a reference —
    /// bigger is a better match — and to spot frames that matched nothing.
    pub corr: f32,
}

/// The half-width of the search window, for a frame and a `maxregshift`.
pub fn lcorr_for(ly: usize, lx: usize, max_shift: f64) -> usize {
    let min_dim = ly.min(lx);
    ((max_shift * min_dim as f64).round() as usize).min(min_dim / 2)
}

/// The correlation window for one frame: `(2*lcorr+1)^2` values, row-major,
/// with no-shift at the centre.
///
/// Split out from [`phase_correlate`] because `smooth_sigma_time` smooths these
/// maps *along time* before the peak is taken — a frame too dim to locate on
/// its own can still be located by where its neighbours agree it is. Taking the
/// peak first and smoothing the answers afterwards is a different and much
/// worse thing: it would average two confident disagreements into a shift that
/// matches neither.
pub fn correlation_map(
    fft: &mut Fft2,
    filters: &RefFilters,
    frame: &[f32],
    max_shift: f64,
    clip: Option<(f32, f32)>,
) -> Vec<f32> {
    let (ly, lx) = (fft.ly, fft.lx);
    let lcorr = lcorr_for(ly, lx, max_shift);

    let mut buf: Vec<Complex32> = frame
        .iter()
        .zip(&filters.mask_mul)
        .zip(&filters.mask_offset)
        .map(|((&v, &m), &o)| {
            // `norm_frames`: clipped to the reference's own range, so one
            // bright speck cannot drag the correlation peak to itself.
            let v = match clip {
                Some((lo, hi)) => v.clamp(lo, hi),
                None => v,
            };
            Complex32::new(v * m + o, 0.0)
        })
        .collect();

    fft.forward(&mut buf);
    for (c, r) in buf.iter_mut().zip(&filters.cf_ref) {
        *c /= 1e-5 + c.norm();
        *c *= r;
    }
    fft.inverse(&mut buf);

    // The wrapped corners nearest the origin, laid out so index `lcorr` is
    // no-shift in each axis.
    let n = 2 * lcorr + 1;
    let mut cc = vec![0.0f32; n * n];
    for iy in 0..n {
        let y = (iy + ly - lcorr) % ly;
        for ix in 0..n {
            let x = (ix + lx - lcorr) % lx;
            cc[iy * n + ix] = buf[y * lx + x].re;
        }
    }
    cc
}

/// The best shift in a correlation window.
pub fn peak_of(cc: &[f32], lcorr: usize) -> Shift {
    let n = 2 * lcorr + 1;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in cc.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    Shift {
        dy: (best.0 / n) as i32 - lcorr as i32,
        dx: (best.0 % n) as i32 - lcorr as i32,
        corr: best.1,
    }
}

/// Phase-correlate one frame against a prepared reference.
///
/// `max_shift` is a fraction of the *smaller* frame dimension, as suite2p's
/// `maxregshift` is: 0.1 of a 512-pixel frame is 51 pixels either way.
pub fn phase_correlate(
    fft: &mut Fft2,
    filters: &RefFilters,
    frame: &[f32],
    max_shift: f64,
) -> Shift {
    let lcorr = lcorr_for(fft.ly, fft.lx, max_shift);
    let cc = correlation_map(fft, filters, frame, max_shift, None);
    peak_of(&cc, lcorr)
}

/// Move a frame by `(dy, dx)`, wrapping at the edges.
///
/// Wrapping rather than filling, because that is what suite2p does (`np.roll`)
/// and because the alternative — a border of zeros or of edge pixels — invents
/// values that later analysis would average in as if they were measurements.
/// A wrapped strip is obviously wrong to a reader; a zero strip is not.
///
/// The sign matches suite2p: a frame found at `+dy` is rolled by `-dy` to bring
/// it back onto the reference.
pub fn shift_frame(frame: &[f32], ly: usize, lx: usize, dy: i32, dx: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; ly * lx];
    let wrap = |v: i32, n: usize| -> usize { v.rem_euclid(n as i32) as usize };
    for y in 0..ly {
        let sy = wrap(y as i32 + dy, ly);
        for x in 0..lx {
            let sx = wrap(x as i32 + dx, lx);
            out[y * lx + x] = frame[sy * lx + sx];
        }
    }
    out
}

#[cfg(test)]
#[path = "rigid_tests.rs"]
mod tests;

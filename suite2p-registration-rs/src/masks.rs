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

//! The masks and filters phase correlation is run through.
//!
//! Three things happen to a frame before it is correlated, and all three matter
//! more than they look:
//!
//! * a **spatial taper** fades the borders out, because an FFT wraps and an
//!   un-tapered edge correlates with the opposite edge;
//! * an **offset** puts the reference's mean back where the taper took the
//!   signal away, so the faded border is flat rather than dark;
//! * the reference's spectrum is **whitened** and then **smoothed** by a
//!   Gaussian, which is what makes it phase correlation rather than plain cross
//!   correlation — whitening throws away how bright a feature is and keeps only
//!   where it is.

use crate::fft::{ifftshift, Fft2};
use rustfft::num_complex::Complex32;

/// A sigmoid taper that fades the border of a `ly * lx` plane to zero.
///
/// `sig` is the slope: bigger fades a wider border. The half-way point sits
/// `2*sig` inside the edge, which is suite2p's choice and is why
/// `spatial_taper` defaults to well over `3 * smooth_sigma` — the taper has to
/// be wider than the smoothing or the smoothing reaches into the wrap.
pub fn spatial_taper(sig: f64, ly: usize, lx: usize) -> Vec<f32> {
    let axis = |n: usize| -> Vec<f64> {
        let mean = (n as f64 - 1.0) / 2.0;
        let m = ((n as f64 - 1.0) / 2.0) - 2.0 * sig;
        (0..n)
            .map(|i| {
                let d = (i as f64 - mean).abs();
                1.0 / (1.0 + ((d - m) / sig).exp())
            })
            .collect()
    };
    let (my, mx) = (axis(ly), axis(lx));
    let mut out = vec![0.0f32; ly * lx];
    for y in 0..ly {
        for x in 0..lx {
            out[y * lx + x] = (my[y] * mx[x]) as f32;
        }
    }
    out
}

/// A normalised 2D Gaussian, centred on the plane.
///
/// Built on the same grid suite2p uses: coordinates run from `-n/2` to `n/2`
/// about the centre, and the result sums to one.
pub fn gaussian_kernel(sigma_y: f64, sigma_x: f64, ly: usize, lx: usize) -> Vec<f32> {
    let coord = |n: usize| -> Vec<f64> {
        let c = (n as f64 - 1.0) / 2.0;
        (0..n).map(|i| i as f64 - c).collect()
    };
    let (cy, cx) = (coord(ly), coord(lx));
    let mut out = vec![0.0f64; ly * lx];
    let mut sum = 0.0;
    for y in 0..ly {
        for x in 0..lx {
            let v = (-(cy[y] * cy[y]) / (2.0 * sigma_y * sigma_y)
                - (cx[x] * cx[x]) / (2.0 * sigma_x * sigma_x))
                .exp();
            out[y * lx + x] = v;
            sum += v;
        }
    }
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

/// The real part of the FFT of a centred Gaussian, for smoothing in frequency.
///
/// `ifftshift` first, because the kernel is built centred and the transform
/// wants the origin in the corner. Skipping it multiplies the spectrum by a
/// checkerboard of signs, which shifts every correlation peak by half the
/// frame — a failure that looks like the registration simply not working.
pub fn gaussian_fft(fft: &mut Fft2, sig: f64, ly: usize, lx: usize) -> Vec<f32> {
    let kernel = ifftshift(&gaussian_kernel(sig, sig, ly, lx), ly, lx);
    let mut buf: Vec<Complex32> = kernel.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    fft.forward(&mut buf);
    buf.iter().map(|c| c.re).collect()
}

/// Everything the reference contributes to a correlation, computed once.
pub struct RefFilters {
    /// The taper, multiplied into every frame.
    pub mask_mul: Vec<f32>,
    /// `mean(ref) * (1 - taper)`, added after it.
    pub mask_offset: Vec<f32>,
    /// The whitened, smoothed, conjugated spectrum of the reference.
    pub cf_ref: Vec<Complex32>,
    /// The range frames are clipped to before correlating, when `norm_frames`
    /// is on. Carried with the filters because it is the *reference's* range —
    /// clipping each frame to its own would be normalising away the very
    /// brightness differences that locate it.
    pub clip: Option<(f32, f32)>,
}

/// Prepare a reference image for correlating frames against.
///
/// `mask_slope` is the taper's slope and `smooth_sigma` the frequency-domain
/// Gaussian — suite2p's `spatial_taper` and `smooth_sigma` respectively.
pub fn reference_filters(
    fft: &mut Fft2,
    reference: &[f32],
    ly: usize,
    lx: usize,
    mask_slope: f64,
    smooth_sigma: f64,
) -> RefFilters {
    reference_filters_normed(fft, reference, ly, lx, mask_slope, smooth_sigma, false)
}

/// [`reference_filters`], optionally clipping to the reference's own 1st and
/// 99th percentile — suite2p's `norm_frames`.
pub fn reference_filters_normed(
    fft: &mut Fft2,
    reference: &[f32],
    ly: usize,
    lx: usize,
    mask_slope: f64,
    smooth_sigma: f64,
    norm_frames: bool,
) -> RefFilters {
    let clip = norm_frames.then(|| crate::pipeline::percentile_range(reference));
    // The reference is clipped too, not just the frames: correlating a clipped
    // frame against an unclipped reference compares two different pictures.
    let clipped: Vec<f32>;
    let reference = match clip {
        Some((lo, hi)) => {
            clipped = reference.iter().map(|v| v.clamp(lo, hi)).collect();
            &clipped[..]
        }
        None => reference,
    };
    let mask_mul = spatial_taper(mask_slope, ly, lx);
    let mean = if reference.is_empty() {
        0.0
    } else {
        reference.iter().map(|&v| v as f64).sum::<f64>() / reference.len() as f64
    } as f32;
    let mask_offset: Vec<f32> = mask_mul.iter().map(|&m| mean * (1.0 - m)).collect();

    // conj(FFT(ref)), whitened, then smoothed.
    let mut cf: Vec<Complex32> = reference.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    fft.forward(&mut cf);
    for c in cf.iter_mut() {
        // The conjugate is what turns a convolution into a correlation.
        *c = c.conj();
        // Whitening: the 1e-5 is suite2p's, and it is what stops an empty
        // frequency bin becoming an infinity that dominates the peak.
        *c /= 1e-5 + c.norm();
    }
    let g = gaussian_fft(fft, smooth_sigma, ly, lx);
    for (c, &g) in cf.iter_mut().zip(&g) {
        *c *= g;
    }

    RefFilters {
        mask_mul,
        mask_offset,
        cf_ref: cf,
        clip,
    }
}

#[cfg(test)]
#[path = "masks_tests.rs"]
mod tests;

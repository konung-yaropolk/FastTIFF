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

//! Measuring a batch of frames, on whichever backend was asked for.

use crate::fft::Fft2;
use crate::masks::RefFilters;
use crate::rigid::{correlation_map, lcorr_for, peak_of, Shift};
use crate::settings::{Backend, Settings};
use rayon::prelude::*;

/// The 1st and 99th percentile of an image, as `norm_frames` uses.
///
/// Percentiles rather than min and max: a two-photon frame has hot pixels, and
/// a single saturated one would set a range the rest of the picture occupies
/// the bottom percent of.
pub fn percentile_range(image: &[f32]) -> (f32, f32) {
    if image.is_empty() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    let mut sorted: Vec<f32> = image.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |p: f64| -> f32 {
        let i = ((sorted.len() - 1) as f64 * p / 100.0).round() as usize;
        sorted[i.min(sorted.len() - 1)]
    };
    (at(1.0), at(99.0))
}

/// Smooth a batch of correlation maps along time, in place.
///
/// One Gaussian per position in the map, running across the batch. It is what
/// `smooth_sigma_time` does, and it works because the peak a dim frame cannot
/// find on its own is usually where its neighbours' peaks already are.
///
/// The batch is the window: suite2p smooths within a batch too, which is why
/// `batch_size` changes the answer when this is non-zero and not otherwise.
pub fn smooth_in_time(maps: &mut [Vec<f32>], sigma: f64) {
    if sigma <= 0.0 || maps.len() < 2 {
        return;
    }
    // Truncated at 4 sigma, as scipy's `gaussian_filter1d` defaults to.
    let radius = ((4.0 * sigma).round() as usize).max(1);
    let kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let d = i as f64 - radius as f64;
            (-0.5 * d * d / (sigma * sigma)).exp()
        })
        .collect();
    let norm: f64 = kernel.iter().sum();

    let n = maps.len();
    let len = maps[0].len();
    let original = maps.to_vec();
    for (t, out) in maps.iter_mut().enumerate() {
        for (i, slot) in out.iter_mut().enumerate().take(len) {
            let mut acc = 0.0f64;
            for (k, w) in kernel.iter().enumerate() {
                // Reflected at the ends, so the first and last frames are not
                // pulled towards nothing.
                let src = (t as isize + k as isize - radius as isize).rem_euclid(2 * n as isize - 2)
                    as usize;
                let src = if src >= n { 2 * n - 2 - src } else { src };
                acc += w * original[src][i] as f64;
            }
            *slot = (acc / norm) as f32;
        }
    }
}

/// Measure the shift of every frame in a batch.
///
/// The backend decides how many are in flight at once and nothing else — the
/// answer is the same either way, which is what makes it safe to offer as a
/// choice.
///
/// [`Backend::Gpu`] runs the correlation on the device when the crate was built
/// with the `gpu` feature and the frame size suits its radix-2 FFT.
/// [`Backend::unavailable_reason`] is what the caller checks first, so nobody
/// selects a device and is quietly handed something else.
pub fn measure_batch(
    ly: usize,
    lx: usize,
    filters: &RefFilters,
    frames: &[Vec<f32>],
    settings: &Settings,
) -> Vec<Shift> {
    let lcorr = lcorr_for(ly, lx, settings.maxregshift);
    let clip = filters.clip;

    // On the device, when that was asked for. An unavailable GPU is refused by
    // the caller (see `Backend::unavailable_reason`), so the only way to arrive
    // here and fail to open one is an adapter that vanished between the check
    // and the run — which takes the CPU path rather than failing the whole
    // registration part-way through.
    #[cfg(feature = "gpu")]
    if settings.backend == Backend::Gpu {
        if let Some(gpu) = crate::gpu::GpuContext::new(ly, lx) {
            gpu.set_reference(filters);
            // One frame at a time: the device is already parallel across the
            // plane, and queueing more would add host bookkeeping for nothing.
            //
            // Note this path does not apply `smooth_sigma_time` — the peak is
            // taken on the device. A run that needs temporal smoothing should
            // use a CPU backend; `measure_batch`'s caller checks that.
            return frames
                .iter()
                .map(|f| gpu.shift_of(f, filters, settings.maxregshift))
                .collect();
        }
    }

    let mut maps: Vec<Vec<f32>> = match settings.backend {
        // One plan, reused; nothing else to schedule.
        Backend::SingleThread => {
            let mut fft = Fft2::new(ly, lx);
            frames
                .iter()
                .map(|f| correlation_map(&mut fft, filters, f, settings.maxregshift, clip))
                .collect()
        }
        // A frame per core. Each worker needs its own plan and scratch — an
        // `Fft2` is stateful — so one is built per chunk rather than per frame.
        //
        // `Gpu` reaches here only in a build without the feature, or after the
        // adapter check above declined; both are already reported to the user.
        Backend::MultiThread | Backend::Gpu => frames
            .par_chunks(8.max(frames.len().div_ceil(rayon::current_num_threads().max(1))))
            .flat_map_iter(|chunk| {
                let mut fft = Fft2::new(ly, lx);
                chunk
                    .iter()
                    .map(|f| correlation_map(&mut fft, filters, f, settings.maxregshift, clip))
                    .collect::<Vec<_>>()
            })
            .collect(),
    };

    smooth_in_time(&mut maps, settings.smooth_sigma_time);
    maps.iter().map(|cc| peak_of(cc, lcorr)).collect()
}

/// Apply a measured correction to a batch of planes, in place and in parallel.
///
/// `frame_of[i]` says which frame `planes[i]` belongs to: a recording with
/// several channels or z-slices has more planes than measurements, and every
/// plane of one timepoint moves by that timepoint's shift. Measuring each
/// channel separately would let them drift apart, which for a two-colour
/// recording is the one thing registration must not do.
///
/// `warp` carries the deformation grid and one field per frame, for a non-rigid
/// run; without it the correction is the rigid shift alone. The rigid shift is
/// folded into the warp rather than applied before it, so a pixel is resampled
/// once.
///
/// This is the other half of the run, next to [`measure_batch`]: a whole
/// recording is resampled here, one output pixel at a time, and leaving it on
/// one thread was enough on its own to hide which backend the measuring used.
pub fn apply_batch(
    planes: &mut [Vec<f32>],
    ly: usize,
    lx: usize,
    frame_of: &[usize],
    shifts: &[Shift],
    warp: Option<(
        &crate::nonrigid::Blocks,
        &[Vec<crate::nonrigid::BlockShift>],
    )>,
    settings: &Settings,
) {
    let correct = |plane: &mut Vec<f32>, t: usize| {
        // A plane with no measurement is left exactly as it is. It cannot
        // happen through the plugin, which pairs them, and silently moving it
        // by someone else's shift would be worse than not moving it.
        let Some(&shift) = shifts.get(t) else {
            return;
        };
        *plane = match warp.and_then(|(blocks, fields)| fields.get(t).map(|f| (blocks, f))) {
            Some((blocks, field)) => crate::nonrigid::warp(plane, ly, lx, blocks, field, shift),
            None => crate::rigid::shift_frame(plane, ly, lx, shift.dy, shift.dx),
        };
    };
    // `Single-thread CPU` means one thread for the whole registration, not just
    // for the measuring — someone who picks it wants a run they can compare
    // against, or a machine they can still use while it runs.
    if settings.backend == Backend::SingleThread {
        for (plane, &t) in planes.iter_mut().zip(frame_of) {
            correct(plane, t);
        }
    } else {
        planes
            .par_iter_mut()
            .zip(frame_of.par_iter())
            .for_each(|(plane, &t)| correct(plane, t));
    }
}

/// Frames whose shift is an outlier, or that barely correlated at all.
///
/// suite2p's `compute_crop`: a frame is bad when its displacement from the
/// local median, divided by its correlation relative to the median correlation,
/// is past `th_badframes * 100` — or when it hit the shift limit, which means
/// the real shift was larger and got clipped.
///
/// Reported rather than dropped. Which frames to throw away is the
/// experimenter's decision, and a registration that quietly discarded some of
/// the recording would be making it for them.
pub fn bad_frames(shifts: &[Shift], ly: usize, lx: usize, settings: &Settings) -> Vec<bool> {
    let n = shifts.len();
    if n == 0 {
        return Vec::new();
    }
    // suite2p's `filter_window`: odd, and never more than 101 frames wide.
    let window = ((n / 2) * 2).saturating_sub(1).clamp(1, 101);
    let median_of = |v: &[f32], i: usize| -> f32 {
        let half = window / 2;
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(v.len());
        let mut w: Vec<f32> = v[lo..hi].to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        w[w.len() / 2]
    };

    let ys: Vec<f32> = shifts.iter().map(|s| s.dy as f32).collect();
    let xs: Vec<f32> = shifts.iter().map(|s| s.dx as f32).collect();
    let cs: Vec<f32> = shifts.iter().map(|s| s.corr).collect();

    let dxy: Vec<f32> = (0..n)
        .map(|i| {
            let dy = ys[i] - median_of(&ys, i);
            let dx = xs[i] - median_of(&xs, i);
            (dy * dy + dx * dx).sqrt()
        })
        .collect();
    let mean = dxy.iter().sum::<f32>() / n as f32;
    let mean = if mean > 0.0 { mean } else { 1.0 };

    (0..n)
        .map(|i| {
            let c = median_of(&cs, i);
            let cxy = if c.abs() > 1e-12 { cs[i] / c } else { 0.0 };
            let px = if cxy > 0.0 {
                dxy[i] / mean / cxy
            } else {
                f32::INFINITY
            };
            px > (settings.th_badframes * 100.0) as f32
                || (shifts[i].dx.abs() as f64) > settings.maxregshift * lx as f64 * 0.95
                || (shifts[i].dy.abs() as f64) > settings.maxregshift * ly as f64 * 0.95
        })
        .collect()
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;

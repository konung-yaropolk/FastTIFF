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

use crate::area::Cut;
use crate::fft::{Fft2, FftPlans2};
use crate::masks::RefFilters;
use crate::nonrigid::{BlockShift, Blocks};
use crate::par::*;
use crate::rigid::{correlation_map, lcorr_for, peak_of, Shift};
use crate::settings::{Backend, Border, Settings};

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

/// The most a batch of frames may hold, in bytes, as `f32`: 256 MiB.
///
/// A host streaming a recording holds a batch while it is measured, and
/// `batch_size` frames of a large field are a lot of memory — the default 100
/// of a 2048-pixel-square recording is a gigabyte and a half. At the frame
/// sizes two-photon recordings come in (512 square and below) the budget is
/// far above the default batch, so it changes nothing there.
pub const BATCH_BUDGET: usize = 256 << 20;

/// How many frames one batch really holds at this frame size: `batch_size`,
/// unless that many would be more than [`BATCH_BUDGET`], and at least one.
///
/// One rule for every way into a registration, and that is why it is here.
/// The batch is the window `smooth_sigma_time` smooths across, so a smaller
/// batch is a shorter window and a different answer; when the streaming
/// viewer bounded its batches and an in-memory caller did not, the two
/// returned different shifts for the same frames and the same settings at
/// large frame sizes. Every caller goes through [`Session`](crate::Session),
/// which asks this, and [`Session::shortened_window`](crate::Session::shortened_window)
/// says when it has shortened the window.
pub fn batch_len(ly: usize, lx: usize, settings: &Settings) -> usize {
    let per_frame = ly
        .saturating_mul(lx)
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);
    settings
        .batch_size
        .max(1)
        .min((BATCH_BUDGET / per_frame).max(1))
}

/// Measure the shift of every frame in a batch.
///
/// The backend decides where the correlation maps are computed and nothing
/// else. Every backend hands its maps to the same smoothing and the same
/// [`peak_of`], so the answer is the same whichever is chosen — which is what
/// makes it safe to offer as a choice.
///
/// [`Backend::Gpu`] runs the correlation on the device when the crate was built
/// with the `gpu` feature and the frame size suits its radix-2 FFT.
/// [`Backend::unavailable_reason`] is what the caller checks first, so nobody
/// selects a device and is quietly handed something else.
///
/// `frames` is anything that lends `ly * lx` samples per frame — `Vec<f32>`s,
/// or slices of one buffer — so a caller holding its frames some other way
/// does not copy them to measure them.
pub fn measure_batch<F: AsRef<[f32]> + Sync>(
    ly: usize,
    lx: usize,
    filters: &RefFilters,
    frames: &[F],
    settings: &Settings,
) -> Vec<Shift> {
    let lcorr = lcorr_for(ly, lx, settings.maxregshift);
    let clip = filters.clip;

    // On the device, when that was asked for. An unavailable GPU is refused by
    // the caller (see `Backend::unavailable_reason`), so arriving here without
    // a context means an adapter that vanished between the check and the run,
    // or a card with no room left for the batch buffers. Either way the batch
    // takes the CPU path rather than failing the whole registration part-way
    // through, and `gpu::take_trouble` says it happened, for a caller that
    // would rather stop.
    #[cfg(feature = "gpu")]
    let on_device = (settings.backend == Backend::Gpu)
        .then(|| crate::gpu::maps_of(ly, lx, filters, frames, settings.maxregshift))
        .flatten();
    #[cfg(not(feature = "gpu"))]
    let on_device: Option<Vec<Vec<f32>>> = None;

    let mut maps: Vec<Vec<f32>> = match (on_device, settings.backend) {
        (Some(maps), _) => maps,
        // One plan, reused; nothing else to schedule.
        (None, Backend::SingleThread) => {
            let mut fft = Fft2::new(ly, lx);
            frames
                .iter()
                .map(|f| correlation_map(&mut fft, filters, f.as_ref(), settings.maxregshift, clip))
                .collect()
        }
        // A frame at a time, to whichever core is free. The plans are built
        // once and shared; each worker keeps only its own scratch (`map_init`
        // makes one per piece of work rayon splits off, which is cheap now that
        // it is scratch alone).
        //
        // `Gpu` reaches here only in a build without the feature, or when the
        // device could not be opened; both are already reported to the user.
        (None, Backend::MultiThread | Backend::Gpu) => {
            let plans = FftPlans2::new(ly, lx);
            frames
                .par_iter()
                .map_init(
                    || Fft2::with_plans(&plans),
                    |fft, f| correlation_map(fft, filters, f.as_ref(), settings.maxregshift, clip),
                )
                .collect()
        }
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
/// The planes are as recorded, as every apply in the crate takes them:
/// `bidiphase` is the offset the run measured them with
/// ([`Registered::bidiphase`](crate::Registered::bidiphase)), undone first; 0
/// for none. An argument rather than the caller's job, because a caller who
/// has to remember to undo the comb can forget it, or do it twice.
///
/// This is the other half of the run, next to [`measure_batch`]: a whole
/// recording is resampled here, one output pixel at a time, and leaving it on
/// one thread was enough on its own to hide which backend the measuring used.
///
/// Under [`Border::Crop`] the planes are resampled as under `Constant`, at
/// full size: the area to cut them to depends on every frame of the
/// recording, and a batch is only some of them.
/// [`Registered::apply_batch`](crate::Registered::apply_batch) moves and cuts
/// them once every frame is measured.
///
/// # Panics
///
/// When `planes` and `frame_of` differ in length, or a plane's frame has no
/// shift (or, in a warp, no field): moved by nothing, or by another frame's
/// correction, it would come back looking registered and not be.
#[allow(clippy::too_many_arguments)]
pub fn apply_batch(
    planes: &mut [Vec<f32>],
    ly: usize,
    lx: usize,
    frame_of: &[usize],
    shifts: &[Shift],
    warp: Option<(&Blocks, &[Vec<BlockShift>])>,
    bidiphase: i32,
    settings: &Settings,
) {
    let how = Moving {
        ly,
        lx,
        shifts,
        warp: warp.map(|(blocks, fields)| Warp {
            blocks,
            fields,
            first: 0,
        }),
        border: settings.border,
        bidiphase,
        cut: None,
    };
    how.apply(planes, frame_of, settings.backend != Backend::SingleThread);
}

/// A block field to warp by: the grid, and the fields held, `fields[0]` being
/// frame `first`'s — every frame's for a finished registration, one batch's
/// for a [`Session`](crate::Session) that does not keep them.
pub(crate) struct Warp<'a> {
    pub blocks: &'a Blocks,
    pub fields: &'a [Vec<BlockShift>],
    pub first: usize,
}

/// How planes are moved onto the reference, for every way the crate moves
/// one — [`apply_batch`], [`Session::apply`](crate::Session::apply),
/// [`Registered::apply`](crate::Registered::apply) and its `apply_batch` — so
/// that the order is written once: the comb undone, then the cut, the warp or
/// the shift.
pub(crate) struct Moving<'a> {
    pub ly: usize,
    pub lx: usize,
    pub shifts: &'a [Shift],
    pub warp: Option<Warp<'a>>,
    pub border: Border,
    /// Undone on each plane before it moves; 0 for none.
    pub bidiphase: i32,
    /// Under [`Border::Crop`], once the area is known: moved and cut in one
    /// resampling, from the plane with its comb undone.
    pub cut: Option<Cut<'a>>,
}

impl Moving<'_> {
    /// Refuse a plane there is no correction for, before any plane is
    /// touched: one of a frame with no shift, or with no block field held.
    fn check(&self, planes: usize, frame_of: &[usize]) {
        assert_eq!(
            planes,
            frame_of.len(),
            "one frame index per plane: {planes} planes and {} indices",
            frame_of.len()
        );
        for &t in frame_of {
            assert!(
                t < self.shifts.len(),
                "a plane of frame {t}, which has no shift: {} frames are measured",
                self.shifts.len()
            );
            if let Some(w) = &self.warp {
                let held = w.first..w.first + w.fields.len();
                assert!(
                    held.contains(&t),
                    "frame {t}'s block field is not held, only frames {held:?}'s: a session \
                     holds one batch's at a time unless asked to keep them (Session::keep_fields)"
                );
            }
        }
    }

    /// Frame `t`'s correction, on a plane with its comb undone already.
    fn moved(&self, combless: &[f32], t: usize) -> Vec<f32> {
        if let Some(cut) = &self.cut {
            return cut.apply(combless, t);
        }
        let shift = self.shifts[t];
        let (ly, lx) = (self.ly, self.lx);
        match &self.warp {
            Some(w) => crate::nonrigid::warp(
                combless,
                ly,
                lx,
                w.blocks,
                &w.fields[t - w.first],
                shift,
                self.border,
            ),
            None => crate::rigid::shift_frame_bordered(
                combless,
                ly,
                lx,
                shift.dy,
                shift.dx,
                self.border,
            ),
        }
    }

    /// A plane of frame `t`, as recorded, moved: a new plane, the one given
    /// left as it is.
    pub fn one(&self, plane: &[f32], t: usize) -> Vec<f32> {
        self.check(1, &[t]);
        if self.bidiphase == 0 {
            return self.moved(plane, t);
        }
        let mut combless = plane.to_vec();
        crate::bidiphase::shift(&mut combless, self.ly, self.lx, self.bidiphase);
        self.moved(&combless, t)
    }

    /// Move `planes[i]`, as recorded, by frame `frame_of[i]`'s correction, in
    /// place, on every core when `parallel`.
    pub fn apply(&self, planes: &mut [Vec<f32>], frame_of: &[usize], parallel: bool) {
        self.check(planes.len(), frame_of);
        let (ly, lx) = (self.ly, self.lx);
        let correct = |plane: &mut Vec<f32>, t: usize| {
            // The comb first: the shift was measured on frames without it.
            crate::bidiphase::shift(plane, ly, lx, self.bidiphase);
            *plane = self.moved(plane, t);
        };
        // `Single-thread CPU` means one thread for the whole registration, not
        // just for the measuring — someone who picks it wants a run they can
        // compare against, or a machine they can still use while it runs.
        if !parallel {
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
    // One window's worth, reused: three medians a frame, and a window
    // allocated for each was three allocations a frame of the recording.
    let mut w: Vec<f32> = Vec::with_capacity(window);
    let mut median_of = |v: &[f32], i: usize| -> f32 {
        let half = window / 2;
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(v.len());
        w.clear();
        w.extend_from_slice(&v[lo..hi]);
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

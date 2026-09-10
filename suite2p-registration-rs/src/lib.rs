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

//! Motion correction for two-photon timelapses, ported from suite2p.
//!
//! A recording of living tissue moves. Breathing, heartbeat, the animal
//! shifting — over minutes a field drifts by tens of pixels, and every
//! measurement made per-pixel afterwards is measuring a different piece of
//! tissue at each timepoint. Registration puts the frames back on top of each
//! other first.
//!
//! # What this is a port of
//!
//! `suite2p/registration` — the reference implementation for this kind of data,
//! and the one the numbers in published papers came out of. The algorithm is
//! reproduced, not the code: this is Rust written against suite2p's method,
//! its parameter names and its defaults, so that a recording registered here
//! and the same recording registered there agree.
//!
//! Because it is a derivative work of GPL-3 software, this crate is GPL-3.
//!
//! # What is here, and what is not
//!
//! * **Bidirectional phase** correction — the comb artefact from a resonant
//!   scanner. See [`bidiphase`].
//! * **Rigid registration** — one whole-frame shift per frame, by phase
//!   correlation against an iteratively-built reference. See [`rigid`].
//! * **Non-rigid registration** — a shift per block, interpolated to a shift
//!   per pixel, for tissue that deforms rather than merely sliding. See
//!   [`nonrigid`].
//!
//! * **Three backends** — one thread, every core, or the graphics card. The
//!   answer is the same on each; only how many frames are in flight differs,
//!   and [`lib_tests`](self) pins that. The GPU path is behind the `gpu`
//!   feature and takes power-of-two frame sizes only (its FFT is radix-2);
//!   [`Backend::unavailable_reason`] says so rather than quietly using the
//!   processor, because a run that said GPU and used the CPU is
//!   indistinguishable from a slow one.

pub mod bidiphase;
pub mod fft;
/// Phase correlation on the graphics card. Behind the `gpu` feature so the
/// algorithm crate does not pull a graphics stack into a CPU-only build.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod masks;
pub mod nonrigid;
pub mod pipeline;
pub mod rigid;
pub mod settings;

use fft::Fft2;
use masks::reference_filters_normed;
pub use rigid::{shift_frame, Shift};
pub use settings::{Backend, Settings};

/// A movie to register: frames of `ly * lx`, row-major.
pub struct Frames<'a> {
    pub ly: usize,
    pub lx: usize,
    pub frames: &'a [Vec<f32>],
}

/// The initial reference: the mean of the frames most like each other.
///
/// Correlate every sampled frame against every other, find the one whose top 20
/// partners agree with it best, and average those 20. It is a way of asking
/// "which part of this recording is the recording at rest", without anyone
/// having to say so.
pub fn pick_initial_reference(frames: &[Vec<f32>], ly: usize, lx: usize) -> Vec<f32> {
    let n = frames.len();
    if n == 0 {
        return vec![0.0; ly * lx];
    }
    if n == 1 {
        return frames[0].clone();
    }
    // Mean-subtracted, so the correlation below is a correlation and not a
    // measure of overall brightness.
    let centred: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| {
            let mean = f.iter().map(|&v| v as f64).sum::<f64>() / f.len().max(1) as f64;
            f.iter().map(|&v| v as f64 - mean).collect()
        })
        .collect();
    let norm: Vec<f64> = centred
        .iter()
        .map(|f| f.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12))
        .collect();

    let mut cc = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i..n {
            let d: f64 = centred[i]
                .iter()
                .zip(&centred[j])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / (norm[i] * norm[j]);
            cc[i * n + j] = d;
            cc[j * n + i] = d;
        }
    }

    // The frame whose 19 best partners (excluding itself) agree most.
    let top = 20.min(n);
    let mut best = (0usize, f64::NEG_INFINITY);
    for i in 0..n {
        let mut row: Vec<f64> = (0..n).map(|j| cc[i * n + j]).collect();
        row.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mean = row[1..top.max(2)].iter().sum::<f64>() / (top.max(2) - 1) as f64;
        if mean > best.1 {
            best = (i, mean);
        }
    }

    // Average that frame's `top` closest partners.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        cc[best.0 * n + b]
            .partial_cmp(&cc[best.0 * n + a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0f64; ly * lx];
    for &i in order.iter().take(top) {
        for (o, v) in out.iter_mut().zip(&centred[i]) {
            *o += v;
        }
    }
    out.iter().map(|v| (v / top as f64) as f32).collect()
}

/// Build the reference frames are registered against.
///
/// Starts from [`pick_initial_reference`] and refines: register everything to
/// the current reference, keep the best-correlated frames, average them, and go
/// again. The number kept grows each pass — a quarter of the frames on the
/// first, all of them by the last — so an early bad reference cannot lock in.
///
/// `on_progress` is called with a fraction and returns `false` to stop.
pub fn compute_reference(
    movie: &Frames<'_>,
    settings: &Settings,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Vec<f32>> {
    let (ly, lx) = (movie.ly, movie.lx);
    if movie.frames.is_empty() || ly == 0 || lx == 0 {
        return Some(vec![0.0; ly * lx]);
    }
    let mut reference = pick_initial_reference(movie.frames, ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let niter = settings.reference_iterations.max(1);
    let n = movie.frames.len();

    // The frames as they are shifted onto the reference, refined each pass.
    let mut aligned: Vec<Vec<f32>> = movie.frames.to_vec();

    for iter in 0..niter {
        if !on_progress(iter as f32 / niter as f32) {
            return None;
        }
        let filters = reference_filters_normed(
            &mut fft,
            &reference,
            ly,
            lx,
            settings.spatial_taper,
            settings.smooth_sigma,
            settings.norm_frames,
        );

        // Through the pipeline, so building the reference uses the backend the
        // caller asked for too — it is eight passes over `nimg_init` frames,
        // which is not free.
        let shifts: Vec<Shift> =
            crate::pipeline::measure_batch(ly, lx, &filters, &aligned, settings);
        for (frame, s) in aligned.iter_mut().zip(&shifts) {
            *frame = shift_frame(frame, ly, lx, s.dy, s.dx);
        }

        // Keep the best-correlated frames — more of them each pass.
        let nmax = ((n as f64 * (1.0 + iter as f64) / (2.0 * niter as f64)) as usize).max(2);
        let nmax = nmax.min(n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            shifts[b]
                .corr
                .partial_cmp(&shifts[a].corr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = &order[..nmax];

        let mut acc = vec![0.0f64; ly * lx];
        for &i in keep {
            for (a, v) in acc.iter_mut().zip(&aligned[i]) {
                *a += *v as f64;
            }
        }
        reference = acc.iter().map(|v| (v / nmax as f64) as f32).collect();

        // Recentre, so the reference does not drift across the frame over the
        // passes and take the whole registration with it.
        let mean_dy = keep.iter().map(|&i| shifts[i].dy as f64).sum::<f64>() / nmax as f64;
        let mean_dx = keep.iter().map(|&i| shifts[i].dx as f64).sum::<f64>() / nmax as f64;
        // suite2p rolls the reference by `+round(mean shift)`; `shift_frame`
        // takes content *from* `y + dy`, which is a roll by `-dy`. So the sign
        // here is the negative of the mean, and getting it backwards moves the
        // reference the wrong way on every pass — the registration then
        // converges on a drifting target and reports about twice the real
        // motion, which is exactly what it did.
        reference = shift_frame(
            &reference,
            ly,
            lx,
            -(mean_dy.round() as i32),
            -(mean_dx.round() as i32),
        );
    }
    Some(reference)
}

/// What a registration run produced.
///
/// Not `Clone`: the block field carries the grid it is defined on, and a copy
/// of a field without its grid would be a set of numbers nobody could apply.
pub struct Registered {
    /// The shift applied to each frame, in input order.
    pub shifts: Vec<Shift>,
    /// The reference every frame was aligned to.
    pub reference: Vec<f32>,
    /// The bidirectional offset applied, if any.
    pub bidiphase: i32,
    /// Which frames look like outliers — a large deviation from the local
    /// median, a poor correlation, or a shift that hit the limit.
    ///
    /// Reported, never dropped. Which frames to discard is the experimenter's
    /// decision, and a registration that quietly removed some of the recording
    /// would be making it for them.
    pub bad_frames: Vec<bool>,
    /// The per-block deformation, when `nonrigid` was on.
    pub nonrigid: Option<NonRigid>,
}

/// The block field a non-rigid pass measured.
pub struct NonRigid {
    /// The grid the field is defined on.
    pub blocks: nonrigid::Blocks,
    /// One entry per frame, each one shift per block.
    pub shifts: Vec<Vec<nonrigid::BlockShift>>,
}

impl Registered {
    /// Put frame `t` back where it belongs.
    ///
    /// The rigid shift alone when there is no block field, and the warp — which
    /// folds the rigid shift in — when there is. One call, so a caller cannot
    /// apply the rigid part twice or forget the deformation.
    pub fn apply(&self, frame: &[f32], ly: usize, lx: usize, t: usize) -> Vec<f32> {
        let rigid = self.shifts.get(t).copied().unwrap_or(Shift {
            dy: 0,
            dx: 0,
            corr: 0.0,
        });
        match &self.nonrigid {
            Some(nr) if t < nr.shifts.len() => {
                nonrigid::warp(frame, ly, lx, &nr.blocks, &nr.shifts[t], rigid)
            }
            _ => shift_frame(frame, ly, lx, rigid.dy, rigid.dx),
        }
    }
}

/// Register a movie: measure each frame's shift against a computed reference.
///
/// The frames themselves are not moved here — [`shift_frame`] does that, and
/// keeping it separate is what lets a caller apply the same shifts to a second
/// channel that was aligned by the first.
///
/// `on_progress` is called with a fraction and returns `false` to stop, which
/// returns `None`.
pub fn register(
    movie: &Frames<'_>,
    settings: &Settings,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Registered> {
    let passes = if settings.two_step_registration { 2 } else { 1 };
    let mut working: Vec<Vec<f32>> = movie.frames.to_vec();
    let mut total: Vec<Shift> = vec![
        Shift {
            dy: 0,
            dx: 0,
            corr: 0.0
        };
        movie.frames.len()
    ];
    let mut last: Option<Registered> = None;

    for pass in 0..passes {
        let span = 1.0 / passes as f32;
        let base = pass as f32 * span;
        let out = register_once(
            &Frames {
                ly: movie.ly,
                lx: movie.lx,
                frames: &working,
            },
            settings,
            &mut |f| on_progress(base + f * span),
        )?;

        // The second pass measures what is left after the first, so the shift
        // a caller applies is the sum. Registering the already-registered
        // frames and reporting only the residual would leave the movie moving.
        for (t, s) in out.shifts.iter().enumerate() {
            total[t].dy += s.dy;
            total[t].dx += s.dx;
            total[t].corr = s.corr;
        }
        if pass + 1 < passes {
            working = working
                .iter()
                .zip(&out.shifts)
                .map(|(f, s)| shift_frame(f, movie.ly, movie.lx, s.dy, s.dx))
                .collect();
        }
        last = Some(out);
    }

    let mut out = last?;
    out.shifts = total;
    out.bad_frames = crate::pipeline::bad_frames(&out.shifts, movie.ly, movie.lx, settings);

    // The deformation is measured *after* the rigid shift, on frames already
    // brought onto the reference — so a block is looking for the few pixels the
    // tissue stretched, not the tens the animal moved. Measuring both at once
    // would need every block's search window to cover the whole motion, and a
    // 64-pixel block cannot see 50 pixels of travel.
    if settings.nonrigid {
        let (ly, lx) = (movie.ly, movie.lx);
        let blocks = nonrigid::make_blocks(ly, lx, settings.block_size, settings.subpixel);
        let mut filters = nonrigid::block_filters(
            &out.reference,
            lx,
            &blocks,
            settings.spatial_taper,
            settings.smooth_sigma,
        );
        let clip = settings
            .norm_frames
            .then(|| crate::pipeline::percentile_range(&out.reference));

        let mut per_frame = Vec::with_capacity(movie.frames.len());
        for (t, frame) in movie.frames.iter().enumerate() {
            if !on_progress(0.9 + 0.1 * t as f32 / movie.frames.len().max(1) as f32) {
                return None;
            }
            let r = out.shifts[t];
            let rigid_corrected = shift_frame(frame, ly, lx, r.dy, r.dx);
            per_frame.push(nonrigid::measure_blocks(
                &mut filters,
                &rigid_corrected,
                lx,
                &blocks,
                &nonrigid::BlockSearch {
                    maxregshift_nr: settings.maxregshift_nr,
                    snr_thresh: settings.snr_thresh,
                    subpixel: settings.subpixel,
                    clip,
                },
            ));
        }
        out.nonrigid = Some(NonRigid {
            blocks,
            shifts: per_frame,
        });
    }
    Some(out)
}

/// One registration pass. [`register`] runs this twice when
/// `two_step_registration` is on.
fn register_once(
    movie: &Frames<'_>,
    settings: &Settings,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Registered> {
    let (ly, lx) = (movie.ly, movie.lx);

    // Bidirectional phase first: it is a within-frame artefact, and measuring
    // motion through a combed picture measures the comb as well.
    //
    // Measured only when asked for *and* no fixed offset was supplied, which is
    // suite2p's rule (`do_bidiphase and settings["bidiphase"] == 0`). Someone
    // who typed in their scope's offset does not want it re-derived every run.
    let bidi = if settings.do_bidiphase && settings.bidiphase == 0 {
        bidiphase::compute(movie.frames, ly, lx)
    } else {
        settings.bidiphase
    };
    let corrected: Vec<Vec<f32>> = if bidi != 0 {
        movie
            .frames
            .iter()
            .map(|f| {
                let mut f = f.clone();
                bidiphase::shift(&mut f, ly, lx, bidi);
                f
            })
            .collect()
    } else {
        movie.frames.to_vec()
    };

    // The reference is built from a sample, not the whole recording: 300 frames
    // is enough to find what the field looks like at rest, and a 20-minute
    // recording is tens of thousands.
    let step = (corrected.len() / settings.nimg_init.max(1)).max(1);
    let sample: Vec<Vec<f32>> = corrected.iter().step_by(step).cloned().collect();
    let reference = compute_reference(
        &Frames {
            ly,
            lx,
            frames: &sample,
        },
        settings,
        &mut |f| on_progress(f * 0.33),
    )?;
    drop(sample);

    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters_normed(
        &mut fft,
        &reference,
        ly,
        lx,
        settings.spatial_taper,
        settings.smooth_sigma,
        settings.norm_frames,
    );
    drop(fft);

    // In batches, because `smooth_sigma_time` smooths *within* one — which is
    // also why `batch_size` changes the answer when that is non-zero, and only
    // then.
    let batch = settings.batch_size.max(1);
    let mut shifts = Vec::with_capacity(corrected.len());
    for (i, chunk) in corrected.chunks(batch).enumerate() {
        if !on_progress(0.33 + 0.67 * (i * batch) as f32 / corrected.len().max(1) as f32) {
            return None;
        }
        shifts.extend(crate::pipeline::measure_batch(
            ly, lx, &filters, chunk, settings,
        ));
    }

    Some(Registered {
        shifts,
        reference,
        bidiphase: bidi,
        bad_frames: Vec::new(),
        nonrigid: None,
    })
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

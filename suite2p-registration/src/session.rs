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

//! A registration fed a batch at a time: the one place the pipeline's order
//! is written down.
//!
//! [`register`](crate::register) has every frame in memory; a host that
//! streams a recording off disk has one batch of it at a time. Both are
//! feeders of a [`Session`], which says which frames it wants and does
//! everything else: it measures the bidirectional phase and undoes it, builds
//! the reference and its filters, decides the batch, runs the second pass of a
//! two-step registration, measures the block field and flags the bad frames.
//! A feeder only reads frames, reports progress and decides when to stop.
//!
//! That is the point of it. When the viewer re-sequenced the crate's public
//! stages itself, the two orderings drifted: the viewer once measured frame
//! by frame on one thread and ignored `smooth_sigma_time` on the recording,
//! bounded its batches where the in-memory path did not, never ran the
//! second pass `two_step_registration` asks for and never flagged a bad
//! frame. Each fix had to be made twice and tested twice, and parity with
//! suite2p was pinned only for the path the viewer did not take.
//!
//! # Feeding one
//!
//! ```
//! use suite2p_registration::{Border, Session, Settings};
//!
//! # let (ly, lx) = (32, 32);
//! # let movie: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0; ly * lx]).collect();
//! # let read = |t: usize| movie[t].clone();
//! let settings = Settings { nonrigid: false, spatial_taper: 5.0, ..Settings::default() };
//! let mut session = Session::new(ly, lx, movie.len(), &settings);
//! // Once, or twice for a two-step registration.
//! while session.needs_reference() {
//!     // Exactly these frames, as recorded, in this order.
//!     let sample: Vec<Vec<f32>> = session.reference_frames().iter().map(|&t| read(t)).collect();
//!     session.build_reference(sample, &mut |_fraction| true).expect("not cancelled");
//!     // Then the recording, a batch at a time, in order.
//!     while let Some(batch) = session.next_batch() {
//!         let mut frames: Vec<Vec<f32>> = batch.clone().map(read).collect();
//!         // The feeder's own buffers: corrected in them, handed back as read.
//!         let measured = session.measure_in_place(&mut frames);
//!         // Each shift is the frame's own on the last pass: move its planes
//!         // now — or, for a crop, once `finish` knows the area to cut to.
//!         if measured.is_final() && settings.border != Border::Crop {
//!             let at: Vec<usize> = batch.collect();
//!             measured.apply(&mut frames, &at);
//!         }
//!     }
//! }
//! let registered = session.finish();
//! assert_eq!(registered.shifts.len(), movie.len());
//! ```
//!
//! Under [`Border::Crop`] the planes are read again
//! after [`finish`](Session::finish) and moved and cut in one resampling by
//! [`Registered::apply_batch`](crate::Registered::apply_batch).
//!
//! # Frames go in as recorded
//!
//! Everything a feeder hands over is a frame or plane as the file holds it.
//! The session undoes the bidirectional phase itself, on the reference sample,
//! on every batch and on every plane it moves ([`Session::apply`],
//! [`Registered::apply`](crate::Registered::apply)), so there is no step at
//! which a caller has to remember to correct the comb, and none at which it
//! can correct it twice.
//!
//! A batch handed to [`measure`](Session::measure) is borrowed and never
//! written — what a caller with read-only frames needs, such as `register`'s
//! borrowed movie (a C caller's `const` buffer could go in as it lies, though
//! `s2p_register` still copies it first). Frames that need nothing done to
//! them (no comb, and the first pass) are measured where they are; the rest
//! are copied into a batch-sized buffer the session keeps. A feeder that owns
//! its batch hands it to [`measure_in_place`](Session::measure_in_place)
//! instead, which corrects the frames where they are and hands them back as
//! recorded, so that it holds one batch rather than two.
//!
//! # What a session holds
//!
//! The reference sample while the reference is built, then a batch: memory
//! bounded by `nimg_init` and [`pipeline::BATCH_BUDGET`], whatever the length
//! of the recording, plus a shift per frame. A non-rigid run's block fields
//! are held a batch at a time too, unless the feeder asks to keep them all
//! ([`keep_fields`](Session::keep_fields)) to move frames after
//! [`finish`](Session::finish).

use std::ops::Range;

use crate::bidiphase;
use crate::fft::Fft2;
use crate::masks::{reference_filters_normed, RefFilters};
use crate::nonrigid::{self, BlockFilters, BlockSearch, BlockShift, Blocks};
use crate::pipeline::{self, Moving, Warp};
use crate::rigid::{roll, Shift};
use crate::settings::{Backend, Border, Settings};
use crate::{NonRigid, Registered};

/// No shift, for a frame no pass has measured yet.
const STILL: Shift = Shift {
    dy: 0,
    dx: 0,
    corr: 0.0,
};

/// Which frames suite2p builds its reference from: `nimg_init` of them,
/// spread evenly over the recording — `np.linspace(0, n, 1 + min(nimg_init,
/// n), dtype=int)[:-1]`.
///
/// In floating point and truncated, as numpy does it, rather than as the exact
/// `k * n / m`: the two differ by a frame now and then (`n = 114`, `m = 100`
/// puts sample 50 on frame 56 in numpy and 57 exactly), and the point of the
/// sample is to be suite2p's. The crate and the viewer used to step by
/// `n / nimg_init` instead, which keeps every frame of a recording a little
/// shorter than twice `nimg_init` — up to `2 * nimg_init - 1` of them, all
/// held at once while the reference is built.
pub fn reference_sample(frames: usize, nimg_init: usize) -> Vec<usize> {
    let m = nimg_init.max(1).min(frames);
    if m == 0 {
        return Vec::new();
    }
    let step = frames as f64 / m as f64;
    (0..m).map(|k| (k as f64 * step) as usize).collect()
}

/// A registration in progress, fed a batch at a time. See the [module
/// documentation](self) for the loop that drives it.
///
/// Nothing a session hands out borrows it across calls but [`Measured`], and
/// everything `Measured` says the session says too ([`shifts`](Self::shifts),
/// [`field`](Self::field), [`apply`](Self::apply)): a caller across a foreign
/// function interface, which cannot hold a borrow from one call to the next,
/// drives it with those.
pub struct Session {
    ly: usize,
    lx: usize,
    frames: usize,
    settings: Settings,
    passes: usize,
    /// The pass under way, from 0.
    pass: usize,
    /// Whether this pass's reference is built.
    built: bool,
    /// Whether a reference build was stopped: the session then wants nothing
    /// more, so that a feeder's loop ends rather than building again.
    stopped: bool,
    sample: Vec<usize>,
    batch: usize,
    /// The bidirectional offset: the settings' own, or measured on the
    /// reference sample of the first pass.
    bidiphase: i32,
    reference: Vec<f32>,
    /// `None` for a frame with no pixels, which has nothing to correlate.
    filters: Option<RefFilters>,
    /// The block grid and a set of block filters per worker, built from the
    /// last pass's reference when the run is non-rigid.
    blocks: Option<(Blocks, Vec<Vec<BlockFilters>>)>,
    /// Each frame's total shift over the passes already finished; empty in
    /// the first. The frames a later pass measures are moved by these first.
    earlier: Vec<Shift>,
    /// The total shift of every frame this pass has measured, in order.
    shifts: Vec<Shift>,
    /// The block fields the last pass holds, `fields[0]` being frame
    /// `fields_from`'s: every frame's when they are kept, the last batch's
    /// otherwise.
    fields: Vec<Vec<BlockShift>>,
    fields_from: usize,
    /// Whether every frame's block field is kept for [`finish`](Self::finish).
    keep_fields: bool,
    /// Where [`measure`](Self::measure) corrects a batch when it needs to be —
    /// a comb to undo, or an earlier pass's shift to apply. Allocated once a
    /// pass, at the batch's size.
    scratch: Vec<Vec<f32>>,
    /// What undoing the comb slides over in a batch
    /// [`measure_in_place`](Self::measure_in_place) corrects, kept to put
    /// back: a few columns of every other line.
    covered: Vec<f32>,
}

impl Session {
    /// Start registering `frames` frames of `ly` rows by `lx` columns.
    ///
    /// Which backend can run is the caller's to check first
    /// ([`Backend::unavailable_reason`](crate::Backend::unavailable_reason)):
    /// a refusal is the caller's to show, and a session has no way to.
    pub fn new(ly: usize, lx: usize, frames: usize, settings: &Settings) -> Session {
        Session {
            ly,
            lx,
            frames,
            settings: *settings,
            passes: if settings.two_step_registration { 2 } else { 1 },
            pass: 0,
            built: false,
            stopped: false,
            sample: reference_sample(frames, settings.nimg_init),
            batch: pipeline::batch_len(ly, lx, settings),
            bidiphase: settings.bidiphase,
            reference: Vec::new(),
            filters: None,
            blocks: None,
            earlier: Vec::new(),
            shifts: Vec::new(),
            fields: Vec::new(),
            fields_from: 0,
            // A crop's area is found from every frame's field.
            keep_fields: settings.border == Border::Crop,
            scratch: Vec::new(),
            covered: Vec::new(),
        }
    }

    /// Keep every frame's block field until [`finish`](Self::finish), so
    /// that the [`Registered`] it returns can move frames afterwards
    /// ([`Registered::apply`]). [`register`](crate::register) asks for this.
    ///
    /// Without it a non-rigid session holds one batch's fields at a time —
    /// the ones [`apply`](Self::apply) moves that batch's planes by — and the
    /// `Registered` names the block grid but carries no field
    /// ([`NonRigid::shifts`]). A feeder that moves each batch's planes as it
    /// goes has no use for the rest, and a field per frame is memory that
    /// grows with the recording: 144 blocks of a 512-square frame are 1.7 KB a
    /// frame, 576 of a 1024-square one about 7 KB. Under
    /// [`Border::Crop`] they are kept regardless: the
    /// area to cut to is found from every frame's field.
    ///
    /// # Panics
    ///
    /// Once a field has been let go of: ask before the last pass measures its
    /// second batch.
    pub fn keep_fields(&mut self) {
        assert!(
            self.fields_from == 0,
            "the block fields of frames before {} are gone already: keep them from the start",
            self.fields_from
        );
        self.keep_fields = true;
    }

    /// How many frames the recording has.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// How many passes over the recording the run makes: two for
    /// `two_step_registration`, one otherwise. Every pass builds a reference
    /// and measures every frame; only the last measures block fields and has
    /// shifts worth applying.
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// The pass under way, from 0.
    pub fn pass(&self) -> usize {
        self.pass
    }

    /// Whether the pass under way is the last, so that the shifts it measures
    /// are the frames' own and their planes can be moved as each batch is
    /// measured. What [`Measured::is_final`] says.
    pub fn is_last_pass(&self) -> bool {
        self.pass + 1 >= self.passes
    }

    /// The frames [`build_reference`](Self::build_reference) wants, by index
    /// into the recording, in order: suite2p's sample ([`reference_sample`]).
    /// The same frames on every pass.
    pub fn reference_frames(&self) -> &[usize] {
        &self.sample
    }

    /// How many frames a batch holds: `batch_size`, bounded by memory at this
    /// frame size ([`pipeline::batch_len`]). The last batch of a pass may be
    /// shorter.
    pub fn batch_len(&self) -> usize {
        self.batch
    }

    /// Why the temporal smoothing's window is shorter than `batch_size` asks,
    /// when it is: a batch bounded by memory, with `smooth_sigma_time` on.
    /// Worth telling the user, since it changes the answer. With the
    /// smoothing off the batch is a scheduling detail and this is `None` —
    /// and so it is for a recording no longer than the shorter batch, whose
    /// one batch is the whole recording either way.
    pub fn shortened_window(&self) -> Option<String> {
        let window = |batch: usize| batch.min(self.frames);
        (window(self.batch) < window(self.settings.batch_size)
            && self.settings.smooth_sigma_time > 0.0)
            .then(|| {
                format!(
                    "batch size reduced from {} to {} to bound memory at this frame size; \
                     smooth_sigma_time smooths within a batch, so its window is shorter",
                    self.settings.batch_size, self.batch
                )
            })
    }

    /// The bidirectional offset the run corrects: the settings' own, or once
    /// the first reference is built, the one measured on its sample.
    pub fn bidiphase(&self) -> i32 {
        self.bidiphase
    }

    fn pass_measured(&self) -> bool {
        self.built && self.shifts.len() >= self.frames
    }

    /// Whether a reference is wanted next: at the start of the run, and once
    /// more after the first pass of a two-step registration. Never once a
    /// build was stopped.
    pub fn needs_reference(&self) -> bool {
        !self.stopped && (!self.built || (self.pass_measured() && !self.is_last_pass()))
    }

    /// Whether every pass is done, so that [`finish`](Self::finish) can be
    /// called.
    pub fn is_done(&self) -> bool {
        self.pass_measured() && self.is_last_pass()
    }

    /// Build this pass's reference from `sample`: the frames
    /// [`reference_frames`](Self::reference_frames) names, as recorded and in
    /// that order.
    ///
    /// On the first pass this is where the bidirectional phase is measured,
    /// on the sample, as suite2p measures it — when `do_bidiphase` asks and no
    /// fixed offset was given. Every pass then corrects the sample, moves it
    /// by the earlier passes' shifts, and refines it into the reference
    /// ([`compute_reference`](crate::compute_reference)). The sample is taken
    /// rather than borrowed because it is the working space for that: held
    /// once, rather than once by the caller and again by the refinement.
    ///
    /// `on_progress` gets the fraction of the build done and returns `false`
    /// to stop, which returns `None`. The session is then of no further use,
    /// and says so: it wants no reference and no batch, so a feeder's loop
    /// ends, and [`finish`](Self::finish) refuses.
    ///
    /// # Panics
    ///
    /// When no reference is wanted ([`needs_reference`](Self::needs_reference)),
    /// or `sample` is not the frames asked for: a different count, or a frame
    /// that is not `ly * lx` samples.
    #[must_use = "None means the build was stopped, and the registration with it"]
    pub fn build_reference(
        &mut self,
        mut sample: Vec<Vec<f32>>,
        on_progress: &mut dyn FnMut(f32) -> bool,
    ) -> Option<()> {
        assert!(
            !self.stopped,
            "the session was stopped while its reference was built, and is of no further use"
        );
        assert!(
            self.needs_reference(),
            "no reference is wanted now: measure the batches next_batch names, or finish"
        );
        assert_eq!(
            sample.len(),
            self.sample.len(),
            "the reference is built from the frames reference_frames names"
        );
        let (ly, lx) = (self.ly, self.lx);
        assert!(
            sample.iter().all(|f| f.len() == ly * lx),
            "every frame of the sample is {ly} x {lx}"
        );
        if self.built {
            // The first pass is measured: its shifts are what the second
            // pass's frames are moved by before it measures them.
            self.earlier = std::mem::take(&mut self.shifts);
            self.pass += 1;
            self.built = false;
            // Building a reference is the run's largest moment — the sample,
            // and the copy the initial pick makes of it — and the batch buffer
            // is idle through it. Let go, it is one allocation per pass;
            // kept, it was a batch more at the peak.
            self.scratch.clear();
        }
        let settings = self.settings;

        // The comb, measured once, on the frames as recorded. A second pass
        // that measured it again would measure frames the first had moved,
        // and a shift by an odd number of rows swaps which lines are the odd
        // ones: the comb came out mirrored, and undoing that doubled it.
        if self.pass == 0 && settings.do_bidiphase && settings.bidiphase == 0 {
            self.bidiphase = bidiphase::compute(&sample, ly, lx);
        }
        for (frame, &t) in sample.iter_mut().zip(&self.sample) {
            bidiphase::shift(frame, ly, lx, self.bidiphase);
            if let Some(s) = self.earlier.get(t) {
                roll(frame, ly, lx, s.dy, s.dx);
            }
        }
        let Some(reference) = crate::refine_reference(&mut sample, ly, lx, &settings, on_progress)
        else {
            self.stopped = true;
            return None;
        };
        drop(sample);

        let empty = ly == 0 || lx == 0;
        self.filters = (!empty).then(|| {
            let mut fft = Fft2::new(ly, lx);
            reference_filters_normed(
                &mut fft,
                &reference,
                ly,
                lx,
                settings.spatial_taper,
                settings.smooth_sigma,
                settings.norm_frames,
            )
        });
        // The deformation is measured on the last pass only, after the rigid
        // shift and against the reference that shift was measured against:
        // every frame's blocks are the same blocks, which is what makes the
        // per-block shifts comparable between frames. One set of block
        // filters per worker, built once for the run — every block carries
        // its own FFT plan and scratch, so workers cannot share a set.
        self.blocks = (self.is_last_pass() && settings.nonrigid && !empty).then(|| {
            let blocks = nonrigid::make_blocks(ly, lx, settings.block_size, settings.subpixel);
            let workers = if settings.backend == Backend::SingleThread {
                1
            } else {
                crate::par::workers()
            };
            let sets = nonrigid::filter_sets(
                &reference,
                lx,
                &blocks,
                settings.spatial_taper,
                settings.smooth_sigma,
                workers,
            );
            (blocks, sets)
        });
        self.reference = reference;
        self.shifts.clear();
        self.shifts.reserve(self.frames);
        self.fields.clear();
        self.fields_from = 0;
        self.built = true;
        Some(())
    }

    /// The frames [`measure`](Self::measure) wants next, or `None` when this
    /// pass has measured them all (or has no reference yet).
    ///
    /// Batches are the session's to choose, not the feeder's: a batch is the
    /// window `smooth_sigma_time` smooths across, so where they fall is part
    /// of the answer.
    pub fn next_batch(&self) -> Option<Range<usize>> {
        let t0 = self.shifts.len();
        (!self.stopped && self.built && t0 < self.frames)
            .then(|| t0..(t0 + self.batch).min(self.frames))
    }

    /// The batch `frames` has to be, checked: the one
    /// [`next_batch`](Self::next_batch) names, as far as a count and a frame
    /// size can tell.
    fn wanted<F: AsRef<[f32]>>(&self, frames: &[F]) -> Range<usize> {
        let batch = self
            .next_batch()
            .expect("no batch is wanted now: build the reference first, or finish");
        assert_eq!(
            frames.len(),
            batch.len(),
            "measure takes as many frames as the batch next_batch names, frames {batch:?}"
        );
        let (ly, lx) = (self.ly, self.lx);
        assert!(
            frames.iter().all(|f| f.as_ref().len() == ly * lx),
            "every frame of the batch is {ly} x {lx}"
        );
        batch
    }

    /// Measure the batch [`next_batch`](Self::next_batch) named: those frames,
    /// as recorded, in order.
    ///
    /// Borrowed, never written. The rigid shift of each, against this pass's
    /// reference and on top of the earlier passes'; on the last pass the block
    /// field too, measured on the frame once that shift is applied — a block
    /// is then looking for the few pixels the tissue stretched, not the tens
    /// the animal moved.
    ///
    /// With a comb to undo or a first pass to apply, the frames are copied
    /// into a batch-sized buffer kept from batch to batch. A feeder that owns
    /// its batch saves that with [`measure_in_place`](Self::measure_in_place).
    ///
    /// # Panics
    ///
    /// When no batch is wanted, or `frames` cannot be the batch asked for: a
    /// different count, or a frame that is not `ly * lx` samples. Which frames
    /// they are is the feeder's to get right; nothing in a frame says which
    /// it is.
    pub fn measure<F: AsRef<[f32]> + Sync>(&mut self, frames: &[F]) -> Measured<'_> {
        let batch = self.wanted(frames);
        let (ly, lx) = (self.ly, self.lx);

        // Copied only when something has to be done to them first: the comb
        // undone, or a second pass's frames moved by the first's shift. A
        // frame that needs neither is measured where the caller keeps it.
        let correct = self.bidiphase != 0 || !self.earlier.is_empty();
        let (moved, fields) = if correct {
            let mut scratch = std::mem::take(&mut self.scratch);
            if scratch.len() < frames.len() {
                scratch.resize_with(frames.len(), Vec::new);
            }
            for ((buf, frame), t) in scratch.iter_mut().zip(frames).zip(batch.clone()) {
                buf.clear();
                buf.extend_from_slice(frame.as_ref());
                bidiphase::shift(buf, ly, lx, self.bidiphase);
                if let Some(s) = self.earlier.get(t) {
                    roll(buf, ly, lx, s.dy, s.dx);
                }
            }
            let measured = self.measure_on(&scratch[..frames.len()]);
            self.scratch = scratch;
            measured
        } else {
            self.measure_on(frames)
        };
        self.record(batch, moved, fields)
    }

    /// [`measure`](Self::measure), with the batch's own buffers as the
    /// working space: the frames are corrected where they are, measured, and
    /// handed back as recorded, bit for bit.
    ///
    /// For a feeder that owns its batch — reads it into buffers of its own —
    /// and so would otherwise hold it twice whenever there is a comb to undo
    /// or a first pass to apply: once as read, and once as `measure`'s
    /// corrected copy. Here the comb is undone in place, keeping aside the few
    /// columns of every other line that it slides over, and a second pass's
    /// frames are rolled by the first pass's shift; after measuring, the roll
    /// is reversed — a roll is a permutation, so exactly — and the comb put
    /// back with what was kept. The same shifts as `measure`, on the same
    /// frames.
    ///
    /// # Panics
    ///
    /// As [`measure`](Self::measure) does. A panic while measuring leaves the
    /// frames corrected (the comb undone, a second pass's roll applied) rather
    /// than as recorded: a caller that catches it must not use the batch.
    pub fn measure_in_place<F: AsRef<[f32]> + AsMut<[f32]> + Sync>(
        &mut self,
        frames: &mut [F],
    ) -> Measured<'_> {
        let batch = self.wanted(frames);
        let (ly, lx, comb) = (self.ly, self.lx, self.bidiphase);
        let mut covered = std::mem::take(&mut self.covered);
        covered.clear();
        for (frame, t) in frames.iter_mut().zip(batch.clone()) {
            let frame = frame.as_mut();
            bidiphase::shift_keeping(frame, ly, lx, comb, &mut covered);
            if let Some(s) = self.earlier.get(t) {
                roll(frame, ly, lx, s.dy, s.dx);
            }
        }
        let (moved, fields) = self.measure_on(&*frames);
        // Back as recorded, in the reverse order.
        let per = bidiphase::covered(ly, lx, comb);
        for (k, (frame, t)) in frames.iter_mut().zip(batch.clone()).enumerate() {
            let frame = frame.as_mut();
            if let Some(s) = self.earlier.get(t) {
                roll(frame, ly, lx, -s.dy, -s.dx);
            }
            bidiphase::unshift(frame, ly, lx, comb, &covered[k * per..(k + 1) * per]);
        }
        self.covered = covered;
        self.record(batch, moved, fields)
    }

    /// Take a measured batch into the run: its totals over the passes, and on
    /// the last pass its block fields.
    fn record(
        &mut self,
        batch: Range<usize>,
        moved: Vec<Shift>,
        fields: Option<Vec<Vec<BlockShift>>>,
    ) -> Measured<'_> {
        // What a caller applies is the total: a second pass measures what is
        // left after the first, and the residual alone would leave the movie
        // moving.
        for (s, t) in moved.iter().zip(batch.clone()) {
            let before = self.earlier.get(t).copied().unwrap_or(STILL);
            self.shifts.push(Shift {
                dy: before.dy + s.dy,
                dx: before.dx + s.dx,
                corr: s.corr,
            });
        }
        if let Some(fields) = fields {
            // The batch before is moved by now, or never will be: a feeder
            // moves a batch's planes before it measures the next.
            if !self.keep_fields {
                self.fields.clear();
                self.fields_from = batch.start;
            }
            self.fields.extend(fields);
        }
        Measured {
            session: self,
            frames: batch,
        }
    }

    /// This pass's shift for each of `frames`, and their block fields on the
    /// last pass. The frames are as this pass measures them: corrected, and
    /// moved by the earlier passes.
    fn measure_on<G: AsRef<[f32]> + Sync>(
        &mut self,
        frames: &[G],
    ) -> (Vec<Shift>, Option<Vec<Vec<BlockShift>>>) {
        let (ly, lx) = (self.ly, self.lx);
        let Some(filters) = &self.filters else {
            // A frame with no pixels has nothing to correlate.
            return (vec![STILL; frames.len()], None);
        };
        let shifts = pipeline::measure_batch(ly, lx, filters, frames, &self.settings);
        // On frames moved by this pass's shift, which with the earlier
        // passes' already applied is the total: whole-pixel rolls compose
        // exactly.
        let fields = self.blocks.as_mut().map(|(blocks, sets)| {
            nonrigid::measure_blocks_batch(
                sets,
                ly,
                lx,
                blocks,
                frames,
                &shifts,
                &BlockSearch {
                    maxregshift_nr: self.settings.maxregshift_nr,
                    snr_thresh: self.settings.snr_thresh,
                    subpixel: self.settings.subpixel,
                    clip: filters.clip,
                },
            )
        });
        (shifts, fields)
    }

    /// Each frame's total shift, in order, for every frame the pass under way
    /// has measured: the earlier passes' and this one's together. Final on
    /// the last pass ([`is_last_pass`](Self::is_last_pass)).
    pub fn shifts(&self) -> &[Shift] {
        &self.shifts
    }

    /// The block grid of a non-rigid run, once the last pass's reference is
    /// built: what a [`field`](Self::field) is a shift per block of.
    pub fn blocks(&self) -> Option<&Blocks> {
        self.blocks.as_ref().map(|(blocks, _)| blocks)
    }

    /// Frame `t`'s block field, on the last pass of a non-rigid run, while the
    /// session holds it: the last batch measured, or every frame's when they
    /// are kept ([`keep_fields`](Self::keep_fields)).
    pub fn field(&self, t: usize) -> Option<&[BlockShift]> {
        self.blocks.as_ref()?;
        let i = t.checked_sub(self.fields_from)?;
        self.fields.get(i).map(Vec::as_slice)
    }

    /// The fields held, as [`Moving`] takes them.
    fn warp(&self) -> Option<Warp<'_>> {
        self.blocks.as_ref().map(|(blocks, _)| Warp {
            blocks,
            fields: &self.fields,
            first: self.fields_from,
        })
    }

    /// Move planes onto the reference, in place: `planes[i]` belongs to frame
    /// `frame_of[i]` of the recording — any channel or slice of it, which is
    /// how every channel follows the one measured.
    ///
    /// The planes are as recorded. The comb is undone first, then the shift
    /// or the warp, filled as [`Settings::border`] says; under
    /// [`Border::Crop`] as under `Constant`, at full
    /// size, since the area to cut to is known only once every frame is
    /// measured ([`Registered::apply_batch`](crate::Registered::apply_batch)
    /// cuts). Parallel unless the run is single-threaded.
    ///
    /// What [`Measured::apply`] does, here for a caller that cannot hold a
    /// `Measured` until it has the planes.
    ///
    /// # Panics
    ///
    /// Before the last pass, whose shifts are the only ones to apply; for a
    /// frame this pass has not measured yet; on a non-rigid run, for a frame
    /// whose block field the session no longer holds — one of an earlier
    /// batch, unless they are kept; and when `planes` and `frame_of` differ in
    /// length. Before any plane is moved.
    pub fn apply(&self, planes: &mut [Vec<f32>], frame_of: &[usize]) {
        assert!(
            self.built && self.is_last_pass(),
            "a pass before the last has no shifts to apply"
        );
        Moving {
            ly: self.ly,
            lx: self.lx,
            shifts: &self.shifts,
            warp: self.warp(),
            border: self.settings.border,
            bidiphase: self.bidiphase,
            cut: None,
        }
        .apply(
            planes,
            frame_of,
            self.settings.backend != Backend::SingleThread,
        );
    }

    /// The registration, once every pass is done ([`is_done`](Self::is_done)):
    /// the shifts, the reference, the bidirectional offset, the block field
    /// and which frames look bad.
    ///
    /// A non-rigid run's field is there for every frame only when it was
    /// kept ([`keep_fields`](Self::keep_fields), or under
    /// [`Border::Crop`]); otherwise the result names the
    /// grid and holds no field ([`NonRigid::shifts`]).
    ///
    /// # Panics
    ///
    /// When a pass is still to run or a batch still to measure, or a build
    /// was stopped.
    pub fn finish(self) -> Registered {
        assert!(
            !self.stopped,
            "the session was stopped while its reference was built: there is no registration"
        );
        assert!(
            self.is_done(),
            "the registration is not finished: a reference or a batch is still wanted"
        );
        let bad_frames = pipeline::bad_frames(&self.shifts, self.ly, self.lx, &self.settings);
        let fields = if self.keep_fields {
            self.fields
        } else {
            Vec::new()
        };
        Registered {
            bad_frames,
            nonrigid: self.blocks.map(|(blocks, _)| NonRigid {
                blocks,
                shifts: fields,
            }),
            shifts: self.shifts,
            reference: self.reference,
            bidiphase: self.bidiphase,
            border: self.settings.border,
            parallel: self.settings.backend != Backend::SingleThread,
            area: Default::default(),
        }
    }
}

/// A batch [`Session::measure`] has just measured: a view of the session,
/// for a feeder that moves the batch's planes before it measures the next.
pub struct Measured<'s> {
    session: &'s Session,
    frames: Range<usize>,
}

impl<'s> Measured<'s> {
    /// Which frames of the recording the batch was.
    pub fn frames(&self) -> Range<usize> {
        self.frames.clone()
    }

    /// Whether this is the last pass, so that the shifts and fields are the
    /// frames' own and their planes can be moved now.
    pub fn is_final(&self) -> bool {
        self.session.is_last_pass()
    }

    /// Each frame's total shift so far, in order: final once
    /// [`is_final`](Self::is_final).
    pub fn shifts(&self) -> &'s [Shift] {
        &self.session.shifts[self.frames.clone()]
    }

    /// Each frame's block field, on the last pass of a non-rigid run.
    pub fn fields(&self) -> Option<&'s [Vec<BlockShift>]> {
        let s = self.session;
        s.blocks
            .as_ref()
            .map(|_| &s.fields[self.frames.start - s.fields_from..self.frames.end - s.fields_from])
    }

    /// Move planes of this batch's frames onto the reference, in place:
    /// [`Session::apply`], which says how and when it panics.
    pub fn apply(&self, planes: &mut [Vec<f32>], frame_of: &[usize]) {
        self.session.apply(planes, frame_of);
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;

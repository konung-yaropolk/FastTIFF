// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! A stack registration fed a batch of frames at a time.
//!
//! # Why the frames come in batches
//!
//! A registration used to take the whole movie at once, as `f32`, and then
//! made a second copy of it for the temporal moving average — even when the
//! window was one frame and nothing was averaged. At 2048 x 2048 for a thousand
//! frames that is 16 GiB of movie and 16 GiB of copy before anything is fitted,
//! and in a browser, which has 4 GiB in all, about 128 frames at that size. Yet
//! nothing in the method needs the movie resident: every frame is fitted
//! independently of the others once the target is fixed, so a batch of them is
//! all that has to be in memory at once.
//!
//! So the [`Session`] asks for frames and the caller hands them over, a batch
//! at a time, from wherever they live — memory ([`crate::register_stack`], the
//! C interface's `tr_register`) or a file read plane by plane (the viewer's
//! built-in). It is the one place the order of the work is decided: which
//! frames build the target and which are fitted, how the moving average
//! reaches across a batch's edges, how large a batch is. Every entry point
//! only reads what it is asked for and reports progress, so none of them can
//! drift from another in what they compute.
//!
//! # What it asks for, and why some frames twice
//!
//! [`Reference::Previous`] fits each frame against the one before it, so one
//! pass over the stack does it: the session keeps the last frame of a batch to
//! fit the first frame of the next against.
//!
//! [`Reference::First`] and [`Reference::Mean`] fit every frame against a
//! target built from the frames themselves — the first
//! [`n_frames`](Settings::n_frames) of them averaged, or all of them — and
//! nothing can be fitted until the target is complete. Holding back the frames
//! that arrive before then would hold the movie again (all of it, for the
//! mean), so instead the session asks for the target's frames first, in a pass
//! of their own ([`Pass::Reference`]), and then for the stack ([`Pass::Fit`])
//! from the first frame fitted, or the first its window reaches back to.
//!
//! What the two passes share is read twice. Under the first-frame default
//! with no smoothing that is nothing: the target is frame 0, and the fits
//! start at frame 1. A longer target, or a moving average, makes it the
//! target's frames and their windows; the mean makes it every frame. That
//! costs reads, not memory — but it does mean that a source which can be read
//! only once, a camera or a pipe, can feed only [`Reference::Previous`] or
//! that default; [`Session::plan`] says which frames each pass reads.
//!
//! # The moving average, without the movie
//!
//! The frames that are measured are the stack smoothed over time
//! ([`Settings::moving_average`]): frame `t` measured is the mean of frames
//! `t - h ..= t + h`, cut off at the ends of the stack. A batch's last few
//! frames therefore need frames the next batch brings, and its first few need
//! frames the last one had. The session keeps the `2h` frames before each
//! batch in a ring, and fits a frame once the batch that completes its window
//! has arrived. The frames themselves are borrowed for the length of
//! [`Session::feed`], never copied whole; what is copied is what has to
//! outlive a batch: the ring's frames, and under [`Reference::Previous`] the
//! one frame the next batch is fitted against.
//!
//! The ring is the window's size, not the batch's: a window of 31 frames keeps
//! 30, which at 2048 x 2048 is more than a CPU batch's 256 MiB, and a window
//! as wide as the stack keeps the stack. Wider than that changes nothing —
//! every frame's window is then the whole stack — so `h` stops at `n - 1`.
//!
//! # Frames the GPU could not fit
//!
//! A card with no room for a batch fails the batch rather than the process
//! (see the `gpu` module): its frames are not fitted, and the device notes why
//! in one slot per process (`gpu::take_trouble`). The session takes that
//! reason after each batch it sends to the device, with the batches of every
//! session taking turns so that each takes only its own, and a run in which
//! the device failed frames finishes as [`Error::Device`] — not as frames that
//! will not move, which a caller who asked for the GPU could not tell from a
//! registration that found nothing to correct.
//!
//! # The same answers
//!
//! Every sum is taken in the order it always was: a smoothed frame's over its
//! window in frame order, the target's over the frames in frame order, each in
//! `f64`. Fits do not depend on each other or on which batch they share, on
//! the CPU or on the GPU, whose lanes are scored separately. So a session
//! gives the matrices the whole-movie driver gave, bit for bit, whatever the
//! batches — `session_oracle_tests.rs` keeps that driver to hold it to.

use std::fmt;
use std::ops::{Range, RangeInclusive};

use crate::{
    optimize, par, Backend, Border, Matrix, Reference, Registered, Settings, Transformation,
};

/// Pixels per piece of a stack-wide sum. Fixed, like the optimiser's row
/// pieces: each pixel's sum runs over the frames in order on whichever thread
/// takes its piece, so the result does not depend on how many there are.
const PIXELS_PER_CHUNK: usize = 4096;

/// The most a CPU batch holds of frames, and of the smoothed frames made from
/// them, in the host's memory. A batch is a few frames per core so the uneven
/// ones do not leave cores idle; at 2048 x 2048 on sixteen cores that was 64
/// frames, a gigabyte, held by every caller, for no gain in speed — each
/// frame's own work is spread across the cores too. The suite2p built-in holds
/// its batches to the same figure. A GPU batch is bounded by what the device
/// takes (`gpu::lanes_for`, against the device's own budget, which the `gpu`
/// module also calls `BATCH_MEMORY`), and that is less.
const HOST_BATCH_MEMORY: usize = 256 << 20;

/// A fit's two frames: what it is fitted against, and what is fitted.
type Pair<'a> = (&'a [f32], &'a [f32]);

/// Frames handed to a [`Session`], borrowed for the length of the call.
#[derive(Clone, Copy, Debug)]
pub enum Batch<'a> {
    /// Frames one after another in one buffer, each `ly * lx` samples: a C
    /// caller's stack, or any run of it.
    Packed(&'a [f32]),
    /// One buffer per frame, each `ly * lx` samples.
    Split(&'a [Vec<f32>]),
    /// One borrowed slice per frame, each `ly * lx` samples: frames that live
    /// in buffers the caller does not own as `Vec`s — planes a host lends, a
    /// mapped file, another language's arrays — fed where they lie.
    Slices(&'a [&'a [f32]]),
}

impl<'a> Batch<'a> {
    /// How many frames of `pixels` samples this is, or why it is not frames
    /// of that size.
    fn count(&self, pixels: usize) -> Result<usize, Error> {
        match *self {
            Batch::Packed(data) => {
                if data.len() % pixels != 0 {
                    return Err(Error::Batch(format!(
                        "{} samples is not a whole number of {pixels}-sample frames",
                        data.len()
                    )));
                }
                Ok(data.len() / pixels)
            }
            Batch::Split(frames) => each_frame(frames.iter().map(|f| f.len()), pixels),
            Batch::Slices(frames) => each_frame(frames.iter().map(|f| f.len()), pixels),
        }
    }

    /// Frame `i` of the batch.
    fn frame(&self, i: usize, pixels: usize) -> &'a [f32] {
        match *self {
            Batch::Packed(data) => &data[i * pixels..(i + 1) * pixels],
            Batch::Split(frames) => &frames[i],
            Batch::Slices(frames) => frames[i],
        }
    }

    /// The batch's frames `frames`, borrowed in turn.
    fn slice(&self, frames: Range<usize>, pixels: usize) -> Batch<'a> {
        match *self {
            Batch::Packed(data) => Batch::Packed(&data[frames.start * pixels..frames.end * pixels]),
            Batch::Split(all) => Batch::Split(&all[frames]),
            Batch::Slices(all) => Batch::Slices(&all[frames]),
        }
    }
}

/// How many frames there are of `lengths`, or which one is not `pixels` long.
fn each_frame(lengths: impl Iterator<Item = usize>, pixels: usize) -> Result<usize, Error> {
    let mut count = 0;
    for (i, len) in lengths.enumerate() {
        if len != pixels {
            return Err(Error::Batch(format!(
                "frame {i} of the batch has {len} samples, not {pixels}"
            )));
        }
        count += 1;
    }
    Ok(count)
}

/// What the frames of a batch are for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pass {
    /// Building the target of [`Reference::First`] or [`Reference::Mean`]:
    /// the frames are averaged into it and nothing is fitted. Cheap next to a
    /// fit; a caller dividing a progress bar should weigh it as reading only.
    Reference,
    /// Fitting each frame against its target.
    Fit,
}

/// The frames a [`Session`] asks for next, and what they are for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Wanted {
    /// Which frames, by index in the stack: at most [`Session::batch_len`].
    pub frames: Range<usize>,
    /// What they are for: the target's frames, or the ones fitted.
    pub pass: Pass,
}

/// Every frame a [`Session`] will ask for, pass by pass: what a caller needs
/// to divide a progress bar before the first batch.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Plan {
    /// The frames [`Pass::Reference`] reads, empty when the mode builds no
    /// target from them ([`Reference::Previous`]).
    pub reference: Range<usize>,
    /// The frames [`Pass::Fit`] reads.
    pub fit: Range<usize>,
}

/// Why a registration did not produce a result.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    /// [`Reference::Previous`] with [`Transformation::Bilinear`]: the
    /// composition of two bilinear maps is not bilinear, so there is no chain
    /// to build. Refused rather than silently substituted, as pystackreg does.
    Unchainable,
    /// The backend cannot run frames of this size here; the words are
    /// [`Backend::unavailable_reason`]'s. Refused rather than downgraded: a
    /// run that said GPU and used the processor would be indistinguishable
    /// from a slow one.
    Unavailable(String),
    /// The progress callback asked to stop.
    Cancelled,
    /// The GPU failed batches of frames — no room on it for their levels, or
    /// a device that stopped answering. Refused as a whole rather than handed
    /// back with those frames unmoved; a run on the processor needs no GPU
    /// memory.
    Device {
        /// Frames not fitted, the device's and any the fit gave up on.
        failed: usize,
        /// Frames in the stack.
        frames: usize,
        /// The device's reason, from the first batch it failed.
        why: String,
    },
    /// A batch that is not the frames the session asked for: frames of the
    /// wrong size, an empty batch, more than is left of the pass, or a batch
    /// after the last.
    Batch(String),
    /// [`Session::finish`] before every frame the session asked for was fed.
    Unfinished,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unchainable => f.write_str(
                "a bilinear warp cannot be chained onto another, so it cannot be registered \
                 against the previous frame",
            ),
            Error::Unavailable(why) => f.write_str(why),
            Error::Cancelled => f.write_str("the registration was stopped"),
            Error::Device {
                failed,
                frames,
                why,
            } => write!(
                f,
                "{failed} of {frames} frames could not be registered on the GPU: {why}"
            ),
            Error::Batch(why) => write!(f, "not the frames the registration asked for: {why}"),
            Error::Unfinished => {
                f.write_str("the registration was finished before every frame was fed")
            }
        }
    }
}

impl std::error::Error for Error {}

/// One pass's frames: the ones read, and the smoothed ones made from them.
#[derive(Clone, Debug)]
struct Stage {
    pass: Pass,
    /// The frames the caller is asked for: `makes` widened by the window at
    /// each end, cut off at the stack's.
    reads: Range<usize>,
    /// The smoothed frames the pass uses.
    makes: Range<usize>,
}

/// A registration in progress: ask it what it wants with [`wanted`], hand it
/// those frames with [`feed`], and when it wants nothing more, [`finish`] it.
///
/// ```
/// use turboreg_registration::{Batch, Session, Settings};
///
/// # let (ly, lx) = (32, 32);
/// # let movie: Vec<Vec<f32>> = (0..4).map(|t| (0..ly * lx).map(|i| ((i + t) % 7) as f32).collect()).collect();
/// # let read = |t: usize| movie[t].clone();
/// let mut session = Session::new(ly, lx, movie.len(), &Settings::default()).unwrap();
/// let mut batch = Vec::new();
/// while let Some(wanted) = session.wanted() {
///     batch.clear();
///     batch.extend(wanted.frames.map(read)); // from memory, a file, ...
///     session.feed(Batch::Split(&batch)).unwrap();
/// }
/// let registered = session.finish().unwrap();
/// assert_eq!(registered.matrices.len(), 4);
/// ```
///
/// On [`Backend::Gpu`] each batch takes the process's
/// turn on the device, a lock held while the batch's fits run on the thread
/// pool. Feed GPU sessions from ordinary threads, not from inside rayon tasks:
/// a pool thread that holds the turn can pick up another task's session while
/// it waits, and that session's batch would wait on the turn for ever.
///
/// [`wanted`]: Session::wanted
/// [`feed`]: Session::feed
/// [`finish`]: Session::finish
pub struct Session {
    settings: Settings,
    ly: usize,
    lx: usize,
    /// Frames in the stack.
    n: usize,
    /// Half the moving-average window: frame `t` measured is the mean of
    /// frames `t - half ..= t + half`, cut off at the ends.
    half: usize,
    /// The batch rule: see [`Session::batch_len`].
    per: usize,
    /// The first frame fitted: frame 0 is the target itself under `First`,
    /// and the start of the chain under `Previous`.
    first_target: usize,
    /// Frames averaged into the target under `First` and `Mean`.
    take: usize,
    /// Within a frame: the smoothing, the target and each fit's own sums.
    threaded: bool,
    /// Across frames: separate fits at the same time.
    parallel: bool,
    /// The passes still to come, the current one first.
    stages: Vec<Stage>,
    /// The next frame the current pass expects.
    next: usize,
    /// The next smoothed frame the current pass makes.
    made: usize,
    /// The last `2 * half` frames fed, frame `i` in slot `i % (2 * half)`:
    /// what a smoothed frame's window takes from before its batch.
    ring: Vec<Vec<f32>>,
    /// The smoothed frames of one piece of a batch, when there is smoothing
    /// to do; without it the batch's own frames are used where they are.
    smoothed: Vec<Vec<f32>>,
    /// Under `Previous`: the frame, smoothed, that the next one fitted is
    /// fitted against.
    previous: Vec<f32>,
    /// The target's running sum, per pixel, during [`Pass::Reference`].
    sum: Vec<f64>,
    /// Where each piece lists its pairs, empty between pieces: kept so that a
    /// batch after the first allocates no list (see [`recycle`]).
    spare_pairs: Vec<Pair<'static>>,
    /// Why the device failed a batch, from the first it failed.
    trouble: Option<String>,
    reference: Option<Vec<f32>>,
    matrices: Vec<Matrix>,
    failed: Vec<usize>,
}

impl Session {
    /// A registration of `frames` frames of `ly` rows by `lx` columns.
    ///
    /// Refused for a combination the method cannot run ([`Error::Unchainable`],
    /// whatever the stack) and for a backend that cannot run here
    /// ([`Error::Unavailable`]) — the first time a GPU is asked about, that
    /// opens the device. A stack with no frames, or frames with no pixels,
    /// asks nothing of a backend: it asks for no frames and finishes with no
    /// matrices, on any backend, without opening a device.
    pub fn new(ly: usize, lx: usize, frames: usize, settings: &Settings) -> Result<Session, Error> {
        if settings.reference == Reference::Previous
            && settings.transformation == Transformation::Bilinear
        {
            return Err(Error::Unchainable);
        }
        let n = if ly == 0 || lx == 0 { 0 } else { frames };
        if n > 0 {
            if let Some(why) = settings.backend.unavailable_reason(ly, lx) {
                return Err(Error::Unavailable(why));
            }
        }
        // An odd window, whatever is asked for: an even one is off-centre by
        // half a frame, which is a drift the registration would faithfully
        // remove. With one frame there is nothing to average. And no wider
        // than the stack: past `n - 1` every frame's window is the whole stack
        // already, cut off at its ends, so the answers are the same — but the
        // ring is `2h` slots, and a window of 2^40 (which nothing upstream
        // refuses) asked for 2^41 of them, an allocation that aborts.
        let half = if n < 2 {
            0
        } else {
            ((settings.moving_average.max(1) | 1) / 2).min(n - 1)
        };
        let take = match settings.reference {
            Reference::First => settings.n_frames.clamp(1, n.max(1)),
            Reference::Mean => n,
            Reference::Previous => 0,
        };
        let first_target = usize::from(settings.reference != Reference::Mean);

        let widened = |makes: Range<usize>| -> Range<usize> {
            if makes.is_empty() {
                return makes.start..makes.start;
            }
            makes.start.saturating_sub(half)..(makes.end + half).min(n)
        };
        let mut stages = Vec::new();
        if take > 0 && n > 0 {
            stages.push(Stage {
                pass: Pass::Reference,
                reads: widened(0..take),
                makes: 0..take,
            });
        }
        if first_target < n {
            // Under `Previous` the first frame fitted needs the one before it.
            let from = if settings.reference == Reference::Previous {
                first_target - 1
            } else {
                first_target
            };
            stages.push(Stage {
                pass: Pass::Fit,
                reads: widened(from..n),
                makes: from..n,
            });
        }
        let (next, made) = stages
            .first()
            .map_or((0, 0), |s| (s.reads.start, s.makes.start));
        Ok(Session {
            settings: *settings,
            ly,
            lx,
            n,
            half,
            // Nothing to register needs no batch rule, and working one out
            // for the GPU opens the device.
            per: if n == 0 {
                1
            } else {
                batch_len(settings, lx, ly)
            },
            first_target,
            take,
            threaded: settings.backend.threaded(),
            parallel: settings.backend.parallel_for(settings.reference),
            stages,
            next,
            made,
            ring: Vec::new(),
            smoothed: Vec::new(),
            previous: Vec::new(),
            sum: Vec::new(),
            spare_pairs: Vec::new(),
            trouble: None,
            reference: None,
            matrices: vec![Matrix::identity(); n],
            failed: Vec::new(),
        })
    }

    /// How many frames a batch holds: what [`wanted`](Self::wanted) asks for
    /// at a time, and the most frames fitted at once whatever is fed.
    ///
    /// Decided here, once, so every caller batches alike. On the GPU, as many
    /// frames as the device holds at once (`gpu::lanes_for`), which is also
    /// the unit it works in. On the CPU, enough to keep every thread busy with
    /// some to spare for the uneven frames, within what memory allows a batch
    /// of frames this size; on one thread, one frame, so progress moves as
    /// each finishes. A scheduling rule only: no answer depends on it.
    pub fn batch_len(&self) -> usize {
        self.per
    }

    /// Every frame this session will ask for, pass by pass.
    pub fn plan(&self) -> Plan {
        let reads = |pass: Pass| {
            self.stages
                .iter()
                .find(|s| s.pass == pass)
                .map_or(0..0, |s| s.reads.clone())
        };
        Plan {
            reference: reads(Pass::Reference),
            fit: reads(Pass::Fit),
        }
    }

    /// The frames to feed next, or `None` once there are none left and the
    /// session is ready to [`finish`](Self::finish).
    pub fn wanted(&self) -> Option<Wanted> {
        let stage = self.stages.first()?;
        Some(Wanted {
            frames: self.next..(self.next + self.per).min(stage.reads.end),
            pass: stage.pass,
        })
    }

    /// Hand over the next frames: the ones [`wanted`](Self::wanted) names, or
    /// fewer of them from its start, or more, up to the end of its pass. They
    /// are borrowed for the call, not kept.
    ///
    /// Whatever is fed, at most [`batch_len`](Self::batch_len) frames are
    /// fitted at once, so what one call holds besides the batch does not grow
    /// with it.
    pub fn feed(&mut self, batch: Batch<'_>) -> Result<(), Error> {
        let Some(stage) = self.stages.first().cloned() else {
            return Err(Error::Batch(
                "the registration has every frame it asked for".to_string(),
            ));
        };
        let pixels = self.ly * self.lx;
        let fed = batch.count(pixels)?;
        let (from, to) = (self.next, stage.reads.end);
        if fed == 0 || fed > to - from {
            return Err(Error::Batch(format!(
                "{fed} frame(s) from frame {from}, where this pass wants 1 to {}",
                to - from
            )));
        }
        let end = from + fed;
        // The smoothed frames whose windows this batch completes.
        let ready = if end == self.n {
            stage.makes.end
        } else {
            end.saturating_sub(self.half).min(stage.makes.end)
        }
        .max(self.made);
        let mut at = self.made;
        while at < ready {
            let piece = at..(at + self.per).min(ready);
            self.piece(stage.pass, batch, from, piece.clone());
            at = piece.end;
        }
        self.made = ready;
        self.next = end;
        if end == to {
            self.end_pass();
            return Ok(());
        }
        // Without smoothing the batch's own frames were used where they are,
        // so the one the next batch's first frame is fitted against has to be
        // kept past this call.
        if self.half == 0
            && stage.pass == Pass::Fit
            && self.settings.reference == Reference::Previous
        {
            let last = batch.frame(ready - 1 - from, pixels);
            self.previous.clear();
            self.previous.extend_from_slice(last);
        }
        self.keep_window(batch, from, end);
        Ok(())
    }

    /// The result, once every frame asked for has been fed — or
    /// [`Error::Device`] when the GPU failed frames of it.
    pub fn finish(self) -> Result<Registered, Error> {
        if !self.stages.is_empty() {
            return Err(Error::Unfinished);
        }
        // A batch the device failed puts every frame of it in `failed`, so a
        // reason of this run's comes with frames lost. Both are checked all
        // the same, as each entry point checked them before the session took
        // this over: a reason with nothing lost is not this run's, but one a
        // direct `register_pair` left in the slot during a batch's turn.
        if let Some(why) = self.trouble {
            if !self.failed.is_empty() {
                return Err(Error::Device {
                    failed: self.failed.len(),
                    frames: self.n,
                    why,
                });
            }
        }
        Ok(Registered {
            matrices: self.matrices,
            reference: self.reference,
            failed: self.failed,
            threaded: self.threaded,
            border: Border::default(),
            area: Default::default(),
        })
    }

    /// Feed the whole stack from memory, batch by batch, and finish.
    ///
    /// `on_progress` is called with the fraction of the frames fitted before
    /// each batch and once at the end, and returns `false` to stop, which is
    /// [`Error::Cancelled`].
    pub fn run(
        mut self,
        movie: Batch<'_>,
        on_progress: &mut dyn FnMut(f32) -> bool,
    ) -> Result<Registered, Error> {
        if self.n == 0 {
            return self.finish();
        }
        let pixels = self.ly * self.lx;
        let frames = movie.count(pixels)?;
        if frames != self.n {
            return Err(Error::Batch(format!(
                "a stack of {frames} frames for a registration of {}",
                self.n
            )));
        }
        while let Some(wanted) = self.wanted() {
            if !on_progress(self.fitted()) {
                return Err(Error::Cancelled);
            }
            self.feed(movie.slice(wanted.frames, pixels))?;
        }
        if !on_progress(1.0) {
            return Err(Error::Cancelled);
        }
        self.finish()
    }

    /// The fraction of the frames to fit that are fitted.
    fn fitted(&self) -> f32 {
        match self.stages.first() {
            None => 1.0,
            Some(stage) if stage.pass == Pass::Reference => 0.0,
            Some(stage) => (self.made - stage.makes.start) as f32 / stage.makes.len().max(1) as f32,
        }
    }

    /// Smoothed frames `piece` — at most a batch of them, every one's window
    /// complete in the ring and `batch` (whose first frame is `from`) — made,
    /// and averaged into the target or fitted.
    fn piece(&mut self, pass: Pass, batch: Batch<'_>, from: usize, piece: Range<usize>) {
        let pixels = self.ly * self.lx;
        if self.half > 0 {
            self.smooth(batch, from, piece.clone());
        }
        let half = self.half;
        let smoothed = &self.smoothed;
        let previous = &self.previous;
        let start = piece.start;
        // Frame `t` smoothed: in place in the batch without smoothing, else
        // made into `smoothed`; the one before the piece, kept from the last.
        let frame = |t: usize| -> &[f32] {
            if half == 0 {
                if t >= from {
                    batch.frame(t - from, pixels)
                } else {
                    previous
                }
            } else if t >= start {
                &smoothed[t - start]
            } else {
                previous
            }
        };
        match pass {
            Pass::Reference => {
                if self.sum.len() != pixels {
                    self.sum = vec![0.0; pixels];
                }
                for t in piece.clone() {
                    add_into(&mut self.sum, frame(t), self.threaded);
                }
            }
            Pass::Fit => {
                let targets = piece.start.max(self.first_target)..piece.end;
                if !targets.is_empty() {
                    let mut pairs = recycle(std::mem::take(&mut self.spare_pairs));
                    pairs.extend(targets.clone().map(|t| {
                        let against = match &self.reference {
                            Some(r) => &r[..],
                            None => frame(t - 1),
                        };
                        (against, frame(t))
                    }));
                    let (fits, trouble) =
                        fit_pairs(&pairs, self.lx, self.ly, &self.settings, self.parallel);
                    self.spare_pairs = recycle(pairs);
                    if self.trouble.is_none() {
                        self.trouble = trouble;
                    }
                    self.record(targets, fits);
                }
            }
        }
        if half > 0 && self.settings.reference == Reference::Previous {
            // Keep the piece's last frame for the next piece's first; the
            // buffer it came from takes the old one's place, so nothing is
            // copied or allocated.
            std::mem::swap(&mut self.previous, &mut self.smoothed[piece.len() - 1]);
        }
    }

    /// Make smoothed frames `piece` into `smoothed`, each the mean of its
    /// window: from the ring before `from`, from `batch` after.
    fn smooth(&mut self, batch: Batch<'_>, from: usize, piece: Range<usize>) {
        let pixels = self.ly * self.lx;
        if self.smoothed.len() < piece.len() {
            self.smoothed.resize_with(piece.len(), Vec::new);
        }
        let ring = &self.ring;
        let slots = ring.len().max(1);
        let raw = |i: usize| -> &[f32] {
            if i >= from {
                batch.frame(i - from, pixels)
            } else {
                &ring[i % slots]
            }
        };
        let (half, n, threaded, start) = (self.half, self.n, self.threaded, piece.start);
        par::for_each_chunk_mut(&mut self.smoothed[..piece.len()], 1, threaded, |j, out| {
            let out = &mut out[0];
            // Sized once; afterwards every buffer here is a frame long.
            out.resize(pixels, 0.0);
            average_into(out, window(start + j, half, n), &raw, threaded);
        });
    }

    /// Keep the frames a later batch's windows reach back to: the last
    /// `2 * half` fed, `batch` being frames `from..end`.
    fn keep_window(&mut self, batch: Batch<'_>, from: usize, end: usize) {
        let slots = 2 * self.half;
        if slots == 0 {
            return;
        }
        let pixels = self.ly * self.lx;
        if self.ring.len() < slots {
            self.ring.resize_with(slots, Vec::new);
        }
        for i in end.saturating_sub(slots).max(from)..end {
            let slot = &mut self.ring[i % slots];
            slot.clear();
            slot.extend_from_slice(batch.frame(i - from, pixels));
        }
    }

    /// Fits for frames `targets`, in order, into the matrices.
    fn record(&mut self, targets: Range<usize>, fits: Vec<Option<optimize::Fit>>) {
        for (t, fit) in targets.zip(fits) {
            match (self.settings.reference, fit) {
                // Composed down the stack, so frame t's matrix is expressed
                // against frame 0: frame t - 1's, followed by the step from
                // t - 1 to t. A frame that could not be fitted keeps its
                // predecessor's matrix, so the chain carries on past it
                // rather than snapping back to the identity.
                (Reference::Previous, Some(f)) => {
                    self.matrices[t] = self.matrices[t - 1].compose(&f.matrix).unwrap_or(f.matrix);
                }
                (Reference::Previous, None) => {
                    self.failed.push(t);
                    self.matrices[t] = self.matrices[t - 1];
                }
                (_, Some(f)) => self.matrices[t] = f.matrix,
                (_, None) => self.failed.push(t),
            }
        }
    }

    /// The current pass is over: the target is complete after
    /// [`Pass::Reference`], and the next pass starts from its own first frame.
    fn end_pass(&mut self) {
        let done = self.stages.remove(0);
        if done.pass == Pass::Reference {
            let take = self.take as f64;
            let reference = self.sum.iter().map(|&s| (s / take) as f32).collect();
            self.reference = Some(reference);
            // Two frames' worth of `f64` that nothing needs any more.
            self.sum = Vec::new();
        }
        if let Some(stage) = self.stages.first() {
            self.next = stage.reads.start;
            self.made = stage.makes.start;
        } else {
            // Nothing reads them again.
            self.ring = Vec::new();
            self.smoothed = Vec::new();
            self.previous = Vec::new();
            self.spare_pairs = Vec::new();
        }
    }
}

/// Fit `pairs`, and say why the device failed any of them.
///
/// On the GPU the batch takes its turn on the device (`gpu::attributed`), so
/// the reason the device notes is this batch's and no other session's; on the
/// CPU there is never one.
fn fit_pairs(
    pairs: &[Pair<'_>],
    lx: usize,
    ly: usize,
    settings: &Settings,
    parallel: bool,
) -> (Vec<Option<optimize::Fit>>, Option<String>) {
    #[cfg(feature = "gpu")]
    if settings.backend == Backend::Gpu {
        return crate::gpu::attributed(|| {
            optimize::register_pairs(pairs, lx, ly, settings, parallel)
        });
    }
    (
        optimize::register_pairs(pairs, lx, ly, settings, parallel),
        None,
    )
}

/// `spare`, emptied, as a list of borrows of another lifetime — in the same
/// allocation.
///
/// A piece's pairs borrow the batch, which lives only as long as one
/// [`Session::feed`], so the list cannot be kept full; kept empty it borrows
/// nothing, and this changes only what it may hold. Collecting a `Vec`'s own
/// `into_iter` into a `Vec` of an item the same size is done in place by the
/// standard library, so the capacity carries over and a batch after the first
/// allocates no list. Were that ever not so, this would allocate, as building
/// the list afresh does: never wrong, only slower, and
/// `a_session_holds_a_batch_not_the_stack` would say so.
fn recycle<'a, 'b>(mut spare: Vec<Pair<'a>>) -> Vec<Pair<'b>> {
    spare.clear();
    spare
        .into_iter()
        .map(|_| -> Pair<'b> { unreachable!("the list was emptied") })
        .collect()
}

/// The frames frame `t` is averaged over: a centred window of `2 * half + 1`,
/// cut off at the ends of a stack of `n`.
fn window(t: usize, half: usize, n: usize) -> RangeInclusive<usize> {
    t.saturating_sub(half)..=(t + half).min(n - 1)
}

/// `out` as the mean of frames `frames`, each pixel's sum taken over them in
/// order, in `f64`.
///
/// Parallel across pixels, never across frames, so the answer is the same on
/// one thread or many; each piece's sums sit on the stack, so nothing is
/// allocated.
fn average_into<'f>(
    out: &mut [f32],
    frames: RangeInclusive<usize>,
    raw: &(impl Fn(usize) -> &'f [f32] + Sync),
    parallel: bool,
) {
    let count = (frames.end() - frames.start() + 1) as f64;
    par::for_each_chunk_mut(out, PIXELS_PER_CHUNK, parallel, |c, piece| {
        let base = c * PIXELS_PER_CHUNK;
        let mut acc = [0.0f64; PIXELS_PER_CHUNK];
        let acc = &mut acc[..piece.len()];
        for i in frames.clone() {
            for (a, &v) in acc.iter_mut().zip(&raw(i)[base..base + piece.len()]) {
                *a += v as f64;
            }
        }
        for (o, &a) in piece.iter_mut().zip(acc.iter()) {
            *o = (a / count) as f32;
        }
    });
}

/// Add `frame` into the running sum `sum`, pixel by pixel. Called once per
/// frame in frame order, so every pixel's sum is over the frames in order.
fn add_into(sum: &mut [f64], frame: &[f32], parallel: bool) {
    par::for_each_chunk_mut(sum, PIXELS_PER_CHUNK, parallel, |c, piece| {
        let base = c * PIXELS_PER_CHUNK;
        let len = piece.len();
        for (a, &v) in piece.iter_mut().zip(&frame[base..base + len]) {
            *a += v as f64;
        }
    });
}

/// How many frames go in one batch; see [`Session::batch_len`].
fn batch_len(settings: &Settings, lx: usize, ly: usize) -> usize {
    match settings.backend {
        #[cfg(feature = "gpu")]
        Backend::Gpu => crate::gpu::lanes_for(lx, ly).max(1),
        Backend::SingleThread => 1,
        _ => {
            let frame = (lx * ly * std::mem::size_of::<f32>()).max(1);
            (4 * par::workers())
                .max(8)
                .min((HOST_BATCH_MEMORY / frame).max(1))
        }
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "session_oracle_tests.rs"]
mod oracle_tests;

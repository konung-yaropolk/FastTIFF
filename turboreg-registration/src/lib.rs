// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.
//
// The algorithm is Thevenaz, Ruttimann & Unser's, published in 1998 and shipped
// by EPFL as the ImageJ plugins TurboReg and StackReg, which EPFL relicensed to
// GPL-3-or-later in April 2023. This is an implementation of that algorithm,
// not a transcription of any particular code: see Cargo.toml for why the
// distinction matters here, and for the citation EPFL asks for.

//! Image registration by the TurboReg/StackReg pyramid method.
//!
//! Aligns one image to another by finding the transformation that minimises the
//! squared intensity difference between them, searched coarse-to-fine over a
//! spline pyramid. Five models, from two degrees of freedom to eight:
//! translation, rigid body, scaled rotation, affine and bilinear.
//!
//! # What this is for
//!
//! The sibling crate `suite2p-registration` answers a different question, and
//! which one to reach for is worth saying plainly. That one measures a *shift*
//! by phase correlation: fast, robust to noise, and translation-only (plus an
//! optional block-wise deformation field). This one solves for a *transformation
//! matrix* by optimisation, so it can take out rotation, scale and shear — and
//! it costs more per frame to do it.
//!
//! A two-photon timelapse that drifts under the objective wants the first. A
//! histology series where each section was placed on the slide by hand, or a
//! widefield recording where the stage rotated, wants this one.
//!
//! # The shape of the API
//!
//! Deliberately pystackreg's, because a result that cannot be compared with the
//! reference implementation is hard to trust. [`Transformation`] is its
//! transformation constants, [`Reference`] its `reference=` argument.
//!
//! Measuring and applying are kept apart, as in the sibling crate: [`register`]
//! returns matrices and moves nothing. That is what lets a multi-channel
//! recording be measured on one channel and moved on all of them, which is the
//! only way to keep channels registered *to each other*.
//!
//! # The matrix direction, which is the thing to get right
//!
//! A returned matrix maps **reference coordinates to moving coordinates**. It
//! is already the inverse map a pull-resampler wants: to produce the registered
//! image, walk the output grid, send each point through the matrix, and sample
//! the moving image there. Inverting it first is the mistake this paragraph
//! exists to prevent — the result looks registered, in the wrong direction, and
//! only a comparison against a known answer shows it.
//!
//! # From C and C++
//!
//! With the `capi` feature the crate is also a C library: `capi` is the
//! interface, `include/turboreg_registration.h` declares it, and
//!
//! ```text
//! cargo rustc -p turboreg-registration --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! builds it as a shared and a static library. The README's "From C and C++"
//! has the rest: what lands where, how to link it, and what a C caller should
//! know.

#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod optimize;
pub mod par;
pub mod pyramid;
pub mod settings;
pub mod spline;
pub mod transform;

pub use optimize::{register_pair, Fit};
pub use settings::{Backend, Border, Reference, Settings, Transformation};
pub use transform::{Matrix, Point};

/// A movie to register: frames of `ly * lx`, row-major.
pub struct Frames<'a> {
    /// Rows in a frame.
    pub ly: usize,
    /// Columns in a frame.
    pub lx: usize,
    /// One frame per timepoint, each `ly * lx` samples long.
    pub frames: &'a [Vec<f32>],
}

/// What a stack registration produced.
pub struct Registered {
    /// One transformation per frame, in input order, each mapping that frame's
    /// reference coordinates to its own.
    pub matrices: Vec<Matrix>,
    /// The image every frame was matched against, for the modes that build one.
    /// `None` under [`Reference::Previous`], where the target changes per frame.
    pub reference: Option<Vec<f32>>,
    /// Frames the fit gave up on, which keep the identity. Reported rather than
    /// dropped: which frames to discard is the experimenter's decision.
    pub failed: Vec<usize>,
    /// Whether [`Registered::apply`] resamples on every core.
    ///
    /// Carried from the backend the registration ran on, so moving the frames
    /// honours the same choice as measuring them did. Applying is the part a
    /// caller does once per *plane* — every channel and slice of every
    /// timepoint — so on a multi-channel stack it is a large share of the run,
    /// and when it ignored the backend it was what kept a "multi-thread" run of
    /// such a stack on one core for most of its length.
    pub threaded: bool,
    /// What [`Registered::apply`] writes where a moved frame no longer covers
    /// the frame. [`Border::Native`] from a run, TurboReg's own mirror; set it
    /// before applying for another.
    pub border: Border,
}

impl Registered {
    /// Put frame `t` back where it belongs, filling what the move uncovers as
    /// [`Registered::border`] says.
    pub fn apply(&self, frame: &[f32], ly: usize, lx: usize, t: usize) -> Vec<f32> {
        let m = self.matrices.get(t).copied().unwrap_or_default();
        transform::warp_bordered(frame, lx, ly, &m, self.border, self.threaded)
    }
}

/// Pixels per piece of a stack-wide average. Fixed, like the optimiser's row
/// pieces: each pixel's sum runs over the frames in order on whichever thread
/// takes its piece, so the result does not depend on how many there are.
const PIXELS_PER_CHUNK: usize = 4096;

/// The mean of the first `len` frames.
///
/// Parallel across pixels rather than across frames: every pixel's sum is
/// still taken over the frames in order, so the answer is bit-identical to a
/// single-threaded mean, and each piece walks the frames contiguously.
fn mean_of(frames: &[Vec<f32>], len: usize, parallel: bool) -> Vec<f32> {
    let take = len.clamp(1, frames.len().max(1));
    let n = frames.first().map_or(0, |f| f.len());
    let mut out = vec![0.0f32; n];
    par::for_each_chunk_mut(&mut out, PIXELS_PER_CHUNK, parallel, |i, piece| {
        let base = i * PIXELS_PER_CHUNK;
        let mut acc = vec![0.0f64; piece.len()];
        for f in frames.iter().take(take) {
            for (a, &v) in acc.iter_mut().zip(&f[base..base + piece.len()]) {
                *a += v as f64;
            }
        }
        for (o, a) in piece.iter_mut().zip(acc) {
            *o = (a / take as f64) as f32;
        }
    });
    out
}

/// Smooth the stack over time with a centred box window.
///
/// For a recording too dim to register frame by frame: averaging neighbours
/// raises what the optimiser is matching without moving it, because the window
/// is centred. An odd window is used whatever is asked for — an even one is
/// off-centre by half a frame, which is a drift.
///
/// Each output frame is independent of the others, so they are built in
/// parallel when `parallel`; each one still sums its window in frame order.
fn moving_average(frames: &[Vec<f32>], window: usize, parallel: bool) -> Vec<Vec<f32>> {
    let w = window.max(1) | 1;
    if w == 1 || frames.len() < 2 {
        return frames.to_vec();
    }
    let half = (w / 2) as isize;
    let n = frames.len() as isize;
    let times: Vec<usize> = (0..frames.len()).collect();
    par::map_in_order(&times, parallel, |&t| {
        let t = t as isize;
        let lo = (t - half).max(0);
        let hi = (t + half).min(n - 1);
        let count = (hi - lo + 1) as f64;
        let len = frames[0].len();
        let mut acc = vec![0.0f64; len];
        for f in frames.iter().take((hi + 1) as usize).skip(lo as usize) {
            for (a, &v) in acc.iter_mut().zip(f.iter()) {
                *a += v as f64;
            }
        }
        acc.iter().map(|&v| (v / count) as f32).collect()
    })
}

/// Register a stack, every frame against whatever [`Settings::reference`] says.
///
/// The frames themselves are not moved: this returns matrices, and
/// [`Registered::apply`] moves one. Keeping those apart is what lets a
/// multi-channel recording be measured on one channel and moved on all of them,
/// which is the only way to keep the channels registered to *each other*.
///
/// `on_progress` is called with a fraction and returns `false` to stop, which
/// returns `None`.
pub fn register_stack(
    movie: &Frames<'_>,
    settings: &Settings,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Registered> {
    let (ly, lx) = (movie.ly, movie.lx);
    let n = movie.frames.len();
    let threaded = settings.backend.threaded();
    if n == 0 || ly == 0 || lx == 0 {
        return Some(Registered {
            matrices: Vec::new(),
            reference: None,
            failed: Vec::new(),
            threaded,
            border: Border::default(),
        });
    }
    // Bilinear cannot be chained: the composition of two bilinear maps is not
    // bilinear. Refused rather than silently substituted, as pystackreg does.
    if settings.reference == Reference::Previous
        && settings.transformation == Transformation::Bilinear
    {
        return None;
    }

    let measured = moving_average(movie.frames, settings.moving_average, threaded);
    let mut matrices = vec![Matrix::identity(); n];
    let mut failed = Vec::new();

    // Which frame each pair registers, and against what. Frame 0 is never
    // fitted under `First` or `Previous`: it is the reference itself, or the
    // start of the chain, and keeps the identity.
    let reference = match settings.reference {
        // The first `n_frames` averaged; one means frame 0 alone.
        Reference::First => Some(mean_of(&measured, settings.n_frames.max(1), threaded)),
        Reference::Mean => Some(mean_of(&measured, n, threaded)),
        Reference::Previous => None,
    };
    let start = usize::from(settings.reference != Reference::Mean);
    let targets: Vec<usize> = (start..n).collect();
    // Under `Previous` each frame is fitted against the raw frame before it —
    // not that frame registered — so these pairs are as independent of each
    // other as any others, and run like them. Only the composition below is a
    // chain.
    let pairs: Vec<(&[f32], &[f32])> = targets
        .iter()
        .map(|&t| {
            let against = match &reference {
                Some(r) => &r[..],
                None => &measured[t - 1][..],
            };
            (against, &measured[t][..])
        })
        .collect();

    // In batches, so the run reports how far it has got and can be stopped
    // between them. A batch is a scheduling unit only: every pair is fitted
    // independently, so the batch size cannot change an answer.
    let per = batch_size(settings, lx, ly);
    let parallel = settings.backend.parallel_for(settings.reference);
    let mut fits: Vec<Option<optimize::Fit>> = Vec::with_capacity(pairs.len());
    for chunk in pairs.chunks(per) {
        if !on_progress(fits.len() as f32 / pairs.len().max(1) as f32) {
            return None;
        }
        fits.extend(optimize::register_pairs(chunk, lx, ly, settings, parallel));
    }

    match settings.reference {
        Reference::Previous => {
            // Compose down the stack, so frame t's matrix is expressed against
            // frame 0: frame t - 1's, followed by the step from t - 1 to t. A
            // frame that could not be fitted keeps its predecessor's matrix, so
            // the chain carries on past it rather than snapping back to the
            // identity.
            for (&t, fit) in targets.iter().zip(fits) {
                match fit {
                    Some(f) => {
                        matrices[t] = matrices[t - 1].compose(&f.matrix).unwrap_or(f.matrix);
                    }
                    None => {
                        failed.push(t);
                        matrices[t] = matrices[t - 1];
                    }
                }
            }
        }
        Reference::First | Reference::Mean => {
            for (&t, fit) in targets.iter().zip(fits) {
                match fit {
                    Some(f) => matrices[t] = f.matrix,
                    None => failed.push(t),
                }
            }
        }
    }
    if !on_progress(1.0) {
        return None;
    }
    Some(Registered {
        matrices,
        reference,
        failed,
        threaded,
        border: Border::default(),
    })
}

/// How many pairs go in one batch: one progress report, and one chance to be
/// stopped, per batch.
///
/// On the GPU, a batch is as many frames as the device holds at once, which is
/// also the unit the device works in. On the CPU, enough to keep every thread
/// busy with some to spare for the uneven ones; on one thread, one frame, so
/// progress moves as each frame finishes.
fn batch_size(settings: &Settings, lx: usize, ly: usize) -> usize {
    match settings.backend {
        #[cfg(feature = "gpu")]
        Backend::Gpu => gpu::lanes_for(lx, ly).max(1),
        Backend::SingleThread => 1,
        _ => {
            let _ = (lx, ly);
            (4 * par::workers()).max(8)
        }
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

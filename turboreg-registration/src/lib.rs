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

pub mod optimize;
pub mod par;
pub mod pyramid;
pub mod settings;
pub mod spline;
pub mod transform;

pub use optimize::{register_pair, Fit};
pub use settings::{Backend, Reference, Settings, Transformation};
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
}

impl Registered {
    /// Put frame `t` back where it belongs.
    pub fn apply(&self, frame: &[f32], ly: usize, lx: usize, t: usize) -> Vec<f32> {
        let m = self.matrices.get(t).copied().unwrap_or_default();
        transform::warp(frame, lx, ly, &m)
    }
}

/// The mean of a run of frames.
fn mean_of(frames: &[Vec<f32>], len: usize) -> Vec<f32> {
    let take = len.clamp(1, frames.len());
    let n = frames.first().map_or(0, |f| f.len());
    let mut acc = vec![0.0f64; n];
    for f in frames.iter().take(take) {
        for (a, &v) in acc.iter_mut().zip(f.iter()) {
            *a += v as f64;
        }
    }
    acc.iter().map(|&v| (v / take as f64) as f32).collect()
}

/// Smooth the stack over time with a centred box window.
///
/// For a recording too dim to register frame by frame: averaging neighbours
/// raises what the optimiser is matching without moving it, because the window
/// is centred. An odd window is used whatever is asked for — an even one is
/// off-centre by half a frame, which is a drift.
fn moving_average(frames: &[Vec<f32>], window: usize) -> Vec<Vec<f32>> {
    let w = window.max(1) | 1;
    if w == 1 || frames.len() < 2 {
        return frames.to_vec();
    }
    let half = (w / 2) as isize;
    let n = frames.len() as isize;
    (0..frames.len())
        .map(|t| {
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
        .collect()
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
    if n == 0 || ly == 0 || lx == 0 {
        return Some(Registered {
            matrices: Vec::new(),
            reference: None,
            failed: Vec::new(),
        });
    }
    // Bilinear cannot be chained: the composition of two bilinear maps is not
    // bilinear. Refused rather than silently substituted, as pystackreg does.
    if settings.reference == Reference::Previous
        && settings.transformation == Transformation::Bilinear
    {
        return None;
    }

    let measured = moving_average(movie.frames, settings.moving_average);
    let mut matrices = vec![Matrix::identity(); n];
    let mut failed = Vec::new();

    match settings.reference {
        Reference::Previous => {
            // A chain: each frame against the one before it, composing as it
            // goes, so frame t's matrix is expressed against frame 0. Nothing
            // here can be parallel.
            for t in 1..n {
                if !on_progress(t as f32 / n as f32) {
                    return None;
                }
                match optimize::register_pair(&measured[t - 1], &measured[t], lx, ly, settings) {
                    Some(fit) => {
                        matrices[t] = matrices[t - 1].compose(&fit.matrix).unwrap_or(fit.matrix);
                    }
                    None => {
                        failed.push(t);
                        matrices[t] = matrices[t - 1];
                    }
                }
            }
            Some(Registered {
                matrices,
                reference: None,
                failed,
            })
        }
        Reference::First | Reference::Mean => {
            let reference = match settings.reference {
                // The first `n_frames` averaged; one means frame 0 alone.
                Reference::First => mean_of(&measured, settings.n_frames.max(1)),
                _ => mean_of(&measured, n),
            };
            // Frame 0 is the reference itself under `First`, so it keeps the
            // identity and is not fitted against itself.
            let start = usize::from(settings.reference == Reference::First);
            let todo: Vec<Vec<f32>> = measured[start..].to_vec();
            if !on_progress(0.05) {
                return None;
            }
            let parallel = settings.backend.parallel_for(settings.reference);
            let fits = optimize::register_each(&todo, &reference, lx, ly, settings, parallel);
            for (i, fit) in fits.into_iter().enumerate() {
                match fit {
                    Some(f) => matrices[start + i] = f.matrix,
                    None => failed.push(start + i),
                }
            }
            if !on_progress(1.0) {
                return None;
            }
            Some(Registered {
                matrices,
                reference: Some(reference),
                failed,
            })
        }
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

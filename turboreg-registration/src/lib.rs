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
//! Measuring and applying are kept apart, as in the sibling crate:
//! [`register_stack`] returns matrices and moves nothing. That is what lets a
//! multi-channel recording be measured on one channel and moved on all of them,
//! which is the only way to keep channels registered *to each other*.
//!
//! A stack does not have to be in memory to be registered. A [`Session`] asks
//! for the frames it needs a batch at a time and is handed them from wherever
//! they are — a file read plane by plane, say — and [`register_stack`] and the
//! C interface are the same session fed from memory. What a run computes, and
//! in what order, is decided there once; see the `session` module, which also
//! says which frames it asks for twice.
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

pub mod area;
#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod optimize;
pub(crate) use image_registration_support::par;
pub mod pyramid;
pub mod session;
pub mod settings;
pub mod spline;
pub mod transform;

pub use area::{Area, NoCommonArea};
pub use optimize::{register_pair, Fit};
pub use session::{Batch, Error, Pass, Plan, Session, Wanted};
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
    /// [`Registered::common_area`], once asked for, with the frame size it was
    /// asked for at. Kept because [`Border::Crop`] needs it for every frame
    /// and finding it reads every frame's transformation.
    area: std::sync::OnceLock<((usize, usize), Result<Area, NoCommonArea>)>,
}

impl Registered {
    /// Put frame `t` back where it belongs, filling what the move uncovers as
    /// [`Registered::border`] says.
    ///
    /// Under [`Border::Crop`] the frame comes back cut to
    /// [`common_area`](Self::common_area) — and when the frames moved so far
    /// apart that there is none, this is [`NoCommonArea`] rather than a frame:
    /// there is no way to crop them that keeps anything, and a frame with its
    /// edges handled some other way would not be what was asked for. Under
    /// every other border it cannot fail.
    pub fn apply(
        &self,
        frame: &[f32],
        ly: usize,
        lx: usize,
        t: usize,
    ) -> Result<Vec<f32>, NoCommonArea> {
        let m = self.matrices.get(t).copied().unwrap_or_default();
        if self.border == Border::Crop {
            let area = self.common_area(ly, lx)?;
            return Ok(transform::warp_area(
                frame,
                lx,
                ly,
                &m,
                &area,
                self.threaded,
            ));
        }
        Ok(transform::warp_bordered(
            frame,
            lx,
            ly,
            &m,
            self.border,
            self.threaded,
        ))
    }

    /// The largest rectangle every frame covers once moved, at any angle
    /// ([`area::common_area`]) — what [`Border::Crop`] keeps — or
    /// [`NoCommonArea`] when the frames share no pixel.
    ///
    /// Found from every frame's transformation the first time it is asked
    /// for, and kept: changing [`matrices`](Self::matrices) afterwards is not
    /// seen.
    pub fn common_area(&self, ly: usize, lx: usize) -> Result<Area, NoCommonArea> {
        if let Some(((h, w), area)) = self.area.get() {
            if (*h, *w) == (ly, lx) {
                return *area;
            }
        }
        let area = area::common_area(&self.matrices, ly, lx, self.threaded);
        let _ = self.area.set(((ly, lx), area));
        area
    }

    /// The size, `(rows, columns)`, of what [`apply`](Self::apply) hands back
    /// for frames of `ly * lx`: the same, except under [`Border::Crop`], where
    /// it is the common area's — or [`NoCommonArea`] when there is none.
    pub fn output_size(&self, ly: usize, lx: usize) -> Result<(usize, usize), NoCommonArea> {
        if self.border == Border::Crop {
            self.common_area(ly, lx).map(|a| (a.height, a.width))
        } else {
            Ok((ly, lx))
        }
    }
}

/// Register a stack held in memory, every frame against whatever
/// [`Settings::reference`] says.
///
/// A [`Session`] fed from `movie`, batch by batch: the frames are read where
/// they are, and the registration holds a batch's worth of working memory
/// beside them — plus, with a moving average of `2h + 1` frames, a copy of the
/// `2h` before each batch — not a copy of the movie, unless the window is as
/// wide as it. A stack too large to hold, or read from a file, goes through a
/// [`Session`] directly.
///
/// The frames themselves are not moved: this returns matrices, and
/// [`Registered::apply`] moves one. Keeping those apart is what lets a
/// multi-channel recording be measured on one channel and moved on all of them,
/// which is the only way to keep the channels registered to *each other*.
///
/// `on_progress` is called with a fraction and returns `false` to stop, which
/// is [`Error::Cancelled`]. A combination the method cannot run, or a backend
/// that cannot run here, is refused before anything is fitted
/// ([`Session::new`]); frames the GPU failed fail the run
/// ([`Error::Device`]).
pub fn register_stack(
    movie: &Frames<'_>,
    settings: &Settings,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<Registered, Error> {
    Session::new(movie.ly, movie.lx, movie.frames.len(), settings)?
        .run(Batch::Split(movie.frames), on_progress)
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

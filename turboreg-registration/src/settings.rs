// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! What to register, how, and where it runs — named as pystackreg names it.

/// The transformation model, and so how many degrees of freedom the search has.
///
/// The discriminants are pystackreg's own constants, which are also the
/// reference implementation's parameter counts for the landmark models. They
/// are written out rather than left to the compiler because a value copied from
/// a Python script should mean the same thing here.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Transformation {
    /// Shift only. Two degrees of freedom.
    ///
    /// The cheapest and the most robust: with nothing to rotate or stretch
    /// there is no way for the search to trade a bad rotation against a bad
    /// shift, which is how the richer models fail on low-contrast data.
    Translation = 2,
    /// Shift and rotation, no scaling. Three degrees of freedom.
    ///
    /// The default, and the right one for most microscopy: a specimen that
    /// moved under the objective has not changed size.
    #[default]
    RigidBody = 3,
    /// Shift, rotation and one isotropic scale. Four degrees of freedom.
    ScaledRotation = 4,
    /// Any linear map plus a shift: shear and anisotropic scale included. Six
    /// degrees of freedom.
    Affine = 6,
    /// A bilinear warp. Eight degrees of freedom.
    ///
    /// The only model that is not a straight line map, so it is also the only
    /// one that cannot be composed with another and stay in the family — which
    /// is why [`Reference::Previous`] refuses it.
    Bilinear = 8,
}

impl Transformation {
    /// What to call this in an interface.
    pub fn label(self) -> &'static str {
        match self {
            Transformation::Translation => "Translation",
            Transformation::RigidBody => "Rigid body",
            Transformation::ScaledRotation => "Scaled rotation",
            Transformation::Affine => "Affine",
            Transformation::Bilinear => "Bilinear",
        }
    }

    /// Every model, in the order a selector should offer them: fewest degrees
    /// of freedom first, which is also least to most likely to go wrong.
    pub fn all() -> [Transformation; 5] {
        [
            Transformation::Translation,
            Transformation::RigidBody,
            Transformation::ScaledRotation,
            Transformation::Affine,
            Transformation::Bilinear,
        ]
    }

    /// How many free parameters the optimiser carries.
    pub fn dof(self) -> usize {
        self as usize
    }

    /// How many landmark pairs the model is parameterised by.
    ///
    /// Not simply `dof / 2`: [`Transformation::RigidBody`] carries three
    /// landmarks but only three degrees of freedom, because the rigid structure
    /// is imposed rather than fitted.
    pub fn landmarks(self) -> usize {
        match self {
            Transformation::Translation => 1,
            Transformation::RigidBody => 3,
            Transformation::ScaledRotation => 2,
            Transformation::Affine => 3,
            Transformation::Bilinear => 4,
        }
    }
}

/// What each frame of a stack is registered *to*.
///
/// pystackreg's `reference=` argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Reference {
    /// Each frame to the one before it, composing the transformations down the
    /// stack.
    ///
    /// The right choice for a recording that drifts steadily, because each pair
    /// is nearly aligned already and the search starts close. The cost is that
    /// error accumulates: frame 500's transformation is the product of 500
    /// estimates, and a systematic bias in each becomes a slow walk.
    ///
    /// Refused for [`Transformation::Bilinear`], where the composition of two
    /// transformations is not in the family and the chain would be meaningless.
    Previous,
    /// Every frame to the first.
    ///
    /// The default. Nothing accumulates, and the target is a frame that was
    /// actually photographed — which matters when the result is compared
    /// against the original, and on a short stack, where the average is mostly
    /// the drift it is meant to remove.
    ///
    /// The cost is that a frame late in a long recording is matched against
    /// something it may have drifted a long way from.
    #[default]
    First,
    /// Every frame to the mean of the whole stack.
    ///
    /// A sharper target than any single frame, because the noise averages down,
    /// and nothing accumulates either. Worth reaching for on a long, noisy
    /// recording — at the price that every frame is then aligned to something
    /// that was never photographed.
    Mean,
}

impl Reference {
    pub fn label(self) -> &'static str {
        match self {
            Reference::Previous => "Previous frame",
            Reference::First => "First frame",
            Reference::Mean => "Mean of the stack",
        }
    }

    pub fn all() -> [Reference; 3] {
        [Reference::Previous, Reference::First, Reference::Mean]
    }
}

/// Where the arithmetic runs.
///
/// The two CPU backends give the same answer bit for bit, not to a tolerance;
/// see `crate::par` for what makes that a rule rather than luck. The GPU backend
/// agrees with them to a tolerance — its per-pixel arithmetic is `f32` — and is
/// reproducible run to run on the same device; see `crate::gpu`.
///
/// The CPU threads are spent at two levels. *Within* a frame, the criterion's
/// sum over pixels, the gradient images, the spline prefilters and the final
/// resampling each run across every core. *Across* frames, independent frames
/// run at the same time too — in every reference mode, the previous-frame one
/// included; see [`Backend::parallel_for`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Backend {
    /// One frame at a time. The one to reach for when a result looks wrong, as
    /// it removes scheduling from the list of suspects.
    #[cfg_attr(not(feature = "threads"), default)]
    SingleThread,
    /// A frame per core, through rayon — and so only in a build with the
    /// `threads` feature, which is where the default also is. Without it this
    /// is still offered and still refused by [`unavailable_reason`], because an
    /// option that silently ran on one thread would be indistinguishable from a
    /// slow machine.
    ///
    /// [`unavailable_reason`]: Backend::unavailable_reason
    #[cfg_attr(feature = "threads", default)]
    MultiThread,
    /// The criterion's sum over pixels on the graphics card, for the pyramid
    /// levels large enough to be worth a round trip; everything else on every
    /// CPU core, as [`Backend::MultiThread`] does it.
    ///
    /// Not the default, and not promised to be faster: on a modest card the
    /// multi-thread CPU path can match it. Offered in every build and refused
    /// by [`unavailable_reason`] where it cannot run — without the `gpu`
    /// feature, in a browser, or on a machine with no usable adapter.
    ///
    /// [`unavailable_reason`]: Backend::unavailable_reason
    Gpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::SingleThread => "Single-thread CPU",
            Backend::MultiThread => "Multi-thread CPU",
            Backend::Gpu => "GPU",
        }
    }

    /// Every backend, in the order a selector should offer them.
    pub fn all() -> [Backend; 3] {
        [Backend::SingleThread, Backend::MultiThread, Backend::Gpu]
    }

    /// Whether this backend can run at all here, whatever the frame size.
    ///
    /// For [`Backend::Gpu`] this opens the device the first time it is asked,
    /// which takes a moment; the answer is remembered.
    pub fn available(self) -> bool {
        match self {
            Backend::SingleThread => true,
            Backend::MultiThread => cfg!(feature = "threads"),
            Backend::Gpu => gpu_ready(),
        }
    }

    /// Why it cannot run frames of `ly` rows by `lx` columns, for a caller to
    /// show.
    ///
    /// Checked rather than silently downgraded: a run that said GPU and used
    /// the processor is indistinguishable from a slow one, and there would be
    /// no way to tell which had happened.
    pub fn unavailable_reason(self, ly: usize, lx: usize) -> Option<String> {
        match self {
            Backend::SingleThread => None,
            Backend::MultiThread => (!cfg!(feature = "threads")).then(|| {
                "this build has no thread pool: it is behind the crate's `threads` \
                 feature, which was not enabled. Use the single-thread CPU backend."
                    .to_string()
            }),
            Backend::Gpu => gpu_unavailable_reason(ly, lx),
        }
    }

    /// Whether this backend uses more than one CPU thread at all.
    ///
    /// True for [`Backend::MultiThread`] and [`Backend::Gpu`] in a build that
    /// has threads, whatever the reference mode — the GPU backend runs
    /// everything it does not send to the device the way the multi-thread one
    /// does. This is the switch for the work *inside* a frame, which every mode
    /// has.
    pub fn threaded(self) -> bool {
        matches!(self, Backend::MultiThread | Backend::Gpu) && cfg!(feature = "threads")
    }

    /// Whether separate *frames* may be registered at the same time.
    ///
    /// In every reference mode, on a threaded backend. That includes
    /// [`Reference::Previous`], which this used to exclude on the grounds that
    /// it is a chain — and it is not, where it matters. Frame `t` is fitted
    /// against the *raw* frame `t - 1`, not against frame `t - 1` registered,
    /// so the fits are as independent as any others. Only composing them into
    /// matrices relative to frame 0 is sequential, and that is a handful of
    /// 3x3 products after the expensive part is over. Treating the whole
    /// registration as a chain kept the chained mode one frame at a time — on
    /// one core, until the work inside each frame was spread too — and on the
    /// GPU one lane per round trip, which made it slower there than on the CPU.
    ///
    /// The `reference` argument stays so callers say which mode they are
    /// asking about; no mode is excluded today.
    pub fn parallel_for(self, reference: Reference) -> bool {
        let _ = reference;
        self.threaded()
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn gpu_ready() -> bool {
    crate::gpu::available()
}

#[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
fn gpu_ready() -> bool {
    false
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn gpu_unavailable_reason(ly: usize, lx: usize) -> Option<String> {
    if !crate::gpu::available() {
        return Some(
            "no usable graphics adapter was found (a software renderer does not count: it \
             would run the GPU path on the processor, slower than the CPU backends). Use a \
             CPU backend."
                .to_string(),
        );
    }
    if crate::gpu::lanes_for(lx, ly) == 0 {
        return Some(format!(
            "a {lx}x{ly} frame does not fit in this graphics card's memory limits. Use a CPU \
             backend."
        ));
    }
    None
}

#[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
fn gpu_unavailable_reason(_ly: usize, _lx: usize) -> Option<String> {
    Some(if cfg!(target_arch = "wasm32") {
        "the GPU backend is not available in a browser. Use a CPU backend.".to_string()
    } else {
        "this build has no GPU backend: it is behind the crate's `gpu` feature, which was \
         not enabled. Use a CPU backend."
            .to_string()
    })
}

/// What a registered frame shows where its content has moved out of view.
///
/// Moving a frame onto the reference uncovers a strip along the edge it moved
/// away from: output pixels with no measurement behind them. Something has to
/// be written there, and this is the choice of what. Only the frames handed
/// back are affected; how frames move while the registration measures is part
/// of the method and stays as it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Border {
    /// What TurboReg's own output shows, and the default: the spline's
    /// mirrored extension, the same one the registration measures with. It
    /// has no seam, but past the edge it repeats the picture backwards, cells
    /// included, so the strip shows tissue that is not there.
    #[default]
    Native,
    /// The nearest edge pixel, repeated outward. The frame keeps the look of
    /// the frame, with no far-side content and no dark band; but what it
    /// writes is a copy of a real pixel, so an analysis that averages over the
    /// strip still takes in a value that was not measured there.
    Edge,
    /// Zero. Unmistakably not a measurement, and easy to mask out afterwards;
    /// a pixel only partly covered by the frame counts as uncovered, so the
    /// strip ends in a hard edge rather than a ramp from data to zero.
    Constant,
}

impl Border {
    /// What to call this option in an interface.
    pub fn label(self) -> &'static str {
        match self {
            Border::Native => "As TurboReg (mirror)",
            Border::Edge => "Edge replicate",
            Border::Constant => "Constant (0)",
        }
    }

    /// Every option, in the order a selector should offer them.
    pub fn all() -> [Border; 3] {
        [Border::Native, Border::Edge, Border::Constant]
    }
}

/// How far past the outermost sample a position may land and still count as
/// on the frame: rounding, not motion. A fitted transformation that is the
/// identity to within `1e-12` would otherwise blank the last row and column
/// under [`Border::Constant`].
pub(crate) const ON_FRAME: f64 = 1e-6;

/// Everything a run needs beyond the pixels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Settings {
    /// Which model to fit.
    pub transformation: Transformation,
    /// What each frame is registered to.
    pub reference: Reference,
    /// How many frames from the start are averaged to build the target under
    /// [`Reference::First`]. pystackreg's `n_frames`.
    ///
    /// One means "the first frame alone", which is the default. Read only for
    /// [`Reference::First`]: [`Reference::Mean`] averages the whole stack, and
    /// [`Reference::Previous`] has no fixed target to average.
    pub n_frames: usize,
    /// Smooth the stack over time before measuring, with this window.
    /// pystackreg's `moving_average`. One disables it.
    ///
    /// For a recording too dim to register frame by frame: averaging a few
    /// neighbouring frames raises the signal the optimiser is matching without
    /// changing what is measured, since the window is centred.
    pub moving_average: usize,
    /// Where the arithmetic runs.
    pub backend: Backend,
    /// Cap on the pyramid depth, counting the full-resolution level.
    ///
    /// Zero means "as deep as the image allows", which is what the reference
    /// does and what this defaults to. A cap exists because a very deep pyramid
    /// on a nearly-aligned stack spends its time on levels where every frame
    /// looks the same.
    pub max_pyramid_depth: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            transformation: Transformation::RigidBody,
            reference: Reference::First,
            n_frames: 1,
            moving_average: 1,
            backend: Backend::default(),
            max_pyramid_depth: 0,
        }
    }
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;

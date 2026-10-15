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
/// The answer is the same on each: a frame's transformation depends on the
/// reference and on that frame, never on another frame's result — except under
/// [`Reference::Previous`], which is a chain by construction and runs on one
/// thread whatever is chosen here. That exception is reported rather than
/// hidden; see [`Backend::unavailable_reason`].
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
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::SingleThread => "Single-thread CPU",
            Backend::MultiThread => "Multi-thread CPU",
        }
    }

    pub fn all() -> [Backend; 2] {
        [Backend::SingleThread, Backend::MultiThread]
    }

    /// Whether this backend can run at all in this build.
    pub fn available(self) -> bool {
        self.unavailable_reason().is_none()
    }

    /// Why it cannot run, for a caller to show.
    pub fn unavailable_reason(self) -> Option<String> {
        match self {
            Backend::MultiThread => {
                #[cfg(not(feature = "threads"))]
                {
                    Some(
                        "this build has no thread pool: it is behind the crate's `threads` \
                         feature, which was not enabled"
                            .to_string(),
                    )
                }
                #[cfg(feature = "threads")]
                {
                    None
                }
            }
            Backend::SingleThread => None,
        }
    }

    /// Whether a stack registered this way can use more than one thread.
    ///
    /// [`Reference::Previous`] cannot: frame `t` is registered against frame
    /// `t - 1`'s *result*, so the frames are a chain and there is nothing to
    /// run in parallel. Stated here rather than discovered as a disappointing
    /// benchmark.
    pub fn parallel_for(self, reference: Reference) -> bool {
        self == Backend::MultiThread
            && self.available()
            && !matches!(reference, Reference::Previous)
    }
}

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

//! The pieces more than one built-in plugin is built from.
//!
//! Three things every tool that hands back an image needs, and which have to
//! agree between them: the "open in a new window" checkbox, the key it is read
//! by, and how a result is stored.
//!
//! These lived in [`stack_tools`](super::stack_tools) while that was the module
//! with three of the four users, on the grounds that a module of their own
//! would have been ceremony. Two things changed. Deconvolution became a fourth
//! user, so no one plugin is the obvious home any more; and each plugin is now
//! a Cargo feature that can be compiled out, which made the old arrangement
//! wrong rather than merely untidy — `plugin-deconvolve` would not build
//! without `plugin-stack-tools`, for two functions and an enum that have
//! nothing to do with slices.
//!
//! So this module is not feature-gated. It is a few hundred lines that cost
//! nothing when no plugin uses them, and being here is what lets every
//! `plugin-*` feature be chosen on its own.
//!
//! Note what is *not* here. `Axis` belongs to `stack_tools` and `plot_axis`
//! has its own, deliberately: they offer different axis sets for different
//! reasons, and merging them would couple two plugins to make one enum.

use fasttiff_plugin_api::{
    ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType, PlaneData,
};

/// The key of the "open in a new window" checkbox.
pub(crate) const NEW_WINDOW: &str = "new_window";

/// The checkbox every tool here offers.
///
/// `Filters > Invert` offers it too, so this is `pub(crate)` rather than
/// private to this module.
///
/// Declared in one place so the four of them cannot drift — a tool that spelled
/// the key differently would read the default on every run and silently always
/// open a window.
pub(crate) fn in_new_window() -> ParamDecl {
    ParamDecl::new(
        "new_window",
        "Open in a new window",
        ParamKind::Bool { default: true },
    )
    .help(
        "On, the result opens in its own window and this one is left alone. Off, it \
         replaces the image in this window — which cannot be undone.",
    )
}

/// Hand an image back the way the dialog asked for it.
pub(crate) fn deliver(image: ImageResult, params: &Params) -> Outcome {
    if params.bool(NEW_WINDOW, true) {
        Outcome::NewDocument(Box::new(image))
    } else {
        Outcome::ReplaceDocument(Box::new(image))
    }
}

/// What a result's samples are stored as, so a tool gives back what it was
/// given.
///
/// A tool here changes which planes there are, not what a sample means, so a
/// 16-bit recording must come back 16-bit. Widening everything to float — which
/// is what reading through `read_plane_f32` and storing the result would do —
/// doubles every file for nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Store {
    U8,
    U16,
    I16,
    F32,
}

impl Store {
    pub(crate) fn of(source: PixelType) -> Self {
        match source {
            PixelType::U8 => Store::U8,
            PixelType::U16 => Store::U16,
            PixelType::I16 => Store::I16,
            PixelType::F32 => Store::F32,
        }
    }

    pub(crate) fn pixel_type(self) -> PixelType {
        match self {
            Store::U8 => PixelType::U8,
            Store::U16 => PixelType::U16,
            Store::I16 => PixelType::I16,
            Store::F32 => PixelType::F32,
        }
    }

    /// The range a sample of this width can hold, for the tools that need one.
    /// `None` for float, which has no range of its own to invert about.
    ///
    /// `Filters > Invert` is the only one that needs it, and can be compiled
    /// out — hence the conditional allow rather than a plain one, so that the
    /// method is still held to being used in the build that has its user.
    #[cfg_attr(not(feature = "plugin-invert"), allow(dead_code))]
    pub(crate) fn range(self) -> Option<(f32, f32)> {
        match self {
            Store::U8 => Some((0.0, 255.0)),
            Store::U16 => Some((0.0, 65535.0)),
            Store::I16 => Some((-32768.0, 32767.0)),
            Store::F32 => None,
        }
    }

    /// Store a plane of `f32` samples as this width.
    ///
    /// The signed arm is the one to be careful with: the contract has no
    /// `PlaneData::I16`, so signed samples travel as the same sixteen bits in
    /// the `U16` lane and are declared `PixelType::I16`. Getting that wrong
    /// makes every negative sample read as a very bright one — a picture that
    /// still looks like a picture.
    pub(crate) fn plane(self, v: Vec<f32>) -> PlaneData {
        match self {
            Store::U8 => PlaneData::U8(v.iter().map(|&x| whole(x, 0.0, 255.0) as u8).collect()),
            Store::U16 => {
                PlaneData::U16(v.iter().map(|&x| whole(x, 0.0, 65535.0) as u16).collect())
            }
            Store::I16 => PlaneData::U16(
                v.iter()
                    .map(|&x| whole(x, -32768.0, 32767.0) as i16 as u16)
                    .collect(),
            ),
            Store::F32 => PlaneData::F32(v),
        }
    }
}

/// Round to the nearest whole sample, clamped into range.
///
/// `NaN` becomes the bottom of the range rather than zero-by-cast: `as u16` on
/// a `NaN` is 0, which for a signed stack is the middle of the range and reads
/// as mid-grey rather than as nothing.
fn whole(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() {
        return lo;
    }
    x.clamp(lo, hi).round()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

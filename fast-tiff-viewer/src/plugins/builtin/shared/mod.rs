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

// Each piece is compiled for the plugins that use it, so a build with only
// some of them carries no unused code — and no warnings saying so.
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
))]
use fasttiff_plugin_api::{ImageResult, Outcome};
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-suite2p-stabilize",
    feature = "plugin-turboreg-stabilize",
))]
use fasttiff_plugin_api::{ParamDecl, ParamKind, Params};
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
    feature = "plugin-crop",
))]
use fasttiff_plugin_api::{PixelType, PlaneData};
#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
use fasttiff_plugin_api::PluginError;

/// The key of the "open in a new window" checkbox.
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
))]
pub(crate) const NEW_WINDOW: &str = "new_window";

/// The checkbox every tool here offers.
///
/// `Filters > Invert` offers it too, so this is `pub(crate)` rather than
/// private to this module.
///
/// Declared in one place so the four of them cannot drift — a tool that spelled
/// the key differently would read the default on every run and silently always
/// open a window.
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
))]
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

#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
/// What a stabilizer writes where a moved frame no longer covers the frame.
///
/// Both registration plugins offer the same four, and each maps them onto its
/// own crate's `Border`: the two crates share the idea, not a type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Edges {
    /// Zero: plainly no data, and easy to mask out. The default here.
    Zero,
    /// The nearest edge pixel, repeated outward.
    Replicate,
    /// Whatever the algorithm writes by itself.
    Raw,
    /// No strip: every frame cut to the largest rectangle every frame covers.
    Crop,
}

#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
/// The key of the edge choice.
pub(crate) const EDGES: &str = "edges";

/// The edge choice, as both stabilizers offer it. `raw` says what the
/// algorithm's own output is, which differs between them.
#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
pub(crate) fn edges_param(raw: &str) -> ParamDecl {
    ParamDecl::new(
        EDGES,
        "Uncovered edges",
        ParamKind::Choice {
            default: 0,
            options: vec![
                "Fill with 0".to_string(),
                "Edge replicate".to_string(),
                "Crop to the common area".to_string(),
                "Algorithm's raw output".to_string(),
            ],
        },
    )
    .help(format!(
        "Moving a frame onto the reference uncovers a strip along the edge it moved away \
         from, with nothing measured behind it. Fill with 0 marks it as no data, which an \
         analysis can mask out. Edge replicate repeats the nearest edge pixel: it looks \
         continuous, but copies real values into the strip. The raw output is what the \
         algorithm writes by itself: {raw}. Crop keeps only the largest rectangle that \
         every frame covers, so nothing is filled: the result is smaller, at the same pixel \
         spacing. When the frames turned, that rectangle can be at an angle; it is then \
         turned upright in the result, which is no longer aligned with the original image. \
         A recording that moved so far that no pixel is covered by every frame cannot be \
         cropped, and the run stops and says so."
    ))
}

/// Read the edge choice back.
#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
pub(crate) fn edges_from(params: &Params) -> Edges {
    match params.choice(EDGES, 0) {
        1 => Edges::Replicate,
        2 => Edges::Crop,
        3 => Edges::Raw,
        _ => Edges::Zero,
    }
}

/// The run's end when a crop was asked for and the frames share no pixel.
///
/// Said in the dialog's terms, with the way out: a fill keeps every frame
/// whole, so any of the other three runs where this one cannot. `if_not_real`
/// is the plugin's own advice for a recording that did not really move that
/// far — the settings that keep its search in bounds.
#[cfg(any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize"))]
pub(crate) fn no_common_area(why: impl std::fmt::Display, if_not_real: &str) -> PluginError {
    PluginError::failed(format!(
        "cannot crop to the common area: {why}. Fill with 0, Edge replicate or the \
         algorithm's raw output for the uncovered edges keep every frame whole. If the \
         recording did not really move that far, {if_not_real}."
    ))
}

/// Hand an image back the way the dialog asked for it.
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
))]
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
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
    feature = "plugin-crop",
))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Store {
    U8,
    U16,
    I16,
    F32,
}

#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
    feature = "plugin-crop",
))]
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
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
    feature = "plugin-turboreg-stabilize",
    feature = "plugin-crop",
))]
fn whole(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() {
        return lo;
    }
    x.clamp(lo, hi).round()
}

// The tests are of the window and storage pieces, so they build with them.
#[cfg(all(
    test,
    any(
        feature = "plugin-invert",
        feature = "plugin-stack-tools",
        feature = "plugin-deconvolve",
        feature = "plugin-turboreg-stabilize",
    )
))]
#[path = "mod_tests.rs"]
mod tests;
#[cfg(all(
    test,
    any(feature = "plugin-suite2p-stabilize", feature = "plugin-turboreg-stabilize")
))]
#[path = "edges_tests.rs"]
mod edges_tests;

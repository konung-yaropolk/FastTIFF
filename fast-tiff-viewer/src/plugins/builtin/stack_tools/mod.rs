//! Stack Tools: the operations that change a stack's *shape* rather than its
//! pixels' meaning.
//!
//! Three entries in the `Stack` menu, beside `Z Project` and `Plot third axis`.
//! [`SliceKeeper`] and [`SliceRemover`] take planes out along an axis, and
//! [`SliceOrderInvert`] reverses the order of the ones that are there. None of
//! them changes a sample value; `Filters > Invert` is the one that does.
//!
//! # Every tool offers a window
//!
//! FastTIFF is one stack per window, so a result either opens in a window of
//! its own or takes the place of the one it came from. Only the user knows
//! which they meant, so every tool here asks — see [`in_new_window`] — and
//! defaults to a new window, because that is the answer that cannot lose
//! anything. The unchecked case is [`Outcome::ReplaceDocument`], which is
//! destructive by design.
//!
//! # Axes, and what "a slice" means
//!
//! ImageJ's stack is a flat list of slices. FastTIFF's has three axes, so
//! "remove slices 3 to 7" is ambiguous until it says *along what*. Each tool
//! that takes planes out therefore offers an axis, exactly as `ZProject` does,
//! and operates along it while keeping every channel: a channel is not a slice,
//! and dropping half of one would leave a stack whose channels no longer line
//! up.
//!
//! # The plane order
//!
//! `ImageResult::planes` is `xyczt` — channel fastest, then Z, then T. The one
//! formula every tool here depends on is therefore
//!
//! ```text
//!     index = t * (slices * channels) + z * channels + c
//! ```
//!
//! # Shared with `Filters > Invert`
//!
//! [`Store`], [`deliver`] and [`in_new_window`] are `pub(crate)` because
//! `Filters > Invert` is built from them as well: it inverts whatever is
//! loaded, which is the same walk over the same planes at the same sample
//! width, and the checkbox means the same thing there as here. They live here
//! rather than in a module of their own because this is where there are three
//! users of them and there one.
//!
//! None of these tools has to *call* that formula: each walks the planes in
//! that order and emits them in that order, so the order is the loop nesting.
//! What they do instead is choose which `(z, t)` pairs to walk, and hand them
//! to [`map_planes`] already in order — which is why that is the one place the
//! ordering has to be right.

use fasttiff_plugin_api::{
    HostContext, ImageInfo, ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType,
    PlaneData, PluginError,
};

mod order;
mod slices;

pub use order::SliceOrderInvert;
pub use slices::{SliceKeeper, SliceRemover};

/// An axis planes can be taken along.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Axis {
    Z,
    T,
}

impl Axis {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Axis::Z => "Z (slices)",
            Axis::T => "T (frames)",
        }
    }

    pub(crate) fn depth(self, info: &ImageInfo) -> usize {
        match self {
            Axis::Z => info.slices.max(1),
            Axis::T => info.frames.max(1),
        }
    }
}

/// The axes worth offering, in dialog order.
///
/// Never empty, for the reason `ZProject` gives: an empty dropdown reads as
/// broken, where a selector plus a refusal reads as inapplicable.
pub(crate) fn axes(info: &ImageInfo) -> Vec<Axis> {
    let found: Vec<Axis> = [Axis::Z, Axis::T]
        .into_iter()
        .filter(|a| a.depth(info) > 1)
        .collect();
    if found.is_empty() {
        vec![Axis::Z]
    } else {
        found
    }
}

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

/// Read every plane of the source, in `xyczt` order, applying `f` to each.
///
/// Shared because three of the four tools are "walk the planes and keep some of
/// them": what differs is which, and what happens to the samples on the way.
///
/// `Ok(None)` means the user cancelled, which is the house convention
/// (`to_tiff_bytes_reporting` reports it the same way) and is not the same
/// thing as an error. Returning `Err` for it — which this did — puts the word
/// "cancelled" in the status bar styled as a failure, for something the user
/// asked for.
pub(crate) fn map_planes(
    host: &mut dyn HostContext,
    keep: &[(usize, usize)],
    store: Store,
    mut f: impl FnMut(&mut Vec<f32>),
) -> Result<Option<Vec<PlaneData>>, PluginError> {
    let info = host.image();
    let channels = info.channels.max(1);
    let mut planes = Vec::with_capacity(keep.len() * channels);
    let mut buf = Vec::new();
    let total = (keep.len() * channels).max(1);
    let mut done = 0usize;
    for &(z, t) in keep {
        for c in 0..channels {
            if !host.progress(done as f32 / total as f32) {
                return Ok(None);
            }
            host.read_plane_f32(fasttiff_plugin_api::Plane::new(c, z, t), &mut buf)?;
            f(&mut buf);
            planes.push(store.plane(std::mem::take(&mut buf)));
            done += 1;
        }
    }
    Ok(Some(planes))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

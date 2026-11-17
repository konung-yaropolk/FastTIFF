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
//! FastTIFF is one document per window, so a result either opens in a window of
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
//! A result's planes are `xyczt` — channel fastest, then Z, then T — whether
//! returned whole or, as every tool here does, streamed to the host as they
//! are read. The one formula every tool here depends on is therefore
//!
//! ```text
//!     index = t * (slices * channels) + z * channels + c
//! ```
//!
//! # Shared with `Filters > Invert` and with deconvolution
//!
//! [`Store`], [`deliver`] and [`in_new_window`] used to live here, because
//! `Filters > Invert` is built from them as well and this was where three of
//! the four users were. They are in [`shared`](super::shared) now: deconvolution
//! became a fourth user, and each of these is a Cargo feature that can be
//! compiled out on its own, which a plugin reaching into another plugin's module
//! does not survive.
//!
//! None of these tools has to *call* that formula: each walks the planes in
//! that order and emits them in that order, so the order is the loop nesting.
//! What they do instead is choose which `(z, t)` pairs to walk, and hand them
//! to [`map_planes`] already in order — which is why that is the one place the
//! ordering has to be right.

use super::shared::{Store, Stored};
use fasttiff_plugin_api::{HostContext, ImageInfo, ImageSpec, ImageStream, PluginError};

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

/// Read every plane of the source, in `xyczt` order, applying `f` to each and
/// pushing it to `out`.
///
/// Shared because three of the four tools are "walk the planes and keep some of
/// them": what differs is which, and what happens to the samples on the way.
/// Kept planes go to the host as they are read, so a tool that keeps most of a
/// recording never holds it — `out` is the result the tool began, and each
/// plane is the host's before the next is read.
///
/// `Ok(false)` means the user cancelled, which is the house convention and is
/// not the same thing as an error. Returning `Err` for it — which this once
/// did — puts the word "cancelled" in the status bar styled as a failure, for
/// something the user asked for.
pub(crate) fn map_planes(
    host: &mut dyn HostContext,
    keep: &[(usize, usize)],
    store: Store,
    out: &mut ImageStream,
    mut f: impl FnMut(&mut Vec<f32>),
) -> Result<bool, PluginError> {
    let info = host.image();
    let channels = info.channels.max(1);
    let mut buf = Vec::new();
    let mut stored = Stored::default();
    let total = (keep.len() * channels).max(1);
    let mut done = 0usize;
    for &(z, t) in keep {
        for c in 0..channels {
            if !host.progress(done as f32 / total as f32) {
                return Ok(false);
            }
            host.read_plane_f32(fasttiff_plugin_api::Plane::new(c, z, t), &mut buf)?;
            f(&mut buf);
            out.push(host, store.encode(&buf, &mut stored))?;
            done += 1;
        }
    }
    Ok(true)
}

/// The result of a tool here: the source's frame and calibration, with the
/// axes the tool left, at the source's own width, named for what was done.
pub(crate) fn result_spec(
    host: &dyn HostContext,
    store: Store,
    slices: usize,
    frames: usize,
    suffix: &str,
) -> ImageSpec {
    let info = host.image();
    ImageSpec {
        width: info.width,
        height: info.height,
        channels: info.channels.max(1),
        slices,
        frames,
        pixel_type: store.pixel_type(),
        channel_colors: Vec::new(),
        metadata: Some(host.stack_info().clone()),
        name: format!("{}-{suffix}", host.stack_info().stem()),
    }
}

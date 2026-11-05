//! Decoding one plane into `f32`, in the file's own units.
//!
//! "The file's own units" is the contract, and it is not what the *display*
//! path does. An 8-bit sample arrives here as `0..255`, not widened to
//! `0..65535`; a signed 16-bit sample arrives as the number the file states,
//! not offset into unsigned. Anything measuring pixels has to get the value
//! that is in the file, or the number it reports is about the viewer rather
//! than about the specimen.
//!
//! The decoding itself is `fast_tiff_lib::read_plane_f32_into`, which reads
//! every depth in those units in one pass. This module used to redo the
//! per-depth dispatch and undo the display transform by hand, and that copy
//! had drifted (signed 8-bit came back as 0..255); there is now one
//! definition, in the library, pinned against the display readers by its
//! `display_map` tests.

use anyhow::{Context, Result};
use fast_tiff_lib::{read_plane_f32_into, TiffStack};

/// Decode `(ifd, sample)` of `tiff` into `out` as `f32`, in the file's own
/// units.
///
/// `out` is resized to the plane's pixel count and its allocation reused.
pub fn plane_f32_into(
    tiff: &TiffStack,
    ifd: usize,
    sample: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let frame = tiff
        .frames
        .get(ifd)
        .with_context(|| format!("IFD {ifd} is past the {} this file has", tiff.frames.len()))?;
    read_plane_f32_into(&tiff.data, frame, tiff.byte_order, sample, out)
}

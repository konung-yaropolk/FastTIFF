//! Decoding one plane into `f32`, in the file's own units.
//!
//! "The file's own units" is the contract, and it is not what the *display*
//! path does. An 8-bit sample arrives here as `0..255`, not widened to
//! `0..65535`; a signed 16-bit sample arrives as the number the file states,
//! not offset into unsigned. Anything measuring pixels has to get the value
//! that is in the file, or the number it reports is about the viewer rather
//! than about the specimen.
//!
//! The decoding itself is fast-tiff-lib's
//! [`File`](fast_tiff_lib::units::File) units, which read every depth in
//! those units in one pass. This module used to redo the per-depth dispatch
//! and undo the display transform by hand, and that copy had drifted (signed
//! 8-bit came back as 0..255); there is now one definition, in the library,
//! pinned against the display readers by its `display_map` tests.

use anyhow::{Context, Result};
use fast_tiff_lib::units::File;
use fast_tiff_lib::{DecodeScratch, Frame, TiffStack};

/// IFD `ifd` of `tiff`, or why there is none.
fn frame(tiff: &TiffStack, ifd: usize) -> Result<Frame<'_>> {
    tiff.frame(ifd)
        .with_context(|| format!("IFD {ifd} is past the {} this file has", tiff.frame_count()))
}

/// How many samples a plane of IFD `ifd` holds: the length
/// [`plane_f32_into`] needs. The frame's size is vetted against its strips
/// first, so a hostile header is refused here rather than sized from.
pub(crate) fn plane_len(tiff: &TiffStack, ifd: usize) -> Result<usize> {
    Ok(frame(tiff, ifd)?.plane_len(File)?)
}

/// Decode `(ifd, sample)` of `tiff` into `out`, exactly
/// [`plane_len`] long, as `f32` in the file's own units.
///
/// `scratch` is the caller's, kept from one plane to the next, so a run over
/// a compressed stack decodes with no allocation once it is warm. Every
/// sample of `out` is written, and none is read first, so it needs no
/// clearing; on `Err` what it holds is unspecified.
pub(crate) fn plane_f32_into(
    tiff: &TiffStack,
    ifd: usize,
    sample: usize,
    out: &mut [f32],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    Ok(frame(tiff, ifd)?.read_into(File, sample, &mut [out], scratch)?)
}

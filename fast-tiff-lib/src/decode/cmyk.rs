//! CMYK (Separated) -> RGB.
//!
//! A Separated image stores *ink coverage*, not light: 0 means no ink, so the
//! paper shows through. Display needs the opposite convention, and the standard
//! conversion with no colour management applied is
//!
//! ```text
//! R = (1 - C)(1 - K)     G = (1 - M)(1 - K)     B = (1 - Y)(1 - K)
//! ```
//!
//! over values normalized to 0..1. (Verified component-for-component against
//! Pillow's `CMYK -> RGB`, including mixed-ink values where the tempting
//! `255 - C - K` form diverges.)
//!
//! These are units of their own ([`CmykRgb8`], [`CmykRgb16`]) rather than a
//! change to what [`Stored`] or [`Display`] give, which is exactly one plane
//! *per sample*, in sample order. Reinterpreting four inks as three colours is
//! a different operation and says so in its name.
//!
//! The conversion runs on a piece's inks as they are decoded, so chunky and
//! planar frames, every codec, tiles and the predictor undo all come along
//! unchanged. Samples past the fourth — an alpha channel, or spot colours in a
//! file that also carries the four process inks — take no part in the
//! composite, matching what a viewer can meaningfully show.
//!
//! [`CmykRgb8`]: crate::units::CmykRgb8
//! [`CmykRgb16`]: crate::units::CmykRgb16
//! [`Stored`]: crate::units::Stored
//! [`Display`]: crate::units::Display

use super::convert::{round_u16, round_u8};
use super::core::{Inks, Src};
use crate::error::{bail, Result};
use crate::index::FrameInfo;
use crate::DecodeScratch;

/// One ink pair -> one display component, in 0..=1 float space.
#[inline]
fn ink_to_component(ink: f32, black: f32) -> f32 {
    (1.0 - ink) * (1.0 - black)
}

/// Gate the CMYK readers on a frame the formula is actually defined for.
pub(crate) fn require_cmyk(frame: &FrameInfo, method: &str) -> Result<()> {
    if frame.is_cmyk() {
        return Ok(());
    }
    bail!(
        InvalidInput,
        "{method} requires a CMYK frame (PhotometricInterpretation=5, InkSet=1, 4+ unsigned \
         8- or 16-bit samples); this frame is photometric={}, ink_set={}, {} sample(s)/px, \
         {}-bit {:?}",
        frame.photometric,
        frame.ink_set,
        frame.samples_per_pixel,
        frame.bits_per_sample,
        frame.sample_format,
    );
}

/// [`CmykRgb8`](crate::units::CmykRgb8) planes `first..`.
pub(crate) fn rgb8(
    src: Src<'_, '_>,
    first: usize,
    outs: &mut [&mut [u8]],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    require_cmyk(src.info, "a CmykRgb8 read")?;
    if src.info.bits_per_sample != 8 {
        bail!(
            InvalidInput,
            "a CmykRgb8 read takes 8-bit inks, and this frame's are {}-bit (read them as \
             CmykRgb16)",
            src.info.bits_per_sample
        );
    }
    Inks { src, scratch }.convert(first, outs, &|b| b[0] as u32, &|ink, black| {
        let black = black as f32 / 255.0;
        round_u8(ink_to_component(ink as f32 / 255.0, black) * 255.0)
    })
}

/// [`CmykRgb16`](crate::units::CmykRgb16) planes `first..`: in the 0..=65535
/// space the crate's `u16` reads use, 8-bit inks widened the way a
/// [`Display`](crate::units::Display) read widens them first, so a caller on
/// the 16-bit path gets consistent units either way.
pub(crate) fn rgb16(
    src: Src<'_, '_>,
    first: usize,
    outs: &mut [&mut [u16]],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    require_cmyk(src.info, "a CmykRgb16 read")?;
    const FULL: f32 = u16::MAX as f32;
    let order = src.order;
    let ink: &(dyn Fn(&[u8]) -> u32 + Sync) = if src.info.bits_per_sample == 8 {
        &|b: &[u8]| ((b[0] as u32) << 8) | b[0] as u32
    } else {
        &move |b: &[u8]| order.u16(b) as u32
    };
    Inks { src, scratch }.convert(first, outs, ink, &|ink, black| {
        let black = black as f32 / FULL;
        round_u16(ink_to_component(ink as f32 / FULL, black) * FULL)
    })
}

//! The units a [`Frame`] read gives its samples in, as types.
//!
//! Each marker fixes the sample type of the destination at compile time —
//! [`Display`] reads into `u16`, [`File`] into `f32`, [`Stored`] into bytes —
//! and what the read reports back. One pair of calls, [`Frame::read`] and
//! [`Frame::read_into`], then covers what used to take a reader per
//! combination of unit, destination and plane selection.
//!
//! | marker | sample | what each element is |
//! |---|---|---|
//! | [`Stored`]`(order)` | `u8` | the plane's samples as the file stores them, each sample's bytes in `order`; 4-bit indices one per byte |
//! | [`Display`] | `u16` | the display scale of [`DisplayMap`](crate::DisplayMap): 8-bit widened, signed offset, wide data rescaled from `range` |
//! | [`File`] | `f32` | the file's own value: 200 is 200.0, -1 is -1.0, a float is itself |
//! | [`CmykRgb8`], [`CmykRgb16`] | `u8`, `u16` | a CMYK frame converted to red, green and blue |

use crate::decode::core;
use crate::frame::{DisplayMaps, Frame};
use crate::ifd::ByteOrder;
use crate::{DecodeScratch, FrameInfo, Result};

pub(crate) mod sealed {
    /// Proof of being inside the crate: the hidden methods of
    /// [`Units`](super::Units) take one, so only this crate can call them, and
    /// only this crate's markers implement the trait.
    pub struct Private;
    pub trait Sealed {}
}

/// The units a read gives its samples in. Implemented by the markers in this
/// module, and only by them.
pub trait Units: Copy + Send + Sync + sealed::Sealed {
    /// One element of a destination.
    type Sample: bytemuck::Pod + Default + Send + Sync;
    /// What a read says about what it did, besides the samples.
    type Report;

    #[doc(hidden)]
    fn planes(self, frame: &Frame<'_>, _: sealed::Private) -> usize;
    #[doc(hidden)]
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize>;
    #[doc(hidden)]
    fn borrow<'a>(
        self,
        frame: &Frame<'a>,
        plane: usize,
        _: sealed::Private,
    ) -> Option<&'a [Self::Sample]>;
    #[doc(hidden)]
    fn borrowed_report(self, frame: &Frame<'_>, _: sealed::Private) -> Self::Report;
    #[doc(hidden)]
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [Self::Sample]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<Self::Report>;
}

/// A plane's samples exactly as the file stores them — decompressed and with
/// the predictor undone, nothing widened, offset or rescaled — each sample's
/// bytes in the given order. Read as `&[i16]`, `&[u32]`, ... through bytemuck
/// with [`ByteOrder::host`]. 4-bit palette indices come one per byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stored(pub ByteOrder);

/// The `u16` display scale every bit depth is put on, so a renderer can treat
/// every channel alike; [`DisplayMap`](crate::DisplayMap) says how a file
/// value maps onto it. A 32- or 64-bit plane is rescaled linearly from
/// `range`, in the file's units, or from its own smallest and largest values
/// when `range` is `None`. 4-bit samples are palette indices, which have no
/// display scale: read them as [`Stored`] or [`File`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Display {
    pub range: Option<(f32, f32)>,
}

/// `f32` in the file's own units, for every depth: an 8-bit 200 is 200.0, a
/// signed 16-bit -1 is -1.0, a float is itself (64-bit narrowed to `f32`), a
/// 4-bit index is its index. 16-bit floats are refused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct File;

/// A CMYK frame ([`FrameInfo::is_cmyk`], 8-bit inks) as red, green and blue
/// bytes: `R = (1 - C)(1 - K)` and so on, over inks normalised to 0..=1.
/// Three planes, whatever the frame's sample count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CmykRgb8;

/// [`CmykRgb8`] on the `u16` display scale, from 8- or 16-bit inks (8-bit
/// inks widened first, as [`Display`] widens them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CmykRgb16;

impl sealed::Sealed for Stored {}
impl sealed::Sealed for Display {}
impl sealed::Sealed for File {}
impl sealed::Sealed for CmykRgb8 {}
impl sealed::Sealed for CmykRgb16 {}

fn spp(info: &FrameInfo) -> usize {
    (info.samples_per_pixel as usize).max(1)
}

impl Units for Stored {
    type Sample = u8;
    type Report = ();
    fn planes(self, frame: &Frame<'_>, _: sealed::Private) -> usize {
        spp(frame.info())
    }
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize> {
        core::stored_plane_len(info)
    }
    fn borrow<'a>(self, frame: &Frame<'a>, plane: usize, _: sealed::Private) -> Option<&'a [u8]> {
        core::borrow_stored(frame.src(), self.0, plane)
    }
    fn borrowed_report(self, _: &Frame<'_>, _: sealed::Private) {}
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [u8]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<()> {
        core::stored(frame.src(), self.0, first, out, scratch)
    }
}

impl Units for Display {
    type Sample = u16;
    type Report = DisplayMaps;
    fn planes(self, frame: &Frame<'_>, _: sealed::Private) -> usize {
        spp(frame.info())
    }
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize> {
        core::display_plane_len(info)
    }
    fn borrow<'a>(self, frame: &Frame<'a>, plane: usize, _: sealed::Private) -> Option<&'a [u16]> {
        core::borrow_display(frame.src(), plane)
    }
    fn borrowed_report(self, _: &Frame<'_>, _: sealed::Private) -> DisplayMaps {
        // Only unsigned 16-bit samples are ever borrowed.
        DisplayMaps::One(crate::DisplayMap::Identity)
    }
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [u16]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<DisplayMaps> {
        core::display(frame.src(), self.range, first, out, scratch)
    }
}

impl Units for File {
    type Sample = f32;
    type Report = ();
    fn planes(self, frame: &Frame<'_>, _: sealed::Private) -> usize {
        spp(frame.info())
    }
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize> {
        core::file_plane_len(info)
    }
    fn borrow<'a>(self, frame: &Frame<'a>, plane: usize, _: sealed::Private) -> Option<&'a [f32]> {
        core::borrow_file(frame.src(), plane)
    }
    fn borrowed_report(self, _: &Frame<'_>, _: sealed::Private) {}
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [f32]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<()> {
        core::file(frame.src(), first, out, scratch)
    }
}

impl Units for CmykRgb8 {
    type Sample = u8;
    type Report = ();
    fn planes(self, _: &Frame<'_>, _: sealed::Private) -> usize {
        3
    }
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize> {
        let n = core::cmyk_plane_len(info, "a CmykRgb8 read")?;
        crate::decode::cmyk::require_8bit_inks(info)?;
        Ok(n)
    }
    fn borrow<'a>(self, _: &Frame<'a>, _: usize, _: sealed::Private) -> Option<&'a [u8]> {
        None
    }
    fn borrowed_report(self, _: &Frame<'_>, _: sealed::Private) {}
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [u8]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<()> {
        crate::decode::cmyk::rgb8(frame.src(), first, out, scratch)
    }
}

impl Units for CmykRgb16 {
    type Sample = u16;
    type Report = ();
    fn planes(self, _: &Frame<'_>, _: sealed::Private) -> usize {
        3
    }
    fn plane_len(self, info: &FrameInfo, _: sealed::Private) -> Result<usize> {
        core::cmyk_plane_len(info, "a CmykRgb16 read")
    }
    fn borrow<'a>(self, _: &Frame<'a>, _: usize, _: sealed::Private) -> Option<&'a [u16]> {
        None
    }
    fn borrowed_report(self, _: &Frame<'_>, _: sealed::Private) {}
    fn read_into(
        self,
        frame: &Frame<'_>,
        first: usize,
        out: &mut [&mut [u16]],
        scratch: &mut DecodeScratch,
        _: sealed::Private,
    ) -> Result<()> {
        crate::decode::cmyk::rgb16(frame.src(), first, out, scratch)
    }
}

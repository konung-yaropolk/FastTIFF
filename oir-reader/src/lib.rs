//! A reader for Olympus/Evident FluoView OIR acquisitions.
//!
//! ```no_run
//! let acquisition = oir_reader::read("recording.oir")?;
//! let shape = acquisition.shape();
//! println!(
//!     "{}x{}, {} channel(s), {} slice(s), {} frame(s)",
//!     shape.width, shape.height, shape.channels, shape.slices, shape.frames
//! );
//! if let Some(planes) = acquisition.planes().as_u16() {
//!     // Channel 0, slice 0, the last time point.
//!     let last = shape.plane_index(0, 0, shape.frames - 1).unwrap();
//!     let plane: &[u16] = &planes[last];
//!     println!("first pixel {}", plane[0]);
//! }
//! println!("{:?} um per pixel", acquisition.record().pixel_x_um());
//! # Ok::<(), oir_reader::Error>(())
//! ```
//!
//! OIR is proprietary and has no published specification. What follows was
//! determined from real FluoView acquisitions and **checked against an
//! oracle**: the same acquisitions exported to TIFF by the acquisition software
//! itself. The planes this reader reassembles from the container are
//! byte-identical to the exported ones, which is the only kind of evidence worth
//! having about a format nobody documents. No vendor SDK, no specification and
//! no code from another reader went into it.
//!
//! # The container
//!
//! ```text
//!   0x00  "OLYMPUSRAWFORMAT"
//!   0x20  u64  total file size
//!   0x28  u64  offset of the block index
//!   0x48  "FLUOVIEW"
//!   ...   blocks
//!   index: u32 0xFFFFFFFF, u32, u32, then one u64 file offset per block
//! ```
//!
//! Every block is `u32 length, u32 type, length bytes`. They come in pairs: a
//! *descriptor* (type 3) naming a plane and the byte range of it that follows,
//! then the block carrying those bytes. A 512x512 16-bit plane arrives as 34
//! chunks of 15,360 bytes and one of 2,048 — so a plane is scattered, and
//! reassembling it is the whole job.
//!
//! Descriptor payload: `u32 offset-within-plane, u32 length, u32 name length,
//! name`.
//!
//! # Axes, and the order of the planes
//!
//! Names look like `t001_0_1_<uid>_<chunk>` in a timelapse and
//! `z001_0_1_<uid>_<chunk>` in a z-stack, and the UID is the *channel* — two
//! channels of one slice differ only there. Dropping it would merge them into
//! one plane whose chunks overlap, which is not a decode error and so does not
//! announce itself; it just produces the wrong picture.
//!
//! The planes come back in [`Planes`] in ImageJ's `xyczt` order — channel
//! fastest, then slice, then time — and [`Shape::plane_index`] says where any
//! one of them is. Channels and slices are counted from the plane names when the
//! names distinguish them, and taken from the acquisition record when they do
//! not; a count from the record that does not divide the planes evenly is
//! ignored rather than believed.
//!
//! # Split acquisitions
//!
//! A long recording is split into `name.oir`, `name_00001`, `name_00002`, …
//! (with or without the extension). Each part is a complete container holding
//! whole planes, so reading the first reads them all — see
//! [`acquisition_parts`].
//!
//! # Files that are not finished
//!
//! The block index is the **last** thing an acquisition writes. A recording
//! still in progress, or one whose tail was lost, has no index — and the
//! offset at `0x28` that should say where it lives is either zero or points
//! past the end of what was written.
//!
//! Blocks are self-delimiting, so when the index is missing or unusable the
//! stream is **walked** from the first block instead, and
//! [`Part::block_source`] says so. The walk stops at the first block that does
//! not fit in what is there, which is exactly what a half-written tail looks
//! like, and everything in front of it is whole. A plane the acquisition did
//! not finish is then left out — [`Acquisition::dropped_planes`] — rather than
//! kept as a mostly-black frame.
//!
//! This is a snapshot, not a subscription: what was on disk when the file was
//! opened is what is read. Reading it again later reads more.
//!
//! # Metadata
//!
//! Everything said about the acquisition comes out of the OIR itself: the
//! [`Record`] is translated from the file's own XML, and
//! [`Acquisition::fluoview_text`] writes it as the `"key"\t"value"` record the
//! acquisition software exports beside an OIR as a `.txt`.
//!
//! That format is kept, and the `.txt` is never read. The format is kept
//! because things downstream parse it: it is the one place an analysis can
//! learn when the stimulus fired, and it should not have to learn two dialects
//! to do it. The file is not read because it is a second, detachable copy of
//! what the OIR already contains — it goes missing, it gets renamed, it is left
//! behind when the `.oir` is moved, and (since the exported name carries a
//! series number) it can belong to a different acquisition in the same folder.
//! Reading the container means the record cannot disagree with the pixels
//! beside it.
//!
//! # Reading: mapped, buffered, or in memory
//!
//! Each part is read **once, front to back**: a descriptor and the pixels it
//! describes sit next to each other, so the pixels are copied out the moment
//! their descriptor has been read, and the metadata documents are picked up in
//! the same pass.
//!
//! How the bytes get there is [`Access`]:
//!
//! * **Mapped**, by default where the `mmap` feature is on and the target is
//!   unix or Windows ([`MMAP_SUPPORTED`]). Everything but the pixels is parsed
//!   in place in the mapping, and each pixel chunk is copied exactly once, into
//!   the plane it belongs to. A plane is thirty-five chunks scattered among the
//!   headers, so it cannot be borrowed from the file; what comes back owns its
//!   buffers, and no mapping outlives the part it maps.
//! * **Buffered**, through a window that slides forward through the part, a few
//!   large reads instead of a quarter of a million small ones. Used where
//!   mapping is not available, for a part whose mapping fails —
//!   [`Part::map_error`] says why — and whenever [`Access::Buffered`] asks.
//! * **In memory**, by [`read_bytes`] and [`Options::gather_bytes`], read in
//!   place. The way in where there is no filesystem, as in a browser.
//!
//! All three produce the same acquisition from the same bytes.
//!
//! A mapping makes one promise the reader cannot keep for itself: that the file
//! is not *truncated* while it is mapped. An acquisition only ever appends, so a
//! recording in progress is safe to map; a file another process may shorten,
//! or one on a network share that may drop, is not, because there an I/O
//! failure is a crash rather than an error. Read those with
//! [`Access::Buffered`].
//!
//! # What it does not do
//!
//! Every plane is held in memory: an acquisition is read whole, and the largest
//! this was built against is about thirteen gigabytes of pixels. It does not
//! write OIR, read the older OIB/OIF formats, or know axes other than `t` and
//! `z`.
//!
//! # From C and C++
//!
//! With the `capi` feature the crate is also a C library: `capi` is the
//! interface, `include/oir_reader.h` declares it, and
//!
//! ```text
//! cargo rustc -p oir-reader --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! builds it as a shared and a static library. That module is allowed
//! `unsafe`, to take C's pointers — the crate's only other `unsafe` is the one
//! call that maps a file — and the reader it calls is the same safe code
//! either way. The README's "From C and C++" has the rest: what
//! lands where, how to link it, and what a C caller should know.

#![warn(missing_docs)]
#![deny(unsafe_code)]

mod acquisition;
#[cfg(feature = "capi")]
pub mod capi;
mod container;
mod error;
#[cfg(test)]
mod fixture;
mod layout;
#[cfg(test)]
mod legacy;
#[cfg(all(feature = "mmap", any(unix, windows)))]
mod map;
mod meta;
mod options;
mod planes;
mod read;
mod source;

pub use acquisition::{
    Acquisition, BlockSource, Gathered, Part, Planes, ReadPath, SampleType, Shape,
};
pub use error::{Error, ErrorKind};
pub use meta::{Channel, Event, Record};
pub use options::{Access, Options};
pub use read::acquisition_parts;

use std::path::Path;

/// Whether this build memory-maps the parts it reads: the `mmap` feature, on a
/// unix or Windows target. When it is false, [`Access::Auto`] reads every part
/// through a buffer.
pub const MMAP_SUPPORTED: bool = cfg!(all(feature = "mmap", any(unix, windows)));

/// Whether `head`, the opening bytes of a file, carries the OIR signature.
///
/// The signature is the first sixteen bytes, so that many is enough to know.
pub fn is_oir(head: &[u8]) -> bool {
    head.starts_with(container::MAGIC)
}

/// Read the acquisition `first` begins, every part of it, with the default
/// [`Options`].
pub fn read(first: impl AsRef<Path>) -> Result<Acquisition, Error> {
    Options::new().gather(first, |_| true)?.assemble()
}

/// Read an acquisition from its parts, already in memory and in order.
///
/// Nothing is copied but the pixels, into their planes. This is the way in on
/// a target with no filesystem.
pub fn read_bytes(parts: &[&[u8]]) -> Result<Acquisition, Error> {
    Options::new().gather_bytes(parts, |_| true)?.assemble()
}

/// The README's examples, compiled as doctests so they cannot drift.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

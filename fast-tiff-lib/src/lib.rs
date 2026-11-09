//! Lazy, memory-mapped reading and streaming writing of multi-frame TIFF
//! stacks: IFD indexing, ImageJ and OME-TIFF metadata, and per-frame decoding.
//! The README has the tour.
//!
//! # From C and C++
//!
//! With the `capi` feature the crate is also a C library: `capi` is the
//! interface, `include/fast_tiff_lib.h` declares it, and
//!
//! ```text
//! cargo rustc -p fast-tiff-lib --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! builds it as a shared and a static library. The README's "From C and C++"
//! has the rest: what lands where, how to link it, and what a C caller should
//! know.
//!
//! # Errors
//!
//! Every fallible call returns this crate's [`Result`], whose [`Error`] says
//! what went wrong in words and, through [`Error::kind`], what kind of failure
//! it was: the bytes are [not a TIFF](ErrorKind::NotTiff) at all, the file is
//! [damaged](ErrorKind::Malformed), it uses something this library
//! [does not read](ErrorKind::Unsupported) or this build
//! [left out](ErrorKind::Unavailable), it is [too large](ErrorKind::TooLarge)
//! for this target, the operating system [failed](ErrorKind::Io) an open, a
//! read or a write, or the call itself [asked for what cannot be
//! done](ErrorKind::InvalidInput). Decide on the kind, never on the words:
//!
//! ```no_run
//! use fast_tiff_lib::{ErrorKind, TiffStack};
//!
//! match TiffStack::from_bytes(std::fs::read("scan.dat")?) {
//!     Ok(stack) => println!("{} frames", stack.frames.len()),
//!     // Some other format: hand it to another reader.
//!     Err(e) if e.kind() == ErrorKind::NotTiff => {}
//!     Err(e) => eprintln!("{e:#}"),
//! }
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! An [`Error`] is a [`std::error::Error`], `Send`, `Sync` and `'static`, so
//! `?` takes it into `anyhow`, `Box<dyn Error>` and the like, and `{:#}`
//! prints its whole cause chain.

#[cfg(feature = "capi")]
pub mod capi;
pub mod decode;
pub mod display;
pub mod encode;
mod error;
pub mod ifd;
pub mod index;
pub mod metadata;

pub use decode::{
    frame_float_minmax, parallel_decode_enabled, parallel_decode_scope, preload_frames_f32,
    preload_frames_u16, preload_frames_u8, read_frame_bytes_into, read_frame_f32,
    read_frame_f32_into, read_frame_u16, read_frame_u16_into, read_frame_u8, read_frame_u8_into,
    read_plane_f32, read_plane_f32_into, read_plane_rgb_u16, read_plane_rgb_u8, read_plane_u16,
    read_plane_u16_into, read_plane_u8, read_plane_u8_into, read_planes_f32, read_planes_f32_into,
    read_planes_rgb_u16, read_planes_rgb_u8, read_planes_u16, read_planes_u16_into, read_planes_u8,
    read_planes_u8_into, set_parallel_decode, set_thread_parallel_decode, ParallelDecodeScope,
};
pub use display::DisplayMap;
pub use encode::{
    SampleType, TiffWriter, WriterOptions, DEFAULT_DEFLATE_LEVEL, DEFAULT_ZSTD_LEVEL,
};
pub use error::{Error, ErrorKind, Result};
pub use ifd::{ByteOrder, TiffFlavor};
pub use index::{
    Compression, FrameInfo, Region, RowBand, SampleFormat, SampledBand, Strips, TiffStack,
};
pub use metadata::roi::{point_rois_from_imagej, roi_set_zip, PointRoi, RoiPoint};
pub use metadata::{
    color_ramp_lut, colormap_to_lut, composite_color, default_composite_lut, default_lut_for,
    grayscale_lut, resolve_dimensions, ChannelDisplay, DisplayMode, MetadataFormat,
    ResolvedDimensions, StackMeta, StackMetaWrite,
};

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

#[cfg(feature = "capi")]
pub mod capi;
pub mod decode;
pub mod encode;
pub mod ifd;
pub mod index;
pub mod metadata;

pub use decode::{
    frame_float_minmax, preload_frames_f32, preload_frames_u16, preload_frames_u8, read_frame_f32,
    read_frame_f32_into, read_frame_u16, read_frame_u16_into, read_frame_u8, read_frame_u8_into,
    read_plane_f32, read_plane_f32_into, read_plane_rgb_u16, read_plane_rgb_u8, read_plane_u16,
    read_plane_u16_into, read_plane_u8, read_plane_u8_into, read_planes_f32, read_planes_f32_into,
    read_planes_rgb_u16, read_planes_rgb_u8, read_planes_u16, read_planes_u16_into, read_planes_u8,
    read_planes_u8_into, set_parallel_decode,
};
pub use encode::{
    SampleType, TiffWriter, WriterOptions, DEFAULT_DEFLATE_LEVEL, DEFAULT_ZSTD_LEVEL,
};
pub use ifd::{ByteOrder, NotTiff, TiffFlavor};
pub use index::{
    Compression, FrameInfo, Region, RowBand, SampleFormat, SampledBand, Strips, TiffStack,
};
pub use metadata::{
    color_ramp_lut, colormap_to_lut, composite_color, default_composite_lut, default_lut_for,
    grayscale_lut, resolve_dimensions, ChannelDisplay, DisplayMode, MetadataFormat,
    ResolvedDimensions, StackMeta, StackMetaWrite,
};

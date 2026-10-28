// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! A C interface, for programs in C, C++, or anything else that can call C.
//!
//! Behind the `capi` feature, and built as a library of its own by
//! `cargo rustc`, which picks the crate type at build time:
//!
//! ```text
//! cargo rustc -p fast-tiff-lib --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! The declarations are in `include/fast_tiff_lib.h`. That header and this
//! module describe the same structures and the same functions, and only the
//! tests hold them together: a field added here goes there too, in the same
//! place.
//!
//! The shape follows the Rust API. Reading: open a stack with [`ftl_open`]
//! (memory-mapped) or [`ftl_open_bytes`] (copied from memory), look at it with
//! [`ftl_frame_info`] and [`ftl_metadata`], and decode a plane into the
//! caller's buffer with [`ftl_read_u8`], [`ftl_read_u16`] or [`ftl_read_f32`],
//! sized by [`ftl_sample_count`]. Writing: fill an [`FtlWriterOptions`] from
//! [`ftl_writer_options_default`], [`ftl_writer_create`], append frames, and
//! [`ftl_writer_finish`]. With the habits a C caller expects: every call that
//! can fail returns a status, `FTL_OK` or why not; [`ftl_last_error`] says why
//! in words; handles are opaque and freed by the library. Nothing unwinds into
//! C: a panic comes back as [`FTL_FAILED`], as long as the library was built
//! with `panic = "unwind"` (the default, and this workspace's release
//! profile).
//!
//! Decoded planes are copied out of the library into the caller's buffer, once:
//! a zero-copy borrow of the mapping cannot cross into C, whose buffer the
//! decoders' `Vec`-based `*_into` forms cannot write to either. The fast path
//! therefore costs a `memcpy` from the mapping and nothing more.

use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs::File;
use std::io::BufWriter;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    frame_float_minmax, read_frame_f32, read_frame_u16, read_frame_u8, read_planes_f32,
    read_planes_rgb_u16, read_planes_rgb_u8, read_planes_u16, read_planes_u8, resolve_dimensions,
    ByteOrder, Compression, DisplayMode, FrameInfo, MetadataFormat, NotTiff, ResolvedDimensions,
    SampleFormat, SampleType, StackMeta, StackMetaWrite, TiffFlavor, TiffStack, TiffWriter,
    WriterOptions,
};

/// The call did what it was asked.
pub const FTL_OK: c_int = 0;
/// A pointer, size, index, buffer or option the call cannot take.
pub const FTL_INVALID_ARGUMENT: c_int = 1;
/// Not in this build or on this target: `ftl_open` without the `mmap`
/// feature, a file written on a target with no filesystem, ZSTD without
/// `codec-zstd`.
pub const FTL_UNAVAILABLE: c_int = 2;
/// Something inside the library went wrong: a panic, or a decoder that
/// produced a size it should not have.
pub const FTL_FAILED: c_int = 3;
/// A file could not be opened, created, mapped, read or written.
pub const FTL_IO: c_int = 4;
/// The bytes are not a TIFF at all (no `II`/`MM` byte-order mark, or a magic
/// number other than 42 and 43).
pub const FTL_NOT_TIFF: c_int = 5;
/// A TIFF this library cannot read, or not the way asked: malformed, an
/// unsupported codec or bit depth, or a frame of the wrong depth for the
/// reader called (a 16-bit frame through [`ftl_read_u8`]).
pub const FTL_FORMAT: c_int = 6;

/// TIFF `SampleFormat` 1, [`SampleFormat::UnsignedInt`].
pub const FTL_SAMPLE_UNSIGNED: u16 = 1;
/// TIFF `SampleFormat` 2, [`SampleFormat::SignedInt`].
pub const FTL_SAMPLE_SIGNED: u16 = 2;
/// TIFF `SampleFormat` 3, [`SampleFormat::Float`].
pub const FTL_SAMPLE_FLOAT: u16 = 3;

/// TIFF `Compression` 1, [`Compression::None`].
pub const FTL_COMPRESSION_NONE: u16 = 1;
/// TIFF `Compression` 5, [`Compression::Lzw`].
pub const FTL_COMPRESSION_LZW: u16 = 5;
/// TIFF `Compression` 8, [`Compression::Deflate`] (32946, the old code for
/// the same thing, reads back as 8).
pub const FTL_COMPRESSION_DEFLATE: u16 = 8;
/// TIFF `Compression` 32773, [`Compression::PackBits`].
pub const FTL_COMPRESSION_PACKBITS: u16 = 32773;
/// TIFF `Compression` 50000, [`Compression::Zstd`] (34926, the withdrawn
/// experimental code, reads back as 50000).
pub const FTL_COMPRESSION_ZSTD: u16 = 50000;

/// [`SampleType::U8`].
pub const FTL_TYPE_U8: c_int = 0;
/// [`SampleType::I8`].
pub const FTL_TYPE_I8: c_int = 1;
/// [`SampleType::U16`].
pub const FTL_TYPE_U16: c_int = 2;
/// [`SampleType::I16`].
pub const FTL_TYPE_I16: c_int = 3;
/// [`SampleType::U32`].
pub const FTL_TYPE_U32: c_int = 4;
/// [`SampleType::I32`].
pub const FTL_TYPE_I32: c_int = 5;
/// [`SampleType::F32`].
pub const FTL_TYPE_F32: c_int = 6;
/// [`SampleType::U64`].
pub const FTL_TYPE_U64: c_int = 7;
/// [`SampleType::I64`].
pub const FTL_TYPE_I64: c_int = 8;
/// [`SampleType::F64`].
pub const FTL_TYPE_F64: c_int = 9;

/// [`MetadataFormat::None`]: no recognised metadata (read), none written
/// (write).
pub const FTL_METADATA_NONE: c_int = 0;
/// [`MetadataFormat::ImageJ`].
pub const FTL_METADATA_IMAGEJ: c_int = 1;
/// [`MetadataFormat::Ome`].
pub const FTL_METADATA_OME: c_int = 2;

/// [`DisplayMode::Grayscale`].
pub const FTL_MODE_GRAYSCALE: c_int = 0;
/// [`DisplayMode::Composite`].
pub const FTL_MODE_COMPOSITE: c_int = 1;
/// [`DisplayMode::Color`].
pub const FTL_MODE_COLOR: c_int = 2;

/// [`FtlWriterOptions::compression_level`] for the library's default level
/// ([`DEFAULT_DEFLATE_LEVEL`](crate::DEFAULT_DEFLATE_LEVEL),
/// [`DEFAULT_ZSTD_LEVEL`](crate::DEFAULT_ZSTD_LEVEL)).
pub const FTL_LEVEL_DEFAULT: i32 = i32::MIN;

/// One plane's layout: [`FrameInfo`], in C's terms, with its `is_*` helpers
/// answered. Codes are TIFF's own: `compression` is an `FTL_COMPRESSION_*`
/// or any other TIFF code the file used, `sample_format` an `FTL_SAMPLE_*`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FtlFrameInfo {
    /// [`FrameInfo::width`].
    pub width: u32,
    /// [`FrameInfo::height`].
    pub height: u32,
    /// [`FrameInfo::bits_per_sample`].
    pub bits_per_sample: u16,
    /// [`FrameInfo::samples_per_pixel`].
    pub samples_per_pixel: u16,
    /// [`FrameInfo::sample_format`], an `FTL_SAMPLE_*`.
    pub sample_format: u16,
    /// [`FrameInfo::compression`], as its TIFF code.
    pub compression: u16,
    /// [`FrameInfo::predictor`]: 1 none, 2 horizontal, 3 floating point.
    pub predictor: u16,
    /// [`FrameInfo::photometric`]: 1 grayscale, 2 RGB, 3 palette, 5 separated.
    pub photometric: u16,
    /// [`FrameInfo::planar_config`]: 1 chunky, 2 planar.
    pub planar_config: u16,
    /// [`FrameInfo::ink_set`].
    pub ink_set: u16,
    /// [`FrameInfo::rows_per_strip`].
    pub rows_per_strip: u32,
    /// The tile width for a tiled frame, 0 for a stripped one.
    pub tile_width: u32,
    /// The tile height for a tiled frame, 0 for a stripped one.
    pub tile_height: u32,
    /// [`FrameInfo::is_tiled`].
    pub tiled: bool,
    /// [`FrameInfo::is_rgb`].
    pub rgb: bool,
    /// [`FrameInfo::is_planar`].
    pub planar: bool,
    /// [`FrameInfo::is_palette`].
    pub palette: bool,
    /// [`FrameInfo::is_cmyk`]: what [`ftl_read_rgb_u8`] and
    /// [`ftl_read_rgb_u16`] take.
    pub cmyk: bool,
}

fn compression_code(c: Compression) -> u16 {
    match c {
        Compression::None => FTL_COMPRESSION_NONE,
        Compression::Lzw => FTL_COMPRESSION_LZW,
        Compression::PackBits => FTL_COMPRESSION_PACKBITS,
        Compression::Deflate => FTL_COMPRESSION_DEFLATE,
        Compression::Zstd => FTL_COMPRESSION_ZSTD,
        Compression::Other(code) => code,
    }
}

impl From<&FrameInfo> for FtlFrameInfo {
    fn from(f: &FrameInfo) -> Self {
        let (tile_width, tile_height) = f.tile_size.unwrap_or((0, 0));
        FtlFrameInfo {
            width: f.width,
            height: f.height,
            bits_per_sample: f.bits_per_sample,
            samples_per_pixel: f.samples_per_pixel,
            sample_format: match f.sample_format {
                SampleFormat::UnsignedInt => FTL_SAMPLE_UNSIGNED,
                SampleFormat::SignedInt => FTL_SAMPLE_SIGNED,
                SampleFormat::Float => FTL_SAMPLE_FLOAT,
            },
            compression: compression_code(f.compression),
            predictor: f.predictor,
            photometric: f.photometric,
            planar_config: f.planar_config,
            ink_set: f.ink_set,
            rows_per_strip: f.rows_per_strip,
            tile_width,
            tile_height,
            tiled: f.is_tiled(),
            rgb: f.is_rgb(),
            planar: f.is_planar(),
            palette: f.is_palette(),
            cmyk: f.is_cmyk(),
        }
    }
}

/// What the file's metadata says: [`StackMeta`], in C's terms, without the
/// display side (LUTs and contrast windows). A value the file does not give
/// is NaN (test with `isnan`), and `loop_playback` is -1.
///
/// `channels`, `slices` and `frames` are the file's own labels — the plane
/// order is `xyczt`, channel fastest — or 1, 1 and the IFD count when there
/// is no recognised metadata. [`ftl_resolve_dimensions`] re-reads them the
/// way the viewer does.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FtlMetadata {
    /// [`StackMeta::channels`].
    pub channels: usize,
    /// [`StackMeta::slices`].
    pub slices: usize,
    /// [`StackMeta::frames`].
    pub frames: usize,
    /// [`StackMeta::source_format`], an `FTL_METADATA_*`.
    pub format: c_int,
    /// [`StackMeta::mode`], an `FTL_MODE_*`.
    pub mode: c_int,
    /// [`StackMeta::pixel_width`], in the unit [`ftl_metadata_unit`] gives.
    pub pixel_width: f64,
    /// [`StackMeta::pixel_height`], in the same unit.
    pub pixel_height: f64,
    /// [`StackMeta::spacing`], the Z step, in the same unit.
    pub spacing: f64,
    /// [`StackMeta::calibration`]: a raw sample `r` stands for
    /// `calibration[0] + calibration[1] * r`.
    pub calibration: [f64; 2],
    /// [`StackMeta::fps`].
    pub fps: f64,
    /// [`StackMeta::frame_interval_s`], seconds.
    pub frame_interval_s: f64,
    /// [`StackMeta::loop_playback`]: 1 yes, 0 no, -1 not said.
    pub loop_playback: c_int,
}

fn metadata_code(f: MetadataFormat) -> c_int {
    match f {
        MetadataFormat::None => FTL_METADATA_NONE,
        MetadataFormat::ImageJ => FTL_METADATA_IMAGEJ,
        MetadataFormat::Ome => FTL_METADATA_OME,
    }
}

fn mode_code(m: DisplayMode) -> c_int {
    match m {
        DisplayMode::Grayscale => FTL_MODE_GRAYSCALE,
        DisplayMode::Composite => FTL_MODE_COMPOSITE,
        DisplayMode::Color => FTL_MODE_COLOR,
    }
}

impl From<&StackMeta> for FtlMetadata {
    fn from(m: &StackMeta) -> Self {
        let nan = |v: Option<f64>| v.unwrap_or(f64::NAN);
        let (c0, c1) = m.calibration.unwrap_or((f64::NAN, f64::NAN));
        FtlMetadata {
            channels: m.channels,
            slices: m.slices,
            frames: m.frames,
            format: metadata_code(m.source_format),
            mode: mode_code(m.mode),
            pixel_width: nan(m.pixel_width),
            pixel_height: nan(m.pixel_height),
            spacing: nan(m.spacing),
            calibration: [c0, c1],
            fps: nan(m.fps),
            frame_interval_s: nan(m.frame_interval_s),
            loop_playback: m.loop_playback.map_or(-1, c_int::from),
        }
    }
}

/// [`ResolvedDimensions`], in C's terms.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FtlDims {
    /// [`ResolvedDimensions::channels`].
    pub channels: usize,
    /// [`ResolvedDimensions::slices`].
    pub slices: usize,
    /// [`ResolvedDimensions::frames`].
    pub frames: usize,
    /// [`ResolvedDimensions::triple_axis_warning`]: several channels with Z
    /// and time both real, which three slots cannot show at once.
    pub triple_axis_warning: bool,
}

impl From<ResolvedDimensions> for FtlDims {
    fn from(d: ResolvedDimensions) -> Self {
        FtlDims {
            channels: d.channels,
            slices: d.slices,
            frames: d.frames,
            triple_axis_warning: d.triple_axis_warning,
        }
    }
}

/// [`WriterOptions`] and the [`StackMetaWrite`] inside it, in C's terms.
///
/// Fill it with [`ftl_writer_options_default`] and change what differs — a
/// structure built field by field would break the first time one is added.
/// A double left NaN, a pointer left NULL and `loop_playback` left -1 are not
/// written. The metadata fields (`mode` to the end) are used only when
/// `metadata_format` is not `FTL_METADATA_NONE`; `pixel_width` and
/// `pixel_height` also go to the resolution tags, as the Rust writer does.
///
/// Not here: [`StackMetaWrite::channel_lut`], [`StackMetaWrite::extra`] and
/// [`StackMetaWrite::trailing`].
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FtlWriterOptions {
    /// Frame width, pixels.
    pub width: u32,
    /// Frame height, pixels.
    pub height: u32,
    /// An `FTL_TYPE_*`; the default is `FTL_TYPE_U16`.
    pub sample_type: c_int,
    /// [`WriterOptions::samples_per_pixel`]: 1 grayscale, 3 RGB, 4 or more
    /// with `cmyk` for CMYK.
    pub samples_per_pixel: u16,
    /// An `FTL_COMPRESSION_*`.
    pub compression: u16,
    /// [`WriterOptions::compression_level`], or `FTL_LEVEL_DEFAULT`.
    pub compression_level: i32,
    /// [`WriterOptions::rows_per_strip`], or 0 for the library's choice.
    pub rows_per_strip: u32,
    /// [`WriterOptions::planar`]: frames go in plane after plane rather than
    /// interleaved.
    pub planar: bool,
    /// [`WriterOptions::cmyk`].
    pub cmyk: bool,
    /// [`WriterOptions::predictor`].
    pub predictor: bool,
    /// [`WriterOptions::bigtiff`]: force BigTIFF (it is automatic past 4 GiB).
    pub bigtiff: bool,
    /// [`WriterOptions::description`]: a verbatim `ImageDescription`, UTF-8,
    /// instead of structured metadata.
    pub description: *const c_char,
    /// [`WriterOptions::metadata_format`], an `FTL_METADATA_*`;
    /// `FTL_METADATA_NONE` writes no structured metadata.
    pub metadata_format: c_int,
    /// [`StackMetaWrite::mode`], an `FTL_MODE_*`.
    pub mode: c_int,
    /// Channels per time point; the planes go in `xyczt` order.
    pub channels: usize,
    /// Z slices per time point. Time points are counted at the finish.
    pub slices: usize,
    /// [`StackMetaWrite::unit`], UTF-8.
    pub unit: *const c_char,
    /// [`StackMetaWrite::pixel_size`]'s width, in `unit`s.
    pub pixel_width: f64,
    /// [`StackMetaWrite::pixel_size`]'s height, in `unit`s.
    pub pixel_height: f64,
    /// [`StackMetaWrite::spacing`].
    pub spacing: f64,
    /// [`StackMetaWrite::calibration`], `c0` and `c1`.
    pub calibration: [f64; 2],
    /// [`StackMetaWrite::range`], the display window.
    pub range: [f64; 2],
    /// [`StackMetaWrite::fps`].
    pub fps: f64,
    /// [`StackMetaWrite::frame_interval_s`].
    pub frame_interval_s: f64,
    /// [`StackMetaWrite::loop_playback`]: 1, 0, or -1 for unsaid.
    pub loop_playback: c_int,
    /// [`StackMetaWrite::channel`]: `named_channels` names, UTF-8 (OME only).
    pub channel_names: *const *const c_char,
    /// `named_channels` RGB colours, three bytes each, for `channel_names`.
    pub channel_colors: *const u8,
    /// How many `channel_names` and `channel_colors` there are.
    pub named_channels: usize,
}

fn sample_type_of(v: c_int) -> Result<SampleType, String> {
    Ok(match v {
        FTL_TYPE_U8 => SampleType::U8,
        FTL_TYPE_I8 => SampleType::I8,
        FTL_TYPE_U16 => SampleType::U16,
        FTL_TYPE_I16 => SampleType::I16,
        FTL_TYPE_U32 => SampleType::U32,
        FTL_TYPE_I32 => SampleType::I32,
        FTL_TYPE_F32 => SampleType::F32,
        FTL_TYPE_U64 => SampleType::U64,
        FTL_TYPE_I64 => SampleType::I64,
        FTL_TYPE_F64 => SampleType::F64,
        _ => {
            return Err(format!(
                "sample_type {v} is not one of the FTL_TYPE_* values"
            ))
        }
    })
}

fn compression_of(v: u16) -> Result<Compression, String> {
    Ok(match v {
        FTL_COMPRESSION_NONE => Compression::None,
        FTL_COMPRESSION_LZW => Compression::Lzw,
        FTL_COMPRESSION_DEFLATE => Compression::Deflate,
        FTL_COMPRESSION_PACKBITS => Compression::PackBits,
        FTL_COMPRESSION_ZSTD => Compression::Zstd,
        _ => {
            return Err(format!(
                "compression {v} is not one the writer takes (FTL_COMPRESSION_*)"
            ))
        }
    })
}

fn metadata_format_of(v: c_int) -> Result<MetadataFormat, String> {
    match v {
        FTL_METADATA_NONE => Ok(MetadataFormat::None),
        FTL_METADATA_IMAGEJ => Ok(MetadataFormat::ImageJ),
        FTL_METADATA_OME => Ok(MetadataFormat::Ome),
        _ => Err(format!(
            "metadata_format {v} is not one of the FTL_METADATA_* values"
        )),
    }
}

fn mode_of(v: c_int) -> Result<DisplayMode, String> {
    match v {
        FTL_MODE_GRAYSCALE => Ok(DisplayMode::Grayscale),
        FTL_MODE_COMPOSITE => Ok(DisplayMode::Composite),
        FTL_MODE_COLOR => Ok(DisplayMode::Color),
        _ => Err(format!("mode {v} is not one of the FTL_MODE_* values")),
    }
}

/// `v`, or nothing for NaN; infinity is refused.
fn optional(name: &str, v: f64) -> Result<Option<f64>, String> {
    if v.is_nan() {
        Ok(None)
    } else if v.is_finite() {
        Ok(Some(v))
    } else {
        Err(format!("{name} is {v}, not a number it can write"))
    }
}

/// Both of a pair, or neither.
fn optional_pair(name: &str, v: [f64; 2]) -> Result<Option<(f64, f64)>, String> {
    match (optional(name, v[0])?, optional(name, v[1])?) {
        (Some(a), Some(b)) => Ok(Some((a, b))),
        (None, None) => Ok(None),
        _ => Err(format!(
            "{name} has one value NaN and the other not; give both or neither"
        )),
    }
}

/// A NUL-terminated UTF-8 string from C, or nothing for NULL.
///
/// # Safety
///
/// `p` is null or points to a NUL-terminated string.
unsafe fn c_str<'a>(name: &str, p: *const c_char) -> Result<Option<&'a str>, String> {
    if p.is_null() {
        return Ok(None);
    }
    CStr::from_ptr(p)
        .to_str()
        .map(Some)
        .map_err(|_| format!("{name} is not UTF-8"))
}

impl FtlWriterOptions {
    /// Back to [`WriterOptions`], refusing what C can say and Rust cannot: an
    /// enumeration value out of range, half a pair, text that is not UTF-8.
    /// What both can say and the writer refuses (a zero width, CMYK on three
    /// samples) is left to [`TiffWriter::new`], which says it the same way to
    /// both.
    ///
    /// # Safety
    ///
    /// The pointers are null or valid as the header describes them.
    unsafe fn to_options(self) -> Result<WriterOptions, String> {
        let sample_type = sample_type_of(self.sample_type)?;
        let compression = compression_of(self.compression)?;
        let format = metadata_format_of(self.metadata_format)?;
        let mut o = WriterOptions::new(self.width, self.height, sample_type)
            .samples_per_pixel(self.samples_per_pixel)
            .compression(compression)
            .planar(self.planar)
            .cmyk(self.cmyk)
            .predictor(self.predictor)
            .bigtiff(self.bigtiff);
        if self.compression_level != FTL_LEVEL_DEFAULT {
            o = o.compression_level(self.compression_level);
        }
        if self.rows_per_strip != 0 {
            o = o.rows_per_strip(self.rows_per_strip);
        }
        if let Some(d) = c_str("description", self.description)? {
            o = o.description(d);
        }
        if format == MetadataFormat::None {
            return Ok(o);
        }

        if self.channels == 0 || self.slices == 0 {
            return Err(format!(
                "channels and slices have to be at least 1 (got {} and {})",
                self.channels, self.slices
            ));
        }
        let mut m = StackMetaWrite::new(self.channels, self.slices).mode(mode_of(self.mode)?);
        if let Some(u) = c_str("unit", self.unit)? {
            m = m.unit(u);
        }
        if let Some((w, h)) = optional_pair("pixel size", [self.pixel_width, self.pixel_height])? {
            m = m.pixel_size(w, h);
        }
        if let Some(s) = optional("spacing", self.spacing)? {
            m = m.spacing(s);
        }
        if let Some((c0, c1)) = optional_pair("calibration", self.calibration)? {
            m = m.calibration(c0, c1);
        }
        if let Some((lo, hi)) = optional_pair("range", self.range)? {
            m = m.range(lo, hi);
        }
        if let Some(fps) = optional("fps", self.fps)? {
            m = m.fps(fps);
        }
        if let Some(dt) = optional("frame_interval_s", self.frame_interval_s)? {
            m = m.frame_interval_s(dt);
        }
        m = match self.loop_playback {
            -1 => m,
            0 => m.loop_playback(false),
            1 => m.loop_playback(true),
            v => return Err(format!("loop_playback {v} is not -1, 0 or 1")),
        };
        if self.named_channels > 0 {
            if self.channel_names.is_null() || self.channel_colors.is_null() {
                return Err(format!(
                    "named_channels is {} but channel_names or channel_colors is NULL",
                    self.named_channels
                ));
            }
            if self.named_channels > isize::MAX as usize / 3 {
                return Err("named_channels is larger than memory can address".to_string());
            }
            let names = std::slice::from_raw_parts(self.channel_names, self.named_channels);
            let colors = std::slice::from_raw_parts(self.channel_colors, 3 * self.named_channels);
            for (i, (&name, rgb)) in names.iter().zip(colors.chunks_exact(3)).enumerate() {
                let name = c_str("a channel name", name)?
                    .ok_or_else(|| format!("channel_names[{i}] is NULL"))?;
                m = m.channel(name, [rgb[0], rgb[1], rgb[2]]);
            }
        }
        Ok(o.metadata(m).metadata_format(format))
    }
}

/// An open stack behind a handle: [`TiffStack`].
pub struct FtlStack {
    inner: TiffStack,
}

/// A writer behind a handle: [`TiffWriter`] on a buffered file.
pub struct FtlWriter {
    inner: TiffWriter<BufWriter<File>>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Record why a call failed, for [`ftl_last_error`], and return its status.
///
/// `try_with`, not `with`: a call made while the thread is being torn down —
/// from a C++ `thread_local` destructor, say — finds the slot already gone,
/// and `with` would panic there, which past `extern "C"` is an abort. The
/// status still comes back; only the message is lost.
fn fail(status: c_int, why: impl Into<String>) -> c_int {
    let why = CString::new(why.into().replace('\0', " ")).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|e| *e.borrow_mut() = why);
    status
}

/// The status for an error from the Rust API: what its type says when it has
/// one ([`NotTiff`], an I/O error anywhere in the chain), `otherwise` when it
/// is only words.
fn status_of(e: &anyhow::Error, otherwise: c_int) -> c_int {
    if e.downcast_ref::<NotTiff>().is_some() {
        FTL_NOT_TIFF
    } else if e.chain().any(|c| c.is::<std::io::Error>()) {
        FTL_IO
    } else {
        otherwise
    }
}

/// [`fail`] with an error from the Rust API, its whole chain in the message.
fn fail_with(e: &anyhow::Error, otherwise: c_int) -> c_int {
    fail(status_of(e, otherwise), format!("{e:#}"))
}

/// Run `work`, turning a panic into [`FTL_FAILED`] rather than letting it
/// unwind into a caller that has no idea what one is.
fn guarded(work: impl FnOnce() -> c_int) -> c_int {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(status) => status,
        Err(panic) => {
            let why = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "no message".to_string());
            fail(FTL_FAILED, format!("fast-tiff-lib panicked: {why}"))
        }
    }
}

/// The stack behind `s`, or the status to return for a null one.
///
/// # Safety
///
/// `s` is null or came from [`ftl_open`] or [`ftl_open_bytes`] and has not
/// been freed.
unsafe fn stack<'a>(s: *const FtlStack) -> Result<&'a TiffStack, c_int> {
    s.as_ref()
        .map(|s| &s.inner)
        .ok_or_else(|| fail(FTL_INVALID_ARGUMENT, "the stack handle is NULL"))
}

/// The stack behind `s` and its frame `index`, or the status to return.
///
/// # Safety
///
/// As for [`stack`].
unsafe fn frame_at<'a>(
    s: *const FtlStack,
    index: usize,
) -> Result<(&'a TiffStack, &'a FrameInfo), c_int> {
    let st = stack(s)?;
    match st.frames.get(index) {
        Some(f) => Ok((st, f)),
        None => Err(fail(
            FTL_INVALID_ARGUMENT,
            format!(
                "frame {index} is past the end of {} frame(s)",
                st.frames.len()
            ),
        )),
    }
}

/// The snprintf convention: as much of `s` as fits in `cap` bytes,
/// NUL-terminated, and the whole length back.
///
/// # Safety
///
/// `buf` is null or points to `cap` writable bytes.
unsafe fn copy_str(s: &str, buf: *mut c_char, cap: usize) -> usize {
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf.cast::<u8>(), n);
        buf.add(n).write(0);
    }
    s.len()
}

/// Why the last call on this thread that did not return `FTL_OK` failed, as a
/// NUL-terminated UTF-8 string. Empty before any has.
///
/// The pointer stays valid until the next failure on the same thread; copy
/// the text to keep it. Empty, too, on a thread being torn down.
#[no_mangle]
pub extern "C" fn ftl_last_error() -> *const c_char {
    LAST_ERROR
        .try_with(|e| e.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}

/// The library's version, `"major.minor.patch"`, NUL-terminated and static.
/// The header is only good for the version it shipped with.
#[no_mangle]
pub extern "C" fn ftl_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// [`set_parallel_decode`](crate::set_parallel_decode): let large decodes
/// (and the writer's compression) split across cores. Process-wide; off by
/// default; the pixels are the same either way.
#[no_mangle]
pub extern "C" fn ftl_set_parallel_decode(enabled: bool) {
    crate::set_parallel_decode(enabled);
}

/// Open and index the TIFF at `path`, memory-mapped ([`TiffStack::open`]).
/// On `FTL_OK`, `*out` is the stack, to be freed with [`ftl_stack_free`]; on
/// anything else it is NULL. `FTL_UNAVAILABLE` in a build without the `mmap`
/// feature: read the file yourself and use [`ftl_open_bytes`].
///
/// The file must not change while it is open.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string; `out` points to writable
/// memory for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ftl_open(path: *const c_char, out: *mut *mut FtlStack) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        let path = match c_str("path", path) {
            Ok(Some(p)) => p,
            Ok(None) => return fail(FTL_INVALID_ARGUMENT, "path is NULL"),
            Err(why) => return fail(FTL_INVALID_ARGUMENT, why),
        };
        #[cfg(not(feature = "mmap"))]
        {
            let _ = path;
            fail(
                FTL_UNAVAILABLE,
                "this build has no `mmap` feature, so no ftl_open; read the file and use ftl_open_bytes",
            )
        }
        #[cfg(feature = "mmap")]
        {
            // `TiffStack::open` words a failed open without keeping the
            // `io::Error`, so open it here first for a status that says I/O.
            if let Err(e) = File::open(path) {
                return fail(FTL_IO, format!("could not open {path}: {e}"));
            }
            match TiffStack::open(path) {
                Ok(inner) => {
                    out.write(Box::into_raw(Box::new(FtlStack { inner })));
                    FTL_OK
                }
                Err(e) => fail_with(&e, FTL_FORMAT),
            }
        }
    })
}

/// Index a TIFF held in memory ([`TiffStack::from_bytes`]). The `len` bytes
/// at `data` are copied, so the caller's buffer is free again as soon as this
/// returns. On `FTL_OK`, `*out` is the stack, to be freed with
/// [`ftl_stack_free`]; on anything else it is NULL.
///
/// # Safety
///
/// `data` points to `len` readable bytes; `out` points to writable memory for
/// one pointer.
#[no_mangle]
pub unsafe extern "C" fn ftl_open_bytes(
    data: *const c_void,
    len: usize,
    out: *mut *mut FtlStack,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        if data.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "data is NULL");
        }
        if len > isize::MAX as usize {
            return fail(
                FTL_INVALID_ARGUMENT,
                "len is larger than memory can address",
            );
        }
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), len).to_vec();
        match TiffStack::from_bytes(bytes) {
            Ok(inner) => {
                out.write(Box::into_raw(Box::new(FtlStack { inner })));
                FTL_OK
            }
            Err(e) => fail_with(&e, FTL_FORMAT),
        }
    })
}

/// Free a stack, and with it its mapping. NULL is ignored.
///
/// # Safety
///
/// `s` is null or came from [`ftl_open`] or [`ftl_open_bytes`] and has not
/// been freed.
#[no_mangle]
pub unsafe extern "C" fn ftl_stack_free(s: *mut FtlStack) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// How many frames (IFDs, planes) the stack has; 0 for NULL.
///
/// # Safety
///
/// `s` is null or a live stack.
#[no_mangle]
pub unsafe extern "C" fn ftl_frame_count(s: *const FtlStack) -> usize {
    s.as_ref().map_or(0, |s| s.inner.frames.len())
}

/// Whether the file is a BigTIFF ([`TiffFlavor::Big`]); false for NULL.
///
/// # Safety
///
/// `s` is null or a live stack.
#[no_mangle]
pub unsafe extern "C" fn ftl_stack_is_bigtiff(s: *const FtlStack) -> bool {
    s.as_ref()
        .is_some_and(|s| s.inner.flavor == TiffFlavor::Big)
}

/// Whether the file is big-endian (`MM`, [`ByteOrder::Big`]); false for NULL.
/// Informational: the readers hand back host-order values either way.
///
/// # Safety
///
/// `s` is null or a live stack.
#[no_mangle]
pub unsafe extern "C" fn ftl_stack_is_big_endian(s: *const FtlStack) -> bool {
    s.as_ref()
        .is_some_and(|s| s.inner.byte_order == ByteOrder::Big)
}

/// Frame `index`'s layout.
///
/// # Safety
///
/// `s` is null or a live stack; `out` points to writable memory for one
/// [`FtlFrameInfo`].
#[no_mangle]
pub unsafe extern "C" fn ftl_frame_info(
    s: *const FtlStack,
    index: usize,
    out: *mut FtlFrameInfo,
) -> c_int {
    let (_, f) = match frame_at(s, index) {
        Ok(v) => v,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(FtlFrameInfo::from(f));
    FTL_OK
}

/// What the file's metadata says (see [`FtlMetadata`]).
///
/// # Safety
///
/// `s` is null or a live stack; `out` points to writable memory for one
/// [`FtlMetadata`].
#[no_mangle]
pub unsafe extern "C" fn ftl_metadata(s: *const FtlStack, out: *mut FtlMetadata) -> c_int {
    let st = match stack(s) {
        Ok(st) => st,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(FtlMetadata::from(&st.meta));
    FTL_OK
}

/// The calibration unit ([`StackMeta::unit`], such as `"µm"`), UTF-8, by the
/// snprintf convention: as much as fits in `cap` bytes, NUL-terminated, and
/// the whole length back, not counting the NUL. 0 when the file names none,
/// and for NULL.
///
/// # Safety
///
/// `s` is null or a live stack; `buf` is null or points to `cap` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_metadata_unit(
    s: *const FtlStack,
    buf: *mut c_char,
    cap: usize,
) -> usize {
    let unit = s.as_ref().and_then(|s| s.inner.meta.unit.as_deref());
    copy_str(unit.unwrap_or(""), buf, cap)
}

/// The first IFD's `ImageDescription` ([`TiffStack::description`]), verbatim,
/// by the snprintf convention (see [`ftl_metadata_unit`]). 0 when the file
/// has none, and for NULL.
///
/// # Safety
///
/// `s` is null or a live stack; `buf` is null or points to `cap` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_description(
    s: *const FtlStack,
    buf: *mut c_char,
    cap: usize,
) -> usize {
    let text = s.as_ref().and_then(|s| s.inner.description.as_deref());
    copy_str(text.unwrap_or(""), buf, cap)
}

/// [`resolve_dimensions`]: how a viewer reads `channels` x `slices` x
/// `frames` — classified by size, not by the file's labels, and with Z folded
/// into time unless both are real. The product never changes.
///
/// # Safety
///
/// `out` points to writable memory for one [`FtlDims`].
#[no_mangle]
pub unsafe extern "C" fn ftl_resolve_dimensions(
    channels: usize,
    slices: usize,
    frames: usize,
    out: *mut FtlDims,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "out is NULL");
        }
        // The promise is that the product never changes, which needs one:
        // a count of zero, or a product past `size_t`, has nothing to keep.
        if channels == 0 || slices == 0 || frames == 0 {
            return fail(
                FTL_INVALID_ARGUMENT,
                format!("{channels} x {slices} x {frames}: every count has to be at least 1"),
            );
        }
        if channels.checked_mul(slices).and_then(|v| v.checked_mul(frames)).is_none() {
            return fail(
                FTL_INVALID_ARGUMENT,
                format!("{channels} x {slices} x {frames} is more frames than size_t counts"),
            );
        }
        out.write(resolve_dimensions(channels, slices, frames).into());
        FTL_OK
    })
}

/// The frame (IFD) that holds channel `c`, slice `z` and time point `t`:
/// `(t * slices + z) * channels + c`, ImageJ's `xyczt` order. `dims` may be
/// NULL for the file's own labels ([`FtlMetadata`]), or a resolution of them
/// from [`ftl_resolve_dimensions`]. A position outside `dims`, or a frame the
/// file does not have, is refused.
///
/// # Safety
///
/// `s` is null or a live stack; `dims` is null or points to one [`FtlDims`];
/// `index` points to one writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn ftl_plane_index(
    s: *const FtlStack,
    dims: *const FtlDims,
    c: usize,
    z: usize,
    t: usize,
    index: *mut usize,
) -> c_int {
    let st = match stack(s) {
        Ok(st) => st,
        Err(status) => return status,
    };
    if index.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "index is NULL");
    }
    let (channels, slices, frames) = match dims.as_ref() {
        Some(d) => (d.channels, d.slices, d.frames),
        None => (st.meta.channels, st.meta.slices, st.meta.frames),
    };
    if c >= channels || z >= slices || t >= frames {
        return fail(
            FTL_INVALID_ARGUMENT,
            format!("(c {c}, z {z}, t {t}) is outside {channels} x {slices} x {frames}"),
        );
    }
    let i = t
        .checked_mul(slices)
        .and_then(|v| v.checked_add(z))
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| v.checked_add(c));
    match i {
        Some(i) if i < st.frames.len() => {
            index.write(i);
            FTL_OK
        }
        _ => fail(
            FTL_INVALID_ARGUMENT,
            format!(
                "(c {c}, z {z}, t {t}) is past the end of the file's {} frame(s)",
                st.frames.len()
            ),
        ),
    }
}

/// How many values [`ftl_read_u8`], [`ftl_read_u16`] and [`ftl_read_f32`]
/// write for frame `index`: `width * height * samples_per_pixel`
/// ([`FrameInfo::sample_count`]). [`ftl_read_rgb_u8`] and
/// [`ftl_read_rgb_u16`] write `3 * width * height` instead.
///
/// # Safety
///
/// `s` is null or a live stack; `count` points to one writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn ftl_sample_count(
    s: *const FtlStack,
    index: usize,
    count: *mut usize,
) -> c_int {
    let (_, f) = match frame_at(s, index) {
        Ok(v) => v,
        Err(status) => return status,
    };
    if count.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "count is NULL");
    }
    match f.sample_count() {
        Ok(n) => {
            count.write(n);
            FTL_OK
        }
        Err(e) => fail_with(&e, FTL_FORMAT),
    }
}

/// A decoder as the readers below drive it: the frame's planes, in sample
/// order, borrowed from the mapping where the Rust API can.
type Decode<T> = for<'a> fn(
    &'a TiffStack,
    &'a FrameInfo,
    Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [T]>>>;

/// Values a reader writes: one plane per sample, or three for the RGB ones.
type Count = fn(&FrameInfo) -> anyhow::Result<usize>;

fn all_samples(f: &FrameInfo) -> anyhow::Result<usize> {
    f.sample_count()
}

fn three_planes(f: &FrameInfo) -> anyhow::Result<usize> {
    f.pixel_count()?
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("frame geometry overflows address space"))
}

fn owned<T: Clone>(planes: Vec<Vec<T>>) -> Vec<Cow<'static, [T]>> {
    planes.into_iter().map(Cow::Owned).collect()
}

fn decode_u8<'a>(
    st: &'a TiffStack,
    f: &'a FrameInfo,
    _: Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [u8]>>> {
    if !matches!(f.bits_per_sample, 4 | 8) {
        anyhow::bail!(
            "the frame is {}-bit; ftl_read_u8 takes 8-bit frames (and 4-bit palette indices), \
             ftl_read_u16 or ftl_read_f32 the rest",
            f.bits_per_sample
        );
    }
    if f.samples_per_pixel <= 1 {
        Ok(vec![read_frame_u8(&st.data, f, st.byte_order)?])
    } else {
        Ok(owned(read_planes_u8(&st.data, f, st.byte_order)?))
    }
}

fn decode_u16<'a>(
    st: &'a TiffStack,
    f: &'a FrameInfo,
    range: Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [u16]>>> {
    if f.samples_per_pixel <= 1 {
        Ok(vec![read_frame_u16(&st.data, f, st.byte_order, range)?])
    } else {
        Ok(owned(read_planes_u16(&st.data, f, st.byte_order, range)?))
    }
}

fn decode_f32<'a>(
    st: &'a TiffStack,
    f: &'a FrameInfo,
    _: Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [f32]>>> {
    if f.samples_per_pixel <= 1 {
        Ok(vec![read_frame_f32(&st.data, f, st.byte_order)?])
    } else {
        Ok(owned(read_planes_f32(&st.data, f, st.byte_order)?))
    }
}

fn decode_rgb_u8<'a>(
    st: &'a TiffStack,
    f: &'a FrameInfo,
    _: Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [u8]>>> {
    Ok(owned(read_planes_rgb_u8(&st.data, f, st.byte_order)?))
}

fn decode_rgb_u16<'a>(
    st: &'a TiffStack,
    f: &'a FrameInfo,
    _: Option<(f32, f32)>,
) -> anyhow::Result<Vec<Cow<'a, [u16]>>> {
    Ok(owned(read_planes_rgb_u16(&st.data, f, st.byte_order)?))
}

/// Decode frame `index` with `decode` and copy its planes, one after another,
/// into the `len` values at `out` — after checking `len` against `count`.
///
/// # Safety
///
/// `s` is null or a live stack; `out` is null or points to `len` writable
/// values.
unsafe fn read_into<T: Copy>(
    s: *const FtlStack,
    index: usize,
    range: Option<(f32, f32)>,
    out: *mut T,
    len: usize,
    count: Count,
    decode: Decode<T>,
) -> c_int {
    guarded(|| {
        let (st, f) = match frame_at(s, index) {
            Ok(v) => v,
            Err(status) => return status,
        };
        if out.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "out is NULL");
        }
        // Reading ZSTD without the codec is the build, not the file: the same
        // status writing it gets.
        if f.compression == Compression::Zstd && !cfg!(feature = "codec-zstd") {
            return fail(
                FTL_UNAVAILABLE,
                format!("frame {index} is ZSTD-compressed, and this build has no `codec-zstd`"),
            );
        }
        let needed = match count(f) {
            Ok(n) => n,
            Err(e) => return fail_with(&e, FTL_FORMAT),
        };
        if len < needed {
            return fail(
                FTL_INVALID_ARGUMENT,
                format!("out holds {len} values; frame {index} decodes to {needed}"),
            );
        }
        let planes = match decode(st, f, range) {
            Ok(p) => p,
            Err(e) => {
                let status = status_of(&e, FTL_FORMAT);
                return fail(status, format!("frame {index}: {e:#}"));
            }
        };
        let total: usize = planes.iter().map(|p| p.len()).sum();
        if total != needed {
            return fail(
                FTL_FAILED,
                format!("frame {index} decoded to {total} values, not the {needed} expected"),
            );
        }
        let mut at = 0;
        for p in &planes {
            std::ptr::copy_nonoverlapping(p.as_ptr(), out.add(at), p.len());
            at += p.len();
        }
        FTL_OK
    })
}

/// Frame `index` as raw 8-bit samples, every sample plane one after another
/// ([`read_frame_u8`], [`read_planes_u8`]): 8-bit frames, and 4-bit palette
/// indices as one byte each; anything wider is `FTL_FORMAT`. `out` holds
/// `len` bytes, at least [`ftl_sample_count`].
///
/// # Safety
///
/// `s` is null or a live stack; `out` is null or points to `len` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_read_u8(
    s: *const FtlStack,
    index: usize,
    out: *mut u8,
    len: usize,
) -> c_int {
    read_into(s, index, None, out, len, all_samples, decode_u8)
}

/// Frame `index` as 16-bit display samples, every sample plane one after
/// another ([`read_frame_u16`], [`read_planes_u16`]): 16-bit as it is, 8-bit
/// widened onto 0..=65535, signed offset into unsigned, 32- and 64-bit
/// rescaled from `range[0]..range[1]` — or from each plane's own minimum and
/// maximum when `range` is NULL. `out` holds `len` values, at least
/// [`ftl_sample_count`].
///
/// # Safety
///
/// `s` is null or a live stack; `range` is null or points to two floats;
/// `out` is null or points to `len` writable values.
#[no_mangle]
pub unsafe extern "C" fn ftl_read_u16(
    s: *const FtlStack,
    index: usize,
    range: *const f32,
    out: *mut u16,
    len: usize,
) -> c_int {
    let range = if range.is_null() {
        None
    } else {
        let (lo, hi) = (range.read(), range.add(1).read());
        if !lo.is_finite() || !hi.is_finite() {
            return fail(
                FTL_INVALID_ARGUMENT,
                format!("range {lo}..{hi} is not finite"),
            );
        }
        Some((lo, hi))
    };
    read_into(s, index, range, out, len, all_samples, decode_u16)
}

/// Frame `index` as raw floats, every sample plane one after another
/// ([`read_frame_f32`], [`read_planes_f32`]): 32- and 64-bit frames, integer
/// or float, cast to `float`; 8- and 16-bit are `FTL_FORMAT`. `out` holds
/// `len` values, at least [`ftl_sample_count`].
///
/// # Safety
///
/// `s` is null or a live stack; `out` is null or points to `len` writable
/// floats.
#[no_mangle]
pub unsafe extern "C" fn ftl_read_f32(
    s: *const FtlStack,
    index: usize,
    out: *mut f32,
    len: usize,
) -> c_int {
    read_into(s, index, None, out, len, all_samples, decode_f32)
}

/// An 8-bit CMYK frame (`FtlFrameInfo.cmyk`) converted to red, green and
/// blue planes, one after another ([`read_planes_rgb_u8`]); anything else is
/// `FTL_FORMAT`. An RGB frame needs no conversion: [`ftl_read_u8`] gives its
/// planes. `out` holds `len` bytes, at least `3 * width * height`.
///
/// # Safety
///
/// `s` is null or a live stack; `out` is null or points to `len` writable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_read_rgb_u8(
    s: *const FtlStack,
    index: usize,
    out: *mut u8,
    len: usize,
) -> c_int {
    read_into(s, index, None, out, len, three_planes, decode_rgb_u8)
}

/// [`ftl_read_rgb_u8`] as 16-bit samples ([`read_planes_rgb_u16`]), for 8-
/// and 16-bit CMYK frames alike.
///
/// # Safety
///
/// `s` is null or a live stack; `out` is null or points to `len` writable
/// values.
#[no_mangle]
pub unsafe extern "C" fn ftl_read_rgb_u16(
    s: *const FtlStack,
    index: usize,
    out: *mut u16,
    len: usize,
) -> c_int {
    read_into(s, index, None, out, len, three_planes, decode_rgb_u16)
}

/// The smallest and largest finite value across a whole 32- or 64-bit frame,
/// every plane of it ([`frame_float_minmax`]); `0` and `1` when the frame is
/// constant or has no finite value. Not what [`ftl_read_u16`] uses when given
/// no range — that scales each plane by its own — but what to give it to put
/// every plane, or several frames, on one scale. NaN and NaN for an 8- or
/// 16-bit frame, whose own integer range is the one to use. Either output may
/// be NULL.
///
/// # Safety
///
/// `s` is null or a live stack; `min` and `max` are each null or writable.
#[no_mangle]
pub unsafe extern "C" fn ftl_float_minmax(
    s: *const FtlStack,
    index: usize,
    min: *mut f32,
    max: *mut f32,
) -> c_int {
    guarded(|| {
        let (st, f) = match frame_at(s, index) {
            Ok(v) => v,
            Err(status) => return status,
        };
        match frame_float_minmax(&st.data, f, st.byte_order) {
            Ok(r) => {
                let (lo, hi) = r.unwrap_or((f32::NAN, f32::NAN));
                if !min.is_null() {
                    min.write(lo);
                }
                if !max.is_null() {
                    max.write(hi);
                }
                FTL_OK
            }
            Err(e) => fail(status_of(&e, FTL_FORMAT), format!("frame {index}: {e:#}")),
        }
    })
}

/// Fill `out` with the writer's defaults: 16-bit grayscale, uncompressed, no
/// metadata, and every optional field unset. Width and height are 0 and have
/// to be set.
///
/// # Safety
///
/// `out` is null or points to writable memory for one [`FtlWriterOptions`].
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_options_default(out: *mut FtlWriterOptions) -> c_int {
    if out.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(FtlWriterOptions {
        width: 0,
        height: 0,
        sample_type: FTL_TYPE_U16,
        samples_per_pixel: 1,
        compression: FTL_COMPRESSION_NONE,
        compression_level: FTL_LEVEL_DEFAULT,
        rows_per_strip: 0,
        planar: false,
        cmyk: false,
        predictor: false,
        bigtiff: false,
        description: std::ptr::null(),
        metadata_format: FTL_METADATA_NONE,
        mode: FTL_MODE_GRAYSCALE,
        channels: 1,
        slices: 1,
        unit: std::ptr::null(),
        pixel_width: f64::NAN,
        pixel_height: f64::NAN,
        spacing: f64::NAN,
        calibration: [f64::NAN; 2],
        range: [f64::NAN; 2],
        fps: f64::NAN,
        frame_interval_s: f64::NAN,
        loop_playback: -1,
        channel_names: std::ptr::null(),
        channel_colors: std::ptr::null(),
        named_channels: 0,
    });
    FTL_OK
}

/// Create (or truncate) the file at `path` and a writer for it
/// ([`TiffWriter::create`]). The options are checked before the file is
/// touched, so options it refuses leave no file behind. On `FTL_OK`, `*out`
/// is the writer, to be finished with [`ftl_writer_finish`] or abandoned
/// with [`ftl_writer_free`]; on anything else it is NULL. The options'
/// strings are copied.
///
/// # Safety
///
/// `path` is null or a NUL-terminated string; `options` is null or points to
/// one [`FtlWriterOptions`] whose pointers are null or valid as documented;
/// `out` points to writable memory for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_create(
    path: *const c_char,
    options: *const FtlWriterOptions,
    out: *mut *mut FtlWriter,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        let path = match c_str("path", path) {
            Ok(Some(p)) => p,
            Ok(None) => return fail(FTL_INVALID_ARGUMENT, "path is NULL"),
            Err(why) => return fail(FTL_INVALID_ARGUMENT, why),
        };
        let Some(options) = options.as_ref() else {
            return fail(FTL_INVALID_ARGUMENT, "options is NULL");
        };
        let opts = match options.to_options() {
            Ok(o) => o,
            Err(why) => return fail(FTL_INVALID_ARGUMENT, why),
        };
        if cfg!(not(feature = "codec-zstd")) && options.compression == FTL_COMPRESSION_ZSTD {
            return fail(
                FTL_UNAVAILABLE,
                "this build has no `codec-zstd` feature, so it cannot write ZSTD",
            );
        }
        if cfg!(all(target_arch = "wasm32", target_os = "unknown")) {
            return fail(FTL_UNAVAILABLE, "this target has no filesystem to write to");
        }
        // Validated against memory first, so a refusal leaves no file behind.
        if let Err(e) = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts.clone()) {
            return fail_with(&e, FTL_INVALID_ARGUMENT);
        }
        let file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return fail(FTL_IO, format!("could not create {path}: {e}")),
        };
        match TiffWriter::new(BufWriter::new(file), opts) {
            Ok(inner) => {
                out.write(Box::into_raw(Box::new(FtlWriter { inner })));
                FTL_OK
            }
            Err(e) => fail_with(&e, FTL_INVALID_ARGUMENT),
        }
    })
}

/// Hand one frame of `count` values at `samples` to `write`, after the
/// checks every typed writer shares.
///
/// # Safety
///
/// `w` is null or a live writer; `samples` is null or points to `count`
/// readable values.
unsafe fn write_with<T>(
    w: *mut FtlWriter,
    samples: *const T,
    count: usize,
    write: impl FnOnce(&mut TiffWriter<BufWriter<File>>, &[T]) -> anyhow::Result<()>,
) -> c_int {
    guarded(|| {
        let Some(w) = w.as_mut() else {
            return fail(FTL_INVALID_ARGUMENT, "the writer handle is NULL");
        };
        if samples.is_null() {
            return fail(FTL_INVALID_ARGUMENT, "samples is NULL");
        }
        if count > isize::MAX as usize / std::mem::size_of::<T>().max(1) {
            return fail(
                FTL_INVALID_ARGUMENT,
                "count is larger than memory can address",
            );
        }
        match write(&mut w.inner, std::slice::from_raw_parts(samples, count)) {
            Ok(()) => FTL_OK,
            Err(e) => fail_with(&e, FTL_INVALID_ARGUMENT),
        }
    })
}

/// Append one frame of `count` 8-bit samples ([`TiffWriter::write_frame_u8`];
/// `FTL_TYPE_U8` only). `count` is `width * height * samples_per_pixel`,
/// interleaved, or plane after plane with `planar`.
///
/// # Safety
///
/// `w` is null or a live writer; `samples` points to `count` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_write_u8(
    w: *mut FtlWriter,
    samples: *const u8,
    count: usize,
) -> c_int {
    write_with(w, samples, count, |w, s| w.write_frame_u8(s))
}

/// [`ftl_writer_write_u8`] for `FTL_TYPE_U16` ([`TiffWriter::write_frame_u16`]).
///
/// # Safety
///
/// `w` is null or a live writer; `samples` points to `count` readable values.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_write_u16(
    w: *mut FtlWriter,
    samples: *const u16,
    count: usize,
) -> c_int {
    write_with(w, samples, count, |w, s| w.write_frame_u16(s))
}

/// [`ftl_writer_write_u8`] for `FTL_TYPE_F32` ([`TiffWriter::write_frame_f32`]).
///
/// # Safety
///
/// `w` is null or a live writer; `samples` points to `count` readable floats.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_write_f32(
    w: *mut FtlWriter,
    samples: *const f32,
    count: usize,
) -> c_int {
    write_with(w, samples, count, |w, s| w.write_frame_f32(s))
}

/// [`ftl_writer_write_u8`] for `FTL_TYPE_F64` ([`TiffWriter::write_frame_f64`]).
///
/// # Safety
///
/// `w` is null or a live writer; `samples` points to `count` readable
/// doubles.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_write_f64(
    w: *mut FtlWriter,
    samples: *const f64,
    count: usize,
) -> c_int {
    write_with(w, samples, count, |w, s| w.write_frame_f64(s))
}

/// Append one frame of any sample type as `len` raw little-endian bytes
/// ([`TiffWriter::write_frame_bytes`]): how the types without a function of
/// their own (`I8`, `I16`, `U32`, `I32`, `U64`, `I64`) are written. `len` is
/// `width * height * samples_per_pixel * bytes_per_sample`.
///
/// # Safety
///
/// `w` is null or a live writer; `data` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_write_bytes(
    w: *mut FtlWriter,
    data: *const c_void,
    len: usize,
) -> c_int {
    write_with(w, data.cast::<u8>(), len, |w, s| w.write_frame_bytes(s))
}

/// How many frames the writer has taken so far; 0 for NULL.
///
/// # Safety
///
/// `w` is null or a live writer.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_frames_written(w: *const FtlWriter) -> usize {
    w.as_ref().map_or(0, |w| w.inner.frames_written())
}

/// Write the directory, the metadata and the header, and close the file
/// ([`TiffWriter::finish`]); the file is not a TIFF until this succeeds.
/// Frees the writer whatever it returns: on a failure the file left behind is
/// not a valid TIFF. A frame count that does not divide into `channels *
/// slices` is refused here, as it is in Rust.
///
/// # Safety
///
/// `w` is null or a live writer, which is not used again.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_finish(w: *mut FtlWriter) -> c_int {
    if w.is_null() {
        return fail(FTL_INVALID_ARGUMENT, "the writer handle is NULL");
    }
    let w = Box::from_raw(w);
    guarded(move || match w.inner.finish() {
        Ok(file) => match file.into_inner() {
            Ok(_) => FTL_OK,
            Err(e) => fail(FTL_IO, format!("could not write the file: {}", e.error())),
        },
        Err(e) => fail_with(&e, FTL_INVALID_ARGUMENT),
    })
}

/// Abandon a writer without finishing: the file stays as far as it got, which
/// is not a valid TIFF — delete it. NULL is ignored.
///
/// # Safety
///
/// `w` is null or a live writer, which is not used again.
#[no_mangle]
pub unsafe extern "C" fn ftl_writer_free(w: *mut FtlWriter) {
    if !w.is_null() {
        drop(Box::from_raw(w));
    }
}

#[cfg(test)]
#[path = "capi_tests.rs"]
mod tests;

//! Turns pixel data back into a multi-frame TIFF file — the write-side
//! counterpart of `decode`. The API shape follows TinyTIFF's proven stack-writer
//! model (open with a fixed frame layout, append frames one at a time, finish),
//! the format coverage follows libtiff (None/LZW/PackBits/Deflate compression,
//! horizontal predictor, unsigned/signed/float samples at 8/16/32 bits, chunky
//! or planar RGB, multi-strip), and the Rust idioms follow the `tiff` crate (a writer
//! generic over `Write + Seek`, so files and in-memory `Cursor`s both work).
//!
//! Layout choices are made for *this* library's reader:
//! - Output is always **little-endian**, and an uncompressed frame defaults to
//!   a **single strip** — exactly the zero-copy fast path of `read_frame_u16` /
//!   `read_frame_u8` / `read_frame_f32` (and ImageJ's own default layout), so
//!   files written here scrub back with no decode work at all.
//! - Pixel data is written append-only as frames arrive (nothing buffered but
//!   the current frame's strips); the IFD chain is assembled at `finish()`,
//!   which needs one seek to patch the header's first-IFD offset. This is also
//!   what makes the ImageJ `images=N` line trivially correct — the plane count
//!   is known by then.
//! - Compressed frames default to ~256 KiB strips, which is what lets the
//!   reader's parallel strip decompression engage; strips of one frame are
//!   themselves compressed in parallel here when the frame is large.
//!
//! Classic TIFF only: offsets are 32-bit, so the file must stay under 4 GiB
//! (checked, with a clear error). Tiles are out of scope — matching the reader.
//! Both sample interleavings are written: chunky by default (the layout the
//! reader's zero-copy and fused paths are tuned for), or planar via
//! `WriterOptions::planar`, which emits `PlanarConfiguration=2` and splits
//! strips per sample plane. An absent tag is chunky per TIFF6, so the default
//! output carries no PlanarConfiguration entry at all.

use crate::ifd::TiffFlavor;
use crate::ij_metadata::DisplayMode;
use crate::index::{
    Compression, TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_DESCRIPTION, TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_PLANAR_CONFIG, TAG_PREDICTOR, TAG_ROWS_PER_STRIP,
    TAG_SAMPLES_PER_PIXEL, TAG_SAMPLE_FORMAT, TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS,
};
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// ExtraSamples (tag 338): declares samples beyond the photometric base, in
/// either interleaving (write-side only — the reader splits planes regardless).
const TAG_EXTRA_SAMPLES: u16 = 338;

/// Classic TIFF stores every offset as a u32, so nothing may live at or past
/// 4 GiB. Files that would exceed this are automatically written as BigTIFF
/// instead (see `finish`).
const MAX_CLASSIC_TIFF: u64 = u32::MAX as u64;

/// The pixel sample layout of every frame in the stack, mapping onto TIFF's
/// (BitsPerSample, SampleFormat) tag pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl SampleType {
    /// TIFF `BitsPerSample` for this type.
    pub fn bits(self) -> u16 {
        match self {
            SampleType::U8 | SampleType::I8 => 8,
            SampleType::U16 | SampleType::I16 => 16,
            SampleType::U32 | SampleType::I32 | SampleType::F32 => 32,
            SampleType::U64 | SampleType::I64 | SampleType::F64 => 64,
        }
    }

    /// Bytes per sample (`bits / 8`).
    pub fn bytes(self) -> usize {
        self.bits() as usize / 8
    }

    /// TIFF `SampleFormat` code: 1 = unsigned, 2 = signed, 3 = IEEE float.
    fn format_code(self) -> u16 {
        match self {
            SampleType::U8 | SampleType::U16 | SampleType::U32 | SampleType::U64 => 1,
            SampleType::I8 | SampleType::I16 | SampleType::I32 | SampleType::I64 => 2,
            SampleType::F32 | SampleType::F64 => 3,
        }
    }
}

/// ImageJ hyperstack metadata to embed in the file's `ImageDescription`
/// (tag 270), so ImageJ — and this crate's own reader — sees the plane chain as
/// channels × Z-slices × time frames instead of a flat image sequence. Planes
/// are expected in ImageJ's default `xyczt` order (channel fastest, then Z,
/// then time); the time-frame count is derived at `finish()` from the number of
/// planes actually written (`planes / (channels * slices)`, which must divide
/// evenly).
#[derive(Clone, Debug)]
pub struct ImageJOptions {
    channels: usize,
    slices: usize,
    mode: DisplayMode,
    fps: Option<f64>,
    frame_interval_s: Option<f64>,
    unit: Option<String>,
    /// Display window written as `min=`/`max=` (applies to all channels).
    range: Option<(f64, f64)>,
    /// Linear calibration `(c0, c1)` written as `cf=0`/`c0=`/`c1=`:
    /// real value = `c0 + c1 * raw`.
    calibration: Option<(f64, f64)>,
    /// Z-step between slices, written as `spacing=`.
    spacing: Option<f64>,
    /// Whether playback should loop, written as `loop=`.
    loop_playback: Option<bool>,
    /// Extra verbatim `key=value` lines appended to the description, for any
    /// documented ImageJ key without a dedicated setter (`vunit`, `tunit`, ...).
    extra: Vec<(String, String)>,
}

impl ImageJOptions {
    /// `channels` and `slices` describe the plane order; both are clamped to a
    /// minimum of 1. Display mode defaults to grayscale (ImageJ's own default
    /// when `mode=` is absent).
    pub fn new(channels: usize, slices: usize) -> Self {
        Self {
            channels: channels.max(1),
            slices: slices.max(1),
            mode: DisplayMode::Grayscale,
            fps: None,
            frame_interval_s: None,
            unit: None,
            range: None,
            calibration: None,
            spacing: None,
            loop_playback: None,
            extra: Vec::new(),
        }
    }

    pub fn mode(mut self, mode: DisplayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Playback rate, written as `fps=`.
    pub fn fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Seconds between time frames, written as `finterval=`.
    pub fn frame_interval_s(mut self, seconds: f64) -> Self {
        self.frame_interval_s = Some(seconds);
        self
    }

    /// Spatial unit label (e.g. `"um"`), written as `unit=`.
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Display window, written as `min=`/`max=`.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Linear pixel calibration: real value = `c0 + c1 * raw`.
    pub fn calibration(mut self, c0: f64, c1: f64) -> Self {
        self.calibration = Some((c0, c1));
        self
    }

    /// Z-step between slices (in `unit`s), written as `spacing=`.
    pub fn spacing(mut self, spacing: f64) -> Self {
        self.spacing = Some(spacing);
        self
    }

    /// Whether playback should loop, written as `loop=`.
    pub fn loop_playback(mut self, looped: bool) -> Self {
        self.loop_playback = Some(looped);
        self
    }

    /// Append a verbatim `key=value` line — the escape hatch for any ImageJ
    /// description key without a dedicated setter. Appended after the built-in
    /// keys, in insertion order.
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.push((key.into(), value.into()));
        self
    }
}

/// Everything fixed about the stack before the first frame is written. Every
/// frame must share this geometry — the same invariant the reader enforces on
/// open. Built with [`WriterOptions::new`] plus chained setters:
///
/// ```
/// use fast_tiff_lib::{Compression, SampleType, WriterOptions};
/// let opts = WriterOptions::new(512, 512, SampleType::U16)
///     .compression(Compression::Deflate)
///     .predictor(true);
/// ```
#[derive(Clone, Debug)]
pub struct WriterOptions {
    width: u32,
    height: u32,
    sample_type: SampleType,
    samples_per_pixel: u16,
    planar: bool,
    compression: Compression,
    compression_level: Option<i32>,
    predictor: bool,
    rows_per_strip: Option<u32>,
    force_bigtiff: bool,
    ij: Option<ImageJOptions>,
    description: Option<String>,
}

impl WriterOptions {
    pub fn new(width: u32, height: u32, sample_type: SampleType) -> Self {
        Self {
            width,
            height,
            sample_type,
            samples_per_pixel: 1,
            planar: false,
            compression: Compression::None,
            compression_level: None,
            predictor: false,
            rows_per_strip: None,
            force_bigtiff: false,
            ij: None,
            description: None,
        }
    }

    /// Samples per pixel: 1 (default) writes single-plane grayscale frames;
    /// 3 writes RGB. Data passed to `write_frame_*` is
    /// `width * height * samples_per_pixel` samples, in the interleaving
    /// [`planar`](Self::planar) selects (interleaved per pixel by default).
    pub fn samples_per_pixel(mut self, spp: u16) -> Self {
        self.samples_per_pixel = spp;
        self
    }

    /// Write multi-sample frames as **planar** (`PlanarConfiguration=2`): each
    /// sample stored as its own whole plane, one after another, instead of the
    /// default chunky interleaving (`PlanarConfiguration=1`).
    ///
    /// **This changes the layout `write_frame_*` expects.** The buffer is
    /// always laid out the way the file is, so no reordering pass sits between
    /// caller and disk. For a 3-sample frame:
    ///
    /// - chunky (default): `R0 G0 B0  R1 G1 B1  R2 G2 B2 …`
    /// - planar: `R0 R1 R2 …  G0 G1 G2 …  B0 B1 B2 …`
    ///
    /// The total length is identical either way, so passing chunky data with
    /// this set (or vice versa) is **not** caught by the length check — it
    /// silently scrambles the image. This is the layout a `(C, H, W)` numpy
    /// array already has, and what `tifffile` writes for one.
    ///
    /// Ignored when `samples_per_pixel` is 1: the two layouts are then
    /// byte-identical, and TIFF6 calls PlanarConfiguration irrelevant in that
    /// case, so the tag is simply not written.
    ///
    /// Chunky remains the default and is what the reader's zero-copy path is
    /// tuned for; reach for this when a consuming tool specifically needs
    /// separate planes.
    pub fn planar(mut self, on: bool) -> Self {
        self.planar = on;
        self
    }

    /// Strip compression: `None` (default), `Lzw`, `PackBits`, `Deflate`, or
    /// `Zstd` (tag 50000, the libtiff/GDAL extension).
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Compression effort level, for the codecs that have one: Deflate takes
    /// 0..=9 (clamped) and Zstd 1..=22 (negative = faster-than-1 modes). Ignored
    /// for `Lzw`/`PackBits`, which have no level. Unset applies the lib defaults:
    /// `DEFAULT_DEFLATE_LEVEL` (6) and `DEFAULT_ZSTD_LEVEL` (3).
    pub fn compression_level(mut self, level: i32) -> Self {
        self.compression_level = Some(level);
        self
    }

    /// Predictor pass before compression — usually shrinks LZW/Deflate output
    /// on continuous-tone data. Integer samples get TIFF Predictor 2
    /// (horizontal differencing, 8/16/32-bit); `F32` gets Predictor 3 (the
    /// TechNote 3 floating-point predictor, as libtiff writes for float data).
    pub fn predictor(mut self, on: bool) -> Self {
        self.predictor = on;
        self
    }

    /// Rows per strip. Default: the whole frame as one strip when uncompressed
    /// (the reader's zero-copy layout), or ~256 KiB strips when compressed
    /// (parallel-decompression-friendly).
    pub fn rows_per_strip(mut self, rows: u32) -> Self {
        self.rows_per_strip = Some(rows);
        self
    }

    /// Force BigTIFF output (magic 43, 64-bit offsets). Rarely needed: the
    /// writer upgrades to BigTIFF **automatically** when the file grows past
    /// classic TIFF's 4 GiB offset limit; this knob forces it for smaller
    /// files (e.g. interop testing).
    pub fn bigtiff(mut self, force: bool) -> Self {
        self.force_bigtiff = force;
        self
    }

    /// Embed ImageJ hyperstack metadata (mutually exclusive with
    /// [`description`](Self::description)).
    pub fn imagej(mut self, ij: ImageJOptions) -> Self {
        self.ij = Some(ij);
        self
    }

    /// Embed a verbatim `ImageDescription` text instead of ImageJ metadata
    /// (mutually exclusive with [`imagej`](Self::imagej)).
    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Some(text.into());
        self
    }
}

/// Streaming multi-frame TIFF writer: construct with a fixed frame layout,
/// append frames with `write_frame_*`, then call [`finish`](Self::finish)
/// (which writes the IFD chain and patches the header — without it the file
/// has no directory and is unreadable).
///
/// ```no_run
/// use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
/// # fn main() -> anyhow::Result<()> {
/// let opts = WriterOptions::new(256, 256, SampleType::U16);
/// let mut writer = TiffWriter::create("stack.tif", opts)?;
/// let frame = vec![0u16; 256 * 256];
/// writer.write_frame_u16(&frame)?;
/// writer.finish()?;
/// # Ok(())
/// # }
/// ```
pub struct TiffWriter<W: Write + Seek> {
    w: W,
    /// Current absolute file position (tracked, so the data path never seeks).
    pos: u64,
    /// Strip locations of every frame written so far, for `finish()`'s IFDs.
    frames: Vec<FrameStrips>,
    // Validated, resolved copies of the options:
    width: u32,
    height: u32,
    sample_type: SampleType,
    spp: usize,
    /// True when samples are written as separate whole planes
    /// (`PlanarConfiguration=2`). Already normalized: never true for `spp == 1`,
    /// where the two layouts are byte-identical.
    planar: bool,
    compression: Compression,
    compression_level: Option<i32>,
    force_bigtiff: bool,
    /// TIFF Predictor tag value: 1 = off, 2 = integer horizontal differencing,
    /// 3 = floating-point (TechNote 3). Resolved from the sample type at
    /// construction.
    predictor_tag: u16,
    rows_per_strip: u32,
    ij: Option<ImageJOptions>,
    description: Option<String>,
    /// Bytes in one strip-splittable row: `width * spp * sample_bytes` chunky,
    /// `width * sample_bytes` planar (a planar row holds one sample plane's
    /// pixels only).
    row_bytes: usize,
    /// Bytes in one strip-splittable *image*: the whole frame when chunky, one
    /// sample plane when planar. Strips never span a plane boundary, so this is
    /// what the strip split resets on.
    plane_bytes: usize,
    frame_bytes: usize,
}

struct FrameStrips {
    offsets: Vec<u64>,
    byte_counts: Vec<u64>,
}

impl TiffWriter<BufWriter<File>> {
    /// Create `path` (truncating any existing file) and write a stack to it
    /// through a buffered writer.
    pub fn create(path: impl AsRef<Path>, options: WriterOptions) -> Result<Self> {
        let file = File::create(path.as_ref())
            .map_err(|e| anyhow!("could not create {}: {e}", path.as_ref().display()))?;
        Self::new(BufWriter::new(file), options)
    }
}

impl<W: Write + Seek> TiffWriter<W> {
    /// Wrap any `Write + Seek` sink (a file, an in-memory
    /// `std::io::Cursor<Vec<u8>>`, ...). Validates the options and writes the
    /// 8-byte TIFF header immediately.
    pub fn new(mut w: W, options: WriterOptions) -> Result<Self> {
        let WriterOptions {
            width,
            height,
            sample_type,
            samples_per_pixel,
            planar,
            compression,
            compression_level,
            predictor,
            rows_per_strip,
            force_bigtiff,
            ij,
            description,
        } = options;

        if width == 0 || height == 0 {
            bail!("frame dimensions must be non-zero (got {width}x{height})");
        }
        if samples_per_pixel == 0 {
            bail!("samples_per_pixel must be at least 1");
        }
        if let Compression::Other(code) = compression {
            bail!("cannot write unsupported compression scheme {code} (use None/Lzw/PackBits/Deflate/Zstd)");
        }
        // Predictor 2 for integers (any width), Predictor 3 for floats —
        // matching what libtiff chooses for the same data.
        let predictor_tag: u16 = match (predictor, sample_type) {
            (false, _) => 1,
            (true, SampleType::F32 | SampleType::F64) => 3,
            (true, _) => 2,
        };
        if ij.is_some() && description.is_some() {
            bail!("imagej(..) and description(..) are mutually exclusive (both write tag 270)");
        }
        // ASCII fields are NUL-terminated, so an interior NUL would silently
        // truncate the text on read-back; ImageJ's format is line-oriented, so
        // stray newlines / '=' in its strings would corrupt the key=value
        // lines. Reject both up front instead of writing a broken file.
        if description.as_deref().is_some_and(|d| d.contains('\0')) {
            bail!("description must not contain NUL bytes (TIFF ASCII fields are NUL-terminated)");
        }
        if let Some(ij) = &ij {
            let clean = |s: &str| !s.contains('\0') && !s.contains('\n');
            if !ij.unit.as_deref().map_or(true, clean) {
                bail!("ImageJ unit must not contain NUL or newline characters");
            }
            for (key, value) in &ij.extra {
                if !clean(key) || key.contains('=') || !clean(value) {
                    bail!("ImageJ extra key/value {key:?} must not contain NUL, newline, or '=' in the key");
                }
            }
        }

        let spp = samples_per_pixel as usize;
        // PlanarConfiguration is irrelevant for single-sample data (TIFF6), and
        // the two layouts are byte-identical there — normalize it away so every
        // check downstream (and the tag emission) has one meaning.
        let planar = planar && spp > 1;
        // Planar splits the frame into `spp` images of `width`-sample rows;
        // chunky is one image of `width * spp`-sample rows. Either way the
        // frame is the same total size — only the strip split differs.
        let n_planes = if planar { spp } else { 1 };
        let row_bytes = width as usize * (spp / n_planes) * sample_type.bytes();
        let plane_bytes = row_bytes * height as usize;
        let frame_bytes = plane_bytes * n_planes;
        let rows_per_strip = match rows_per_strip {
            Some(r) => r.clamp(1, height),
            // Uncompressed: one strip per frame — the reader's zero-copy fast
            // path. Compressed: ~256 KiB strips, so large frames decompress in
            // parallel on the way back in.
            None if compression == Compression::None => height,
            None => (((256 * 1024) / row_bytes.max(1)).max(1) as u32).min(height),
        };

        // Reserve 16 zero bytes and start pixel data at offset 16: enough
        // room for either header. finish() writes the real one — classic
        // (8 bytes + 8 legal pad bytes) or BigTIFF (16 bytes) — once it knows
        // whether the file outgrew classic TIFF's 4 GiB offsets. This is what
        // makes the BigTIFF upgrade automatic with zero re-writing of pixel
        // data: offsets are identical under both headers.
        w.write_all(&[0u8; 16])?;

        Ok(Self {
            w,
            pos: 16,
            frames: Vec::new(),
            width,
            height,
            sample_type,
            spp,
            planar,
            compression,
            compression_level,
            force_bigtiff,
            predictor_tag,
            rows_per_strip,
            ij,
            description,
            row_bytes,
            plane_bytes,
            frame_bytes,
        })
    }

    /// Number of frames (planes) written so far.
    pub fn frames_written(&self) -> usize {
        self.frames.len()
    }

    /// Append one frame from raw **little-endian** sample bytes — the
    /// escape hatch that covers every `SampleType`, including the signed and
    /// 32-bit-integer ones without a typed method (on a little-endian host,
    /// `bytemuck::cast_slice` turns any `&[i16]`/`&[u32]`/... into these bytes
    /// for free). Length must be exactly
    /// `width * height * samples_per_pixel * bytes_per_sample`; multi-sample
    /// data is chunky (interleaved per pixel) unless [`WriterOptions::planar`]
    /// was set, in which case it is plane-major (all of sample 0, then all of
    /// sample 1, …). Both layouts are the same length, so the check below
    /// cannot tell them apart — see that method.
    pub fn write_frame_bytes(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.frame_bytes {
            bail!(
                "frame data is {} bytes but the configured frame layout ({}x{}, {} sample(s)/px, \
                 {}-bit) needs exactly {}",
                data.len(),
                self.width,
                self.height,
                self.spp,
                self.sample_type.bits(),
                self.frame_bytes
            );
        }

        // Predictor first (a per-row operation, so strip boundaries — always
        // whole rows — don't affect it), then split into strips and compress
        // each strip independently, as the TIFF spec requires. The differencing
        // stride is one pixel to the left: `spp` samples away when interleaved,
        // 1 when each plane is stored whole.
        let stride = if self.planar { 1 } else { self.spp };
        let processed: Cow<[u8]> = match self.predictor_tag {
            2 => {
                let mut owned = data.to_vec();
                apply_predictor(&mut owned, self.row_bytes, stride, self.sample_type.bytes());
                Cow::Owned(owned)
            }
            3 => {
                let mut owned = data.to_vec();
                apply_float_predictor(&mut owned, self.row_bytes, stride, self.sample_type.bytes());
                Cow::Owned(owned)
            }
            _ => Cow::Borrowed(data),
        };

        // Strips never span a plane boundary: chunky yields one plane (the
        // whole frame), planar yields `spp`, each split independently —
        // StripsPerImage x SamplesPerPixel strips, as TIFF6 requires and as the
        // reader's `strip_dest_lens` expects on the way back in.
        let strip_len = self.rows_per_strip as usize * self.row_bytes;
        let plane_bytes = self.plane_bytes;
        let mut strips = FrameStrips { offsets: Vec::new(), byte_counts: Vec::new() };

        if self.compression == Compression::None {
            // Raw strips stream straight from the (possibly borrowed) frame
            // buffer — for the default single-strip layout this is one
            // contiguous write with no intermediate allocation.
            for plane in processed.chunks(plane_bytes) {
                for chunk in plane.chunks(strip_len) {
                    self.push_strip_bytes(chunk, &mut strips)?;
                }
            }
        } else {
            let chunks: Vec<&[u8]> =
                processed.chunks(plane_bytes).flat_map(|plane| plane.chunks(strip_len)).collect();
            let compression = self.compression;
            let level = self.compression_level;
            let row_bytes = self.row_bytes;
            // Strips are independent compressed units, so a big frame's strips
            // compress in parallel (ordered collect preserves row order) —
            // under the same process-wide hint + size floor as decoding
            // (`set_parallel_decode`), so the host has one threading switch.
            let n_pixels = self.width as usize * self.height as usize;
            let compressed: Vec<Vec<u8>> = if chunks.len() > 1 && crate::decode::should_parallelize(n_pixels) {
                chunks
                    .par_iter()
                    .map(|c| compress_strip(c, compression, row_bytes, level))
                    .collect::<Result<_>>()?
            } else {
                chunks
                    .iter()
                    .map(|c| compress_strip(c, compression, row_bytes, level))
                    .collect::<Result<_>>()?
            };
            for strip in &compressed {
                self.push_strip_bytes(strip, &mut strips)?;
            }
        }

        self.frames.push(strips);
        Ok(())
    }

    /// Append one frame of `u8` samples. Requires `SampleType::U8`.
    pub fn write_frame_u8(&mut self, samples: &[u8]) -> Result<()> {
        self.expect_type(SampleType::U8, "write_frame_u8")?;
        self.write_frame_bytes(samples)
    }

    /// Append one frame of `u16` samples. Requires `SampleType::U16`.
    pub fn write_frame_u16(&mut self, samples: &[u16]) -> Result<()> {
        self.expect_type(SampleType::U16, "write_frame_u16")?;
        self.write_samples(samples)
    }

    /// Append one frame of `f32` samples. Requires `SampleType::F32`.
    pub fn write_frame_f32(&mut self, samples: &[f32]) -> Result<()> {
        self.expect_type(SampleType::F32, "write_frame_f32")?;
        self.write_samples(samples)
    }

    /// Append one frame of `f64` samples. Requires `SampleType::F64`.
    pub fn write_frame_f64(&mut self, samples: &[f64]) -> Result<()> {
        self.expect_type(SampleType::F64, "write_frame_f64")?;
        self.write_samples(samples)
    }

    /// Write the IFD chain, patch the header's first-IFD offset, flush, and
    /// return the underlying writer (so `Cursor` users can take the buffer
    /// back). The file is not a valid TIFF until this has run.
    pub fn finish(mut self) -> Result<W> {
        if self.frames.is_empty() {
            bail!("no frames were written — a TIFF needs at least one image");
        }

        // The ImageDescription for the first IFD: ImageJ metadata (with the
        // now-known plane count), or the verbatim text, or nothing.
        let description: Option<String> = match (&self.ij, &self.description) {
            (Some(ij), _) => Some(build_ij_description(self.frames.len(), ij)?),
            (None, Some(text)) => Some(text.clone()),
            (None, None) => None,
        };

        // IFD tables must start on a word boundary.
        if self.pos % 2 == 1 {
            self.w.write_all(&[0])?;
            self.pos += 1;
        }

        // Decide classic vs BigTIFF: classic when everything — pixel data AND
        // the IFD region — fits below the 4 GiB offset limit. The sizing pass
        // uses classic-shaped entries; entry sizes don't depend on the offset
        // *values*, so this is exact even when those values wouldn't fit.
        let build_all = |big: bool| -> Vec<Vec<Entry>> {
            (0..self.frames.len())
                .map(|i| self.build_entries(i, if i == 0 { description.as_deref() } else { None }, big))
                .collect()
        };
        let size_region = |entry_lists: &[Vec<Entry>], flavor: TiffFlavor| -> (Vec<u64>, u64) {
            let mut table_offsets = Vec::with_capacity(entry_lists.len());
            let mut cursor = self.pos;
            for entries in entry_lists {
                let (end, table) = layout_ifd(entries, cursor, 0, flavor, None);
                table_offsets.push(table);
                cursor = end;
            }
            (table_offsets, cursor)
        };

        let classic_lists = build_all(false);
        let (_, classic_end) = size_region(&classic_lists, TiffFlavor::Classic);
        let big = self.force_bigtiff || classic_end > MAX_CLASSIC_TIFF;

        let (flavor, entry_lists) = if big {
            (TiffFlavor::Big, build_all(true))
        } else {
            (TiffFlavor::Classic, classic_lists)
        };
        let (table_offsets, end) = size_region(&entry_lists, flavor);

        let mut region = Vec::with_capacity((end - self.pos) as usize);
        let mut start = self.pos;
        for (i, entries) in entry_lists.iter().enumerate() {
            let next = if i + 1 < table_offsets.len() { table_offsets[i + 1] } else { 0 };
            let (new_start, _) = layout_ifd(entries, start, next, flavor, Some(&mut region));
            start = new_start;
        }
        self.w.write_all(&region)?;

        // The one seek: write the real header over the 16 reserved bytes.
        // Both layouts leave pixel data untouched (it starts at offset 16);
        // classic files just carry 8 legal pad bytes after their header.
        self.w.flush()?;
        self.w.seek(SeekFrom::Start(0))?;
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(b"II");
        match flavor {
            TiffFlavor::Classic => {
                header.extend_from_slice(&42u16.to_le_bytes());
                header.extend_from_slice(&(table_offsets[0] as u32).to_le_bytes());
            }
            TiffFlavor::Big => {
                header.extend_from_slice(&43u16.to_le_bytes());
                header.extend_from_slice(&8u16.to_le_bytes()); // offset byte size
                header.extend_from_slice(&0u16.to_le_bytes()); // reserved
                header.extend_from_slice(&table_offsets[0].to_le_bytes());
            }
        }
        self.w.write_all(&header)?;
        self.w.flush()?;
        Ok(self.w)
    }

    // ---- internals ----

    fn expect_type(&self, wanted: SampleType, method: &str) -> Result<()> {
        if self.sample_type != wanted {
            bail!(
                "{method} requires SampleType::{wanted:?} but the writer was configured with \
                 SampleType::{:?} (use write_frame_bytes for other layouts)",
                self.sample_type
            );
        }
        Ok(())
    }

    /// Typed samples → little-endian bytes. On little-endian hosts (i.e.
    /// everywhere this realistically runs) this is a free `bytemuck` cast; a
    /// big-endian host pays one conversion pass.
    fn write_samples<T: bytemuck::Pod>(&mut self, samples: &[T]) -> Result<()> {
        if cfg!(target_endian = "little") {
            self.write_frame_bytes(bytemuck::cast_slice(samples))
        } else {
            let size = std::mem::size_of::<T>();
            let mut le = Vec::with_capacity(samples.len() * size);
            for s in samples {
                let bytes = bytemuck::bytes_of(s);
                le.extend(bytes.iter().rev()); // native BE -> LE
            }
            self.write_frame_bytes(&le)
        }
    }

    /// Record and write one strip. No size limit here: offsets are tracked as
    /// u64, and finish() picks BigTIFF automatically when they outgrow u32.
    fn push_strip_bytes(&mut self, bytes: &[u8], strips: &mut FrameStrips) -> Result<()> {
        strips.offsets.push(self.pos);
        strips.byte_counts.push(bytes.len() as u64);
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// The IFD entries for frame `i`, in ascending tag order (required by the
    /// TIFF spec). `description` is only passed for the first frame. `big`
    /// selects LONG8 (BigTIFF) vs LONG storage for the strip locations.
    fn build_entries(&self, i: usize, description: Option<&str>, big: bool) -> Vec<Entry> {
        let strips = &self.frames[i];
        let spp = self.spp as u16;
        let bits = self.sample_type.bits();
        let compression_code: u16 = match self.compression {
            Compression::None => 1,
            Compression::Lzw => 5,
            Compression::Deflate => 8,
            Compression::PackBits => 32773,
            Compression::Zstd => 50000, // libtiff/GDAL registered extension
            Compression::Other(code) => code, // rejected at construction
        };
        // 3+ samples is RGB (photometric 2) in either interleaving; everything
        // else is BlackIsZero grayscale — mirroring what the reader's `is_rgb`
        // keys on.
        let photometric: u16 = if spp >= 3 { 2 } else { 1 };

        let mut entries = Vec::with_capacity(12);
        entries.push(Entry::long(TAG_IMAGE_WIDTH, self.width));
        entries.push(Entry::long(TAG_IMAGE_LENGTH, self.height));
        entries.push(Entry::shorts(TAG_BITS_PER_SAMPLE, &vec![bits; spp as usize]));
        entries.push(Entry::short(TAG_COMPRESSION, compression_code));
        entries.push(Entry::short(TAG_PHOTOMETRIC, photometric));
        if let Some(text) = description {
            entries.push(Entry::ascii(TAG_IMAGE_DESCRIPTION, text));
        }
        // Strip locations: LONG8 in BigTIFF (offsets can exceed u32), LONG in
        // classic (the flavor decision guarantees they fit — truncation below
        // can only happen in the discarded classic sizing pass of a file that
        // will be BigTIFF).
        if big {
            entries.push(Entry::long8s(TAG_STRIP_OFFSETS, &strips.offsets));
        } else {
            let offs: Vec<u32> = strips.offsets.iter().map(|&v| v as u32).collect();
            entries.push(Entry::longs(TAG_STRIP_OFFSETS, &offs));
        }
        entries.push(Entry::short(TAG_SAMPLES_PER_PIXEL, spp));
        entries.push(Entry::long(TAG_ROWS_PER_STRIP, self.rows_per_strip));
        if big {
            entries.push(Entry::long8s(TAG_STRIP_BYTE_COUNTS, &strips.byte_counts));
        } else {
            let counts: Vec<u32> = strips.byte_counts.iter().map(|&v| v as u32).collect();
            entries.push(Entry::longs(TAG_STRIP_BYTE_COUNTS, &counts));
        }
        // PlanarConfiguration is only written when it isn't the default: TIFF6
        // defines an absent tag as 1 (chunky), and calls it irrelevant for
        // single-sample data (which `self.planar` has already normalized away).
        if self.planar {
            entries.push(Entry::short(TAG_PLANAR_CONFIG, 2));
        }
        if self.predictor_tag != 1 {
            entries.push(Entry::short(TAG_PREDICTOR, self.predictor_tag));
        }
        // Samples beyond what the photometric interpretation accounts for
        // (3 for RGB, 1 for grayscale) must be declared in ExtraSamples per
        // TIFF6; value 0 = unspecified data (not premultiplied alpha).
        let base_samples: usize = if photometric == 2 { 3 } else { 1 };
        if spp as usize > base_samples {
            entries.push(Entry::shorts(TAG_EXTRA_SAMPLES, &vec![0u16; spp as usize - base_samples]));
        }
        entries.push(Entry::shorts(
            TAG_SAMPLE_FORMAT,
            &vec![self.sample_type.format_code(); spp as usize],
        ));
        entries
    }
}

/// One IFD entry, value bytes already little-endian.
struct Entry {
    tag: u16,
    ftype: u16,
    count: u32,
    data: Vec<u8>,
}

impl Entry {
    fn short(tag: u16, v: u16) -> Self {
        Entry { tag, ftype: 3, count: 1, data: v.to_le_bytes().to_vec() }
    }
    fn shorts(tag: u16, vals: &[u16]) -> Self {
        let mut data = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Entry { tag, ftype: 3, count: vals.len() as u32, data }
    }
    fn long(tag: u16, v: u32) -> Self {
        Entry { tag, ftype: 4, count: 1, data: v.to_le_bytes().to_vec() }
    }
    fn longs(tag: u16, vals: &[u32]) -> Self {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Entry { tag, ftype: 4, count: vals.len() as u32, data }
    }
    /// LONG8 (type 16) — BigTIFF's 64-bit unsigned, for strip locations.
    fn long8s(tag: u16, vals: &[u64]) -> Self {
        let mut data = Vec::with_capacity(vals.len() * 8);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Entry { tag, ftype: 16, count: vals.len() as u32, data }
    }
    fn ascii(tag: u16, text: &str) -> Self {
        // ASCII fields are NUL-terminated; the count includes the terminator.
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        let count = data.len() as u32;
        Entry { tag, ftype: 2, count, data }
    }
}

/// Lay out (and optionally serialize) one IFD at absolute offset `start`:
/// external value blobs first (even-aligned, for entries whose data exceeds
/// the flavor's inline field — 4 bytes classic, 8 BigTIFF), then the entry
/// table (12-byte entries / u16 count / u32 next for classic; 20-byte entries
/// / u64 count / u64 next for BigTIFF). Returns `(end_offset, table_offset)`
/// — the table offset is what next-IFD pointers and the header must point at.
/// Sizing and serialization share this one function so they can't drift apart.
fn layout_ifd(
    entries: &[Entry],
    start: u64,
    next_ifd: u64,
    flavor: TiffFlavor,
    mut out: Option<&mut Vec<u8>>,
) -> (u64, u64) {
    let (inline_cap, count_len, entry_len, next_len) = match flavor {
        TiffFlavor::Classic => (4usize, 2u64, 12u64, 4u64),
        TiffFlavor::Big => (8, 8, 20, 8),
    };
    let mut ext_len = 0u64;
    let mut value_fields: Vec<[u8; 8]> = Vec::with_capacity(entries.len());
    for e in entries {
        if e.data.len() <= inline_cap {
            let mut field = [0u8; 8];
            field[..e.data.len()].copy_from_slice(&e.data);
            value_fields.push(field);
        } else {
            if ext_len % 2 == 1 {
                ext_len += 1;
                if let Some(out) = out.as_deref_mut() {
                    out.push(0);
                }
            }
            let mut field = [0u8; 8];
            field.copy_from_slice(&(start + ext_len).to_le_bytes());
            value_fields.push(field);
            ext_len += e.data.len() as u64;
            if let Some(out) = out.as_deref_mut() {
                out.extend_from_slice(&e.data);
            }
        }
    }
    if ext_len % 2 == 1 {
        ext_len += 1;
        if let Some(out) = out.as_deref_mut() {
            out.push(0);
        }
    }
    let table_offset = start + ext_len;
    let end = table_offset + count_len + entries.len() as u64 * entry_len + next_len;
    if let Some(out) = out {
        match flavor {
            TiffFlavor::Classic => out.extend_from_slice(&(entries.len() as u16).to_le_bytes()),
            TiffFlavor::Big => out.extend_from_slice(&(entries.len() as u64).to_le_bytes()),
        }
        for (e, field) in entries.iter().zip(&value_fields) {
            out.extend_from_slice(&e.tag.to_le_bytes());
            out.extend_from_slice(&e.ftype.to_le_bytes());
            match flavor {
                TiffFlavor::Classic => {
                    out.extend_from_slice(&e.count.to_le_bytes());
                    out.extend_from_slice(&field[..4]);
                }
                TiffFlavor::Big => {
                    out.extend_from_slice(&(e.count as u64).to_le_bytes());
                    out.extend_from_slice(field);
                }
            }
        }
        match flavor {
            TiffFlavor::Classic => out.extend_from_slice(&(next_ifd as u32).to_le_bytes()),
            TiffFlavor::Big => out.extend_from_slice(&next_ifd.to_le_bytes()),
        }
    }
    (end, table_offset)
}

/// The ImageJ `ImageDescription` block, using exactly the keys
/// `ij_metadata::build_stack_meta` parses back (`images`, `channels`, `slices`,
/// `frames`, `mode`, `unit`, `finterval`, `fps`, `min`/`max`, `cf`/`c0`/`c1`).
fn build_ij_description(planes: usize, ij: &ImageJOptions) -> Result<String> {
    let per_frame = ij.channels * ij.slices;
    if planes % per_frame != 0 {
        bail!(
            "{planes} plane(s) written, which doesn't divide evenly into {} channel(s) x {} \
             slice(s) — an ImageJ hyperstack needs channels x slices planes per time frame",
            ij.channels,
            ij.slices
        );
    }
    let frames = planes / per_frame;

    let mut s = String::from("ImageJ=1.54f\n");
    s += &format!("images={planes}\n");
    if ij.channels > 1 {
        s += &format!("channels={}\n", ij.channels);
    }
    if ij.slices > 1 {
        s += &format!("slices={}\n", ij.slices);
    }
    if frames > 1 {
        s += &format!("frames={frames}\n");
    }
    if [ij.channels > 1, ij.slices > 1, frames > 1].iter().filter(|&&b| b).count() >= 2 {
        s += "hyperstack=true\n";
    }
    if ij.channels > 1 || ij.mode != DisplayMode::Grayscale {
        let mode = match ij.mode {
            DisplayMode::Grayscale => "grayscale",
            DisplayMode::Composite => "composite",
            DisplayMode::Color => "color",
        };
        s += &format!("mode={mode}\n");
    }
    if let Some(unit) = &ij.unit {
        s += &format!("unit={unit}\n");
    }
    if let Some(fi) = ij.frame_interval_s {
        s += &format!("finterval={fi}\n");
    }
    if let Some(fps) = ij.fps {
        s += &format!("fps={fps}\n");
    }
    if let Some(spacing) = ij.spacing {
        s += &format!("spacing={spacing}\n");
    }
    if let Some(looped) = ij.loop_playback {
        s += &format!("loop={looped}\n");
    }
    if let Some((lo, hi)) = ij.range {
        s += &format!("min={lo}\nmax={hi}\n");
    }
    if let Some((c0, c1)) = ij.calibration {
        s += &format!("cf=0\nc0={c0}\nc1={c1}\n");
    }
    for (key, value) in &ij.extra {
        s += &format!("{key}={value}\n");
    }
    Ok(s)
}

/// Apply TIFF Predictor 2 (horizontal differencing) in place — the exact
/// inverse of `decode::undo_predictor`, in little-endian (the only order this
/// writer emits). Per row, per sample plane; iterates right-to-left so each
/// difference reads the original (not already-differenced) left neighbor.
fn apply_predictor(data: &mut [u8], row_bytes: usize, stride: usize, sample_bytes: usize) {
    let row_samples = row_bytes / sample_bytes;
    match sample_bytes {
        1 => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (stride..row_samples).rev() {
                    row[i] = row[i].wrapping_sub(row[i - stride]);
                }
            }
        }
        2 => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (stride..row_samples).rev() {
                    let cur = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
                    let prev = u16::from_le_bytes([row[(i - stride) * 2], row[(i - stride) * 2 + 1]]);
                    let diff = cur.wrapping_sub(prev).to_le_bytes();
                    row[i * 2] = diff[0];
                    row[i * 2 + 1] = diff[1];
                }
            }
        }
        4 => {
            // 32-bit integers (U32/I32) — floats route to predictor 3 instead.
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (stride..row_samples).rev() {
                    let cur = u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
                    let prev = u32::from_le_bytes(row[(i - stride) * 4..(i - stride) * 4 + 4].try_into().unwrap());
                    let diff = cur.wrapping_sub(prev).to_le_bytes();
                    row[i * 4..i * 4 + 4].copy_from_slice(&diff);
                }
            }
        }
        8 => {
            // 64-bit integers (U64/I64) — floats route to predictor 3 instead.
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (stride..row_samples).rev() {
                    let cur = u64::from_le_bytes(row[i * 8..i * 8 + 8].try_into().unwrap());
                    let prev = u64::from_le_bytes(row[(i - stride) * 8..(i - stride) * 8 + 8].try_into().unwrap());
                    let diff = cur.wrapping_sub(prev).to_le_bytes();
                    row[i * 8..i * 8 + 8].copy_from_slice(&diff);
                }
            }
        }
        _ => unreachable!("sample widths are 1/2/4/8 bytes"),
    }
}

/// Apply TIFF Predictor 3 (TechNote 3 floating-point differencing) in place —
/// the exact inverse of `decode::undo_float_predictor`. Per row: split each
/// float into the row's `sample_bytes` byte-significance planes (MSB plane
/// first, as the spec requires regardless of file byte order), then difference
/// the plane bytes horizontally with stride = samples per pixel, mirroring
/// libtiff's `fpDiff`. Handles f32 (`sample_bytes == 4`) and f64 (`== 8`).
fn apply_float_predictor(data: &mut [u8], row_bytes: usize, stride: usize, sample_bytes: usize) {
    let wc = row_bytes / sample_bytes; // float values per row
    let mut scratch = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for v in 0..wc {
            // Values arrive little-endian (this writer's only order); plane p
            // (0 = MSB) takes the LE byte at significance `sample_bytes-1-p`.
            for p in 0..sample_bytes {
                scratch[p * wc + v] = row[v * sample_bytes + (sample_bytes - 1 - p)];
            }
        }
        for i in (stride..row_bytes).rev() {
            scratch[i] = scratch[i].wrapping_sub(scratch[i - stride]);
        }
        row.copy_from_slice(&scratch);
    }
}

/// Default effort levels used when the caller doesn't set `compression_level`:
/// a balanced Deflate 6 and a fast Zstd 3 (both the usual library defaults),
/// chosen here explicitly so the write path's behavior is fixed and documented
/// rather than inherited from whatever the codec crate happens to default to.
pub const DEFAULT_DEFLATE_LEVEL: i32 = 6;
pub const DEFAULT_ZSTD_LEVEL: i32 = 3;

/// Compress one strip. Mirrors `decode::decompress` codec-for-codec so
/// everything written here reads back with the sibling decoder (and libtiff,
/// ImageJ, etc.). `level` is the optional effort knob for the codecs that have
/// one (Deflate 0..=9 clamped, Zstd 1..=22); `None` = the lib default
/// (`DEFAULT_DEFLATE_LEVEL` / `DEFAULT_ZSTD_LEVEL`).
fn compress_strip(strip: &[u8], compression: Compression, row_bytes: usize, level: Option<i32>) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(strip.to_vec()),
        Compression::Lzw => {
            // TIFF-flavored LZW: MSb-first with the early size switch —
            // the mirror of the decoder's `with_tiff_size_switch`.
            let mut out = Vec::with_capacity(strip.len() / 2);
            let mut encoder = weezl::encode::Encoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            encoder
                .into_stream(&mut out)
                .encode_all(strip)
                .status
                .map_err(|e| anyhow!("LZW encode failed: {e:?}"))?;
            Ok(out)
        }
        Compression::Deflate => {
            let l = level.unwrap_or(DEFAULT_DEFLATE_LEVEL).clamp(0, 9);
            let flate_level = flate2::Compression::new(l as u32);
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::with_capacity(strip.len() / 2), flate_level);
            encoder.write_all(strip).map_err(|e| anyhow!("Deflate encode failed: {e}"))?;
            encoder.finish().map_err(|e| anyhow!("Deflate encode failed: {e}"))
        }
        Compression::PackBits => Ok(packbits_encode(strip, row_bytes)),
        Compression::Zstd => {
            zstd::stream::encode_all(strip, level.unwrap_or(DEFAULT_ZSTD_LEVEL))
                .map_err(|e| anyhow!("ZSTD encode failed: {e}"))
        }
        Compression::Other(code) => bail!("unsupported TIFF compression scheme: {code}"),
    }
}

/// PackBits-compress a strip. Per the TIFF spec, each row is packed
/// independently (runs never cross row boundaries), which is also what libtiff
/// emits.
fn packbits_encode(strip: &[u8], row_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(strip.len() / 2 + 8);
    for row in strip.chunks(row_bytes) {
        packbits_encode_row(row, &mut out);
    }
    out
}

fn packbits_encode_row(row: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < row.len() {
        // Length of the run of identical bytes starting at i (capped at 128,
        // the most one PackBits record can express).
        let mut run = 1;
        while run < 128 && i + run < row.len() && row[i + run] == row[i] {
            run += 1;
        }
        if run >= 2 {
            // Replicate record: count byte -(run-1), then the byte.
            out.push((1i32 - run as i32) as i8 as u8);
            out.push(row[i]);
            i += run;
        } else {
            // Literal record: gather bytes until a run of 3+ starts (2 is
            // break-even, not worth interrupting a literal for) or 128 bytes.
            let start = i;
            i += 1;
            while i < row.len() && i - start < 128 {
                if i + 2 < row.len() && row[i] == row[i + 1] && row[i] == row[i + 2] {
                    break;
                }
                i += 1;
            }
            out.push((i - start - 1) as u8);
            out.extend_from_slice(&row[start..i]);
        }
    }
}

#[cfg(test)]
#[path = "encode_tests.rs"]
mod tests;

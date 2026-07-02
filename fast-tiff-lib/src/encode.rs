//! Turns pixel data back into a multi-frame TIFF file — the write-side
//! counterpart of `decode`. The API shape follows TinyTIFF's proven stack-writer
//! model (open with a fixed frame layout, append frames one at a time, finish),
//! the format coverage follows libtiff (None/LZW/PackBits/Deflate compression,
//! horizontal predictor, unsigned/signed/float samples at 8/16/32 bits, chunky
//! RGB, multi-strip), and the Rust idioms follow the `tiff` crate (a writer
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
//! (checked, with a clear error). BigTIFF, tiles, and planar (non-chunky)
//! layouts are out of scope — matching the reader.

use crate::ij_metadata::DisplayMode;
use crate::index::{
    Compression, TAG_BITS_PER_SAMPLE, TAG_COMPRESSION, TAG_IMAGE_DESCRIPTION, TAG_IMAGE_LENGTH,
    TAG_IMAGE_WIDTH, TAG_PHOTOMETRIC, TAG_PREDICTOR, TAG_ROWS_PER_STRIP, TAG_SAMPLES_PER_PIXEL,
    TAG_SAMPLE_FORMAT, TAG_STRIP_BYTE_COUNTS, TAG_STRIP_OFFSETS,
};
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Classic TIFF stores every offset as a u32, so nothing may live at or past
/// 4 GiB. (BigTIFF, which lifts this, is not supported — matching the reader.)
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
}

impl SampleType {
    /// TIFF `BitsPerSample` for this type.
    pub fn bits(self) -> u16 {
        match self {
            SampleType::U8 | SampleType::I8 => 8,
            SampleType::U16 | SampleType::I16 => 16,
            SampleType::U32 | SampleType::I32 | SampleType::F32 => 32,
        }
    }

    /// Bytes per sample (`bits / 8`).
    pub fn bytes(self) -> usize {
        self.bits() as usize / 8
    }

    /// TIFF `SampleFormat` code: 1 = unsigned, 2 = signed, 3 = IEEE float.
    fn format_code(self) -> u16 {
        match self {
            SampleType::U8 | SampleType::U16 | SampleType::U32 => 1,
            SampleType::I8 | SampleType::I16 | SampleType::I32 => 2,
            SampleType::F32 => 3,
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
    compression: Compression,
    predictor: bool,
    rows_per_strip: Option<u32>,
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
            compression: Compression::None,
            predictor: false,
            rows_per_strip: None,
            ij: None,
            description: None,
        }
    }

    /// Samples per pixel: 1 (default) writes single-plane grayscale frames;
    /// 3 writes chunky (interleaved) RGB. Data passed to `write_frame_*` is
    /// `width * height * samples_per_pixel` samples, interleaved per pixel.
    pub fn samples_per_pixel(mut self, spp: u16) -> Self {
        self.samples_per_pixel = spp;
        self
    }

    /// Strip compression: `None` (default), `Lzw`, `PackBits`, or `Deflate`.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
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
    compression: Compression,
    /// TIFF Predictor tag value: 1 = off, 2 = integer horizontal differencing,
    /// 3 = floating-point (TechNote 3). Resolved from the sample type at
    /// construction.
    predictor_tag: u16,
    rows_per_strip: u32,
    ij: Option<ImageJOptions>,
    description: Option<String>,
    row_bytes: usize,
    frame_bytes: usize,
}

struct FrameStrips {
    offsets: Vec<u32>,
    byte_counts: Vec<u32>,
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
            compression,
            predictor,
            rows_per_strip,
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
            bail!("cannot write unsupported compression scheme {code} (use None/Lzw/PackBits/Deflate)");
        }
        // Predictor 2 for integers (any width), Predictor 3 for floats —
        // matching what libtiff chooses for the same data.
        let predictor_tag: u16 = match (predictor, sample_type) {
            (false, _) => 1,
            (true, SampleType::F32) => 3,
            (true, _) => 2,
        };
        if ij.is_some() && description.is_some() {
            bail!("imagej(..) and description(..) are mutually exclusive (both write tag 270)");
        }

        let spp = samples_per_pixel as usize;
        let row_bytes = width as usize * spp * sample_type.bytes();
        let frame_bytes = row_bytes * height as usize;
        let rows_per_strip = match rows_per_strip {
            Some(r) => r.clamp(1, height),
            // Uncompressed: one strip per frame — the reader's zero-copy fast
            // path. Compressed: ~256 KiB strips, so large frames decompress in
            // parallel on the way back in.
            None if compression == Compression::None => height,
            None => (((256 * 1024) / row_bytes.max(1)).max(1) as u32).min(height),
        };

        // Header: "II" (little-endian), magic 42, first-IFD offset patched in
        // finish() once the chain's location is known.
        w.write_all(b"II")?;
        w.write_all(&42u16.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;

        Ok(Self {
            w,
            pos: 8,
            frames: Vec::new(),
            width,
            height,
            sample_type,
            spp,
            compression,
            predictor_tag,
            rows_per_strip,
            ij,
            description,
            row_bytes,
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
    /// `width * height * samples_per_pixel * bytes_per_sample`;
    /// multi-sample data is chunky (interleaved per pixel).
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
        // each strip independently, as the TIFF spec requires.
        let processed: Cow<[u8]> = match self.predictor_tag {
            2 => {
                let mut owned = data.to_vec();
                apply_predictor(&mut owned, self.row_bytes, self.spp, self.sample_type.bytes());
                Cow::Owned(owned)
            }
            3 => {
                let mut owned = data.to_vec();
                apply_float_predictor(&mut owned, self.row_bytes, self.spp);
                Cow::Owned(owned)
            }
            _ => Cow::Borrowed(data),
        };

        let strip_len = self.rows_per_strip as usize * self.row_bytes;
        let mut strips = FrameStrips { offsets: Vec::new(), byte_counts: Vec::new() };

        if self.compression == Compression::None {
            // Raw strips stream straight from the (possibly borrowed) frame
            // buffer — for the default single-strip layout this is one
            // contiguous write with no intermediate allocation.
            for chunk in processed.chunks(strip_len) {
                self.push_strip_bytes(chunk, &mut strips)?;
            }
        } else {
            let chunks: Vec<&[u8]> = processed.chunks(strip_len).collect();
            let compression = self.compression;
            let row_bytes = self.row_bytes;
            // Strips are independent compressed units, so a big frame's strips
            // compress in parallel (ordered collect preserves row order) —
            // under the same process-wide hint + size floor as decoding
            // (`set_parallel_decode`), so the host has one threading switch.
            let n_pixels = self.width as usize * self.height as usize;
            let compressed: Vec<Vec<u8>> = if chunks.len() > 1 && crate::decode::should_parallelize(n_pixels) {
                chunks
                    .par_iter()
                    .map(|c| compress_strip(c, compression, row_bytes))
                    .collect::<Result<_>>()?
            } else {
                chunks
                    .iter()
                    .map(|c| compress_strip(c, compression, row_bytes))
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

        // Build every frame's entry list, then lay the region out twice: a
        // sizing pass to learn each IFD table's offset (needed for the
        // next-IFD pointers), and a serialization pass using those offsets.
        let entry_lists: Vec<Vec<Entry>> = (0..self.frames.len())
            .map(|i| self.build_entries(i, if i == 0 { description.as_deref() } else { None }))
            .collect();

        let mut table_offsets = Vec::with_capacity(entry_lists.len());
        let mut cursor = self.pos;
        for entries in &entry_lists {
            let (end, table) = layout_ifd(entries, cursor, 0, None);
            table_offsets.push(table);
            cursor = end;
        }
        if cursor > MAX_CLASSIC_TIFF {
            bail!(
                "file would be {} bytes, past the 4 GiB classic-TIFF offset limit \
                 (BigTIFF is not supported)",
                cursor
            );
        }

        let mut region = Vec::with_capacity((cursor - self.pos) as usize);
        let mut start = self.pos;
        for (i, entries) in entry_lists.iter().enumerate() {
            let next = if i + 1 < table_offsets.len() { table_offsets[i + 1] as u32 } else { 0 };
            let (end, _) = layout_ifd(entries, start, next, Some(&mut region));
            start = end;
        }
        self.w.write_all(&region)?;

        // The one seek: point the header at the first IFD table.
        self.w.flush()?;
        self.w.seek(SeekFrom::Start(4))?;
        self.w.write_all(&(table_offsets[0] as u32).to_le_bytes())?;
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

    /// Record and write one strip, guarding the classic-TIFF offset limit.
    fn push_strip_bytes(&mut self, bytes: &[u8], strips: &mut FrameStrips) -> Result<()> {
        if self.pos + bytes.len() as u64 > MAX_CLASSIC_TIFF {
            bail!(
                "writing this strip would push the file past the 4 GiB classic-TIFF offset \
                 limit (BigTIFF is not supported) — split the stack across several files"
            );
        }
        strips.offsets.push(self.pos as u32);
        strips.byte_counts.push(bytes.len() as u32);
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    /// The IFD entries for frame `i`, in ascending tag order (required by the
    /// TIFF spec). `description` is only passed for the first frame.
    fn build_entries(&self, i: usize, description: Option<&str>) -> Vec<Entry> {
        let strips = &self.frames[i];
        let spp = self.spp as u16;
        let bits = self.sample_type.bits();
        let compression_code: u16 = match self.compression {
            Compression::None => 1,
            Compression::Lzw => 5,
            Compression::Deflate => 8,
            Compression::PackBits => 32773,
            Compression::Other(code) => code, // rejected at construction
        };
        // Chunky data with 3+ samples is RGB (photometric 2); everything else
        // is BlackIsZero grayscale — mirroring what the reader's `is_rgb` keys on.
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
        entries.push(Entry::longs(TAG_STRIP_OFFSETS, &strips.offsets));
        entries.push(Entry::short(TAG_SAMPLES_PER_PIXEL, spp));
        entries.push(Entry::long(TAG_ROWS_PER_STRIP, self.rows_per_strip));
        entries.push(Entry::longs(TAG_STRIP_BYTE_COUNTS, &strips.byte_counts));
        if self.predictor_tag != 1 {
            entries.push(Entry::short(TAG_PREDICTOR, self.predictor_tag));
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
    fn ascii(tag: u16, text: &str) -> Self {
        // ASCII fields are NUL-terminated; the count includes the terminator.
        let mut data = text.as_bytes().to_vec();
        data.push(0);
        let count = data.len() as u32;
        Entry { tag, ftype: 2, count, data }
    }
}

/// Lay out (and optionally serialize) one IFD at absolute offset `start`:
/// external value blobs first (even-aligned, for entries whose data exceeds the
/// 4-byte inline field), then the entry table. Returns `(end_offset,
/// table_offset)` — the table offset is what next-IFD pointers and the header
/// must point at. Sizing and serialization share this one function so they
/// can't drift apart.
fn layout_ifd(entries: &[Entry], start: u64, next_ifd: u32, mut out: Option<&mut Vec<u8>>) -> (u64, u64) {
    let mut ext_len = 0u64;
    let mut value_fields: Vec<[u8; 4]> = Vec::with_capacity(entries.len());
    for e in entries {
        if e.data.len() <= 4 {
            let mut field = [0u8; 4];
            field[..e.data.len()].copy_from_slice(&e.data);
            value_fields.push(field);
        } else {
            if ext_len % 2 == 1 {
                ext_len += 1;
                if let Some(out) = out.as_deref_mut() {
                    out.push(0);
                }
            }
            value_fields.push(((start + ext_len) as u32).to_le_bytes());
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
    let end = table_offset + 2 + entries.len() as u64 * 12 + 4;
    if let Some(out) = out {
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (e, field) in entries.iter().zip(&value_fields) {
            out.extend_from_slice(&e.tag.to_le_bytes());
            out.extend_from_slice(&e.ftype.to_le_bytes());
            out.extend_from_slice(&e.count.to_le_bytes());
            out.extend_from_slice(field);
        }
        out.extend_from_slice(&next_ifd.to_le_bytes());
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
fn apply_predictor(data: &mut [u8], row_bytes: usize, spp: usize, sample_bytes: usize) {
    let row_samples = row_bytes / sample_bytes;
    match sample_bytes {
        1 => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (spp..row_samples).rev() {
                    row[i] = row[i].wrapping_sub(row[i - spp]);
                }
            }
        }
        2 => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (spp..row_samples).rev() {
                    let cur = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
                    let prev = u16::from_le_bytes([row[(i - spp) * 2], row[(i - spp) * 2 + 1]]);
                    let diff = cur.wrapping_sub(prev).to_le_bytes();
                    row[i * 2] = diff[0];
                    row[i * 2 + 1] = diff[1];
                }
            }
        }
        4 => {
            // 32-bit integers (U32/I32) — floats route to predictor 3 instead.
            for row in data.chunks_exact_mut(row_bytes) {
                for i in (spp..row_samples).rev() {
                    let cur = u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
                    let prev = u32::from_le_bytes(row[(i - spp) * 4..(i - spp) * 4 + 4].try_into().unwrap());
                    let diff = cur.wrapping_sub(prev).to_le_bytes();
                    row[i * 4..i * 4 + 4].copy_from_slice(&diff);
                }
            }
        }
        _ => unreachable!("sample widths are 1/2/4 bytes"),
    }
}

/// Apply TIFF Predictor 3 (TechNote 3 floating-point differencing) in place —
/// the exact inverse of `decode::undo_float_predictor`. Per row: split each
/// f32 into the row's four byte-significance planes (MSB plane first, as the
/// spec requires regardless of file byte order), then difference the plane
/// bytes horizontally with stride = samples per pixel, mirroring libtiff's
/// `fpDiff`.
fn apply_float_predictor(data: &mut [u8], row_bytes: usize, spp: usize) {
    let wc = row_bytes / 4; // f32 values per row
    let mut scratch = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for v in 0..wc {
            // Values arrive little-endian (this writer's only order); the
            // planes want big-endian byte significance.
            scratch[v] = row[v * 4 + 3];
            scratch[wc + v] = row[v * 4 + 2];
            scratch[2 * wc + v] = row[v * 4 + 1];
            scratch[3 * wc + v] = row[v * 4];
        }
        for i in (spp..row_bytes).rev() {
            scratch[i] = scratch[i].wrapping_sub(scratch[i - spp]);
        }
        row.copy_from_slice(&scratch);
    }
}

/// Compress one strip. Mirrors `decode::decompress` codec-for-codec so
/// everything written here reads back with the sibling decoder (and libtiff,
/// ImageJ, etc.).
fn compress_strip(strip: &[u8], compression: Compression, row_bytes: usize) -> Result<Vec<u8>> {
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
            let mut encoder = flate2::write::ZlibEncoder::new(
                Vec::with_capacity(strip.len() / 2),
                flate2::Compression::default(),
            );
            encoder.write_all(strip).map_err(|e| anyhow!("Deflate encode failed: {e}"))?;
            encoder.finish().map_err(|e| anyhow!("Deflate encode failed: {e}"))
        }
        Compression::PackBits => Ok(packbits_encode(strip, row_bytes)),
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

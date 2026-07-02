# fast-tiff-lib

[![Crates.io](https://img.shields.io/crates/v/fast-tiff-lib?color=green)](https://crates.io/crates/fast-tiff-lib)
[![Downloads](https://img.shields.io/crates/d/fast-tiff-lib)](https://crates.io/crates/fast-tiff-lib)
[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/release.yml?label=build)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)

A lazy, memory-mapped reader — and a streaming writer — for multi-frame
(ImageJ hyperstack) TIFF files: IFD-chain indexing, ImageJ metadata/LUT
parsing, and per-frame strip decoding, with a zero-copy fast path for the
common uncompressed case. The [writer](#writing) emits exactly that fast-path
layout by default, so files it produces scrub back with no decode work.

It's the decode/parsing engine behind [FastTIFF](https://github.com/konung-yaropolk/FastTIFF),
split out so it can be used on its own. No GUI, no GPU — just file → pixels +
metadata (and back).

## What it does

- **Memory-maps** the file and walks the whole IFD chain, treating each IFD as
  one *plane* (one channel at one Z/T position), which is how ImageJ writes
  hyperstacks (`xyczt` order). It's a generic multi-page TIFF walker, not tied
  to any specific writer.
- **Decodes a plane to 16-bit or 32-bit-float samples** on demand. For the
  common case — uncompressed, single strip, native byte order — decoding a
  16-bit frame is a **zero-copy reinterpret** of the mapped bytes (no allocation,
  no decode pass).
- **Parses display metadata** from ImageJ's `ImageDescription` (tag 270) and,
  as a fallback, the binary `IJMetadata` block: channel/slice/frame counts,
  display mode, per-channel LUTs + contrast ranges, calibration, `fps`, etc.
- **Writes multi-frame stacks** streamingly (append a frame at a time, nothing
  buffered but the current frame): 8/16/32-bit integer or float, grayscale or
  chunky RGB, None/LZW/PackBits/Deflate compression with optional predictor,
  and ImageJ hyperstack metadata — see [Writing](#writing).

### Supported pixel formats

- 8-, 16-, and 32-bit; integer (signed or unsigned) or 32-bit IEEE float.
- Compression: none, LZW, PackBits, Deflate/zip. Predictor 2 (horizontal
  differencing, any integer width) and Predictor 3 (TechNote 3 floating-point,
  as libtiff writes for float data) are undone.
- Chunky (interleaved) RGB is deinterleaved per sample plane.

### Not supported

Tiled TIFFs, planar (non-chunky) multi-sample data, BigTIFF, and pyramidal /
mixed-size stacks (every frame must share frame 0's geometry — this is enforced
with a clear error at `open`).

## Quick start

```rust
use fast_tiff_lib::{read_frame_u16, TiffStack};

let stack = TiffStack::open("movie.tif")?;
println!(
    "{} planes, {}x{}, {}-bit",
    stack.frames.len(),
    stack.frames[0].width,
    stack.frames[0].height,
    stack.frames[0].bits_per_sample,
);

// Decode the first plane to display-ready u16 samples (0..=65535).
let frame = &stack.frames[0];
let pixels = read_frame_u16(&stack.mmap, frame, stack.byte_order, None)?;
assert_eq!(pixels.len(), (frame.width * frame.height) as usize);
# Ok::<(), anyhow::Error>(())
```

All fallible calls return `anyhow::Result`.

## Core API

### `TiffStack::open(path) -> Result<TiffStack>`

Opens and indexes the file. The returned struct exposes everything decoding
needs:

```rust
pub struct TiffStack {
    pub mmap: memmap2::Mmap,        // the mapped file bytes
    pub byte_order: ByteOrder,      // Little or Big (the file's endianness)
    pub frames: Vec<FrameInfo>,     // one entry per IFD/plane, in file order
    pub meta: StackMeta,            // parsed ImageJ display metadata
}
```

### `FrameInfo`

Everything needed to locate and decode one plane. Key fields:

```rust
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub sample_format: SampleFormat,   // UnsignedInt | SignedInt | Float
    pub compression: Compression,      // None | Lzw | PackBits | Deflate | Other(u16)
    pub predictor: u16,                // 1 = none, 2 = horizontal differencing
    pub photometric: u16,              // 2 = RGB
    pub planar_config: u16,            // 1 = chunky
    pub strip_offsets: Vec<u64>,
    pub strip_byte_counts: Vec<u64>,
    pub rows_per_strip: u32,
}
```

`frame.is_rgb()` is true for a chunky RGB frame whose samples are color
components you can deinterleave.

### Decoding

All decoders take the mapped bytes, the `FrameInfo`, and the file's
`ByteOrder`. They return owned data or a borrow of the mapping (`Cow`).

```rust
// 16-bit display path. 8-bit is upcast to 0..=65535; signed ints are offset
// into unsigned display space; 32-bit (int/float) is linearly rescaled into
// 0..=65535 using `float_range` (or the frame's own min/max when `None`).
// Zero-copy (Cow::Borrowed) for uncompressed, single-strip, native-order 16-bit.
fn read_frame_u16(mmap, frame, order, float_range: Option<(f32, f32)>) -> Result<Cow<[u16]>>;

// As above but for a single sample plane of a chunky multi-sample (e.g. RGB)
// frame; `plane` selects the component.
fn read_plane_u16(mmap, frame, order, float_range: Option<(f32, f32)>, plane: usize) -> Result<Vec<u16>>;

// Raw unsigned 8-bit bytes, *no* widening to 16-bit — for callers that scale to
// 0..=255 themselves (e.g. on the GPU). Zero-copy for uncompressed, single-strip
// 8-bit. Only valid for unsigned single-sample 8-bit frames.
fn read_frame_u8(mmap, frame, order) -> Result<Cow<[u8]>>;

// The deinterleaving sibling of read_frame_u8: one raw 8-bit sample plane of a
// chunky frame (e.g. 8-bit RGB), un-widened. Always allocates (a chunky plane is
// a strided gather, so no zero-copy borrow). 8-bit only.
fn read_plane_u8(mmap, frame, order, plane: usize) -> Result<Vec<u8>>;

// Raw 32-bit float samples, *no* rescaling — for callers that do window/level
// themselves (e.g. on the GPU). Zero-copy for uncompressed, single-strip,
// native-order float; integer 32-bit is cast to f32.
fn read_frame_f32(mmap, frame, order) -> Result<Cow<[f32]>>;
fn read_plane_f32(mmap, frame, order, plane: usize) -> Result<Vec<f32>>;

// Actual min/max of a 32-bit frame's raw values (for auto-ranging the display);
// `None` for non-32-bit frames.
fn frame_float_minmax(mmap, frame, order) -> Result<Option<(f32, f32)>>;

// Eager counterparts: decode *every* frame at once, in parallel across frames,
// into owned buffers (one per frame). For consumers that need the whole stack
// resident in RAM rather than scrubbing it lazily (see "Design & prior art").
// `_u16` covers any depth (8-bit is widened, 32-bit rescaled); `_u8` is the
// half-memory raw-byte loader for unsigned 8-bit stacks; `_f32` is raw float.
fn preload_frames_u16(stack: &TiffStack, float_range: Option<(f32, f32)>) -> Result<Vec<Vec<u16>>>;
fn preload_frames_u8(stack: &TiffStack) -> Result<Vec<Vec<u8>>>;   // unsigned 8-bit stacks
fn preload_frames_f32(stack: &TiffStack) -> Result<Vec<Vec<f32>>>;
```

To index a plane in a hyperstack (ImageJ's default `xyczt` order, channel
fastest), the position in `frames` is `frame_index * slices * channels + channel`.

### Parallel decoding (runtime hint)

Decoding can use [rayon](https://crates.io/crates/rayon) to split a frame's
strip decompression and 32-bit conversion across cores. It's controlled by a
**process-wide hint**, switchable at any time:

```rust
fast_tiff_lib::set_parallel_decode(true);  // split large decodes across cores
fast_tiff_lib::set_parallel_decode(false); // serial (default)
```

The same hint governs the [writer](#writing)'s per-strip compression, so one
switch controls all CPU-heavy pixel work in both directions.

This is a **performance hint only** — decoded pixels are identical either way.
Parallel decode spreads load across cores but uses *more total CPU* (fork-join
overhead), so it's only a win when a single core can't keep up (e.g. real-time
playback of a large compressed stack dropping frames). It's **off by default**,
and a small frame-size floor means tiny frames always decode serially regardless.
The host application is expected to flip it on only when needed.

## Writing

The write side is a streaming stack writer in the spirit of
[TinyTIFF](https://github.com/jkriege2/TinyTIFF): fix the frame layout up
front, append frames one at a time, `finish()`. Format coverage follows
libtiff (compression, predictor, sample formats, strips); the Rust shape
follows the `tiff` crate (`TiffWriter<W: Write + Seek>`, so files and
in-memory `Cursor`s both work).

```rust
use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};

let opts = WriterOptions::new(512, 512, SampleType::U16);
let mut writer = TiffWriter::create("stack.tif", opts)?;
for frame in frames {
    writer.write_frame_u16(&frame)?; // one plane per call, streamed to disk
}
writer.finish()?; // writes the IFD chain; the file isn't valid without it
# Ok::<(), anyhow::Error>(())
```

### What it writes

- **Samples:** `SampleType::{U8, I8, U16, I16, U32, I32, F32}` — TIFF
  unsigned/signed/float at 8/16/32 bits.
- **Layout:** grayscale planes (`samples_per_pixel(1)`, default) or chunky
  interleaved RGB (`samples_per_pixel(3)`, tagged photometric=RGB).
- **Compression:** `None` (default), `Lzw`, `PackBits`, `Deflate` — plus
  optional `predictor(true)`, which usually shrinks LZW/Deflate on
  continuous-tone data: integers get TIFF Predictor 2 (any width), `F32`
  gets Predictor 3 (the TechNote 3 floating-point predictor libtiff uses).
- **Strips:** uncompressed frames are one strip; compressed frames default to
  ~256 KiB strips. A big frame's strips compress in parallel under the same
  `set_parallel_decode` hint + size floor that governs decoding — one
  threading switch for all CPU-heavy pixel work — and decompress in parallel
  on the way back in. Override with `rows_per_strip(n)`.
- **Metadata:** `imagej(ImageJOptions...)` embeds an ImageJ hyperstack
  description — channels/slices (time frames are derived from the plane count
  at `finish()`), display mode, `fps`, frame interval, unit, `min=`/`max=`
  display range, linear calibration, Z `spacing`, playback `loop`, plus
  `extra(key, value)` for any other documented key. Or `description(text)`
  for a fully verbatim `ImageDescription` — either way, the reader hands the
  whole tag 270 text back via `TiffStack::description`.

```rust
use fast_tiff_lib::{Compression, DisplayMode, ImageJOptions, SampleType, WriterOptions};

// A 2-channel composite time series, Deflate-compressed:
let opts = WriterOptions::new(1024, 1024, SampleType::U16)
    .compression(Compression::Deflate)
    .predictor(true)
    .imagej(ImageJOptions::new(2, 1).mode(DisplayMode::Composite).fps(20.0));
```

Typed appenders `write_frame_u8` / `write_frame_u16` / `write_frame_f32`
mirror the readers; `write_frame_bytes` takes raw little-endian sample bytes
and covers the other sample types (`bytemuck::cast_slice` on any `&[i16]`,
`&[u32]`, ... produces them for free on little-endian hosts). Planes go in
ImageJ's `xyczt` order, matching the reader's indexing.

### Guarantees & limits

- Output is always **little-endian**, IFD entries in spec-required ascending
  tag order, ASCII fields NUL-terminated, IFDs word-aligned — standard TIFF6
  any reader accepts (libtiff, ImageJ, this crate).
- The uncompressed default (single strip, native order) is **exactly this
  reader's zero-copy path**: `read_frame_u16`/`_u8`/`_f32` borrow straight
  from the mapping, no decode pass. Verified by round-trip tests.
- Pixel data streams to the sink as frames arrive; `finish()` buffers only
  the IFD tables (~150 bytes/frame) and seeks once, to patch the header.
- **Classic TIFF only:** the file must stay under 4 GiB (offsets are u32) —
  exceeded writes fail with a clear error rather than corrupt. BigTIFF,
  tiles, and planar (non-chunky) layouts are not written — the same envelope
  the reader accepts. Binary `IJMetadata` LUT blocks aren't written (contrast
  ranges go in the description; LUT writing may come later).

### Metadata (`StackMeta`)

```rust
pub struct StackMeta {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub mode: DisplayMode,                  // Grayscale | Composite | Color
    pub unit: Option<String>,
    pub frame_interval_s: Option<f64>,
    pub channel_display: Vec<ChannelDisplay>,   // per-channel LUT + range
    pub calibration: Option<(f64, f64)>,        // linear (c0, c1): value = c0 + c1*raw
    pub fps: Option<f64>,
    pub spacing: Option<f64>,               // Z-step between slices (spacing=)
    pub loop_playback: Option<bool>,        // playback looping (loop=)
}
```

This is the *parsed ImageJ view* of the metadata. The raw `ImageDescription`
(tag 270) text is also exposed verbatim — whatever the writer put there,
ImageJ-formatted or not:

```rust
let stack = fast_tiff_lib::TiffStack::open("movie.tif")?;
if let Some(desc) = &stack.description {
    println!("{desc}"); // the whole tag 270 text, unparsed
}
# Ok::<(), anyhow::Error>(())

pub struct ChannelDisplay {
    pub lut: [[u8; 3]; 256],          // 256-entry RGB lookup table
    pub range: Option<(f64, f64)>,    // display window (min, max); None = auto-contrast
}
```

Helpers:

- `resolve_dimensions(c, z, f) -> ResolvedDimensions` — sanity-resolves
  channel/slice/frame counts against the actual plane count (and flags the
  ambiguous channels×Z×time case).
- `default_lut_for(mode, channel)`, `default_composite_lut(channel)`,
  `grayscale_lut()` — standard ImageJ LUTs for when a file carries none.

## Design & prior art

`fast-tiff-lib` is built around **lazy, on-demand decoding**: `open` only indexes the
IFD chain (offsets, dimensions, metadata), and pixels are decoded per frame when
asked — for uncompressed, native-order data that decode is a **zero-copy
reinterpret** of the mapped bytes (no allocation, no copy). This suits a
viewer/scrubber, which shouldn't pull a multi-gigabyte stack fully into RAM just
to show one frame.

A 2025 paper — Hang Lv, Longlong Zhang, Qiusheng Cao, *"A Method for
Accelerating the Loading of TIFF Files Based on Memory-Mapped File Technology"*,
Frontiers in Computing and Intelligent Systems **11(2), 100–105** (the method is
also, independently, named "FastTIFF") — shares the memory-mapped foundation but
takes the opposite *loading* strategy: it **eagerly decodes every frame into heap
buffers at load time, in parallel across frames**, using a custom work-stealing
thread pool. That targets a different use case — loading a whole file resident
for downstream batch processing (e.g. aerial/remote-sensing recognition).

| | Lv et al. "FastTIFF" | `fast-tiff-lib` |
| --- | --- | --- |
| Memory-mapped access | yes | yes |
| Eager full-stack load at open | yes (the point) | no — lazy / on-demand |
| Parallel **across frames** | yes (thread pool) | opt-in (`preload_frames_*`) |
| Parallel **within a frame** | no | yes (rayon, adaptive) |
| Zero-copy for uncompressed | no (always copies) | yes |

If you do want the paper's load-everything-at-once behavior, `fast-tiff-lib` offers
it explicitly via `preload_frames_u16` / `preload_frames_u8` / `preload_frames_f32`
— they decode all frames in parallel across frames into owned buffers — while the
default path stays lazy and zero-copy.

## Testing

Besides unit and writer→reader round-trip tests, the reader is
cross-validated against **independently-produced files**: a committed fixture
matrix (`tests/fixtures/`, ~30 files) generated by Python's `tifffile` and by
Pillow (whose compressed path runs the actual libtiff encoders) — covering
every sample type, all codecs, predictor 2/3, big-endian, RGB, multi-strip,
ImageJ hyperstack metadata, and an unsupported-tiled error case. Regenerate
with `tests/fixtures/generate_fixtures.py`.

## Dependencies

`memmap2`, `weezl` (LZW), `flate2` (Deflate), `anyhow`, `bytemuck`, `rayon`.

## License

Mozilla Public License 2.0. See the [LICENSE](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-lib/LICENSE).

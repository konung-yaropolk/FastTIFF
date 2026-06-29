# fast-tiff-lib

[![Crates.io](https://img.shields.io/crates/v/fast-tiff-lib?color=green)](https://crates.io/crates/fast-tiff-lib)
[![Downloads](https://img.shields.io/crates/d/fast-tiff-lib)](https://crates.io/crates/fast-tiff-lib)
[![License](https://img.shields.io/crates/l/fast-tiff-lib?color=green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/release.yml?label=build)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/ci.yml)

A lazy, memory-mapped reader for multi-frame (ImageJ hyperstack) TIFF files:
IFD-chain indexing, ImageJ metadata/LUT parsing, and per-frame strip decoding —
with a zero-copy fast path for the common uncompressed case.

It's the decode/parsing engine behind [FastTIFF](https://github.com/konung-yaropolk/FastTIFF),
split out so it can be used on its own. No GUI, no GPU — just file → pixels +
metadata.

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

### Supported pixel formats

- 8-, 16-, and 32-bit; integer (signed or unsigned) or 32-bit IEEE float.
- Compression: none, LZW, PackBits, Deflate/zip; horizontal predictor (2) undone.
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

This is a **performance hint only** — decoded pixels are identical either way.
Parallel decode spreads load across cores but uses *more total CPU* (fork-join
overhead), so it's only a win when a single core can't keep up (e.g. real-time
playback of a large compressed stack dropping frames). It's **off by default**,
and a small frame-size floor means tiny frames always decode serially regardless.
The host application is expected to flip it on only when needed.

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
}

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

## Dependencies

`memmap2`, `weezl` (LZW), `flate2` (Deflate), `anyhow`, `bytemuck`, `rayon`.

## License

LGPL-3.0-only. See the [LICENSE](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-lib/LICENSE).

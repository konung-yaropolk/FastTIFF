# fast-tiff-lib

[![Crates.io](https://img.shields.io/crates/v/fast-tiff-lib?color=green)](https://crates.io/crates/fast-tiff-lib)
[![Downloads](https://img.shields.io/crates/d/fast-tiff-lib)](https://crates.io/crates/fast-tiff-lib)
[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Docs](https://img.shields.io/docsrs/fast-tiff-lib?label=docs.rs)](https://docs.rs/fast-tiff-lib)

A lazy, memory-mapped reader — and a streaming writer — for multi-frame
scientific TIFF files: IFD-chain indexing, ImageJ + OME-TIFF metadata parsing,
and per-frame strip decoding, with a zero-copy fast path for the common
uncompressed case. It also reads from a plain byte buffer, so it works with no
filesystem at all — including on `wasm32-unknown-unknown`. The [writer](#writing) emits exactly that fast-path layout by
default, so files it produces scrub back with no decode work.

It's the decode/parsing engine behind [FastTIFF](https://github.com/konung-yaropolk/FastTIFF),
split out so it can be used on its own. No GUI, no GPU — just file → pixels +
metadata (and back).

## What it does

- **Memory-maps** the file and walks the whole IFD chain, treating each IFD as
  one *plane* (one channel at one Z/T position), which is how ImageJ writes
  hyperstacks (`xyczt` order). It's a generic multi-page TIFF walker, not tied
  to any specific writer.
- **Decodes a plane to 16-bit or 32-bit-float samples** on demand (64-bit
  int/float are down-converted to these — see [Supported pixel
  formats](#supported-pixel-formats)). For the common case — uncompressed,
  single strip, native byte order — decoding a 16-bit frame is a **zero-copy
  reinterpret** of the mapped bytes (no allocation, no decode pass).
- **Parses display metadata** from several dialects into one normalized
  `StackMeta`, picking the dialect by inspecting `ImageDescription` (tag 270):
  **ImageJ** (`key=value` + the binary `IJMetadata` LUT block) and **OME-TIFF**
  (OME-XML). Either way you get channel/slice/frame counts, display mode,
  per-channel LUTs + contrast ranges, calibration, `fps`, Z `spacing`, and x/y
  pixel size — enough to reconstruct the physical voxel scale. `meta.source_format`
  says which dialect a file used; new dialects slot in without changing consumers.
- **Writes multi-frame stacks** streamingly (append a frame at a time, nothing
  buffered but the current frame): 8/16/32/64-bit integer or 32/64-bit float,
  grayscale or RGB (chunky or planar), None/LZW/PackBits/Deflate/ZSTD
  compression with optional predictor, and structured metadata in the ImageJ or
  OME dialect from one neutral builder — see [Writing](#writing).

### Supported pixel formats

- 8-, 16-, 32-, and 64-bit; integer (signed or unsigned) or 32-/64-bit IEEE
  float.
- Compression: none, LZW, PackBits, Deflate/zip, ZSTD (tag 50000, the
  libtiff/GDAL extension). Predictor 2 (horizontal differencing, any integer
  width, 8–64-bit) and Predictor 3 (TechNote 3 floating-point, f32 and f64, as
  libtiff writes for float data) are undone.
- Multi-sample (e.g. RGB) data is split per sample plane in either
  interleaving: chunky (`PlanarConfiguration=1`, samples interleaved per pixel)
  and planar (`PlanarConfiguration=2`, each sample stored as its own whole
  plane — what `tifffile` writes for a `(3|4, H, W)` array and what libtiff
  writes as `PLANARCONFIG_SEPARATE`).

Because the display pipeline (and GPUs) work at ≤32 bits, **64-bit samples are
decoded through the same paths as 32-bit**: a 64-bit float is downcast to `f32`,
and a 64-bit integer is rescaled into the 0..=65535 display window exactly like a
32-bit integer. The full 64-bit values aren't widened past `f32` — more than
enough precision for display/contrast, but not a lossless numeric round-trip for
values beyond `f32`'s range. (The bytes themselves round-trip losslessly through
the *writer*; the down-conversion is only in the decode-to-display readers.)

### Also handled

- **BigTIFF** (magic 43, 64-bit offsets) reads exactly like classic TIFF —
  `TiffStack::flavor` says which one you got.
- **ColorMap (tag 320) as a display LUT**, in the two situations ImageJ writes
  one — the map becomes the channel's display LUT (`channel_display[0].lut`), the
  pixels decode unchanged, and the color is applied on display:
  - **4-/8-bit palette** images (PhotometricInterpretation=3): the pixel is a
    direct index into the map (`2^BitsPerSample` entries). `FrameInfo::is_palette()`
    flags these (the viewer maps index → entry through an identity contrast
    window); 4-bit indices are unpacked two-per-byte on decode.
  - **grayscale** images (PhotometricInterpretation=1) that ImageJ has colored
    with a Fire/Ice-style LUT: the 256-entry map is applied through the normal
    contrast window (`min=`/`max=`), not as an index.

  This is **sample-format agnostic** — it applies to unsigned, signed, and
  float grayscale alike (8/16/32/64-bit). 16-bit- and 8-bit-scaled ColorMaps are
  both handled.
- **CMYK (`PhotometricInterpretation=5`, "Separated")**, 8- and 16-bit
  unsigned, chunky or planar, under any supported codec.

  Separated files store *ink coverage*, not light: more ink means a darker
  pixel. `read_planes_rgb_u8` / `read_planes_rgb_u16` (and the single-component
  `read_plane_rgb_*`) convert the four plates into **three** RGB planes using
  `component = (1 - ink) x (1 - black)`, matching Pillow and libtiff. Gate on
  `FrameInfo::is_cmyk()`, which additionally requires `InkSet=1` — the tag is
  usually absent, and 1 is its spec default, but `InkSet=2` means some *other*
  set of inks (hi-fi printing adds orange/green/violet plates) that must not be
  read as C, M, Y, K.

  The conversion runs after the plane gather, so chunky/planar and every codec
  and predictor come out the same. **The raw readers are unaffected**:
  `read_planes_u8` and friends still return one plane per sample, so the four
  ink plates remain available unchanged. 32-bit and float Separated frames are
  deliberately excluded — the wide-sample readers auto-range each plane
  independently, which would rescale the four plates against four different
  ranges before combining them.

- **ImageJ's contiguous big-stack layout**: ImageJ writes its own >4 GiB
  stacks as a classic TIFF with a *single* IFD, `images=N` in the
  description, and the remaining frames appended as raw contiguous data.
  `open` detects this and expands it into N virtual frames (clamped to what
  the file actually contains), each on the zero-copy fast path.

### Not supported

Tiled TIFFs and pyramidal / mixed-size stacks (every frame must share frame 0's
geometry — this is enforced with a clear error at `open`). A 64-bit target is
assumed: offset arithmetic uses `usize`.

### Robustness against malformed input

Every input is untrusted, so for **any** bytes at all the only outcomes are `Ok`
or `Err`. Specifically, a malformed or hostile file must never panic, never
abort, and never allocate unboundedly — an oversized `vec![0u8; n]` calls
`abort()`, which no caller can catch, so it would take the host process down.

Header fields are attacker-controlled, and `width × height × samples_per_pixel ×
bytes_per_sample` is easy to inflate, so sizes are constrained on three sides
before anything is allocated:

- **Checked arithmetic.** `FrameInfo::decoded_len` is the single place the frame
  size is computed, and it cannot wrap.
- **Index-time validation.** `open` rejects empty images and any strip that
  falls outside the file, so every `FrameInfo` a caller can reach is already
  sane, and the strip totals below are bounded by the file.
- **Two size bounds at decode.** What the declared strips can structurally
  *cover* (`strips × rows_per_strip × row bytes`), and what they actually
  *supply* — exact for uncompressed data, with codec headroom for compressed.
  Both are needed: `rows_per_strip` is itself file-declared, so the structural
  bound alone is satisfiable by any frame claiming one enormous strip.
- **Fallible allocation.** What survives the bounds is allocated with
  `try_reserve`, so a genuinely large frame on a memory-starved machine returns
  an error instead of aborting.

This is enforced two ways. [`tests/malformed_robustness.rs`](tests/malformed_robustness.rs)
pins the known cases — truncation at every prefix, single-byte corruption at
every offset, a seeded mutation sweep, hostile metadata, and inflated
geometry — and runs in ordinary CI on every platform. [`fuzz/`](fuzz/) adds
`cargo-fuzz` targets over `from_bytes` and every decoder that explore for new
ones; see [`fuzz/README.md`](fuzz/README.md).

### Compared with other TIFF libraries

How the format coverage lines up against the readers/writers this crate is
[benchmarked](#benchmarks) against — the pure-Rust
[`tiff`](https://crates.io/crates/tiff) crate (image-rs), the C
[TinyTIFF](https://github.com/jkriege2/TinyTIFF), and
[libtiff](http://www.libtiff.org/) (the de-facto reference). `fast-tiff-lib` is a
*specialized* engine for lazily scrubbing scientific hyperstacks, not a
general-purpose TIFF library — the table is meant to show where that focus adds
reach (ImageJ metadata, lazy/zero-copy) and where it deliberately stops
(tiled):

| Feature | `fast-tiff-lib` | `tiff` (Rust) | TinyTIFF (C) | libtiff (C)|
| --- | :---: | :---: | :---: | :---: |
| 8 / 16 / 32-bit integer | ✓ | ✓ | ✓ ¹ | ✓ |
| 64-bit integer | ✓ | ✓ | ✓ ¹ | ✓ |
| Signed integer | ✓ | ✓ | ✗ ¹ | ✓ |
| 32-bit float | ✓ | ✓ | ✗ ¹ | ✓ |
| 64-bit float | ✓ | ✓ | ✗ ¹ | ✓ |
| LZW · PackBits · Deflate | ✓ | ✓ | ✗ | ✓ |
| ZSTD (tag 50000) | ✓ | read only ² | ✗ | ✓ ³ |
| Predictor 2 · 3 (float) | ✓ | ✓ | ✗ | ✓ |
| Chunky RGB (deinterleaved) | ✓ | ✓ | ✓ | ✓ |
| Planar RGB | ✓ | ✗ ⁴ | ✓ | ✓ |
| Tiled | ✗ | ✗ ⁴ | ✗ | ✓ |
| BigTIFF | ✓ | ✓ | ✗ | ✓ |
| ImageJ hyperstack metadata | ✓ | ✗ | ✗ | ✗ ⁵ |
| OME-TIFF metadata (OME-XML) | ✓ | ✗ | ✗ | ✗ ⁵ |
| Memory-mapped, lazy per-frame decode | ✓ | ✗ | ✗ | ✗ ⁶ |
| Zero-copy uncompressed frame | ✓ | ✗ | ✗ | ✗ |
| Streaming (append-a-frame) writer | ✓ | ✗ ⁷ | ✓ | ✓ |

<sub>✓ supported · ✗ not supported. Reflects each library's documented
capabilities at the time of writing; general-purpose libtiff/`tiff` evolve, so
check their current docs.</sub>

1. TinyTIFF reads/writes **uncompressed unsigned-integer** data only (8/16/32/64-bit) — no signed, no float, no compression.
2. `tiff` *decodes* ZSTD behind a cargo feature but does not *encode* it.
3. libtiff's ZSTD codec requires building against libzstd (added in libtiff 4.0.7).
4. `tiff` doesn't handle planar layout; tiled *reading* isn't listed in its README.
5. libtiff hands back the raw `ImageDescription` text but doesn't parse ImageJ/OME hyperstack semantics (channels/slices/LUTs/calibration). This crate reads a minimal but useful OME-XML subset (the `Pixels` core + per-`Channel` name/color); ROIs, instruments, and per-plane records are not modeled.
6. libtiff reads through its strip/tile API into owned buffers; memory-mapping is left to the caller.
7. `tiff`'s encoder writes a whole image per call rather than appending frames one at a time.

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
let pixels = read_frame_u16(&stack.data, frame, stack.byte_order, None)?;
assert_eq!(pixels.len(), (frame.width * frame.height) as usize);
# Ok::<(), anyhow::Error>(())
```

All fallible calls return `anyhow::Result`.

## Core API

### Opening a stack

Two entry points, both returning the same fully-indexed `TiffStack`:

```rust
// Memory-mapped from a path. Requires the `mmap` feature (on by default).
let stack = TiffStack::open("movie.tif")?;

// From bytes already in memory — no filesystem needed, so this is the way in
// on wasm, or for a stack that arrived over the network.
let bytes: Vec<u8> = std::fs::read("movie.tif")?;
let stack = TiffStack::from_bytes(bytes)?;
```

They share one parser, so a mapped and an in-memory stack can never disagree.
See [Feature flags](#feature-flags) for the trade-off between them.

```rust
pub struct TiffStack {
    pub data: Bytes,                // the file's bytes (mapped or owned)
    pub byte_order: ByteOrder,      // Little or Big (the file's endianness)
    pub frames: Vec<FrameInfo>,     // one entry per IFD/plane, in file order
    pub meta: StackMeta,            // normalized display metadata (ImageJ / OME / inferred)
}
```

#### Migrating from 0.9.6 and earlier

`TiffStack::mmap` is now `TiffStack::data`, typed [`Bytes`](#bytes) rather than
`memmap2::Mmap`. Since `Bytes` derefs to `&[u8]`, call sites that passed the
field to a decoder need only the rename:

```diff
- let pixels = read_frame_u16(&stack.mmap, frame, stack.byte_order, None)?;
+ let pixels = read_frame_u16(&stack.data, frame, stack.byte_order, None)?;
```

Code that named the type (`let m: &Mmap = &stack.mmap`) has to match on `Bytes`
or go through the deref. Nothing else moved, and `TiffStack::open` is unchanged
on a default build.

### `Bytes`

Where the file's bytes live. It derefs to `&[u8]`, so it's passed straight to
every decoder and you rarely name the type:

```rust
pub enum Bytes {
    Mapped(memmap2::Mmap),   // only with the `mmap` feature
    Owned(Vec<u8>),
}
```

The choice does **not** affect decoding — the zero-copy fast path borrows out of
either one. What differs is when you pay for the bytes: `Mapped` reads pages
lazily on first touch and lets the OS evict them again, so opening a multi-GB
stack is instant and resident memory tracks what you actually looked at.
`Owned` reads everything up front and keeps it.

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
    pub photometric: u16,              // 2 = RGB, 3 = palette, 5 = separated (CMYK)
    pub planar_config: u16,            // 1 = chunky, 2 = planar
    pub ink_set: u16,                  // tag 332; 1 = CMYK (the default), 2 = other inks
    pub strip_offsets: Vec<u64>,
    pub strip_byte_counts: Vec<u64>,
    pub rows_per_strip: u32,
}
```

`frame.is_rgb()` is true for an RGB frame whose samples are color components you
can split out; `frame.is_planar()` says which interleaving they're stored in.
The plane readers below handle both, so callers rarely need to check.
`frame.is_cmyk()` is true for a Separated frame the converting readers can
handle — see the CMYK note above for exactly what that requires.

### Decoding

All decoders take the file's bytes (`&stack.data`), the `FrameInfo`, and the file's
`ByteOrder`. They return owned data or a borrow of the mapping (`Cow`).

```rust
// 16-bit display path. 8-bit is upcast to 0..=65535; signed ints are offset
// into unsigned display space; 32- and 64-bit (int/float) are linearly rescaled
// into 0..=65535 using `float_range` (or the frame's own min/max when `None`).
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
// native-order f32; 32-bit int is cast, and 64-bit int/float is down-converted,
// to f32.
fn read_frame_f32(mmap, frame, order) -> Result<Cow<[f32]>>;
fn read_plane_f32(mmap, frame, order, plane: usize) -> Result<Vec<f32>>;

// All sample planes in ONE decompression pass — for chunky RGB this is ~3x
// cheaper than three read_plane_* calls on compressed data (each of which
// decompresses the whole frame again). One Vec per sample plane. With
// Predictor 2, the undo is fused into the per-plane gather (one pass).
fn read_planes_u16(mmap, frame, order, float_range) -> Result<Vec<Vec<u16>>>;
fn read_planes_u8(mmap, frame, order) -> Result<Vec<Vec<u8>>>;
fn read_planes_f32(mmap, frame, order) -> Result<Vec<Vec<f32>>>;

// CMYK only (`frame.is_cmyk()`; anything else is an error, never a silent
// misread). Gathers the four ink plates and returns THREE converted RGB
// planes, in the source's own width. The raw readers above are unchanged on
// the same frame -- they still give you the four plates.
fn read_planes_rgb_u8(mmap, frame, order) -> Result<Vec<Vec<u8>>>;
fn read_planes_rgb_u16(mmap, frame, order) -> Result<Vec<Vec<u16>>>;
fn read_plane_rgb_u8(mmap, frame, order, component: usize) -> Result<Vec<u8>>;
fn read_plane_rgb_u16(mmap, frame, order, component: usize) -> Result<Vec<u16>>;
```

Every reader above also has a **`*_into` variant** (`read_frame_u16_into(...,
&mut buf)`, `read_planes_u8_into(..., &mut planes)`, ...) that decodes into a
caller-provided buffer, reusing its allocation — for hot per-frame loops this
skips the allocation, the zero-fill, and fresh-page faults of a new `Vec`
every frame. For uncompressed predictor-free frames the `_into` paths convert
each strip **straight from the mapping into the output** (a plain memcpy in
the native-order case) with no intermediate assembly buffer at all; compressed
strips likewise decompress directly into their row ranges, single-pass.

`TiffStack::prefetch_frame(&frame)` is a performance hint that touches the
frame's mapped pages so a subsequent decode doesn't stall on page faults —
call it from a read-ahead thread for the *next* frame (first-touch faults are
cheap on Linux but cost real time on Windows).

```rust

// Actual min/max of a 32- or 64-bit frame's raw values (for auto-ranging the
// display); `None` for 8/16-bit frames (use their native integer min/max).
fn frame_float_minmax(mmap, frame, order) -> Result<Option<(f32, f32)>>;

// Eager counterparts: decode *every* frame at once, in parallel across frames,
// into owned buffers (one per frame). For consumers that need the whole stack
// resident in RAM rather than scrubbing it lazily (see "Design & prior art").
// `_u16` covers any depth (8-bit is widened, 32/64-bit rescaled); `_u8` is the
// half-memory raw-byte loader for unsigned 8-bit stacks; `_f32` is raw float.
fn preload_frames_u16(stack: &TiffStack, float_range: Option<(f32, f32)>) -> Result<Vec<Vec<u16>>>;
fn preload_frames_u8(stack: &TiffStack) -> Result<Vec<Vec<u8>>>;   // unsigned 8-bit stacks
fn preload_frames_f32(stack: &TiffStack) -> Result<Vec<Vec<f32>>>;
```

To index a plane in a hyperstack (ImageJ's default `xyczt` order, channel
fastest), the position in `frames` is `frame_index * slices * channels + channel`.

### Parallel decoding (runtime hint)

Decoding can use [rayon](https://crates.io/crates/rayon) to split a frame's
strip decompression and 32-/64-bit per-pixel conversion across cores. It's controlled by a
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

- **Samples:** `SampleType::{U8, I8, U16, I16, U32, I32, F32, U64, I64, F64}` —
  TIFF unsigned/signed/float at 8/16/32/64 bits.
- **Layout:** grayscale planes (`samples_per_pixel(1)`, default), RGB
  (`samples_per_pixel(3)`, tagged photometric=RGB), or CMYK
  (`samples_per_pixel(4).cmyk(true)`, tagged photometric=Separated with
  `InkSet=1` — the samples are written through as ink coverage, in C, M, Y, K
  order, with no conversion; `cmyk()` is rejected for anything the reader would
  not take back, i.e. fewer than 4 samples or a non-unsigned/non-8-or-16-bit
  sample type). Any of these interleaved per pixel by
  default or as separate sample planes with `planar(true)`
  (`PlanarConfiguration=2`). **The frame buffer you pass matches the file
  layout** — chunky is `RGB RGB RGB…`, planar is `RRR… GGG… BBB…` — and both are
  the same length, so a mismatch is not caught by the length check.
- **Compression:** `None` (default), `Lzw`, `PackBits`, `Deflate`, `Zstd`
  (tag 50000) — plus optional `predictor(true)`, which usually shrinks
  LZW/Deflate/ZSTD output on continuous-tone data: integers get TIFF
  Predictor 2 (any width, 8–64-bit), `F32`/`F64` get Predictor 3 (the TechNote 3
  floating-point predictor libtiff uses). `compression_level(n)` sets the
  effort for the codecs that have one (Deflate 0–9, ZSTD 1–22); when left unset
  the lib applies its own defaults — `DEFAULT_DEFLATE_LEVEL` (6) and
  `DEFAULT_ZSTD_LEVEL` (3) — so a codec choice alone always compresses sensibly.
- **Strips:** uncompressed frames are one strip; compressed frames default to
  ~256 KiB strips. A big frame's strips compress in parallel under the same
  `set_parallel_decode` hint + size floor that governs decoding — one
  threading switch for all CPU-heavy pixel work — and decompress in parallel
  on the way back in. Override with `rows_per_strip(n)`.
- **Metadata:** fill one neutral `metadata(StackMetaWrite...)` builder —
  channels/slices (time frames are derived from the plane count at `finish()`),
  display mode, `fps`, frame interval, unit, `min`/`max` display range, linear
  calibration, Z `spacing`, playback `loop`, physical `pixel_size` (written to
  the resolution tags), per-`channel(name, color)` (OME) or
  per-`channel_lut(lut)` (ImageJ — a 256-entry color LUT written to the binary
  `IJMetadata` block, so a colorized 8-/16-/32-bit stack keeps its colors),
  plus `extra(key, value)` —
  then pick the dialect with `metadata_format(MetadataFormat::ImageJ | Ome)`
  (ImageJ by default). The same input serializes to either an ImageJ
  `key=value` block or an OME-XML document. Or `description(text)` for a fully
  verbatim `ImageDescription`. Either way the reader hands the whole tag 270
  text back via `TiffStack::description`, with the parsed view in `meta`.

```rust
use fast_tiff_lib::{Compression, DisplayMode, MetadataFormat, SampleType, StackMetaWrite, WriterOptions};

// A 2-channel composite time series, Deflate-compressed, written as OME-TIFF.
// Drop the `.metadata_format(..)` line to write the same metadata as ImageJ.
let opts = WriterOptions::new(1024, 1024, SampleType::U16)
    .compression(Compression::Deflate)
    .predictor(true)
    .metadata_format(MetadataFormat::Ome)
    .metadata(
        StackMetaWrite::new(2, 1)
            .mode(DisplayMode::Composite)
            .fps(20.0)
            .pixel_size(0.1, 0.1)
            .channel("DAPI", [0, 0, 255])
            .channel("GFP", [0, 255, 0]),
    );
```

Typed appenders `write_frame_u8` / `write_frame_u16` / `write_frame_f32` /
`write_frame_f64` mirror the readers; `write_frame_bytes` takes raw
little-endian sample bytes and covers the other sample types
(`bytemuck::cast_slice` on any `&[i16]`, `&[u32]`, `&[i64]`, ... produces them
for free on little-endian hosts). Planes go in ImageJ's `xyczt` order, matching
the reader's indexing.

### Guarantees & limits

- Output is always **little-endian**, IFD entries in spec-required ascending
  tag order, ASCII fields NUL-terminated, IFDs word-aligned, and samples
  beyond the photometric base declared in `ExtraSamples` — standard TIFF6
  any reader accepts (libtiff, ImageJ, this crate).
- The uncompressed default (single strip, native order) is **exactly this
  reader's zero-copy path**: `read_frame_u16`/`_u8`/`_f32` borrow straight
  from the mapping, no decode pass. Verified by round-trip tests.
- Pixel data streams to the sink as frames arrive; `finish()` buffers only
  the IFD tables (~150 bytes/frame) and seeks once, to patch the header.
- **BigTIFF is automatic:** files are classic TIFF while they fit under the
  4 GiB offset limit and upgrade to BigTIFF (magic 43, 64-bit offsets) when
  they outgrow it — decided at `finish()`, with no re-writing of pixel data
  (a 16-byte header slot is reserved up front; classic files carry 8 legal
  pad bytes in it). `bigtiff(true)` forces BigTIFF for smaller files. Tiles are
  not written — the same envelope the reader accepts. `planar(true)` writes
  `PlanarConfiguration=2` (separate sample planes, strips split per plane);
  chunky is the default and writes no tag at all, which TIFF6 defines as
  chunky. Physical `pixel_size` is written to the XResolution/YResolution tags
  (in either dialect); the unit name travels in the metadata text (ImageJ) or
  OME-XML `PhysicalSize` (OME). ImageJ output writes the binary `IJMetadata` LUT
  block (tags 50839/50838, the real `IJIJ`/`JIJI`-magic format ImageJ and
  tifffile use) only when `channel_lut` LUTs are supplied — so a colorized
  16/32-bit stack round-trips its colors; contrast ranges otherwise go in the
  description. OME output carries per-channel colors as `Channel/@Color`.

### Metadata (`StackMeta`)

```rust
#[non_exhaustive] // fields are added over time; construct it only via `open`
pub struct StackMeta {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub mode: DisplayMode,                  // Grayscale | Composite | Color
    pub unit: Option<String>,               // calibration unit (\uXXXX escapes decoded, e.g. µm)
    pub frame_interval_s: Option<f64>,
    pub channel_display: Vec<ChannelDisplay>,   // per-channel LUT + range
    pub calibration: Option<(f64, f64)>,        // linear (c0, c1): value = c0 + c1*raw
    pub fps: Option<f64>,
    pub spacing: Option<f64>,               // Z-step between slices (spacing=)
    pub loop_playback: Option<bool>,        // playback looping (loop=)
    pub pixel_width: Option<f64>,           // x pixel size in `unit`s (OME PhysicalSize, else XResolution)
    pub pixel_height: Option<f64>,          // y pixel size in `unit`s (OME PhysicalSize, else YResolution)
    pub has_explicit_luts: bool,            // file supplied real (colored) per-channel LUTs/colors
    pub source_format: MetadataFormat,      // ImageJ | Ome | None (which dialect was parsed)
}
```

`has_explicit_luts` reports whether the file carried genuine colored LUTs (the
ImageJ `IJMetadata` block, or OME channel `Color`s) — which take priority over
the mode-derived default, including over grayscale — so a consumer knows not to
override the file's own channel colors. `source_format` names the dialect the
values came from (`None` if the description matched no known dialect and the
dimensions were inferred from the IFD count).

This is the *normalized view* — the same shape regardless of which dialect the
file used. The raw `ImageDescription` (tag 270) text is also exposed verbatim —
whatever the writer put there, ImageJ / OME / neither:

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

- `meta.calibrate(raw) -> f64` — apply the linear calibration (`c0 + c1·raw`),
  or return the value unchanged when the file has none.
- `meta.voxel_scale() -> [f32; 3]` — physical x:y:z voxel scale from the pixel
  calibration and Z `spacing` (the raw calibrated values, all in `unit`;
  `1:1:1` when uncalibrated) — for anisotropy-correct 3D display.
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
BigTIFF (including big-endian BigTIFF), ImageJ hyperstack metadata, and an
unsupported-tiled error case. Regenerate with
`tests/fixtures/generate_fixtures.py`.

## Benchmarks

[`bench/`](https://github.com/konung-yaropolk/FastTIFF/tree/main/fast-tiff-lib/bench)
holds a reproducible benchmark comparing per-frame read speed against the
pure-Rust [`tiff`](https://crates.io/crates/tiff) crate, the C
[TinyTIFF](https://github.com/jkriege2/TinyTIFF) reader (vendored), optionally
system libtiff, and a raw `fread` floor — across every sample format, codec,
predictor, strip layout, BigTIFF, and several frame counts, on stacks written
by this crate's own writer. Numbers below: Ryzen 7 5800X, Windows 10,
rustc 1.96.

- **Scrubbing — the design target.** After the up-front IFD index, reading a
  frame of a **1,000,000-frame** stack takes **0.08 µs**, vs 1.5–2.4 µs for
  every other reader (≈20–30×). Even *including* its ~0.4 s open/index cost,
  the full million-frame pass finishes in ~0.6 s — fastest in the field
  (tiff-rs 2.2 s, TinyTIFF 2.5 s, raw `fread` 1.6 s).
- **Coverage matrix** (45 runs): geometric-mean relative speed **2.0×** of
  each run's fastest reader — ahead of tiff-rs (2.2×). TinyTIFF leads the
  uncompressed-only subset it supports (see the caveat below).
- **Outright wins:** PackBits (46 µs vs tiff-rs's 4 200 µs per frame, ≈90×)
  and compressed RGB with predictor (the fused `read_planes_*` path).
- **Batch loading:** `preload_frames_*` (rayon, across frames) wins 17 of its
  39 runs — the fastest way to slurp a compressed stack into RAM.
- **Writer throughput:** ≈2.7 GB/s uncompressed, 1.0 GB/s PackBits,
  460 MB/s ZSTD, 150 MB/s Deflate, 115 MB/s LZW.

![Benchmark summary](https://raw.githubusercontent.com/konung-yaropolk/FastTIFF/main/fast-tiff-lib/bench/bench_summary.png)

![Frame-count sweep](https://raw.githubusercontent.com/konung-yaropolk/FastTIFF/main/fast-tiff-lib/bench/graphs/sweep_combined.png)

![All tests](https://raw.githubusercontent.com/konung-yaropolk/FastTIFF/main/fast-tiff-lib/bench/graphs/all_tests.png)

One honest caveat: the benchmark forces every reader to produce owned buffers
and reads each frame exactly once, which bills mmap's one-time page-fault
cost (expensive on Windows, minor on Linux) entirely to this crate while
giving its zero-copy design no credit — the uncompressed single-pass rows are
a *lower bound*. Methodology, machine details, and how to run are in
[`bench/README.md`](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-lib/bench/README.md).

## Feature flags

| Feature | Default | What it does |
|---|---|---|
| `mmap` | ✅ | Memory-mapped file access, and with it `TiffStack::open`. Pulls in `memmap2`. |
| `codec-zstd` | ✅ | The ZSTD codec (compression 50000). Pulls in `zstd`, which builds a C library via `zstd-sys`. |

Both are on by default, so nothing changes for an ordinary native build.

Turning one off narrows what the crate can do, and says so clearly rather than
failing obscurely:

- Without `mmap`, `TiffStack::open` doesn't exist; use `TiffStack::from_bytes`.
- Without `codec-zstd`, a ZSTD-compressed frame returns an error naming the
  missing feature instead of decoding. Every other codec is pure Rust and
  unaffected.

### wasm

`--no-default-features` is exactly the shape needed for
`wasm32-unknown-unknown`, where there's no filesystem to map and no C toolchain
for `zstd-sys`:

```bash
cargo build -p fast-tiff-lib --target wasm32-unknown-unknown --no-default-features
```

`rayon` still compiles for wasm and stays in the dependency tree, but no
parallel path is ever entered there — `should_parallelize` returns `false` on
`wasm32`, since the target has no threads to fork onto.

Both configurations are covered in CI, and `tests/from_bytes.rs` deliberately
runs in each so the filesystem-free path can't silently rot.

## Dependencies

`weezl` (LZW), `flate2` (Deflate), `anyhow`, `bytemuck`, `rayon`, plus two
optional ones: `memmap2` (feature `mmap`) and `zstd` (feature `codec-zstd`,
which builds the C library via `zstd-sys`).

## License

Mozilla Public License 2.0. See the [LICENSE](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-lib/LICENSE).

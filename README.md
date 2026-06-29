# FastTIFF - a lightning-fast multi-frame TIFF viewer with ImageJ-compatible rendering

[![Release](https://img.shields.io/github/v/release/konung-yaropolk/FastTIFF?label=release)](https://github.com/konung-yaropolk/FastTIFF/releases)
[![License](https://img.shields.io/github/license/konung-yaropolk/FastTIFF/FastTIFF)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/release.yml?label=build)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/ci.yml)

A fast multi-frame TIFF stack viewer for huge ImageJ hyperstacks: a horizontal
scrubber instead of ImageJ's slice slider, GPU-side LUT/contrast rendering,
and (for the common uncompressed case) zero CPU-side image processing per
frame change.

Open a stack via the "Open TIFF..." button or by dragging a `.tif`/`.tiff`
file onto the window. Scrub with the bottom slider, the mouse wheel while
hovering over the image (one frame per notch; hold **Shift** for fast
continuous scrolling), or the left/right arrow keys.

Downloads here:
https://github.com/konung-yaropolk/FastTIFF/releases

<img width="1137" height="923" alt="Untitled" src="https://github.com/user-attachments/assets/e3fe4619-bd53-4b68-bf7f-019231cf20a6" />

## Build & run

```sh
cargo run --release
```

### Renderer (glow vs wgpu)

The GPU backend is chosen at compile time. 

**glow** (OpenGL) is the default:
```sh
cargo run --release
```
**wgpu** (DX12/Vulkan/Metal) is opt-in:
```sh
cargo run --release --no-default-features --features renderer-wgpu
```

glow is the default because wgpu pegs a CPU core while idle on some Windows 10
machines; wgpu may be preferable on macOS (Metal). Only the selected backend is
compiled in — the other's dependencies are excluded entirely.

## Why it's fast

ImageJ re-renders each slice on the CPU (Java `BufferedImage`/AWT) every time
you move the slider. This viewer instead:

- Memory-maps the TIFF file. For uncompressed strips (ImageJ's default when
  saving raw stacks), reading a frame is a direct reinterpret of file bytes
  already sitting in mapped memory - no decode step, no allocation.
- Uploads the raw 16-bit samples straight to the GPU as a texture.
- Does window/level (contrast) and LUT color mapping in a fragment shader,
  per pixel, on the GPU. The CPU never touches pixel values.


## Project layout

- **`fast-tiff-lib/`** - pure parsing/decoding library, no GUI or GPU
  dependencies. IFD-chain walking, ImageJ metadata parsing, strip decoding
  (uncompressed fast path + LZW/Deflate/PackBits + predictor undo). Has a
  real test suite (`cargo test -p fast-tiff-lib`) that builds synthetic
  multi-frame TIFFs in memory and round-trips them through the whole
  pipeline - this is the part most worth trusting blind, since it's
  actually verified.
- **`FastTIFF/`** — the GUI binary: eframe/egui for the window and controls,
  a custom GPU render pipeline for the image itself, with interchangeable
  glow (OpenGL) and wgpu backends selected at build time (see Renderer above).

## What v1 covers

- Multi-frame grayscale, multi-channel composite, and chunky RGB TIFFs in
  8-bit, 16-bit, and 32-bit (integer or float) - 32-bit and float data is
  auto-ranged into the display, RGB is deinterleaved into R/G/B planes.
- ImageJ `ImageDescription` parsing (channels/slices/frames, mode,
  min/max, unit, frame interval, linear calibration `c0`/`c1`, `fps`) —
  solid, well-documented format.
- Composite-channel colors from a standard cycling palette; contrast from
  `min=`/`max=` in `ImageDescription` (or auto-contrast from the data).
- Signed-integer images offset into ImageJ's unsigned display space, so a
  signed file and the equivalent unsigned+calibration file render the same.
- Horizontal frame scrubber + mouse-wheel scrubbing + arrow keys, plus a
  play button for looped playback (uses `fps=` from metadata, else 30 fps).
- Per-channel enable/disable + a two-handle contrast range slider; the
  values shown are calibrated (`c0 + c1·raw`) when the file has calibration.
- Z-slice selector when `slices > 1` (the scrubber itself always drives
  the time/frame axis).

## What it doesn't do (intentionally out of scope for a "viewer")

ROIs, measurements, image processing, saving/exporting, zoom/pan. All
straightforward to add later on top of this structure if you want them —
the render pipeline already separates "decode" from "display" cleanly.

## Known caveat: plane ordering assumption

For multi-channel/multi-slice stacks, the formula mapping (frame, slice,
channel) to a position in the IFD chain assumes ImageJ's default `xyczt`
plane order (channel varies fastest, then Z, then T) - see
`ifd_index()` in `FastTIFF/src/app.rs`. This is what ImageJ's TIFF writer
uses by default. If a particular file was produced with reordered planes,
this is the one-line formula to change.

## Tuning knobs if you need to go further

- A small LRU texture cache for *compressed* (LZW) stacks would help if
  you have large compressed movies - the uncompressed path doesn't need
  one (it's already near the theoretical floor), but decode cost dominates
  for compressed strips. The frame-access layer (`read_frame_u16`) is
  already isolated cleanly enough to slot a cache in front of without
  restructuring anything.
- Background/threaded loading for opening extremely large stacks (the IFD
  walk itself is fast - pure memory access - but hasn't been measured
  against anything with hundreds of thousands of frames).

## To Do:

- Done: Fix bug with skewed first frame when loading some tifs through command
- Done: add label with version, and gpu backend info
- Done: add suppport to open multiple files if passed in command - open needed number of processes and open eah image in it
- Done: Hide slider for single-frame tiffs
- Done: add label in channels slider to hold shift to synchronize adjustments
- Add zstd compression support
- Fix viewing >6Gb tifs (no frames change when scrolling)
- Add bigtiff support
- Port to linux and mac
- Add windows installer with files association
- publish fast-tiff-lib as FastTiffLib in to crates.io




- Solved: issue with performance in optimized version - 16 bit compressed tiff playback holds 12% cpu spreaded by multiple cores, but unoptimized - 4-5% which is ~50% single core load
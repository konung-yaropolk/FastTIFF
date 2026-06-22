# FastTIFF

A fast multi-frame TIFF stack viewer for huge ImageJ hyperstacks: a horizontal
scrubber instead of ImageJ's slice slider, GPU-side LUT/contrast rendering,
and (for the common uncompressed case) zero CPU-side image processing per
frame change.

## Why it's fast

ImageJ re-renders each slice on the CPU (Java `BufferedImage`/AWT) every time
you move the slider. This viewer instead:

- Memory-maps the TIFF file. For uncompressed strips (ImageJ's default when
  saving raw stacks), reading a frame is a direct reinterpret of file bytes
  already sitting in mapped memory — no decode step, no allocation.
- Uploads the raw 16-bit samples straight to the GPU as a texture.
- Does window/level (contrast) and LUT color mapping in a WGSL fragment
  shader, per pixel, on the GPU. The CPU never touches pixel values.

So scrubbing cost is dominated by "mmap read + texture upload," not by any
per-pixel CPU work — that's the entire reason this is faster than ImageJ for
large stacks.

## Build & run

Requires a current Rust toolchain (this was developed against the latest
stable as of June 2026 — eframe/egui/egui_wgpu 0.34, wgpu 29). Get one via
[rustup](https://rustup.rs) if you don't have it.

```sh
cargo run --release
```

First build will take a couple of minutes (wgpu has a large dependency
tree). Subsequent builds are incremental.

Open a stack via the "Open TIFF..." button or by dragging a `.tif`/`.tiff`
file onto the window. Scrub with the bottom slider, the mouse wheel while
hovering over the image, or the left/right arrow keys.

## Project layout

- **`tiff_core/`** — pure parsing/decoding library, no GUI or GPU
  dependencies. IFD-chain walking, ImageJ metadata parsing, strip decoding
  (uncompressed fast path + LZW/Deflate/PackBits + predictor undo). Has a
  real test suite (`cargo test -p tiff_core`) that builds synthetic
  multi-frame TIFFs in memory and round-trips them through the whole
  pipeline — this is the part most worth trusting blind, since it's
  actually verified.
- **`FastTIFF/`** — the GUI binary: eframe/egui for the window and controls,
  a custom wgpu render pipeline (via `egui_wgpu::CallbackTrait`) for the
  image itself.

## What v1 covers

- Multi-frame grayscale, multi-channel composite, and chunky RGB TIFFs in
  8-bit, 16-bit, and 32-bit (integer or float) — 32-bit and float data is
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

## Deprecated: the binary IJMetadata block (tags 50838/50839)

ImageJ's `ImageDescription` text block (channels/slices/frames/min/max) is
well documented and the parser for it is solid. The binary `IJMetadata` /
`IJMetadataByteCounts` tags (which carry per-channel custom LUTs and
display ranges for composite stacks) are **not officially documented**, and
in practice two otherwise-identical files could render differently purely
because of inconsistencies in this block.

These tags are therefore **no longer read**. All display metadata comes
from `ImageDescription` alone: composite-channel colors use a standard
cycling palette, and contrast uses `min=`/`max=` from the description (or
auto-contrast from the data when that's absent). The former best-effort
binary parser was removed — see git history if it ever needs reviving.

## Known caveat: plane ordering assumption

For multi-channel/multi-slice stacks, the formula mapping (frame, slice,
channel) to a position in the IFD chain assumes ImageJ's default `xyczt`
plane order (channel varies fastest, then Z, then T) — see
`ifd_index()` in `FastTIFF/src/app.rs`. This is what ImageJ's TIFF writer
uses by default. If a particular file was produced with reordered planes,
this is the one-line formula to change.

## Tuning knobs if you need to go further

- `MAX_CHANNELS` in `FastTIFF/src/render/pipeline.rs` (currently 4) — bump
  it, the bind-group/shader pattern extends trivially.
- A small LRU texture cache for *compressed* (LZW) stacks would help if
  you have large compressed movies — the uncompressed path doesn't need
  one (it's already near the theoretical floor), but decode cost dominates
  for compressed strips. The frame-access layer (`read_frame_u16`) is
  already isolated cleanly enough to slot a cache in front of without
  restructuring anything.
- Background/threaded loading for opening extremely large stacks (the IFD
  walk itself is fast — pure memory access — but hasn't been measured
  against anything with hundreds of thousands of frames).

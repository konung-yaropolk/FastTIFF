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

- Multi-frame grayscale and multi-channel composite TIFFs, 8/16/32-bit.
- ImageJ `ImageDescription` parsing (channels/slices/frames, mode,
  min/max, unit, frame interval) — solid, well-documented format.
- Per-channel LUT and display-range extraction from the `IJMetadata` block
  — best-effort (see caveat below).
- Horizontal frame scrubber + mouse-wheel scrubbing + arrow keys.
- Per-channel enable/disable + manual contrast (min/max) override.
- Z-slice selector when `slices > 1` (the scrubber itself always drives
  the time/frame axis).

## What it doesn't do (intentionally out of scope for a "viewer")

ROIs, measurements, image processing, saving/exporting, zoom/pan. All
straightforward to add later on top of this structure if you want them —
the render pipeline already separates "decode" from "display" cleanly.

## Known caveat: IJMetadata LUT parsing is best-effort

ImageJ's `ImageDescription` text block (channels/slices/frames/min/max) is
well documented and the parser for it is solid. The binary `IJMetadata` /
`IJMetadataByteCounts` tags (which carry per-channel custom LUTs and
display ranges for composite stacks) are **not officially documented** —
`tiff_core/src/ij_metadata.rs` reconstructs the format from known reader
implementations, with two defensive properties:

1. It tries both header endiannesses and bails out to defaults (grayscale,
   auto-contrast from the first frame) on any structural inconsistency,
   rather than risk silently misreading the directory.
2. If it's wrong on one of your real files, the failure mode is "default
   colors instead of your custom LUT" — not a crash, not corrupted pixels.

This only matters for **multi-channel composite** stacks with custom
per-channel LUTs/ranges saved in ImageJ. Single-channel grayscale (the
common case for calcium imaging time series) never touches this code path
at all — it just uses `min=`/`max=` from `ImageDescription`, or
auto-contrast from the first frame if that's absent too.

If composite-mode colors look wrong on a real file, that's the function to
look at (`try_parse_ij_blocks`) — happy to fix it against an actual
example if you hit this.

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

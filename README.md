# FastTIFF - a lightning-fast multi-frame TIFF viewer with ImageJ-compatible rendering

A fast multi-frame TIFF stack viewer for huge ImageJ hyperstacks: a horizontal
scrubber instead of ImageJ's slice slider, GPU-side LUT/contrast rendering,
and (for the common uncompressed case) zero CPU-side image processing per
frame change.

Open a stack via the "Open TIFF..." button or by dragging a `.tif`/`.tiff`
file onto the window. Scrub with the bottom slider, the mouse wheel while
hovering over the image, or the left/right arrow keys.

<img width="1137" height="923" alt="Untitled" src="https://github.com/user-attachments/assets/e3fe4619-bd53-4b68-bf7f-019231cf20a6" />

## Build & run

```sh
cargo run --release
```

## Why it's fast

ImageJ re-renders each slice on the CPU (Java `BufferedImage`/AWT) every time
you move the slider. This viewer instead:

- Memory-maps the TIFF file. For uncompressed strips (ImageJ's default when
  saving raw stacks), reading a frame is a direct reinterpret of file bytes
  already sitting in mapped memory - no decode step, no allocation.
- Uploads the raw 16-bit samples straight to the GPU as a texture.
- Does window/level (contrast) and LUT color mapping in a WGSL fragment
  shader, per pixel, on the GPU. The CPU never touches pixel values.


## Project layout

- **`tiff_core/`** - pure parsing/decoding library, no GUI or GPU
  dependencies. IFD-chain walking, ImageJ metadata parsing, strip decoding
  (uncompressed fast path + LZW/Deflate/PackBits + predictor undo). Has a
  real test suite (`cargo test -p tiff_core`) that builds synthetic
  multi-frame TIFFs in memory and round-trips them through the whole
  pipeline - this is the part most worth trusting blind, since it's
  actually verified.
- **`FastTIFF/`** — the GUI binary: eframe/egui for the window and controls,
  a custom wgpu render pipeline (via `egui_wgpu::CallbackTrait`) for the
  image itself.

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

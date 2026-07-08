# FastTIFF - a lightning-fast multi-frame 2D- and 3D-viewer with ImageJ-compatible GPU-rendering

[![Release](https://img.shields.io/github/v/release/konung-yaropolk/FastTIFF?label=release)](https://github.com/konung-yaropolk/FastTIFF/releases)
[![License](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green)](LICENSE)
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

<img width="320" height="404" alt="2D" src="https://github.com/user-attachments/assets/5ba646bc-f451-4872-978b-bc0d5cf8b056" />  

<p align="center">
  <video src="[https://github.com](https://github.com/user-attachments/assets/5f338459-6fbd-4ea2-8fb8-7e1a31b7e045)" autoplay loop muted playsinline width="100%"></video>
</p>

<img width="1137" height="923" alt="Untitled" src="https://github.com/user-attachments/assets/e3fe4619-bd53-4b68-bf7f-019231cf20a6" />

## Build & run

```sh
cargo run --release
```

### Renderer (wgpu vs glow)

The GPU backend is chosen at compile time.

**wgpu** (DX12/Vulkan/Metal) is the default:
```sh
cargo run --release
```
**glow** (OpenGL) is opt-in:
```sh
cargo run --release --no-default-features --features renderer-glow
```

wgpu is the default: it's the more actively developed backend and preferable on
macOS (Metal, since OpenGL is deprecated there). glow is the portable fallback —
it links only OpenGL (near-universal on Linux) and avoids a Windows 10 idle-CPU
spin that wgpu triggers on some machines. Only the selected backend is compiled
in — the other's dependencies are excluded entirely.

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
- Zoom (Ctrl+scroll) and pan (drag) of the 2D image, with the window sized
  to fit on open.

## 3D volume view

A **2D / 3D** toggle in the top toolbar switches the stack between the movie
view and a GPU-ray-marched 3D volume, built from every frame and composited
with the same per-channel LUTs and contrast as the 2D view.

- Two rendering modes: **Max intensity** (MIP) and a translucent **Volume**
  mode modelled on ImageJ's 3D Viewer (emission–absorption alpha compositing,
  with a density control).
- **Navigation styles** — CAD, Blender, Maya, and a first-person **WASD Fly** —
  selectable in the render-settings window (⚙). Orbit modes rotate around the
  point where the view center enters the volume; the wheel zooms and WASD moves
  in every mode.
- **Interpolation**: none (nearest), trilinear, or tricubic B-spline.
- **Voxel scale** (x:y:z) seeded from the file's pixel calibration
  (XResolution/YResolution) and Z `spacing`, editable in the settings window.
- **4D**: for channels+Z+time stacks, playing the movie animates the volume
  through time.
- Runs on both the glow and wgpu backends.

## What it doesn't do (intentionally out of scope for a "viewer")

ROIs, measurements, image processing, saving/exporting. All straightforward to
add later on top of this structure if you want them — the render pipeline
already separates "decode" from "display" cleanly.

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
- Done: add suppport to open multiple files if passed in command - open needed number of processes and open each image in it
- Done: Hide slider for single-frame tiffs
- Done: add label in channels slider to hold shift to synchronize adjustments
- Done: publish fast-tiff-lib as FastTiffLib in to crates.io
- Done: added read_plane_u8 to lib
- Done: optimization 8bit rgb halved in occupied memory
- Done: change fast scroll to 10% of movie length instead of fixed frames number  
- Done: Port to linux
- Done: Add zstd compression support
- Done: Add tiff write support
- Done: Fix viewing >6Gb tifs (no frames change when scrolling)
- Done: Add bigtiff support
- Done: make inactive decode mode for when it is actual unneeded, make single mode default
- Solved: issue with performance in optimized version - 16 bit compressed tiff playback holds 12% cpu spreaded by multiple cores, but unoptimized - 4-5% which is ~50% single core load
- Done: 2D zoom (Ctrl+scroll) and pan
- Done: 3D volume view (MIP + ImageJ-style alpha), navigation modes, interpolation, 4D playback
- Done: 3D volume view on the wgpu backend (was blank on Windows 10)
- Done: move to wgpu default
- Done: Set default compression rates on write in lib
- Done: add shift and space keys navigation in CAD and Maya modes
- Done: change mouse wheel zoom logic - outside of the box like zoom, inside the box - linear like in spectator mode
- Done: add orbiting mechanism in to spectator mode by pressing right mouse button
- Done: add right mouse button camera angle change as in spectator mode 
- Done: add color selector for grayscale images applying for both 2d and 3d
- Done: add different colormaps to the selector like magma, plasma, viridis, turbo etc
- Done: add adjustable WASD and mouse scroll speed input into 3d settings window in navigation section


- Port to mac and publish at Brew
- Add windows installer with files association




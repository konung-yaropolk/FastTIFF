# fast-tiff-viewer

[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-viewer/LICENSE)

The frontend-agnostic viewer core for multi-frame scientific TIFF stacks:
everything between "a TIFF on disk" and "pixels on the GPU", with no GUI toolkit
in sight.

```text
  fast-tiff-lib     file I/O, IFD index, decode, metadata
        │
  scivis-render     GPU pipelines, textures, ray-marching
        │
  fast-tiff-viewer  ← this crate: document model, channel settings,
        │             c/z/t interpretation, 2D view, 3D camera,
        │             decode→GPU sync
  frontend          egui desktop app, or a wasm/JS web UI
```

Split out of [FastTIFF](https://github.com/konung-yaropolk/FastTIFF) so a second
frontend doesn't have to reinvent any of it. A frontend owns its windows,
widgets and input; it borrows everything else from here.

## What it does

- **Opens a stack and derives its whole display model** — resolved channel / Z /
  time roles (including the "metadata says `channels=100`" case), RGB plane
  deinterleaving, palette-image handling, per-channel contrast windows seeded
  from metadata or the actual data range, and the file's own LUT.
- **Maps display channels to file data.** Which IFD and sample plane a given
  channel of a given frame lives in — the thing that differs between RGB,
  hyperstacks and 4D stacks, and the easiest thing to get subtly wrong.
- **Runs the per-frame GPU sync**: allocate textures for the frame size and
  channel layout, upload LUTs and this frame's pixels (or the whole volume in
  3D), rescale each contrast window into the units its texture actually holds,
  and push the uniforms.
- **Assembles 3D volumes**, subsampled to fit the GPU's 3D-texture limit and a
  memory budget, parallel across output slices, on a background thread.
- **Reads ahead** while playback keeps up: page-touch for frames the view
  reads straight from the file (an uncompressed plane in the host's byte
  order, in one strip or many laid end to end, where decoding is zero-copy),
  decode-ahead for every other — into the buffers of frames already shown,
  so a steady movie allocates no frame-sized buffer, as the frame decoded on
  the interface thread does not either.
- **Keeps the 2D view** — the zoom ladder and the fit that joins it, pan,
  the zoom and swipe glides, autoscroll, wheel travel — as plain arithmetic in
  logical points (`view2d`), and works out where that puts the image and the
  part of it `sync` samples.
- **Drives the 3D camera** — four navigation styles, orbit/fly conversion,
  ray/box re-pivoting — as plain `f32` math over `(dx, dy)` deltas, with the
  user's navigation preferences (speeds, the resolution while moving, the
  coordinate box) beside it.
- **Keeps the playback clock**, advancing by real elapsed time so a movie runs at
  the file's fps regardless of render cadence, and latching parallel decode when
  it starts dropping frames.
- **Hosts plugins** (`plugins`): the registry, the host a plugin sees, the C
  boundary every plugin crosses, and on the desktop the loader for plugin
  libraries. It names no plugin itself. The ones an application compiles in
  are handed to it at start-up — FastTIFF's are the
  [`fasttiff-builtin-plugins`](../fasttiff-builtin-plugins) crate:

  ```rust,ignore
  let registry = Registry::with_installed(fasttiff_builtin_plugins::register);
  ```

  The registry implements `fasttiff_plugin::Registrar`, and installs every
  plugin type it is handed there through the C ABI, as a built-in.
- **Models the point counter** (`point_counter`): points clicked onto the
  picture in named groups, with undo, exported as ImageJ point ROIs. A tool
  rather than a plugin, until the plugin contract has a kind of plugin for
  one.

## Usage

```rust,ignore
use fast_tiff_viewer::view2d::{ImagePx, LogicalRect, LogicalVec, View2d};
use fast_tiff_viewer::{FrameInputs, Viewer};

let mut viewer = Viewer::new();
viewer.open(path)?;
let mut view = View2d::default();
let mut inputs = FrameInputs::default();

// each frame:
if viewer.playback.playing() {
    viewer.tick_playback(now_seconds);
}
// Your input, through `view` (a wheel notch: `view.step_zoom`, a drag:
// `view.drag`, ...). Once a frame, the zoom glide, which also says whether the
// view is moving (a glide or a pinch: no new window is decoded for a zoom it
// is passing through). Then the panel it is drawn in: the part of the image on
// screen, and its size in physical pixels.
view.advance_zoom_glide(dt);
inputs.moving = view.is_moving();
view.set_panel(panel);
let extent = view.measure(ImagePx::new(width, height), room);
view.clamp_pan(&extent);
let placed = view.place(&extent);
inputs.image = placed.inputs(pixels_per_point);

// `renderer` is either scivis-render backend's `ImageRenderResources`, or
// anything else implementing its `VolumeRenderer`: `sync` is generic over it.
let outcome = viewer.sync(&mut renderer, &inputs);
// …then draw, and repaint again if outcome.needs_repaint
for notice in viewer.notices() {
    // one line each on a status bar: a failed open, a frame that would not
    // decode, a 3D volume that did not fit, a note about the file's shape
}
```

What the user changes goes through operations, not fields: the frame on
screen is `set_frame` / `step_frame` on `viewer.document_mut()`, a channel's
contrast window and switch are `set_window` / `set_enabled` on its
`display_mut()`, and the axes and LUTs are `Viewer::set_dimension_order`,
`set_pseudocolor` and `set_gray_lut`. Those two are handles
(`DocumentMut`, `DisplayMut`) that lend these operations and nothing that
could replace the document or its display; a document arrives only through an
open. What a display channel is — an IFD's
intensities, an RGB or CMYK sample plane, palette indices — is
`Display::layout()`, a `ChannelLayout`.

`Viewer` holds no window, panel, zoom or pointer state — those are the
frontend's, and what they come to each frame is handed to `sync` as
`FrameInputs`. The 2D view's zoom and pan are a `View2d` the frontend keeps
and feeds its input to, so that a second frontend reuses the arithmetic and
not only the result. The dividing line is *"would a browser UI need this to show the
right pixels?"*; if yes, it's here.

## Features

| Feature | Default | Effect |
|---|---|---|
| `threads` | on | Background read-ahead and volume assembly. Turn it off for a single-threaded host such as `wasm32-unknown-unknown`; the synchronous fallbacks are already the paths taken when a worker fails to spawn, so nothing else changes. |
| `mmap` | on | Memory-mapped files, and with them `Viewer::open` by path. Off for a browser, where a document arrives as bytes (`Viewer::load_bytes`). |
| `codec-zstd` | on | fast-tiff-lib's ZSTD codec, a C dependency, so not for wasm. |

No feature picks a plugin: the plugins are another crate's, handed in by the
application, so this one compiles the same whichever of them it carries.

No feature picks a GPU backend. `Viewer::sync` takes any renderer that
implements scivis-render's `VolumeRenderer` (which includes its
`ImageRenderer`), and is compiled for the one the frontend hands it — static
dispatch, as if it named it — so this crate builds no backend at all, and the
frontend picks one where it constructs it. The sync's own tests drive it with
a double that records what it is sent (`src/sync_tests.rs`), with no GPU.

## License

MPL-2.0, as the crates it builds on (`fast-tiff-lib`, `scivis-render`) and the
plugin SDK are. It was GPL-3.0-only while the built-in plugins were compiled
into it: two of them link GPL-3.0-only registration crates
(`suite2p-registration`, `turboreg-registration`, and the
`image-registration-support` they share). Those plugins are the
[`fasttiff-builtin-plugins`](../fasttiff-builtin-plugins) crate now, which is
GPL-3.0-only, and this crate links nothing copyleft. An application that hands
it those plugins links them, and is GPL-3.0 for that — `FastTIFF` and
`FastTIFF-web` are.

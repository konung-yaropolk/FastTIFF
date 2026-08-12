# fast-tiff-viewer

[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/LICENSE)

The frontend-agnostic viewer core for multi-frame scientific TIFF stacks:
everything between "a TIFF on disk" and "pixels on the GPU", with no GUI toolkit
in sight.

```text
  fast-tiff-lib     file I/O, IFD index, decode, metadata
        │
  scivis-render     GPU pipelines, textures, ray-marching
        │
  fast-tiff-viewer  ← this crate: stack model, channel settings,
        │             c/z/t interpretation, camera, decode→GPU sync
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
- **Reads ahead** while playback keeps up: decode-ahead for compressed stacks,
  page-touch for uncompressed ones (where decoding is already zero-copy).
- **Drives the 3D camera** — four navigation styles, orbit/fly conversion,
  ray/box re-pivoting — as plain `f32` math over `(dx, dy)` deltas.
- **Keeps the playback clock**, advancing by real elapsed time so a movie runs at
  the file's fps regardless of render cadence, and latching parallel decode when
  it starts dropping frames.

## Usage

```rust,ignore
use fast_tiff_viewer::Viewer;

let mut viewer = Viewer::new();
viewer.open(path)?;

// each frame:
if viewer.playback.playing {
    viewer.tick_playback(now_seconds);
}
viewer.uv_offset = /* from your pan/zoom */;
viewer.uv_scale  = /* from your pan/zoom */;

let outcome = viewer.sync(&mut renderer);
// …then draw, and repaint again if outcome.needs_repaint
```

`Viewer` holds no window, panel, zoom or pointer state — those are the
frontend's. The dividing line is *"would a browser UI need this to show the
right pixels?"*; if yes, it's here.

## Features

| Feature | Default | Effect |
|---|---|---|
| `backend-wgpu` / `backend-glow` | — | Which GPU backend `Viewer::sync` drives. Exactly one is needed to sync; with neither, the CPU-side model still compiles and is fully usable headless. |
| `threads` | on | Background read-ahead and volume assembly. Turn it off for a single-threaded host such as `wasm32-unknown-unknown`; the synchronous fallbacks are already the paths taken when a worker fails to spawn, so nothing else changes. |

Note that the backend feature must match the one the frontend selects — in the
FastTIFF app, the `renderer-*` features forward to both crates, which is what
keeps them in step.

## License

MPL-2.0. File-level copyleft: changes to these files stay open, but a consuming
application doesn't have to be MPL.

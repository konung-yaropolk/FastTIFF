# FastTIFF-web

FastTIFF's **egui** interface, compiled to WebAssembly and drawn on a browser
canvas. This is what the GitHub Pages deploy publishes.

Files are decoded and rendered entirely in the browser — nothing is uploaded.

```text
  fast-tiff-lib      TIFF parsing + decode          ┐
  scivis-render      GPU compositing + ray-marching ├─ shared, unchanged
  fast-tiff-viewer   stack model, camera, GPU sync  ┘
        │
  src/render.rs      eframe ↔ scivis-render adapter (~100 lines)
  src/app.rs         the egui UI
```

## Two web builds, one core

| | `FastTIFF-web/` (React) | `FastTIFF-web/` (this) |
|---|---|---|
| UI | React + TypeScript, DOM widgets | egui, drawn on the canvas |
| Bundle | 2.3 MB wasm + 219 kB JS | 6.5 MB wasm, no JS framework |
| Toolchain | wasm-pack + npm + Vite | wasm-pack only |
| UI shares code with desktop | no — reimplemented | **yes — same toolkit, same idioms** |
| Native-feeling DOM (a11y, text selection, mobile) | yes | limited |

Both sit on the identical `fast-tiff-viewer` core; neither reimplements any
stack, channel, contrast, dimension-order, playback or camera logic. The
difference is only which toolkit paints.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --release
```

Then serve the folder — `index.html` loads `./pkg/` relatively, so any static
server works and the site is happy at any subpath:

```bash
python -m http.server 4200
```

## Controls

Same as the desktop app, minus window management:

| | 2D (movie) | 3D (volume) |
|---|---|---|
| Drag | pan (when zoomed in) | left: orbit · middle/right: pan |
| Scroll | scrub frames | fly along the view axis |
| Shift+scroll | scrub ~10% of the stack | — |
| Ctrl+scroll | zoom | — |
| ← → | — | orbit |
| WASD / Space / Shift | — | fly (per nav mode) |

## Notes on the wasm build

Identical to the React build's constraints, and for the same reasons — no
threads, no filesystem, no ZSTD. See
[`../FastTIFF-web/README.md`](../FastTIFF-web/README.md#notes-on-the-wasm-build)
for the detail; both crates depend on `fast-tiff-viewer` with
`default-features = false`.

One extra thing this build needs that the React one didn't:
`render::tune_web_options` replaces eframe's default device limits with the
adapter's real ones. eframe otherwise requests `wgpu::Limits::default()`, which
is smaller than this renderer needs (the composite pass binds 13 sampled
textures; the volume pass allocates 3D textures). It's the web twin of the
desktop adapter's `tune_native_options`.

This crate is deliberately **not** a workspace member — Cargo unifies features
across a workspace, and one build covering this and the desktop app would union
`threads`/`mmap`/`codec-zstd` back in and break the wasm build.

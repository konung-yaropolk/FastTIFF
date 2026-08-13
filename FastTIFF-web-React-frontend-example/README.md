# FastTIFF-web

FastTIFF in the browser: the **same** Rust viewer core and GPU renderer as the
desktop app, compiled to WebAssembly, behind a React UI.

Files are decoded and rendered entirely in the browser — nothing is uploaded.

```text
  fast-tiff-lib      TIFF parsing + decode          ┐
  scivis-render      GPU compositing + ray-marching ├─ shared with the desktop app,
  fast-tiff-viewer   stack model, camera, GPU sync  ┘  compiled to wasm32
        │
  crate/             wasm-bindgen bindings (~600 lines)
        │
  src/               React UI
```

The only new Rust here is `crate/` — the browser's counterpart to the desktop
app's `FastTIFF/src/render.rs`. Both are thin adapters doing the same three jobs
(get GPU handles from the host, own the render pass, translate input) over an
identical core. No stack, channel, contrast, dimension-order, playback or camera
logic is reimplemented for the web.

## Requirements

A browser with **WebGPU** (Chrome/Edge 113+, Firefox 141+) or **WebGL2** —
`wgpu` picks WebGPU when available and falls back automatically.

## Develop

```bash
npm install
npm run dev
```

`npm run dev` rebuilds the wasm first, then starts Vite. After changing anything
in `crate/` or the Rust crates above it, re-run `npm run wasm` (or just
`npm run dev` again).

Prerequisites: [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) and the
wasm target —

```bash
rustup target add wasm32-unknown-unknown && cargo install wasm-pack
```

## Build

```bash
npm run build      # wasm-pack -> tsc -> vite build, output in dist/
npm run preview    # serve dist/ locally
```

## Deploy to GitHub Pages

`.github/workflows/deploy-web.yml` does this on every push to `main` that
touches the web app or any crate it depends on. Enable it once:

**Settings → Pages → Source → GitHub Actions.**

The workflow sets `BASE_PATH=/<repo>/` because a project site is served from a
subpath; `vite.config.ts` reads it. A user/organisation site (`<user>.github.io`)
is served from `/`, so drop the env var there.

## Controls

| | 2D (movie) | 3D (volume) |
|---|---|---|
| Drag | pan (when zoomed in) | left: orbit · middle/right: pan |
| Scroll | scrub frames | fly along the view axis |
| Shift+scroll | scrub ~10% of the stack | — |
| Ctrl+scroll | zoom about the cursor | — |
| ← → | step frame (Shift: fast) | orbit |
| Space | play/pause | play/pause (4D stacks) |

## Notes on the wasm build

Three things differ from the desktop build, all forced by the target and all
handled by turning features off rather than by forking any code:

- **No threads.** `wasm32-unknown-unknown` has no `std::thread::spawn`, so the
  `threads` feature is off: no read-ahead worker, no background volume builder.
  Both already had synchronous fallbacks — the path taken whenever a worker
  failed to spawn — so scrubbing and 3D still work, just on one core. Rayon
  still *compiles*, and `should_parallelize` returns `false` on wasm so no
  parallel path is ever entered.
- **No filesystem.** The `mmap` feature is off; stacks arrive as bytes through
  `TiffStack::from_bytes` instead of being memory-mapped. The whole file sits in
  memory, so very large stacks are limited by the browser's ~4 GB wasm heap
  rather than by the OS page cache.
- **No ZSTD.** `zstd-sys` is a C dependency and cannot build for this target, so
  `codec-zstd` is off. A ZSTD-compressed stack reports a clear error instead of
  decoding. LZW, Deflate and PackBits all work.

## Layout

```
FastTIFF-web/
├── crate/              wasm-bindgen bindings (Rust)
│   └── src/lib.rs
├── public/samples/     a small demo stack, so the page is usable with no file
├── src/
│   ├── useViewer.ts    owns the wasm instance + the render loop
│   ├── App.tsx         layout, canvas input, keyboard
│   └── components/     Toolbar, ScrubBar, ChannelPanel, VolumeSettings, MetadataPanel
└── vite.config.ts
```

`crate/` is deliberately **not** a member of the repo's Cargo workspace: Cargo
unifies features across a workspace, so one build covering both this crate and
the desktop app would union `threads`/`mmap`/`codec-zstd` back in and break the
wasm build.

The render loop in `useViewer.ts` is demand-driven, not a permanent
`requestAnimationFrame`: a static 2D frame needs no redraws, so the loop parks
itself and any mutation calls `redraw()`. Playback and in-flight volume builds
keep it running by returning `true` from the Rust `render()`.

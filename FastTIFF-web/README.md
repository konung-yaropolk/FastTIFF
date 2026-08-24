# FastTIFF-web

FastTIFF's **egui** interface, compiled to WebAssembly and drawn on a browser
canvas. This is what the GitHub Pages deploy publishes.

### ▶ [FastTIFF Online](https://konung-yaropolk.github.io/FastTIFF/)

**Your files never leave your machine.** There is no server and nothing is
uploaded — the page is static, and the TIFF is decoded and rendered locally by
WebAssembly and your GPU.

Needs a current browser with WebGPU or WebGL2 (Chrome/Edge 113+, Firefox 141+).

> **Worth knowing before you rely on it:** the whole file is held in memory (no
> `mmap` in a browser), so file size is capped by wasm's 32-bit address space —
> comfortable up to a few hundred MB, not multi-GB. And with no threads, volume
> building runs inline and **freezes the page** while "Building the volume…" is
> up; read-ahead decoding and parallel decode don't run either. Ray-marching is
> GPU work and stays fast once built. ZSTD-compressed stacks can't be opened.
> The [root README](../README.md#browser-limitations) has the full picture.

```text
  fast-tiff-lib      TIFF parsing + decode          ┐
  scivis-render      GPU compositing + ray-marching ├─ shared with the desktop app
  fast-tiff-viewer   stack model, camera, GPU sync  │
  FastTIFF (lib)     the egui UI + eframe adapter   ┘
        │
  src/lib.rs         this crate: ~57 lines of browser host
```

**There is no UI code in this crate.** The interface is the *same*
`fasttiff::ViewerApp` the desktop binary runs — `FastTIFF` is a lib + bin, and
both hosts construct the same app. All this crate does is hand eframe a canvas
instead of a window.

What actually differs between the two hosts is 31 `#[cfg(target_arch = "wasm32")]`
sites across ~3,100 lines of shared UI, and every one of them is about the
*host* rather than the viewer:

| | Desktop | Web |
|---|---|---|
| Window sizing, position, title | `ViewerApp::manage_window` | nothing — CSS sizes the canvas |
| Opening a file | blocking dialog, argv, Apple Events → a path | async picker / drop → bytes |
| GPU option hook | `render::tune_native_options` | `render::tune_web_options` |
| Extra files at once | launched as sibling processes | n/a |
| Interface scale | native (`app::scale::UI_SCALE` = 1.0) | **150%**, pop-ups excepted |
| Empty-viewport prompt | drop/open + scroll hints | the same, plus the local-processing line |
| Opening zoom | window resized to the image, at a fixed level | image fitted to the canvas, at whatever factor that takes |

Both routes to a file meet at one `Opened` enum, so everything downstream —
loading, fit, resetting the chrome — is shared.

## Why the opening zoom is not a round number

The desktop build resizes its window around the image, so it can always open at
one of the fixed zoom levels — 100%, 50%, 25%. A canvas has no window to
resize, so the image is fitted to the canvas instead, and the factor that takes
is whatever the two aspect ratios happen to demand: 53.91%, say. Snapping that
to the nearest fixed level would either crop a sliver off the image or leave a
margin you did not ask for.

The exact factor is then inserted into that image's zoom ladder
(`app::zoom_ladder`) for as long as the file is open, so wheeling out returns to
the fitted view rather than stepping past it to 50%. It is dropped on the next
open, since it describes one picture in one canvas.

## Why the web build is drawn larger

`install_chrome` sets egui's zoom factor to 1.5 here and 1.0 natively. The
desktop window is sized to the image and read at arm's length; a canvas is one
element on a page, competing with the browser's own furniture, and the same
widgets come out noticeably small. Zoom factor scales `pixels_per_point`, so
fonts, padding, hit targets and line widths all grow together — the layout is
enlarged rather than stretched.

**Pop-up windows opt out.** The 3D settings, file metadata and histogram windows
are dense and already sized to be a tight fit; half again as large pushes them
past the bottom of a laptop viewport, which is the one place a browser cannot
simply be resized. `app::scale::unscaled` scales the global style by the
reciprocal for the duration of the window, which cancels the zoom exactly —
title bar included, since `Window` takes those metrics from the context. Note
that this scales the *style*, so anything a pop-up sizes in literal points has
to be expressed relative to the text (see `PLOT_TEXT_HEIGHTS`) or it will keep
the enlarged scale while the text around it shrinks back.

The panel's control row wraps to as many lines as it needs, which is what keeps
the enlarged chrome usable in a narrow window.

## Two web builds, one core

| | `FastTIFF-web-React-frontend-example/` | `FastTIFF-web/` (this, deployed) |
|---|---|---|
| UI | React + TypeScript, DOM widgets | egui, drawn on the canvas |
| Bundle | 2.3 MB wasm + 219 kB JS | 6.6 MB wasm, no JS framework |
| Toolchain | wasm-pack + npm + Vite | wasm-pack only |
| UI code | reimplemented (~700 lines TSX) | **literally the desktop app's** |
| Fixing a UI bug | fix it twice | fix it once |
| Native-feeling DOM (a11y, text selection, mobile) | yes | limited |

Both sit on the identical `fast-tiff-viewer` core and neither reimplements any
stack, channel, contrast, dimension-order, playback or camera logic. The React
build is kept as a worked example of driving the core from JavaScript — useful
if you want a DOM UI — but it does duplicate the interface, which is why the
egui build is the one deployed.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --out-dir pkg --release
```

Then serve the folder — `index.html` loads `./pkg/` relatively, so any static
server works and the site is happy at any subpath:

```bash
cp ../FastTIFF/icon/icon.svg ../FastTIFF/icon/icon32.png ../FastTIFF/icon/icon256.png .
python -m http.server 4200
```

The `cp` is only for the favicons — they live in `FastTIFF/icon/` so the tab
icon and the desktop build share one source, and the deploy workflow copies
them the same way. Skip it and you just get a default icon.

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
[`../FastTIFF-web-React-frontend-example/README.md`](../FastTIFF-web-React-frontend-example/README.md#notes-on-the-wasm-build)
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

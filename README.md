# FastTIFF - a lightning-fast multi-frame 2D- and 3D-viewer with ImageJ-compatible GPU-rendering

[![Try it online](https://img.shields.io/badge/Try%20it-FastTIFF%20Online-5b9dd9?logo=googlechrome&logoColor=white)](https://konung-yaropolk.github.io/FastTIFF/)
[![Release](https://img.shields.io/github/v/release/konung-yaropolk/FastTIFF?label=release)](https://github.com/konung-yaropolk/FastTIFF/releases)
[![License](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/release.yml?label=build)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/ci.yml)


[![FastTIFF](https://github.com/user-attachments/assets/b935b2fa-86cc-4edf-ab5a-255a1aa73e4d)](https://github.com/konung-yaropolk/FastTIFF/releases)  
A fast multi-frame TIFF stack viewer for huge ImageJ hyperstacks: a horizontal
scrubber instead of ImageJ's slice slider, GPU-side LUT/contrast rendering,
and (for the common uncompressed case) zero CPU-side image processing per
frame change.

Open a stack via the "Open TIFF..." button or by dragging a `.tif`/`.tiff`
file onto the window. Scrub with the bottom slider, the mouse wheel while
hovering over the image (one frame per notch; hold **Shift** for fast
continuous scrolling), or the left/right arrow keys.



<table align="center" style="border: none;">
  <tr>
    <td style="border: none; padding: 5px;">
      🎥 2D.gif  <br>
      <img width="320" height="404" alt="2D" src="https://github.com/user-attachments/assets/5ba646bc-f451-4872-978b-bc0d5cf8b056" />  
    </td>
    <td style="border: none; padding: 5px;">
      <video src="https://github.com/user-attachments/assets/0bec868f-cf4f-4b9c-8c7b-c4f123f09b27" autoplay loop muted playsinline width="100%"></video>
    </td>
  </tr>
</table>

## 🌐 [FastTIFF Online](https://konung-yaropolk.github.io/FastTIFF/) - no install

The full viewer runs in the browser: same Rust decoder, same GPU renderer, same
ImageJ-compatible display. **Your files never leave your machine** — there is no
server and nothing is uploaded. The page is static; the TIFF is decoded and
rendered locally by WebAssembly and your GPU.

Good for a quick look, a second opinion on someone else's stack, or a machine
you can't install software on. For day-to-day work on large stacks, the
[desktop build](#downloads) is substantially faster — see
[what the browser gives up](#browser-limitations).

Needs a current browser with WebGPU or WebGL2 (Chrome/Edge 113+, Firefox 141+).

## ⬇️ Desktop App - Download links

Every link below points at the **latest release**. You can either browse the
[**Releases**](https://github.com/konung-yaropolk/FastTIFF/releases) page for a
specific or older version.

| Download | For | Install / run |
|----------|-----|---------------|
|[**Installer**](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-setup.exe) or [**Portable**](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF.exe) |<img width="16" height="16" alt="microsoft-windows-icon" src="https://github.com/user-attachments/assets/506a3842-c123-4f21-b571-ef9769573b04" /> **Windows 10 / 11** — 64-bit | Installer adds a Start-menu entry + "Open with" for TIFFs; the portable `.exe` just runs. |
|[`FastTIFF-arm64.dmg`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-arm64.dmg) |<img width="16" height="16" alt="apple" src="https://github.com/user-attachments/assets/0d2c799e-8aac-4e07-9ee8-96bbccff1693" />&nbsp;**macOS 11+** — Apple Silicon (M-series chipsets) | Open the `.dmg`, drag **FastTIFF** into Applications. |
|[`FastTIFF-x86_64.dmg`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.dmg) |<img width="16" height="16" alt="apple" src="https://github.com/user-attachments/assets/0d2c799e-8aac-4e07-9ee8-96bbccff1693" />&nbsp;**macOS 11+** — Intel | Open the `.dmg`, drag **FastTIFF** into Applications. |
|[`FastTIFF-arm64.deb`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-arm64.deb) |<img width="16" height="16" alt="debian" src="https://github.com/user-attachments/assets/2fefe976-d357-4abb-a476-6a8a31b422fe" />&nbsp;**Debian / <img width="16" height="16" alt="ubuntu" src="https://github.com/user-attachments/assets/e2bd815d-709d-4571-9665-f957c90e8300" />&nbsp;Ubuntu** — ARM64 | `sudo apt install ./FastTIFF-arm64.deb` |
|[`FastTIFF-x86_64.deb`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.deb) |<img width="16" height="16" alt="debian" src="https://github.com/user-attachments/assets/2fefe976-d357-4abb-a476-6a8a31b422fe" />&nbsp;**Debian / <img width="16" height="16" alt="ubuntu" src="https://github.com/user-attachments/assets/e2bd815d-709d-4571-9665-f957c90e8300" />&nbsp;Ubuntu** — x86-64 | `sudo apt install ./FastTIFF-x86_64.deb` |
|[`FastTIFF-arm64.rpm`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-arm64.rpm) |<img width="16" height="16" alt="fedora" src="https://github.com/user-attachments/assets/97817114-81e7-40bd-8b4d-165fc856cba4" />&nbsp;**Fedora / <img width="16" height="16" alt="redhat-icon" src="https://github.com/user-attachments/assets/aaf64e10-797c-4cb7-87d9-04293d761cc8" />&nbsp;RHEL / <img width="16" height="16" alt="suse" src="https://github.com/user-attachments/assets/2f9e5316-fbea-4394-b801-784672831993" />&nbsp;openSUSE** — ARM64 | `sudo dnf install ./FastTIFF-arm64.rpm` |
|[`FastTIFF-x86_64.rpm`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.rpm) |<img width="16" height="16" alt="fedora" src="https://github.com/user-attachments/assets/97817114-81e7-40bd-8b4d-165fc856cba4" />&nbsp;**Fedora / <img width="16" height="16" alt="redhat-icon" src="https://github.com/user-attachments/assets/aaf64e10-797c-4cb7-87d9-04293d761cc8" />&nbsp;RHEL / <img width="16" height="16" alt="suse" src="https://github.com/user-attachments/assets/2f9e5316-fbea-4394-b801-784672831993" />&nbsp;openSUSE** — x86-64 | `sudo dnf install ./FastTIFF-x86_64.rpm` |
|[`FastTIFF-x86_64.pkg.tar.zst`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.pkg.tar.zst) |<img width="16" height="16" alt="archlinux" src="https://github.com/user-attachments/assets/9bbaa231-f10d-4642-a9a9-a31a8a980e3d" />&nbsp;**Arch Linux** — x86-64 | `sudo pacman -U ./FastTIFF-x86_64.pkg.tar.zst` |
|[`FastTIFF-arm64.flatpak`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-arm64.flatpak) |<img width="16" height="16" alt="linux-tux" src="https://github.com/user-attachments/assets/78810b28-56ac-4d72-b0b1-ad8a93b9d2a9" />&nbsp;**Any Linux** · Flatpak — ARM64 | `flatpak install --user ./FastTIFF-arm64.flatpak` |
|[`FastTIFF-x86_64.flatpak`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.flatpak) |<img width="16" height="16" alt="linux-tux" src="https://github.com/user-attachments/assets/78810b28-56ac-4d72-b0b1-ad8a93b9d2a9" />&nbsp;**Any Linux** · Flatpak — x86-64 | `flatpak install --user ./FastTIFF-x86_64.flatpak` |
|[`FastTIFF-arm64.AppImage`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-arm64.AppImage) |<img width="16" height="16" alt="linux-tux" src="https://github.com/user-attachments/assets/78810b28-56ac-4d72-b0b1-ad8a93b9d2a9" />&nbsp;**Any Linux** · AppImage — ARM64 | `chmod +x FastTIFF-arm64.AppImage && ./FastTIFF-arm64.AppImage` |
|[`FastTIFF-x86_64.AppImage`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-x86_64.AppImage) |<img width="16" height="16" alt="linux-tux" src="https://github.com/user-attachments/assets/78810b28-56ac-4d72-b0b1-ad8a93b9d2a9" />&nbsp;**Any Linux** · AppImage — x86-64 | `chmod +x FastTIFF-x86_64.AppImage && ./FastTIFF-x86_64.AppImage` |

**Which one?** On Linux, the **.deb** / **.rpm** / **pacman** packages integrate
with your system package manager; the **Flatpak** and **AppImage** are
distro-agnostic and self-contained — handy when no native package matches your
distro. For the CPU: 64-bit Intel/AMD → the `x86_64` files; 64-bit ARM
(Raspberry Pi, Ampere, Apple Silicon) → the `arm64` files. On Linux, `uname -m`
prints `x86_64` or `aarch64` (which is the `arm64` file); on macOS, *About This
Mac* shows "Apple M…" (arm64) or "Intel" (x86_64). After installing on Linux,
launch it from the apps menu or by running `FastTIFF` (or `fasttiff`) in a
terminal.

**First-launch security prompt** (the binaries aren't signed with a paid
developer certificate):
- **Windows** — SmartScreen may warn: click *More info → Run anyway*.
- **macOS** — Gatekeeper blocks unsigned apps the first time: **right-click the
  app → Open** (then confirm), or run
  `xattr -dr com.apple.quarantine /Applications/FastTIFF.app` once.

Not listed (32-bit or a different CPU)? Build from source.

## Browser limitations

[FastTIFF Online](https://konung-yaropolk.github.io/FastTIFF/) runs the same code as the
desktop build, but the browser withholds three things the native app relies on. None of it is a
missing feature — it's the platform:

**File size.** The desktop build memory-maps the file, so opening a 20 GB stack
is instant and only the frames you actually visit occupy RAM. A browser has no
filesystem to map, so the whole file is read into WebAssembly memory. That
memory is 32-bit — a hard ceiling around **4 GB**, and in practice browsers
start failing well before it. Loading also costs roughly double transiently,
while the bytes are copied from the browser's buffer into wasm. **Rule of
thumb: comfortable up to a few hundred MB; multi-GB stacks want the desktop
build.**

**Speed, especially in 3D.** `wasm32` has no threads, so three things the
desktop does in the background all happen inline on one core:

- *Read-ahead decoding* is gone. Scrubbing an uncompressed stack is barely
  affected (decoding is a zero-copy borrow either way), but a compressed one is
  noticeably slower because each frame decodes on demand.
- *Volume building is synchronous.* On the desktop a worker thread assembles the
  3D volume while the UI stays live; in the browser the page **freezes** during
  "Building the volume…", and again on every timepoint step of a 4D stack.
- *Parallel decode* never engages, so wide 32-/64-bit frames take the
  single-core path.

Ray-marching itself is GPU work and stays fast once the volume is built — it's
getting there that's slower. A large 3D stack can take several seconds and a
visible stall.

**ZSTD-compressed stacks won't open.** The ZSTD codec is a C library that can't
be built for this target. LZW, Deflate, PackBits and uncompressed all work; a
ZSTD file reports a clear error rather than failing silently.

Everything else — per-channel contrast and LUTs, ImageJ/OME metadata,
dimension-order correction, playback, all three volume modes — behaves exactly
as it does natively, because it *is* the same code.

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
in - the other's dependencies are excluded entirely.


## Test and lint

Unit Test:
```sh
cargo test --workspace
```

Lint:
```sh
cargo clippy --workspace --all-targets
```


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

Four crates, layered bottom-up. Each one below the app is free of any GUI
toolkit, so the same engine can drive a different frontend (a wasm/web UI is the
intended second one):

```text
  fast-tiff-lib     file I/O, IFD index, decode, metadata
        │
  scivis-render     GPU pipelines, textures, ray-marching
        │
  fast-tiff-viewer  stack model, channel settings, decode → GPU sync
        │
  FastTIFF          the egui UI (lib) + the native binary
        │
  FastTIFF-web      a browser host around the same UI
```

`FastTIFF` is a lib + bin: the library half holds the egui interface, and both
the desktop binary and the web build construct the same `ViewerApp`. The UI is
written once; what differs per host — window management, how a file is opened,
the GPU option hook — is confined to a handful of `#[cfg(target_arch = "wasm32")]`
sites.

- **`fast-tiff-lib/`** - pure parsing/decoding library, no GUI or GPU
  dependencies. IFD-chain walking, ImageJ metadata parsing, strip decoding
  (uncompressed fast path + LZW/Deflate/PackBits + predictor undo). Reads
  either a memory-mapped path (`TiffStack::open`) or a plain byte buffer
  (`TiffStack::from_bytes`), so it also builds for
  `wasm32-unknown-unknown` with `--no-default-features`. Has a
  real test suite (`cargo test -p fast-tiff-lib`) that builds synthetic
  multi-frame TIFFs in memory and round-trips them through the whole
  pipeline - this is the part most worth trusting blind, since it's
  actually verified. **Published on crates.io** — see below.
- **`scivis-render/`** — all GPU work, free of any GUI toolkit: the
  compositing pipeline, per-channel textures and LUTs, and the volume
  ray-marcher. Never creates a device, a surface or a window — construction
  takes the host's device, painting takes the host's render pass. Two additive
  backends (`backend-wgpu`, `backend-glow`). Its WGSL shaders are validated
  offline with naga, so a broken shader fails `cargo test` instead of showing a
  blank canvas.
- **`fast-tiff-viewer/`** — everything between "a TIFF on disk" and "pixels on the
  GPU": the loaded-stack model, per-channel contrast/LUT derivation, c/z/t
  interpretation, the display-channel → IFD mapping, 3D volume assembly,
  read-ahead, the camera, the playback clock, and the per-frame sync that drives
  the renderer. A `threads` feature (on by default) can be turned off for a
  single-threaded host such as `wasm32-unknown-unknown`.
- **`FastTIFF/`** — the GUI binary: eframe/egui for the window and controls.
  It holds a `fast_tiff_viewer::Viewer` plus the things only a desktop window has
  (zoom, pan, window sizing, which panels are open). `src/render.rs` is the sole
  file bridging the renderer to eframe — a browser frontend writes its own
  ~150-line equivalent and reuses everything else.

## The TIFF engine is a standalone crate

The viewer isn't built on an existing TIFF library — the whole reader/writer
was written from scratch for it, and it's published separately as
[**`fast-tiff-lib`**](https://crates.io/crates/fast-tiff-lib) so it can be used
without any of the GUI:

```sh
cargo add fast-tiff-lib
```

[![Crates.io](https://img.shields.io/crates/v/fast-tiff-lib?color=green)](https://crates.io/crates/fast-tiff-lib)
[![Downloads](https://img.shields.io/crates/d/fast-tiff-lib)](https://crates.io/crates/fast-tiff-lib)
[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/fast-tiff-lib/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Docs](https://img.shields.io/docsrs/fast-tiff-lib?label=docs.rs)](https://docs.rs/fast-tiff-lib)

It's a *specialized* engine for lazily scrubbing large scientific hyperstacks
rather than a general-purpose TIFF library: memory-mapped and lazy by default
(frames decode on demand, never the whole stack — or read from a byte buffer
where there's no filesystem), zero-copy for uncompressed data, and
it parses **ImageJ and OME-TIFF** hyperstack metadata (channels/slices/frames,
LUTs, calibration) — into one normalized view — that general TIFF readers hand
back as an opaque string. It also writes: streaming multi-frame output with
automatic BigTIFF upgrade, and metadata in either dialect from one neutral
builder. Full format coverage, a comparison against libtiff / the `tiff` crate /
TinyTIFF, and benchmarks are in [`fast-tiff-lib/README.md`](fast-tiff-lib/README.md).

Note the licenses differ: the viewer is GPLv3, but the crate is **MPL-2.0**, so
it can be used in projects that couldn't take a GPL dependency.

## What v1 covers

- Multi-frame grayscale, multi-channel composite, and RGB TIFFs (chunky or
  planar) in 8-bit, 16-bit, and 32-bit (integer or float) - 32-bit and float
  data is auto-ranged into the display, RGB is split into R/G/B planes.
- CMYK (Separated) TIFFs, 8- and 16-bit: the four ink plates are converted to
  RGB for display, since ink subtracts from white where the compositing shader
  adds. The status bar says `CMYK->RGB` so it is clear the three channels shown
  are derived rather than stored.
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
- A **histogram** window (the bar-chart button in the panel) plotting the
  current frame's intensity distribution, one filled curve per channel in that
  channel's own LUT colour, all on a shared axis so a dim channel visibly sits
  left of a bright one. The same contrast sliders sit beneath it, each one
  full-width under its channel's name and window values, so a slider spans
  exactly the part of the plot it clips — and the parts each channel's window
  throws away are drawn faded, so you can see what a setting is costing you.
  Unticking a channel takes its curve off the plot and greys out its slider, in
  both the panel and the window. Resizable — the plot takes whatever height the
  window is given. Log scaling is on by default: a 16-bit frame is mostly
  background, and linear puts all of it in one bin at the edge.
- Z-slice selector when `slices > 1` (the scrubber itself always drives
  the time/frame axis).
- Zoom (Ctrl+scroll) and pan (drag) of the 2D image, with the window sized
  to fit on open. A wheel notch **glides** to the next level over about a tenth
  of a second, about the point under the cursor, rather than jumping — long
  enough to see which way the picture moved, short enough that a second notch
  never queues up behind the first. Turning the wheel again mid-glide advances
  another rung rather than restarting the same one. **Pinch to zoom and two fingers to pan** on a touch screen,
  in both the 2D and 3D views — a pinch zooms continuously rather than in
  fixed steps, since a gesture is continuous.
- A **navigator** in the corner of the 2D view once the whole frame no longer
  fits: two nested rectangles, the frame and the part of it on screen, in the
  manner of ImageJ's. At 800% on a mosaic every screenful looks like every
  other, and this is what says which one you are looking at.
- **Physical size** in the toolbar next to the pixel count, when the file says
  how big a pixel is (OME `PhysicalSize`, or the TIFF resolution tags). Both
  axes always, since anisotropic pixels are common enough that collapsing them
  would be a lie half the time.
- **Opening does not block the interface.** Walking the IFD chain and measuring
  each channel's display range takes seconds on a large stack, and it happens
  on a worker thread with a progress readout in the panel — a counted bar for
  the channel scans, whose length is known, and a spinner for the index walk,
  whose length is not. The file already open stays on screen and usable until
  the new one lands. (A browser has no threads to spawn, so there the load
  still blocks.)
- **Frames larger than one GPU texture, at full resolution** — every GPU caps a
  texture at 16384 or 32768 pixels per axis, and mosaics run well past it
  (Hubble's Andromeda is 40000 x 12788). Such a frame is shown through a
  *window*: the part you are looking at is kept on the GPU, and re-cut as you
  pan. Zoomed out that is a reduced view of the whole frame, flagged
  `1/N scale` in the toolbar; zoom in and the flag disappears, because you are
  then looking at the file's own pixels.

  A **Large image** selector appears in the panel for these frames, choosing
  between two ways of spending your machine on the problem:

  - **Tiled** (default) trades CPU for RAM. Only the region on screen is
    loaded, at exactly the resolution the zoom calls for and no finer, so
    memory follows the viewport rather than the file — which is why it is the
    default: a frame of any size opens. Every zoom step is then a fresh decode
    of the region, which is where the stutter comes from, and the coarse levels
    it picks when zoomed out alias more.
  - **Preload** trades RAM for smoothness. One reduced copy of the whole frame
    — the finest that fits, usually 1/2 or 1/4 — is decoded once and kept, and
    full resolution is cut for whatever is on screen. There are only ever those
    two levels, so a zoom crosses one boundary instead of a dozen, panning and
    zooming out decode nothing at all, and the reduced view aliases far less:
    point-sampling every second pixel of a stitched mosaic moirés where every
    sixteenth does. Costs up to 512 MB held for the life of the file, so it is
    worth choosing when you know the file fits that.

  The rest of this section describes the machinery both share.

  Nothing is held in memory to make that work. Moving the window decodes only
  the *strips* it covers (`FrameInfo::crop_rows`), a band at a time, so both
  the time and the peak memory follow what is on screen rather than the size of
  the file — tenths of a second and a couple of hundred megabytes for a window
  of Andromeda, against 4 seconds and 1.5 GB for the whole frame. Files far
  larger than RAM are no different in kind.

  How finely it is sampled follows the **display**, not the file: showing a
  40000-pixel mosaic on a 1900-pixel panel keeps about 1/16 of the pixels,
  because the rest cannot be drawn. Sampling to the texture budget instead — as
  it did at first — built a 384 MB texture to draw 1900 columns, and rebuilt it
  on every zoom step. The sampling levels are powers of two, so a zoom from fit
  to 1:1 crosses five of them rather than changing continuously.

  A coarse view skips whole **strips**: at 1/8 sampling only every eighth row
  is kept, so seven strips in eight are never decompressed at all
  (`FrameInfo::crop_rows_step`). Decompressing them and throwing the rows away
  is what made the zoomed-out view the slowest one to build, which is backwards.

  Bands come off a grid fixed to the frame rather than to the window, and the
  last few are kept (capped, a few hundred MB), so moving the view re-uses what
  it already decompressed: pan sideways and the rows on screen do not change at
  all, so nothing is decoded again; pan vertically and everything but the newly
  exposed edge is re-used. The grid is independent of zoom, so zooming over one
  spot re-uses its bands too.

  A **tiled** TIFF does better still. A strip spans the whole image width, so a
  narrow window of a wide mosaic decodes about ten times what it displays — the
  file's layout, not the reader's doing. A tile is bounded on both axes, so the
  window narrows in both: on the same 40000 x 12788 mosaic a 3840 x 2200 window
  costs 153 ms and 37 MB decoded, against 824 ms and 264 MB from the stripped
  original. Tiled files open natively, so converting a large image to tiles is
  worth it if you plan to explore it.

  Re-cutting the window happens on a **background thread**, and the window
  already on screen keeps being drawn — stretched — until the new one lands. A
  zoom crosses a sampling level every few notches, and building the next window
  is tenths of a second at best, so doing it inline stopped the program
  repeatedly during exactly the gesture where that is most obvious: a 14-notch
  zoom of Andromeda stalled for 2.25 seconds at a stretch. Off-thread the same
  sweep does not stall at all; the picture goes soft for a moment instead.

  That covers zooming *in*, where the window on screen still covers the ground
  the view wants. Zooming out, or jumping across the image at full resolution,
  it does not — a window of one corner cannot be stretched over the whole frame
  — so there would be nothing to draw and the rebuild would have to be waited
  for. What is kept instead is the **fit view**: the whole frame at a coarse
  sampling, decoded once when the file opens (which happens anyway) and held in
  RAM, about 5 MB for Andromeda. Leaving the resident window then costs an
  upload rather than a decode, and the picture stays correct — coarse, but the
  right pixels, not the resident window smeared past its edge. A 14-notch zoom
  out stalled for 0.75 seconds at a stretch before; it no longer stalls at all.
  Only the very first view of a file is waited for, because until then there is
  nothing to keep showing.

  None of this machinery runs for an image that fits a texture, which is very
  nearly all of them: those take the same path they always did, decoded and
  uploaded whole, with no planning, cropping or extra copy.

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
`build_jobs()` in `fast-tiff-viewer/src/sync.rs`. This is what ImageJ's TIFF writer
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
- Done: publish fast-tiff-lib to crates.io
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
- Done: Port to macOS
- Done: close all opened dialog windows when opened new file
- Done: surface rendering mode added
- Done: Add windows installer with files association
- publish in Brew
- Add 3D volume save in best suitable format for the gpu rendering (not necessary now, need to study the idea)



<img width="1137" height="923" alt="Untitled" src="https://github.com/user-attachments/assets/e3fe4619-bd53-4b68-bf7f-019231cf20a6" />

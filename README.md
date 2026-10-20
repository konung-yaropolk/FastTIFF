# FastTIFF - a lightning-fast multi-frame 2D- and 3D-viewer with ImageJ-compatible GPU-rendering
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner-direct-single.svg)](https://stand-with-ukraine.pp.ua)

[![Try it online](https://img.shields.io/badge/Try%20it-FastTIFF%20Online-5b9dd9?logo=googlechrome&logoColor=white)](https://konung-yaropolk.github.io/FastTIFF/)
[![Release](https://img.shields.io/github/v/release/konung-yaropolk/FastTIFF?label=release)](https://github.com/konung-yaropolk/FastTIFF/releases)
[![License](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/release.yml?label=build)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/release.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/konung-yaropolk/FastTIFF/ci.yml?branch=main&label=tests)](https://github.com/konung-yaropolk/FastTIFF/actions/workflows/ci.yml)


[![FastTIFF](https://github.com/user-attachments/assets/b935b2fa-86cc-4edf-ab5a-255a1aa73e4d)](https://github.com/konung-yaropolk/FastTIFF/releases)  
A fast multi-frame TIFF stack viewer optimized for fast viewing huge hyperstacks: movie playback, 3D volumetric reconstruction, GPU-side rendering,
and zero CPU-side image processing per
frame change.

**Key Features**
- **Memory-Mapped Lazy Loading**: Instead of loading an entire massive file into RAM, the desktop build memory-maps the file. This means a 20 GB TIFF stack can open instantly, and the system only loads the specific frames you are currently viewing.
- **Zero-Copy Rendering**: For common uncompressed files, it processes frames with zero CPU-side image processing or allocation, simply reading directly from the mapped bytes.  Advanced GPU Rendering: It utilizes wgpu (DirectX 12, Vulkan, Metal) to offload tasks to the GPU. This includes applying Look-Up Tables (LUTs), adjusting contrast, and generating ray-marched 3D volume views from 2D stacks in real-time.
- **Web Accessibility**: The browser version runs the exact same Rust decoder and GPU renderer natively in your browser. Files are processed completely locally—meaning sensitive scientific data never leaves your machine.  



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
|[`FastTIFF-universal.dmg`](https://github.com/konung-yaropolk/FastTIFF/releases/latest/download/FastTIFF-universal.dmg) |<img width="16" height="16" alt="apple" src="https://github.com/user-attachments/assets/0d2c799e-8aac-4e07-9ee8-96bbccff1693" />&nbsp;**macOS 11+** — Apple Silicon *and* Intel | Open the `.dmg`, drag **FastTIFF** into Applications. |
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
(Raspberry Pi, Ampere) → the `arm64` files. `uname -m` prints `x86_64` or
`aarch64` (which is the `arm64` file). **On macOS there is nothing to choose**
— the `.dmg` holds a universal binary, so the same download runs natively on
Apple Silicon and on Intel, and macOS picks the right half when the app
launches. After installing on Linux, launch it from the apps menu or by running
`FastTIFF` (or `fasttiff`) in a terminal.

**First-launch security prompt** (the binaries aren't signed with a paid
developer certificate):
- **Windows** — SmartScreen may warn: click *More info → Run anyway*.
- **macOS** — Gatekeeper blocks unsigned apps the first time: **right-click the
  app → Open** (then confirm), or run
  `xattr -dr com.apple.quarantine /Applications/FastTIFF.app` once.

Not listed (32-bit or a different CPU)? Build from source.

## Browser limitations

[FastTIFF Online](https://konung-yaropolk.github.io/FastTIFF/) runs the same code as the
desktop build, but a tab is not a desktop and four things follow from that. None of it is a
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

**Saving is a download, and the file is built in memory.** Same encoder, same
bytes — a stack saved in the browser is byte-for-byte the file the desktop
would have written, which is asserted in the tests rather than hoped for. Two
differences, both the platform again. A tab cannot ask where a file should go,
so the browser's download manager decides (turn on "ask where to save each
file" to get the dialog back), and the name is taken from what the window is
showing. And the whole file is assembled in memory instead of streamed, so
saving peaks at roughly the stack's own size on top of what is already loaded,
against the 4 GB ceiling above — and the page freezes while it encodes, for the
same no-threads reason the volume builder does. Plugin **exporters** (PNG and
anything a `.dll` adds) are desktop-only: an exporter's contract is that it
writes the file itself, given a path, and there is no path to give it.

Everything else — per-channel contrast and LUTs, ImageJ/OME metadata,
dimension-order correction, playback, all three volume modes, the built-in
plugins — behaves exactly as it does natively, because it *is* the same code.

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
cargo run --release --no-default-features --features renderer-glow,builtin-plugins
```

wgpu is the default: it's the more actively developed backend and preferable on
macOS (Metal, since OpenGL is deprecated there). glow is the portable fallback —
it links only OpenGL (near-universal on Linux) and avoids a Windows 10 idle-CPU
spin that wgpu triggers on some machines. Only the selected backend is compiled
in - the other's dependencies are excluded entirely.


## Deprecated Windows 7 build

Install the Nightly Toolchain — **a pinned one**, see the note below:
```sh
rustup toolchain install nightly-2026-09-08
rustup component add rust-src --toolchain nightly-2026-09-08
```

Compile for the Tier 3 (deprecated) target, **with the OpenGL backend**:
```sh
cargo +nightly-2026-09-08 build --target x86_64-win7-windows-msvc -Z build-std=std,panic_abort --release --no-default-features --features renderer-glow,builtin-plugins
```

> **Why a pinned nightly.** `-Z build-std` compiles the standard library from
> source, so a regression in *std* breaks this build and nothing in this
> repository can work around it. A nightly after 2026-09-08 put
> `#[cfg(not(target_vendor = "win7"))]` on `is_console` in
> `library/std/src/sys/stdio/windows.rs` while leaving its two call sites
> unconditional, so std stops compiling for this target with
> `cannot find function is_console in this scope`. 2026-09-08 is pinned because
> it was tested against this target. Once that `cfg` is gone upstream, try a
> current nightly and drop the pin here and in
> `.github/workflows/release.yml`.


Check a build with — the path is relative to the `FastTIFF` crate, which is
where the test process runs, hence the `..`:
```sh
FASTTIFF_EXE=../target/x86_64-win7-windows-msvc/release/FastTIFF.exe cargo test -p FastTIFF --bin FastTIFF -- --ignored no_combase
```

Released builds run this too — `build-windows7` in `.github/workflows/release.yml`
attaches a `FastTIFF-win7-x86_64.exe` only if it passes.

## Test and lint

Unit Test:
```sh
cargo test --workspace
```

Two things sit outside the workspace on purpose and so are missed by the line
above — both are worked examples rather than parts of the app, and both are
kept building so they cannot quietly rot:

```sh
cargo test --manifest-path plugins/example/Cargo.toml
cargo test -p fast-tiff-viewer --features plugin-netpbm
```

Lint:
```sh
cargo clippy --workspace --all-targets
```

The web build is a different target and has its own `#[cfg]` paths, so lint it
separately:
```sh
cargo clippy --manifest-path FastTIFF-web/Cargo.toml --target wasm32-unknown-unknown --all-targets
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

Layered bottom-up. Each crate below the app is free of any GUI toolkit, so the
same engine can drive a different frontend (the browser build is the second
one):

```text
  fast-tiff-lib     file I/O, IFD index, decode, metadata
        │
  scivis-render     GPU pipelines, textures, ray-marching
        │
  fast-tiff-viewer  stack model, channel settings, decode → GPU sync,
        │           and the plugin host
  FastTIFF          the egui UI (lib) + the native binary
        │
  FastTIFF-web      a browser host around the same UI
```

Off to one side, the plugin contract — three crates that a third party depends
on to build a plugin, and which therefore depend on nothing of ours:

```text
  fasttiff-plugin-abi   the frozen #[repr(C)] contract. Never changes.
  fasttiff-plugin-api   the plain-Rust traits both sides implement
  fasttiff-plugin       the macro that generates the boundary from them
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
- **`fasttiff-plugin-abi/`**, **`fasttiff-plugin-api/`**,
  **`fasttiff-plugin/`** — the plugin contract, split by what may change.
  `-abi` is the frozen binary interface (`#[repr(C)]`, `extern "C"`, zero
  dependencies, layout pinned by tests); `-api` is the ordinary Rust traits a
  plugin author actually writes against; `fasttiff-plugin` is the macro that
  turns one into the other. Only `-abi` is permanent — the other two compile
  *into* the plugin, so they can change freely. All three are MPL-2.0 rather
  than the app's GPL: a plugin SDK that set the licence of every plugin written
  against it would be making a decision that is not this project's to make.
  `plugins/` holds plugin *implementations* that ship as separate
  libraries. `plugins/example/` is the worked example for authors — a filter,
  an importer with a dialog, and a raw-binary reader — and it is the oracle for
  the one test that exercises a real library across the ABI. It is deliberately
  *outside* the workspace, so it is built the way a third party's plugin is
  built rather than as part of the app; that also means `cargo test
  --workspace` does not reach it, and it has its own line in
  [Test and lint](#test-and-lint).

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
- **Copy the view as an image** — the toolbar's copy button, or **Ctrl+C**
  (**⌘C** on macOS). What gets copied differs by mode, because the two modes
  have different right answers:
  - In **2D** it is the *image*: the current frame at its **original size**,
    every enabled channel through its own contrast window and LUT. Not a grab of
    what is on screen — that is at whatever zoom and pan you left it, and on a
    stack too large to upload whole it is only the window currently resident.
    The frame is recomposed on the CPU for the copy, which is also why it does
    not need a working GPU.
  - In **3D** it is the *view*, because for a ray-march there is nothing else to
    copy: there is no original size, and the camera, projection and brightness
    are the picture. So it takes the 3D panel exactly as framed.

  The hotkey yields to whatever has keyboard focus, so Ctrl+C in a text box
  still copies the text.
- Zoom (Ctrl+scroll) and pan (drag, or middle-drag) of the 2D image, with the window sized
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
  other, and this is what says which one you are looking at. **Click it to jump
  there, or drag to sweep the view** — anywhere in the frame rectangle counts,
  not just the inner box, which at high zoom is only a few points across. It
  stays visible on extreme aspect ratios: a 20000 x 4 line scan gets a map with
  a floor on both sides, squashed rather than invisible, and the box's *position*
  in it is still exact even where its shape cannot be.
- **Middle-click to autoscroll**, the gesture browsers and file managers have:
  the cursor becomes a four-way arrow, a ring marks the neutral point, and the
  view drifts towards wherever the pointer sits — further out is faster. Click
  again, or take hold of the picture, or press Escape to stop. It exists because
  a drag can only move the view as far as the hand can travel: crossing a
  20000-pixel mosaic by dragging is a dozen grab-release cycles, where
  autoscroll is one click and a wait. Middle-*drag* still pans directly, as it
  does in the 3D view's CAD mode.
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
- **Brightness is per mode, because the modes do not accumulate the same thing.**
  A MIP takes the brightest sample along each ray, so it is independent of how
  far the ray travelled but runs hot — a maximum over a few hundred samples is a
  high-order statistic, and it reads as blown out next to the same data in 2-D.
  It gets a plain gain, defaulting to **78%**. Volume mode instead accumulates
  `1 - exp(-density · intensity · dt)`, which means dragging **Density** changes
  how *bright* the volume is as much as how *solid* — dark at the low end,
  saturated at the high one. **Auto-level brightness** (on by default) divides
  that out, so density controls solidity alone. It is an approximation, not a
  guarantee: the correction assumes a typical optical thickness, so on a volume
  much thinner or more opaque than typical it flattens the response only partly —
  turn it off and the manual Brightness slider appears.
- **The settings window shows the current mode's controls and hides the others**,
  rather than greying them out: Density belongs to Volume, Threshold to Surface,
  and each mode has its own brightness. It also stays **on top**, so it can be
  left open while you drag the volume underneath it.
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

## Plugins

FastTIFF loads plugins from shared libraries — `.dll` on Windows, `.so` on
Linux, `.dylib` on macOS. **Plugins ▸ Open plugin folder…** opens the folder to
put them in; they appear in the menu on the next start.

Three kinds:

- **Filters** run against the stack that is open. They get the image's shape in
  file coordinates, the file's own scale (pixel size, Z step, frame interval,
  intensity calibration, channel names), the viewer's current display state
  (contrast window, 3D camera), and a plane-at-a-time reader — then return a new
  document, a file to write, or a message. A plugin declares its dialog as a
  list of typed controls and the host draws it, so there is no UI toolkit in a
  plugin's dependency tree.
- **Importers** read formats FastTIFF does not know. An importer declares the
  extensions it handles, and the host adds them to the Open dialog and to
  drag-and-drop *before the plugin has run*; opening such a file calls the
  importer, and what it returns — pixels and the scale it read out of the file —
  becomes an ordinary FastTIFF document.
- **Exporters** write formats FastTIFF does not. The mirror of an importer, and
  the simpler half: the formats it declares become rows in the Save-as dialog,
  and the extension the user ends up with picks which one runs. It is handed the
  path and reads the open stack back through the host, so nothing returns but
  success or a reason — a file that does not exist yet cannot be probed, which
  is why an exporter is chosen by name where an importer is chosen by content.
  Not for saving a stack faithfully; the host does that itself, in TIFF. An
  exporter is for handing the data to something else — a figure, a movie, a
  colleague's software — where losing the axes is usually the point.

The boundary is a frozen C ABI, not Rust's, because Rust has no stable ABI: a
plugin built next year by a different compiler must still load. Nothing crosses
but fixed-width integers, `#[repr(C)]` structs and function pointers; every
struct carries its own size so fields can be appended without breaking either
side; the ABI's major version is part of the exported symbol name, so a version
mismatch is a clean "not a plugin for this FastTIFF" instead of two binaries
disagreeing about a struct layout mid-call. No allocator is shared — the host
copies everything during the call that supplies it. Every enumeration crosses as
a plain integer rather than a Rust enum, because a value from a newer plugin
than the host has heard of has to be *data* to be checked, not undefined
behaviour on arrival. The layouts are pinned by compile-time assertions, so they
are checked on every target the crate builds for rather than only the one that
happened to run the tests.

Writing one is still ordinary Rust: implement a trait, call one macro. See
[`fasttiff-plugin/README.md`](fasttiff-plugin/README.md) for the twenty-line
version, and [`plugins/example/`](plugins/example/src/lib.rs)
for a filter, an importer with a dialog, a raw-binary reader and a CSV
exporter. That crate is
not in the workspace, so build and test it on its own:

```sh
cargo test --manifest-path plugins/example/Cargo.toml
```

### Built-in plugins, and choosing which get compiled in

Some plugins ship *inside* the binary rather than as libraries: Invert, Z
Project, Plot third axis, the three Slice tools, Generate PSF, Deconvolve,
suite2p stabilization, TurboReg stabilization, the PNG importer/exporter, the
Olympus OIR reader and the two MRI importers below.
They are written against the same contract a `.dll` is — being compiled in is a
lane, not a different kind of plugin — and they are what the browser build has,
since it can load nothing.

Each one is a Cargo feature named `plugin-<name>`, and `builtin-plugins` is all
of them. Every one is on by default. To leave some out, name the ones you want:

```sh
cargo build --release -p FastTIFF --no-default-features \
    --features renderer-wgpu,plugin-invert,plugin-zproject
```

A plugin left out is *gone*: no module, no menu entry, and no dependency of its
own compiled or linked. Two carry enough to notice — `plugin-deconvolve` brings
an FFT, and `plugin-stabilize` and `plugin-turboreg` a registration crate
each. Dropping the FFT and suite2p takes about 460 KB off the web bundle. `plugin-dicom` brings a JPEG decoder, for
the two JPEG codecs DICOM actually uses. `plugin-nifti` brings only gzip, which
the Deflate TIFF codec already needs, so it costs nothing — and the same is true
of `plugin-oir`'s XML parser, which OME-XML metadata already pulls in. (A plugin
that reuses a dependency the library needs anyway cannot be dropped from the
graph by turning the plugin off; `cargo tree --invert` shows who the real user
is.) `plugin-stabilize` is also the only
GPL-3 code in the tree, which is why `fast-tiff-viewer` is licensed GPL-3 where
the other library crates are MPL-2.0.

Because these features live in `default`, `--no-default-features` drops the
plugins along with whatever it was really for — notably the glow renderer, which
is how the Linux packages are built. Every such invocation in this repository
names `builtin-plugins` again, and the application refuses to compile with no
plugins at all unless `--features plugins-none` says that was deliberate, so a
forgotten one stops the build rather than shipping a binary that quietly cannot
do half of what it is for.

A worked example lives among them:
[`plugins/builtin/netpbm/`](fast-tiff-viewer/src/plugins/builtin/netpbm/mod.rs)
reads PBM/PGM/PPM as a built-in importer. It is a demonstration rather than a
format this viewer needs, so it is off by default and deliberately not part of
`builtin-plugins`:

```sh
cargo test -p fast-tiff-viewer --features plugin-netpbm
```

The shared-library lane is native-only — there is no `dlopen` in a browser — so
the web build gets the built-ins and no way to add to them.

### Registration: two plugins, two different questions

Both put frames back on top of each other, and which to reach for depends on
*how* they moved.

| | measures | takes out | cost |
|---|---|---|---|
| **suite2p stabilization** | a shift, by phase correlation | drift, plus an optional block-wise deformation | fast |
| **TurboReg stabilization** | a transformation, by optimisation | rotation, scale, shear or a bilinear warp | slower per frame |

A two-photon timelapse that drifted under the objective wants the first. A
histology series where each section was placed on the slide by hand, or a
recording where the stage rotated, wants the second.

TurboReg offers five models — translation, rigid body, scaled rotation, affine
and bilinear — and three targets in time: the previous frame, the first, or the
mean of the stack. The first frame is the default: nothing accumulates down the
series, and the target is a frame that was actually photographed rather than an
average of frames that each need correcting. The stack mean is sharper on a
long, noisy recording. Aligning to the previous frame suits a steady drift but
compounds its own error, and it is not offered for a bilinear warp, which cannot
be chained onto another.

**Pick the smallest model that covers the motion.** More degrees of freedom is
not better, and the cost is measurable. Registering a series built by rotating
and shifting one image, as residual difference between frames afterwards:

| model | ordinary image | self-similar pattern |
|---|---|---|
| Translation | 3.8% → 1.4% | 33.6% → 22.1% |
| Rigid body | 3.8% → **0.075%** | 33.6% → **3.8%** |
| Affine | 3.8% → 0.091% | 33.6% → 19.9% |

Affine contains rigid body, so on an ordinary image it does just as well. On a
*repeating* pattern it collapses, because such an image has a local minimum
everywhere one period lines up with the next and the extra freedom finds them.
A grid, a regular array of wells, a tissue microarray: use the model the motion
actually needs.

**Measured on one channel, applied to all.** A multi-channel recording is one
specimen photographed several ways at once, so the channels moved together: the
transformation is measured on the channel you name — or on their mean, the
default, since averaging raises what the search is matching — and applied to
every channel. Registering each channel on its own would move them relative to
each other, which is the one thing a multi-channel recording must not do. With a
single channel there is nothing to choose and the selector says so.

Both run single- or multi-threaded, selectable in the dialog, and the answer is
the same either way. For TurboReg that is *bit for bit*, not to a tolerance: the
work is split into pieces fixed by the image, never by the thread count, and
summed in a fixed order.

TurboReg spends its threads at two levels: inside each frame (the pixel sums,
the gradient images, the spline prefilters and the final resampling) and across
frames, in every reference mode. Aligning to the previous frame looks like a
chain, but not where the time goes: frame *t* is fitted against the raw frame
*t − 1*, not against frame *t − 1* registered, so the fits are independent.
Only composing them into one transformation per frame is sequential, and that
is a few 3×3 products.

TurboReg also has a **GPU** backend. The search stays on the processor; what
goes to the card is the sum over pixels that scores each candidate — every
pixel's spline interpolation, residual and derivatives — which is where the time
is. Frames are scored in batches, every frame's next candidate in one dispatch,
because one round trip to the card is long next to one candidate's arithmetic.
The coarse pyramid levels, too small to repay a round trip, are fitted on the
CPU threads instead. The card computes in `f32`, so the GPU agrees with the CPU
to a tolerance rather than bit for bit — within a fiftieth of a pixel in the
tests — and repeats itself exactly, run to run, on the same card. It is in the
builds with the wgpu renderer (the default), not in the glow build or the
browser, and the dialog refuses it with the reason wherever it cannot run.

32 frames of 512×512, rigid body, median of five runs, 16 threads and a Quadro
P620:

| reference | single-thread | multi-thread | GPU |
|---|---|---|---|
| first frame | 12.0 s | 1.45 s | 0.79 s |
| previous frame | 7.15 s | 0.87 s | 0.65 s |
| stack mean | 15.5 s | 1.92 s | 0.77 s |
| resampling the result | 0.55 s | 0.08 s | (CPU) |

On that machine the GPU won at every size tried, 256×256 to 1024×1024, by 1.2×
to 2.5× over multi-thread depending on size and target. That is an entry-level
card, but the margin is not a promise: a stronger processor or a weaker card
narrows it, so multi-thread CPU stays the default. The previous frame is the
quickest target here because neighbouring frames start out nearly aligned, and
the search has least to do.

The algorithm is Thévenaz, Rüttimann & Unser's, shipped by EPFL as TurboReg and
StackReg. [`turboreg-registration/`](turboreg-registration/) is an
implementation of it with the API of
[pystackreg](https://github.com/glichtner/pystackreg), **not** a port of that
project's code: pystackreg carries the pre-2023 EPFL terms ("you should not
redistribute it without our consent"), which are not GPL-compatible, whereas
EPFL relicensed TurboReg/StackReg itself to GPL-3-or-later in April 2023. It
does not claim bit-parity with StackReg; see the crate's README for exactly
where it differs and why.

### MRI: NIfTI and DICOM

Two built-in importers read the formats MRI comes in. Both map the scan onto the
viewer's own axes — **z becomes the slice axis**, so a structural scan opens as a
stack you scroll through — and both bring the voxel size with them, so the 3-D
view and the readouts are in millimetres rather than in voxels.

| | reads | notes |
|---|---|---|
| **NIfTI** | `.nii`, `.nii.gz` | NIfTI-1, either byte order. A 4-D fMRI series opens with both a slice and a frame axis, and `pixdim[4]` becomes the frame interval. NIfTI-2 is detected and refused by name rather than mis-read. |
| **DICOM** | `.dcm`, and extensionless scanner exports | Gathers the **whole series** from the folder and orders it by slice position. Reads native (implicit/explicit, either byte order), **RLE**, **baseline JPEG** and **lossless JPEG** (processes 14 and 1/selection-value 1), and expands palette-colour images. |

Three things worth knowing before you rely on either:

- **The slice order is from the geometry, not the file numbering.** DICOM slices
  are sorted by their position along the slice normal
  (`ImagePositionPatient` projected onto the normal of
  `ImageOrientationPatient`), because `InstanceNumber` is acquisition order and
  an interleaved acquisition sorted that way comes out as a comb. When a series
  carries no position the reader falls back to `InstanceNumber` **and says so**,
  since that is exactly when the result might be interleaved.
- **Most compressed DICOM is read; the rest is refused by name.** RLE, baseline
  JPEG and lossless JPEG decode in-process, which covers the common clinical
  export. What is left — JPEG-LS, JPEG 2000, 12-bit extended JPEG, deflated, and
  the encapsulated-but-uncompressed syntax — each produces a message naming the
  syntax it actually is and the command that converts it (`dcmdjpeg`,
  `gdcmconv --raw`, or `dcm2niix` to NIfTI, which this also reads). The
  transfer-syntax UID names *two* things, the element encoding and the pixel
  codec, so a file can be perfectly ordinary except for its codec; the refusal
  says which part it tripped on.

  Lossless JPEG is the one to be slightly careful with, because it is not the
  JPEG most people mean: process 14 is *predictive*, not DCT, and a decoder that
  guesses wrong produces a plausible-looking image made of noise. The test for
  it here is quantitative rather than visual — mean absolute difference between
  horizontally adjacent pixels, which on a correct decode of a skull CT comes out
  at 0.5% of range (anatomy) against 3.7% for a known-good MRI, and at tens of
  percent for a wrong one.
- **Left and right are not guaranteed.** Both formats carry the transform from
  voxel indices to the patient's frame — NIfTI's qform/sform, DICOM's
  orientation cosines — and this viewer shows planes with a voxel size, so there
  is nothing here that could honour a direction matrix. The planes are in the
  file's own storage order. If laterality matters for what you are doing, it
  matters more than this viewer can promise.

Nothing identifying is read from a DICOM. `PatientName`, `PatientID` and the
rest are in every one of those files and none of them are touched: the document
is named after the file on disk, and its description comes from `Modality` and
`SeriesDescription`. A viewer that put a patient's name in a window title would
put it in screenshots and in bug reports.

Both are importers, so both are desktop-only — an importer is handed a
filesystem path, and a browser tab has none.

## What it doesn't do (intentionally out of scope for a "viewer")

ROIs and measurements. Image processing and export are not built in either, but
are what the [plugin interface](#plugins) exists for — the render pipeline
separates "decode" from "display" cleanly enough that a plugin can read the
former without disturbing the latter.

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
- Done: TurboReg stabilization plugin (rigid/affine/bilinear) + turboreg-registration crate
- Done: copy the view as an image (2D frame at original size, 3D as framed), button + Ctrl/Cmd+C
- Done: built-in plugins are per-plugin Cargo features, each with its own deps
- Done: universal (arm64 + x86_64) macOS binary, so there is one download
- Done: minimap has a minimum size, and click/drag moves the field of view
- Done: sticky autoscroll mode on the middle mouse button (2D)
- Done: NIfTI importer (.nii, .nii.gz)
- Done: DICOM importer (.dcm), with RLE, baseline JPEG and lossless JPEG
- Done: per-mode 3D brightness, auto-levelled against Density in Volume mode
- Done: 3D settings window is mode-aware and stays on top
- publish in Brew
- Add 3D volume save in best suitable format for the gpu rendering (not necessary now, need to study the idea)



<img width="1137" height="923" alt="Untitled" src="https://github.com/user-attachments/assets/e3fe4619-bd53-4b68-bf7f-019231cf20a6" />

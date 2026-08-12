# scivis-render

[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/scivis-render/LICENSE)

**Sci**entific **vis**ualization rendering, on the GPU: per-channel
window/level → LUT compositing of multi-channel image stacks in 2D, and volume
ray-marching in 3D.

If you have 16-bit multi-channel or volumetric data — a confocal z-stack, a CT
series, a spectral cube — this draws it the way the field expects: independent
contrast and colormap per channel, composited in one pass.

No GUI toolkit and no file format. It never creates a device, a surface or a
window, and never draws a frame on its own — you bring the pixels and the render
pass. That makes it equally at home under a native desktop app, a headless
encoder, or a browser canvas.

Extracted from [FastTIFF](https://github.com/konung-yaropolk/FastTIFF), but it
has no TIFF dependency of any kind: it takes `&[u8]`, `&[u16]` and `&[f32]`, so
Zarr, HDF5, NIfTI, DICOM or a synthetic array work just as well.

## What it does

- **Composites up to 6 channels** in one pass: each channel is sampled, windowed
  (min/max), colored through its own 256-entry LUT, and the results summed.
  Zoom-out uses a bounded box filter so minification doesn't shimmer.
- **Uploads each channel in its cheapest format.** 8-bit sources go to `R8Uint`
  with no CPU widening; 16-bit and rescaled integers to `R16Uint`; 32-bit floats
  to `R32F`, with window/level applied on the GPU in the data's own units. The
  unused texture of each pair shrinks to a 1×1 dummy.
- **Ray-marches a 3D volume** in three modes — maximum-intensity projection,
  emission-absorption alpha compositing (the ImageJ 3D Viewer's "Volume" look),
  and a gradient-shaded isosurface — with nearest / trilinear / in-shader
  tricubic sampling, anisotropic voxel scale, and per-channel LUT compositing.
- **Pans and zooms via UVs**, not an oversized viewport (which a backend would
  clamp to the framebuffer, squashing the image instead of zooming).

## Backends

Two, selected by feature and **additive** — nothing stops you compiling both:

| Feature | Targets |
|---|---|
| `backend-wgpu` | DX12 / Vulkan / Metal / WebGPU |
| `backend-glow` | OpenGL 3.x / WebGL2 |

With neither feature you still get the shared parameter types — enough for
CPU-side code that budgets volume memory or computes window/level.

Both backends expose the **same inherent method set** on their
`ImageRenderResources`, so a host picks one with a `#[cfg]` re-export and writes
its call sites once. There is deliberately **no** `Renderer` trait: the two
`paint` methods can't share a signature (a wgpu render pass vs. a live GL
context), and a cfg-selected façade already gives static dispatch for free.

---

# API

Everything lives on one type, `ImageRenderResources`, in either
`scivis_render::wgpu_backend` or `scivis_render::glow_backend`.

## Lifecycle

The API is a state machine in three tiers, from least to most frequent:

```text
  new()                                   once, at startup
    └─ ensure_size(w, h, kinds)           when the image shape or channel layout changes
         └─ upload_lut(ch, lut)           when a channel's colors change
              └─ upload_channel_*()       per frame
                 set_params(...)
                 paint(...)
```

Calls at a coarser tier invalidate finer ones: `ensure_size` reallocates
textures, so every channel must be re-uploaded after it changes anything.
`ensure_size` is cheap to call every frame — it compares the requested size and
per-channel kinds against what's allocated and returns immediately if they
match, so you can call it unconditionally.

**Mutability tells you the cost.** Uploads take `&self` — they only queue GPU
writes. Anything that can reallocate or restage takes `&mut self`.

## Construction

```rust,ignore
// wgpu — Device and Queue are refcounted handles, so these clones cost nothing.
// `target_format` must match the color attachment you will later paint into;
// the pipelines are compiled against it.
let mut r = wgpu_backend::ImageRenderResources::new(device.clone(), queue.clone(), target_format);

// glow — the Arc is kept, so later uploads need no context argument.
let mut r = glow_backend::ImageRenderResources::new(gl.clone());
```

## Shared types

| Type | Meaning |
|---|---|
| `MAX_CHANNELS` | `6`. Channels past this are ignored by every method. |
| `Lut` | `[[u8; 3]; 256]` — RGB indexed by windowed intensity. |
| `ChannelKind` | `Int8` / `Int16` / `Float` — which texture a channel uploads to. |
| `ChannelUniform` | `{ min, max, enabled, is_float }` for one channel. |
| `VolumeKind` | `U8` / `U16` / `F32` — a volume channel's sample format. |
| `VolumeInterp` | `Nearest` / `Linear` / `Cubic`. |
| `VolumeRender` | `Mip` / `Alpha` / `Surface`. |
| `VolumeParams` | Camera basis + per-channel windows + mode, for one 3D frame. |

## 2D image

```rust,ignore
fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind])
```
Allocate for an image of `width × height` with one `ChannelKind` per display
channel. Channels past `kinds.len()` become unused (1×1 dummies). No-op when
nothing changed; otherwise reallocates and **invalidates all uploaded pixels**.

```rust,ignore
fn upload_channel_u8 (&self, channel: usize, width: u32, height: u32, samples: &[u8])
fn upload_channel_u16(&self, channel: usize, width: u32, height: u32, samples: &[u16])
fn upload_channel_f32(&self, channel: usize, width: u32, height: u32, samples: &[f32])
```
Upload one channel's pixels, row-major, tightly packed, `width * height` samples.

**Use the variant matching the `ChannelKind` you declared.** The kinds passed to
`ensure_size` are a contract: an `Int8` channel must be fed by
`upload_channel_u8`, and so on. A mismatch is not a graceful no-op — depending
on which way it goes you get a GPU validation error (uploading a full-size image
into the 1×1 dummy that the unused format was shrunk to) or garbled pixels
(feeding 16-bit data to an 8-bit texture). Out-of-range `channel` *is* ignored
silently.

```rust,ignore
fn upload_lut(&self, channel: usize, lut: &Lut)
```
Set one channel's color table. Sticky — upload once and it persists across
frames until `ensure_size` reallocates.

```rust,ignore
fn set_params(&mut self, channels: &[ChannelUniform], num_channels: u32,
              uv_offset: [f32; 2], uv_scale: [f32; 2])
```
Per-frame state: contrast windows, on/off flags, how many channels to composite,
and the visible sub-rect. `sampled_uv = uv_offset + uv * uv_scale`, so
`([0,0], [1,1])` shows the whole image; a smaller `uv_scale` zooms in.

**Window/level units follow the texture, not the source.** This is the one place
it's easy to go wrong:

| `ChannelKind` | `min`/`max` are in |
|---|---|
| `Int8` | raw `0..255` |
| `Int16` | raw `0..65535` |
| `Float` | the data's own float units |

If your UI keeps one `0..65535` slider for every channel (as FastTIFF does),
divide by `257` before handing an `Int8` channel's window over.

```rust,ignore
fn paint(&self, render_pass: &mut wgpu::RenderPass)   // wgpu
fn paint(&self, gl: &glow::Context)                   // glow
```
Draw the composited image. Assumes **you** have already set the viewport and
scissor to the target rect — that's host state, and this crate won't touch it.

## 3D volume

```rust,ignore
fn max_3d_texture_size(&self) -> u32
```
The device's per-axis 3D-texture limit. Subsample your volume to fit it before
uploading; there is no automatic downscale.

```rust,ignore
fn upload_volumes(&mut self, w: u32, h: u32, d: u32, channels: &[(VolumeKind, Vec<u8>)])
```
Upload the whole volume, one entry per channel, each `w * h * d` samples in
native-endian bytes of its `VolumeKind`, z-major. Channels past `channels.len()`
shrink to 1×1×1 dummies. Re-uploading the same shape and kind refills in place
with no reallocation — which is what makes stepping through 4D timepoints cheap.

```rust,ignore
fn volume_gpu_bps(kind: VolumeKind) -> usize   // free function, per backend
```
GPU bytes per sample on this backend — wgpu stores everything as 16-bit, glow
stores natively. Budget on `max(cpu_bps, volume_gpu_bps(kind))` so an 8-bit
source doesn't silently double past your limit in VRAM.

```rust,ignore
fn set_volume_interp(&mut self, interp: VolumeInterp)
fn set_volume_params(&mut self, params: VolumeParams)
```
Sampling mode and the per-frame camera + windows. Both are cheap and idempotent;
call them every 3D frame.

`VolumeParams` window units differ from the 2D path: **integer volumes are
unorm-normalized**, so divide a `0..65535` window by `65535`. Float volumes keep
their own units. The camera arrives as an explicit basis (`eye`, `forward`,
`right`, `up`, `tan_half_fov`) rather than matrices, so the shader builds rays
with no matrix inverse.

```rust,ignore
fn paint_volume(&self, render_pass: &mut wgpu::RenderPass)   // wgpu
fn paint_volume(&self, gl: &glow::Context)                   // glow
```

## Backend-specific extras

Two members exist only on `wgpu_backend`:

```rust,ignore
fn write_volume_uniform(&self)
```
**Ordering requirement:** must run once per 3D frame *before* `paint_volume`.
`set_volume_params` only stashes; this is what reaches the GPU. Under
`egui_wgpu` that's the callback's `prepare` step, then `paint`. The glow backend
has no separate uniform buffer and applies everything inside `paint_volume`, so
it needs no equivalent.

```rust,ignore
fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features
```
The optional device features worth requesting, intersected with what the adapter
offers — pass as `required_features` at device creation. Today that's
`TEXTURE_FORMAT_16BIT_NORM`, which keeps 16-bit volume data at full precision
instead of rounding through f16's ~11 bits. Everything degrades cleanly without
it, so this is a quality knob, not a requirement.

Also `BACKEND: &str` on both (`"wgpu"` / `"glow"`), for a UI that wants to show
which one is live.

## Worked example

```rust,ignore
use scivis_render::{wgpu_backend::ImageRenderResources, ChannelKind, ChannelUniform};

let mut r = ImageRenderResources::new(device.clone(), queue.clone(), target_format);

// Two 16-bit channels, red and green.
let kinds = [ChannelKind::Int16, ChannelKind::Int16];
r.ensure_size(width, height, &kinds);
r.upload_lut(0, &red_lut);
r.upload_lut(1, &green_lut);

// Per frame:
r.upload_channel_u16(0, width, height, &ch0);
r.upload_channel_u16(1, width, height, &ch1);
r.set_params(
    &[
        ChannelUniform { min: 100.0, max: 4000.0, enabled: true, is_float: false },
        ChannelUniform { min: 0.0,   max: 8000.0, enabled: true, is_float: false },
    ],
    2,
    [0.0, 0.0],  // uv_offset — whole image
    [1.0, 1.0],  // uv_scale
);

// ...inside your own render pass:
r.paint(&mut render_pass);
```

## Threading

On native platforms `ImageRenderResources` is both `Send` and `Sync` — asserted
at compile time in `tests/markers.rs`, so a dependency bump that drops either
fails this crate's build rather than yours. On wasm the glow backend is neither,
because a WebGL context is bound to one thread.

It is **not internally synchronized**, though, and the methods that reallocate
take `&mut self`. If your drawing layer needs a shared `'static` callback (as
egui's paint callbacks do), wrap it in `Arc<Mutex<…>>` on your side — see
`FastTIFF/src/render.rs` for a worked adapter. Uploads and painting are expected
to happen on the same thread and never overlap.

## Version pinning

You hand over your own `wgpu::Device`/`Queue` (or `glow::Context`), so these must
resolve to the **same** crate versions your windowing layer uses — `egui-wgpu`
0.34 → wgpu 29, `egui_glow` 0.34 → glow 0.17. Cargo unifies them automatically
while the majors match; a host on a different major gets a type mismatch at the
`new()` call, not a subtle bug.

## Testing

The WGSL shaders are validated offline with `naga` — the same front-end wgpu uses
at runtime — so a broken shader is a failing `cargo test` rather than a blank
canvas at startup. No GPU required, and it runs regardless of which backend
feature is enabled.

## License

MPL-2.0. File-level copyleft: changes to these files stay open, but a consuming
application doesn't have to be MPL.

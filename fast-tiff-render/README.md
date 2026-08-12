# fast-tiff-render

[![License](https://img.shields.io/badge/License-MPL--2.0-green)](https://github.com/konung-yaropolk/FastTIFF/blob/main/LICENSE)

GPU rendering for multi-channel scientific image stacks: per-channel
window/level → LUT compositing in 2D, and volume ray-marching in 3D.
**No GUI toolkit** — it never creates a device, a surface or a window, and never
draws a frame on its own.

It's the rendering half of [FastTIFF](https://github.com/konung-yaropolk/FastTIFF),
split out so the same renderer can sit under a native desktop app and a browser
canvas. Pair it with [`fast-tiff-lib`](https://crates.io/crates/fast-tiff-lib)
(file → pixels) and `fast-tiff-core` (the viewer model that drives both).

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

Two, selected by feature and additive — nothing stops you compiling both:

| Feature | Targets |
|---|---|
| `backend-wgpu` | DX12 / Vulkan / Metal / WebGPU |
| `backend-glow` | OpenGL 3.x / WebGL2 |

Both expose the same inherent method set on their `ImageRenderResources`, so a
host picks one with a `#[cfg]` re-export and writes its call sites once. There
is deliberately **no** `dyn Renderer` trait: the two `paint` methods can't share
a signature (a wgpu render pass vs. a live GL context), and the cfg-selected
façade already gives static dispatch for free.

With neither feature you get just the shared parameter types — enough for
CPU-side code that budgets volume memory or computes window/level.

## Usage

```rust,ignore
use fast_tiff_render::{wgpu_backend::ImageRenderResources, ChannelKind, ChannelUniform};

// Once, from whatever created your device (eframe, winit + wgpu, a canvas…).
// `Device`/`Queue` are refcounted handles, so these clones cost nothing.
let mut r = ImageRenderResources::new(device.clone(), queue.clone(), target_format);

// Whenever the stack or its pixel layout changes:
r.ensure_size(width, height, &[ChannelKind::Int16, ChannelKind::Int16]);
r.upload_lut(0, &lut);

// Per frame:
r.upload_channel_u16(0, width, height, &pixels);
r.set_params(&uniforms, 2, uv_offset, uv_scale);

// …then, inside your own render pass:
r.paint(&mut render_pass);
```

Uploads take `&self` (they only queue GPU writes); anything that can reallocate
or restage takes `&mut self`. For the 3D view, call `write_volume_uniform()`
before `paint_volume()` — under `egui_wgpu` that's the callback's `prepare` then
`paint`.

## Version pinning

The host hands over its own `wgpu::Device`/`Queue` (or `glow::Context`), so
these must resolve to the **same** crate versions the host's windowing layer
uses — `egui-wgpu` 0.34 → wgpu 29, `egui_glow` 0.34 → glow 0.17. Cargo unifies
them automatically while the majors match; a host on a different major gets a
type mismatch at the `new()` call, not a subtle bug.

## Testing

The WGSL shaders are validated offline with `naga` — the same front-end wgpu
uses at runtime — so a broken shader is a failing `cargo test` rather than a
blank canvas at startup. No GPU required, and it runs regardless of which
backend feature is enabled.

## License

MPL-2.0. File-level copyleft: changes to these files stay open, but a consuming
application doesn't have to be MPL.

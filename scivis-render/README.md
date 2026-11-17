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

With neither feature you still get the shared parameter types and the two
renderer traits — enough for CPU-side code that budgets volume memory,
computes window/level, or is generic over a renderer it is handed.

Both backends' `ImageRenderResources` implement two traits, `ImageRenderer`
(the 2D composite) and `VolumeRenderer` (the 3D view, which includes the
first), so a host writes its per-frame code once, generic over them, and names
a backend only where it constructs one and hands it to its drawing layer.
Generic, not `dyn`: the host's code is compiled for the backend it is given,
and a call costs what a call on the backend's own type does. What cannot share
a signature stays on each type: construction (a wgpu device, or a GL context)
and painting (into a wgpu render pass, or with a live GL context).

What both backends work out on the CPU is done once, in this crate, and both
consume it: the composite's uniform block (`uniforms::CompositeParams`, laid out
as both shading languages read it), the LUT texels, the volume's acceleration
uniform, and how a device stores a volume (`VolumeCaps`). The backends differ
only in what they hand the GPU, and where they store volumes differently, on
purpose, `VolumeCaps` says so.

---

# API

Each backend is one type, `ImageRenderResources`, in either
`scivis_render::wgpu_backend` or `scivis_render::glow_backend`. Most of what it
does is the two traits below, `ImageRenderer` and `VolumeRenderer`; bring them
into scope (`use scivis_render::{ImageRenderer as _, VolumeRenderer as _};`) to
call their methods on a backend's own type, or write a function generic over
them:

```rust,ignore
fn show<R: scivis_render::ImageRenderer>(r: &mut R, frame: &Frame) {
    r.ensure_size(frame.width, frame.height, &frame.kinds);
    // ...
}
```

Construction and painting are each type's own (see Construction, and
`paint` / `paint_volume` below), since they take the backend's GPU objects.

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

`ensure_size` reallocates the channel textures, so every channel's pixels
must be re-uploaded after it changes anything; the LUTs survive it.
`ensure_size` is cheap to call every frame — it compares the requested size and
per-channel kinds against what's allocated and returns immediately if they
match, so you can call it unconditionally.

Everything that changes what the renderer holds takes `&mut self`, uploads
included. The tiers above, not the receiver, say what a call costs: the two
backends only queue an upload's write, and only `ensure_size` and a volume
upload allocate. A `&self` upload would have made any renderer that keeps
its pixels itself (a test double, a CPU compositor) reach for interior
mutability.

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
| `MAX_CHANNELS` | `6`. Channels past this are ignored by every method. The shaders take it (and the other numbers they share with Rust) from `shaders`, which assembles their sources; see Testing. |
| `Lut` | `[[u8; 3]; 256]` — RGB indexed by windowed intensity. |
| `ChannelKind` | `Int8` / `Int16` / `Float` — which texture a channel uploads to. |
| `ChannelUniform` | `{ min, max, enabled, is_float }` for one channel. |
| `VolumeKind` | `U8` / `U16` / `F32` — a volume channel's sample format. |
| `VolumeInterp` | `Nearest` / `Linear` / `Cubic`. |
| `VolumeRender` | `Mip` / `Alpha` / `Surface`. |
| `VolumeParams` | Camera basis + per-channel windows + mode, for one 3D frame. |
| `VolumeCaps` | What a device holds of a volume: its 3D limit, and how it stores each kind of sample. |
| `uniforms::CompositeParams` | The 2D composite's uniform block, as both backends hand it to their shaders. |

## 2D image

`ImageRenderer`:

```rust,ignore
fn max_2d_texture_size(&self) -> u32
```
The device's per-axis 2D-texture limit, asked once when the backend is made.
A frame past it cannot be uploaded at all: subsample or window it to fit, as
FastTIFF's viewer does.

```rust,ignore
fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind])
```
Allocate for an image of `width × height` with one `ChannelKind` per display
channel. Channels past `kinds.len()` become unused (1×1 dummies). No-op when
nothing changed; otherwise reallocates and **invalidates all uploaded pixels**.

```rust,ignore
fn upload_channel_u8 (&mut self, channel: usize, width: u32, height: u32, samples: &[u8])
fn upload_channel_u16(&mut self, channel: usize, width: u32, height: u32, samples: &[u16])
fn upload_channel_f32(&mut self, channel: usize, width: u32, height: u32, samples: &[f32])
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
fn upload_lut(&mut self, channel: usize, lut: &Lut)
```
Set one channel's color table. Sticky — upload once and it persists across
frames until it is replaced. `ensure_size` leaves the LUTs alone (it
reallocates only the channel textures), which is what lets FastTIFF's viewer
resize to another window of a large frame without sending its LUTs again.

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

Both backends pack these into one `uniforms::CompositeParams`: the wgpu one
writes it to its uniform buffer here, the glow one keeps it and sets it as
uniforms when it paints.

```rust,ignore
fn paint(&self, render_pass: &mut wgpu::RenderPass)   // wgpu, inherent
fn paint(&self, gl: &glow::Context)                   // glow, inherent
```
Draw the composited image. Assumes **you** have already set the viewport and
scissor to the target rect — that's host state, and this crate won't touch it.

## 3D volume

`VolumeRenderer`, which includes `ImageRenderer`: the volume is drawn through
the same LUTs.

```rust,ignore
fn volume_caps(&self) -> VolumeCaps
```
What the device holds of a volume, asked once when the backend is made:
`max_3d_size`, its per-axis 3D-texture limit (reduce your volume to fit it
before uploading; there is no automatic downscale), and whether it filters
16-bit unorm (`unorm16`) and full-float (`float32`) textures, which decides
how each kind of sample is stored.

### Packing a volume (`pack`)

A volume reaches the GPU **packed**: up to four channels share one RGBA texture
(channels 5 and 6 a second one), so a single texture read returns every channel
at a point. The packed volume also carries a mip chain and per-brick maxima.
Building all that costs time in proportion to the volume, so do it where it
won't stall a frame — FastTIFF does it on its volume-building thread — and the
upload is then a plain copy.

```rust,ignore
VolumeCaps::texel_format(&self, kind: VolumeKind) -> pack::TexelFormat
VolumeCaps::check(&self, layout: &pack::VolumeLayout) -> Result<(), VolumeUploadError>
```
How this device stores a volume of `kind` (use `pack::common_kind` for mixed
channels), and whether a packed volume is one it can take. The backends differ
here on purpose. glow: native widths (`Unorm8`, `Unorm16`, `Float32`), since
OpenGL 3 filters all three as core. wgpu: 8-bit → `Unorm8`; 16-bit → `Unorm16`
with the `TEXTURE_FORMAT_16BIT_NORM` feature, else `Float16`; float → `Float16`,
since a full float is filterable only under `FLOAT32_FILTERABLE`, which many
devices lack and the backend does not ask for.

```rust,ignore
let layout = pack::VolumeLayout::new((w, h, d), channels, format);
let mut volume = pack::PackedVolume::new(layout);
// fill volume.level0_mut() — slice z of texture t is the `layout.slice_bytes(t)`
// chunk at z, so slices can be written in parallel with chunks_mut — then:
volume.finish(); // mip chain + brick maxima
// or, with whole channels in hand: PackedVolume::from_channels(dims, &channels, format)
```
Values are in **texture units**: `raw/255` or `raw/65535` for integer sources,
the raw value for float ones. `pack::bytes_per_voxel(channels, format)` is what
a budget divides by (packing padding and the mip chain included).

The mip levels hold the **maximum** of the voxels below them, not the mean: they
are read by the maximum-intensity projection where a voxel is smaller than a
pixel, and averaging would dim the thin bright structures a MIP is for.

```rust,ignore
fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError>
fn upload_volumes(&mut self, w: u32, h: u32, d: u32, channels: &[(VolumeKind, Vec<u8>)])
    -> Result<(), VolumeUploadError>   // packs on the calling thread, then uploads
```
Re-uploading the same shape and format refills in place with no reallocation —
which is what makes stepping through 4D timepoints cheap; a volume of another
shape frees the old one's memory before it asks for its own. A volume whose
buffers are not the sizes its layout says (`PackedVolume::check`: one never
`finish`ed, say), or that `VolumeCaps::check` refuses, is refused as
`Unsupported`. Fails with
`VolumeUploadError::OutOfMemory` when the GPU has no room (wgpu reports it per
texture, before anything is written to it), leaving placeholders and no volume:
build a smaller one and try again. Don't rely on that alone, though: a driver
may overcommit rather than fail (DX12 does, by several GB on a 2 GB card, and
OpenGL only fails for absurd sizes).

```rust,ignore
fn release_volume(&mut self)
```
Give the volume's GPU memory back, placeholders in its place, for a host whose
volume has no further use (the stack it was built from closed or replaced).

```rust,ignore
fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)>
fn volume_pending(&mut self) -> bool
```
In a browser, WebGPU says whether there was room through a promise an upload
cannot wait for. The upload then returns `Ok`, the volume is not drawn until
the answer is in (a texture there was no room for would invalidate the whole
frame, the host's UI included), and a refusal turns up here, with the bytes
the volume took: the volume is gone, as after a failed upload. Ask once a
frame. Always `None` elsewhere. While `volume_pending` says the answer is not
in, keep frames coming: nothing else asks for the one that draws the volume.

Size the volume against:

```rust,ignore
fn volume_memory_available(&self) -> Option<u64>
fn set_adapter(&mut self, adapter: wgpu::Adapter)   // wgpu only, inherent: what to ask
```
GPU memory a new volume could take: the driver's budget less what the process
uses, plus what the current volume holds (a new one replaces it). wgpu asks
`VK_EXT_memory_budget` on Vulkan and DXGI on DX12 (and DXGI for Vulkan on
Windows when the extension is missing); glow asks `GL_NVX_gpu_memory_info` or
`GL_ATI_meminfo`. `None` where there's no way to ask (Metal, the browser), the
wgpu adapter wasn't handed over, or the GPU is integrated: its memory is the
host's RAM, which the volume already takes once on the CPU side, and the
drivers' figures for it are the carve-out (or nothing). Take a share of it,
not all of it.

### Drawing it

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
with no matrix inverse. `max_steps` caps the samples per ray: `max_steps(pixels,
channels, interp)` keeps a frame within `FRAME_SAMPLE_BUDGET` reads, so no frame
runs long enough for the system to reset the GPU.

```rust,ignore
fn set_volume_target(&mut self, width_px: u32, height_px: u32, downscale: u32)
```
The size the view is painted at, and how many times smaller per axis to render
it: `MOVING_DOWNSCALE` (4) while the user is moving the view renders a sixteenth
of the rays into a texture of its own, which `paint_volume` scales up; 1 renders
every pixel. Set it every frame. Until it is set, wgpu does not know how large
a pixel is and reads no coarser mip level. (glow takes the size from the
viewport the host has set and uses only `downscale`.) Work out
`VolumeParams::max_steps` from the full size even when reduced: a reduced frame
then takes as many steps per ray as a sharp one and a sixteenth of the work.

```rust,ignore
fn set_volume_acceleration(&mut self, skip_empty: bool, mip_levels: bool)
```
Both on unless switched off, for comparing, measuring or ruling them out:

- **Empty-space skipping.** From the brick maxima and the current windows the
  renderer works out which 16³ bricks can contribute anything (grown by one
  brick, so filtering across a face is covered), and rays jump over the rest,
  staying on the sample lattice they would have walked. **Exact**: the picture
  is the same either way, in every mode.
- **Mip levels** (MIP only). Where a voxel is smaller than a pixel across the
  ray at the ray's nearest sample, the ray reads a coarser level and steps
  accordingly. Conservative on anisotropic stacks: a thick-sliced stack seen
  from the side keeps its full resolution.

A MIP ray also stops once every channel is at the top of its window.

```rust,ignore
fn prepare_volume(&mut self, encoder: &mut wgpu::CommandEncoder)   // wgpu, inherent
fn paint_volume(&self, render_pass: &mut wgpu::RenderPass)         // wgpu, inherent
fn paint_volume(&mut self, gl: &glow::Context)                     // glow, inherent
```
**wgpu ordering:** `prepare_volume` must run once per 3D frame *before*
`paint_volume`: it writes the uniform and, when reduced, records the reduced
render into `encoder` so it runs ahead of your pass. Under `egui_wgpu` that's
the callback's `prepare` (with the encoder it is handed), then `paint`.
`write_volume_uniform(&mut self)` remains for a host that never reduces the
view. glow does it all in `paint_volume`, inside the viewport the host has set,
and leaves the host's framebuffer, viewport and scissor as they were.

```rust,ignore
fn capture_volume(&mut self, width: u32, height: u32) -> Result<(), String>
fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>>
```
Render the volume off screen at any size and read it back, for a still or a
movie frame sharper than the window. Every pixel is a ray, so a larger render
has more detail, not stretched pixels. The aspect ratio comes from the size,
and nothing the host draws over its view is included. Call `capture_volume`
after `set_volume_params`, then call `take_volume_capture` once per frame until
it returns the RGBA picture. On wgpu the readback lands a frame or more later,
and a browser cannot wait for it; glow reads back at once. Only one capture can
be in flight at a time. Always full size, whatever `set_volume_target` says.

A capture is rendered in tiles of `CAPTURE_TILE` (1024) pixels, each read back
before the next is drawn. Each GPU job then costs about what a frame of the view
costs, however large the picture. Drawn in one go, an 8192-pixel frame of a
large volume can run long enough for Windows to reset the GPU, which loses the
device. GPU memory holds one tile, not the whole picture. `capture_tiles` and
`tile_params` are public if you want to render the tiles yourself.

## Backend-specific extras

Inherent on each type, beside `new`, `paint` and `paint_volume`. Only on
`wgpu_backend`, as well as `prepare_volume`, `write_volume_uniform` and
`set_adapter`:

```rust,ignore
fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features
```
The optional device features worth requesting, intersected with what the adapter
offers — pass as `required_features` at device creation. Today that's
`TEXTURE_FORMAT_16BIT_NORM`, which keeps 16-bit volume data at full precision
instead of rounding through f16's ~11 bits. Everything degrades cleanly without
it, so this is a quality knob, not a requirement.

Also `BACKEND: &str` on both (`"wgpu"` / `"glow"`), for a UI that wants to show
which one is live. A volume's bytes per sample are those of the format
`volume_caps().texel_format(kind)` returns (`TexelFormat::bytes`); a 2D
channel's are `ChannelKind::bytes_per_sample`.

Features: `backend-wgpu` and `backend-glow` (additive), and `rayon` to build a
packed volume's mip chain and brick maxima across threads.

## Worked example

```rust,ignore
use scivis_render::{wgpu_backend::ImageRenderResources, ChannelKind, ChannelUniform};
use scivis_render::ImageRenderer as _;

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
at compile time in `tests/it/markers.rs`, so a dependency bump that drops either
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

Every shader is written twice, in WGSL (`src/shaders/*.wgsl`) and in GLSL 1.40
(`src/shaders/*.glsl`), and neither file is complete on its own: the numbers
both share with Rust — `MAX_CHANNELS`, the step cap, the isosurface look, the
mode codes — are defined once in `src/shaders.rs` and written in ahead of each
when a backend builds its pipelines. `cargo test -p scivis-render` builds both
backends whatever features are asked for, and checks:

* the WGSL, as assembled, with `naga` — the front-end wgpu uses at runtime —
  with no optional capabilities (`tests/it/wgsl.rs`);
* the GLSL, as assembled, compiled and linked by glslang, Khronos' reference
  compiler, held to the version it declares, and the interface between each
  program's two stages, which glslang leaves to the driver (`tests/it/glsl.rs`).
  glslang is a dev-dependency built from C++ by `cc`, once per target
  directory and profile: 12 s debug and 35 s release on 16 threads;
* that the Rust uniform structs are laid out field for field as naga lays out
  the WGSL ones: names, offsets, sizes and types (`src/wgpu_backend_tests.rs`),
  and the composite's shared one as the GLSL reads it, a `vec4` a channel
  (`src/uniforms_tests.rs`);
* what both backends work out on the CPU, against what each one's own copy of
  it did: the uniform block, the LUT texels, the acceleration uniform, when the
  textures are remade, the occupancy worked out again and a capture may start
  (`src/uniforms_tests.rs`, `src/cpu_state_tests.rs`), and the texel policy,
  `VolumeCaps` as a function of the kind, for caps written out as each
  backend reports them (`src/renderer_tests.rs`);
* that the per-channel lines both languages make us write out by hand cover
  exactly `MAX_CHANNELS` channels, each line one channel's alone, and read each
  from where the volume packer put it (`tests/it/unrolled.rs`).

None of these needs a GPU. Three more do. `tests/it/gl.rs` has the glow backend
build its programs on the machine's own OpenGL driver (a headless WGL context
on Windows, surfaceless EGL on Linux, GL 3.1 or later) and draw a fixed scene
in every mode, with no GL error. `tests/it/parity.rs` draws that scene, a 2D
composite and a volume of `MAX_CHANNELS` channels each, in every render and
sampling mode, through both backends and compares the pictures within
tolerances it documents: the check that the two copies of each shader still
mean the same thing. That one also needs a hardware wgpu adapter.
`tests/it/caps.rs` holds each backend, made on a real device, to the caps it
reports (set in its own `new`, which nothing else reads), and draws the
scene's volume as 16-bit and as float samples standing for the same values,
held to the 8-bit picture: the only draw of those volume textures in the
suites. Without what they need, these skip, saying so on the test run's
output whether it passes or not; with `SCIVIS_RENDER_REQUIRE_GL=1` in the
environment, a missing GL context fails them instead, as the Linux CI job has
it.

A copy of this crate from crates.io builds only the backends it is asked
for, and runs the suites those allow: the GPU ones need `backend-glow`, and
`parity` and the wgpu half of `caps` `backend-wgpu` as well.

## License

MPL-2.0. File-level copyleft: changes to these files stay open, but a consuming
application doesn't have to be MPL.

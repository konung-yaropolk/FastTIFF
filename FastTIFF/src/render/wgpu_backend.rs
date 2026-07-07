//! wgpu rendering for the composited image: the render pipeline, two per-channel
//! texture sets (R16Uint for integer sources, R32F for 32-bit-float sources),
//! the LUT texture array, and the uniform buffer carrying per-channel
//! window/level + enabled + is-float flags. Uploads happen from `app.rs` via
//! `UploadCtx` (the device + queue from `Frame::wgpu_render_state`); `paint` is
//! invoked from the `egui_wgpu::CallbackTrait` built by `paint_callback`.
//!
//! Each channel is either integer or float: the one carrying its data is a
//! full-size texture, the other a 1x1 dummy (`is_float` in the uniform tells the
//! shader which to sample). Float channels skip the per-frame CPU rescale —
//! window/level is done on the GPU in the data's own units.
//!
//! Unlike egui's `custom3d_wgpu` example (which parks resources in egui_wgpu's
//! `CallbackResources` map), we keep them in an `Arc<Mutex>` owned by the app and
//! captured by the callback — matching the glow backend so the two are
//! interchangeable behind one app-side interface.
//!
//! The 3D volume view shares this handle: per-channel R16Float 3D textures (all
//! channels normalized to a common unit, so window/level applies directly), a
//! separate ray-march pipeline (`volume.wgsl`), and its own uniform. R16Float is
//! core-filterable, so linear/cubic sampling use the hardware sampler.

use super::{ChannelKind, ChannelUniform, VolumeInterp, VolumeKind, VolumeParams, MAX_CHANNELS};
use eframe::egui_wgpu::{self, wgpu};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

const LUT_WIDTH: u32 = 256;
/// Bind-group binding indices. Integer channel textures occupy `1..=MAX_CHANNELS`,
/// then the LUT array, the sampler, and finally the float channel textures.
const LUT_BINDING: u32 = MAX_CHANNELS as u32 + 1;
const SAMPLER_BINDING: u32 = MAX_CHANNELS as u32 + 2;
const FTEX_BINDING_BASE: u32 = MAX_CHANNELS as u32 + 3;

/// Per-channel texture-allocation kind (which set is full-size + the integer
/// format), tracked so `ensure_size` only rebuilds when something changes.
const KIND_UNUSED: u8 = 0; // channel not present: both textures are 1x1 dummies
const KIND_INT8: u8 = 1; // integer channel: R8Uint full-size, R32F dummy
const KIND_INT16: u8 = 2; // integer channel: R16Uint full-size, R32F dummy
const KIND_FLOAT: u8 = 3; // float channel: R32F full-size, int texture dummy

/// The `eframe::Renderer` this backend needs requested in `NativeOptions`.
pub const RENDERER: eframe::Renderer = eframe::Renderer::Wgpu;

/// Short human-readable backend name, shown in the UI.
pub const BACKEND: &str = "wgpu";

/// Shared handle to the wgpu render resources. `Arc<Mutex>` because the
/// egui_wgpu paint callback (which draws) must be `Send + Sync + 'static`;
/// uploads happen in `app::sync_gpu`, so the lock is uncontended (both on the
/// UI thread, and never overlap — uploads finish before the callback paints).
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// Build the render resources from eframe's creation context (its wgpu render
/// state). Called once at startup.
pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
    let rs = cc
        .wgpu_render_state
        .as_ref()
        .expect("FastTIFF requires the wgpu backend (NativeOptions::renderer = Wgpu)");
    Arc::new(Mutex::new(ImageRenderResources::new(&rs.device, rs.target_format)))
}

/// Per-frame upload handle: the device + queue, pulled from `eframe::Frame`.
/// `None` before the backend is up (shouldn't happen after init).
pub struct UploadCtx<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

pub fn upload_ctx(frame: &eframe::Frame) -> Option<UploadCtx<'_>> {
    frame
        .wgpu_render_state()
        .map(|rs| UploadCtx { device: &rs.device, queue: &rs.queue })
}

/// The egui paint callback that draws the current image into `rect`. Captures a
/// clone of the shared resources and locks them at paint time.
pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
        rect,
        ImagePaintCallback { resources: render.clone() },
    ))
}

/// GPU bytes per volume sample on this backend: every channel is stored as a
/// 16-bit texel (R16Unorm or R16Float), whatever the source kind. The volume
/// builder budgets on the larger of CPU/GPU footprint, so 8-bit sources count
/// as 2 bytes here instead of silently doubling past the budget in VRAM.
pub fn volume_gpu_bps(_kind: VolumeKind) -> usize {
    2
}

/// Backend hook for `eframe::NativeOptions`: request the 16-bit-norm texture
/// feature when the adapter has it (nearly universal on desktop), so 16-bit
/// volume data keeps its full precision (R16Unorm) instead of rounding through
/// f16 (~11 bits). The device init falls back cleanly when it's missing.
pub fn tune_native_options(options: &mut eframe::NativeOptions) {
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(|adapter| {
            // Mirror eframe's default limits choice, plus our optional feature.
            let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            };
            wgpu::DeviceDescriptor {
                label: Some("egui wgpu device"),
                required_features: adapter.features() & wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: 8192,
                    ..base_limits
                },
                ..Default::default()
            }
        });
    }
}

/// The egui paint callback that ray-marches the 3D volume into `rect`.
pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
        rect,
        VolumePaintCallback { resources: render.clone() },
    ))
}

/// The volume-view callback. `prepare` writes the camera/window uniform (it has
/// the queue; `set_volume_params` only stashed the params), then `paint` draws.
struct VolumePaintCallback {
    resources: Render,
}

impl egui_wgpu::CallbackTrait for VolumePaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Ok(r) = self.resources.lock() {
            r.write_volume_uniform(queue);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        if let Ok(r) = self.resources.lock() {
            r.paint_volume(render_pass);
        }
    }
}

/// The `egui_wgpu::CallbackTrait` impl invoked once per egui frame to draw the
/// image. Holds its own clone of the resources (not egui_wgpu's resource map).
struct ImagePaintCallback {
    resources: Render,
}

impl egui_wgpu::CallbackTrait for ImagePaintCallback {
    // `prepare` is left as the trait default (no-op): all GPU state updates
    // (texture uploads, uniform writes) happen synchronously in app.rs before
    // this callback is queued, via direct queue.write_* calls.
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        if let Ok(r) = self.resources.lock() {
            r.paint(render_pass);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChannelParamsGpu {
    min_max: [f32; 2],
    enabled: f32,
    is_float: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    channels: [ChannelParamsGpu; MAX_CHANNELS],
    /// UV sub-rect of the image to display: `sampled_uv = uv_offset + uv * uv_scale`.
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
    num_channels: u32,
    _pad: [u32; 3],
}

/// Volume uniform, matching `VolParams` in `volume.wgsl`. Every field is a vec4
/// (or vec4-packed) so the std140-style uniform layout is unambiguous.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolParamsGpu {
    /// Per channel: `(window.min, window.max, enabled, unused)`.
    channels: [[f32; 4]; MAX_CHANNELS],
    cam_eye: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    box_he: [f32; 4],
    /// `(tan_half_fov, aspect, density, unused)`.
    misc: [f32; 4],
    /// `(num_channels, render_mode, interp, unused)`.
    modes: [i32; 4],
}

pub struct ImageRenderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    channel_textures: [wgpu::Texture; MAX_CHANNELS], // R16Uint (integer channels)
    channel_ftextures: [wgpu::Texture; MAX_CHANNELS], // R32Float (float channels)
    lut_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    /// Cached so we only rebuild textures + bind group when the frame size or the
    /// per-channel int/float layout changes (e.g. opening a different stack).
    current_size: (u32, u32),
    current_kinds: [u8; MAX_CHANNELS],

    // --- 3D volume ---------------------------------------------------------
    volume_pipeline: wgpu::RenderPipeline,
    volume_bind_group_layout: wgpu::BindGroupLayout,
    volume_bind_group: wgpu::BindGroup,
    volume_uniform_buffer: wgpu::Buffer,
    /// Per-channel 16-bit 3D textures (1x1x1 dummies until a volume is built):
    /// R16Unorm for integer sources when the device has the 16-bit-norm feature
    /// (full 16-bit precision, and u16 uploads become plain memcpys), else
    /// R16Float; float sources are always R16Float.
    volume_textures: [wgpu::Texture; MAX_CHANNELS],
    /// Whether the device has `TEXTURE_FORMAT_16BIT_NORM` (see `tune_native_options`).
    volume_unorm16: bool,
    /// `interp` for the shader: 0 = nearest, 1 = linear, 2 = cubic.
    volume_interp_mode: i32,
    /// The camera/window params, stashed by `set_volume_params` and written to
    /// the uniform buffer by the paint callback's `prepare`.
    volume_params: Option<VolumeParams>,
}

impl ImageRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FastTIFFcomposite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        let mut layout_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<ParamsGpu>() as u64),
            },
            count: None,
        }];
        for c in 0..MAX_CHANNELS as u32 {
            layout_entries.push(int_texture_entry(c + 1));
        }
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: LUT_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: SAMPLER_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        for c in 0..MAX_CHANNELS as u32 {
            layout_entries.push(float_texture_entry(FTEX_BINDING_BASE + c));
        }
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FastTIFFbind group layout"),
            entries: &layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FastTIFFpipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FastTIFFcomposite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(target_format.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FastTIFFparams"),
            size: std::mem::size_of::<ParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1x1 placeholder textures to start; resized via `ensure_size`.
        let channel_textures =
            std::array::from_fn(|i| {
                create_int_texture(device, 1, 1, wgpu::TextureFormat::R16Uint, &format!("FastTIFFchannel {i}"))
            });
        let channel_ftextures =
            std::array::from_fn(|i| create_float_texture(device, 1, 1, &format!("FastTIFFchannel-f {i}")));
        let lut_texture = create_lut_texture(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FastTIFFlut sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = build_bind_group(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &channel_textures,
            &channel_ftextures,
            &lut_texture,
            &sampler,
        );

        let (volume_pipeline, volume_bind_group_layout, volume_bind_group, volume_uniform_buffer, volume_textures) =
            create_volume_resources(device, target_format, &lut_texture, &sampler);
        let volume_unorm16 = device.features().contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            channel_textures,
            channel_ftextures,
            lut_texture,
            sampler,
            current_size: (1, 1),
            current_kinds: [KIND_UNUSED; MAX_CHANNELS],
            volume_pipeline,
            volume_bind_group_layout,
            volume_bind_group,
            volume_uniform_buffer,
            volume_textures,
            volume_unorm16,
            volume_interp_mode: 1, // linear
            volume_params: None,
        }
    }

    // --- 3D volume ---------------------------------------------------------

    /// Largest per-axis 3D-texture dimension the device supports; the app uses it
    /// (with a memory cap) to decide whether the volume must be subsampled.
    pub fn max_3d_texture_size(&self, ctx: &UploadCtx) -> u32 {
        ctx.device.limits().max_texture_dimension_3d
    }

    /// (Re)upload the volume, one entry per channel: each `bytes` is `w*h*d`
    /// samples in native `kind` format, uploaded to 16-bit 3D textures (see the
    /// `volume_textures` field for the format choice). Channels past
    /// `channels.len()` become 1x1x1 dummies. Textures whose size + format are
    /// unchanged (a 4D timepoint step) are refilled in place; the bind group is
    /// only rebuilt when a texture was recreated. Conversion happens in bounded
    /// z-slab chunks (or not at all: u16 -> R16Unorm is a straight copy).
    pub fn upload_volumes(&mut self, ctx: &UploadCtx, w: u32, h: u32, d: u32, channels: &[(VolumeKind, Vec<u8>)]) {
        let device = ctx.device;
        let mut rebind = false;
        for c in 0..MAX_CHANNELS {
            let entry = channels.get(c);
            let (want_w, want_h, want_d) = if entry.is_some() { (w, h, d) } else { (1, 1, 1) };
            let want = wgpu::Extent3d {
                width: want_w,
                height: want_h,
                depth_or_array_layers: want_d,
            };
            let want_fmt = match entry {
                Some((kind, _)) => self.volume_format(*kind),
                None => wgpu::TextureFormat::R16Float,
            };
            if self.volume_textures[c].size() != want || self.volume_textures[c].format() != want_fmt {
                self.volume_textures[c] =
                    create_volume_texture(device, want_w, want_h, want_d, want_fmt, &format!("FastTIFFvol {c}"));
                rebind = true;
                if entry.is_none() {
                    write_volume_region(ctx.queue, &self.volume_textures[c], 1, 1, 0, 1, &[0u8, 0u8]);
                }
            }
            if let Some((kind, bytes)) = entry {
                self.upload_volume_channel(ctx.queue, c, (w, h, d), *kind, bytes);
            }
        }
        if rebind {
            self.volume_bind_group = build_volume_bind_group(
                device,
                &self.volume_bind_group_layout,
                &self.volume_uniform_buffer,
                &self.volume_textures,
                &self.lut_texture,
                &self.sampler,
            );
        }
    }

    /// The texture format a volume channel of `kind` uses on this device.
    fn volume_format(&self, kind: VolumeKind) -> wgpu::TextureFormat {
        match kind {
            VolumeKind::U8 | VolumeKind::U16 if self.volume_unorm16 => wgpu::TextureFormat::R16Unorm,
            _ => wgpu::TextureFormat::R16Float,
        }
    }

    /// Fill one channel's volume texture from native-format samples. u16 data on
    /// an R16Unorm texture uploads verbatim (no conversion pass at all); every
    /// other combination converts z-slab by z-slab into a reused buffer, so the
    /// transient allocation is bounded by `VOLUME_UPLOAD_CHUNK_BYTES` instead of
    /// a second full-volume copy.
    fn upload_volume_channel(&self, queue: &wgpu::Queue, c: usize, dims: (u32, u32, u32), kind: VolumeKind, bytes: &[u8]) {
        let (w, h, d) = dims;
        let texture = &self.volume_textures[c];
        let unorm = texture.format() == wgpu::TextureFormat::R16Unorm;
        if kind == VolumeKind::U16 && unorm {
            // Raw u16 samples are exactly R16Unorm texels: one straight upload.
            write_volume_region(queue, texture, w, h, 0, d, bytes);
            return;
        }
        let slice_texels = w as usize * h as usize;
        let src_bps = match kind {
            VolumeKind::U8 => 1,
            VolumeKind::U16 => 2,
            VolumeKind::F32 => 4,
        };
        let slices_per_chunk = (VOLUME_UPLOAD_CHUNK_BYTES / (slice_texels * 2).max(1)).clamp(1, d as usize);
        let mut buf: Vec<u16> = Vec::with_capacity(slices_per_chunk * slice_texels);
        let mut z0 = 0usize;
        while z0 < d as usize {
            let zn = slices_per_chunk.min(d as usize - z0);
            let src = &bytes[z0 * slice_texels * src_bps..(z0 + zn) * slice_texels * src_bps];
            buf.clear();
            convert_volume_texels(kind, unorm, src, &mut buf);
            write_volume_region(queue, texture, w, h, z0 as u32, zn as u32, bytemuck::cast_slice(&buf));
            z0 += zn;
        }
    }

    /// Store the sampling mode for the shader (0 = nearest, 1 = linear, 2 = cubic).
    /// A single linear sampler serves all three (nearest snaps to texel centers,
    /// cubic reconstructs in-shader), so no GPU state changes here.
    pub fn set_volume_interp(&mut self, _ctx: &UploadCtx, interp: VolumeInterp) {
        self.volume_interp_mode = match interp {
            VolumeInterp::Nearest => 0,
            VolumeInterp::Linear => 1,
            VolumeInterp::Cubic => 2,
        };
    }

    /// Stash the ray-march params; the paint callback's `prepare` writes them to
    /// the uniform buffer (which is where the queue is available).
    pub fn set_volume_params(&mut self, params: VolumeParams) {
        self.volume_params = Some(params);
    }

    /// Marshal the stashed params + interp mode into the volume uniform buffer.
    /// Called from the paint callback's `prepare`.
    fn write_volume_uniform(&self, queue: &wgpu::Queue) {
        let Some(p) = self.volume_params else { return };
        let mut gpu = VolParamsGpu {
            channels: [[0.0; 4]; MAX_CHANNELS],
            cam_eye: [p.eye[0], p.eye[1], p.eye[2], 0.0],
            cam_forward: [p.forward[0], p.forward[1], p.forward[2], 0.0],
            cam_right: [p.right[0], p.right[1], p.right[2], 0.0],
            cam_up: [p.up[0], p.up[1], p.up[2], 0.0],
            box_he: [p.box_he[0], p.box_he[1], p.box_he[2], 0.0],
            misc: [p.tan_half_fov, p.aspect, p.density, 0.0],
            modes: [p.num_channels, p.render_mode, self.volume_interp_mode, 0],
        };
        for c in 0..MAX_CHANNELS {
            gpu.channels[c] = [p.windows[c * 2], p.windows[c * 2 + 1], p.enabled[c], 0.0];
        }
        queue.write_buffer(&self.volume_uniform_buffer, 0, bytemuck::bytes_of(&gpu));
    }

    /// Ray-march the current volume. `prepare` has already written the uniform.
    pub fn paint_volume(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.volume_params.is_none() {
            return;
        }
        render_pass.set_pipeline(&self.volume_pipeline);
        render_pass.set_bind_group(0, &self.volume_bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }

    /// (Re)allocate channel textures for the current frame size and per-channel
    /// `kind`. An `Int8`/`Int16` channel gets a full-size R8Uint/R16Uint integer
    /// texture (float texture a 1x1 dummy); a `Float` channel gets a full-size
    /// R32F float texture (integer texture a 1x1 dummy); channels past
    /// `kinds.len()` are unused (both 1x1). Rebuilds the bind group when anything
    /// changed; no-op otherwise.
    pub fn ensure_size(&mut self, ctx: &UploadCtx, width: u32, height: u32, kinds: &[ChannelKind]) {
        let mut want = [KIND_UNUSED; MAX_CHANNELS];
        for (c, slot) in want.iter_mut().enumerate() {
            *slot = match kinds.get(c) {
                None => KIND_UNUSED,
                Some(ChannelKind::Int8) => KIND_INT8,
                Some(ChannelKind::Int16) => KIND_INT16,
                Some(ChannelKind::Float) => KIND_FLOAT,
            };
        }
        if self.current_size == (width, height) && self.current_kinds == want {
            return;
        }
        let device = ctx.device;
        self.channel_textures = std::array::from_fn(|c| {
            let label = format!("FastTIFFchannel {c}");
            // R8Uint or R16Uint at full size for an integer channel, else a 1x1
            // R16Uint dummy. Both are sampled through the same `texture_2d<u32>`.
            match want[c] {
                KIND_INT8 => create_int_texture(device, width, height, wgpu::TextureFormat::R8Uint, &label),
                KIND_INT16 => create_int_texture(device, width, height, wgpu::TextureFormat::R16Uint, &label),
                _ => create_int_texture(device, 1, 1, wgpu::TextureFormat::R16Uint, &label),
            }
        });
        self.channel_ftextures = std::array::from_fn(|c| {
            let (w, h) = if want[c] == KIND_FLOAT { (width, height) } else { (1, 1) };
            create_float_texture(device, w, h, &format!("FastTIFFchannel-f {c}"))
        });
        self.bind_group = build_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.channel_textures,
            &self.channel_ftextures,
            &self.lut_texture,
            &self.sampler,
        );
        self.current_size = (width, height);
        self.current_kinds = want;
    }

    /// Upload one integer channel's raw 16-bit samples (R16Uint texture).
    pub fn upload_channel(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.channel_textures[channel],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(samples),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 2),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload one integer channel's raw 8-bit samples (R8Uint texture). Skips
    /// the CPU `0..255 -> 0..65535` widening; the window/level is scaled to
    /// 0..255 units on the app side instead.
    pub fn upload_channel_u8(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[u8]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.channel_textures[channel],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            samples,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload one float channel's raw 32-bit float samples (R32F texture).
    pub fn upload_channel_f32(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[f32]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.channel_ftextures[channel],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(samples),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload one channel's LUT (256 RGB entries) into the LUT texture array.
    pub fn upload_lut(&self, ctx: &UploadCtx, channel: usize, lut: &[[u8; 3]; 256]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let mut rgba = vec![0u8; LUT_WIDTH as usize * 4];
        for (i, px) in lut.iter().enumerate() {
            rgba[i * 4] = px[0];
            rgba[i * 4 + 1] = px[1];
            rgba[i * 4 + 2] = px[2];
            rgba[i * 4 + 3] = 255;
        }
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: channel as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LUT_WIDTH * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: LUT_WIDTH,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Update per-channel window/level + enabled + is-float flags, the active
    /// channel count, and the visible UV sub-rect (pan/zoom), in one uniform
    /// write.
    pub fn set_params(
        &mut self,
        ctx: &UploadCtx,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) {
        let mut gpu = ParamsGpu {
            channels: [ChannelParamsGpu {
                min_max: [0.0, 65535.0],
                enabled: 0.0,
                is_float: 0.0,
            }; MAX_CHANNELS],
            uv_offset,
            uv_scale,
            num_channels,
            _pad: [0; 3],
        };
        for (i, c) in channels.iter().take(MAX_CHANNELS).enumerate() {
            gpu.channels[i] = ChannelParamsGpu {
                min_max: [c.min, c.max],
                enabled: if c.enabled { 1.0 } else { 0.0 },
                is_float: if c.is_float { 1.0 } else { 0.0 },
            };
        }
        ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&gpu));
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

fn int_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn float_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // Read via textureLoad (no filtering), so `filterable: false` — this
            // keeps R32Float in core wgpu without the FLOAT32_FILTERABLE feature.
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_int_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_float_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_lut_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("FastTIFFlut array"),
        size: wgpu::Extent3d {
            width: LUT_WIDTH,
            height: 1,
            depth_or_array_layers: MAX_CHANNELS as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    channel_textures: &[wgpu::Texture; MAX_CHANNELS],
    channel_ftextures: &[wgpu::Texture; MAX_CHANNELS],
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let channel_views: Vec<wgpu::TextureView> = channel_textures
        .iter()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let fchannel_views: Vec<wgpu::TextureView> = channel_ftextures
        .iter()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
    }];
    for (i, view) in channel_views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: LUT_BINDING,
        resource: wgpu::BindingResource::TextureView(&lut_view),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: SAMPLER_BINDING,
        resource: wgpu::BindingResource::Sampler(sampler),
    });
    for (i, view) in fchannel_views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: FTEX_BINDING_BASE + i as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FastTIFFbind group"),
        layout,
        entries: &entries,
    })
}

// --- 3D volume helpers -----------------------------------------------------

/// Build the volume pipeline, its bind-group layout + initial (1x1x1 dummy)
/// bind group, and the uniform buffer. Reuses the 2D LUT texture + sampler.
fn create_volume_resources(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::BindGroup, wgpu::Buffer, [wgpu::Texture; MAX_CHANNELS]) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FastTIFFvolume shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/volume.wgsl").into()),
    });

    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(std::mem::size_of::<VolParamsGpu>() as u64),
        },
        count: None,
    }];
    for c in 0..MAX_CHANNELS as u32 {
        entries.push(volume_texture_entry(c + 1));
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: MAX_CHANNELS as u32 + 1,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    });
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: MAX_CHANNELS as u32 + 2,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FastTIFFvolume bind group layout"),
        entries: &entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FastTIFFvolume pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("FastTIFFvolume pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(target_format.into())],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FastTIFFvolume params"),
        size: std::mem::size_of::<VolParamsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let textures = std::array::from_fn(|c| {
        create_volume_texture(device, 1, 1, 1, wgpu::TextureFormat::R16Float, &format!("FastTIFFvol {c}"))
    });
    let bind_group = build_volume_bind_group(device, &layout, &uniform_buffer, &textures, lut_texture, sampler);

    (pipeline, layout, bind_group, uniform_buffer, textures)
}

fn volume_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // R16Float is core-filterable; R16Unorm is filterable under the
            // TEXTURE_FORMAT_16BIT_NORM feature. Both fit this entry.
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn create_volume_texture(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    d: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: d.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Bound on the converted-texel scratch buffer for chunked volume uploads.
const VOLUME_UPLOAD_CHUNK_BYTES: usize = 32 << 20;

/// Write a z-slab of 16-bit texels (`data` = `w*h*d` texels starting at `z0`)
/// into a 3D texture.
fn write_volume_region(queue: &wgpu::Queue, texture: &wgpu::Texture, w: u32, h: u32, z0: u32, d: u32, data: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: z0 },
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 2),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        },
    );
}

/// Convert native-format samples to 16-bit texels, appended to `out`:
///   * R16Unorm target: U8 widens ×257 (exact: 255·257 = 65535).
///   * R16Float target: U8 -> raw/255, U16 -> raw/65535, F32 -> raw — matching
///     the glow unorm/float units, so the per-channel window applies unchanged.
///
/// The input holds native-endian samples (filled via `bytemuck` casts).
fn convert_volume_texels(kind: VolumeKind, unorm: bool, bytes: &[u8], out: &mut Vec<u16>) {
    use half::f16;
    match kind {
        VolumeKind::U8 if unorm => out.extend(bytes.iter().map(|&b| b as u16 * 257)),
        VolumeKind::U8 => out.extend(bytes.iter().map(|&b| f16::from_f32(b as f32 / 255.0).to_bits())),
        VolumeKind::U16 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_f32(u16::from_ne_bytes([c[0], c[1]]) as f32 / 65535.0).to_bits()),
        ),
        VolumeKind::F32 => out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| f16::from_f32(f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).to_bits()),
        ),
    }
}

#[cfg(test)]
mod wgsl_tests {
    /// Parse + validate a WGSL source with naga (what wgpu does at runtime), so a
    /// shader error is a failing test rather than a blank 3D canvas at startup.
    fn validate(src: &str, name: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: validate: {e:?}"));
    }

    #[test]
    fn volume_shader_is_valid() {
        validate(include_str!("../shaders/volume.wgsl"), "volume.wgsl");
    }

    #[test]
    fn composite_shader_is_valid() {
        validate(include_str!("../shaders/composite.wgsl"), "composite.wgsl");
    }
}

fn build_volume_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    textures: &[wgpu::Texture; MAX_CHANNELS],
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let views: Vec<wgpu::TextureView> = textures
        .iter()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
        .collect();
    let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
    }];
    for (i, view) in views.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: MAX_CHANNELS as u32 + 1,
        resource: wgpu::BindingResource::TextureView(&lut_view),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: MAX_CHANNELS as u32 + 2,
        resource: wgpu::BindingResource::Sampler(sampler),
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FastTIFFvolume bind group"),
        layout,
        entries: &entries,
    })
}

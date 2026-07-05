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

/// 3D volume rendering is not implemented on the wgpu backend yet — the glow
/// backend (the default) carries it. This paints a plain black canvas and logs
/// once so a wgpu user sees a clear blank rather than stale 2D pixels.
pub fn paint_volume_callback(_render: &Render, rect: egui::Rect) -> egui::Shape {
    use std::sync::Once;
    static WARN: Once = Once::new();
    WARN.call_once(|| log::warn!("3D volume view is only implemented on the glow renderer; wgpu shows a blank canvas"));
    egui::Shape::rect_filled(rect, 0.0, egui::Color32::BLACK)
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
        }
    }

    // --- 3D volume: not implemented on wgpu; these keep the backend-agnostic
    // surface identical to glow's so app.rs compiles unchanged. See
    // `paint_volume_callback` above. ---
    pub fn max_3d_texture_size(&self, _ctx: &UploadCtx) -> u32 {
        2048 // a safe conservative default; the volume path is unused here
    }
    pub fn upload_volume(&mut self, _ctx: &UploadCtx, _w: u32, _h: u32, _d: u32, _kind: VolumeKind, _bytes: &[u8]) {}
    pub fn set_volume_interp(&mut self, _ctx: &UploadCtx, _interp: VolumeInterp) {}
    pub fn set_volume_params(&mut self, _params: VolumeParams) {}

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

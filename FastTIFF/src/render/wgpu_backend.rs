//! wgpu rendering for the composited image: the render pipeline, the fixed set
//! of `MAX_CHANNELS` raw-sample textures (R16Uint), the LUT texture array, and
//! the uniform buffer carrying per-channel window/level + enabled flags. Uploads
//! happen from `app.rs` via `UploadCtx` (the device + queue from
//! `Frame::wgpu_render_state`); `paint` is invoked from the
//! `egui_wgpu::CallbackTrait` built by `paint_callback`.
//!
//! Unlike egui's `custom3d_wgpu` example (which parks resources in
//! egui_wgpu's `CallbackResources` map), we keep them in an `Arc<Mutex>` owned
//! by the app and captured by the callback — matching the glow backend so the
//! two are interchangeable behind one app-side interface.

use super::{ChannelUniform, MAX_CHANNELS};
use eframe::egui_wgpu::{self, wgpu};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

const LUT_WIDTH: u32 = 256;
/// Bind-group binding indices: channel textures occupy `1..=MAX_CHANNELS`, then
/// the LUT array and sampler follow.
const LUT_BINDING: u32 = MAX_CHANNELS as u32 + 1;
const SAMPLER_BINDING: u32 = MAX_CHANNELS as u32 + 2;

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
    _pad: f32,
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
    channel_textures: [wgpu::Texture; MAX_CHANNELS],
    lut_texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    /// Cached so we know whether channel textures need to be recreated
    /// (only on a frame-size change, e.g. opening a different stack).
    current_size: (u32, u32),
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
            layout_entries.push(channel_texture_entry(c + 1));
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
        let channel_textures = std::array::from_fn(|i| {
            create_channel_texture(device, 1, 1, &format!("FastTIFFchannel {i}"))
        });
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
            &lut_texture,
            &sampler,
        );

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            channel_textures,
            lut_texture,
            sampler,
            current_size: (1, 1),
        }
    }

    /// Recreate channel textures if the frame dimensions changed (new
    /// stack opened, or stack with different per-frame size).
    pub fn ensure_size(&mut self, ctx: &UploadCtx, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
        let device = ctx.device;
        self.channel_textures = std::array::from_fn(|i| {
            create_channel_texture(device, width, height, &format!("FastTIFFchannel {i}"))
        });
        self.bind_group = build_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.channel_textures,
            &self.lut_texture,
            &self.sampler,
        );
        self.current_size = (width, height);
    }

    /// Upload one channel's raw 16-bit samples.
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

    /// Update per-channel window/level + enabled flags, the active channel
    /// count, and the visible UV sub-rect (pan/zoom), in one uniform write.
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
                _pad: 0.0,
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
                _pad: 0.0,
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

fn channel_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn create_channel_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
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
        format: wgpu::TextureFormat::R16Uint,
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
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let channel_views: Vec<wgpu::TextureView> = channel_textures
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

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FastTIFFbind group"),
        layout,
        entries: &entries,
    })
}

//! OpenGL (glow) rendering for the composited image: up to `MAX_CHANNELS`
//! single-channel raw-sample textures (R16UI), a LUT texture (one row per
//! channel), and a fragment shader that does per-channel window/level → LUT →
//! additive blend, with a minification box filter for clean zoom-out. Uploads
//! happen from `app.rs` via `UploadCtx` (the live GL context from `Frame::gl`);
//! `paint` is invoked from the egui_glow callback built by `paint_callback`.

use super::{ChannelUniform, MAX_CHANNELS};
use eframe::glow::{self, HasContext as _};
use eframe::egui_glow;
use std::sync::{Arc, Mutex};

const LUT_WIDTH: i32 = 256;

/// The `eframe::Renderer` this backend needs requested in `NativeOptions`.
pub const RENDERER: eframe::Renderer = eframe::Renderer::Glow;

/// Shared handle to the GL render resources. `Arc<Mutex>` because the egui_glow
/// paint callback (which draws) must be `Send + Sync + 'static`; uploads happen
/// in `app::sync_gpu`, so the lock is uncontended (both on the UI thread).
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// Build the render resources from eframe's creation context (its glow
/// context). Called once at startup.
pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
    let gl = cc
        .gl
        .as_ref()
        .expect("FastTIFF requires the glow backend (NativeOptions::renderer = Glow)");
    Arc::new(Mutex::new(ImageRenderResources::new(gl)))
}

/// Per-frame upload handle: the live GL context, pulled from `eframe::Frame`.
/// `None` before the backend is up (shouldn't happen after init).
pub struct UploadCtx<'a> {
    gl: &'a glow::Context,
}

pub fn upload_ctx(frame: &eframe::Frame) -> Option<UploadCtx<'_>> {
    frame.gl().map(|gl| UploadCtx { gl })
}

/// The egui paint callback that draws the current image into `rect`. Captures a
/// clone of the shared resources and locks them at paint time.
pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    let res = render.clone();
    let callback = egui_glow::CallbackFn::new(move |_info, painter| {
        if let Ok(r) = res.lock() {
            r.paint(painter.gl());
        }
    });
    egui::Shape::Callback(egui::PaintCallback {
        rect,
        callback: Arc::new(callback),
    })
}

/// egui_glow gives us a desktop GL 3.x context, for which it uses GLSL `#version
/// 140`. We match that so usampler2D / texelFetch / gl_VertexID are available.
const VERTEX_SRC: &str = r#"#version 140
out vec2 v_uv;
void main() {
    const vec2 POS[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0));
    const vec2 UV[6] = vec2[6](
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
    gl_Position = vec4(POS[gl_VertexID], 0.0, 1.0);
    v_uv = UV[gl_VertexID];
}
"#;

const FRAGMENT_SRC: &str = r#"#version 140
in vec2 v_uv;
out vec4 frag_color;

uniform usampler2D ch0_tex;
uniform usampler2D ch1_tex;
uniform usampler2D ch2_tex;
uniform usampler2D ch3_tex;
uniform usampler2D ch4_tex;
uniform usampler2D ch5_tex;
uniform sampler2D lut_tex;
uniform vec2 ch_min_max[6];
uniform float ch_enabled[6];
uniform int num_channels;
uniform vec2 uv_offset;
uniform vec2 uv_scale;

float load_texel(usampler2D tex, vec2 uv, vec2 dims) {
    ivec2 c = clamp(ivec2(uv * dims), ivec2(0), ivec2(dims) - ivec2(1));
    return float(texelFetch(tex, c, 0).r);
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
    pub fn ensure_size(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
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
    pub fn upload_channel(&self, queue: &wgpu::Queue, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        queue.write_texture(
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
    pub fn upload_lut(&self, queue: &wgpu::Queue, channel: usize, lut: &[[u8; 3]; 256]) {
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
        queue.write_texture(
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
    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
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
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&gpu));
    }

    pub fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

pub struct ImageRenderResources {
    program: glow::NativeProgram,
    vao: glow::NativeVertexArray,
    channel_textures: [glow::NativeTexture; MAX_CHANNELS],
    lut_texture: glow::NativeTexture,
    current_size: (u32, u32),

    u_ch_tex: [Option<glow::NativeUniformLocation>; MAX_CHANNELS],
    u_lut: Option<glow::NativeUniformLocation>,
    u_min_max: Option<glow::NativeUniformLocation>,
    u_enabled: Option<glow::NativeUniformLocation>,
    u_num_channels: Option<glow::NativeUniformLocation>,
    u_uv_offset: Option<glow::NativeUniformLocation>,
    u_uv_scale: Option<glow::NativeUniformLocation>,

    // Current draw params, set from app.rs and applied in `paint`.
    min_max: [f32; MAX_CHANNELS * 2],
    enabled: [f32; MAX_CHANNELS],
    num_channels: i32,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
}

impl ImageRenderResources {
    pub fn new(gl: &glow::Context) -> Self {
        unsafe {
            let program = link_program(gl, VERTEX_SRC, FRAGMENT_SRC);
            let vao = gl.create_vertex_array().expect("create VAO");
            let channel_textures = std::array::from_fn(|_| create_channel_texture(gl, 1, 1));
            let lut_texture = create_lut_texture(gl);

            let u_ch_tex = std::array::from_fn(|c| gl.get_uniform_location(program, &format!("ch{c}_tex")));

            Self {
                program,
                vao,
                channel_textures,
                lut_texture,
                current_size: (1, 1),
                u_ch_tex,
                u_lut: gl.get_uniform_location(program, "lut_tex"),
                u_min_max: gl.get_uniform_location(program, "ch_min_max"),
                u_enabled: gl.get_uniform_location(program, "ch_enabled"),
                u_num_channels: gl.get_uniform_location(program, "num_channels"),
                u_uv_offset: gl.get_uniform_location(program, "uv_offset"),
                u_uv_scale: gl.get_uniform_location(program, "uv_scale"),
                min_max: [0.0; MAX_CHANNELS * 2],
                enabled: [0.0; MAX_CHANNELS],
                num_channels: 0,
                uv_offset: [0.0, 0.0],
                uv_scale: [1.0, 1.0],
            }
        }
    }

    /// Reallocate the channel textures when the frame size changes.
    pub fn ensure_size(&mut self, ctx: &UploadCtx, width: u32, height: u32) {
        if self.current_size == (width, height) {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            for &tex in &self.channel_textures {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::R16UI as i32,
                    width as i32,
                    height as i32,
                    0,
                    glow::RED_INTEGER,
                    glow::UNSIGNED_SHORT,
                    glow::PixelUnpackData::Slice(None),
                );
            }
        }
        self.current_size = (width, height);
    }

    /// Upload one channel's raw 16-bit samples.
    pub fn upload_channel(&self, ctx: &UploadCtx, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[channel]));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                width as i32,
                height as i32,
                glow::RED_INTEGER,
                glow::UNSIGNED_SHORT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(samples))),
            );
        }
    }

    /// Upload one channel's LUT (256 RGB entries) into row `channel`.
    pub fn upload_lut(&self, ctx: &UploadCtx, channel: usize, lut: &[[u8; 3]; 256]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        let gl = ctx.gl;
        let mut rgba = [0u8; (LUT_WIDTH * 4) as usize];
        for (i, px) in lut.iter().enumerate() {
            rgba[i * 4] = px[0];
            rgba[i * 4 + 1] = px[1];
            rgba[i * 4 + 2] = px[2];
            rgba[i * 4 + 3] = 255;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                channel as i32,
                LUT_WIDTH,
                1,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&rgba)),
            );
        }
    }

    /// Stash the per-channel window/level + enabled flags, channel count, and
    /// the visible UV sub-rect (pan/zoom); applied as uniforms in `paint`. The
    /// glow backend uploads nothing here (it has no per-frame uniform buffer),
    /// so `ctx` is unused — the signature matches the wgpu backend, which does.
    pub fn set_params(
        &mut self,
        _ctx: &UploadCtx,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) {
        self.min_max = [0.0; MAX_CHANNELS * 2];
        self.enabled = [0.0; MAX_CHANNELS];
        for (i, c) in channels.iter().take(MAX_CHANNELS).enumerate() {
            self.min_max[i * 2] = c.min;
            self.min_max[i * 2 + 1] = c.max;
            self.enabled[i] = if c.enabled { 1.0 } else { 0.0 };
        }
        self.num_channels = num_channels as i32;
        self.uv_offset = uv_offset;
        self.uv_scale = uv_scale;
    }

    /// Draw the composited image. The egui_glow callback sets the viewport and
    /// scissor to the image's on-screen rect for us.
    pub fn paint(&self, gl: &glow::Context) {
        unsafe {
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            for c in 0..MAX_CHANNELS {
                gl.active_texture(glow::TEXTURE0 + c as u32);
                gl.bind_texture(glow::TEXTURE_2D, Some(self.channel_textures[c]));
                gl.uniform_1_i32(self.u_ch_tex[c].as_ref(), c as i32);
            }
            gl.active_texture(glow::TEXTURE0 + MAX_CHANNELS as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.lut_texture));
            gl.uniform_1_i32(self.u_lut.as_ref(), MAX_CHANNELS as i32);

            gl.uniform_2_f32_slice(self.u_min_max.as_ref(), &self.min_max);
            gl.uniform_1_f32_slice(self.u_enabled.as_ref(), &self.enabled);
            gl.uniform_1_i32(self.u_num_channels.as_ref(), self.num_channels);
            gl.uniform_2_f32(self.u_uv_offset.as_ref(), self.uv_offset[0], self.uv_offset[1]);
            gl.uniform_2_f32(self.u_uv_scale.as_ref(), self.uv_scale[0], self.uv_scale[1]);

            // The image is opaque (alpha 1); draw straight, no blending.
            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 6);
        }
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

/// Registers the pipeline in egui_wgpu's per-frame resource map, following
/// the same pattern as the egui repo's `custom3d_wgpu` example.
pub fn install(render_state: &egui_wgpu::RenderState) {
    let resources = ImageRenderResources::new(&render_state.device, render_state.target_format);
    render_state
        .renderer
        .write()
        .callback_resources
        .insert(resources);
}

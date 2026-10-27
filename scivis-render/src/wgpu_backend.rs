//! wgpu rendering for the composited image: the render pipeline, two per-channel
//! texture sets (R16Uint for integer sources, R32F for 32-bit-float sources),
//! the LUT texture array, and the uniform buffer carrying per-channel
//! window/level + enabled + is-float flags.
//!
//! Each channel is either integer or float: the one carrying its data is a
//! full-size texture, the other a 1x1 dummy (`is_float` in the uniform tells the
//! shader which to sample). Float channels skip the per-frame CPU rescale —
//! window/level is done on the GPU in the data's own units.
//!
//! The 3D volume view shares the LUT and sampler: the channels packed four to
//! an RGBA 3D texture with a mip chain (see [`crate::pack`]), a small 3D
//! texture saying which bricks of the volume can be seen, a separate ray-march
//! pipeline (`volume.wgsl`), and its own uniform. Every format used is
//! filterable, so linear/cubic sampling use the hardware sampler.
//!
//! # Host responsibilities
//!
//! The host creates the device and queue (both are cheap `Clone` handles, so we
//! keep our own) and owns the render pass. Two entry points need its
//! cooperation each 3D frame: [`ImageRenderResources::prepare_volume`] must run
//! before [`ImageRenderResources::paint_volume`] — under egui_wgpu that's
//! `prepare` then `paint`, elsewhere just call them in order. `prepare_volume`
//! is also where the view is rendered at reduced size while it moves (see
//! [`ImageRenderResources::set_volume_target`]), into a texture `paint_volume`
//! then scales up.

use crate::pack::{self, BrickMaxima, PackedVolume, TexelFormat, Visibility, VolumeLayout};
use crate::{
    capture_tiles, tile_params, ChannelKind, ChannelUniform, Lut, VolumeImage, VolumeInterp,
    VolumeKind, VolumeParams, VolumeUploadError, CAPTURE_TILE, MAX_CHANNELS,
};
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

/// Short human-readable backend name, for a host that wants to show it.
pub const BACKEND: &str = "wgpu";

/// GPU bytes per volume sample on this backend, per channel and before the
/// mip chain: 8-bit sources stay 8-bit, everything else is stored in 16 bits
/// (unorm or half float). See [`ImageRenderResources::volume_texel_format`]
/// for the exact format, which a volume builder should use instead.
pub fn volume_gpu_bps(kind: VolumeKind) -> usize {
    match kind {
        VolumeKind::U8 => 1,
        VolumeKind::U16 | VolumeKind::F32 => 2,
    }
}

/// The optional device features this backend benefits from, intersected with
/// what `adapter` actually offers — pass the result as `required_features` when
/// you create the device.
///
/// Today that's `TEXTURE_FORMAT_16BIT_NORM` (nearly universal on desktop, often
/// absent on WebGL): with it, 16-bit volume data keeps full precision in
/// R16Unorm instead of rounding through f16's ~11 bits. Everything falls back
/// cleanly when it's missing, so this is a quality knob, not a requirement.
pub fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
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
    /// Per channel: `(window.min, window.max, enabled, isosurface albedo t)`.
    channels: [[f32; 4]; MAX_CHANNELS],
    cam_eye: [f32; 4],
    cam_forward: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    box_he: [f32; 4],
    /// `(tan_half_fov, aspect, density, iso)`.
    misc: [f32; 4],
    gain: [f32; 4],
    /// `(num_channels, render_mode, interp, max_steps)`.
    modes: [i32; 4],
    /// `(size of one pixel at unit distance, highest mip level a MIP may read,
    /// brick size, skip empty bricks)`.
    accel: [f32; 4],
}

pub struct ImageRenderResources {
    /// Our own handles on the host's device/queue. Both are `Clone` in wgpu
    /// (internally reference-counted), so this is a refcount bump, not a second
    /// device — and it's what lets every upload method drop the per-call
    /// context parameter the eframe-coupled version needed.
    device: wgpu::Device,
    queue: wgpu::Queue,
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
    /// The packed volume: channels 0..3 and channels 4..5 (1x1x1 dummies until
    /// a volume is uploaded, and the second one whenever there are at most four
    /// channels). See [`crate::pack`].
    volume_textures: [wgpu::Texture; 2],
    /// One texel per brick of the volume: whether a ray must sample there.
    occupancy_texture: wgpu::Texture,
    /// The uploaded volume's layout and brick maxima.
    volume_layout: Option<VolumeLayout>,
    volume_bricks: Option<BrickMaxima>,
    /// What `occupancy_texture` was last worked out for.
    occupancy_for: Option<Visibility>,
    /// Per volume texture, the device's answer to whether there was room for
    /// it, when that had not come by the time the upload returned: on WebGPU
    /// it comes through a promise. The volume is not drawn until both have
    /// come, since a texture there was no room for is invalid, and drawing it
    /// would invalidate the host's whole frame, its own UI included.
    volume_checks: [Option<LateCheck>; 2],
    /// An upload that returned `Ok` and was refused after all, with the bytes
    /// its volume took, for `take_late_upload_error`.
    volume_lost: Option<(VolumeUploadError, usize)>,
    /// Whether the device has `TEXTURE_FORMAT_16BIT_NORM` (see `optional_features`).
    volume_unorm16: bool,
    /// `interp` for the shader: 0 = nearest, 1 = linear, 2 = cubic.
    volume_interp_mode: i32,
    /// Empty-space skipping and mip levels on (both, unless switched off with
    /// `set_volume_acceleration`).
    accel_skip: bool,
    accel_lod: bool,
    /// The camera/window params, stashed by `set_volume_params` and marshalled
    /// into the uniform buffer by `prepare_volume`.
    volume_params: Option<VolumeParams>,
    /// The size the view is painted at, in pixels, and how much smaller to
    /// render it (see `set_volume_target`).
    volume_target: (u32, u32),
    volume_downscale: u32,
    /// The reduced-size render and what scales it up.
    lowres: Option<LowRes>,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    /// Whether `prepare_volume` rendered this frame at reduced size, for
    /// `paint_volume` to scale up instead of ray-marching.
    lowres_ready: bool,
    /// The adapter the device came from, for asking how much memory it has
    /// (see `set_adapter`).
    adapter: Option<wgpu::Adapter>,
    /// The format the pipelines were built for, which an offscreen render has
    /// to use too.
    target_format: wgpu::TextureFormat,
    /// The tile texture and readback buffer `capture_volume` renders through,
    /// made on the first capture and kept.
    capture_target: Option<CaptureTarget>,
    /// The capture under way, if one is.
    capture: Option<CaptureJob>,
}

/// The 3D view rendered at reduced size, and the bind group that reads it.
struct LowRes {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

/// One tile's worth of render target, and the buffer it is read back through.
struct CaptureTarget {
    texture: wgpu::Texture,
    buffer: wgpu::Buffer,
    /// Bytes per row in `buffer`: a tile's row padded to wgpu's copy alignment.
    padded_row: u32,
}

/// An off-screen capture under way, a tile at a time (see [`CAPTURE_TILE`]).
struct CaptureJob {
    width: u32,
    height: u32,
    /// The params it was asked for with, kept: the view's own may move on
    /// while the tiles are being rendered.
    params: VolumeParams,
    tiles: Vec<(u32, u32, u32, u32)>,
    /// The tile being rendered and read back; those before it are in `rgba`.
    next: usize,
    rgba: Vec<u8>,
    /// Set by the map callback when the tile's readback has landed (or
    /// failed). `None` while it is on its way.
    mapped: Arc<Mutex<Option<Result<(), String>>>>,
    /// Whether the first tile has been submitted: held back while the device
    /// has not yet said there was room for the volume (`volume_checks`).
    started: bool,
}

impl ImageRenderResources {
    /// Build every pipeline, texture and buffer on `device`. `target_format` is
    /// the format of the color attachment the host will later paint into — the
    /// pipelines are compiled against it, so it must match the render pass.
    ///
    /// Both handles are kept (cheap refcounted clones); the host does not need
    /// to hand them back on later calls.
    pub fn new(gpu: wgpu::Device, queue: wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        let device = &gpu;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FastTIFFcomposite shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
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
        let channel_textures = std::array::from_fn(|i| {
            create_int_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::R16Uint,
                &format!("FastTIFFchannel {i}"),
            )
        });
        let channel_ftextures = std::array::from_fn(|i| {
            create_float_texture(device, 1, 1, &format!("FastTIFFchannel-f {i}"))
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
            &channel_ftextures,
            &lut_texture,
            &sampler,
        );

        let volume = create_volume_resources(device, target_format, &lut_texture, &sampler);
        let (blit_pipeline, blit_bind_group_layout) = create_blit_pipeline(device, target_format);
        let volume_unorm16 = device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);

        Self {
            device: gpu.clone(),
            queue,
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
            volume_pipeline: volume.pipeline,
            volume_bind_group_layout: volume.layout,
            volume_bind_group: volume.bind_group,
            volume_uniform_buffer: volume.uniform_buffer,
            volume_textures: volume.textures,
            occupancy_texture: volume.occupancy,
            volume_layout: None,
            volume_bricks: None,
            occupancy_for: None,
            volume_checks: [None, None],
            volume_lost: None,
            volume_unorm16,
            volume_interp_mode: 1, // linear
            accel_skip: true,
            accel_lod: true,
            volume_params: None,
            volume_target: (0, 0),
            volume_downscale: 1,
            lowres: None,
            blit_pipeline,
            blit_bind_group_layout,
            lowres_ready: false,
            adapter: None,
            target_format,
            capture_target: None,
            capture: None,
        }
    }

    // --- 3D volume ---------------------------------------------------------

    /// Largest per-axis 3D-texture dimension the device supports; the app uses it
    /// (with a memory cap) to decide whether the volume must be subsampled.
    pub fn max_3d_texture_size(&self) -> u32 {
        self.device.limits().max_texture_dimension_3d
    }

    /// Hand over the adapter the device came from, so the renderer can ask
    /// the driver how much GPU memory there is ([`volume_memory_available`]).
    /// Optional: without it the answer is "unknown".
    ///
    /// [`volume_memory_available`]: Self::volume_memory_available
    pub fn set_adapter(&mut self, adapter: wgpu::Adapter) {
        self.adapter = Some(adapter);
    }

    /// GPU memory a new volume could take, in bytes: what the driver says is
    /// left of this process's budget, plus what the current volume holds (a
    /// new one replaces it). `None` when the driver will not say — no adapter
    /// was handed over, or the backend has no way to ask (Metal, the browser).
    /// A builder should take a share of this, not all of it.
    pub fn volume_memory_available(&self) -> Option<u64> {
        let budget = crate::vram::wgpu_budget(self.adapter.as_ref()?)?;
        let current = self
            .volume_layout
            .as_ref()
            .map_or(0, |l| l.total_bytes() as u64);
        Some(budget.headroom() + current)
    }

    /// How this device stores a volume whose channels are `kind`: what a
    /// builder should pack it as (see [`crate::pack`]). 8-bit stays 8-bit;
    /// 16-bit is a 16-bit unorm when the device has the 16-bit-norm feature
    /// and a half float when it has not; float is a half float (a full float
    /// is not filterable without an optional feature).
    pub fn volume_texel_format(&self, kind: VolumeKind) -> TexelFormat {
        match kind {
            VolumeKind::U8 => TexelFormat::Unorm8,
            VolumeKind::U16 if self.volume_unorm16 => TexelFormat::Unorm16,
            VolumeKind::U16 | VolumeKind::F32 => TexelFormat::Float16,
        }
    }

    /// Upload a volume packed by [`crate::pack`], every level of every texture,
    /// and the brick maxima the renderer skips empty space with. Textures whose
    /// size and format are unchanged (a 4D timepoint step) are refilled in
    /// place; the bind group is only rebuilt when one was recreated.
    ///
    /// Fails with [`VolumeUploadError::OutOfMemory`] when the GPU had no room,
    /// leaving 1x1x1 placeholders and no volume: build a smaller one and try
    /// again.
    pub fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError> {
        volume.check().map_err(VolumeUploadError::Unsupported)?;
        let layout = &volume.layout;
        match layout.format {
            TexelFormat::Float32 => {
                return Err(VolumeUploadError::Unsupported(
                    "this device stores float volumes as half floats; pack them as Float16".into(),
                ))
            }
            TexelFormat::Unorm16 if !self.volume_unorm16 => {
                return Err(VolumeUploadError::Unsupported(
                    "this device has no 16-bit normalized textures; pack 16-bit volumes as Float16"
                        .into(),
                ))
            }
            _ => {}
        }
        let max = self.max_3d_texture_size();
        let (w, h, d) = layout.dims;
        if w > max || h > max || d > max {
            return Err(VolumeUploadError::Unsupported(format!(
                "a {w} x {h} x {d} volume is beyond this GPU's 3D texture limit of {max}"
            )));
        }

        // Per texture, what it must be, and whether the one there is not that.
        let specs: [_; 2] = std::array::from_fn(|t| {
            let (size, format, levels) = match layout.textures.get(t) {
                Some(tex) => (
                    layout.dims,
                    packed_format(layout.format, tex.components),
                    layout.levels,
                ),
                None => ((1, 1, 1), wgpu::TextureFormat::R8Unorm, 1),
            };
            let current = &self.volume_textures[t];
            let stale = current.size()
                != wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: size.2,
                }
                || current.format() != format
                || current.mip_level_count() != levels;
            (size, format, levels, stale)
        });
        if specs.iter().any(|s| s.3) {
            // Let go of the textures being replaced before asking for room for
            // their successors, and wait until the GPU has: the old volume and
            // the new are then never in memory together. A budget counts the
            // old one as free (`volume_memory_available`), and holding both
            // would spend the margin it leaves.
            for (t, spec) in specs.iter().enumerate() {
                if spec.3 {
                    self.volume_textures[t].destroy();
                }
            }
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        }

        let mut rebind = false;
        for (t, &(size, format, levels, stale)) in specs.iter().enumerate() {
            if stale {
                // Each texture made inside a scope of its own, and checked
                // before anything is written to it: a texture the device had
                // no room for comes back invalid, and writing to it would be a
                // validation error nothing catches, which wgpu treats as fatal.
                let scope = OutOfMemoryScope::push(&self.device);
                let texture = create_volume_texture(
                    &self.device,
                    size,
                    format,
                    levels,
                    if t == 0 {
                        "FastTIFFvol a"
                    } else {
                        "FastTIFFvol b"
                    },
                );
                match scope.pop() {
                    Ok(check) => self.volume_checks[t] = check,
                    Err(why) => {
                        drop(texture);
                        self.clear_volume();
                        return Err(why);
                    }
                }
                self.volume_textures[t] = texture;
                rebind = true;
                if layout.textures.get(t).is_none() {
                    write_volume_region(
                        &self.queue,
                        &self.volume_textures[t],
                        0,
                        (1, 1),
                        1,
                        0,
                        1,
                        &[0],
                    );
                }
            }
            if layout.textures.get(t).is_some() {
                let texel = layout.texel_bytes(t);
                for (level, data) in volume.levels[t].iter().enumerate() {
                    let dims = layout.level_dims(level as u32);
                    write_volume_level(
                        &self.queue,
                        &self.volume_textures[t],
                        level as u32,
                        dims,
                        texel,
                        data,
                    );
                }
            }
        }
        self.volume_layout = Some(layout.clone());
        self.volume_bricks = Some(volume.bricks.clone());
        self.occupancy_for = None;
        self.refresh_occupancy();
        if rebind {
            self.rebuild_volume_bind_group();
        }
        Ok(())
    }

    /// Drop the volume: placeholders in its place and nothing bound that a
    /// failed upload left half made.
    fn clear_volume(&mut self) {
        for (t, label) in ["FastTIFFvol a", "FastTIFFvol b"].into_iter().enumerate() {
            self.volume_textures[t] = create_volume_texture(
                &self.device,
                (1, 1, 1),
                wgpu::TextureFormat::R8Unorm,
                1,
                label,
            );
            write_volume_region(
                &self.queue,
                &self.volume_textures[t],
                0,
                (1, 1),
                1,
                0,
                1,
                &[0],
            );
        }
        self.volume_layout = None;
        self.volume_bricks = None;
        self.occupancy_for = None;
        self.volume_checks = [None, None];
        self.rebuild_volume_bind_group();
    }

    /// Collect the answers of [`volume_checks`](Self::volume_checks) that have
    /// come. A refusal drops the volume, as a failed upload does, and is kept
    /// for [`take_late_upload_error`](Self::take_late_upload_error). Returns
    /// whether the volume may be drawn: every answer is in.
    fn settle_volume_checks(&mut self) -> bool {
        let mut refused = None;
        for check in &mut self.volume_checks {
            let answer = check
                .as_ref()
                .and_then(|c| c.lock().ok().and_then(|a| a.clone()));
            match answer {
                None => {}
                Some(None) => *check = None,
                Some(Some(why)) => refused = Some(why),
            }
        }
        if let Some(why) = refused {
            let bytes = self.volume_layout.as_ref().map_or(0, |l| l.total_bytes());
            self.clear_volume();
            self.volume_lost = Some((VolumeUploadError::OutOfMemory(why), bytes));
        }
        self.volume_checks.iter().all(Option::is_none)
    }

    /// An upload that returned `Ok` and was then refused for want of memory,
    /// and the bytes its volume took. Only WebGPU does this: it says whether
    /// there was room through a promise an upload cannot wait for. The volume
    /// is gone, as after an upload that failed, and `None` is returned from
    /// then on: build a smaller one. Ask once a frame; until the answer is in,
    /// the volume is not drawn.
    pub fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)> {
        self.settle_volume_checks();
        self.volume_lost.take()
    }

    /// Whether the volume is waiting for the device to say there was room for
    /// it (WebGPU only; see [`take_late_upload_error`]). It is not drawn until
    /// then, and nothing but the host can ask for the frame that draws it:
    /// keep frames coming while this is true.
    ///
    /// [`take_late_upload_error`]: Self::take_late_upload_error
    pub fn volume_pending(&mut self) -> bool {
        !self.settle_volume_checks()
    }

    /// (Re)upload the volume, one entry per channel: each `bytes` is `w*h*d`
    /// samples in native `kind` format. Packs it here, on the calling thread —
    /// a builder with a thread of its own should pack it there instead (see
    /// [`crate::pack`]) and call [`upload_packed_volume`].
    ///
    /// [`upload_packed_volume`]: Self::upload_packed_volume
    pub fn upload_volumes(
        &mut self,
        w: u32,
        h: u32,
        d: u32,
        channels: &[(VolumeKind, Vec<u8>)],
    ) -> Result<(), VolumeUploadError> {
        let kinds: Vec<VolumeKind> = channels.iter().map(|(k, _)| *k).collect();
        let format = self.volume_texel_format(pack::common_kind(&kinds));
        let borrowed: Vec<(VolumeKind, &[u8])> =
            channels.iter().map(|(k, b)| (*k, b.as_slice())).collect();
        let packed = PackedVolume::from_channels((w, h, d), &borrowed, format);
        self.upload_packed_volume(&packed)
    }

    /// Rebuild the volume bind group after a texture was recreated.
    fn rebuild_volume_bind_group(&mut self) {
        self.volume_bind_group = build_volume_bind_group(
            &self.device,
            &self.volume_bind_group_layout,
            &self.volume_uniform_buffer,
            &self.volume_textures,
            &self.occupancy_texture,
            &self.lut_texture,
            &self.sampler,
        );
    }

    /// Work out which bricks can be seen under the current params, if that
    /// changed, and upload it.
    fn refresh_occupancy(&mut self) {
        let (Some(bricks), Some(params)) = (&self.volume_bricks, &self.volume_params) else {
            return;
        };
        let vis = Visibility::of(params);
        if self.occupancy_for == Some(vis) {
            return;
        }
        let occ = pack::occupancy(bricks, &vis);
        let (bx, by, bz) = bricks.dims;
        let want = wgpu::Extent3d {
            width: bx,
            height: by,
            depth_or_array_layers: bz,
        };
        if self.occupancy_texture.size() != want {
            self.occupancy_texture = create_volume_texture(
                &self.device,
                (bx, by, bz),
                wgpu::TextureFormat::R8Unorm,
                1,
                "FastTIFFvol occupancy",
            );
            self.rebuild_volume_bind_group();
        }
        write_volume_level(
            &self.queue,
            &self.occupancy_texture,
            0,
            (bx, by, bz),
            1,
            &occ,
        );
        self.occupancy_for = Some(vis);
    }

    /// Switch the two accelerations on or off: skipping parts of the volume
    /// that cannot be seen (exact: the picture is the same either way) and, in
    /// the maximum projection, reading coarser mip levels where a voxel is
    /// smaller than a pixel. Both are on unless switched off; off is for
    /// comparing, measuring, or ruling them out.
    pub fn set_volume_acceleration(&mut self, skip_empty: bool, mip_levels: bool) {
        self.accel_skip = skip_empty;
        self.accel_lod = mip_levels;
    }

    /// The size the 3D view is painted at, in pixels, and how many times
    /// smaller per axis to render it: 1 renders every pixel, and more renders a
    /// reduced picture that [`paint_volume`] scales up — for a view that is
    /// moving, where a sharp frame is wasted (see [`crate::MOVING_DOWNSCALE`]).
    /// Set every frame, before [`prepare_volume`]. Until it is set the size of
    /// a pixel is unknown, so the view renders full size and reads no coarser
    /// mip level than the finest.
    ///
    /// [`paint_volume`]: Self::paint_volume
    /// [`prepare_volume`]: Self::prepare_volume
    pub fn set_volume_target(&mut self, width: u32, height: u32, downscale: u32) {
        self.volume_target = (width, height);
        self.volume_downscale = downscale.max(1);
    }

    /// Store the sampling mode for the shader (0 = nearest, 1 = linear, 2 = cubic).
    /// A single linear sampler serves all three (nearest snaps to texel centers,
    /// cubic reconstructs in-shader), so no GPU state changes here.
    pub fn set_volume_interp(&mut self, interp: VolumeInterp) {
        self.volume_interp_mode = match interp {
            VolumeInterp::Nearest => 0,
            VolumeInterp::Linear => 1,
            VolumeInterp::Cubic => 2,
        };
    }

    /// Stash the ray-march params. They reach the GPU in `prepare_volume`,
    /// which the host must call before `paint_volume`.
    pub fn set_volume_params(&mut self, params: VolumeParams) {
        self.volume_params = Some(params);
        self.refresh_occupancy();
    }

    /// Marshal the stashed params into the volume uniform buffer for a
    /// full-size render. Kept for a host that does not reduce the view: it is
    /// [`prepare_volume`] with nothing to record. No-op until
    /// `set_volume_params` has run.
    ///
    /// [`prepare_volume`]: Self::prepare_volume
    pub fn write_volume_uniform(&mut self) {
        self.lowres_ready = false;
        self.settle_volume_checks();
        if let Some(p) = self.volume_params {
            self.write_volume_uniform_with(&p, self.volume_target.1);
        }
    }

    /// Get the frame ready: marshal the params into the uniform buffer and,
    /// when [`set_volume_target`] asked for a reduced render, record that
    /// render into `encoder` (it then runs ahead of the host's own pass, which
    /// [`paint_volume`] scales it into). Under egui_wgpu this is the
    /// callback's `prepare`, with the encoder it is handed.
    ///
    /// [`set_volume_target`]: Self::set_volume_target
    /// [`paint_volume`]: Self::paint_volume
    pub fn prepare_volume(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.lowres_ready = false;
        if !self.settle_volume_checks() {
            return;
        }
        let Some(p) = self.volume_params else { return };
        let (w, h) = self.volume_target;
        let f = self.volume_downscale;
        if f <= 1 || w == 0 || h == 0 {
            self.write_volume_uniform_with(&p, h);
            return;
        }
        let size = (w.div_ceil(f).max(1), h.div_ceil(f).max(1));
        self.write_volume_uniform_with(&p, size.1);
        self.ensure_lowres(size);
        let lowres = self.lowres.as_ref().expect("made above");
        let view = lowres.texture.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FastTIFFvolume reduced"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.volume_pipeline);
        pass.set_bind_group(0, &self.volume_bind_group, &[]);
        pass.draw(0..6, 0..1);
        drop(pass);
        self.lowres_ready = true;
    }

    /// The reduced-size render target, `size` pixels, made or remade to fit.
    fn ensure_lowres(&mut self, size: (u32, u32)) {
        let fits = self.lowres.as_ref().is_some_and(|l| {
            let s = l.texture.size();
            (s.width, s.height) == size
        });
        if fits {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FastTIFFvolume reduced"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FastTIFFvolume reduced bind group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.lowres = Some(LowRes {
            texture,
            bind_group,
        });
    }

    /// Marshal `p` + the interp mode and the accelerations into the volume
    /// uniform buffer, for a render `height` pixels tall (0: not known, and
    /// then no coarser mip level is read).
    ///
    /// Empty space is skipped only where the occupancy was worked out for
    /// `p`'s own windows: a capture renders the params it was asked for over
    /// several frames, while the view's change underneath it.
    fn write_volume_uniform_with(&self, p: &VolumeParams, height: u32) {
        let levels = self.volume_layout.as_ref().map_or(1, |l| l.levels);
        let mut gpu = VolParamsGpu {
            channels: [[0.0; 4]; MAX_CHANNELS],
            cam_eye: [p.eye[0], p.eye[1], p.eye[2], 0.0],
            cam_forward: [p.forward[0], p.forward[1], p.forward[2], 0.0],
            cam_right: [p.right[0], p.right[1], p.right[2], 0.0],
            cam_up: [p.up[0], p.up[1], p.up[2], 0.0],
            box_he: [p.box_he[0], p.box_he[1], p.box_he[2], 0.0],
            misc: [p.tan_half_fov, p.aspect, p.density, p.iso],
            gain: [p.brightness, 0.0, 0.0, 0.0],
            modes: [
                p.num_channels,
                p.render_mode,
                self.volume_interp_mode,
                p.max_steps,
            ],
            accel: accel_uniform(
                p,
                height,
                if self.accel_lod && height > 0 {
                    levels - 1
                } else {
                    0
                },
                self.accel_skip && self.occupancy_for == Some(Visibility::of(p)),
            ),
        };
        for c in 0..MAX_CHANNELS {
            gpu.channels[c] = [
                p.windows[c * 2],
                p.windows[c * 2 + 1],
                p.enabled[c],
                p.albedo_t[c],
            ];
        }
        self.queue
            .write_buffer(&self.volume_uniform_buffer, 0, bytemuck::bytes_of(&gpu));
    }

    /// Draw the volume into the host's pass: the ray-march itself, or — when
    /// `prepare_volume` rendered a reduced picture this frame — that picture,
    /// scaled up to fill the pass's viewport.
    pub fn paint_volume(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Nothing until the device has said there was room for the volume
        // (see `volume_checks`), which `prepare_volume` waits for.
        if self.volume_params.is_none() || self.volume_checks.iter().any(Option::is_some) {
            return;
        }
        if let (true, Some(lowres)) = (self.lowres_ready, &self.lowres) {
            render_pass.set_pipeline(&self.blit_pipeline);
            render_pass.set_bind_group(0, &lowres.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
            return;
        }
        render_pass.set_pipeline(&self.volume_pipeline);
        render_pass.set_bind_group(0, &self.volume_bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }

    /// Render the volume off screen at `width` x `height` pixels, with the
    /// params last given to `set_volume_params`, and read it back.
    ///
    /// The ray-march has no resolution of its own (every pixel is a ray), so
    /// this is the view at any size: sharper than the window when larger, and
    /// with nothing the host draws over the view in it. The aspect ratio is
    /// taken from the size, not from the params.
    ///
    /// Rendered in tiles of [`CAPTURE_TILE`] pixels, one GPU job and one
    /// readback each, so that no single job runs long enough for the system
    /// to reset the GPU, and GPU memory holds one tile rather than the
    /// picture. The first tile starts here; [`take_volume_capture`] collects
    /// each as it lands and starts the next, and returns the picture once the
    /// last is in. A readback cannot be waited for in a browser, so a capture
    /// takes a frame or more a tile. One capture at a time.
    ///
    /// [`take_volume_capture`]: Self::take_volume_capture
    pub fn capture_volume(&mut self, width: u32, height: u32) -> Result<(), String> {
        let Some(params) = self.volume_params else {
            return Err("there is no volume to render".into());
        };
        let max = self.max_2d_texture_size();
        if width == 0 || height == 0 || width > max || height > max {
            return Err(format!(
                "a {width} x {height} picture is beyond what this GPU can render (up to {max} a side)"
            ));
        }
        if swizzle(self.target_format).is_none() {
            return Err(format!(
                "cannot read back a render in {:?}",
                self.target_format
            ));
        }
        if self.capture.is_some() {
            return Err("the last capture has not been taken yet".into());
        }
        if self.capture_target.is_none() {
            self.capture_target = Some(self.create_capture_target());
        }
        self.capture = Some(CaptureJob {
            width,
            height,
            params: VolumeParams {
                aspect: width as f32 / height as f32,
                ..params
            },
            tiles: capture_tiles(width, height),
            next: 0,
            rgba: vec![0; width as usize * height as usize * 4],
            mapped: Arc::new(Mutex::new(None)),
            started: false,
        });
        // Not before the device has said there was room for the volume:
        // `take_volume_capture` starts it then.
        if self.settle_volume_checks() {
            self.submit_capture_tile();
        }
        Ok(())
    }

    /// The picture [`capture_volume`] asked for, once every tile of it has
    /// arrived: `None` while it is still being rendered, or when none was
    /// asked for. Call once a frame until it answers; each call collects the
    /// tile that has landed and starts the next.
    ///
    /// [`capture_volume`]: Self::capture_volume
    pub fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>> {
        if !self.capture.as_ref()?.started {
            if !self.settle_volume_checks() {
                return None;
            }
            if self.volume_layout.is_none() {
                // Refused after its upload returned: gone, not black.
                self.capture = None;
                return Some(Err("the GPU had no room for the volume".into()));
            }
            self.submit_capture_tile();
            return None;
        }
        // Drives the map callback on native; in a browser the browser does.
        let _ = self.device.poll(wgpu::PollType::Poll);
        let job = self.capture.as_mut().expect("checked above");
        let result = job
            .mapped
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        if let Err(why) = result {
            self.capture = None;
            return Some(Err(format!("the render could not be read back: {why}")));
        }
        let target = self.capture_target.as_ref().expect("made with the job");
        let order = swizzle(self.target_format).expect("checked when captured");
        let (x, y, w, h) = job.tiles[job.next];
        let stride = job.width as usize * 4;
        {
            let data = target.buffer.slice(..).get_mapped_range();
            for (r, row) in data
                .chunks(target.padded_row as usize)
                .take(h as usize)
                .enumerate()
            {
                let at = (y as usize + r) * stride + x as usize * 4;
                let out = &mut job.rgba[at..at + w as usize * 4];
                for (o, px) in out.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                    o.copy_from_slice(&[px[order[0]], px[order[1]], px[order[2]], 255]);
                }
            }
        }
        target.buffer.unmap();
        job.next += 1;
        if job.next < job.tiles.len() {
            self.submit_capture_tile();
            return None;
        }
        let job = self.capture.take().expect("checked above");
        Some(Ok(VolumeImage {
            width: job.width,
            height: job.height,
            rgba: job.rgba,
        }))
    }

    /// Render the job's next tile into the tile texture, copy it to the
    /// readback buffer and start mapping it.
    fn submit_capture_tile(&mut self) {
        let Some(job) = self.capture.as_mut() else {
            return;
        };
        job.started = true;
        let job = &*job;
        let tile = job.tiles[job.next];
        let params = tile_params(&job.params, (job.width, job.height), tile);
        let mapped = Arc::clone(&job.mapped);
        let (_, _, w, h) = tile;
        // This tile's uniform, ahead of its submit. The view's own paint
        // writes the uniform again before its draw, from its own params.
        self.write_volume_uniform_with(&params, h);

        let target = self.capture_target.as_ref().expect("made with the job");
        let view = target.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("volume capture"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volume capture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            // An edge tile is drawn into the corner of the tile texture, at
            // its own size, so its pixels are the picture's pixels.
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_pipeline(&self.volume_pipeline);
            pass.set_bind_group(0, &self.volume_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &target.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        *mapped.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let slot = Arc::clone(&mapped);
        target
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                *slot.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(result.map_err(|e| e.to_string()));
            });
    }

    fn create_capture_target(&self) -> CaptureTarget {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume capture tile"),
            size: wgpu::Extent3d {
                width: CAPTURE_TILE,
                height: CAPTURE_TILE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let padded_row = (CAPTURE_TILE * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume capture readback"),
            size: padded_row as u64 * CAPTURE_TILE as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        CaptureTarget {
            texture,
            buffer,
            padded_row,
        }
    }

    /// (Re)allocate channel textures for the current frame size and per-channel
    /// `kind`. An `Int8`/`Int16` channel gets a full-size R8Uint/R16Uint integer
    /// texture (float texture a 1x1 dummy); a `Float` channel gets a full-size
    /// R32F float texture (integer texture a 1x1 dummy); channels past
    /// `kinds.len()` are unused (both 1x1). Rebuilds the bind group when anything
    /// changed; no-op otherwise.
    /// Largest per-axis 2D-texture dimension the device supports.
    ///
    /// Frames wider or taller than this cannot be uploaded at all — `wgpu`
    /// rejects the texture and, since errors are fatal by default, takes the
    /// process with it. The app subsamples to fit rather than letting that
    /// happen; see `fast_tiff_viewer::sync::fit_stride`.
    pub fn max_2d_texture_size(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind]) {
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
        let device = &self.device;
        self.channel_textures = std::array::from_fn(|c| {
            let label = format!("FastTIFFchannel {c}");
            // R8Uint or R16Uint at full size for an integer channel, else a 1x1
            // R16Uint dummy. Both are sampled through the same `texture_2d<u32>`.
            match want[c] {
                KIND_INT8 => {
                    create_int_texture(device, width, height, wgpu::TextureFormat::R8Uint, &label)
                }
                KIND_INT16 => {
                    create_int_texture(device, width, height, wgpu::TextureFormat::R16Uint, &label)
                }
                _ => create_int_texture(device, 1, 1, wgpu::TextureFormat::R16Uint, &label),
            }
        });
        self.channel_ftextures = std::array::from_fn(|c| {
            let (w, h) = if want[c] == KIND_FLOAT {
                (width, height)
            } else {
                (1, 1)
            };
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
    pub fn upload_channel_u16(&self, channel: usize, width: u32, height: u32, samples: &[u16]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        self.queue.write_texture(
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
    pub fn upload_channel_u8(&self, channel: usize, width: u32, height: u32, samples: &[u8]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        self.queue.write_texture(
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
    pub fn upload_channel_f32(&self, channel: usize, width: u32, height: u32, samples: &[f32]) {
        if channel >= MAX_CHANNELS {
            return;
        }
        self.queue.write_texture(
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
    pub fn upload_lut(&self, channel: usize, lut: &Lut) {
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
        self.queue.write_texture(
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
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&gpu));
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

fn create_float_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
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

/// The `accel` uniform: the size of a pixel at unit distance for a render
/// `height` pixels tall (a capture tile's narrowed params give the same value
/// as the whole picture's), the highest mip level a MIP may read, the brick
/// size, and whether to skip empty bricks.
fn accel_uniform(p: &VolumeParams, height: u32, max_level: u32, skip: bool) -> [f32; 4] {
    [
        2.0 * p.tan_half_fov / height.max(1) as f32,
        max_level as f32,
        pack::BRICK as f32,
        if skip { 1.0 } else { 0.0 },
    ]
}

/// The texture format for a packed texture of `components` components stored
/// as `format`. One or two components keep their own; three or four are RGBA.
fn packed_format(format: TexelFormat, components: usize) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    match (format, components) {
        (TexelFormat::Unorm8, 1) => F::R8Unorm,
        (TexelFormat::Unorm8, 2) => F::Rg8Unorm,
        (TexelFormat::Unorm8, _) => F::Rgba8Unorm,
        (TexelFormat::Unorm16, 1) => F::R16Unorm,
        (TexelFormat::Unorm16, 2) => F::Rg16Unorm,
        (TexelFormat::Unorm16, _) => F::Rgba16Unorm,
        (TexelFormat::Float16, 1) => F::R16Float,
        (TexelFormat::Float16, 2) => F::Rg16Float,
        (TexelFormat::Float16, _) => F::Rgba16Float,
        (TexelFormat::Float32, 1) => F::R32Float,
        (TexelFormat::Float32, 2) => F::Rg32Float,
        (TexelFormat::Float32, _) => F::Rgba32Float,
    }
}

/// What `create_volume_resources` makes.
struct VolumeResources {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    textures: [wgpu::Texture; 2],
    occupancy: wgpu::Texture,
}

/// Build the volume pipeline, its bind-group layout + initial (1x1x1 dummy)
/// bind group, and the uniform buffer. Reuses the 2D LUT texture + sampler.
fn create_volume_resources(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> VolumeResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FastTIFFvolume shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/volume.wgsl").into()),
    });

    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<VolParamsGpu>() as u64),
            },
            count: None,
        },
        volume_texture_entry(1),
        volume_texture_entry(2),
        volume_texture_entry(3),
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
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

    let placeholder = |label: &str| {
        create_volume_texture(device, (1, 1, 1), wgpu::TextureFormat::R8Unorm, 1, label)
    };
    let textures = [placeholder("FastTIFFvol a"), placeholder("FastTIFFvol b")];
    let occupancy = placeholder("FastTIFFvol occupancy");
    let bind_group = build_volume_bind_group(
        device,
        &layout,
        &uniform_buffer,
        &textures,
        &occupancy,
        lut_texture,
        sampler,
    );

    VolumeResources {
        pipeline,
        layout,
        bind_group,
        uniform_buffer,
        textures,
        occupancy,
    }
}

/// The pipeline that scales a reduced render of the view up into the host's
/// pass, and its bind-group layout (the render, a sampler).
fn create_blit_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("FastTIFFvolume blit shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FastTIFFvolume blit bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("FastTIFFvolume blit pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("FastTIFFvolume blit pipeline"),
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
    (pipeline, layout)
}

fn volume_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            // Every volume format used here is filterable: the 8-bit unorms
            // and half floats in core, the 16-bit unorms under the
            // TEXTURE_FORMAT_16BIT_NORM feature they need anyway.
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn create_volume_texture(
    device: &wgpu::Device,
    (w, h, d): (u32, u32, u32),
    format: wgpu::TextureFormat,
    levels: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: d.max(1),
        },
        mip_level_count: levels.max(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Bound on one write of a volume level: a level is written in z-slabs of at
/// most this many bytes, so no single staging copy is the whole volume.
const VOLUME_UPLOAD_CHUNK_BYTES: usize = 32 << 20;

/// Write one whole mip `level` (`dims` texels of `texel` bytes, x fastest) of a
/// 3D texture, in z-slabs, each submitted before the next is written.
///
/// The submit is what bounds the staging memory: a write is copied into a
/// staging buffer that is only freed once a submit has carried it to the GPU,
/// so without one every slab of a large volume is staged at once — and a
/// staging buffer the device has no room for loses the device.
fn write_volume_level(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    level: u32,
    (w, h, d): (u32, u32, u32),
    texel: usize,
    data: &[u8],
) {
    let slice = w as usize * h as usize * texel;
    let per_chunk = (VOLUME_UPLOAD_CHUNK_BYTES / slice.max(1)).clamp(1, d.max(1) as usize);
    let mut z0 = 0usize;
    while z0 < d as usize {
        let zn = per_chunk.min(d as usize - z0);
        write_volume_region(
            queue,
            texture,
            level,
            (w, h),
            texel,
            z0 as u32,
            zn as u32,
            &data[z0 * slice..(z0 + zn) * slice],
        );
        queue.submit(std::iter::empty());
        z0 += zn;
    }
}

/// Write a z-slab of level `level` (`data` = `w*h*d` texels of `texel` bytes,
/// starting at `z0`) into a 3D texture.
#[allow(clippy::too_many_arguments)]
fn write_volume_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    level: u32,
    (w, h): (u32, u32),
    texel: usize,
    z0: u32,
    d: u32,
    data: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d { x: 0, y: 0, z: z0 },
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * texel as u32),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        },
    );
}

fn build_volume_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    textures: &[wgpu::Texture; 2],
    occupancy: &wgpu::Texture,
    lut_texture: &wgpu::Texture,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
    let (a, b, occ) = (view(&textures[0]), view(&textures[1]), view(occupancy));
    let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("FastTIFFvolume bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&a),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&b),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&occ),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&lut_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Where an out-of-memory answer that comes late is left: `None` until it
/// comes, then `Some(None)` for room and `Some(Some(why))` for none.
type LateCheck = std::sync::Arc<std::sync::Mutex<Option<Option<String>>>>;

/// Catches the device running out of memory while a volume is made, so the
/// upload can say so instead of the device being lost.
struct OutOfMemoryScope {
    guard: wgpu::ErrorScopeGuard,
}

impl OutOfMemoryScope {
    fn push(device: &wgpu::Device) -> Self {
        OutOfMemoryScope {
            guard: device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
        }
    }

    /// The device's answer: an error if there was no room. Natively it is
    /// there at once. WebGPU gives it through a promise, which resolves only
    /// once the page's code yields; then this returns where it will be left
    /// (see `volume_checks`).
    fn pop(self) -> Result<Option<LateCheck>, VolumeUploadError> {
        use std::future::Future as _;
        let mut answer = Box::pin(self.guard.pop());
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match answer.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(Some(e)) => Err(VolumeUploadError::OutOfMemory(e.to_string())),
            std::task::Poll::Ready(None) => Ok(None),
            #[cfg(target_arch = "wasm32")]
            std::task::Poll::Pending => {
                let check = LateCheck::default();
                let slot = check.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let why = answer.await.map(|e| e.to_string());
                    if let Ok(mut slot) = slot.lock() {
                        *slot = Some(why);
                    }
                });
                Ok(Some(check))
            }
            // Never pending off the web: wgpu-core answers synchronously.
            #[cfg(not(target_arch = "wasm32"))]
            std::task::Poll::Pending => Ok(None),
        }
    }
}

/// Where red, green and blue sit in a pixel of `format`, for the formats a
/// render can be read back from. Both are 8-bit; an sRGB one holds the bytes
/// the screen shows, the same as a plain one does.
fn swizzle(format: wgpu::TextureFormat) -> Option<[usize; 3]> {
    use wgpu::TextureFormat as F;
    match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => Some([0, 1, 2]),
        F::Bgra8Unorm | F::Bgra8UnormSrgb => Some([2, 1, 0]),
        _ => None,
    }
}

//! GPU rendering for a composited multi-channel image and its 3D volume,
//! independent of any GUI toolkit.
//!
//! The crate is two interchangeable backends over one set of parameter types:
//!   * [`wgpu_backend`] — DX12 / Vulkan / Metal / WebGPU (feature `backend-wgpu`)
//!   * [`glow_backend`] — desktop OpenGL 3.1 or later (feature `backend-glow`);
//!     its shaders are GLSL 1.40, which WebGL2 (GLSL ES 3.00) does not take
//!
//! Both backends' `ImageRenderResources` implement the two traits of
//! [`renderer`], [`ImageRenderer`] (the 2D composite) and [`VolumeRenderer`]
//! (the 3D view), so a host writes its per-frame code once, generic over them,
//! and names a backend only where it constructs one and hands it to its
//! drawing layer: the two `paint` methods cannot share a signature (a wgpu
//! render pass vs. a live GL context), and they stay inherent. Generic, not
//! `dyn`: the host's code is compiled for the backend it hands in. The
//! backends are additive, so a host may build both.
//!
//! What both backends work out on the CPU — the composite's uniform block,
//! the LUT texels, the volume's acceleration uniform — is in [`uniforms`],
//! and how a device stores a volume is [`VolumeCaps`], so the backends differ
//! only in what they hand the GPU.
//!
//! # What a host owns, and what this crate owns
//!
//! This crate owns GPU resources — pipelines, textures, uniform buffers — and
//! the math that fills them. It never creates a device, a surface, or a window,
//! and it never draws a frame on its own: construction takes the host's already
//! initialized device (or GL context), and painting takes the host's render
//! pass. That's the whole boundary, and it's what lets the same renderer sit
//! under a native eframe app and a browser canvas.
//!
//! ```ignore
//! use scivis_render::ImageRenderer as _;
//!
//! // Once, at startup — from whatever created your device:
//! let mut r = ImageRenderResources::new(device.clone(), queue.clone(), target_format);
//!
//! // Whenever the stack or its pixel layout changes:
//! r.ensure_size(width, height, &[ChannelKind::Int16, ChannelKind::Int16]);
//! r.upload_lut(0, &lut);
//!
//! // Per frame:
//! r.upload_channel_u16(0, width, height, &pixels);
//! r.set_params(&uniforms, n_channels, uv_offset, uv_scale);
//! // ...then, inside your own render pass:
//! r.paint(&mut render_pass);
//! ```
//!
//! Everything that changes what the renderer holds takes `&mut self`, the
//! uploads included (see [`ImageRenderer`] for why).

#[cfg(any(feature = "backend-wgpu", feature = "backend-glow"))]
mod cpu_state;
#[cfg(feature = "backend-glow")]
pub mod glow_backend;
pub mod pack;
pub mod renderer;
pub mod shaders;
pub mod uniforms;
#[cfg(feature = "backend-wgpu")]
pub mod vram;
#[cfg(feature = "backend-wgpu")]
pub mod wgpu_backend;

pub use renderer::{ImageRenderer, VolumeCaps, VolumeRenderer};

/// Maximum number of display channels composited at once. Shared by both
/// backends (texture/uniform array sizes, written into their shaders by
/// [`shaders`]) and by the host's channel state.
///
/// At most 7, which the build asserts: the 2D composite binds two textures a
/// channel and the LUTs to one shader, of the 16 every GPU allows it, and
/// the volume packs its channels four to a texture, in two. Within that, a
/// change here reaches every array, LUT row and binding on its own; the
/// shader tests then say which per-channel lines need writing out for it.
pub const MAX_CHANNELS: usize = 6;

/// A channel's display lookup table: 256 RGB entries, indexed by windowed
/// intensity (0 = `lut[0]`, 255 = `lut[255]`).
pub type Lut = [[u8; 3]; 256];

/// Where along `lut` its brightest entry sits, as a 0..1 sample position —
/// what [`VolumeParams::albedo_t`] wants.
///
/// For an ordinary ramp this is 1.0 (the top entry), so nothing changes. It
/// differs only for a LUT that peaks early and falls away, which is exactly the
/// case that made the isosurface render black.
pub fn brightest_lut_t(lut: &Lut) -> f32 {
    let brightest = lut
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| c[0] as u32 + c[1] as u32 + c[2] as u32)
        .map(|(i, _)| i)
        .unwrap_or(255);
    brightest as f32 / 255.0
}

/// How a channel's pixels are stored in its GPU texture. Picked per channel from
/// the source format so each gets the cheapest upload, while the shader stays
/// uniform (the two integer kinds share one `usampler2D`/`texture_2d<u32>` — the
/// window/level units differ, which the host accounts for):
///   * `Int8` — `R8Uint`, raw unsigned 8-bit bytes (zero-copy, no widening).
///   * `Int16` — `R16Uint`, the default integer path (16-bit native, or 8-bit
///     signed / 32-bit int rescaled into 0..65535 on the CPU).
///   * `Float` — `R32F`, raw 32-bit float (window/level done on the GPU).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelKind {
    Int8,
    Int16,
    Float,
}

impl ChannelKind {
    /// Bytes one sample of this kind takes in its 2D texture.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            ChannelKind::Int8 => 1,
            ChannelKind::Int16 => 2,
            ChannelKind::Float => 4,
        }
    }

    /// How a channel of this kind is stored in the 3D volume, which mirrors
    /// the 2D display.
    pub const fn volume_kind(self) -> VolumeKind {
        match self {
            ChannelKind::Int8 => VolumeKind::U8,
            ChannelKind::Int16 => VolumeKind::U16,
            ChannelKind::Float => VolumeKind::F32,
        }
    }
}

impl From<ChannelKind> for VolumeKind {
    fn from(kind: ChannelKind) -> Self {
        kind.volume_kind()
    }
}

/// How the 3D volume's scalar samples are stored in its GPU texture. Chosen
/// from channel 0's `ChannelKind` so the volume mirrors the 2D display:
///   * `U8` — `R8` unorm (8-bit source)
///   * `U16` — `R16` unorm (16-bit source, or CPU-widened 8-bit/rescaled ints)
///   * `F32` — a float texture (32-bit float source): `R32F` on glow, and a
///     half float on wgpu, where full float is not filterable without an
///     optional feature ([`VolumeCaps`] says which a device has). A builder
///     should normalize float samples to a span of about `0..=1` before
///     packing them, as the viewer does, since a half float holds nothing
///     past 65504.
///
/// Unlike the 2D integer path (which uses `usampler`, NEAREST-only), the volume
/// uses *normalized* textures so trilinear interpolation is available.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VolumeKind {
    U8,
    U16,
    F32,
}

/// Volume texture sampling: `Nearest` (crisp voxels), `Linear` (hardware
/// trilinear), or `Cubic` (in-shader tricubic B-spline — smoother than linear,
/// 8 trilinear taps per sample). `Nearest`/`Linear` set the GL min/mag filter;
/// `Cubic` uses the GL linear filter plus the shader's cubic reconstruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VolumeInterp {
    Nearest,
    Linear,
    Cubic,
}

impl VolumeInterp {
    /// The code both backends' volume shaders branch on (`INTERP_NEAREST`,
    /// `INTERP_LINEAR` and `INTERP_CUBIC` in their sources, written in by
    /// [`shaders`]). The glow backend leaves nearest and linear to the GL
    /// filter and branches only on cubic; the wgpu backend, with one linear
    /// sampler for all three, snaps a nearest read to the texel itself.
    pub const fn shader_mode(self) -> i32 {
        match self {
            VolumeInterp::Nearest => 0,
            VolumeInterp::Linear => 1,
            VolumeInterp::Cubic => 2,
        }
    }
}

/// How the ray-marcher turns samples along a ray into a pixel:
///   * `Mip` — maximum-intensity projection (brightest sample wins; the default,
///     order-independent, good for sparse/bright structures).
///   * `Alpha` — emission-absorption alpha compositing, à la the ImageJ 3D
///     Viewer's "Volume" mode: a translucent, depth-cued render where intensity
///     drives both color (LUT) and opacity.
///   * `Surface` — an opaque isosurface: the ray stops at the first voxel whose
///     windowed intensity crosses the `iso` threshold, and the hit is shaded from
///     the field gradient (a solid, depth-cued surface).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VolumeRender {
    Mip,
    Alpha,
    Surface,
}

impl VolumeRender {
    /// The code both backends' volume shaders branch on (`MODE_MIP`,
    /// `MODE_ALPHA` and `MODE_SURFACE` in their sources, written in by
    /// [`shaders`]).
    pub const fn shader_mode(self) -> i32 {
        match self {
            VolumeRender::Mip => 0,
            VolumeRender::Alpha => 1,
            VolumeRender::Surface => 2,
        }
    }
}

/// Everything the ray-march fragment shader needs for one 3D frame. The camera
/// is passed as an explicit basis (rather than matrices) so the shader builds
/// per-pixel rays with no matrix inverse; the host computes it from its orbit
/// angles + zoom. Distances/positions are in the volume's own normalized box
/// space, whose half-extents `box_he` already fold in the per-axis dimension
/// scale (voxel anisotropy). The per-channel arrays mirror the 2D compositor:
/// each channel MIP-projects independently, then colors through its own LUT row
/// (= channel index) and the results are summed.
#[derive(Clone, Copy, Debug)]
pub struct VolumeParams {
    /// Number of channels composited (≤ [`MAX_CHANNELS`]).
    pub num_channels: i32,
    /// Per-channel window/level as flat `(min, max)` pairs, in the sampled
    /// texture's units: raw value for `F32`; the 0..65535 display window divided
    /// by 65535 for `U8`/`U16`.
    pub windows: [f32; MAX_CHANNELS * 2],
    /// Per-channel on/off (1.0 / 0.0), so toggling a channel needs no rebuild.
    pub enabled: [f32; MAX_CHANNELS],
    /// Per-channel: 1.0 if the channel's data is in the float texture, else 0.0.
    pub is_float: [f32; MAX_CHANNELS],
    /// Per-channel LUT position (0..1) to take the **isosurface albedo** from.
    ///
    /// Surface mode colours the whole surface with one fixed colour rather than
    /// the colour at the crossing value, so raising the threshold doesn't also
    /// darken the surface. That fixed colour must be a *bright* point on the
    /// LUT: sampling the top entry looks right for an ordinary ramp but renders
    /// a black surface for any LUT that ends dark — a contrast-stretched
    /// palette, say, which maxes out partway along and blacks out the rest.
    ///
    /// [`brightest_lut_t`] computes the right value; `1.0` reproduces the old
    /// top-entry behaviour. Ignored by MIP and alpha DVR.
    pub albedo_t: [f32; MAX_CHANNELS],
    /// How the ray-march composites its samples. The sample count is derived
    /// in-shader from the voxel size.
    pub render_mode: VolumeRender,
    /// Alpha-DVR opacity scale (higher = more solid). Ignored by MIP/surface.
    pub density: f32,
    /// A gain applied to the composited colour, for MIP and alpha DVR.
    ///
    /// One uniform rather than one per mode, because the shader only ever runs
    /// one of them: the host decides what the gain means for the mode it asked
    /// for. Surface mode ignores it — it returns a shaded hit rather than an
    /// accumulation, and has its own shading constants.
    ///
    /// 1.0 is "as composited". See `fast_tiff_viewer::brightness` for where the
    /// values come from and why MIP's default is below 1.
    pub brightness: f32,
    /// Isosurface threshold in windowed units (0..1). Only used by surface mode.
    pub iso: f32,
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub tan_half_fov: f32,
    pub aspect: f32,
    /// Half-extents of the volume box (largest scaled axis = 0.5).
    pub box_he: [f32; 3],
    /// The most samples one ray may take; see [`max_steps`]. Fewer than the
    /// ray would take at half-voxel spacing means a coarser spacing for that
    /// ray. `0` or less means only the shaders' own ceiling,
    /// [`SHADER_MAX_STEPS`], applies.
    pub max_steps: i32,
}

/// Texture reads one frame of the ray-march may make, across every pixel,
/// channel and sample.
///
/// The ceiling that keeps a frame finite. Without it a large volume shown
/// large asks for whatever its voxels add up to: 4096 x 4096 x 69 planes of
/// two channels, on a two-million-pixel panel, came to some fifteen billion
/// reads a frame, which a modest GPU takes many seconds over, and Windows
/// resets a GPU that is busy for more than two (the device is lost, and the
/// app's next draw panics). At this many a frame takes a fraction of a second
/// even on a small workstation card, and an ordinary volume never comes near
/// it.
pub const FRAME_SAMPLE_BUDGET: f64 = 1.5e9;

/// The ceiling the shaders put on steps per ray, whatever `max_steps` says
/// (`MAX_STEPS` in their sources, written in by [`shaders`]).
///
/// High enough that the largest volume the memory budget admits (a diagonal
/// of some 1600 voxels) still gets half-voxel sampling. The march loops run
/// as many steps as a ray needs, at most this many, so an ordinary volume
/// never pays for the headroom.
pub const SHADER_MAX_STEPS: i32 = 4096;

/// The fewest steps [`max_steps`] allows, so a huge panel still shows a
/// volume rather than a smear.
pub const MIN_STEPS: i32 = 32;

/// [`VolumeParams::max_steps`] for a frame of `pixels` pixels compositing
/// `channels` channels with `interp`: as many steps as keep the frame within
/// [`FRAME_SAMPLE_BUDGET`]. Tricubic sampling reads eight texels a sample,
/// the others one.
pub fn max_steps(pixels: f64, channels: usize, interp: VolumeInterp) -> i32 {
    let taps = match interp {
        VolumeInterp::Cubic => 8.0,
        VolumeInterp::Nearest | VolumeInterp::Linear => 1.0,
    };
    let per_step = pixels.max(1.0) * channels.clamp(1, MAX_CHANNELS) as f64 * taps;
    (FRAME_SAMPLE_BUDGET / per_step).clamp(MIN_STEPS as f64, SHADER_MAX_STEPS as f64) as i32
}

/// The side of the square tiles an off-screen capture is rendered in, in
/// pixels.
///
/// A capture is not drawn in one go. A movie frame can be 8192 pixels wide,
/// and ray-marching that many pixels through a large volume is one GPU job
/// of many seconds, which Windows ends by resetting the GPU (its two-second
/// timeout): the device is lost, and the next thing to draw panics. In tiles
/// about the size of a window, each job costs what a frame of the view on
/// screen costs, and a tile at a time is all the GPU memory a capture needs.
pub const CAPTURE_TILE: u32 = 1024;

/// The tiles a `width` x `height` capture is rendered in, row by row, as
/// `(x, y, w, h)` in pixels from the top-left. The last column and row are
/// narrower where the picture does not divide evenly.
pub fn capture_tiles(width: u32, height: u32) -> Vec<(u32, u32, u32, u32)> {
    let mut tiles = Vec::new();
    for y in (0..height).step_by(CAPTURE_TILE as usize) {
        for x in (0..width).step_by(CAPTURE_TILE as usize) {
            tiles.push((
                x,
                y,
                CAPTURE_TILE.min(width - x),
                CAPTURE_TILE.min(height - y),
            ));
        }
    }
    tiles
}

/// `params` narrowed to `tile` of a `width` x `height` picture: rendered on
/// its own, at the tile's size, it is exactly that part of the picture.
///
/// Every pixel is a ray, `forward + ndc.x * aspect * tan * right + ndc.y *
/// tan * up` with `ndc` across the target, so a part of the picture is the
/// same camera looking a little to the side (`forward` moved to the tile's
/// centre) with a narrower view (`tan` and `aspect` scaled to the tile's
/// share). The shaders normalise the ray, so `forward` need not be a unit
/// vector. The picture's aspect ratio is `width / height`, whatever `params`
/// says.
pub fn tile_params(
    params: &VolumeParams,
    (width, height): (u32, u32),
    (x, y, w, h): (u32, u32, u32, u32),
) -> VolumeParams {
    let (fw, fh) = (width.max(1) as f32, height.max(1) as f32);
    let aspect = fw / fh;
    let tan = params.tan_half_fov;
    // The tile's edges in the picture's normalised coordinates: x to the
    // right and y up, -1..1 across.
    let (left, right) = (2.0 * x as f32 / fw - 1.0, 2.0 * (x + w) as f32 / fw - 1.0);
    let (top, bottom) = (1.0 - 2.0 * y as f32 / fh, 1.0 - 2.0 * (y + h) as f32 / fh);
    let (cx, sx) = (0.5 * (left + right), 0.5 * (right - left));
    let (cy, sy) = (0.5 * (top + bottom), 0.5 * (top - bottom));
    let forward: [f32; 3] = std::array::from_fn(|i| {
        params.forward[i] + cx * aspect * tan * params.right[i] + cy * tan * params.up[i]
    });
    VolumeParams {
        forward,
        tan_half_fov: tan * sy,
        aspect: aspect * sx / sy,
        ..*params
    }
}

/// Why a volume could not be uploaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VolumeUploadError {
    /// The GPU had no room for it. A smaller volume may fit: build one and try
    /// again.
    OutOfMemory(String),
    /// Anything else: a format this backend cannot hold, a size beyond its
    /// limits.
    Unsupported(String),
}

impl std::fmt::Display for VolumeUploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VolumeUploadError::OutOfMemory(m) => write!(f, "out of GPU memory: {m}"),
            VolumeUploadError::Unsupported(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for VolumeUploadError {}

/// How much smaller than its target the 3D view is rendered while it moves,
/// per axis (then scaled up): a quarter, so a sixteenth of the rays.
pub const MOVING_DOWNSCALE: u32 = 1;

/// The volume rendered off screen, read back: what `capture_volume` produces.
///
/// The picture the 3D view would show at this size, with nothing the host
/// draws over the view: `rgba` is `width * height` pixels of 8-bit RGBA, rows
/// top to bottom, alpha 255 (the ray-march is opaque, black where a ray misses
/// the volume).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VolumeImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One channel's window/level + on/off state, as the host produces it each
/// frame. Both backends pack it into [`uniforms::CompositeParams`].
#[derive(Clone, Copy, Debug)]
pub struct ChannelUniform {
    pub min: f32,
    pub max: f32,
    pub enabled: bool,
    /// True if this channel's data is uploaded as a float (R32F) texture — i.e.
    /// 32-bit float source. The shader then samples it as a float and applies
    /// window/level in the data's own units. False = integer (R8Uint/R16Uint)
    /// channel, where `min`/`max` are in raw sample units.
    pub is_float: bool,
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

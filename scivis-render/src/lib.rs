//! GPU rendering for a composited multi-channel image and its 3D volume,
//! independent of any GUI toolkit.
//!
//! The crate is two interchangeable backends over one set of parameter types:
//!   * [`wgpu_backend`] — DX12 / Vulkan / Metal / WebGPU (feature `backend-wgpu`)
//!   * [`glow_backend`] — OpenGL / WebGL2 (feature `backend-glow`)
//!
//! Both expose the same inherent method set on their `ImageRenderResources`, so
//! a host picks one with a `#[cfg]` re-export and writes its call sites once.
//! There is deliberately **no** `dyn Renderer` trait: the two `paint` methods
//! can't share a signature (a wgpu render pass vs. a live GL context), and the
//! cfg-selected façade already gives static dispatch for free. Add a trait only
//! if you ever need both backends live in one binary.
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
//! Uploads take `&self` (they only queue GPU writes); anything that can
//! reallocate or restage takes `&mut self`.

#[cfg(feature = "backend-glow")]
pub mod glow_backend;
#[cfg(feature = "backend-wgpu")]
pub mod wgpu_backend;

/// Maximum number of display channels composited at once. Shared by both
/// backends (texture/uniform array sizes) and by the host's channel state.
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

/// How the 3D volume's scalar samples are stored in its GPU texture. Chosen
/// from channel 0's `ChannelKind` so the volume mirrors the 2D display:
///   * `U8` — `R8` unorm (8-bit source)
///   * `U16` — `R16` unorm (16-bit source, or CPU-widened 8-bit/rescaled ints)
///   * `F32` — `R32F` (32-bit float source, window/level in its own units)
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
    /// The `u_interp` value the fragment shader branches on (0 = point/linear via
    /// the GL filter, 1 = in-shader tricubic). Only the glow backend consumes it.
    pub fn shader_mode(self) -> i32 {
        match self {
            VolumeInterp::Nearest | VolumeInterp::Linear => 0,
            VolumeInterp::Cubic => 1,
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
    /// The `u_mode` value the fragment shader branches on.
    pub fn shader_mode(self) -> i32 {
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
    /// Ray-march compositing mode (see [`VolumeRender::shader_mode`]): 0 = MIP,
    /// 1 = alpha DVR, 2 = isosurface. The sample count is derived in-shader from
    /// the voxel size.
    pub render_mode: i32,
    /// Alpha-DVR opacity scale (higher = more solid). Ignored by MIP/surface.
    pub density: f32,
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
}

/// One channel's window/level + on/off state, as the host produces it each
/// frame. The backend maps it to whatever GPU representation it uses.
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

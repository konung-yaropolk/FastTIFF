//! GPU rendering for the composited image. The implementation is one of two
//! interchangeable backends, picked at compile time by the `renderer-glow` /
//! `renderer-wgpu` features (see Cargo.toml): glow (OpenGL) or wgpu
//! (DX12/Vulkan/Metal).
//!
//! Both backends expose the same backend-agnostic surface so `app.rs` and
//! `main.rs` never mention glow or wgpu:
//!   * `Render`         — a shared handle to the GPU resources (`Arc<Mutex<…>>`),
//!                        stored in the app and captured by the paint callback.
//!   * `RENDERER`       — which `eframe::Renderer` to request in `NativeOptions`.
//!   * `init`           — build the resources from the eframe creation context.
//!   * `upload_ctx`     — per-frame upload handle, pulled from `eframe::Frame`.
//!   * `paint_callback` — the egui paint callback that draws the image.
//! plus `ImageRenderResources` (the resources themselves) and the shared
//! `ChannelUniform` / `MAX_CHANNELS` defined here.

// Exactly one renderer must be selected. These guards turn an accidental
// both/neither feature set into a clear compile error instead of a confusing
// "unresolved import" cascade from the re-exports below.
#[cfg(all(feature = "renderer-glow", feature = "renderer-wgpu"))]
compile_error!(
    "features `renderer-glow` and `renderer-wgpu` are mutually exclusive — enable exactly one"
);
#[cfg(not(any(feature = "renderer-glow", feature = "renderer-wgpu")))]
compile_error!(
    "no renderer selected — enable feature `renderer-glow` (default) or `renderer-wgpu`"
);

#[cfg(feature = "renderer-glow")]
mod glow_backend;
#[cfg(feature = "renderer-glow")]
pub use glow_backend::{init, paint_callback, paint_volume_callback, upload_ctx, Render, BACKEND, RENDERER};

// The `not(renderer-glow)` guard means that turning on *both* features (a hard
// error, above) selects only the glow backend here — so the build fails with
// the clean `compile_error!` message instead of a duplicate-symbol cascade.
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
mod wgpu_backend;
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
pub use wgpu_backend::{init, paint_callback, paint_volume_callback, upload_ctx, Render, BACKEND, RENDERER};

/// Maximum number of display channels composited at once. Shared by both
/// backends (texture/uniform array sizes) and by `app.rs`.
pub const MAX_CHANNELS: usize = 6;

/// How a channel's pixels are stored in its GPU texture. Picked per channel from
/// the source format so each gets the cheapest upload, while the shader stays
/// uniform (the two integer kinds share one `usampler2D`/`texture_2d<u32>` — the
/// window/level units differ, which `app.rs` accounts for):
///   * `Int8`  — `R8Uint`,  raw unsigned 8-bit bytes (zero-copy, no widening).
///   * `Int16` — `R16Uint`, the default integer path (16-bit native, or 8-bit
///               signed / 32-bit int rescaled into 0..65535 on the CPU).
///   * `Float` — `R32F`,    raw 32-bit float (window/level done on the GPU).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    Int8,
    Int16,
    Float,
}

/// How the 3D volume's scalar samples are stored in its GPU texture. Chosen
/// from channel 0's `ChannelKind` so the volume mirrors the 2D display:
///   * `U8`  — `R8`  unorm (8-bit source)
///   * `U16` — `R16` unorm (16-bit source, or CPU-widened 8-bit/rescaled ints)
///   * `F32` — `R32F`      (32-bit float source, window/level in its own units)
/// Unlike the 2D integer path (which uses `usampler`, NEAREST-only), the volume
/// uses *normalized* textures so trilinear interpolation is available.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VolumeKind {
    U8,
    U16,
    F32,
}

/// Volume texture sampling: `Nearest` (crisp voxels), `Linear` (hardware
/// trilinear), or `Cubic` (in-shader tricubic B-spline — smoother than linear,
/// 8 trilinear taps per sample). `Nearest`/`Linear` set the GL min/mag filter;
/// `Cubic` uses the GL linear filter plus the shader's cubic reconstruction.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VolumeInterp {
    Nearest,
    Linear,
    Cubic,
}

impl VolumeInterp {
    /// The `u_interp` value the fragment shader branches on (0 = point/linear via
    /// the GL filter, 1 = in-shader tricubic). Only the glow backend consumes it.
    #[cfg_attr(not(feature = "renderer-glow"), allow(dead_code))]
    pub fn shader_mode(self) -> i32 {
        match self {
            VolumeInterp::Nearest | VolumeInterp::Linear => 0,
            VolumeInterp::Cubic => 1,
        }
    }
}

/// How the ray-marcher turns samples along a ray into a pixel:
///   * `Mip`   — maximum-intensity projection (brightest sample wins; the
///               default, order-independent, good for sparse/bright structures).
///   * `Alpha` — emission-absorption alpha compositing, à la the ImageJ 3D
///               Viewer's "Volume" mode: a translucent, depth-cued render where
///               intensity drives both color (LUT) and opacity.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VolumeRender {
    Mip,
    Alpha,
}

impl VolumeRender {
    /// The `u_mode` value the fragment shader branches on.
    pub fn shader_mode(self) -> i32 {
        match self {
            VolumeRender::Mip => 0,
            VolumeRender::Alpha => 1,
        }
    }
}

/// Everything the ray-march fragment shader needs for one 3D frame. The camera
/// is passed as an explicit basis (rather than matrices) so the shader builds
/// per-pixel rays with no matrix inverse; `app.rs` computes it from the orbit
/// angles + zoom. Distances/positions are in the volume's own normalized box
/// space, whose half-extents `box_he` already fold in the per-axis dimension
/// scale (voxel anisotropy). The per-channel arrays mirror the 2D compositor:
/// each channel MIP-projects independently, then colors through its own LUT row
/// (= channel index) and the results are summed.
#[derive(Clone, Copy)]
// The wgpu backend's 3D path is a stub that ignores these (glow reads them all),
// so under a wgpu-only build the fields are legitimately never read.
#[cfg_attr(not(feature = "renderer-glow"), allow(dead_code))]
pub struct VolumeParams {
    /// Number of channels composited (≤ `MAX_CHANNELS`).
    pub num_channels: i32,
    /// Per-channel window/level as flat `(min, max)` pairs, in the sampled
    /// texture's units: raw value for `F32`; the 0..65535 display window divided
    /// by 65535 for `U8`/`U16`.
    pub windows: [f32; MAX_CHANNELS * 2],
    /// Per-channel on/off (1.0 / 0.0), so toggling a channel needs no rebuild.
    pub enabled: [f32; MAX_CHANNELS],
    /// Per-channel: 1.0 if the channel's data is in the float texture, else 0.0.
    pub is_float: [f32; MAX_CHANNELS],
    /// Ray-march compositing mode (see `VolumeRender::shader_mode`): 0 = MIP,
    /// 1 = alpha DVR. The sample count is derived in-shader from the voxel size.
    pub render_mode: i32,
    /// Alpha-DVR opacity scale (higher = more solid). Ignored by the MIP mode.
    pub density: f32,
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub tan_half_fov: f32,
    pub aspect: f32,
    /// Half-extents of the volume box (largest scaled axis = 0.5).
    pub box_he: [f32; 3],
}

/// One channel's window/level + on/off state, as `app.rs` produces it each
/// frame. The backend maps it to whatever GPU representation it uses.
#[derive(Clone, Copy)]
pub struct ChannelUniform {
    pub min: f32,
    pub max: f32,
    pub enabled: bool,
    /// True if this channel's data is uploaded as a float (R32F) texture — i.e.
    /// 32-bit float source. The shader then samples it as a float and applies
    /// window/level in the data's own units. False = integer (R16Uint) channel,
    /// where `min`/`max` are in raw 0..65535 sample units (unchanged path).
    pub is_float: bool,
}

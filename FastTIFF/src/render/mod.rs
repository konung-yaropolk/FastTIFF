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
pub use glow_backend::{init, paint_callback, upload_ctx, Render, BACKEND, RENDERER};

// The `not(renderer-glow)` guard means that turning on *both* features (a hard
// error, above) selects only the glow backend here — so the build fails with
// the clean `compile_error!` message instead of a duplicate-symbol cascade.
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
mod wgpu_backend;
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
pub use wgpu_backend::{init, paint_callback, upload_ctx, Render, BACKEND, RENDERER};

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

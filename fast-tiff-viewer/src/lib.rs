//! The frontend-agnostic viewer core: everything between "a TIFF on disk" and
//! "pixels on the GPU", with no GUI toolkit in sight.
//!
//! ```text
//!   fast-tiff-lib     file I/O, IFD index, decode, metadata
//!         │
//!   scivis-render     GPU pipelines, textures, ray-marching
//!         │
//!   fast-tiff-viewer  ← you are here: stack model, channel settings,
//!         │             c/z/t interpretation, camera, decode→GPU sync
//!   frontend          egui app, or a wasm/JS web UI
//! ```
//!
//! A frontend owns its windows, widgets and input; it borrows everything else
//! from here. The typical loop is:
//!
//! ```ignore
//! let mut viewer = Viewer::new();
//! viewer.open(path)?;
//!
//! // each frame:
//! if viewer.playback.playing { viewer.tick_playback(now_seconds); }
//! viewer.uv_offset = ...;  // from your pan/zoom
//! viewer.uv_scale  = ...;
//! let outcome = viewer.sync(&mut renderer);
//! // ...then draw, and repaint again if outcome.needs_repaint
//! ```
//!
//! # Features
//!
//! * `backend-wgpu` / `backend-glow` — which GPU backend [`sync`] drives.
//!   Exactly one is needed for [`Viewer::sync`]; with neither, the CPU-side
//!   model still compiles and is fully usable headless.
//! * `threads` (default) — background read-ahead and volume assembly. Turn it
//!   off for a single-threaded host such as `wasm32-unknown-unknown`; the
//!   synchronous fallbacks are already the paths taken when a worker fails to
//!   spawn, so nothing else changes.

pub mod camera;
pub mod channels;
pub mod colormap;
pub mod dimensions;
pub mod display;
pub mod histogram;
pub mod prefetch;
pub mod roi;
pub mod stack;
pub mod viewer;
pub mod volume;

#[cfg(any(feature = "backend-wgpu", feature = "backend-glow"))]
pub mod sync;

// The GPU backend this crate was built against. Selected the same way the
// frontend selects one, and it must be the *same* one — the app's `renderer-*`
// features forward to both crates, which is what keeps them in step.
#[cfg(feature = "backend-glow")]
pub use scivis_render::glow_backend as backend;
#[cfg(all(feature = "backend-wgpu", not(feature = "backend-glow")))]
pub use scivis_render::wgpu_backend as backend;

/// The concrete renderer [`Viewer::sync`] drives, so callers name it once.
#[cfg(any(feature = "backend-wgpu", feature = "backend-glow"))]
pub type Renderer = backend::ImageRenderResources;

pub use display::{Dims, Display};
pub use histogram::Histogram;
pub use stack::{ChannelSettings, Stack};
pub use viewer::{DecodeMode, Playback, ViewMode, Viewer, VolumeView, DEFAULT_FPS};

// Re-exported so a frontend needs only this crate in its imports for the common
// path; reach for `fast_tiff_lib` / `scivis_render` directly when you need
// something more specific.
pub use fast_tiff_lib::{self, TiffStack};
pub use scivis_render::{self, ChannelKind, Lut, VolumeInterp, VolumeKind, VolumeRender, MAX_CHANNELS};

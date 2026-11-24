//! The frontend-agnostic viewer core: everything between "a TIFF on disk" and
//! "pixels on the GPU", with no GUI toolkit in sight.
//!
//! ```text
//!   fast-tiff-lib     file I/O, IFD index, decode, metadata
//!         │
//!   scivis-render     GPU pipelines, textures, ray-marching
//!         │
//!   fast-tiff-viewer  ← you are here: document model, channel settings,
//!         │             c/z/t interpretation, 2D view, 3D camera,
//!         │             decode→GPU sync
//!   frontend          egui app, or a wasm/JS web UI
//! ```
//!
//! A frontend owns its windows, widgets and input; it borrows everything else
//! from here. The typical loop is:
//!
//! ```ignore
//! use fast_tiff_viewer::view2d::View2d;
//! use fast_tiff_viewer::{FrameInputs, Viewer};
//!
//! let mut viewer = Viewer::new();
//! viewer.open(path)?;
//! let mut view = View2d::default();
//! let mut inputs = FrameInputs::default();
//!
//! // each frame:
//! if viewer.playback.playing() { viewer.tick_playback(now_seconds); }
//! // your input, through `view` (see `view2d` for a frame of it), then:
//! view.advance_zoom_glide(dt);
//! inputs.moving = view.is_moving();
//! view.set_panel(panel);
//! let extent = view.measure(image_px, room);
//! view.clamp_pan(&extent);
//! let placed = view.place(&extent);
//! inputs.image = placed.inputs(pixels_per_point);
//! let outcome = viewer.sync(&mut renderer, &inputs);
//! // ...then draw, and repaint again if outcome.needs_repaint;
//! // show viewer.notices() on a status line
//! ```
//!
//! What the user changes goes through operations rather than fields: the
//! frame with [`DocumentMut::set_frame`] and [`DocumentMut::step_frame`], a
//! channel's window and switch with [`DisplayMut::set_window`] and
//! [`DisplayMut::set_enabled`] (handles from [`Viewer::document_mut`] and
//! [`DocumentMut::display_mut`], which lend those and nothing that could
//! replace the document or its display), the interpretation and the LUTs
//! with [`Viewer::set_dimension_order`], [`Viewer::set_pseudocolor`] and
//! [`Viewer::set_gray_lut`]; a document arrives only through an open.
//!
//! # The renderer
//!
//! [`Viewer::sync`] drives whatever renderer it is handed, through the two
//! traits of `scivis_render` ([`ImageRenderer`] for the 2D frame,
//! [`VolumeRenderer`] for the 3D view): it is generic over them, so it is
//! compiled for the backend the frontend constructs, and calls it directly.
//! This crate names no backend and needs none to build: the frontend picks
//! one where it makes it (FastTIFF in `FastTIFF/src/render.rs`), and the
//! tests here hand the sync a double that records what it is sent.
//!
//! # Plugins
//!
//! The plugin host is here ([`plugins`]): the registry, the host a plugin
//! sees, the C boundary and, on the desktop, the loader for plugin libraries.
//! The plugins the application compiles in are not: they are the
//! `fasttiff-builtin-plugins` crate, which the application hands to the
//! registry when it builds it ([`plugins::Registry::new`]). So this crate
//! names no plugin, has no plugin features, and links none of their
//! dependencies — the registration crates, the FFT, the image codecs.
//!
//! The one plugin-like thing that is here is the [`point_counter`]: a tool on
//! the picture rather than a registry plugin, written against the document
//! because the plugin contract has no kind of plugin for a tool yet.
//!
//! # Features
//!
//! * `threads` (default) — background read-ahead, window and volume assembly.
//!   Turn it off for a single-threaded host such as `wasm32-unknown-unknown`;
//!   the synchronous fallbacks are already the paths taken when a worker
//!   fails to spawn, so nothing else changes.
//! * `mmap` (default) — memory-mapped files, and with them `Viewer::open`
//!   by path; without it a document arrives as bytes.
//! * `codec-zstd` (default) — fast-tiff-lib's ZSTD codec, a C dependency, so
//!   not for wasm.
//!
//! # Licence
//!
//! MPL-2.0, as `fast-tiff-lib`, `scivis-render` and the plugin SDK are. It
//! was GPL-3.0 while the built-in plugins were compiled into it, two of which
//! link GPL-3.0-only registration crates; those are in
//! `fasttiff-builtin-plugins` now, and this crate links nothing copyleft.

// Everything `pub` is reachable from outside, and so is this crate's API: an
// item only the crate uses says `pub(crate)`. And a type a public signature
// hands out can be named by the frontend it is handed to.
#![warn(unreachable_pub, unnameable_types)]

// The modules a frontend uses as areas of their own are public; everything
// else is the crate's, and what a frontend needs of it is re-exported below.
// An item is reachable by one path.
mod bandcache;
/// The batch readers — the volume build, the save, a plugin's plane reads —
/// against what they gave before they moved onto the library's `Frame`.
#[cfg(test)]
mod batch_oracle_tests;
mod brightness;
pub mod camera;
pub mod channels;
mod colormap;
mod dimensions;
mod display;
mod document;
mod frame;
pub mod histogram;
mod loader;
mod notice;
/// Decoding one plane as `f32` in the file's own units. One definition, so the
/// plugin host and anything else measuring pixels cannot disagree about what a
/// sample is worth.
mod planes;
// Plugins — the registry, the host a plugin sees, and the C boundary every
// plugin crosses; the plugins themselves are handed in.
//
// Built for every target, including wasm. What is desktop-only is narrower
// than the whole module and is gated inside it: finding and loading a
// third-party shared library (`plugins::library`, `plugins::install_dir`).
// The browser build gets the rest, which is what `plugins::Registry::new` has
// always been for.
//
// Nothing in the decode, sync or render path refers to this module, so a build
// that never opens the Plugins menu pays for it only in binary size. (A
// comment, not docs: outer docs would have rustdoc resolve the module's own
// links from here.)
pub mod plugins;
// The point counter: points clicked onto the picture, counted in groups and
// exported as ImageJ point ROIs. In the Plugins menu, but a tool rather than
// a registry plugin; see the module for why it is here. (A comment, not docs:
// outer docs would have rustdoc resolve the module's own links from here.)
pub mod point_counter;
mod prefetch;
/// A renderer that records what the sync sends it, for its tests.
#[cfg(test)]
mod recorder;
/// What one renderer holds of the document it shows, reconciled with the
/// display model once per sync. Crate-private: a frontend asks the
/// [`Viewer`] what it needs of it.
mod residency;
pub mod roi;
/// Writing the open stack back out as a TIFF. Native only: it writes to a
/// path, and a browser has none.
pub mod save;
mod screen;
pub mod snapshot;
mod sync;
mod units;
pub mod view2d;
mod viewer;
mod volume;
mod window;
mod worker;

#[doc(hidden)]
pub mod testing;

pub use display::{ChannelLayout, Dims, Display, DisplayMut, GrayLuts, Revision};
pub use document::{ChannelSettings, Document, DocumentMut, SampleUnits, Window};
pub use frame::{FrameInputs, ImageInputs, Pose, VolumeInputs};
pub use loader::{LoadSource, LoadStage};
pub use notice::{Notice, NoticeKind, ShapeNote, VolumeNote};
pub use screen::PhysicalPx;
pub use sync::SyncOutcome;
pub use viewer::{DecodeMode, Playback, ViewMode, Viewer, VolumeSettings, DEFAULT_FPS};

// Re-exported so a frontend needs only this crate in its imports for the common
// path; reach for `fast_tiff_lib` / `scivis_render` directly when you need
// something more specific.
pub use fast_tiff_lib::{self, TiffStack};
pub use scivis_render::{
    self, ChannelKind, ImageRenderer, Lut, VolumeInterp, VolumeRender, VolumeRenderer, MAX_CHANNELS,
};

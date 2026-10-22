//! The contract between FastTIFF and a plugin.
//!
//! A plugin implements [`Plugin`]: it says who it is ([`PluginInfo`]), declares
//! the dialog it wants ([`ParamDecl`]), and does the work in
//! [`Plugin::run`], reaching the open stack through a [`HostContext`] and
//! handing back an [`Outcome`].
//!
//! ```no_run
//! use fasttiff_plugin_api::*;
//!
//! struct Invert;
//!
//! impl Plugin for Invert {
//!     fn info(&self) -> PluginInfo {
//!         PluginInfo::new("org.example.invert", "Invert").menu_path("Filters")
//!     }
//!
//!     fn run(&mut self, host: &mut dyn HostContext, _p: &Params)
//!         -> Result<Outcome, PluginError>
//!     {
//!         let info = host.image();
//!         let t = host.view().frame_index;
//!         let mut px = Vec::new();
//!         host.read_plane_f32(Plane::new(0, 0, t), &mut px)?;
//!         let hi = px.iter().copied().fold(f32::MIN, f32::max);
//!         for v in &mut px { *v = hi - *v; }
//!         Ok(Outcome::NewDocument(Box::new(ImageResult {
//!             width: info.width, height: info.height,
//!             channels: 1, slices: 1, frames: 1,
//!             pixel_type: PixelType::F32,
//!             planes: vec![PlaneData::F32(px)],
//!             channel_colors: Vec::new(),
//!             metadata: None,
//!             name: "Inverted".into(),
//!         })))
//!     }
//! }
//! ```
//!
//! # Why the API looks like this
//!
//! Three constraints shaped every awkward-looking corner of it, and they are
//! worth knowing before proposing a nicer-looking alternative.
//!
//! **Rust has no stable ABI.** A plugin compiled by a different `rustc`, at a
//! different optimisation level, or against different dependency versions,
//! cannot safely exchange `Vec`, `String`, `&mut` or a trait object with the
//! host. This crate is the *Rust-native* face of the contract, and every
//! plugin reaches the host through a narrow `extern "C"` boundary that only
//! `#[repr(C)]` types cross — a plugin in a `.dll`/`.so`, and equally one
//! compiled into the app, which the host registers through the same boundary
//! in its own process. Everything here is therefore shaped so it can be
//! marshalled: owned data, no borrows of host memory, no callbacks holding host
//! types.
//!
//! # One set of rules
//!
//! Because there is only one way in, a plugin is held to the same rules
//! wherever its code lives:
//!
//! * **A fresh value for every call.** [`Plugin::params`] and [`Plugin::run`]
//!   are each called on a newly built `Default` value; nothing carries over in
//!   the plugin's fields from one call to the next. State that must outlive a
//!   call belongs in a static of the plugin's own.
//! * **One thread at a time** for any one plugin; different plugins may run at
//!   once. The trait is `Send`, not `Sync`.
//! * **Results are checked.** An [`ImageResult`] must [`validate`]; a plot
//!   needs a finite x axis; both have size limits a runaway loop would hit
//!   first. A result that fails is an error naming the plugin, not a document.
//! * **A panic is contained** and reported as a failure naming the plugin, in
//!   the dialog as well as the run. A dialog that cannot be built is reported,
//!   and the plugin is not run.
//! * **What crosses, crosses whole**: the stack's shape, calibration, channel
//!   windows and colour tables, the selection, the dialog's pending values in;
//!   pixels, colours, channel names and metadata, plots and messages out.
//!
//! [`validate`]: ImageResult::validate
//!
//! **A plugin cannot draw.** Handing an `&mut egui::Ui` across that boundary is
//! precisely the unsound case, so dialogs are *declared* and the host renders
//! them — the same choice ImageJ made with `GenericDialog`. See [`params`].
//!
//! **Stacks do not fit in memory.** A 4 GB stack cannot be copied to a plugin,
//! and lending a slice of the host's memory map stops being sound the moment
//! the plugin is a separate library. So pixels are pulled one plane at a time
//! into a buffer the plugin owns. See [`HostContext`].
//!
//! **A run is one call that returns.** So a plugin that wants something to
//! keep happening — a chart that follows the selection, a volume that turns —
//! declares it and the host carries it out: a [`Plot`] the host re-asks for
//! when the regions change, an [`Animation`] the host plays on the 3D view.
//! And a dialog that is used more than once, Play then Stop, declares buttons
//! ([`ParamKind::Action`]) and becomes a panel that stays open, each press a
//! run of its own. See [`animation`]. A tool whose input is a region asks for
//! one with [`Outcome::SelectRegion`] and is run again once it is drawn.
//!
//! # Licence
//!
//! MPL-2.0 — deliberately weaker copyleft than the GPL-3.0 application, because
//! a plugin must depend on this crate and a copyleft SDK would force its licence
//! on every plugin, including ones a hardware vendor might ship.

pub mod animation;
pub mod export;
pub mod host;
pub mod image;
pub mod import;
pub mod meta;
pub mod params;
pub mod plot;
pub mod plugin;
pub mod selection;

pub use animation::{Animation, AnimationOutput, Pose};
pub use export::{ExportRequest, Exporter};
pub use host::{HostContext, HostContextExt};
pub use image::{
    ChannelView, ImageInfo, Lut, PixelType, Plane, ViewParams, VolumeMode, VolumeView,
};
pub use import::{Confidence, FileType, ImportHost, ImportRequest, ImportResult, Importer};
pub use meta::{DisplayMode, Spacing, StackInfo};
pub use params::{ParamDecl, ParamKind, ParamValue, Params};
pub use plot::{Plot, Selection, SelectionKind, Series};
pub use plugin::{ImageResult, Outcome, PlaneData, Plugin, PluginError, PluginInfo};
pub use selection::{Roi, Shape};

/// The contract's version.
///
/// Bumped only when the *shape* of the contract changes. The `.dll` lane puts
/// the major version in the exported symbol name, so a host looking for v2 does
/// not find a v1 plugin's entry point at all — the mismatch is a clean missing
/// symbol at load rather than two sides disagreeing about a struct layout after
/// the call has already begun.
pub const API_VERSION_MAJOR: u32 = 1;
pub const API_VERSION_MINOR: u32 = 3;

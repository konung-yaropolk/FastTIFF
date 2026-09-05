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
//! host. This crate is the *Rust-native* face of the contract — used directly
//! by plugins compiled into the app — and the `.dll`/`.so` lane reaches it
//! through a narrow `extern "C"` boundary that only `#[repr(C)]` types cross.
//! Everything here is therefore shaped so it can be marshalled: owned data,
//! no borrows of host memory, no callbacks holding host types.
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
//! # Licence
//!
//! MPL-2.0 — deliberately weaker copyleft than the GPL-3.0 application, because
//! a plugin must depend on this crate and a copyleft SDK would force its licence
//! on every plugin, including ones a hardware vendor might ship.

pub mod host;
pub mod image;
pub mod import;
pub mod meta;
pub mod params;
pub mod plugin;

pub use host::{HostContext, HostContextExt};
pub use image::{
    ChannelView, ImageInfo, Lut, PixelType, Plane, ViewParams, VolumeMode, VolumeView,
};
pub use import::{Confidence, FileType, ImportHost, ImportRequest, ImportResult, Importer};
pub use meta::{DisplayMode, Spacing, StackInfo};
pub use params::{ParamDecl, ParamKind, ParamValue, Params};
pub use plugin::{ImageResult, Outcome, PlaneData, Plugin, PluginError, PluginInfo};

/// The contract's version.
///
/// Bumped only when the *shape* of the contract changes. The `.dll` lane puts
/// the major version in the exported symbol name, so a host looking for v2 does
/// not find a v1 plugin's entry point at all — the mismatch is a clean missing
/// symbol at load rather than two sides disagreeing about a struct layout after
/// the call has already begun.
pub const API_VERSION_MAJOR: u32 = 1;
pub const API_VERSION_MINOR: u32 = 0;

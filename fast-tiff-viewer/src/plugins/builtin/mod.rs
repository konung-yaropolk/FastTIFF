//! The plugins compiled into this build.
//!
//! Everything in this directory is a *plugin*. Everything in the directory
//! above it is the *interface* plugins are written against — the registry, the
//! search path, the host context, the shared-library loader. The split is worth
//! keeping sharp: a file here may only use `fasttiff_plugin_api`, exactly as a
//! third-party plugin would, so if one of these ever needs something the API
//! does not offer, that is a gap in the contract rather than a reason to reach
//! sideways into the viewer.
//!
//! Being compiled in is a *lane*, not a different kind of plugin. These
//! implement the same [`Plugin`] and [`Importer`] traits a `.dll` does, and the
//! registry cannot tell them apart except by the [`Origin`](super::Origin) tag
//! it stores alongside. They exist for four reasons:
//!
//! * The browser build can never load a shared library, and still gets a
//!   working Plugins menu.
//! * The contract gets a real consumer before any loading mechanism exists,
//!   which is how a plugin API avoids being designed in the abstract and then
//!   failing on the first real plugin.
//! * [`Invert`] is the oracle for the `.dll` lane: the same filter, run both
//!   ways, must produce byte-identical output.
//! * [`Oir`] is the importer that earns its place: a proprietary format with no
//!   spec at all, worked out from a real acquisition and checked against the
//!   vendor software's own export.
//! * [`PngImport`] is the one people actually reach for, because a figure, a
//!   mask or a screenshot arrives as a PNG and has to be got *in* — at the
//!   depth the file stores it, which no screenshot-shaped reader does.
//! * [`deconvolve`] is the one that is a real piece of science rather than a
//!   convenience, and the proof that the contract is wide enough to carry one:
//!   a dialog of twenty controls, a file the plugin opens for itself, an hour
//!   of arithmetic that has to stay cancellable, and nine published algorithms
//!   behind one menu entry.
//!
//! [`Netpbm`] is a fifth thing — a worked example of the other shape the job
//! takes, a documented format implemented straight from its spec. It is behind
//! the off-by-default `netpbm-example` feature: read it when writing an
//! importer, enable it (`--features netpbm-example`) to run it, but it is not
//! part of the product and nobody opens a `.pgm` in a TIFF viewer.
//!
//! # Adding one
//!
//! Give it a directory here — `<name>/mod.rs`, its tests in `mod_tests.rs`
//! beside that, and any further files it needs alongside them — write it using
//! nothing but `fasttiff_plugin_api`, and add it to [`all`] or [`importers`]
//! below. That is the whole procedure.
//!
//! A directory each rather than a file each, even for the two that are one file
//! long. A plugin grows parts — [`oir`] has a container reader and a metadata
//! translator, and would have had them as `oir.rs`, `oir_meta.rs`,
//! `oir_tests.rs` and `oir_meta_tests.rs` in a shared directory, four files
//! whose relationship is a naming convention. Keeping the boundary in the
//! filesystem is also what makes the last step here cheap: if a plugin would
//! rather ship separately, its directory becomes a `cdylib` crate's `src/` and
//! nothing inside it has to change.

pub mod deconvolve;
pub mod invert;
#[cfg(feature = "netpbm-example")]
pub mod netpbm;
/// Desktop-only: it memory-maps the container and parses the vendor's XML, and
/// a browser has no file to map.
#[cfg(not(target_arch = "wasm32"))]
pub mod oir;
pub mod plot_axis;
pub mod png;
/// Desktop-only. It compiles for wasm, which is the trap: `suite2p-registration`
/// uses rayon unconditionally, and `std::thread::spawn` is unsupported on
/// `wasm32-unknown-unknown` — so the browser build would offer the menu entry
/// and then panic on the first frame it registered. Absent is better than
/// present and fatal.
#[cfg(not(target_arch = "wasm32"))]
pub mod stabilize;
pub mod stack_tools;
pub mod zproject;

pub use deconvolve::{Deconvolve, GeneratePsf};
pub use invert::Invert;
#[cfg(feature = "netpbm-example")]
pub use netpbm::Netpbm;
#[cfg(not(target_arch = "wasm32"))]
pub use oir::Oir;
pub use plot_axis::PlotAxis;
pub use png::{Png, PngImport};
#[cfg(not(target_arch = "wasm32"))]
pub use stabilize::Stabilize;
pub use stack_tools::{SliceKeeper, SliceOrderInvert, SliceRemover};
pub use zproject::ZProject;

use fasttiff_plugin_api::{Exporter, Importer, Plugin};

/// The filters compiled into this build, in registration order.
///
/// Pushed rather than written as one literal because `Stabilize` is not in the
/// browser build; see its module. Order is not load-bearing here — the menu
/// sorts by path and name, and `add` only uses order to break an id clash —
/// but keeping the desktop list in its historical order keeps the diff honest.
pub fn all() -> Vec<Box<dyn Plugin>> {
    let mut v: Vec<Box<dyn Plugin>> =
        vec![Box::new(Invert), Box::new(ZProject), Box::new(PlotAxis)];
    #[cfg(not(target_arch = "wasm32"))]
    v.push(Box::new(Stabilize));
    v.push(Box::new(SliceKeeper));
    v.push(Box::new(SliceRemover));
    v.push(Box::new(SliceOrderInvert));
    v.push(Box::new(GeneratePsf));
    v.push(Box::new(Deconvolve));
    v
}

/// The exporters compiled into this build, in registration order.
///
/// Order is the tie-break when two claim one extension, as it is for importers
/// — though an exporter cannot be probed, so order is the *only* tie-break.
pub fn exporters() -> Vec<Box<dyn Exporter>> {
    vec![Box::new(Png)]
}

/// The importers compiled into this build, in registration order.
///
/// Order is the tie-break when two importers are equally confident about a
/// file, so it is a real decision rather than a list: the more specific format
/// goes first.
/// `vec_init_then_push`: the literal clippy asks for cannot be written here,
/// because the *first* element is the conditional one and the order is the
/// documented tie-break. Seeding the vector with the second element would put
/// the general format ahead of the specific one on every target.
#[allow(clippy::vec_init_then_push)]
pub fn importers() -> Vec<Box<dyn Importer>> {
    // OIR first, being the specific format; both answer on a signature of
    // their own, so the order between them never actually decides anything.
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Importer>> = Vec::new();
    #[cfg(not(target_arch = "wasm32"))]
    v.push(Box::new(Oir));
    v.push(Box::new(PngImport));
    #[cfg(feature = "netpbm-example")]
    v.push(Box::new(Netpbm));
    v
}

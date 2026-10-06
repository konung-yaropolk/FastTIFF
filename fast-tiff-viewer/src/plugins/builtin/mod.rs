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
//! takes, a documented format implemented straight from its spec. Read it when
//! writing an importer, enable it (`--features plugin-netpbm`) to run it, but
//! it is not part of the product and nobody opens a `.pgm` in a TIFF viewer.
//!
//! # Which of them are here
//!
//! One Cargo feature per directory, named `plugin-<directory>`, and
//! `builtin-plugins` is every one that ships. All of them are on by default;
//! the manifest is where the list lives and what it costs to drop one is
//! written there. The effect is total: a plugin whose feature is off has no
//! module, no entry in the lists below, no menu entry, and — for the four that
//! bring one — no dependency compiled or linked.
//!
//! Two kinds of `cfg` therefore appear below, and they mean different things:
//!
//! * `feature = "plugin-…"` is *whether this build wants it*, a product
//!   decision taken in the manifest.
//! * `not(target_arch = "wasm32")` is *whether this target can run it at all*,
//!   a fact about the platform, written next to the reason it is true.
//!
//! Only [`oir`] still has the second, because it memory-maps a file. Keeping
//! both means enabling a plugin for a target that cannot serve it is inert
//! rather than broken: the feature is simply ignored there.
//!
//! # Adding one
//!
//! Give it a directory here — `<name>/mod.rs`, its tests in `mod_tests.rs`
//! beside that, and any further files it needs alongside them — write it using
//! nothing but `fasttiff_plugin_api`, add a `plugin-<name>` feature to the
//! manifest (with any dependency of its own named by that feature and nothing
//! else), put it in `builtin-plugins`, and add it to [`all`] or [`importers`]
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

#[cfg(feature = "plugin-deconvolve")]
pub mod deconvolve;
#[cfg(feature = "plugin-invert")]
pub mod invert;
#[cfg(feature = "plugin-netpbm")]
pub mod netpbm;
/// Desktop-only on top of its feature: it memory-maps the container and parses
/// the vendor's XML, and a browser has no file to map.
#[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
pub mod oir;
#[cfg(feature = "plugin-plot-axis")]
pub mod plot_axis;
#[cfg(feature = "plugin-png")]
pub mod png;
/// What more than one of them is built from, and the reason each of them can be
/// chosen on its own.
///
/// The condition is its users, so that a build with none of them does not carry
/// it and warn that it is unused. Nothing has to remember to update this list:
/// a fourth user that forgot would fail to find the module.
#[cfg(any(
    feature = "plugin-invert",
    feature = "plugin-stack-tools",
    feature = "plugin-deconvolve",
))]
pub(crate) mod shared;
/// Every target, now that `suite2p-registration` can be built without rayon.
///
/// This was desktop-only, on the grounds that the registration crate depended
/// on rayon unconditionally and would therefore panic in a browser. That was
/// wrong twice over: rayon documents a fallback that runs a `par_iter`
/// sequentially on `wasm32-unknown-unknown` rather than failing, and nothing in
/// that crate spawns a thread directly. The plugin would have worked.
///
/// What it would not have done is tell the truth, because under that fallback
/// `Backend::MultiThread` is single-threaded and still calls itself
/// multi-threaded. So rayon is now behind the registration crate's `threads`
/// feature, which this crate asks for only on targets that have threads; a
/// browser gets the same arithmetic on one thread, and `MultiThread` reports
/// itself unavailable there the way the GPU backend already does.
///
/// One thing to know before reaching for it in a browser: a plugin run on the
/// web is synchronous, so a long registration will hold the tab.
#[cfg(feature = "plugin-stabilize")]
pub mod stabilize;
#[cfg(feature = "plugin-stack-tools")]
pub mod stack_tools;
#[cfg(feature = "plugin-zproject")]
pub mod zproject;

#[cfg(feature = "plugin-deconvolve")]
pub use deconvolve::{Deconvolve, GeneratePsf};
#[cfg(feature = "plugin-invert")]
pub use invert::Invert;
#[cfg(feature = "plugin-netpbm")]
pub use netpbm::Netpbm;
#[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
pub use oir::Oir;
#[cfg(feature = "plugin-plot-axis")]
pub use plot_axis::PlotAxis;
#[cfg(feature = "plugin-png")]
pub use png::{Png, PngImport};
#[cfg(feature = "plugin-stabilize")]
pub use stabilize::Stabilize;
#[cfg(feature = "plugin-stack-tools")]
pub use stack_tools::{SliceKeeper, SliceOrderInvert, SliceRemover};
#[cfg(feature = "plugin-zproject")]
pub use zproject::ZProject;

use fasttiff_plugin_api::{Exporter, Importer, Plugin};

/// The filters compiled into this build, in registration order.
///
/// Pushed rather than written as one literal because any of them may be absent;
/// see the feature note above. Order is not load-bearing here — the menu sorts
/// by path and name, and `add` only uses order to break an id clash — but
/// keeping the historical order keeps the diff honest.
///
/// `vec_init_then_push`: the literal clippy asks for cannot be written when
/// every element is conditional.
#[allow(clippy::vec_init_then_push)]
pub fn all() -> Vec<Box<dyn Plugin>> {
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Plugin>> = Vec::new();
    #[cfg(feature = "plugin-invert")]
    v.push(Box::new(Invert));
    #[cfg(feature = "plugin-zproject")]
    v.push(Box::new(ZProject));
    #[cfg(feature = "plugin-plot-axis")]
    v.push(Box::new(PlotAxis));
    #[cfg(feature = "plugin-stabilize")]
    v.push(Box::new(Stabilize));
    #[cfg(feature = "plugin-stack-tools")]
    {
        v.push(Box::new(SliceKeeper));
        v.push(Box::new(SliceRemover));
        v.push(Box::new(SliceOrderInvert));
    }
    #[cfg(feature = "plugin-deconvolve")]
    {
        v.push(Box::new(GeneratePsf));
        v.push(Box::new(Deconvolve));
    }
    v
}

/// The exporters compiled into this build, in registration order.
///
/// Order is the tie-break when two claim one extension, as it is for importers
/// — though an exporter cannot be probed, so order is the *only* tie-break.
#[allow(clippy::vec_init_then_push)]
pub fn exporters() -> Vec<Box<dyn Exporter>> {
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Exporter>> = Vec::new();
    #[cfg(feature = "plugin-png")]
    v.push(Box::new(Png));
    v
}

/// The importers compiled into this build, in registration order.
///
/// Order is the tie-break when two importers are equally confident about a
/// file, so it is a real decision rather than a list: the more specific format
/// goes first.
/// `vec_init_then_push`: the literal clippy asks for cannot be written here,
/// because the *first* element is conditional and the order is the documented
/// tie-break. Seeding the vector with the second element would put the general
/// format ahead of the specific one.
#[allow(clippy::vec_init_then_push)]
pub fn importers() -> Vec<Box<dyn Importer>> {
    // OIR first, being the specific format; both answer on a signature of
    // their own, so the order between them never actually decides anything.
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Importer>> = Vec::new();
    #[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
    v.push(Box::new(Oir));
    #[cfg(feature = "plugin-png")]
    v.push(Box::new(PngImport));
    #[cfg(feature = "plugin-netpbm")]
    v.push(Box::new(Netpbm));
    v
}

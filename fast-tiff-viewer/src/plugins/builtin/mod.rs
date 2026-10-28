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
//! Being compiled in is a *place*, not a different kind of plugin. These
//! implement the same [`Plugin`], [`Importer`] and [`Exporter`] traits a `.dll`
//! does, and [`register`] installs them through the same C ABI a `.dll` is
//! installed through — the same registration, run in this process instead of
//! from a library's query function. From then on every call they receive goes
//! through the host table and sinks a library's does, so there is nothing a
//! built-in can do that a library cannot, and no rule one is held to that the
//! other is not. The registry knows the difference only by the
//! [`Origin`](super::Origin) tag it stores alongside. They exist for four
//! reasons:
//!
//! * The browser build can never load a shared library, and still gets a
//!   working Plugins menu.
//! * The contract gets a real consumer before any loading mechanism exists,
//!   which is how a plugin API avoids being designed in the abstract and then
//!   failing on the first real plugin.
//! * [`Invert`] is the oracle for the `.dll` boundary: the same filter, run
//!   from a real library and from here, must produce byte-identical output.
//! * [`Oir`] is the importer that earns its place: a proprietary format with no
//!   spec at all, worked out from a real acquisition and checked against the
//!   vendor software's own export. Its reader is the `oir-reader` crate,
//!   published on its own; what is here is the importer around it.
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
//! [`oir`], [`dicom`] and [`nifti`] have the second: an importer is handed a
//! path, and a browser has none. Keeping both means enabling a plugin for a
//! target that cannot serve it is inert rather than broken: the feature is
//! simply ignored there.
//!
//! # Adding one
//!
//! Give it a directory here — `<name>/mod.rs`, its tests in `mod_tests.rs`
//! beside that, and any further files it needs alongside them — write it using
//! nothing but `fasttiff_plugin_api`, add a `plugin-<name>` feature to the
//! manifest (with any dependency of its own named by that feature and nothing
//! else), put it in `builtin-plugins`, derive `Default` on its type, and add
//! it to [`register`] below. That is this crate's part.
//!
//! The applications choose their plugins by feature too, so the feature has
//! to reach them: forward it, and list it in `builtin-plugins`, in
//! `FastTIFF/Cargo.toml`; add it to the no-plugins guard in
//! `FastTIFF/src/lib.rs`; and forward it in `FastTIFF-web/Cargo.toml` if it
//! can run in a browser. Miss those and the plugin builds, passes its tests and
//! appears in neither application. Last, add it to
//! `tests/plugin_parity.rs`, which fails until every installed built-in is
//! checked through the C boundary.
//!
//! A directory each rather than a file each, even for the two that are one file
//! long. A plugin grows parts — [`deconvolve`] has its algorithms, its
//! transforms, its optics and its point-spread functions, and would have had
//! them as `deconvolve.rs`, `deconvolve_fft.rs`, `deconvolve_psf.rs` and so on
//! in a shared directory, files whose relationship is a naming convention.
//! Keeping the boundary in the filesystem is also what makes the last step here
//! cheap: if a plugin would rather ship separately, its directory becomes a
//! `cdylib` crate's `src/` and nothing inside it has to change. [`oir`] went
//! one step further and gave its reader away: the container and record code
//! became the `oir-reader` crate, and the directory kept the importer.

/// Every target: it computes poses and the host plays them, so it needs
/// nothing a browser lacks.
#[cfg(feature = "plugin-animation-3d")]
pub mod animation_3d;
#[cfg(feature = "plugin-crop")]
pub mod crop;
#[cfg(feature = "plugin-deconvolve")]
pub mod deconvolve;
/// Desktop-only on top of its feature, as every importer is: an
/// [`ImportRequest`](fasttiff_plugin_api::ImportRequest) carries a filesystem
/// path, and a browser tab has none. See the module for what that costs and why
/// the contract is right to be that way.
#[cfg(all(feature = "plugin-dicom", not(target_arch = "wasm32")))]
pub mod dicom;
#[cfg(feature = "plugin-invert")]
pub mod invert;
#[cfg(feature = "plugin-netpbm")]
pub mod netpbm;
/// Desktop-only on top of its feature, for the reason [`dicom`] is.
#[cfg(all(feature = "plugin-nifti", not(target_arch = "wasm32")))]
pub mod nifti;
/// Desktop-only on top of its feature, for the reason [`dicom`] is. The reader
/// itself is the `oir-reader` crate, which does build for wasm; only this
/// importer around it is gated.
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
    feature = "plugin-crop",
    feature = "plugin-suite2p-stabilize",
    feature = "plugin-turboreg-stabilize",
))]
pub(crate) mod shared;
#[cfg(feature = "plugin-suite2p-stabilize")]
pub mod stabilize;
#[cfg(feature = "plugin-stack-tools")]
pub mod stack_tools;
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
#[cfg(feature = "plugin-turboreg-stabilize")]
pub mod turboreg;
#[cfg(feature = "plugin-zproject")]
pub mod zproject;

#[cfg(feature = "plugin-animation-3d")]
pub use animation_3d::Animation3d;
#[cfg(feature = "plugin-crop")]
pub use crop::Crop;
#[cfg(feature = "plugin-deconvolve")]
pub use deconvolve::{Deconvolve, GeneratePsf};
#[cfg(all(feature = "plugin-dicom", not(target_arch = "wasm32")))]
pub use dicom::Dicom;
#[cfg(feature = "plugin-invert")]
pub use invert::Invert;
#[cfg(feature = "plugin-netpbm")]
pub use netpbm::Netpbm;
#[cfg(all(feature = "plugin-nifti", not(target_arch = "wasm32")))]
pub use nifti::Nifti;
#[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
pub use oir::Oir;
#[cfg(feature = "plugin-plot-axis")]
pub use plot_axis::PlotAxis;
#[cfg(feature = "plugin-png")]
pub use png::{Png, PngImport};
#[cfg(feature = "plugin-suite2p-stabilize")]
pub use stabilize::Stabilize;
#[cfg(feature = "plugin-stack-tools")]
pub use stack_tools::{SliceKeeper, SliceOrderInvert, SliceRemover};
#[cfg(feature = "plugin-turboreg-stabilize")]
pub use turboreg::TurboReg;
#[cfg(feature = "plugin-zproject")]
pub use zproject::ZProject;

use super::{Origin, Registry};

/// Install every plugin compiled into this build, through the C ABI.
///
/// One list for all three kinds, in registration order. Order is not
/// load-bearing for the filters — the menu sorts by path and name, and `add`
/// only uses order to break an id clash — but it is the tie-break for the
/// other two, so it is a real decision there:
///
/// * **Importers**, when two are equally confident about a file: the more
///   specific format goes first. OIR first, being the specific one; every
///   importer here answers on a signature of its own, so the order between
///   them rarely decides anything. The exception, and the reason the order is
///   still worth keeping deliberate: NIfTI claims `.gz` as well as `.nii`,
///   because `.nii.gz` has `gz` for its extension and there is no way to
///   declare a two-part one. It is therefore the one importer here that is
///   offered files it will decline, and it declines them in `probe` by looking
///   at the name and the magic rather than by being asked last. Anything added
///   after it that also wants `.gz` should be put *before* it.
/// * **Exporters**, when two claim one extension. An exporter cannot be
///   probed, so order is the *only* tie-break.
///
/// Each plugin is pushed under its own `cfg` because any of them may be
/// absent; see the feature note above.
pub fn register(reg: &mut Registry) {
    let o = Origin::BuiltIn;
    // Every line below is conditional, so a build with every plugin left out
    // uses neither of these — and must still build without a warning.
    let _ = (&reg, o);
    // ---- filters
    #[cfg(feature = "plugin-invert")]
    reg.add::<Invert>(o);
    #[cfg(feature = "plugin-zproject")]
    reg.add::<ZProject>(o);
    #[cfg(feature = "plugin-plot-axis")]
    reg.add::<PlotAxis>(o);
    #[cfg(feature = "plugin-suite2p-stabilize")]
    reg.add::<Stabilize>(o);
    #[cfg(feature = "plugin-turboreg-stabilize")]
    reg.add::<TurboReg>(o);
    #[cfg(feature = "plugin-stack-tools")]
    {
        reg.add::<SliceKeeper>(o);
        reg.add::<SliceRemover>(o);
        reg.add::<SliceOrderInvert>(o);
    }
    #[cfg(feature = "plugin-animation-3d")]
    reg.add::<Animation3d>(o);
    #[cfg(feature = "plugin-crop")]
    reg.add::<Crop>(o);
    #[cfg(feature = "plugin-deconvolve")]
    {
        reg.add::<GeneratePsf>(o);
        reg.add::<Deconvolve>(o);
    }

    // ---- importers, most specific first
    #[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
    reg.add_importer::<Oir>(o);
    #[cfg(all(feature = "plugin-dicom", not(target_arch = "wasm32")))]
    reg.add_importer::<Dicom>(o);
    #[cfg(all(feature = "plugin-nifti", not(target_arch = "wasm32")))]
    reg.add_importer::<Nifti>(o);
    #[cfg(feature = "plugin-png")]
    reg.add_importer::<PngImport>(o);
    #[cfg(feature = "plugin-netpbm")]
    reg.add_importer::<Netpbm>(o);

    // ---- exporters
    #[cfg(feature = "plugin-png")]
    reg.add_exporter::<Png>(o);
}

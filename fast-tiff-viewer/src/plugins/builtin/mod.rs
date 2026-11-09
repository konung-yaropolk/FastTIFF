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
//! nothing but `fasttiff_plugin_api`, and derive `Default` on its type. Then:
//!
//! * one line in the `builtins!` table below, which declares the module,
//!   re-exports its types and registers them;
//! * a `plugin-<name>` feature in this crate's manifest, naming any dependency
//!   of its own and any `_shared-*` facility it uses, and an entry in
//!   `builtin-plugins`;
//! * the same feature forwarded in `FastTIFF/Cargo.toml` (with
//!   `_any-builtin`, which keeps that crate's no-plugins guard true) and
//!   listed in its `builtin-plugins`, and forwarded in
//!   `FastTIFF-web/Cargo.toml` if it can run in a browser;
//! * a case in `tests/it/plugin_parity.rs`, which fails until every installed
//!   built-in is checked through the C boundary.
//!
//! A directory each rather than a file each, even for the two that are one file
//! long. A plugin grows parts — [`deconvolve`] has its algorithms, its
//! transforms, its optics and its point-spread functions, and would have had
//! them as a run of `deconvolve_*.rs` files in a shared directory, files whose
//! relationship is a naming convention.
//! Keeping the boundary in the filesystem is also what makes the last step here
//! cheap: if a plugin would rather ship separately, its directory becomes a
//! `cdylib` crate's `src/` and nothing inside it has to change. [`oir`] went
//! one step further and gave its reader away: the container and record code
//! became the `oir-reader` crate, and the directory kept the importer.

/// The one table of built-ins: for each, the condition it is compiled under,
/// its module, and the plugins, importers and exporters it registers.
///
/// Everything else is made from it — the `pub mod` and `pub use` for each,
/// and the body of [`register`], in table order — so a plugin is added by one
/// entry here and its feature, and cannot be declared but left out of the
/// registry, or registered under a condition its module was not compiled
/// under. The condition is a `cfg` predicate, with the target gate written
/// next to the feature when a plugin cannot run everywhere.
macro_rules! builtins {
    ($(
        $(#[$doc:meta])*
        $module:ident if ($cfg:meta) =>
            $(plugins($($plugin:ident),+))?
            $(importers($($importer:ident),+))?
            $(exporters($($exporter:ident),+))?;
    )*) => {
        $(
            $(#[$doc])*
            #[cfg($cfg)]
            pub mod $module;
            #[cfg($cfg)]
            pub use $module::{
                $($($plugin,)+)?
                $($($importer,)+)?
                $($($exporter,)+)?
            };
        )*

        /// [`register`]'s body: every entry of the table, in its order.
        fn register_table(reg: &mut Registry, o: Origin) {
            // Every line is conditional, so a build with every plugin left
            // out uses neither — and must still build without a warning.
            let _ = (&reg, o);
            $(
                #[cfg($cfg)]
                {
                    $($(reg.add::<$plugin>(o);)+)?
                    $($(reg.add_importer::<$importer>(o);)+)?
                    $($(reg.add_exporter::<$exporter>(o);)+)?
                }
            )*
        }
    };
}

builtins! {
    // ---- filters
    invert if (feature = "plugin-invert") => plugins(Invert);
    zproject if (feature = "plugin-zproject") => plugins(ZProject);
    plot_axis if (feature = "plugin-plot-axis") => plugins(PlotAxis);
    stabilize if (feature = "plugin-suite2p-stabilize") => plugins(Stabilize);
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
    turboreg if (feature = "plugin-turboreg-stabilize") => plugins(TurboReg);
    stack_tools if (feature = "plugin-stack-tools")
        => plugins(SliceKeeper, SliceRemover, SliceOrderInvert);
    /// Every target: it computes poses and the host plays them, so it needs
    /// nothing a browser lacks.
    animation_3d if (feature = "plugin-animation-3d") => plugins(Animation3d);
    crop if (feature = "plugin-crop") => plugins(Crop);
    deconvolve if (feature = "plugin-deconvolve") => plugins(GeneratePsf, Deconvolve);

    // ---- importers, most specific first (see `register`)
    /// Desktop-only on top of its feature, for the reason [`dicom`] is. The reader
    /// itself is the `oir-reader` crate, which does build for wasm; only this
    /// importer around it is gated.
    oir if (all(feature = "plugin-oir", not(target_arch = "wasm32"))) => importers(Oir);
    /// Desktop-only on top of its feature, as every importer is: an
    /// [`ImportRequest`](fasttiff_plugin_api::ImportRequest) carries a filesystem
    /// path, and a browser tab has none. See the module for what that costs and why
    /// the contract is right to be that way.
    dicom if (all(feature = "plugin-dicom", not(target_arch = "wasm32"))) => importers(Dicom);
    /// Desktop-only on top of its feature, for the reason [`dicom`] is.
    nifti if (all(feature = "plugin-nifti", not(target_arch = "wasm32"))) => importers(Nifti);
    png if (feature = "plugin-png") => importers(PngImport) exporters(Png);
}

/// The one directory here that is not a registry plugin, and so has no
/// feature and no entry in [`register`]: a tool on the picture, kept with the
/// built-ins because it is listed with them. See the module for why.
pub mod point_counter;
/// What more than one of them is built from, and the reason each of them can be
/// chosen on its own.
///
/// Compiled when a plugin that uses one of its facilities is. Each plugin's
/// feature names the facilities it uses (`_shared-deliver`, `_shared-store`,
/// `_shared-edges` in the manifest), so the conditions here and inside the
/// module name three facilities rather than lists of plugins to keep in step.
#[cfg(any(
    feature = "_shared-deliver",
    feature = "_shared-store",
    feature = "_shared-edges"
))]
pub(crate) mod shared;

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
/// The order is the `builtins!` table's above; each plugin is installed under
/// its own `cfg`, since any of them may be absent (see the feature note above).
pub fn register(reg: &mut Registry) {
    register_table(reg, Origin::BuiltIn);
}

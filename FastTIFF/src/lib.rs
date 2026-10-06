//! The FastTIFF viewer's egui interface, shared by the desktop binary and the
//! browser build.
//!
//! `src/main.rs` is the native entry point (a window, argv, file associations);
//! `FastTIFF-web` is the wasm one (a canvas). Both construct the same
//! [`ViewerApp`], so the UI is written once.
//!
//! What differs between them is confined to `#[cfg(target_arch = "wasm32")]`
//! at a handful of places, all of them about the *host* rather than the viewer:
//!
//!   * window management — sizing, positioning and titling the OS window has no
//!     canvas equivalent (`ViewerApp::manage_window`),
//!   * opening files — a blocking dialog and argv versus an async picker and
//!     drop events carrying bytes (`Opened`),
//!   * the GPU adapter's option hook (`render::tune_native_options` /
//!     `render::tune_web_options`),
//!   * how large the chrome is drawn — the web build runs at 150% (`app::scale`).
//!
//! Everything below the UI — the stack model, channel settings, contrast,
//! dimension order, playback and the 3D camera — comes from `fast_tiff_viewer`
//! and is identical on every target.

// An application whose Plugins menu is empty is almost always an accident: the
// plugin features live in `default`, and `--no-default-features` is also how the
// glow renderer is selected, so one forgotten `builtin-plugins` would produce a
// binary that builds, runs, and quietly cannot do half of what it is for. The
// original objection to putting plugins behind a feature at all was exactly
// that CI would not notice, because a release step only compiles — so this is
// the thing that notices, at the only moment where noticing is free.
#[cfg(not(any(
    feature = "plugin-invert",
    feature = "plugin-zproject",
    feature = "plugin-plot-axis",
    feature = "plugin-stack-tools",
    feature = "plugin-stabilize",
    feature = "plugin-deconvolve",
    feature = "plugin-png",
    feature = "plugin-oir",
    feature = "plugin-netpbm",
    feature = "plugins-none",
)))]
compile_error!(
    "FastTIFF is being built with no built-in plugins, which is almost always a \
     `--no-default-features` that forgot to name them again: add `builtin-plugins` \
     to the feature list (e.g. `--features renderer-glow,builtin-plugins`, or \
     `FastTIFF/builtin-plugins` from the workspace root). It has to be this \
     crate's own feature: naming `fast-tiff-viewer/builtin-plugins` instead \
     compiles the plugins into the viewer and leaves this binary unable to \
     see them. If a binary without plugins is what you actually want, say \
     so with \
     `--features plugins-none`."
);

pub mod app;
pub mod render;

// Native-only host integrations: launching sibling processes for extra files,
// and the macOS Apple Event that delivers "Open With" documents.
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
pub mod macos_open;
#[cfg(not(target_arch = "wasm32"))]
pub mod process;
/// The browser's counterpart to [`process`]: handing a document to a second
/// tab instead of to a second process.
#[cfg(target_arch = "wasm32")]
pub mod web_open;
/// The browser's counterpart to a save dialog: handing a file to the user as a
/// download. Compiled under `test` on every target for the one part of it that
/// is not about a browser; see the module.
#[cfg(any(target_arch = "wasm32", test))]
pub mod web_save;

pub use app::{install_chrome, ViewerApp};

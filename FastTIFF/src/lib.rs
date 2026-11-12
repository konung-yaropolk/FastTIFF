//! The FastTIFF viewer's egui interface, shared by the desktop binary and the
//! browser build.
//!
//! `src/main.rs` is the native entry point (a window, argv, file associations);
//! `FastTIFF-web` is the wasm one (a canvas). Both construct the same
//! [`ViewerApp`], so the UI is written once.
//!
//! What differs between them is all about the *host* rather than the viewer —
//! window management, how files come in and go out, workers, a new process or
//! a new tab, how large the chrome is drawn — and it is behind one seam,
//! `app::platform`: what each host decides is a value tested on every target,
//! and what each host does is one implementation per target, picked there by
//! the one `cfg` that picks it. Beyond it, only what cannot compile on the
//! other target is gated: the host modules below, and the GPU adapter's
//! per-target option hooks (`render::tune_native_options` /
//! `render::tune_web_options`).
//!
//! Everything below the UI — the document model, channel settings, contrast,
//! dimension order, playback and the 3D camera — comes from `fast_tiff_viewer`
//! and is identical on every target.

// An application whose Plugins menu is empty is almost always an accident: the
// plugin features live in `default`, and `--no-default-features` is also how the
// glow renderer is selected, so one forgotten `builtin-plugins` would produce a
// binary that builds, runs, and quietly cannot do half of what it is for. The
// original objection to putting plugins behind a feature at all was exactly
// that CI would not notice, because a release step only compiles — so this is
// the thing that notices, at the only moment where noticing is free.
#[cfg(not(any(feature = "_any-builtin", feature = "plugins-none")))]
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

// The hosts' own integrations, each on the one target it can compile for;
// `app::platform` is what uses them. The desktop's: launching sibling
// processes for extra files and plugin results, and the macOS Apple Event
// that delivers "Open With" documents (AppKit's, so macOS's alone).
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
pub mod macos_open;
#[cfg(not(target_arch = "wasm32"))]
pub mod process;
/// The browser's counterpart to [`process`]: handing a document to a second
/// tab instead of to a second process. Written against `web-sys`, which only
/// wasm32 has.
#[cfg(target_arch = "wasm32")]
pub mod web_open;
/// The browser's counterpart to a save dialog: handing a file to the user as a
/// download. `web-sys` too; what a download is called is decided, and tested,
/// in `app/platform/naming.rs` on every target.
#[cfg(target_arch = "wasm32")]
pub mod web_save;

pub use app::{install_chrome, ViewerApp};

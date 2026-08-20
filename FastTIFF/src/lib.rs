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

pub mod app;
pub mod render;

// Native-only host integrations: launching sibling processes for extra files,
// and the macOS Apple Event that delivers "Open With" documents.
#[cfg(not(target_arch = "wasm32"))]
pub mod process;
#[cfg(all(target_os = "macos", not(target_arch = "wasm32")))]
pub mod macos_open;

pub use app::{install_chrome, ViewerApp};

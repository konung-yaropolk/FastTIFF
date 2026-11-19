//! The integration suites of the built-in plugins, as one test binary.
//!
//! Each plugin's own arithmetic is unit-tested beside it in `src/`. These run
//! the plugins the way an application does: against a real document, through
//! the viewer's [`StackHost`](fast_tiff_viewer::plugins::StackHost), and
//! through the registry's C boundary — `fast-tiff-viewer` is a dev-dependency
//! for that and for nothing else. A module that only applies to some builds
//! says so with its own `#![cfg(...)]`.
//!
//! Run one suite with `cargo test -p fasttiff-builtin-plugins --test it <module>::`.

mod plugin_parity;
mod plugins;
mod registry;
mod rgb_plugins;
mod streamed_results;

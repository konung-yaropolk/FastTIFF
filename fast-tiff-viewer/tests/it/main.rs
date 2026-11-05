//! The integration suites of the viewer core, as one test binary.
//!
//! Each module was its own `tests/*.rs` file, and so its own test binary:
//! a separate link of this crate and everything it depends on, and in the
//! release workflow's `cargo test --release` a separate single-threaded
//! fat-LTO link. As modules of one binary they link once. A module that
//! only applies to some builds says so with its own `#![cfg(...)]`.
//!
//! Run one suite with `cargo test -p fast-tiff-viewer --test it <module>::`.

mod auto_contrast;
mod band_cache;
mod cmyk_display;
mod display_model;
mod histogram;
mod open_failure;
mod open_signal;
mod overview;
mod palette_lut;
mod plane_addressing;
mod plane_clamp;
mod plugin_parity;
mod plugins;
mod rgb_plugins;
mod roi_plan;
mod roi_preload;
mod shift_sync;
mod stack_model;
mod window_assembly;
mod window_worker;

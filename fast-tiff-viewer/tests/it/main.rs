//! The integration suites of the viewer core, as one test binary.
//!
//! Each module was its own `tests/*.rs` file, and so its own test binary:
//! a separate link of this crate and everything it depends on, and in the
//! release workflow's `cargo test --release` a separate single-threaded
//! fat-LTO link. As modules of one binary they link once. A module that
//! only applies to some builds says so with its own `#![cfg(...)]`.
//!
//! They test what a frontend can do, through the public API alone: that is
//! what being outside the crate is for. A suite about the crate's own
//! machinery (the band cache, residency planning, window assembly, plane
//! addressing) is a unit test beside the code it tests, in `src/`.
//!
//! The built-in plugins' suites, which run them against this crate's host,
//! are in `fasttiff-builtin-plugins`, with the plugins.
//!
//! Run one suite with `cargo test -p fast-tiff-viewer --test it <module>::`.

mod auto_contrast;
mod display_model;
mod document_model;
mod facade;
mod histogram;
mod open_failure;
mod open_signal;
mod palette_lut;
mod shift_sync;

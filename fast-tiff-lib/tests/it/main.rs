//! The integration suites of the reader and writer, as one test binary.
//!
//! Each module was its own `tests/*.rs` file, and so its own test binary:
//! a separate link of this crate and everything it depends on, and in the
//! release workflow's `cargo test --release` a separate single-threaded
//! fat-LTO link. As modules of one binary they link once. A module that
//! only applies to some builds says so with its own `#![cfg(...)]`.
//!
//! Run one suite with `cargo test -p fast-tiff-lib --test it <module>::`.

mod cmyk;
mod common;
mod display_map;
mod from_bytes;
mod fused_predictor;
mod fuzz_regressions;
mod imagej_contiguous;
mod libtiff_fixtures;
mod malformed_robustness;
mod row_bands;
mod synthetic_tiff;
mod tiled;
mod write_roundtrip;

//! The integration suites of the renderer, as one test binary.
//!
//! Each module was its own `tests/*.rs` file, and so its own test binary:
//! a separate link of this crate and everything it depends on, and in the
//! release workflow's `cargo test --release` a separate single-threaded
//! fat-LTO link. As modules of one binary they link once. A module that
//! only applies to some builds says so with its own `#![cfg(...)]`.
//!
//! Run one suite with `cargo test -p scivis-render --test it <module>::`.

mod markers;
mod wgsl;

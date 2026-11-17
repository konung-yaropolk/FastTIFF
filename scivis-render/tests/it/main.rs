//! The integration suites of the renderer, as one test binary.
//!
//! Each module was its own `tests/*.rs` file, and so its own test binary:
//! a separate link of this crate and everything it depends on, and in the
//! release workflow's `cargo test --release` a separate single-threaded
//! fat-LTO link. As modules of one binary they link once. A module that
//! only applies to some builds says so with its own `#![cfg(...)]`, or here
//! where it is a backend it needs.
//!
//! In this repository every build of these tests has both backends (the
//! crate is its own dev-dependency, with both on: see its `Cargo.toml`). A
//! copy built from crates.io has only the features it is asked for, so the
//! modules that drive a backend are gated on it, and the rest, which need
//! none, run in every build.
//!
//! Run one suite with `cargo test -p scivis-render --test it <module>::`.

// The GPU suites share `headless` and `scene`, which are built with glow, so
// each is gated on it: `gl` draws with glow alone, `parity` against wgpu, and
// `caps` checks each backend's volume caps on its own (its wgpu half gated on
// wgpu as well).
#[cfg(feature = "backend-glow")]
mod caps;
#[cfg(feature = "backend-glow")]
mod gl;
mod glsl;
#[cfg(feature = "backend-glow")]
mod headless;
mod markers;
#[cfg(all(feature = "backend-glow", feature = "backend-wgpu"))]
mod parity;
#[cfg(feature = "backend-glow")]
mod scene;
mod unrolled;
mod wgsl;

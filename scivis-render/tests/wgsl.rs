//! Offline WGSL validation: `cargo test` catches shader errors without a GPU,
//! instead of a blank canvas at startup.
//!
//! This is an integration test rather than a `#[cfg(test)] mod` inside the wgpu
//! backend, so it runs on **every** build of this crate — including one with no
//! backend feature enabled at all. As a unit test in the backend module it was
//! silently skipped whenever the wgpu backend wasn't compiled in, which is
//! exactly when a broken shader is easiest to miss.

/// Parse + validate a WGSL source with naga (what wgpu does at runtime), so a
/// shader error is a failing test rather than a blank canvas at startup.
fn validate(src: &str, name: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: validate: {e:?}"));
}

#[test]
fn volume_shader_is_valid() {
    validate(include_str!("../src/shaders/volume.wgsl"), "volume.wgsl");
}

#[test]
fn composite_shader_is_valid() {
    validate(include_str!("../src/shaders/composite.wgsl"), "composite.wgsl");
}

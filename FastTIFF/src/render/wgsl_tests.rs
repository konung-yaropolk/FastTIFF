//! Offline WGSL validation: `cargo test` catches shader errors without a
//! GPU, instead of a blank canvas at startup. Split from `wgpu_backend.rs`;
//! `include_str!` paths resolve the same from this sibling file.

/// Parse + validate a WGSL source with naga (what wgpu does at runtime), so a
/// shader error is a failing test rather than a blank 3D canvas at startup.
fn validate(src: &str, name: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: parse: {e}"));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: validate: {e:?}"));
}

#[test]
fn volume_shader_is_valid() {
    validate(include_str!("../shaders/volume.wgsl"), "volume.wgsl");
}

#[test]
fn composite_shader_is_valid() {
    validate(include_str!("../shaders/composite.wgsl"), "composite.wgsl");
}

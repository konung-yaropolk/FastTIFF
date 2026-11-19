//! Offline WGSL validation: `cargo test` catches shader errors without a GPU,
//! instead of a blank canvas at startup.
//!
//! This is an integration test rather than a `#[cfg(test)] mod` inside the wgpu
//! backend, so it runs on **every** build of this crate — including one with no
//! backend feature enabled at all. As a unit test in the backend module it was
//! silently skipped whenever the wgpu backend wasn't compiled in, which is
//! exactly when a broken shader is easiest to miss.
//!
//! What is validated is what the backend compiles: each shader file with the
//! constants `scivis_render::shaders` writes in ahead of it, not the file
//! alone, which is not a complete module.

use crate::unrolled::{
    assert_every_channel, assert_every_channel_arg, assert_one_channel_a_statement,
    assert_reads_where_packed, Arg,
};
use scivis_render::shaders::{self, composite_bindings};
use scivis_render::MAX_CHANNELS;

/// Parse + validate a WGSL source with naga (what wgpu does at runtime), so a
/// shader error is a failing test rather than a blank canvas at startup.
///
/// With no optional capabilities: what a WebGPU device without features
/// accepts, the least a host may hand the renderer.
///
/// What it does not catch: naga 29 does not hold a fragment shader's
/// derivatives and implicitly levelled samples (`fwidth`, `textureSample`) to
/// uniform control flow, as WGSL does (its analyzer turns those requirements
/// off, `DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE`); a `textureSample` under
/// a branch on an interpolated value passes here. A browser's WebGPU on Tint
/// (Chrome's) refuses such a module. The composite is the one shader with
/// such calls, and keeps them in uniform control flow by how it is written
/// (its `fs_main` says how).
fn validate(name: &str, src: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
        panic!("{name}: parse: {}", e.emit_to_string(src));
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: validate: {}", e.emit_to_string(src)));
    module
}

#[test]
fn every_shader_is_valid() {
    for (name, src) in shaders::wgsl_modules() {
        validate(name, &src);
    }
}

/// The composite's bindings are those the wgpu backend's bind group layout is
/// built from (`composite_bindings`), one integer and one float texture per
/// channel, and nothing else: a binding the layout lacks fails pipeline
/// creation, which in wgpu is fatal.
#[test]
fn composite_bindings_are_the_ones_the_backend_binds() {
    use composite_bindings::{FLOAT_TEXTURE, INT_TEXTURE, LUT, PARAMS, SAMPLER};
    let module = validate("composite", &shaders::composite_wgsl());
    let mut found: Vec<(u32, String)> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| {
            let b = var.binding.as_ref()?;
            assert_eq!(b.group, 0, "composite: `{:?}` is not in group 0", var.name);
            Some((b.binding, var.name.clone().unwrap_or_default()))
        })
        .collect();
    found.sort();
    let mut want = vec![
        (PARAMS, "params".to_string()),
        (LUT, "lut_tex".to_string()),
        (SAMPLER, "lut_sampler".to_string()),
    ];
    for c in 0..MAX_CHANNELS as u32 {
        want.push((INT_TEXTURE + c, format!("ch{c}_tex")));
        want.push((FLOAT_TEXTURE + c, format!("ch{c}_ftex")));
    }
    want.sort();
    assert_eq!(found, want);
}

/// Every uniform array the shaders index by channel is `MAX_CHANNELS` long.
/// (Their layout against the Rust structs that fill them is checked in
/// `wgpu_backend_tests.rs`, where those structs are.)
#[test]
fn channel_arrays_are_max_channels_long() {
    for (name, src, ty) in [
        ("composite", shaders::composite_wgsl(), "Params"),
        ("volume", shaders::volume_wgsl(), "VolParams"),
    ] {
        let module = validate(name, &src);
        let (_, params) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(ty))
            .unwrap_or_else(|| panic!("{name}: no struct `{ty}`"));
        let naga::TypeInner::Struct { members, .. } = &params.inner else {
            panic!("{name}: `{ty}` is not a struct");
        };
        let channels = members
            .iter()
            .find(|m| m.name.as_deref() == Some("channels"))
            .unwrap_or_else(|| panic!("{name}: `{ty}` has no `channels`"));
        let naga::TypeInner::Array {
            size: naga::ArraySize::Constant(n),
            ..
        } = module.types[channels.ty].inner
        else {
            panic!("{name}: `{ty}.channels` is not a sized array");
        };
        assert_eq!(n.get() as usize, MAX_CHANNELS, "{name}: `{ty}.channels`");
    }
}

/// The per-channel lines written out by hand: see `unrolled.rs`.
#[test]
fn per_channel_code_covers_every_channel() {
    let what = "composite.wgsl";
    let composite = include_str!("../../src/shaders/composite.wgsl");
    assert_every_channel(what, composite, "ch", "_tex");
    assert_every_channel(what, composite, "ch", "_ftex");
    assert_every_channel(what, composite, "params.channels[", "]");
    assert_every_channel_arg(what, composite, "apply_channel", Arg::At(1));
    assert_one_channel_a_statement(what, composite);
    let what = "volume.wgsl";
    let volume = include_str!("../../src/shaders/volume.wgsl");
    assert_every_channel(what, volume, "on(", ")");
    assert_every_channel(what, volume, "albedo_t(", ")");
    assert_every_channel_arg(what, volume, "lut_col", Arg::Last);
    assert_reads_where_packed(what, volume, "norm_ch(r.");
    assert_one_channel_a_statement(what, volume);
}

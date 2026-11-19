//! Compile-time assertions on the thread-safety of `ImageRenderResources`,
//! and on the traits each backend's implements.
//!
//! A host that wraps the renderer for its drawing layer (egui's paint callbacks
//! demand `Send + Sync + 'static`) needs to know exactly which markers hold.
//! Asserting it here means the README can't drift from the truth, and a
//! dependency bump that silently drops `Sync` fails the build instead of
//! surfacing as a confusing error in someone else's crate.

#![allow(dead_code)]

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_renders<T: scivis_render::VolumeRenderer>() {}

#[cfg(feature = "backend-wgpu")]
#[test]
fn wgpu_resources_are_send_and_sync() {
    assert_send::<scivis_render::wgpu_backend::ImageRenderResources>();
    assert_sync::<scivis_render::wgpu_backend::ImageRenderResources>();
}

/// Each backend is a renderer a host can be generic over: the 2D composite
/// and the volume, `VolumeRenderer` being the both of them. On every target,
/// wasm included, where only these traits stand between a host's sync and
/// the backend it was built with.
#[test]
fn every_backend_is_a_renderer() {
    #[cfg(feature = "backend-wgpu")]
    assert_renders::<scivis_render::wgpu_backend::ImageRenderResources>();
    #[cfg(feature = "backend-glow")]
    assert_renders::<scivis_render::glow_backend::ImageRenderResources>();
}

// Native only: on wasm, `glow::Context` wraps web-sys handles that are neither
// `Send` nor `Sync`, so the renderer isn't either — correctly, since a WebGL
// context is bound to one thread.
#[cfg(all(feature = "backend-glow", not(target_arch = "wasm32")))]
#[test]
fn glow_resources_are_send_and_sync() {
    assert_send::<scivis_render::glow_backend::ImageRenderResources>();
    assert_sync::<scivis_render::glow_backend::ImageRenderResources>();
}

/// The parameter types cross thread boundaries in a host's own plumbing, so
/// they must be freely shareable.
#[test]
fn parameter_types_are_send_and_sync() {
    assert_send::<scivis_render::VolumeParams>();
    assert_sync::<scivis_render::VolumeParams>();
    assert_send::<scivis_render::ChannelUniform>();
    assert_sync::<scivis_render::ChannelUniform>();
}

//! Compile-time assertions on the thread-safety of `ImageRenderResources`.
//!
//! A host that wraps the renderer for its drawing layer (egui's paint callbacks
//! demand `Send + Sync + 'static`) needs to know exactly which markers hold.
//! Asserting it here means the README can't drift from the truth, and a
//! dependency bump that silently drops `Sync` fails the build instead of
//! surfacing as a confusing error in someone else's crate.

#![allow(dead_code)]

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[cfg(feature = "backend-wgpu")]
#[test]
fn wgpu_resources_are_send_and_sync() {
    assert_send::<stack_renderer::wgpu_backend::ImageRenderResources>();
    assert_sync::<stack_renderer::wgpu_backend::ImageRenderResources>();
}

// Native only: on wasm, `glow::Context` wraps web-sys handles that are neither
// `Send` nor `Sync`, so the renderer isn't either — correctly, since a WebGL
// context is bound to one thread.
#[cfg(all(feature = "backend-glow", not(target_arch = "wasm32")))]
#[test]
fn glow_resources_are_send_and_sync() {
    assert_send::<stack_renderer::glow_backend::ImageRenderResources>();
    assert_sync::<stack_renderer::glow_backend::ImageRenderResources>();
}

/// The parameter types cross thread boundaries in a host's own plumbing, so
/// they must be freely shareable.
#[test]
fn parameter_types_are_send_and_sync() {
    assert_send::<stack_renderer::VolumeParams>();
    assert_sync::<stack_renderer::VolumeParams>();
    assert_send::<stack_renderer::ChannelUniform>();
    assert_sync::<stack_renderer::ChannelUniform>();
}

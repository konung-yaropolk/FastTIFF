//! The `egui_wgpu::CallbackTrait` impl that gets invoked once per egui
//! frame to draw the current image into the rect `app.rs` allocates for
//! it. Mirrors the structure of egui's own `custom3d_wgpu` example.

use crate::render::pipeline::ImageRenderResources;
use eframe::egui_wgpu::{self, wgpu};

pub struct ImagePaintCallback;

impl egui_wgpu::CallbackTrait for ImagePaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // All GPU state updates (texture uploads, uniform writes) happen
        // synchronously in app.rs before this callback is queued, via
        // direct queue.write_texture/write_buffer calls — there's nothing
        // left to do here. Kept as a no-op rather than removed so the
        // intended extension point (e.g. moving uploads here if profiling
        // ever shows a benefit) stays obvious.
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &ImageRenderResources = match resources.get() {
            Some(r) => r,
            None => return, // not yet installed (shouldn't happen after app init)
        };
        resources.paint(render_pass);
    }
}

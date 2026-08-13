//! The eframe/egui adapter over [`scivis_render`], for the web.
//!
//! The browser twin of `FastTIFF/src/render.rs`. Same three jobs — pull the GPU
//! handles out of eframe's `CreationContext`, wrap them in the `Arc<Mutex<…>>`
//! egui's paint callbacks require, and turn a draw request into an
//! [`egui::Shape`] — minus the parts that only mean something natively (the
//! glow backend, and `tune_native_options`, which takes an `eframe::NativeOptions`
//! that doesn't exist on this target).
//!
//! It's a separate file rather than a shared module because the two differ in
//! exactly those places, and a `#[path]` include reaching into another crate's
//! `src/` would be invisible coupling for the ~40 lines it saves.

use eframe::egui_wgpu::{self, wgpu};
use std::sync::{Arc, Mutex};

/// The web counterpart of the desktop adapter's `tune_native_options`.
///
/// eframe would otherwise create the device with `wgpu::Limits::default()`,
/// which is *smaller* than what this renderer needs: the composite pass binds
/// 13 sampled textures (6 integer + 6 float channels + the LUT array) and the
/// volume pass allocates 3D textures. Exceeding any single limit makes wgpu
/// reject resource creation, and the resulting device errors surface far from
/// the cause — egui's own font atlas fails to allocate, and the next frame
/// panics in `egui-wgpu` with "tried to update a texture that has not been
/// allocated yet".
///
/// Asking for the adapter's own limits is always safe: they are by definition
/// what the hardware supports, and every texture here is already sized against
/// a memory budget.
pub fn tune_web_options(options: &mut eframe::WebOptions) {
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(|adapter| wgpu::DeviceDescriptor {
            label: Some("fasttiff wgpu device"),
            required_features: scivis_render::wgpu_backend::optional_features(adapter),
            required_limits: adapter.limits(),
            ..Default::default()
        });
    }
}

pub use scivis_render::wgpu_backend::ImageRenderResources;

/// Shared handle to the GPU resources. `Arc<Mutex>` because an egui paint
/// callback must be `Send + Sync + 'static`; uploads happen before the callback
/// paints, on the same thread, so the lock is uncontended.
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// Build the render resources from eframe's wgpu render state. `device`/`queue`
/// are refcounted handles, so the clones create nothing.
pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
    let rs = cc
        .wgpu_render_state
        .as_ref()
        .expect("FastTIFF needs the wgpu backend");
    Arc::new(Mutex::new(ImageRenderResources::new(
        rs.device.clone(),
        rs.queue.clone(),
        rs.target_format,
    )))
}

/// The egui paint callback that draws the composited image into `rect`.
pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
        rect,
        ImagePaint { resources: render.clone() },
    ))
}

/// The egui paint callback that ray-marches the 3D volume into `rect`.
pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
    egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
        rect,
        VolumePaint { resources: render.clone() },
    ))
}

struct ImagePaint {
    resources: Render,
}

impl egui_wgpu::CallbackTrait for ImagePaint {
    // `prepare` stays the trait default: every upload already happened in
    // `Viewer::sync` before this callback was queued.
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        if let Ok(r) = self.resources.lock() {
            r.paint(pass);
        }
    }
}

struct VolumePaint {
    resources: Render,
}

impl egui_wgpu::CallbackTrait for VolumePaint {
    /// The volume uniform is staged here rather than during `sync` — this is
    /// the wgpu backend's one ordering requirement, and `prepare` is where a
    /// queue is available.
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Ok(r) = self.resources.lock() {
            r.write_volume_uniform();
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        if let Ok(r) = self.resources.lock() {
            r.paint_volume(pass);
        }
    }
}

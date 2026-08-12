//! The eframe/egui adapter over [`fast_tiff_render`].
//!
//! All GPU work — pipelines, textures, shaders, the ray-marcher — lives in the
//! `fast-tiff-render` crate, which knows nothing about egui. This module is the
//! only place that bridges the two, and it does exactly three things:
//!
//!   * pulls the GPU handles out of eframe's `CreationContext` ([`init`]),
//!   * wraps the resources in the `Arc<Mutex<…>>` egui's paint callbacks
//!     require (`Send + Sync + 'static`), and
//!   * turns a draw request into an [`egui::Shape`] ([`paint_callback`],
//!     [`paint_volume_callback`]).
//!
//! Everything else is a re-export, so the rest of the app keeps saying
//! `crate::render::ChannelKind` and never names glow, wgpu, *or* the render
//! crate. A different host (a browser canvas, an offscreen encoder) writes its
//! own module of this size and reuses the renderer untouched — that is the
//! whole reason the GPU code moved out.

// Exactly one renderer must be selected. This is an *app* constraint, not a
// renderer one: eframe initializes a single GPU backend per process, so the
// `Render` alias and `init` below can only resolve to one. (The render crate
// itself is happy to compile both.) These guards turn an accidental
// both/neither feature set into a clear error instead of a confusing
// "unresolved import" cascade from the re-exports.
#[cfg(all(feature = "renderer-glow", feature = "renderer-wgpu"))]
compile_error!(
    "features `renderer-glow` and `renderer-wgpu` are mutually exclusive — enable exactly one"
);
#[cfg(not(any(feature = "renderer-glow", feature = "renderer-wgpu")))]
compile_error!(
    "no renderer selected — enable feature `renderer-wgpu` (default) or `renderer-glow`"
);

use std::sync::{Arc, Mutex};

// The two parameter types the UI itself names (the 3D settings pop-up edits
// them). The rest of the renderer's vocabulary reaches the app through
// `fast_tiff_core`, which is what actually drives the GPU.
pub use fast_tiff_render::{VolumeInterp, VolumeRender};

// Pick the one backend the selected eframe renderer can drive. The
// `not(renderer-glow)` guard on the wgpu arm means enabling *both* features (a
// hard error, above) selects only glow here — so the build fails with the clean
// `compile_error!` message rather than a duplicate-definition cascade.
#[cfg(feature = "renderer-glow")]
use fast_tiff_render::glow_backend as backend;
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
use fast_tiff_render::wgpu_backend as backend;

pub use backend::{ImageRenderResources, BACKEND};

/// Shared handle to the GPU resources. `Arc<Mutex>` because an egui paint
/// callback must be `Send + Sync + 'static`; uploads happen in `app::sync_gpu`,
/// so the lock is uncontended — both run on the UI thread and never overlap
/// (uploads finish before the callback paints).
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// The `eframe::Renderer` the compiled-in backend needs requested in
/// `NativeOptions`.
#[cfg(feature = "renderer-glow")]
pub const RENDERER: eframe::Renderer = eframe::Renderer::Glow;
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
pub const RENDERER: eframe::Renderer = eframe::Renderer::Wgpu;

// --- glow ------------------------------------------------------------------

#[cfg(feature = "renderer-glow")]
mod glue {
    use super::{ImageRenderResources, Render};
    use eframe::egui_glow;
    use std::sync::{Arc, Mutex};

    /// Build the render resources from eframe's glow context. Called once at
    /// startup.
    pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
        let gl = cc
            .gl
            .clone()
            .expect("FastTIFF requires the glow backend (NativeOptions::renderer = Glow)");
        Arc::new(Mutex::new(ImageRenderResources::new(gl)))
    }

    /// The egui paint callback that draws the current image into `rect`.
    /// egui_glow has already set the viewport/scissor to `rect` when this runs.
    pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        shape(render, rect, |r, gl| r.paint(gl))
    }

    /// The egui paint callback that ray-marches the 3D volume into `rect`.
    pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        shape(render, rect, |r, gl| r.paint_volume(gl))
    }

    /// Shared body of the two callbacks: capture a clone of the resources, lock
    /// them at paint time, and hand `draw` the live GL context egui gives us.
    fn shape(
        render: &Render,
        rect: egui::Rect,
        draw: impl Fn(&ImageRenderResources, &eframe::glow::Context) + Send + Sync + 'static,
    ) -> egui::Shape {
        let res = render.clone();
        let callback = egui_glow::CallbackFn::new(move |_info, painter| {
            if let Ok(r) = res.lock() {
                draw(&r, painter.gl());
            }
        });
        egui::Shape::Callback(egui::PaintCallback { rect, callback: Arc::new(callback) })
    }

    /// Backend hook for `eframe::NativeOptions`; the glow backend needs none.
    pub fn tune_native_options(_options: &mut eframe::NativeOptions) {}
}

// --- wgpu ------------------------------------------------------------------

#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
mod glue {
    use super::{ImageRenderResources, Render};
    use eframe::egui_wgpu::{self, wgpu};
    use std::sync::{Arc, Mutex};

    /// Build the render resources from eframe's wgpu render state. Called once
    /// at startup. `device`/`queue` are cheap refcounted handles, so the clones
    /// hand the renderer its own without creating anything.
    pub fn init(cc: &eframe::CreationContext<'_>) -> Render {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("FastTIFF requires the wgpu backend (NativeOptions::renderer = Wgpu)");
        Arc::new(Mutex::new(ImageRenderResources::new(
            rs.device.clone(),
            rs.queue.clone(),
            rs.target_format,
        )))
    }

    /// The egui paint callback that draws the current image into `rect`.
    pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
            rect,
            ImagePaintCallback { resources: render.clone() },
        ))
    }

    /// The egui paint callback that ray-marches the 3D volume into `rect`.
    pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
            rect,
            VolumePaintCallback { resources: render.clone() },
        ))
    }

    /// Backend hook for `eframe::NativeOptions`: ask the renderer which optional
    /// device features are worth having (16-bit-norm textures, for full-precision
    /// volume data) and request them when the adapter offers them.
    pub fn tune_native_options(options: &mut eframe::NativeOptions) {
        if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
            setup.device_descriptor = Arc::new(|adapter| {
                wgpu::DeviceDescriptor {
                    label: Some("egui wgpu device"),
                    required_features: fast_tiff_render::wgpu_backend::optional_features(adapter),
                    // Request exactly the adapter's own limits rather than wgpu's
                    // generic defaults. The defaults ask for more than some low-end
                    // GPUs provide — e.g. the Raspberry Pi's V3DV Vulkan driver caps
                    // `max_color_attachments` at 4 vs the default 8 — and wgpu rejects
                    // the entire device request when any single limit is exceeded. We
                    // only ever draw to one color target and size every texture against
                    // a memory budget, so the adapter's real limits are always enough
                    // and, by definition, never exceed what the hardware allows. (On
                    // the GL backend this also reports the driver's true max 2D texture
                    // size instead of WebGL2's conservative 2048 floor.)
                    required_limits: adapter.limits(),
                    ..Default::default()
                }
            });
        }
    }

    /// The volume-view callback. `prepare` marshals the stashed camera/window
    /// params into the uniform buffer, then `paint` draws.
    struct VolumePaintCallback {
        resources: Render,
    }

    impl egui_wgpu::CallbackTrait for VolumePaintCallback {
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
            render_pass: &mut wgpu::RenderPass<'static>,
            _resources: &egui_wgpu::CallbackResources,
        ) {
            if let Ok(r) = self.resources.lock() {
                r.paint_volume(render_pass);
            }
        }
    }

    /// The 2D image callback. Holds its own clone of the resources rather than
    /// parking them in egui_wgpu's `CallbackResources` map (which the upstream
    /// `custom3d_wgpu` example uses), so both backends share one app-side shape.
    struct ImagePaintCallback {
        resources: Render,
    }

    impl egui_wgpu::CallbackTrait for ImagePaintCallback {
        // `prepare` is left as the trait default (no-op): all GPU state updates
        // (texture uploads, uniform writes) happen synchronously in
        // `app::sync_gpu` before this callback is queued.
        fn paint(
            &self,
            _info: egui::PaintCallbackInfo,
            render_pass: &mut wgpu::RenderPass<'static>,
            _resources: &egui_wgpu::CallbackResources,
        ) {
            if let Ok(r) = self.resources.lock() {
                r.paint(render_pass);
            }
        }
    }
}

pub use glue::{init, paint_callback, paint_volume_callback, tune_native_options};

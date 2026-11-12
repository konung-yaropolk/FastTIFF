//! The eframe/egui adapter over [`scivis_render`].
//!
//! All GPU work — pipelines, textures, shaders, the ray-marcher — lives in the
//! `scivis-render` crate, which knows nothing about egui. This module is the
//! only place that bridges the two, and it does exactly three things:
//!
//!   * pulls the GPU handles out of eframe's `CreationContext` ([`init`]),
//!   * wraps the resources in the `Arc<Mutex<…>>` egui's paint callbacks
//!     require (`Send + Sync + 'static`), and
//!   * turns a draw request into an [`egui::Shape`] ([`paint_callback`],
//!     [`paint_volume_callback`]).
//!
//! It also re-exports the handful of renderer types the UI itself names, so no
//! other file in the app mentions glow, wgpu, *or* `scivis_render`. Everything
//! else reaches the GPU through `fast_tiff_viewer`, which is what actually drives
//! it. A different host (a browser canvas, an offscreen encoder) writes its own
//! module of this size and reuses the renderer untouched — that is the whole
//! reason the GPU code moved out.

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
// `fast_tiff_viewer`, which is what actually drives the GPU.
pub use scivis_render::{VolumeInterp, VolumeRender, MOVING_DOWNSCALE};

// Pick the one backend the selected eframe renderer can drive. The
// `not(renderer-glow)` guard on the wgpu arm means enabling *both* features (a
// hard error, above) selects only glow here — so the build fails with the clean
// `compile_error!` message rather than a duplicate-definition cascade.
#[cfg(feature = "renderer-glow")]
use scivis_render::glow_backend as backend;
#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
use scivis_render::wgpu_backend as backend;

pub use backend::{ImageRenderResources, BACKEND};

/// Shared handle to the GPU resources. `Arc<Mutex>` because an egui paint
/// callback must be `Send + Sync + 'static`; uploads happen in `app::sync_gpu`,
/// so the lock is uncontended — both run on the UI thread and never overlap
/// (uploads finish before the callback paints).
pub type Render = Arc<Mutex<ImageRenderResources>>;

/// The `eframe::Renderer` the compiled-in backend needs requested in
/// `NativeOptions`. Only the desktop's options ask: `WebOptions` has no
/// equivalent field, the browser build being always wgpu.
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

    /// The egui paint callback that ray-marches the 3D volume into `rect` —
    /// at reduced size and scaled up, while the view is moving (the renderer
    /// does both; see `set_volume_target`).
    pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        shape(render, rect, |r, gl| r.paint_volume(gl))
    }

    /// Shared body of the two callbacks: capture a clone of the resources, lock
    /// them at paint time, and hand `draw` the live GL context egui gives us.
    fn shape(
        render: &Render,
        rect: egui::Rect,
        draw: impl Fn(&mut ImageRenderResources, &eframe::glow::Context) + Send + Sync + 'static,
    ) -> egui::Shape {
        let res = render.clone();
        let callback = egui_glow::CallbackFn::new(move |_info, painter| {
            if let Ok(mut r) = res.lock() {
                draw(&mut r, painter.gl());
            }
        });
        egui::Shape::Callback(egui::PaintCallback {
            rect,
            callback: Arc::new(callback),
        })
    }

    /// Backend hook for `eframe::NativeOptions`; the glow backend needs none.
    #[cfg(not(target_arch = "wasm32"))]
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
        let mut resources =
            ImageRenderResources::new(rs.device.clone(), rs.queue.clone(), rs.target_format);
        // So the renderer can ask the driver how much GPU memory a volume may
        // take, rather than guess.
        resources.set_adapter(rs.adapter.clone());
        // And the GPU registrations run on this device rather than opening
        // their own: a second device can deadlock against this one on a card
        // short of memory (see `share_gpu_device`). The registrations' GPU
        // paths are the desktop's alone, so the viewer has this nowhere else.
        #[cfg(not(target_arch = "wasm32"))]
        fast_tiff_viewer::share_gpu_device(&rs.device, &rs.queue, &rs.adapter.get_info());
        // A lost device is noted, so the app can close saying why rather than
        // let egui panic on its next draw (see `gpu_lost`) — where the
        // platform closes over it at all, which the browser does not
        // (`Platform::gpu_lost`). Dropping the device at exit reports
        // `Destroyed`, which is not news.
        rs.device.set_device_lost_callback(|reason, message| {
            if !matches!(reason, wgpu::DeviceLostReason::Destroyed) {
                super::note_gpu_lost(message);
            }
        });
        Arc::new(Mutex::new(resources))
    }

    /// The egui paint callback that draws the current image into `rect`.
    pub fn paint_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
            rect,
            ImagePaintCallback {
                resources: render.clone(),
            },
        ))
    }

    /// The egui paint callback that ray-marches the 3D volume into `rect`.
    pub fn paint_volume_callback(render: &Render, rect: egui::Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(
            rect,
            VolumePaintCallback {
                resources: render.clone(),
            },
        ))
    }

    /// Backend hook for `eframe::NativeOptions`: ask the renderer which optional
    /// device features are worth having (16-bit-norm textures, for full-precision
    /// volume data) and request them when the adapter offers them.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn tune_native_options(options: &mut eframe::NativeOptions) {
        if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
            setup.device_descriptor = Arc::new(|adapter| {
                wgpu::DeviceDescriptor {
                    label: Some("egui wgpu device"),
                    required_features: scivis_render::wgpu_backend::optional_features(adapter),
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

    /// The web twin of `tune_native_options`, and not optional: eframe would
    /// otherwise build the device with `wgpu::Limits::default()`, which is
    /// *smaller* than this renderer needs — the composite pass binds 13 sampled
    /// textures and the volume pass allocates 3D textures. Exceeding a limit
    /// makes wgpu reject resource creation, and the fallout surfaces far from
    /// the cause: egui's own font atlas fails to allocate and the next frame
    /// panics inside egui-wgpu. The adapter's own limits are by definition
    /// supported, and every texture here is already sized against a budget.
    #[cfg(target_arch = "wasm32")]
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

    /// The volume-view callback. `prepare` marshals the stashed camera/window
    /// params into the uniform buffer and, while the view is moving, renders
    /// it at reduced size into egui's encoder — which runs ahead of egui's own
    /// pass — then `paint` draws it (or scales the reduced render up).
    struct VolumePaintCallback {
        resources: Render,
    }

    impl egui_wgpu::CallbackTrait for VolumePaintCallback {
        fn prepare(
            &self,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _screen: &egui_wgpu::ScreenDescriptor,
            encoder: &mut wgpu::CommandEncoder,
            _resources: &mut egui_wgpu::CallbackResources,
        ) -> Vec<wgpu::CommandBuffer> {
            if let Ok(mut r) = self.resources.lock() {
                r.prepare_volume(encoder);
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

// Each option hook takes its own target's eframe options, a type that exists
// on that target alone.
#[cfg(not(target_arch = "wasm32"))]
pub use glue::tune_native_options;
#[cfg(target_arch = "wasm32")]
pub use glue::tune_web_options;
pub use glue::{init, paint_callback, paint_volume_callback};

// --- a lost graphics device -------------------------------------------------

/// What wgpu said when the graphics device was lost, once it has been.
///
/// A lost device cannot be got back: eframe has no way to make a new one, and
/// nothing in the window can be drawn without it — egui panics on its next
/// draw ("Failed to create staging buffer"). What can be done is to close
/// saying why, in words, rather than with a renderer's internals.
///
/// Recorded on every target; whether the app closes over it is the
/// platform's to say, and only the desktop does (`Platform::gpu_lost`).
static GPU_LOST: Mutex<Option<String>> = Mutex::new(None);

#[cfg(all(feature = "renderer-wgpu", not(feature = "renderer-glow")))]
fn note_gpu_lost(message: String) {
    *GPU_LOST.lock().unwrap_or_else(|e| e.into_inner()) = Some(message);
}

/// What to close with, once the graphics device is gone: for the frame loop,
/// which ends there rather than let egui draw.
pub fn gpu_lost() -> Option<String> {
    let lost = GPU_LOST.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    Some(gpu_failure_text(&lost))
}

/// For the panic hook: whether `panic` (its text) is the graphics device going
/// away, and if so what to say instead of it.
///
/// Three ways it shows: the frame loop ending on a loss it was told about
/// ([`gpu_lost`]); egui panicking first, on a device lost before anyone
/// heard; and wgpu's own panic for running out of video memory.
pub fn explain_gpu_failure(panic: &str) -> Option<String> {
    let recorded = GPU_LOST.lock().unwrap_or_else(|e| e.into_inner()).clone();
    explain(recorded.as_deref(), panic)
}

fn explain(recorded: Option<&str>, panic: &str) -> Option<String> {
    // egui-wgpu's words when the device it writes to is gone, and wgpu's
    // default handler's when an allocation found no video memory.
    let egui_lost = panic.contains("Failed to create staging buffer");
    let out_of_memory = panic.contains("wgpu error") && panic.contains("Out of Memory");
    match recorded {
        Some(message) => Some(gpu_failure_text(message)),
        None if egui_lost || out_of_memory => Some(gpu_failure_text(panic)),
        None => None,
    }
}

fn gpu_failure_text(detail: &str) -> String {
    let detail = detail.trim();
    format!(
        "The graphics card stopped working for FastTIFF, so it cannot draw any more and \
         has to close.\n\n\
         This usually means the card ran out of video memory: two FastTIFF windows each \
         showing a large 3D view can fill a card between them, and so can other programs \
         using the GPU. It can also mean the system reset the card because one job on it \
         took too long.\n\n\
         Reopen the file to carry on. To avoid it, show large stacks in 3D in one window \
         at a time, or close other programs that use the GPU.{}",
        if detail.is_empty() {
            String::new()
        } else {
            format!("\n\nDetails: {detail}")
        }
    )
}

#[cfg(test)]
mod gpu_lost_tests {
    use super::explain;

    #[test]
    fn a_lost_device_is_explained_whichever_way_it_shows() {
        // Told of by wgpu, with or without a message.
        let told = explain(Some("device lost: timeout"), "anything").expect("explained");
        assert!(told.contains("ran out of video memory") && told.contains("device lost: timeout"));
        assert!(!explain(Some(""), "x")
            .expect("explained")
            .contains("Details"));
        // egui first, on a device nobody had heard was lost.
        let egui = "panicked at egui-wgpu-0.34.3\\src\\renderer.rs:971:17:\n\
                    Failed to create staging buffer for index data. Index count: 3480.";
        assert!(explain(None, egui)
            .expect("explained")
            .contains("Index count: 3480"));
        // wgpu's own out-of-memory panic.
        assert!(explain(None, "wgpu error: Out of Memory\n\nCaused by: ...").is_some());
    }

    #[test]
    fn any_other_panic_is_left_as_it_is() {
        assert_eq!(
            explain(None, "index out of bounds: the len is 3 but the index is 7"),
            None
        );
        assert_eq!(explain(None, "wgpu error: Validation Error"), None);
    }
}

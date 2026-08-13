//! FastTIFF's egui interface, compiled to WebAssembly.
//!
//! The desktop app and this one are the *same* UI toolkit over the *same*
//! core — the only difference is the host: eframe on winit there, eframe on a
//! browser canvas here. Compare with `FastTIFF-web/`, which puts a React UI
//! over the identical core.

mod app;
mod render;

use wasm_bindgen::prelude::*;

/// Boot the viewer onto `canvas`.
///
/// Async because WebGPU adapter/device requests are promises. Resolves once the
/// app is running; the returned handle keeps eframe's event loop alive, so JS
/// must hold onto it.
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<WebHandle, JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);

    let mut web_options = eframe::WebOptions::default();
    render::tune_web_options(&mut web_options);

    let runner = eframe::WebRunner::new();
    runner
        .start(
            canvas,
            web_options,
            Box::new(|cc| Ok(Box::new(app::WebApp::new(cc)))),
        )
        .await?;
    Ok(WebHandle { runner })
}

/// Keeps the running app alive, and lets the page tear it down.
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[wasm_bindgen]
impl WebHandle {
    /// Stop the app and release its GPU resources.
    pub fn destroy(&self) {
        self.runner.destroy();
    }
}

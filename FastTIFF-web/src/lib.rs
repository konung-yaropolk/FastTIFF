//! FastTIFF's egui interface, compiled to WebAssembly.
//!
//! There is no UI code here: the interface is the **same** [`fasttiff::ViewerApp`]
//! the desktop binary runs. This crate is only the browser host — it hands
//! eframe a canvas instead of a window, and hands the shared adapter the
//! device-limit tuning the web needs.
//!
//! The desktop entry point is `FastTIFF/src/main.rs`; compare the two and the
//! whole difference between the platforms is visible at a glance.

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
    fasttiff::render::tune_web_options(&mut web_options);

    let runner = eframe::WebRunner::new();
    runner
        .start(
            canvas,
            web_options,
            Box::new(|cc| {
                // Dark by default, as on the desktop: a light chrome throws
                // stray light onto the canvas and skews how dim structures read.
                cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
                let render = fasttiff::render::init(cc);
                // No initial path — a browser has no argv and no filesystem;
                // files arrive from the picker or a drop.
                Ok(Box::new(fasttiff::ViewerApp::new(None, render)))
            }),
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

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

    // A document this tab was opened to show, if it was. That is how a plugin
    // result reaches a new tab: the tab that ran the plugin put the encoded
    // TIFF in a blob and opened us with its URL in the query. An ordinary
    // visit answers `None` and nothing below changes. See
    // `fasttiff::web_open`.
    //
    // Awaited before the app is built rather than after, so the viewer comes
    // up with the image already in it — the same way the desktop binary opens
    // a path from its argv — instead of appearing empty and filling in.
    let opened = fasttiff::web_open::take_pending()
        .await
        .map(|(bytes, name)| fasttiff::app::Opened::Bytes(bytes, name));

    let mut web_options = eframe::WebOptions::default();
    fasttiff::render::tune_web_options(&mut web_options);

    let runner = eframe::WebRunner::new();
    runner
        .start(
            canvas,
            web_options,
            Box::new(|cc| {
                // Theme + interface scale. The same call the desktop binary
                // makes — it is what decides that the web build draws its
                // chrome at 150%, so the difference lives in the shared crate
                // rather than here.
                fasttiff::install_chrome(&cc.egui_ctx);
                let render = fasttiff::render::init(cc);
                // No initial *path* — a browser has no argv and no filesystem.
                // Files arrive from the picker, from a drop, or, for a plugin
                // result handed over from another tab, as the bytes read above.
                Ok(Box::new(fasttiff::ViewerApp::new(opened, render)))
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

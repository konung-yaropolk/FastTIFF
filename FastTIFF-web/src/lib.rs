//! FastTIFF's egui interface, compiled to WebAssembly.
//!
//! There is no UI code here: the interface is the **same** [`fasttiff::ViewerApp`]
//! the desktop binary runs. This crate is only the browser host — it hands
//! eframe a canvas instead of a window, hands the shared adapter the
//! device-limit tuning the web needs, and keeps the browser's page zoom off
//! the keys that zoom the picture.
//!
//! The desktop entry point is `FastTIFF/src/main.rs`; compare the two and the
//! whole difference between the platforms is visible at a glance.
//!
//! Empty on every other target. `eframe::WebRunner` exists only on wasm32, and
//! this crate is a workspace member, so a host `cargo test --workspace` has to
//! be able to build it — as nothing.
#![cfg(target_arch = "wasm32")]

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
        .map(|(bytes, name)| fasttiff::app::Opened::Bytes(bytes.into(), name));

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
                // chrome at 125%, so the difference lives in the shared crate
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
    // The app is running by now. Without the listener a zoom key zooms the
    // page as well, as it used to, which is no reason to fail the start.
    if let Err(err) = keep_zoom_keys_from_page(&runner) {
        log::warn!("the zoom keys will zoom the page too: {err:?}");
    }
    Ok(WebHandle { runner })
}

/// Keep the browser from zooming the page on the presses that zoom the
/// picture, so that one press zooms only the picture.
///
/// Ctrl or Cmd with Plus, Equals or Minus steps the picture's zoom
/// ([`fasttiff::zoom_key`]), and the same chords are the browser's page zoom.
/// eframe prevents the default of a few chords an app is likely to want
/// (Ctrl+S, Ctrl+O, Ctrl+P, Ctrl+Comma) and not of these, and has no option
/// for more: its web runner's `should_prevent_default_for_key` decides for a
/// key, and `WebOptions::should_prevent_default` is not asked about one. So
/// one press zoomed the picture and the page around it, which, by changing
/// the device pixel ratio, scaled the whole interface as well.
///
/// Cmd, in a browser, is the Meta key on every platform, so on Windows and
/// Linux it is the Windows or Super key: eframe reports it as `mac_cmd` and
/// `command` wherever it runs, and egui's wheel zoom already took it there.
///
/// This listener prevents the default of exactly the presses `zoom_key`
/// takes, judged on the modifiers and key eframe will read from the same
/// event, and does nothing else. It does not stop the event, which eframe
/// still has to see for the picture to zoom.
///
/// - Where: on the document, in the capture phase. eframe takes keys on the
///   canvas, and on its text agent, a hidden `<input>` it adds to the page,
///   which has the focus while one of the app's text fields does. Each of
///   those stops the event there (`WebOptions::should_stop_propagation` is
///   true for every event), so a listener on the document in the bubble phase
///   would hear no key; in the capture phase it hears each one on the way
///   down, before either of eframe's, whichever has the focus. That is soon
///   enough: a default is taken after the event has been dispatched.
/// - When: only while eframe counts the app as focused (`has_focus`: the
///   canvas or the text agent), so anything else on the page keeps the
///   browser's chords. That is whenever the app has the focus, whether or not
///   it has a picture to zoom: on the empty viewport and in the 3D view these
///   chords do nothing at all, and the page zooms only from the browser's
///   menu. While the app has the focus the chords are the picture's and never
///   the page's, so a page zoom cannot slip in from one view, where the app
///   ignores them, and stay to scale the chrome in the next. Asking the app
///   instead would need it to tell the host, each frame, whether it would
///   zoom.
/// - Not with Alt held. A browser on Windows reports AltGr as Ctrl and Alt,
///   and a character typed with AltGr into one of the app's text fields
///   arrives as the text agent's `input` event, which preventing the keydown
///   would cancel. The browsers' own zoom chords have no Alt in them.
/// - Ctrl+0 is not among them (see `zoom_key`): it still resets the page.
///
/// eframe keeps the closure with its own listeners (`add_event_listener_ex`)
/// for as long as the app runs, drops it with them when the app is destroyed
/// or panics, and does not call it once the app has panicked. Dropping it is
/// not enough to take it off the document, though: eframe removes each of
/// its listeners with `removeEventListener` and no capture flag, which
/// matches only a listener added without one, so this one would stay, and
/// every key pressed anywhere on the page afterwards would throw (wasm-bindgen
/// refuses to call a closure that has been dropped). So it is added with an
/// abort signal, and the closure owns the controller, as [`AbortOnDrop`],
/// which aborts it, and with it removes the listener, when the closure is
/// dropped.
fn keep_zoom_keys_from_page(runner: &eframe::WebRunner) -> Result<(), JsValue> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| JsValue::from_str("no document to listen on"))?;
    let removal = AbortOnDrop(web_sys::AbortController::new()?);
    let options = web_sys::AddEventListenerOptions::new();
    options.set_capture(true);
    options.set_signal(&removal.0.signal());
    runner.add_event_listener_ex(
        &document,
        "keydown",
        &options,
        move |event: web_sys::KeyboardEvent, app, _| {
            // Only so that the closure owns it, and drops it when eframe
            // drops the closure.
            let _removal = &removal;
            // What eframe passes over: every key while the app does not have
            // the focus, and any while a text is being composed.
            if event.is_composing() || !app.has_focus() {
                return;
            }
            // As eframe's web runner reads them from a key event (its
            // `modifiers_from_kb_event`, which it keeps to itself), and the
            // key as it names it (`egui::Key::from_name`, which its
            // `translate_key` is): what the app will be handed for this
            // press.
            let modifiers = egui::Modifiers {
                alt: event.alt_key(),
                ctrl: event.ctrl_key(),
                shift: event.shift_key(),
                mac_cmd: event.meta_key(),
                command: event.ctrl_key() || event.meta_key(),
            };
            let zooms = egui::Key::from_name(&event.key())
                .is_some_and(|key| fasttiff::zoom_key(modifiers, key).is_some());
            if zooms && !modifiers.alt {
                event.prevent_default();
            }
        },
    )
}

/// An abort controller that is aborted when this is dropped, which removes
/// every listener added with its signal. See `keep_zoom_keys_from_page` for
/// why that listener needs one. If eframe fails to add the listener, it drops
/// the closure there and then, and the abort removes nothing.
struct AbortOnDrop(web_sys::AbortController);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
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

//! Handing a file to the user — the browser's answer to a save dialog.
//!
//! On the desktop, saving asks where to put the file and then writes it there.
//! A browser tab cannot do either half: it has no filesystem to write to, and
//! no way to ask where. What it has instead is a *download* — a buffer handed
//! to the user agent with a suggested name, which the browser then puts
//! wherever its own settings say, asking the user if they have told it to.
//!
//! So the two halves swap places. The encode is the same
//! ([`fast_tiff_viewer::save`] produces identical bytes either way); what
//! changes is that the name is decided by the app rather than by the user (in
//! `app/platform/naming.rs`, which is compiled and tested on every target),
//! and the destination is decided by the browser rather than by us. Someone
//! who wants to choose both turns on "ask where to save each file" and gets
//! the dialog back — from the browser, which is the only thing that can show
//! it.
//!
//! Browser-only, as a module: everything here is `web-sys`.
//!
//! # How a `Vec<u8>` becomes a download
//!
//! The same object-URL route [`crate::web_open`] uses to hand a document to a
//! second tab, pointed at the user instead:
//!
//!   1. wrap the bytes in a `Blob` and take an object URL for it;
//!   2. make an `<a href=… download="name.tif">` and click it;
//!   3. revoke the URL once the download has had a chance to start.
//!
//! An `<a download>` rather than `window.open`: a download is not a popup, so
//! it is not subject to the popup blocker, and the `download` attribute is what
//! makes the browser save the file rather than try to display it. (Nothing
//! displays a TIFF, so a browser handed one without that attribute would
//! usually download it anyway — but "usually" is not a thing to build on, and
//! the attribute is also where the file's name comes from.)
//!
//! # On revoking
//!
//! Not immediately after the click. The click starts a download that reads the
//! blob, and revoking the URL in the same task can cancel it before it has
//! read anything — a race that depends on the browser and on how large the file
//! is, which is the worst kind. The revoke therefore goes on a timer, long
//! enough that the download has certainly begun and short enough that a stack
//! is not held in memory twice for the life of the tab.
//!
//! Not *never*, either: these are whole microscopy stacks, and a blob that is
//! never released is a copy of one retained until the tab is closed.

use wasm_bindgen::JsCast;

/// How long the blob is kept alive after the click, in milliseconds.
///
/// The download only needs the URL to resolve, which happens as the click is
/// dispatched; this is slack for a browser that defers that to a later task.
/// Ten seconds is far more than any of them take and still bounded.
const REVOKE_AFTER_MS: i32 = 10_000;

/// Offer `bytes` to the user as a download named `name`, of media type
/// `mime` — which a download manager may go by in naming or opening it.
///
/// `Err` is "the browser would not take it", which is a thing to report rather
/// than retry — the status line is where it belongs, because by this point the
/// encode has already succeeded and the work is not lost, merely undelivered.
pub fn download(bytes: &[u8], name: &str, mime: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or("there is no window to save from")?;
    let document = window.document().ok_or("there is no page to save from")?;

    // `Uint8Array::from` copies into the JS heap. Unavoidable: the blob has to
    // live outside this wasm instance's memory for the browser to read it, and
    // for a moment the stack is therefore in memory twice. That is the cost of
    // this route, and the reason the URL is revoked rather than left.
    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array);
    let options = web_sys::BlobPropertyBag::new();
    // Stated, unlike in `web_open`, where the only reader was our own `fetch`.
    // Here the reader is the browser's download manager, which may use it to
    // decide what it has.
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(|e| format!("could not wrap the file for download: {e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("could not make a URL for the file: {e:?}"))?;

    let anchor = document
        .create_element("a")
        .ok()
        .and_then(|e| e.dyn_into::<web_sys::HtmlAnchorElement>().ok());
    let Some(anchor) = anchor else {
        // The URL is dead the moment this returns, and nothing else will ever
        // reach it.
        let _ = web_sys::Url::revoke_object_url(&url);
        return Err("could not make a download link".to_string());
    };
    anchor.set_href(&url);
    anchor.set_download(name);

    // Into the document and straight back out. Not every browser follows a
    // click on an element that was never in the page, and leaving it there
    // would put an invisible link in the body for each save.
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    } else {
        anchor.click();
    }

    revoke_later(&window, url);
    Ok(())
}

/// Release the object URL once the download has had time to start.
///
/// A failure to *schedule* the revoke is not reported: the download itself has
/// already been handed over, and the only consequence is that one blob is held
/// until the tab closes. Telling the user their file did not save, when it did,
/// would be the worse of the two outcomes.
fn revoke_later(window: &web_sys::Window, url: String) {
    let revoke = wasm_bindgen::closure::Closure::once_into_js(move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
        revoke.unchecked_ref(),
        REVOKE_AFTER_MS,
    );
}

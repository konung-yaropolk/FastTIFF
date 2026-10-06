//! Handing an image to a second tab — the browser's answer to
//! [`crate::process`].
//!
//! FastTIFF is one stack per window, and on the desktop that means one stack
//! per *process*: a plugin result opens by re-launching the binary and handing
//! the encoded TIFF over on its stdin (see [`crate::process`]). A browser has
//! no processes to launch, but it has tabs, and a tab is the same thing for
//! this purpose — a second instance of the app with its own document.
//!
//! # How the bytes get across
//!
//! The result is a `Vec<u8>` in wasm memory and the new tab is a separate wasm
//! instance with its own. The route is a blob:
//!
//!   1. the opener wraps the bytes in a `Blob` and takes an object URL for it;
//!   2. it opens `<this page>?open=<that url>&name=<the title>`;
//!   3. the child reads the query, `fetch`es the URL, and loads the bytes.
//!
//! An object URL is same-origin and readable from the opened tab, and it stays
//! valid while the document that created it lives — so the opener does not
//! revoke it; the child does, once it has the bytes. Nothing is copied through
//! a string, so size is bounded by memory rather than by a quota: the
//! alternatives were `sessionStorage` (strings, and about five megabytes, where
//! these stacks are tens) and `postMessage` (which works, but needs the child
//! to announce itself first and so adds a handshake to go wrong).
//!
//! # Popups
//!
//! `window.open` is refused unless the browser thinks a user asked for it.
//! That is why the plugin runs *synchronously* on this target: the run happens
//! in the same task as the click that started it, so the activation is still
//! current when this is called. When a browser refuses anyway, `window.open`
//! answers `null` rather than throwing — this reports that as an error, and
//! the caller then shows the result in the current tab instead
//! (`PluginProduct::Inline`), which is the path that already existed for a
//! desktop handover that could not go through.

use wasm_bindgen::JsCast;

/// The query parameter carrying the object URL.
const PARAM_URL: &str = "open";
/// The query parameter carrying what to call the document.
const PARAM_NAME: &str = "name";

/// Open a new tab showing `bytes`, titled `name`.
///
/// `Err` is "it did not open" — a blocked popup, or no `window` at all — and
/// is a thing to report rather than a thing to retry.
pub fn open_bytes_in_new_tab(bytes: &[u8], name: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or("there is no window to open from")?;

    // `Uint8Array::from` copies into the JS heap. That is the one unavoidable
    // copy on this path: the blob has to live outside this wasm instance's
    // memory, because the instance that reads it is a different one.
    let bytes = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&bytes);
    // No `BlobPropertyBag`: the type would only matter to something that
    // sniffs it, and the only reader is our own `fetch` below.
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts)
        .map_err(|e| format!("could not wrap the result for the new tab: {e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("could not make a URL for the result: {e:?}"))?;

    // The same page, with the result named in the query. `pathname` rather
    // than the whole href so an existing query of our own is not carried over.
    let path = window
        .location()
        .pathname()
        .map_err(|e| format!("could not read this page's address: {e:?}"))?;
    let target = format!(
        "{path}?{PARAM_URL}={}&{PARAM_NAME}={}",
        String::from(js_sys::encode_uri_component(&url)),
        String::from(js_sys::encode_uri_component(name)),
    );

    match window.open_with_url_and_target(&target, "_blank") {
        Ok(Some(_)) => Ok(()),
        // Not an exception — this is what a blocked popup looks like.
        Ok(None) => Err("the browser blocked the new tab".to_string()),
        Err(e) => Err(format!("the new tab could not be opened: {e:?}")),
    }
}

/// The object URL and title this tab was opened with, if it was.
fn pending() -> Option<(String, String)> {
    let window = web_sys::window()?;
    let search = window.location().search().ok()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    let url = params.get(PARAM_URL)?;
    if url.is_empty() {
        return None;
    }
    let name = params
        .get(PARAM_NAME)
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| "result.tif".to_string());
    Some((url, name))
}

/// Take the document this tab was opened to show, if there is one.
///
/// `None` for an ordinary visit, which is what makes this safe to call
/// unconditionally at start-up. Any failure also answers `None`: a tab that
/// cannot fetch its result should open empty, the way it would have if nobody
/// had passed it one, rather than refuse to start.
pub async fn take_pending() -> Option<(Vec<u8>, String)> {
    let (url, name) = pending()?;
    let window = web_sys::window()?;

    let response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&url))
        .await
        .ok()?;
    let response: web_sys::Response = response.dyn_into().ok()?;
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().ok()?)
        .await
        .ok()?;
    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();

    // The blob has been read, so let it go — the opener deliberately does not,
    // because it cannot know when the child is done.
    let _ = web_sys::Url::revoke_object_url(&url);
    // And take the query off the address bar. Without this, a reload would try
    // to fetch a URL that has just been revoked and open an empty tab with no
    // explanation; afterwards a reload simply opens an empty tab, which is
    // what reloading a tab whose document came from memory has to mean.
    if let Ok(history) = window.history() {
        if let Ok(path) = window.location().pathname() {
            let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&path));
        }
    }

    (!bytes.is_empty()).then_some((bytes, name))
}

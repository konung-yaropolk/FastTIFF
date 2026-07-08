//! macOS "Open With" / double-click file handling.
//!
//! On macOS, opening a document from Finder (double-click, drag-onto-icon, or
//! "Open With") does NOT put the path in `argv` the way Windows and Linux do —
//! it's delivered to the app as a `kAEOpenDocuments` Apple Event. winit 0.30
//! doesn't surface that event, so without this the app launches but never sees
//! the file (an empty window).
//!
//! We install a Carbon Apple Event Manager handler (still supported on macOS 14)
//! for `kCoreEventClass` / `kAEOpenDocuments`, pull the file paths out of the
//! event, and queue them. The egui update loop drains the queue and opens them
//! through the same `process::open_all` path as argv / drag-drop. No
//! Objective-C class or extra crate is needed — just a C callback and a few
//! framework calls.
//!
//! `install()` is called both before the event loop starts (to catch the
//! cold-launch event that's already queued) and again once the app is up (in
//! case AppKit's own launch-time handler install clobbered the first one).

use std::ffi::{c_void, OsString};
use std::os::raw::c_long;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

// --- Apple Event Manager FFI (CoreServices.framework) --------------------

/// `AEDesc` / `AppleEvent` / `AEDescList` are all the same opaque descriptor
/// struct in the Apple Event Manager.
#[repr(C)]
struct AEDesc {
    descriptor_type: u32,
    data_handle: *mut c_void,
}

impl AEDesc {
    const fn null() -> Self {
        AEDesc { descriptor_type: 0, data_handle: std::ptr::null_mut() }
    }
}

/// `AEEventHandlerUPP`: on 64-bit macOS a UPP is just the function pointer, so
/// the handler can be passed straight to `AEInstallEventHandler`.
type AEEventHandlerProc =
    extern "C" fn(event: *const AEDesc, reply: *mut AEDesc, refcon: *mut c_void) -> i16;

#[link(name = "CoreServices", kind = "framework")]
extern "C" {
    fn AEInstallEventHandler(
        event_class: u32,
        event_id: u32,
        handler: AEEventHandlerProc,
        refcon: *mut c_void,
        is_sys_handler: u8,
    ) -> i16;

    fn AEGetParamDesc(apple_event: *const AEDesc, keyword: u32, desired_type: u32, result: *mut AEDesc) -> i16;

    fn AECountItems(list: *const AEDesc, count: *mut c_long) -> i16;

    #[allow(clippy::too_many_arguments)]
    fn AEGetNthPtr(
        list: *const AEDesc,
        index: c_long,
        desired_type: u32,
        keyword: *mut u32,
        type_code: *mut u32,
        data_ptr: *mut c_void,
        maximum_size: c_long,
        actual_size: *mut c_long,
    ) -> i16;

    fn AEDisposeDesc(desc: *mut AEDesc) -> i16;
}

/// Pack four ASCII bytes into a big-endian `FourCharCode`.
const fn fourcc(code: &[u8; 4]) -> u32 {
    ((code[0] as u32) << 24) | ((code[1] as u32) << 16) | ((code[2] as u32) << 8) | (code[3] as u32)
}

// --- state ---------------------------------------------------------------

/// Paths delivered by "Open With" / double-click, waiting for the update loop.
fn queue() -> &'static Mutex<Vec<PathBuf>> {
    static Q: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    Q.get_or_init(|| Mutex::new(Vec::new()))
}

/// The egui context, stashed once the app is up, so a handler that fires while
/// the app is idle can wake the event loop to drain the queue.
fn ctx_slot() -> &'static OnceLock<egui::Context> {
    static C: OnceLock<egui::Context> = OnceLock::new();
    &C
}

/// Remember the egui context (and (re)install the handler now that AppKit has
/// finished its own launch-time setup). Call once, from the eframe creator.
pub fn set_ctx(ctx: egui::Context) {
    let _ = ctx_slot().set(ctx);
    install();
}

/// Drain any paths delivered since the last call.
pub fn take_opened_files() -> Vec<PathBuf> {
    match queue().lock() {
        Ok(mut q) => std::mem::take(&mut *q),
        Err(_) => Vec::new(),
    }
}

/// Install the `kAEOpenDocuments` handler. Idempotent — installing the same
/// class/id/handler just replaces it, so calling twice is harmless.
pub fn install() {
    // SAFETY: a plain Apple Event Manager registration; the handler is a
    // 'static `extern "C"` fn, and no arguments outlive the call.
    let err = unsafe {
        AEInstallEventHandler(fourcc(b"aevt"), fourcc(b"odoc"), handle_open_documents, std::ptr::null_mut(), 0)
    };
    if err != 0 {
        log::error!("macOS: failed to install open-file handler (AE error {err})");
    }
}

/// C callback: runs on the main thread when Finder asks us to open documents.
extern "C" fn handle_open_documents(event: *const AEDesc, _reply: *mut AEDesc, _refcon: *mut c_void) -> i16 {
    // SAFETY: `event` is a valid AppleEvent for the lifetime of this call.
    let paths = unsafe { extract_paths(event) };
    if !paths.is_empty() {
        if let Ok(mut q) = queue().lock() {
            q.extend(paths);
        }
        // Wake the UI so `take_opened_files` runs even if the app was idle.
        if let Some(ctx) = ctx_slot().get() {
            ctx.request_repaint();
        }
    }
    0 // noErr
}

/// Pull the file list out of a `kAEOpenDocuments` event as filesystem paths.
///
/// # Safety
/// `event` must be a valid, non-null pointer to the AppleEvent passed to the
/// handler (or null, which yields an empty result).
unsafe fn extract_paths(event: *const AEDesc) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if event.is_null() {
        return out;
    }

    // keyDirectObject '----', coerced to a descriptor list 'list'.
    let mut list = AEDesc::null();
    if AEGetParamDesc(event, fourcc(b"----"), fourcc(b"list"), &mut list) != 0 {
        return out;
    }

    let mut count: c_long = 0;
    if AECountItems(&list, &mut count) == 0 {
        for i in 1..=count {
            // Each item as a file URL ('furl'): UTF-8 bytes of a file:// URL.
            let mut buf = [0u8; 4096];
            let mut keyword: u32 = 0;
            let mut type_code: u32 = 0;
            let mut actual: c_long = 0;
            let err = AEGetNthPtr(
                &list,
                i,
                fourcc(b"furl"),
                &mut keyword,
                &mut type_code,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as c_long,
                &mut actual,
            );
            if err == 0 && actual > 0 && (actual as usize) <= buf.len() {
                if let Some(path) = file_url_to_path(&buf[..actual as usize]) {
                    out.push(path);
                }
            }
        }
    }

    AEDisposeDesc(&mut list);
    out
}

/// Convert the UTF-8 bytes of a `file://` URL to a filesystem path, undoing
/// percent-encoding. `None` if the bytes aren't a usable local file URL.
fn file_url_to_path(bytes: &[u8]) -> Option<PathBuf> {
    let s = std::str::from_utf8(bytes).ok()?;
    // Strip scheme + authority: the path starts at the first '/' after "file://"
    // ("file:///p" -> "/p"; "file://localhost/p" -> "/p").
    let rest = s.strip_prefix("file://")?;
    let path = &rest[rest.find('/')?..];
    Some(PathBuf::from(OsString::from_vec(percent_decode(path.as_bytes()))))
}

/// Decode `%XX` escapes in a URL path. `+` is left as-is (file URLs don't use
/// form encoding); a malformed `%` escape is passed through literally.
fn percent_decode(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(hi << 4 | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_path() {
        assert_eq!(file_url_to_path(b"file:///Users/me/scan.tif"), Some(PathBuf::from("/Users/me/scan.tif")));
    }

    #[test]
    fn percent_encoded_spaces_and_unicode() {
        // "/Users/me/My Scan é.tif"
        let url = b"file:///Users/me/My%20Scan%20%C3%A9.tif";
        assert_eq!(file_url_to_path(url), Some(PathBuf::from("/Users/me/My Scan é.tif")));
    }

    #[test]
    fn localhost_authority_is_stripped() {
        assert_eq!(file_url_to_path(b"file://localhost/tmp/a.tiff"), Some(PathBuf::from("/tmp/a.tiff")));
    }

    #[test]
    fn non_file_url_rejected() {
        assert_eq!(file_url_to_path(b"http://example.com/a.tif"), None);
    }

    #[test]
    fn malformed_escape_is_literal() {
        assert_eq!(percent_decode(b"a%2z%"), b"a%2z%");
    }
}

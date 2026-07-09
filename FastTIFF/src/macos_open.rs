//! macOS "Open With" / double-click file handling.
//!
//! On macOS, opening a document from Finder (double-click, drag-onto-icon, or
//! "Open With") does NOT put the path in `argv` the way Windows and Linux do —
//! it's delivered as a `kAEOpenDocuments` Apple Event. winit 0.30 doesn't
//! surface that event, so without this the app launches but never sees the
//! file (an empty window).
//!
//! Two delivery windows need covering, and they need different mechanisms:
//!
//! * **Cold launch** (double-click starts the app): AppKit delivers the open
//!   event *between* `applicationWillFinishLaunching` and
//!   `applicationDidFinishLaunching`, through its own Apple Event handler,
//!   which forwards to the app delegate's `application:openURLs:`. winit's
//!   delegate doesn't implement that selector, so stock AppKit replies
//!   "not handled" — Finder then shows a "does not support this file type"
//!   dialog and the app opens empty. Any handler we install before `run` is
//!   clobbered when AppKit installs its own during `finishLaunching`, and
//!   anything we install from the eframe creator (= `didFinishLaunching`, via
//!   winit's `resumed`) runs *after* the event was already dispatched. The fix:
//!   observe `NSApplicationWillFinishLaunchingNotification` (fires after
//!   winit's delegate exists, before the event dispatch) and inject an
//!   `application:openURLs:` method onto the delegate's class with
//!   `class_addMethod`, so AppKit's own handler delivers the URLs to us and
//!   replies success.
//!
//! * **App already running** (warm open): `set_ctx` — called from the eframe
//!   creator — installs a classic Carbon Apple Event handler for
//!   `kCoreEventClass`/`kAEOpenDocuments`, replacing AppKit's dispatch entry,
//!   so later open events come straight to us. (The injected delegate method
//!   would cover this too if the Carbon install ever failed — belt and
//!   suspenders.)
//!
//! Both paths queue the paths here; the egui update loop drains the queue and
//! opens them through the same `process::open_all` path as argv / drag-drop.
//! No Objective-C class of our own and no extra crates — just C FFI into
//! CoreServices (Apple Event Manager), CoreFoundation (notification center),
//! and libobjc (runtime).

use std::ffi::{c_void, CStr, OsString};
use std::os::raw::{c_char, c_long};
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

// --- CoreFoundation notification FFI --------------------------------------

type CFNotificationCallback = extern "C" fn(
    center: *mut c_void,
    observer: *mut c_void,
    name: *const c_void,
    object: *const c_void,
    user_info: *const c_void,
);

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFNotificationCenterGetLocalCenter() -> *mut c_void;
    fn CFNotificationCenterAddObserver(
        center: *mut c_void,
        observer: *const c_void,
        callback: CFNotificationCallback,
        name: *const c_void,
        object: *const c_void,
        suspension_behavior: isize,
    );
    fn CFStringCreateWithCString(alloc: *const c_void, c_str: *const c_char, encoding: u32) -> *const c_void;
}

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const CF_NOTIFICATION_DELIVER_IMMEDIATELY: isize = 4;

// --- Objective-C runtime FFI (libobjc) -------------------------------------

#[link(name = "objc")]
extern "C" {
    /// Untyped; cast per call site to the exact signature before invoking (the
    /// ABI Apple documents for `objc_msgSend` dispatch from C).
    fn objc_msgSend();
    fn objc_getClass(name: *const c_char) -> *mut c_void;
    fn object_getClass(obj: *mut c_void) -> *mut c_void;
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn class_addMethod(cls: *mut c_void, name: *const c_void, imp: *mut c_void, types: *const c_char) -> u8;
}

/// `[obj sel]` returning an object pointer.
///
/// # Safety
/// `obj` must be a valid Objective-C object (or class) and `sel` a selector it
/// responds to with a `()`-args, object-return signature.
unsafe fn msg_obj(obj: *mut c_void, sel: *const c_void) -> *mut c_void {
    let f: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    f(obj, sel)
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

/// Queue opened paths and wake the UI so `take_opened_files` runs promptly.
/// (During cold launch the context isn't stashed yet — that's fine, the first
/// frames render unconditionally and drain the queue.)
fn deliver(paths: Vec<PathBuf>) {
    if paths.is_empty() {
        return;
    }
    if let Ok(mut q) = queue().lock() {
        q.extend(paths);
    }
    if let Some(ctx) = ctx_slot().get() {
        ctx.request_repaint();
    }
}

/// Remember the egui context (and install the Carbon handler now that AppKit
/// has finished its own launch-time setup — this replaces AppKit's dispatch
/// entry, covering warm opens). Call once, from the eframe creator.
pub fn set_ctx(ctx: egui::Context) {
    let _ = ctx_slot().set(ctx);
    install_ae_handler();
}

/// Drain any paths delivered since the last call.
pub fn take_opened_files() -> Vec<PathBuf> {
    match queue().lock() {
        Ok(mut q) => std::mem::take(&mut *q),
        Err(_) => Vec::new(),
    }
}

/// Launch-time setup; call from `main` before the event loop starts. Registers
/// the `willFinishLaunching` observer that injects `application:openURLs:` onto
/// winit's app delegate — the piece that catches the *cold-launch* open event
/// (see the module docs for why nothing later works for that one).
pub fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        let name = CFStringCreateWithCString(
            std::ptr::null(),
            c"NSApplicationWillFinishLaunchingNotification".as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if name.is_null() {
            log::error!("macOS: couldn't create launch-notification name");
            return;
        }
        // Observer `NULL` = non-removable, which is fine: the notification
        // fires exactly once per process lifetime.
        CFNotificationCenterAddObserver(
            CFNotificationCenterGetLocalCenter(),
            std::ptr::null(),
            on_will_finish_launching,
            name,
            std::ptr::null(),
            CF_NOTIFICATION_DELIVER_IMMEDIATELY,
        );
    });
}

/// Fires after winit created `NSApplication` and set its delegate, but before
/// AppKit dispatches the queued open-documents event — the one moment where
/// adding the delegate method helps the cold-launch case.
extern "C" fn on_will_finish_launching(
    _center: *mut c_void,
    _observer: *mut c_void,
    _name: *const c_void,
    _object: *const c_void,
    _user_info: *const c_void,
) {
    // SAFETY: called on the main thread by the local notification center while
    // NSApplication exists; all pointers checked before use.
    unsafe { inject_open_urls_method() };
}

/// Add `application:openURLs:` to winit's application-delegate class so
/// AppKit's own open-documents handler delivers files to us (and replies
/// success — no "does not support this file type" dialog).
///
/// # Safety
/// Must run on the main thread with `NSApplication` initialized.
unsafe fn inject_open_urls_method() {
    let app_cls = objc_getClass(c"NSApplication".as_ptr());
    if app_cls.is_null() {
        log::error!("macOS: NSApplication class not found");
        return;
    }
    let app = msg_obj(app_cls, sel_registerName(c"sharedApplication".as_ptr()));
    let delegate = if app.is_null() {
        std::ptr::null_mut()
    } else {
        msg_obj(app, sel_registerName(c"delegate".as_ptr()))
    };
    if delegate.is_null() {
        log::error!("macOS: no app delegate at willFinishLaunching; cold-launch open won't work");
        return;
    }
    let cls = object_getClass(delegate);
    let added = class_addMethod(
        cls,
        sel_registerName(c"application:openURLs:".as_ptr()),
        handle_open_urls as *mut c_void,
        c"v@:@@".as_ptr(), // void (id self, SEL _cmd, id app, id urls)
    );
    if added == 0 {
        // The delegate (a future winit?) already implements it; leave theirs.
        log::warn!("macOS: app delegate already implements application:openURLs:");
    }
}

/// The injected `application:openURLs:` implementation. `urls` is an
/// `NSArray<NSURL *>`.
extern "C" fn handle_open_urls(_this: *mut c_void, _cmd: *const c_void, _app: *mut c_void, urls: *mut c_void) {
    // SAFETY: AppKit passes a valid NSArray for the duration of this call.
    deliver(unsafe { ns_urls_to_paths(urls) });
}

/// Read an `NSArray<NSURL *>` into filesystem paths via each URL's
/// `fileSystemRepresentation` (bytes are copied out immediately, before the
/// autorelease pool can reclaim them).
///
/// # Safety
/// `urls` must be a valid `NSArray<NSURL *>` (or null, which yields empty).
unsafe fn ns_urls_to_paths(urls: *mut c_void) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if urls.is_null() {
        return out;
    }
    let count: unsafe extern "C" fn(*mut c_void, *const c_void) -> usize =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let at_index: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> *mut c_void =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let fs_repr: unsafe extern "C" fn(*mut c_void, *const c_void) -> *const c_char =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

    let n = count(urls, sel_registerName(c"count".as_ptr()));
    for i in 0..n {
        let url = at_index(urls, sel_registerName(c"objectAtIndex:".as_ptr()), i);
        if url.is_null() {
            continue;
        }
        let repr = fs_repr(url, sel_registerName(c"fileSystemRepresentation".as_ptr()));
        if repr.is_null() {
            continue;
        }
        let bytes = CStr::from_ptr(repr).to_bytes().to_vec();
        out.push(PathBuf::from(OsString::from_vec(bytes)));
    }
    out
}

// --- Carbon Apple Event handler (warm opens) -------------------------------

/// Install the `kAEOpenDocuments` handler, replacing whatever entry (ours or
/// AppKit's) currently holds that class/id. Idempotent.
fn install_ae_handler() {
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
    deliver(unsafe { extract_paths(event) });
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

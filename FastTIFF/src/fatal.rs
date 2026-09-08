//! Making a failed start visible on Windows.
//!
//! `#![windows_subsystem = "windows"]` detaches the process from a console, so
//! it has no stderr. Anything Rust would normally print there — the `Err` a
//! `main` returning `Result` reports, a panic message, a backtrace — is written
//! to a handle that goes nowhere. The process exits with a non-zero code and
//! not one character reaches the user.
//!
//! That is how a wgpu build on Windows 7 presents: no adapter, `run_native`
//! returns an error, and the program vanishes on launch with no dialog, no log
//! and nothing in Event Viewer. Diagnosing it needs a debugger or an import
//! dump, neither of which is a reasonable thing to ask of someone who just
//! wants to open a TIFF.
//!
//! A message box is the one output channel a windowed process always has, so
//! this puts the text there as well. Cheap, and it turns "it does not start"
//! into a sentence naming the reason.
//!
//! Only on Windows: everywhere else stderr is a real stream and the default
//! behaviour is already right.

use std::ffi::c_void;

#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(hwnd: *mut c_void, text: *const u16, caption: *const u16, u_type: u32) -> i32;
}

const MB_OK: u32 = 0x0000_0000;
const MB_ICONERROR: u32 = 0x0000_0010;
/// Put the box in front of whatever else is on screen. Without it a dialog from
/// a process that never got a window of its own can open behind everything.
const MB_TOPMOST: u32 = 0x0004_0000;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Show `body` under `title` and wait for the user to dismiss it.
pub fn show(title: &str, body: &str) {
    // The box is modal to nothing (`hwnd` null) on purpose: it is used when
    // there is no window yet, or when the one there is is about to die with the
    // process.
    let (body, title) = (wide(body), wide(title));
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            body.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONERROR | MB_TOPMOST,
        );
    }
}

/// What to suggest when startup fails, which on an older machine it usually
/// does for one reason.
///
/// Named rather than inlined so the error path and the panic path say the same
/// thing: a user reading one has no way to know the other exists.
pub const GPU_HINT: &str = "This build needs a GPU backend the system can \
     provide. The default Windows build uses wgpu (Direct3D 12 or Vulkan); \
     Windows 7 has neither, and needs the OpenGL build instead.";

/// Route panics to a message box as well as to stderr.
///
/// Chains rather than replaces: the default hook still runs, so a build started
/// from a console — or with a debugger attached — keeps the full message and
/// backtrace it would otherwise print. This only adds the copy that a windowed
/// process can actually show.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        previous(info);
        show("FastTIFF stopped unexpectedly", &info.to_string());
    }));
}

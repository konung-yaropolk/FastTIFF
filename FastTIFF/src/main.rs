#![windows_subsystem = "windows"]

// The UI lives in the library half of this crate so the web build can reuse
// it; see `src/lib.rs`. This file is only the native host.
use fasttiff::{app, render};
#[cfg(target_os = "macos")]
use fasttiff::macos_open;
use fasttiff::process;

/// The PNG baked into the binary for the application icon.
///
/// Bigger on macOS because that is the one platform where this image is drawn at
/// size: it becomes the Dock icon (see [`window_icon`]), which is rendered far
/// larger than any title bar or task switcher asks for. Everywhere else the OS
/// only ever scales it down.
#[cfg(target_os = "macos")]
const ICON_PNG: &[u8] = include_bytes!("../icon/icon512.png");
#[cfg(not(target_os = "macos"))]
const ICON_PNG: &[u8] = include_bytes!("../icon/icon256.png");

/// Decode [`ICON_PNG`] into the RGBA image `ViewportBuilder::with_icon` wants.
///
/// Baked into the binary, so it needs no icon file at runtime; the OS scales it
/// for each context (title bar, taskbar, alt-tab, Dock). `None` if the embedded
/// PNG somehow fails to decode, which only costs the icon.
fn window_icon() -> Option<egui::IconData> {
    let image = image::load_from_memory(ICON_PNG).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData { rgba: image.into_raw(), width, height })
}

fn main() -> eframe::Result {
    env_logger::init();

    // macOS delivers "Open With" / double-clicked files as an Apple Event, not
    // argv. Register the launch observer that hooks the open-documents event
    // during AppKit's launch sequence (the cold-launch open fires before the
    // eframe creator runs, so it must be armed here). The app's update loop
    // drains what it queues. See `macos_open` for the full timing story.
    #[cfg(target_os = "macos")]
    macos_open::install();

    // On Linux, default to winit's X11 backend so file drag-and-drop works:
    // winit's Wayland backend doesn't reliably deliver file drops (notably on
    // KDE), and running under XWayland costs nothing here. Override by setting
    // WINIT_UNIX_BACKEND=wayland to force native Wayland.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WINIT_UNIX_BACKEND").is_none() {
        std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    }

    // argv[0] is the program path itself; argv[1..] are the files passed when
    // the program is launched via "Open with", a file association, or files
    // dragged onto the .exe / its shortcut. Selecting several files at once
    // passes them all to a single invocation — open the first here and launch
    // each of the rest in its own process so they all appear side by side.
    let files: Vec<std::path::PathBuf> =
        std::env::args_os().skip(1).map(std::path::PathBuf::from).collect();
    let initial_path = process::open_all(&files).cloned();

    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([320.0, 320.0])
        // Keep in sync with `app::MIN_WINDOW` — the floor for both manual
        // resizing and zoom-out (which letterboxes below this size).
        .with_min_inner_size([256.0, 256.0])
        .with_title("FastTIFF");
    // Set the application icon from the bundled PNG, so it isn't the egui
    // default. On Windows and Linux/X11 this is the per-window icon winit sets
    // — title bar, taskbar, task switcher.
    //
    // On macOS winit ignores per-window icons, so it is tempting to skip this
    // there and let the .app bundle's .icns speak for itself. That is exactly
    // what put the egui logo in the Dock: eframe substitutes *its own* default
    // icon when the viewport carries none, then installs it at runtime with
    // `NSApplication::setApplicationIconImage:`, which overrides whatever the
    // bundle declared. Passing ours means the Dock gets ours. (Leaving the
    // bundle to it would mean passing `IconData::default()`, which eframe reads
    // as "no icon" and skips — but that only works for a bundled build, and
    // leaves a bare `cargo run` iconless.)
    let viewport = match window_icon() {
        Some(icon) => viewport.with_icon(std::sync::Arc::new(icon)),
        None => viewport,
    };

    let mut native_options = eframe::NativeOptions {
        viewport,
        // glow or wgpu, picked at compile time by the `renderer-*` features.
        renderer: render::RENDERER,
        ..Default::default()
    };
    // Backend-specific option tweaks (wgpu: request the optional 16-bit-norm
    // texture feature for full-precision volume textures; glow: no-op).
    render::tune_native_options(&mut native_options);

    eframe::run_native(
        "FastTIFF",
        native_options,
        Box::new(|cc| {
            // Now that the event loop is up, hand the macOS open-file machinery
            // the egui context (so it can wake an idle UI) and install the Apple
            // Event handler that covers opens while the app is already running.
            #[cfg(target_os = "macos")]
            macos_open::set_ctx(cc.egui_ctx.clone());
            // Theme + interface scale, shared with the web host so the two
            // cannot drift.
            fasttiff::install_chrome(&cc.egui_ctx);
            let render = render::init(cc);
            Ok(Box::new(app::ViewerApp::new(initial_path, render)))
        }),
    )
}
#[cfg(test)]
mod icon_tests {
    use super::*;

    /// The icon this build embeds has to actually decode, or the window and the
    /// Dock fall back to eframe's own logo — which is the bug this replaced.
    #[test]
    fn the_embedded_icon_decodes() {
        let icon = window_icon().expect("the embedded icon should decode");
        assert!(icon.width >= 256 && icon.height >= 256, "{}x{}", icon.width, icon.height);
        assert_eq!(icon.width, icon.height, "an app icon should be square");
        assert_eq!(
            icon.rgba.len(),
            icon.width as usize * icon.height as usize * 4,
            "RGBA buffer does not match the stated size"
        );
        assert!(icon.rgba.iter().any(|&b| b != 0), "the icon decoded to nothing but zeroes");
    }

    /// Both candidates, whichever this platform compiled.
    ///
    /// macOS embeds the 512 and every other platform the 256, so on any one
    /// machine only one of them is reachable through `window_icon` — and a
    /// missing or corrupt file for the *other* would not fail the build until
    /// someone released from that platform. Checking the bytes directly covers
    /// the macOS icon from a Windows or Linux checkout, which is the only place
    /// the Dock bug could come back unseen.
    #[test]
    fn both_platform_icons_are_present_and_sound() {
        for (name, bytes, side) in [
            ("icon256.png", &include_bytes!("../icon/icon256.png")[..], 256u32),
            ("icon512.png", &include_bytes!("../icon/icon512.png")[..], 512),
        ] {
            let image = image::load_from_memory(bytes)
                .unwrap_or_else(|e| panic!("{name} does not decode: {e}"))
                .into_rgba8();
            assert_eq!(image.dimensions(), (side, side), "{name} is not {side}x{side}");
            assert!(
                image.pixels().any(|p| p.0[3] != 0),
                "{name} is fully transparent, so it would show as nothing at all"
            );
        }
    }
}

#![windows_subsystem = "windows"]

mod app;
mod colormap;
#[cfg(target_os = "macos")]
mod macos_open;
mod prefetch;
mod process;
mod render;
mod volume;

/// Decode the bundled 256×256 PNG into the RGBA image `ViewportBuilder::with_icon`
/// wants. Baked into the binary with `include_bytes!`, so it needs no icon file
/// at runtime; winit / the OS downscales it for each context (title bar, taskbar,
/// alt-tab). Returns `None` if the embedded PNG somehow fails to decode, in which
/// case the window just falls back to the default icon.
fn window_icon() -> Option<egui::IconData> {
    let image = image::load_from_memory(include_bytes!("../icon/icon256.png")).ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData { rgba: image.into_raw(), width, height })
}

fn main() -> eframe::Result {
    env_logger::init();

    // macOS delivers "Open With" / double-clicked files as an Apple Event, not
    // argv — install the handler before the event loop so the cold-launch open
    // is captured. The app's update loop drains what it queues. See `macos_open`.
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

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([320.0, 320.0])
        // Keep in sync with `app::MIN_WINDOW` — the floor for both manual
        // resizing and zoom-out (which letterboxes below this size).
        .with_min_inner_size([256.0, 256.0])
        .with_title("FastTIFF");
    // Set the window icon (title bar / taskbar / task switcher) from the bundled
    // PNG, so it isn't the egui default. On macOS the Dock icon comes from the
    // .app bundle's .icns instead (winit ignores per-window icons there), so this
    // is effectively a no-op on macOS but fixes Windows and Linux/X11.
    if let Some(icon) = window_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

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
            // Now that the event loop is up, hand the macOS open-file handler the
            // egui context (so it can wake an idle UI) and re-assert the handler
            // in case AppKit's own launch-time install replaced ours.
            #[cfg(target_os = "macos")]
            macos_open::set_ctx(cc.egui_ctx.clone());
            let render = render::init(cc);
            Ok(Box::new(app::ViewerApp::new(initial_path, render)))
        }),
    )
}
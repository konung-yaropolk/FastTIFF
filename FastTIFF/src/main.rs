#![windows_subsystem = "windows"]

mod app;
mod prefetch;
mod process;
mod render;

fn main() -> eframe::Result {
    env_logger::init();

    // argv[0] is the program path itself; argv[1..] are the files passed when
    // the program is launched via "Open with", a file association, or files
    // dragged onto the .exe / its shortcut. Selecting several files at once
    // passes them all to a single invocation — open the first here and launch
    // each of the rest in its own process so they all appear side by side.
    let files: Vec<std::path::PathBuf> =
        std::env::args_os().skip(1).map(std::path::PathBuf::from).collect();
    let initial_path = process::open_all(&files).cloned();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([320.0, 320.0])
            // Keep in sync with `app::MIN_WINDOW` — the floor for both manual
            // resizing and zoom-out (which letterboxes below this size).
            .with_min_inner_size([256.0, 256.0])
            .with_title("FastTIFF"),
        // glow or wgpu, picked at compile time by the `renderer-*` features.
        renderer: render::RENDERER,
        ..Default::default()
    };

    eframe::run_native(
        "FastTIFF",
        native_options,
        Box::new(|cc| {
            let render = render::init(cc);
            Ok(Box::new(app::ViewerApp::new(initial_path, render)))
        }),
    )
}
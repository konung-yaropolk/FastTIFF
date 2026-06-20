#![windows_subsystem = "windows"]

mod app;
mod render;

fn main() -> eframe::Result {
    env_logger::init();

    // argv[0] is the program path itself; argv[1], if present, is the file
    // Windows passes when the program is launched via "Open with", a file
    // association, or a file dragged onto the .exe / its shortcut.
    let initial_path = std::env::args_os().nth(1).map(std::path::PathBuf::from);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([512.0, 512.0])
            .with_title("FastTIFF"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "FastTIFF",
        native_options,
        Box::new(|cc| {
            let render_state = cc
                .wgpu_render_state
                .as_ref()
                .expect("FastTIFF requires the wgpu backend (NativeOptions::renderer = Wgpu)");
            render::pipeline::install(render_state);
            Ok(Box::new(app::ViewerApp::new(initial_path)))
        }),
    )
}
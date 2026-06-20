#![windows_subsystem = "windows"]

mod app;
mod render;

fn main() -> eframe::Result {
    env_logger::init();

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
                .expect("FastTIFFrequires the wgpu backend (NativeOptions::renderer = Wgpu)");
            render::pipeline::install(render_state);
            Ok(Box::new(app::ViewerApp::new()))
        }),
    )
}

//! The pop-up windows: 3D render settings and the file-metadata viewer.
//! Split from `app.rs`.

use super::*;

use super::camera::NavMode;
use crate::render;
use egui::RichText;

/// The 3D render-settings pop-up: rendering method (+ alpha density), per-axis
/// voxel scale (x:y:z), interpolation, navigation style, and a camera-reset
/// button. Scale defaults to the stack's pixel calibration (with a button to
/// re-seed it). `reset_position` is set true when the user clicks Reset position.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_settings_window(
    ctx: &egui::Context,
    open: &mut bool,
    scale: &mut [f32; 3],
    interp: &mut render::VolumeInterp,
    nav: &mut NavMode,
    move_speed: &mut f32,
    scroll_speed: &mut f32,
    render_mode: &mut render::VolumeRender,
    density: &mut f32,
    reset_position: &mut bool,
    loaded: Option<&LoadedStack>,
) {
    egui::Window::new("3D render settings")
        .open(open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.label(RichText::new("Rendering").strong());
            ui.horizontal(|ui| {
                ui.selectable_value(render_mode, render::VolumeRender::Mip, "Max intensity")
                    .on_hover_text("Maximum-intensity projection: brightest sample along each ray");
                ui.selectable_value(render_mode, render::VolumeRender::Alpha, "Volume (ImageJ)")
                    .on_hover_text("ImageJ 3D Viewer style: translucent alpha-blended volume");
            });
            // Density only affects the alpha DVR — disabled for MIP.
            ui.add_enabled_ui(*render_mode == render::VolumeRender::Alpha, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Density");
                    ui.add(egui::Slider::new(density, 1.0..=1000.0).logarithmic(true))
                        .on_hover_text("Opacity of the alpha volume (higher = brighter/more solid)");
                });
            });

            ui.separator();
            ui.label(RichText::new("Voxel scale (x : y : z)").strong());
            ui.horizontal(|ui| {
                for (i, axis) in ["x", "y", "z"].iter().enumerate() {
                    ui.label(*axis);
                    ui.add(
                        egui::DragValue::new(&mut scale[i])
                            .speed(0.01)
                            .range(0.0001..=100_000.0)
                            .max_decimals(4),
                    );
                }
            });
            if let Some(loaded) = loaded {
                ui.horizontal(|ui| {
                    if ui
                        .button("Reset from metadata")
                        .on_hover_text("Re-seed x:y:z from the file's pixel calibration + spacing (else 1:1:1)")
                        .clicked()
                    {
                        *scale = loaded.tiff.meta.voxel_scale();
                    }
                    if let Some(unit) = loaded.tiff.meta.unit.as_deref().filter(|u| !u.is_empty()) {
                        ui.label(RichText::new(format!("unit: {unit}")).weak());
                    }
                });
            }

            ui.separator();
            ui.label(RichText::new("Interpolation").strong());
            ui.horizontal(|ui| {
                ui.selectable_value(interp, render::VolumeInterp::Nearest, "None (nearest)")
                    .on_hover_text("Crisp voxels, no smoothing");
                ui.selectable_value(interp, render::VolumeInterp::Linear, "Trilinear")
                    .on_hover_text("Smoothly interpolated samples");
                ui.selectable_value(interp, render::VolumeInterp::Cubic, "Cubic")
                    .on_hover_text("Tricubic B-spline: smoothest, but slower (8 taps/sample)");
            });

            ui.separator();
            ui.label(RichText::new("Navigation").strong());
            ui.horizontal_wrapped(|ui| {
                for mode in [NavMode::Cad, NavMode::Blender, NavMode::Maya, NavMode::WasdFly] {
                    ui.selectable_value(nav, mode, mode.label()).on_hover_text(mode.help());
                }
            });
            // Controls hint for the selected mode.
            ui.label(RichText::new(nav.help()).small().weak());
            // Speed multipliers on the built-in WASD / wheel base rates.
            // Logarithmic so 1.0 (the default) sits mid-track with symmetric
            // slower/faster range; the reset button restores both to 1×.
            ui.horizontal(|ui| {
                ui.label("Move speed");
                ui.add(egui::Slider::new(move_speed, 0.1..=10.0).logarithmic(true))
                    .on_hover_text("WASD / Space / Shift movement speed (× the default)");
            });
            ui.horizontal(|ui| {
                ui.label("Scroll speed");
                ui.add(egui::Slider::new(scroll_speed, 0.1..=10.0).logarithmic(true))
                    .on_hover_text("Mouse-wheel fly speed (× the default)");
                if ui.small_button("Reset").on_hover_text("Restore both speeds to 1×").clicked() {
                    *move_speed = 1.0;
                    *scroll_speed = 1.0;
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(RichText::new("Position").strong());
                if ui
                    .button("Reset position")
                    .on_hover_text("Recenter the camera to the default three-quarter view")
                    .clicked()
                {
                    *reset_position = true;
                }
            });
        });
}

pub(super) fn metadata_window(ctx: &egui::Context, open: &mut bool, loaded: &LoadedStack) {
    let tiff = &loaded.tiff;
    egui::Window::new("File metadata")
        .open(open)
        .resizable(true)
        .default_width(256.0)
        .vscroll(true)
        .show(ctx, |ui| {
            fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
                ui.label(RichText::new(k).strong());
                ui.label(v.into());
                ui.end_row();
            }

            ui.heading("File");
            egui::Grid::new("meta_file").num_columns(2).striped(true).show(ui, |ui| {
                kv(ui, "Size", human_bytes(tiff.mmap.len() as u64));
                let container = match tiff.flavor {
                    fast_tiff_lib::TiffFlavor::Classic => "classic TIFF",
                    fast_tiff_lib::TiffFlavor::Big => "BigTIFF",
                };
                kv(ui, "Container", container);
                let order = match tiff.byte_order {
                    fast_tiff_lib::ByteOrder::Little => "little-endian (II)",
                    fast_tiff_lib::ByteOrder::Big => "big-endian (MM)",
                };
                kv(ui, "Byte order", order);
                kv(ui, "Planes (IFDs)", tiff.frames.len().to_string());
            });

            if let Some(f) = tiff.frames.first() {
                ui.add_space(12.0);
                ui.heading("Frame format");
                egui::Grid::new("meta_frame").num_columns(2).striped(true).show(ui, |ui| {
                    kv(ui, "Dimensions", format!("{} x {} px", f.width, f.height));
                    let format = match f.sample_format {
                        fast_tiff_lib::SampleFormat::UnsignedInt => "unsigned integer",
                        fast_tiff_lib::SampleFormat::SignedInt => "signed integer",
                        fast_tiff_lib::SampleFormat::Float => "IEEE float",
                    };
                    kv(ui, "Pixel type", format!("{}-bit {format}", f.bits_per_sample));
                    kv(
                        ui,
                        "Samples/pixel",
                        match (f.is_rgb(), f.is_planar()) {
                            (true, false) => format!("{} (chunky RGB)", f.samples_per_pixel),
                            (true, true) => format!("{} (planar RGB)", f.samples_per_pixel),
                            (false, true) => format!("{} (planar)", f.samples_per_pixel),
                            (false, false) => f.samples_per_pixel.to_string(),
                        },
                    );
                    let photometric = match f.photometric {
                        0 => "0 (WhiteIsZero)".into(),
                        1 => "1 (BlackIsZero)".into(),
                        2 => "2 (RGB)".into(),
                        3 => "3 (palette)".into(),
                        other => format!("{other}"),
                    };
                    kv(ui, "Photometric", photometric);
                    let compression = match f.compression {
                        fast_tiff_lib::Compression::None => "uncompressed".into(),
                        fast_tiff_lib::Compression::Lzw => "LZW".into(),
                        fast_tiff_lib::Compression::PackBits => "PackBits".into(),
                        fast_tiff_lib::Compression::Deflate => "Deflate (zip)".into(),
                        fast_tiff_lib::Compression::Zstd => "ZSTD".into(),
                        other => format!("{other:?}"),
                    };
                    kv(ui, "Compression", compression);
                    let predictor = match f.predictor {
                        1 => "none".into(),
                        2 => "2 (horizontal differencing)".into(),
                        3 => "3 (floating-point)".into(),
                        other => format!("{other}"),
                    };
                    kv(ui, "Predictor", predictor);
                    kv(
                        ui,
                        "Strips/frame",
                        format!("{} ({} rows/strip)", f.strip_offsets.len(), f.rows_per_strip),
                    );
                    let bpf = f.width as u64
                        * f.height as u64
                        * f.samples_per_pixel as u64
                        * (f.bits_per_sample as u64 / 8);
                    kv(ui, "Decoded frame", human_bytes(bpf));
                });
            }

            ui.add_space(12.0);
            egui::CollapsingHeader::new("ImageDescription (tag 270)")
                .default_open(true)
                .show(ui, |ui| match &tiff.description {
                    Some(desc) => {
                        // Read-only TextEdit: selectable + copyable.
                        let mut text = desc.as_str();
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(desc.lines().count().clamp(2, 16)),
                        );
                    }
                    None => {
                        ui.label(RichText::new("(this file carries no ImageDescription)").weak());
                    }
                });

            ui.add_space(12.0);
            ui.heading("ImageJ metadata");
            let meta = &tiff.meta;
            egui::Grid::new("meta_ij").num_columns(2).striped(true).show(ui, |ui| {
                kv(
                    ui,
                    "Dimensions",
                    format!(
                        "{} channel(s) x {} slice(s) x {} frame(s)",
                        meta.channels, meta.slices, meta.frames
                    ),
                );
                let mode = match meta.mode {
                    fast_tiff_lib::DisplayMode::Grayscale => "grayscale",
                    fast_tiff_lib::DisplayMode::Composite => "composite",
                    fast_tiff_lib::DisplayMode::Color => "color",
                };
                kv(ui, "Display mode", mode);
                if let Some(unit) = &meta.unit {
                    kv(ui, "Unit", unit.clone());
                }
                if let Some(fi) = meta.frame_interval_s {
                    kv(ui, "Frame interval", format!("{fi} s"));
                }
                if let Some(fps) = meta.fps {
                    kv(ui, "Playback fps", fps.to_string());
                }
                if let Some(spacing) = meta.spacing {
                    kv(ui, "Z spacing", spacing.to_string());
                }
                if let Some(looped) = meta.loop_playback {
                    kv(ui, "Loop playback", looped.to_string());
                }
                if let Some((c0, c1)) = meta.calibration {
                    kv(ui, "Calibration", format!("value = {c0} + {c1} x raw"));
                }
                for (i, cd) in meta.channel_display.iter().enumerate() {
                    let range = match cd.range {
                        Some((lo, hi)) => format!("{lo} .. {hi}"),
                        None => "auto-contrast".into(),
                    };
                    kv(ui, &format!("Ch {} display range", i + 1), range);
                }
            });
        });
}

/// `1234567` -> `"1.2 MiB (1234567 bytes)"`.
pub(super) fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{:.1} {} ({n} bytes)", v, UNITS[u])
    }
}

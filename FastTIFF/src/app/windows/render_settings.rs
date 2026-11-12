//! The 3D render-settings pop-up, opened from the toolbar.

use super::*;

use crate::render;
use egui::RichText;
use fast_tiff_viewer::camera::{NavMode, OrbitPoint};

impl WindowId {
    pub(in crate::app) const RENDER_SETTINGS: WindowId = WindowId("render_settings");
}

/// The window. Everything it edits is the core's volume settings and the 3D
/// preferences, both handed in each frame, so it keeps nothing of its own.
pub(super) struct RenderSettingsWindow;

impl AppWindow for RenderSettingsWindow {
    fn id(&self) -> WindowId {
        WindowId::RENDER_SETTINGS
    }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool, host: &mut Host<'_>) {
        let core = &mut *host.core;
        let prefs = &mut *host.prefs;
        let prev_nav = core.volume.cam.nav;
        let mut reset_position = false;
        render_settings_window(
            ctx,
            open,
            &mut core.volume.scale,
            &mut core.volume.interp,
            &mut prefs.moving_downscale,
            &mut core.volume.cam.nav,
            &mut core.volume.cam.orbit_point,
            &mut prefs.move_speed,
            &mut prefs.scroll_speed,
            &mut core.volume.render,
            &mut core.volume.density,
            &mut core.volume.iso,
            &mut core.volume.mip_brightness,
            &mut core.volume.volume_brightness,
            &mut core.volume.volume_autolevel,
            &mut prefs.show_coord_box,
            &mut reset_position,
            core.document.as_ref(),
        );
        // Keep the view continuous across a fly⇄orbit switch (the two use
        // different eye representations) instead of snapping to a default.
        if core.volume.cam.nav != prev_nav {
            core.volume.cam.sync_for_nav(prev_nav.is_fly());
        }
        if reset_position {
            core.volume.cam.reset();
        }
    }

    /// Up on the desktop for as long as the new document has a volume. It is
    /// reachable only through a button the toolbar hides for a stack with
    /// none, so leaving it up over a single-frame file would strand it:
    /// closed, it could not be reopened. In the browser it closes, as every
    /// pop-up there does (see [`MetadataWindow`](super::metadata::MetadataWindow)).
    fn outlives_document(&self, core: &Viewer, web: bool) -> bool {
        !web && core.can_show_volume()
    }
}

/// The 3D render-settings pop-up: rendering method and the controls belonging
/// to it, per-axis voxel scale (x:y:z), interpolation, the resolution while the
/// view moves (`moving_downscale`: how many times smaller per axis, 1 for
/// full), navigation style, and a camera-reset button. Scale defaults to the stack's pixel calibration (with a
/// button to re-seed it). `reset_position` is set true when the user clicks
/// Reset position.
///
/// # The mode's controls, and only the mode's
///
/// Density belongs to Volume, Threshold to Surface, and each mode has its own
/// brightness. They used to be laid out together and greyed out when they did
/// not apply, which is the wrong choice here: a disabled control is a promise
/// that some reachable state will enable it, and reading three greyed rows to
/// find the one live one is work on every visit. They are **hidden** instead,
/// so the panel shows what the current mode actually has.
///
/// The exception that proves the rule is the mode selector itself, which is
/// always all three — that *is* the reachable state.
#[allow(clippy::too_many_arguments)]
fn render_settings_window(
    ctx: &egui::Context,
    open: &mut bool,
    scale: &mut [f32; 3],
    interp: &mut render::VolumeInterp,
    moving_downscale: &mut u32,
    nav: &mut NavMode,
    orbit_point: &mut OrbitPoint,
    move_speed: &mut f32,
    scroll_speed: &mut f32,
    render_mode: &mut render::VolumeRender,
    density: &mut f32,
    iso: &mut f32,
    mip_brightness: &mut f32,
    volume_brightness: &mut f32,
    volume_autolevel: &mut bool,
    show_coord_box: &mut bool,
    reset_position: &mut bool,
    loaded: Option<&Document>,
) {
    let spec = dialog::Dialog {
        id: WindowId::RENDER_SETTINGS.0,
        title: "3D render settings",
        // The height only the first time it opens; then it is the content's.
        size: egui::vec2(300.0, 520.0),
        scroll: true,
        // The rows change with the render mode, and a window kept at the
        // tallest mode's height would sit half empty in the others.
        fit_height: true,
        resizable: false,
        // Every slider in here changes the render while it is being dragged, so
        // the picture and the controls have to be visible at once.
        always_on_top: true,
    };
    dialog::show(ctx, spec, open, |ui| {
        ui.label(RichText::new("Rendering").strong());
        ui.horizontal(|ui| {
            ui.selectable_value(render_mode, render::VolumeRender::Mip, "Max intensity")
                .on_hover_text("Maximum-intensity projection: brightest sample along each ray");
            ui.selectable_value(render_mode, render::VolumeRender::Alpha, "Volume")
                .on_hover_text("ImageJ 3D Viewer style: translucent alpha-blended volume");
            ui.selectable_value(render_mode, render::VolumeRender::Surface, "Surface")
                .on_hover_text("Opaque isosurface: a shaded solid at the intensity threshold");
        });
        match *render_mode {
            // Max intensity: a plain gain, and nothing else to set. The default
            // is below 100% because a maximum over a few hundred samples per
            // ray is a high-order statistic and comes out hotter than the same
            // data in 2-D — see `fast_tiff_viewer::brightness`.
            render::VolumeRender::Mip => {
                ui.horizontal(|ui| {
                    ui.label("Brightness");
                    ui.add(
                        egui::Slider::new(mip_brightness, 0.1..=2.0)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                            .custom_parser(|s| {
                                s.trim_end_matches('%')
                                    .trim()
                                    .parse::<f64>()
                                    .ok()
                                    .map(|v| v / 100.0)
                            }),
                    )
                    .on_hover_text(
                        "Gain on the projection. Below 100% by default: a \
                         maximum along each ray runs brighter than the same \
                         data in 2D.",
                    );
                });
            }
            render::VolumeRender::Alpha => {
                ui.horizontal(|ui| {
                    ui.label("Density");
                    ui.add(egui::Slider::new(density, 1.0..=1000.0).logarithmic(true))
                        .on_hover_text("How solid the volume is (higher = more opaque)");
                });
                ui.checkbox(volume_autolevel, "Auto-level brightness")
                    .on_hover_text(
                        "Keeps the brightness steady as Density moves. Without \
                         it, dragging Density changes how bright the volume is \
                         as much as how solid — dark at the low end and \
                         saturated at the high one.",
                    );
                // The manual gain appears only when the automatic one is off,
                // because two brightness controls fighting each other is worse
                // than either alone.
                if !*volume_autolevel {
                    ui.horizontal(|ui| {
                        ui.label("Brightness");
                        ui.add(
                            egui::Slider::new(volume_brightness, 0.1..=4.0)
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0))
                                .custom_parser(|s| {
                                    s.trim_end_matches('%')
                                        .trim()
                                        .parse::<f64>()
                                        .ok()
                                        .map(|v| v / 100.0)
                                }),
                        )
                        .on_hover_text("Gain on the composited volume");
                    });
                }
            }
            render::VolumeRender::Surface => {
                ui.horizontal(|ui| {
                    ui.label("Threshold");
                    ui.add(egui::Slider::new(iso, 0.01..=1.0)).on_hover_text(
                        "Isosurface level in windowed units (higher = only brighter voxels)",
                    );
                });
            }
        }

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
                    .on_hover_text(
                        "Re-seed x:y:z from the file's pixel calibration + spacing (else 1:1:1)",
                    )
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
        ui.label(RichText::new("Performance").strong());
        ui.horizontal_wrapped(|ui| {
            ui.label("Resolution while moving");
            // Per axis, as the view's size is: 50% is a quarter of the rays.
            for (downscale, label, help) in [
                (1, "Full", "Every pixel, moving or not"),
                (
                    2,
                    "50%",
                    "Half the resolution while the view moves (a quarter of the \
                     rays), sharp again as soon as it stops",
                ),
                (
                    4,
                    "25%",
                    "A quarter of the resolution while the view moves (a \
                     sixteenth of the rays), sharp again as soon as it stops. \
                     The most responsive on large stacks",
                ),
            ] {
                ui.selectable_value(moving_downscale, downscale, label)
                    .on_hover_text(help);
            }
        });

        ui.separator();
        ui.label(RichText::new("Navigation").strong());
        ui.horizontal_wrapped(|ui| {
            for mode in [
                NavMode::Cad,
                NavMode::Blender,
                NavMode::Maya,
                NavMode::WasdFly,
            ] {
                ui.selectable_value(nav, mode, mode.label())
                    .on_hover_text(mode.help());
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
            if ui
                .small_button("Reset")
                .on_hover_text("Restore both speeds to 1×")
                .clicked()
            {
                *move_speed = 1.0;
                *scroll_speed = 1.0;
            }
        });
        // What an orbit drag rotates around.
        ui.label("Orbiting point:");
        ui.radio_value(orbit_point, OrbitPoint::VolumeCenter, "Volume center")
            .on_hover_text(
                "Rotate around the volume's center — a turntable that re-centers the box",
            );
        ui.radio_value(orbit_point, OrbitPoint::ScreenCenter, "Screen center")
            .on_hover_text("Rotate around whatever the view center is aimed at");

        ui.separator();
        ui.label(RichText::new("Overlay").strong());
        ui.checkbox(show_coord_box, "Coordinate box").on_hover_text(
            "Draw the volume's bounding box with x/y/z tick coordinates \
             (pixels, or calibrated length if the file has a physical unit)",
        );

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

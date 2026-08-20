//! The pop-up windows: 3D render settings, the file-metadata viewer, and the
//! channel histogram. Split from `app.rs`.
//!
//! All three are opened through [`super::scale::unscaled`], so they keep their
//! native size on the web build where the surrounding chrome is enlarged.

use super::scale::unscaled;
use super::*;

use fast_tiff_viewer::camera::{NavMode, OrbitPoint};
use fast_tiff_viewer::histogram::{fill_alpha, fill_tint, Histogram, BINS};
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
    orbit_point: &mut OrbitPoint,
    move_speed: &mut f32,
    scroll_speed: &mut f32,
    render_mode: &mut render::VolumeRender,
    density: &mut f32,
    iso: &mut f32,
    show_coord_box: &mut bool,
    reset_position: &mut bool,
    loaded: Option<&Stack>,
) {
    unscaled(ctx, |ctx| {
    egui::Window::new("3D render settings")
        .open(open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.label(RichText::new("Rendering").strong());
            ui.horizontal(|ui| {
                ui.selectable_value(render_mode, render::VolumeRender::Mip, "Max intensity")
                    .on_hover_text("Maximum-intensity projection: brightest sample along each ray");
                ui.selectable_value(render_mode, render::VolumeRender::Alpha, "Volume")
                    .on_hover_text("ImageJ 3D Viewer style: translucent alpha-blended volume");
                ui.selectable_value(render_mode, render::VolumeRender::Surface, "Surface")
                    .on_hover_text("Opaque isosurface: a shaded solid at the intensity threshold");
            });
            // Density only affects the alpha DVR — disabled for the other modes.
            ui.add_enabled_ui(*render_mode == render::VolumeRender::Alpha, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Density");
                    ui.add(egui::Slider::new(density, 1.0..=1000.0).logarithmic(true))
                        .on_hover_text("Opacity of the alpha volume (higher = brighter/more solid)");
                });
            });
            // Iso threshold only affects the surface mode — disabled otherwise.
            ui.add_enabled_ui(*render_mode == render::VolumeRender::Surface, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Threshold");
                    ui.add(egui::Slider::new(iso, 0.01..=1.0))
                        .on_hover_text("Isosurface level in windowed units (higher = only brighter voxels)");
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
            // What an orbit drag rotates around.
            ui.label("Orbiting point:");
            ui.radio_value(orbit_point, OrbitPoint::VolumeCenter, "Volume center")
                .on_hover_text("Rotate around the volume's center — a turntable that re-centers the box");
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
    });
}

pub(super) fn metadata_window(ctx: &egui::Context, open: &mut bool, loaded: &Stack) {
    let tiff = &loaded.tiff;
    unscaled(ctx, |ctx| {
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
                kv(ui, "Size", human_bytes(tiff.data.len() as u64));
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
                let meta_format = match tiff.meta.source_format {
                    fast_tiff_lib::MetadataFormat::ImageJ => "ImageJ",
                    fast_tiff_lib::MetadataFormat::Ome => "OME-XML",
                    _ => "—",
                };
                kv(ui, "Metadata", meta_format);
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
                        {
                            let model = if f.is_rgb() {
                                Some("RGB")
                            } else if f.is_cmyk() {
                                Some("CMYK")
                            } else {
                                None
                            };
                            match (model, f.is_planar()) {
                                (Some(m), false) => format!("{} (chunky {m})", f.samples_per_pixel),
                                (Some(m), true) => format!("{} (planar {m})", f.samples_per_pixel),
                                (None, true) => format!("{} (planar)", f.samples_per_pixel),
                                (None, false) => f.samples_per_pixel.to_string(),
                            }
                        },
                    );
                    let photometric = match f.photometric {
                        0 => "0 (WhiteIsZero)".into(),
                        1 => "1 (BlackIsZero)".into(),
                        2 => "2 (RGB)".into(),
                        3 => "3 (palette)".into(),
                        // Separated. InkSet says *which* inks; only set 1 is
                        // CMYK, and only that one the viewer converts. Anything
                        // else stays raw ink planes, so label it honestly
                        // rather than calling every separated file CMYK.
                        5 if f.ink_set == 1 => "5 (CMYK)".into(),
                        5 => format!("5 (separated, InkSet {})", f.ink_set),
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
                // Everything in this panel is the *file's* own metadata, left
                // exactly as parsed. The viewer's interpretation is separate
                // (see `fast_tiff_viewer::display`) and routinely differs — a
                // mislabeled `channels=100` is really a frame count, and the
                // user can reassign the axes by hand. Show that only when the
                // two disagree, so the panel stays quiet for ordinary files.
                let shown = loaded.display.dims;
                if (shown.channels, shown.slices, shown.frames)
                    != (meta.channels, meta.slices, meta.frames)
                {
                    kv(
                        ui,
                        "Shown as",
                        format!(
                            "{} channel(s) x {} slice(s) x {} frame(s)",
                            shown.channels, shown.slices, shown.frames
                        ),
                    );
                }
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

/// Histogram plot height when the window opens, and the floor it can be
/// dragged down to — both in multiples of the body text height.
///
/// Relative rather than fixed point counts because [`unscaled`] resizes a
/// pop-up by scaling its *style*: a hard-coded length would keep the enlarged
/// chrome's scale while the text around it shrank back to native, and the plot
/// would come out half again too tall for its own window.
const PLOT_TEXT_HEIGHTS: f32 = 9.0;
const MIN_PLOT_TEXT_HEIGHTS: f32 = 3.0;

/// How much of its fill and outline a curve keeps where the channel's contrast
/// window clips it away.
///
/// Faded rather than hidden: what a window is throwing away is exactly what you
/// need to see to judge whether it is set right — a tail cut off here is
/// detail crushed to black or blown to white in the image. Dim enough to read
/// as excluded at a glance, present enough to count.
const CLIPPED_FILL: f32 = 0.3;
const CLIPPED_STROKE_ALPHA: u8 = 70;

/// The channel-histogram pop-up: the intensity distribution of the frame on
/// screen, with the contrast sliders directly beneath it.
///
/// The two belong together — a contrast window is a choice about where the data
/// actually is, and until now that choice was made blind. The plot is drawn
/// across exactly the span the sliders below it cover (see
/// [`contrast_controls`]), so a handle sits above the part of the distribution
/// it clips.
///
/// `hists` is computed by the caller and cached; this only draws.
pub(super) fn histogram_window(
    ctx: &egui::Context,
    open: &mut bool,
    loaded: &mut Stack,
    hists: &[Histogram],
    log_scale: &mut bool,
) {
    unscaled(ctx, |ctx| {
        // Opening size only — `Resize` remembers whatever the user drags it to
        // afterwards. The height is the plot at its default size plus what the
        // controls need: `Stacked` spends two lines per channel, plus two more
        // for the hint and log-scale lines. Without this a six-channel stack
        // would open with its plot already squeezed onto the minimum.
        let (row_h, text_h) = {
            let style = ctx.global_style();
            let font = style.text_styles[&egui::TextStyle::Body].clone();
            let text_h = ctx.fonts_mut(|f| f.row_height(&font));
            (style.spacing.interact_size.y + style.spacing.item_spacing.y, text_h)
        };
        let rows = loaded.display.settings.len().max(1) as f32 * 2.0 + 2.0;
        egui::Window::new("Histogram")
            .open(open)
            .resizable(true)
            // An explicit size, rather than auto-sizing to content: the plot
            // claims whatever vertical space is left over, so the window needs a
            // height of its own for it to claim. Dragging the corner then grows
            // the plot instead of the empty space around it.
            .default_size([420.0 / super::scale::UI_SCALE, text_h * PLOT_TEXT_HEIGHTS + row_h * rows])
            .show(ctx, |ui| {
                // How tall everything *below* the plot is depends on the channel
                // count, and it is drawn after the plot has to be sized. Rather
                // than guess, reuse the height it came to last frame — one
                // frame of lag while a resize handle is being dragged, which is
                // invisible, and no circular dependency between the two.
                let id = ui.id().with("controls_h");
                let controls_h = ui.data(|d| d.get_temp::<f32>(id)).unwrap_or(0.0);
                let plot_h =
                    (ui.available_height() - controls_h).max(text_h * MIN_PLOT_TEXT_HEIGHTS);

                // Reserve the plot's rect now and paint into it further down,
                // once the sliders have reported the span to align with. Nothing
                // else draws there, so the out-of-order painting is invisible.
                let (plot, _) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), plot_h),
                    egui::Sense::hover(),
                );

                let controls_top = ui.cursor().top();
                // Every slider spans the same range the plot is binned over, so
                // a handle's x means the same thing as a point on the curve
                // above it — which is what lets the plot fade exactly where the
                // handle sits.
                let axis = fast_tiff_viewer::histogram::shared_track(loaded);
                let track = contrast_controls(ui, loaded, ContrastLayout::Stacked, Some(axis));

                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(log_scale, "Log scale").on_hover_text(
                        "Plot log(1 + count). Microscopy frames are mostly background, \
                         and one enormous bin at the dark end flattens everything else \
                         into the axis on a linear plot.",
                    );
                    if let Some(h) = hists.first() {
                        ui.separator();
                        ui.label(
                            RichText::new(format!("{} bins · {} px sampled", BINS, h.counted))
                                .small()
                                .weak(),
                        );
                    }
                });

                ui.data_mut(|d| d.insert_temp(id, ui.cursor().top() - controls_top));

                // Align the plot with the sliders when there are any; a palette
                // stack draws no slider, so fall back to the full width.
                let span = track.unwrap_or_else(|| plot.x_range());
                let area = egui::Rect::from_x_y_ranges(span, plot.y_range());
                draw_plot(ui, area, hists, &loaded.display, *log_scale);
            });
    });
}

/// Paint every channel's histogram over one set of axes.
fn draw_plot(
    ui: &egui::Ui,
    area: egui::Rect,
    hists: &[Histogram],
    display: &fast_tiff_viewer::Display,
    log_scale: bool,
) {
    let painter = ui.painter_at(area);
    painter.rect_filled(area, 2.0, ui.visuals().extreme_bg_color);

    if hists.is_empty() {
        painter.text(
            area.center(),
            egui::Align2::CENTER_CENTER,
            "No frame to histogram",
            egui::FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
        return;
    }

    // Every plot shares these axes, so they stack; thin them accordingly.
    let alpha = fill_alpha(hists.len());
    // One vertical scale for all channels, matching the one horizontal one.
    // Per-channel scaling would stretch every curve to full height and erase
    // exactly the comparison the shared axis was introduced to show — that this
    // channel is concentrated and that one spread thin.
    let peak = hists.iter().map(|h| h.peak).max().unwrap_or(0).max(1) as f32;
    // Log compresses the tall background bin so the rest of the distribution is
    // visible. Normalising by the *scaled* peak keeps the tallest bar at full
    // height either way, so switching modes rescales rather than shrinks.
    let height = |v: f32| if log_scale { (1.0 + v * 1000.0).ln() / 1000.0_f32.ln_1p() } else { v };

    for h in hists {
        let [r, g, b] = display.lut(h.channel).map(fill_tint).unwrap_or([200, 200, 200]);
        let solid = egui::Color32::from_rgb(r, g, b);
        let fill = egui::Color32::from_rgba_unmultiplied(r, g, b, alpha);
        let dim_fill =
            egui::Color32::from_rgba_unmultiplied(r, g, b, (alpha as f32 * CLIPPED_FILL) as u8);
        let dim_stroke = egui::Color32::from_rgba_unmultiplied(r, g, b, CLIPPED_STROKE_ALPHA);
        let bar_w = area.width() / BINS as f32;

        // One mesh of `BINS` quads rather than `BINS` separate rect shapes, and
        // one polyline across the tops. A filled *polygon* would be the obvious
        // choice, but epaint's closed-path fill assumes convexity and a
        // histogram is the opposite of convex — the tops would web over.
        let mut mesh = egui::epaint::Mesh::default();
        let mut dim_mesh = egui::epaint::Mesh::default();
        let mut top: Vec<egui::Pos2> = Vec::with_capacity(BINS);
        for i in 0..BINS {
            let x = area.left() + i as f32 * bar_w;
            let y = area.bottom() - height(h.bins[i] as f32 / peak) * area.height();
            let bar =
                egui::Rect::from_min_max(egui::pos2(x, y), egui::pos2(x + bar_w, area.bottom()));
            mesh.add_colored_rect(bar, fill);
            dim_mesh.add_colored_rect(bar, dim_fill);
            top.push(egui::pos2(x + bar_w * 0.5, y));
        }

        // The part of this channel the contrast window keeps, as a span of the
        // plot. Both are on the same axis (the window's sliders are given the
        // shared track), so this lands exactly under the channel's handles.
        let span = (h.hi - h.lo).max(f32::EPSILON);
        let x_of = |v: f32| area.left() + ((v - h.lo) / span).clamp(0.0, 1.0) * area.width();
        let (win_min, win_max) = display
            .settings
            .get(h.channel)
            .map(|s| (s.min, s.max))
            .unwrap_or((h.lo, h.hi));
        let kept = egui::Rect::from_x_y_ranges(
            egui::Rangef::new(x_of(win_min), x_of(win_max)),
            area.y_range(),
        );

        // Draw the curve three times under different clips rather than
        // recolouring bin by bin: the boundary then falls exactly on the handle
        // even when it lands mid-bin, and the kept and clipped parts never
        // overlap, so neither is composited on top of the other.
        painter.with_clip_rect(kept).add(egui::Shape::mesh(mesh));
        painter
            .with_clip_rect(kept)
            .add(egui::Shape::line(top.clone(), egui::Stroke::new(1.0, solid)));
        for tail in [
            egui::Rect::from_x_y_ranges(
                egui::Rangef::new(area.left(), kept.left()),
                area.y_range(),
            ),
            egui::Rect::from_x_y_ranges(
                egui::Rangef::new(kept.right(), area.right()),
                area.y_range(),
            ),
        ] {
            if tail.width() <= 0.0 {
                continue;
            }
            let p = painter.with_clip_rect(tail);
            p.add(egui::Shape::mesh(dim_mesh.clone()));
            p.add(egui::Shape::line(top.clone(), egui::Stroke::new(1.0, dim_stroke)));
        }
    }

    painter.rect_stroke(
        area,
        2.0,
        ui.visuals().widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );
}

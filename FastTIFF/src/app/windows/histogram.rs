//! The channel-histogram pop-up, opened from the bottom bar.

use super::*;

use fast_tiff_viewer::histogram::{fill_alpha, fill_tint, Histogram, HistogramSource, BINS};

impl WindowId {
    pub(in crate::app) const HISTOGRAM: WindowId = WindowId("histogram");
}

/// The window, and its one preference.
pub(super) struct HistogramWindow {
    /// Plot log(1 + count). A preference, so it outlives both the window being
    /// closed and the file being changed.
    ///
    /// On by default. A 16-bit microscopy frame is mostly background sitting in
    /// the bottom percent of a 0..65535 track, so a linear plot is one spike at
    /// the left edge and a flat line — which reads as a broken widget rather
    /// than as data. Log shows the distribution that is actually there; the
    /// checkbox is right under the plot for anyone who wants true counts.
    pub(super) log: bool,
}

impl Default for HistogramWindow {
    fn default() -> Self {
        HistogramWindow { log: true }
    }
}

impl AppWindow for HistogramWindow {
    fn id(&self) -> WindowId {
        WindowId::HISTOGRAM
    }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool, host: &mut Host<'_>) {
        let resident = Resident {
            over_texture_limit: host.core.over_texture_limit(),
            window: host.core.resident_window(),
        };
        match host.core.document.as_mut() {
            Some(loaded) => histogram_window(ctx, open, loaded, resident, &mut self.log),
            None => *open = false,
        }
    }

    /// Up on the desktop, for the reasons the metadata window is (see
    /// [`MetadataWindow`](super::metadata::MetadataWindow)) — this is the one most worth
    /// parking on a second monitor while stepping through a folder.
    fn outlives_document(&self, _core: &Viewer, web: bool) -> bool {
        !web
    }
}

/// Histogram plot height when the window opens, and the floor it can be
/// dragged down to — both in multiples of the body text height.
///
/// Relative rather than fixed point counts because of how the browser build
/// keeps pop-ups at native size while enlarging the chrome around them: it
/// scales the *style* rather than the zoom (see [`crate::app::scale::unscaled`]). A
/// hard-coded length would keep the enlarged scale while the text beside it
/// shrank back to native, and the plot would come out half again too tall for
/// its own window.
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

/// Hover text for the histogram Auto button.
const AUTO_HINT: &str = "Clip the outermost 0.35% of samples at each end.\
                         Fitted to the frame you are looking at, not to frame 0.";

/// Hover text for the histogram Reset button.
const RESET_HINT: &str = "Put every contrast window back where it was when the file was opened.";

/// What the viewer's last sync put on the GPU of the frame on screen, as far as
/// the histogram window says anything about it.
#[derive(Clone, Copy)]
struct Resident {
    /// Whether the frame is past the device's texture limit
    /// ([`Viewer::over_texture_limit`](fast_tiff_viewer::Viewer::over_texture_limit)).
    over_texture_limit: bool,
    /// The window of it the textures hold
    /// ([`Viewer::resident_window`](fast_tiff_viewer::Viewer::resident_window)).
    window: Option<fast_tiff_viewer::roi::Roi>,
}

/// The channel-histogram pop-up: the intensity distribution of the frame on
/// screen, with the contrast sliders directly beneath it.
///
/// The two belong together — a contrast window is a choice about where the data
/// actually is, and until now that choice was made blind. The plot is drawn
/// across exactly the span the sliders below it cover (see
/// [`contrast_controls`]), so a handle sits above the part of the distribution
/// it clips.
///
/// What it plots is the document's own histograms ([`Document::histograms`]),
/// which the viewer bins from the frames it decodes while the window is open;
/// this only draws. They are borrowed from the document only where nothing
/// edits it (the note on where they came from, and the plot), so the contrast
/// controls drawn between the two edit it with nothing copied out. `resident`
/// is what the viewer's last sync put on the GPU, which says why a frame too
/// large for one texture may have none.
fn histogram_window(
    ctx: &egui::Context,
    open: &mut bool,
    loaded: &mut Document,
    resident: Resident,
    log_scale: &mut bool,
) {
    // Past the texture limit the frame is never decoded whole, and what is
    // binned is the overview kept for zooming out: say so, since its counts
    // are not the frame's.
    let held = loaded.histograms();
    let overview = match held.source() {
        Some(HistogramSource::Overview { stride }) if !held.shown(loaded).is_empty() => {
            Some(stride)
        }
        _ => None,
    };
    // Such a frame is binned from a view of the whole of it as that view is
    // built, so with none there are two reasons: zoomed in on it, there is no
    // such view; zoomed out, the one on screen was built while the histogram
    // was closed and was too large to keep.
    let zoomed_out = resident
        .window
        .zip(loaded.dimensions())
        .is_some_and(|(roi, (w, h))| fast_tiff_viewer::window::spans_whole_frame(&roi, w, h));
    let empty_note = if !resident.over_texture_limit {
        "No frame to histogram"
    } else if zoomed_out {
        "A frame this large is histogrammed from a view of the whole\nof it as that is built: zoom in and back out to build one"
    } else {
        "Zoom out to see the whole frame: a frame this large\nis histogrammed from its overview"
    };
    // Opening size only — the host remembers whatever the user drags it to
    // afterwards. The height is the plot at its default size plus what the
    // controls need: `Stacked` spends two lines per channel, plus two more for
    // the hint and the log-scale/button line. Without this a six-channel stack
    // would open with its plot already squeezed onto the minimum.
    let (row_h, text_h) = {
        let style = ctx.global_style();
        let font = style.text_styles[&egui::TextStyle::Body].clone();
        let text_h = ctx.fonts_mut(|f| f.row_height(&font));
        (
            style.spacing.interact_size.y + style.spacing.item_spacing.y,
            text_h,
        )
    };
    let rows = loaded.display.settings().len().max(1) as f32 * 2.0 + 2.0;
    let spec = dialog::Dialog {
        id: WindowId::HISTOGRAM.0,
        title: "Histogram",
        // An explicit size, rather than auto-sizing to content: the plot claims
        // whatever vertical space is left over, so the window needs a height of
        // its own for it to claim. Dragging the corner then grows the plot
        // instead of the empty space around it.
        //
        // Divided by the interface scale because the metrics above are measured
        // under the enlarged style the chrome uses, while the pop-up itself is
        // drawn at native size (see `dialog`).
        size: egui::vec2(420.0, text_h * PLOT_TEXT_HEIGHTS + row_h * rows) / UI_SCALE,
        // No scroll area: the plot sizes itself from the height left over after
        // the controls, and inside a scroll area that is unbounded.
        scroll: false,
        fit_height: false,
        resizable: true,
        // Read *about* the picture rather than used to change it, so it
        // stays an ordinary window a user can put behind the app.
        always_on_top: false,
    };
    dialog::show(ctx, spec, open, |ui| {
        // How tall everything *below* the plot is depends on the channel
        // count, and it is drawn after the plot has to be sized. Rather
        // than guess, reuse the height it came to last frame — one
        // frame of lag while a resize handle is being dragged, which is
        // invisible, and no circular dependency between the two.
        let id = ui.id().with("controls_h");
        let controls_h = ui.data(|d| d.get_temp::<f32>(id)).unwrap_or(0.0);
        let plot_h = (ui.available_height() - controls_h).max(text_h * MIN_PLOT_TEXT_HEIGHTS);

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
            if let Some(stride) = overview {
                ui.separator();
                ui.weak(format!("From the 1/{stride} overview"))
                    .on_hover_text(format!(
                        "This frame is too large to decode whole, so its \
                         histogram is binned from the coarse copy kept for \
                         zooming out: one pixel in {stride} across and down. \
                         The shape is the frame's; the counts are the \
                         overview's, and a feature narrower than {stride} \
                         pixels can be missed."
                    ));
            }
            // Auto and Reset drive the very sliders above them, so they
            // are shown only when there are any: a palette stack has a
            // fixed index-to-colour window rather than a contrast one,
            // and `contrast_controls` draws no slider for it.
            if track.is_some() {
                ui.separator();
                if ui.button("Auto").on_hover_text(AUTO_HINT).clicked() {
                    // Fitted to them while it edits the document that holds
                    // them, so it is handed a copy: on a click, not a repaint.
                    let hists = loaded.histograms().shown(loaded).to_vec();
                    fast_tiff_viewer::channels::auto_contrast(loaded, &hists);
                }
                if ui.button("Reset").on_hover_text(RESET_HINT).clicked() {
                    fast_tiff_viewer::channels::reset_contrast(loaded);
                }
            }
        });

        ui.data_mut(|d| d.insert_temp(id, ui.cursor().top() - controls_top));

        // Align the plot with the sliders when there are any; a palette
        // stack draws no slider, so fall back to the full width.
        let span = track.unwrap_or_else(|| plot.x_range());
        let area = egui::Rect::from_x_y_ranges(span, plot.y_range());
        let hists = loaded.histograms().shown(loaded);
        draw_plot(ui, area, hists, &loaded.display, *log_scale, empty_note);
    });
}

/// Paint every channel's histogram over one set of axes. `empty_note` is what
/// to say when there are none at all.
fn draw_plot(
    ui: &egui::Ui,
    area: egui::Rect,
    hists: &[Histogram],
    display: &fast_tiff_viewer::Display,
    log_scale: bool,
    empty_note: &str,
) {
    let painter = ui.painter_at(area);
    painter.rect_filled(area, 2.0, ui.visuals().extreme_bg_color);

    // Only the channels still switched on. Unticking one takes its curve off
    // the plot, matching the image, where it stops being composited. The
    // histograms are computed for every channel regardless, so this is a
    // redraw rather than another decode — toggling is instant.
    let shown: Vec<&Histogram> = hists
        .iter()
        .filter(|h| display.settings().get(h.channel).is_none_or(|s| s.enabled))
        .collect();

    if shown.is_empty() {
        painter.text(
            area.center(),
            egui::Align2::CENTER_CENTER,
            if hists.is_empty() {
                empty_note
            } else {
                "Every channel is switched off"
            },
            egui::FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
        painter.rect_stroke(
            area,
            2.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
        return;
    }

    // The curves that remain share these axes, so they stack; thin them
    // accordingly. Counting only what is drawn means hiding a channel makes the
    // rest more solid, rather than leaving a gap in the budget.
    let alpha = fill_alpha(shown.len());
    // One vertical scale for all channels, matching the one horizontal one.
    // Per-channel scaling would stretch every curve to full height and erase
    // exactly the comparison the shared axis was introduced to show — that this
    // channel is concentrated and that one spread thin.
    let peak = shown.iter().map(|h| h.peak).max().unwrap_or(0).max(1) as f32;
    // Log compresses the tall background bin so the rest of the distribution is
    // visible. Normalising by the *scaled* peak keeps the tallest bar at full
    // height either way, so switching modes rescales rather than shrinks.
    let height = |v: f32| {
        if log_scale {
            (1.0 + v * 1000.0).ln() / 1000.0_f32.ln_1p()
        } else {
            v
        }
    };

    for h in shown {
        let [r, g, b] = display
            .lut(h.channel)
            .map(fill_tint)
            .unwrap_or([200, 200, 200]);
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
            .settings()
            .get(h.channel)
            .map_or((h.lo, h.hi), |s| s.window.into());
        let kept = egui::Rect::from_x_y_ranges(
            egui::Rangef::new(x_of(win_min), x_of(win_max)),
            area.y_range(),
        );

        // Draw the curve three times under different clips rather than
        // recolouring bin by bin: the boundary then falls exactly on the handle
        // even when it lands mid-bin, and the kept and clipped parts never
        // overlap, so neither is composited on top of the other.
        painter.with_clip_rect(kept).add(egui::Shape::mesh(mesh));
        painter.with_clip_rect(kept).add(egui::Shape::line(
            top.clone(),
            egui::Stroke::new(1.0_f32, solid),
        ));
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
            p.add(egui::Shape::line(
                top.clone(),
                egui::Stroke::new(1.0_f32, dim_stroke),
            ));
        }
    }

    painter.rect_stroke(
        area,
        2.0,
        ui.visuals().widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );
}

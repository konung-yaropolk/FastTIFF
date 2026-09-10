//! Drawing a [`Plot`] a plugin declared, and the canvas tool that feeds it.
//!
//! Everything here is generic. This module knows how to draw a run of numbers
//! with labelled axes, and how to turn a drag into a rectangle or an ellipse
//! snapped to pixels. It does not know what is being measured, what the axes
//! mean, or why a region is interesting — all of that is the plugin's, and the
//! only thing that crosses is a [`Plot`].
//!
//! Which is the point: a second plugin wanting a line profile, or a histogram
//! of a region, or a bleach-correction curve, uses this unchanged.
//!
//! # No plotting crate
//!
//! The histogram window next door already draws its own axes and bars with
//! [`egui::Painter`], so the idiom is established, and a few polylines are not
//! worth a dependency.

use super::*;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use fasttiff_plugin_api::{Plot, Roi, SelectionKind, Shape};

/// Colours for regions and the traces that describe them.
///
/// Held apart for the common kinds of colour blindness, and legible on both
/// themes. These are drawn on the picture and on the chart's background, so
/// they cannot borrow the channel LUTs — those are the image's colours and
/// would vanish into it.
const PALETTE: [Color32; 8] = [
    Color32::from_rgb(0x1f, 0x9e, 0xd8),
    Color32::from_rgb(0xe6, 0x7e, 0x22),
    Color32::from_rgb(0x2e, 0xcc, 0x71),
    Color32::from_rgb(0xe0, 0x4f, 0x5f),
    Color32::from_rgb(0x9b, 0x59, 0xb6),
    Color32::from_rgb(0xf1, 0xc4, 0x0f),
    Color32::from_rgb(0x1a, 0xbc, 0x9c),
    Color32::from_rgb(0x95, 0xa5, 0xa6),
];

/// The colour for the region or series at `i`.
///
/// Regions and the series describing them arrive in the same order, so indexing
/// one palette from both places is what makes trace 3 in the legend and region
/// 3 on the picture the same colour — without the plugin and the host having to
/// agree a colour between them. A plugin that genuinely wants a specific colour
/// says so with [`Series::color`](fasttiff_plugin_api::Series::color).
pub(super) fn palette(i: usize) -> Color32 {
    PALETTE[i % PALETTE.len()]
}

/// A plot on screen, and the selection it was computed for.
pub(super) struct PlotWindow {
    /// Which plugin produced it, so it can be run again for a new selection.
    pub plugin: usize,
    /// The values that plugin's dialog was answered with, reused verbatim on
    /// every re-run — a re-run is the same question about different regions,
    /// not a new question.
    pub params: fasttiff_plugin_api::Params,
    pub plot: Plot,
    /// The regions drawn, in image pixels.
    pub rois: Vec<Roi>,
    /// The shape the next drag makes.
    pub shape: Shape,
    /// A drag in flight: the pixel it started on and the pixel it is over.
    pub drag: Option<((i64, i64), (i64, i64))>,
    /// Set when the selection changed and the plot no longer describes it.
    pub stale: bool,
}

impl PlotWindow {
    pub fn new(plugin: usize, params: fasttiff_plugin_api::Params, plot: Plot) -> Self {
        PlotWindow {
            plugin,
            params,
            plot,
            rois: Vec::new(),
            shape: Shape::Rect,
            drag: None,
            stale: false,
        }
    }

    /// Whether this plot asked for a canvas tool.
    pub fn wants_regions(&self) -> bool {
        self.plot.wants == SelectionKind::Regions
    }
}

// ------------------------------------------------------------ the canvas tool

/// The image pixel under a point on screen.
///
/// Floored, not rounded: the pixel under the cursor is the one it is *within*,
/// and rounding would give the top-left quarter of a pixel to its neighbour.
pub(super) fn pixel_at(pos: Pos2, origin: Pos2, zoom: f32) -> (i64, i64) {
    (
        ((pos.x - origin.x) / zoom).floor() as i64,
        ((pos.y - origin.y) / zoom).floor() as i64,
    )
}

/// The region a drag between two pixels covers.
///
/// The far corner is pushed out by one, so a drag that starts and ends on the
/// same pixel selects that pixel. Without it the smallest possible selection
/// would be two pixels across, and a careful click on one bright spot would
/// select nothing.
pub(super) fn drag_region(
    from: (i64, i64),
    to: (i64, i64),
    shape: Shape,
    width: u32,
    height: u32,
) -> Option<Roi> {
    let lo = (from.0.min(to.0), from.1.min(to.1));
    let hi = (from.0.max(to.0) + 1, from.1.max(to.1) + 1);
    Roi::from_corners(lo, hi, width, height).map(|r| r.with_shape(shape))
}

fn roi_rect(roi: &Roi, origin: Pos2, zoom: f32) -> Rect {
    Rect::from_min_size(
        origin + Vec2::new(roi.x as f32 * zoom, roi.y as f32 * zoom),
        Vec2::new(roi.w as f32 * zoom, roi.h as f32 * zoom),
    )
}

/// Outline one region.
///
/// An ellipse is a closed polyline because egui has no ellipse primitive and
/// `circle_stroke` cannot be squashed. Forty segments is past where the corners
/// show at any zoom this view reaches.
fn stroke_roi(painter: &egui::Painter, rect: Rect, shape: Shape, stroke: Stroke) {
    match shape {
        Shape::Rect => {
            painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Middle);
        }
        Shape::Ellipse => {
            const N: usize = 40;
            let c = rect.center();
            let (rx, ry) = (rect.width() / 2.0, rect.height() / 2.0);
            let pts: Vec<Pos2> = (0..N)
                .map(|i| {
                    let a = i as f32 / N as f32 * std::f32::consts::TAU;
                    Pos2::new(c.x + rx * a.cos(), c.y + ry * a.sin())
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
    }
}

/// Draw the committed regions and the one being dragged, over the picture.
pub(super) fn draw_rois(
    painter: &egui::Painter,
    win: &PlotWindow,
    origin: Pos2,
    zoom: f32,
    size: (u32, u32),
) {
    for (i, roi) in win.rois.iter().enumerate() {
        let rect = roi_rect(roi, origin, zoom);
        let color = palette(i);
        stroke_roi(painter, rect, roi.shape, Stroke::new(1.5_f32, color));
        // The number, so a line in the legend can be found on the picture.
        painter.text(
            rect.left_top() + Vec2::new(3.0, 1.0),
            egui::Align2::LEFT_TOP,
            format!("{}", i + 1),
            egui::FontId::proportional(12.0),
            color,
        );
    }
    // The drag in flight, thinner so it reads as provisional.
    if let Some((from, to)) = win.drag {
        if let Some(preview) = drag_region(from, to, win.shape, size.0, size.1) {
            stroke_roi(
                painter,
                roi_rect(&preview, origin, zoom),
                preview.shape,
                Stroke::new(1.0_f32, palette(win.rois.len())),
            );
        }
    }
}

// ------------------------------------------------------------------ the chart

/// Draw `plot` into `area`.
fn draw_chart(ui: &mut egui::Ui, area: Rect, plot: &Plot) {
    let painter = ui.painter_at(area);
    let visuals = ui.visuals().clone();
    painter.rect_filled(area, 2.0, visuals.extreme_bg_color);

    let points = plot.points();
    let Some((mut lo, mut hi)) = plot.range().filter(|_| points > 0) else {
        painter.text(
            area.center(),
            egui::Align2::CENTER_CENTER,
            "Nothing to plot",
            egui::FontId::proportional(13.0),
            visuals.weak_text_color(),
        );
        return;
    };
    // A flat trace would be a division by zero and a line along the top edge.
    if (hi - lo).abs() < f32::EPSILON {
        let pad = if hi.abs() > 1.0 { hi.abs() * 0.05 } else { 0.5 };
        lo -= pad;
        hi += pad;
    }

    // Room for the y labels, measured rather than guessed so a six-figure
    // intensity is not clipped.
    let font = egui::FontId::proportional(11.0);
    let label_w = ui
        .ctx()
        .fonts_mut(|f| f.layout_no_wrap(format!("{hi:.0}"), font.clone(), visuals.text_color()))
        .size()
        .x
        .max(28.0);
    let plot_rect = Rect::from_min_max(
        area.min + Vec2::new(label_w + 8.0, 8.0),
        area.max - Vec2::new(8.0, 30.0),
    );
    if !plot_rect.is_positive() {
        return;
    }

    let faint = visuals.weak_text_color();
    let grid = Stroke::new(1.0_f32, faint.gamma_multiply(0.3));
    for k in 0..3 {
        let f = k as f32 / 2.0;
        let y = plot_rect.bottom() - f * plot_rect.height();
        painter.line_segment(
            [
                Pos2::new(plot_rect.left(), y),
                Pos2::new(plot_rect.right(), y),
            ],
            grid,
        );
        painter.text(
            Pos2::new(plot_rect.left() - 5.0, y),
            egui::Align2::RIGHT_CENTER,
            format!("{:.0}", lo + f * (hi - lo)),
            font.clone(),
            faint,
        );
    }

    // The x extent, in the plugin's own units.
    let last = points.saturating_sub(1);
    for (i, align) in [
        (0usize, egui::Align2::LEFT_TOP),
        (last, egui::Align2::RIGHT_TOP),
    ] {
        let f = if last == 0 {
            0.0
        } else {
            i as f32 / last as f32
        };
        painter.text(
            Pos2::new(
                plot_rect.left() + f * plot_rect.width(),
                plot_rect.bottom() + 4.0,
            ),
            align,
            fmt_x(plot.x_at(i)),
            font.clone(),
            faint,
        );
    }
    if !plot.x_label.is_empty() {
        painter.text(
            Pos2::new(plot_rect.center().x, plot_rect.bottom() + 4.0),
            egui::Align2::CENTER_TOP,
            &plot.x_label,
            font.clone(),
            faint,
        );
    }
    if !plot.y_label.is_empty() {
        painter.text(
            Pos2::new(area.left() + 2.0, area.top() + 2.0),
            egui::Align2::LEFT_TOP,
            &plot.y_label,
            font.clone(),
            faint,
        );
    }

    let span = last.max(1) as f32;
    for (i, series) in plot.series.iter().enumerate() {
        let color = series
            .color
            .map(|[r, g, b]| Color32::from_rgb(r, g, b))
            .unwrap_or_else(|| palette(i));
        // Split on non-finite points rather than drawing through them: a gap in
        // the data is a gap, and a line across it asserts a measurement nobody
        // made.
        let mut run: Vec<Pos2> = Vec::new();
        let flush = |run: &mut Vec<Pos2>| {
            if run.len() > 1 {
                painter.add(egui::Shape::line(
                    std::mem::take(run),
                    Stroke::new(1.5_f32, color),
                ));
            } else if let Some(p) = run.first() {
                // A lone point is still a measurement; draw it rather than
                // dropping it.
                painter.circle_filled(*p, 2.0, color);
                run.clear();
            }
        };
        for (k, v) in series.values.iter().enumerate() {
            if v.is_finite() {
                run.push(Pos2::new(
                    plot_rect.left() + k as f32 / span * plot_rect.width(),
                    plot_rect.bottom() - (v - lo) / (hi - lo) * plot_rect.height(),
                ));
            } else {
                flush(&mut run);
                run.clear();
            }
        }
        flush(&mut run);
    }

    painter.rect_stroke(
        plot_rect,
        0.0,
        Stroke::new(1.0_f32, faint.gamma_multiply(0.6)),
        egui::StrokeKind::Middle,
    );
}

/// An x tick, without trailing noise on a whole number.
///
/// A frame index should read `120`, not `120.000`; a calibrated axis in seconds
/// should keep its decimals.
fn fmt_x(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.2}")
    }
}

/// What the window asks the app to do next.
pub(super) enum PlotAction {
    None,
    /// The selection changed; run the plugin again.
    Rerun,
    /// The window was closed.
    Close,
}

/// The plot window.
pub(super) fn plot_window(ctx: &egui::Context, win: &mut PlotWindow, busy: bool) -> PlotAction {
    let mut action = PlotAction::None;
    let mut open = true;
    // Cloned rather than borrowed: the closure below needs `win` mutably, and
    // the title is a handful of characters.
    let title = win.plot.title.clone();
    let spec = dialog::Dialog {
        id: "plugin_plot",
        title: &title,
        // An explicit size: the chart claims the height left over, so the window
        // needs height of its own for it to claim.
        size: egui::vec2(480.0, 320.0) / super::scale::UI_SCALE,
        scroll: false,
        resizable: true,
    };
    dialog::show(ctx, spec, &mut open, |ui| {
        // The tool controls, only when the plugin asked for a selection.
        if win.wants_regions() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Draw:");
                for shape in [Shape::Rect, Shape::Ellipse] {
                    if ui
                        .selectable_label(win.shape == shape, shape.label())
                        .on_hover_text("Drag on the image. Hold Shift to add another.")
                        .clicked()
                    {
                        win.shape = shape;
                    }
                }
                ui.separator();
                let n = win.rois.len();
                ui.label(
                    RichText::new(if n == 0 {
                        "Whole frame".to_string()
                    } else {
                        format!("{n} region{}", if n == 1 { "" } else { "s" })
                    })
                    .weak()
                    .small(),
                );
                if n > 0 {
                    if ui.small_button("Remove last").clicked() {
                        win.rois.pop();
                        action = PlotAction::Rerun;
                    }
                    if ui.small_button("Clear").clicked() {
                        win.rois.clear();
                        action = PlotAction::Rerun;
                    }
                }
                if busy {
                    ui.add(egui::Spinner::new().size(12.0));
                }
            });
            ui.separator();
        }

        let h = (ui.available_height() - 4.0).max(70.0);
        let area = Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), h));
        ui.allocate_rect(area, egui::Sense::hover());
        draw_chart(ui, area, &win.plot);

        // The legend, so a colour can be matched to a region on the picture.
        if win.plot.series.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                for (i, s) in win.plot.series.iter().enumerate() {
                    let c = s
                        .color
                        .map(|[r, g, b]| Color32::from_rgb(r, g, b))
                        .unwrap_or_else(|| palette(i));
                    ui.colored_label(c, "\u{25cf}");
                    ui.label(RichText::new(&s.label).small());
                }
            });
        }
    });
    if !open {
        return PlotAction::Close;
    }
    action
}

#[cfg(test)]
#[path = "plot_tests.rs"]
mod tests;

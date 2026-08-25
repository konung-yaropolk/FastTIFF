//! The 2D navigator: a small two-rectangle map of where the view sits in the
//! image, in the manner of ImageJ's.
//!
//! It answers a question a zoomed-in view cannot: *which part of the picture is
//! this?* At 800% on a mosaic every screenful looks like every other, and the
//! scrollbar-less pan gesture gives no sense of position at all. Two nested
//! rectangles — the frame, and the part of it on screen — answer it at a glance
//! and take a corner of the canvas to do it.
//!
//! Only drawn when it has something to say. With the whole frame visible the
//! inner rectangle is the outer one, which is a box that conveys nothing while
//! still covering a corner of the image.

/// Longest side of the map, in points, and the fraction of the panel it may
/// take. Whichever is smaller wins, so it stays legible on a large canvas and
/// gets out of the way on a small one.
const MAX_SIDE: f32 = 128.0;
const PANEL_FRACTION: f32 = 0.2;

/// Gap between the map and the corner of the canvas.
const MARGIN: f32 = 10.0;

/// Draw the navigator into the top-left of `panel`, given where the whole image
/// would be (`full`) and which part of it is on screen (`visible`).
///
/// Both rects are in screen space, so this needs no knowledge of zoom, pan or
/// UVs — it is the same geometry the image was just drawn from, scaled down.
///
/// **The caller decides whether to draw at all.** It already has to work out
/// whether the image overflows the panel, to know whether dragging should pan,
/// and that answer carries a tolerance for the sub-pixel disagreement between
/// a window's size and the panel inside it. Asking the same question a second
/// time here, from the rects and with a different tolerance, is what made the
/// map appear over images that visibly fitted: the two answers differed by a
/// fraction of a pixel. One question, asked once, in the place that already
/// needs it.
pub(super) fn draw(
    painter: &egui::Painter,
    visuals: &egui::Visuals,
    panel: egui::Rect,
    full: egui::Rect,
    visible: egui::Rect,
) {
    let (fw, fh) = (full.width(), full.height());
    // `is_finite` first, so the comparisons below are meaningful: a NaN
    // extent — which a zoom that went through a bad gesture can produce —
    // compares false against everything, so `fw < 1.0` alone would wave it
    // through and the map would be drawn from nonsense.
    if !fw.is_finite() || !fh.is_finite() || fw < 1.0 || fh < 1.0 || !visible.is_positive() {
        return;
    }
    // The visible fraction of the frame on each axis.
    let (sx, sy) = (visible.width() / fw, visible.height() / fh);
    // Defensive only — the caller has already decided this is worth drawing.
    // An exactly-covering view would put the inner rectangle on top of the
    // outer one, which says nothing while covering a corner of the image.
    if visible.contains_rect(full) {
        return;
    }

    // Aspect-preserving box for the frame.
    let side = (panel.width().min(panel.height()) * PANEL_FRACTION).min(MAX_SIDE);
    let scale = (side / fw).min(side / fh);
    let outer = egui::Rect::from_min_size(
        panel.min + egui::vec2(MARGIN, MARGIN),
        egui::vec2(fw * scale, fh * scale),
    );
    if !outer.is_positive() || !panel.contains_rect(outer) {
        return;
    }

    // Where the view sits inside it, clamped so a view dragged past the edge
    // mid-gesture cannot draw the inner box outside the outer one.
    // Width and height are clamped to a minimum size.
    let inner = egui::Rect::from_min_size(
        outer.min
            + egui::vec2(
                ((visible.min.x - full.min.x) / fw).clamp(0.0, 1.0) * outer.width(),
                ((visible.min.y - full.min.y) / fh).clamp(0.0, 1.0) * outer.height(),
            ),
        egui::vec2(
            (sx.clamp(0.0, 1.0) * outer.width()).max(4.0),
            (sy.clamp(0.0, 1.0) * outer.height()).max(4.0),
        ),
    )
    .intersect(outer);

    // Ink opposite the interface background, so the map reads in either theme.
    // A backdrop behind it as well, because the map sits over the *image*, which
    // can be any brightness — ink alone vanishes against a matching picture.
    let (ink, opposite) = contrasting_ink(visuals);
    let backdrop = opposite.gamma_multiply(0.45);

    painter.rect_filled(outer, 2.0, backdrop);
    painter.rect_stroke(outer, 2.0, egui::Stroke::new(1.0, ink), egui::StrokeKind::Inside);
    // The view box is the thing being read, so it is the heavier of the two.
    painter.rect_filled(inner, 1.0, ink.gamma_multiply(0.25));
    painter.rect_stroke(inner, 1.0, egui::Stroke::new(1.5, ink), egui::StrokeKind::Inside);
}

/// `(ink, opposite)` — black on a light interface and white on a dark one,
/// with the other one for the backdrop behind it.
///
/// Taken from the panel fill rather than from a text colour: the requirement is
/// contrast against the background, and a theme's text colour is only ever
/// *approximately* that — several of egui's are mid-greys.
fn contrasting_ink(visuals: &egui::Visuals) -> (egui::Color32, egui::Color32) {
    let c = visuals.panel_fill;
    // Rec. 601 luma, which is what "how bright does this look" means closely
    // enough for a two-way choice.
    let luma = 0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32;
    if luma < 128.0 {
        (egui::Color32::WHITE, egui::Color32::BLACK)
    } else {
        (egui::Color32::BLACK, egui::Color32::WHITE)
    }
}

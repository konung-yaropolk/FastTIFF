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

/// Stroke widths of the two rectangles, both drawn *inside* their own bounds.
///
/// Named because the view box's minimum size is derived from its own stroke:
/// the two are the same measurement, and letting them drift apart is how the
/// box ends up thinner than the line drawing it.
const OUTER_STROKE: f32 = 1.0;
const INNER_STROKE: f32 = 1.5;

/// Smallest the view box may be drawn, on either axis.
///
/// Its own stroke on both sides plus a pixel of fill between them, so that at
/// any zoom it still reads as a small rectangle rather than a thick dot. Zoomed
/// far enough in, the true fraction of the frame on screen is a small part of
/// one pixel, and something has to be drawn.
const MIN_VIEW_BOX: f32 = 2.0 * INNER_STROKE + 1.0;

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

    let inner = view_box(outer, full, visible);

    // Ink opposite the interface background, so the map reads in either theme.
    // A backdrop behind it as well, because the map sits over the *image*, which
    // can be any brightness — ink alone vanishes against a matching picture.
    let (ink, opposite) = contrasting_ink(visuals);
    let backdrop = opposite.gamma_multiply(0.45);

    painter.rect_filled(outer, 2.0, backdrop);
    painter.rect_stroke(outer, 2.0, egui::Stroke::new(OUTER_STROKE, ink), egui::StrokeKind::Inside);
    // The view box is the thing being read, so it is the heavier of the two.
    painter.rect_filled(inner, 1.0, ink.gamma_multiply(0.25));
    painter.rect_stroke(inner, 1.0, egui::Stroke::new(INNER_STROKE, ink), egui::StrokeKind::Inside);
}

/// Where the view box goes inside the frame box: the same fraction of `outer`
/// that `visible` is of `full`, kept whole and kept inside.
///
/// Two rules, and the second is the one that was missing.
///
/// It is never smaller than [`MIN_VIEW_BOX`], because zoomed far enough in the
/// true fraction is a sliver of a pixel and a box that small is not a box.
///
/// And when that minimum pushes it past an edge it is **slid** back inside, not
/// trimmed to fit. Trimming is the obvious thing — intersect with `outer` and be
/// done — and it breaks precisely where the navigator is most needed: zoomed in
/// near an edge of the image, the box is already at its minimum, so intersecting
/// shaves it thinner than its own outline. It merges into the frame's border,
/// and a little further out it disappears into it altogether.
///
/// The room it is slid within is `outer` less the frame's own stroke, so the two
/// outlines never share ink at all. Stopping at `outer` itself would be within
/// the letter of "overlap by no more than the line thickness", but only just:
/// the rectangles are drawn with rounded corners, and epaint widens a corner
/// radius to at least the stroke width, so a box flush into a corner puts about
/// a third of a point of its outline *outside* the frame's arc — the same
/// "partially goes outside" in miniature. A stroke's worth of clearance costs
/// under 1% of the map's width in positional accuracy and removes the question.
fn view_box(outer: egui::Rect, full: egui::Rect, visible: egui::Rect) -> egui::Rect {
    let (fw, fh) = (full.width(), full.height());
    // The room inside the frame's outline.
    //
    // The inset is capped at half the map on either axis, so a map smaller than
    // its own border collapses to a point at its centre rather than turning
    // inside out — an unguarded subtraction gives a negative extent, and a
    // rectangle of negative width is one epaint will tessellate happily and
    // wrongly. Reachable: the map is a fifth of the panel's shorter side, so a
    // small enough window gets one only a couple of points tall.
    let inset = OUTER_STROKE.min(outer.width() * 0.5).min(outer.height() * 0.5).max(0.0);
    let room = outer.shrink(inset);
    let size = egui::vec2(
        (visible.width() / fw).clamp(0.0, 1.0) * room.width(),
        (visible.height() / fh).clamp(0.0, 1.0) * room.height(),
    );
    // `max` then `min`, never `clamp`: on a frame so long and thin that `room`
    // is itself under the minimum on one axis, `clamp` would be handed a low
    // bound above its high one and panic.
    let size = egui::vec2(
        size.x.max(MIN_VIEW_BOX).min(room.width()),
        size.y.max(MIN_VIEW_BOX).min(room.height()),
    );
    let at = egui::vec2(
        ((visible.min.x - full.min.x) / fw).clamp(0.0, 1.0) * room.width(),
        ((visible.min.y - full.min.y) / fh).clamp(0.0, 1.0) * room.height(),
    );
    // The upper bounds are floored at `room.min` for the same reason.
    egui::Rect::from_min_size(
        egui::pos2(
            (room.min.x + at.x).clamp(room.min.x, (room.max.x - size.x).max(room.min.x)),
            (room.min.y + at.y).clamp(room.min.y, (room.max.y - size.y).max(room.min.y)),
        ),
        size,
    )
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

#[cfg(test)]
#[path = "minimap_tests.rs"]
mod minimap_tests;

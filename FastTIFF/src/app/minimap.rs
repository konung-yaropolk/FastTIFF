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

/// Smallest the frame box may be drawn, on either axis.
///
/// Derived rather than chosen, from what has to fit inside it: the frame's own
/// outline on both sides, and then a view box at its own minimum in the room
/// that leaves. Anything less and the map cannot show the thing it exists to
/// show. (`view_box` insets by exactly [`OUTER_STROKE`] per side, so the two
/// terms here are its `room` run backwards.)
///
/// # Why a minimum at all
///
/// An aspect-preserving map of an extremely elongated frame is invisible. A
/// 20000 x 4 image in a 1200 x 800 panel takes `side = 128`, so
/// `scale = 128 / 20000`, and the map comes out **128 x 0.026 points** — a
/// hairline thinner than the stroke drawing it, with a view box of literally
/// zero height inside. The navigator was there, drawn, and unreadable.
///
/// # What it costs
///
/// The map stops being a scale model: at 20000 x 4 it is drawn 128 x 6, an
/// aspect of 21:1 standing in for 5000:1. That is a real loss and it is the
/// right trade, because the alternative is a map nobody can see.
///
/// The loss is confined to *shape*. [`view_box`] works in fractions of the
/// frame, so where the box sits stays exactly truthful — a view halfway across
/// the image still puts the box halfway across the map. And the box's own size
/// was already clamped to [`MIN_VIEW_BOX`] at this sort of aspect, so the
/// distortion takes nothing from it that the clamp had not already taken.
///
/// Which is also why the distortion is left uncapped. Capping the aspect would
/// mean shrinking the long axis to keep the ratio honest, trading away the
/// positional precision that *is* still truthful to protect a shape that
/// cannot be.
const MIN_FRAME_SIDE: f32 = 2.0 * OUTER_STROKE + MIN_VIEW_BOX;

/// Draw the navigator into `outer`, given where the whole image would be
/// (`full`) and which part of it is on screen (`visible`).
///
/// All three rects are in screen space, so this needs no knowledge of zoom, pan
/// or UVs — it is the same geometry the image was just drawn from, scaled down.
///
/// **The caller decides whether to draw at all**, and supplies `outer` from
/// [`frame_box`] rather than letting this derive it. Two reasons, and the second
/// arrived later.
///
/// The caller already has to work out whether the image overflows the panel, to
/// know whether dragging should pan, and that answer carries a tolerance for the
/// sub-pixel disagreement between a window's size and the panel inside it.
/// Asking the same question a second time here, from the rects and with a
/// different tolerance, is what made the map appear over images that visibly
/// fitted: the two answers differed by a fraction of a pixel.
///
/// And the map is now clickable, so the caller needs the frame box *before* it
/// draws — a press has to be tested against it early enough to stop the image
/// panning underneath. Passing it in is what guarantees the rectangle a click is
/// measured against and the rectangle on screen are the same one.
pub(super) fn draw(
    painter: &egui::Painter,
    visuals: &egui::Visuals,
    outer: egui::Rect,
    full: egui::Rect,
    visible: egui::Rect,
) {
    if !visible.is_positive() {
        return;
    }
    // Defensive only — the caller has already decided this is worth drawing.
    // An exactly-covering view would put the inner rectangle on top of the
    // outer one, which says nothing while covering a corner of the image.
    if visible.contains_rect(full) {
        return;
    }

    let inner = view_box(outer, full, visible);

    // Ink opposite the interface background, so the map reads in either theme.
    // A backdrop behind it as well, because the map sits over the *image*, which
    // can be any brightness — ink alone vanishes against a matching picture.
    let (ink, opposite) = contrasting_ink(visuals);
    let backdrop = opposite.gamma_multiply(0.45);

    painter.rect_filled(outer, 2.0, backdrop);
    painter.rect_stroke(
        outer,
        2.0,
        egui::Stroke::new(OUTER_STROKE, ink),
        egui::StrokeKind::Inside,
    );
    // The view box is the thing being read, so it is the heavier of the two.
    painter.rect_filled(inner, 1.0, ink.gamma_multiply(0.25));
    painter.rect_stroke(
        inner,
        1.0,
        egui::Stroke::new(INNER_STROKE, ink),
        egui::StrokeKind::Inside,
    );
}

/// Where the frame box goes and how big it is, or `None` if there is no room
/// for one.
///
/// Aspect-preserving, up to a point: scaled so the longer side is `side` and
/// then floored at [`MIN_FRAME_SIDE`] on both axes, which is what keeps an
/// extremely elongated frame's map visible. See that constant for the trade.
///
/// `None` rather than a degenerate rect when it will not fit the panel, which
/// is the honest answer for a window too small to hold a navigator — the same
/// answer the caller got before, and the reason this returns an `Option` rather
/// than clamping itself into a space it does not have.
///
/// Pure, and separate from [`draw`] for the same reason [`view_box`] is: the
/// arithmetic is where the bug was, and it should be checkable without a
/// display, at aspect ratios nobody is going to sit down and reproduce by hand.
pub(super) fn frame_box(panel: egui::Rect, fw: f32, fh: f32) -> Option<egui::Rect> {
    // Checked here rather than by the caller because this is where they are
    // divided by, and checked with `is_finite` first so the comparisons that
    // follow are meaningful: a NaN extent — which a zoom through a bad gesture
    // can produce — compares false against everything, so `fw < 1.0` alone
    // would wave it through.
    //
    // Floating-point `min`/`max` make that worse rather than better, and this
    // is worth knowing because the obvious reading is the wrong way round:
    // IEEE-754 `max` *swallows* a NaN, returning the other operand. So on a NaN
    // width the clamp below would not propagate the NaN — it would quietly
    // substitute `MIN_FRAME_SIDE` and hand back a perfectly plausible map of an
    // image whose size is not a number. A map drawn from nonsense that looks
    // like a map is worse than none, so the guard cannot be left to the
    // arithmetic.
    if !fw.is_finite() || !fh.is_finite() || fw < 1.0 || fh < 1.0 {
        return None;
    }
    let side = (panel.width().min(panel.height()) * PANEL_FRACTION).min(MAX_SIDE);
    let scale = (side / fw).min(side / fh);
    let size = egui::vec2(
        (fw * scale).max(MIN_FRAME_SIDE),
        (fh * scale).max(MIN_FRAME_SIDE),
    );
    let outer = egui::Rect::from_min_size(panel.min + egui::vec2(MARGIN, MARGIN), size);
    (outer.is_positive() && panel.contains_rect(outer)).then_some(outer)
}

/// The room inside the frame's outline — the rectangle the whole image maps
/// onto, and the one the view box is confined to.
///
/// The inset is capped at half the map on either axis, so a map smaller than
/// its own border collapses to a point at its centre rather than turning inside
/// out — an unguarded subtraction gives a negative extent, and a rectangle of
/// negative width is one epaint will tessellate happily and wrongly. Reachable:
/// the map is a fifth of the panel's shorter side, so a small enough window
/// gets one only a couple of points tall.
///
/// Shared by [`view_box`] and [`pan_for`] because it is the coordinate system
/// both work in, and a click has to land where the box is drawn. Two copies of
/// this inset that drifted apart would put the box a point or two from the
/// cursor — which is the kind of thing that reads as the map being imprecise
/// rather than as a bug.
fn room_in(outer: egui::Rect) -> egui::Rect {
    let inset = OUTER_STROKE
        .min(outer.width() * 0.5)
        .min(outer.height() * 0.5)
        .max(0.0);
    outer.shrink(inset)
}

/// The pan that puts the point `at` of the map at the centre of the view.
///
/// The inverse of [`view_box`]: that turns a pan into a box, this turns a point
/// into the pan that would put the box there. `frame` is the image's size at the
/// current zoom (what `full.size()` is to `view_box`) and `view` is how much of
/// it is on screen.
///
/// # Centred, not cornered
///
/// A click says "show me *there*", and the natural reading of a point is the
/// middle of what you get, not its top-left corner — cornering would put the
/// thing you clicked at the edge of the screen, or off it. Half the view is
/// therefore subtracted, which is also what makes dragging feel like moving the
/// box rather than dragging its corner around.
///
/// # Unclamped, deliberately
///
/// The result can be negative or past the end of the image, and is handed back
/// that way. The caller clamps `pan` to the overflow on every frame regardless
/// — it has to, for every other way the pan can change — so clamping here as
/// well would be the same question answered twice, and the second answer would
/// be the one computed without knowing the panel.
///
/// That also handles an axis with nothing to pan: its overflow is zero, so the
/// caller's clamp pins the pan to zero and the image stays centred. Which is
/// right — on such an axis the whole image is already on screen, and there is
/// no part of it for the map to move to.
pub(super) fn pan_for(
    outer: egui::Rect,
    frame: egui::Vec2,
    view: egui::Vec2,
    at: egui::Pos2,
) -> egui::Vec2 {
    let room = room_in(outer);
    // A zero-extent axis — a map collapsed to a line by the cap in `room_in` —
    // has no fraction to read, and 0 is the only answer that is not a division
    // by zero. The caller's clamp makes it the image's own origin.
    let frac = |p: f32, lo: f32, len: f32| {
        if len > 0.0 {
            ((p - lo) / len).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    egui::vec2(
        frac(at.x, room.min.x, room.width()) * frame.x - view.x * 0.5,
        frac(at.y, room.min.y, room.height()) * frame.y - view.y * 0.5,
    )
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
    let room = room_in(outer);
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
pub(super) fn contrasting_ink(visuals: &egui::Visuals) -> (egui::Color32, egui::Color32) {
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

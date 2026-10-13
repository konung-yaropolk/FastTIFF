//! Where the navigator's view box goes.
//!
//! The map is two nested rectangles, and the inner one is the whole point: it
//! says which part of the picture is on screen. It is also the one that is
//! hardest to draw, because the further you zoom in the smaller it gets, and it
//! is smallest exactly when it is being relied on most.
//!
//! Two things have to hold at every zoom and every position, and the tests here
//! are all one or the other:
//!
//! - it stays **readable** — never thinner than the line that draws it;
//! - it stays **inside** — never hanging out of the frame box, and never even
//!   reaching the frame's own outline, so the two can never merge.
//!
//! The bug these were written for satisfied the first only in the middle of the
//! map: near an edge the box was trimmed against the boundary instead of being
//! slid back in, so the minimum size was quietly taken away again and the box
//! thinned into the border and then vanished.

use super::*;

/// The frame box, as `draw` builds it: a 128 x 80 map in the corner.
fn outer() -> egui::Rect {
    egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(128.0, 80.0))
}

/// The room the view box is confined to: the map less the frame's own outline,
/// so the two never share ink. Mirrors `view_box`'s own inset.
fn room() -> egui::Rect {
    outer().shrink(OUTER_STROKE)
}

/// A 4000 x 2500 image, drawn somewhere on screen at some zoom.
fn full() -> egui::Rect {
    egui::Rect::from_min_size(egui::pos2(-3000.0, -1200.0), egui::vec2(4000.0, 2500.0))
}

/// The visible rect for a view `frac` of the frame across, with its top-left at
/// `(u, v)` as a fraction of the frame.
fn visible(u: f32, v: f32, frac: f32) -> egui::Rect {
    let f = full();
    egui::Rect::from_min_size(
        f.min + egui::vec2(f.width() * u, f.height() * v),
        egui::vec2(f.width() * frac, f.height() * frac),
    )
}

/// Every corner and edge of the map, plus the middle, at a given zoom.
fn positions() -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for v in [0.0f32, 0.5, 1.0] {
        for u in [0.0f32, 0.5, 1.0] {
            out.push((u, v));
        }
    }
    // And a few just off the edge, which a drag past the boundary produces.
    out.extend([(-0.2, 0.5), (1.2, 0.5), (0.5, -0.2), (0.5, 1.2), (1.5, 1.5)]);
    out
}

/// The headline property: wherever the view is and however far it is zoomed in,
/// the box is inside the map and still big enough to see.
#[test]
fn the_view_box_is_always_inside_and_always_readable() {
    let o = outer();
    for frac in [0.9f32, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.000_001] {
        for (u, v) in positions() {
            let b = view_box(o, full(), visible(u, v, frac));
            let case = format!("frac {frac}, at ({u}, {v}) -> {b:?}");

            assert!(b.width() >= MIN_VIEW_BOX - 1e-3, "too thin: {case}");
            assert!(b.height() >= MIN_VIEW_BOX - 1e-3, "too short: {case}");

            let r = room();
            assert!(b.min.x >= r.min.x - 1e-3, "hangs off the left: {case}");
            assert!(b.min.y >= r.min.y - 1e-3, "hangs off the top: {case}");
            assert!(b.max.x <= r.max.x + 1e-3, "hangs off the right: {case}");
            assert!(b.max.y <= r.max.y + 1e-3, "hangs off the bottom: {case}");
        }
    }
}

/// The specific failure that prompted this: zoomed right in, against an edge.
///
/// The old code positioned the box and then intersected it with the map, so a
/// box already at its minimum was trimmed by however far it stuck out — down to
/// nothing in the corner. Sliding it back in keeps it whole.
#[test]
fn a_tiny_view_box_against_the_edge_keeps_its_size() {
    let o = outer();
    // A view a ten-thousandth of the frame across, hard against the far corner.
    let b = view_box(o, full(), visible(1.0, 1.0, 0.000_1));
    assert!(
        (b.width() - MIN_VIEW_BOX).abs() < 1e-3 && (b.height() - MIN_VIEW_BOX).abs() < 1e-3,
        "the box was trimmed instead of slid: {b:?}"
    );
    // Slid flush against the corner of the room inside the frame's outline —
    // not past it, and not onto the outline.
    let r = room();
    assert!(
        (b.max.x - r.max.x).abs() < 1e-3,
        "should sit against the right edge: {b:?}"
    );
    assert!(
        (b.max.y - r.max.y).abs() < 1e-3,
        "should sit against the bottom edge: {b:?}"
    );
}

/// The two outlines never share ink. Both are drawn *inside* their own
/// rectangles, so keeping the view box out of the frame's stroke entirely is
/// what makes "overlap by no more than the line thickness" true with room to
/// spare — and, unlike stopping at `outer`, it survives the rounded corners,
/// where epaint widens the radius to at least the stroke and a flush box would
/// otherwise put a fraction of a point of ink outside the frame's arc.
#[test]
fn the_view_box_never_reaches_the_frame_outline() {
    let o = outer();
    for (u, v) in positions() {
        for frac in [0.5f32, 0.05, 0.000_1] {
            let b = view_box(o, full(), visible(u, v, frac));
            let case = format!("frac {frac}, at ({u}, {v}) -> {b:?}");
            // How far the view box reaches into the frame's stroke, per side.
            // Never at all.
            for (into, side) in [
                (o.min.x + OUTER_STROKE - b.min.x, "left"),
                (o.min.y + OUTER_STROKE - b.min.y, "top"),
                (b.max.x - (o.max.x - OUTER_STROKE), "right"),
                (b.max.y - (o.max.y - OUTER_STROKE), "bottom"),
            ] {
                assert!(
                    into <= 1e-3,
                    "{side} side reaches {into} into the frame outline: {case}"
                );
            }
        }
    }
}

/// Zoomed out far enough that the view is the whole frame, the box is the map.
/// (`draw` returns before this, but the geometry must not depend on that.)
#[test]
fn a_view_of_the_whole_frame_fills_the_map() {
    let r = room();
    let b = view_box(outer(), full(), full());
    assert!((b.width() - r.width()).abs() < 1e-3, "{b:?}");
    assert!((b.height() - r.height()).abs() < 1e-3, "{b:?}");
    assert!((b.min - r.min).length() < 1e-3, "{b:?}");
}

/// A view larger than the frame — which a zoom-out past the fitted level
/// produces — is the map, not something bigger than it.
#[test]
fn a_view_larger_than_the_frame_does_not_overflow_the_map() {
    let o = outer();
    let f = full();
    let huge = egui::Rect::from_min_size(
        f.min - egui::vec2(f.width(), f.height()),
        egui::vec2(f.width() * 3.0, f.height() * 3.0),
    );
    let b = view_box(o, f, huge);
    assert!(o.contains_rect(b), "{b:?} is not inside {o:?}");
}

/// The box tracks the view: sliding the view right moves the box right, and it
/// reaches the far edge only when the view does.
#[test]
fn the_box_tracks_the_view_across_the_map() {
    let o = outer();
    let mut last = f32::NEG_INFINITY;
    for u in [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let b = view_box(o, full(), visible(u, 0.0, 0.1));
        assert!(
            b.min.x >= last - 1e-3,
            "moving the view right moved the box left at u={u}"
        );
        last = b.min.x;
    }
    let r = room();
    let leftmost = view_box(o, full(), visible(0.0, 0.0, 0.1));
    let rightmost = view_box(o, full(), visible(0.9, 0.0, 0.1));
    assert!(
        (leftmost.min.x - r.min.x).abs() < 1e-3,
        "should start flush left"
    );
    assert!(
        (rightmost.max.x - r.max.x).abs() < 1e-3,
        "should end flush right"
    );
}

/// A map narrower than the minimum on one axis — a very long, thin image — must
/// not panic. `clamp` with a low bound above its high one does; `max` then `min`
/// does not.
#[test]
fn a_map_thinner_than_the_minimum_does_not_panic() {
    let thin = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(128.0, 2.0));
    let b = view_box(thin, full(), visible(0.5, 0.5, 0.001));
    assert!(
        b.height() <= thin.height() + 1e-3,
        "cannot be taller than the map: {b:?}"
    );
    assert!(thin.contains_rect(b), "{b:?} is not inside {thin:?}");
}

/// A map smaller than its own outline has no room inside it at all. Subtracting
/// the border unguarded gives a negative extent, and a rectangle with negative
/// width is not a rectangle — epaint will happily tessellate one inside out.
/// There is nothing sensible to draw here, but it must be a degenerate rect
/// inside the map rather than a wrong one.
#[test]
fn a_map_smaller_than_its_own_border_yields_nothing_negative() {
    for side in [0.0f32, 0.5, 1.0, 1.9] {
        let tiny = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(side, side));
        let b = view_box(tiny, full(), visible(0.5, 0.5, 0.001));
        assert!(b.width() >= 0.0, "negative width on a {side}pt map: {b:?}");
        assert!(
            b.height() >= 0.0,
            "negative height on a {side}pt map: {b:?}"
        );
        assert!(
            b.min.x >= tiny.min.x - 1e-3 && b.max.x <= tiny.max.x + 1e-3,
            "{b:?} left {tiny:?}"
        );
        assert!(
            b.min.y >= tiny.min.y - 1e-3 && b.max.y <= tiny.max.y + 1e-3,
            "{b:?} left {tiny:?}"
        );
    }
}

// ------------------------------------------------- the frame box's own size
//
// The view box above is confined to a frame box, and these are about that
// frame box existing at all. The bug: an aspect-preserving map of a 20000 x 4
// image is 128 x 0.026 points, which is thinner than the line drawing it.

/// A realistic desktop panel, big enough that `MAX_SIDE` is what caps the map.
fn panel() -> egui::Rect {
    egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0))
}

/// The reported case, and its transpose. The map is squashed — it has to be,
/// there is no honest way to draw 5000:1 in 128 points — but it is *there*.
#[test]
fn an_extremely_elongated_frame_still_gets_a_map_with_size_in_both_axes() {
    for (fw, fh, what) in [
        (20000.0, 4.0, "very wide"),
        (4.0, 20000.0, "very tall"),
        // A 10-megapixel-wide line scan, which is the same shape taken further.
        (10_000_000.0, 5.0, "absurdly wide"),
        (1.0, 1_000_000.0, "a single column"),
    ] {
        let b = frame_box(panel(), fw, fh).unwrap_or_else(|| panic!("no map for {what}"));
        assert!(
            b.width() >= MIN_FRAME_SIDE - 1e-3,
            "{what}: map {b:?} is narrower than the minimum"
        );
        assert!(
            b.height() >= MIN_FRAME_SIDE - 1e-3,
            "{what}: map {b:?} is shorter than the minimum"
        );
    }
}

/// The guarantee the minimum exists to buy, end to end: whatever the aspect,
/// the map is big enough that the view box inside it is still a rectangle.
///
/// This is the one that would have caught the bug. `view_box` was already
/// correct — it clamps to `MIN_VIEW_BOX` — but it was being handed a frame box
/// with no room in it, and `min(room.height())` then took the clamp straight
/// back off. Testing either function alone misses that; the two have to be
/// composed, as `draw` composes them.
#[test]
fn the_map_of_any_frame_has_room_for_a_readable_view_box() {
    for (fw, fh) in [
        (20000.0, 4.0),
        (4.0, 20000.0),
        (10_000_000.0, 5.0),
        (4000.0, 2500.0),
        (512.0, 512.0),
        (1.0, 1.0),
    ] {
        let o = frame_box(panel(), fw, fh).expect("a map");
        // Zoomed far in, which is when the view box is at its smallest.
        let f = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(fw, fh));
        let v = egui::Rect::from_min_size(f.min, egui::vec2(fw * 0.0001, fh * 0.0001));
        let b = view_box(o, f, v);
        assert!(
            b.width() >= MIN_VIEW_BOX - 1e-3 && b.height() >= MIN_VIEW_BOX - 1e-3,
            "{fw}x{fh}: view box {b:?} in map {o:?} is not readable"
        );
        assert!(o.contains_rect(b), "{b:?} is not inside {o:?}");
    }
}

/// The minimum is a floor, not a resize: a frame of ordinary proportions is
/// still drawn to scale, because that is the whole value of the map.
#[test]
fn an_ordinary_frame_keeps_its_aspect_ratio() {
    for (fw, fh) in [(4000.0f32, 2500.0f32), (512.0, 512.0), (1920.0, 1080.0)] {
        let b = frame_box(panel(), fw, fh).expect("a map");
        let want = fw / fh;
        let got = b.width() / b.height();
        assert!(
            (got - want).abs() < 1e-2,
            "{fw}x{fh}: map {b:?} has aspect {got}, not {want}"
        );
        // And the longer side is what `MAX_SIDE` capped, so nothing was floored.
        assert!(
            (b.width().max(b.height()) - MAX_SIDE).abs() < 1e-3,
            "{fw}x{fh}: map {b:?} should be MAX_SIDE on its long side"
        );
    }
}

/// Position is the part that stays truthful when the shape cannot, so it is
/// worth asserting on exactly the frame whose shape is sacrificed.
#[test]
fn the_box_still_tracks_the_view_on_a_squashed_map() {
    let (fw, fh) = (20000.0f32, 4.0f32);
    let o = frame_box(panel(), fw, fh).expect("a map");
    let f = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(fw, fh));
    let at = |u: f32| {
        view_box(
            o,
            f,
            egui::Rect::from_min_size(
                f.min + egui::vec2(fw * u, 0.0),
                egui::vec2(fw * 0.01, fh * 0.5),
            ),
        )
    };
    // Halfway across the image is halfway across the map, to within the box's
    // own width — the fraction is exact, the box has size.
    let mid = at(0.5);
    let room_w = o.width() - 2.0 * OUTER_STROKE;
    let expected = o.min.x + OUTER_STROKE + room_w * 0.5;
    assert!(
        (mid.min.x - expected).abs() < 1.0,
        "the box should sit at the middle of the map: {mid:?} vs {expected}"
    );
    // And it is monotonic across the whole width.
    let mut last = f32::NEG_INFINITY;
    for u in [0.0f32, 0.25, 0.5, 0.75, 0.99] {
        let x = at(u).min.x;
        assert!(x >= last - 1e-3, "went backwards at u={u}");
        last = x;
    }
}

/// A window too small to hold a map gets none, rather than one drawn over the
/// image or outside the panel. Floored sizes make this reachable where
/// aspect-preserving ones were not: the map no longer shrinks without limit.
#[test]
fn a_panel_with_no_room_for_a_map_gets_none() {
    for side in [1.0f32, 8.0, 15.0] {
        let tiny = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(side, side));
        assert!(
            frame_box(tiny, 20000.0, 4.0).is_none(),
            "a {side}pt panel should get no map"
        );
    }
}

/// Whatever the frame's shape, the map stays within the panel it was asked to
/// sit in — the invariant the caller relies on to clip nothing.
#[test]
fn the_map_never_leaves_the_panel() {
    for p in [
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0)),
        egui::Rect::from_min_size(egui::pos2(37.0, 91.0), egui::vec2(300.0, 200.0)),
        egui::Rect::from_min_size(egui::pos2(-50.0, -50.0), egui::vec2(60.0, 60.0)),
    ] {
        for (fw, fh) in [(20000.0, 4.0), (4.0, 20000.0), (4000.0, 2500.0), (1.0, 1.0)] {
            if let Some(b) = frame_box(p, fw, fh) {
                assert!(p.contains_rect(b), "{b:?} is not inside {p:?}");
                assert!(b.is_positive(), "{b:?} is degenerate");
            }
        }
    }
}

/// A zoom through a bad gesture can hand this a non-finite extent, and the
/// floor cannot be relied on to deal with it — the wrong way round from the
/// obvious reading. IEEE-754 `max` *swallows* a NaN and returns the other
/// operand, so `(NaN).max(MIN_FRAME_SIDE)` is `MIN_FRAME_SIDE`: without an
/// explicit guard this would hand back a plausible-looking map of an image
/// whose width is not a number. This test is why that guard exists.
#[test]
fn a_non_finite_frame_extent_yields_no_map() {
    for bad in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
        assert!(
            frame_box(panel(), bad, 100.0).is_none(),
            "width {bad} should get no map"
        );
        assert!(
            frame_box(panel(), 100.0, bad).is_none(),
            "height {bad} should get no map"
        );
    }
}

// ------------------------------------------------------ clicking on the map
//
// `pan_for` is the inverse of `view_box`: that turns a pan into a box, this
// turns a point into the pan that would put the box there. Most of what is
// below is that relationship, asserted directly — it is the property the
// gesture actually depends on, and the one that would break silently if either
// function's idea of `room` drifted from the other's.

/// The pan the caller ends up with: `pan_for`, then the clamp `app.rs` applies
/// to `view.pan` on every frame however it changed.
fn clamped_pan(
    outer: egui::Rect,
    frame: egui::Vec2,
    view: egui::Vec2,
    at: egui::Pos2,
) -> egui::Vec2 {
    let p = pan_for(outer, frame, view, at);
    let overflow = egui::vec2((frame.x - view.x).max(0.0), (frame.y - view.y).max(0.0));
    egui::vec2(p.x.clamp(0.0, overflow.x), p.y.clamp(0.0, overflow.y))
}

/// The view box that results from panning to `pan` — the geometry `app.rs`
/// builds, reduced to what `view_box` needs.
fn box_after(
    outer: egui::Rect,
    frame: egui::Vec2,
    view: egui::Vec2,
    pan: egui::Vec2,
) -> egui::Rect {
    // `full.min` is the panel origin less the pan, so `visible.min - full.min`
    // is the pan — which is the relationship `view_box` reads.
    let full = egui::Rect::from_min_size(egui::pos2(-pan.x, -pan.y), frame);
    let visible = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), view);
    view_box(outer, full, visible)
}

/// The headline property: click the map, and the box lands under the pointer.
///
/// Checked away from the edges, where the clamps have nothing to do — at an
/// edge the box stops at the boundary while the pointer can keep going, which
/// is correct and is asserted separately below.
#[test]
fn clicking_the_map_puts_the_view_box_under_the_pointer() {
    let frame = egui::vec2(4000.0, 2500.0);
    let view = egui::vec2(800.0, 600.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let r = room_in(o);
    for u in [0.3f32, 0.4, 0.5, 0.6, 0.7] {
        for v in [0.3f32, 0.5, 0.7] {
            let at = egui::pos2(r.min.x + r.width() * u, r.min.y + r.height() * v);
            let b = box_after(o, frame, view, clamped_pan(o, frame, view, at));
            let off = (b.center() - at).length();
            assert!(
                off < 1.5,
                "clicked ({u}, {v}) at {at:?} but the box centred at {:?} ({off} off)",
                b.center()
            );
        }
    }
}

/// Centred, not cornered: clicking the middle of the map shows the middle of
/// the image. Cornering would put what you clicked at the edge of the screen.
#[test]
fn clicking_the_centre_of_the_map_centres_the_image() {
    let frame = egui::vec2(4000.0, 2500.0);
    let view = egui::vec2(800.0, 600.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let p = clamped_pan(o, frame, view, room_in(o).center());
    assert!(
        ((p.x + view.x * 0.5) - frame.x * 0.5).abs() < 2.0,
        "horizontal centre is off: pan {p:?}"
    );
    assert!(
        ((p.y + view.y * 0.5) - frame.y * 0.5).abs() < 2.0,
        "vertical centre is off: pan {p:?}"
    );
}

/// The ends reach the ends. A map you cannot use to get to the corner of the
/// image has a dead margin, which near an edge is worse than no map.
#[test]
fn clicking_the_corners_of_the_map_reaches_the_corners_of_the_image() {
    let frame = egui::vec2(20000.0, 9000.0);
    let view = egui::vec2(800.0, 600.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let r = room_in(o);
    let overflow = egui::vec2(frame.x - view.x, frame.y - view.y);

    let top_left = clamped_pan(o, frame, view, r.min);
    assert_eq!(
        (top_left.x, top_left.y),
        (0.0, 0.0),
        "the top-left of the map must be the top-left of the image"
    );
    let bottom_right = clamped_pan(o, frame, view, r.max);
    assert!(
        (bottom_right.x - overflow.x).abs() < 1e-3 && (bottom_right.y - overflow.y).abs() < 1e-3,
        "the bottom-right of the map must be the bottom-right of the image: \
         {bottom_right:?} vs {overflow:?}"
    );
}

/// Dragging the box past the edge of the map keeps going to the edge of the
/// image and stops there, rather than wrapping or jumping back.
#[test]
fn a_pointer_dragged_off_the_map_clamps_to_the_edge_of_the_image() {
    let frame = egui::vec2(4000.0, 2500.0);
    let view = egui::vec2(800.0, 600.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let r = room_in(o);
    let overflow = frame.x - view.x;
    for (at, want) in [
        (egui::pos2(r.min.x - 500.0, r.center().y), 0.0),
        (egui::pos2(r.max.x + 500.0, r.center().y), overflow),
    ] {
        let p = clamped_pan(o, frame, view, at);
        assert!(
            (p.x - want).abs() < 1e-3,
            "dragged to {at:?} should give pan.x {want}, got {}",
            p.x
        );
    }
}

/// The map is monotonic along both axes: moving the pointer right never moves
/// the view left. Obvious, and exactly what a sign error in the centring offset
/// would break while every centred-case test above still passed.
#[test]
fn moving_the_pointer_across_the_map_moves_the_view_the_same_way() {
    let frame = egui::vec2(4000.0, 2500.0);
    let view = egui::vec2(800.0, 600.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let r = room_in(o);
    let (mut lastx, mut lasty) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for t in [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let p = clamped_pan(
            o,
            frame,
            view,
            egui::pos2(r.min.x + r.width() * t, r.min.y + r.height() * t),
        );
        assert!(p.x >= lastx - 1e-3, "pan.x went backwards at t={t}");
        assert!(p.y >= lasty - 1e-3, "pan.y went backwards at t={t}");
        (lastx, lasty) = (p.x, p.y);
    }
}

/// The squashed map is still a usable map. Its shape is a lie (see
/// [`MIN_FRAME_SIDE`]) but its *positions* are not, and this is the test that
/// says so for the case the whole minimum exists to serve.
#[test]
fn a_squashed_map_still_reaches_the_whole_image() {
    let frame = egui::vec2(20000.0, 4.0);
    let view = egui::vec2(800.0, 4.0);
    let o = frame_box(panel(), frame.x, frame.y).expect("a map");
    let r = room_in(o);
    let overflow = frame.x - view.x;

    assert_eq!(clamped_pan(o, frame, view, r.min).x, 0.0);
    assert!((clamped_pan(o, frame, view, r.max).x - overflow).abs() < 1e-3);
    // The middle is the middle, to within one of the map's own points — which
    // on a 126-point map of a 20000-pixel image is about 160 pixels, and is all
    // the precision a map that size can possibly carry.
    let mid = clamped_pan(o, frame, view, r.center()).x;
    let want = (frame.x * 0.5 - view.x * 0.5).clamp(0.0, overflow);
    assert!(
        (mid - want).abs() < frame.x / r.width(),
        "middle of the map gave {mid}, wanted about {want}"
    );
    // The vertical axis has nothing to pan — the image is 4 pixels tall and all
    // of it is on screen — so the caller's clamp pins it to zero wherever the
    // click lands. Which is the right answer, not a lost gesture: there is no
    // other vertical position for the view to be in.
    for t in [0.0f32, 0.5, 1.0] {
        let at = egui::pos2(r.center().x, r.min.y + r.height() * t);
        let p = clamped_pan(o, frame, view, at);
        assert_eq!(p.y, 0.0, "nothing to pan vertically, yet pan.y is {}", p.y);
    }
}

/// A map collapsed to a line by [`room_in`]'s cap has no fraction to read along
/// the collapsed axis. Dividing by its zero extent gives a NaN pan, and a NaN
/// survives the caller's `clamp` — which returns NaN rather than a bound — and
/// would poison the transform for every frame after.
#[test]
fn a_map_with_a_collapsed_axis_yields_a_usable_pan() {
    let frame = egui::vec2(4000.0, 2500.0);
    let view = egui::vec2(800.0, 600.0);
    for side in [0.0f32, 1.0, 2.0] {
        let flat = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(128.0, side));
        let p = pan_for(flat, frame, view, egui::pos2(50.0, 10.0));
        assert!(
            p.x.is_finite() && p.y.is_finite(),
            "a {side}pt-tall map gave a non-finite pan: {p:?}"
        );
    }
}

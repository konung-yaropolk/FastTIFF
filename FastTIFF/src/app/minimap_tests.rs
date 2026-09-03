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

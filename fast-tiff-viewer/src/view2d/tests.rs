//! A frame of the 2D view: what a new image resets, how the image is laid out
//! and placed in its panel, and the gestures that move it.
//!
//! The arithmetic itself is checked against the code it was moved from — the
//! app's, in egui's types — by `FastTIFF/src/app/view2d_oracle_tests.rs`, on
//! grids of inputs. These pin what each step promises.

use super::*;

const PANEL: LogicalRect = LogicalRect {
    min: LogicalPt::new(0.0, 30.0),
    max: LogicalPt::new(640.0, 510.0),
};
const ROOM: LogicalVec = LogicalVec::new(640.0, 480.0);

/// A view with every field moved off its fresh value.
fn used_view() -> View2d {
    let mut glide = kinetic::Glide::default();
    glide.push(LogicalVec::new(40.0, -25.0), 1.0 / 60.0);
    assert!(glide.is_moving(), "the fixture's glide should be moving");
    View2d {
        zoom: 3.0,
        pan: LogicalVec::new(12.0, 34.0),
        awaiting_fit: false,
        fit_rung: Some(0.4),
        zoom_glide: Some(ZoomGlide {
            target: 4.0,
            anchor: LogicalPt::new(7.0, 8.0),
        }),
        pinched: true,
        moving: true,
        scroll_accum: 0.75,
        image_origin: LogicalPt::new(-3.0, 9.0),
        panel: PANEL,
        pan_glide: glide,
        was_pannable: true,
        autoscroll: Some(LogicalPt::new(100.0, 200.0)),
    }
}

/// The transform and every gesture over it go back to where a freshly opened
/// picture starts, with the opening fit still to come.
#[test]
fn a_new_image_resets_the_view_to_a_fresh_one_awaiting_its_fit() {
    let mut view = used_view();
    view.reset();
    let fresh = View2d::default();
    assert_eq!(view.zoom, fresh.zoom);
    assert_eq!(view.pan, fresh.pan);
    assert!(view.awaits_fit(), "the opening fit is still to come");
    assert_eq!(view.fit_rung, fresh.fit_rung);
    assert_eq!(view.zoom_glide, fresh.zoom_glide);
    assert_eq!(view.was_pannable, fresh.was_pannable);
    assert_eq!(view.autoscroll, fresh.autoscroll);
    assert!(
        !view.pan_glide.is_moving(),
        "no momentum from the last picture"
    );
}

/// What the reset leaves alone: the last layout's geometry, which the next
/// frame recomputes, and wheel travel short of a frame. None of it was ever
/// reset, and a reset that started dropping it would be a change of
/// behaviour.
#[test]
fn a_new_image_keeps_the_views_layout_bookkeeping() {
    let used = used_view();
    let mut view = used_view();
    view.reset();
    assert_eq!(view.image_origin, used.image_origin);
    assert_eq!(view.panel, used.panel);
    assert_eq!(view.scroll_accum, used.scroll_accum);
}

/// Fitting on open: the zoom and a scroll to the corner, and the fit a rung
/// only when it was asked to be one.
#[test]
fn opening_at_a_zoom_ends_the_wait_for_the_fit() {
    let mut view = used_view();
    view.reset();
    view.open_fitted(0.287);
    assert!(!view.awaits_fit());
    assert_eq!((view.zoom(), view.pan()), (0.287, LogicalVec::ZERO));
    assert_eq!(view.fit_rung(), Some(0.287));
    assert!(view.ladder().rungs().contains(&0.287));

    let mut view = View2d::default();
    view.open_at_level(0.5);
    assert_eq!(view.fit_rung(), None, "a fixed level is no rung of its own");
}

#[test]
fn an_image_that_fits_is_centred_and_cannot_be_panned() {
    let mut view = View2d::default();
    view.set_panel(PANEL);
    let layout = view.measure(ImagePx::new(320, 240), ROOM);
    assert_eq!(layout.size, LogicalVec::new(320.0, 240.0));
    assert_eq!(layout.overflow, LogicalVec::ZERO);
    assert!(!layout.pannable());
    assert_eq!(layout.shown(), layout.size);
    let placed = view.place(&layout);
    assert_eq!(placed.origin, LogicalPt::new(160.0, 150.0));
    assert_eq!(placed.visible, placed.full, "all of it is on screen");
    assert!(placed.is_visible());
    let inputs = placed.inputs(2.0);
    assert_eq!(
        (inputs.uv_offset, inputs.uv_scale),
        ([0.0, 0.0], [1.0, 1.0])
    );
    assert_eq!(inputs.viewport, crate::PhysicalPx::new(640.0, 480.0));
    assert_eq!(
        view.image_origin(),
        placed.origin,
        "noted for the next frame"
    );
}

/// A point of slack before an overflow counts: a window sized to the image
/// is not a pannable view of it.
#[test]
fn a_point_of_overflow_is_not_an_overflow() {
    let view = View2d::default();
    assert!(!view.measure(ImagePx::new(641, 481), ROOM).pannable());
    assert!(view.measure(ImagePx::new(642, 480), ROOM).pannable());
}

#[test]
fn an_overflowing_image_is_scrolled_by_the_pan_and_cut_to_the_panel() {
    let mut view = View2d {
        zoom: 2.0,
        ..Default::default()
    };
    view.set_panel(PANEL);
    let layout = view.measure(ImagePx::new(1000, 200), ROOM);
    assert_eq!(layout.overflow, LogicalVec::new(1359.0, 0.0));
    assert!(layout.pannable());
    assert_eq!(layout.shown(), LogicalVec::new(640.0, 400.0));

    view.drag(LogicalVec::new(-500.0, -500.0));
    view.clamp_pan(&layout);
    assert_eq!(
        view.pan(),
        LogicalVec::new(500.0, 0.0),
        "y has nowhere to go"
    );
    let placed = view.place(&layout);
    // Scrolled on x, centred on y.
    assert_eq!(placed.origin, LogicalPt::new(-500.0, 70.0));
    assert_eq!(placed.visible.min, LogicalPt::new(0.0, 70.0));
    assert_eq!(placed.visible.max, LogicalPt::new(640.0, 470.0));
    let inputs = placed.inputs(1.0);
    let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6;
    assert!(
        near(inputs.uv_offset, [0.25, 0.0]),
        "{:?}",
        inputs.uv_offset
    );
    assert!(near(inputs.uv_scale, [0.32, 1.0]), "{:?}", inputs.uv_scale);
    assert_eq!(inputs.viewport, crate::PhysicalPx::new(640.0, 400.0));

    // Dragged past either end, it stops at the end.
    view.drag(LogicalVec::new(-5000.0, 0.0));
    view.clamp_pan(&layout);
    assert_eq!(view.pan().x, 1359.0);
    view.pan_to(LogicalVec::new(-40.0, 0.0));
    view.clamp_pan(&layout);
    assert_eq!(view.pan().x, 0.0);
}

/// Once a frame of overflow is not enough to show a navigator, and nor is an
/// image still waiting for its opening fit.
#[test]
fn the_view_is_steadily_pannable_from_the_second_frame_of_overflow() {
    let mut view = View2d::default();
    let over = view.measure(ImagePx::new(2000, 2000), ROOM);
    let fits = view.measure(ImagePx::new(10, 10), ROOM);
    assert!(!view.note_pannable(&over), "the first frame of it");
    assert!(view.note_pannable(&over), "the second");
    assert!(!view.note_pannable(&fits));
    assert!(!view.note_pannable(&over), "starting over");

    view.reset();
    assert!(!view.note_pannable(&over));
    assert!(!view.note_pannable(&over), "still waiting for the fit");
}

/// A step glides about the pointer when it is over the panel, the middle of
/// the panel when it is not, and is no step at all at the end of the ladder.
#[test]
fn a_zoom_step_turns_about_the_pointer_over_the_panel() {
    let mut view = View2d::default();
    view.set_panel(PANEL);
    let at = LogicalPt::new(100.0, 100.0);
    let step = view.step_zoom(1, Some(at)).expect("a rung above 1:1");
    assert_eq!(
        step,
        ZoomStep {
            from: 1.0,
            anchor: at
        }
    );
    assert_eq!((view.zoom(), view.zoom_settled()), (1.0, 1.5));
    assert!(view.is_zoom_gliding());

    let mut view = View2d::default();
    view.set_panel(PANEL);
    let off = LogicalPt::new(5000.0, 100.0);
    let step = view.step_zoom(-1, Some(off)).unwrap();
    assert_eq!(step.anchor, PANEL.center());
    assert_eq!(view.step_zoom(-1, None).unwrap().anchor, PANEL.center());

    let mut top = View2d {
        zoom: 32.0,
        ..Default::default()
    };
    assert_eq!(top.step_zoom(1, None), None);
    assert!(!top.is_zoom_gliding());
    top.stop_zoom_glide();
}

/// A pinch is held within the ladder's ends, keeps the point between the
/// fingers under them, slides with them, and ends a zoom glide.
#[test]
fn a_pinch_zooms_freely_within_the_ladder() {
    let mut view = View2d::default();
    view.set_panel(PANEL);
    view.place(&view.measure(ImagePx::new(2000, 2000), ROOM));
    view.step_zoom(1, None);
    let center = LogicalPt::new(320.0, 270.0);
    let p = (center - view.image_origin()) / view.zoom();
    view.pinch(center, 1.37, LogicalVec::ZERO);
    assert_eq!(view.zoom(), 1.37, "between the rungs");
    assert!(!view.is_zoom_gliding());
    let origin = view.panel().min - view.pan();
    let here = origin + p * view.zoom();
    assert!(
        (here - center).length() < 1e-3,
        "{here:?} drifted from {center:?}"
    );

    view.pinch(center, 1000.0, LogicalVec::ZERO);
    assert_eq!(view.zoom(), 32.0);
    view.pinch(center, 1.0e-6, LogicalVec::ZERO);
    assert_eq!(view.zoom(), 0.031);

    // The fingers' slide, on top of the same pinch.
    let (mut still, mut slid) = (view, view);
    let slide = LogicalVec::new(10.0, -4.0);
    still.pinch(center, 1.2, LogicalVec::ZERO);
    slid.pinch(center, 1.2, slide);
    assert_eq!(slid.pan(), still.pan() - slide);
    assert_eq!(slid.zoom(), still.zoom());
}

/// A pinch whose ratio is not a number zooms nothing, and still pans with
/// the fingers: the zoom stays a number a step can leave from. Any other
/// ratio is held within the ladder.
#[test]
fn a_pinch_that_is_not_a_ratio_zooms_nothing() {
    let center = LogicalPt::new(320.0, 270.0);
    let slide = LogicalVec::new(10.0, -4.0);
    let mut view = View2d::default();
    view.set_panel(PANEL);
    let mut still = view;
    view.pinch(center, f32::NAN, slide);
    still.pinch(center, 1.0, slide);
    assert_eq!(view, still);
    assert_eq!(view.step_zoom(1, None).map(|s| s.from), Some(1.0));

    for (ratio, end) in [(f32::INFINITY, 32.0), (0.0, 0.031), (-2.0, 0.031)] {
        let mut view = View2d::default();
        view.set_panel(PANEL);
        view.pinch(center, ratio, slide);
        assert_eq!(view.zoom(), end, "a ratio of {ratio}");
    }
}

/// The view is moving while a step glides out or a pinch is live, as the
/// frame's advance notes it, and still on the frame a glide lands on.
#[test]
fn the_view_is_moving_while_it_glides_or_is_pinched() {
    let dt = 1.0 / 60.0;
    let mut view = View2d::default();
    view.set_panel(PANEL);
    assert!(!view.is_moving());
    view.step_zoom(1, None);
    assert!(view.advance_zoom_glide(dt));
    assert!(view.is_moving(), "gliding");
    let mut frames = 0;
    while view.advance_zoom_glide(dt) {
        frames += 1;
        assert!(frames < 100, "the glide should land");
    }
    assert!(!view.is_moving(), "the frame it lands on is still");

    view.pinch(PANEL.center(), 1.1, LogicalVec::ZERO);
    assert!(!view.advance_zoom_glide(dt));
    assert!(view.is_moving(), "pinched this frame");
    view.advance_zoom_glide(dt);
    assert!(!view.is_moving(), "and not the next");
}

/// Middle-click on, middle-click off, and nothing to turn on over an image
/// that fits.
#[test]
fn autoscroll_turns_on_only_where_there_is_something_to_scroll() {
    let mut view = View2d::default();
    let over = view.measure(ImagePx::new(2000, 2000), ROOM);
    let fits = view.measure(ImagePx::new(10, 10), ROOM);
    let at = Some(LogicalPt::new(300.0, 300.0));

    view.toggle_autoscroll(at, &fits);
    assert_eq!(view.autoscroll(), None);
    view.toggle_autoscroll(at, &over);
    assert_eq!(view.autoscroll(), at);
    view.toggle_autoscroll(Some(LogicalPt::new(1.0, 1.0)), &over);
    assert_eq!(view.autoscroll(), None, "a second click stops it");

    view.toggle_autoscroll(at, &over);
    view.end_autoscroll();
    assert_eq!(view.autoscroll(), None);
    assert!(
        !view.autoscroll_by(LogicalPt::new(900.0, 300.0), 0.1),
        "off"
    );
}

/// Parked in the dead zone nothing moves and no frame is wanted; past it the
/// view drifts towards the pointer, frame time by frame time.
#[test]
fn autoscroll_drifts_towards_the_pointer() {
    let mut view = View2d::default();
    let over = view.measure(ImagePx::new(2000, 2000), ROOM);
    let anchor = LogicalPt::new(300.0, 300.0);
    view.toggle_autoscroll(Some(anchor), &over);
    assert!(!view.autoscroll_by(anchor + LogicalVec::new(3.0, 0.0), 0.1));
    assert_eq!(view.pan(), LogicalVec::ZERO);
    assert!(view.autoscroll_by(anchor + LogicalVec::new(0.0, 100.0), 0.1));
    assert_eq!(view.pan().x, 0.0);
    assert!(view.pan().y > 0.0, "below the anchor shows a lower part");
}

/// Wheel travel comes out in whole frames, both ways, with the remainder
/// kept until the pointer leaves the picture.
#[test]
fn scrubbing_keeps_the_fraction_of_a_frame() {
    let mut view = View2d::default();
    assert_eq!(view.scrub(0.4), 0);
    assert_eq!(view.scrub(0.4), 0);
    assert_eq!(view.scrub(0.4), 1);
    assert_eq!(view.scrub(-3.0), -2, "0.2 left, so -2.8");
    view.forget_scrub();
    assert_eq!(view.scrub(0.9), 0);
    assert_eq!(view.scrub(-0.9), 0, "back where it started");
}

/// A swipe pans the view by exactly what it was, coasts on after, stops at an
/// edge, and has nothing to do over an image that fits.
#[test]
fn a_swipe_pans_and_coasts_within_the_overflow() {
    let mut view = View2d::default();
    let over = view.measure(ImagePx::new(4000, 4000), ROOM);
    view.pan_to(LogicalVec::new(1000.0, 1000.0));
    let dt = 1.0 / 60.0;
    for _ in 0..6 {
        view.glide_pan(LogicalVec::new(-20.0, 0.0), dt, &over);
    }
    assert_eq!(view.pan(), LogicalVec::new(1120.0, 1000.0));
    assert!(view.glide_pan(LogicalVec::ZERO, dt, &over), "it coasts");
    assert!(view.pan().x > 1120.0);
    let mut coasting = true;
    for _ in 0..600 {
        coasting = view.glide_pan(LogicalVec::ZERO, dt, &over);
        if !coasting {
            break;
        }
    }
    assert!(!coasting, "and stops");

    // Into the edge: stopped there, not wound up against it.
    view.pan_to(LogicalVec::new(over.overflow.x - 5.0, 1000.0));
    for _ in 0..6 {
        view.glide_pan(LogicalVec::new(-20.0, 0.0), dt, &over);
    }
    assert_eq!(view.pan().x, over.overflow.x);
    assert!(!view.glide_pan(LogicalVec::ZERO, dt, &over));

    // An image that fits: a swipe passes through, and nothing coasts.
    let fits = view.measure(ImagePx::new(10, 10), ROOM);
    let mut view = View2d::default();
    for _ in 0..6 {
        view.glide_pan(LogicalVec::new(-20.0, 0.0), dt, &fits);
    }
    assert_eq!(view.pan(), LogicalVec::ZERO);
}

/// A drag or a navigator's jump ends a coast: the hand decides now.
#[test]
fn taking_hold_of_the_picture_ends_a_coast() {
    let over = View2d::default().measure(ImagePx::new(4000, 4000), ROOM);
    let dt = 1.0 / 60.0;
    let flung = || {
        let mut view = View2d::default();
        view.pan_to(LogicalVec::new(1000.0, 1000.0));
        for _ in 0..6 {
            view.glide_pan(LogicalVec::new(-20.0, 0.0), dt, &over);
        }
        view
    };
    let mut view = flung();
    view.drag(LogicalVec::ZERO);
    assert!(!view.glide_pan(LogicalVec::ZERO, dt, &over));
    let mut view = flung();
    view.pan_to(view.pan());
    assert!(!view.glide_pan(LogicalVec::ZERO, dt, &over));
}

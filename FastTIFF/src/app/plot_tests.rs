//! The canvas tool's geometry, and the chart's number formatting.
//!
//! The drawing itself is not tested here — what is worth pinning is the
//! drag-to-region mapping, because a plot of the wrong pixels looks exactly
//! like a plot of the right ones. The screen-to-pixel mapping before it is
//! `ImageXform`'s, and is tested with it, in `canvas_tool_tests.rs`; so are
//! these gestures as egui reports them.

use super::*;

/// The smallest possible selection is one pixel. A drag that starts and ends
/// inside the same pixel must select it, not nothing.
#[test]
fn a_drag_within_one_pixel_selects_that_pixel() {
    let r = drag_region((3, 4), (3, 4), Shape::Rect, 64, 64).expect("a region");
    assert_eq!((r.x, r.y, r.w, r.h), (3, 4, 1, 1));
}

#[test]
fn a_drag_covers_both_end_pixels() {
    // From pixel 2 to pixel 5 inclusive is four pixels, not three.
    let r = drag_region((2, 0), (5, 0), Shape::Rect, 64, 64).expect("a region");
    assert_eq!((r.x, r.w), (2, 4));
}

#[test]
fn a_backwards_drag_is_the_same_region() {
    let a = drag_region((5, 7), (2, 3), Shape::Rect, 64, 64).expect("a");
    let b = drag_region((2, 3), (5, 7), Shape::Rect, 64, 64).expect("b");
    assert_eq!(a, b);
}

#[test]
fn a_drag_keeps_the_shape_it_was_started_with() {
    let e = drag_region((0, 0), (7, 7), Shape::Ellipse, 64, 64).expect("a region");
    assert_eq!(e.shape, Shape::Ellipse);
    // And the ellipse rule really applies to it.
    assert!(!e.contains(0, 0));
    assert!(e.contains(4, 4));
}

#[test]
fn a_drag_is_clamped_to_the_image() {
    let r = drag_region((-10, -10), (3, 3), Shape::Rect, 8, 8).expect("a region");
    assert_eq!((r.x, r.y, r.w, r.h), (0, 0, 4, 4));
    // Entirely off the image is not a region.
    assert!(drag_region((50, 50), (60, 60), Shape::Rect, 8, 8).is_none());
}

// ------------------------------------------------------------- the drag rule

/// A 40 x 20 picture at 1x from the window's corner: pixel `(x, y)` is under
/// screen `(x + 0.5, y + 0.5)`.
const XF: ImageXform = ImageXform {
    origin: Pos2::ZERO,
    zoom: 1.0,
    size: (40, 20),
};

/// The middle of pixel `(x, y)` on screen.
fn on((x, y): (i64, i64)) -> Pos2 {
    Pos2::new(x as f32 + 0.5, y as f32 + 0.5)
}

/// A primary drag first seen over `px`, the button having gone down on
/// `pressed`.
fn start(px: (i64, i64), pressed: (i64, i64)) -> CanvasGesture {
    CanvasGesture {
        at: Some(on(px)),
        pressed_at: Some(on(pressed)),
        drag_started: true,
        dragging: true,
        ..Default::default()
    }
}

fn moving(px: (i64, i64)) -> CanvasGesture {
    CanvasGesture {
        at: Some(on(px)),
        dragging: true,
        ..Default::default()
    }
}

/// The button coming up over `px`: egui no longer calls it a drag, and has
/// let go of where it was pressed.
fn stop(px: (i64, i64)) -> CanvasGesture {
    CanvasGesture {
        at: Some(on(px)),
        drag_stopped: true,
        ..Default::default()
    }
}

/// egui reports a drag once it is past the click threshold, so the first
/// frame of one already has the pointer some way from the press. The region
/// starts on the press.
#[test]
fn a_drag_starts_on_the_pixel_the_button_went_down_on() {
    let mut drag = None;
    assert_eq!(track_drag(&mut drag, &start((7, 5), (4, 4)), &XF), None);
    assert_eq!(drag, Some(((4, 4), (7, 5))), "the press, not the pointer");
    // Pressed off the picture, it starts there, not at its corner.
    let mut drag = None;
    track_drag(&mut drag, &start((2, 2), (-3, -1)), &XF);
    assert_eq!(drag, Some(((-3, -1), (2, 2))));
}

/// A flick moves and lets go in one frame: the far corner is where the
/// button came up, not where the last frame of the drag left it.
#[test]
fn a_drag_ends_on_the_pixel_the_button_came_up_on() {
    let mut drag = None;
    track_drag(&mut drag, &start((7, 5), (4, 4)), &XF);
    track_drag(&mut drag, &moving((10, 8)), &XF);
    let corners = track_drag(&mut drag, &stop((15, 12)), &XF);
    assert_eq!(corners, Some(((4, 4), (15, 12))));
    assert_eq!(drag, None, "nothing in flight once it ends");
}

/// A press the navigator had, or that ended autoscroll, is spent on that.
/// One already in flight goes on, wherever the pointer is.
#[test]
fn a_taken_press_starts_no_drag_and_a_drag_in_flight_goes_on() {
    let taken = |g: CanvasGesture| CanvasGesture { taken: true, ..g };
    let mut drag = None;
    assert_eq!(
        track_drag(&mut drag, &taken(start((7, 5), (4, 4))), &XF),
        None
    );
    assert_eq!(drag, None);
    track_drag(&mut drag, &moving((10, 8)), &XF);
    assert_eq!(track_drag(&mut drag, &stop((10, 8)), &XF), None);

    track_drag(&mut drag, &start((7, 5), (4, 4)), &XF);
    track_drag(&mut drag, &taken(moving((30, 15))), &XF);
    let corners = track_drag(&mut drag, &taken(stop((31, 16))), &XF);
    assert_eq!(corners, Some(((4, 4), (31, 16))));
}

/// Only the primary button draws: another one's drag passes, and its end
/// commits nothing.
#[test]
fn another_buttons_drag_draws_nothing() {
    let mut drag = None;
    let other = CanvasGesture {
        at: Some(on((7, 5))),
        pressed_at: Some(on((4, 4))),
        ..Default::default()
    };
    track_drag(&mut drag, &other, &XF);
    assert_eq!(track_drag(&mut drag, &stop((9, 9)), &XF), None);
}

fn regions_plot() -> PlotWindow {
    let mut plot = Plot::new("trace");
    plot.wants = SelectionKind::Regions;
    PlotWindow::new(0, fasttiff_plugin_api::Params::new(), plot)
}

/// A plain drag replaces the regions, Shift adds one; either leaves the plot
/// stale for its plugin to run again.
#[test]
fn a_drag_replaces_the_regions_and_shift_adds_one() {
    let mut win = regions_plot();
    let drag = |win: &mut PlotWindow, from, to, shift| {
        let held = |g: CanvasGesture| CanvasGesture { shift, ..g };
        assert_eq!(CanvasTool::track(win, &held(start(to, from)), &XF, 0), None);
        assert_eq!(CanvasTool::track(win, &held(stop(to)), &XF, 0), None);
    };
    drag(&mut win, (1, 1), (2, 2), false);
    drag(&mut win, (5, 5), (9, 7), false);
    let replaced = Roi::from_corners((5, 5), (10, 8), 40, 20).unwrap();
    assert_eq!(win.rois, vec![replaced]);
    assert!(win.stale);
    win.stale = false;
    win.shape = Shape::Ellipse;
    drag(&mut win, (20, 10), (24, 14), true);
    let added = Roi::from_corners((20, 10), (25, 15), 40, 20)
        .unwrap()
        .with_shape(Shape::Ellipse);
    assert_eq!(win.rois, vec![replaced, added]);
    assert!(win.stale);
}

// ------------------------------------------------------------- the x labels

#[test]
fn a_whole_x_value_has_no_decimals() {
    // A frame index should read as one.
    assert_eq!(fmt_x(0.0), "0");
    assert_eq!(fmt_x(120.0), "120");
    assert_eq!(fmt_x(-3.0), "-3");
}

#[test]
fn a_calibrated_x_value_keeps_its_decimals() {
    assert_eq!(fmt_x(0.266), "0.27");
    assert_eq!(fmt_x(12.5), "12.50");
}

// --------------------------------------------------------------- the palette

/// Region 3 on the picture and series 3 in the legend have to be the same
/// colour, which is only true if both sides index one palette.
#[test]
fn the_palette_is_stable_and_wraps() {
    assert_eq!(palette(0), palette(8), "the palette wraps");
    assert_ne!(palette(0), palette(1));
    // Region 3 on the picture and series 3 in the legend take the same entry,
    // which is the whole mechanism keeping them matched.
    assert_eq!(palette(2), palette(2));
}

//! The gestures, one frame of the button at a time: what a click, a drag and
//! an Alt+click do to the points, on the frame in view and off it.

use super::*;

/// A 100 x 50 image, and a press reaching 3 image pixels.
const SIZE: (u32, u32) = (100, 50);
const R: f32 = 3.0;

fn tool() -> Session {
    Session::new(PointCounter::new())
}

fn click(x: f32, y: f32) -> Gesture {
    Gesture {
        at: Some((x, y)),
        pressed_at: Some((x, y)),
        clicked: true,
        ..Default::default()
    }
}

fn alt_click(x: f32, y: f32) -> Gesture {
    Gesture {
        alt: true,
        ..click(x, y)
    }
}

/// The frame a drag is first seen on: pressed at `from`, already at `to`.
fn drag_start(from: (f32, f32), to: (f32, f32)) -> Gesture {
    Gesture {
        at: Some(to),
        pressed_at: Some(from),
        drag_started: true,
        dragging: true,
        ..Default::default()
    }
}

fn drag_move(to: (f32, f32)) -> Gesture {
    Gesture {
        at: Some(to),
        dragging: true,
        ..Default::default()
    }
}

fn drag_stop(to: (f32, f32)) -> Gesture {
    Gesture {
        at: Some(to),
        drag_stopped: true,
        ..Default::default()
    }
}

fn positions(t: &Session) -> Vec<(f32, f32)> {
    t.counter.points.iter().map(|p| (p.x, p.y)).collect()
}

/// A click adds a point where it landed — continuous, not floored to the
/// pixel — on the frame in view, in the selected group.
#[test]
fn a_click_adds_a_point_on_the_frame_in_view() {
    let mut t = tool();
    t.counter.add_group();
    t.track(click(10.5, 20.25), 4, R, SIZE);
    assert_eq!(t.counter.points.len(), 1);
    let p = t.counter.points[0];
    assert_eq!((p.x, p.y, p.frame, p.group), (10.5, 20.25, Some(4), 1));
}

/// The margin around a letterboxed image is not the image.
#[test]
fn a_click_off_the_image_adds_nothing() {
    let mut t = tool();
    for (x, y) in [(-0.5, 10.0), (100.0, 10.0), (10.0, -0.1), (10.0, 50.0)] {
        t.track(click(x, y), 0, R, SIZE);
    }
    assert!(t.counter.points.is_empty());
    // The edges that are in it are in it.
    t.track(click(0.0, 0.0), 0, R, SIZE);
    t.track(click(99.9, 49.9), 0, R, SIZE);
    assert_eq!(t.counter.points.len(), 2);
}

/// A click on a point is taken as aiming at that point, not as a second one
/// stacked on it — which a double-click would otherwise make.
#[test]
fn a_click_on_a_point_adds_nothing() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(click(11.0, 12.0), 0, R, SIZE);
    assert_eq!(t.counter.points.len(), 1);
    // Just out of reach is a new point.
    t.track(click(14.0, 10.0), 0, R, SIZE);
    assert_eq!(t.counter.points.len(), 2);
}

#[test]
fn alt_click_deletes_the_point_under_it_and_nothing_elsewhere() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(click(30.0, 10.0), 0, R, SIZE);
    t.track(alt_click(60.0, 10.0), 0, R, SIZE);
    assert_eq!(
        t.counter.points.len(),
        2,
        "Alt+click on nothing deletes nothing, and adds nothing either"
    );
    t.track(alt_click(29.0, 11.0), 0, R, SIZE);
    assert_eq!(positions(&t), vec![(10.0, 10.0)]);
}

/// The navigator's click and the click that ends autoscroll are theirs.
#[test]
fn a_click_something_else_took_adds_nothing() {
    let mut t = tool();
    t.track(
        Gesture {
            taken: true,
            ..click(10.0, 10.0)
        },
        0,
        R,
        SIZE,
    );
    t.track(
        Gesture {
            taken: true,
            ..drag_start((10.0, 10.0), (20.0, 10.0))
        },
        0,
        R,
        SIZE,
    );
    assert!(t.counter.points.is_empty());
    assert!(t.drag.is_none());
}

/// A drag taken from a point moves it, keeping the offset it was taken hold
/// of at, and the whole drag is one undo step back to where it began.
#[test]
fn a_drag_moves_a_point_and_undoes_in_one_step() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    // Pressed a pixel to the right of the point, first seen 4 further on.
    t.track(drag_start((11.0, 10.0), (15.0, 10.0)), 0, R, SIZE);
    assert_eq!(
        positions(&t),
        vec![(14.0, 10.0)],
        "moves with the grab offset"
    );
    t.track(drag_move((21.0, 15.0)), 0, R, SIZE);
    t.track(drag_move((31.0, 20.0)), 0, R, SIZE);
    assert_eq!(positions(&t), vec![(30.0, 20.0)]);
    t.track(drag_stop((31.0, 20.0)), 0, R, SIZE);
    assert!(t.drag.is_none());
    assert_eq!(t.counter.points.len(), 1, "a drag adds nothing");
    assert!(t.undo());
    assert_eq!(
        positions(&t),
        vec![(10.0, 10.0)],
        "back where the drag began"
    );
    assert!(t.undo(), "then the add");
    assert!(t.counter.points.is_empty());
}

/// A drag from empty space neither pans (the canvas's rule) nor adds or
/// moves anything here.
#[test]
fn a_drag_from_nowhere_does_nothing() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(drag_start((50.0, 30.0), (55.0, 30.0)), 0, R, SIZE);
    t.track(drag_move((10.0, 10.0)), 0, R, SIZE);
    t.track(drag_stop((10.0, 10.0)), 0, R, SIZE);
    assert_eq!(positions(&t), vec![(10.0, 10.0)]);
    assert!(t.drag.is_none());
}

/// A point dragged past the edge stays on the image — on its last pixel, not
/// on the edge beyond it, which is on no pixel, as it is for a click.
#[test]
fn a_dragged_point_stays_on_the_image() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(drag_start((10.0, 10.0), (-40.0, 400.0)), 0, R, SIZE);
    assert_eq!(positions(&t), vec![(0.0, 50.0f32.next_down())]);
    t.track(drag_stop((500.0, -3.0)), 0, R, SIZE);
    let (x, y) = positions(&t)[0];
    assert!((99.99..100.0).contains(&x) && y == 0.0, "{x}, {y}");
}

/// A drag whose release was never seen — the view left 2D mid-drag — ends
/// on the next frame the canvas reports, still as one undo step.
#[test]
fn a_drag_ends_when_the_button_is_no_longer_dragging() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(drag_start((10.0, 10.0), (20.0, 10.0)), 0, R, SIZE);
    t.track(Gesture::default(), 0, R, SIZE);
    assert!(t.drag.is_none());
    assert!(t.undo());
    assert_eq!(positions(&t), vec![(10.0, 10.0)]);
}

/// With "Show only current frame's tags" on, a point placed on another frame
/// is not on screen: it cannot be grabbed or deleted from this one, and a
/// click where it would be is a new point.
#[test]
fn only_the_points_on_the_frame_in_view_are_under_the_pointer() {
    let mut t = tool();
    t.counter.per_frame = true;
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(alt_click(10.0, 10.0), 1, R, SIZE);
    assert_eq!(t.counter.points.len(), 1, "not deleted from another frame");
    t.track(drag_start((10.0, 10.0), (20.0, 10.0)), 1, R, SIZE);
    assert!(t.drag.is_none(), "not grabbed from another frame");
    t.track(click(10.0, 10.0), 1, R, SIZE);
    assert_eq!(t.counter.points.len(), 2, "a new point on this frame");

    // Off, every point is on every frame.
    t.counter.per_frame = false;
    t.track(alt_click(10.0, 10.0), 7, R, SIZE);
    assert_eq!(t.counter.points.len(), 1);
}

/// Ctrl+Z in the middle of a drag puts the point back where the drag began
/// and lets go of it: the rest of the gesture moves nothing.
#[test]
fn undo_during_a_drag_puts_the_point_back_and_lets_go() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(drag_start((10.0, 10.0), (20.0, 10.0)), 0, R, SIZE);
    assert!(t.undo());
    assert!(t.drag.is_none());
    assert_eq!(positions(&t), vec![(10.0, 10.0)]);
    t.track(drag_move((40.0, 40.0)), 0, R, SIZE);
    t.track(drag_stop((40.0, 40.0)), 0, R, SIZE);
    assert_eq!(positions(&t), vec![(10.0, 10.0)], "the gesture is spent");
    assert!(t.undo(), "the add is still there to undo");
    assert!(!t.undo());
}

/// A delete undone comes back at its place, numbered as it was.
#[test]
fn undoing_a_delete_restores_the_numbering() {
    let mut t = tool();
    for x in [10.0, 20.0, 30.0] {
        t.track(click(x, 10.0), 0, R, SIZE);
    }
    t.track(alt_click(20.0, 10.0), 0, R, SIZE);
    assert_eq!(
        t.counter.number_in_group(1),
        2,
        "the third is now the second"
    );
    assert!(t.undo());
    assert_eq!(
        positions(&t),
        vec![(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)]
    );
    assert_eq!(t.counter.number_in_group(2), 3);
}

/// Reassigning the axes keeps each point on the plane its frame began at: a
/// time series of six read as 2 channels x 3 frames puts frame 2 (IFD 4) on
/// what is now frame 1 of 3 channels x 2 frames, and a point on every frame
/// stays there. A delete waiting to be undone moves with the rest.
#[test]
fn points_follow_the_axes_they_were_placed_on() {
    use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(2, 2, SampleType::U8).metadata(StackMetaWrite::new(2, 1));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for p in 0..6u8 {
        w.write_frame_u8(&[p; 4]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    let mut stack = Document::from_bytes(bytes, "axes.tif".into(), false).unwrap();
    assert_eq!(stack.frame_count(), 3);

    let mut t = Session::new(PointCounter::new());
    t.counter.add(0.5, 0.5, 2);
    t.counter.add(1.5, 1.5, 1);
    t.counter.add(1.5, 0.5, 0);
    t.counter.points[2].frame = None;
    t.counter.add(0.5, 1.5, 2);
    t.counter.delete(3);

    let before = super::super::frame_starts(&stack);
    assert_eq!(before, [1, 3, 5]);
    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    assert_eq!(stack.frame_count(), 2);
    t.follow_axes(&before, &stack);
    let frames: Vec<Option<usize>> = t.counter.points.iter().map(|p| p.frame).collect();
    // IFD 4 is frame 1 now, IFD 2 frame 0.
    assert_eq!(frames, [Some(1), Some(0), None]);
    assert!(t.undo());
    assert_eq!(
        t.counter.points[3].frame,
        Some(1),
        "the deleted one moved too"
    );
}

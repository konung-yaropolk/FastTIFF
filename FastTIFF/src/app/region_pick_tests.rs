//! The rectangle a plugin asked for: what a drag draws, and what the status
//! line says about it. Where a drag starts and ends is the rule it shares
//! with a plot's regions, `plot::track_drag`, and is pinned there.

use super::*;
use egui::Pos2;

fn pick() -> RegionPick {
    RegionPick {
        plugin: 0,
        params: Params::new(),
        name: "Crop".into(),
        roi: None,
        drag: None,
    }
}

/// A 20 x 10 picture at 1x from the window's corner: pixel `(x, y)` is
/// under screen `(x + 0.5, y + 0.5)`.
const XF: ImageXform = ImageXform {
    origin: Pos2::ZERO,
    zoom: 1.0,
    size: (20, 10),
};

/// The middle of pixel `(x, y)` on screen.
fn on((x, y): (i64, i64)) -> Pos2 {
    Pos2::new(x as f32 + 0.5, y as f32 + 0.5)
}

/// A primary drag first seen over `px`, the button having gone down on
/// `pressed`.
fn start(px: (i64, i64), pressed: Option<(i64, i64)>) -> CanvasGesture {
    CanvasGesture {
        at: Some(on(px)),
        pressed_at: pressed.map(on),
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

/// The button coming up over `px`, or with the pointer nowhere egui says.
fn stop(px: Option<(i64, i64)>) -> CanvasGesture {
    CanvasGesture {
        at: px.map(on),
        drag_stopped: true,
        ..Default::default()
    }
}

/// One frame of `g`: the rectangle, on the frame one is drawn, after
/// checking the status line says so.
fn track(p: &mut RegionPick, g: CanvasGesture) -> Option<Roi> {
    let said = CanvasTool::track(p, &g, &XF, 0)?;
    let roi = p.roi.expect("a status only once a rectangle is drawn");
    assert_eq!(said, ToolEvent::Status(p.drawn_message(&roi)));
    Some(roi)
}

#[test]
fn a_drag_draws_the_rectangle_it_covers() {
    let mut p = pick();
    assert_eq!(track(&mut p, start((2, 3), None)), None);
    assert_eq!(track(&mut p, moving((6, 5))), None);
    assert!(p.drag.is_some(), "the drag in flight is drawn");
    let roi = track(&mut p, stop(Some((6, 5)))).expect("a rectangle");
    // Both corner pixels inside.
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (2, 3, 5, 3));
    assert_eq!(roi.shape, Shape::Rect);
    assert_eq!(p.roi, Some(roi));
    assert!(p.drag.is_none());
}

/// One rectangle at a time: drawing again replaces it.
#[test]
fn a_new_drag_replaces_the_rectangle() {
    let mut p = pick();
    track(&mut p, start((0, 0), None));
    track(&mut p, stop(Some((3, 3))));
    track(&mut p, start((10, 1), None));
    track(&mut p, moving((8, 4)));
    let roi = track(&mut p, stop(None)).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (8, 1, 3, 4));
    assert_eq!(p.roi, Some(roi));
}

/// Off the image, it keeps the part on it; wholly off, it keeps the last one.
#[test]
fn a_drag_is_clamped_to_the_image() {
    let mut p = pick();
    track(&mut p, start((-5, -5), None));
    let roi = track(&mut p, stop(Some((30, 4)))).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (0, 0, 20, 5));

    track(&mut p, start((40, 40), None));
    assert_eq!(track(&mut p, stop(Some((50, 50)))), None);
    assert_eq!(p.roi, Some(roi), "a drag off the image draws nothing");
}

#[test]
fn the_status_says_what_was_drawn_and_what_to_press() {
    let mut p = pick();
    track(&mut p, start((1, 2), None));
    let roi = track(&mut p, stop(Some((4, 2)))).unwrap();
    let m = p.drawn_message(&roi);
    assert!(m.starts_with("Crop: 4 x 1 pixels from (1, 2)"), "{m}");
    assert!(m.contains("Enter") && m.contains("Esc"), "{m}");
}

/// The corner is where the button went down, even when the drag was only
/// noticed further on.
#[test]
fn the_rectangle_starts_where_the_button_went_down() {
    let mut p = pick();
    track(&mut p, start((9, 9), Some((2, 3))));
    let roi = track(&mut p, stop(Some((9, 9)))).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (2, 3, 8, 7));
}

/// A drag whose press ended autoscroll, or was the navigator's, draws no
/// rectangle and keeps the one there was.
#[test]
fn a_taken_press_draws_nothing() {
    let mut p = pick();
    track(&mut p, start((1, 1), None));
    let drawn = track(&mut p, stop(Some((3, 3)))).unwrap();
    let taken = CanvasGesture {
        taken: true,
        ..start((5, 5), None)
    };
    assert_eq!(track(&mut p, taken), None);
    assert!(p.drag.is_none(), "no drag in flight");
    assert_eq!(track(&mut p, moving((9, 8))), None);
    assert_eq!(track(&mut p, stop(Some((9, 8)))), None);
    assert_eq!(p.roi, Some(drawn));
}

use super::*;

fn pick() -> RegionPick {
    RegionPick {
        plugin: 0,
        params: Params::new(),
        name: "Crop".into(),
        roi: None,
        drag: None,
    }
}

const START: (bool, bool, bool) = (true, true, false);
const MOVE: (bool, bool, bool) = (false, true, false);
const STOP: (bool, bool, bool) = (false, false, true);

#[test]
fn a_drag_draws_the_rectangle_it_covers() {
    let mut p = pick();
    assert_eq!(p.track(Some((2, 3)), None, START, (20, 10)), None);
    assert_eq!(p.track(Some((6, 5)), None, MOVE, (20, 10)), None);
    assert!(p.drag.is_some(), "the drag in flight is drawn");
    let roi = p
        .track(Some((6, 5)), None, STOP, (20, 10))
        .expect("a rectangle");
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
    p.track(Some((0, 0)), None, START, (20, 10));
    p.track(Some((3, 3)), None, STOP, (20, 10));
    p.track(Some((10, 1)), None, START, (20, 10));
    p.track(Some((8, 4)), None, MOVE, (20, 10));
    let roi = p.track(None, None, STOP, (20, 10)).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (8, 1, 3, 4));
    assert_eq!(p.roi, Some(roi));
}

/// Off the image, it keeps the part on it; wholly off, it keeps the last one.
#[test]
fn a_drag_is_clamped_to_the_image() {
    let mut p = pick();
    p.track(Some((-5, -5)), None, START, (20, 10));
    let roi = p.track(Some((30, 4)), None, STOP, (20, 10)).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (0, 0, 20, 5));

    p.track(Some((40, 40)), None, START, (20, 10));
    assert_eq!(p.track(Some((50, 50)), None, STOP, (20, 10)), None);
    assert_eq!(p.roi, Some(roi), "a drag off the image draws nothing");
}

#[test]
fn the_status_says_what_was_drawn_and_what_to_press() {
    let mut p = pick();
    p.track(Some((1, 2)), None, START, (20, 10));
    let roi = p.track(Some((4, 2)), None, STOP, (20, 10)).unwrap();
    let m = p.drawn_message(&roi);
    assert!(m.starts_with("Crop: 4 x 1 pixels from (1, 2)"), "{m}");
    assert!(m.contains("Enter") && m.contains("Esc"), "{m}");
}

/// The corner is where the button went down, even when the drag was only
/// noticed further on.
#[test]
fn the_rectangle_starts_where_the_button_went_down() {
    let mut p = pick();
    p.track(Some((9, 9)), Some((2, 3)), START, (20, 10));
    let roi = p.track(Some((9, 9)), Some((2, 3)), STOP, (20, 10)).unwrap();
    assert_eq!((roi.x, roi.y, roi.w, roi.h), (2, 3, 8, 7));
}

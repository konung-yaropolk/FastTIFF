//! The geometry of a region, checked against counts worked out by hand.
//!
//! This arithmetic decides which pixels a measurement includes, and getting it
//! wrong produces a number that is wrong by a plausible-looking amount rather
//! than one that is obviously broken. Every expectation below is derived
//! independently of the code under test.

use super::*;

#[test]
fn a_rectangle_covers_its_whole_extent() {
    let r = Roi {
        shape: Shape::Rect,
        x: 2,
        y: 3,
        w: 4,
        h: 5,
    };
    assert_eq!(r.pixel_count(), 20);
    assert!(r.contains(2, 3));
    assert!(r.contains(5, 7));
    // One past each edge is outside.
    assert!(!r.contains(1, 3));
    assert!(!r.contains(2, 2));
    assert!(!r.contains(6, 7));
    assert!(!r.contains(5, 8));
}

/// An ellipse is not its bounding box, and the corners are exactly the
/// difference.
#[test]
fn an_ellipse_excludes_its_corners() {
    let box_ = Roi {
        shape: Shape::Rect,
        x: 0,
        y: 0,
        w: 8,
        h: 8,
    };
    let e = box_.with_shape(Shape::Ellipse);

    assert!(box_.contains(0, 0) && box_.contains(7, 7));
    assert!(
        !e.contains(0, 0),
        "the top-left corner is outside an ellipse"
    );
    assert!(!e.contains(7, 0));
    assert!(!e.contains(0, 7));
    assert!(!e.contains(7, 7));
    assert!(e.contains(4, 4), "the centre is inside");
    assert!(e.pixel_count() < box_.pixel_count());
}

/// The count is of pixel centres inside, not the continuous area. For an 8x8
/// circle those differ: pi*4*4 = 50.27, and the centres come to a whole number
/// that is its own answer.
#[test]
fn the_pixel_count_is_counted_not_computed() {
    let e = Roi {
        shape: Shape::Ellipse,
        x: 0,
        y: 0,
        w: 8,
        h: 8,
    };

    // Worked out independently, by the same rule stated in the doc comment.
    let mut n = 0;
    for y in 0..8u32 {
        for x in 0..8u32 {
            let dx = (x as f64 + 0.5 - 4.0) / 4.0;
            let dy = (y as f64 + 0.5 - 4.0) / 4.0;
            if dx * dx + dy * dy <= 1.0 {
                n += 1;
            }
        }
    }
    assert_eq!(e.pixel_count(), n);
    assert_ne!(
        e.pixel_count(),
        (std::f64::consts::PI * 16.0).round() as usize,
        "the continuous area is a different number, and using it would scale \
         every mean by a few percent"
    );
}

/// A one-pixel region is a region. The smallest useful selection is one bright
/// spot, and an ellipse that small must still contain its own single pixel.
#[test]
fn a_single_pixel_region_covers_that_pixel() {
    for shape in [Shape::Rect, Shape::Ellipse] {
        let r = Roi {
            shape,
            x: 5,
            y: 6,
            w: 1,
            h: 1,
        };
        assert_eq!(r.pixel_count(), 1, "{shape:?}");
        assert!(r.contains(5, 6), "{shape:?}");
        assert!(!r.contains(6, 6), "{shape:?}");
    }
}

/// An ellipse thinner than a pixel in one axis still has to behave — this is
/// what a careless drag produces, and a divide-by-zero here would be a panic in
/// the middle of the user's selection.
#[test]
fn a_degenerate_ellipse_does_not_panic() {
    let thin = Roi {
        shape: Shape::Ellipse,
        x: 0,
        y: 0,
        w: 1,
        h: 9,
    };
    assert!(thin.pixel_count() >= 1);
    let wide = Roi {
        shape: Shape::Ellipse,
        x: 0,
        y: 0,
        w: 9,
        h: 1,
    };
    assert!(wide.pixel_count() >= 1);
}

// ------------------------------------------------------------- from a drag

#[test]
fn a_backwards_drag_is_the_same_region() {
    let a = Roi::from_corners((1, 1), (5, 4), 16, 16).expect("a");
    let b = Roi::from_corners((5, 4), (1, 1), 16, 16).expect("b");
    assert_eq!(a, b);
    assert_eq!((a.x, a.y, a.w, a.h), (1, 1, 4, 3));
}

#[test]
fn a_drag_off_the_image_keeps_the_part_that_is_on_it() {
    let r = Roi::from_corners((-5, -5), (3, 3), 8, 6).expect("a region");
    assert_eq!((r.x, r.y, r.w, r.h), (0, 0, 3, 3));

    let past = Roi::from_corners((6, 4), (100, 100), 8, 6).expect("a region");
    assert_eq!((past.x, past.y, past.w, past.h), (6, 4, 2, 2));
}

#[test]
fn a_click_or_a_drag_entirely_outside_is_not_a_region() {
    assert!(Roi::from_corners((3, 3), (3, 3), 8, 6).is_none(), "a click");
    assert!(
        Roi::from_corners((20, 20), (30, 30), 8, 6).is_none(),
        "entirely past the right edge"
    );
    assert!(
        Roi::from_corners((-30, -30), (-20, -20), 8, 6).is_none(),
        "entirely past the top-left"
    );
}

/// The shape a drag starts as is the caller's to set — `from_corners` is
/// geometry, and defaulting to a rectangle keeps it from guessing.
#[test]
fn a_drag_starts_rectangular_and_is_reshaped_by_the_caller() {
    let r = Roi::from_corners((0, 0), (4, 4), 8, 8).expect("a region");
    assert_eq!(r.shape, Shape::Rect);
    assert_eq!(r.with_shape(Shape::Ellipse).shape, Shape::Ellipse);
    // Reshaping keeps the extent — it is the same drag, read differently.
    assert_eq!((r.x, r.y, r.w, r.h), {
        let e = r.with_shape(Shape::Ellipse);
        (e.x, e.y, e.w, e.h)
    });
}

// --------------------------------------------------------------- addressing

#[test]
fn indices_address_a_row_major_plane() {
    let r = Roi {
        shape: Shape::Rect,
        x: 1,
        y: 2,
        w: 2,
        h: 2,
    };
    // Rows 2 and 3, columns 1 and 2, of a 5-wide plane.
    assert_eq!(r.indices(5, 5), vec![11, 12, 16, 17]);
}

/// A region reaching past the plane must not address past its end — the
/// clamping in `from_corners` is not the only way one can be built.
#[test]
fn indices_stay_inside_the_plane() {
    let r = Roi {
        shape: Shape::Rect,
        x: 3,
        y: 3,
        w: 10,
        h: 10,
    };
    let idx = r.indices(5, 5);
    assert!(
        idx.iter().all(|&i| i < 25),
        "an index past the plane would read another row's pixels, or past the buffer"
    );
    // Rows 3 and 4, columns 3 and 4.
    assert_eq!(idx, vec![18, 19, 23, 24]);
}

#[test]
fn an_ellipses_indices_match_its_own_containment_rule() {
    let e = Roi {
        shape: Shape::Ellipse,
        x: 0,
        y: 0,
        w: 6,
        h: 6,
    };
    let idx = e.indices(6, 6);
    assert_eq!(idx.len(), e.pixel_count());
    for &i in &idx {
        let (x, y) = ((i % 6) as u32, (i / 6) as u32);
        assert!(e.contains(x, y), "index {i} is outside the ellipse");
    }
}

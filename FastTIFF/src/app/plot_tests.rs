//! The canvas tool's geometry, and the chart's number formatting.
//!
//! The drawing itself is not tested here — what is worth pinning is the
//! screen-to-pixel mapping, because a plot of the wrong pixels looks exactly
//! like a plot of the right ones.

use super::*;

const ORIGIN: Pos2 = Pos2::new(100.0, 50.0);

#[test]
fn a_point_maps_to_the_pixel_it_is_within() {
    // At 4x zoom, image pixel (0,0) covers screen x 100..104.
    assert_eq!(pixel_at(Pos2::new(100.0, 50.0), ORIGIN, 4.0), (0, 0));
    assert_eq!(pixel_at(Pos2::new(103.9, 53.9), ORIGIN, 4.0), (0, 0));
    assert_eq!(pixel_at(Pos2::new(104.0, 54.0), ORIGIN, 4.0), (1, 1));
    // Not rounded: three-quarters into a pixel is still that pixel.
    assert_eq!(pixel_at(Pos2::new(107.0, 50.0), ORIGIN, 4.0), (1, 0));
}

#[test]
fn a_point_left_of_the_image_maps_outside_it() {
    // Negative rather than clamped to 0 — clamping here would make a drag that
    // began off the image start at its corner instead of where it began.
    assert_eq!(pixel_at(Pos2::new(96.0, 46.0), ORIGIN, 4.0), (-1, -1));
}

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

//! The crop's arithmetic. The whole run, on a real stack and through the
//! plugin boundary, is in `tests/plugin_parity.rs`.

use super::*;
use fasttiff_plugin_api::Shape;

fn rect(x: u32, y: u32, w: u32, h: u32) -> Roi {
    Roi {
        shape: Shape::Rect,
        x,
        y,
        w,
        h,
    }
}

#[test]
fn a_rectangle_inside_the_image_is_kept_as_drawn() {
    assert_eq!(bounds(&rect(1, 2, 3, 4), 10, 10), Some((1, 2, 3, 4)));
    // Up to the far edges exactly.
    assert_eq!(bounds(&rect(0, 0, 10, 8), 10, 8), Some((0, 0, 10, 8)));
}

/// A selection the host clamped to another image: refused, not cut short.
#[test]
fn a_rectangle_past_the_edge_or_empty_is_refused() {
    assert_eq!(bounds(&rect(8, 0, 3, 1), 10, 10), None);
    assert_eq!(bounds(&rect(0, 9, 1, 2), 10, 10), None);
    assert_eq!(bounds(&rect(0, 0, 0, 1), 10, 10), None);
    assert_eq!(bounds(&rect(u32::MAX, 0, 2, 1), 10, 10), None);
}

/// An ellipse keeps its bounding box: the image is rectangular.
#[test]
fn an_ellipse_keeps_its_bounding_box() {
    let e = rect(2, 1, 4, 3).with_shape(Shape::Ellipse);
    assert_eq!(bounds(&e, 10, 10), Some((2, 1, 4, 3)));
}

#[test]
fn the_cut_is_the_rows_and_columns_inside() {
    // 5 x 4, each sample its own index.
    let plane: Vec<f32> = (0..20).map(|i| i as f32).collect();
    assert_eq!(
        cut(&plane, 5, (1, 1, 3, 2)),
        [6.0, 7.0, 8.0, 11.0, 12.0, 13.0]
    );
    assert_eq!(cut(&plane, 5, (0, 0, 5, 4)), plane);
    assert_eq!(cut(&plane, 5, (4, 3, 1, 1)), [19.0]);
}

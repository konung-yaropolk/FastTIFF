//! The shape of a declared plot.

use super::*;

#[test]
fn a_plot_scales_its_own_x_axis() {
    // Uncalibrated: the index is the x value.
    let p = Plot::new("t").push(Series::new("a", vec![1.0, 2.0, 3.0]));
    assert_eq!(p.points(), 3);
    assert_eq!(p.x_at(0), 0.0);
    assert_eq!(p.x_at(2), 2.0);

    // Calibrated: seconds, from a frame interval the file stated.
    let p = p.scale(0.0, 0.133);
    assert_eq!(p.x_at(0), 0.0);
    assert!((p.x_at(2) - 0.266).abs() < 1e-9, "{}", p.x_at(2));
}

#[test]
fn the_range_spans_every_series_so_they_share_a_scale() {
    let p = Plot::new("t")
        .push(Series::new("a", vec![1.0, 5.0]))
        .push(Series::new("b", vec![-2.0, 3.0]));
    assert_eq!(p.range(), Some((-2.0, 5.0)));
}

/// A gap is a gap. The host breaks the line at a non-finite point, so the range
/// must not be dragged to infinity by one.
#[test]
fn gaps_do_not_take_part_in_the_range() {
    let s = Series::new("a", vec![f32::NAN, 2.0, f32::INFINITY, 4.0]);
    assert_eq!(s.range(), Some((2.0, 4.0)));
}

#[test]
fn a_plot_with_nothing_finite_has_no_range() {
    assert_eq!(Plot::new("t").range(), None);
    assert_eq!(
        Plot::new("t")
            .push(Series::new("a", vec![f32::NAN]))
            .range(),
        None
    );
    // And no points to draw, rather than one that cannot be placed.
    assert_eq!(Plot::new("t").points(), 0);
}

/// The longest series sets the axis: two regions measured over different
/// numbers of frames must not clip the longer one.
#[test]
fn the_axis_spans_the_longest_series() {
    let p = Plot::new("t")
        .push(Series::new("a", vec![1.0, 2.0]))
        .push(Series::new("b", vec![1.0, 2.0, 3.0, 4.0]));
    assert_eq!(p.points(), 4);
}

#[test]
fn a_plot_asks_for_no_tool_unless_it_says_so() {
    assert_eq!(Plot::new("t").wants, SelectionKind::None);
    assert_eq!(
        Plot::new("t").wants(SelectionKind::Regions).wants,
        SelectionKind::Regions
    );
}

/// A series may name its own colour so the legend can be matched to the region
/// drawn on the picture; leaving it unset hands the choice to the host.
#[test]
fn a_series_colour_is_optional() {
    assert_eq!(Series::new("a", vec![]).color, None);
    assert_eq!(
        Series::new("a", vec![]).color([1, 2, 3]).color,
        Some([1, 2, 3])
    );
}

//! The plugin's arithmetic, checked against means worked out by hand.
//!
//! A plot of the wrong number looks exactly like a plot of the right one, so
//! every expectation here is derived independently of the code under test.

use super::*;
use fasttiff_plugin_api::Shape;

fn info(channels: usize, slices: usize, frames: usize) -> ImageInfo {
    ImageInfo {
        width: 4,
        height: 3,
        channels,
        slices,
        frames,
        samples_per_pixel: 1,
        pixel_type: fasttiff_plugin_api::PixelType::U16,
    }
}

// -------------------------------------------------------------- which axes

#[test]
fn only_axes_with_more_than_one_plane_are_offered() {
    assert!(axes(&info(1, 1, 1)).is_empty(), "a single plane has none");
    assert_eq!(axes(&info(1, 1, 9)), vec![Axis::T]);
    assert_eq!(axes(&info(2, 7, 1)), vec![Axis::Z]);
}

/// Channels are not a third axis: a two-channel single-plane stack has nothing
/// to plot against, and offering one would draw a two-point line whose x axis
/// is a list of dyes.
#[test]
fn channels_are_not_a_third_axis() {
    assert!(axes(&info(4, 1, 1)).is_empty());
}

/// Both present: the longer is first, so the default shows the axis with more
/// to see.
#[test]
fn the_longer_axis_is_offered_first() {
    assert_eq!(axes(&info(1, 3, 20)), vec![Axis::T, Axis::Z]);
    assert_eq!(axes(&info(1, 30, 4)), vec![Axis::Z, Axis::T]);
}

/// The plane an index maps to differs by axis, and confusing them gives a plot
/// of the right length and the wrong data.
#[test]
fn each_axis_walks_its_own_planes() {
    assert_eq!(Axis::Z.plane(1, 5), Plane::new(1, 5, 0));
    assert_eq!(Axis::T.plane(1, 5), Plane::new(1, 0, 5));
}

// ------------------------------------------------------------- the mean

#[test]
fn the_whole_plane_mean_is_over_every_pixel() {
    let plane: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(mean_of(&plane, None), 2.5);
}

#[test]
fn a_region_mean_is_over_the_pixels_it_names() {
    // A 4x3 plane whose value is its own index.
    let plane: Vec<f32> = (0..12).map(|i| i as f32).collect();
    // Columns 1..3 of row 0: indices 1 and 2, mean 1.5.
    assert_eq!(mean_of(&plane, Some(&[1, 2])), 1.5);
    // The whole thing, named explicitly, agrees with the whole-plane path.
    let all: Vec<usize> = (0..12).collect();
    assert_eq!(mean_of(&plane, Some(&all)), mean_of(&plane, None));
}

/// A region naming nothing has no mean. NaN rather than 0: the host breaks the
/// line at a non-finite point, and 0 would draw a measurement of zero
/// brightness that was never made.
#[test]
fn a_region_covering_nothing_is_a_gap_not_a_zero() {
    let plane: Vec<f32> = vec![5.0; 4];
    assert!(mean_of(&plane, Some(&[])).is_nan());
    // And indices past the plane are skipped rather than counted as zero.
    assert_eq!(mean_of(&plane, Some(&[0, 999])), 5.0);
}

/// The running sum is `f64`. In `f32` a large plane of large samples loses
/// counts, and the mean of a region is a number people subtract from another.
#[test]
fn a_large_bright_plane_does_not_lose_counts() {
    // 100k pixels at 60000: sums to 6e9, past f32's integer precision (2^24).
    let plane: Vec<f32> = vec![60000.0; 100_000];
    assert_eq!(mean_of(&plane, None), 60000.0);
    let idx: Vec<usize> = (0..100_000).collect();
    assert_eq!(mean_of(&plane, Some(&idx)), 60000.0);
}

// ------------------------------------------------- regions reach the mean

/// The region geometry the host draws and the pixels the plugin averages must
/// be the same set — they come from one `Roi`, which is why it lives in the
/// shared contract.
#[test]
fn an_ellipse_averages_only_the_pixels_it_covers() {
    let (w, h) = (8u32, 8u32);
    let plane: Vec<f32> = (0..w * h).map(|i| (i % w) as f32).collect();
    let e = Roi {
        shape: Shape::Ellipse,
        x: 0,
        y: 0,
        w,
        h,
    };
    let got = mean_of(&plane, Some(&e.indices(w, h)));

    // Worked out independently by the stated rule: the column index, over the
    // pixels whose centres are inside.
    let (mut sum, mut n) = (0f64, 0usize);
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 + 0.5 - 4.0) / 4.0;
            let dy = (y as f64 + 0.5 - 4.0) / 4.0;
            if dx * dx + dy * dy <= 1.0 {
                sum += x as f64;
                n += 1;
            }
        }
    }
    assert_eq!(got, (sum / n as f64) as f32);
    // Not the same as its bounding box, or the ellipse was ignored.
    assert_ne!(n, (w * h) as usize);
}

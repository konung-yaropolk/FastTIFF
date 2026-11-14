//! The frame's inputs: their units, and the helpers a frontend builds them with.

use super::*;
use scivis_render::VolumeInterp;

/// The image at `min`, `size` across, seen through `visible`.
fn of_visible(
    min: [f32; 2],
    size: [f32; 2],
    vis_min: [f32; 2],
    vis_size: [f32; 2],
    ppp: f32,
) -> ImageInputs {
    let visible = LogicalRect::from_min_size(
        LogicalPt::new(vis_min[0], vis_min[1]),
        LogicalVec::new(vis_size[0], vis_size[1]),
    );
    ImageInputs::of_visible(
        LogicalPt::new(min[0], min[1]),
        LogicalVec::new(size[0], size[1]),
        visible,
        ppp,
    )
}

#[test]
fn physical_pixels_are_points_times_the_scale() {
    let px = LogicalVec::new(800.0, 600.0).to_physical(1.25);
    assert_eq!(px, PhysicalPx::new(1000.0, 750.0));
    assert!(px.is_known());
}

#[test]
fn an_unknown_size_is_not_a_size() {
    assert!(!PhysicalPx::UNKNOWN.is_known());
    assert!(!PhysicalPx::new(0.5, 100.0).is_known(), "under a pixel");
    assert!(!PhysicalPx::new(f32::NAN, 100.0).is_known());
    assert!(!PhysicalPx::new(f32::INFINITY, 100.0).is_known());
    assert_eq!(PhysicalPx::default(), PhysicalPx::UNKNOWN);
}

/// What a render target is made at: rounded, and never empty, since a GPU
/// will not make a texture of no pixels.
#[test]
fn whole_pixels_round_and_never_reach_zero() {
    assert_eq!(PhysicalPx::new(1919.6, 1080.4).whole(), [1920, 1080]);
    assert_eq!(PhysicalPx::UNKNOWN.whole(), [1, 1]);
    assert_eq!(PhysicalPx::new(0.2, 3.5).whole(), [1, 4]);
}

/// The screen's longer side, which caps a volume's detail: 0 says "no cap".
#[test]
fn the_longest_side_is_whole_and_zero_when_unknown() {
    assert_eq!(PhysicalPx::new(2560.0, 1440.0).longest_side(), 2560);
    assert_eq!(PhysicalPx::new(1440.0, 2560.4).longest_side(), 2560);
    assert_eq!(PhysicalPx::UNKNOWN.longest_side(), 0);
}

/// Taking the longer side after the scale is the same number as scaling the
/// longer side, to the bit: what a frontend that used to send the latter now
/// sends as a size.
#[test]
fn the_longest_side_scales_like_the_side_it_was() {
    for (w, h, ppp) in [
        (1920.0_f32, 1080.0, 1.25_f32),
        (1366.0, 768.0, 1.5),
        (3.3, 7.7, 2.75),
    ] {
        let before = (w.max(h) * ppp).round().max(0.0) as u32;
        assert_eq!(
            LogicalVec::new(w, h).to_physical(ppp).longest_side(),
            before
        );
    }
}

#[test]
fn the_whole_image_is_the_default_view() {
    let i = ImageInputs::default();
    assert_eq!((i.uv_offset, i.uv_scale), ([0.0, 0.0], [1.0, 1.0]));
    assert_eq!(i.viewport, PhysicalPx::UNKNOWN);
    let f = FrameInputs::default();
    assert!(!f.moving && !f.histograms);
    assert_eq!(f.image, i);
    assert_eq!(f.volume.pose, None);
    assert_eq!(f.volume.downscale, 1);
}

/// An image that fits: all of it, in as many pixels as it is drawn at.
#[test]
fn a_fitted_image_is_all_on_screen() {
    let i = of_visible(
        [10.0, 20.0],
        [400.0, 300.0],
        [10.0, 20.0],
        [400.0, 300.0],
        2.0,
    );
    assert_eq!(i.uv_offset, [0.0, 0.0]);
    assert_eq!(i.uv_scale, [1.0, 1.0]);
    assert_eq!(i.viewport, PhysicalPx::new(800.0, 600.0));
}

/// Zoomed in and panned: the panel shows the window of the image under it,
/// in fractions of the image, and its own size in pixels.
#[test]
fn an_overflowing_image_shows_the_window_under_the_panel() {
    // The image is drawn 1000 x 500 points from (-200, -100): panned right
    // and down by a fifth. The panel is 500 x 250 points at (0, 0).
    let i = of_visible(
        [-200.0, -100.0],
        [1000.0, 500.0],
        [0.0, 0.0],
        [500.0, 250.0],
        1.0,
    );
    assert_eq!(i.uv_offset, [0.2, 0.2]);
    assert_eq!(i.uv_scale, [0.5, 0.5]);
    assert_eq!(i.viewport, PhysicalPx::new(500.0, 250.0));
}

/// The same arithmetic, to the bit, as the canvas did inline before it moved
/// here, so the residency planned from it is the same.
#[test]
fn the_window_is_the_canvas_arithmetic() {
    let (origin, img, vis_min, vis_max, ppp) = (
        [13.7_f32, -41.3],
        [2345.6_f32, 1234.5],
        [20.0_f32, 30.0],
        [831.3_f32, 637.9],
        1.25_f32,
    );
    // The visible rect's size as the canvas took it, `max - min`.
    let vis = [vis_max[0] - vis_min[0], vis_max[1] - vis_min[1]];
    let inv = [1.0 / img[0].max(1.0), 1.0 / img[1].max(1.0)];
    let i = ImageInputs::of_visible(
        LogicalPt::new(origin[0], origin[1]),
        LogicalVec::new(img[0], img[1]),
        LogicalRect {
            min: LogicalPt::new(vis_min[0], vis_min[1]),
            max: LogicalPt::new(vis_max[0], vis_max[1]),
        },
        ppp,
    );
    assert_eq!(
        i.uv_offset,
        [
            (vis_min[0] - origin[0]) * inv[0],
            (vis_min[1] - origin[1]) * inv[1]
        ]
    );
    assert_eq!(i.uv_scale, [vis[0] * inv[0], vis[1] * inv[1]]);
    assert_eq!(i.viewport, PhysicalPx::new(vis[0] * ppp, vis[1] * ppp));
}

/// A degenerate image (nothing laid out yet) gives a finite window rather
/// than dividing by zero.
#[test]
fn an_empty_image_gives_a_finite_window() {
    let i = of_visible([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 1.0);
    assert!(i.uv_offset.iter().chain(&i.uv_scale).all(|v| v.is_finite()));
    assert!(!i.viewport.is_known());
}

#[test]
fn the_volume_aspect_follows_its_viewport_within_bounds() {
    let v = |w, h| VolumeInputs {
        viewport: PhysicalPx::new(w, h),
        ..Default::default()
    };
    assert!((v(1600.0, 900.0).aspect() - 16.0 / 9.0).abs() < 1e-6);
    assert_eq!(v(10_000.0, 10.0).aspect(), 10.0, "clamped wide");
    assert_eq!(v(10.0, 10_000.0).aspect(), 0.1, "clamped tall");
    assert_eq!(
        v(100.0, 0.0).aspect(),
        10.0,
        "no height is not a division by zero"
    );
}

/// A reduced frame's rays take the steps a sharp frame's do, so it does a
/// sixteenth of the work at a quarter the size.
#[test]
fn a_reduced_view_takes_as_many_steps_per_ray_as_a_sharp_one() {
    let mut v = VolumeInputs {
        viewport: PhysicalPx::new(1921.0, 1081.0),
        ..Default::default()
    };
    let sharp = v.max_steps(2, VolumeInterp::Linear);
    assert_eq!(
        sharp,
        scivis_render::max_steps(1921.0 * 1081.0, 2, VolumeInterp::Linear),
        "the whole viewport's"
    );
    v.downscale = 4;
    assert_eq!(v.max_steps(2, VolumeInterp::Linear), sharp);
}

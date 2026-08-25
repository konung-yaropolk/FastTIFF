//! The zoom ladder, the fit factor that joins it for one image, and the glide
//! a step takes to get there.
//!
//! The property that matters for the ladder is that stepping is *monotonic and
//! reaches the fitted view*: the whole reason the fit factor is inserted as a
//! rung rather than applied once is so that scrolling back out lands on it
//! again.
//!
//! For the glide it is that it ends, ends where it was aimed, and behaves the
//! same however the frames fall. An animation that overshoots, drifts, or runs
//! for ever is worse than the jump it replaced, and one that depends on frame
//! rate looks different on every machine.

use super::*;

#[test]
fn without_a_fit_the_ladder_is_the_fixed_one() {
    assert_eq!(zoom_ladder(None), ZOOM_LEVELS.to_vec());
    // A nonsense factor is ignored rather than poisoning the ladder.
    for bad in [f32::NAN, f32::INFINITY, 0.0, -0.5] {
        assert_eq!(zoom_ladder(Some(bad)), ZOOM_LEVELS.to_vec(), "{bad} was inserted");
    }
}

#[test]
fn a_fit_between_two_levels_becomes_a_rung_in_order() {
    let ladder = zoom_ladder(Some(0.287));
    assert_eq!(ladder.len(), ZOOM_LEVELS.len() + 1);
    assert!(ladder.windows(2).all(|w| w[0] < w[1]), "ladder not sorted: {ladder:?}");
    let at = ladder.iter().position(|z| *z == 0.287).expect("the fit is a rung");
    assert_eq!((ladder[at - 1], ladder[at + 1]), (0.25, 0.333));
}

#[test]
fn a_fit_that_is_already_a_level_adds_nothing() {
    // Exactly on one, and near enough to one to be indistinguishable.
    for f in [0.5, 0.501, 0.4995, 1.0] {
        assert_eq!(zoom_ladder(Some(f)).len(), ZOOM_LEVELS.len(), "{f} duplicated a level");
    }
    // Just outside the snap tolerance, so it does earn its own rung.
    assert_eq!(zoom_ladder(Some(0.52)).len(), ZOOM_LEVELS.len() + 1);
}

#[test]
fn zooming_out_from_the_fit_and_back_returns_to_it() {
    // The behaviour the feature is for: the fitted view is reachable by wheel,
    // not just at load.
    let fit = 0.287;
    let inned = stepped_zoom(fit, 1, Some(fit));
    assert!(inned > fit, "zooming in did not move: {inned}");
    assert_eq!(stepped_zoom(inned, -1, Some(fit)), fit, "did not come back to the fit");

    let outed = stepped_zoom(fit, -1, Some(fit));
    assert!(outed < fit, "zooming out did not move: {outed}");
    assert_eq!(stepped_zoom(outed, 1, Some(fit)), fit);
}

#[test]
fn a_fit_below_every_level_becomes_the_new_floor() {
    // A mosaic can need less than the ladder's 3.1%; the fitted view must still
    // be reachable, and zooming out must stop there rather than above it.
    let fit = 0.004;
    let ladder = zoom_ladder(Some(fit));
    assert_eq!(ladder[0], fit);
    assert_eq!(stepped_zoom(fit, -1, Some(fit)), fit, "zoom-out should clamp at the fit");
    assert!(stepped_zoom(fit, 1, Some(fit)) > fit);
}

#[test]
fn stepping_is_unchanged_when_there_is_no_fit() {
    assert_eq!(stepped_zoom(1.0, 1, None), 1.5);
    assert_eq!(stepped_zoom(1.0, -1, None), 0.75);
    assert_eq!(stepped_zoom(32.0, 1, None), 32.0, "clamps at the top");
    assert_eq!(stepped_zoom(0.031, -1, None), 0.031, "clamps at the bottom");
}

#[test]
fn the_fit_fills_the_shorter_axis_and_never_magnifies() {
    // Letterboxed: limited by height.
    let z = fit_to_panel(1000.0, 1000.0, egui::vec2(2000.0, 500.0)).unwrap();
    assert!((z - 0.5).abs() < 1e-6, "{z}");
    // Pillarboxed: limited by width.
    let z = fit_to_panel(1000.0, 1000.0, egui::vec2(400.0, 5000.0)).unwrap();
    assert!((z - 0.4).abs() < 1e-6, "{z}");
    // A small image is left at 1:1 rather than blown up — nearest sampling
    // makes a magnified thumbnail a block of squares, not a bigger picture.
    assert_eq!(fit_to_panel(100.0, 80.0, egui::vec2(1600.0, 900.0)), Some(1.0));
}

#[test]
fn the_fit_declines_a_canvas_that_has_no_size_yet() {
    assert_eq!(fit_to_panel(1000.0, 1000.0, egui::vec2(0.0, 0.0)), None);
    assert_eq!(fit_to_panel(1000.0, 1000.0, egui::vec2(800.0, 0.0)), None);
    assert_eq!(fit_to_panel(0.0, 0.0, egui::vec2(800.0, 600.0)), None);
    assert_eq!(fit_to_panel(f32::NAN, 10.0, egui::vec2(800.0, 600.0)), None);
}

// ---------------------------------------------------------------------------
// The glide
// ---------------------------------------------------------------------------

/// A view zoomed to `from`, with the geometry a laid-out panel would have.
fn gliding(from: f32, target: f32) -> View2d {
    View2d {
        zoom: from,
        panel_rect: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 800.0)),
        image_origin: egui::pos2(0.0, 0.0),
        zoom_glide: Some(ZoomGlide { target, anchor: egui::pos2(500.0, 400.0) }),
        ..Default::default()
    }
}

/// Run a glide to completion at a fixed frame time, returning the zoom after
/// each frame. Bounded so a glide that never finishes fails the test instead of
/// hanging it.
fn run(v: &mut View2d, dt: f32) -> Vec<f32> {
    let mut seen = Vec::new();
    for _ in 0..600 {
        let running = v.advance_zoom_glide(dt);
        seen.push(v.zoom);
        if !running {
            return seen;
        }
    }
    panic!("glide still running after 600 frames of {dt}s");
}

/// The basic contract: it arrives, exactly, and stops.
#[test]
fn a_glide_reaches_its_target_and_stops() {
    let mut v = gliding(1.0, 4.0);
    let seen = run(&mut v, 1.0 / 60.0);
    assert_eq!(v.zoom, 4.0, "a glide must land on the rung, not near it");
    assert!(v.zoom_glide.is_none(), "and must not still be running");
    assert!(!v.advance_zoom_glide(1.0 / 60.0), "nor restart when advanced again");
    assert!(seen.len() > 2, "it should take more than a frame, or it is not an animation");
}

/// Fast enough not to be in the way. At 60 Hz a step should be visually done
/// inside about a fifth of a second: long enough to see, short enough that a
/// second notch never queues up behind it.
#[test]
fn a_glide_is_over_quickly() {
    for (from, to) in [(1.0, 2.0), (1.0, 8.0), (8.0, 1.0), (0.125, 4.0)] {
        let mut v = gliding(from, to);
        let frames = run(&mut v, 1.0 / 60.0).len();
        assert!(
            frames <= 15,
            "{from} -> {to} took {frames} frames at 60 Hz, which is over a quarter of a second"
        );
    }
}

/// Monotonic, and never past the target. An overshoot on a zoom reads as a
/// wobble, which is worse than the snap this replaced.
#[test]
fn a_glide_never_overshoots_or_backtracks() {
    for (from, to) in [(1.0, 8.0), (8.0, 1.0), (0.5, 0.75), (3.0, 0.125)] {
        let mut v = gliding(from, to);
        let seen = run(&mut v, 1.0 / 60.0);
        let up = to > from;
        for w in seen.windows(2) {
            if up {
                assert!(w[1] >= w[0] - 1e-6, "{from} -> {to} went backwards: {seen:?}");
                assert!(w[1] <= to + 1e-6, "{from} -> {to} overshot: {seen:?}");
            } else {
                assert!(w[1] <= w[0] + 1e-6, "{from} -> {to} went backwards: {seen:?}");
                assert!(w[1] >= to - 1e-6, "{from} -> {to} overshot: {seen:?}");
            }
        }
    }
}

/// The same gesture on a slow machine and a fast one must take the same *time*,
/// not the same number of frames. Advancing by a fixed fraction per frame, the
/// obvious way to write this, would make the zoom crawl at 30 Hz and race at
/// 240 Hz.
#[test]
fn a_glide_follows_the_clock_rather_than_the_frame_rate() {
    let elapsed: Vec<f32> = [30.0f32, 60.0, 144.0]
        .iter()
        .map(|hz| {
            let dt = 1.0 / hz;
            let mut v = gliding(1.0, 8.0);
            run(&mut v, dt).len() as f32 * dt
        })
        .collect();
    let lo = elapsed.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = elapsed.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        hi - lo < 0.05,
        "arrival time varies with frame rate: {elapsed:?} seconds at 30/60/144 Hz"
    );
}

/// A stalled frame must not be integrated whole. Without a ceiling on `dt`, one
/// slow frame (a decode, a window drag) teleports the zoom and undoes the point
/// of animating it.
#[test]
fn one_very_long_frame_does_not_swallow_the_whole_glide() {
    let mut v = gliding(1.0, 8.0);
    v.advance_zoom_glide(2.0);
    assert!(v.zoom < 8.0, "a two-second frame jumped straight to the target");
    assert!(v.zoom_glide.is_some(), "and ended the glide with it");
}

/// The anchor is the point under the cursor, and the whole reason a zoom is
/// anchored at all: whatever is under the pointer must stay under it for every
/// frame of the glide, not merely at the end.
#[test]
fn the_anchored_point_stays_put_for_the_whole_glide() {
    let anchor = egui::pos2(500.0, 400.0);
    let mut v = gliding(1.0, 4.0);
    let p0 = (anchor - v.image_origin) / v.zoom;
    for _ in 0..600 {
        let running = v.advance_zoom_glide(1.0 / 60.0);
        // `image_origin` is the previous frame's cache; the drawing recomputes
        // it as `panel_rect.min - pan` on an overflowing axis, which is what
        // this reproduces. (Getting the sign wrong here is how this test first
        // failed against correct code.)
        v.image_origin = v.panel_rect.min - v.pan;
        let here = v.image_origin + p0 * v.zoom;
        assert!(
            (here - anchor).length() < 0.5,
            "the anchored point drifted to {here:?} from {anchor:?} at zoom {}",
            v.zoom
        );
        if !running {
            return;
        }
    }
    panic!("glide never finished");
}

/// Nothing outside the drawing should see the intermediate values: the window
/// is sized once, for where the zoom is going.
#[test]
fn the_settled_zoom_is_the_destination_while_gliding() {
    let mut v = gliding(1.0, 4.0);
    assert_eq!(v.zoom_settled(), 4.0, "mid-glide it is the target");
    run(&mut v, 1.0 / 60.0);
    assert_eq!(v.zoom_settled(), 4.0, "and afterwards it is simply the zoom");
    assert_eq!(v.zoom, 4.0);

    let still = View2d { zoom: 2.5, ..Default::default() };
    assert_eq!(still.zoom_settled(), 2.5, "with no glide it is what is on screen");
}

/// Stepping again mid-glide must advance from the destination, or a fast flick
/// of the wheel would keep re-deciding the same rung and go nowhere.
#[test]
fn a_second_step_mid_glide_advances_another_rung() {
    let mut v = gliding(1.0, 2.0);
    v.advance_zoom_glide(1.0 / 60.0);
    assert!(v.zoom_glide.is_some(), "still gliding");
    let next = stepped_zoom(v.zoom_settled(), 1, None);
    assert!(next > 2.0, "a second notch should pass the rung being approached, got {next}");
}

/// A glide asked to go where it already is finishes rather than spinning, and a
/// nonsensical one is dropped instead of producing NaN.
#[test]
fn a_degenerate_glide_ends_immediately() {
    let mut v = gliding(3.0, 3.0);
    assert!(!v.advance_zoom_glide(1.0 / 60.0), "already there");
    assert!(v.zoom_glide.is_none());

    for bad in [0.0f32, -1.0] {
        let mut v = gliding(bad, 2.0);
        assert!(!v.advance_zoom_glide(1.0 / 60.0), "zoom {bad} should not glide");
        assert!(v.zoom_glide.is_none());
        let mut v = gliding(1.0, bad);
        assert!(!v.advance_zoom_glide(1.0 / 60.0), "target {bad} should not glide");
        assert!(v.zoom_glide.is_none());
    }
}

/// Zooming in and zooming out must feel the same.
///
/// Zoom is geometric — the ladder is a sequence of ratios, and 1 to 2 is the
/// same gesture as 2 to 1 — so the glide has to be integrated in the logarithm.
/// Interpolating the zoom *linearly* passes every other test here and still
/// gets this wrong: from 1 to 8 the first frame would cover 2.7x while the
/// mirror image from 8 to 1 covered only 1.5x, so zooming out would crawl where
/// zooming in raced.
///
/// Stated as the property rather than the implementation: however far a glide
/// has left to run, measured as a ratio, must fall by the same factor each
/// frame whichever direction it is going.
#[test]
fn zooming_in_and_out_are_mirror_images() {
    let remaining = |from: f32, to: f32| -> Vec<f32> {
        let mut v = gliding(from, to);
        let mut out = Vec::new();
        for _ in 0..600 {
            let running = v.advance_zoom_glide(1.0 / 60.0);
            // How far there is still to go, as a ratio, in log units.
            out.push((v.zoom.ln() - to.ln()).abs());
            if !running {
                break;
            }
        }
        out
    };

    let up = remaining(1.0, 8.0);
    let down = remaining(8.0, 1.0);
    assert_eq!(up.len(), down.len(), "in and out took different numbers of frames");
    for (i, (a, b)) in up.iter().zip(&down).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "frame {i}: {a} left going in against {b} going out — the two directions differ"
        );
    }

    // And the decay really is constant, which is what makes the motion read as
    // one smooth movement rather than a lunge that peters out.
    // Excluding the last frame, which snaps exactly onto the target and so has
    // no ratio to speak of.
    let ratios: Vec<f32> = up
        .windows(2)
        .filter(|w| w[0] > 1e-3 && w[1] > 1e-3)
        .map(|w| w[1] / w[0])
        .collect();
    assert!(ratios.len() >= 3, "too few frames to say anything about the decay");
    let lo = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = ratios.iter().cloned().fold(0.0f32, f32::max);
    assert!(hi - lo < 0.01, "the per-frame decay is not constant: {ratios:?}");
}

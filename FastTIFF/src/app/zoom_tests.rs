//! The zoom ladder, and the fit factor that joins it for one image.
//!
//! The property that matters is that stepping is *monotonic and reaches the
//! fitted view*: the whole reason the fit factor is inserted as a rung rather
//! than applied once is so that scrolling back out lands on it again.

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

//! The glide's arithmetic. Feel is a matter of taste; these pin the properties
//! that are not — that it always stops, never reverses, and never travels
//! further than the flick that started it.

use super::*;

/// Drive `frames` of steady movement at `dt`, then let go and coast to a stop.
/// Returns (distance while pushing, distance while coasting, frames coasted).
fn flick(delta: egui::Vec2, dt: f32, frames: usize) -> (egui::Vec2, egui::Vec2, usize) {
    let mut g = Glide::default();
    let mut pushed = egui::Vec2::ZERO;
    for _ in 0..frames {
        pushed += g.push(delta, dt);
    }
    let mut coasted = egui::Vec2::ZERO;
    let mut n = 0;
    while g.is_moving() && n < 10_000 {
        coasted += g.coast(dt);
        n += 1;
    }
    (pushed, coasted, n)
}

#[test]
fn input_is_passed_through_untouched() {
    // The gesture itself must track the fingers exactly; only what follows is
    // ours to invent.
    let mut g = Glide::default();
    let d = egui::vec2(13.0, -7.0);
    assert_eq!(g.push(d, 1.0 / 60.0), d);
}

#[test]
fn a_flick_coasts_on_and_then_stops() {
    let (_, coasted, frames) = flick(egui::vec2(0.0, -20.0), 1.0 / 60.0, 8);
    assert!(
        coasted.y < 0.0,
        "the coast should continue in the same direction"
    );
    assert!(frames > 1, "a flick should glide for more than one frame");
    assert!(
        frames < 120,
        "a glide should be over inside two seconds, was {frames} frames"
    );
}

#[test]
fn the_glide_never_outruns_the_gesture() {
    // Coasting further than the flick that caused it feels like losing control
    // of the picture.
    let (pushed, coasted, _) = flick(egui::vec2(0.0, -20.0), 1.0 / 60.0, 30);
    assert!(
        coasted.y.abs() <= pushed.y.abs(),
        "coasted {} vs pushed {}",
        coasted.y.abs(),
        pushed.y.abs()
    );
}

#[test]
fn it_keeps_the_direction_it_was_given() {
    let cases = [
        egui::vec2(30.0, 0.0),
        egui::vec2(-30.0, 0.0),
        egui::vec2(0.0, 25.0),
        egui::vec2(11.0, -9.0),
    ];
    for d in cases {
        let (_, coasted, _) = flick(d, 1.0 / 60.0, 10);
        assert!(coasted.x * d.x >= 0.0, "x reversed for {d:?}: {coasted:?}");
        assert!(coasted.y * d.y >= 0.0, "y reversed for {d:?}: {coasted:?}");
    }
}

#[test]
fn a_slow_drag_does_not_glide() {
    // Placing the view somewhere deliberately should leave it there.
    let mut g = Glide::default();
    for _ in 0..20 {
        g.push(egui::vec2(0.0, -0.2), 1.0 / 60.0);
    }
    assert!(!g.is_moving(), "a crawl should not fling");
}

#[test]
fn stopping_is_immediate() {
    let mut g = Glide::default();
    for _ in 0..10 {
        g.push(egui::vec2(0.0, -30.0), 1.0 / 60.0);
    }
    assert!(g.is_moving());
    g.stop();
    assert!(!g.is_moving());
    assert_eq!(g.coast(1.0 / 60.0), egui::Vec2::ZERO);
}

#[test]
fn an_axis_can_be_stopped_alone() {
    // Running into the left edge must not also stop vertical motion.
    let mut g = Glide::default();
    for _ in 0..10 {
        g.push(egui::vec2(-30.0, -30.0), 1.0 / 60.0);
    }
    g.stop_axes(true, false);
    let step = g.coast(1.0 / 60.0);
    assert_eq!(step.x, 0.0, "x should be stopped");
    assert!(step.y < 0.0, "y should still be running");
}

#[test]
fn a_degenerate_frame_time_is_survivable() {
    // `stable_dt` is zero on the first frame, and a stalled app can report
    // something absurd. Neither may produce a velocity that never stops.
    let mut g = Glide::default();
    g.push(egui::vec2(10.0, 10.0), 0.0);
    assert!(!g.is_moving(), "a zero frame time should not fling");
    g.push(egui::vec2(10.0, 10.0), f32::NAN);
    assert!(!g.is_moving());

    let mut g = Glide::default();
    for _ in 0..10 {
        g.push(egui::vec2(0.0, -30.0), 1.0 / 60.0);
    }
    assert_eq!(g.coast(0.0), egui::Vec2::ZERO);
    assert_eq!(g.coast(f32::NAN), egui::Vec2::ZERO);
    assert!(
        g.is_moving(),
        "a bad frame should not silently end the glide"
    );
}

//! The autoscroll response curve.
//!
//! Three promises, and each has a way of being broken that would only show up
//! as "the mode feels wrong" rather than as anything failing:
//!
//! - it **rests**, so there is a neutral position and the view can be stopped;
//! - it **points where the hand points**, exactly, at every angle;
//! - it is **bounded**, so the far corner of a large display is fast and not a
//!   strobe.

use super::*;

const AT: egui::Pos2 = egui::Pos2 { x: 400.0, y: 300.0 };

/// Resting. The pointer never sits exactly where it was clicked — the click
/// itself nudges it — so a curve through the origin would make the view creep
/// with no way to stop it, and the mode would be unusable rather than merely
/// twitchy.
#[test]
fn inside_the_dead_zone_nothing_moves() {
    for (dx, dy) in [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, -3.0),
        (5.0, 5.0),
        (DEAD_ZONE - 0.1, 0.0),
        (0.0, DEAD_ZONE - 0.1),
        // On the boundary itself: resting, so the mode has a definite edge
        // rather than a sliver where it scrolls immeasurably slowly.
        (DEAD_ZONE, 0.0),
    ] {
        let v = velocity(AT, AT + egui::vec2(dx, dy));
        assert_eq!(
            v,
            egui::Vec2::ZERO,
            "({dx}, {dy}) is within the dead zone but gave {v:?}"
        );
    }
}

/// Just past the dead zone it moves, and slowly — the gesture has to start
/// gently or there is no low gear to steer with.
#[test]
fn just_outside_the_dead_zone_it_creeps() {
    let v = velocity(AT, AT + egui::vec2(DEAD_ZONE + 1.0, 0.0));
    assert!(v.x > 0.0, "should have started moving: {v:?}");
    assert!(
        v.x < GAIN * 2.0,
        "one point past the zone should be one point's worth of speed: {v:?}"
    );
    assert_eq!(v.y, 0.0, "a horizontal offset must not scroll vertically");
}

/// The direction is the direction, at every angle. This is the test that would
/// catch a per-axis dead zone: subtracting it from each component separately
/// leaves the velocity pointing somewhere other than at the pointer for every
/// angle that is not a multiple of 45°.
#[test]
fn the_view_moves_exactly_towards_the_pointer() {
    for i in 0..32 {
        let a = i as f32 * std::f32::consts::TAU / 32.0;
        let dir = egui::vec2(a.cos(), a.sin());
        // Well outside the dead zone, so the cap is not in play either.
        let v = velocity(AT, AT + dir * 100.0);
        let got = v.length();
        assert!(got > 0.0, "angle {a} gave no motion");
        // Compare unit vectors: the magnitude is the next test's business.
        let off = ((v / got) - dir).length();
        assert!(
            off < 1e-4,
            "angle {a}: moving towards {:?}, not {dir:?}",
            v / got
        );
    }
}

/// Twice as far is twice as fast, measured past the dead zone rather than past
/// the anchor — the distinction that makes the low gear exist.
#[test]
fn speed_is_proportional_to_the_offset_past_the_dead_zone() {
    let one = velocity(AT, AT + egui::vec2(DEAD_ZONE + 50.0, 0.0)).x;
    let two = velocity(AT, AT + egui::vec2(DEAD_ZONE + 100.0, 0.0)).x;
    assert!(
        (two - one * 2.0).abs() < 1e-2,
        "{one} then {two}: should have doubled"
    );
    assert!(
        (one - 50.0 * GAIN).abs() < 1e-2,
        "50 points past the zone should be {} points/s, got {one}",
        50.0 * GAIN
    );
}

/// Monotonic, so pushing further never scrolls slower — including across the
/// cap, where a naive clamp written the wrong way round would fold back.
#[test]
fn pushing_further_never_scrolls_slower() {
    let mut last = -1.0;
    for d in [0.0f32, 6.0, 12.0, 13.0, 50.0, 200.0, 700.0, 2000.0, 20000.0] {
        let got = velocity(AT, AT + egui::vec2(d, 0.0)).length();
        assert!(
            got >= last - 1e-3,
            "slowed down at {d} points: {got} < {last}"
        );
        last = got;
    }
}

/// Bounded. A pointer in the corner of a 4K display is some 2000 points out,
/// which ungoverned would be eight screens a second — a strobe rather than a
/// motion, and impossible to stop in the right place.
#[test]
fn the_speed_is_capped() {
    for d in [1000.0f32, 5000.0, 100_000.0] {
        let got = velocity(AT, AT + egui::vec2(d, d)).length();
        assert!(
            got <= MAX_SPEED + 1e-2,
            "{d} points out gave {got} points/s, over the {MAX_SPEED} cap"
        );
    }
    // And the cap is actually reached, or it is not a cap but a coincidence.
    let far = velocity(AT, AT + egui::vec2(100_000.0, 0.0)).length();
    assert!(
        (far - MAX_SPEED).abs() < 1e-2,
        "expected the cap, got {far}"
    );
}

/// Both signs, on both axes. A dropped sign here scrolls the view away from the
/// pointer, which is the one failure that makes the mode actively hostile.
#[test]
fn every_direction_carries_its_own_sign() {
    let d = DEAD_ZONE + 100.0;
    assert!(velocity(AT, AT + egui::vec2(d, 0.0)).x > 0.0, "right");
    assert!(velocity(AT, AT + egui::vec2(-d, 0.0)).x < 0.0, "left");
    assert!(velocity(AT, AT + egui::vec2(0.0, d)).y > 0.0, "down");
    assert!(velocity(AT, AT + egui::vec2(0.0, -d)).y < 0.0, "up");
}

/// A non-finite pointer position — which a lost or synthesised event can
/// produce — must not reach the pan. Unlike a wrong number, a NaN pan cannot be
/// undone by moving the mouse back: it poisons the transform for every frame
/// after, and the window has to be closed.
#[test]
fn a_non_finite_pointer_does_not_move_the_view() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for p in [egui::pos2(bad, 300.0), egui::pos2(400.0, bad)] {
            let v = velocity(AT, p);
            assert_eq!(v, egui::Vec2::ZERO, "pointer {p:?} gave {v:?}");
        }
        // And a non-finite anchor, which is the same arithmetic the other way.
        let v = velocity(egui::pos2(bad, bad), AT);
        assert_eq!(v, egui::Vec2::ZERO, "anchor ({bad}, {bad}) gave {v:?}");
    }
}

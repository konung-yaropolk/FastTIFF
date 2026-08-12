//! Unit tests for the volume camera math. These became testable when the camera
//! moved out of the egui app — the ray/box and basis math never needed a GPU or
//! a window, only a caller that wasn't holding an `egui::Ui`.

use super::*;

/// Squared distance between two points, for approximate comparisons.
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum()
}

#[test]
fn basis_is_orthonormal() {
    for &(yaw, pitch) in &[(0.0, 0.0), (0.7, 0.5), (-2.1, -1.2), (3.0, 1.5)] {
        let (f, r, u) = volume_basis(yaw, pitch);
        for v in [f, r, u] {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "not unit length at ({yaw}, {pitch}): {len}");
        }
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!(dot(f, r).abs() < 1e-4, "forward·right at ({yaw}, {pitch})");
        assert!(dot(f, u).abs() < 1e-4, "forward·up at ({yaw}, {pitch})");
        assert!(dot(r, u).abs() < 1e-4, "right·up at ({yaw}, {pitch})");
    }
}

#[test]
fn pitch_is_clamped_away_from_the_pole() {
    // Straight up would make `cross(forward, world_up)` degenerate.
    let (f, r, u) = volume_basis(0.0, 100.0);
    for v in [f, r, u] {
        assert!(v.iter().all(|c| c.is_finite()), "non-finite basis at extreme pitch");
    }
}

#[test]
fn focal_box_entry_hits_misses_and_insides() {
    let he = [0.5, 0.5, 0.5];
    // Looking at the box from outside along -Z: enters at z = 0.5, so t = 1.5.
    let t = focal_box_entry([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], he).expect("should hit");
    assert!((t - 1.5).abs() < 1e-4, "expected 1.5, got {t}");
    // Eye inside the box: entry distance is 0.
    assert_eq!(focal_box_entry([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], he), Some(0.0));
    // Aimed away from the box entirely.
    assert_eq!(focal_box_entry([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], he), None);
    // Parallel to a slab and outside it.
    assert_eq!(focal_box_entry([0.0, 5.0, 2.0], [0.0, 0.0, -1.0], he), None);
}

#[test]
fn orbit_eye_sits_at_distance_from_the_pivot() {
    let cam = CameraState { nav: NavMode::Cad, dist: 3.0, target: [0.0, 0.0, 0.0], ..Default::default() };
    let (forward, _, _) = cam.basis();
    let eye = cam.eye(forward);
    assert!((dist2(eye, cam.target).sqrt() - 3.0).abs() < 1e-4);
}

#[test]
fn nav_switch_keeps_the_eye_put() {
    // The orbit and fly modes store the eye differently; switching must not move
    // it, or the view visibly jumps.
    let mut cam = CameraState { nav: NavMode::Cad, dist: 2.5, target: [0.3, -0.2, 0.1], ..Default::default() };
    let (forward, _, _) = cam.basis();
    let before = cam.eye(forward);

    cam.nav = NavMode::WasdFly;
    cam.sync_for_nav(/* was_fly */ false);
    let after = cam.eye(cam.basis().0);
    assert!(dist2(before, after) < 1e-8, "orbit -> fly moved the eye: {before:?} -> {after:?}");

    cam.nav = NavMode::Cad;
    cam.sync_for_nav(/* was_fly */ true);
    let back = cam.eye(cam.basis().0);
    assert!(dist2(before, back) < 1e-6, "fly -> orbit moved the eye: {before:?} -> {back:?}");
}

#[test]
fn orbiting_the_volume_center_preserves_the_pivot_radius() {
    // The volume-center orbit rotates the target about the origin, so its
    // distance from the origin is invariant — that's what keeps an off-center
    // framing from snapping back to center.
    let mut cam = CameraState { orbit_point: OrbitPoint::VolumeCenter, target: [0.4, 0.1, -0.2], ..Default::default() };
    let r0 = dist2(cam.target, [0.0; 3]).sqrt();
    for _ in 0..20 {
        cam.orbit_drag(7.0, -3.0);
    }
    let r1 = dist2(cam.target, [0.0; 3]).sqrt();
    assert!((r0 - r1).abs() < 1e-4, "radius drifted: {r0} -> {r1}");
}

#[test]
fn box_half_extents_normalize_the_longest_scaled_axis() {
    let cam = CameraState::default();
    // Anisotropic voxels: Z is 4x the physical spacing, and only 10 slices.
    let v = volume_camera(&cam, [1.0, 1.0, 4.0], (100, 50, 10));
    // Longest physical axis (x: 100) maps to half-extent 0.5.
    assert!((v.box_he[0] - 0.5).abs() < 1e-4, "{:?}", v.box_he);
    assert!((v.box_he[1] - 0.25).abs() < 1e-4, "{:?}", v.box_he);
    // z: 10 * 4 = 40 physical units vs. 100 -> 0.2
    assert!((v.box_he[2] - 0.2).abs() < 1e-4, "{:?}", v.box_he);
}

#[test]
fn degenerate_dimensions_stay_finite() {
    let cam = CameraState::default();
    let v = volume_camera(&cam, [0.0, 0.0, 0.0], (0, 0, 0));
    assert!(v.box_he.iter().all(|c| c.is_finite() && *c > 0.0), "{:?}", v.box_he);
}

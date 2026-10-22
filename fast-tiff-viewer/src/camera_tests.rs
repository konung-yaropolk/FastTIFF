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
            assert!(
                (len - 1.0).abs() < 1e-4,
                "not unit length at ({yaw}, {pitch}): {len}"
            );
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
        assert!(
            v.iter().all(|c| c.is_finite()),
            "non-finite basis at extreme pitch"
        );
    }
}

#[test]
fn focal_box_entry_hits_misses_and_insides() {
    let he = [0.5, 0.5, 0.5];
    // Looking at the box from outside along -Z: enters at z = 0.5, so t = 1.5.
    let t = focal_box_entry([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], he).expect("should hit");
    assert!((t - 1.5).abs() < 1e-4, "expected 1.5, got {t}");
    // Eye inside the box: entry distance is 0.
    assert_eq!(
        focal_box_entry([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], he),
        Some(0.0)
    );
    // Aimed away from the box entirely.
    assert_eq!(focal_box_entry([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], he), None);
    // Parallel to a slab and outside it.
    assert_eq!(focal_box_entry([0.0, 5.0, 2.0], [0.0, 0.0, -1.0], he), None);
}

#[test]
fn orbit_eye_sits_at_distance_from_the_pivot() {
    let cam = CameraState {
        nav: NavMode::Cad,
        dist: 3.0,
        target: [0.0, 0.0, 0.0],
        ..Default::default()
    };
    let (forward, _, _) = cam.basis();
    let eye = cam.eye(forward);
    assert!((dist2(eye, cam.target).sqrt() - 3.0).abs() < 1e-4);
}

#[test]
fn nav_switch_keeps_the_eye_put() {
    // The orbit and fly modes store the eye differently; switching must not move
    // it, or the view visibly jumps.
    let mut cam = CameraState {
        nav: NavMode::Cad,
        dist: 2.5,
        target: [0.3, -0.2, 0.1],
        ..Default::default()
    };
    let (forward, _, _) = cam.basis();
    let before = cam.eye(forward);

    cam.nav = NavMode::WasdFly;
    cam.sync_for_nav(/* was_fly */ false);
    let after = cam.eye(cam.basis().0);
    assert!(
        dist2(before, after) < 1e-8,
        "orbit -> fly moved the eye: {before:?} -> {after:?}"
    );

    cam.nav = NavMode::Cad;
    cam.sync_for_nav(/* was_fly */ true);
    let back = cam.eye(cam.basis().0);
    assert!(
        dist2(before, back) < 1e-6,
        "fly -> orbit moved the eye: {before:?} -> {back:?}"
    );
}

#[test]
fn orbiting_the_volume_center_preserves_the_pivot_radius() {
    // The volume-center orbit rotates the target about the origin, so its
    // distance from the origin is invariant — that's what keeps an off-center
    // framing from snapping back to center.
    let mut cam = CameraState {
        orbit_point: OrbitPoint::VolumeCenter,
        target: [0.4, 0.1, -0.2],
        ..Default::default()
    };
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
    assert!(
        v.box_he.iter().all(|c| c.is_finite() && *c > 0.0),
        "{:?}",
        v.box_he
    );
}

// ------------------------------------------------------------ a posed camera

/// A quarter turn about the volume's y axis, row-major.
const QUARTER_Y: [[f32; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];

#[test]
fn the_identity_pose_is_the_users_camera() {
    let cam = volume_camera(&CameraState::default(), [1.0; 3], (64, 64, 32));
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let p = cam.posed(&id, [0.1, 0.2, 0.3]);
    assert!(dist2(p.eye, cam.eye) < 1e-10);
    assert!(dist2(p.forward, cam.forward) < 1e-10);
    assert!(dist2(p.up, cam.up) < 1e-10);
    assert!(dist2(p.right, cam.right) < 1e-10);
    assert_eq!(p.box_he, cam.box_he);
}

/// Turning the volume, not the camera: what was in front of the camera before
/// the volume turned is what the camera sees of the turned volume at the
/// position the turn took it to. Checked through the picture's own projection
/// — the depth along `forward` and the offsets along `right` and `up` — which
/// is all the shader does with the basis.
#[test]
fn a_posed_camera_sees_the_turned_volume_where_the_turn_put_it() {
    let cam = volume_camera(&CameraState::default(), [1.0; 3], (64, 64, 64));
    let posed = cam.posed(&QUARTER_Y, [0.0; 3]);
    let project = |c: &VolumeCamera, p: [f32; 3]| {
        let v = [p[0] - c.eye[0], p[1] - c.eye[1], p[2] - c.eye[2]];
        let dot = |a: [f32; 3]| a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
        [dot(c.right), dot(c.up), dot(c.forward)]
    };
    for p in [[0.3, 0.1, -0.2], [0.5, 0.5, 0.5], [-0.4, 0.0, 0.25]] {
        // The point p of the volume, after the turn, is at R·p in the world.
        // Seen by the unposed camera there, it must project where the posed
        // camera projects p itself.
        let turned = [
            QUARTER_Y[0][0] * p[0] + QUARTER_Y[0][1] * p[1] + QUARTER_Y[0][2] * p[2],
            QUARTER_Y[1][0] * p[0] + QUARTER_Y[1][1] * p[1] + QUARTER_Y[1][2] * p[2],
            QUARTER_Y[2][0] * p[0] + QUARTER_Y[2][1] * p[1] + QUARTER_Y[2][2] * p[2],
        ];
        let a = project(&cam, turned);
        let b = project(&posed, p);
        assert!(dist2(a, b) < 1e-8, "{p:?}: {a:?} vs {b:?}");
    }
}

/// The turn is about the point at the centre of the view, so that point stays
/// in the middle of the frame — however far the user had panned the volume
/// away from it — and at the same depth.
#[test]
fn the_centre_of_the_view_stays_put_under_a_pose() {
    let state = CameraState {
        target: [0.2, -0.1, 0.05],
        ..Default::default()
    };
    let cam = volume_camera(&state, [1.0; 3], (64, 64, 64));
    let focus = cam.focus(state.dist);
    assert!(
        dist2(focus, state.target) < 1e-8,
        "the focus is the orbit target"
    );
    let posed = cam.posed(&QUARTER_Y, focus);
    // Where the posed camera sees the volume point that is now under the
    // centre of the frame: the pivot itself, since the turn leaves it fixed.
    let on_screen = |c: &VolumeCamera, p: [f32; 3]| {
        let v = [p[0] - c.eye[0], p[1] - c.eye[1], p[2] - c.eye[2]];
        let dot = |a: [f32; 3]| a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
        let depth = dot(c.forward);
        [dot(c.right) / depth, dot(c.up) / depth, depth]
    };
    let (a, b) = (on_screen(&cam, focus), on_screen(&posed, focus));
    assert!(
        a[0].abs() < 1e-5 && a[1].abs() < 1e-5,
        "the focus is centred: {a:?}"
    );
    assert!(dist2(a, b) < 1e-8, "{a:?} vs {b:?}");
    // And the volume's own centre, off to the side, swings about it rather
    // than staying where it was.
    let (c0, c1) = (on_screen(&cam, [0.0; 3]), on_screen(&posed, [0.0; 3]));
    assert!(dist2(c0, c1) > 1e-4);
}

#[test]
fn a_posed_basis_is_still_orthonormal() {
    let cam = volume_camera(&CameraState::default(), [1.0; 3], (10, 20, 30));
    // Slightly off a rotation, as a pose accepted at the API's tolerance can
    // be: the rays must still be unit length.
    let mut r = QUARTER_Y;
    r[0][2] = 1.0004;
    let p = cam.posed(&r, [0.0; 3]);
    for v in [p.forward, p.right, p.up] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "{len}");
    }
}

#[test]
fn detail_height_follows_what_is_in_view() {
    // A 512 x 256 plane: the box is 1.0 x 0.5 world units, a pixel 1/512.
    let cam = volume_camera(&CameraState::default(), [1.0; 3], (512, 256, 16));
    let at = |dist: f32| cam.detail_height(dist, (512, 256));
    // The picture spans 2 * dist * tan(fov/2) world units top to bottom.
    let span = |dist: f32| 2.0 * dist * cam.tan_half_fov;
    assert!((at(2.0) - span(2.0) * 512.0).abs() < 1e-2);
    // Zoomed out until the whole plane fits with room to spare, the picture
    // has more pixels across than the plane: the plane keeps every pixel.
    let whole = 1.0 / (2.0 * cam.tan_half_fov);
    assert!(at(whole * 1.5) > 512.0);
    // Zoomed in on a quarter of it: about a quarter of the pixels.
    assert!((at(whole / 4.0) - 128.0).abs() < 1.0);
    // Twice as close, half the pixels in view.
    assert!((at(1.0) * 2.0 - at(2.0)).abs() < 1e-2);
}

#[test]
fn detail_height_keeps_the_finer_axis() {
    // Pixels twice as wide as tall: the tall axis decides.
    let cam = volume_camera(&CameraState::default(), [2.0, 1.0, 1.0], (100, 100, 4));
    let fine = cam.detail_height(1.0, (100, 100));
    let pixel_y = 2.0 * cam.box_he[1] / 100.0;
    assert!((fine - 2.0 * cam.tan_half_fov / pixel_y).abs() < 1e-3);
}

/// Whether `p` is inside the box `[-he, he]`, give or take rounding.
fn inside(p: [f32; 3], he: [f32; 3]) -> bool {
    (0..3).all(|i| p[i].abs() <= he[i] + 1e-4)
}

#[test]
fn the_pivot_is_the_middle_of_the_view_axis_through_the_volume() {
    // Aimed at the centre of a 1 x 1 x 0.25 box: whatever the angle, the
    // middle of the axis's stretch through it is the centre.
    let state = CameraState::default();
    let cam = volume_camera(&state, [1.0; 3], (64, 64, 16));
    let p = cam.focus(cam.pivot_distance());
    assert!(dist2(p, [0.0; 3]) < 1e-8, "{p:?}");

    // The orbit target pushed far behind the volume, off to one side: the
    // pivot stays where the axis crosses the volume, not out at the target.
    let state = CameraState {
        yaw: 0.0,
        pitch: 0.0,
        target: [0.2, -0.1, -3.0],
        dist: 5.0,
        ..Default::default()
    };
    let cam = volume_camera(&state, [1.0; 3], (64, 64, 16));
    let p = cam.focus(cam.pivot_distance());
    assert!(inside(p, cam.box_he), "{p:?} is outside {:?}", cam.box_he);
    assert!(
        (p[0] - 0.2).abs() < 1e-4 && (p[1] + 0.1).abs() < 1e-4,
        "{p:?}"
    );
    assert!(p[2].abs() < 1e-4, "the middle of the slab: {p:?}");
}

#[test]
fn the_pivot_is_inside_the_volume_from_any_side_it_is_seen() {
    for (yaw, pitch) in [
        (0.0, 0.0),
        (37.0, 20.0),
        (-120.0, -60.0),
        (90.0, 0.0),
        (200.0, 80.0),
    ] {
        let state = CameraState {
            yaw,
            pitch,
            target: [0.1, 0.05, -0.02],
            ..Default::default()
        };
        let cam = volume_camera(&state, [1.0, 1.0, 2.0], (100, 80, 30));
        let p = cam.focus(cam.pivot_distance());
        assert!(inside(p, cam.box_he), "yaw {yaw} pitch {pitch}: {p:?}");
    }
}

#[test]
fn with_the_eye_inside_the_pivot_is_the_middle_of_what_lies_ahead() {
    let state = CameraState {
        dist: 0.0,
        target: [0.0; 3],
        ..Default::default()
    };
    let mut cam = volume_camera(&state, [1.0; 3], (64, 64, 64));
    cam.eye = [0.0, 0.0, 0.0];
    cam.forward = [0.0, 0.0, -1.0];
    // From the centre to the far face at z = -0.5.
    assert!((cam.pivot_distance() - 0.25).abs() < 1e-6);
}

#[test]
fn an_axis_that_misses_the_volume_pivots_nearest_its_centre() {
    let mut cam = volume_camera(&CameraState::default(), [1.0; 3], (64, 64, 64));
    cam.eye = [2.0, 0.0, 3.0];
    cam.forward = [0.0, 0.0, -1.0];
    // Passes beside the box; nearest the centre level with it, 3 ahead.
    assert!((cam.pivot_distance() - 3.0).abs() < 1e-6);
    // Looking away from it: no point behind the eye.
    cam.forward = [0.0, 0.0, 1.0];
    assert_eq!(cam.pivot_distance(), 0.0);
}

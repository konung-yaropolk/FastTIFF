use super::*;
use fasttiff_plugin_api::{
    AnimationOutput, ImageInfo, ParamValue, PixelType, Plane, StackInfo, ViewParams, VolumeMode,
    VolumeView,
};

/// A host with nothing behind it but the shape, the dialog's pending values and
/// whether this plugin's animation is running: all the plugin reads.
struct Fake {
    frames: usize,
    pending: Params,
    animating: bool,
    view: ViewParams,
    info: StackInfo,
}

impl Fake {
    fn new(frames: usize) -> Fake {
        Fake {
            frames,
            pending: Params::new(),
            animating: false,
            view: ViewParams {
                frame_index: 0,
                volume_view: false,
                channels: Vec::new(),
                luts: Vec::new(),
                volume: VolumeView {
                    mode: VolumeMode::Mip,
                    density: 1.0,
                    iso: 0.5,
                    eye: [0.0, 0.0, 3.0],
                    forward: [0.0, 0.0, -1.0],
                    up: [0.0, 1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                },
            },
            info: StackInfo::default(),
        }
    }
}

impl HostContext for Fake {
    fn image(&self) -> ImageInfo {
        ImageInfo {
            width: 8,
            height: 8,
            channels: 1,
            slices: 1,
            frames: self.frames,
            samples_per_pixel: 1,
            pixel_type: PixelType::U16,
        }
    }
    fn view(&self) -> &ViewParams {
        &self.view
    }
    fn stack_info(&self) -> &StackInfo {
        &self.info
    }
    fn read_plane_u16(&mut self, _p: Plane, _o: &mut Vec<u16>) -> Result<(), PluginError> {
        Err(PluginError::failed("the animation never reads pixels"))
    }
    fn read_plane_f32(&mut self, _p: Plane, _o: &mut Vec<f32>) -> Result<(), PluginError> {
        Err(PluginError::failed("the animation never reads pixels"))
    }
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn pending_params(&self) -> &Params {
        &self.pending
    }
    fn animating(&self) -> bool {
        self.animating
    }
    fn log(&mut self, _m: &str) {}
}

fn keys(d: &[ParamDecl]) -> Vec<&str> {
    d.iter().map(|d| d.key.as_str()).collect()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn a_cycle_is_as_many_frames_as_the_speed_takes_at_thirty_a_second() {
    assert_eq!(frames_per_cycle(30.0), 360);
    assert_eq!(frames_per_cycle(360.0), 30);
    assert_eq!(frames_per_cycle(1.0), 10_800);
    assert_eq!(frames_per_cycle(45.0), 240);
    // Out of range is clamped to what the dialog declares.
    assert_eq!(frames_per_cycle(1.0e6), 30);
    assert_eq!(frames_per_cycle(0.0), 10_800);
}

/// The seam rule, for every motion: one more step after the last frame lands
/// on the first, and no two frames are the same orientation.
#[test]
fn every_motion_loops_without_showing_any_orientation_twice() {
    for motion in Motion::ALL {
        let a = animation(motion, 45.0, 15.0);
        let n = a.poses.len();
        assert_eq!(n, 240, "{motion:?}");
        assert!(a.validate().is_ok(), "{motion:?}");
        assert_eq!(a.fps, FPS);
        // The pose one step past the end is the first pose again.
        let tilt = 15f32.to_radians();
        let past_the_end = motion.pose(std::f32::consts::TAU, tilt);
        let first = a.poses[0];
        for (r, row) in past_the_end.rotation.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!(
                    (v - first.rotation[r][c]).abs() < 1e-5,
                    "{motion:?}: the cycle does not close"
                );
            }
        }
        // And the last frame is not the first one: the seam is not shown.
        let probe = [0.3, -0.5, 0.8];
        let (p0, pl) = (a.poses[0].apply(probe), a.poses[n - 1].apply(probe));
        let d2: f32 = (0..3).map(|i| (p0[i] - pl[i]).powi(2)).sum();
        assert!(d2 > 1e-6, "{motion:?}: the last frame repeats the first");
    }
}

#[test]
fn the_turns_start_from_the_users_view() {
    for motion in [Motion::RotateX, Motion::RotateY] {
        assert_eq!(animation(motion, 30.0, 15.0).poses[0], Pose::IDENTITY);
    }
}

/// Rotation about X keeps X where it is and turns the rest; likewise Y.
#[test]
fn each_rotation_is_about_its_own_axis() {
    let x = animation(Motion::RotateX, 90.0, 15.0);
    let y = animation(Motion::RotateY, 90.0, 15.0);
    for p in x.poses.iter().chain(y.poses.iter()).take(1) {
        assert!(p.is_rotation(1e-5));
    }
    for p in &x.poses {
        let v = p.apply([1.0, 0.0, 0.0]);
        assert!(
            (v[0] - 1.0).abs() < 1e-5,
            "X moved under a rotation about X"
        );
    }
    for p in &y.poses {
        let v = p.apply([0.0, 1.0, 0.0]);
        assert!(
            (v[1] - 1.0).abs() < 1e-5,
            "Y moved under a rotation about Y"
        );
    }
    // A quarter of the way round about Y takes the depth axis to the width
    // axis: right-handed, as the API says.
    let quarter = y.poses[y.poses.len() / 4].apply([0.0, 0.0, 1.0]);
    assert!((quarter[0] - 1.0).abs() < 1e-4, "{quarter:?}");
}

/// Precession: the z axis keeps exactly the tilt from where it started, all the
/// way round, and goes all the way round.
#[test]
fn precession_swings_the_tilted_z_axis_round_a_cone() {
    let tilt = 15f64;
    let a = animation(Motion::Precession, 30.0, tilt);
    let mut directions = Vec::new();
    for p in &a.poses {
        let z = p.apply([0.0, 0.0, 1.0]);
        let angle = dot(z, [0.0, 0.0, 1.0]).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            (angle as f64 - tilt).abs() < 1e-3,
            "the lean drifted: {angle}"
        );
        directions.push(z[1].atan2(z[0]));
    }
    // Every quadrant is visited: it goes round, not back and forth.
    for q in 0..4 {
        let lo = -std::f32::consts::PI + q as f32 * std::f32::consts::FRAC_PI_2;
        let hi = lo + std::f32::consts::FRAC_PI_2;
        assert!(
            directions.iter().any(|d| (lo..hi).contains(d)),
            "quadrant {q} was never leaned into"
        );
    }
}

/// And it is a wobble, not a spin: the axis the volume leans about lies in the
/// image plane, so nothing turns about z. Checked as the twist of the image's
/// own x axis about the tilted z axis, which a spin would accumulate and a pure
/// lean keeps small.
#[test]
fn precession_does_not_spin_the_volume_about_its_own_axis() {
    let a = animation(Motion::Precession, 30.0, 15.0);
    for (i, p) in a.poses.iter().enumerate() {
        let phase = std::f32::consts::TAU * i as f32 / a.poses.len() as f32;
        // The axis of the lean is unmoved by it.
        let axis = [phase.cos(), phase.sin(), 0.0];
        let v = p.apply(axis);
        assert!(
            dot(v, axis) > 1.0 - 1e-5,
            "frame {i} turned its own lean axis"
        );
        // The image's x axis stays within the tilt of where it was.
        let x = p.apply([1.0, 0.0, 0.0]);
        assert!(dot(x, [1.0, 0.0, 0.0]) > 15f32.to_radians().cos() - 1e-5);
    }
}

#[test]
fn the_dialog_offers_play_then_stop_and_tilt_only_for_precession() {
    let host = Fake::new(10);
    let d = Animation3d.params(&host);
    assert_eq!(keys(&d), ["motion", "speed", "cycle", "play", "save"]);
    assert!(d.iter().any(|d| d.label.contains("360 frames")));

    let mut host = Fake::new(10);
    host.pending.set("motion", ParamValue::Choice(2));
    host.pending.set("speed", ParamValue::Float(90.0));
    let d = Animation3d.params(&host);
    assert_eq!(
        keys(&d),
        ["motion", "tilt", "speed", "cycle", "play", "save"]
    );
    assert!(d.iter().any(|d| d.label.contains("120 frames")));

    host.animating = true;
    let d = Animation3d.params(&host);
    assert!(keys(&d).contains(&"stop") && !keys(&d).contains(&"play"));
    let stop = d.iter().find(|d| d.key == "stop").unwrap();
    assert_eq!(stop.label, "Stop");
    assert!(matches!(stop.kind, ParamKind::Action));
}

#[test]
fn each_button_asks_for_what_it_says() {
    let mut host = Fake::new(10);
    let mut p = Params::new();
    p.set("motion", ParamValue::Choice(0));
    p.set("speed", ParamValue::Float(60.0));

    let mut play = p.clone();
    play.set("play", ParamValue::Bool(true));
    match Animation3d.run(&mut host, &play).expect("play") {
        Outcome::Animate(a) => {
            assert_eq!(a.output, AnimationOutput::Play);
            assert_eq!(a.poses.len(), 180);
        }
        other => panic!("{other:?}"),
    }

    let mut save = p.clone();
    save.set("save", ParamValue::Bool(true));
    match Animation3d.run(&mut host, &save).expect("save") {
        Outcome::Animate(a) => assert_eq!(
            a.output,
            AnimationOutput::SaveTiff {
                path: String::new()
            },
            "a save with no path asks the host to ask where"
        ),
        other => panic!("{other:?}"),
    }

    let mut stop = p.clone();
    stop.set("stop", ParamValue::Bool(true));
    assert_eq!(
        Animation3d.run(&mut host, &stop).expect("stop"),
        Outcome::StopAnimation
    );

    // No button at all plays.
    assert!(matches!(
        Animation3d.run(&mut host, &p).expect("run"),
        Outcome::Animate(_)
    ));
}

#[test]
fn a_single_plane_has_no_volume_to_turn() {
    let mut host = Fake::new(1);
    let mut p = Params::new();
    p.set("play", ParamValue::Bool(true));
    let e = Animation3d.run(&mut host, &p).expect_err("one plane");
    assert!(matches!(e, PluginError::Unsupported(_)), "{e}");
    // Stopping still works, whatever is open: it touches nothing.
    let mut stop = Params::new();
    stop.set("stop", ParamValue::Bool(true));
    assert_eq!(
        Animation3d.run(&mut host, &stop).expect("stop"),
        Outcome::StopAnimation
    );
}

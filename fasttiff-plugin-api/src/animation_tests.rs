use super::*;
use std::f32::consts::{FRAC_PI_2, PI};

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// The convention every plugin relies on, pinned: right-handed, so a quarter
/// turn about z takes x to y.
#[test]
fn a_quarter_turn_about_each_axis_is_right_handed() {
    let z = Pose::rotation([0.0, 0.0, 1.0], FRAC_PI_2);
    assert!(close(z.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    let x = Pose::rotation([1.0, 0.0, 0.0], FRAC_PI_2);
    assert!(close(x.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]));
    let y = Pose::rotation([0.0, 1.0, 0.0], FRAC_PI_2);
    assert!(close(y.apply([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]));
}

#[test]
fn the_axis_need_not_be_unit_length_and_a_null_one_does_nothing() {
    let a = Pose::rotation([0.0, 0.0, 5.0], 0.3);
    let b = Pose::rotation([0.0, 0.0, 1.0], 0.3);
    assert!(close(a.apply([1.0, 2.0, 3.0]), b.apply([1.0, 2.0, 3.0])));
    assert_eq!(Pose::rotation([0.0; 3], 1.0), Pose::IDENTITY);
    assert_eq!(Pose::rotation([1.0, 0.0, 0.0], f32::NAN), Pose::IDENTITY);
}

#[test]
fn then_applies_the_first_pose_first() {
    let x = Pose::rotation([1.0, 0.0, 0.0], FRAC_PI_2);
    let z = Pose::rotation([0.0, 0.0, 1.0], FRAC_PI_2);
    let p = [0.0, 1.0, 0.0];
    // About x, y goes to z; then about z, z stays.
    assert!(close(x.then(&z).apply(p), [0.0, 0.0, 1.0]));
    // About z, y goes to -x; then about x, -x stays.
    assert!(close(z.then(&x).apply(p), [-1.0, 0.0, 0.0]));
}

#[test]
fn rotations_are_recognised_and_stretches_and_mirrors_are_not() {
    assert!(Pose::IDENTITY.is_rotation(1e-6));
    assert!(Pose::rotation([1.0, 2.0, 3.0], 2.0).is_rotation(1e-5));
    let stretch = Pose::from_matrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(!stretch.is_rotation(1e-3));
    let mirror = Pose::from_matrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(!mirror.is_rotation(1e-3), "orthonormal, but inside out");
    let nan = Pose::from_matrix([[f32::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert!(!nan.is_rotation(1e-3));
}

/// The seam rule: a full turn in `n` frames never includes the turn's end,
/// which is its start again.
#[test]
fn a_cycle_samples_one_period_without_repeating_its_first_frame() {
    let mut phases = Vec::new();
    let a = Animation::cycle(8, 30.0, |phase| {
        phases.push(phase);
        Pose::rotation([0.0, 1.0, 0.0], phase)
    });
    assert_eq!(a.poses.len(), 8);
    assert_eq!(phases[0], 0.0);
    assert!((phases[4] - PI).abs() < 1e-6);
    assert!((phases[7] - 7.0 * std::f32::consts::TAU / 8.0).abs() < 1e-6);
    assert_eq!(a.poses[0], Pose::IDENTITY);
    // The step after the last frame lands on the first one: one step, not two
    // and not zero.
    let next = a.poses[7].then(&Pose::rotation(
        [0.0, 1.0, 0.0],
        std::f32::consts::TAU / 8.0,
    ));
    assert!(close(next.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]));
    assert_eq!(a.output, AnimationOutput::Play);
    assert!(a.validate().is_ok());
    assert!((a.period_s() - 8.0 / 30.0).abs() < 1e-9);
}

#[test]
fn a_cycle_of_no_frames_is_still_one_frame() {
    let a = Animation::cycle(0, 30.0, |_| Pose::IDENTITY);
    assert_eq!(a.poses.len(), 1);
}

#[test]
fn saving_is_a_destination_not_another_animation() {
    let a = Animation::new(vec![Pose::IDENTITY], 25.0).save_tiff("");
    assert_eq!(
        a.output,
        AnimationOutput::SaveTiff {
            path: String::new()
        }
    );
    let b = Animation::new(vec![Pose::IDENTITY], 25.0).save_tiff("out.tif");
    assert_eq!(
        b.output,
        AnimationOutput::SaveTiff {
            path: "out.tif".into()
        }
    );
}

#[test]
fn validation_refuses_what_a_host_cannot_carry_out() {
    assert!(Animation::new(Vec::new(), 30.0).validate().is_err());
    assert!(Animation::new(vec![Pose::IDENTITY], 0.0)
        .validate()
        .is_err());
    assert!(Animation::new(vec![Pose::IDENTITY], f32::NAN)
        .validate()
        .is_err());
    assert!(Animation::new(vec![Pose::IDENTITY], 1000.0)
        .validate()
        .is_err());
    let too_many = vec![Pose::IDENTITY; Animation::MAX_POSES + 1];
    assert!(Animation::new(too_many, 30.0).validate().is_err());
    let skew = Pose::from_matrix([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let e = Animation::new(vec![Pose::IDENTITY, skew], 30.0)
        .validate()
        .expect_err("a skew is not a pose");
    assert!(e.to_string().contains("pose 1"), "{e}");
}

//! One evaluation on the device, checked against the same evaluation on the
//! CPU.
//!
//! This is the level where a mistake in the kernel would be visible as a
//! number rather than as a registration that is merely a little off: a
//! mirrored index taken the wrong way, a weight paired with the wrong tap, a
//! derivative with its sign flipped. Each of those changes the sums by far more
//! than `f32` rounding does.
//!
//! Every test here skips rather than fails without a usable adapter. Whether a
//! machine has one is a property of the machine; a red test on a headless CI
//! box would say nothing about the code.

use super::*;
use crate::optimize::{accumulate, prepare, Sampler};
use crate::pyramid::Level;
use crate::settings::Transformation;
use crate::transform::{initial_landmarks, Point};

/// A level with structure at several scales and no symmetry.
fn level(w: usize, h: usize) -> Level {
    Level {
        w,
        h,
        img: (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                300.0
                    + ((x * 0.13).sin() * (y * 0.11).cos()) * 90.0
                    + (x * 0.37 + y * 0.29).sin() * 25.0
            })
            .collect(),
        octave: 0,
    }
}

/// A candidate a little away from the identity: every model's landmarks,
/// nudged by an amount that differs per landmark so no derivative is zero by
/// symmetry.
fn nudged(model: Transformation, w: usize, h: usize) -> (Vec<Point>, Vec<Point>) {
    let fixed = initial_landmarks(model, w, h);
    let moved = fixed
        .iter()
        .enumerate()
        .map(|(i, p)| Point::new(p.x + 0.7 + 0.3 * i as f64, p.y - 0.4 + 0.2 * i as f64))
        .collect();
    (fixed, moved)
}

/// The largest difference relative to the largest magnitude.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().chain(b).fold(1e-12f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// The shader compiles for every model's parameter count. A naga or driver
/// rejection shows up here, by model, rather than as a backend that quietly
/// reports itself unavailable.
#[test]
fn the_shader_builds_for_every_model() {
    let Some((device, _)) = shared_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    for model in Transformation::all() {
        assert!(
            pipeline(&device, model.dof()).is_some(),
            "{model:?}: the shader did not build"
        );
    }
}

/// One candidate, every model: the device's sums match the CPU's to within
/// `f32` arithmetic.
#[test]
fn one_evaluation_matches_the_cpu() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let (w, h) = (97, 83);
    let lv = level(w, h);
    let reference = level(w, h)
        .img
        .iter()
        .map(|v| v * 0.98 + 3.0)
        .collect::<Vec<_>>();
    let sampler = Sampler::of(&lv, false);
    for model in Transformation::all() {
        let (fixed, moved) = nudged(model, w, h);
        for derivatives in [true, false] {
            let prep = prepare(model, &fixed, &moved, derivatives).expect("candidate");
            let cpu = accumulate(&prep, &reference, w, h, &sampler, false);
            let gpu = GpuLevel::new(
                model.dof(),
                w,
                h,
                &[[
                    &reference[..],
                    &sampler.img[..],
                    &sampler.gx[..],
                    &sampler.gy[..],
                ]],
            )
            .expect("level")
            .evaluate(&[(0, &prep)])
            .expect("round")
            .remove(0);
            assert_eq!(gpu.counted, cpu.counted);
            let ss = (gpu.sum_sq - cpu.sum_sq).abs() / cpu.sum_sq.max(1e-12);
            assert!(ss < 1e-4, "{model:?}: criterion off by {ss:e} relative");
            if derivatives {
                let g = rel(&gpu.gradient, &cpu.gradient);
                let hh = rel(&gpu.hessian, &cpu.hessian);
                assert!(g < 1e-3, "{model:?}: gradient off by {g:e} relative");
                assert!(hh < 1e-3, "{model:?}: curvature off by {hh:e} relative");
            } else {
                assert!(gpu.gradient.iter().all(|v| *v == 0.0));
            }
        }
    }
}

/// A candidate that reaches well outside the frame reads the mirrored
/// extension the same way on both — the boundary is where an index error
/// would hide.
#[test]
fn the_mirror_boundary_matches_the_cpu() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let (w, h) = (64, 48);
    let lv = level(w, h);
    let sampler = Sampler::of(&lv, false);
    let model = Transformation::Affine;
    let fixed = initial_landmarks(model, w, h);
    // A shift of more than a frame, and a shear: every tap lands off the grid
    // somewhere, many of them past an edge.
    let moved: Vec<Point> = fixed
        .iter()
        .map(|p| Point::new(p.x * 1.3 - 70.0, p.y + 0.2 * p.x + 55.0))
        .collect();
    let prep = prepare(model, &fixed, &moved, true).expect("candidate");
    let cpu = accumulate(&prep, &lv.img, w, h, &sampler, false);
    let gpu = GpuLevel::new(
        model.dof(),
        w,
        h,
        &[[
            &lv.img[..],
            &sampler.img[..],
            &sampler.gx[..],
            &sampler.gy[..],
        ]],
    )
    .expect("level")
    .evaluate(&[(0, &prep)])
    .expect("round")
    .remove(0);
    let ss = (gpu.sum_sq - cpu.sum_sq).abs() / cpu.sum_sq.max(1e-12);
    assert!(ss < 1e-4, "criterion off by {ss:e} past the edge");
    assert!(rel(&gpu.gradient, &cpu.gradient) < 1e-3);
}

/// Lanes do not affect each other: three frames scored in one dispatch give
/// exactly what each gives scored alone. This is what makes batching a
/// scheduling choice rather than a numerical one.
#[test]
fn lanes_in_one_dispatch_are_independent() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let (w, h) = (80, 60);
    let model = Transformation::RigidBody;
    let levels: Vec<Level> = (0..3)
        .map(|k| {
            let mut lv = level(w, h);
            for v in lv.img.iter_mut() {
                *v = *v * (1.0 + 0.05 * k as f32) + 10.0 * k as f32;
            }
            lv
        })
        .collect();
    let samplers: Vec<Sampler> = levels.iter().map(|lv| Sampler::of(lv, false)).collect();
    // Each lane its own reference as well, as the previous-frame mode has it.
    let references: Vec<Vec<f32>> = (0..3)
        .map(|k| level(w, h).img.iter().map(|v| v + 4.0 * k as f32).collect())
        .collect();
    let planes: Vec<[&[f32]; 4]> = samplers
        .iter()
        .zip(&references)
        .map(|(s, r)| [&r[..], &s.img[..], &s.gx[..], &s.gy[..]])
        .collect();
    let gpu = GpuLevel::new(model.dof(), w, h, &planes).expect("level");
    let (fixed, moved) = nudged(model, w, h);
    let preps: Vec<_> = (0..3)
        .map(|k| {
            let m: Vec<Point> = moved
                .iter()
                .map(|p| Point::new(p.x + k as f64 * 0.5, p.y))
                .collect();
            prepare(model, &fixed, &m, true).expect("candidate")
        })
        .collect();
    let together = gpu
        .evaluate(&[(0, &preps[0]), (1, &preps[1]), (2, &preps[2])])
        .expect("round");
    for k in 0..3 {
        let alone = gpu.evaluate(&[(k, &preps[k])]).expect("round").remove(0);
        assert_eq!(together[k], alone, "lane {k} depends on its neighbours");
        // And each lane is scored against its own reference, not slot 0's: the
        // same candidate on the CPU, with that reference, agrees.
        let cpu = accumulate(&preps[k], &references[k], w, h, &samplers[k], false);
        let ss = (alone.sum_sq - cpu.sum_sq).abs() / cpu.sum_sq.max(1e-12);
        assert!(
            ss < 1e-4,
            "lane {k} read the wrong reference: {ss:e} relative"
        );
    }
}

/// The same input gives the same answer, run after run.
#[test]
fn repeated_rounds_are_identical() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let (w, h) = (90, 70);
    let lv = level(w, h);
    let sampler = Sampler::of(&lv, false);
    let model = Transformation::ScaledRotation;
    let (fixed, moved) = nudged(model, w, h);
    let prep = prepare(model, &fixed, &moved, true).expect("candidate");
    let gpu = GpuLevel::new(
        model.dof(),
        w,
        h,
        &[[
            &lv.img[..],
            &sampler.img[..],
            &sampler.gx[..],
            &sampler.gy[..],
        ]],
    )
    .expect("level");
    let first = gpu.evaluate(&[(0, &prep)]).expect("round");
    for _ in 0..5 {
        assert_eq!(gpu.evaluate(&[(0, &prep)]).expect("round"), first);
    }
}

/// More lanes than the level was built for, or a slot it has no planes for, is
/// refused rather than read or written past the end of a buffer.
#[test]
fn an_oversized_round_is_refused() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let (w, h) = (32, 32);
    let lv = level(w, h);
    let sampler = Sampler::of(&lv, false);
    let model = Transformation::Translation;
    let (fixed, moved) = nudged(model, w, h);
    let prep = prepare(model, &fixed, &moved, true).expect("candidate");
    let gpu = GpuLevel::new(
        model.dof(),
        w,
        h,
        &[[
            &lv.img[..],
            &sampler.img[..],
            &sampler.gx[..],
            &sampler.gy[..],
        ]],
    )
    .expect("level");
    assert!(gpu.evaluate(&[(0, &prep), (0, &prep)]).is_none());
    assert!(gpu.evaluate(&[(1, &prep)]).is_none());
    assert_eq!(gpu.evaluate(&[]).expect("empty"), Vec::new());
}

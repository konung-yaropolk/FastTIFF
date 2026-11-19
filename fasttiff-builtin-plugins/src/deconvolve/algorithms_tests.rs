//! The nine algorithms, against a problem whose answer is known.
//!
//! The fixture is a blur done by the same operator the deconvolution will
//! use, on a periodic grid, with no noise. That makes it the ideal case — the
//! forward model is exactly right — which is deliberate: a method that cannot
//! improve on the ideal case is broken, and no amount of arguing about
//! regularisation parameters can rescue it. Noise and model error make every
//! method worse by amounts that are properly the subject of a paper, not of a
//! unit test.
//!
//! The assertion is therefore always the same shape: the deconvolved image is
//! closer to the truth than the blurred image was. That is the one claim
//! every one of these methods makes.

use super::super::fft::Transform;
use super::super::grid::{normalise, place_psf, Operator};
use super::*;

/// Relative error against the truth, as a fraction.
fn error(got: &[f32], truth: &[f32]) -> f64 {
    let num: f64 = got
        .iter()
        .zip(truth)
        .map(|(a, b)| ((*a - *b) as f64).powi(2))
        .sum();
    let den: f64 = truth.iter().map(|&v| (v as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

/// Bright blobs and a hard edge on a dark field: non-negative, so the
/// multiplicative methods are in their domain, and sharp, so a blur is
/// something to undo.
fn phantom(d: Dims) -> Vec<f32> {
    let mut v = vec![0.05f32; d.len()];
    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                let i = d.at(x, y, z);
                if x >= d.x / 2 {
                    v[i] = 0.4;
                }
                let near = |cx: usize, cy: usize, r: f32| {
                    let (dx, dy) = (x as f32 - cx as f32, y as f32 - cy as f32);
                    (dx * dx + dy * dy).sqrt() < r
                };
                if near(d.x / 4, d.y / 4, 2.5) || near(3 * d.x / 4, 3 * d.y / 4, 1.5) {
                    v[i] = 1.0;
                }
            }
        }
    }
    v
}

/// A small Gaussian kernel, normalised, deliberately not isotropic so that a
/// transposed axis somewhere would show.
fn kernel(d: Dims, sx: f32, sy: f32, sz: f32) -> Vec<f32> {
    let g = |delta: f32, s: f32| {
        if s <= 0.0 {
            if delta == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            (-0.5 * (delta / s).powi(2)).exp()
        }
    };
    let mut k = vec![0.0f32; d.len()];
    let (cx, cy, cz) = (d.x / 2, d.y / 2, d.z / 2);
    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                k[d.at(x, y, z)] = g(x as f32 - cx as f32, sx)
                    * g(y as f32 - cy as f32, sy)
                    * g(z as f32 - cz as f32, sz);
            }
        }
    }
    normalise(&mut k);
    k
}

/// The truth, the operator that blurred it, and the blurred data.
struct Problem {
    op: Operator,
    truth: Vec<f32>,
    blurred: Vec<f32>,
    baseline: f64,
}

fn problem_2d() -> Problem {
    build(Dims::new(32, 32, 1), Dims::new(7, 7, 1), (1.3, 1.0, 0.0))
}

fn problem_3d() -> Problem {
    build(Dims::new(16, 16, 8), Dims::new(5, 5, 5), (1.1, 1.1, 1.4))
}

fn build(grid: Dims, pd: Dims, sigma: (f32, f32, f32)) -> Problem {
    let truth = phantom(grid);
    let k = kernel(pd, sigma.0, sigma.1, sigma.2);
    let placed = place_psf(grid, &k, pd, (pd.x / 2, pd.y / 2, pd.z / 2));
    let mut op = Operator::new(Transform::new(grid), &placed);
    let mut blurred = vec![0.0f32; grid.len()];
    op.blur(&truth, &mut blurred);
    let baseline = error(&blurred, &truth);
    assert!(
        baseline > 0.05,
        "the fixture must actually be blurred: {baseline}"
    );
    Problem {
        op,
        truth,
        blurred,
        baseline,
    }
}

fn never_cancel(_f: f32) -> bool {
    true
}

/// Settings that give each method a fair run on a noiseless problem.
fn settings(method: Method) -> Settings {
    Settings {
        method,
        iterations: 60,
        // Noiseless data wants little regularisation; these are the values a
        // user would reach for on clean data, not ones tuned until the test
        // passes.
        lambda: match method {
            Method::RichardsonLucyTv => 0.002,
            _ => 0.0001,
        },
        gamma: 0.0001,
        step: 1.0,
        threshold: 0.001,
        low_pass: 0.0,
        // Off: these tests are about where a method gets to, not about when
        // it decides to stop.
        stop_delta: 0.0,
        nonneg: true,
    }
}

// ------------------------------------------------------- every method helps

#[test]
fn every_method_gets_closer_to_the_truth_in_2d() {
    for method in Method::ALL {
        let mut p = problem_2d();
        let out = run(&mut p.op, &p.blurred, &settings(method), &mut never_cancel)
            .expect("not cancelled");
        let after = error(&out.image, &p.truth);
        assert!(
            after < p.baseline,
            "{}: error {after:.4} is no better than the blurred input's {:.4}",
            method.label(),
            p.baseline
        );
    }
}

#[test]
fn every_method_gets_closer_to_the_truth_in_3d() {
    for method in Method::ALL {
        let mut p = problem_3d();
        let out = run(&mut p.op, &p.blurred, &settings(method), &mut never_cancel)
            .expect("not cancelled");
        let after = error(&out.image, &p.truth);
        assert!(
            after < p.baseline,
            "{}: error {after:.4} is no better than the blurred input's {:.4}",
            method.label(),
            p.baseline
        );
    }
}

/// Richardson-Lucy is the default, so it is held to more than "better than
/// nothing": it has to keep converging.
///
/// Stated as a trend rather than a threshold, because a threshold here would
/// be a number copied from whatever this fixture happened to produce. RL on a
/// hard edge converges slowly and without ever quite arriving — 0.17 of the
/// original error at 50 iterations, 0.15 at 200, 0.13 at 600 — and that
/// *shape* is the claim. A run that had stalled, started oscillating, or
/// settled on the wrong answer would fail this and would pass any single
/// bound loose enough to be safe.
#[test]
fn richardson_lucy_keeps_converging() {
    let mut errors = Vec::new();
    for n in [50usize, 200, 600] {
        let mut p = problem_2d();
        let mut s = settings(Method::RichardsonLucy);
        s.iterations = n;
        let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
        errors.push((n, error(&out.image, &p.truth), p.baseline));
    }
    for w in errors.windows(2) {
        assert!(
            w[1].1 < w[0].1,
            "{} iterations ({:.4}) did not beat {} ({:.4})",
            w[1].0,
            w[1].1,
            w[0].0,
            w[0].1
        );
    }
    let (n, err, baseline) = errors[errors.len() - 1];
    assert!(
        err < baseline * 0.6,
        "{n} iterations left {err:.4} of a starting {baseline:.4}"
    );
}

// --------------------------------------------------------- the invariants

/// With a delta for a PSF there is nothing to undo, so every method must hand
/// back what it was given. A method that "sharpens" here is sharpening noise.
#[test]
fn a_delta_psf_leaves_the_image_alone() {
    let grid = Dims::new(16, 16, 1);
    for method in Method::ALL {
        let placed = place_psf(grid, &[1.0], Dims::new(1, 1, 1), (0, 0, 0));
        let mut op = Operator::new(Transform::new(grid), &placed);
        let data = phantom(grid);
        let out = run(&mut op, &data, &settings(method), &mut never_cancel).expect("not cancelled");
        let e = error(&out.image, &data);
        assert!(
            e < 0.02,
            "{} changed an unblurred image by {e:.4}",
            method.label()
        );
    }
}

/// Richardson-Lucy conserves total intensity by construction — it is a
/// likelihood maximisation over a fixed photon budget — and that is the
/// property that makes a deconvolved stack still quantitative.
#[test]
fn richardson_lucy_conserves_the_total() {
    let mut p = problem_2d();
    let out = run(
        &mut p.op,
        &p.blurred,
        &settings(Method::RichardsonLucy),
        &mut never_cancel,
    )
    .expect("not cancelled");
    let before: f64 = p.blurred.iter().map(|&v| v as f64).sum();
    let after: f64 = out.image.iter().map(|&v| v as f64).sum();
    assert!(
        (after - before).abs() < 0.01 * before,
        "{before} became {after}"
    );
}

#[test]
fn the_non_negativity_constraint_is_honoured() {
    for method in Method::ALL {
        let mut p = problem_2d();
        let mut s = settings(method);
        s.nonneg = true;
        let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
        assert!(
            out.image.iter().all(|&v| v >= 0.0),
            "{} produced a negative intensity",
            method.label()
        );
    }
}

/// The inverse filter without regularisation is the one that overshoots, so
/// turning the constraint off has to actually let it: a test where every
/// method is non-negative anyway would not be testing the flag.
#[test]
fn turning_the_constraint_off_lets_a_method_overshoot() {
    let mut p = problem_2d();
    let mut s = settings(Method::Wiener);
    s.nonneg = false;
    s.gamma = 1e-6;
    let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
    assert!(
        out.image.iter().any(|&v| v < 0.0),
        "an unconstrained inverse filter on a hard edge should ring below zero"
    );
}

/// A deconvolution must not move the picture. One voxel of shift is the
/// signature of a misplaced PSF origin, and it survives every other check
/// here because a shifted image is still a sharp image.
#[test]
fn the_result_is_not_shifted() {
    let grid = Dims::new(32, 32, 1);
    let pd = Dims::new(9, 9, 1);
    let k = kernel(pd, 1.5, 1.5, 0.0);
    let placed = place_psf(grid, &k, pd, (pd.x / 2, pd.y / 2, 0));
    let mut op = Operator::new(Transform::new(grid), &placed);

    let mut truth = vec![0.0f32; grid.len()];
    let spot = (11usize, 19usize);
    truth[grid.at(spot.0, spot.1, 0)] = 1.0;
    let mut blurred = vec![0.0f32; grid.len()];
    op.blur(&truth, &mut blurred);

    let out = run(
        &mut op,
        &blurred,
        &settings(Method::RichardsonLucy),
        &mut never_cancel,
    )
    .expect("not cancelled");

    let peak = out
        .image
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaNs"))
        .expect("non-empty")
        .0;
    assert_eq!(
        (peak % grid.x, peak / grid.x),
        spot,
        "the restored point moved"
    );
}

// --------------------------------------------------------------- stopping

#[test]
fn cancelling_returns_nothing_rather_than_half_an_answer() {
    let mut p = problem_2d();
    let mut calls = 0;
    let out = run(
        &mut p.op,
        &p.blurred,
        &settings(Method::RichardsonLucy),
        &mut |_| {
            calls += 1;
            calls < 3
        },
    );
    assert!(out.is_none(), "a cancelled run must not produce an image");
}

#[test]
fn an_estimate_that_stops_moving_ends_the_run() {
    let mut p = problem_2d();
    let mut s = settings(Method::RichardsonLucy);
    s.iterations = 500;
    // Any movement at all counts as convergence, so this stops immediately
    // and the test does not depend on how fast RL actually converges.
    s.stop_delta = 100.0;
    let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
    match out.stopped {
        Stopped::Converged { iterations, .. } => assert!(iterations < 10, "{iterations}"),
        other => panic!("expected convergence, got {other:?}"),
    }
}

#[test]
fn a_run_that_blows_up_says_so_instead_of_returning_noise() {
    let mut p = problem_2d();
    let mut s = settings(Method::VanCittert);
    s.step = 2000.0;
    s.iterations = 60;
    s.nonneg = false;
    let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
    assert!(
        matches!(out.stopped, Stopped::Diverged(_)),
        "a step of 2000 must be detected, not reported as a result: {:?}",
        out.stopped
    );
}

#[test]
fn a_full_run_reports_the_iterations_it_did() {
    let mut p = problem_2d();
    let mut s = settings(Method::Landweber);
    s.iterations = 7;
    let out = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("not cancelled");
    assert_eq!(out.stopped, Stopped::Completed(7));
}

// ---------------------------------------------------------------- options

/// The low-pass is there to stop an iterative method turning noise into
/// texture, so it has to actually smooth the result.
#[test]
fn the_low_pass_smooths_the_result() {
    let roughness = |v: &[f32], d: Dims| -> f64 {
        let mut s = 0.0;
        for y in 0..d.y {
            for x in 1..d.x {
                s += ((v[d.at(x, y, 0)] - v[d.at(x - 1, y, 0)]) as f64).abs();
            }
        }
        s
    };
    let grid = Dims::new(32, 32, 1);

    let mut sharp = problem_2d();
    let a = run(
        &mut sharp.op,
        &sharp.blurred,
        &settings(Method::RichardsonLucy),
        &mut never_cancel,
    )
    .expect("not cancelled");

    let mut soft = problem_2d();
    let mut s = settings(Method::RichardsonLucy);
    s.low_pass = 1.5;
    let b = run(&mut soft.op, &soft.blurred, &s, &mut never_cancel).expect("not cancelled");

    assert!(
        roughness(&b.image, grid) < roughness(&a.image, grid),
        "the low-pass left the result no smoother"
    );
}

/// Total variation is the thing RLTV adds, so it must produce a flatter
/// answer than plain Richardson-Lucy from the same start.
#[test]
fn total_variation_flattens_more_than_plain_richardson_lucy() {
    let variation = |v: &[f32]| -> f64 {
        let d = Dims::new(32, 32, 1);
        let mut s = 0.0;
        for y in 0..d.y {
            for x in 1..d.x {
                s += ((v[d.at(x, y, 0)] - v[d.at(x - 1, y, 0)]) as f64).abs();
            }
        }
        s
    };
    let mut plain = problem_2d();
    let a = run(
        &mut plain.op,
        &plain.blurred,
        &settings(Method::RichardsonLucy),
        &mut never_cancel,
    )
    .expect("not cancelled");

    let mut tv = problem_2d();
    let mut s = settings(Method::RichardsonLucyTv);
    // The weight the literature uses, and the range the dialog's help names.
    // Much above this the term over-regularises into staircase artefacts,
    // which have a total variation of their own — so "more weight is always
    // flatter" is false, and is not what is claimed here.
    s.lambda = 0.01;
    let b = run(&mut tv.op, &tv.blurred, &s, &mut never_cancel).expect("not cancelled");

    assert!(
        variation(&b.image) < variation(&a.image),
        "the TV term did nothing: {} vs {}",
        variation(&b.image),
        variation(&a.image)
    );
    // And at that weight it is not merely flatter but closer to the truth,
    // which is the only reason to want it.
    assert!(
        error(&b.image, &tv.truth) < error(&a.image, &plain.truth),
        "total variation made the answer worse"
    );
}

/// Wiener's one parameter has to do something monotone: more damping, smoother
/// answer. A gamma that was ignored would make these identical.
#[test]
fn a_larger_wiener_gamma_damps_more() {
    let energy = |v: &[f32]| -> f64 { v.iter().map(|&x| (x as f64).powi(2)).sum() };
    let mut p = problem_2d();
    let mut s = settings(Method::Wiener);
    s.gamma = 1e-6;
    let sharp = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("ok");
    s.gamma = 0.5;
    let damped = run(&mut p.op, &p.blurred, &s, &mut never_cancel).expect("ok");
    assert!(
        energy(&damped.image) < energy(&sharp.image),
        "gamma made no difference"
    );
}

/// The spectral filters do not loop, so the iteration count must not reach
/// them. Stated as a test because the dialog shows the control for every
/// method and a user will set it.
#[test]
fn the_one_shot_filters_ignore_the_iteration_count() {
    for method in [
        Method::Wiener,
        Method::RegularisedInverse,
        Method::NaiveInverse,
    ] {
        assert!(!method.is_iterative(), "{}", method.label());
        let mut p = problem_2d();
        let mut one = settings(method);
        one.iterations = 1;
        let a = run(&mut p.op, &p.blurred, &one, &mut never_cancel).expect("ok");
        let mut many = settings(method);
        many.iterations = 500;
        let b = run(&mut p.op, &p.blurred, &many, &mut never_cancel).expect("ok");
        assert_eq!(a.image, b.image, "{} looked at the count", method.label());
    }
}

#[test]
fn every_method_has_its_own_name_and_tag() {
    let mut labels: Vec<&str> = Method::ALL.iter().map(|m| m.label()).collect();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(labels.len(), Method::ALL.len());

    let mut tags: Vec<&str> = Method::ALL.iter().map(|m| m.tag()).collect();
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), Method::ALL.len(), "tags name the result window");
}

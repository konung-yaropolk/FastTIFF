//! The search, checked by recovery.
//!
//! Every test here is the same experiment: take an image, move it by a
//! transformation we chose, and ask the optimiser to find it. That is the only
//! assertion about a registration worth making — a fit that "converged" and
//! landed in the wrong local minimum reports a small criterion and a wrong
//! answer, and nothing short of a known ground truth distinguishes the two.

use super::*;
use crate::settings::{Backend, Reference};

/// A test image with structure at several scales and no symmetry.
///
/// Symmetry is the thing to avoid: a pattern that repeats every few pixels has
/// a local minimum per period, and a test built on one would measure the
/// pyramid's luck rather than the optimiser's correctness.
fn textured(w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let blobs = (x * 0.21).sin() * (y * 0.17).cos() * 60.0;
            let ramp = x * 0.4 + y * 0.25;
            let detail = ((x * 0.9).sin() + (y * 1.1).cos()) * 12.0;
            // A bright asymmetric feature, so there is one unambiguous answer.
            let spot = if (x - w as f32 * 0.37).abs() < 4.0 && (y - h as f32 * 0.61).abs() < 4.0 {
                120.0
            } else {
                0.0
            };
            128.0 + blobs + ramp + detail + spot
        })
        .collect()
}

/// Build a moving image by displacing the reference, and return the
/// transformation the registration is expected to find.
///
/// The direction is the thing to keep straight, and it is worth spelling out
/// because it reads backwards at first. `warp(img, g)` produces `out[p] =
/// img[g(p)]`, so warping the reference by `g` makes an image whose content has
/// moved by *minus* `g`. The matrix that puts it back — the one `register_pair`
/// returns, mapping reference coordinates to moving ones — is therefore `g`
/// inverted.
///
/// So a test asks for a displacement, gets the moving image, and compares the
/// fit against the inverse. Writing the comparison against `g` itself is the
/// sign error this helper exists to make impossible.
fn displaced(reference: &[f32], w: usize, h: usize, g: &Matrix) -> (Vec<f32>, Matrix) {
    let moving = transform::warp(reference, w, h, g);
    // Inverse of a 3x3 affine, which is all these tests use.
    let m = g.to_3x3();
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    assert!(det.abs() > 1e-12, "the displacement must be invertible");
    let (a, b, c, d) = (m[1][1] / det, -m[0][1] / det, -m[1][0] / det, m[0][0] / det);
    let (tx, ty) = (m[0][2], m[1][2]);
    let inverse = Matrix::from_3x3([
        [a, b, -(a * tx + b * ty)],
        [c, d, -(c * tx + d * ty)],
        [0.0, 0.0, 1.0],
    ]);
    (moving, inverse)
}

fn settings(model: Transformation) -> Settings {
    Settings {
        transformation: model,
        backend: Backend::SingleThread,
        ..Settings::default()
    }
}

/// Registering an image against itself finds the identity. If this fails
/// nothing else is worth reading.
#[test]
fn an_image_registers_to_itself_as_the_identity() {
    let (w, h) = (96, 80);
    let img = textured(w, h);
    for model in [
        Transformation::Translation,
        Transformation::RigidBody,
        Transformation::ScaledRotation,
        Transformation::Affine,
    ] {
        let fit = register_pair(&img, &img, w, h, &settings(model))
            .unwrap_or_else(|| panic!("{model:?} should fit"));
        let p = fit.matrix.apply(Point::new(w as f64 / 2.0, h as f64 / 2.0));
        assert!(
            (p.x - w as f64 / 2.0).abs() < 0.05 && (p.y - h as f64 / 2.0).abs() < 0.05,
            "{model:?}: centre moved to {p:?} registering an image to itself"
        );
    }
}

/// A known shift comes back, to well under a pixel.
#[test]
fn a_known_translation_is_recovered() {
    let (w, h) = (96, 80);
    let reference = textured(w, h);
    for (dx, dy) in [(3.0, 0.0), (0.0, -2.0), (4.0, 5.0), (-6.0, 3.0)] {
        // The moving image is the reference shifted, built by warping with the
        // matrix that maps reference -> moving.
        let (moving, want) = displaced(
            &reference,
            w,
            h,
            &Matrix {
                c: [dx, 1.0, 0.0, 0.0, dy, 0.0, 1.0, 0.0],
            },
        );
        let fit = register_pair(
            &reference,
            &moving,
            w,
            h,
            &settings(Transformation::Translation),
        )
        .expect("fit");
        let got = fit.matrix;
        assert!(
            (got.c[0] - want.c[0]).abs() < 0.15 && (got.c[4] - want.c[4]).abs() < 0.15,
            "displacement ({dx}, {dy}) should be undone by ({}, {}), got ({}, {})",
            want.c[0],
            want.c[4],
            got.c[0],
            got.c[4]
        );
    }
}

/// Subpixel: a fractional shift is recovered as a fractional shift, which is
/// the whole reason for the spline.
#[test]
fn a_subpixel_translation_is_recovered() {
    let (w, h) = (96, 80);
    let reference = textured(w, h);
    let (dx, dy) = (1.4, -0.65);
    let (moving, want) = displaced(
        &reference,
        w,
        h,
        &Matrix {
            c: [dx, 1.0, 0.0, 0.0, dy, 0.0, 1.0, 0.0],
        },
    );
    let fit = register_pair(
        &reference,
        &moving,
        w,
        h,
        &settings(Transformation::Translation),
    )
    .expect("fit");
    assert!(
        (fit.matrix.c[0] - want.c[0]).abs() < 0.1 && (fit.matrix.c[4] - want.c[4]).abs() < 0.1,
        "subpixel displacement ({dx}, {dy}) should be undone by ({}, {}), got ({}, {})",
        want.c[0],
        want.c[4],
        fit.matrix.c[0],
        fit.matrix.c[4]
    );
}

/// A rotation is recovered by the rigid-body model, and the answer is still a
/// rotation.
#[test]
fn a_known_rotation_is_recovered() {
    let (w, h) = (96, 96);
    let reference = textured(w, h);
    let angle: f64 = 0.06; // ~3.4 degrees
                           // Rotate about the centre.
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (s, c) = (angle.sin(), angle.cos());
    let m = Matrix {
        c: [
            cx - c * cx + s * cy,
            c,
            -s,
            0.0,
            cy - s * cx - c * cy,
            s,
            c,
            0.0,
        ],
    };
    let (moving, _) = displaced(&reference, w, h, &m);
    let fit = register_pair(
        &reference,
        &moving,
        w,
        h,
        &settings(Transformation::RigidBody),
    )
    .expect("fit");
    let got = fit.matrix;
    let got_angle = got.c[5].atan2(got.c[1]);
    assert!(
        (got_angle + angle).abs() < 0.02,
        "a displacement of {angle} rad should be undone by {} rad, got {got_angle}",
        -angle
    );
    // Still a rotation: the model's constraint survived the search.
    assert!(
        (got.c[1] * got.c[1] + got.c[5] * got.c[5] - 1.0).abs() < 1e-6,
        "the fit should still be a rotation: {:?}",
        got.c
    );
}

/// The fit is better than where it started — the criterion a registration
/// reports is the one belonging to the answer it returns.
#[test]
fn registering_improves_on_doing_nothing() {
    let (w, h) = (96, 80);
    let reference = textured(w, h);
    let m = Matrix {
        c: [4.0, 1.0, 0.0, 0.0, -3.0, 0.0, 1.0, 0.0],
    };
    let moving = transform::warp(&reference, w, h, &m);
    let fit = register_pair(
        &reference,
        &moving,
        w,
        h,
        &settings(Transformation::Translation),
    )
    .expect("fit");
    // The criterion of leaving it alone, for comparison.
    let unmoved: f64 = reference
        .iter()
        .zip(moving.iter())
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / (w * h) as f64;
    assert!(
        fit.criterion < unmoved / 4.0,
        "registering should beat doing nothing by a lot: {} vs {unmoved}",
        fit.criterion
    );
    assert!(fit.evaluations > 1, "the search should have taken steps");
}

/// Applying the fit puts the moving image back on the reference.
#[test]
fn the_fit_can_be_applied_to_undo_the_motion() {
    let (w, h) = (96, 80);
    let reference = textured(w, h);
    let m = Matrix {
        c: [3.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0],
    };
    let moving = transform::warp(&reference, w, h, &m);
    let fit = register_pair(
        &reference,
        &moving,
        w,
        h,
        &settings(Transformation::Translation),
    )
    .expect("fit");
    let restored = transform::warp(&moving, w, h, &fit.matrix);
    // Compare away from the border, where the warp pulls in mirrored content
    // that was never in the reference.
    let mut worst = 0.0f32;
    for y in 10..h - 10 {
        for x in 10..w - 10 {
            worst = worst.max((restored[y * w + x] - reference[y * w + x]).abs());
        }
    }
    let range = reference.iter().cloned().fold(f32::MIN, f32::max)
        - reference.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        worst < range * 0.05,
        "restored image differs by {worst} on a range of {range}"
    );
}

/// Empty or mis-sized input is refused rather than indexed past.
#[test]
fn degenerate_input_is_refused() {
    let s = settings(Transformation::Translation);
    assert!(register_pair(&[], &[], 0, 0, &s).is_none());
    assert!(register_pair(&[1.0; 4], &[1.0; 4], 0, 2, &s).is_none());
    // A buffer shorter than the declared size.
    assert!(register_pair(&[1.0; 3], &[1.0; 4], 2, 2, &s).is_none());
}

/// A flat image has nothing to register against: there is no gradient anywhere,
/// so every transformation scores the same. It must come back without a step
/// rather than wandering off or dividing by zero.
#[test]
fn a_featureless_image_does_not_wander() {
    let (w, h) = (48, 48);
    let flat = vec![100.0f32; w * h];
    let fit = register_pair(&flat, &flat, w, h, &settings(Transformation::Translation))
        .expect("a flat pair should still produce a fit");
    assert!(
        fit.matrix.c[0].abs() < 1.0 && fit.matrix.c[4].abs() < 1.0,
        "a featureless pair should not move: {:?}",
        fit.matrix.c
    );
    assert!(fit.criterion.is_finite());
}

/// The parallel path computes the same answers as the sequential one. It is the
/// same function per frame either way, so this is a check that the plumbing
/// does not reorder or drop results.
#[test]
fn the_parallel_path_agrees_with_the_sequential_one() {
    let (w, h) = (64, 64);
    let reference = textured(w, h);
    let frames: Vec<Vec<f32>> = [(2.0, 1.0), (-3.0, 2.0), (0.0, 0.0), (1.5, -1.5)]
        .iter()
        .map(|&(dx, dy)| {
            transform::warp(
                &reference,
                w,
                h,
                &Matrix {
                    c: [dx, 1.0, 0.0, 0.0, dy, 0.0, 1.0, 0.0],
                },
            )
        })
        .collect();
    let s = settings(Transformation::Translation);
    let seq = register_each(&frames, &reference, w, h, &s, false);
    let par = register_each(&frames, &reference, w, h, &s, true);
    assert_eq!(seq.len(), par.len());
    for (i, (a, b)) in seq.iter().zip(par.iter()).enumerate() {
        match (a, b) {
            (Some(a), Some(b)) => assert!(
                a.matrix
                    .c
                    .iter()
                    .zip(b.matrix.c.iter())
                    .all(|(x, y)| (x - y).abs() < 1e-12),
                "frame {i} differs between backends"
            ),
            (None, None) => {}
            _ => panic!("frame {i}: one backend fitted and the other did not"),
        }
    }
}

/// `Reference::Previous` is never parallel, so a stack registered that way is a
/// chain whatever backend is asked for. Checked on the settings rather than by
/// timing, which would be flaky.
#[test]
fn the_chained_reference_is_never_parallel() {
    assert!(!Backend::MultiThread.parallel_for(Reference::Previous));
}

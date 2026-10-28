//! The models, checked by recovery.
//!
//! The sharpest test available here is a round trip: take a transformation of a
//! given family, push the model's landmarks through it, and ask the model to
//! recover the transformation from that correspondence. It must come back
//! exactly, because the landmarks are *exactly* a parameterisation of the
//! family — and if it does not, either the matrix construction or the landmark
//! count is wrong.
//!
//! The constrained models get a second test each, on the constraint itself: a
//! rigid-body fit must be a rotation and a scaled-rotation fit must be a
//! similarity, whatever correspondence they are handed. That is what the
//! virtual third landmark in the scaled-rotation solve is for, and nothing in a
//! recovery test would notice if it were dropped.

use super::*;
use crate::settings::{Border, Transformation};

fn through(m: &Matrix, pts: &[Point]) -> Vec<Point> {
    pts.iter().map(|p| m.apply(*p)).collect()
}

fn close(a: &Matrix, b: &Matrix, tol: f64) -> bool {
    a.c.iter().zip(b.c.iter()).all(|(x, y)| (x - y).abs() < tol)
}

/// Landmarks mapped onto themselves are the identity, in every model. A model
/// that cannot express "nothing moved" cannot express anything.
#[test]
fn an_unmoved_correspondence_is_the_identity() {
    for model in Transformation::all() {
        let l = initial_landmarks(model, 64, 48);
        let m = matrix_from_landmarks(model, &l, &l)
            .unwrap_or_else(|| panic!("{model:?} should fit its own landmarks"));
        assert!(
            close(&m, &Matrix::identity(), 1e-9),
            "{model:?} gave {:?} for an unmoved correspondence",
            m.c
        );
    }
}

/// Each model recovers a transformation of its own family exactly.
#[test]
fn every_model_recovers_its_own_family() {
    let cases: [(Transformation, Matrix); 5] = [
        (
            Transformation::Translation,
            Matrix {
                c: [7.5, 1.0, 0.0, 0.0, -3.25, 0.0, 1.0, 0.0],
            },
        ),
        (Transformation::RigidBody, {
            let a: f64 = 0.3;
            Matrix {
                c: [5.0, a.cos(), -a.sin(), 0.0, -2.0, a.sin(), a.cos(), 0.0],
            }
        }),
        (Transformation::ScaledRotation, {
            let (a, s): (f64, f64) = (-0.2, 1.4);
            Matrix {
                c: [
                    3.0,
                    s * a.cos(),
                    -s * a.sin(),
                    0.0,
                    1.5,
                    s * a.sin(),
                    s * a.cos(),
                    0.0,
                ],
            }
        }),
        (
            Transformation::Affine,
            Matrix {
                c: [2.0, 1.1, 0.3, 0.0, -1.0, -0.2, 0.9, 0.0],
            },
        ),
        (
            Transformation::Bilinear,
            Matrix {
                c: [1.0, 1.05, 0.02, 0.0012, -2.0, -0.03, 0.98, -0.0007],
            },
        ),
    ];
    for (model, want) in cases {
        let from = initial_landmarks(model, 128, 96);
        let to = through(&want, &from);
        let got = matrix_from_landmarks(model, &from, &to)
            .unwrap_or_else(|| panic!("{model:?} should fit"));
        assert!(
            close(&got, &want, 1e-7),
            "{model:?}\n  want {:?}\n  got  {:?}",
            want.c,
            got.c
        );
    }
}

/// A rigid-body fit is a rotation, even when the correspondence it is given is
/// not rigid at all. The constraint is imposed by construction, not fitted.
#[test]
fn a_rigid_body_fit_is_always_a_rotation() {
    let from = initial_landmarks(Transformation::RigidBody, 100, 100);
    // A correspondence with scale and shear in it, which a rigid body must
    // refuse to represent.
    let messy = Matrix {
        c: [4.0, 1.7, 0.4, 0.0, -6.0, -0.25, 0.6, 0.0],
    };
    let to = through(&messy, &from);
    let m = matrix_from_landmarks(Transformation::RigidBody, &from, &to).expect("fit");
    let (a, b, c, d) = (m.c[1], m.c[2], m.c[5], m.c[6]);
    assert!(
        (a * a + b * b - 1.0).abs() < 1e-9 && (c * c + d * d - 1.0).abs() < 1e-9,
        "rows should be unit length: {:?}",
        m.c
    );
    assert!((a * c + b * d).abs() < 1e-9, "rows should be orthogonal");
    assert!(
        (a * d - b * c - 1.0).abs() < 1e-9,
        "and a rotation, not a flip"
    );
}

/// A scaled-rotation fit is a similarity: one scale, one angle, no shear. This
/// is what the virtual third correspondence buys, and dropping it would leave a
/// fit that passed the recovery test above and sheared on real data.
#[test]
fn a_scaled_rotation_fit_is_always_a_similarity() {
    let from = initial_landmarks(Transformation::ScaledRotation, 100, 100);
    let messy = Matrix {
        c: [1.0, 1.3, 0.5, 0.0, 2.0, -0.1, 0.8, 0.0],
    };
    let to = through(&messy, &from);
    let m = matrix_from_landmarks(Transformation::ScaledRotation, &from, &to).expect("fit");
    let (a, b, c, d) = (m.c[1], m.c[2], m.c[5], m.c[6]);
    // A similarity has the form [[a, -b], [b, a]].
    assert!(
        (a - d).abs() < 1e-9 && (b + c).abs() < 1e-9,
        "should be [[a,-b],[b,a]], got [[{a},{b}],[{c},{d}]]"
    );
}

/// Degenerate landmarks produce no transformation rather than infinities. Three
/// collinear points do not determine an affine map, and a run that carried NaN
/// forward would fill the output with it.
#[test]
fn a_degenerate_correspondence_is_refused() {
    let collinear = [
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(2.0, 2.0),
    ];
    let to = [
        Point::new(0.0, 0.0),
        Point::new(1.0, 2.0),
        Point::new(2.0, 1.0),
    ];
    assert!(
        matrix_from_landmarks(Transformation::Affine, &collinear, &to).is_none(),
        "collinear landmarks do not determine an affine map"
    );
    // And too few landmarks is refused rather than indexed past.
    assert!(matrix_from_landmarks(Transformation::Affine, &collinear[..2], &to[..2]).is_none());
}

/// Composition is matrix multiplication in the right order: `a.compose(b)`
/// means a point goes through `a` and then through `b`.
#[test]
fn composition_applies_in_order() {
    let shift = Matrix {
        c: [10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    let scale = Matrix {
        c: [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    };
    let both = shift.compose(&scale).expect("linear pair composes");
    let p = Point::new(1.0, 3.0);
    let stepwise = scale.apply(shift.apply(p));
    let composed = both.apply(p);
    assert!(
        (stepwise.x - composed.x).abs() < 1e-12 && (stepwise.y - composed.y).abs() < 1e-12,
        "composed {composed:?} should equal stepwise {stepwise:?}"
    );
}

/// A bilinear pair does not compose, and says so rather than dropping the terms
/// it cannot hold. This is why `Reference::Previous` refuses the model.
#[test]
fn bilinear_refuses_to_compose() {
    let bilinear = Matrix {
        c: [0.0, 1.0, 0.0, 0.001, 0.0, 0.0, 1.0, 0.0],
    };
    let linear = Matrix::identity();
    assert!(bilinear.compose(&linear).is_none());
    assert!(linear.compose(&bilinear).is_none());
    assert!(
        linear.compose(&linear).is_some(),
        "linear pairs still compose"
    );
}

/// The 3x3 form round-trips, so a caller exporting matrices and reading them
/// back gets what it had.
#[test]
fn the_three_by_three_form_round_trips() {
    let m = Matrix {
        c: [2.0, 1.1, 0.3, 0.0, -1.0, -0.2, 0.9, 0.0],
    };
    let back = Matrix::from_3x3(m.to_3x3());
    assert!(close(&m, &back, 1e-15), "{:?} -> {:?}", m.c, back.c);
    // And the layout is pystackreg's: translation in the last column.
    let g = m.to_3x3();
    assert_eq!([g[0][2], g[1][2]], [2.0, -1.0], "translation in column 2");
    assert_eq!(g[2], [0.0, 0.0, 1.0], "homogeneous bottom row");
}

/// The landmark placements are StackReg's, computed from the full image size
/// with integer division. Pinned by value because moving them changes the
/// optimiser's path, so a tidy-looking "simplification" here is a behaviour
/// change that no other test would catch.
#[test]
fn the_landmarks_are_stackregs() {
    let (w, h) = (100usize, 60usize);
    assert_eq!(
        initial_landmarks(Transformation::Translation, w, h),
        vec![Point::new(50.0, 30.0)]
    );
    assert_eq!(
        initial_landmarks(Transformation::RigidBody, w, h),
        vec![
            Point::new(50.0, 30.0),
            Point::new(50.0, 15.0),
            Point::new(50.0, 45.0),
        ]
    );
    assert_eq!(
        initial_landmarks(Transformation::ScaledRotation, w, h),
        vec![Point::new(25.0, 30.0), Point::new(75.0, 30.0)]
    );
    assert_eq!(
        initial_landmarks(Transformation::Affine, w, h),
        vec![
            Point::new(50.0, 15.0),
            Point::new(25.0, 45.0),
            Point::new(75.0, 45.0),
        ]
    );
    // Integer division, so an odd size truncates rather than landing on a half
    // pixel.
    assert_eq!(
        initial_landmarks(Transformation::Translation, 101, 61),
        vec![Point::new(50.0, 30.0)]
    );
}

/// Bilinear's landmarks are the golden-ratio insets, inside the frame and
/// spanning it — four corners, not four points in a line.
#[test]
fn bilinear_landmarks_are_inset_corners() {
    let l = initial_landmarks(Transformation::Bilinear, 200, 120);
    assert_eq!(l.len(), 4);
    for p in &l {
        assert!(
            p.x > 0.0 && p.x < 200.0 && p.y > 0.0 && p.y < 120.0,
            "landmark {p:?} should be inside the frame"
        );
    }
    // Two distinct x values and two distinct y values: a proper quad.
    let xs: std::collections::BTreeSet<_> = l.iter().map(|p| p.x.to_bits()).collect();
    let ys: std::collections::BTreeSet<_> = l.iter().map(|p| p.y.to_bits()).collect();
    assert_eq!(
        (xs.len(), ys.len()),
        (2, 2),
        "should be a rectangle of four"
    );
}

/// Warping by the identity returns the image. The spline reconstructs exactly
/// on the grid, so this is a real equality rather than a blur.
#[test]
fn warping_by_the_identity_returns_the_image() {
    let (w, h) = (12, 9);
    let img: Vec<f32> = (0..w * h).map(|i| ((i * 13) % 97) as f32).collect();
    let out = warp(&img, w, h, &Matrix::identity());
    for (i, (a, b)) in img.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 2e-3,
            "pixel {i}: {a} became {b} under the identity"
        );
    }
}

/// And by a whole-pixel shift, moves it by that many pixels — in the direction
/// the matrix says, which is the one that catches a sign error.
#[test]
fn warping_by_a_whole_pixel_shift_moves_the_image() {
    let (w, h) = (10, 8);
    // A single bright pixel, so where it lands is unambiguous.
    let mut img = vec![0.0f32; w * h];
    img[4 * w + 5] = 100.0;
    // The matrix maps reference -> moving, so to make the feature appear two
    // columns further right in the output, the output grid must read two
    // columns further left in the input.
    let m = Matrix {
        c: [-2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    let out = warp(&img, w, h, &m);
    let brightest = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| (i % w, i / w))
        .expect("an output pixel");
    assert_eq!(
        brightest,
        (7, 4),
        "a reference->moving offset of -2 should move the feature right by 2"
    );
}

/// The output warp fills what the move uncovers instead of mirroring into it:
/// the edge pixel repeated, or zero. What it still covers reads exactly what
/// the mirrored warp reads.
#[test]
fn a_bordered_warp_fills_what_the_move_uncovers() {
    let (w, h) = (10, 6);
    // Bright on the left, so a mirror would bring it back in on the left.
    let img: Vec<f32> = (0..w * h).map(|i| (10 - (i % w)) as f32).collect();
    // Output x reads input x - 3: the three left columns are uncovered.
    let m = Matrix {
        c: [-3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    let mirrored = warp(&img, w, h, &m);
    let edged = warp_bordered(&img, w, h, &m, Border::Edge, false);
    let zeroed = warp_bordered(&img, w, h, &m, Border::Constant, false);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x < 3 {
                assert!((edged[i] - 10.0).abs() < 2e-3, "({x},{y}) edge {}", edged[i]);
                assert_eq!(zeroed[i], 0.0, "({x},{y}) not filled");
                // The mirror reads -1 as column 0, so it differs from a fill
                // only further out.
                if x < 2 {
                    assert!((mirrored[i] - 10.0).abs() > 0.5, "the mirror is not a fill");
                }
            } else {
                assert_eq!(edged[i], mirrored[i], "({x},{y}) covered, but changed");
                assert_eq!(zeroed[i], mirrored[i], "({x},{y}) covered, but changed");
            }
        }
    }
    // Unless asked, it is TurboReg's mirror.
    assert_eq!(warp_bordered(&img, w, h, &m, Border::Native, false), mirrored);
    // The identity uncovers nothing, whatever the border — rounding included.
    for border in Border::all() {
        assert_eq!(
            warp_bordered(&img, w, h, &Matrix::identity(), border, false),
            warp(&img, w, h, &Matrix::identity())
        );
    }
    // And threads change nothing here either.
    let a: f64 = 0.07;
    let r = Matrix {
        c: [2.3, a.cos(), -a.sin(), 0.0, -1.1, a.sin(), a.cos(), 0.0],
    };
    for border in Border::all() {
        assert_eq!(
            warp_bordered(&img, w, h, &r, border, false),
            warp_bordered(&img, w, h, &r, border, true)
        );
    }
}

/// Resampling gives the same image on every core as on one. Every output
/// pixel is read independently from the same coefficients.
#[test]
fn warping_does_not_depend_on_threads() {
    let (w, h) = (53usize, 31usize);
    let img: Vec<f32> = (0..w * h).map(|i| ((i * 13) % 251) as f32).collect();
    let a: f64 = 0.07;
    let m = Matrix {
        c: [2.3, a.cos(), -a.sin(), 0.0, -1.1, a.sin(), a.cos(), 0.0],
    };
    let seq = warp_with(&img, w, h, &m, false);
    let par = warp_with(&img, w, h, &m, true);
    assert_eq!(seq, par, "threaded warp differs");
    assert_eq!(
        warp(&img, w, h, &m),
        seq,
        "the plain entry point is sequential"
    );
    // Degenerate sizes produce an empty image rather than a panic.
    assert!(warp_with(&[], 0, 0, &m, true).is_empty());
}

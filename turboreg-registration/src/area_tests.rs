//! The common area against slow ways of finding it, and cutting to it against
//! a picture whose every point is known.

use super::*;
use crate::transform::{warp_area, warp_bordered};
use crate::Border;

/// A small deterministic generator, so a failure names its own inputs.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// Uniform in `-r .. r`.
    fn within(&mut self, r: f64) -> f64 {
        ((self.next() % 20_001) as f64 / 10_000.0 - 1.0) * r
    }
}

fn brute_largest(mask: &[bool], rows: usize, cols: usize) -> usize {
    let mut best = 0;
    for y0 in 0..rows {
        for x0 in 0..cols {
            for y1 in y0..rows {
                for x1 in x0..cols {
                    if (y0..=y1).all(|y| (x0..=x1).all(|x| mask[y * cols + x])) {
                        best = best.max((y1 - y0 + 1) * (x1 - x0 + 1));
                    }
                }
            }
        }
    }
    best
}

#[test]
fn the_largest_rectangle_is_the_largest() {
    let mut rng = Lcg(5);
    for case in 0..300 {
        let (rows, cols) = (1 + rng.next() as usize % 8, 1 + rng.next() as usize % 9);
        let mask: Vec<bool> = (0..rows * cols).map(|_| rng.next() % 5 != 0).collect();
        let want = brute_largest(&mask, rows, cols);
        match largest_rectangle(&mask, rows, cols) {
            None => assert_eq!(want, 0, "case {case}"),
            Some(r) => {
                assert_eq!(r.height * r.width, want, "case {case}: {r:?}");
                let all = (r.row..r.row + r.height)
                    .all(|y| (r.col..r.col + r.width).all(|x| mask[y * cols + x]));
                assert!(all, "case {case}: {r:?} is not all covered");
            }
        }
    }
}

/// A rotation by `degrees` about the frame's middle, then a shift.
fn turn(degrees: f64, ty: f64, tx: f64, ly: usize, lx: usize) -> Matrix {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let (cy, cx) = (ly as f64 / 2.0, lx as f64 / 2.0);
    Matrix {
        c: [
            cx - cos * cx + sin * cy + tx,
            cos,
            -sin,
            0.0,
            cy - sin * cx - cos * cy + ty,
            sin,
            cos,
            0.0,
        ],
    }
}

/// A random transformation near the identity: a shift, a small rotation and
/// scale, a little shear, and sometimes a bilinear term.
fn wobble(rng: &mut Lcg, ly: usize, lx: usize) -> Matrix {
    let mut m = turn(rng.within(4.0), rng.within(4.0), rng.within(4.0), ly, lx);
    let s = 1.0 + rng.within(0.03);
    m.c[1] *= s;
    m.c[6] *= s;
    m.c[2] += rng.within(0.02);
    if rng.next() % 3 == 0 {
        m.c[3] = rng.within(0.0005);
        m.c[7] = -m.c[3];
    }
    m
}

/// Every pixel of `area` is inside the frame and read from inside it by every
/// transformation.
fn all_covered(area: &Area, ly: usize, lx: usize, matrices: &[Matrix]) -> bool {
    (0..area.height).all(|r| {
        (0..area.width).all(|c| {
            let (y, x) = area.at(r, c);
            on_frame(Point::new(x, y), lx, ly)
                && matrices
                    .iter()
                    .all(|m| on_frame(m.apply(Point::new(x, y)), lx, ly))
        })
    })
}

/// The area holds only covered pixels, and when it is upright it is the
/// largest rectangle of what resampling never fills.
#[test]
fn the_common_area_is_covered_and_largest() {
    let mut rng = Lcg(29);
    for case in 0..30 {
        let (ly, lx) = (16 + rng.next() as usize % 40, 16 + rng.next() as usize % 40);
        let frames = 1 + rng.next() as usize % 5;
        let matrices: Vec<Matrix> = (0..frames).map(|_| wobble(&mut rng, ly, lx)).collect();
        let ones = vec![1.0f32; ly * lx];
        let mut covered = vec![true; ly * lx];
        for m in &matrices {
            let moved = warp_bordered(&ones, lx, ly, m, Border::Constant, false);
            for (c, v) in covered.iter_mut().zip(&moved) {
                *c &= *v != 0.0;
            }
        }
        let upright = largest_rectangle(&covered, ly, lx).map_or(0, |r| r.height * r.width);
        match common_area(&matrices, ly, lx, true) {
            Err(_) => assert_eq!(upright, 0, "case {case}"),
            Ok(a) => {
                assert!(all_covered(&a, ly, lx, &matrices), "case {case}: {a:?}");
                if a.is_upright() {
                    assert_eq!(a.height * a.width, upright, "case {case}: {a:?}");
                } else {
                    assert!(a.height * a.width > upright, "case {case}: {a:?} vs {upright}");
                }
                assert_eq!(common_area(&matrices, ly, lx, false), Ok(a), "case {case}");
            }
        }
    }
}

/// A stack that turned a long way: the common area is slanted, a turned
/// rectangle holds clearly more than any upright one, and it is found.
#[test]
fn a_turned_stack_is_cut_at_an_angle() {
    let (ly, lx) = (60usize, 160usize);
    // The reference itself, and the field turned 30 degrees: what both cover
    // is the long frame and a long frame across it at 30 degrees.
    let matrices = [Matrix::identity(), turn(30.0, 0.0, 0.0, ly, lx)];
    let area = common_area(&matrices, ly, lx, true).expect("an area");
    assert!(all_covered(&area, ly, lx, &matrices), "{area:?}");
    let up = {
        let ones = vec![1.0f32; ly * lx];
        let moved = warp_bordered(&ones, lx, ly, &matrices[1], Border::Constant, false);
        let mask: Vec<bool> = moved.iter().map(|&v| v != 0.0).collect();
        largest_rectangle(&mask, ly, lx).map_or(0, |r| r.height * r.width)
    };
    assert!(!area.is_upright(), "{area:?} against an upright {up}");
    assert!(area.height * area.width > up + up / 10, "{area:?} against {up}");
}

/// Translation and scale only: upright, without searching.
#[test]
fn a_shifted_stack_is_cut_upright() {
    let shift = |tx: f64, ty: f64| Matrix {
        c: [tx, 1.0, 0.0, 0.0, ty, 0.0, 1.0, 0.0],
    };
    let a = common_area(&[shift(2.5, -1.0), shift(-3.0, 4.0)], 40, 50, true).expect("an area");
    assert!(a.is_upright());
    // Columns 3 ..= 46 and rows 1 ..= 35 are read from inside both frames.
    assert_eq!((a.y, a.x, a.height, a.width), (1.0, 3.0, 35, 44));
}

#[test]
fn frames_that_share_nothing_have_no_area() {
    let shift = |tx: f64| Matrix {
        c: [tx, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    let err = common_area(&[shift(7.0), shift(-7.0)], 10, 12, true).expect_err("none shared");
    assert_eq!(
        err,
        NoCommonArea {
            ly: 10,
            lx: 12,
            frames: 2
        }
    );
    assert!(err.to_string().contains("no common area"), "{err}");
    // A transformation that is not a number covers nothing — and is not
    // waved through by a band too narrow to look at it.
    let broken = Matrix {
        c: [f64::NAN, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };
    assert!(common_area(&[Matrix::identity(), broken], 10, 12, true).is_err());
    assert_eq!(
        common_area(&[Matrix::identity()], 10, 12, true),
        Ok(Area {
            y: 0.0,
            x: 0.0,
            height: 10,
            width: 12,
            angle: 0.0
        })
    );
}

/// A cut reads each result pixel from where the transformation sends it: on
/// a plane, which the cubic spline reproduces exactly, every value is known.
#[test]
fn a_cut_reads_where_the_transformation_says() {
    let (ly, lx) = (40, 50);
    let plane: Vec<f32> = (0..ly * lx)
        .map(|i| (3 * (i / lx) + 2 * (i % lx)) as f32)
        .collect();
    let m = turn(12.0, 1.5, -2.0, ly, lx);
    let area = Area {
        y: 12.0,
        x: 15.0,
        height: 10,
        width: 14,
        angle: 25f64.to_radians(),
    };
    let out = warp_area(&plane, lx, ly, &m, &area, false);
    for r in 0..area.height {
        for c in 0..area.width {
            let (y, x) = area.at(r, c);
            let p = m.apply(Point::new(x, y));
            let want = 3.0 * p.y + 2.0 * p.x;
            let got = out[r * area.width + c] as f64;
            assert!((got - want).abs() < 0.05, "({r}, {c}): {got} vs {want}");
        }
    }
    assert_eq!(warp_area(&plane, lx, ly, &m, &area, true), out);
}

/// The reference and the field turned 10 degrees: the largest rectangle they
/// share sits on the bisector, at 5 degrees, where a brute-force scan finds
/// 118 by 118 pixels. The rough ranking alone settled on 4.6 degrees and 116
/// by 116; the candidates it suggests are now checked exactly, and land on the
/// bisector within a row of the best.
#[test]
fn the_best_angle_is_checked_not_guessed() {
    let (ly, lx) = (128usize, 128usize);
    let matrices = [Matrix::identity(), turn(10.0, 0.0, 0.0, ly, lx)];
    let a = common_area(&matrices, ly, lx, true).expect("an area");
    assert!(all_covered(&a, ly, lx, &matrices), "{a:?}");
    assert!((a.angle.to_degrees().abs() - 5.0).abs() < 0.15, "{a:?}");
    assert!(
        a.height * a.width >= 118 * 117,
        "{} x {} at {:.2} degrees",
        a.height,
        a.width,
        a.angle.to_degrees()
    );
}

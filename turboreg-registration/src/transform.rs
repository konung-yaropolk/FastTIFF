// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The five transformation models, as matrices and as landmark pairs.
//!
//! # Two representations, and why both exist
//!
//! A transformation is a matrix, and that is what a caller gets back. But the
//! optimiser does not search over matrix entries — it searches over the
//! positions of a handful of **landmarks**, and re-derives the matrix from them
//! at every step.
//!
//! That indirection is not decoration. It is what makes the constrained models
//! expressible: a rigid body has three degrees of freedom but a 2x3 matrix has
//! six, and searching the six would wander off the rigid manifold at the first
//! step. Parameterising by landmarks and *constructing* the matrix from them
//! keeps every iterate inside the family by arithmetic rather than by
//! projection. It is also why the models differ in how many landmarks they
//! carry — three for a rigid body with only three degrees of freedom — and why
//! the landmark positions themselves matter to the result: they set the
//! parameterisation, and so the path the search takes.
//!
//! # Direction
//!
//! Every matrix here maps **reference to moving**: give it a point in the fixed
//! image and it says where to look in the one being registered. That is the
//! direction a pull-resampler wants, so [`warp`] uses it as it stands.

use crate::settings::Transformation;

/// A point, in pixels, `x` across and `y` down.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

/// A transformation, in the one form that covers all five models.
///
/// `x' = c[0] + c[1] * x + c[2] * y + c[3] * x * y`
/// `y' = c[4] + c[5] * x + c[6] * y + c[7] * x * y`
///
/// The `x * y` terms are zero for every model but [`Transformation::Bilinear`],
/// so one shape serves all of them and the linear cases cost two multiplies by
/// zero. Holding bilinear in its own type instead would double every function
/// below for the one model that is used least.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Matrix {
    pub c: [f64; 8],
}

impl Default for Matrix {
    fn default() -> Self {
        Matrix::identity()
    }
}

impl Matrix {
    /// The transformation that moves nothing.
    pub fn identity() -> Self {
        Matrix {
            c: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Where `p` in the reference lands in the moving image.
    pub fn apply(&self, p: Point) -> Point {
        let c = &self.c;
        Point {
            x: c[0] + c[1] * p.x + c[2] * p.y + c[3] * p.x * p.y,
            y: c[4] + c[5] * p.x + c[6] * p.y + c[7] * p.x * p.y,
        }
    }

    /// The 3x3 homogeneous form, as pystackreg's `get_matrix` returns it:
    /// `[[a, b, tx], [c, d, ty], [0, 0, 1]]`.
    ///
    /// Only meaningful for the four linear models; a bilinear transformation
    /// has no 3x3 form and this drops its `x * y` terms, so it is the caller's
    /// business not to ask. Stated rather than enforced because the natural
    /// caller — someone exporting matrices for a report — knows which model
    /// they ran.
    pub fn to_3x3(&self) -> [[f64; 3]; 3] {
        let c = &self.c;
        [[c[1], c[2], c[0]], [c[5], c[6], c[4]], [0.0, 0.0, 1.0]]
    }

    /// Build from the 3x3 form.
    pub fn from_3x3(m: [[f64; 3]; 3]) -> Self {
        Matrix {
            c: [
                m[0][2], m[0][0], m[0][1], 0.0, m[1][2], m[1][0], m[1][1], 0.0,
            ],
        }
    }

    /// `self` followed by `then`, for the models where that is meaningful.
    ///
    /// Used by [`Reference::Previous`], which chains each frame's transformation
    /// onto the last. Returns `None` for a bilinear pair, because the
    /// composition of two bilinear maps is not bilinear — the product has
    /// `x^2 y` terms this representation cannot hold, and silently dropping
    /// them would give a transformation that is plausible and wrong.
    ///
    /// [`Reference::Previous`]: crate::settings::Reference::Previous
    pub fn compose(&self, then: &Matrix) -> Option<Matrix> {
        if self.is_bilinear() || then.is_bilinear() {
            return None;
        }
        let a = self.to_3x3();
        let b = then.to_3x3();
        let mut m = [[0.0f64; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| b[i][k] * a[k][j]).sum();
            }
        }
        Some(Matrix::from_3x3(m))
    }

    /// Whether either `x * y` term is live.
    pub fn is_bilinear(&self) -> bool {
        self.c[3] != 0.0 || self.c[7] != 0.0
    }
}

/// Where the landmarks start, for an image of this size.
///
/// These positions are StackReg's, computed from the **full** image dimensions
/// with integer division. They are not arbitrary and not interchangeable with
/// the interactive TurboReg plugin's golden-ratio insets: the landmarks are the
/// search's parameters, so moving them changes the Jacobian normaliser and the
/// path the optimiser takes, and a run started from different landmarks lands
/// somewhere slightly different. Matching StackReg is the point.
///
/// Bilinear is the exception that uses golden-ratio insets in both programs.
pub fn initial_landmarks(model: Transformation, w: usize, h: usize) -> Vec<Point> {
    let (wf, hf) = (w as f64, h as f64);
    // Integer division throughout, as the reference does it.
    let half_w = (w / 2) as f64;
    let half_h = (h / 2) as f64;
    let quarter_w = (w / 4) as f64;
    let quarter_h = (h / 4) as f64;
    let three_quarter_w = (3 * w / 4) as f64;
    let three_quarter_h = (3 * h / 4) as f64;
    match model {
        Transformation::Translation => vec![Point::new(half_w, half_h)],
        Transformation::RigidBody => vec![
            Point::new(half_w, half_h),
            Point::new(half_w, quarter_h),
            Point::new(half_w, three_quarter_h),
        ],
        Transformation::ScaledRotation => vec![
            Point::new(quarter_w, half_h),
            Point::new(three_quarter_w, half_h),
        ],
        Transformation::Affine => vec![
            Point::new(half_w, quarter_h),
            Point::new(quarter_w, three_quarter_h),
            Point::new(three_quarter_w, three_quarter_h),
        ],
        Transformation::Bilinear => {
            // The golden ratio, inset by a quarter.
            const GR: f64 = 0.618_033_988_749_894_9;
            let ix = (0.25 * GR * wf).floor();
            let iy = (0.25 * GR * hf).floor();
            let fx = wf - (0.25 * GR * wf).ceil();
            let fy = hf - (0.25 * GR * hf).ceil();
            vec![
                Point::new(ix, iy),
                Point::new(ix, fy),
                Point::new(fx, iy),
                Point::new(fx, fy),
            ]
        }
    }
}

/// Solve `a * x = b` in place by Gauss-Jordan with partial pivoting.
///
/// Returns `None` for a singular system, which a degenerate landmark set
/// produces — three collinear landmarks for an affine fit, say. The caller
/// keeps its previous estimate rather than propagating infinities, which is the
/// difference between a run that stalls and one that fills the image with NaN.
fn solve(a: &mut [Vec<f64>], b: &mut [Vec<f64>]) -> Option<()> {
    let n = a.len();
    let rhs = b[0].len();
    for col in 0..n {
        // Pivot on the largest magnitude available, which keeps the elimination
        // stable when a landmark set is nearly degenerate.
        let (mut best, mut best_at) = (a[col][col].abs(), col);
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                best_at = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(col, best_at);
        b.swap(col, best_at);
        let pivot = a[col][col];
        for v in a[col].iter_mut() {
            *v /= pivot;
        }
        for v in b[col].iter_mut() {
            *v /= pivot;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            let (pivot_row, target) = {
                let (lo, hi) = if col < r {
                    a.split_at_mut(r)
                } else {
                    a.split_at_mut(col)
                };
                if col < r {
                    (&lo[col], &mut hi[0])
                } else {
                    (&hi[0], &mut lo[r])
                }
            };
            for (t, p) in target.iter_mut().zip(pivot_row.iter()) {
                *t -= factor * p;
            }
            let (pivot_row, target) = {
                let (lo, hi) = if col < r {
                    b.split_at_mut(r)
                } else {
                    b.split_at_mut(col)
                };
                if col < r {
                    (&lo[col], &mut hi[0])
                } else {
                    (&hi[0], &mut lo[r])
                }
            };
            for (t, p) in target.iter_mut().take(rhs).zip(pivot_row.iter()) {
                *t -= factor * p;
            }
        }
    }
    Some(())
}

/// The transformation carrying `from` onto `to`.
///
/// Both slices are the model's landmarks in order; the result maps a point in
/// the space `from` lives in to the space `to` lives in. Which of those is the
/// reference depends on the caller — the optimiser works in one direction and
/// reports the other (see the module header of [`crate::optimize`]).
///
/// Returns `None` when the landmarks are degenerate and no transformation of
/// this family carries one set onto the other.
pub fn matrix_from_landmarks(
    model: Transformation,
    from: &[Point],
    to: &[Point],
) -> Option<Matrix> {
    if from.len() < model.landmarks() || to.len() < model.landmarks() {
        return None;
    }
    let mut c = [0.0f64; 8];
    match model {
        Transformation::Translation => {
            c[1] = 1.0;
            c[6] = 1.0;
            c[0] = to[0].x - from[0].x;
            c[4] = to[0].y - from[0].y;
        }
        Transformation::RigidBody => {
            // The angle between the two landmark *baselines*. The arguments to
            // `atan2` are (dx, dy) rather than the usual (dy, dx): that is the
            // reference's own spelling, and it is harmless because the two
            // calls are subtracted — both measure from the same axis, so the
            // difference is the rotation either way.
            let a_from = (from[2].x - from[1].x).atan2(from[2].y - from[1].y);
            let a_to = (to[2].x - to[1].x).atan2(to[2].y - to[1].y);
            let angle = a_from - a_to;
            let (s, k) = (angle.sin(), angle.cos());
            c[1] = k;
            c[2] = -s;
            c[5] = s;
            c[6] = k;
            // Fixed so landmark 0 maps exactly. Not a least-squares fit over
            // the three pairs: with the rotation already decided, one point
            // pins the translation, and the reference pins it on the first.
            c[0] = to[0].x - k * from[0].x + s * from[0].y;
            c[4] = to[0].y - s * from[0].x - k * from[0].y;
        }
        Transformation::ScaledRotation => {
            // Two real correspondences plus one *virtual* one, which is what
            // imposes the similarity constraint. The third row is the point
            // obtained by rotating the baseline a quarter turn: requiring it to
            // map correctly forbids shear and anisotropic scale, leaving
            // rotation and one isotropic factor.
            let mut a = vec![
                vec![1.0, from[0].x, from[0].y],
                vec![1.0, from[1].x, from[1].y],
                // A point like any other, so its translation column is 1. It is
                // only the *position* that is invented: p1 rotated a quarter
                // turn about p0. Writing 0 here instead — treating the row as a
                // direction rather than a point — drops the translation from
                // the constraint and lets a shear back in, which still fits the
                // two real pairs and is no longer a similarity.
                vec![
                    1.0,
                    from[0].y - from[1].y + from[1].x,
                    from[1].x + from[1].y - from[0].x,
                ],
            ];
            let mut b = vec![
                vec![to[0].x, to[0].y],
                vec![to[1].x, to[1].y],
                vec![to[0].y - to[1].y + to[1].x, to[1].x + to[1].y - to[0].x],
            ];
            solve(&mut a, &mut b)?;
            c[0] = b[0][0];
            c[1] = b[1][0];
            c[2] = b[2][0];
            c[4] = b[0][1];
            c[5] = b[1][1];
            c[6] = b[2][1];
        }
        Transformation::Affine => {
            let mut a = vec![
                vec![1.0, from[0].x, from[0].y],
                vec![1.0, from[1].x, from[1].y],
                vec![1.0, from[2].x, from[2].y],
            ];
            let mut b = vec![
                vec![to[0].x, to[0].y],
                vec![to[1].x, to[1].y],
                vec![to[2].x, to[2].y],
            ];
            solve(&mut a, &mut b)?;
            c[0] = b[0][0];
            c[1] = b[1][0];
            c[2] = b[2][0];
            c[4] = b[0][1];
            c[5] = b[1][1];
            c[6] = b[2][1];
        }
        Transformation::Bilinear => {
            let mut a: Vec<Vec<f64>> = (0..4)
                .map(|k| vec![1.0, from[k].x, from[k].y, from[k].x * from[k].y])
                .collect();
            let mut b: Vec<Vec<f64>> = (0..4).map(|k| vec![to[k].x, to[k].y]).collect();
            solve(&mut a, &mut b)?;
            c[0] = b[0][0];
            c[1] = b[1][0];
            c[2] = b[2][0];
            c[3] = b[3][0];
            c[4] = b[0][1];
            c[5] = b[1][1];
            c[6] = b[2][1];
            c[7] = b[3][1];
        }
    }
    if c.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(Matrix { c })
}

/// Resample `moving` onto the reference's grid.
///
/// A **pull**: walk the output grid, send each point through `m` to find where
/// it came from, and read the spline there. The alternative — pushing each
/// input pixel to where it goes — leaves holes wherever the transformation
/// spreads the grid out, which is why nobody does it.
///
/// `moving` is samples, not coefficients; this prefilters a copy. Out-of-frame
/// positions read the mirrored extension rather than a fill value, because the
/// spline is defined there and a hard edge would show as a seam.
pub fn warp(moving: &[f32], w: usize, h: usize, m: &Matrix) -> Vec<f32> {
    let mut coeff = moving.to_vec();
    crate::spline::prefilter_2d(&mut coeff, w, h, &crate::spline::POLES_3);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let p = m.apply(Point::new(x as f64, y as f64));
            out[y * w + x] = crate::spline::interpolate(&coeff, w, h, p.x, p.y) as f32;
        }
    }
    out
}

#[cfg(test)]
#[path = "transform_tests.rs"]
mod tests;

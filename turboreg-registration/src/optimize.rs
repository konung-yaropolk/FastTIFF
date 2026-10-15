// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The search: finding the transformation that lines two images up.
//!
//! # What is minimised
//!
//! The sum of squared intensity differences between the reference and the
//! moving image resampled onto the reference's grid, over a transformation
//! parameterised by landmark positions (see [`crate::transform`]).
//!
//! # The damping is diagonal, and that is deliberate
//!
//! The inner step is
//!
//! ```text
//!     step_k = gradient_k / ((1 + lambda) * hessian_kk)
//! ```
//!
//! which uses only the diagonal of the curvature matrix — a Jacobi-preconditioned
//! gradient step, not the Levenberg-Marquardt step the name suggests. This is
//! not a simplification taken here: the reference implementation builds a full
//! Hessian, writes only its diagonal into the damped system, and inverts that,
//! so every one of its inner steps is diagonal too. Assembling the full damped
//! system instead converges in fewer iterations and lands somewhere slightly
//! different, which would make results incomparable with StackReg for no gain
//! that anyone asked for.
//!
//! The full Hessian is used in exactly one place, and it is worth keeping: a
//! single Gauss-Newton step after the loop, accepted only if it improves the
//! criterion. That is what turns a run of damped gradient steps into a fit with
//! subpixel accuracy.
//!
//! # How this differs from the reference, stated plainly
//!
//! The reference optimises the *inverse* transformation applied to the
//! reference image (Thevenaz's "ML*"), which lets it sample image gradients
//! once, on the grid, instead of at moving off-grid positions every iteration.
//! It is a speed reformulation.
//!
//! This implementation optimises the forward map directly and interpolates the
//! gradients where it needs them. The minimum being sought is the same and the
//! schedule around it — damping, lambda, iteration caps, stopping rule — is the
//! reference's. The *path* taken to the minimum is not identical, so results
//! agree to within the convergence tolerance rather than bit for bit. Anyone
//! needing bit-parity with StackReg should verify against it directly; this
//! crate does not claim it, and a claim that cannot be checked here would be
//! worth less than saying so.

// Only the threaded path names rayon's iterators; without the feature the
// sequential branch below is the only one compiled and the import is dead.
#[cfg(feature = "threads")]
use crate::par::*;
use crate::pyramid::{self, Level};
use crate::settings::{Settings, Transformation};
use crate::spline;
use crate::transform::{self, Matrix, Point};

/// Where the inner loop stops, in pixels of landmark movement.
const PIXEL_PRECISION: f64 = 0.001;

/// Iterations allowed at full resolution. Coarser levels get `MANY_ITERATIONS *
/// 2^octave`, because a step there is worth twice as much in full-resolution
/// pixels for each octave of reduction.
const MANY_ITERATIONS: usize = 10;

/// How far a landmark is nudged when measuring the Jacobian numerically.
///
/// The transformation is affine (or bilinear) in the landmark coordinates, so
/// the difference quotient is *exact* for any step — there is no truncation
/// error to trade against rounding. A step near 1 pixel keeps it well clear of
/// the subtraction's rounding.
const JACOBIAN_STEP: f64 = 1.0;

/// What a fit produced.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Maps reference coordinates to moving coordinates.
    pub matrix: Matrix,
    /// The final normalised criterion: lower is better, zero is identical.
    pub criterion: f64,
    /// How many criterion evaluations it took, over the whole pyramid.
    pub evaluations: usize,
}

/// A level's resampling tables: the moving image and its gradients, as spline
/// coefficients ready to be read at fractional positions.
struct Sampler {
    w: usize,
    h: usize,
    img: Vec<f32>,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl Sampler {
    fn of(level: &Level) -> Self {
        let (gx, gy) = pyramid::gradients(level);
        let mut gxc = gx;
        let mut gyc = gy;
        spline::prefilter_2d(&mut gxc, level.w, level.h, &spline::POLES_3);
        spline::prefilter_2d(&mut gyc, level.w, level.h, &spline::POLES_3);
        Sampler {
            w: level.w,
            h: level.h,
            img: pyramid::coefficients(level),
            gx: gxc,
            gy: gyc,
        }
    }

    fn value(&self, p: Point) -> f64 {
        spline::interpolate(&self.img, self.w, self.h, p.x, p.y)
    }

    fn grad(&self, p: Point) -> (f64, f64) {
        (
            spline::interpolate(&self.gx, self.w, self.h, p.x, p.y),
            spline::interpolate(&self.gy, self.w, self.h, p.x, p.y),
        )
    }
}

/// The criterion's normaliser for a model, given its current landmarks.
///
/// Translation, rigid body and bilinear divide by the pixel count alone. The
/// two models that can change area do more: dividing by how much the
/// transformation shrinks the image stops the search from improving its score
/// by collapsing the picture to a point, which is a real minimum of the raw sum
/// of squares and a useless one.
fn normaliser(model: Transformation, from: &[Point], to: &[Point], area: f64) -> f64 {
    let sq = |a: Point, b: Point| (a.x - b.x).powi(2) + (a.y - b.y).powi(2);
    match model {
        Transformation::ScaledRotation => {
            let uv2 = sq(from[0], from[1]);
            let target = sq(to[0], to[1]);
            if target.abs() < f64::EPSILON {
                area
            } else {
                area * uv2 / target
            }
        }
        Transformation::Affine => {
            // Twice the signed area of the landmark triangle, source over
            // target.
            let det = |p: &[Point]| {
                (p[1].x - p[2].x) * p[0].y + (p[2].x - p[0].x) * p[1].y + (p[0].x - p[1].x) * p[2].y
            };
            let target = det(to);
            if target.abs() < f64::EPSILON {
                area
            } else {
                area * (det(from) / target).abs()
            }
        }
        _ => area,
    }
}

/// Evaluate the criterion for a candidate landmark set, and optionally the
/// gradient and curvature with it.
///
/// Returns `None` when the landmarks do not determine a transformation.
///
/// `want_derivatives` is not an optimisation flag so much as a correctness one:
/// a rejected candidate must not overwrite the gradient the next step is built
/// from, and the cheapest way to guarantee that is not to compute it.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    model: Transformation,
    reference: &[f32],
    rw: usize,
    rh: usize,
    sampler: &Sampler,
    fixed: &[Point],
    landmarks: &[Point],
    want_derivatives: bool,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let m = transform::matrix_from_landmarks(model, fixed, landmarks)?;
    let n = model.dof();

    // How the sampling position moves when each parameter moves. The
    // transformation is linear in the landmark coordinates, so a difference
    // quotient here is the exact derivative rather than an approximation of it
    // — which is why this is done numerically instead of by transcribing a
    // closed form per model, where a sign error would be invisible.
    let mut dm: Vec<Matrix> = Vec::new();
    if want_derivatives {
        for k in 0..n {
            let mut nudged = landmarks.to_vec();
            // Parameter k is the x of landmark k/2 for even k, the y for odd.
            // Rigid body is the exception: three landmarks, three parameters,
            // so its parameters are the first landmark's pair and the second's
            // x — enough to span rotation and translation, with the matrix
            // construction imposing rigidity.
            let (idx, axis) = (k / 2, k % 2);
            if axis == 0 {
                nudged[idx].x += JACOBIAN_STEP;
            } else {
                nudged[idx].y += JACOBIAN_STEP;
            }
            let mk = transform::matrix_from_landmarks(model, fixed, &nudged)?;
            let mut diff = [0.0f64; 8];
            for (i, d) in diff.iter_mut().enumerate() {
                *d = (mk.c[i] - m.c[i]) / JACOBIAN_STEP;
            }
            dm.push(Matrix { c: diff });
        }
    }

    let mut sum_sq = 0.0f64;
    let mut gradient = vec![0.0f64; n];
    let mut hessian = vec![0.0f64; n * n];
    let mut counted = 0usize;

    for y in 0..rh {
        for x in 0..rw {
            let p = Point::new(x as f64, y as f64);
            let q = m.apply(p);
            let diff = reference[y * rw + x] as f64 - sampler.value(q);
            sum_sq += diff * diff;
            counted += 1;
            if !want_derivatives {
                continue;
            }
            let (gx, gy) = sampler.grad(q);
            // d(residual)/d(p_k), by the chain rule through the sampling
            // position. The leading minus is the residual's own sign: the model
            // enters through the sampled value, which is subtracted.
            let mut dk = vec![0.0f64; n];
            for (k, d) in dk.iter_mut().enumerate() {
                let v = dm[k].apply(p);
                *d = -(gx * v.x + gy * v.y);
            }
            for a in 0..n {
                gradient[a] -= diff * dk[a];
                for b in a..n {
                    hessian[a * n + b] += dk[a] * dk[b];
                }
            }
        }
    }
    if counted == 0 {
        return None;
    }
    // Mirror the upper triangle.
    if want_derivatives {
        for a in 0..n {
            for b in a..n {
                hessian[b * n + a] = hessian[a * n + b];
            }
        }
    }
    let norm = normaliser(model, landmarks, fixed, counted as f64);
    let criterion = if norm.abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        sum_sq / norm
    };
    Some((criterion, gradient, hessian))
}

/// Solve `h * x = g` for the Gauss-Newton clean-up step.
fn solve_full(h: &[f64], g: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a: Vec<Vec<f64>> = (0..n).map(|r| h[r * n..(r + 1) * n].to_vec()).collect();
    let mut b: Vec<Vec<f64>> = (0..n).map(|r| vec![g[r]]).collect();
    // Gauss-Jordan with partial pivoting, as in `transform::solve`.
    for col in 0..n {
        let (mut best, mut at) = (a[col][col].abs(), col);
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                at = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(col, at);
        b.swap(col, at);
        let pivot = a[col][col];
        a[col].iter_mut().for_each(|v| *v /= pivot);
        b[col][0] /= pivot;
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
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
                *t -= f * p;
            }
            b[r][0] -= f * b[col][0];
        }
    }
    let out: Vec<f64> = b.iter().map(|r| r[0]).collect();
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// Move a landmark set by a parameter step.
fn stepped(landmarks: &[Point], step: &[f64]) -> Vec<Point> {
    let mut out = landmarks.to_vec();
    for (k, s) in step.iter().enumerate() {
        let (idx, axis) = (k / 2, k % 2);
        if idx >= out.len() {
            break;
        }
        if axis == 0 {
            out[idx].x += s;
        } else {
            out[idx].y += s;
        }
    }
    out
}

/// How far a step moves the landmarks, in pixels, averaged over them.
fn displacement(step: &[f64]) -> f64 {
    let pairs = step.len() / 2;
    if pairs == 0 {
        return step.iter().map(|v| v.abs()).sum();
    }
    let mut total = 0.0;
    for k in 0..pairs {
        total += (step[2 * k].powi(2) + step[2 * k + 1].powi(2)).sqrt();
    }
    // An odd parameter out — rigid body's third — counts its own magnitude.
    if step.len() % 2 == 1 {
        total += step[step.len() - 1].abs();
    }
    total / pairs as f64
}

/// Fit one level, starting from `landmarks` and returning where it got to.
fn fit_level(
    model: Transformation,
    reference: &Level,
    moving: &Level,
    fixed: &[Point],
    landmarks: &[Point],
    evaluations: &mut usize,
) -> Vec<Point> {
    let sampler = Sampler::of(moving);
    let n = model.dof();
    let cap = MANY_ITERATIONS * (1usize << reference.octave);

    let mut best = landmarks.to_vec();
    let Some((mut best_criterion, mut gradient, mut hessian)) = evaluate(
        model,
        &reference.img,
        reference.w,
        reference.h,
        &sampler,
        fixed,
        &best,
        true,
    ) else {
        return best;
    };
    *evaluations += 1;

    let mut lambda = 1.0f64;
    let mut iteration = 1usize;
    loop {
        // The damped step, diagonal by construction. See the module header:
        // this is the reference's behaviour, not a shortcut.
        let mut step = vec![0.0f64; n];
        for k in 0..n {
            let d = (1.0 + lambda) * hessian[k * n + k];
            step[k] = if d.abs() < 1e-12 {
                0.0
            } else {
                gradient[k] / d
            };
        }
        let moved = displacement(&step);
        let candidate = stepped(&best, &step);
        if let Some((c, g, hh)) = evaluate(
            model,
            &reference.img,
            reference.w,
            reference.h,
            &sampler,
            fixed,
            &candidate,
            true,
        ) {
            *evaluations += 1;
            if c < best_criterion {
                best_criterion = c;
                best = candidate;
                gradient = g;
                hessian = hh;
                lambda /= 4.0;
            } else {
                lambda *= 4.0;
            }
        } else {
            lambda *= 4.0;
        }
        iteration += 1;
        if iteration >= cap || moved < PIXEL_PRECISION {
            break;
        }
    }

    // One Gauss-Newton step with the full curvature, taken only if it helps.
    // This is where the subpixel accuracy comes from: the damped steps above
    // approach along the gradient, and this one jumps to the quadratic
    // model's minimum.
    if let Some(step) = solve_full(&hessian, &gradient, n) {
        let candidate = stepped(&best, &step);
        if let Some((c, _, _)) = evaluate(
            model,
            &reference.img,
            reference.w,
            reference.h,
            &sampler,
            fixed,
            &candidate,
            false,
        ) {
            *evaluations += 1;
            if c < best_criterion {
                best = candidate;
            }
        }
    }
    best
}

/// Register `moving` onto `reference`, both `w * h`.
///
/// Returns the transformation mapping reference coordinates to moving ones, or
/// `None` if the images are empty or the fit degenerated.
pub fn register_pair(
    reference: &[f32],
    moving: &[f32],
    w: usize,
    h: usize,
    settings: &Settings,
) -> Option<Fit> {
    if w == 0 || h == 0 || reference.len() < w * h || moving.len() < w * h {
        return None;
    }
    let model = settings.transformation;
    let depth = pyramid::depth_for(w, h, settings.max_pyramid_depth);
    let ref_levels = pyramid::build(reference, w, h, depth);
    let mov_levels = pyramid::build(moving, w, h, depth);

    // Landmarks are placed on the full-resolution image and scaled down to
    // whichever level is being worked on; `fixed` is where they sit in the
    // reference, `moving_at` where the search has pushed their images.
    let full = transform::initial_landmarks(model, w, h);
    let mut moving_at = full.clone();

    let mut evaluations = 0usize;
    // Coarsest first.
    for octave in (0..depth).rev() {
        let scale = 1.0 / (1u32 << octave) as f64;
        let fixed: Vec<Point> = full
            .iter()
            .map(|p| Point::new(p.x * scale, p.y * scale))
            .collect();
        let start: Vec<Point> = moving_at
            .iter()
            .map(|p| Point::new(p.x * scale, p.y * scale))
            .collect();
        let got = fit_level(
            model,
            &ref_levels[octave],
            &mov_levels[octave],
            &fixed,
            &start,
            &mut evaluations,
        );
        // Back to full-resolution coordinates for the next level down.
        moving_at = got
            .iter()
            .map(|p| Point::new(p.x / scale, p.y / scale))
            .collect();
    }

    let matrix = transform::matrix_from_landmarks(model, &full, &moving_at)?;
    // One last criterion at full resolution, so the number reported describes
    // the answer returned rather than the last candidate tried.
    let sampler = Sampler::of(&mov_levels[0]);
    let criterion = evaluate(
        model,
        &ref_levels[0].img,
        w,
        h,
        &sampler,
        &full,
        &moving_at,
        false,
    )
    .map(|(c, _, _)| c)
    .unwrap_or(f64::INFINITY);
    Some(Fit {
        matrix,
        criterion,
        evaluations,
    })
}

/// Register every frame of a stack, in parallel where the reference mode allows.
///
/// `frames` are `w * h` each. `reference` is the image every frame is matched
/// against; building it is the caller's business (see [`crate::register_stack`]).
pub fn register_each(
    frames: &[Vec<f32>],
    reference: &[f32],
    w: usize,
    h: usize,
    settings: &Settings,
    parallel: bool,
) -> Vec<Option<Fit>> {
    if parallel {
        #[cfg(feature = "threads")]
        {
            return frames
                .par_iter()
                .map(|f| register_pair(reference, f, w, h, settings))
                .collect();
        }
    }
    frames
        .iter()
        .map(|f| register_pair(reference, f, w, h, settings))
        .collect()
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
mod tests;

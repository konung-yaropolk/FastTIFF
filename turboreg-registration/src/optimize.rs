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

use crate::par::map_in_order;
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

/// The most parameters any model has: bilinear's eight. Sized so the
/// per-pixel derivative scratch can live on the stack — it used to be a
/// `Vec` allocated for every pixel of every evaluation, which is millions of
/// allocations per frame for no reason.
const MAX_DOF: usize = 8;

/// Rows per piece of the criterion's sum.
///
/// Fixed, and that is the property that matters: the sum over pixels is split
/// into pieces of this many rows, each piece is summed on whichever thread
/// takes it, and the pieces are added together in row order on the calling
/// thread. Because the split depends only on the image, never on how many
/// threads there are, the floating-point result is the same on one core or on
/// sixteen — which is what lets "Single-thread CPU" stand as the reference for
/// "Multi-thread CPU" rather than as an approximation of it.
///
/// Eight rows is a few thousand pixels at typical sizes: large enough that
/// scheduling a piece costs nothing next to computing it, small enough that a
/// 64-pixel frame still has eight pieces to share out.
const ROWS_PER_CHUNK: usize = 8;

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
pub(crate) struct Sampler {
    w: usize,
    h: usize,
    pub(crate) img: Vec<f32>,
    pub(crate) gx: Vec<f32>,
    pub(crate) gy: Vec<f32>,
}

impl Sampler {
    /// Build a level's tables, on every core when `parallel`.
    ///
    /// The image is prefiltered once and its gradients taken from those same
    /// coefficients. They used to be prefiltered twice — once here and once
    /// inside the gradient routine — which was a full pass per level per frame
    /// that produced a copy of something already in hand.
    pub(crate) fn of(level: &Level, parallel: bool) -> Self {
        let (w, h) = (level.w, level.h);
        let img = pyramid::coefficients_with(level, parallel);
        let (mut gx, mut gy) = pyramid::gradients_from(&img, w, h, parallel);
        spline::prefilter_2d_with(&mut gx, w, h, &spline::POLES_3, parallel);
        spline::prefilter_2d_with(&mut gy, w, h, &spline::POLES_3, parallel);
        Sampler { w, h, img, gx, gy }
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

/// One piece's share of the criterion and its derivatives.
///
/// Fixed-size, so a piece allocates nothing: the Hessian is held at the
/// largest model's size and a smaller model uses its top-left corner. Only the
/// upper triangle is accumulated; the caller mirrors it once at the end.
struct Partial {
    sum_sq: f64,
    counted: usize,
    gradient: [f64; MAX_DOF],
    hessian: [f64; MAX_DOF * MAX_DOF],
}

impl Default for Partial {
    fn default() -> Self {
        Partial {
            sum_sq: 0.0,
            counted: 0,
            gradient: [0.0; MAX_DOF],
            hessian: [0.0; MAX_DOF * MAX_DOF],
        }
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

/// A candidate made ready to score: its transformation, and how the sampling
/// position moves with each parameter.
///
/// Split out of the evaluation because it is the part that does not depend on
/// the pixels. The CPU and the GPU paths share it, so both score exactly the
/// same transformation and the same derivatives — only the sum over pixels is
/// done in two places.
#[derive(Clone, Copy)]
pub(crate) struct Prepared {
    /// Reference coordinates to moving coordinates.
    pub m: Matrix,
    /// d(position)/d(parameter k), for `k < n`; the rest are zero.
    pub dm: [Matrix; MAX_DOF],
    /// Parameters in the model.
    pub n: usize,
    /// Whether the gradient and curvature are wanted, or only the criterion.
    pub derivatives: bool,
}

/// The transformation for `landmarks`, and its derivatives when asked for.
///
/// `None` when the landmarks do not determine a transformation.
pub(crate) fn prepare(
    model: Transformation,
    fixed: &[Point],
    landmarks: &[Point],
    derivatives: bool,
) -> Option<Prepared> {
    let m = transform::matrix_from_landmarks(model, fixed, landmarks)?;
    let n = model.dof();

    // How the sampling position moves when each parameter moves. The
    // transformation is linear in the landmark coordinates, so a difference
    // quotient here is the exact derivative rather than an approximation of it
    // — which is why this is done numerically instead of by transcribing a
    // closed form per model, where a sign error would be invisible.
    let mut dm = [Matrix { c: [0.0; 8] }; MAX_DOF];
    if derivatives {
        for (k, slot) in dm.iter_mut().enumerate().take(n) {
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
            for (i, d) in slot.c.iter_mut().enumerate() {
                *d = (mk.c[i] - m.c[i]) / JACOBIAN_STEP;
            }
        }
    }
    Some(Prepared {
        m,
        dm,
        n,
        derivatives,
    })
}

/// The sum over pixels: squared residual, and with derivatives the gradient and
/// the upper triangle of the curvature.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sums {
    pub sum_sq: f64,
    pub counted: usize,
    /// `n` entries.
    pub gradient: Vec<f64>,
    /// `n * n`, row-major, upper triangle only; [`finish`] mirrors it.
    pub hessian: Vec<f64>,
}

/// [`Sums`] on the CPU.
///
/// `parallel` spreads the pixels over every core. It changes nothing about the
/// answer — see [`ROWS_PER_CHUNK`] for why the sum is the same sum either way.
pub(crate) fn accumulate(
    prep: &Prepared,
    reference: &[f32],
    rw: usize,
    rh: usize,
    sampler: &Sampler,
    parallel: bool,
) -> Sums {
    let (m, n, want_derivatives) = (prep.m, prep.n, prep.derivatives);
    let dm = &prep.dm[..n];

    // The pixels, in fixed pieces of rows. Each piece sums its own share; the
    // shares come back in row order and are added in that order below. See
    // `ROWS_PER_CHUNK` for why that makes the answer independent of threads.
    let pieces: Vec<std::ops::Range<usize>> = (0..rh)
        .step_by(ROWS_PER_CHUNK)
        .map(|y0| y0..(y0 + ROWS_PER_CHUNK).min(rh))
        .collect();
    let partials = map_in_order(&pieces, parallel, |rows| {
        let mut part = Partial::default();
        let mut dk = [0.0f64; MAX_DOF];
        for y in rows.clone() {
            for x in 0..rw {
                let p = Point::new(x as f64, y as f64);
                let q = m.apply(p);
                let diff = reference[y * rw + x] as f64 - sampler.value(q);
                part.sum_sq += diff * diff;
                part.counted += 1;
                if !want_derivatives {
                    continue;
                }
                let (gx, gy) = sampler.grad(q);
                // d(residual)/d(p_k), by the chain rule through the sampling
                // position. The leading minus is the residual's own sign: the
                // model enters through the sampled value, which is subtracted.
                for (d, mk) in dk.iter_mut().zip(dm) {
                    let v = mk.apply(p);
                    *d = -(gx * v.x + gy * v.y);
                }
                for a in 0..n {
                    part.gradient[a] -= diff * dk[a];
                    for b in a..n {
                        part.hessian[a * MAX_DOF + b] += dk[a] * dk[b];
                    }
                }
            }
        }
        part
    });

    // In piece order, on this thread.
    let mut sums = Sums {
        sum_sq: 0.0,
        counted: 0,
        gradient: vec![0.0f64; n],
        hessian: vec![0.0f64; n * n],
    };
    for part in &partials {
        sums.sum_sq += part.sum_sq;
        sums.counted += part.counted;
        if want_derivatives {
            for a in 0..n {
                sums.gradient[a] += part.gradient[a];
                for b in a..n {
                    sums.hessian[a * n + b] += part.hessian[a * MAX_DOF + b];
                }
            }
        }
    }
    sums
}

/// Turn the sums into the criterion, gradient and full curvature.
///
/// `None` when no pixel was counted.
pub(crate) fn finish(
    model: Transformation,
    fixed: &[Point],
    landmarks: &[Point],
    sums: Sums,
    want_derivatives: bool,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let Sums {
        sum_sq,
        counted,
        gradient,
        mut hessian,
    } = sums;
    if counted == 0 {
        return None;
    }
    let n = gradient.len();
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

/// Evaluate the criterion for a candidate landmark set, and optionally the
/// gradient and curvature with it, on the CPU.
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
    parallel: bool,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let prep = prepare(model, fixed, landmarks, want_derivatives)?;
    let sums = accumulate(&prep, reference, rw, rh, sampler, parallel);
    finish(model, fixed, landmarks, sums, want_derivatives)
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

/// How many evaluations a level at this octave may spend.
fn iteration_cap(octave: usize) -> usize {
    MANY_ITERATIONS * (1usize << octave)
}

/// What a level fit wants scored next.
pub(crate) struct Request {
    pub landmarks: Vec<Point>,
    pub derivatives: bool,
}

/// Where the search is in a level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// The starting point has not been scored yet.
    Init,
    /// Damped steps, accepted or rejected one at a time.
    Iterate,
    /// The single full-curvature step after the loop.
    Newton,
    Done,
}

/// One level of the search, as a state machine.
///
/// The search is a loop — score, step, score, accept or reject — and on the
/// CPU it is naturally written as one. It is written as a state machine instead
/// because the GPU path has to run *many* such loops in lockstep: one dispatch
/// scores the next candidate of every frame in a batch at once, and a round
/// trip to the device is expensive enough that doing it once per frame per
/// step would cost more than the arithmetic it saves.
///
/// So the loop is turned inside out. [`request`](Self::request) says what to
/// score next and [`absorb`](Self::absorb) takes the result; whoever drives it
/// decides where the scoring happens. Both paths drive this one type, so the
/// accept/reject logic, the damping schedule and the stopping rule have a single
/// definition — a GPU frame and a CPU frame cannot follow different rules.
pub(crate) struct LevelFit {
    n: usize,
    cap: usize,
    phase: Phase,
    best: Vec<Point>,
    best_criterion: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
    lambda: f64,
    iteration: usize,
    /// The candidate handed out by the last `request`, with how far its step
    /// moved the landmarks, awaiting its score.
    pending: Option<(Vec<Point>, f64)>,
    evaluations: usize,
}

impl LevelFit {
    pub fn new(model: Transformation, start: &[Point], octave: usize) -> Self {
        LevelFit {
            n: model.dof(),
            cap: iteration_cap(octave),
            phase: Phase::Init,
            best: start.to_vec(),
            best_criterion: f64::INFINITY,
            gradient: Vec::new(),
            hessian: Vec::new(),
            lambda: 1.0,
            iteration: 1,
            pending: None,
            evaluations: 0,
        }
    }

    /// The next candidate to score, or `None` when the level is finished.
    ///
    /// Every `Some` must be answered by exactly one [`absorb`](Self::absorb)
    /// before the next call.
    pub fn request(&mut self) -> Option<Request> {
        debug_assert!(self.pending.is_none(), "request() twice without absorb()");
        match self.phase {
            Phase::Init => {
                self.pending = Some((self.best.clone(), 0.0));
                Some(Request {
                    landmarks: self.best.clone(),
                    derivatives: true,
                })
            }
            Phase::Iterate => {
                // The damped step, diagonal by construction. See the module
                // header: this is the reference's behaviour, not a shortcut.
                let n = self.n;
                let mut step = vec![0.0f64; n];
                for (k, s) in step.iter_mut().enumerate() {
                    let d = (1.0 + self.lambda) * self.hessian[k * n + k];
                    *s = if d.abs() < 1e-12 {
                        0.0
                    } else {
                        self.gradient[k] / d
                    };
                }
                let moved = displacement(&step);
                let candidate = stepped(&self.best, &step);
                self.pending = Some((candidate.clone(), moved));
                Some(Request {
                    landmarks: candidate,
                    derivatives: true,
                })
            }
            Phase::Newton => {
                // One Gauss-Newton step with the full curvature, taken only if
                // it helps. This is where the subpixel accuracy comes from: the
                // damped steps approach along the gradient, and this one jumps
                // to the quadratic model's minimum.
                let Some(step) = solve_full(&self.hessian, &self.gradient, self.n) else {
                    self.phase = Phase::Done;
                    return None;
                };
                let candidate = stepped(&self.best, &step);
                self.pending = Some((candidate.clone(), 0.0));
                Some(Request {
                    landmarks: candidate,
                    derivatives: false,
                })
            }
            Phase::Done => None,
        }
    }

    /// The score of the last requested candidate: `None` when it could not be
    /// scored, which for a candidate means its landmarks were degenerate.
    pub fn absorb(&mut self, scored: Option<(f64, Vec<f64>, Vec<f64>)>) {
        let Some((candidate, moved)) = self.pending.take() else {
            debug_assert!(false, "absorb() without a request()");
            return;
        };
        match self.phase {
            Phase::Init => match scored {
                Some((c, g, h)) => {
                    self.evaluations += 1;
                    self.best_criterion = c;
                    self.gradient = g;
                    self.hessian = h;
                    self.phase = Phase::Iterate;
                }
                // Nothing to step from: the level ends where it started, with
                // no score.
                None => self.phase = Phase::Done,
            },
            Phase::Iterate => {
                match scored {
                    Some((c, g, h)) => {
                        self.evaluations += 1;
                        if c < self.best_criterion {
                            self.best_criterion = c;
                            self.best = candidate;
                            self.gradient = g;
                            self.hessian = h;
                            self.lambda /= 4.0;
                        } else {
                            self.lambda *= 4.0;
                        }
                    }
                    None => self.lambda *= 4.0,
                }
                self.iteration += 1;
                if self.iteration >= self.cap || moved < PIXEL_PRECISION {
                    self.phase = Phase::Newton;
                }
            }
            Phase::Newton => {
                if let Some((c, _, _)) = scored {
                    self.evaluations += 1;
                    if c < self.best_criterion {
                        self.best = candidate;
                        // Kept in step with `best`, so the criterion returned
                        // is the one belonging to the landmarks returned.
                        self.best_criterion = c;
                    }
                }
                self.phase = Phase::Done;
            }
            Phase::Done => {}
        }
    }

    /// Where the level got to, the criterion it scored there, and how many
    /// evaluations it spent.
    pub fn outcome(self) -> (Vec<Point>, f64, usize) {
        (self.best, self.best_criterion, self.evaluations)
    }
}

/// Fit one level on the CPU, starting from `landmarks`, and return where it got
/// to with the criterion it scored there and the evaluations it spent.
///
/// The criterion comes back because at full resolution it *is* the number a
/// caller reports, and this function has just computed it. Recomputing it
/// afterwards meant building the full-resolution sampler a second time —
/// gradient images and four prefilters — for every frame, to arrive at a
/// value already in hand.
fn fit_level(
    model: Transformation,
    reference: &Level,
    moving: &Level,
    fixed: &[Point],
    landmarks: &[Point],
    parallel: bool,
) -> (Vec<Point>, f64, usize) {
    let sampler = Sampler::of(moving, parallel);
    let mut fit = LevelFit::new(model, landmarks, reference.octave);
    while let Some(req) = fit.request() {
        let scored = evaluate(
            model,
            &reference.img,
            reference.w,
            reference.h,
            &sampler,
            fixed,
            &req.landmarks,
            req.derivatives,
            parallel,
        );
        fit.absorb(scored);
    }
    fit.outcome()
}

/// Register `moving` onto `reference`, both `w * h`.
///
/// Returns the transformation mapping reference coordinates to moving ones, or
/// `None` if the images are empty or the fit degenerated.
///
/// With [`Backend::Gpu`] this is a batch of one, so every round trip to the
/// device serves a single frame; prefer [`register_each`] for several frames
/// against one reference.
///
/// [`Backend::Gpu`]: crate::Backend::Gpu
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
    #[cfg(feature = "gpu")]
    {
        if settings.backend == crate::Backend::Gpu {
            return register_batch_gpu(&[(reference, moving)], w, h, settings)
                .and_then(|mut fits| fits.pop().flatten());
        }
    }
    register_pair_cpu(reference, moving, w, h, settings)
}

/// [`register_pair`] on the CPU.
fn register_pair_cpu(
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
    // Inside the frame, every mode can use every core: this is the level of
    // parallelism that a chained reference still has. See `Backend::threaded`.
    let parallel = settings.backend.threaded();
    let depth = pyramid::depth_for(w, h, settings.max_pyramid_depth);
    let ref_levels = pyramid::build(reference, w, h, depth);
    let mov_levels = pyramid::build(moving, w, h, depth);

    // Landmarks are placed on the full-resolution image and scaled down to
    // whichever level is being worked on; `fixed` is where they sit in the
    // reference, `moving_at` where the search has pushed their images.
    let full = transform::initial_landmarks(model, w, h);
    let mut moving_at = full.clone();

    let mut evaluations = 0usize;
    let mut criterion = f64::INFINITY;
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
        let (got, scored, spent) = fit_level(
            model,
            &ref_levels[octave],
            &mov_levels[octave],
            &fixed,
            &start,
            parallel,
        );
        evaluations += spent;
        // Full resolution is the last level, and its score is the one that
        // describes the answer returned. At scale 1 the landmarks pass through
        // the scaling below unchanged, so this is the criterion *of*
        // `moving_at`, not of some nearby candidate.
        if octave == 0 {
            criterion = scored;
        }
        // Back to full-resolution coordinates for the next level down.
        moving_at = got
            .iter()
            .map(|p| Point::new(p.x / scale, p.y / scale))
            .collect();
    }

    let matrix = transform::matrix_from_landmarks(model, &full, &moving_at)?;
    Some(Fit {
        matrix,
        criterion,
        evaluations,
    })
}

/// Register every frame of a stack against one reference.
///
/// `frames` are `w * h` each. `reference` is the image every frame is matched
/// against; building it is the caller's business (see [`crate::register_stack`]).
///
/// A [`register_pairs`] in which every pair shares its reference.
pub fn register_each(
    frames: &[Vec<f32>],
    reference: &[f32],
    w: usize,
    h: usize,
    settings: &Settings,
    parallel: bool,
) -> Vec<Option<Fit>> {
    let pairs: Vec<(&[f32], &[f32])> = frames.iter().map(|f| (reference, &f[..])).collect();
    register_pairs(&pairs, w, h, settings, parallel)
}

/// Register each `(reference, moving)` pair, all `w * h`.
///
/// The pairs are independent: no fit depends on another's result. That holds
/// for the previous-frame reference too — frame `t` is fitted against the raw
/// frame `t - 1`, not against frame `t - 1` registered — so the expensive part
/// of a chained registration runs like any other; only composing the answers
/// afterwards is sequential, and that is a handful of 3x3 products.
///
/// `parallel` runs separate pairs at the same time on the CPU threads. Each
/// pair's own work is parallel or not according to `settings.backend`
/// independently of this — the two levels nest, and rayon's work-stealing keeps
/// the nesting from oversubscribing the cores. With [`Backend::Gpu`] the pairs
/// go to the device as lanes of a batch instead.
///
/// [`Backend::Gpu`]: crate::Backend::Gpu
pub fn register_pairs(
    pairs: &[(&[f32], &[f32])],
    w: usize,
    h: usize,
    settings: &Settings,
    parallel: bool,
) -> Vec<Option<Fit>> {
    #[cfg(feature = "gpu")]
    {
        if settings.backend == crate::Backend::Gpu {
            // As many pairs per batch as the device's memory takes. A batch the
            // device failed on is reported as failed frames, never quietly
            // refitted on the CPU.
            let per = crate::gpu::lanes_for(w, h).max(1);
            let mut out = Vec::with_capacity(pairs.len());
            for chunk in pairs.chunks(per) {
                match register_batch_gpu(chunk, w, h, settings) {
                    Some(fits) => out.extend(fits),
                    None => out.extend(chunk.iter().map(|_| None)),
                }
            }
            return out;
        }
    }
    map_in_order(pairs, parallel, |&(r, m)| {
        register_pair_cpu(r, m, w, h, settings)
    })
}

/// Register a batch of pairs, the expensive levels on the device.
///
/// Every pair of the batch is a *lane*. Level by level, coarsest first, each
/// lane runs the same [`LevelFit`] search the CPU runs — but in lockstep: every
/// round, each lane still searching hands over its next candidate, one dispatch
/// scores all of them, and each lane takes its own score back. A round trip to
/// the device is long next to one candidate's arithmetic, so it is shared by
/// the whole batch rather than paid by each frame.
///
/// Levels smaller than [`crate::gpu::MIN_PIXELS`] are fitted on the CPU, lanes
/// side by side on the threads: there the search takes most of its steps on
/// the fewest pixels, and a round trip would cost more than it saved.
///
/// `None` if the device failed partway; the caller reports those frames as not
/// fitted.
#[cfg(feature = "gpu")]
fn register_batch_gpu(
    pairs: &[(&[f32], &[f32])],
    w: usize,
    h: usize,
    settings: &Settings,
) -> Option<Vec<Option<Fit>>> {
    let model = settings.transformation;
    let threaded = settings.backend.threaded();
    let depth = pyramid::depth_for(w, h, settings.max_pyramid_depth);
    let lanes = pairs.len();
    // Each lane's two pyramids. A shared reference is built once per lane,
    // which is a few reductions wasted against seconds of search; keeping the
    // lanes uniform is what lets one driver serve every reference mode.
    let levels: Vec<(Vec<Level>, Vec<Level>)> = map_in_order(pairs, threaded, |&(r, m)| {
        (pyramid::build(r, w, h, depth), pyramid::build(m, w, h, depth))
    });
    let full = transform::initial_landmarks(model, w, h);
    let mut moving_at = vec![full.clone(); lanes];
    let mut evaluations = vec![0usize; lanes];
    let mut criterion = vec![f64::INFINITY; lanes];
    let lane_ids: Vec<usize> = (0..lanes).collect();

    for octave in (0..depth).rev() {
        let scale = 1.0 / (1u32 << octave) as f64;
        let fixed: Vec<Point> = full
            .iter()
            .map(|p| Point::new(p.x * scale, p.y * scale))
            .collect();
        let starts: Vec<Vec<Point>> = moving_at
            .iter()
            .map(|at| at.iter().map(|p| Point::new(p.x * scale, p.y * scale)).collect())
            .collect();
        // Every lane's level has the same size: the frames are all `w * h`.
        let (lw, lh) = (levels[0].0[octave].w, levels[0].0[octave].h);
        let outcomes: Vec<(Vec<Point>, f64, usize)> = if lw * lh < crate::gpu::MIN_PIXELS {
            map_in_order(&lane_ids, threaded, |&l| {
                let (refs, movs) = &levels[l];
                fit_level(model, &refs[octave], &movs[octave], &fixed, &starts[l], threaded)
            })
        } else {
            let samplers: Vec<Sampler> = map_in_order(&lane_ids, threaded, |&l| {
                Sampler::of(&levels[l].1[octave], threaded)
            });
            let references: Vec<&Level> = levels.iter().map(|(r, _)| &r[octave]).collect();
            fit_level_gpu(model, &references, &samplers, &fixed, &starts)?
        };
        for (l, (got, scored, spent)) in outcomes.into_iter().enumerate() {
            evaluations[l] += spent;
            if octave == 0 {
                criterion[l] = scored;
            }
            moving_at[l] = got
                .iter()
                .map(|p| Point::new(p.x / scale, p.y / scale))
                .collect();
        }
    }

    Some(
        (0..lanes)
            .map(|l| {
                let matrix = transform::matrix_from_landmarks(model, &full, &moving_at[l])?;
                Some(Fit {
                    matrix,
                    criterion: criterion[l],
                    evaluations: evaluations[l],
                })
            })
            .collect(),
    )
}

/// One level of a batch, every lane's candidates scored on the device.
#[cfg(feature = "gpu")]
fn fit_level_gpu(
    model: Transformation,
    references: &[&Level],
    samplers: &[Sampler],
    fixed: &[Point],
    starts: &[Vec<Point>],
) -> Option<Vec<(Vec<Point>, f64, usize)>> {
    let (w, h, octave) = (references[0].w, references[0].h, references[0].octave);
    let planes: Vec<[&[f32]; 4]> = references
        .iter()
        .zip(samplers)
        .map(|(r, s)| [&r.img[..], &s.img[..], &s.gx[..], &s.gy[..]])
        .collect();
    let level = crate::gpu::GpuLevel::new(model.dof(), w, h, &planes)?;
    let mut fits: Vec<LevelFit> = starts
        .iter()
        .map(|s| LevelFit::new(model, s, octave))
        .collect();
    loop {
        // Each lane's next candidate. A candidate whose landmarks do not make a
        // transformation is answered on the spot, as the CPU path answers it,
        // and that lane asked again.
        let mut round: Vec<(usize, Prepared, Vec<Point>)> = Vec::new();
        for (l, fit) in fits.iter_mut().enumerate() {
            while let Some(req) = fit.request() {
                match prepare(model, fixed, &req.landmarks, req.derivatives) {
                    Some(prep) => {
                        round.push((l, prep, req.landmarks));
                        break;
                    }
                    None => fit.absorb(None),
                }
            }
        }
        if round.is_empty() {
            break;
        }
        let batch: Vec<(usize, &Prepared)> = round.iter().map(|(l, p, _)| (*l, p)).collect();
        let sums = level.evaluate(&batch)?;
        for ((l, prep, landmarks), s) in round.iter().zip(sums) {
            fits[*l].absorb(finish(model, fixed, landmarks, s, prep.derivatives));
        }
    }
    Some(fits.into_iter().map(LevelFit::outcome).collect())
}

#[cfg(test)]
#[path = "optimize_tests.rs"]
mod tests;

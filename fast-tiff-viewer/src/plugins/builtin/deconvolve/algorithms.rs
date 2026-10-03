//! The nine ways of undoing a convolution.
//!
//! Every one of them is after the same thing: an estimate `x` whose blur by
//! the PSF matches what the microscope recorded. They differ in what else they
//! insist on, and that is the whole reason there are nine rather than one.
//!
//! * **Spectral**, one shot: divide by the PSF's spectrum, having first done
//!   something about the frequencies where that spectrum is ~0 and dividing
//!   would amplify nothing but noise. [`Method::NaiveInverse`] does nothing
//!   about them and is here to show why the others exist;
//!   [`Method::Wiener`] adds a constant; [`Method::RegularisedInverse`] adds a
//!   penalty that grows with frequency.
//! * **Iterative**, stepping towards a fit: [`Method::Landweber`] and
//!   [`Method::VanCittert`] walk downhill on the squared error,
//!   [`Method::TikhonovMiller`] on the squared error plus a smoothness
//!   penalty, [`Method::Mrnsd`] downhill in a way that cannot go negative, and
//!   [`Method::RichardsonLucy`] — the one most people mean by "deconvolution"
//!   — maximises the likelihood of a *Poisson* measurement, which is what
//!   photon counting actually is.
//!
//! # The two things that matter more than the choice
//!
//! A deconvolution is wrong in a way no amount of iterating fixes if the PSF
//! does not match the optics, or if the edges were not handled — see
//! [`super::grid`]. Between methods, the honest summary is that
//! Richardson-Lucy with a reasonable iteration count is the default for
//! fluorescence for good reason, Wiener is instant and good enough to look at,
//! and the rest are here because ImageJ's deconvolution plugins offer them and
//! a result is only reproducible against the method that produced it.
//!
//! # Provenance
//!
//! The set is the union of what the ImageJ deconvolution plugins expose:
//! DeconvolutionLab2 (naive/regularised inverse, Landweber, Van Cittert,
//! Tikhonov-Miller, Richardson-Lucy, RLTV), Parallel Iterative Deconvolution
//! (MRNSD), Parallel Spectral Deconvolution (Tikhonov) and Bob Dougherty's
//! Iterative Deconvolution 3D (the Wiener-filter damping, the low-pass and the
//! early-stop criterion, which here are options on the iterative methods
//! rather than a method of their own).

use super::fft::Dims;
use super::grid::Operator;
use super::par;
use rustfft::num_complex::Complex32;

/// Which algorithm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Method {
    RichardsonLucy,
    RichardsonLucyTv,
    Wiener,
    RegularisedInverse,
    NaiveInverse,
    Landweber,
    VanCittert,
    TikhonovMiller,
    Mrnsd,
}

impl Method {
    /// In dialog order: the ones worth reaching for first, first.
    pub(crate) const ALL: [Method; 9] = [
        Method::RichardsonLucy,
        Method::RichardsonLucyTv,
        Method::Wiener,
        Method::RegularisedInverse,
        Method::NaiveInverse,
        Method::Landweber,
        Method::VanCittert,
        Method::TikhonovMiller,
        Method::Mrnsd,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Method::RichardsonLucy => "Richardson-Lucy",
            Method::RichardsonLucyTv => "Richardson-Lucy + total variation",
            Method::Wiener => "Wiener filter",
            Method::RegularisedInverse => "Regularized inverse (Tikhonov)",
            Method::NaiveInverse => "Naive inverse filter",
            Method::Landweber => "Landweber",
            Method::VanCittert => "Van Cittert",
            Method::TikhonovMiller => "Tikhonov-Miller (ICTM)",
            Method::Mrnsd => "MRNSD (non-negative least squares)",
        }
    }

    /// A short name for the result's window title.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Method::RichardsonLucy => "rl",
            Method::RichardsonLucyTv => "rltv",
            Method::Wiener => "wiener",
            Method::RegularisedInverse => "rif",
            Method::NaiveInverse => "inverse",
            Method::Landweber => "landweber",
            Method::VanCittert => "vancittert",
            Method::TikhonovMiller => "tm",
            Method::Mrnsd => "mrnsd",
        }
    }

    /// Whether this one loops, and so reads `iterations` and can be stopped
    /// early.
    pub(crate) fn is_iterative(self) -> bool {
        !matches!(
            self,
            Method::Wiener | Method::RegularisedInverse | Method::NaiveInverse
        )
    }
}

/// Everything the dialog decided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Settings {
    pub method: Method,
    pub iterations: usize,
    /// Tikhonov weight, and the TV weight for RLTV.
    pub lambda: f32,
    /// Wiener noise-to-signal ratio — Dougherty's "Wiener filter gamma".
    pub gamma: f32,
    /// Relaxation factor for the methods that take a step.
    pub step: f32,
    /// Below this fraction of the PSF spectrum's peak, the naive inverse gives
    /// up rather than dividing.
    pub threshold: f32,
    /// Gaussian low-pass applied to the estimate, as sigma in pixels. 0 is off.
    pub low_pass: f32,
    /// Stop when an iteration changes the estimate by less than this percent.
    pub stop_delta: f32,
    pub nonneg: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            method: Method::RichardsonLucy,
            iterations: 10,
            lambda: 0.01,
            gamma: 0.001,
            step: 1.0,
            threshold: 0.001,
            low_pass: 0.0,
            stop_delta: 0.01,
            nonneg: true,
        }
    }
}

/// Why a run stopped, for the log.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stopped {
    /// Ran every iteration asked for.
    Completed(usize),
    /// The estimate stopped moving.
    Converged { iterations: usize, delta: f32 },
    /// The estimate blew up. Carries the iteration it happened on.
    Diverged(usize),
}

/// The outcome of a run: the estimate, and how it ended.
pub(crate) struct Deconvolved {
    pub image: Vec<f32>,
    pub stopped: Stopped,
}

/// Deconvolve `data` on the operator's grid.
///
/// `data` is the padded observation; the result is on the same grid and the
/// caller crops it. `progress` is called once per iteration and returning
/// `false` from it cancels, which this reports as `None` — not an error, and
/// not a partial result either, because a half-converged estimate presented as
/// a finished one is worse than nothing.
pub(crate) fn run(
    op: &mut Operator,
    data: &[f32],
    s: &Settings,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Deconvolved> {
    let n = op.dims().len();
    debug_assert_eq!(data.len(), n);

    // One rule for which family a method belongs to, read here and nowhere
    // else. A wildcard arm below would route a newly added method to whichever
    // branch it fell through to, which is how a method ends up quietly not
    // running the algorithm it is named after.
    if s.method.is_iterative() {
        return iterate(op, data, s, progress);
    }
    let mut out = match s.method {
        Method::NaiveInverse => {
            let thr = s.threshold.max(0.0);
            let peak = op
                .spectrum()
                .iter()
                .map(|h| h.norm_sqr())
                .fold(0.0f32, f32::max)
                .sqrt();
            let cut = (thr * peak).max(f32::MIN_POSITIVE);
            let mut out = vec![0.0f32; n];
            op.through_spectrum(data, &mut out, |spec, h| {
                par::zip(spec, h, |y, hv| {
                    // Below the cut the PSF has passed nothing, so the
                    // recorded value there is noise and dividing by ~0 would
                    // return it amplified a thousandfold. Dropping the band is
                    // what makes this usable at all; it is still the worst of
                    // the three spectral filters, which is the lesson.
                    *y = if hv.norm() <= cut {
                        Complex32::new(0.0, 0.0)
                    } else {
                        *y / *hv
                    };
                });
            });
            out
        }
        Method::Wiener => {
            let g = s.gamma.max(0.0);
            let mut out = vec![0.0f32; n];
            op.through_spectrum(data, &mut out, |spec, h| {
                par::zip(spec, h, |y, hv| {
                    *y = *y * hv.conj() / (hv.norm_sqr() + g);
                });
            });
            out
        }
        Method::RegularisedInverse => {
            let lap = laplacian_power(op);
            let lam = s.lambda.max(0.0);
            let mut out = vec![0.0f32; n];
            op.through_spectrum(data, &mut out, |spec, h| {
                par::zip3(spec, h, &lap, |y, hv, l| {
                    // The difference from Wiener in one line: the penalty
                    // grows with frequency instead of being flat, so smooth
                    // detail is restored and the high-frequency noise the
                    // inverse would amplify is held down.
                    *y = *y * hv.conj() / (hv.norm_sqr() + lam * *l);
                });
            });
            out
        }
        other => unreachable!("{other:?} is iterative and was handled above"),
    };

    if s.low_pass > 0.0 {
        let g = gaussian_response(op.dims(), s.low_pass);
        let mut smoothed = vec![0.0f32; n];
        op.through_spectrum(&out, &mut smoothed, |spec, _| {
            par::zip(spec, &g, |v, k| *v *= *k);
        });
        out = smoothed;
    }
    if s.nonneg {
        par::each(&mut out, |v| *v = v.max(0.0));
    }
    Some(Deconvolved {
        image: out,
        stopped: Stopped::Completed(0),
    })
}

/// The loop the six iterative methods share.
fn iterate(
    op: &mut Operator,
    data: &[f32],
    s: &Settings,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Deconvolved> {
    let n = op.dims().len();
    let dims = op.dims();

    // A floor under the starting estimate. Richardson-Lucy and MRNSD are
    // multiplicative: a voxel that starts at exactly zero is multiplied by
    // something and stays zero for every iteration that follows, so a
    // background-subtracted stack would come back with its background frozen
    // black and nothing able to move into it. The floor is six orders below
    // the mean, which is far under the noise and cannot be seen.
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64;
    let floor = (mean.abs() * 1e-6) as f32;
    let positive = matches!(s.method, Method::RichardsonLucy | Method::RichardsonLucyTv)
        || s.method == Method::Mrnsd;

    let mut est: Vec<f32> = data
        .iter()
        .map(|&v| if positive { v.max(floor) } else { v })
        .collect();

    let mut blurred = vec![0.0f32; n];
    let mut work = vec![0.0f32; n];
    let mut update = vec![0.0f32; n];
    // Only RLTV and Tikhonov-Miller need these, and they are a padded grid
    // each, so they are not allocated for the methods that do not.
    let mut tv = if s.method == Method::RichardsonLucyTv {
        Some(TotalVariation::new(n))
    } else {
        None
    };
    let lap = (s.method == Method::TikhonovMiller).then(|| laplacian_power(op));
    let smooth = (s.low_pass > 0.0).then(|| gaussian_response(dims, s.low_pass));

    let iterations = s.iterations.max(1);
    let mut stopped = Stopped::Completed(iterations);
    // The smallest residual seen so far. A method that is working drives this
    // down; one that is coming apart drives it up, and it does so long before
    // any number overflows. Each branch below fills `residual` from a
    // difference it had to compute anyway, so detecting this is free.
    let mut best_residual = f32::INFINITY;
    let data_rms = rms(data);
    for it in 0..iterations {
        if !progress(it as f32 / iterations as f32) {
            return None;
        }
        let previous_rms = rms(&est);
        // Declared without a value so that the compiler, not a reviewer,
        // checks that every branch below computes one. A default here would
        // make "this method forgot to measure its residual" indistinguishable
        // from "this method is converging".
        let residual: f32;

        match s.method {
            Method::RichardsonLucy | Method::RichardsonLucyTv => {
                op.blur(&est, &mut blurred);
                residual = rms_difference(data, &blurred);
                // work := data / blur. The division is the whole method: it
                // asks "by what factor is the model short of what we saw?".
                par::zip3(&mut work, data, &blurred, |w, d, b| {
                    *w = if *b > floor { *d / *b } else { 0.0 };
                });
                op.correlate(&work, &mut update);
                match &mut tv {
                    None => {
                        par::zip(&mut est, &update, |e, u| *e *= *u);
                    }
                    Some(tv) => {
                        // Dey et al. 2006: the correction is divided by
                        // `1 - lambda * div(grad u / |grad u|)`, which damps
                        // the update where the estimate is already flat and
                        // leaves it alone across an edge. That is what stops
                        // RL from turning read noise into a field of dots
                        // while keeping it able to sharpen a membrane.
                        tv.divergence(&est, dims);
                        par::zip3(&mut est, &update, &tv.div, |e, &u, &d| {
                            // Clamped, and the bounds are not arbitrary. The
                            // derivation assumes `lambda * div` stays well
                            // under 1; at a weight the dialog permits it does
                            // not, and an unclamped denominator then passes
                            // through zero and changes sign — multiplying the
                            // estimate by something enormous and negative,
                            // which no later iteration recovers from. Bounded
                            // to a factor of two either way, a weight that is
                            // too high merely stops helping, and Richardson-
                            // Lucy's own feedback holds the result together.
                            let damp = (1.0 - s.lambda * d).clamp(0.5, 2.0);
                            *e *= u / damp;
                        });
                    }
                }
            }
            Method::Landweber | Method::VanCittert | Method::TikhonovMiller => {
                op.blur(&est, &mut blurred);
                par::zip3(&mut work, data, &blurred, |w, d, b| *w = *d - *b);
                residual = rms(&work);
                if s.method == Method::VanCittert {
                    // Van Cittert pushes the residual straight back without
                    // passing it through the adjoint. That is the whole
                    // difference from Landweber, and the reason it is the
                    // faster and the less stable of the two.
                    update.copy_from_slice(&work);
                } else {
                    op.correlate(&work, &mut update);
                }
                if let Some(lap) = &lap {
                    // Tikhonov-Miller: the same step, minus the gradient of a
                    // smoothness penalty.
                    let lam = s.lambda;
                    op.through_spectrum(&est, &mut blurred, |spec, _| {
                        par::zip(spec, lap, |v, l| *v *= lam * *l);
                    });
                    par::zip(&mut update, &blurred, |u, r| *u -= *r);
                }
                let step = s.step;
                par::zip(&mut est, &update, |e, u| *e += step * *u);
            }
            Method::Mrnsd => {
                // Nagy and Strakos' modified residual-norm steepest descent.
                // The scaling by the estimate is what keeps it non-negative:
                // the search direction vanishes wherever the estimate does, so
                // a voxel can approach zero but never cross it, and the step
                // is cut short if it would.
                op.blur(&est, &mut blurred);
                par::zip3(&mut work, data, &blurred, |w, d, b| *w = *b - *d);
                residual = rms(&work);
                op.correlate(&work, &mut update); // the gradient
                                                  // One pass, not two: the scaling factor is measured from the
                                                  // same values the search direction is built from.
                let gamma = par::zip_sum(&mut update, &est, |u, &e| {
                    let g = (e * *u * *u) as f64;
                    *u *= -e; // the search direction
                    g
                });
                op.blur(&update, &mut blurred);
                let denom: f64 = par::sum_by(&blurred, |&v| (v * v) as f64);
                if denom <= 0.0 || gamma <= 0.0 {
                    stopped = Stopped::Converged {
                        iterations: it,
                        delta: 0.0,
                    };
                    break;
                }
                let alpha = par::zip_min(&update, &est, (gamma / denom) as f32, |&d, &e| {
                    if d < 0.0 {
                        -e / d
                    } else {
                        f32::INFINITY
                    }
                });
                par::zip(&mut est, &update, |e, d| *e += alpha * *d);
            }
            // The spectral three never reach here.
            Method::Wiener | Method::RegularisedInverse | Method::NaiveInverse => unreachable!(),
        }

        if s.nonneg {
            par::each(&mut est, |v| *v = v.max(0.0));
        }
        if let Some(g) = &smooth {
            op.through_spectrum(&est, &mut work, |spec, _| {
                par::zip(spec, g, |v, k| *v *= *k);
            });
            est.copy_from_slice(&work);
        }

        // Dougherty's exits, which matter more the longer a run is: an
        // estimate that has stopped moving is finished whatever the iteration
        // count said, and one that has blown up will not recover.
        //
        // Divergence is caught by the residual rather than by waiting for a
        // number to overflow. Van Cittert at a step of 2 is the case that
        // taught this: it produces values a thousand times the data's and
        // every one of them is finite, so an overflow check calls it a result
        // and hands back noise shaped like an image. The factor is wide
        // enough that a regularised method — which trades data fit for
        // smoothness on purpose, and so is *expected* to let the residual
        // rise — cannot trip it.
        // Two checks, because they catch different failures and neither
        // catches the other's.
        //
        // The residual catches a method walking away from the data. It cannot
        // catch Van Cittert at a step of 2, which is the instructive case:
        // the modes that run away are the ones the PSF passes nothing of, so
        // they are invisible to `H x` and the residual stays respectable
        // while the estimate fills with alternating values a thousand times
        // the data's. That one is caught by the magnitude instead.
        //
        // The magnitude bound is loose on purpose — concentrating a smeared
        // point back into one voxel genuinely raises the RMS, by roughly the
        // square root of the number of voxels it was spread over — and a
        // hundredfold is far past anything that is still a picture.
        const RESIDUAL_GROWTH: f32 = 10.0;
        const MAGNITUDE: f32 = 100.0;
        let now = rms(&est);
        if residual.is_finite() {
            best_residual = best_residual.min(residual);
        }
        let blew_up = !now.is_finite()
            || !residual.is_finite()
            || (best_residual > 0.0 && residual > RESIDUAL_GROWTH * best_residual)
            || (data_rms > 0.0 && now > MAGNITUDE * data_rms);
        if blew_up {
            stopped = Stopped::Diverged(it + 1);
            break;
        }
        let delta = if previous_rms > 0.0 {
            100.0 * (now - previous_rms).abs() / previous_rms
        } else {
            f32::INFINITY
        };
        if s.stop_delta > 0.0 && delta < s.stop_delta {
            stopped = Stopped::Converged {
                iterations: it + 1,
                delta,
            };
            break;
        }
    }

    Some(Deconvolved {
        image: est,
        stopped,
    })
}

/// The root-mean-square of `a - b`, which is the residual every iterative
/// method is trying to make small.
fn rms_difference(a: &[f32], b: &[f32]) -> f32 {
    let s = par::zip_sum_by(a, b, |x, y| ((*x - *y) as f64).powi(2));
    (s / a.len().max(1) as f64).sqrt() as f32
}

fn rms(v: &[f32]) -> f32 {
    let s = par::sum_by(v, |&x| (x as f64) * (x as f64));
    (s / v.len().max(1) as f64).sqrt() as f32
}

/// `|L|^2` for the discrete Laplacian on this grid — the Tikhonov penalty.
///
/// Built as a kernel and transformed rather than written out as a formula, so
/// that a 2-D grid gets the 2-D Laplacian without a second code path: the
/// neighbours along an axis of length one simply are not there.
fn laplacian_power(op: &mut Operator) -> Vec<f32> {
    let dims = op.dims();
    let mut k = vec![0.0f32; dims.len()];
    let mut centre = 0.0f32;
    let mut poke = |x: usize, y: usize, z: usize| {
        k[dims.at(x, y, z)] += 1.0;
        centre -= 1.0;
    };
    if dims.x > 1 {
        poke(1, 0, 0);
        poke(dims.x - 1, 0, 0);
    }
    if dims.y > 1 {
        poke(0, 1, 0);
        poke(0, dims.y - 1, 0);
    }
    if dims.z > 1 {
        poke(0, 0, 1);
        poke(0, 0, dims.z - 1);
    }
    k[0] = centre;
    op.spectrum_of(&k)
        .iter()
        .map(|c| c.norm_sqr())
        .collect::<Vec<f32>>()
}

/// A Gaussian's frequency response on this grid, for the low-pass option.
///
/// Analytic rather than a sampled kernel: a Gaussian truncated to fit the grid
/// has a discontinuity at the cut, and that discontinuity is exactly the
/// ringing the low-pass was added to avoid.
fn gaussian_response(dims: Dims, sigma: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; dims.len()];
    let axis = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                if n <= 1 {
                    return 1.0;
                }
                let k = if i <= n / 2 {
                    i as f32
                } else {
                    i as f32 - n as f32
                };
                let f = k / n as f32;
                (-2.0 * std::f32::consts::PI * std::f32::consts::PI * sigma * sigma * f * f).exp()
            })
            .collect()
    };
    let (gx, gy, gz) = (axis(dims.x), axis(dims.y), axis(dims.z));
    for (z, &wz) in gz.iter().enumerate() {
        for (y, &wy) in gy.iter().enumerate() {
            let row = dims.at(0, y, z);
            for (o, &wx) in out[row..row + dims.x].iter_mut().zip(&gx) {
                *o = wx * wy * wz;
            }
        }
    }
    out
}

/// Scratch for the total-variation term, kept so RLTV does not allocate four
/// padded grids per iteration.
struct TotalVariation {
    px: Vec<f32>,
    py: Vec<f32>,
    pz: Vec<f32>,
    div: Vec<f32>,
}

impl TotalVariation {
    fn new(n: usize) -> Self {
        TotalVariation {
            px: vec![0.0; n],
            py: vec![0.0; n],
            pz: vec![0.0; n],
            div: vec![0.0; n],
        }
    }

    /// `div(grad u / |grad u|)`, by forward differences for the gradient and
    /// backward for the divergence — the pairing that makes the discrete
    /// divergence the adjoint of the discrete gradient, which is what the
    /// derivation assumes.
    fn divergence(&mut self, u: &[f32], d: Dims) {
        // The epsilon under the gradient's norm is not a guard against
        // dividing by zero; it is the gradient below which a region counts as
        // flat. That distinction is the whole behaviour of the term.
        //
        // With a merely numerical epsilon, a flat background has gradients of
        // order 1e-7 from rounding, which normalise to a unit vector pointing
        // in a direction decided by the rounding — so the divergence of that
        // field is large and arbitrary, and the term that was added to
        // suppress noise texture *writes* noise texture into the background.
        // Scaling it to the image puts real edges far above the threshold and
        // rounding far below it.
        let scale = {
            let s: f64 = u.iter().map(|&x| (x as f64) * (x as f64)).sum();
            (s / u.len().max(1) as f64).sqrt() as f32
        };
        let eps = (scale * 1e-3).max(f32::MIN_POSITIVE);
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let i = d.at(x, y, z);
                    let gx = if x + 1 < d.x {
                        u[d.at(x + 1, y, z)] - u[i]
                    } else {
                        0.0
                    };
                    let gy = if y + 1 < d.y {
                        u[d.at(x, y + 1, z)] - u[i]
                    } else {
                        0.0
                    };
                    let gz = if z + 1 < d.z {
                        u[d.at(x, y, z + 1)] - u[i]
                    } else {
                        0.0
                    };
                    let norm = (gx * gx + gy * gy + gz * gz + eps * eps).sqrt();
                    self.px[i] = gx / norm;
                    self.py[i] = gy / norm;
                    self.pz[i] = gz / norm;
                }
            }
        }
        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    let i = d.at(x, y, z);
                    let bx = if x > 0 {
                        self.px[i] - self.px[d.at(x - 1, y, z)]
                    } else {
                        self.px[i]
                    };
                    let by = if y > 0 {
                        self.py[i] - self.py[d.at(x, y - 1, z)]
                    } else {
                        self.py[i]
                    };
                    let bz = if z > 0 {
                        self.pz[i] - self.pz[d.at(x, y, z - 1)]
                    } else {
                        self.pz[i]
                    };
                    self.div[i] = bx + by + bz;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "algorithms_tests.rs"]
mod tests;

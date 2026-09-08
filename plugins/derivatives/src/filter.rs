//! The 3-D Gaussian derivative, matching SciPy's `gaussian_filter`.
//!
//! The analysis this reproduces computes
//!
//! ```text
//! dz = gaussian_filter(stack[start:end], sigma=[s, s, s], order=[1, 0, 0])
//! out = sum(maximum(dz, 0), axis=0)
//! ```
//!
//! — a first-derivative-of-Gaussian along time and a plain Gaussian across the
//! frame, then the positive part summed over the window. Matching SciPy matters
//! more than it might look: these maps are compared against ones the Python
//! produced, so a different kernel, a different truncation radius or a
//! different edge rule changes every pixel by a little and every published
//! figure by a little more.
//!
//! Three details are therefore copied exactly rather than reinvented:
//!
//! * **Radius** is `int(truncate * sigma + 0.5)` with `truncate = 4.0`, which
//!   for `sigma = 2.3` gives 9 — not "however wide the window happens to be".
//! * **The kernel is reversed before correlating.** `gaussian_filter1d` builds
//!   `-x/sigma^2 * phi(x)` and then hands `weights[::-1]` to `correlate1d`,
//!   which for an antisymmetric kernel flips its sign. Miss that and a *rising*
//!   signal differentiates negative, so taking the positive part would map
//!   where the tissue went dark instead of where it responded — a perfectly
//!   plausible-looking image of the wrong thing.
//! * **Reflect** at the edges (`d c b a | a b c d | d c b a`), the SciPy
//!   default, which along a short time window is most of the window.

/// SciPy's default kernel truncation, in standard deviations.
const TRUNCATE: f64 = 4.0;

/// The 1-D kernel radius SciPy uses for this sigma.
pub fn radius(sigma: f64) -> usize {
    (TRUNCATE * sigma + 0.5) as usize
}

/// The weights to correlate with, matching `gaussian_filter1d` exactly.
///
/// SciPy builds `_gaussian_kernel1d` — the normalised Gaussian for order 0, and
/// `-x/sigma^2` times it for order 1, which is the one term its polynomial
/// recurrence produces for a single derivative — and then passes it to
/// `correlate1d` **reversed**. The reversal is a no-op for the symmetric order-0
/// kernel and a sign flip for the antisymmetric order-1 one, which is exactly
/// the difference between measuring a rise and measuring a fall.
pub fn kernel1d(sigma: f64, order: u32) -> Vec<f64> {
    let r = radius(sigma) as isize;
    let s2 = sigma * sigma;
    let mut phi: Vec<f64> = (-r..=r)
        .map(|x| (-0.5 / s2 * (x * x) as f64).exp())
        .collect();
    let sum: f64 = phi.iter().sum();
    for v in &mut phi {
        *v /= sum;
    }
    if order == 0 {
        // Symmetric, so reversing it would change nothing.
        return phi;
    }
    let mut d: Vec<f64> = (-r..=r)
        .zip(phi)
        .map(|(x, p)| -(x as f64) / s2 * p)
        .collect();
    // `weights[::-1]`, as `gaussian_filter1d` does.
    d.reverse();
    d
}

/// `reflect` indexing: `d c b a | a b c d | d c b a`.
///
/// Written as a loop rather than a modulus because a window can be shorter than
/// the kernel — a 3-frame response with a radius-9 kernel is the normal case
/// here — and the reflection then has to bounce more than once.
fn reflect(mut i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// A stack of `frames` planes of `w * h`, filtered and reduced the way the
/// analysis does it: derivative along time, Gaussian across the frame, positive
/// part summed over time.
///
/// Separable, so the three passes are three 1-D correlations rather than one
/// 3-D one — the same result, and the difference between a kernel of 19 taps
/// and one of 19³.
pub fn positive_derivative_sum(
    planes: &[Vec<f32>],
    width: usize,
    height: usize,
    sigma: f64,
) -> Vec<f32> {
    let n = planes.len();
    let px = width * height;
    let dt = kernel1d(sigma, 1);
    let g = kernel1d(sigma, 0);
    let r = radius(sigma) as isize;

    // Pass 1: derivative along time. Each output frame is a weighted sum of
    // input frames, so this is where the stack collapses from `n` planes of
    // history into `n` planes of rate-of-change.
    let mut dz: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut out = vec![0f32; px];
        for (k, &w) in dt.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let src = &planes[reflect(i as isize + k as isize - r, n as isize)];
            for (o, s) in out.iter_mut().zip(src) {
                *o += (w * *s as f64) as f32;
            }
        }
        dz.push(out);
    }

    // Pass 2 and 3: Gaussian across the frame, rows then columns.
    let mut row = vec![0f32; px];
    for frame in &mut dz {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0f64;
                for (k, &w) in g.iter().enumerate() {
                    let sx = reflect(x as isize + k as isize - r, width as isize);
                    acc += w * frame[y * width + sx] as f64;
                }
                row[y * width + x] = acc as f32;
            }
        }
        for x in 0..width {
            for y in 0..height {
                let mut acc = 0f64;
                for (k, &w) in g.iter().enumerate() {
                    let sy = reflect(y as isize + k as isize - r, height as isize);
                    acc += w * row[sy * width + x] as f64;
                }
                frame[y * width + x] = acc as f32;
            }
        }
    }

    // The reduction: only rises count, and they are summed rather than
    // averaged, so a response spread over more frames reads as larger.
    let mut out = vec![0f32; px];
    for frame in &dz {
        for (o, v) in out.iter_mut().zip(frame) {
            if *v > 0.0 {
                *o += *v;
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "filter_tests.rs"]
mod tests;

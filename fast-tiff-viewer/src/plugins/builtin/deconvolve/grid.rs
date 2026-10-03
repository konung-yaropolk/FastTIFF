//! Turning a finite image into something an FFT can convolve, and the blur
//! operator itself.
//!
//! An FFT convolution is *circular*: what leaves the right edge comes back in
//! at the left. Microscope images are not periodic, so convolving one directly
//! wraps the bright side of a cell onto the dark side opposite and leaves a
//! bright rim there — the single most recognisable artefact of a deconvolution
//! done without thinking about edges, and the reason every serious
//! implementation pads.
//!
//! So the image is embedded in a larger grid with a halo half a PSF wide on
//! each side, the halo is filled by extending the image ([`Edge`]), and the
//! result is cropped back out. What wraps then is halo onto halo, and it is
//! discarded.
//!
//! # The PSF goes in wrapped, not centred
//!
//! The other half of the contract, and the one that silently shifts a result
//! by half the grid when it is got wrong. A DFT treats index 0 as the origin,
//! so a kernel laid out with its peak in the middle of the array is a kernel
//! that translates by half the grid as well as blurring. [`place_psf`] writes
//! each PSF voxel at its offset *from the peak*, modulo the grid, which puts
//! the peak at index 0 with the rest wrapped around the far edges. That is
//! what the literature means by `ifftshift`, and it is not optional.

use super::fft::{self, Dims, Transform};
use rustfft::num_complex::Complex32;

/// How the halo around an image is filled.
///
/// The names and the behaviour are ImageJ's: Parallel Iterative Deconvolution
/// offers reflexive, periodic and zero, and reflexive is its default because
/// it is the one that does not invent an edge. Zero padding states that the
/// specimen stops at the field of view, which is false for nearly every
/// microscope image and shows up as a dark halo pulled in from the border.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Edge {
    /// Mirror at the border, repeating the edge pixel: `abc|cba`.
    Mirror,
    /// Hold the edge pixel: `abc|ccc`.
    Replicate,
    /// Outside is nothing.
    Zero,
    /// Wrap around — the circular convolution the FFT would have done anyway.
    Wrap,
}

impl Edge {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Edge::Mirror => "Mirror (reflexive)",
            Edge::Replicate => "Replicate edge",
            Edge::Zero => "Zero",
            Edge::Wrap => "Wrap (periodic)",
        }
    }

    pub(crate) const ALL: [Edge; 4] = [Edge::Mirror, Edge::Replicate, Edge::Zero, Edge::Wrap];

    /// Where the sample for padded coordinate `p` comes from, or `None` for
    /// "outside, and outside is zero".
    fn source(self, p: isize, n: isize) -> Option<usize> {
        if n <= 0 {
            return None;
        }
        if (0..n).contains(&p) {
            return Some(p as usize);
        }
        match self {
            Edge::Zero => None,
            Edge::Replicate => Some(p.clamp(0, n - 1) as usize),
            Edge::Wrap => Some(p.rem_euclid(n) as usize),
            Edge::Mirror => {
                // Fold repeatedly: one fold is not enough when the halo is
                // wider than the image, which happens for a PSF deeper than
                // the stack it is deconvolving.
                let period = 2 * n;
                let mut q = p.rem_euclid(period);
                if q >= n {
                    q = period - 1 - q;
                }
                Some(q as usize)
            }
        }
    }
}

/// The padded grid an image is deconvolved in.
pub(crate) struct Grid {
    /// The padded extent — what the FFT runs on.
    padded: Dims,
    /// The image's own extent.
    src: Dims,
    /// Where the image sits inside the padded grid.
    off: (usize, usize, usize),
    edge: Edge,
}

impl Grid {
    /// Pad `src` enough to hold a linear convolution with a `psf`-sized kernel.
    ///
    /// Half a PSF on each side is what a linear convolution needs: a voxel at
    /// the border is reached by kernel taps up to `psf/2` away, and beyond that
    /// nothing from the far edge can arrive. [`Edge::Wrap`] asks for the
    /// circular convolution on purpose and so gets no halo at all — only
    /// whatever rounding up to a fast length adds.
    pub(crate) fn plan(src: Dims, psf: Dims, edge: Edge) -> Grid {
        let halo = |p: usize| if edge == Edge::Wrap { 0 } else { p / 2 };
        let axis = |s: usize, p: usize| fft::next_fast_len(s + 2 * halo(p));
        let padded = Dims::new(
            axis(src.x, psf.x),
            axis(src.y, psf.y),
            // A single-slice image stays single-slice: a 2-D problem must not
            // grow a third axis just because the PSF has one.
            if src.z == 1 && psf.z == 1 {
                1
            } else {
                axis(src.z, psf.z)
            },
        );
        Grid {
            padded,
            src,
            off: (
                (padded.x - src.x) / 2,
                (padded.y - src.y) / 2,
                (padded.z - src.z) / 2,
            ),
            edge,
        }
    }

    pub(crate) fn padded(&self) -> Dims {
        self.padded
    }

    /// Copy `img` into the middle of a padded buffer and fill the halo.
    pub(crate) fn embed(&self, img: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(img.len(), self.src.len());
        out.clear();
        out.resize(self.padded.len(), 0.0);
        let map = |n_pad: usize, off: usize, n_src: usize| -> Vec<Option<usize>> {
            (0..n_pad)
                .map(|p| self.edge.source(p as isize - off as isize, n_src as isize))
                .collect()
        };
        let mx = map(self.padded.x, self.off.0, self.src.x);
        let my = map(self.padded.y, self.off.1, self.src.y);
        let mz = map(self.padded.z, self.off.2, self.src.z);

        for (z, sz) in mz.iter().enumerate() {
            let Some(sz) = sz else { continue };
            for (y, sy) in my.iter().enumerate() {
                let Some(sy) = sy else { continue };
                let row = self.src.at(0, *sy, *sz);
                let dst = self.padded.at(0, y, z);
                for (x, sx) in mx.iter().enumerate() {
                    if let Some(sx) = sx {
                        out[dst + x] = img[row + sx];
                    }
                }
            }
        }
    }

    /// Take the image back out of a padded buffer.
    pub(crate) fn crop(&self, padded: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(padded.len(), self.padded.len());
        out.clear();
        out.resize(self.src.len(), 0.0);
        for z in 0..self.src.z {
            for y in 0..self.src.y {
                let from = self.padded.at(self.off.0, y + self.off.1, z + self.off.2);
                let to = self.src.at(0, y, z);
                out[to..to + self.src.x].copy_from_slice(&padded[from..from + self.src.x]);
            }
        }
    }
}

/// Write a PSF into a padded grid with its peak at the origin, wrapped.
///
/// `peak` is the voxel of `psf` that should land at index 0. Everything else
/// goes at its offset from that, modulo the grid — so the half of the kernel
/// that is "to the left" of the peak appears at the high end of each axis.
///
/// A PSF larger than the grid would alias onto itself; the caller is expected
/// to have padded for it, and the modulo here means a PSF that is too big
/// degrades into a wrapped one rather than reading out of bounds.
pub(crate) fn place_psf(
    grid: Dims,
    psf: &[f32],
    psf_dims: Dims,
    peak: (usize, usize, usize),
) -> Vec<f32> {
    let mut out = vec![0.0f32; grid.len()];
    if grid.len() == 0 {
        return out;
    }
    for z in 0..psf_dims.z {
        let dz = (z as isize - peak.2 as isize).rem_euclid(grid.z as isize) as usize;
        for y in 0..psf_dims.y {
            let dy = (y as isize - peak.1 as isize).rem_euclid(grid.y as isize) as usize;
            for x in 0..psf_dims.x {
                let dx = (x as isize - peak.0 as isize).rem_euclid(grid.x as isize) as usize;
                // `+=`, not `=`: a PSF wider than the grid folds onto itself,
                // and summing is what a circular convolution does with it.
                out[grid.at(dx, dy, dz)] += psf[psf_dims.at(x, y, z)];
            }
        }
    }
    out
}

/// The brightest voxel, as `(x, y, z)`.
///
/// The default origin for a PSF read from a file. A measured PSF — a bead
/// imaged and cropped by hand — is rarely centred in its own array to the
/// voxel, and every voxel it is off by shifts the whole deconvolved stack by
/// exactly that much. Taking the peak instead of the geometric centre makes
/// that failure impossible to have silently.
pub(crate) fn brightest(psf: &[f32], dims: Dims) -> (usize, usize, usize) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in psf.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    let i = best.0;
    let x = i % dims.x.max(1);
    let y = (i / dims.x.max(1)) % dims.y.max(1);
    let z = i / (dims.x.max(1) * dims.y.max(1));
    (x, y, z)
}

/// The geometric centre, as `(x, y, z)`.
///
/// What a *synthetic* PSF wants, and what ImageJ's own tools assume: the model
/// put the emitter at the centre of the array, so that is the origin whether
/// or not the brightest voxel agrees. For a PSF with strong spherical
/// aberration they genuinely disagree, and the model is right.
pub(crate) fn centre(dims: Dims) -> (usize, usize, usize) {
    (dims.x / 2, dims.y / 2, dims.z / 2)
}

/// Scale a PSF so its samples sum to one.
///
/// Convolution by a kernel summing to one preserves total intensity, which is
/// what makes a deconvolved stack comparable to the raw one — the point of
/// quantitative deconvolution. A kernel summing to 1000 produces a result
/// 1000 times too dim, which looks like the algorithm failed.
///
/// Returns false when there is nothing to normalise: an all-zero or negative
/// PSF, which is a file the user did not mean to pick.
pub(crate) fn normalise(psf: &mut [f32]) -> bool {
    let sum: f64 = psf.iter().map(|&v| v as f64).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return false;
    }
    let k = (1.0 / sum) as f32;
    for v in psf.iter_mut() {
        *v *= k;
    }
    true
}

/// The blur a PSF applies, and its adjoint.
///
/// Holds the PSF's spectrum, so the hundreds of blurs an iterative method
/// performs cost one forward and one inverse transform each rather than three.
pub(crate) struct Operator {
    tf: Transform,
    h: Vec<Complex32>,
    work: Vec<Complex32>,
}

impl Operator {
    /// `psf` must already be placed on the grid by [`place_psf`].
    pub(crate) fn new(tf: Transform, psf: &[f32]) -> Operator {
        let n = tf.dims().len();
        debug_assert_eq!(psf.len(), n);
        let mut h = vec![Complex32::new(0.0, 0.0); n];
        fft::lift(psf, &mut h);
        tf.forward(&mut h);
        Operator {
            tf,
            h,
            work: vec![Complex32::new(0.0, 0.0); n],
        }
    }

    pub(crate) fn dims(&self) -> Dims {
        self.tf.dims()
    }

    /// The PSF's spectrum, for the one-shot spectral filters.
    pub(crate) fn spectrum(&self) -> &[Complex32] {
        &self.h
    }

    /// Transform a real buffer, hand the spectrum to `f`, transform back.
    ///
    /// The one place the spectral filters and the iterative ones meet: both
    /// are "go to the frequency domain, do something, come back", and sharing
    /// the plumbing is what keeps the algorithms in `algorithms.rs` down to
    /// the few lines that are actually the algorithm.
    pub(crate) fn through_spectrum(
        &mut self,
        src: &[f32],
        dst: &mut [f32],
        f: impl FnOnce(&mut [Complex32], &[Complex32]),
    ) {
        fft::lift(src, &mut self.work);
        self.tf.forward(&mut self.work);
        f(&mut self.work, &self.h);
        self.tf.inverse(&mut self.work);
        fft::lower(&self.work, dst);
    }

    /// The spectrum of some other kernel on the same grid, reusing this
    /// operator's plan.
    ///
    /// The regularised methods need a second kernel — the Laplacian whose
    /// power spectrum is their penalty — and planning a whole second
    /// transform for one use of it would be waste.
    pub(crate) fn spectrum_of(&mut self, kernel: &[f32]) -> Vec<Complex32> {
        let mut spec = vec![Complex32::new(0.0, 0.0); self.tf.dims().len()];
        fft::lift(kernel, &mut spec);
        self.tf.forward(&mut spec);
        spec
    }

    /// Blur: convolve with the PSF.
    pub(crate) fn blur(&mut self, src: &[f32], dst: &mut [f32]) {
        self.through_spectrum(src, dst, fft::multiply);
    }

    /// The adjoint: correlate with the PSF.
    pub(crate) fn correlate(&mut self, src: &[f32], dst: &mut [f32]) {
        self.through_spectrum(src, dst, fft::multiply_conj);
    }
}

#[cfg(test)]
#[path = "grid_tests.rs"]
mod tests;

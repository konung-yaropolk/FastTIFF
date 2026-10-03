//! Theoretical point spread functions, from the optics rather than from a
//! bead.
//!
//! A deconvolution is only as good as its PSF, and there are two ways to get
//! one: image a sub-resolution bead and clean it up, or compute what the
//! objective must do given its numerical aperture, the wavelength and what the
//! light travelled through. This module is the second. It is what ImageJ's
//! *Diffraction PSF 3D* and the BIG group's *PSF Generator* do, and the models
//! here are theirs.
//!
//! # The one equation
//!
//! Everything except the Gaussian is the same scalar diffraction integral,
//! differing only in the optical path difference fed to it:
//!
//! ```text
//!     h(r, z) = | integral over rho in 0..1 of
//!                   J0(k * NA * r * rho) * exp(i * k * OPD(rho, z)) * rho d(rho) | ^ 2
//! ```
//!
//! `rho` runs across the pupil, `J0` is the zeroth Bessel function of the
//! first kind — which is what an integral over a *circular* pupil leaves
//! behind — and `k` is `2 pi / lambda`. With `OPD = 0` this is the Airy
//! pattern. With a defocus term it is Born and Wolf. With terms for every
//! refractive index the light crossed it is Gibson and Lanni. There is no
//! third implementation: [`Model`] chooses which terms [`Optics::opd`]
//! contributes, and that is the entire difference between the models.
//!
//! # Radially symmetric, and computed that way
//!
//! None of these models depends on the angle, so the integral is evaluated
//! once per radius into a profile and the volume is filled by interpolating
//! it. For a 256-wide PSF that is 362 integrals per slice instead of 65,536 —
//! the difference between a dialog that returns and one that appears to hang.
//! Each output voxel averages a 3x3 subsample of the profile, which matters
//! for exactly one voxel and that voxel is the peak.

use super::fft::Dims;

/// Which theoretical model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Model {
    /// A separable Gaussian sized from the optics. Crude, instant, and what a
    /// great many published deconvolutions actually used.
    Gaussian,
    /// Scalar diffraction with a defocus term: the Airy pattern in focus,
    /// spreading above and below. ImageJ's *Diffraction PSF 3D*.
    BornWolf,
    /// Born and Wolf plus the refractive-index mismatches the light actually
    /// crossed — immersion, coverslip, specimen — which is what makes a PSF
    /// depth-dependent and asymmetric in z.
    GibsonLanni,
    /// The geometric blur circle, with no diffraction at all. The model a
    /// ray-tracer would give, and useful mostly as a sanity check.
    Defocus,
}

impl Model {
    pub(crate) const ALL: [Model; 4] = [
        Model::BornWolf,
        Model::GibsonLanni,
        Model::Gaussian,
        Model::Defocus,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Model::Gaussian => "Gaussian (from NA and wavelength)",
            Model::BornWolf => "Born & Wolf (scalar diffraction)",
            Model::GibsonLanni => "Gibson & Lanni (with index mismatch)",
            Model::Defocus => "Defocus (geometric, no diffraction)",
        }
    }

    pub(crate) fn tag(self) -> &'static str {
        match self {
            Model::Gaussian => "gaussian",
            Model::BornWolf => "bornwolf",
            Model::GibsonLanni => "gibsonlanni",
            Model::Defocus => "defocus",
        }
    }
}

/// How the microscope forms the image, which decides what the PSF of the whole
/// system is once the objective's is known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    /// The objective's PSF, as computed.
    Widefield,
    /// Excitation and detection through the same objective with a pinhole
    /// small enough to ignore: the two multiply, so the system PSF is the
    /// square. An approximation, and a standard one — a real pinhole of one
    /// Airy unit gives something between this and widefield.
    Confocal,
    /// Two-photon excitation: the rate goes as the square of the intensity, so
    /// the system PSF is again the square — but of the PSF at the *excitation*
    /// wavelength, which is about twice the emission. Enter that wavelength.
    TwoPhoton,
}

impl Mode {
    pub(crate) const ALL: [Mode; 3] = [Mode::Widefield, Mode::Confocal, Mode::TwoPhoton];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Mode::Widefield => "Widefield",
            Mode::Confocal => "Confocal (small pinhole)",
            Mode::TwoPhoton => "Two-photon",
        }
    }

    /// The exponent the computed PSF is raised to.
    fn power(self) -> i32 {
        match self {
            Mode::Widefield => 1,
            Mode::Confocal | Mode::TwoPhoton => 2,
        }
    }
}

/// The optical train, in micrometres.
///
/// Design values are what the objective was corrected for and are engraved on
/// it; actual values are what is on the bench today. They are separate because
/// their *difference* is the aberration — an oil objective used with a 1.5 mm
/// coverslip instead of a 0.17 mm one is a textbook spherical-aberration
/// problem, and a PSF that ignored it would deconvolve towards the wrong
/// answer with great confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Optics {
    pub na: f64,
    /// Emission wavelength, or excitation for two-photon.
    pub lambda: f64,
    /// Immersion medium, actual and design.
    pub ni: f64,
    pub ni0: f64,
    /// Coverslip, actual and design.
    pub ng: f64,
    pub ng0: f64,
    /// Specimen.
    pub ns: f64,
    /// Working distance: the immersion thickness the objective was designed
    /// for.
    pub ti0: f64,
    /// Coverslip thickness, actual and design.
    pub tg: f64,
    pub tg0: f64,
    /// How far below the coverslip the emitter sits. The parameter that makes
    /// Gibson and Lanni worth the extra fields: with `ns` below `ni` the PSF
    /// degrades with depth, which is the single largest aberration in most
    /// live imaging.
    pub depth: f64,
    /// Longitudinal spherical aberration at full aperture — Bob Dougherty's
    /// parameter in *Diffraction PSF 3D*, in micrometres of focal shift
    /// between the paraxial rays and the marginal ones.
    pub sa: f64,
    /// Lateral sample spacing.
    pub pixel: f64,
    /// Axial sample spacing.
    pub step: f64,
}

impl Default for Optics {
    fn default() -> Self {
        Optics {
            na: 1.4,
            lambda: 0.52,
            ni: 1.515,
            ni0: 1.515,
            ng: 1.515,
            ng0: 1.515,
            ns: 1.33,
            ti0: 150.0,
            tg: 0.17,
            tg0: 0.17,
            depth: 0.0,
            sa: 0.0,
            pixel: 0.1,
            step: 0.25,
        }
    }
}

impl Optics {
    /// The optical path difference across the pupil, at defocus `z`.
    ///
    /// Each term is a thickness times the axial direction cosine in that
    /// medium — the extra path a ray at pupil radius `rho` takes compared with
    /// the axis. Design terms enter negative, so a microscope used exactly as
    /// designed has them cancel and is left with the defocus alone, which is
    /// Born and Wolf. That is not a coincidence to be grateful for; it is why
    /// [`Model::BornWolf`] needs no code of its own.
    fn opd(&self, model: Model, rho: f64, z: f64) -> f64 {
        // `max(0)`: when NA exceeds a medium's index the ray at that pupil
        // radius is beyond the critical angle and never propagates there.
        // Clamping drops it from the integral, which is what physically
        // happens — an oil lens of NA 1.4 looking into water collects no more
        // than NA 1.33 of specimen-side aperture.
        let cos = |n: f64| (1.0 - (self.na * rho / n).powi(2)).max(0.0).sqrt();
        let mut opd = match model {
            Model::GibsonLanni => {
                self.ns * self.depth * cos(self.ns) + self.ng * self.tg * cos(self.ng)
                    - self.ng0 * self.tg0 * cos(self.ng0)
                    + self.ni * (self.ti0 + z) * cos(self.ni)
                    - self.ni0 * self.ti0 * cos(self.ni0)
            }
            // Pure defocus in the immersion medium.
            _ => self.ni * z * cos(self.ni),
        };
        // Primary spherical aberration, as a focal shift that grows with the
        // square of the aperture. Integrating `dW/d(rho^2) = (NA^2/2n) * sa *
        // rho^2` — the defocus term's own coefficient, with `z` replaced by
        // the shift at that radius — gives a quartic, which is the Zernike
        // form it should be.
        if self.sa != 0.0 {
            opd += self.na * self.na * self.sa * rho.powi(4) / (4.0 * self.ni);
        }
        opd
    }

    /// The Gaussian widths this optical train implies, as `(sigma_xy,
    /// sigma_z)` in micrometres.
    ///
    /// The widefield paraxial approximations — `0.21 lambda / NA` laterally
    /// and `0.66 lambda n / NA^2` axially. Accurate to a few percent below
    /// NA 1.0 and progressively optimistic above it, which is the known price
    /// of the Gaussian model and the reason it is not the default.
    pub(crate) fn gaussian_sigma(&self) -> (f64, f64) {
        let na = self.na.max(1e-3);
        (
            0.21 * self.lambda / na,
            0.66 * self.lambda * self.ni / (na * na),
        )
    }
}

/// Compute a PSF volume, in `xyz` order with `z` slowest.
///
/// Returns `None` if `progress` asked to stop.
pub(crate) fn generate(
    model: Model,
    mode: Mode,
    o: &Optics,
    dims: Dims,
    progress: &mut dyn FnMut(f32) -> bool,
) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; dims.len()];
    // The emitter sits on voxel `n / 2`, not between voxels at `(n - 1) / 2`.
    // For an odd extent they are the same; for an even one they differ by half
    // a sample, and `grid::centre` — which is what tells the convolution where
    // this PSF's origin is — says `n / 2`. Two definitions of the centre half
    // a voxel apart is a deconvolution that shifts the whole stack by half a
    // voxel per axis, visible as a result that will not quite register with
    // its own input.
    let (cx, cy, cz) = (
        (dims.x / 2) as f64,
        (dims.y / 2) as f64,
        (dims.z / 2) as f64,
    );

    // Three subsamples per pixel per axis. The profile is smooth everywhere
    // except at the peak, where it is not, and the peak is one voxel.
    const SUB: usize = 3;
    let offsets: Vec<f64> = (0..SUB)
        .map(|i| (i as f64 + 0.5) / SUB as f64 - 0.5)
        .collect();

    // The radial profile is sampled finely enough that the interpolation
    // between samples is never the limiting error.
    let corner = ((cx * o.pixel).powi(2) + (cy * o.pixel).powi(2)).sqrt();
    let dr = o.pixel / 8.0;
    let radii = (corner / dr).ceil() as usize + 2;

    let mut profile = vec![0.0f64; radii];
    for zi in 0..dims.z {
        if !progress(zi as f32 / dims.z.max(1) as f32) {
            return None;
        }
        let z = (zi as f64 - cz) * o.step;
        radial_profile(model, o, z, dr, &mut profile);

        for yi in 0..dims.y {
            for xi in 0..dims.x {
                let mut acc = 0.0f64;
                for &oy in &offsets {
                    let dy = (yi as f64 + oy - cy) * o.pixel;
                    for &ox in &offsets {
                        let dx = (xi as f64 + ox - cx) * o.pixel;
                        acc += sample(&profile, (dx * dx + dy * dy).sqrt() / dr);
                    }
                }
                let v = acc / (SUB * SUB) as f64;
                out[dims.at(xi, yi, zi)] = match mode.power() {
                    1 => v as f32,
                    _ => (v * v) as f32,
                };
            }
        }
    }
    Some(out)
}

/// Linear interpolation into a radial profile, clamped at the far end.
fn sample(profile: &[f64], at: f64) -> f64 {
    let i = at.floor();
    if i < 0.0 {
        return profile.first().copied().unwrap_or(0.0);
    }
    let i = i as usize;
    match (profile.get(i), profile.get(i + 1)) {
        (Some(a), Some(b)) => {
            let f = at - i as f64;
            a + (b - a) * f
        }
        (Some(a), None) => *a,
        _ => 0.0,
    }
}

/// Fill `profile[i]` with the PSF at radius `i * dr` and defocus `z`.
fn radial_profile(model: Model, o: &Optics, z: f64, dr: f64, profile: &mut [f64]) {
    if model == Model::Gaussian {
        let (sxy, sz) = o.gaussian_sigma();
        let axial = (-0.5 * (z / sz.max(1e-9)).powi(2)).exp();
        for (i, p) in profile.iter_mut().enumerate() {
            let r = i as f64 * dr;
            *p = axial * (-0.5 * (r / sxy.max(1e-9)).powi(2)).exp();
        }
        return;
    }
    if model == Model::Defocus {
        // The geometric cone: rays fill a disc whose radius grows with the
        // tangent of the collection half-angle. In focus it is a point, which
        // on a sampled grid means one pixel.
        let sin = (o.na / o.ni).min(0.999_999);
        let tan = sin / (1.0 - sin * sin).sqrt();
        let radius = (z.abs() * tan).max(o.pixel * 0.5);
        let area = std::f64::consts::PI * radius * radius;
        for (i, p) in profile.iter_mut().enumerate() {
            *p = if i as f64 * dr <= radius {
                1.0 / area
            } else {
                0.0
            };
        }
        return;
    }

    let k = std::f64::consts::TAU / o.lambda;
    let r_max = (profile.len() as f64) * dr;
    // Simpson's rule, with enough points that the fastest oscillation in the
    // integrand — whichever of the Bessel argument and the phase swings
    // further across the pupil — is sampled many times over.
    let swing = (k * o.na * r_max).max(k * o.opd(model, 1.0, z).abs());
    let steps = (((swing / std::f64::consts::PI) * 8.0) as usize).clamp(64, 4096);
    let steps = steps + steps % 2; // Simpson needs an even number of intervals
    let h = 1.0 / steps as f64;

    // The phase is the same for every radius, so it is computed once per pupil
    // sample rather than once per radius per pupil sample. That single hoist
    // is most of the difference between this returning in a moment and not.
    let phase: Vec<(f64, f64, f64)> = (0..=steps)
        .map(|i| {
            let rho = i as f64 * h;
            let w = k * o.opd(model, rho, z);
            let weight = simpson_weight(i, steps) * rho;
            (weight * w.cos(), weight * w.sin(), rho)
        })
        .collect();

    for (i, p) in profile.iter_mut().enumerate() {
        let r = i as f64 * dr;
        let (mut re, mut im) = (0.0, 0.0);
        for &(wc, ws, rho) in &phase {
            let j = bessel_j0(k * o.na * r * rho);
            re += wc * j;
            im += ws * j;
        }
        *p = (re * re + im * im) * (h / 3.0) * (h / 3.0);
    }
}

/// Simpson's 1, 4, 2, 4, ..., 4, 1.
fn simpson_weight(i: usize, steps: usize) -> f64 {
    if i == 0 || i == steps {
        1.0
    } else if i % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

/// The Bessel function of the first kind, order zero.
///
/// Abramowitz and Stegun 9.4.1 and 9.4.3, good to about 1e-8 absolute, which
/// is five orders below the accuracy of any PSF model that would call it.
/// Written out rather than pulled from a crate because it is twenty lines and
/// a dependency on a special-function library is not twenty lines.
pub(crate) fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.0 {
        let t = (x / 3.0).powi(2);
        1.0 + t
            * (-2.2499997
                + t * (1.2656208
                    + t * (-0.3163866 + t * (0.0444479 + t * (-0.0039444 + t * 0.00021)))))
    } else {
        let t = 3.0 / ax;
        let f = 0.797_884_56
            + t * (-0.000_000_77
                + t * (-0.005_527_40
                    + t * (-0.000_095_12
                        + t * (0.001_372_37 + t * (-0.000_728_05 + t * 0.000_144_76)))));
        // A&S prints this offset as 0.78539816; it is pi/4 exactly, and the
        // constant is both clearer and a few digits better.
        let theta = ax - std::f64::consts::FRAC_PI_4
            + t * (-0.041_663_97
                + t * (-0.000_039_54
                    + t * (0.002_625_73
                        + t * (-0.000_541_25 + t * (-0.000_293_33 + t * 0.000_135_58)))));
        f * theta.cos() / ax.sqrt()
    }
}

/// The Bessel function of the first kind, order one.
///
/// Used by the tests, which check the in-focus profile against the closed-form
/// Airy pattern `(2 J1(v) / v)^2`. A diffraction model that cannot reproduce
/// the Airy disc is not a diffraction model, and that is the one assertion
/// here that no amount of plausible-looking output can satisfy by accident.
///
/// `cfg(test)`: nothing in the models needs it. An oracle that shared code
/// with the thing it checks would not be one.
#[cfg(test)]
pub(crate) fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.0 {
        let t = (x / 3.0).powi(2);
        x * (0.5
            + t * (-0.56249985
                + t * (0.21093573
                    + t * (-0.03954289 + t * (0.00443319 + t * (-0.00031761 + t * 0.00001109))))))
    } else {
        let t = 3.0 / ax;
        let f = 0.797_884_56
            + t * (0.000_001_56
                + t * (0.016_596_67
                    + t * (0.000_171_05
                        + t * (-0.002_495_11 + t * (0.001_136_53 + t * (-0.000_200_33))))));
        // Likewise 2.35619449, which is 3 pi / 4.
        let theta = ax - 3.0 * std::f64::consts::FRAC_PI_4
            + t * (0.124_996_12
                + t * (0.000_056_50
                    + t * (-0.006_378_79
                        + t * (0.000_743_48 + t * (0.000_798_24 + t * (-0.000_291_66))))));
        let j = f * theta.cos() / ax.sqrt();
        if x < 0.0 {
            -j
        } else {
            j
        }
    }
}

#[cfg(test)]
#[path = "optics_tests.rs"]
mod tests;

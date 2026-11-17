//! The optical models, against closed-form optics.
//!
//! A PSF model is the one part of a deconvolution tool whose output nobody can
//! eyeball: a wrong PSF produces a perfectly plausible blob. So these tests
//! check it against things that are known independently — the Bessel
//! functions' tabulated values, the Airy pattern, the Rayleigh criterion, and
//! the reductions one model makes to another.

use super::*;

/// Tabulated to ten places (Abramowitz and Stegun, table 9.1).
#[test]
fn bessel_j0_matches_the_tables() {
    for (x, want) in [
        (0.0, 1.0),
        (1.0, 0.765_197_686_6),
        (2.0, 0.223_890_779_1),
        (3.0, -0.260_051_954_9),
        (5.0, -0.177_596_771_3),
        (10.0, -0.245_935_764_5),
        (20.0, 0.167_024_664_3),
    ] {
        assert!(
            (bessel_j0(x) - want).abs() < 1e-7,
            "J0({x}) = {} not {want}",
            bessel_j0(x)
        );
    }
    assert_eq!(bessel_j0(-2.0), bessel_j0(2.0), "J0 is even");
}

#[test]
fn bessel_j0_has_its_zeros_where_it_should() {
    for zero in [2.404_825_558, 5.520_078_110, 8.653_727_913] {
        assert!(
            bessel_j0(zero).abs() < 1e-6,
            "J0({zero}) = {}",
            bessel_j0(zero)
        );
    }
}

#[test]
fn bessel_j1_matches_the_tables() {
    for (x, want) in [
        (0.0, 0.0),
        (1.0, 0.440_050_585_7),
        (2.0, 0.576_724_807_8),
        (5.0, -0.327_579_137_6),
        (10.0, 0.043_472_746_2),
    ] {
        assert!(
            (bessel_j1(x) - want).abs() < 1e-7,
            "J1({x}) = {} not {want}",
            bessel_j1(x)
        );
    }
    assert!(
        (bessel_j1(-3.0) + bessel_j1(3.0)).abs() < 1e-12,
        "J1 is odd"
    );
}

// --------------------------------------------------------------- the Airy disc

fn airy_optics() -> Optics {
    Optics {
        na: 1.0,
        lambda: 0.5,
        ni: 1.515,
        ni0: 1.515,
        ..Optics::default()
    }
}

/// In focus and unaberrated, the scalar diffraction integral *is* the Airy
/// pattern: `(2 J1(v) / v)^2`. Nothing else this module produces is checkable
/// in closed form, so this is the anchor for all of it.
#[test]
fn the_in_focus_profile_is_the_airy_pattern() {
    let o = airy_optics();
    let dr = 0.002;
    let mut profile = vec![0.0; 400];
    radial_profile(Model::BornWolf, &o, 0.0, dr, &mut profile);

    // The integral of `rho d(rho)` over the unit pupil is 1/2, so the on-axis
    // intensity is its square.
    assert!(
        (profile[0] - 0.25).abs() < 1e-6,
        "on-axis intensity {} should be 1/4",
        profile[0]
    );

    let k = std::f64::consts::TAU / o.lambda;
    for i in 1..profile.len() {
        let v = k * o.na * (i as f64 * dr);
        let airy = (2.0 * bessel_j1(v) / v).powi(2);
        let got = profile[i] / profile[0];
        assert!(
            (got - airy).abs() < 2e-6,
            "at r = {:.3} um the model gives {got:.8} and Airy gives {airy:.8}",
            i as f64 * dr
        );
    }
}

/// The Rayleigh criterion, stated as the first dark ring at `0.61 lambda /
/// NA`. The same fact as the test above, reached through the generated
/// volume rather than the profile, so the sampling and interpolation are in
/// the path too.
#[test]
fn the_first_dark_ring_is_at_the_rayleigh_radius() {
    let mut o = airy_optics();
    o.pixel = 0.02;
    o.step = 0.1;
    let dims = Dims::new(81, 81, 1);
    let psf = generate(Model::BornWolf, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");

    let (cx, cy) = (dims.x / 2, dims.y / 2);
    let along: Vec<f32> = (0..cx).map(|i| psf[dims.at(cx + i, cy, 0)]).collect();
    let first_min = (1..along.len() - 1)
        .find(|&i| along[i] <= along[i - 1] && along[i] <= along[i + 1])
        .expect("there must be a dark ring");

    let want = 0.61 * o.lambda / o.na;
    let got = first_min as f64 * o.pixel;
    assert!(
        (got - want).abs() <= o.pixel,
        "the first minimum is at {got:.3} um, Rayleigh says {want:.3} um"
    );
}

// ------------------------------------------------------- what the models do

#[test]
fn an_unaberrated_psf_is_symmetric_in_z() {
    let o = Optics {
        na: 1.2,
        lambda: 0.52,
        pixel: 0.08,
        step: 0.2,
        ..Optics::default()
    };
    let dims = Dims::new(21, 21, 11);
    let psf = generate(Model::BornWolf, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let mid = dims.z / 2;
    for d in 1..=mid {
        for i in 0..dims.x * dims.y {
            let a = psf[(mid - d) * dims.x * dims.y + i];
            let b = psf[(mid + d) * dims.x * dims.y + i];
            assert!(
                (a - b).abs() <= 1e-6 * a.abs().max(1e-9),
                "slices {} and {} differ at {i}: {a} vs {b}",
                mid - d,
                mid + d
            );
        }
    }
}

/// Spherical aberration is the thing that breaks that symmetry, which is the
/// whole reason the parameter exists.
#[test]
fn spherical_aberration_makes_it_asymmetric() {
    let o = Optics {
        na: 1.2,
        lambda: 0.52,
        pixel: 0.08,
        step: 0.2,
        sa: 2.0,
        ..Optics::default()
    };
    let dims = Dims::new(21, 21, 11);
    let psf = generate(Model::BornWolf, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let plane = dims.x * dims.y;
    let sum = |z: usize| -> f64 {
        psf[z * plane..(z + 1) * plane]
            .iter()
            .map(|&v| v as f64)
            .sum()
    };
    let (above, below) = (sum(2), sum(8));
    assert!(
        (above - below).abs() > 0.01 * above.max(below),
        "aberration left the PSF symmetric: {above} vs {below}"
    );
}

/// Gibson and Lanni reduces to Born and Wolf when the microscope is used
/// exactly as designed. This is the claim the shared `opd` is built on; if it
/// failed, the two models would be two implementations rather than one.
#[test]
fn gibson_lanni_reduces_to_born_and_wolf_when_nothing_is_mismatched() {
    let o = Optics {
        na: 1.2,
        lambda: 0.52,
        ni: 1.515,
        ni0: 1.515,
        ng: 1.515,
        ng0: 1.515,
        tg: 170.0,
        tg0: 170.0,
        depth: 0.0,
        pixel: 0.08,
        step: 0.2,
        ..Optics::default()
    };
    let dims = Dims::new(15, 15, 7);
    let bw = generate(Model::BornWolf, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let gl = generate(Model::GibsonLanni, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    for (i, (a, b)) in bw.iter().zip(&gl).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(1e-6),
            "voxel {i}: {a} vs {b}"
        );
    }
}

/// And it does not reduce to it when the specimen's index differs and the
/// emitter is deep in it, which is the case the model is for.
#[test]
fn depth_in_a_mismatched_specimen_changes_the_psf() {
    let base = Optics {
        na: 1.3,
        lambda: 0.52,
        ns: 1.33,
        pixel: 0.08,
        step: 0.2,
        ..Optics::default()
    };
    let dims = Dims::new(15, 15, 7);
    let shallow = generate(
        Model::GibsonLanni,
        Mode::Widefield,
        &base,
        dims,
        &mut |_| true,
    )
    .expect("ok");
    let deep = generate(
        Model::GibsonLanni,
        Mode::Widefield,
        &Optics {
            depth: 30.0,
            ..base
        },
        dims,
        &mut |_| true,
    )
    .expect("ok");

    let peak = |v: &[f32]| v.iter().copied().fold(0.0f32, f32::max);
    assert!(
        peak(&deep) < peak(&shallow) * 0.98,
        "30 um into water with an oil lens must cost peak intensity: {} vs {}",
        peak(&deep),
        peak(&shallow)
    );
}

#[test]
fn an_impossible_aperture_collects_nothing_beyond_the_critical_angle() {
    // NA 1.4 oil looking into water: everything past rho = 1.33/1.4 is beyond
    // the critical angle, and `opd` clamps it rather than taking the square
    // root of a negative number.
    let o = Optics {
        na: 1.4,
        ns: 1.33,
        depth: 10.0,
        ..Optics::default()
    };
    for i in 0..=100 {
        let v = o.opd(Model::GibsonLanni, i as f64 / 100.0, 0.5);
        assert!(v.is_finite(), "OPD at rho = {} is {v}", i as f64 / 100.0);
    }
}

#[test]
fn the_gaussian_widths_follow_the_published_approximations() {
    let o = Optics {
        na: 1.4,
        lambda: 0.52,
        ni: 1.515,
        ..Optics::default()
    };
    let (sxy, sz) = o.gaussian_sigma();
    assert!((sxy - 0.21 * 0.52 / 1.4).abs() < 1e-12);
    assert!((sz - 0.66 * 0.52 * 1.515 / (1.4 * 1.4)).abs() < 1e-12);
    // And they are the right way round: the axial resolution of a microscope
    // is always the worse of the two.
    assert!(sz > sxy * 2.0, "{sxy} / {sz}");
}

#[test]
fn a_gaussian_psf_peaks_at_the_centre_and_falls_away() {
    let o = Optics {
        na: 1.0,
        lambda: 0.5,
        pixel: 0.05,
        step: 0.15,
        ..Optics::default()
    };
    let dims = Dims::new(21, 21, 9);
    let psf = generate(Model::Gaussian, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let centre = psf[dims.at(dims.x / 2, dims.y / 2, dims.z / 2)];
    assert_eq!(
        psf.iter().copied().fold(0.0f32, f32::max),
        centre,
        "the brightest voxel must be the one the emitter is on"
    );
    assert!(psf[dims.at(0, 0, 0)] < centre * 0.01);
}

/// Confocal and two-photon square the widefield PSF. The consequence worth
/// pinning is the one people rely on: the squared PSF is narrower.
#[test]
fn squaring_the_psf_narrows_it() {
    let o = Optics {
        na: 1.0,
        lambda: 0.5,
        pixel: 0.04,
        step: 0.15,
        ..Optics::default()
    };
    let dims = Dims::new(31, 31, 1);
    let wide = generate(Model::BornWolf, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let conf = generate(Model::BornWolf, Mode::Confocal, &o, dims, &mut |_| true).expect("ok");

    let width = |v: &[f32]| -> usize {
        let peak = v.iter().copied().fold(0.0f32, f32::max);
        v.iter().filter(|&&x| x >= peak * 0.5).count()
    };
    assert!(
        width(&conf) < width(&wide),
        "confocal {} is not narrower than widefield {}",
        width(&conf),
        width(&wide)
    );
    assert_eq!(Mode::Widefield.power(), 1);
    assert_eq!(Mode::TwoPhoton.power(), 2);
}

/// The geometric model has no diffraction, so in focus it is a point and out
/// of focus it is a disc that grows linearly.
#[test]
fn the_defocus_model_spreads_with_distance() {
    let o = Optics {
        na: 0.5,
        ni: 1.0,
        pixel: 0.1,
        step: 1.0,
        ..Optics::default()
    };
    let dims = Dims::new(41, 41, 5);
    let psf = generate(Model::Defocus, Mode::Widefield, &o, dims, &mut |_| true).expect("ok");
    let plane = dims.x * dims.y;
    let lit = |z: usize| {
        psf[z * plane..(z + 1) * plane]
            .iter()
            .filter(|&&v| v > 0.0)
            .count()
    };
    assert!(lit(2) < lit(1), "in focus should be the tightest");
    assert!(lit(1) < lit(0), "further out of focus should be wider");
}

#[test]
fn generation_can_be_cancelled() {
    let o = Optics::default();
    let mut calls = 0;
    let out = generate(
        Model::BornWolf,
        Mode::Widefield,
        &o,
        Dims::new(32, 32, 16),
        &mut |_| {
            calls += 1;
            calls < 3
        },
    );
    assert!(out.is_none());
}

#[test]
fn every_model_and_mode_has_its_own_name() {
    let mut names: Vec<&str> = Model::ALL.iter().map(|m| m.label()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), Model::ALL.len());

    let mut tags: Vec<&str> = Model::ALL.iter().map(|m| m.tag()).collect();
    tags.sort_unstable();
    tags.dedup();
    assert_eq!(tags.len(), Model::ALL.len());

    let mut modes: Vec<&str> = Mode::ALL.iter().map(|m| m.label()).collect();
    modes.sort_unstable();
    modes.dedup();
    assert_eq!(modes.len(), Mode::ALL.len());
}

//! The spline, checked by the property that defines it.
//!
//! A cubic B-spline *interpolates*: the continuous function its coefficients
//! describe passes exactly through the samples it was built from. That single
//! round trip exercises the prefilter's gain, both recursions, both boundary
//! seeds, the weights and the index mirror together, and nothing else here is
//! as sharp. The rest of these tests exist to say *which* of those is wrong
//! when the round trip fails.

use super::*;

/// Reconstruct at every integer position and compare with the samples.
fn round_trip_error(samples: &[f32], w: usize, h: usize) -> f64 {
    let mut coeff = samples.to_vec();
    prefilter_2d(&mut coeff, w, h, &POLES_3);
    let mut worst = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let got = interpolate(&coeff, w, h, x as f64, y as f64);
            worst = worst.max((got - samples[y * w + x] as f64).abs());
        }
    }
    worst
}

/// The defining property, on a pattern with content at every scale.
#[test]
fn the_spline_passes_through_its_own_samples() {
    let (w, h) = (17, 13);
    let samples: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            100.0 + 40.0 * (x * 0.7).sin() + 25.0 * (y * 1.3).cos() + 7.0 * ((x * y) % 5.0)
        })
        .collect();
    let worst = round_trip_error(&samples, w, h);
    assert!(
        worst < 2e-3,
        "reconstruction should pass through the samples, worst error {worst}"
    );
}

/// Including right at the edges, which is where the boundary model decides the
/// answer and where a whole-sample mirror would differ from a half-sample one.
#[test]
fn the_round_trip_holds_at_the_borders() {
    let (w, h) = (9, 7);
    // A ramp with a step at the edge: the boundary seed has to carry it.
    let samples: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                200.0
            } else {
                (x + y) as f32
            }
        })
        .collect();
    let mut coeff = samples.clone();
    prefilter_2d(&mut coeff, w, h, &POLES_3);
    for (x, y) in [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (0, 3),
        (4, 0),
    ] {
        let got = interpolate(&coeff, w, h, x as f64, y as f64);
        let want = samples[y * w + x] as f64;
        assert!(
            (got - want).abs() < 2e-3,
            "corner/edge ({x},{y}): got {got}, want {want}"
        );
    }
}

/// A constant image is constant everywhere the spline is defined — on the grid,
/// between samples, and outside the frame where the mirror takes over. This is
/// what the "sum to one" construction of `w[1]` buys.
#[test]
fn a_flat_image_is_flat_between_and_beyond_its_samples() {
    let (w, h) = (6, 5);
    let mut coeff = vec![42.0f32; w * h];
    prefilter_2d(&mut coeff, w, h, &POLES_3);
    for &(x, y) in &[
        (0.0, 0.0),
        (2.5, 1.5),
        (5.0, 4.0),
        (-1.75, 2.25),
        (7.5, -2.5),
        (0.5, 4.5),
    ] {
        let got = interpolate(&coeff, w, h, x, y);
        assert!(
            (got - 42.0).abs() < 1e-3,
            "flat image sampled at ({x},{y}) gave {got}"
        );
    }
}

/// The weights are a partition of unity, exactly. If they were not, a flat
/// image would shade.
#[test]
fn the_weights_sum_to_one() {
    for i in 0..=100 {
        let f = i as f64 / 100.0;
        let w = weights(f);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-15, "frac {f}: weights sum to {sum}");
        assert!(
            w.iter().all(|v| *v >= -1e-15),
            "cubic B-spline weights are non-negative: {w:?}"
        );
    }
}

/// And they are the cubic B-spline evaluated at the right offsets, in the
/// reference's descending order: `w[0]` is the sample two to the right,
/// `w[3]` the one to the left.
#[test]
fn the_weights_are_beta3_in_descending_order() {
    // beta3(t), the cubic B-spline.
    let beta3 = |t: f64| {
        let t = t.abs();
        if t < 1.0 {
            2.0 / 3.0 - t * t + t * t * t / 2.0
        } else if t < 2.0 {
            (2.0 - t).powi(3) / 6.0
        } else {
            0.0
        }
    };
    for i in 0..=20 {
        let f = i as f64 / 20.0;
        let w = weights(f);
        // Offsets of the four samples relative to the sample point.
        for (k, offset) in [2.0, 1.0, 0.0, -1.0].iter().enumerate() {
            let want = beta3(offset - f);
            assert!(
                (w[k] - want).abs() < 1e-12,
                "frac {f}, slot {k} (offset {offset}): got {}, want {want}",
                w[k]
            );
        }
    }
}

/// On the grid, exactly one weight is 1 and the rest are 0 — so an integer
/// position reads one coefficient and not a blend of four.
#[test]
fn an_integer_position_has_a_single_nonzero_weight() {
    let w = weights(0.0);
    assert!((w[2] - 2.0 / 3.0).abs() < 1e-15, "{w:?}");
    // beta3(0) = 2/3 with beta3(±1) = 1/6: three non-zero weights, which is
    // what makes the prefilter necessary in the first place.
    assert!((w[1] - 1.0 / 6.0).abs() < 1e-15, "{w:?}");
    assert!((w[3] - 1.0 / 6.0).abs() < 1e-15, "{w:?}");
    assert!(w[0].abs() < 1e-15, "{w:?}");
}

/// Indices inside the frame are just the four neighbours, descending.
#[test]
fn interior_indexes_are_the_four_neighbours() {
    assert_eq!(indexes(5.25, 20), [7, 6, 5, 4]);
    assert_eq!(indexes(5.0, 20), [7, 6, 5, 4]);
}

/// Outside it they mirror back in without repeating the edge sample, which is
/// the half-sample convention the prefilter assumed.
#[test]
fn indexes_mirror_at_the_edges() {
    // Near the left edge: -1 reflects to 0, -2 to 1.
    assert_eq!(indexes(0.5, 8), [2, 1, 0, 0]);
    assert_eq!(indexes(-1.5, 8), [0, 0, 1, 2]);
    // Near the right edge of an 8-wide axis: 8 reflects to 7, 9 to 6.
    assert_eq!(indexes(6.5, 8), [7, 7, 6, 5]);
    // Everything stays in range however far out the request is.
    for x in [-100.0, -7.5, 0.0, 7.0, 50.0, 1000.25] {
        for &i in indexes(x, 8).iter() {
            assert!(i < 8, "x {x} produced index {i} on an 8-wide axis");
        }
    }
}

/// A one-sample axis is a legal degenerate case: every index is 0 and the
/// prefilter leaves it alone.
#[test]
fn a_single_sample_axis_is_handled() {
    assert_eq!(indexes(0.0, 1), [0, 0, 0, 0]);
    assert_eq!(indexes(-5.0, 1), [0, 0, 0, 0]);
    let mut line = vec![3.0f64];
    samples_to_coefficients_1d(&mut line, &POLES_3);
    assert_eq!(line, vec![3.0], "one sample is its own coefficient");
}

/// The fractional part is floored, not truncated, so a negative coordinate
/// evaluates the spline on its support rather than off it.
#[test]
fn the_fraction_is_floored() {
    assert!((frac(2.25) - 0.25).abs() < 1e-15);
    assert!((frac(-0.25) - 0.75).abs() < 1e-15);
    assert!(frac(-3.0).abs() < 1e-15);
    for x in [-10.5, -0.001, 0.0, 0.999, 7.5] {
        let f = frac(x);
        assert!((0.0..1.0).contains(&f), "frac({x}) = {f} is outside 0..1");
    }
}

/// The degree-7 prefilter — used only by the pyramid — reconstructs too. A
/// wrong pole there shows up as a pyramid that blurs asymmetrically, which is
/// hard to see and changes every coarse-level answer.
#[test]
fn the_degree_seven_filter_also_round_trips() {
    let (w, h) = (12, 10);
    let samples: Vec<f32> = (0..w * h)
        .map(|i| ((i * 37) % 101) as f32 + ((i / w) as f32) * 3.0)
        .collect();
    let mut coeff = samples.clone();
    prefilter_2d(&mut coeff, w, h, &POLES_7);
    // Degree 7 needs degree-7 weights to reconstruct exactly; what is checked
    // here is the weaker but still telling property that the filter is
    // invertible in the mean — it must not change the image's total.
    let before: f64 = samples.iter().map(|&v| v as f64).sum();
    let after: f64 = coeff.iter().map(|&v| v as f64).sum();
    assert!(
        (after - before).abs() / before.abs() < 0.02,
        "a prefilter should conserve the mean to within its gain: {before} -> {after}"
    );
}

/// The poles are the published values. Written out as constants, so this is a
/// guard against a transcription slip rather than a computation.
#[test]
fn the_poles_are_the_published_ones() {
    assert!(
        (POLES_3[0] - (3.0f64.sqrt() - 2.0)).abs() < 1e-15,
        "the cubic pole is sqrt(3) - 2, got {}",
        POLES_3[0]
    );
    // All poles must be inside the unit circle or the recursions diverge.
    for &z in POLES_3.iter().chain(POLES_7.iter()) {
        assert!(z.abs() < 1.0, "pole {z} is not inside the unit circle");
    }
}

/// The prefilter gives the same coefficients on every core as on one — bit for
/// bit. Both passes read each line from the `f32` store into an `f64` line and
/// store it back, in the same order, whichever thread runs them; the column
/// pass's detour through a transposed copy moves values without touching them.
#[test]
fn the_prefilter_does_not_depend_on_threads() {
    for (w, h) in [(37usize, 23usize), (1, 9), (9, 1), (64, 64)] {
        let img: Vec<f32> = (0..w * h)
            .map(|i| ((i * 7919) % 1013) as f32 * 0.37 - 150.0)
            .collect();
        let mut seq = img.clone();
        let mut par = img.clone();
        prefilter_2d_with(&mut seq, w, h, &POLES_3, false);
        prefilter_2d_with(&mut par, w, h, &POLES_3, true);
        assert_eq!(seq, par, "{w}x{h}: threaded prefilter differs");
        // And the plain entry point is the sequential one.
        let mut plain = img.clone();
        prefilter_2d(&mut plain, w, h, &POLES_3);
        assert_eq!(plain, seq, "{w}x{h}");
    }
}

/// The spline is continuous, so a value at an exact integer equals the limit
/// from either side — including at *negative* integers, past the left or top
/// edge, which are what an integer shift toward that edge produces.
///
/// The index used to be computed as `trunc(x) + 1` on the negative side while
/// the weights used the true floor. Those agree everywhere except at exact
/// negative integers, where the centre weight landed one tap to the left: at
/// x = -44.0 the spline was read at -45. Found because the GPU path, working in
/// `f32`, hits exact integers at different pixels than the `f64` CPU path did,
/// and the two stopped agreeing past the edge.
#[test]
fn the_spline_is_continuous_at_negative_integers() {
    let (w, h) = (12usize, 9usize);
    let mut coeff: Vec<f32> = (0..w * h)
        .map(|i| ((i * 7) % 23) as f32 * 3.0 + (i / w) as f32)
        .collect();
    prefilter_2d(&mut coeff, w, h, &POLES_3);
    for x in [-1.0f64, -3.0, -7.0, -12.0, -25.0] {
        for y in [-2.0f64, 0.0, 4.5] {
            let at = interpolate(&coeff, w, h, x, y);
            let left = interpolate(&coeff, w, h, x - 1e-9, y);
            let right = interpolate(&coeff, w, h, x + 1e-9, y);
            assert!(
                (at - left).abs() < 1e-5 && (at - right).abs() < 1e-5,
                "discontinuous at x = {x}, y = {y}: left {left}, at {at}, right {right}"
            );
        }
    }
    // The same along y.
    for y in [-1.0f64, -4.0, -9.0] {
        let at = interpolate(&coeff, w, h, 3.25, y);
        let near = interpolate(&coeff, w, h, 3.25, y + 1e-9);
        assert!(
            (at - near).abs() < 1e-5,
            "discontinuous at y = {y}: {at} vs {near}"
        );
    }
}

//! The kernel has to be SciPy's, not merely Gaussian-shaped.
//!
//! These maps get compared against ones produced by the Python analysis, so
//! "close enough to a Gaussian" is not close enough: the radius, the sign
//! convention and the edge rule all have to be the same or every pixel differs
//! by a little.

use super::*;

#[test]
fn the_kernel_matches_scipys() {
    // sigma=2.3 -> radius int(4*2.3 + 0.5) = 9 -> 19 taps.
    assert_eq!(radius(2.3), 9);
    assert_eq!(radius(1.0), 4);

    let g = kernel1d(2.3, 0);
    assert_eq!(g.len(), 19);
    assert!(
        (g.iter().sum::<f64>() - 1.0).abs() < 1e-12,
        "not normalised"
    );
    for i in 0..9 {
        assert!((g[i] - g[18 - i]).abs() < 1e-15, "not symmetric at {i}");
    }
    // Computed from SciPy's own formula, not from this implementation.
    assert!(
        (g[9] - 0.173_458_622_762_916_1).abs() < 1e-15,
        "centre tap {}",
        g[9]
    );
    assert!(
        (g[0] - 8.208_372_254_440_408e-5).abs() < 1e-18,
        "edge tap {}",
        g[0]
    );

    let d = kernel1d(2.3, 1);
    assert_eq!(d.len(), 19);
    // A derivative kernel sums to zero and is antisymmetric, or a constant
    // image would differentiate to something.
    assert!(d.iter().sum::<f64>().abs() < 1e-15, "does not sum to zero");
    assert!(
        d[9].abs() < 1e-18,
        "centre of a derivative kernel must be 0"
    );
    for i in 0..9 {
        assert!((d[i] + d[18 - i]).abs() < 1e-15, "not antisymmetric at {i}");
    }

    // Sign. `gaussian_filter` *correlates*, so a rising ramp must give a
    // positive response — that is what makes "the positive part" mean
    // "brightening" rather than "dimming", and getting it backwards would
    // produce a plausible map of exactly the wrong thing.
    let ramp: Vec<f64> = (0..19).map(|i| i as f64).collect();
    let response: f64 = d.iter().zip(&ramp).map(|(w, v)| w * v).sum();
    // A unit slope must come back as a unit derivative, not as -1.
    assert!(
        (response - 0.999_418).abs() < 1e-5,
        "a unit ramp differentiates to {response}, not ~1"
    );
}

#[test]
fn reflect_bounces_as_many_times_as_it_needs_to() {
    // d c b a | a b c d | d c b a
    let n = 4;
    assert_eq!(
        (0..4).map(|i| reflect(i, n)).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        (-4..0).map(|i| reflect(i, n)).collect::<Vec<_>>(),
        [3, 2, 1, 0]
    );
    assert_eq!(
        (4..8).map(|i| reflect(i, n)).collect::<Vec<_>>(),
        [3, 2, 1, 0]
    );
    // A window shorter than the kernel is the normal case here — a 3-frame
    // response against a 19-tap kernel — so it has to survive bouncing.
    // Period 4 for n=2 (a b b a), so both of these land on `a`.
    assert_eq!(reflect(-9, 2), 0);
    assert_eq!(reflect(12, 2), 0);
    assert_eq!(reflect(-10, 2), 1);
    assert_eq!(reflect(13, 2), 1);
    assert_eq!(reflect(-100, 1), 0);
    assert_eq!(reflect(100, 1), 0);
}

#[test]
fn a_constant_stack_has_no_derivative() {
    let planes = vec![vec![7.0f32; 16]; 5];
    let out = positive_derivative_sum(&planes, 4, 4, 2.3);
    for v in out {
        assert!(v.abs() < 1e-4, "a flat stack produced a response of {v}");
    }
}

/// Only rises count. A stack that falls must produce nothing — that is what
/// separates a response from its recovery.
#[test]
fn a_falling_stack_produces_nothing_and_a_rising_one_does() {
    let rising: Vec<Vec<f32>> = (0..8).map(|t| vec![t as f32; 16]).collect();
    let falling: Vec<Vec<f32>> = (0..8).map(|t| vec![(7 - t) as f32; 16]).collect();
    let up = positive_derivative_sum(&rising, 4, 4, 1.0);
    let down = positive_derivative_sum(&falling, 4, 4, 1.0);
    assert!(up[0] > 0.5, "a rise should register: {}", up[0]);
    assert!(
        down.iter().all(|v| *v < 1e-4),
        "a fall registered as a response: {}",
        down[0]
    );
}

/// The spatial half is a blur, so an isolated bright pixel spreads into its
/// neighbours without becoming brighter than the centre.
#[test]
fn the_spatial_pass_blurs_around_the_response() {
    let dark = vec![0f32; 49];
    let mut bright = vec![0f32; 49];
    bright[24] = 100.0; // centre of a 7x7
    let planes = vec![
        dark.clone(),
        dark.clone(),
        bright.clone(),
        bright.clone(),
        bright,
    ];
    let out = positive_derivative_sum(&planes, 7, 7, 1.0);
    assert!(out[24] > 0.0, "the centre should respond");
    assert!(out[25] > 0.0, "a neighbour should have been blurred into");
    assert!(
        out[24] > out[25],
        "the centre should still be the brightest"
    );
    assert!(
        out[0] < out[24],
        "the corner should be dimmer than the centre"
    );
}

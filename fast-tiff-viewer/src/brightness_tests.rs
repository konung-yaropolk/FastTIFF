//! The gain curve, which is the only part of this with arithmetic to get wrong.

use super::*;

/// The property the whole thing exists for: a volume at the default density
/// looks exactly as it did before any of this. If the gain were not 1 there,
/// opening a file would silently change its appearance — which is a worse
/// problem than the one being fixed.
#[test]
fn the_default_density_is_left_alone() {
    assert!(
        (autolevel(DENSITY_DEFAULT) - 1.0).abs() < 1e-6,
        "gain at the default should be exactly 1, got {}",
        autolevel(DENSITY_DEFAULT)
    );
}

/// Below the default the volume is thinner and therefore darker, so the gain
/// goes **up**. A sign error here would make the dark end darker still, which
/// is the complaint this was written for, doubled.
#[test]
fn a_thinner_volume_is_brightened_and_a_denser_one_is_not() {
    for low in [1.0f32, 5.0, 20.0, 50.0] {
        assert!(
            autolevel(low) > 1.0,
            "density {low} is below the default and should be brightened, got {}",
            autolevel(low)
        );
    }
    // Above the default the accumulation is already saturating, so the gain
    // comes down — but only a little, because there is little left to take: at
    // the default the volume is already most of the way to opaque.
    for high in [200.0f32, 500.0, 1000.0] {
        let g = autolevel(high);
        assert!(g <= 1.0, "density {high} should not be brightened, got {g}");
        assert!(g > 0.9, "density {high} has little left to give back: {g}");
    }
}

/// Monotonic, so dragging the density never reverses the brightness trend.
/// Without this the slider would have a hump in it, which is harder to use than
/// a slider with no compensation at all.
#[test]
fn the_gain_falls_monotonically_as_density_rises() {
    let mut last = f32::INFINITY;
    for d in [
        0.5f32, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0,
    ] {
        let g = autolevel(d);
        assert!(
            g <= last + 1e-6,
            "gain rose at density {d}: {g} after {last}"
        );
        last = g;
    }
}

/// The compensation halves the variation on an ordinary volume. This is the
/// claim the module makes, stated as arithmetic: the brightness a ray
/// accumulates is `1 - exp(-density * thickness)`, so the compensated product
/// should vary markedly less across the slider than the raw accumulation does.
#[test]
fn the_compensated_brightness_varies_much_less_on_a_typical_volume() {
    // The thickness the module assumes, which is the case it is tuned for.
    let accumulated = |d: f32| 1.0 - (-d * 0.2f32).exp();
    let densities = [1.0f32, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0];

    let spread = |f: &dyn Fn(f32) -> f32| {
        let v: Vec<f32> = densities.iter().map(|&d| f(d)).collect();
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        hi / lo.max(1e-6)
    };
    let before = spread(&|d| accumulated(d));
    let after = spread(&|d| accumulated(d) * autolevel(d));
    assert!(
        before > 4.0,
        "the raw accumulation should vary a lot, or there is nothing to fix: {before}x"
    );
    assert!(
        after < before / 2.0,
        "compensation should at least halve the variation: {before}x became {after}x"
    );
    assert!(after < 2.5, "and should leave it small: {after}x");
}

/// On a volume much **more opaque** than typical the correction overshoots,
/// because such a volume was already near-saturated across the whole slider and
/// is being corrected as though it were not.
///
/// This test exists to pin that it stays *bounded* rather than to pretend it
/// does not happen. It is the cost of a correction that cannot measure the
/// volume it is correcting, it is why [`MAX_GAIN`] is 2.5 rather than large
/// enough to flatten the typical case completely, and it is why the manual gain
/// exists beside the automatic one. Anyone tempted to raise the cap should read
/// the table on `MAX_GAIN` first — this is the row that gets worse.
#[test]
fn an_unusually_opaque_volume_overshoots_but_stays_bounded() {
    let accumulated = |d: f32| 1.0 - (-d * 2.0f32).exp();
    let v: Vec<f32> = [1.0f32, 10.0, 100.0, 1000.0]
        .iter()
        .map(|&d| accumulated(d) * autolevel(d))
        .collect();
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let spread = hi / lo.max(1e-6);
    assert!(
        spread > 1.5,
        "the overshoot is real and should be visible here"
    );
    assert!(
        spread < 3.0,
        "but bounded by the cap: {spread}x across the slider"
    );
}

/// On a volume much **thinner** than typical the correction helps but cannot
/// finish the job, for the same reason in the other direction.
#[test]
fn an_unusually_thin_volume_is_helped_but_not_flattened() {
    let accumulated = |d: f32| 1.0 - (-d * 0.02f32).exp();
    let densities = [1.0f32, 10.0, 100.0, 1000.0];
    let raw: Vec<f32> = densities.iter().map(|&d| accumulated(d)).collect();
    let fixed: Vec<f32> = densities
        .iter()
        .map(|&d| accumulated(d) * autolevel(d))
        .collect();
    let spread = |v: &[f32]| {
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        hi / lo.max(1e-6)
    };
    assert!(
        spread(&fixed) < spread(&raw),
        "should still be an improvement: {}x became {}x",
        spread(&raw),
        spread(&fixed)
    );
}

/// Bounded. At a density approaching zero the volume genuinely has nothing in
/// it, and the compensation needed to hold brightness constant runs away — so
/// the cap is where honesty takes over, rather than multiplying noise by twenty.
#[test]
fn the_gain_is_capped_at_both_ends() {
    for d in [1e-6f32, 0.001, 0.01] {
        let g = autolevel(d);
        assert!(g.is_finite(), "density {d} gave {g}");
        assert!(g <= 2.5 + 1e-6, "density {d} gave {g}, over the cap");
    }
    for d in [1e9f32, f32::MAX] {
        let g = autolevel(d);
        assert!(g.is_finite() && g >= 0.4 - 1e-6, "density {d} gave {g}");
    }
}

/// A density that is not a number cannot produce a gain that is not a number:
/// a NaN gain multiplies the whole render to nothing and persists until the
/// slider is touched, which looks like a broken file rather than a broken
/// setting.
#[test]
fn a_density_that_is_not_a_number_yields_no_gain_rather_than_a_nan() {
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -1.0,
        -100.0,
    ] {
        let g = autolevel(bad);
        assert!(g.is_finite(), "density {bad} gave {g}");
        assert!(
            g > 0.0,
            "density {bad} gave {g}, which would black the render"
        );
    }
    assert_eq!(autolevel(f32::NAN), 1.0, "nothing to compensate for");
}

// The max-intensity default's range is asserted at compile time in the module
// itself — see the `const _: ()` beside `MIP_DEFAULT`. A runtime test of two
// constants only moves the failure later.

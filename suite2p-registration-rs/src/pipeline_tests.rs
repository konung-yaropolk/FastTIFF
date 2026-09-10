//! Batching, backends, normalisation and bad-frame flagging.

use super::*;

#[test]
fn the_percentile_range_ignores_a_hot_pixel() {
    // 100 values 0..99, plus one saturated speck. Min/max would give (0, 9999);
    // the 1st/99th percentile keeps the picture's own range.
    let mut v: Vec<f32> = (0..100).map(|i| i as f32).collect();
    v.push(9999.0);
    let (lo, hi) = percentile_range(&v);
    assert!(lo <= 1.0, "lo {lo}");
    assert!(hi < 200.0, "a hot pixel set the top of the range: {hi}");
}

#[test]
fn an_empty_image_has_an_unbounded_range() {
    let (lo, hi) = percentile_range(&[]);
    assert_eq!((lo, hi), (f32::NEG_INFINITY, f32::INFINITY));
}

// ------------------------------------------------------- temporal smoothing

#[test]
fn zero_sigma_leaves_the_maps_alone() {
    let mut maps = vec![vec![1.0f32, 2.0], vec![5.0, 6.0]];
    let before = maps.clone();
    smooth_in_time(&mut maps, 0.0);
    assert_eq!(maps, before);
}

/// A spike in one frame's map is spread into its neighbours — which is the
/// point: a frame too dim to locate on its own borrows its neighbours' peak.
#[test]
fn smoothing_spreads_a_peak_into_neighbouring_frames() {
    // Nine frames with the spike in the middle. Long enough that the kernel
    // (radius 4 at sigma 1) fits without the reflection folding back on itself
    // — with a series shorter than the kernel, reflect padding samples the same
    // frames repeatedly and the total is legitimately not preserved.
    let mut maps: Vec<Vec<f32>> = (0..9)
        .map(|t| vec![if t == 4 { 10.0 } else { 0.0 }])
        .collect();
    smooth_in_time(&mut maps, 1.0);

    assert!(
        maps[3][0] > 1.0,
        "the peak did not reach frame 3: {:?}",
        maps[3]
    );
    assert!(maps[5][0] > 1.0, "nor frame 5: {:?}", maps[5]);
    assert!(maps[4][0] < 10.0, "and frame 4 gave some away");
    // It falls off with distance, rather than being spread flat.
    assert!(
        maps[3][0] > maps[2][0] && maps[2][0] > maps[1][0],
        "{maps:?}"
    );
    // Nothing is created: away from the edges the total is preserved.
    let total: f32 = maps.iter().map(|m| m[0]).sum();
    assert!((total - 10.0).abs() < 0.5, "total {total}");
}

/// Smoothing must not move a peak that every frame already agrees on.
#[test]
fn smoothing_a_constant_run_changes_nothing() {
    let mut maps = vec![vec![4.0f32]; 5];
    smooth_in_time(&mut maps, 2.0);
    for m in &maps {
        assert!((m[0] - 4.0).abs() < 1e-4, "{m:?}");
    }
}

// ------------------------------------------------------------- bad frames

fn shifts(v: &[(i32, i32, f32)]) -> Vec<Shift> {
    v.iter()
        .map(|&(dy, dx, corr)| Shift { dy, dx, corr })
        .collect()
}

#[test]
fn a_steady_recording_has_no_bad_frames() {
    let s = shifts(&[(0, 0, 1.0); 20]);
    let bad = bad_frames(&s, 512, 512, &Settings::default());
    assert!(!bad.iter().any(|&b| b), "a still movie has no outliers");
}

/// A frame that hit the shift limit is bad: the real shift was larger and got
/// clipped, so the number reported is not the motion.
#[test]
fn a_frame_at_the_shift_limit_is_flagged() {
    let mut v = vec![(0i32, 0i32, 1.0f32); 20];
    // maxregshift 0.1 of 512 is 51.2; 0.95 of that is 48.6.
    v[7] = (0, 50, 1.0);
    let bad = bad_frames(&shifts(&v), 512, 512, &Settings::default());
    assert!(bad[7], "a clipped shift should be flagged");
    assert!(!bad[6] && !bad[8], "only that frame");
}

#[test]
fn an_empty_recording_flags_nothing() {
    assert!(bad_frames(&[], 512, 512, &Settings::default()).is_empty());
}

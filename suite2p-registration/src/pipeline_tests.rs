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

/// The parallel apply gives exactly what a serial loop would.
///
/// Exactly, not nearly: `shift_frame` is a permutation of the pixels, so there
/// is no arithmetic to reorder and no excuse for a difference. This is what
/// makes it safe to offer the backend as a choice rather than a trade.
#[test]
fn apply_batch_matches_a_serial_loop() {
    let (ly, lx) = (16, 12);
    let planes: Vec<Vec<f32>> = (0..7)
        .map(|k| (0..ly * lx).map(|i| (i * 7 + k * 13) as f32).collect())
        .collect();
    let shifts: Vec<Shift> = [(0, 0), (2, -3), (-1, 4)]
        .iter()
        .map(|&(dy, dx)| Shift { dy, dx, corr: 1.0 })
        .collect();
    // Three shifts, seven planes: two channels' worth of some frames, which is
    // the arrangement `frame_of` exists for.
    let frame_of = [0usize, 0, 1, 1, 2, 2, 0];

    let expected: Vec<Vec<f32>> = planes
        .iter()
        .zip(&frame_of)
        .map(|(p, &t)| crate::rigid::shift_frame(p, ly, lx, shifts[t].dy, shifts[t].dx))
        .collect();

    for backend in Backend::all() {
        let settings = Settings {
            backend,
            ..Settings::default()
        };
        let mut got = planes.clone();
        super::apply_batch(&mut got, ly, lx, &frame_of, &shifts, None, &settings);
        assert_eq!(got, expected, "{} disagreed", backend.label());
    }
}

/// Every plane of one timepoint moves by that timepoint's shift.
///
/// The guarantee a two-colour recording depends on: measure on one channel,
/// move them all together. A per-plane measurement would let the channels
/// drift apart, and then nothing measured from their ratio means anything.
#[test]
fn every_channel_of_a_frame_moves_together() {
    let (ly, lx) = (8, 8);
    let mut planes: Vec<Vec<f32>> = (0..4)
        .map(|k| {
            let mut p = vec![0.0f32; ly * lx];
            // A single bright pixel per plane, all in the same place.
            p[3 * lx + 3] = 1.0 + k as f32;
            p
        })
        .collect();
    // Two frames, two channels each.
    let frame_of = [0usize, 0, 1, 1];
    let shifts = vec![
        Shift {
            dy: 1,
            dx: 2,
            corr: 1.0,
        },
        Shift {
            dy: -2,
            dx: 0,
            corr: 1.0,
        },
    ];
    super::apply_batch(
        &mut planes,
        ly,
        lx,
        &frame_of,
        &shifts,
        None,
        &Settings::default(),
    );

    let bright = |p: &Vec<f32>| {
        p.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| (i / lx, i % lx))
            .unwrap()
    };
    assert_eq!(bright(&planes[0]), bright(&planes[1]), "frame 0's channels");
    assert_eq!(bright(&planes[2]), bright(&planes[3]), "frame 1's channels");
    assert_ne!(
        bright(&planes[0]),
        bright(&planes[2]),
        "the two frames had different shifts and should not have landed together"
    );
}

/// A plane with no measurement behind it is left alone rather than moved by
/// somebody else's shift.
#[test]
fn a_plane_without_a_measurement_is_untouched() {
    let (ly, lx) = (4, 4);
    let original: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut planes = vec![original.clone()];
    super::apply_batch(
        &mut planes,
        ly,
        lx,
        &[9],
        &[Shift {
            dy: 1,
            dx: 1,
            corr: 1.0,
        }],
        None,
        &Settings::default(),
    );
    assert_eq!(planes[0], original);
}

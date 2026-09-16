//! Rigid registration, against shifts that were put there on purpose.
//!
//! The test that matters: take a frame, move it by a known amount, and check
//! the algorithm reports exactly that amount back. A registration that is
//! subtly wrong produces a plausible-looking movie of the wrong tissue.

use super::*;
use crate::masks::reference_filters;

/// A frame with a few bright blobs — something with structure to lock onto.
/// Flat noise has no features and would register to anywhere.
fn blobs(ly: usize, lx: usize) -> Vec<f32> {
    let mut f = vec![10.0f32; ly * lx];
    for (cy, cx, amp) in [
        (20usize, 24usize, 200.0f32),
        (40, 44, 150.0),
        (30, 12, 120.0),
    ] {
        for y in 0..ly {
            for x in 0..lx {
                let d2 = ((y as f32 - cy as f32).powi(2) + (x as f32 - cx as f32).powi(2)) / 8.0;
                f[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    f
}

fn find(reference: &[f32], frame: &[f32], ly: usize, lx: usize) -> Shift {
    let mut fft = Fft2::new(ly, lx);
    // A taper of 5 rather than the default 40: the default is sized for a
    // 512-pixel two-photon frame, and on a 64-pixel test frame it would fade
    // the whole picture out. The algorithm is what is under test, not the
    // suitability of a default to a size nobody records at.
    let filters = reference_filters(&mut fft, reference, ly, lx, 5.0, 1.15);
    phase_correlate(&mut fft, &filters, frame, 0.1)
}

#[test]
fn a_frame_matched_against_itself_reports_no_shift() {
    let (ly, lx) = (64, 64);
    let r = blobs(ly, lx);
    let s = find(&r, &r, ly, lx);
    assert_eq!((s.dy, s.dx), (0, 0), "{s:?}");
}

/// The headline behaviour: a known displacement comes back exactly.
#[test]
fn a_known_shift_is_recovered_exactly() {
    let (ly, lx) = (64, 64);
    let r = blobs(ly, lx);
    for (dy, dx) in [(3i32, 0i32), (0, 4), (-5, 2), (2, -6), (-3, -3)] {
        // `shift_frame` with (dy, dx) moves content by (-dy, -dx), so a frame
        // built this way is one the algorithm should answer (dy, dx) for.
        let moved = shift_frame(&r, ly, lx, -dy, -dx);
        let s = find(&r, &moved, ly, lx);
        assert_eq!(
            (s.dy, s.dx),
            (dy, dx),
            "a frame displaced by ({dy},{dx}) reported {s:?}"
        );
    }
}

/// And the shift undoes: applying what was found puts the frame back.
#[test]
fn applying_the_found_shift_restores_the_frame() {
    let (ly, lx) = (64, 64);
    let r = blobs(ly, lx);
    let moved = shift_frame(&r, ly, lx, -4, 3);
    let s = find(&r, &moved, ly, lx);
    let back = shift_frame(&moved, ly, lx, s.dy, s.dx);
    // Compare away from the border, where the wrap brings in the far side.
    for y in 10..ly - 10 {
        for x in 10..lx - 10 {
            let (a, b) = (back[y * lx + x], r[y * lx + x]);
            assert!((a - b).abs() < 1e-3, "({y},{x}): {a} vs {b}");
        }
    }
}

/// The search is bounded. A displacement past `maxregshift` cannot be reported,
/// which is the point — a wild answer is worse than a clipped one.
#[test]
fn a_shift_is_never_reported_beyond_the_limit() {
    let (ly, lx) = (64, 64);
    let r = blobs(ly, lx);
    // 0.1 of 64 rounds to 6, so nothing may exceed 6.
    let moved = shift_frame(&r, ly, lx, -20, 0);
    let s = find(&r, &moved, ly, lx);
    assert!(
        s.dy.abs() <= 6 && s.dx.abs() <= 6,
        "{s:?} exceeded the limit"
    );
}

/// A better match correlates higher — that ordering is what picks the frames a
/// reference is built from, so it has to hold.
#[test]
fn a_closer_frame_correlates_higher() {
    let (ly, lx) = (64, 64);
    let r = blobs(ly, lx);
    let exact = find(&r, &r, ly, lx);
    let moved = find(&r, &shift_frame(&r, ly, lx, -5, 5), ly, lx);
    let mut noise = r.clone();
    for (i, v) in noise.iter_mut().enumerate() {
        *v += ((i * 7919) % 101) as f32;
    }
    let noisy = find(&r, &noise, ly, lx);
    assert!(exact.corr >= moved.corr, "{exact:?} vs {moved:?}");
    assert!(exact.corr > noisy.corr, "{exact:?} vs {noisy:?}");
}

// ---------------------------------------------------------------- shifting

#[test]
fn shifting_wraps_rather_than_filling() {
    let (ly, lx) = (2, 3);
    // 0 1 2
    // 3 4 5
    let f: Vec<f32> = (0..6).map(|i| i as f32).collect();
    // dx = 1 takes the column to the right, so column 0 becomes old column 1.
    assert_eq!(
        shift_frame(&f, ly, lx, 0, 1),
        vec![1.0, 2.0, 0.0, 4.0, 5.0, 3.0]
    );
    // And dy = 1 takes the row below, wrapping.
    assert_eq!(
        shift_frame(&f, ly, lx, 1, 0),
        vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn a_zero_shift_changes_nothing() {
    let (ly, lx) = (4, 5);
    let f: Vec<f32> = (0..20).map(|i| i as f32).collect();
    assert_eq!(shift_frame(&f, ly, lx, 0, 0), f);
}

#[test]
fn shifting_by_the_frame_size_is_the_identity() {
    let (ly, lx) = (4, 5);
    let f: Vec<f32> = (0..20).map(|i| i as f32).collect();
    assert_eq!(shift_frame(&f, ly, lx, ly as i32, lx as i32), f);
    // And a negative shift is the mirror of the positive one.
    assert_eq!(
        shift_frame(&shift_frame(&f, ly, lx, 2, 3), ly, lx, -2, -3),
        f
    );
}

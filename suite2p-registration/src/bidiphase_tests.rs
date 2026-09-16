//! Bidirectional phase, against combs put in on purpose.

use super::*;

/// A frame whose odd lines are *displaced* right by `displaced_by` — what a
/// mis-timed resonant scanner produces.
///
/// Note the direction. [`compute`] returns the **correction**, not the
/// displacement: suite2p's pipeline is `b = compute(frames); shift(frames, b)`
/// (register.py:924-931), so the value it hands back is the one that undoes the
/// comb. For a comb displaced right by `k`, that is `-k`.
fn combed(ly: usize, lx: usize, displaced_by: i32) -> Vec<f32> {
    let offset = displaced_by;
    // A vertical-edge pattern, so a horizontal displacement is visible at all.
    let value = |x: i32| -> f32 {
        let x = x.rem_euclid(lx as i32) as usize;
        if (10..20).contains(&x) || (30..34).contains(&x) {
            200.0
        } else {
            10.0
        }
    };
    let mut f = vec![0.0f32; ly * lx];
    for y in 0..ly {
        for x in 0..lx {
            // Odd lines are drawn shifted right by `offset`.
            let sx = if y % 2 == 1 {
                x as i32 - offset
            } else {
                x as i32
            };
            f[y * lx + x] = value(sx);
        }
    }
    f
}

#[test]
fn an_uncombed_frame_reports_no_offset() {
    let (ly, lx) = (32, 64);
    assert_eq!(compute(&[combed(ly, lx, 0)], ly, lx), 0);
}

/// The headline behaviour: a comb of a known size yields the correction that
/// undoes it.
#[test]
fn a_known_comb_yields_the_correction_that_undoes_it() {
    let (ly, lx) = (32, 64);
    for displaced_by in [-4i32, -2, -1, 1, 2, 3, 5] {
        let frames = vec![combed(ly, lx, displaced_by)];
        let got = compute(&frames, ly, lx);
        assert_eq!(
            got, -displaced_by,
            "lines displaced by {displaced_by} need a correction of {}, got {got}",
            -displaced_by
        );
    }
}

/// And undoing it removes the comb: after shifting, odd and even lines agree.
#[test]
fn applying_the_offset_removes_the_comb() {
    let (ly, lx) = (32, 64);
    let displaced_by = 3;
    let mut f = combed(ly, lx, displaced_by);
    let correction = compute(&[f.clone()], ly, lx);
    assert_eq!(correction, -displaced_by);
    // Applied as suite2p applies it: `shift(frames, compute(frames))`.
    shift(&mut f, ly, lx, correction);

    // Compare an odd line with the even one above it, away from the edges the
    // slide leaves untouched.
    for y in (1..ly).step_by(2) {
        for x in 10..lx - 10 {
            let (odd, even) = (f[y * lx + x], f[(y - 1) * lx + x]);
            assert!(
                (odd - even).abs() < 1e-3,
                "line {y} col {x}: {odd} vs {even}"
            );
        }
    }
}

#[test]
fn shifting_by_zero_changes_nothing() {
    let (ly, lx) = (4, 6);
    let original: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut f = original.clone();
    shift(&mut f, ly, lx, 0);
    assert_eq!(f, original);
}

/// Even lines are never touched — only every other line is displaced by the
/// scanner, and moving both would be moving the whole frame.
#[test]
fn shifting_leaves_the_even_lines_alone() {
    let (ly, lx) = (4, 6);
    let original: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut f = original.clone();
    shift(&mut f, ly, lx, 2);
    for y in (0..ly).step_by(2) {
        assert_eq!(
            &f[y * lx..(y + 1) * lx],
            &original[y * lx..(y + 1) * lx],
            "even line {y} was moved"
        );
    }
    // And an odd line really did move: row 1 was 6,7,8,9,10,11.
    assert_eq!(&f[lx..2 * lx], &[6.0, 7.0, 6.0, 7.0, 8.0, 9.0]);
}

/// An offset past the frame width would address outside it. Refusing is right;
/// panicking in the middle of a registration is not.
#[test]
fn an_absurd_offset_is_ignored_rather_than_panicking() {
    let (ly, lx) = (4, 6);
    let original: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut f = original.clone();
    shift(&mut f, ly, lx, 100);
    assert_eq!(f, original);
    shift(&mut f, ly, lx, -100);
    assert_eq!(f, original);
}

#[test]
fn an_empty_or_tiny_movie_reports_no_offset() {
    assert_eq!(compute(&[], 16, 16), 0);
    assert_eq!(compute(&[vec![0.0; 16]], 1, 16), 0);
}

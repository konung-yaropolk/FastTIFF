//! The parallel primitives, against the serial loops they replace.
//!
//! Every test runs at two lengths: one under [`FLOOR`], which takes the
//! serial path, and one well over it, which takes the parallel one. That is
//! the whole point — the two paths are different code, and a helper that is
//! correct only below the threshold would pass a test suite built on small
//! fixtures and be wrong on every real stack.

use super::*;

/// Both sides of the threshold, so neither path is ever untested.
const LENGTHS: [usize; 4] = [0, 1, 1000, FLOOR * 3 + 7];

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 97) as f32 - 48.0).collect()
}

#[test]
fn each_touches_every_element_exactly_once() {
    for n in LENGTHS {
        let mut v = ramp(n);
        let want: Vec<f32> = v.iter().map(|x| x * 2.0 + 1.0).collect();
        each(&mut v, |x| *x = *x * 2.0 + 1.0);
        assert_eq!(v, want, "n = {n}");
    }
}

#[test]
fn zip_pairs_by_index() {
    for n in LENGTHS {
        let mut a = ramp(n);
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // Not a symmetric operation: `a - b` would catch a pairing that was
        // off by one, where `a + b` on a ramp might not.
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        zip(&mut a, &b, |x, y| *x -= *y);
        assert_eq!(a, want, "n = {n}");
    }
}

#[test]
fn zip3_pairs_all_three_by_index() {
    for n in LENGTHS {
        let mut a = ramp(n);
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let c: Vec<f32> = (0..n).map(|i| (i * i % 13) as f32).collect();
        let want: Vec<f32> = a
            .iter()
            .zip(&b)
            .zip(&c)
            .map(|((x, y), z)| x * 2.0 - y + z * 3.0)
            .collect();
        zip3(&mut a, &b, &c, |x, y, z| *x = *x * 2.0 - *y + *z * 3.0);
        assert_eq!(a, want, "n = {n}");
    }
}

#[test]
fn sum_by_agrees_with_the_serial_sum() {
    for n in LENGTHS {
        let v = ramp(n);
        let serial: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let got = sum_by(&v, |&x| (x as f64) * (x as f64));
        // Not bit-equal, and not meant to be: a chunked reduction is a
        // different (shallower) tree. Close to the limit of f64 over this many
        // terms is the right claim.
        assert!(
            (got - serial).abs() <= 1e-9 * serial.abs().max(1.0),
            "n = {n}: {got} vs {serial}"
        );
    }
}

/// The reason the chunking is there: the same input must give the same answer
/// every time, however the work happened to be shared out.
#[test]
fn a_reduction_is_reproducible() {
    let v = ramp(FLOOR * 3 + 7);
    let first = sum_by(&v, |&x| x as f64);
    for _ in 0..20 {
        assert_eq!(sum_by(&v, |&x| x as f64), first);
    }
}

#[test]
fn zip_sum_both_mutates_and_measures() {
    for n in LENGTHS {
        let mut a = ramp(n);
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let want_sum: f64 = a.iter().zip(&b).map(|(x, y)| (x * y) as f64).sum();
        let want: Vec<f32> = a.iter().map(|x| -x).collect();

        let got = zip_sum(&mut a, &b, |x, y| {
            let m = (*x * *y) as f64;
            *x = -*x;
            m
        });
        assert_eq!(a, want, "n = {n}: the mutation");
        assert!(
            (got - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0),
            "n = {n}: {got} vs {want_sum}"
        );
    }
}

#[test]
fn zip_min_finds_the_smallest() {
    for n in LENGTHS {
        let a = ramp(n);
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 1.0).collect();
        let want = a
            .iter()
            .zip(&b)
            .fold(f32::INFINITY, |m: f32, (x, y)| m.min(x / y));
        let got = zip_min(&a, &b, f32::INFINITY, |x, y| x / y);
        assert_eq!(got, want, "n = {n}");
    }
}

/// The MRNSD case: only some pairs are candidates, and the rest must not drag
/// the answer down.
#[test]
fn zip_min_skips_what_the_closure_excludes() {
    let n = FLOOR * 2;
    let a: Vec<f32> = (0..n)
        .map(|i| if i == 12345 { -4.0 } else { 1.0 })
        .collect();
    let b = vec![2.0f32; n];
    let got = zip_min(&a, &b, f32::INFINITY, |x, y| {
        if *x < 0.0 {
            -*y / *x
        } else {
            f32::INFINITY
        }
    });
    assert_eq!(got, 0.5, "the one candidate pair should decide it");

    // And with no candidates at all, the starting value survives.
    let none = zip_min(&b, &b, 7.0, |_, _| f32::INFINITY);
    assert_eq!(none, 7.0);
}

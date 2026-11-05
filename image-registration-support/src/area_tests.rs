// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The search on masks it can be checked against by brute force.

use super::*;

/// A small deterministic generator, so a failure names its own inputs.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn brute_largest(mask: &[bool], rows: usize, cols: usize) -> usize {
    let mut best = 0;
    for y0 in 0..rows {
        for x0 in 0..cols {
            for y1 in y0..rows {
                for x1 in x0..cols {
                    if (y0..=y1).all(|y| (x0..=x1).all(|x| mask[y * cols + x])) {
                        best = best.max((y1 - y0 + 1) * (x1 - x0 + 1));
                    }
                }
            }
        }
    }
    best
}

#[test]
fn the_largest_rectangle_is_the_largest() {
    let mut rng = Lcg(5);
    for case in 0..300 {
        let (rows, cols) = (1 + rng.next() as usize % 8, 1 + rng.next() as usize % 9);
        let mask: Vec<bool> = (0..rows * cols)
            .map(|_| !rng.next().is_multiple_of(5))
            .collect();
        let want = brute_largest(&mask, rows, cols);
        match largest_rectangle(&mask, rows, cols) {
            None => assert_eq!(want, 0, "case {case}"),
            Some(r) => {
                assert_eq!(r.height * r.width, want, "case {case}: {r:?}");
                let all = (r.row..r.row + r.height)
                    .all(|y| (r.col..r.col + r.width).all(|x| mask[y * cols + x]));
                assert!(all, "case {case}: {r:?} is not all covered");
            }
        }
    }
}

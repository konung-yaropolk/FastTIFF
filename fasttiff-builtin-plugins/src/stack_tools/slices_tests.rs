//! The selection arithmetic, on its own.
//!
//! This is where the off-by-one lives, so it is tested without a stack in the
//! way: [`selected`] takes the dialog's numbers already converted to indices
//! and returns the planes they name. Every case here is one someone would get
//! wrong writing it from the description.

use super::*;

/// The ordinary case, and the two edges that are easiest to drop.
#[test]
fn a_range_includes_both_of_its_ends() {
    // First 1, last 3 in the dialog -> indices 0..=2.
    assert_eq!(selected(0, 2, 1, 10), vec![0, 1, 2]);
    // The first plane, alone.
    assert_eq!(selected(0, 0, 1, 10), vec![0]);
    // The last plane, alone.
    assert_eq!(selected(9, 9, 1, 10), vec![9]);
    // The whole stack.
    assert_eq!(selected(0, 9, 1, 10), (0..10).collect::<Vec<_>>());
}

/// The increment counts from the first plane, and both ends still bound it.
///
/// The property that separates these tools from "remove a block": with an
/// increment the selection is sparse *within* the range, so Remover leaves the
/// planes between rather than deleting through them.
#[test]
fn an_increment_steps_from_the_first_and_stops_at_the_last() {
    assert_eq!(selected(0, 9, 2, 10), vec![0, 2, 4, 6, 8]);
    assert_eq!(selected(1, 9, 2, 10), vec![1, 3, 5, 7, 9]);
    assert_eq!(selected(0, 8, 3, 10), vec![0, 3, 6]);
    // The step overshoots the end rather than landing on it, which must not
    // add a plane past `last`.
    assert_eq!(selected(0, 7, 3, 10), vec![0, 3, 6]);
    // An increment larger than the range selects the first plane only.
    assert_eq!(selected(2, 4, 99, 10), vec![2]);
}

/// A range answered the wrong way round is still a range.
///
/// The dialog has two independent spinners and nothing stops someone setting
/// last below first. Treating that as empty would delete nothing, or keep
/// nothing, with no indication why.
#[test]
fn a_reversed_range_is_read_the_way_round_it_was_meant() {
    assert_eq!(selected(5, 2, 1, 10), vec![2, 3, 4, 5]);
    assert_eq!(selected(5, 2, 2, 10), vec![2, 4]);
}

/// Numbers past the end of the axis are clamped, not followed.
///
/// The declared range spans the *longest* axis on offer, because the dialog is
/// built before the axis is chosen — so choosing the shorter one hands this
/// numbers the axis does not have.
#[test]
fn numbers_past_the_end_of_the_axis_are_clamped() {
    assert_eq!(selected(0, 99, 1, 3), vec![0, 1, 2]);
    assert_eq!(selected(50, 99, 1, 3), vec![2]);
    assert_eq!(selected(0, 99, 2, 5), vec![0, 2, 4]);
}

/// An increment of zero would be an endless loop rather than an error.
#[test]
fn a_zero_increment_is_treated_as_one() {
    assert_eq!(selected(0, 3, 0, 10), vec![0, 1, 2, 3]);
}

/// A stack with no planes selects nothing, rather than indexing into it.
#[test]
fn an_empty_axis_selects_nothing() {
    assert!(selected(0, 0, 1, 0).is_empty());
}

/// Keeper and Remover partition the axis between them: every plane is in
/// exactly one of the two results.
///
/// The invariant that makes the pair trustworthy, checked over every range and
/// increment on a small axis rather than at a few hand-picked points.
#[test]
fn keeper_and_remover_partition_the_axis() {
    let depth = 7;
    for first in 0..depth {
        for last in 0..depth {
            for increment in 1..=4 {
                let kept = selected(first, last, increment, depth);
                let removed: Vec<usize> = (0..depth).filter(|i| !kept.contains(i)).collect();
                let mut union = kept.clone();
                union.extend(&removed);
                union.sort_unstable();
                assert_eq!(
                    union,
                    (0..depth).collect::<Vec<_>>(),
                    "first {first} last {last} inc {increment} lost or duplicated a plane"
                );
                assert!(
                    kept.iter().all(|k| !removed.contains(k)),
                    "first {first} last {last} inc {increment} put a plane in both"
                );
            }
        }
    }
}

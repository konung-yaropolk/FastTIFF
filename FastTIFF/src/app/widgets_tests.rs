//! Unit tests for the range slider's pure decision logic. The painting and
//! interaction halves need a live egui context; this is the part that decides
//! *what* a drag does, and the part that kept getting it wrong.

use super::*;

/// `dragged_handle` on the first frame of a drag, with no choice made yet.
fn fresh(min: f32, max: f32, v: f32, grabbed: bool) -> Option<bool> {
    dragged_handle(min, max, v, None, grabbed)
}

#[test]
fn apart_the_grabbed_handle_is_the_one_that_moves() {
    // Nothing clever when the two are separated: grab min, move min.
    assert_eq!(fresh(100.0, 900.0, 200.0, false), Some(false));
    assert_eq!(fresh(100.0, 900.0, 800.0, true), Some(true));
    // Even when the pointer has run past the other handle — from there the
    // grabbed handle pushes, rather than the other one taking over.
    assert_eq!(fresh(100.0, 900.0, 5000.0, false), Some(false));
    assert_eq!(fresh(100.0, 900.0, -5000.0, true), Some(true));
}

#[test]
fn stacked_the_pointer_side_decides_not_the_grab() {
    // Stacked at the low end, min winning the grab, dragging right: min clamps
    // against max and nothing moves. Whichever handle is caught, a pointer to
    // the right has to mean max.
    assert_eq!(
        fresh(0.0, 0.0, 50.0, false),
        Some(true),
        "grabbed min, dragging right -> max"
    );
    assert_eq!(fresh(0.0, 0.0, 50.0, true), Some(true));
    // And to the left, min — so a stack at the top of the track is no worse.
    assert_eq!(
        fresh(1000.0, 1000.0, 900.0, true),
        Some(false),
        "grabbed max, dragging left -> min"
    );
    assert_eq!(fresh(1000.0, 1000.0, 900.0, false), Some(false));
}

#[test]
fn a_committed_drag_keeps_its_handle_so_it_pushes() {
    // min has been the one moving; it has now caught up with max. It must stay
    // the one moving — that is what shoves max along — rather than handing over
    // to max and being left behind.
    assert_eq!(
        dragged_handle(500.0, 500.0, 600.0, Some(false), false),
        Some(false)
    );
    // Mirror: max driven leftwards into min keeps pushing min.
    assert_eq!(
        dragged_handle(500.0, 500.0, 400.0, Some(true), true),
        Some(true)
    );
    // And it holds even where the grabbed widget disagrees, which it will once
    // the two are stacked and the top one is taking the pointer.
    assert_eq!(
        dragged_handle(500.0, 500.0, 600.0, Some(false), true),
        Some(false)
    );
}

#[test]
fn the_choice_survives_the_pair_separating_again() {
    assert_eq!(
        dragged_handle(500.0, 700.0, 800.0, Some(true), false),
        Some(true)
    );
    assert_eq!(
        dragged_handle(300.0, 500.0, 200.0, Some(false), true),
        Some(false)
    );
}

#[test]
fn stacked_with_no_direction_yet_commits_to_nothing() {
    // Pointer still exactly on the pair: wait rather than guess, or the guess
    // gets committed for the whole drag.
    assert_eq!(fresh(500.0, 500.0, 500.0, true), None);
    assert_eq!(fresh(500.0, 500.0, 500.0, false), None);
    // Unless a choice was already made, which always wins.
    assert_eq!(
        dragged_handle(500.0, 500.0, 500.0, Some(true), false),
        Some(true)
    );
    assert_eq!(
        dragged_handle(500.0, 500.0, 500.0, Some(false), true),
        Some(false)
    );
}

#[test]
fn a_stack_at_either_edge_can_always_be_separated() {
    // The whole point: from a stack, *some* drag direction always moves
    // something. At the low edge that is rightwards, at the high edge leftwards.
    let (lo, hi) = (0.0_f32, 1000.0_f32);
    for at in [lo, hi, 500.0] {
        assert_eq!(
            fresh(at, at, at + 10.0, false),
            Some(true),
            "right from a stack at {at}"
        );
        assert_eq!(
            fresh(at, at, at - 10.0, true),
            Some(false),
            "left from a stack at {at}"
        );
    }
}

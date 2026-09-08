//! Growing the window instead of the canvas, when the bottom bar changes size.
//!
//! The rule: anything that appears in the bottom bar — the collapsible panel,
//! a status message — takes its height from the *window*, not from the image.
//! Thirty pixels is enough to put a picture that was fitted to its window into
//! pan mode, which is a strange thing for "saved it" to do.
//!
//! These drive [`PanelLayout`] a frame at a time, because the whole difficulty
//! is which frame knows what. The toggle is clicked on a frame that still draws
//! the bar in its old state, so its delta is only visible on the next one; a
//! status message is already in the height being measured when it is noticed.
//! The two need opposite "before" heights, and getting that backwards produces
//! a window that grows by nothing, or twice.

use super::{PanelLayout, GROW_WAIT};

/// One frame: the bar draws at `height` with `status`, and the window grows by
/// whatever comes back.
fn frame(p: &mut PanelLayout, status: Option<&str>, height: f32) -> Option<f32> {
    p.note_status(status.map(str::to_string), height);
    p.grow_delta(height)
}

const BARE: f32 = 40.0;
const WITH_STATUS: f32 = 58.0;

#[test]
fn a_message_appearing_grows_the_window_by_its_height() {
    let mut p = PanelLayout::default();
    assert_eq!(frame(&mut p, None, BARE), None, "nothing has changed yet");

    let grew = frame(
        &mut p,
        Some("Invert: opened it in a new window"),
        WITH_STATUS,
    );
    assert_eq!(grew, Some(WITH_STATUS - BARE));
    assert!(
        !p.waiting(),
        "the change was seen, so nothing is still expected"
    );

    // And it stays put while the message does.
    assert_eq!(
        frame(
            &mut p,
            Some("Invert: opened it in a new window"),
            WITH_STATUS
        ),
        None,
        "the window grew again for a message that had not changed"
    );
}

#[test]
fn a_message_going_away_gives_the_height_back() {
    let mut p = PanelLayout::default();
    frame(&mut p, None, BARE);
    frame(&mut p, Some("something"), WITH_STATUS);

    let shrank = frame(&mut p, None, BARE).expect("the height should come back");
    assert_eq!(shrank, BARE - WITH_STATUS);
    assert!(shrank < 0.0);
}

/// A longer message wraps to more lines, and that is a height change like any
/// other — which is why the comparison is on the text and not on whether there
/// is a message at all.
#[test]
fn a_message_that_wraps_to_another_line_grows_the_window_again() {
    let mut p = PanelLayout::default();
    frame(&mut p, None, BARE);
    frame(&mut p, Some("short"), WITH_STATUS);

    let grew = frame(&mut p, Some("a much longer message"), WITH_STATUS + 14.0);
    assert_eq!(grew, Some(14.0));
}

/// The one that nearly shipped: a message replaced by another of the same
/// height has no delta to find, and a flag would sit armed for one forever —
/// asking for a repaint every frame, which costs a core for as long as the
/// window is open.
#[test]
fn a_change_that_turns_out_to_be_no_change_stops_asking_for_frames() {
    let mut p = PanelLayout::default();
    frame(&mut p, None, BARE);
    frame(&mut p, Some("first"), WITH_STATUS);

    // Same height, different text: armed, and nothing to measure.
    assert_eq!(frame(&mut p, Some("second"), WITH_STATUS), None);
    assert!(p.waiting(), "it should look again before giving up");

    for _ in 0..GROW_WAIT {
        frame(&mut p, Some("second"), WITH_STATUS);
    }
    assert!(
        !p.waiting(),
        "still waiting for a height change that is never coming"
    );
}

/// The panel toggle needs the opposite reading: it is clicked on a frame that
/// still draws the bar closed, so the height in front of it is the "before".
#[test]
fn the_toggle_measures_from_the_height_it_was_clicked_at() {
    let mut p = PanelLayout::default();
    frame(&mut p, None, BARE);

    // The click frame: the bar is still its old size.
    p.note_status(None, BARE);
    p.arm_grow(BARE);
    assert_eq!(
        p.grow_delta(BARE),
        None,
        "the panel has not been redrawn yet"
    );
    assert!(p.waiting());

    // The next frame draws it open.
    let expanded = 180.0;
    assert_eq!(frame(&mut p, None, expanded), Some(expanded - BARE));
}

/// A status arriving on the same frame as a toggle must not steal the toggle's
/// "before" height — that one is waiting for a delta this frame cannot show.
#[test]
fn a_toggle_in_flight_keeps_its_own_starting_height() {
    let mut p = PanelLayout::default();
    frame(&mut p, None, BARE);
    p.arm_grow(BARE);

    // A message turns up while the toggle is still waiting.
    assert_eq!(frame(&mut p, Some("saved"), BARE), None);
    let expanded = 180.0;
    assert_eq!(
        frame(&mut p, Some("saved"), expanded),
        Some(expanded - BARE),
        "the toggle's growth was measured from the wrong height"
    );
}

/// Before the bar has ever been drawn there is no previous height, and treating
/// zero as one would grow the window by the whole bar.
#[test]
fn the_first_frame_does_not_grow_the_window_by_the_whole_bar() {
    let mut p = PanelLayout::default();
    assert_eq!(
        frame(&mut p, Some("opened"), WITH_STATUS),
        None,
        "grew by a height that was never on screen"
    );
    assert!(!p.waiting());
}

//! The epoch arithmetic, which is where a quiet mistake would be worst.
//!
//! Getting a window wrong by one frame does not fail — it produces a map of the
//! wrong moment, which looks exactly as convincing as the right one. So the
//! windows are checked against numbers worked out by hand from a real
//! acquisition's timing rather than against whatever the code currently does.

use super::*;

#[test]
fn the_pattern_says_which_steps_fire() {
    // The default: stimulator A on step 0, stimulator C on step 1.
    assert_eq!(parse_pattern("10,01").unwrap(), vec![true, true]);
    // Both stimulators on step 0, neither on step 1.
    assert_eq!(parse_pattern("10,10").unwrap(), vec![true, false]);
    assert_eq!(
        parse_pattern("1100,0110").unwrap(),
        vec![true, true, true, false]
    );
    // A single row is a single stimulator.
    assert_eq!(parse_pattern("101").unwrap(), vec![true, false, true]);
    // Separators are interchangeable, since the config format varies.
    assert_eq!(parse_pattern("10;01").unwrap(), vec![true, true]);
    assert_eq!(parse_pattern("10/01").unwrap(), vec![true, true]);
    assert_eq!(parse_pattern(" 10 , 01 ").unwrap(), vec![true, true]);
}

#[test]
fn a_malformed_pattern_says_what_is_wrong_with_it() {
    for (text, want) in [
        ("", "empty"),
        ("10,0", "different lengths"),
        ("12,01", "only contain 0 and 1"),
        ("ab,cd", "only contain 0 and 1"),
    ] {
        let e = parse_pattern(text).expect_err(&format!("{text:?} should be refused"));
        assert!(
            e.to_string().contains(want),
            "{text:?} gave {e}, expected something about {want:?}"
        );
    }
}

/// A plan built from the numbers of a real recording, so the windows can be
/// checked against arithmetic done by hand.
fn plan(step_duration: f64, resp_duration: f64, frames: usize, duration_s: f64) -> Plan {
    let spf = duration_s / frames as f64;
    let sync = -0.003;
    let spf_adj = spf * (1.0 + sync);
    let mut p = Plan {
        steps: vec![0, 1],
        epochs: 0,
        start_from_epoch: 1,
        trigger_s: 27.772762,
        step_duration,
        resp_duration,
        epoch_duration: step_duration * 2.0,
        fps: 1.0 / spf_adj,
        duration_s: duration_s * (1.0 + sync),
        frame_lag: -1,
        sigma: 2.3,
        channel: 0,
        frames,
    };
    p.epochs = p.fit_epochs();
    p
}

#[test]
fn the_window_lands_where_the_arithmetic_says() {
    // 3000 frames over 300 s: 0.1 s per frame, so windows are several frames.
    let p = plan(10.0, 0.8, 3000, 300.0);
    let spf: f64 = 300.0 / 3000.0 * (1.0 - 0.003);

    // Epoch 0 here is `start_from_epoch` = 1, so the first window is one whole
    // epoch after the trigger — the trigger epoch itself is deliberately
    // skipped.
    let at: f64 = 27.772762 + 20.0;
    let want_start = (at / spf).floor() as i64 - 1;
    let want_end = ((at + 0.8) / spf).floor() as i64 - 1;
    assert_eq!(p.window(0, 0), (want_start as usize, want_end as usize));

    // The second firing step is one step later, not one epoch.
    let at1 = at + 10.0;
    assert_eq!(
        p.window(1, 0).0,
        ((at1 / spf).floor() as i64 - 1) as usize,
        "the step shift is not a step long"
    );
    // And the next epoch is one epoch later.
    let at2 = at + 20.0;
    assert_eq!(p.window(0, 1).0, ((at2 / spf).floor() as i64 - 1) as usize);
}

/// The epoch count is derived, not configured: as many whole epochs as fit
/// after the trigger with their response windows inside the recording.
#[test]
fn the_epoch_count_is_the_most_that_fit() {
    let p = plan(10.0, 0.8, 3000, 300.0);
    // Trigger at 27.77 s, epochs of 20 s, the last step 10 s in plus a 0.8 s
    // window, all inside 300 s * 0.997 = 299.1 s. Epoch e ends at
    // 27.772762 + 20e + 10.8, so e <= 13.0 -> epochs 1..=13 is 13 of them.
    assert_eq!(p.epochs, 13);

    // A recording half as long fits half as many.
    let short = plan(10.0, 0.8, 1500, 150.0);
    assert!(
        short.epochs < p.epochs && short.epochs > 0,
        "a shorter recording should fit fewer epochs, got {}",
        short.epochs
    );

    // The last epoch's last window must really be inside the recording.
    let (_, end) = p.window(*p.steps.last().unwrap(), p.epochs - 1);
    assert!(
        end <= p.frames,
        "the last window runs past the end of the recording"
    );
}

/// A recording that stops before the first epoch completes has nothing to
/// average, and must say so rather than average one truncated window.
#[test]
fn a_recording_too_short_for_one_epoch_fits_none() {
    let p = plan(10.0, 0.8, 300, 30.0);
    assert_eq!(p.epochs, 0);
}

#[test]
fn a_timestamp_past_the_end_clamps_to_the_frame_count() {
    let p = plan(10.0, 0.8, 3000, 300.0);
    assert_eq!(p.sec_to_frame(1e9), p.frames as i64);
    assert_eq!(p.sec_to_frame(0.0), 0);
}

/// The clock correction has to actually move the windows, or it is decoration.
#[test]
fn the_clock_correction_shifts_later_windows_more_than_earlier_ones() {
    let uncorrected = {
        let mut p = plan(10.0, 0.8, 3000, 300.0);
        p.fps = 1.0 / (300.0 / 3000.0);
        p.duration_s = 300.0;
        p.epochs = p.fit_epochs();
        p
    };
    let corrected = plan(10.0, 0.8, 3000, 300.0);

    let first = corrected.window(0, 0).0 as i64 - uncorrected.window(0, 0).0 as i64;
    let last = corrected.window(0, 9).0 as i64 - uncorrected.window(0, 9).0 as i64;
    assert!(
        last.abs() > first.abs(),
        "a 0.3% clock correction should accumulate: first {first}, tenth {last}"
    );
}

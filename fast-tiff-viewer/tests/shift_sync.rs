//! Tests for Shift-dragging one channel's contrast handles to move them all.

use fast_tiff_viewer::channels::shift_sync;
use fast_tiff_viewer::{ChannelKind, ChannelSettings};

fn ch(min: f32, max: f32, bounds: (f32, f32), enabled: bool) -> ChannelSettings {
    ChannelSettings { min, max, enabled, bounds, kind: ChannelKind::Int16 }
}

fn snapshot(settings: &[ChannelSettings]) -> Vec<(f32, f32)> {
    settings.iter().map(|s| (s.min, s.max)).collect()
}

#[test]
fn the_delta_carries_to_the_other_enabled_channels() {
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), true),
        ch(500.0, 600.0, (0.0, 1000.0), true),
    ];
    let before = snapshot(&settings);
    // Channel 0 dragged +50 on both handles.
    settings[0].min += 50.0;
    settings[0].max += 50.0;
    shift_sync(&mut settings, &before);

    assert_eq!((settings[0].min, settings[0].max), (150.0, 250.0));
    assert_eq!((settings[1].min, settings[1].max), (350.0, 450.0));
    assert_eq!((settings[2].min, settings[2].max), (550.0, 650.0));
}

#[test]
fn a_switched_off_channel_does_not_move() {
    // The regression: a disabled channel's handles are drawn inert and take no
    // input, so a Shift-drag on a neighbour must not be a back door that moves
    // them anyway — invisibly, since that channel is not even on screen.
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), false),
        ch(500.0, 600.0, (0.0, 1000.0), true),
    ];
    let before = snapshot(&settings);
    settings[0].min += 50.0;
    settings[0].max += 50.0;
    shift_sync(&mut settings, &before);

    assert_eq!((settings[1].min, settings[1].max), (300.0, 400.0), "disabled channel moved");
    assert_eq!((settings[2].min, settings[2].max), (550.0, 650.0), "enabled channel did not move");
}

#[test]
fn each_channel_is_clamped_to_its_own_bounds() {
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        // Room for only 20 more before this one hits its ceiling.
        ch(800.0, 900.0, (0.0, 920.0), true),
    ];
    let before = snapshot(&settings);
    settings[0].min += 500.0;
    settings[0].max += 500.0;
    shift_sync(&mut settings, &before);

    assert_eq!((settings[1].min, settings[1].max), (920.0, 920.0));
    assert!(settings[1].min <= settings[1].max, "handles crossed");
}

#[test]
fn nothing_moving_is_a_no_op() {
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), true),
    ];
    let before = snapshot(&settings);
    shift_sync(&mut settings, &before);
    assert_eq!(snapshot(&settings), before);
}

#[test]
fn one_handle_moving_carries_only_that_handle() {
    // Dragging just the max handle must not drag everyone's min along with it.
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), true),
    ];
    let before = snapshot(&settings);
    settings[0].max += 30.0;
    shift_sync(&mut settings, &before);

    assert_eq!((settings[1].min, settings[1].max), (300.0, 430.0));
}

#[test]
fn a_short_before_snapshot_does_not_panic() {
    // Defensive: the channel count can change between frames.
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), true),
    ];
    shift_sync(&mut settings, &[(100.0, 200.0)]);
    shift_sync(&mut settings, &[]);
}

//! Tests for Shift-dragging one channel's contrast handles to move them all.
//!
//! The windows here are in the file's own units, as every window the viewer
//! holds is: 16-bit channels, whose values those are.

use fast_tiff_viewer::channels::shift_sync;
use fast_tiff_viewer::fast_tiff_lib::DisplayMap;
use fast_tiff_viewer::{ChannelSettings, SampleUnits, Window};

fn ch(min: f32, max: f32, bounds: (f32, f32), enabled: bool) -> ChannelSettings {
    ChannelSettings {
        enabled,
        bounds: bounds.into(),
        ..ChannelSettings::new(
            Window::new(min, max),
            SampleUnits::display(DisplayMap::Identity),
        )
    }
}

fn snapshot(settings: &[ChannelSettings]) -> Vec<Window> {
    settings.iter().map(|s| s.window).collect()
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
    settings[0].window.lo += 50.0;
    settings[0].window.hi += 50.0;
    shift_sync(&mut settings, &before);

    assert_eq!(
        (settings[0].window.lo, settings[0].window.hi),
        (150.0, 250.0)
    );
    assert_eq!(
        (settings[1].window.lo, settings[1].window.hi),
        (350.0, 450.0)
    );
    assert_eq!(
        (settings[2].window.lo, settings[2].window.hi),
        (550.0, 650.0)
    );
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
    settings[0].window.lo += 50.0;
    settings[0].window.hi += 50.0;
    shift_sync(&mut settings, &before);

    assert_eq!(
        (settings[1].window.lo, settings[1].window.hi),
        (300.0, 400.0),
        "disabled channel moved"
    );
    assert_eq!(
        (settings[2].window.lo, settings[2].window.hi),
        (550.0, 650.0),
        "enabled channel did not move"
    );
}

#[test]
fn each_channel_is_clamped_to_its_own_bounds() {
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        // Room for only 20 more before this one hits its ceiling.
        ch(800.0, 900.0, (0.0, 920.0), true),
    ];
    let before = snapshot(&settings);
    settings[0].window.lo += 500.0;
    settings[0].window.hi += 500.0;
    shift_sync(&mut settings, &before);

    assert_eq!(
        (settings[1].window.lo, settings[1].window.hi),
        (920.0, 920.0)
    );
    assert!(
        settings[1].window.lo <= settings[1].window.hi,
        "handles crossed"
    );
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
    settings[0].window.hi += 30.0;
    shift_sync(&mut settings, &before);

    assert_eq!(
        (settings[1].window.lo, settings[1].window.hi),
        (300.0, 430.0)
    );
}

#[test]
fn a_short_before_snapshot_does_not_panic() {
    // Defensive: the channel count can change between frames.
    let mut settings = vec![
        ch(100.0, 200.0, (0.0, 1000.0), true),
        ch(300.0, 400.0, (0.0, 1000.0), true),
    ];
    shift_sync(&mut settings, &[Window::new(100.0, 200.0)]);
    shift_sync(&mut settings, &[]);
}

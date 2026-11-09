//! Integration tests for the display model's public API.
//!
//! These were unit tests inside the egui app, back when `DecodeMode` and the
//! RGB channel plan were private to it. Both are public `fast-tiff-viewer` API
//! now, so they belong in `tests/` — which also proves the API is reachable
//! from outside the crate, something a `#[cfg(test)] mod` can never check.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::dimensions::rgb_channel_plan;
use fast_tiff_viewer::{DecodeMode, Display, Viewer, Window, MAX_CHANNELS};

#[test]
fn decode_mode_drives_parallel_flag() {
    // Serial is always off and Threaded always on, regardless of whether
    // Auto's "falling behind" latch happens to be set.
    assert!(!DecodeMode::Serial.parallel(false));
    assert!(!DecodeMode::Serial.parallel(true));
    assert!(DecodeMode::Threaded.parallel(false));
    assert!(DecodeMode::Threaded.parallel(true));
    // Auto follows the latch: serial until playback falls behind, then parallel.
    assert!(!DecodeMode::Auto.parallel(false));
    assert!(DecodeMode::Auto.parallel(true));
}

#[test]
fn rgb_extra_samples_get_channels_but_start_disabled() {
    // Plain RGB: three channels, all on.
    assert_eq!(rgb_channel_plan(3), vec![true, true, true]);

    // RGBA (what tifffile writes for a (4, H, W) array): the fourth sample is
    // reachable as a channel — the regression this guards — but starts off, so
    // an opaque alpha plane can't wash out a genuine RGBA image on open.
    assert_eq!(rgb_channel_plan(4), vec![true, true, true, false]);

    // More extras stay individually addressable...
    assert_eq!(rgb_channel_plan(5), vec![true, true, true, false, false]);
    // ...up to the shader's channel limit, past which samples are dropped.
    let many = rgb_channel_plan(MAX_CHANNELS + 3);
    assert_eq!(many.len(), MAX_CHANNELS);
    assert_eq!(
        many.iter().filter(|&&on| on).count(),
        3,
        "only R/G/B start on"
    );
}

// ---------------------------------------------------------------------------
// Revisions: what moves which counter
// ---------------------------------------------------------------------------

/// Two channels by three timepoints of 8 x 8 u16, as a viewer opened on it.
fn two_channel_movie() -> Viewer {
    let opts = WriterOptions::new(8, 8, SampleType::U16).metadata(StackMetaWrite::new(2, 1));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for plane in 0..6u16 {
        w.write_frame_u16(&[plane * 1000; 64]).unwrap();
    }
    let mut viewer = Viewer::new();
    let bytes = w.finish().unwrap().into_inner();
    viewer.load_bytes(bytes, "movie.tif".into()).expect("open");
    viewer
}

fn display(viewer: &mut Viewer) -> &mut Display {
    &mut viewer.stack.as_mut().expect("a stack is open").display
}

/// Switching a channel on or off changes which planes a frame decodes, not
/// which planes a channel reads: the `enabled` revision moves and the `layout`
/// one does not, so whatever covers every channel regardless (the histograms)
/// stays valid while whatever was decoded for the old set does not.
#[test]
fn a_channel_toggle_moves_the_enabled_revision_only() {
    let mut viewer = two_channel_movie();
    let d = display(&mut viewer);
    let before = d.revision();
    assert!(
        d.set_enabled(1, false),
        "switching a channel off is a change"
    );
    let after = d.revision();
    assert_eq!(after.layout, before.layout, "the same planes are read");
    assert_ne!(after.enabled, before.enabled, "but a different set of them");
    assert!(!d.settings()[1].enabled);

    // Asking for what is already so changes nothing, and neither does a
    // channel that is not there.
    assert!(!d.set_enabled(1, false));
    assert!(!d.set_enabled(9, true));
    assert_eq!(d.revision(), after);

    // Switching it back is a change again, and to a revision never seen
    // before, so a key from before the first toggle stays stale.
    assert!(d.set_enabled(1, true));
    let back = d.revision();
    assert_eq!(back.layout, before.layout);
    assert_ne!(back.enabled, after.enabled);
    assert_ne!(back.enabled, before.enabled);
}

/// Contrast and colour change how a frame is drawn, not what is decoded, so
/// nothing keyed on a revision has to be rebuilt for them.
#[test]
fn a_window_or_a_lut_moves_no_revision() {
    let mut viewer = two_channel_movie();
    let before = display(&mut viewer).revision();
    assert!(display(&mut viewer).set_window(0, Window::new(10.0, 20.0)));
    assert_eq!(
        display(&mut viewer).settings()[0].window,
        Window::new(10.0, 20.0)
    );
    viewer.set_pseudocolor(true);
    viewer.set_pseudocolor(false);
    assert_eq!(display(&mut viewer).revision(), before);
}

/// Reassigning the axes moves which IFD every display channel reads, and
/// rebuilds the channels (formats, windows, flags) to match: both counters
/// move.
#[test]
fn a_dimension_override_moves_the_layout_revision() {
    let mut viewer = two_channel_movie();
    let dims = display(&mut viewer).dims();
    assert_eq!((dims.channels, dims.slices, dims.frames), (2, 1, 3));
    let before = display(&mut viewer).revision();

    viewer.set_dimension_order(3, 1, 2);

    let d = display(&mut viewer);
    assert_eq!(d.dims().channels, 3);
    assert_eq!(d.channel_count(), 3, "the settings follow the new count");
    let after = d.revision();
    assert_ne!(
        after.layout, before.layout,
        "every channel reads another IFD"
    );
    assert_ne!(after.enabled, before.enabled, "and has new flags");
}

/// Revisions are drawn from one sequence for the whole process, so a key kept
/// from one stack can never match the next one opened — even the same file,
/// reopened, which would otherwise start from the same numbers.
#[test]
fn no_two_stacks_share_a_revision() {
    let a = display(&mut two_channel_movie()).revision();
    let b = display(&mut two_channel_movie()).revision();
    assert_ne!(a.layout, b.layout);
    assert_ne!(a.enabled, b.enabled);
}

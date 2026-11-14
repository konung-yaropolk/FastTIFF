//! The operations a frontend changes the model through, and the invariants
//! they keep: a frame index that stays on the stack, a window or a switch
//! that moves only the revision it should, a channel layout that is decided
//! at load and survives a reinterpretation, and a LUT selector that cannot be
//! read while it does not apply nor outlive the LUT it chose.

use fast_tiff_lib::{DisplayMode, SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::{ChannelLayout, Document, DocumentMut, Lut, Viewer, Window};
use std::io::Cursor;

const W: u32 = 8;
const H: u32 = 6;

/// `channels` x `frames` planes of 16-bit samples, plane `p` filled with
/// `p * 1000`, as a viewer opened on them.
fn gray(channels: usize, frames: usize) -> Viewer {
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(StackMetaWrite::new(channels, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for plane in 0..(channels * frames) as u16 {
        w.write_frame_u16(&[plane * 1000; (W * H) as usize])
            .unwrap();
    }
    let mut viewer = Viewer::new();
    let bytes = w.finish().unwrap().into_inner();
    viewer.load_bytes(bytes, "gray.tif".into()).expect("open");
    viewer
}

/// `frames` chunky 8-bit RGB frames, as a viewer opened on them.
fn rgb(frames: usize) -> Viewer {
    let opts = WriterOptions::new(W, H, SampleType::U8)
        .samples_per_pixel(3)
        .metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames as u32 {
        let px: Vec<u8> = (0..W * H * 3).map(|i| (i * 7 + f * 50) as u8).collect();
        w.write_frame_u8(&px).unwrap();
    }
    let mut viewer = Viewer::new();
    let bytes = w.finish().unwrap().into_inner();
    viewer.load_bytes(bytes, "rgb.tif".into()).expect("open");
    viewer
}

fn doc(viewer: &Viewer) -> &Document {
    viewer.document().expect("a stack is open")
}

fn doc_mut(viewer: &mut Viewer) -> DocumentMut<'_> {
    viewer.document_mut().expect("a stack is open")
}

// ---------------------------------------------------------------------------
// The frame on screen
// ---------------------------------------------------------------------------

/// A step or a jump off either end lands on that end: the frame index never
/// names a frame the stack does not have, and never wraps round.
#[test]
fn frame_steps_stop_at_the_first_and_the_last_frame() {
    let mut viewer = gray(1, 4);
    let mut d = doc_mut(&mut viewer);
    assert_eq!((d.frame(), d.frame_count()), (0, 4));

    d.step_frame(-1);
    assert_eq!(d.frame(), 0, "back from the first frame stays on it");
    d.step_frame(2);
    assert_eq!(d.frame(), 2);
    d.step_frame(5);
    assert_eq!(d.frame(), 3, "past the end is the last frame");
    d.step_frame(1);
    assert_eq!(d.frame(), 3, "and stepping on from it stays there");
    d.step_frame(-2);
    assert_eq!(d.frame(), 1);

    // Nothing a frontend can pass overflows the arithmetic.
    d.step_frame(isize::MAX);
    assert_eq!(d.frame(), 3);
    d.step_frame(isize::MIN);
    assert_eq!(d.frame(), 0);

    d.set_frame(99);
    assert_eq!(d.frame(), 3, "a jump past the end is the last frame too");
}

/// Reassigning the axes changes how many frames there are; the frame on
/// screen goes back to the first, which every interpretation has.
#[test]
fn a_dimension_override_puts_the_frame_back_to_the_first() {
    let mut viewer = gray(2, 3);
    doc_mut(&mut viewer).set_frame(2);
    viewer.set_dimension_order(3, 1, 2);
    assert_eq!(doc(&viewer).frame_count(), 2);
    assert_eq!(doc(&viewer).frame(), 0);
}

// ---------------------------------------------------------------------------
// A channel's window and switch
// ---------------------------------------------------------------------------

/// The setters say whether they changed anything, the getters read back what
/// they set, and a channel that is not there is a no-op that moves nothing.
#[test]
fn window_and_switch_read_back_and_ignore_a_missing_channel() {
    let mut viewer = gray(2, 1);
    let mut doc = doc_mut(&mut viewer);
    let mut d = doc.display_mut();
    let before = d.revision();

    let w = Window::new(100.0, 900.0);
    assert!(d.set_window(1, w));
    assert_eq!(d.window(1), Some(w));
    assert!(!d.set_window(1, w), "the same window again is no change");
    assert_eq!(d.revision(), before, "a window moves no revision");

    assert!(d.enabled(1));
    assert!(d.set_enabled(1, false));
    assert!(!d.enabled(1));
    let after = d.revision();
    assert_eq!(
        after.layout, before.layout,
        "a switch reads the same planes"
    );
    assert_ne!(
        after.enabled, before.enabled,
        "but decodes another set of them"
    );

    assert!(!d.set_window(2, w), "no third channel");
    assert!(!d.set_enabled(2, true));
    assert_eq!(d.window(2), None);
    assert!(!d.enabled(2));
    assert_eq!(d.revision(), after, "a missing channel moves nothing");
}

/// Auto, Reset and the single-channel LUT change how a frame is drawn, not
/// what is decoded for it, so nothing keyed on a revision is rebuilt.
#[test]
fn contrast_buttons_and_the_lut_selector_move_no_revision() {
    let mut viewer = gray(1, 2);
    let before = doc(&viewer).display().revision();
    let hists = fast_tiff_viewer::histogram::frame_histograms(doc(&viewer));
    doc_mut(&mut viewer).display_mut().auto_contrast(&hists);
    doc_mut(&mut viewer).display_mut().reset_contrast();
    viewer.set_gray_lut(3);
    assert_eq!(doc(&viewer).display().revision(), before);
}

// ---------------------------------------------------------------------------
// The channel layout
// ---------------------------------------------------------------------------

/// What a display channel is follows from the file, and only RGB and CMYK
/// read sample planes of one IFD.
#[test]
fn the_layout_is_the_files() {
    let gray = gray(2, 1);
    let layout = doc(&gray).display().layout();
    assert_eq!(layout, ChannelLayout::Intensity);
    assert!(!layout.shown_as_rgb());
    assert!(layout.has_contrast());

    let rgb = rgb(2);
    let layout = doc(&rgb).display().layout();
    assert_eq!(layout, ChannelLayout::Rgb);
    assert!(layout.shown_as_rgb());
    assert!(layout.has_contrast());
    assert_eq!(doc(&rgb).display().channel_count(), 3);

    assert!(ChannelLayout::Cmyk.shown_as_rgb());
    assert!(ChannelLayout::Cmyk.has_contrast());
    assert!(!ChannelLayout::Palette.shown_as_rgb());
    assert!(
        !ChannelLayout::Palette.has_contrast(),
        "a palette window is pinned"
    );
}

/// A reinterpretation only moves planes between the axes: counts that would
/// address more or fewer of them, or none, are no reading of the file and
/// change nothing.
#[test]
fn a_dimension_order_that_is_not_a_reassignment_is_ignored() {
    let mut viewer = gray(2, 3);
    let (dims, revision) = {
        let d = doc(&viewer).display();
        (d.dims(), d.revision())
    };
    for (c, z, t) in [(1, 1, 1), (0, 0, 0), (2, 3, 2), (usize::MAX, 2, 3)] {
        viewer.set_dimension_order(c, z, t);
        let d = doc(&viewer).display();
        assert_eq!(d.dims(), dims, "({c}, {z}, {t})");
        assert_eq!(d.revision(), revision, "({c}, {z}, {t})");
        assert_eq!(d.channel_count(), 2);
    }
    viewer.set_dimension_order(6, 1, 1);
    assert_eq!(doc(&viewer).display().dims().channels, 6, "a reassignment");
}

/// Reassigning the axes moves which IFD a channel reads, not what a channel
/// is.
#[test]
fn a_dimension_override_keeps_the_layout() {
    let mut viewer = gray(2, 3);
    let before = doc(&viewer).display().revision();
    viewer.set_dimension_order(3, 1, 2);
    let d = doc(&viewer).display();
    assert_eq!(d.dims().channels, 3);
    assert_eq!(d.layout(), ChannelLayout::Intensity);
    assert_ne!(d.revision().layout, before.layout, "but the planes moved");
}

// ---------------------------------------------------------------------------
// The single-channel LUT selector
// ---------------------------------------------------------------------------

/// The selector exists only while it applies: one channel, not RGB. There is
/// no stale choice to read for a stack it does not apply to.
#[test]
fn the_lut_selector_applies_to_one_channel_that_is_not_rgb() {
    assert!(doc(&gray(1, 2)).display().gray_luts().is_some());
    assert!(
        doc(&gray(2, 1)).display().gray_luts().is_none(),
        "two channels"
    );
    assert!(doc(&rgb(1)).display().gray_luts().is_none(), "RGB");

    // And asking for it where it does not apply changes nothing.
    let mut two = gray(2, 1);
    let luts = doc(&two).display().luts().to_vec();
    two.set_gray_lut(3);
    assert_eq!(doc(&two).display().luts(), &luts[..]);
}

/// Picking an option shows its LUT and is what the selector then says; an
/// option past the last is ignored rather than shown.
#[test]
fn the_lut_selector_shows_what_was_picked() {
    let mut viewer = gray(1, 2);
    let luts = doc(&viewer).display().gray_luts().unwrap();
    assert_eq!((luts.selected(), luts.name(0)), (0, "Grayscale"));
    let count = luts.count();

    viewer.set_gray_lut(1);
    let d = doc(&viewer).display();
    assert_eq!(d.gray_luts().unwrap().selected(), 1);
    assert_eq!(d.gray_luts().unwrap().name(1), "Red");
    assert_eq!(d.lut(0), Some(&fast_tiff_lib::default_composite_lut(0)));

    viewer.set_gray_lut(count);
    let d = doc(&viewer).display();
    assert_eq!(d.gray_luts().unwrap().selected(), 1, "no such option");
    assert_eq!(d.lut(0), Some(&fast_tiff_lib::default_composite_lut(0)));
}

/// A reinterpretation that changes the channel count regenerates the LUTs,
/// and with them goes the choice the selector made: back on one channel, it
/// says what is shown — its first option — not the colour picked before.
#[test]
fn the_lut_choice_goes_with_the_luts_it_chose() {
    let mut viewer = gray(1, 4);
    viewer.set_gray_lut(3);
    assert_eq!(doc(&viewer).display().gray_luts().unwrap().selected(), 3);

    viewer.set_dimension_order(4, 1, 1);
    assert!(
        doc(&viewer).display().gray_luts().is_none(),
        "four channels"
    );

    viewer.set_dimension_order(1, 1, 4);
    let d = doc(&viewer).display();
    let luts = d.gray_luts().expect("one channel again");
    assert_eq!(luts.selected(), 0);
    assert_eq!(
        d.lut(0),
        Some(&fast_tiff_lib::grayscale_lut()),
        "what option 0 is"
    );
}

/// `channels` x `frames` planes as a viewer opened on them, the file saying
/// `mode` and, when `lut` is given, that LUT for every channel.
fn with_mode(channels: usize, frames: usize, mode: DisplayMode, lut: Option<Lut>) -> Viewer {
    let mut meta = StackMetaWrite::new(channels, 1).mode(mode);
    for _ in 0..channels {
        if let Some(lut) = lut {
            meta = meta.channel_lut(lut);
        }
    }
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(meta);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for plane in 0..(channels * frames) as u16 {
        w.write_frame_u16(&[plane * 1000; (W * H) as usize])
            .unwrap();
    }
    let mut viewer = Viewer::new();
    let bytes = w.finish().unwrap().into_inner();
    viewer.load_bytes(bytes, "mode.tif".into()).expect("open");
    viewer
}

/// The option the selector says is on show is the LUT shown: its name,
/// and the LUT that picking it again installs.
fn assert_selector_names_the_lut(viewer: &mut Viewer, name: &str) {
    let d = doc(viewer).display();
    let shown = *d.lut(0).unwrap();
    let luts = d.gray_luts().expect("one channel");
    let selected = luts.selected();
    assert_eq!(luts.name(selected), name);
    viewer.set_gray_lut(selected);
    assert_eq!(doc(viewer).display().lut(0), Some(&shown), "{name}");
}

/// A composite file's first channel is red, so collapsed to that channel
/// the stack wears red, and the selector says so rather than "Grayscale".
#[test]
fn a_composite_stack_collapsed_to_one_channel_is_said_to_be_red() {
    let mut viewer = with_mode(3, 1, DisplayMode::Composite, None);
    viewer.set_dimension_order(1, 1, 3);
    let d = doc(&viewer).display();
    assert_eq!(d.lut(0), Some(&fast_tiff_lib::default_composite_lut(0)));
    assert_eq!(
        d.gray_luts().unwrap().name(0),
        "Grayscale",
        "no Built-in option: red is one of the others"
    );
    assert_selector_names_the_lut(&mut viewer, "Red");
}

/// The same for a single-channel composite file, as it opens.
#[test]
fn a_single_channel_composite_file_opens_said_to_be_red() {
    let mut viewer = with_mode(1, 2, DisplayMode::Composite, None);
    assert_eq!(
        doc(&viewer).display().lut(0),
        Some(&fast_tiff_lib::default_composite_lut(0))
    );
    assert_selector_names_the_lut(&mut viewer, "Red");
}

/// A LUT of the file's own that is none of the options is the "Built-in"
/// one once the stack is down to that channel, and is what it puts back.
#[test]
fn a_files_own_lut_collapsed_to_one_channel_is_the_built_in_option() {
    let mut own = [[0u8; 3]; 256];
    for (i, entry) in own.iter_mut().enumerate() {
        *entry = [i as u8, (255 - i) as u8, 40];
    }
    let mut viewer = with_mode(3, 1, DisplayMode::Composite, Some(own));
    assert!(
        doc(&viewer).display().gray_luts().is_none(),
        "three channels"
    );
    viewer.set_dimension_order(1, 1, 3);
    assert_eq!(doc(&viewer).display().lut(0), Some(&own));
    assert_selector_names_the_lut(&mut viewer, "Built-in");

    viewer.set_gray_lut(1);
    assert_eq!(
        doc(&viewer).display().gray_luts().unwrap().name(1),
        "Grayscale"
    );
    viewer.set_gray_lut(0);
    assert_eq!(doc(&viewer).display().lut(0), Some(&own), "put back");
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

/// Playing, pausing and stopping are operations on the clock; resuming
/// starts it clean, so the time spent paused is not played out in a burst.
#[test]
fn play_pause_and_stop_run_the_clock_from_a_clean_start() {
    let mut viewer = gray(1, 20);
    assert!(!viewer.playback.playing());
    viewer.playback.set_fps(4.0);
    assert_eq!(viewer.playback.fps(), 4.0);

    viewer.playback.toggle();
    assert!(viewer.playback.playing());
    assert_eq!(
        viewer.tick_playback(0.0),
        0,
        "the first tick seeds the clock"
    );
    assert_eq!(viewer.tick_playback(0.25), 1);

    viewer.playback.toggle();
    assert!(!viewer.playback.playing());
    assert_eq!(viewer.tick_playback(3.0), 0, "paused");

    viewer.playback.toggle();
    assert_eq!(viewer.tick_playback(8.0), 0, "resumed: seeds again");
    assert_eq!(viewer.tick_playback(8.25), 1, "not the time it was paused");
    assert_eq!(doc(&viewer).frame(), 2);

    viewer.playback.stop();
    assert!(!viewer.playback.playing());
    assert_eq!(viewer.tick_playback(12.0), 0);
}

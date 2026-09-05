//! Tag 270 is a mixture, and the parser has to be deaf to most of it.

use super::*;

/// A description exactly as FastTIFF writes one for a converted OIR: the ImageJ
/// block first, then the instrument's own export appended after it.
const REAL_SHAPE: &str = concat!(
    "ImageJ=1.54f\n",
    "images=298\n",
    "channels=1\n",
    "slices=1\n",
    "frames=298\n",
    "hyperstack=true\n",
    "mode=grayscale\n",
    "unit=micron\n",
    "finterval=1.0862\n",
    "\"[General]\"\t\"\"\n",
    "\"Name\"\t\"Field_1.oir\"\n",
    "\"Scan Mode\"\t\"XYT\"\n",
    "\"[Dimensions]\"\t\"\"\n",
    "\"X Dimension\"\t\"512, 0.0 - 318.198 [um], 0.621 [um/pixel]\"\n",
    "\"T Dimension\"\t\"298, 0.000 - 322.701 [s], Interval FreeRun\"\n",
    "\"[Acquisition]\"\t\"\"\n",
    "\"Sampling Speed\"\t\"2.0 [us/pixel]\"\n",
    "\"[Channel 1]\"\t\"\"\n",
    "\"Channel Name\"\t\"RNDD3G\"\n",
    "\"[Event 1]\"\t\"\"\n",
    "\"Event Contents\"\t\"DRS\"\n",
    "\"Event Timer\"\t\"27772.762000[ms]\"\n",
    "\"[Event 2]\"\t\"\"\n",
    "\"Event Contents\"\t\"DRS\"\n",
    "\"Event Timer\"\t\"52285.595000[ms]\"\n",
    "\"[Event 3]\"\t\"\"\n",
    "\"Event Contents\"\t\"Drug Application trigger\"\n",
    "\"Event Timer\"\t\"75246.282000[ms]\"\n",
);

#[test]
fn the_timing_is_read_out_of_a_real_description() {
    let t = Timing::parse(REAL_SHAPE);
    assert_eq!(t.frames, Some(298));
    assert_eq!(t.duration_s, Some(322.701));
    assert_eq!(t.events.len(), 3);
    assert_eq!(t.events[0].0, "DRS");
    // Milliseconds in the file, seconds here.
    assert!(
        (t.events[0].1 - 27.772762).abs() < 1e-9,
        "{:?}",
        t.events[0]
    );
    assert!((t.events[2].1 - 75.246282).abs() < 1e-9);
    assert_eq!(t.events[2].0, "Drug Application trigger");
    assert!((t.seconds_per_frame().unwrap() - 322.701 / 298.0).abs() < 1e-12);
}

/// The point of the exercise: ImageJ's own keys live in the same string, and
/// several of them are numbers that would be ruinous to mistake for the
/// acquisition's.
#[test]
fn the_imagej_block_in_the_same_string_is_ignored() {
    let t = Timing::parse(REAL_SHAPE);
    assert_eq!(
        t.duration_s,
        Some(322.701),
        "finterval was read as the duration"
    );
    // A description that is *only* an ImageJ block yields nothing at all rather
    // than a plausible-looking guess.
    let bare = Timing::parse("ImageJ=1.54f\nimages=298\nframes=298\nfinterval=1.0862\n");
    assert_eq!(bare.frames, None);
    assert_eq!(bare.duration_s, None);
    assert!(bare.events.is_empty());
    assert_eq!(bare.seconds_per_frame(), None);
}

#[test]
fn unrelated_lines_are_ignored_however_much_they_look_like_a_match() {
    let t = Timing::parse(concat!(
        "some free text about T Dimension: 999\n",
        "<xml><T-Dimension>888</T-Dimension></xml>\n",
        "\"T Dimension \"\t\"777, 0 - 1 [s]\"\n",
        "Event Timer 123456\n",
        "\"[Events]\"\t\"\"\n",
        "\"Event Timer\"\t\"5000[ms]\"\n",
    ));
    assert_eq!(t.frames, None, "a near-miss key was accepted");
    assert!(
        t.events.is_empty(),
        "an `Event Timer` outside an `[Event N]` section was taken as a trigger: {:?}",
        t.events
    );
}

#[test]
fn an_event_without_a_timer_is_not_an_event() {
    let t = Timing::parse(concat!(
        "\"[Event 1]\"\t\"\"\n",
        "\"Event Contents\"\t\"DRS\"\n",
        "\"Something Else\"\t\"x\"\n",
    ));
    assert!(t.events.is_empty());
}

#[test]
fn a_zero_or_missing_duration_yields_no_sampling_interval() {
    for value in ["0, 0.000 - 0.000 [s]", "298", "nonsense"] {
        let t = Timing::parse(&format!("\"T Dimension\"\t\"{value}\"\n"));
        assert_eq!(
            t.seconds_per_frame(),
            None,
            "{value:?} should not produce an interval"
        );
    }
}

#[test]
fn numbers_are_found_wherever_they_sit() {
    assert_eq!(first_number("298, 0.000 - 322.701 [s]"), Some(298.0));
    assert_eq!(first_number("27772.762000[ms]"), Some(27772.762));
    assert_eq!(first_number("-1.5 and more"), Some(-1.5));
    assert_eq!(first_number("no digits"), None);
    assert_eq!(first_number(""), None);
}

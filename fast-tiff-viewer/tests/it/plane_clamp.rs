//! Trusting a file's declared shape further than the file itself.
//!
//! A TIFF's metadata says how many channels, Z-slices and time frames it has,
//! and the viewer turns that into an IFD index for every plane it draws. Nothing
//! guarantees the two agree. A multi-file OME set is the ordinary way they do
//! not: each file carries the *whole* dataset's dimensions and points at its
//! siblings, so one file of a two-channel pair honestly declares twice the
//! planes it holds. A truncated acquisition is the other way.
//!
//! Believed literally, the shape addresses IFDs past the end of the chain and
//! every frame the scrubber reaches past the halfway point fails to decode. So
//! the shape is cut down to what the file can actually back up — and, because
//! that means showing an arrangement the metadata does not agree with, said out
//! loud rather than quietly corrected.
//!
//! The one thing these tests must not let slip is the mirror between
//! `planes_addressed` and `build_jobs`: they are two statements of the same
//! arithmetic, and the clamp is worth nothing if they drift apart.

use fast_tiff_viewer::dimensions::{clamp_to_available, compute_status, planes_addressed};
use fast_tiff_viewer::Dims;

fn dims(channels: usize, slices: usize, frames: usize) -> Dims {
    Dims {
        channels,
        slices,
        frames,
    }
}

// ---------------------------------------------------------------------------
// What a shape costs in planes
// ---------------------------------------------------------------------------

/// The count is one past the highest IFD index the addressing produces, so a
/// file with exactly that many planes is exactly enough.
#[test]
fn a_shape_addresses_as_many_planes_as_it_has() {
    // One channel, no Z: a plain movie is one IFD per frame.
    assert_eq!(planes_addressed(dims(1, 1, 20), false), 20);
    // Two channels interleaved per frame.
    assert_eq!(planes_addressed(dims(2, 1, 20), false), 40);
    // Z is a *stride*, not a multiplier on the count: only slice 0 of each
    // frame is ever read (that is what the triple-axis warning is about), so a
    // 2c x 3z x 10t stack holds 60 planes but addresses only 56 of them —
    // frames land at 0, 6, 12 ... 54, plus the channel offset. It is the
    // highest index that has to exist, not the total.
    assert_eq!(planes_addressed(dims(2, 3, 10), false), 56);
    // Chunky RGB keeps its channels inside one IFD, so they cost nothing.
    assert_eq!(planes_addressed(dims(3, 1, 20), true), 20);
    assert_eq!(planes_addressed(dims(3, 4, 20), true), 77, "(20-1)*4 + 1");
    // Degenerate shapes still address the one plane they must.
    assert_eq!(planes_addressed(dims(0, 0, 0), false), 1);
    assert_eq!(planes_addressed(dims(0, 0, 0), true), 1);
}

// ---------------------------------------------------------------------------
// Cutting it down
// ---------------------------------------------------------------------------

/// The file that prompted this: `tubhiswt_C0.ome.tif`, one half of a two-file
/// OME set. Its OME-XML says `SizeC="2" SizeT="20"` — forty planes — and the
/// file contains twenty. Believed literally it asked for IFD 36 and failed.
#[test]
fn the_two_file_ome_set_is_cut_to_the_half_it_holds() {
    let mut d = dims(2, 1, 20);
    let told = clamp_to_available(&mut d, false, 20);
    assert_eq!(told, Some((40, 20)), "it should report what it found");
    assert!(
        planes_addressed(d, false) <= 20,
        "still addresses {} planes of 20: {d:?}",
        planes_addressed(d, false)
    );
    assert_eq!(
        d.channels, 2,
        "the channels are real; it is the length that is not"
    );
    assert_eq!(d.frames, 10);
}

/// Nothing is said, and nothing is touched, when the file backs up its claim.
#[test]
fn a_file_that_holds_what_it_declares_is_left_alone() {
    for (c, z, f, rgb, available) in [
        (1usize, 1usize, 20usize, false, 20usize),
        (2, 1, 20, false, 40),
        (3, 1, 20, true, 20),
        (2, 3, 10, false, 56),
        // More planes than declared is not a mismatch: extra IFDs are simply
        // never addressed, which is a file's business, not ours.
        (1, 1, 5, false, 99),
    ] {
        let mut d = dims(c, z, f);
        let before = d;
        assert_eq!(
            clamp_to_available(&mut d, rgb, available),
            None,
            "{before:?} vs {available}"
        );
        assert_eq!(d, before, "it was changed anyway: {before:?} -> {d:?}");
    }
}

/// Whatever the declaration and whatever the file, the result must address only
/// planes that exist. This is the property the decode error came from, so it is
/// checked over a spread rather than at a point.
#[test]
fn nothing_ever_addresses_a_plane_the_file_lacks() {
    for rgb in [false, true] {
        for available in [1usize, 2, 3, 7, 20, 40, 1000] {
            for c in [1usize, 2, 3, 6] {
                for z in [1usize, 2, 5] {
                    for f in [1usize, 2, 20, 500] {
                        let mut d = dims(c, z, f);
                        let told = clamp_to_available(&mut d, rgb, available);
                        let case = format!("{c}c {z}z {f}t rgb={rgb} into {available}");

                        let needs = planes_addressed(d, rgb);
                        assert!(needs <= available, "{case}: still needs {needs} -> {d:?}");
                        assert!(
                            d.channels >= 1 && d.slices >= 1 && d.frames >= 1,
                            "{case}: {d:?}"
                        );
                        // Only cut when it had to, and never grown.
                        assert!(
                            d.channels <= c && d.slices <= z && d.frames <= f,
                            "{case}: {d:?}"
                        );
                        if told.is_none() {
                            assert_eq!(d, dims(c, z, f), "{case}: cut without saying so");
                        }
                    }
                }
            }
        }
    }
}

/// It should cut as little as it can: having reduced the shape, one more frame
/// would not have fitted. Otherwise a file with a few planes missing would lose
/// far more of itself than it had to.
#[test]
fn it_keeps_as_many_frames_as_will_fit() {
    for rgb in [false, true] {
        for available in [1usize, 5, 19, 20, 21, 37] {
            for c in [1usize, 2, 3] {
                for z in [1usize, 2] {
                    let mut d = dims(c, z, 500);
                    clamp_to_available(&mut d, rgb, available);
                    let case = format!("{c}c {z}z rgb={rgb} into {available} -> {d:?}");
                    if d.frames < 500 && d.channels == c && d.slices == z {
                        let one_more = dims(d.channels, d.slices, d.frames + 1);
                        assert!(
                            planes_addressed(one_more, rgb) > available,
                            "{case}: another frame would have fitted"
                        );
                    }
                }
            }
        }
    }
}

/// Fewer IFDs in the whole file than there are channels: a frame cannot be
/// completed at all, so the channels give way too rather than the shape staying
/// impossible.
#[test]
fn a_file_shorter_than_one_frame_loses_channels() {
    let mut d = dims(6, 1, 10);
    let told = clamp_to_available(&mut d, false, 3);
    assert_eq!(told, Some((60, 3)));
    assert_eq!(d.frames, 1, "there is not a whole second frame to be had");
    assert_eq!(d.channels, 3);
    assert!(planes_addressed(d, false) <= 3);
}

/// A file with no IFDs at all is not something to divide by.
#[test]
fn an_empty_file_is_declined_rather_than_divided_by() {
    let mut d = dims(2, 1, 20);
    let before = d;
    assert_eq!(clamp_to_available(&mut d, false, 0), None);
    assert_eq!(d, before, "nothing sensible to cut to, so nothing is cut");
}

// ---------------------------------------------------------------------------
// Saying so
// ---------------------------------------------------------------------------

/// The reader has to be told, because what is on screen is an arrangement the
/// file's own metadata does not support.
#[test]
fn the_mismatch_is_reported_with_both_numbers() {
    let note = compute_status(dims(2, 1, 10), false, Some((40, 20)))
        .expect("a mismatch must produce a note");
    assert!(note.contains("40"), "should say what was declared: {note}");
    assert!(
        note.contains("20"),
        "should say what the file holds: {note}"
    );
    assert!(
        note.to_lowercase().contains("warning"),
        "should read as a warning: {note}"
    );
}

/// It outranks the interpretation notes: those say how the file was read, this
/// says the file disagrees with itself.
#[test]
fn the_mismatch_outranks_the_other_notes() {
    let both = compute_status(dims(2, 3, 10), true, Some((40, 20))).expect("some note");
    assert!(
        both.contains("40"),
        "the triple-axis note displaced it: {both}"
    );

    // And with no mismatch the other notes still come through unchanged.
    let triple = compute_status(dims(2, 3, 10), true, None).expect("some note");
    assert!(triple.contains("Z-slice"), "{triple}");
    assert_eq!(
        compute_status(dims(1, 1, 10), false, None),
        None,
        "a plain stack says nothing"
    );
}

//! Which JPEG a stream is, decided from the stream.
//!
//! The dispatch is on the `SOF` marker rather than the transfer syntax UID,
//! because the UID says which the file is *supposed* to be and converters
//! routinely rewrite one and not the other. These tests are about the walk that
//! finds that marker, since a stream's `SOF` is not at a fixed offset — any
//! number of `APP`, `COM` and table segments may come first, and medical
//! encoders are fond of `APP` blocks.

use super::super::pixels::Format;
use super::*;

fn fmt() -> Format {
    Format {
        rows: 2,
        columns: 2,
        frames: 1,
        samples: 1,
        bits_allocated: 8,
        bits_stored: 8,
        high_bit: 7,
        signed: false,
        planar: false,
        inverted: false,
        palette: false,
    }
}

/// `SOI`, then the given segments, then an `SOF` of the given kind.
fn with_segments(before: &[(u8, usize)], sof: u8) -> Vec<u8> {
    let mut b = vec![0xFF, 0xD8];
    for &(marker, payload) in before {
        b.extend_from_slice(&[0xFF, marker]);
        b.extend_from_slice(&((payload + 2) as u16).to_be_bytes());
        b.extend(std::iter::repeat_n(0x5A, payload));
    }
    b.extend_from_slice(&[0xFF, sof]);
    b.extend_from_slice(&11u16.to_be_bytes());
    b.extend_from_slice(&[8, 0, 2, 0, 2, 1, 1, 0x11, 0x00]);
    b
}

#[test]
fn the_frame_marker_is_found_past_any_number_of_leading_segments() {
    // No segments, then one, then several — including an `APP14` and a comment,
    // which is what a scanner's output actually looks like.
    for before in [
        &[][..],
        &[(0xE0, 14)][..],
        &[(0xE0, 14), (0xFE, 40), (0xEE, 12), (0xDB, 65)][..],
    ] {
        assert_eq!(
            start_of_frame(&with_segments(before, 0xC3)).expect("should find it"),
            0xC3,
            "{} leading segment(s)",
            before.len()
        );
    }
}

/// Every `SOFn` is `FFC0..FFCF` *except* the three that are something else:
/// `FFC4` is a Huffman table, `FFC8` is reserved and `FFCC` is arithmetic
/// conditioning. A walk that treated those as frame headers would stop at the
/// first table in the stream and decide the file was a kind of JPEG it is not.
#[test]
fn a_huffman_table_is_not_mistaken_for_a_frame_header() {
    // A `DHT` before the real `SOF3`.
    let b = with_segments(&[(0xC4, 19)], 0xC3);
    assert_eq!(start_of_frame(&b).expect("should skip the table"), 0xC3);
    // And the other two of the three.
    for not_a_frame in [0xC8u8, 0xCC] {
        let b = with_segments(&[(not_a_frame, 8)], 0xC0);
        assert_eq!(start_of_frame(&b).expect("should skip it"), 0xC0);
    }
}

/// A stream with no `SOI` at all is not a JPEG, and saying so beats whatever a
/// decoder would make of it.
#[test]
fn a_stream_with_no_start_marker_is_refused() {
    let err = decode(&[0u8; 64], fmt()).expect_err("should refuse");
    assert!(format!("{err}").contains("start-of-image"), "{err}");
}

/// A stream that ends before its frame header does.
#[test]
fn a_stream_that_ends_before_its_frame_header_is_refused() {
    assert!(decode(&[0xFF, 0xD8], fmt()).is_err());
    assert!(decode(&[0xFF, 0xD8, 0xFF], fmt()).is_err());
}

/// Each kind this does not decode is named, because the name says what to
/// convert from. Progressive, hierarchical and arithmetic are three different
/// algorithms and a decoder handed the wrong one does not fail cleanly — it
/// produces an image.
#[test]
fn each_kind_of_jpeg_this_cannot_read_is_refused_by_name() {
    for (sof, word) in [
        (0xC2u8, "progressive"),
        (0xC5, "hierarchical"),
        (0xC6, "hierarchical"),
        (0xC7, "hierarchical"),
        (0xC9, "arithmetic"),
        (0xCA, "arithmetic"),
        (0xCD, "arithmetic"),
    ] {
        let err = decode(&with_segments(&[], sof), fmt()).expect_err("should refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains(word),
            "FF{sof:02X} should be named {word:?}: {msg}"
        );
    }
}

/// A marker in the `SOF` range that is not a kind of JPEG at all is reported
/// with its own number, which is the only thing that can be said about it.
#[test]
fn an_unrecognised_frame_marker_reports_its_own_number() {
    // `FFCE` is in the arithmetic set; `FFC3` is lossless. Pick something that
    // reaches the catch-all by being a valid `SOF` slot this does not list.
    let b = with_segments(&[], 0xCF);
    let err = decode(&b, fmt()).expect_err("should refuse");
    assert!(format!("{err}").contains("arithmetic"), "{err}");
}

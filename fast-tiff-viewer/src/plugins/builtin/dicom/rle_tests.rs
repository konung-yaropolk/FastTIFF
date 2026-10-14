//! PackBits, and the segment-to-byte-position mapping that is the actual trap.

use super::super::pixels::Format;
use super::*;

/// A `Format` with the given geometry, built directly — the tags it would come
/// from are tested in `pixels_tests`.
fn fmt(cols: u32, rows: u32, bits: u16, samples: usize) -> Format {
    Format {
        rows,
        columns: cols,
        frames: 1,
        samples,
        bits_allocated: bits,
        bits_stored: bits,
        high_bit: bits - 1,
        signed: false,
        planar: false,
        inverted: false,
        palette: false,
    }
}

/// Assemble a fragment: the 64-byte segment table, then the segments.
fn fragment(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    out[0..4].copy_from_slice(&(segments.len() as u32).to_le_bytes());
    let mut at = 64u32;
    for (i, s) in segments.iter().enumerate() {
        out[4 + i * 4..8 + i * 4].copy_from_slice(&at.to_le_bytes());
        at += s.len() as u32;
    }
    for s in segments {
        out.extend_from_slice(s);
    }
    out
}

/// PackBits a run of identical bytes.
fn repeat(b: u8, n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut left = n;
    while left > 0 {
        let run = left.min(128);
        out.push((1i32 - run as i32) as i8 as u8);
        out.push(b);
        left -= run;
    }
    out
}

/// PackBits a run of literal bytes.
fn literal(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in bytes.chunks(128) {
        out.push((chunk.len() - 1) as u8);
        out.extend_from_slice(chunk);
    }
    out
}

#[test]
fn a_literal_run_comes_back_as_itself() {
    let want: Vec<u8> = (0..16u8).collect();
    let f = fragment(&[literal(&want)]);
    assert_eq!(decode(&f, fmt(16, 1, 8, 1)).expect("decode"), want);
}

#[test]
fn a_repeat_run_expands() {
    let f = fragment(&[repeat(0xAB, 40)]);
    assert_eq!(
        decode(&f, fmt(40, 1, 8, 1)).expect("decode"),
        vec![0xAB; 40]
    );
}

/// A run longer than one control byte can express is split, and the splits have
/// to join seamlessly.
#[test]
fn runs_longer_than_a_control_byte_are_rejoined() {
    for n in [128usize, 129, 300, 1000] {
        let f = fragment(&[repeat(7, n)]);
        assert_eq!(
            decode(&f, fmt(n as u32, 1, 8, 1)).expect("decode"),
            vec![7u8; n],
            "{n} bytes"
        );
    }
}

/// `-128` is a no-op and is in the scheme only because the byte exists.
/// Treating it as a run of 129 shifts every byte after it.
#[test]
fn the_no_op_control_byte_is_skipped_rather_than_decoded() {
    let mut seg = literal(&[1, 2, 3]);
    seg.push(0x80); // -128
    seg.extend(literal(&[4, 5, 6]));
    let f = fragment(&[seg]);
    assert_eq!(
        decode(&f, fmt(6, 1, 8, 1)).expect("decode"),
        vec![1, 2, 3, 4, 5, 6]
    );
}

/// **The trap.** A 16-bit frame is two segments: the first holds the *high*
/// byte of every pixel and the second the low byte. A decoder that concatenated
/// them, or that put them the other way round, produces recognisable anatomy
/// with the contrast of noise — every sample's halves swapped.
#[test]
fn a_sixteen_bit_frame_recomposes_its_samples_from_two_byte_planes() {
    let want: [u16; 4] = [0x0102, 0xABCD, 0x0000, 0xFFFF];
    let high: Vec<u8> = want.iter().map(|v| (v >> 8) as u8).collect();
    let low: Vec<u8> = want.iter().map(|v| (v & 0xFF) as u8).collect();
    // Segment order is most-significant byte first, per PS3.5 G.2.
    let f = fragment(&[literal(&high), literal(&low)]);

    let out = decode(&f, fmt(4, 1, 16, 1)).expect("decode");
    // Native little-endian: low byte first within each sample.
    let got: Vec<u16> = out
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got, want.to_vec());
}

/// Colour is one segment per component, and the output is interleaved — which
/// is what the rest of the reader expects and why `PlanarConfiguration` stops
/// applying to a decoded frame.
#[test]
fn an_eight_bit_colour_frame_interleaves_its_three_planes() {
    let r = [1u8, 2, 3];
    let g = [10u8, 20, 30];
    let b = [100u8, 200, 250];
    let f = fragment(&[literal(&r), literal(&g), literal(&b)]);
    assert_eq!(
        decode(&f, fmt(3, 1, 8, 3)).expect("decode"),
        vec![1, 10, 100, 2, 20, 200, 3, 30, 250]
    );
}

/// Sixteen-bit colour is six segments, two per component, and the MSB-first
/// rule applies within each component.
#[test]
fn a_sixteen_bit_colour_frame_takes_six_segments() {
    let want: [[u16; 2]; 3] = [[0x0111, 0x0222], [0x1333, 0x1444], [0x2555, 0x2666]];
    let mut segs = Vec::new();
    for c in want {
        segs.push(literal(
            &c.iter().map(|v| (v >> 8) as u8).collect::<Vec<u8>>(),
        ));
        segs.push(literal(
            &c.iter().map(|v| (v & 0xFF) as u8).collect::<Vec<u8>>(),
        ));
    }
    let out = decode(&fragment(&segs), fmt(2, 1, 16, 3)).expect("decode");
    let got: Vec<u16> = out
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    // Interleaved: pixel 0's three components, then pixel 1's.
    assert_eq!(got, vec![0x0111, 0x1333, 0x2555, 0x0222, 0x1444, 0x2666]);
}

/// The segment count *is* the layout. A 16-bit frame declaring two segments has
/// one byte position missing and there is no way to guess which, so this is a
/// refusal rather than a best effort.
#[test]
fn a_segment_count_that_disagrees_with_the_header_is_refused() {
    // One segment for a 16-bit image, which needs two.
    let f = fragment(&[literal(&[1, 2, 3, 4])]);
    let err = decode(&f, fmt(2, 1, 16, 1)).expect_err("should refuse");
    assert!(format!("{err}").contains("segment"), "{err}");
}

#[test]
fn a_fragment_shorter_than_its_own_segment_table_is_refused() {
    assert!(decode(&[0u8; 20], fmt(4, 1, 8, 1)).is_err());
}

/// A segment that unpacks short is a truncated frame, and padding it would
/// finish the image with a plausible run of one colour.
#[test]
fn a_segment_that_unpacks_short_is_refused() {
    let f = fragment(&[literal(&[1, 2])]);
    let err = decode(&f, fmt(10, 1, 8, 1)).expect_err("should refuse");
    assert!(format!("{err}").contains("unpacked to"), "{err}");
}

/// Segments are allowed to overrun by a byte, because each is padded to an even
/// length. The extra must be dropped rather than shifting the next row.
#[test]
fn a_segment_that_unpacks_long_is_truncated_to_the_image() {
    let f = fragment(&[literal(&[1, 2, 3, 4, 5, 6])]);
    assert_eq!(
        decode(&f, fmt(5, 1, 8, 1)).expect("decode"),
        vec![1, 2, 3, 4, 5]
    );
}

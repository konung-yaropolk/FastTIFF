//! Lossless JPEG, decoded from a stream assembled bit by bit.
//!
//! There is no small real file to borrow and a hospital's CT is not something
//! to commit, so the fixture is built here — which is also the only way to pin
//! the prediction rules one at a time. A real file exercises all of them at
//! once and tells you nothing about which is wrong.
//!
//! The Huffman table below is deliberately trivial: two codes, one bit each.
//! That makes the bit stream short enough to write out by hand and reason about
//! in the comments, which is the whole point — the arithmetic under test is the
//! *prediction*, not the entropy coding.

use super::super::pixels::Format;
use super::*;

fn fmt(cols: u32, rows: u32, bits: u16) -> Format {
    Format {
        rows,
        columns: cols,
        frames: 1,
        samples: 1,
        bits_allocated: if bits > 8 { 16 } else { 8 },
        bits_stored: bits,
        high_bit: bits - 1,
        signed: false,
        planar: false,
        inverted: false,
        palette: false,
    }
}

/// Build a lossless JPEG: `SOI`, `SOF3`, `DHT`, `SOS`, the bits, `EOI`.
///
/// The table has two codes of one bit: `0` means magnitude category 0 (a
/// difference of exactly zero, no extra bits) and `1` means category 1 (one
/// extra bit: `1` for +1, `0` for −1).
fn stream(w: u16, h: u16, precision: u8, predictor: u8, bits: &str) -> Vec<u8> {
    let mut out = vec![0xFF, 0xD8]; // SOI

    // SOF3: Lf, P, Y, X, Nf, then one component (Ci, H/V, Tq).
    out.extend_from_slice(&[0xFF, 0xC3, 0x00, 0x0B, precision]);
    out.extend_from_slice(&h.to_be_bytes());
    out.extend_from_slice(&w.to_be_bytes());
    out.extend_from_slice(&[0x01, 0x01, 0x11, 0x00]);

    // DHT: Lh, Tc/Th = 0, sixteen counts, then the values.
    // Two codes of length 1 → counts[0] = 2, values = [0, 1].
    let mut counts = [0u8; 16];
    counts[0] = 2;
    let values = [0u8, 1];
    let lh = 2 + 1 + 16 + values.len();
    out.extend_from_slice(&[0xFF, 0xC4]);
    out.extend_from_slice(&(lh as u16).to_be_bytes());
    out.push(0x00);
    out.extend_from_slice(&counts);
    out.extend_from_slice(&values);

    // SOS: Ls, Ns, (Cs, Td/Ta), Ss = the predictor, Se = 0, Ah/Al = 0.
    out.extend_from_slice(&[
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, predictor, 0x00, 0x00,
    ]);

    // The entropy-coded bits, padded to a byte with 1s as JPEG requires.
    let mut packed = Vec::new();
    let mut acc = 0u8;
    let mut n = 0u8;
    for c in bits.chars().filter(|c| *c == '0' || *c == '1') {
        acc = (acc << 1) | u8::from(c == '1');
        n += 1;
        if n == 8 {
            packed.push(acc);
            // A real `FF` in the data is stuffed with a following `00`.
            if acc == 0xFF {
                packed.push(0x00);
            }
            acc = 0;
            n = 0;
        }
    }
    if n > 0 {
        packed.push((acc << (8 - n)) | ((1u8 << (8 - n)) - 1));
    }
    out.extend_from_slice(&packed);
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    out
}

/// Decode and read back as `u16` samples.
fn samples(b: &[u8], f: Format) -> Vec<u16> {
    let out = decode(b, f).expect("decode");
    if f.sample_bytes() == 1 {
        out.into_iter().map(u16::from).collect()
    } else {
        out.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

/// The first sample of the image has no neighbours, so T.81 gives it a fixed
/// prediction of half the range — `2^(P-1)`, which for 8-bit precision is 128.
/// A decoder that started from zero would make the whole image too dark by
/// exactly that much, which on a windowed CT looks like a contrast problem
/// rather than a decode one.
#[test]
fn the_first_sample_is_predicted_from_half_the_range() {
    // One pixel, category 0 → difference 0 → the prediction itself.
    let s = samples(&stream(1, 1, 8, 1, "0"), fmt(1, 1, 8));
    assert_eq!(s, vec![128]);
    // And at 12-bit precision it is 2048, not 128.
    let s = samples(&stream(1, 1, 12, 1, "0"), fmt(1, 1, 12));
    assert_eq!(s, vec![2048]);
}

/// Predictor 1 is the sample to the left, which is what the `.70` transfer
/// syntax pins and what essentially every encoder emits.
#[test]
fn predictor_one_takes_the_sample_to_its_left() {
    // 128, then +1, then +1, then 0 → 128, 129, 130, 130.
    let s = samples(&stream(4, 1, 8, 1, "0 1 1 1 1 0"), fmt(4, 1, 8));
    assert_eq!(s, vec![128, 129, 130, 130]);
}

/// The sign convention. Category 1 with its extra bit clear is −1, not +0 and
/// not +1: T.81's extension makes the *low* half of each magnitude range
/// negative. A decoder with this backwards produces an image that drifts the
/// wrong way and looks like a gradient artefact.
#[test]
fn a_difference_is_sign_extended_the_way_the_spec_says() {
    // 128, then −1, then −1 → 128, 127, 126.
    let s = samples(&stream(3, 1, 8, 1, "0 1 0 1 0"), fmt(3, 1, 8));
    assert_eq!(s, vec![128, 127, 126]);
}

/// The first sample of every row after the first is predicted from the sample
/// **above**, not from the end of the previous row. Using predictor 1 there
/// would carry the right-hand edge of each row into the left-hand edge of the
/// next, which on anatomy shows as a diagonal smear.
#[test]
fn the_first_sample_of_a_row_is_predicted_from_the_row_above() {
    // Row 0: 128, 129 (via +1). Row 1: first sample predicted from 128 (above)
    // with +1 → 129; second predicted from 129 (left, predictor 1) with 0 → 129.
    let s = samples(&stream(2, 2, 8, 1, "0 1 1   1 1 0"), fmt(2, 2, 8));
    assert_eq!(s, vec![128, 129, 129, 129]);
}

/// Predictor 2 is the sample above. Within the first row there is nothing
/// above, so the row falls back to the left — which is a rule of its own and
/// not predictor 2 applied to a zero.
#[test]
fn predictor_two_takes_the_sample_above() {
    // Row 0: 128, then +1 → 129 (first row uses the left neighbour whatever
    // the predictor says). Row 1: above-left is 128 → +1 → 129; then above is
    // 129 → 0 → 129.
    let s = samples(&stream(2, 2, 8, 2, "0 1 1   1 1 0"), fmt(2, 2, 8));
    assert_eq!(s, vec![128, 129, 129, 129]);
}

/// Reconstruction is modulo `2^P`, which is the format's own arithmetic — T.81
/// H.2.2 reduces every sample that way. It is also what makes a decode of a
/// deliberately wrapping difference well-defined rather than a clamp.
#[test]
fn reconstruction_wraps_at_the_declared_precision() {
    // 2-bit precision: the first sample is 2^(2-1) = 2, then +1 → 3, then +1
    // wraps to 0.
    let s = samples(&stream(3, 1, 2, 1, "0 1 1 1 1"), fmt(3, 1, 2));
    assert_eq!(s, vec![2, 3, 0]);
}

/// A stream that ends mid-sample is a truncated frame. Finishing it with zeros
/// would complete the image with a plausible flat region instead.
#[test]
fn a_truncated_stream_is_reported() {
    // Four pixels promised, one difference supplied, and the padding bits read
    // as more categories than there is data for.
    let mut b = stream(64, 64, 8, 1, "0");
    b.truncate(b.len() - 2); // drop EOI, so the reader runs off the end
    assert!(decode(&b, fmt(64, 64, 8)).is_err());
}

/// The geometry has to agree with the dataset's, because the slice spacing and
/// the series assembly were built from the dataset.
#[test]
fn a_frame_whose_size_disagrees_with_the_header_is_refused() {
    let b = stream(4, 1, 8, 1, "0 0 0 0");
    let err = decode(&b, fmt(8, 1, 8)).expect_err("should refuse");
    assert!(format!("{err}").contains("4x1"), "{err}");
}

/// Colour lossless is rare enough that an untested path for it would be worse
/// than a message naming the conversion.
///
/// The `SOF3` is rebuilt rather than patched in place: its own length field
/// covers the component descriptors, so raising `Nf` without adding them makes
/// a *truncated* header and tests the wrong refusal.
#[test]
fn a_multi_component_lossless_frame_is_refused_by_name() {
    let one = stream(2, 2, 8, 1, "0 0 0 0");
    let sof = one
        .windows(2)
        .position(|w| w == [0xFF, 0xC3])
        .expect("the frame header");
    // Lf, P, Y, X, Nf, then three (Ci, H/V, Tq) triples. Lf is 8 + 3 * Nf.
    let mut three = one[..sof + 2].to_vec();
    three.extend_from_slice(&(8u16 + 9).to_be_bytes());
    three.push(8); // P
    three.extend_from_slice(&2u16.to_be_bytes()); // Y
    three.extend_from_slice(&2u16.to_be_bytes()); // X
    three.push(3); // Nf
    for c in 1..=3u8 {
        three.extend_from_slice(&[c, 0x11, 0x00]);
    }
    // Everything after the one-component SOF3, which was 2 + 11 bytes long.
    three.extend_from_slice(&one[sof + 2 + 11..]);

    let err = decode(&three, fmt(2, 2, 8)).expect_err("should refuse");
    let msg = format!("{err}");
    assert!(msg.contains("3-component"), "{msg}");
    assert!(msg.contains("gdcmconv"), "should say how: {msg}");
}

/// A predictor outside 1..=7 is a stream asking for something the format does
/// not define, and guessing one would give a picture built from the wrong
/// neighbour.
#[test]
fn a_predictor_the_format_does_not_define_is_refused() {
    // Two rows, so the predictor is actually consulted — the first row uses the
    // left neighbour whatever it says.
    let b = stream(2, 2, 8, 9, "0 1 1 1 1 0");
    assert!(decode(&b, fmt(2, 2, 8)).is_err());
}

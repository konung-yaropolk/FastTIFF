//! Five tags deciding what a sample means, and the ways of getting it wrong
//! that still produce a picture.
//!
//! Each test here fixes one of those tags against a value the others would
//! otherwise hide. The 12-bits-in-16 cases are the ones worth reading: that is
//! the commonest real configuration outside MRI, and the two ways of mishandling
//! it — not masking, and masking without sign-extending — both give images that
//! look like images.

use super::super::parse::tests::{Dcm, EXPLICIT_LE, IMPLICIT_LE};
use super::super::parse::{meta, scan};
use super::*;

/// Build a `Found` holding the given (0028,xxxx) values.
fn fmt_from(set: &[(Tag, u16)], text: &[(Tag, &str)]) -> Found {
    let mut d = Dcm::new(EXPLICIT_LE);
    for (tag, v) in set {
        d.us(*tag, *v);
    }
    for (tag, v) in text {
        d.text(*tag, b"CS", v);
    }
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    let wanted: Vec<Tag> = set
        .iter()
        .map(|(t, _)| *t)
        .chain(text.iter().map(|(t, _)| *t))
        .collect();
    scan(&file, at, syntax, &wanted).expect("scan").0
}

/// The common MRI shape: 16 bits allocated, 16 stored, unsigned.
fn plain_u16(rows: u16, cols: u16) -> Format {
    Format::of(
        &fmt_from(
            &[
                (ROWS, rows),
                (COLUMNS, cols),
                (BITS_ALLOCATED, 16),
                (BITS_STORED, 16),
                (HIGH_BIT, 15),
                (PIXEL_REPRESENTATION, 0),
            ],
            &[],
        ),
        false,
    )
    .expect("a plain format")
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn as_u16(p: &PlaneData) -> Vec<u16> {
    match p {
        PlaneData::U16(v) => v.clone(),
        other => panic!("expected the u16 lane, got {other:?}"),
    }
}

#[test]
fn an_unsigned_sixteen_bit_frame_arrives_unchanged() {
    let fmt = plain_u16(2, 3);
    let want = [0u16, 1, 100, 1000, 40000, 65535];
    let planes = planes_of_frame(&u16_bytes(&want), fmt, 0).expect("planes");
    assert_eq!(planes.len(), 1);
    assert_eq!(as_u16(&planes[0]), want.to_vec());
    assert_eq!(fmt.pixel_type(), PixelType::U16);
}

/// Hounsfield units are negative for air, so `PixelRepresentation` ignored
/// turns a CT's background from black to brilliant white. On MRI the same bug
/// shows up as a sprinkling of hot pixels indistinguishable from noise.
#[test]
fn signed_samples_keep_their_sign() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 5),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 16),
            (HIGH_BIT, 15),
            (PIXEL_REPRESENTATION, 1),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    assert_eq!(fmt.pixel_type(), PixelType::I16);

    let want: [i16; 5] = [-32768, -1000, -1, 0, 32767];
    let raw: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    let back: Vec<i16> = as_u16(&planes[0]).iter().map(|&u| u as i16).collect();
    assert_eq!(back, want.to_vec());
}

/// 12 bits in 16 is what CT and CR write. The top four bits are padding and are
/// often zero — which is exactly what makes this dangerous: the reader that
/// does not mask works on most files and scales by sixteen on the rest.
#[test]
fn stored_bits_narrower_than_the_allocation_are_masked() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 12),
            (HIGH_BIT, 11),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    // Each word has junk in its top four bits, as a scanner that did not clear
    // them leaves it.
    let raw = u16_bytes(&[0xF000, 0xF001, 0xFFFF, 0x7ABC]);
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    assert_eq!(
        as_u16(&planes[0]),
        vec![0x000, 0x001, 0xFFF, 0xABC],
        "only the 12 stored bits are the sample"
    );
}

/// And the subtle one: a *signed* 12-bit sample's sign bit is bit 11, not bit
/// 15. Masking without extending from the stored width leaves 0xFFF as 4095
/// instead of −1 — a reading that is wrong by 4096 and entirely plausible.
#[test]
fn a_signed_narrow_sample_is_sign_extended_from_its_stored_width() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 12),
            (HIGH_BIT, 11),
            (PIXEL_REPRESENTATION, 1),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    // 0xFFF is −1 in 12-bit two's complement; 0x800 is −2048, the most
    // negative; 0x7FF is 2047, the most positive.
    let raw = u16_bytes(&[0x0FFF, 0x0800, 0x07FF, 0x0000]);
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    let back: Vec<i16> = as_u16(&planes[0]).iter().map(|&u| u as i16).collect();
    assert_eq!(back, vec![-1i16, -2048, 2047, 0]);
}

/// `HighBit` is normally the top of the stored range, and when it is not the
/// samples sit shifted up inside the word. Honoured rather than assumed, since
/// assuming it costs a factor of two per bit of shift.
#[test]
fn a_high_bit_above_the_stored_width_shifts_the_samples_down() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 3),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 8),
            (HIGH_BIT, 11), // eight bits sitting at bits 4..11
            (PIXEL_REPRESENTATION, 0),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    let raw = u16_bytes(&[0x0FF0, 0x0010, 0x0AB0]);
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    assert_eq!(as_u16(&planes[0]), vec![0xFF, 0x01, 0xAB]);
}

/// An 8-bit unsigned image stays in the byte lane, which halves what the stack
/// costs in memory and is what the file actually holds.
#[test]
fn an_eight_bit_frame_stays_in_the_byte_lane() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    assert_eq!(fmt.pixel_type(), PixelType::U8);
    let planes = planes_of_frame(&[0, 7, 128, 255], fmt, 0).expect("planes");
    assert_eq!(planes[0], PlaneData::U8(vec![0, 7, 128, 255]));
}

/// A *signed* byte has to be widened, because the contract has no signed-byte
/// lane and `U8` would make every negative sample bright.
#[test]
fn a_signed_eight_bit_frame_is_widened_to_the_signed_lane() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 3),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
            (PIXEL_REPRESENTATION, 1),
        ],
        &[],
    );
    let fmt = Format::of(&f, false).expect("format");
    assert_eq!(fmt.pixel_type(), PixelType::I16);
    let planes = planes_of_frame(&[0x80, 0xFF, 0x7F], fmt, 0).expect("planes");
    let back: Vec<i16> = as_u16(&planes[0]).iter().map(|&u| u as i16).collect();
    assert_eq!(back, vec![-128i16, -1, 127]);
}

/// Colour, interleaved — `PlanarConfiguration` 0, which is what almost
/// everything writes. The channels come out as separate planes because that is
/// what `xyczt` wants and what lets each be windowed.
#[test]
fn an_interleaved_colour_frame_splits_into_three_planes() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 3),
            (SAMPLES_PER_PIXEL, 3),
            (PLANAR_CONFIG, 0),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[(PHOTOMETRIC, "RGB")],
    );
    let fmt = Format::of(&f, false).expect("format");
    // r0 g0 b0 r1 g1 b1 r2 g2 b2
    let raw = [10u8, 20, 30, 11, 21, 31, 12, 22, 32];
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    assert_eq!(planes.len(), 3);
    assert_eq!(planes[0], PlaneData::U8(vec![10, 11, 12]));
    assert_eq!(planes[1], PlaneData::U8(vec![20, 21, 22]));
    assert_eq!(planes[2], PlaneData::U8(vec![30, 31, 32]));
}

/// Colour, planar — each channel's whole plane one after another. Reading one
/// layout as the other gives a picture in three vertical bands of wrong
/// colour, which is obvious once seen and easy to never test for.
#[test]
fn a_planar_colour_frame_splits_into_the_same_three_planes() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 3),
            (SAMPLES_PER_PIXEL, 3),
            (PLANAR_CONFIG, 1),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[(PHOTOMETRIC, "RGB")],
    );
    let fmt = Format::of(&f, false).expect("format");
    // rrr ggg bbb
    let raw = [10u8, 11, 12, 20, 21, 22, 30, 31, 32];
    let planes = planes_of_frame(&raw, fmt, 0).expect("planes");
    assert_eq!(planes[0], PlaneData::U8(vec![10, 11, 12]));
    assert_eq!(planes[1], PlaneData::U8(vec![20, 21, 22]));
    assert_eq!(planes[2], PlaneData::U8(vec![30, 31, 32]));
}

/// A multi-frame file's frames are at consecutive offsets, and reading frame
/// `n` from offset zero would show the first slice `n` times.
#[test]
fn each_frame_of_a_multi_frame_file_comes_from_its_own_offset() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 2),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 16),
            (HIGH_BIT, 15),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[],
    );
    let mut fmt = Format::of(&f, false).expect("format");
    fmt.frames = 3;
    let raw = u16_bytes(&[1, 2, 3, 4, 5, 6]);
    for (frame, want) in [(0, vec![1, 2]), (1, vec![3, 4]), (2, vec![5, 6])] {
        let planes = planes_of_frame(&raw, fmt, frame).expect("planes");
        assert_eq!(as_u16(&planes[0]), want, "frame {frame}");
    }
    // And one past the end is an error, not a wrap.
    assert!(planes_of_frame(&raw, fmt, 3).is_err());
}

#[test]
fn a_frame_shorter_than_its_declared_size_is_an_error() {
    let fmt = plain_u16(8, 8);
    let err = planes_of_frame(&u16_bytes(&[0; 10]), fmt, 0).expect_err("should refuse");
    assert!(format!("{err}").contains("ends in the middle"), "{err}");
}

/// Rescale is *reported*, not applied: the stored values stay as the file has
/// them and the host carries the calibration, which is the same arrangement
/// the TIFF reader uses. Applying it here would make the stored values
/// disagree with every other tool's reading of the same file.
#[test]
fn rescale_is_reported_rather_than_applied() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(RESCALE_SLOPE, b"DS", "2.5")
        .text(RESCALE_INTERCEPT, b"DS", "-1024");
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    let (found, _) = scan(&file, at, syntax, &[RESCALE_SLOPE, RESCALE_INTERCEPT]).expect("scan");
    assert_eq!(rescale(&found), Some((-1024.0, 2.5)));
}

/// An identity rescale is no rescale, so an ordinary MRI does not arrive
/// carrying a redundant calibration — and a file with no rescale tags at all
/// defaults to identity, which is also nothing.
#[test]
fn an_identity_or_absent_rescale_is_reported_as_none() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(RESCALE_SLOPE, b"DS", "1")
        .text(RESCALE_INTERCEPT, b"DS", "0");
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    let (found, _) = scan(&file, at, syntax, &[RESCALE_SLOPE, RESCALE_INTERCEPT]).expect("scan");
    assert_eq!(rescale(&found), None);
    assert_eq!(rescale(&Found::default()), None);
}

/// `MONOCHROME1` is flagged, not inverted. Flipping the samples here would
/// make every value disagree with the rescale calibration stated against them,
/// and nothing downstream could tell.
#[test]
fn monochrome1_is_flagged_and_the_samples_are_left_alone() {
    let f = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 2),
            (BITS_ALLOCATED, 16),
            (BITS_STORED, 16),
            (HIGH_BIT, 15),
            (PIXEL_REPRESENTATION, 0),
        ],
        &[(PHOTOMETRIC, "MONOCHROME1")],
    );
    let fmt = Format::of(&f, false).expect("format");
    assert!(fmt.inverted, "should have noticed");
    let planes = planes_of_frame(&u16_bytes(&[7, 9]), fmt, 0).expect("planes");
    assert_eq!(
        as_u16(&planes[0]),
        vec![7, 9],
        "the samples must be the file's own"
    );
}

/// Palette colour is an index image plus lookup tables, which is a third
/// pipeline beside greyscale and RGB. Refused by name, with the conversion.
#[test]
fn palette_colour_is_recognised_as_indices_rather_than_intensities() {
    let f = fmt_from(
        &[
            (ROWS, 4),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
        ],
        &[(PHOTOMETRIC, "PALETTE COLOR")],
    );
    let fmt = Format::of(&f, false).expect("read, not refused");
    assert!(fmt.palette, "should have noticed");
    // The file still says one sample per pixel — the index. The three channels
    // appear only once the table has been applied.
    assert_eq!(fmt.samples, 1);
}

/// The combinations that cannot be read are refused rather than approximated.
#[test]
fn what_cannot_be_read_is_refused_with_a_reason() {
    // 32 bits per sample.
    let f = fmt_from(&[(ROWS, 4), (COLUMNS, 4), (BITS_ALLOCATED, 32)], &[]);
    assert!(format!("{:?}", Format::of(&f, false)).contains("32 bits"));

    // More bits stored than allocated.
    let f = fmt_from(
        &[
            (ROWS, 4),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 16),
        ],
        &[],
    );
    assert!(Format::of(&f, false).is_err());

    // Two samples per pixel, which is neither greyscale nor colour.
    let f = fmt_from(
        &[
            (ROWS, 4),
            (COLUMNS, 4),
            (SAMPLES_PER_PIXEL, 2),
            (BITS_ALLOCATED, 8),
        ],
        &[],
    );
    assert!(Format::of(&f, false).is_err());

    // No dimensions at all.
    assert!(Format::of(&Found::default(), false).is_err());

    // A zero dimension.
    let f = fmt_from(&[(ROWS, 0), (COLUMNS, 4), (BITS_ALLOCATED, 8)], &[]);
    assert!(Format::of(&f, false).is_err());
}

/// Big-endian pixel data is swapped once per file. A 16-bit image read with its
/// bytes the wrong way round is noise rather than a subtly wrong picture, which
/// at least fails loudly — but it should not fail at all.
#[test]
fn big_endian_pixel_data_is_swapped_to_the_readers_order() {
    let fmt = plain_u16(1, 3);
    let mut data: Vec<u8> = [1u16, 256, 65535]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    to_little_endian(&mut data, fmt);
    let planes = planes_of_frame(&data, fmt, 0).expect("planes");
    assert_eq!(as_u16(&planes[0]), vec![1, 256, 65535]);

    // An 8-bit image has nothing to swap, and swapping it would reverse pairs
    // of pixels.
    let eight = fmt_from(
        &[
            (ROWS, 1),
            (COLUMNS, 4),
            (BITS_ALLOCATED, 8),
            (BITS_STORED, 8),
            (HIGH_BIT, 7),
        ],
        &[],
    );
    let eight = Format::of(&eight, false).expect("format");
    let mut bytes = vec![1u8, 2, 3, 4];
    to_little_endian(&mut bytes, eight);
    assert_eq!(bytes, vec![1, 2, 3, 4]);
}

/// Implicit VR carries no VR for `US` either, so the dimensions have to be read
/// as the two little-endian bytes they are rather than looked up.
#[test]
fn dimensions_are_read_from_an_implicit_vr_dataset_too() {
    let mut d = Dcm::new(IMPLICIT_LE);
    d.us(ROWS, 320).us(COLUMNS, 260).us(BITS_ALLOCATED, 16);
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    let (found, _) = scan(&file, at, syntax, &[ROWS, COLUMNS, BITS_ALLOCATED]).expect("scan");
    let fmt = Format::of(&found, false).expect("format");
    assert_eq!((fmt.rows, fmt.columns), (320, 260));
}

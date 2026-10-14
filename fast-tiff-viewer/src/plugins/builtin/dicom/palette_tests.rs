//! The colour table, and the clamping rule that is easy to miss.

use super::super::parse::tests::{Dcm, EXPLICIT_LE};
use super::super::parse::{meta, scan};
use super::super::pixels::Format;
use super::*;

fn fmt(cols: u32, rows: u32, bits: u16) -> Format {
    Format {
        rows,
        columns: cols,
        frames: 1,
        samples: 1,
        bits_allocated: bits,
        bits_stored: bits,
        high_bit: bits - 1,
        signed: false,
        planar: false,
        inverted: false,
        palette: true,
    }
}

/// A `Found` holding a three-channel table of `entries` entries starting at
/// `first`, with `bits` per entry. Each channel's entry `i` is
/// `i * 1 | 2 | 3` so a channel mix-up is visible.
fn table(entries: u16, first: u16, bits: u16) -> Found {
    let mut d = Dcm::new(EXPLICIT_LE);
    let n = if entries == 0 {
        65536usize
    } else {
        entries as usize
    };
    for (ch, mult) in [(0x1101u16, 1u32), (0x1102, 2), (0x1103, 3)] {
        let desc: Vec<u8> = [entries, first, bits]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        d.el((0x0028, ch), b"US", &desc);
        let data: Vec<u8> = if bits > 8 {
            (0..n)
                .flat_map(|i| ((i as u32 * mult) as u16).to_le_bytes())
                .collect()
        } else {
            (0..n).map(|i| (i as u32 * mult) as u8).collect()
        };
        d.el((0x0028, ch + 0x100), b"OW", &data);
    }
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    scan(&file, at, syntax, &TAGS).expect("scan").0
}

fn as_u16(p: &PlaneData) -> Vec<u16> {
    match p {
        PlaneData::U16(v) => v.clone(),
        other => panic!("expected u16, got {other:?}"),
    }
}

/// The headline: an index becomes three colour components, and each comes from
/// its own channel's table. A reader that used the red table for all three
/// would produce a grey image that looks like a successful decode.
#[test]
fn an_index_is_looked_up_in_each_channels_own_table() {
    let p = Palette::of(&table(256, 0, 16), false).expect("a table");
    let planes = p.expand(&[0u8, 1, 2, 10], fmt(4, 1, 8), 0).expect("expand");
    assert_eq!(planes.len(), 3);
    assert_eq!(as_u16(&planes[0]), vec![0, 1, 2, 10], "red is i * 1");
    assert_eq!(as_u16(&planes[1]), vec![0, 2, 4, 20], "green is i * 2");
    assert_eq!(as_u16(&planes[2]), vec![0, 3, 6, 30], "blue is i * 3");
}

/// Indices outside the table are **clamped** to its ends, which is the
/// standard's rule. Wrapping instead — which an unchecked cast does — paints
/// the out-of-range pixels with whatever colour sits at the other end, and on a
/// Doppler palette that is a bright splash of the wrong colour in the corners.
#[test]
fn an_index_outside_the_table_is_clamped_to_its_ends() {
    // A 16-entry table whose first mapped value is 8, so indices 0..7 are
    // below it and 24.. are above.
    let p = Palette::of(&table(16, 8, 16), false).expect("a table");
    let planes = p
        .expand(&[0u8, 7, 8, 9, 23, 24, 255], fmt(7, 1, 8), 0)
        .expect("expand");
    assert_eq!(
        as_u16(&planes[0]),
        // 0 and 7 clamp to entry 0; 8 is entry 0; 9 is entry 1; 23 is entry 15;
        // 24 and 255 clamp to entry 15.
        vec![0, 0, 0, 1, 15, 15, 15]
    );
}

/// A count of zero means 65536 — the one place DICOM uses zero for a
/// full-range value, because 65536 does not fit the `US` it is stored in.
/// Reading it literally gives an empty table and then a division or a panic.
#[test]
fn a_declared_count_of_zero_means_a_full_sixteen_bit_table() {
    let p = Palette::of(&table(0, 0, 8), false).expect("a table");
    // Index 300 is in range for a 65536-entry table, so it is not clamped.
    let planes = p.expand(&[0u8, 255], fmt(2, 1, 8), 0).expect("expand");
    assert_eq!(planes.len(), 3);
    assert!(matches!(planes[0], PlaneData::U8(_)), "8-bit entries");
}

/// Eight-bit entries give eight-bit output, which halves what the stack costs
/// and is what the file holds.
#[test]
fn the_entry_width_decides_the_output_width() {
    let eight = Palette::of(&table(256, 0, 8), false).expect("a table");
    assert_eq!(eight.pixel_type(), PixelType::U8);
    let wide = Palette::of(&table(256, 0, 16), false).expect("a table");
    assert_eq!(wide.pixel_type(), PixelType::U16);
}

/// The segmented form is run-length-coded rather than a plain table, so it is
/// refused by name with the conversion rather than read as if it were one.
#[test]
fn a_segmented_colour_table_is_refused_by_name() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.el((0x0028, 0x1221), b"OW", &[1, 2, 3, 4]);
    let file = d.build();
    let (syntax, _codec, at) = meta(&file).expect("meta");
    let found = scan(&file, at, syntax, &TAGS).expect("scan").0;
    let err = Palette::of(&found, false).expect_err("should refuse");
    assert!(format!("{err}").contains("segmented"), "{err}");
    assert!(
        format!("{err}").contains("gdcmconv"),
        "should say how: {err}"
    );
}

/// A file that says it is palette-colour and then carries no table is a file
/// whose halves disagree, and there is nothing to guess.
#[test]
fn a_palette_image_with_no_table_is_refused() {
    let err = Palette::of(&Found::default(), false).expect_err("should refuse");
    assert!(format!("{err}").contains("descriptors"), "{err}");
}

/// A bits-per-entry the format does not define is refused rather than rounded
/// to one that is.
#[test]
fn an_impossible_entry_width_is_refused() {
    let err = Palette::of(&table(256, 0, 12), false).expect_err("should refuse");
    assert!(format!("{err}").contains("12 bits"), "{err}");
}

/// Each frame of a cine loop is at its own offset, and reading frame `n` from
/// offset zero would show the first frame `n` times.
#[test]
fn each_frame_of_a_palette_cine_comes_from_its_own_offset() {
    let p = Palette::of(&table(256, 0, 16), false).expect("a table");
    let indices = [1u8, 2, 3, 4, 5, 6];
    let mut f = fmt(2, 1, 8);
    f.frames = 3;
    for (frame, want) in [(0, vec![1, 2]), (1, vec![3, 4]), (2, vec![5, 6])] {
        let planes = p.expand(&indices, f, frame).expect("expand");
        assert_eq!(as_u16(&planes[0]), want, "frame {frame}");
    }
    assert!(
        p.expand(&indices, f, 3).is_err(),
        "past the end is an error"
    );
}

//! Decoding a horizontal band of a frame instead of the whole thing.
//!
//! This exists so a frame far too large to hold in memory can still be looked
//! at: the viewer shows a window of it, and decoding the rest is the dominant
//! cost of moving that window. `crop_rows` turns a frame into a smaller frame
//! covering only the strips it names, which every reader then handles as if it
//! were an ordinary file.
//!
//! Two properties carry the whole idea, and both are checked against a *full*
//! decode of the same file rather than against restated arithmetic:
//!
//! 1. a band holds exactly the pixels those rows hold — same values, same
//!    order, whatever the codec, predictor or interleaving;
//! 2. the band is snapped outward to strip boundaries, and says so, because a
//!    strip is the smallest thing that can be decompressed.
//!
//! A band that quietly returned the wrong rows would be the worst kind of bug
//! here: the picture would look perfectly plausible and be of somewhere else.

use fast_tiff_lib::{Compression, SampleType, TiffStack, TiffWriter, WriterOptions};
use std::io::Cursor;

const W: u32 = 37; // deliberately not a multiple of anything
const H: u32 = 51;

/// A single-frame stack whose pixel values are a known function of position, so
/// a band can be checked against where it claims to come from.
fn build(
    rows_per_strip: u32,
    compression: Compression,
    predictor: bool,
    planar: bool,
    spp: u16,
) -> Vec<u8> {
    let opts = WriterOptions::new(W, H, SampleType::U8)
        .samples_per_pixel(spp)
        .planar(planar)
        .compression(compression)
        .predictor(predictor)
        .rows_per_strip(rows_per_strip);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let n = (W * H) as usize * spp as usize;
    // Vary with the flat index so any misplacement shows up as a mismatch.
    let px: Vec<u8> = (0..n).map(|i| (i * 7 + 13) as u8).collect();
    w.write_frame_u8(&px).unwrap();
    w.finish().unwrap().into_inner()
}

/// Decode a whole frame, and the same rows as a band, and compare.
fn check(bytes: Vec<u8>, want_rows: std::ops::Range<u32>, label: &str) {
    let stack = TiffStack::from_bytes(bytes).unwrap();
    let frame = &stack.frames[0];
    let full = fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap();

    let band = frame.crop_rows(want_rows.clone()).unwrap();
    let cut = fast_tiff_lib::read_planes_u8(&stack.data, &band.frame, stack.byte_order).unwrap();

    // Snapped outward, never inward: the band must contain what was asked for.
    assert!(
        band.rows.start <= want_rows.start && band.rows.end >= want_rows.end.min(H),
        "{label}: band {:?} does not cover the requested {want_rows:?}",
        band.rows
    );
    assert_eq!(
        band.frame.height,
        band.len(),
        "{label}: height must match the row range"
    );
    assert_eq!(cut.len(), full.len(), "{label}: plane count");

    let w = W as usize;
    for (p, (cut_plane, full_plane)) in cut.iter().zip(&full).enumerate() {
        assert_eq!(
            cut_plane.len(),
            w * band.len() as usize,
            "{label}: plane {p} is not the band size"
        );
        let from = band.rows.start as usize * w;
        assert_eq!(
            cut_plane,
            &full_plane[from..from + cut_plane.len()],
            "{label}: plane {p} does not match rows {:?} of the whole frame",
            band.rows
        );
    }
}

#[test]
fn a_band_matches_the_same_rows_of_the_whole_frame() {
    check(
        build(1, Compression::None, false, false, 1),
        10..20,
        "uncompressed, 1 row per strip",
    );
}

/// Every codec, because the band is cut from the *strip table* and each codec
/// decompresses strips its own way. A band that worked uncompressed and not
/// under LZW would be a very confusing bug to meet in the wild.
#[test]
fn bands_work_under_every_codec() {
    for (name, c) in [
        ("none", Compression::None),
        ("lzw", Compression::Lzw),
        ("deflate", Compression::Deflate),
        ("packbits", Compression::PackBits),
    ] {
        check(build(4, c, false, false, 1), 12..30, name);
    }
}

/// Predictors difference each pixel from its neighbour *within a row*, never
/// across rows, which is exactly why a strip can be decoded on its own. If that
/// were not so, a band would come out with its contrast progressively wrong.
#[test]
fn bands_work_under_the_horizontal_predictor() {
    check(
        build(4, Compression::Lzw, true, false, 1),
        12..30,
        "lzw + predictor, gray",
    );
    check(
        build(4, Compression::Deflate, true, false, 3),
        12..30,
        "deflate + predictor, rgb",
    );
}

/// Planar frames keep one run of strips per sample plane, so a band has to take
/// the matching slice out of *each* run. Taking a single contiguous slice would
/// read plane 0 twice and never reach plane 2 — pixels that decode cleanly and
/// are the wrong colour.
#[test]
fn bands_take_the_matching_strips_from_every_plane() {
    check(
        build(4, Compression::None, false, true, 3),
        12..30,
        "planar rgb",
    );
    check(
        build(3, Compression::Lzw, true, true, 3),
        9..27,
        "planar rgb, lzw + predictor",
    );
}

#[test]
fn bands_work_for_chunky_multi_sample_frames() {
    check(
        build(5, Compression::Lzw, false, false, 3),
        7..29,
        "chunky rgb",
    );
}

/// A request that does not land on strip boundaries is widened to them, and the
/// returned range says so. A caller indexing against what it *asked for* rather
/// than what it *got* would be off by up to a strip.
#[test]
fn a_request_is_snapped_outward_to_strip_boundaries() {
    let stack = TiffStack::from_bytes(build(8, Compression::None, false, false, 1)).unwrap();
    let band = stack.frames[0].crop_rows(10..20).unwrap();
    assert_eq!(
        band.rows,
        8..24,
        "rows 10..20 live in the strips covering 8..24"
    );
    assert_eq!(band.frame.height, 16);
}

/// The last strip of a frame is usually short. A band ending there must stop at
/// the frame, not at the strip boundary past it.
#[test]
fn a_band_at_the_end_stops_at_the_last_row() {
    let stack = TiffStack::from_bytes(build(8, Compression::None, false, false, 1)).unwrap();
    let band = stack.frames[0].crop_rows(H - 3..H).unwrap();
    assert_eq!(band.rows.end, H, "the band cannot run past the frame");
    assert_eq!(band.frame.height, band.len());
    check(
        build(8, Compression::None, false, false, 1),
        H - 3..H,
        "final short strip",
    );
}

/// Asking past the end, or for nothing at all, still has to produce something
/// decodable — a frontend mid-gesture can ask for either.
#[test]
fn degenerate_requests_still_yield_a_decodable_band() {
    let stack = TiffStack::from_bytes(build(4, Compression::None, false, false, 1)).unwrap();
    let frame = &stack.frames[0];
    // The last one is backwards on purpose — built rather than written as a
    // literal, which the lints reject as a mistake. Here it is the input a
    // frontend produces when a drag inverts, and it must not be a panic.
    let backwards = std::ops::Range { start: 20, end: 5 };
    for rows in [0..0, H..H, H + 100..H + 200, backwards] {
        let band = frame
            .crop_rows(rows.clone())
            .unwrap_or_else(|e| panic!("{rows:?}: {e:#}"));
        assert!(!band.is_empty(), "{rows:?} produced an empty band");
        assert_eq!(band.frame.height, band.len());
        let cut = fast_tiff_lib::read_planes_u8(&stack.data, &band.frame, stack.byte_order)
            .unwrap_or_else(|e| panic!("{rows:?} did not decode: {e:#}"));
        assert_eq!(cut[0].len(), W as usize * band.len() as usize);
    }
}

/// The whole frame as a band is the whole frame.
#[test]
fn asking_for_everything_returns_everything() {
    let stack = TiffStack::from_bytes(build(4, Compression::Lzw, false, false, 1)).unwrap();
    let frame = &stack.frames[0];
    let band = frame.crop_rows(0..H).unwrap();
    assert_eq!(band.rows, 0..H);
    assert_eq!(band.frame.strip_offsets.len(), frame.strip_offsets.len());
}

/// A strip table too short to describe the frame is refused. The table comes
/// from the file and the geometry comes from other tags in the same file, so an
/// untrusted one can disagree; slicing a table that does not add up would read
/// whatever bytes happened to follow.
#[test]
fn a_strip_table_that_does_not_describe_the_frame_is_refused() {
    let stack = TiffStack::from_bytes(build(4, Compression::None, false, false, 1)).unwrap();
    let mut frame = stack.frames[0].clone();
    let full = frame.strip_offsets.len();
    assert!(full > 2, "need a few strips for this to mean anything");

    frame.strip_offsets.truncate(full - 1);
    let err = frame.crop_rows(0..H).unwrap_err().to_string();
    assert!(err.contains("strip table"), "unhelpful error: {err}");

    let mut frame = stack.frames[0].clone();
    frame.strip_byte_counts.truncate(full - 1);
    assert!(
        frame.crop_rows(0..H).is_err(),
        "a short byte-count table is just as wrong"
    );
}

/// Bands compose with the rest of the crate: a band of a CMYK frame converts
/// like any other, because it *is* an ordinary frame.
#[test]
fn a_band_of_a_cmyk_frame_still_converts() {
    let opts = WriterOptions::new(W, H, SampleType::U8)
        .samples_per_pixel(4)
        .cmyk(true)
        .rows_per_strip(4);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let px: Vec<u8> = (0..(W * H) as usize * 4)
        .map(|i| (i * 7 + 13) as u8)
        .collect();
    w.write_frame_u8(&px).unwrap();
    let stack = TiffStack::from_bytes(w.finish().unwrap().into_inner()).unwrap();

    let frame = &stack.frames[0];
    assert!(frame.is_cmyk());
    let band = frame.crop_rows(8..20).unwrap();
    assert!(
        band.frame.is_cmyk(),
        "a band keeps the frame's photometric identity"
    );

    let full = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order).unwrap();
    let cut =
        fast_tiff_lib::read_planes_rgb_u8(&stack.data, &band.frame, stack.byte_order).unwrap();
    let from = band.rows.start as usize * W as usize;
    for (c, (cut_plane, full_plane)) in cut.iter().zip(&full).enumerate() {
        assert_eq!(
            cut_plane,
            &full_plane[from..from + cut_plane.len()],
            "component {c}"
        );
    }
}

// ---------------------------------------------------------------------------
// Stepping over pieces that will not be sampled
// ---------------------------------------------------------------------------

/// Sampling a frame coarsely means most strips hold nothing that will be drawn.
/// Stepping over them must return exactly the rows of the strips it *did* take —
/// getting that mapping wrong shows a different part of the image, convincingly.
#[test]
fn a_stepped_band_holds_the_rows_of_the_strips_it_took() {
    let bytes = build(2, Compression::Lzw, false, false, 1);
    let stack = TiffStack::from_bytes(bytes).unwrap();
    let frame = &stack.frames[0];
    let full = fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap();

    let step = 4; // every 4th strip of 2 rows: source rows 0,1, 8,9, 16,17, ...
    let band = frame.crop_rows_step(0..H, step).unwrap();
    let cut = fast_tiff_lib::read_planes_u8(&stack.data, &band.frame, stack.byte_order).unwrap();
    assert_eq!(band.rows_per_piece, 2);
    assert_eq!(band.step, step);

    let w = W as usize;
    for piece in 0..band.pieces as usize {
        for r in 0..band.rows_per_piece as usize {
            let src_row = band.first_row as usize + piece * step as usize * 2 + r;
            if src_row >= H as usize {
                break;
            }
            let got = &cut[0][(piece * 2 + r) * w..(piece * 2 + r) * w + w];
            let want = &full[0][src_row * w..src_row * w + w];
            assert_eq!(
                got, want,
                "piece {piece} row {r} should be source row {src_row}"
            );
        }
    }
}

/// The mapping in the other direction, which is what a caller actually uses.
#[test]
fn decoded_row_of_finds_sampled_rows_and_refuses_skipped_ones() {
    let stack = TiffStack::from_bytes(build(2, Compression::None, false, false, 1)).unwrap();
    let band = stack.frames[0].crop_rows_step(0..H, 4).unwrap();

    // Strips 0, 4, 8... were taken; each holds 2 rows.
    assert_eq!(band.decoded_row_of(0), Some(0));
    assert_eq!(band.decoded_row_of(1), Some(1));
    assert_eq!(
        band.decoded_row_of(8),
        Some(2),
        "the second strip taken follows the first"
    );
    assert_eq!(band.decoded_row_of(9), Some(3));
    assert_eq!(band.decoded_row_of(16), Some(4));
    // Rows in the strips that were stepped over simply are not there.
    for skipped in [2, 3, 4, 5, 6, 7, 10, 11] {
        assert_eq!(
            band.decoded_row_of(skipped),
            None,
            "row {skipped} was in a skipped strip"
        );
    }
}

/// A row index the result does not have is worse than no answer: the caller
/// uses it to index the decoded plane. The last strip of a frame is usually
/// short — 51 rows in strips of 2 leaves one holding a single row — and being
/// inside a strip that was taken does not put a row inside what that strip
/// actually held.
///
/// Found by the fuzzer, which reached it as a 1-row frame declaring 2 rows per
/// strip: `decoded_row_of(1)` answered 1 for a result one row tall.
#[test]
fn decoded_row_of_stays_inside_the_frame_it_built() {
    let stack = TiffStack::from_bytes(build(2, Compression::None, false, false, 1)).unwrap();
    // Step 5 over 26 strips takes strips 0, 5, 10, 15, 20, 25 — and 25 is the
    // short one, holding only row 50.
    let band = stack.frames[0].crop_rows_step(0..H, 5).unwrap();
    assert_eq!(
        band.decoded_row_of(50),
        Some(band.frame.height - 1),
        "the short strip's one row"
    );

    // Every answer, for every row anyone might ask about, has to be one the
    // decoded frame actually has.
    for source in 0..H + 4 {
        if let Some(row) = band.decoded_row_of(source) {
            assert!(
                row < band.frame.height,
                "source row {source} answered {row}, but the band is only {} rows",
                band.frame.height
            );
        }
    }
    assert_eq!(
        band.decoded_row_of(51),
        None,
        "past the end of the short strip that was taken"
    );
}

/// The shape the fuzzer actually found: a frame claiming more rows per strip
/// than it has rows at all. `RowsPerStrip` and `ImageLength` are separate tags,
/// so nothing but a reader stops them disagreeing — and this crate's own writer
/// never produces it, which is why it takes a hostile file to reach.
#[test]
fn a_frame_claiming_more_rows_per_strip_than_it_has() {
    let opts = WriterOptions::new(4, 1, SampleType::U8);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_u8(&[1, 2, 3, 4]).unwrap();
    let stack = TiffStack::from_bytes(w.finish().unwrap().into_inner()).unwrap();

    let mut frame = stack.frames[0].clone();
    assert_eq!(frame.height, 1);
    frame.rows_per_strip = 2; // the file says its one strip holds two rows

    for step in [1u32, 2, 3] {
        let band = frame.crop_rows_step(0..1, step).unwrap();
        assert_eq!(
            band.frame.height, 1,
            "step {step}: the file only has one row"
        );
        assert_eq!(band.decoded_row_of(0), Some(0), "step {step}");
        assert_eq!(
            band.decoded_row_of(1),
            None,
            "step {step}: the strip claims a second row the frame does not have"
        );
    }
}

/// The saving, stated as the property that motivates it.
#[test]
fn stepping_decodes_proportionally_fewer_strips() {
    let stack = TiffStack::from_bytes(build(2, Compression::None, false, false, 1)).unwrap();
    let frame = &stack.frames[0];
    let all = frame.crop_rows(0..H).unwrap().frame.strip_offsets.len();
    for step in [2u32, 4, 8] {
        let taken = frame
            .crop_rows_step(0..H, step)
            .unwrap()
            .frame
            .strip_offsets
            .len();
        assert!(
            taken <= all.div_ceil(step as usize) + 1,
            "step {step}: took {taken} of {all} strips, expected about {}",
            all.div_ceil(step as usize)
        );
        assert!(taken >= 1);
    }
}

/// A step of one has to be exactly the contiguous crop, since that is the path
/// every non-sampled read takes.
#[test]
fn a_step_of_one_is_the_ordinary_crop() {
    let stack = TiffStack::from_bytes(build(4, Compression::Lzw, true, false, 3)).unwrap();
    let frame = &stack.frames[0];
    let plain = frame.crop_rows(8..24).unwrap();
    let stepped = frame.crop_rows_step(8..24, 1).unwrap();
    assert_eq!(stepped.first_row, plain.rows.start);
    assert_eq!(stepped.frame.height, plain.frame.height);
    assert_eq!(stepped.frame.strip_offsets, plain.frame.strip_offsets);
}

/// Stepping over a *tiled* frame would skip columns as well as rows, because a
/// tile row is several consecutive entries. Refused rather than silently
/// scrambling the image.
#[test]
fn stepping_a_tiled_frame_is_refused() {
    let mut frame = TiffStack::from_bytes(build(4, Compression::None, false, false, 1))
        .unwrap()
        .frames[0]
        .clone();
    frame.tile_size = Some((16, 16));
    let err = frame.crop_rows_step(0..H, 4).unwrap_err().to_string();
    assert!(err.contains("stripped frames"), "unhelpful error: {err}");
}

/// Degenerate steps must not produce an empty or out-of-range band.
#[test]
fn a_step_past_the_frame_still_yields_one_piece() {
    let stack = TiffStack::from_bytes(build(4, Compression::None, false, false, 1)).unwrap();
    let frame = &stack.frames[0];
    for step in [1u32, 1_000, u32::MAX] {
        let band = frame.crop_rows_step(0..H, step).unwrap();
        assert!(band.pieces >= 1, "step {step} took nothing");
        // The final strip of a frame is usually short, so the height is at most
        // the round number and the decode has to agree with whichever it is.
        assert!(band.frame.height <= band.pieces * band.rows_per_piece);
        assert!(band.frame.height > (band.pieces - 1) * band.rows_per_piece);
        let cut = fast_tiff_lib::read_planes_u8(&stack.data, &band.frame, stack.byte_order)
            .unwrap_or_else(|e| panic!("step {step} did not decode: {e:#}"));
        assert_eq!(cut[0].len(), W as usize * band.frame.height as usize);
    }
}

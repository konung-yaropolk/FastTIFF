//! Tiled TIFFs: a grid more than one tile deep.
//!
//! A tile is bounded on both axes where a strip spans the full image width, so
//! a window of a huge image can be read without touching the rest of its rows.
//! That is the whole reason to support them — see [`fast_tiff_lib::FrameInfo::crop`].
//!
//! These fixtures are 100 x 70 in 16 x 16 tiles: a 7 x 5 grid where the right
//! column and bottom row are both partial. That combination is what catches the
//! two mistakes a tiled reader makes:
//!
//! - **assuming edge tiles are trimmed.** TIFF6 stores every tile full size and
//!   pads the ones hanging off the edge, so a tile always decompresses to
//!   `tile_w * tile_h` samples and only part of it belongs in the image. Trim
//!   instead of pad and every row after the first tile column is shifted.
//! - **undoing the predictor on the assembled frame.** It is applied inside each
//!   tile, across the tile's own width including that padding, so undoing it
//!   afterwards differences across tile seams and the image comes out streaked.
//!
//! The main fixture matrix cannot express any of this: its frames are 11 rows,
//! shorter than a single tile. Hence a file of their own, and the `tld_` prefix
//! that tells `libtiff_fixtures.rs` to leave them to this.
//!
//! Requires the `mmap` feature, like the rest of the fixture tests.
#![cfg(feature = "mmap")]

use fast_tiff_lib::TiffStack;
use std::path::{Path, PathBuf};

const W: usize = 100;
const H: usize = 70;

fn fixture(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    p.exists().then_some(p)
}

/// The generator's formulas, over the flat sample index.
fn expect_u8(g: usize) -> u8 {
    ((g * 7 + 13) % 256) as u8
}
fn expect_u16(g: usize) -> u16 {
    ((g * 131 + 17) % 65536) as u16
}

/// Check every plane of a tiled fixture against the formula.
fn check(name: &str, spp: usize, sixteen_bit: bool) {
    let Some(path) = fixture(name) else { return };
    let stack = TiffStack::open(&path).unwrap_or_else(|e| panic!("{name}: failed to open: {e:#}"));
    let frame = &stack.frames[0];

    assert!(frame.is_tiled(), "{name}: should be recognised as tiled");
    assert_eq!(frame.tile_size, Some((16, 16)), "{name}: tile size");
    let (across, down, tw, th) = frame.tile_grid().expect("tiled");
    assert_eq!((across, down, tw, th), (7, 5, 16, 16), "{name}: 100x70 in 16s is a 7x5 grid");
    assert_eq!(frame.width as usize, W);
    assert_eq!(frame.height as usize, H);

    // Where plane `p`'s i-th sample sits in the flat sequence depends on the
    // interleaving, exactly as it does for strips.
    let planar = frame.is_planar();
    let g_of = |p: usize, i: usize| if planar { p * W * H + i } else { i * spp + p };

    if sixteen_bit {
        let planes = fast_tiff_lib::read_planes_u16(&stack.data, frame, stack.byte_order, None).unwrap();
        assert_eq!(planes.len(), spp, "{name}: plane count");
        for (p, plane) in planes.iter().enumerate() {
            assert_eq!(plane.len(), W * H, "{name}: plane {p} size");
            let want: Vec<u16> = (0..W * H).map(|i| expect_u16(g_of(p, i))).collect();
            assert_eq!(plane, &want, "{name}: plane {p}");
        }
    } else {
        let planes = fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap();
        assert_eq!(planes.len(), spp, "{name}: plane count");
        for (p, plane) in planes.iter().enumerate() {
            assert_eq!(plane.len(), W * H, "{name}: plane {p} size");
            let want: Vec<u8> = (0..W * H).map(|i| expect_u8(g_of(p, i))).collect();
            assert_eq!(plane, &want, "{name}: plane {p}");
        }
    }
}

#[test]
fn an_uncompressed_tile_grid_decodes() {
    check("tld_u16_spp1_p1_grid.tif", 1, true);
}

/// Each tile is its own compressed unit, so the codec has to be driven per tile
/// rather than per image.
#[test]
fn a_compressed_tile_grid_decodes() {
    check("tld_u16_spp1_p1_grid-lzw.tif", 1, true);
}

/// The case that catches undoing the predictor in the wrong place: it runs
/// inside a tile, so doing it on the assembled frame differences across the
/// seams between tile columns.
#[test]
fn a_predictor_differenced_tile_grid_decodes() {
    check("tld_u8_spp3_p1_grid-pred2.tif", 3, false);
}

/// Planar tiles repeat the whole grid once per sample plane, so plane `p`'s
/// tiles start at `p * across * down`. Reading one contiguous run instead would
/// decode plane 0 three times.
#[test]
fn a_planar_tile_grid_decodes() {
    check("tld_u8_spp3_p1_grid-planar.tif", 3, false);
}

/// Cropping to a rectangle is what tiles are *for*: unlike a strip band, the
/// columns narrow too, so reading a window costs the window rather than its
/// full-width rows.
#[test]
fn a_tiled_frame_crops_on_both_axes() {
    let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else { return };
    let stack = TiffStack::open(&path).unwrap();
    let frame = &stack.frames[0];
    let full = fast_tiff_lib::read_planes_u16(&stack.data, frame, stack.byte_order, None).unwrap();

    // A window in the middle, deliberately not on tile boundaries.
    let region = frame.crop(20..50, 20..50).unwrap();
    assert!(region.cols.start <= 20 && region.cols.end >= 50, "columns must cover the request");
    assert!(region.rows.start <= 20 && region.rows.end >= 50, "rows must cover the request");
    assert_eq!(region.cols.start % 16, 0, "snapped to the tile grid");
    assert_eq!(region.rows.start % 16, 0);
    assert!(
        region.frame.width < frame.width,
        "a tiled crop should narrow the columns too, got {} of {}",
        region.frame.width,
        frame.width
    );

    let cut = fast_tiff_lib::read_planes_u16(&stack.data, &region.frame, stack.byte_order, None).unwrap();
    let cw = region.frame.width as usize;
    for y in 0..region.frame.height as usize {
        let src = (region.rows.start as usize + y) * W + region.cols.start as usize;
        assert_eq!(
            &cut[0][y * cw..y * cw + cw],
            &full[0][src..src + cw],
            "row {y} of the crop does not match the same rows and columns of the whole frame"
        );
    }
}

/// A crop of the whole image is the whole image, and must not lose the tiling.
#[test]
fn cropping_to_everything_keeps_the_frame_intact() {
    let Some(path) = fixture("tld_u16_spp1_p1_grid-lzw.tif") else { return };
    let stack = TiffStack::open(&path).unwrap();
    let frame = &stack.frames[0];
    let region = frame.crop(0..frame.width, 0..frame.height).unwrap();
    assert_eq!(region.cols, 0..frame.width);
    assert_eq!(region.rows, 0..frame.height);
    assert!(region.frame.is_tiled());
    assert_eq!(region.frame.strip_offsets.len(), frame.strip_offsets.len(), "all tiles kept");

    let a = fast_tiff_lib::read_planes_u16(&stack.data, frame, stack.byte_order, None).unwrap();
    let b = fast_tiff_lib::read_planes_u16(&stack.data, &region.frame, stack.byte_order, None).unwrap();
    assert_eq!(a, b);
}

/// A tile table too short to describe the grid is refused rather than indexed
/// into. The geometry and the table are separate tags of an untrusted file, so
/// they can disagree.
#[test]
fn a_tile_table_that_does_not_describe_the_grid_is_refused() {
    let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else { return };
    let stack = TiffStack::open(&path).unwrap();
    let mut frame = stack.frames[0].clone();
    frame.strip_offsets.truncate(frame.strip_offsets.len() - 1);
    let err = fast_tiff_lib::read_planes_u16(&stack.data, &frame, stack.byte_order, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("tile table"), "unhelpful error: {err}");
}

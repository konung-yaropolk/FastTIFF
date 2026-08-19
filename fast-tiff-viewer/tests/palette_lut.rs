//! Regression tests for palette (indexed) stacks and the LUT selector.
//!
//! A palette image's ColorMap *is* its display transfer: the pixels are indices
//! and the map turns them into greys or colours. Files written by ImageJ after
//! a contrast stretch are routinely **grayscale but non-identity** — index 16
//! maps to grey 46, and everything above the stretched range is black.
//!
//! Such a file must offer a way back to its own map. Losing it is unrecoverable
//! for the user, because a palette channel's contrast slider is deliberately
//! suppressed (its window is a fixed index -> entry identity, so there is
//! nothing meaningful to drag).

use fast_tiff_viewer::channels::{gray_lut_count, gray_lut_sel_lut, gray_lut_sel_name};
use fast_tiff_viewer::{Stack, Viewer};

/// A minimal uncompressed 8-bit palette TIFF whose ColorMap is a grayscale ramp
/// stretched over indices `0..=top` (everything above is black) — i.e. what a
/// contrast-stretched indexed export looks like. The writer can't emit
/// photometric=3, so this is built by hand.
fn palette_tiff(w: u32, h: u32, top: u32) -> Vec<u8> {
    let mut ifd: Vec<u8> = Vec::new();
    let n_entries: u16 = 10;
    // header(8) + count(2) + entries + next(4)
    let cmap_off = 8 + 2 + n_entries as u32 * 12 + 4;
    let px_off = cmap_off + 768 * 2;

    let mut entry = |tag: u16, ty: u16, count: u32, val: u32| {
        ifd.extend_from_slice(&tag.to_le_bytes());
        ifd.extend_from_slice(&ty.to_le_bytes());
        ifd.extend_from_slice(&count.to_le_bytes());
        // A SHORT with count 1 sits in the low half of the value field.
        ifd.extend_from_slice(&val.to_le_bytes());
    };
    entry(256, 3, 1, w); // ImageWidth
    entry(257, 3, 1, h); // ImageLength
    entry(258, 3, 1, 8); // BitsPerSample
    entry(259, 3, 1, 1); // Compression = none
    entry(262, 3, 1, 3); // Photometric = palette
    entry(273, 4, 1, px_off); // StripOffsets
    entry(277, 3, 1, 1); // SamplesPerPixel
    entry(278, 3, 1, h); // RowsPerStrip
    entry(279, 4, 1, w * h); // StripByteCounts
    entry(320, 3, 768, cmap_off); // ColorMap

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&n_entries.to_le_bytes());
    out.extend_from_slice(&ifd);
    out.extend_from_slice(&0u32.to_le_bytes()); // no next IFD

    // ColorMap: all reds, then greens, then blues — 16-bit, so <<8.
    let grey = |i: u32| -> u16 {
        let v = if i <= top { (i * 255 / top).min(255) } else { 0 };
        (v as u16) << 8
    };
    for _ in 0..3 {
        for i in 0..256u32 {
            out.extend_from_slice(&grey(i).to_le_bytes());
        }
    }
    assert_eq!(out.len() as u32, px_off, "pixel data must start where the IFD says");
    out.extend((0..w * h).map(|i| (i % (top + 1)) as u8));
    out
}

fn open(bytes: Vec<u8>) -> Stack {
    Stack::from_bytes(bytes, "palette.tif".into(), false).expect("palette stack should open")
}

#[test]
fn the_fixture_really_is_a_stretched_grayscale_palette() {
    // Guards the premise: if this stopped being a non-identity grey ramp, the
    // tests below would pass without proving anything.
    let stack = open(palette_tiff(16, 8, 87));
    assert!(stack.display.palette, "photometric=3 should be recognised as palette");
    let lut = stack.tiff.meta.channel_display[0].lut;
    assert!(lut.iter().all(|p| p[0] == p[1] && p[1] == p[2]), "ramp is grayscale");
    assert_ne!(lut, fast_tiff_lib::grayscale_lut(), "but it is NOT the identity ramp");
    assert_eq!(lut[16][0], 46, "index 16 stretches to grey 46");
    assert_eq!(lut[128][0], 0, "past the stretched range it goes black");
}

#[test]
fn a_palette_file_offers_its_own_map_in_the_selector() {
    let stack = open(palette_tiff(16, 8, 87));
    assert!(
        stack.display.builtin_lut.is_some(),
        "the ColorMap is this file's display transfer — it must be offered, \
         grayscale ramp or not, since the contrast slider is suppressed here"
    );
    assert_eq!(gray_lut_sel_name(&stack.display, 0), "Built-in");
    assert_eq!(
        gray_lut_sel_lut(&stack.display, 0),
        stack.tiff.meta.channel_display[0].lut,
        "option 0 restores the file's own map exactly"
    );
}

#[test]
fn switching_colormap_and_back_restores_the_file_ramp() {
    // The reported bug: pick a colormap, go back, and the image is left dark
    // with no way to recover.
    let mut viewer = Viewer::new();
    viewer.load_bytes(palette_tiff(16, 8, 87), "palette.tif".into()).unwrap();
    let original = viewer.stack.as_ref().unwrap().display.luts[0];
    assert_ne!(original, fast_tiff_lib::grayscale_lut(), "loads through the file's ramp");

    let n = gray_lut_count(&viewer.stack.as_ref().unwrap().display);
    // Walk every option, then come back to "Built-in" each time.
    for sel in 1..n {
        viewer.set_gray_lut(sel);
        assert_ne!(viewer.stack.as_ref().unwrap().display.luts[0], original, "option {sel} should change the display");
        viewer.set_gray_lut(0);
        assert_eq!(
            viewer.stack.as_ref().unwrap().display.luts[0],
            original,
            "returning from option {sel} must restore the file's ramp"
        );
    }
}

#[test]
fn a_plain_grayscale_file_gains_no_spurious_builtin_option() {
    // The fix must not add a duplicate "Built-in" entry to ordinary files that
    // carry no map of their own.
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    use std::io::Cursor;
    let (w, h) = (8u32, 4u32);
    let opts = WriterOptions::new(w, h, SampleType::U8);
    let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    wr.write_frame_bytes(&vec![7u8; (w * h) as usize]).unwrap();
    let stack = Stack::from_bytes(wr.finish().unwrap().into_inner(), "plain.tif".into(), false).unwrap();

    assert!(!stack.display.palette);
    assert!(stack.display.builtin_lut.is_none(), "no ColorMap, so no Built-in option");
    assert_eq!(gray_lut_sel_name(&stack.display, 0), "Grayscale");
}

/// The isosurface paints one fixed colour for the whole surface, so raising the
/// threshold doesn't also darken it. Taking that colour from the LUT's *top*
/// entry works for an ordinary ramp but renders a completely black — i.e.
/// invisible — surface for a contrast-stretched palette, whose top entry is
/// black because the map peaks partway along and blacks out the unused tail.
#[test]
fn isosurface_albedo_lands_on_a_visible_part_of_the_lut() {
    let stack = open(palette_tiff(16, 8, 87));
    let lut = stack.display.luts[0];
    assert_eq!(lut[255], [0, 0, 0], "this palette really does end black");

    let t = scivis_render::brightest_lut_t(&lut);
    let sampled = lut[(t * 255.0).round() as usize];
    assert!(
        sampled.iter().any(|&c| c > 0),
        "albedo sampled at t={t} gives {sampled:?} — a black surface is invisible"
    );
    assert_eq!(sampled, [255, 255, 255], "it should land on the ramp's peak");
}

#[test]
fn an_ordinary_ramp_still_uses_its_top_entry() {
    // The fix must not move the albedo for the files that already worked.
    for lut in [fast_tiff_lib::grayscale_lut(), fast_tiff_lib::default_composite_lut(0)] {
        assert_eq!(scivis_render::brightest_lut_t(&lut), 1.0, "a rising ramp peaks at the top");
    }
}

#[test]
fn volume_params_carry_the_albedo_per_channel() {
    // End-to-end: the value actually reaches the uniform the shader reads.
    use fast_tiff_viewer::camera::{build_volume_params, volume_camera, VolumeChannel};
    let stack = open(palette_tiff(16, 8, 87));
    let albedo = scivis_render::brightest_lut_t(&stack.display.luts[0]);
    assert!(albedo < 1.0, "the stretched palette peaks before its top entry");

    let cam = volume_camera(&Default::default(), [1.0, 1.0, 1.0], (16, 8, 4));
    let ch = [VolumeChannel { min: 0.0, max: 1.0, is_float: false, enabled: true, albedo_t: albedo }];
    let params = build_volume_params(&cam, &ch, 1.0, scivis_render::VolumeRender::Surface, 100.0, 0.1);
    assert_eq!(params.albedo_t[0], albedo);
    // Absent channels keep the top entry, which is what an unused slot wants.
    assert_eq!(params.albedo_t[1], 1.0);
}

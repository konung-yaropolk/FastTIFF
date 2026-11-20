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

use fast_tiff_viewer::{ChannelLayout, Document, Viewer};

/// A minimal uncompressed 8-bit palette TIFF whose ColorMap is a grayscale ramp
/// stretched over indices `0..=top` (everything above is black) — i.e. what a
/// contrast-stretched indexed export looks like. The writer can't emit
/// photometric=3, so this is built by hand.
fn palette_tiff(w: u32, h: u32, top: u32) -> Vec<u8> {
    palette_stack(w, h, top, 1, None)
}

/// [`palette_tiff`]'s image `frames` times: one IFD each, all of them
/// pointing at the same ColorMap and pixels, and at the same
/// ImageDescription when there is one (an ImageJ header, say).
fn palette_stack(w: u32, h: u32, top: u32, frames: u32, description: Option<&str>) -> Vec<u8> {
    // NUL-terminated, and longer than the 4 bytes that would sit in the entry.
    let description = description.map(|d| format!("{d}\0"));
    assert!(description.as_ref().is_none_or(|d| d.len() > 4));
    let n_entries: u16 = 10 + u16::from(description.is_some());
    // count(2) + entries + next(4)
    let ifd_len = 2 + n_entries as u32 * 12 + 4;
    // After the header(8) and the IFDs, the ColorMap, the description, then
    // the pixels.
    let cmap_off = 8 + frames * ifd_len;
    let desc_off = cmap_off + 768 * 2;
    let px_off = desc_off + description.as_ref().map_or(0, |d| d.len() as u32);

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    for f in 0..frames {
        let mut ifd: Vec<u8> = Vec::new();
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
        if let Some(d) = &description {
            entry(270, 2, d.len() as u32, desc_off); // ImageDescription
        }
        entry(273, 4, 1, px_off); // StripOffsets
        entry(277, 3, 1, 1); // SamplesPerPixel
        entry(278, 3, 1, h); // RowsPerStrip
        entry(279, 4, 1, w * h); // StripByteCounts
        entry(320, 3, 768, cmap_off); // ColorMap
        out.extend_from_slice(&n_entries.to_le_bytes());
        out.extend_from_slice(&ifd);
        // The next IFD, or none after the last.
        let next = if f + 1 < frames {
            8 + (f + 1) * ifd_len
        } else {
            0
        };
        out.extend_from_slice(&next.to_le_bytes());
    }

    // ColorMap: all reds, then greens, then blues — 16-bit, so <<8.
    let grey = |i: u32| -> u16 {
        let v = if i <= top {
            (i * 255 / top).min(255)
        } else {
            0
        };
        (v as u16) << 8
    };
    for _ in 0..3 {
        for i in 0..256u32 {
            out.extend_from_slice(&grey(i).to_le_bytes());
        }
    }
    if let Some(d) = &description {
        out.extend_from_slice(d.as_bytes());
    }
    assert_eq!(
        out.len() as u32,
        px_off,
        "pixel data must start where the IFD says"
    );
    out.extend((0..w * h).map(|i| (i % (top + 1)) as u8));
    out
}

fn open(bytes: Vec<u8>) -> Document {
    Document::from_bytes(bytes, "palette.tif".into(), false).expect("palette stack should open")
}

#[test]
fn the_fixture_really_is_a_stretched_grayscale_palette() {
    // Guards the premise: if this stopped being a non-identity grey ramp, the
    // tests below would pass without proving anything.
    let stack = open(palette_tiff(16, 8, 87));
    assert_eq!(
        stack.display().layout(),
        ChannelLayout::Palette,
        "photometric=3 should be recognised as palette"
    );
    let lut = stack.tiff().meta.channel_display[0].lut;
    assert!(
        lut.iter().all(|p| p[0] == p[1] && p[1] == p[2]),
        "ramp is grayscale"
    );
    assert_ne!(
        lut,
        fast_tiff_lib::grayscale_lut(),
        "but it is NOT the identity ramp"
    );
    assert_eq!(lut[16][0], 46, "index 16 stretches to grey 46");
    assert_eq!(lut[128][0], 0, "past the stretched range it goes black");
}

#[test]
fn a_palette_file_offers_its_own_map_in_the_selector() {
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(palette_tiff(16, 8, 87), "palette.tif".into())
        .unwrap();
    let luts = viewer.document().unwrap().display().gray_luts().unwrap();
    assert_eq!(
        luts.name(0),
        "Built-in",
        "the ColorMap is this file's display transfer — it must be offered, \
         grayscale ramp or not, since the contrast slider is suppressed here"
    );
    assert_eq!(luts.selected(), 0, "and shown to begin with");
    viewer.set_gray_lut(1);
    viewer.set_gray_lut(0);
    let stack = viewer.document().unwrap();
    assert_eq!(
        stack.display().luts()[0],
        stack.tiff().meta.channel_display[0].lut,
        "option 0 restores the file's own map exactly"
    );
}

/// A palette stack has no contrast window to fit. Its window is a pinned
/// index-to-LUT identity, and nudging it would remap every colour in the
/// image — so neither a drag, nor Auto, nor Reset may move it. (That Reset
/// leaves even a moved one alone is tested beside the display, which can
/// move it: through the API nothing can.)
#[test]
fn palette_windows_are_left_alone_by_a_drag_auto_and_reset() {
    let mut stack = open(palette_tiff(16, 8, 87));
    assert!(!stack.display().layout().has_contrast());
    let before = stack.display().window(0);
    let hists = fast_tiff_viewer::histogram::frame_histograms(&stack);

    assert!(
        !stack
            .display_mut()
            .set_window(0, fast_tiff_viewer::Window::new(-0.5, 254.5)),
        "a palette window is refused"
    );
    assert_eq!(stack.display().window(0), before);
    stack.display_mut().auto_contrast(&hists);
    stack.display_mut().reset_contrast();

    assert_eq!(
        stack.display().window(0),
        before,
        "a palette window is an identity mapping, not a contrast choice"
    );
}

/// A palette stack's channels are IFDs of their own, as an intensity stack's
/// are, so its axes can be reassigned (unlike RGB's, which are sample planes):
/// as two channels each keeps the pinned identity window and wears the file's
/// ColorMap, pseudocolor or not, since each one's pixels index that map; and
/// back on one the ColorMap is its LUT again, offered as "Built-in".
#[test]
fn a_palette_stack_can_be_reassigned() {
    for pseudocolor in [false, true] {
        let mut viewer = Viewer::new();
        viewer.set_pseudocolor(pseudocolor);
        viewer
            .load_bytes(palette_stack(16, 8, 87, 4, None), "palette.tif".into())
            .unwrap();
        let stack = viewer.document().unwrap();
        let map = stack.tiff().meta.channel_display[0].lut;
        assert_ne!(map, fast_tiff_lib::grayscale_lut(), "the stretched ramp");
        let d = stack.display();
        assert_eq!((d.dims().channels, d.dims().frames), (1, 4));
        assert_eq!(d.luts(), [map]);
        let identity = d.window(0).expect("one channel");

        viewer.set_dimension_order(2, 1, 2);
        let stack = viewer.document().unwrap();
        let d = stack.display();
        assert_eq!(d.layout(), ChannelLayout::Palette);
        assert_eq!((d.dims().channels, d.dims().frames), (2, 2));
        assert_eq!((d.window(0), d.window(1)), (Some(identity), Some(identity)));
        assert_eq!(d.luts(), [map, map], "pseudocolor {pseudocolor}");
        assert!(!stack.pseudocolor_applicable());
        viewer.set_pseudocolor(!pseudocolor);
        assert_eq!(viewer.document().unwrap().display().luts(), [map, map]);
        viewer.set_pseudocolor(pseudocolor);

        viewer.set_dimension_order(1, 1, 4);
        let stack = viewer.document().unwrap();
        assert_eq!(stack.display().layout(), ChannelLayout::Palette);
        assert_eq!(stack.display().luts(), [map]);
        let luts = stack.display().gray_luts().expect("one channel again");
        assert_eq!((luts.selected(), luts.name(0)), (0, "Built-in"));
    }
}

/// Opened as several channels, because its ImageJ header says so, a palette
/// stack's channels all wear its ColorMap, and pseudocolor does not tint
/// them: the open seeds them as a reassignment does.
#[test]
fn a_palette_stack_opened_as_channels_wears_its_colormap_on_each() {
    let header = "ImageJ=1.54p\nimages=4\nchannels=2\nframes=2\nhyperstack=true\n";
    for pseudocolor in [false, true] {
        let mut viewer = Viewer::new();
        viewer.set_pseudocolor(pseudocolor);
        viewer
            .load_bytes(
                palette_stack(16, 8, 87, 4, Some(header)),
                "palette.tif".into(),
            )
            .unwrap();
        let stack = viewer.document().unwrap();
        let map = stack.tiff().meta.channel_display[0].lut;
        assert_ne!(map, fast_tiff_lib::grayscale_lut(), "the stretched ramp");
        let d = stack.display();
        assert_eq!(d.layout(), ChannelLayout::Palette);
        assert_eq!((d.dims().channels, d.dims().frames), (2, 2));
        assert_eq!(d.luts(), [map, map], "pseudocolor {pseudocolor}");
        assert!(!stack.pseudocolor_applicable());
    }
}

#[test]
fn imagej_composite_uses_channel_luts_and_adjustable_windows() {
    let header = "ImageJ=1.54p\nimages=4\nchannels=2\nframes=2\nmode=composite\nmin=2\nmax=60\n";
    let mut stack = open(palette_stack(16, 8, 87, 4, Some(header)));
    assert!(stack.tiff().frames[0].is_palette());
    assert_eq!(
        stack.display().mode(),
        fast_tiff_lib::DisplayMode::Composite
    );
    assert!(stack.display().layout().has_contrast());
    for c in 0..2 {
        assert_eq!(
            stack.display().luts()[c],
            fast_tiff_lib::default_composite_lut(c)
        );
        assert_eq!(
            stack.display().window(c),
            Some(fast_tiff_viewer::Window::new(2.0, 60.0))
        );
        assert!(stack
            .display_mut()
            .set_window(c, fast_tiff_viewer::Window::new(4.0, 55.0)));
    }
}

#[test]
fn switching_colormap_and_back_restores_the_file_ramp() {
    // The reported bug: pick a colormap, go back, and the image is left dark
    // with no way to recover.
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(palette_tiff(16, 8, 87), "palette.tif".into())
        .unwrap();
    let original = viewer.document().unwrap().display().luts()[0];
    assert_ne!(
        original,
        fast_tiff_lib::grayscale_lut(),
        "loads through the file's ramp"
    );

    let n = viewer
        .document()
        .unwrap()
        .display()
        .gray_luts()
        .unwrap()
        .count();
    // Walk every option, then come back to "Built-in" each time.
    for sel in 1..n {
        viewer.set_gray_lut(sel);
        assert_ne!(
            viewer.document().unwrap().display().luts()[0],
            original,
            "option {sel} should change the display"
        );
        viewer.set_gray_lut(0);
        assert_eq!(
            viewer.document().unwrap().display().luts()[0],
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
    let stack =
        Document::from_bytes(wr.finish().unwrap().into_inner(), "plain.tif".into(), false).unwrap();

    assert_eq!(stack.display().layout(), ChannelLayout::Intensity);
    assert_eq!(
        stack.display().gray_luts().unwrap().name(0),
        "Grayscale",
        "no ColorMap, so no Built-in option"
    );
}

/// The isosurface paints one fixed colour for the whole surface, so raising the
/// threshold doesn't also darken it. Taking that colour from the LUT's *top*
/// entry works for an ordinary ramp but renders a completely black — i.e.
/// invisible — surface for a contrast-stretched palette, whose top entry is
/// black because the map peaks partway along and blacks out the unused tail.
#[test]
fn isosurface_albedo_lands_on_a_visible_part_of_the_lut() {
    let stack = open(palette_tiff(16, 8, 87));
    let lut = stack.display().luts()[0];
    assert_eq!(lut[255], [0, 0, 0], "this palette really does end black");

    let t = scivis_render::brightest_lut_t(&lut);
    let sampled = lut[(t * 255.0).round() as usize];
    assert!(
        sampled.iter().any(|&c| c > 0),
        "albedo sampled at t={t} gives {sampled:?} — a black surface is invisible"
    );
    assert_eq!(
        sampled,
        [255, 255, 255],
        "it should land on the ramp's peak"
    );
}

#[test]
fn an_ordinary_ramp_still_uses_its_top_entry() {
    // The fix must not move the albedo for the files that already worked.
    for lut in [
        fast_tiff_lib::grayscale_lut(),
        fast_tiff_lib::default_composite_lut(0),
    ] {
        assert_eq!(
            scivis_render::brightest_lut_t(&lut),
            1.0,
            "a rising ramp peaks at the top"
        );
    }
}

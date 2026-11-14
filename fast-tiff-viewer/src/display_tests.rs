//! The display model's own invariants: what each change moves, and a LUT
//! selector that cannot be read while it does not apply nor outlive the
//! LUTs it chose.

use super::*;
use crate::units::SampleUnits;

/// A display of `n` switched-on 8-bit channels, each with a LUT.
fn display(n: usize) -> Display {
    let mut d = Display::empty();
    d.set_settings(
        (0..n)
            .map(|_| ChannelSettings::new(Window::new(0.0, 255.0), SampleUnits::RAW8))
            .collect(),
    );
    d.set_luts((0..n).map(fast_tiff_lib::default_composite_lut).collect());
    d
}

#[test]
fn a_new_display_reads_ifds_of_intensities() {
    assert_eq!(Display::empty().layout(), ChannelLayout::Intensity);
}

/// Installing a layout moves the layout revision, since under RGB and CMYK a
/// channel reads another plane; which channels are on is untouched.
#[test]
fn installing_a_layout_moves_the_layout_revision_only() {
    let mut d = display(3);
    for layout in [
        ChannelLayout::Rgb,
        ChannelLayout::Cmyk,
        ChannelLayout::Palette,
    ] {
        let before = d.revision();
        d.set_layout(layout);
        assert_eq!(d.layout(), layout);
        assert_ne!(d.revision().layout, before.layout, "{layout:?}");
        assert_eq!(d.revision().enabled, before.enabled, "{layout:?}");
    }
}

/// Only RGB and CMYK read sample planes, and only a palette has no window.
#[test]
fn what_each_layout_means() {
    use ChannelLayout::*;
    let table = [
        (Intensity, false, true),
        (Rgb, true, true),
        (Cmyk, true, true),
        (Palette, false, false),
    ];
    for (layout, rgb, contrast) in table {
        assert_eq!(layout.shown_as_rgb(), rgb, "{layout:?}");
        assert_eq!(layout.has_contrast(), contrast, "{layout:?}");
    }
}

/// A palette window is moved by none of the contrast operations: not a
/// window set, not Auto, and not Reset, even of one that is somewhere other
/// than where it opened — which only this module can arrange.
#[test]
fn no_contrast_operation_moves_a_palette_window() {
    let mut d = display(1);
    d.set_layout(ChannelLayout::Palette);
    let moved = Window::new(10.0, 20.0);
    d.settings[0].window = moved;

    assert!(!d.set_window(0, Window::new(0.0, 255.0)));
    d.reset_contrast();
    assert_eq!(d.window(0), Some(moved));

    d.set_layout(ChannelLayout::Intensity);
    assert!(
        d.set_window(0, Window::new(0.0, 100.0)),
        "but any other can"
    );
}

/// The selector is there for one channel that is not RGB — intensities or a
/// palette — and for nothing else, and choosing where it is not there
/// changes nothing.
#[test]
fn the_selector_applies_to_one_channel_that_is_not_rgb() {
    assert!(Display::empty().gray_luts().is_none(), "no channels");
    assert!(display(2).gray_luts().is_none(), "two channels");
    for layout in [ChannelLayout::Intensity, ChannelLayout::Palette] {
        let mut d = display(1);
        d.set_layout(layout);
        assert!(d.gray_luts().is_some(), "{layout:?}");
    }
    for layout in [ChannelLayout::Rgb, ChannelLayout::Cmyk] {
        let mut d = display(1);
        d.set_layout(layout);
        assert!(d.gray_luts().is_none(), "{layout:?}");
        let luts = d.luts().to_vec();
        assert!(!d.select_gray_lut(1));
        assert_eq!(d.luts(), &luts[..], "{layout:?}");
    }
}

/// A choice installs its LUT and moves the LUT revision, not the display's;
/// one past the last option is refused.
#[test]
fn a_choice_installs_its_lut() {
    let mut d = display(1);
    let (before, lut_before) = (d.revision(), d.lut_revision());
    assert!(d.select_gray_lut(2));
    assert_eq!(d.gray_luts().unwrap().selected(), 2);
    assert_eq!(d.lut(0), Some(&crate::channels::gray_lut_for(2)));
    assert_eq!(d.revision(), before);
    assert_ne!(d.lut_revision(), lut_before);

    let count = d.gray_luts().unwrap().count();
    assert!(!d.select_gray_lut(count));
    assert_eq!(d.gray_luts().unwrap().selected(), 2);
}

/// The file's own LUT leads the options as "Built-in", and is what option 0
/// puts back.
#[test]
fn a_builtin_lut_leads_the_options() {
    let mut d = display(1);
    let own = fast_tiff_lib::default_composite_lut(4);
    d.builtin_lut = Some(own);
    let luts = d.gray_luts().unwrap();
    assert_eq!(luts.count(), crate::channels::gray_lut_option_count() + 1);
    assert_eq!((luts.name(0), luts.name(1)), ("Built-in", "Grayscale"));
    assert_eq!(luts.tint(0), crate::channels::ui_tint(&own));
    assert!(d.select_gray_lut(3));
    assert!(d.select_gray_lut(0));
    assert_eq!(d.lut(0), Some(&own));
}

/// Reseeding the LUTs for another channel count discards the user's choice,
/// and the selector's with it; reseeding for the same count keeps both.
#[test]
fn reseeding_for_another_count_resets_the_choice() {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(4, 4, SampleType::U8);
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_bytes(&[7; 16]).unwrap();
    let bytes = w.finish().unwrap().into_inner();
    let mut doc = crate::Document::from_bytes(bytes, "one.tif".into(), false).unwrap();

    assert!(doc.display.select_gray_lut(3));
    doc.display.reseed_luts(&doc.tiff.meta, 1);
    assert_eq!(doc.display.gray_lut_sel, 3, "the same count keeps the LUTs");

    doc.display.reseed_luts(&doc.tiff.meta, 2);
    assert_eq!(
        doc.display.gray_lut_sel, 0,
        "another count regenerates them"
    );
}

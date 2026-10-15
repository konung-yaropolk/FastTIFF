//! The two pieces of copy-the-view with arithmetic or precedence to get wrong:
//! cutting the 3-D panel out of a window screenshot, and deciding whether the
//! copy chord belongs to the picture or to a text field.
//!
//! Both are worth pinning because both fail in ways that are hard to see. A
//! crop that is off by a device-pixel-ratio factor copies a quarter of the
//! view and looks like a rendering bug; a chord that is taken too eagerly makes
//! every text box in the program quietly stop copying.

use super::*;

/// A flat image of a known size, with each pixel encoding its own column so a
/// crop can be checked for *position* rather than only for size.
fn image(w: usize, h: usize) -> ColorImage {
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            rgba.extend_from_slice(&[x as u8, y as u8, 0, 255]);
        }
    }
    ColorImage::from_rgba_unmultiplied([w, h], &rgba)
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> egui::Rect {
    egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1))
}

/// The ordinary case, at 1:1: the crop is where it was asked for and the right
/// size.
#[test]
fn a_crop_takes_the_rect_it_was_given() {
    let img = image(100, 80);
    let cut = crop(&img, rect(10.0, 20.0, 40.0, 60.0), 1.0).expect("a crop inside the image");
    assert_eq!([cut.width(), cut.height()], [30, 40]);
    // Top-left pixel of the crop should be the source pixel at (10, 20).
    assert_eq!(cut.pixels[0].r(), 10, "column should come from x = 10");
    assert_eq!(cut.pixels[0].g(), 20, "row should come from y = 20");
}

/// The rect is in points and the screenshot in physical pixels, so the scale
/// has to be applied. Getting this wrong is the classic HiDPI bug: on a 2x
/// display the copy would be the top-left quarter of the view.
#[test]
fn the_rect_is_scaled_by_the_pixel_ratio() {
    let img = image(200, 160);
    let cut = crop(&img, rect(10.0, 20.0, 40.0, 60.0), 2.0).expect("a crop inside the image");
    assert_eq!(
        [cut.width(), cut.height()],
        [60, 80],
        "a 30x40 point rect at 2x is 60x80 pixels"
    );
    assert_eq!(cut.pixels[0].r(), 20, "x = 10 points is pixel 20");
}

/// A rect larger than the image is clamped, not asserted on. This is the
/// stale-rect case: the panel rect was recorded before a resize, so by the time
/// the screenshot lands it describes a window that no longer exists.
/// `ColorImage::region` panics on that, which would take the program down for
/// what should be a slightly smaller copy.
#[test]
fn a_rect_past_the_edge_is_clamped_rather_than_fatal() {
    let img = image(50, 40);
    let cut = crop(&img, rect(30.0, 20.0, 5000.0, 5000.0), 1.0).expect("the overlap is non-empty");
    assert_eq!(
        [cut.width(), cut.height()],
        [20, 20],
        "clamped to the image's own bounds"
    );
}

/// Likewise a rect that starts past the edge: nothing overlaps, so there is
/// nothing to copy and the caller is told so.
#[test]
fn a_rect_entirely_outside_the_image_is_nothing_to_copy() {
    let img = image(50, 40);
    assert!(crop(&img, rect(60.0, 50.0, 90.0, 90.0), 1.0).is_none());
}

/// An empty or inverted rect is `None` rather than a zero-pixel image. A
/// clipboard entry of no pixels is worse than no clipboard entry: it silently
/// replaces whatever the user had copied before.
#[test]
fn an_empty_or_inverted_rect_copies_nothing() {
    let img = image(50, 40);
    assert!(
        crop(&img, rect(10.0, 10.0, 10.0, 30.0), 1.0).is_none(),
        "zero width"
    );
    assert!(
        crop(&img, rect(10.0, 10.0, 30.0, 10.0), 1.0).is_none(),
        "zero height"
    );
    assert!(
        crop(&img, rect(30.0, 30.0, 10.0, 10.0), 1.0).is_none(),
        "inverted"
    );
    assert!(
        crop(&img, egui::Rect::NOTHING, 1.0).is_none(),
        "the empty rect"
    );
}

/// Non-finite geometry cannot panic or produce a nonsense crop. `Rect::NOTHING`
/// is infinite, and a layout that has not run yet can hand out NaN — neither is
/// exotic enough to be allowed to abort a copy.
#[test]
fn non_finite_geometry_is_survivable() {
    let img = image(50, 40);
    for r in [
        rect(f32::NAN, 0.0, 10.0, 10.0),
        rect(0.0, 0.0, f32::NAN, f32::NAN),
        rect(
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::INFINITY,
        ),
    ] {
        // Either a clamped crop or nothing, but never a panic.
        let _ = crop(&img, r, 1.0);
    }
    // A broken scale falls back to 1:1 rather than scaling by NaN.
    let cut = crop(&img, rect(0.0, 0.0, 10.0, 10.0), f32::NAN).expect("fallback to 1:1");
    assert_eq!([cut.width(), cut.height()], [10, 10]);
    let cut = crop(&img, rect(0.0, 0.0, 10.0, 10.0), 0.0).expect("fallback to 1:1");
    assert_eq!([cut.width(), cut.height()], [10, 10]);
}

/// Run one frame with the copy chord held, and report what `hotkey` decided.
fn chord_taken(focus_a_widget: bool) -> bool {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::C,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        }],
        ..Default::default()
    };
    let mut taken = false;
    let _ = ctx.run_ui(input, |ui| {
        let ctx = ui.ctx();
        if focus_a_widget {
            // Stand in for a text field holding the caret.
            let id = egui::Id::new("a text field");
            ctx.memory_mut(|m| m.request_focus(id));
        }
        taken = hotkey(ctx);
    });
    taken
}

/// The chord copies the view when the picture is what the user is looking at.
#[test]
fn the_copy_chord_is_taken_when_nothing_has_focus() {
    assert!(chord_taken(false), "Ctrl/Cmd+C should copy the view");
}

/// And is left alone when a widget holds the caret, so typing in the frame
/// number box and copying the number still works. This is the precedence that
/// matters: the picture is the fallback, not the priority.
#[test]
fn a_focused_widget_keeps_the_copy_chord() {
    assert!(
        !chord_taken(true),
        "a text field's own copy must win over copying the view"
    );
}

/// A frame with no chord in it copies nothing — the guard is reading the input,
/// not firing on every frame.
#[test]
fn no_chord_means_no_copy() {
    let ctx = egui::Context::default();
    let mut taken = true;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| taken = hotkey(ui.ctx()));
    assert!(!taken);
}

/// The composed snapshot survives the handoff to egui at the right size and
/// fully opaque — the two things that would make a paste look broken.
#[test]
fn a_snapshot_becomes_an_image_of_the_same_size() {
    let snap = fast_tiff_viewer::snapshot::Snapshot {
        width: 3,
        height: 2,
        rgba: vec![255; 3 * 2 * 4],
    };
    let img = image_of(&snap);
    assert_eq!([img.width(), img.height()], [3, 2]);
    assert!(img.pixels.iter().all(|p| p.a() == 255), "opaque");
}

/// Every toolbar icon is a glyph the bundled font actually has.
///
/// This test exists because writing the copy icon by picking a plausible
/// codepoint nearly shipped a tofu box: egui's icon font has the overlapping
/// *pages* glyph but not the shadowed square or the joined squares, and the
/// difference is invisible until the button is on screen. Covering the other
/// three as well costs nothing and pins the same trap for the next icon.
#[test]
fn every_toolbar_icon_has_a_glyph() {
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let id = egui::FontId::proportional(crate::app::ICON_SIZE);
    for (icon, name) in [
        (crate::app::ICON_COPY, "copy"),
        (crate::app::ICON_SAVE, "save"),
        (crate::app::ICON_OPEN, "open"),
        (crate::app::ICON_PLUGINS, "plugins"),
        (crate::app::ICON_SETTINGS, "settings"),
    ] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&id, icon)),
            "the {name} icon ({icon:?}) has no glyph and would draw as a box"
        );
    }
}

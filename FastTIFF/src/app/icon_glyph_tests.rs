//! Every toolbar icon is a character the bundled fonts actually have.
//!
//! A character they do not have is not a compile error and not a runtime error.
//! It is a tofu box — ▯ — sitting in the toolbar, which nothing but a person
//! looking at the window would ever notice, and which looks like a broken build
//! rather than a missing glyph. egui ships Ubuntu-Light, NotoEmoji and
//! emoji-icon-font, and what is in them is not guessable: `📂`, `💾` and `☰` are
//! there; `≡`, `▼` and `🖫`, which are the obvious alternatives to two of them,
//! are not.

use super::{ICON_OPEN, ICON_PLUGINS, ICON_SAVE, ICON_SETTINGS};

#[test]
fn every_toolbar_icon_has_a_glyph() {
    let mut fonts = egui::epaint::text::Fonts::new(
        egui::epaint::text::TextOptions {
            max_texture_side: 8192,
            alpha_from_coverage: Default::default(),
            font_hinting: true,
        },
        egui::FontDefinitions::default(),
    );
    // The size the toolbar draws them at, since coverage is per font family
    // rather than per size — but asking at the real size costs nothing and
    // keeps the test honest if that ever stops being true.
    let id = egui::FontId::proportional(super::ICON_SIZE);
    for (what, icon) in [
        ("open", ICON_OPEN),
        ("save", ICON_SAVE),
        ("plugins", ICON_PLUGINS),
        ("3D settings", ICON_SETTINGS),
    ] {
        assert!(
            fonts.has_glyphs(&id, icon),
            "the {what} icon ({icon:?}) is not in any bundled font, so the toolbar \
             will draw a tofu box where it should be"
        );
    }
}

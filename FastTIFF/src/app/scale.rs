//! Interface scale: how large the chrome is drawn, and how a pop-up opts out.
//!
//! The two hosts want different sizes for the *same* UI. On the desktop the
//! window is sized to the image and the user sits at arm's length, so egui's
//! native sizing is right. In a browser the canvas is one element on a page the
//! user has already zoomed to taste, the controls compete with the page's own
//! furniture, and the same widgets come out noticeably small — so the web build
//! draws everything at [`UI_SCALE`].
//!
//! Pop-up windows are the exception (see [`unscaled`]).

/// How much larger than native the interface is drawn.
///
/// Applied as egui's zoom factor, which multiplies `pixels_per_point` — so it
/// scales fonts, padding, hit targets and line widths together, rather than
/// just enlarging text into a layout built for smaller text.
#[cfg(target_arch = "wasm32")]
pub const UI_SCALE: f32 = 1.25;
#[cfg(not(target_arch = "wasm32"))]
pub const UI_SCALE: f32 = 1.0;

/// Draw a pop-up window at native size even when the chrome around it is
/// scaled.
///
/// The panels benefit from being bigger; the pop-ups do not. They are dense,
/// information-heavy and already sized to be a tight fit — the metadata window
/// is a long two-column grid, the 3D settings a stack of labelled sliders — so
/// enlarging them by half pushes their contents past the bottom of a laptop
/// viewport, which is the one place a browser cannot simply be resized. Keeping
/// them at native size also keeps them honest about how much room they need.
///
/// Because [`UI_SCALE`] is a zoom factor and zoom changes only take effect on
/// the *next* pass, this cannot be done by toggling the zoom. Instead the global
/// style is scaled by the reciprocal for the duration of the call, which cancels
/// the zoom exactly: a 12-point font at 8 points, drawn at 1.5x, is 12 points
/// again. Scaling the *global* style rather than the window's own `Ui` is what
/// gets the title bar and frame padding too — `Window` takes those from the
/// context when it opens.
///
/// A no-op on the desktop, where the scale is already 1 — and unreachable
/// there too, since a desktop pop-up is its own window rather than something
/// drawn inside the scaled canvas (see [`super::dialog`]).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(super) fn unscaled<R>(ctx: &egui::Context, show: impl FnOnce(&egui::Context) -> R) -> R {
    if UI_SCALE == 1.0 {
        return show(ctx);
    }
    let outer = ctx.global_style();
    let mut inner = (*outer).clone();
    scale_style(&mut inner, 1.0 / UI_SCALE);
    ctx.set_global_style(inner);
    let result = show(ctx);
    ctx.set_global_style(outer);
    result
}

/// Multiply every length in `style` by `k`: text sizes, spacing, padding and
/// the fixed widget metrics.
///
/// Only lengths — colours, opacities and flags are left alone, so the result is
/// the same style at a different size rather than a different style.
fn scale_style(style: &mut egui::Style, k: f32) {
    for font in style.text_styles.values_mut() {
        font.size *= k;
    }
    let s = &mut style.spacing;
    s.item_spacing *= k;
    s.button_padding *= k;
    s.interact_size *= k;
    s.default_area_size *= k;
    s.window_margin = scale_margin(s.window_margin, k);
    s.menu_margin = scale_margin(s.menu_margin, k);
    for v in [
        &mut s.indent,
        &mut s.slider_width,
        &mut s.slider_rail_height,
        &mut s.combo_width,
        &mut s.text_edit_width,
        &mut s.icon_width,
        &mut s.icon_width_inner,
        &mut s.icon_spacing,
        &mut s.tooltip_width,
        &mut s.menu_width,
        &mut s.menu_spacing,
        &mut s.combo_height,
    ] {
        *v *= k;
    }
    let sc = &mut s.scroll;
    sc.content_margin = scale_margin(sc.content_margin, k);
    for v in [
        &mut sc.bar_width,
        &mut sc.handle_min_length,
        &mut sc.bar_inner_margin,
        &mut sc.bar_outer_margin,
        &mut sc.floating_width,
        &mut sc.floating_allocated_width,
    ] {
        *v *= k;
    }
}

/// `Margin` is stored as whole points (`i8`), so this rounds rather than
/// scaling exactly — a margin is padding, and a point either way is invisible.
fn scale_margin(m: egui::Margin, k: f32) -> egui::Margin {
    let f = |v: i8| (v as f32 * k).round() as i8;
    egui::Margin { left: f(m.left), right: f(m.right), top: f(m.top), bottom: f(m.bottom) }
}

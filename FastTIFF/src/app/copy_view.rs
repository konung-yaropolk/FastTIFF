//! Putting the view on the clipboard as an image.
//!
//! The two view modes need opposite treatment, and the split is the whole
//! design here:
//!
//! * **2-D** is composed on the CPU by [`fast_tiff_viewer::snapshot`] — the
//!   current frame at its own size, every enabled channel through its own
//!   contrast window and LUT. Not a grab of what is on screen, because that is
//!   at whatever zoom and pan the user left it and, on a stack too large to
//!   upload whole, is only a window of the frame. Copying a picture means
//!   copying the image.
//! * **3-D** is the opposite: the view *is* the picture. Its camera, its
//!   projection, its interpolation and its brightness have no meaning outside
//!   the viewport, and there is no "original size" for a ray-march. So that one
//!   is a screenshot of the 3-D panel, cropped out of the window.
//!
//! The 3-D path is therefore asynchronous, and has to be: a screenshot is taken
//! by the backend at the end of a frame, so asking for one and reading it back
//! are necessarily different frames. [`request_volume`] asks,
//! [`take_screenshot`] collects, and the rect to cut is remembered in between.

use egui::ColorImage;

/// Ask the backend for a screenshot of this frame.
///
/// The result arrives as an [`egui::Event::Screenshot`] in a later frame — the
/// image is read back after the frame has been painted, so it cannot be
/// returned here. Callers remember the rect they want cropped out of it.
pub(super) fn request_volume(ctx: &egui::Context) {
    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
}

/// The screenshot from this frame's input, if one landed.
pub(super) fn take_screenshot(ctx: &egui::Context) -> Option<std::sync::Arc<ColorImage>> {
    ctx.input(|i| {
        i.events.iter().rev().find_map(|e| match e {
            egui::Event::Screenshot { image, .. } => Some(std::sync::Arc::clone(image)),
            _ => None,
        })
    })
}

/// Put `image` on the clipboard.
///
/// Goes out as an ordinary frame output command, which is what the existing
/// text copies do, so it travels the same path on every backend — arboard on
/// the desktop, the async clipboard API in the browser — and needs no platform
/// code here.
pub(super) fn put(ctx: &egui::Context, image: ColorImage) {
    ctx.output_mut(|o| o.commands.push(egui::OutputCommand::CopyImage(image)));
}

/// `rect` (in points) cut out of `image` (in physical pixels).
///
/// Clamped rather than trusted, and that is not defensiveness for its own sake:
/// [`ColorImage::region`] *asserts* on a region outside the image, so a rect
/// that is stale by one frame — the window was resized, or the panel moved
/// after the screenshot was asked for — would abort the program rather than
/// produce a slightly wrong crop. The rect is also whatever the layout
/// produced, which can be empty or inverted when the panel is collapsed to
/// nothing.
///
/// Returns `None` when the overlap is empty, which the caller reports as having
/// nothing to copy rather than putting a zero-pixel image on the clipboard.
pub(super) fn crop(
    image: &ColorImage,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Option<ColorImage> {
    let (w, h) = (image.width(), image.height());
    // A non-finite or absurd scale would make every coordinate garbage; 1.0 is
    // the honest fallback, and it crops *something* rather than nothing.
    let scale = if pixels_per_point.is_finite() && pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let to_px = |v: f32, limit: usize| {
        if v.is_finite() {
            ((v * scale).round().max(0.0) as usize).min(limit)
        } else {
            0
        }
    };
    let x0 = to_px(rect.min.x, w);
    let y0 = to_px(rect.min.y, h);
    let x1 = to_px(rect.max.x, w);
    let y1 = to_px(rect.max.y, h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(image.region_by_pixels([x0, y0], [x1 - x0, y1 - y0]))
}

/// Whether the copy hotkey was pressed this frame.
///
/// `Modifiers::COMMAND` is egui's platform-correct chord — ⌘ on macOS, Ctrl
/// everywhere else — so this is one line rather than a `cfg` per platform.
///
/// Skipped while a widget holds keyboard focus, because there the same chord
/// means "copy the text I have selected" and a text field's own copy must win.
/// Taking it unconditionally would make Ctrl+C in the frame-number box copy the
/// picture instead of the number, which reads as the box being broken.
pub(super) fn hotkey(ctx: &egui::Context) -> bool {
    if ctx.memory(|m| m.focused()).is_some() {
        return false;
    }
    ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::C))
}

/// An [`egui::ColorImage`] of a composed 2-D snapshot.
pub(super) fn image_of(snap: &fast_tiff_viewer::snapshot::Snapshot) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([snap.width as usize, snap.height as usize], &snap.rgba)
}

#[cfg(test)]
#[path = "copy_view_tests.rs"]
mod tests;

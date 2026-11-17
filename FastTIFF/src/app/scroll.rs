//! Deciding what a scroll gesture means over the image.
//!
//! Two devices arrive through the same event, and they want opposite things. A
//! mouse wheel is a discrete ratchet, ideal for stepping through frames — which
//! is what it has always done here, and what it should keep doing. A trackpad
//! is a two-dimensional surface with no detents, ideal for shoving a picture
//! around, and using it to step frames wastes the axis it has and the precision
//! it reports.
//!
//! egui distinguishes them: a trackpad reports pixel deltas
//! ([`MouseWheelUnit::Point`]), a wheel reports notches (`Line`, or `Page` for
//! the rare device that pages). That is the whole basis of the split. It is
//! platform-dependent in one direction — macOS reports trackpads as `Point`,
//! while Windows Precision Touchpads come through as `Line` and therefore keep
//! scrubbing — which is a limitation of what the platform tells us, not a
//! choice made here.
//!
//! Pure, and separate from the event loop for that reason: a wheel that stopped
//! changing frames would be an immediately obvious regression, and it should be
//! catchable without a trackpad, a Mac, or a display.

/// What a frame's worth of scroll events amounts to.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub(super) struct Wheel {
    /// Trackpad movement, in points, to shift the field of view by.
    pub(super) swipe: egui::Vec2,
    /// Wheel notches to scrub frames by. Fractional for a trackpad, which has
    /// no notches of its own.
    pub(super) notches: f32,
}

/// Pixels of trackpad scroll that count as one frame step, when a trackpad is
/// scrubbing rather than panning.
const POINTS_PER_FRAME: f32 = 50.0;

/// Split `events` into panning and frame-scrubbing.
///
/// `pannable` is whether the image is actually larger than the panel. When it
/// is not there is nothing to pan, so a trackpad scrubs frames like a wheel
/// rather than doing nothing at all.
///
/// A scroll with Ctrl or Cmd held is left alone entirely ([`zoom_held`]): that
/// is the zoom gesture, handled elsewhere, and counting it here would zoom and
/// scrub at once.
pub(super) fn classify(events: &[egui::Event], pannable: bool) -> Wheel {
    let mut out = Wheel::default();
    for event in events {
        let egui::Event::MouseWheel {
            unit,
            delta,
            modifiers,
            ..
        } = event
        else {
            continue;
        };
        if zoom_held(*modifiers) {
            continue;
        }
        match unit {
            egui::MouseWheelUnit::Point if pannable => out.swipe += *delta,
            egui::MouseWheelUnit::Point => out.notches += delta.y / POINTS_PER_FRAME,
            // Line / Page: one frame per unit.
            _ => out.notches += delta.y,
        }
    }
    out
}

/// Whether `modifiers` make a scroll zoom the picture: Ctrl, or Cmd, on
/// every platform. The zoom keys take the same ones (`zoom_key`).
///
/// It is egui's own test, the one that sends a scroll to
/// [`zoom_delta`](egui::InputState::zoom_delta) instead of
/// `smooth_scroll_delta`: the wheel's modifiers
/// [`matches_any`](egui::Modifiers::matches_any) the
/// [`zoom_modifier`](egui::InputOptions::zoom_modifier), which is
/// [`Modifiers::COMMAND`](egui::Modifiers::COMMAND), and `matches_any` takes
/// `command` in a pattern to mean any of `ctrl`, `command` and `mac_cmd`. So
/// egui zooms on both Ctrl and Cmd on a Mac too, where egui-winit reports
/// Ctrl as `ctrl` alone and Cmd as `mac_cmd` and `command`. Elsewhere it
/// reports Ctrl as `ctrl` and `command` together; a browser reports Ctrl that
/// way on every platform, and Cmd as egui-winit does on a Mac. A browser's
/// Cmd is the Meta key, which on Windows and Linux is the Windows or Super
/// key, so in a browser there that key zooms too, as it already did for egui.
/// Testing `ctrl` alone, as `classify` once did, left a Mac's Cmd+scroll to
/// zoom and scrub frames at once.
///
/// Nothing sets the zoom modifier, so it is egui's default, and
/// `what_egui_zooms_is_what_is_left_alone` holds this to egui's answer for
/// every combination of modifiers.
pub(super) fn zoom_held(modifiers: egui::Modifiers) -> bool {
    modifiers.matches_any(egui::Modifiers::COMMAND)
}

#[cfg(test)]
#[path = "scroll_tests.rs"]
mod tests;

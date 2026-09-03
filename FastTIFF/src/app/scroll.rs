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
/// Ctrl+scroll is left alone entirely: that is the zoom gesture, handled
/// elsewhere, and counting it here would zoom and scrub at once.
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
        if modifiers.ctrl {
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

#[cfg(test)]
#[path = "scroll_tests.rs"]
mod tests;

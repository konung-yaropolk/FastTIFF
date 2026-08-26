//! Inertia for the panned view: motion that keeps going briefly after the
//! fingers leave the trackpad, and dies away.
//!
//! Kept apart from the input handling because it is arithmetic and nothing
//! else — no egui events, no state beyond a velocity — which is what makes the
//! feel of it adjustable, and checkable, without a display.

/// How quickly a glide dies away, as an e-folding time in seconds.
///
/// The glide is down to about a third of its speed after this long and
/// imperceptible after roughly three times it, so this is a third of the total
/// coast. Short, deliberately: this is a *supplement* to whatever the platform
/// already does. macOS sends its own momentum events after a flick and those
/// arrive as ordinary input, so on a Mac this extends the tail rather than
/// providing it — a long time constant there would feel like the picture had
/// got away from you.
const TAU: f32 = 0.12;

/// Speed below which a glide simply stops, in points per second.
///
/// Without a floor the exponential never reaches zero: the picture would creep
/// for many seconds at a fraction of a pixel per frame, and every one of those
/// frames costs a repaint.
const MIN_SPEED: f32 = 24.0;

/// How much of the newest sample enters the velocity, per frame.
///
/// A flick's speed is read from the last few frames of contact, and raw
/// per-frame deltas are noisy — one short frame at the end of a gesture would
/// otherwise decide the whole glide. Blending keeps the recent history without
/// making a deliberate stop feel sticky.
const SMOOTHING: f32 = 0.4;

/// Movement that continues after its input stops.
#[derive(Default, Clone, Copy, Debug)]
pub(super) struct Glide {
    velocity: egui::Vec2,
}

impl Glide {
    /// Feed in a frame's worth of user movement. `dt` is that frame's duration.
    ///
    /// Returns the movement to apply now, which is exactly what was asked for —
    /// input is never damped, only what follows it.
    pub(super) fn push(&mut self, delta: egui::Vec2, dt: f32) -> egui::Vec2 {
        if dt > 0.0 && dt.is_finite() {
            let sample = delta / dt;
            self.velocity += (sample - self.velocity) * SMOOTHING;
        }
        delta
    }

    /// Advance one frame with no input, returning the movement to apply.
    pub(super) fn coast(&mut self, dt: f32) -> egui::Vec2 {
        if !(dt > 0.0 && dt.is_finite() && self.is_moving()) {
            return egui::Vec2::ZERO;
        }
        let step = self.velocity * dt;
        self.velocity *= (-dt / TAU).exp();
        if self.velocity.length() < MIN_SPEED {
            self.velocity = egui::Vec2::ZERO;
        }
        step
    }

    /// Whether a glide is still running, and so whether the frame after this
    /// one has to be drawn.
    pub(super) fn is_moving(&self) -> bool {
        self.velocity.length() >= MIN_SPEED
    }

    /// Stop dead. For anything that makes the motion meaningless — a grab on
    /// the image, a new file, a jump to another view.
    pub(super) fn stop(&mut self) {
        self.velocity = egui::Vec2::ZERO;
    }

    /// Stop on whichever axes are named, for running into the edge of the
    /// image: a glide that kept its speed while pinned against a bound would
    /// spring back the moment the bound moved.
    pub(super) fn stop_axes(&mut self, x: bool, y: bool) {
        if x {
            self.velocity.x = 0.0;
        }
        if y {
            self.velocity.y = 0.0;
        }
    }
}

#[cfg(test)]
#[path = "kinetic_tests.rs"]
mod tests;

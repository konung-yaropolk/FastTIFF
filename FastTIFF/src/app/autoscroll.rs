//! Autoscroll: the sticky mode a middle-click puts the view into, where moving
//! the pointer scrolls instead of dragging.
//!
//! The gesture browsers and Windows file managers have: click the wheel, the
//! cursor becomes a four-way arrow, and from then on the view drifts in
//! whatever direction the pointer sits relative to where you clicked. Further
//! from the anchor means faster. Click again to stop.
//!
//! It is a different thing from dragging, and worth having beside it rather
//! than instead of it. A drag can only move the view by as far as the hand can
//! travel before it runs out of desk, so crossing a 20000-pixel mosaic means a
//! dozen grab-drag-release cycles. Autoscroll crosses it by holding the pointer
//! an inch from the anchor and waiting, which is the whole point of the mode —
//! distance covered is a function of *time*, not of hand travel.
//!
//! # Why this is its own module
//!
//! Because it is a feel, expressed as arithmetic, and the two reasons to keep
//! those separate from the event loop both apply. A response curve is tuned by
//! reading numbers, not by watching a picture; and nothing here needs a
//! display, a mouse or an egui context to be checked, so the dead zone and the
//! cap can be held to their promises in ordinary tests. [`kinetic`] and
//! [`scroll`] are next door for the same reason.
//!
//! [`kinetic`]: super::kinetic
//! [`scroll`]: super::scroll

/// Radius around the anchor, in points, inside which nothing scrolls.
///
/// Not a nicety — without it the mode is unusable. The pointer never sits
/// exactly where it was clicked (a click itself moves it by a pixel or two),
/// so a curve through the origin means the view creeps continuously and there
/// is no way to make it stop short of leaving the mode. The dead zone is what
/// gives the gesture a neutral position to come back to.
///
/// Twelve points: comfortably larger than the jitter of a hand resting on a
/// mouse, and small enough that the scroll starts as soon as you mean it to.
///
/// Visible to the caller because the marker it draws at the anchor is a ring of
/// exactly this radius. The thing on screen and the thing the arithmetic uses
/// have to be the same number, or the ring becomes a decoration that lies about
/// where the gesture is parked.
pub(super) const DEAD_ZONE: f32 = 18.0;

/// Points of scroll per second, per point of pointer offset past
/// [`DEAD_ZONE`].
///
/// Linear in the offset, like the platforms' own, because that is the curve a
/// hand can aim: twice as far is twice as fast, and holding a speed means
/// holding a position. An exponential curve covers more range from less desk
/// but makes the fast end unsteerable, which on a gigapixel mosaic means
/// overshooting the thing you were looking for.
const GAIN: f32 = 24.0;

/// The fastest it will ever scroll, in points per second.
///
/// A pointer flung into the corner of a 4K display is some 2000 points from an
/// anchor near the middle, which at [`GAIN`] would be 12000 points a second —
/// about eight screens, far enough that a frame's worth is a jump rather than a
/// motion and the picture becomes a strobe. The cap keeps the top speed fast
/// but followable.
const MAX_SPEED: f32 = 40000.0;

/// How fast, and in which direction, the view should scroll — in points per
/// second, to be multiplied by the frame's duration.
///
/// `anchor` is where the mode was entered and `pointer` is where the pointer is
/// now. Returns zero inside the dead zone, which is the resting state and not
/// an error.
///
/// # Radial, not per-axis
///
/// The dead zone is a circle and the speed comes off the vector's length, so
/// the direction the view takes is exactly the direction from the anchor to the
/// pointer. Subtracting the dead zone per axis instead would bend the motion
/// towards whichever axis cleared the zone first: a pointer dragged out at 45°
/// would start scrolling diagonally, correctly, but a pointer at 20° would
/// start off horizontal and swing round as the vertical component cleared — the
/// view curving away from the direction the hand is pointing, which reads as
/// the control fighting you.
pub(super) fn velocity(anchor: egui::Pos2, pointer: egui::Pos2) -> egui::Vec2 {
    let offset = pointer - anchor;
    let len = offset.length();
    // `is_finite` before the comparison, for the same reason the minimap checks
    // its extents: a NaN position compares false against everything, so
    // `len <= DEAD_ZONE` alone would wave it through and the view would be
    // panned by a NaN — which, unlike a wrong number, cannot be recovered from
    // by moving the mouse back.
    if !len.is_finite() || len <= DEAD_ZONE {
        return egui::Vec2::ZERO;
    }
    let speed = ((len - DEAD_ZONE) * GAIN).min(MAX_SPEED);
    // `offset / len` rather than `normalized()`: the length is already computed
    // and known non-zero here, which `normalized()` cannot assume.
    (offset / len) * speed
}

#[cfg(test)]
#[path = "autoscroll_tests.rs"]
mod tests;

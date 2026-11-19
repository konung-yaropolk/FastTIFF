//! The 2D view: how far the image is zoomed and scrolled, where that puts it
//! on screen, and how it moves from one frame to the next.
//!
//! Pan and zoom are not applied by moving the viewport — a GPU backend would
//! clamp an oversized one to the framebuffer and squash the picture. The image
//! is drawn into the part of the panel it covers, and pan and zoom are the
//! part of the image sampled there: [`ImagePlacement::inputs`] turns this view into
//! the [`ImageInputs`] of [`Viewer::sync`](crate::Viewer::sync).
//!
//! [`View2d`] is plain arithmetic in logical points ([`LogicalPt`],
//! [`LogicalVec`], [`LogicalRect`]) and the image's own pixels ([`ImagePx`]),
//! so a frontend's job is to read its input and call the method for it,
//! converting its toolkit's positions at its edge. What is the frontend's own stays there: how it reads
//! the wheel, the window it may size around the image, the navigator it may
//! draw.
//!
//! A frame of the view, in the order the methods are meant to be called:
//!
//! 1. before the panels are laid out, the zoom input: [`View2d::pinch`],
//!    [`View2d::step_zoom`], then [`View2d::advance_zoom_glide`], every frame
//!    — against the last frame's panel and placement, which is where the
//!    pointer was aimed. After it, [`View2d::is_moving`] is the frame's
//!    [`FrameInputs::moving`](crate::FrameInputs::moving);
//! 2. once the panel is: [`View2d::set_panel`], the opening fit
//!    ([`View2d::open_fitted`] or [`View2d::open_at_level`]), and
//!    [`View2d::measure`] for this frame's
//!    [`ImageExtent`];
//! 3. what moves the view by hand: [`View2d::pan_to`], the autoscroll
//!    ([`View2d::toggle_autoscroll`], [`View2d::autoscroll_by`]),
//!    [`View2d::drag`], then [`View2d::clamp_pan`];
//! 4. [`View2d::place`], which says where the image is drawn and which part of
//!    it is on screen;
//! 5. what carries on after the hand: [`View2d::scrub`] and
//!    [`View2d::glide_pan`].

mod autoscroll;
mod kinetic;
pub(crate) mod zoom;

pub use crate::screen::{ImagePx, LogicalPt, LogicalRect, LogicalVec};
pub use autoscroll::DEAD_ZONE as AUTOSCROLL_DEAD_ZONE;
pub use zoom::{fit_zoom, opening_zoom, ZOOM_LEVELS};

use zoom::ZoomLadder;

use crate::frame::ImageInputs;

/// Time constant of the glide a zoom step takes to its new level.
///
/// The approach is exponential, so a step is about 63% done after this long.
/// With [`ZOOM_GLIDE_SNAP`] ending it, one rung of the ladder takes about
/// 130 ms and the longest run across the whole ladder about 190 ms — long
/// enough to read as movement rather than a jump, short enough that a second
/// notch never feels held up. Slower starts to feel like the program is
/// deciding whether to obey.
const ZOOM_GLIDE_TAU: f32 = 0.035;

/// How close to the target ends the glide. A ratio, not a difference, because
/// zoom is geometric.
///
/// One percent, which is well under [`FIT_SNAP`](zoom::FIT_SNAP)'s two — already described
/// there as below what anyone can see in a zoom. Chasing the last percent is
/// pure latency: an exponential approach spends as long covering it as it did
/// covering the first three quarters.
const ZOOM_GLIDE_SNAP: f32 = 0.01;

/// Longest frame the glide will integrate. A hitch — a slow decode, a window
/// drag — must not be turned into one enormous zoom jump on the frame after.
const ZOOM_GLIDE_MAX_DT: f32 = 1.0 / 20.0;

/// A zoom step in flight: the level being glided to, and the point it turns
/// about.
///
/// Only the *drawn* zoom is animated. Everything a frontend does around it —
/// resizing a window to the image, keeping the cursor's point on the same
/// desktop spot — is decided once from the target ([`View2d::zoom_settled`]),
/// so a window still moves once per step rather than sixty times.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ZoomGlide {
    /// The ladder rung being approached.
    target: f32,
    /// Screen point held fixed while the picture scales about it.
    anchor: LogicalPt,
}

/// A zoom step that was taken: what [`View2d::step_zoom`] tells a frontend
/// that sizes a window around the image, and keeps the point under the
/// cursor on the same spot of the desktop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomStep {
    /// The zoom on screen when the step was taken.
    pub from: f32,
    /// The point the picture scales about: the pointer, or the middle of the
    /// panel when the pointer is not over it.
    pub anchor: LogicalPt,
}

/// The image at this frame's zoom, measured against the room it is given:
/// what [`View2d::measure`] works out once a frame, and every step of the
/// frame after it goes by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageExtent {
    /// The whole image's size on screen at the zoom on screen.
    pub size: LogicalVec,
    /// The room the panel gives it.
    pub room: LogicalVec,
    /// How far the image overflows the room each way, 0 where it fits —
    /// which is also how far it can be panned.
    pub overflow: LogicalVec,
}

impl ImageExtent {
    /// Whether the image overflows the room on either axis, and so can be
    /// panned.
    pub fn pannable(&self) -> bool {
        self.overflow.x > 0.0 || self.overflow.y > 0.0
    }

    /// How much of the image is on screen: the room on an axis that
    /// overflows, the whole image on one that fits.
    pub fn shown(&self) -> LogicalVec {
        self.size.min(self.room)
    }
}

/// Where the image is drawn this frame: what [`View2d::place`] decides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePlacement {
    /// Where the image's top-left corner would be drawn at full size:
    /// scrolled by the pan on an axis that overflows, centred on one that
    /// fits. May lie outside the panel.
    pub origin: LogicalPt,
    /// The whole image's size on screen ([`ImageExtent::size`]).
    pub size: LogicalVec,
    /// The whole image's rect on screen, `origin` + `size`.
    pub full: LogicalRect,
    /// The part of [`full`](Self::full) inside the panel: all that is drawn.
    pub visible: LogicalRect,
}

impl ImagePlacement {
    /// Whether any of the image is on screen.
    pub fn is_visible(&self) -> bool {
        self.visible.is_positive()
    }

    /// Which part of the image is on screen and how much room it is drawn
    /// in, for the frame's [`FrameInputs`](crate::FrameInputs), on a display
    /// of `pixels_per_point`.
    pub fn inputs(&self, pixels_per_point: f32) -> ImageInputs {
        ImageInputs::of_visible(self.origin, self.size, self.visible, pixels_per_point)
    }
}

/// The 2D view of one image in one panel: its zoom and pan, the zoom step and
/// the swipe being glided out, the autoscroll anchor, and what the last frame
/// laid out. See the [module docs](self) for a frame of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View2d {
    /// Points per image pixel on screen: 1.0 = one point per pixel.
    zoom: f32,
    /// How far the image is scrolled, from its top-left corner, on an axis
    /// that overflows the panel.
    pan: LogicalVec,
    /// Set when an image opens: the frontend fits it once it knows the room,
    /// and says so with [`open_fitted`](Self::open_fitted) or
    /// [`open_at_level`](Self::open_at_level).
    awaiting_fit: bool,
    /// The exact zoom at which the open image fits the view, when the
    /// frontend fitted it so, and an extra rung on this image's
    /// [`ZoomLadder`].
    ///
    /// Per image, so it is cleared on open: it is a property of this picture
    /// in this panel, and carrying a previous file's factor over would leave a
    /// rung that fits nothing.
    fit_rung: Option<f32>,
    /// A zoom step being glided out, if one is in flight.
    zoom_glide: Option<ZoomGlide>,
    /// Whether a pinch has zoomed the view since the last
    /// [`advance_zoom_glide`](Self::advance_zoom_glide), the end of a frame's
    /// zoom input.
    pinched: bool,
    /// Whether the view was moving at the end of the last frame's zoom input:
    /// see [`is_moving`](Self::is_moving).
    moving: bool,
    /// Inertia for swipe-panning, so the view keeps moving for a moment after
    /// the fingers leave the trackpad. See [`kinetic`].
    pan_glide: kinetic::Glide,
    /// Where an autoscroll gesture is anchored, when that mode is on.
    ///
    /// The anchor is the whole of the mode's state: the velocity is a function
    /// of where the pointer is *relative to it* (see [`autoscroll`]), so there
    /// is no accumulated speed to keep and nothing to decay. Holding the point
    /// rather than a flag is also what lets the marker be drawn where the
    /// gesture started, which is the only thing telling the user where neutral
    /// is.
    autoscroll: Option<LogicalPt>,
    /// Sub-notch wheel travel accumulated until it adds up to a frame step.
    scroll_accum: f32,
    /// Top-left of the drawn image, from the last frame.
    image_origin: LogicalPt,
    /// The panel the image is drawn in, from the last frame.
    panel: LogicalRect,
    /// Whether the image overflowed the panel on the *previous* frame.
    ///
    /// A zoom step changes the image's size immediately but a window that
    /// holds it only on the frame after, so between the two the image
    /// genuinely overflows a panel that is about to grow to fit it. Requiring
    /// the overflow to survive a frame is what stops a navigator blinking on
    /// during a zoom between two views that both fit.
    was_pannable: bool,
}

impl Default for View2d {
    /// One point per pixel, scrolled nowhere, with no image laid out yet.
    fn default() -> Self {
        View2d {
            zoom: 1.0,
            pan: LogicalVec::ZERO,
            awaiting_fit: false,
            fit_rung: None,
            zoom_glide: None,
            pinched: false,
            moving: false,
            pan_glide: kinetic::Glide::default(),
            autoscroll: None,
            scroll_accum: 0.0,
            image_origin: LogicalPt::ZERO,
            panel: LogicalRect::default(),
            was_pannable: false,
        }
    }
}

impl View2d {
    /// Forget the picture that was open, now that another has landed.
    ///
    /// Everything goes back to [`Default`] but the fields named here, so a
    /// field added to the view is reset unless someone decides it should not
    /// be. What that drops, and why it has to:
    ///
    /// * the zoom and pan: the new picture starts at 1:1 until the frontend
    ///   fits it ([`awaits_fit`](Self::awaits_fit)); the fit factor belongs to
    ///   the last picture;
    /// * whether it was pannable: a new file starts with no history of
    ///   overflowing, or a navigator would draw on its first frame from the
    ///   previous file's answer;
    /// * the pan glide: momentum from the last picture means nothing in this
    ///   one;
    /// * the autoscroll anchor: a position on screen tied to a transform that
    ///   has just been reset, which would park the new picture's neutral
    ///   point wherever the last click happened to land;
    /// * the zoom glide: a glide belongs to the picture that was on screen,
    ///   and carrying one into a new file would slide the fresh image away
    ///   from the zoom it just opened at.
    pub fn reset(&mut self) {
        *self = View2d {
            // The opening fit, still to come.
            awaiting_fit: true,
            // Kept: the last layout's geometry, which the next frame lays out
            // afresh, and wheel travel short of one frame.
            image_origin: self.image_origin,
            panel: self.panel,
            scroll_accum: self.scroll_accum,
            ..View2d::default()
        };
    }

    /// The zoom on screen: points per image pixel.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// The zoom the picture is heading for: the glide's target while one is in
    /// flight, otherwise what is on screen.
    ///
    /// Everything outside the drawing itself asks this rather than
    /// [`zoom`](Self::zoom) — a window's size, the readout, the next step off
    /// the ladder — so a glide changes when the picture arrives, never where
    /// it is going.
    pub fn zoom_settled(&self) -> f32 {
        self.zoom_glide.map_or(self.zoom, |g| g.target)
    }

    /// How far the image is scrolled.
    pub fn pan(&self) -> LogicalVec {
        self.pan
    }

    /// The rung the open image's fit added to its ladder: its fit-to-panel
    /// zoom, when it was fitted so ([`open_fitted`](Self::open_fitted)).
    pub fn fit_rung(&self) -> Option<f32> {
        self.fit_rung
    }

    /// The zoom levels a step moves along for this image.
    pub(crate) fn ladder(&self) -> ZoomLadder {
        ZoomLadder::of(self.fit_rung)
    }

    /// Whether the image that opened has still to be fitted.
    pub fn awaits_fit(&self) -> bool {
        self.awaiting_fit
    }

    /// Whether a zoom step is gliding out.
    pub fn is_zoom_gliding(&self) -> bool {
        self.zoom_glide.is_some()
    }

    /// Whether the view is moving this frame: a zoom step still gliding, or a
    /// pinch, as of the frame's [`advance_zoom_glide`](Self::advance_zoom_glide).
    /// What a frontend sets [`FrameInputs::moving`](crate::FrameInputs::moving)
    /// from, so the sync does not start decoding a window for a zoom the view
    /// is about to leave behind.
    pub fn is_moving(&self) -> bool {
        self.moving
    }

    /// Where the autoscroll is anchored, while that mode is on.
    pub fn autoscroll(&self) -> Option<LogicalPt> {
        self.autoscroll
    }

    /// Where the image's top-left corner was drawn on the last frame.
    pub fn image_origin(&self) -> LogicalPt {
        self.image_origin
    }

    /// The panel the image was last laid out in.
    pub fn panel(&self) -> LogicalRect {
        self.panel
    }

    /// The image has been fitted to its panel: show it at `zoom`, scrolled to
    /// its corner, with `zoom` a rung of its ladder for as long as it is open,
    /// so stepping back out returns to it — for a fit that need not be one of
    /// the fixed levels ([`fit_zoom`]).
    pub fn open_fitted(&mut self, zoom: f32) {
        self.fit_rung = Some(zoom);
        self.open_at_level(zoom);
    }

    /// The image has been given its opening zoom, one of the fixed levels
    /// ([`opening_zoom`]): show it at `zoom`, scrolled to its corner, with no
    /// rung of its own.
    pub fn open_at_level(&mut self, zoom: f32) {
        self.zoom = zoom;
        self.pan = LogicalVec::ZERO;
        self.awaiting_fit = false;
    }

    /// Scale the picture about `anchor`, keeping whatever sits under that point
    /// under it. The pan is expressed against the previous frame's placement,
    /// which is where the anchor was aimed.
    fn zoom_about(&mut self, anchor: LogicalPt, new_zoom: f32) {
        let p = (anchor - self.image_origin) / self.zoom;
        self.pan = self.panel.min - (anchor - p * new_zoom);
        self.zoom = new_zoom;
    }

    /// A pinch: the zoom times `ratio`, within the ends of the ladder, about
    /// `center`, the point between the fingers — and translated by however far
    /// they slid (`slide`), so a pinch can pan and zoom at once the way it
    /// does everywhere else.
    ///
    /// Free between the ends of the ladder, not stepped: a pinch is
    /// continuous, and at whole levels a slow pinch would do nothing until it
    /// suddenly jumped a step. It ends any zoom glide, which would be a second
    /// thing steering the same value. A `ratio` that is not a number zooms
    /// nothing: kept, it would leave a zoom no step could leave from.
    pub fn pinch(&mut self, center: LogicalPt, ratio: f32, slide: LogicalVec) {
        let old_zoom = self.zoom;
        let (lo, hi) = self.ladder().bounds();
        let ratio = if ratio.is_nan() { 1.0 } else { ratio };
        let new_zoom = (old_zoom * ratio).clamp(lo, hi);
        let p = (center - self.image_origin) / old_zoom;
        self.pan = self.panel.min - (center - p * new_zoom) - slide;
        self.zoom = new_zoom;
        self.zoom_glide = None;
        self.pinched = true;
    }

    /// A step along the ladder in `dir` (+1 in, −1 out), glided to about
    /// `pointer` when it is over the panel and the panel's middle when it is
    /// not. `None` at the end of the ladder.
    ///
    /// Stepped off wherever the picture is *heading*, not where it has got to,
    /// so a second notch during a glide advances a second rung instead of
    /// re-deciding the first one.
    pub fn step_zoom(&mut self, dir: i32, pointer: Option<LogicalPt>) -> Option<ZoomStep> {
        let old_zoom = self.zoom;
        let settled = self.zoom_settled();
        let new_zoom = self.ladder().step(settled, dir);
        if (new_zoom - settled).abs() > f32::EPSILON {
            let anchor = pointer
                .filter(|p| self.panel.contains(*p))
                .unwrap_or_else(|| self.panel.center());
            self.zoom_glide = Some(ZoomGlide {
                target: new_zoom,
                anchor,
            });
            return Some(ZoomStep {
                from: old_zoom,
                anchor,
            });
        }
        None
    }

    /// Drop the zoom glide in flight, leaving the zoom where it has got to:
    /// for when something else has taken the zoom over, or there is no 2D
    /// view left for it to animate.
    pub fn stop_zoom_glide(&mut self) {
        self.zoom_glide = None;
    }

    /// Advance a zoom glide by `dt` seconds, returning whether one is still
    /// running (and so wants the next frame drawn).
    ///
    /// Exponential approach, integrated in the logarithm so the picture changes
    /// by the same *ratio* every frame: zoom is geometric, and interpolating it
    /// linearly would crawl at the bottom of a step and race at the top.
    ///
    /// Called once a frame, glide or not, after the frame's pinch and step:
    /// it is also where the frame's [`is_moving`](Self::is_moving) is noted,
    /// after the advance, so the frame a glide lands on is already still.
    pub fn advance_zoom_glide(&mut self, dt: f32) -> bool {
        let running = self.advance_glide(dt);
        self.moving = self.zoom_glide.is_some() || self.pinched;
        self.pinched = false;
        running
    }

    /// The advance itself: whether the glide is still running.
    fn advance_glide(&mut self, dt: f32) -> bool {
        let Some(glide) = self.zoom_glide else {
            return false;
        };
        let dt = dt.clamp(0.0, ZOOM_GLIDE_MAX_DT);
        let from = self.zoom;
        if !(from > 0.0 && glide.target > 0.0) {
            self.zoom_glide = None;
            return false;
        }
        let alpha = 1.0 - (-dt / ZOOM_GLIDE_TAU).exp();
        let next = (from.ln() + (glide.target.ln() - from.ln()) * alpha).exp();
        let arrived = !next.is_finite() || (next / glide.target - 1.0).abs() <= ZOOM_GLIDE_SNAP;
        let new_zoom = if arrived { glide.target } else { next };
        self.zoom_about(glide.anchor, new_zoom);
        if arrived {
            self.zoom_glide = None;
        }
        !arrived
    }

    /// The panel the image is laid out in this frame.
    pub fn set_panel(&mut self, panel: LogicalRect) {
        self.panel = panel;
    }

    /// `image` at the zoom on screen, against `room`: the panel's size as it
    /// was asked for, which a toolkit's layout may round differently from the
    /// rect it hands back.
    ///
    /// A point of tolerance on the overflow, so sub-pixel rounding between a
    /// window's size and the panel inside it does not register as a pannable
    /// overflow.
    pub fn measure(&self, image: ImagePx, room: LogicalVec) -> ImageExtent {
        let size = image.at_zoom(self.zoom);
        let overflow = LogicalVec::new(
            (size.x - room.x - 1.0).max(0.0),
            (size.y - room.y - 1.0).max(0.0),
        );
        ImageExtent {
            size,
            room,
            overflow,
        }
    }

    /// Note whether this frame's `extent` overflows the panel, once a frame,
    /// and say whether the image has now overflowed it steadily — this frame
    /// and the one before, with its opening fit done.
    ///
    /// The question a navigator asks before it shows itself: see
    /// `was_pannable` for why one frame of overflow is not enough. A second
    /// call in the same frame would count it twice.
    pub fn note_pannable(&mut self, extent: &ImageExtent) -> bool {
        let settled = !self.awaiting_fit;
        let pannable = extent.pannable();
        let steady = pannable && self.was_pannable && settled;
        self.was_pannable = pannable && settled;
        steady
    }

    /// Scroll to `pan` — a navigator's answer to "show me there" — ending any
    /// glide. Clamped with the rest by [`clamp_pan`](Self::clamp_pan).
    pub fn pan_to(&mut self, pan: LogicalVec) {
        self.pan_glide.stop();
        self.pan = pan;
    }

    /// Drag the picture by `delta`: it follows the pointer. A hand on it ends
    /// any glide.
    pub fn drag(&mut self, delta: LogicalVec) {
        self.pan_glide.stop();
        self.pan -= delta;
    }

    /// A click that turns autoscroll on or off: off from anywhere, so a second
    /// click always means stop; on, anchored `at`, only where there is
    /// something to scroll — a mode that swallowed the click and then did
    /// nothing would be indistinguishable from a dead button.
    pub fn toggle_autoscroll(&mut self, at: Option<LogicalPt>, extent: &ImageExtent) {
        self.autoscroll = match self.autoscroll {
            Some(_) => None,
            None if extent.pannable() => at,
            None => None,
        };
    }

    /// Leave autoscroll, if it is on.
    pub fn end_autoscroll(&mut self) {
        self.autoscroll = None;
    }

    /// One frame of autoscroll, `dt` seconds long, with the pointer at
    /// `pointer`: the view drifts towards it. Returns whether it moved, which
    /// is when it wants the next frame — parked in the dead zone it should
    /// cost no more than any other idle view.
    ///
    /// The opposite sign to a drag, and deliberately: a drag moves the
    /// *picture* under the pointer, while this moves the *view* towards it.
    pub fn autoscroll_by(&mut self, pointer: LogicalPt, dt: f32) -> bool {
        let Some(anchor) = self.autoscroll else {
            return false;
        };
        let v = autoscroll::velocity(anchor, pointer);
        self.pan += v * dt;
        v != LogicalVec::ZERO
    }

    /// Keep the pan within how far the image overflows.
    pub fn clamp_pan(&mut self, extent: &ImageExtent) {
        self.pan.x = self.pan.x.clamp(0.0, extent.overflow.x);
        self.pan.y = self.pan.y.clamp(0.0, extent.overflow.y);
    }

    /// Where the image goes this frame, in the panel set for it, noting the
    /// origin for the next frame's zoom input.
    pub fn place(&mut self, extent: &ImageExtent) -> ImagePlacement {
        let panel = self.panel;
        let origin = LogicalPt::new(
            if extent.overflow.x > 0.0 {
                panel.min.x - self.pan.x
            } else {
                panel.min.x + (extent.room.x - extent.size.x) * 0.5
            },
            if extent.overflow.y > 0.0 {
                panel.min.y - self.pan.y
            } else {
                panel.min.y + (extent.room.y - extent.size.y) * 0.5
            },
        );
        self.image_origin = origin;
        let full = LogicalRect::from_min_size(origin, extent.size);
        ImagePlacement {
            origin,
            size: extent.size,
            full,
            visible: full.intersect(panel),
        }
    }

    /// Wheel travel towards the next frame: `by` frames' worth, added to what
    /// is left over from before, and the whole frames that makes, to step by.
    /// Fractions are kept, so a touchpad's small deltas and a short stack's
    /// fast scroll still move.
    pub fn scrub(&mut self, by: f32) -> i32 {
        self.scroll_accum += by;
        let steps = self.scroll_accum.trunc();
        self.scroll_accum -= steps;
        steps as i32
    }

    /// Drop wheel travel short of a frame: the pointer has left the picture.
    pub fn forget_scrub(&mut self) {
        self.scroll_accum = 0.0;
    }

    /// Move the view by a frame's swipe — `swipe`, 0 when there was none and
    /// the last one coasts on — over `dt` seconds, within the overflow.
    /// Returns whether it is still coasting, which is motion with no input
    /// behind it, and so wants the next frame drawn.
    ///
    /// The sign matches a drag — the picture follows the fingers — because they
    /// are the same gesture done two ways. An axis that runs into the edge of
    /// the image stops there: a glide left wound up against a bound would
    /// spring forward again the moment the bound moved. An image that fits has
    /// nothing to glide.
    pub fn glide_pan(&mut self, swipe: LogicalVec, dt: f32, extent: &ImageExtent) -> bool {
        if !extent.pannable() {
            self.pan_glide.stop();
        }
        let motion = if swipe == LogicalVec::ZERO {
            self.pan_glide.coast(dt)
        } else {
            self.pan_glide.push(swipe, dt)
        };
        if motion != LogicalVec::ZERO {
            let want = self.pan - motion;
            self.pan.x = want.x.clamp(0.0, extent.overflow.x);
            self.pan.y = want.y.clamp(0.0, extent.overflow.y);
            self.pan_glide.stop_axes(
                (self.pan.x - want.x).abs() > 0.01,
                (self.pan.y - want.y).abs() > 0.01,
            );
        }
        self.pan_glide.is_moving()
    }
}

#[cfg(test)]
mod tests;

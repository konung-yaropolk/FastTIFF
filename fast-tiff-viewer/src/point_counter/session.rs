//! The gestures on the picture: what one frame of the primary button does to
//! the points — add, drag, delete — worked out in image pixels, so it is tested
//! without a window.
//!
//! While the counter's window is open the primary button on the picture is the
//! counter's: a click adds a point to the selected group on the frame in view,
//! a press on a point drags it, and Alt+click on a point deletes it. The
//! frontend fills a [`Gesture`] from its toolkit's pointer state each frame and
//! hands it to [`Session::track`]; everything else about the pointer — panning
//! with the other buttons, the wheel — stays the frontend's.

use super::{frame_of_position, PointCounter};
use crate::document::Document;

/// A point being dragged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    /// Which point.
    pub index: usize,
    /// Where it was when the drag began, which is where an undo puts it back.
    pub from: (f32, f32),
    /// The point minus where the press landed, in image pixels. The point
    /// keeps this offset from the pointer, so taking hold of it a few pixels
    /// off-centre does not make it jump under the cursor — which, at a high
    /// zoom, would be a move nobody made.
    pub offset: (f32, f32),
}

/// The counter while its window is open: the points, and a drag in flight.
pub struct Session {
    pub counter: PointCounter,
    pub drag: Option<Drag>,
}

/// One frame of the primary button over the picture, in image pixels. Taken
/// from egui's response by the canvas, so [`Session::track`] can be
/// tested without one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gesture {
    /// Where the pointer is acting: over the picture, or holding it.
    pub at: Option<(f32, f32)>,
    /// Where the button went down. egui reports a drag only once the pointer
    /// has moved past its click threshold, so this — not `at` — is where the
    /// user aimed when the drag is first seen.
    pub pressed_at: Option<(f32, f32)>,
    /// A primary click: pressed and released without moving.
    pub clicked: bool,
    /// Alt was held.
    pub alt: bool,
    /// A primary drag began, went on, or ended this frame.
    pub drag_started: bool,
    pub dragging: bool,
    pub drag_stopped: bool,
    /// Something else had the click or the press: the navigator, or leaving
    /// autoscroll, which a primary click or drag does. Counting it as well
    /// would make one gesture do two things.
    pub taken: bool,
}

impl Session {
    /// A session on `counter`, with nothing being dragged.
    pub fn new(counter: PointCounter) -> Self {
        Session {
            counter,
            drag: None,
        }
    }

    /// A session on `stack`: its overlay's points, each on the frame its
    /// plane is on ([`PointCounter::from_overlay`]), or a fresh counter when
    /// it has none.
    pub fn for_stack(stack: &Document) -> Self {
        Self::new(PointCounter::from_overlay(stack))
    }

    /// Act on one frame of the primary button over the picture: `frame` is
    /// the frame in view, `radius` the reach of a press in image pixels, and
    /// `size` the image's, which a new point has to be on and a dragged one
    /// is kept on.
    ///
    /// * a click adds a point, unless it is on one;
    /// * Alt+click on a point deletes it, and anywhere else does nothing;
    /// * a drag that starts on a point moves it, and is one undo step when it
    ///   ends; a drag that starts anywhere else does nothing.
    ///
    /// Only the points shown on `frame` are under the pointer: one placed on
    /// another frame is not on screen to be grabbed.
    pub fn track(&mut self, g: Gesture, frame: usize, radius: f32, size: (u32, u32)) {
        let (w, h) = (size.0 as f32, size.1 as f32);
        if let Some(drag) = self.drag {
            if let Some((x, y)) = g.at {
                let (x, y) = (x + drag.offset.0, y + drag.offset.1);
                // On the image, which ends just short of `w`: the right edge
                // itself is on no pixel, as it is for a click.
                let (right, bottom) = (w.next_down().max(0.0), h.next_down().max(0.0));
                self.counter
                    .drag_to(drag.index, x.clamp(0.0, right), y.clamp(0.0, bottom));
            }
            if g.drag_stopped || !g.dragging {
                self.counter.finish_drag(drag.index, drag.from);
                self.drag = None;
            }
            return;
        }
        if g.taken {
            return;
        }
        if g.drag_started {
            let Some((px, py)) = g.pressed_at.or(g.at) else {
                return;
            };
            let Some(index) = self.counter.hit(px, py, frame, radius) else {
                return;
            };
            let p = self.counter.points[index];
            self.drag = Some(Drag {
                index,
                from: (p.x, p.y),
                offset: (p.x - px, p.y - py),
            });
            // The frame the drag is first seen on already has the pointer
            // somewhere past the press.
            let rest = Gesture {
                drag_started: false,
                ..g
            };
            self.track(rest, frame, radius, size);
            return;
        }
        if !g.clicked {
            return;
        }
        let Some((x, y)) = g.at else {
            return;
        };
        match (self.counter.hit(x, y, frame, radius), g.alt) {
            (Some(i), true) => self.counter.delete(i),
            // On the image only: a mark in the margin around it is on no
            // pixel, and would be exported as one.
            (None, false) if (0.0..w).contains(&x) && (0.0..h).contains(&y) => {
                self.counter.add(x, y, frame);
            }
            // A click on a point is not a second point on top of it, and
            // Alt+click on nothing deletes nothing.
            _ => {}
        }
    }

    /// Keep every point on its plane while the stack's axes are reassigned:
    /// `before` is [`frame_starts`](super::frame_starts) taken just before,
    /// and `stack` the stack as it is now. A point is pinned to its plane —
    /// the one it came with from a file, or the one its frame began at — which
    /// is what stays put; it goes to the frame that plane is on now, and onto
    /// every frame when it is on none. Pinned, it stays pinned, so changing
    /// the axes and changing them back puts every point where it was
    /// ([`PointCounter::reanchor`]).
    pub fn follow_axes(&mut self, before: &[u32], stack: &Document) {
        self.counter.reanchor(
            |frame| before.get(frame).copied().filter(|&p| p != 0),
            |plane| frame_of_position(stack, plane),
        );
    }

    /// Take back the last add, move or delete. A drag in flight is the latest
    /// move: it ends where it began, and the gesture carrying it on does
    /// nothing more.
    pub fn undo(&mut self) -> bool {
        if let Some(drag) = self.drag.take() {
            self.counter.finish_drag(drag.index, drag.from);
        }
        self.counter.undo()
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;

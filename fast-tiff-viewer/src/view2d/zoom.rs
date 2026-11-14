//! The zoom levels: the fixed ladder a step moves along, the rung an image's
//! fit adds to it, and the zoom a freshly opened image is shown at.

use crate::screen::{ImagePx, LogicalVec};

/// Discrete zoom levels the viewer snaps to (3.1% … 3200%). Zooming in/out
/// steps between adjacent levels. Above 100% the levels are mostly whole-number
/// magnifications (200%, 300%, 400%, …), where one source pixel maps to an exact
/// NxN block of screen pixels — crisp and uniform under our nearest sampling —
/// with 150% as the one fractional step for finer control. The stored values
/// are rounded to the percentages shown in the UI (e.g. 0.333 reads as 33.3%).
pub const ZOOM_LEVELS: [f32; 21] = [
    0.031, 0.042, 0.063, 0.083, 0.125, 0.167, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    8.0, 12.0, 16.0, 24.0, 32.0,
];

/// How close a fit factor must be to a fixed level before it counts as that
/// level rather than being inserted next to it.
///
/// Two steps a percent apart are one step as far as the eye and the wheel are
/// concerned, and the pair would cost a wheel notch to cross for no visible
/// change. 2% is below what anyone can see in a zoom and well above the
/// rounding in the table.
pub(crate) const FIT_SNAP: f32 = 0.02;

/// The zoom levels in effect: the fixed ladder, plus this image's exact
/// fit-to-window factor when it has one and it is not already on the ladder.
///
/// Inserted in order rather than appended, so stepping through it is
/// monotonic. Held in place rather than in a `Vec`: a pinch asks for the
/// ladder's ends on every frame it runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoomLadder {
    rungs: [f32; ZOOM_LEVELS.len() + 1],
    len: usize,
}

impl ZoomLadder {
    /// The ladder for an image whose fit-to-window factor is `fit`, if it has
    /// one. A factor that is not a positive number is ignored rather than
    /// poisoning the ladder.
    pub fn of(fit: Option<f32>) -> Self {
        let mut rungs = [0.0; ZOOM_LEVELS.len() + 1];
        rungs[..ZOOM_LEVELS.len()].copy_from_slice(&ZOOM_LEVELS);
        let mut len = ZOOM_LEVELS.len();
        if let Some(f) = fit.filter(|f| f.is_finite() && *f > 0.0) {
            if !ZOOM_LEVELS.iter().any(|z| (z - f).abs() <= f * FIT_SNAP) {
                let at = ZOOM_LEVELS.partition_point(|z| *z < f);
                rungs.copy_within(at..len, at + 1);
                rungs[at] = f;
                len += 1;
            }
        }
        ZoomLadder { rungs, len }
    }

    /// The rungs, smallest first.
    pub fn rungs(&self) -> &[f32] {
        &self.rungs[..self.len]
    }

    /// The next rung in `dir` (+1 = in, −1 = out) from whichever rung is
    /// nearest `current`, clamped to the ends of the ladder.
    ///
    /// The fit factor is a rung for as long as its image is open, so zooming
    /// back out lands on the fitted view instead of stepping past it to the
    /// nearest fixed level. A `current` that is not a number is nearest the
    /// bottom rung.
    pub fn step(&self, current: f32, dir: i32) -> f32 {
        let levels = self.rungs();
        // `total_cmp`, which orders a NaN distance too; for the distances of
        // numbers, never negative zero, the order `partial_cmp` gives.
        let nearest = levels
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - current).abs().total_cmp(&(**b - current).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let next = (nearest as i32 + dir).clamp(0, levels.len() as i32 - 1) as usize;
        levels[next]
    }

    /// The range a continuous zoom — a pinch — may take: the ends of the
    /// ladder. A step cannot leave the ladder, so this is what keeps a pinch
    /// from running off to a scale no wheel notch could ever return from.
    pub fn bounds(&self) -> (f32, f32) {
        let levels = self.rungs();
        (levels[0], levels[levels.len() - 1])
    }
}

/// The zoom at which `image` exactly fits `room`, capped at 1:1 so a small
/// image is not magnified into a blur. `None` for a degenerate size on either
/// side — a room not laid out yet, most often.
pub fn fit_zoom(image: ImagePx, room: LogicalVec) -> Option<f32> {
    let (img_w, img_h) = (image.width as f32, image.height as f32);
    if !(img_w >= 1.0 && img_h >= 1.0 && room.x >= 1.0 && room.y >= 1.0) {
        return None;
    }
    let z = (room.x / img_w).min(room.y / img_h).min(1.0);
    z.is_finite().then_some(z).filter(|z| *z > 0.0)
}

/// The largest fixed level at most 100% at which `image`, with `chrome_height`
/// points of bars above and below it, still fits `room` — so a normal image
/// opens at 100% and a big one at 50% or 25%. Where even the smallest level
/// overflows, the smallest: the image opens there and is panned.
///
/// For a window sized around the image, where `room` is the monitor's work
/// area; a canvas the frontend does not size is [`fit_zoom`]'s.
pub fn opening_zoom(image: ImagePx, room: LogicalVec, chrome_height: f32) -> f32 {
    let (img_w, img_h) = (image.width as f32, image.height as f32);
    for &z in ZOOM_LEVELS.iter().rev().filter(|&&z| z <= 1.0) {
        if img_w * z <= room.x && img_h * z + chrome_height <= room.y {
            return z;
        }
    }
    ZOOM_LEVELS[0]
}

#[cfg(test)]
#[path = "zoom_tests.rs"]
mod tests;

//! How a progress fraction travels between a worker and the bar that draws it.
//!
//! A worker and the interface share one `AtomicU32`: the worker stores into it,
//! the interface reads it every frame. Both ends have to agree on what the
//! integer means, and this module is the only place that says.
//!
//! It exists because they once did not. The plugin host stored basis points
//! (`fraction * 10_000`) while the app read permille (`/ 1_000`) and clamped to
//! one, so a plugin that had done a tenth of its work drew a full bar — and a
//! stabilization reached "100%" a fraction of a second after it started, then
//! sat there for the rest of the run. Two copies of a scale factor, each right
//! on its own.

use std::sync::atomic::{AtomicU32, Ordering};

/// What the integer is multiplied by. Permille: finer than a pixel of any bar
/// this is drawn as, coarse enough that a `u32` never comes near overflowing.
const SCALE: f32 = 1000.0;

/// Nothing reported yet — which is drawn as a spinner rather than as 0%.
///
/// A worker is not obliged to report at all, and a bar parked at nought for a
/// whole run reads as "stuck" rather than as "no estimate".
pub const UNKNOWN: u32 = u32::MAX;

/// Record how far along a worker is. Out-of-range fractions are clamped.
pub fn store(progress: &AtomicU32, fraction: f32) {
    let fraction = if fraction.is_nan() {
        0.0
    } else {
        fraction.clamp(0.0, 1.0)
    };
    progress.store((fraction * SCALE).round() as u32, Ordering::Relaxed);
}

/// Forget any estimate, so the bar shows a spinner until the next [`store`].
pub fn clear(progress: &AtomicU32) {
    progress.store(UNKNOWN, Ordering::Relaxed);
}

/// How far along, or `None` when nothing has been reported.
pub fn load(progress: &AtomicU32) -> Option<f32> {
    match progress.load(Ordering::Relaxed) {
        UNKNOWN => None,
        stored => Some((stored as f32 / SCALE).clamp(0.0, 1.0)),
    }
}

#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;

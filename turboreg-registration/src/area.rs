// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The area every frame of a registered stack covers.
//!
//! Resampling a frame onto the reference uncovers a strip of it — a different
//! one for every frame, and for a rotation or a shear a slanted one. What is
//! left when every frame's strip is taken away is the part of the field the
//! whole stack saw, and [`Border::Crop`] keeps the largest rectangle inside it
//! — at whatever angle holds the most, which is upright unless a turned one
//! holds clearly more. A turned rectangle is turned upright in the result, so
//! the result's rows run along the rectangle's, not the frame's.
//!
//! A pixel counts as covered exactly when [`Border::Constant`] would not fill
//! it: every frame's transformation reads it from inside the frame. A stack
//! whose transformations only shift and scale covers an upright rectangle, and
//! its area is upright; one that rotates, shears or warps is searched, angle by
//! angle on what every frame covers, and then checked frame by frame at the
//! very points the result will be read from.
//!
//! [`Border::Crop`]: crate::Border::Crop
//! [`Border::Constant`]: crate::Border::Constant

use crate::transform::{on_frame, Matrix, Point};
#[cfg(test)]
use image_registration_support::area::largest_rectangle;
use image_registration_support::area::{largest_turned, largest_upright};

pub use image_registration_support::area::{Area, NoCommonArea};

/// The largest rectangle of an `ly * lx` frame that every one of `matrices`
/// covers — each maps reference coordinates to the frame's own, as
/// [`Registered::matrices`](crate::Registered::matrices) do — at any angle.
///
/// Upright, unless a turned rectangle holds more than one row or column's
/// worth of pixels more than the best upright one — less than that is the
/// lattice, not the motion, and not worth turning the result for.
///
/// `parallel` spreads the frames over every core; the answer is the same.
pub fn common_area(
    matrices: &[Matrix],
    ly: usize,
    lx: usize,
    parallel: bool,
) -> Result<Area, NoCommonArea> {
    let none = NoCommonArea {
        ly,
        lx,
        frames: matrices.len(),
    };
    if ly == 0 || lx == 0 {
        return Err(none);
    }
    let reach: Vec<f64> = matrices.iter().map(|m| reach(m, ly, lx)).collect();
    let reads = |t: usize, y: f64, x: f64| on_frame(matrices[t].apply(Point::new(x, y)), lx, ly);
    // Shifting and scaling cover an upright rectangle, and nothing turned fits
    // inside a rectangle better than the rectangle.
    let upright_only = matrices
        .iter()
        .all(|m| m.c[2] == 0.0 && m.c[3] == 0.0 && m.c[5] == 0.0 && m.c[7] == 0.0);
    let found = if upright_only {
        largest_upright(ly, lx, &reach, &reads, parallel)
    } else {
        largest_turned(ly, lx, &reach, &reads, parallel)
    };
    found.ok_or(none)
}

/// How far a transformation moves any pixel of an `ly * lx` frame along
/// either axis. Largest at a corner: the displacement of an affine or a
/// bilinear map, `x' - x` and `y' - y`, is itself bilinear in the position,
/// and a bilinear function is largest on a rectangle at a corner. Unbounded
/// for a map that is not finite.
fn reach(m: &Matrix, ly: usize, lx: usize) -> f64 {
    let (w, h) = ((lx - 1) as f64, (ly - 1) as f64);
    let mut most = 0.0f64;
    for (x, y) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
        let p = m.apply(Point::new(x, y));
        let (dx, dy) = ((p.x - x).abs(), (p.y - y).abs());
        if !(dx.is_finite() && dy.is_finite()) {
            return f64::INFINITY;
        }
        most = most.max(dx).max(dy);
    }
    most
}

#[cfg(test)]
#[path = "area_tests.rs"]
mod tests;

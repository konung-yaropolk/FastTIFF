// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The area every frame of a registered recording covers, and cutting frames
//! down to it.
//!
//! Moving frames onto a reference uncovers a strip of each, a different one
//! for every frame. What is left when every frame's strip is taken away is the
//! part of the field the whole recording saw, and [`Border::Crop`] keeps the
//! largest rectangle inside it — at whatever angle holds the most, which is
//! upright unless a turned one holds clearly more. A turned rectangle is turned
//! upright in the result, so the result's rows run along the rectangle's, not
//! the frame's.
//!
//! A pixel counts as covered exactly when [`Border::Constant`] would not fill
//! it: every frame's correction reads it from inside the frame. A rigid shift
//! covers an upright rectangle, so for a rigid run the answer is the
//! intersection of rectangles, exactly, and upright. A warp covers an irregular
//! shape, so a warped run is searched: angle by angle on what every frame
//! covers, and then checked frame by frame at the very points the result will
//! be read from.
//!
//! [`Border::Crop`]: crate::Border::Crop
//! [`Border::Constant`]: crate::Border::Constant

use crate::nonrigid::{on_frame, sample, BlockShift, Blocks, Grid};
use crate::par::*;
use crate::rigid::Shift;
use crate::settings::Border;
#[cfg(test)]
use image_registration_support::area::largest_rectangle;
use image_registration_support::area::largest_turned;

pub use image_registration_support::area::{Area, NoCommonArea};

/// How each frame is moved onto the reference: its rigid shift and, when it
/// has one, its block field — the pairing
/// [`apply_batch`](crate::pipeline::apply_batch) moves frames by.
struct Moves<'a> {
    shifts: &'a [Shift],
    warp: Option<(Grid, &'a [Vec<BlockShift>])>,
    /// Columns at the left and right of the raw frame that hold no measured
    /// sample: what the bidirectional-phase correction leaves behind on
    /// alternate lines ([`bidiphase::shift`](crate::bidiphase::shift) slides
    /// them and keeps what was there). A rectangle covers alternate lines
    /// alike, so the whole frame is narrowed by them.
    stale: (usize, usize),
}

impl<'a> Moves<'a> {
    fn new(
        shifts: &'a [Shift],
        warp: Option<(&Blocks, &'a [Vec<BlockShift>])>,
        bidiphase: i32,
        lx: usize,
    ) -> Moves<'a> {
        let n = bidiphase.unsigned_abs() as usize;
        let stale = match bidiphase {
            // `shift` leaves an offset as wide as the frame alone.
            _ if n >= lx => (0, 0),
            b if b > 0 => (n, 0),
            _ => (0, n),
        };
        Moves {
            shifts,
            warp: warp.map(|(blocks, fields)| (Grid::of(blocks), fields)),
            stale,
        }
    }

    fn shift(&self, t: usize) -> Shift {
        self.shifts.get(t).copied().unwrap_or(Shift {
            dy: 0,
            dx: 0,
            corr: 0.0,
        })
    }

    fn field(&self, t: usize) -> Option<(&Grid, &[BlockShift])> {
        self.warp
            .as_ref()
            .and_then(|(grid, fields)| fields.get(t).map(|f| (grid, f.as_slice())))
    }

    fn warped(&self) -> bool {
        (0..self.shifts.len()).any(|t| self.field(t).is_some())
    }

    /// Where frame `t`'s correction reads the registered frame's point `(y,
    /// x)` from: the arithmetic of [`nonrigid::warp`](crate::nonrigid::warp)
    /// and [`shift_frame`](crate::shift_frame), which is what makes "covered"
    /// mean "not filled".
    fn source(&self, t: usize, y: f64, x: f64) -> (f64, f64) {
        let s = self.shift(t);
        match self.field(t) {
            Some((grid, field)) => {
                let (dy, dx) = grid.shift(field, grid.row(y), grid.col(x));
                (y + dy + s.dy as f64, x + dx + s.dx as f64)
            }
            None => (y + s.dy as f64, x + s.dx as f64),
        }
    }

    /// Whether frame `t` reads the registered point `(y, x)` from a measured
    /// sample of an `ly * lx` frame.
    fn reads(&self, t: usize, y: f64, x: f64, ly: usize, lx: usize) -> bool {
        let (left, right) = self.stale;
        let (sy, sx) = self.source(t, y, x);
        on_frame(sy, sx - left as f64, ly, lx - left - right)
    }

    /// How far frame `t` moves anything along either axis, counting the stale
    /// columns as distance: its largest block shift — the interpolation
    /// between blocks never goes past it — the rigid shift on top, and the
    /// wider of the two stale strips. Unbounded for a field that is not
    /// finite.
    fn reach(&self, t: usize) -> f64 {
        let s = self.shift(t);
        let worst = |most: f64, v: f32| {
            if v.is_finite() {
                most.max(v.abs() as f64)
            } else {
                f64::INFINITY
            }
        };
        let (fy, fx) = match self.field(t) {
            Some((_, field)) => field
                .iter()
                .fold((0.0, 0.0), |(y, x), b| (worst(y, b.dy), worst(x, b.dx))),
            None => (0.0, 0.0),
        };
        let strip = self.stale.0.max(self.stale.1) as f64;
        (fy + s.dy.unsigned_abs() as f64).max(fx + s.dx.unsigned_abs() as f64) + strip
    }
}

/// The largest rectangle every frame covers once moved by `shifts` (and
/// `warp`'s fields, where there are), in an `ly * lx` frame: what
/// [`Border::Crop`] keeps, and what to give [`Cut`].
///
/// `bidiphase` is the offset the frames were corrected by before they were
/// measured, as [`Registered::bidiphase`](crate::Registered::bidiphase): the
/// correction slides alternate lines and leaves that many columns at one edge
/// unmeasured, and the area keeps clear of them. Zero for none.
///
/// Upright, unless a turned rectangle holds more than one row or column's
/// worth of pixels more than the best upright one — less than that is the
/// lattice, not the motion, and not worth turning the result for. A rigid run
/// is always upright: its frames cover upright rectangles, and nothing turned
/// fits inside a rectangle better than the rectangle.
///
/// `parallel` spreads the frames over every core; the answer is the same.
pub fn common_area(
    ly: usize,
    lx: usize,
    shifts: &[Shift],
    warp: Option<(&Blocks, &[Vec<BlockShift>])>,
    bidiphase: i32,
    parallel: bool,
) -> Result<Area, NoCommonArea> {
    let none = NoCommonArea {
        ly,
        lx,
        frames: shifts.len(),
    };
    if ly == 0 || lx == 0 {
        return Err(none);
    }
    let moves = Moves::new(shifts, warp, bidiphase, lx);
    let (left, right) = moves.stale;
    if !moves.warped() {
        // Output row `y` reads row `y + dy`, which is on the frame for `y` in
        // `-dy ..= ly - 1 - dy`; every frame's range, met. The same for
        // columns, between the stale strips.
        let (mut y0, mut y1) = (0i64, ly as i64 - 1);
        let (mut x0, mut x1) = (left as i64, (lx - 1 - right) as i64);
        for s in shifts {
            let (dy, dx) = (s.dy as i64, s.dx as i64);
            (y0, y1) = (y0.max(-dy), y1.min(ly as i64 - 1 - dy));
            (x0, x1) = (
                x0.max(left as i64 - dx),
                x1.min((lx - 1 - right) as i64 - dx),
            );
        }
        if y0 > y1 || x0 > x1 {
            return Err(none);
        }
        return Ok(Area {
            y: y0 as f64,
            x: x0 as f64,
            height: (y1 - y0 + 1) as usize,
            width: (x1 - x0 + 1) as usize,
            angle: 0.0,
        });
    }
    let reach: Vec<f64> = (0..shifts.len()).map(|t| moves.reach(t)).collect();
    largest_turned(
        ly,
        lx,
        &reach,
        &|t, y, x| moves.reads(t, y, x, ly, lx),
        parallel,
    )
    .ok_or(none)
}

/// Frames moved onto the reference and cut to an [`Area`], in one resampling:
/// each pixel of the result is read from the raw frame where that frame's
/// correction says it came from, so turning the area costs no second
/// interpolation.
pub struct Cut<'a> {
    ly: usize,
    lx: usize,
    area: Area,
    moves: Moves<'a>,
}

impl<'a> Cut<'a> {
    /// Cut `ly * lx` frames, moved by `shifts` and `warp` as
    /// [`apply_batch`](crate::pipeline::apply_batch) moves them, to `area` —
    /// the [`common_area`] of the same corrections. The frames are the ones
    /// measured: corrected for bidirectional phase already, when they were.
    pub fn new(
        ly: usize,
        lx: usize,
        area: Area,
        shifts: &'a [Shift],
        warp: Option<(&Blocks, &'a [Vec<BlockShift>])>,
    ) -> Cut<'a> {
        Cut {
            ly,
            lx,
            area,
            moves: Moves::new(shifts, warp, 0, lx),
        }
    }

    /// The area cut to.
    pub fn area(&self) -> &Area {
        &self.area
    }

    /// Frame `t` — or any plane of timepoint `t`, which is how a second
    /// channel follows the first — moved and cut: `area.height * area.width`
    /// samples, row by row.
    ///
    /// Read bilinearly, as [`nonrigid::warp`](crate::nonrigid::warp) reads;
    /// an upright cut of a rigidly shifted frame reads whole pixels, so it is
    /// exactly that frame's samples.
    pub fn apply(&self, frame: &[f32], t: usize) -> Vec<f32> {
        let a = &self.area;
        let (ly, lx) = (self.ly, self.lx);
        let whole = a.is_upright() && self.moves.field(t).is_none();
        let mut out = Vec::with_capacity(a.height * a.width);
        for r in 0..a.height {
            for c in 0..a.width {
                let (y, x) = a.at(r, c);
                let (sy, sx) = self.moves.source(t, y, x);
                out.push(if whole {
                    let iy = sy.clamp(0.0, (ly - 1) as f64) as usize;
                    let ix = sx.clamp(0.0, (lx - 1) as f64) as usize;
                    frame[iy * lx + ix]
                } else {
                    // Inside the area every read lands on the frame; the clamp
                    // only keeps a misuse from reading past it.
                    sample(frame, ly, lx, sy, sx, Border::Edge)
                });
            }
        }
        out
    }

    /// [`apply`](Self::apply) for a batch of planes, `frame_of[i]` the
    /// timepoint `planes[i]` belongs to, on every core when `parallel`.
    pub fn apply_batch(
        &self,
        planes: &[Vec<f32>],
        frame_of: &[usize],
        parallel: bool,
    ) -> Vec<Vec<f32>> {
        let pairs: Vec<(&Vec<f32>, usize)> = planes.iter().zip(frame_of.iter().copied()).collect();
        if parallel {
            pairs.par_iter().map(|&(p, t)| self.apply(p, t)).collect()
        } else {
            pairs.iter().map(|&(p, t)| self.apply(p, t)).collect()
        }
    }
}

#[cfg(test)]
#[path = "area_tests.rs"]
mod tests;

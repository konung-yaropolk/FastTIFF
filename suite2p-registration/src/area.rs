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
use crate::settings::{Border, ON_FRAME};

/// A rectangle of the registered frame, at any angle: what [`Border::Crop`]
/// keeps.
///
/// The result's pixel at `row`, `col` is the frame point [`Area::at`] gives;
/// an upright area's are whole pixels, a turned one's fall between them.
///
/// [`Border::Crop`]: crate::Border::Crop
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Area {
    /// Where the result's first pixel — row 0, column 0 — sits in the frame:
    /// the row.
    pub y: f64,
    /// And the column.
    pub x: f64,
    /// Rows kept.
    pub height: usize,
    /// Columns kept.
    pub width: usize,
    /// How far the area's rows are turned from the frame's, in radians, within
    /// a quarter turn either way; positive turns them clockwise as an image is
    /// shown, with rows running down. `0.0` for an upright area.
    pub angle: f64,
}

impl Area {
    /// Where the result's pixel at `row`, `col` sits in the frame, as
    /// `(row, column)`: one pixel apart along the area's rows and down its
    /// columns.
    pub fn at(&self, row: usize, col: usize) -> (f64, f64) {
        let (sin, cos) = self.angle.sin_cos();
        let (r, c) = (row as f64, col as f64);
        (self.y + c * sin + r * cos, self.x + c * cos - r * sin)
    }

    /// Whether the area's rows are the frame's.
    pub fn is_upright(&self) -> bool {
        self.angle == 0.0
    }
}

/// The frames moved so far apart that no pixel is covered by all of them.
///
/// Then there is nothing to crop to: every pixel of the frame is uncovered in
/// at least one frame, so cropping cannot keep anything that was measured
/// throughout. A recording like that can still be registered with a border
/// that fills — [`Border::Constant`], [`Border::Edge`] or
/// [`Border::Native`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NoCommonArea {
    /// Rows in a frame.
    pub ly: usize,
    /// Columns in a frame.
    pub lx: usize,
    /// How many frames were compared.
    pub frames: usize,
}

impl std::fmt::Display for NoCommonArea {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the frames moved so far apart that no pixel of the {} x {} frame is covered by \
             all {} of them, so there is no common area to crop to",
            self.lx, self.ly, self.frames
        )
    }
}

impl std::error::Error for NoCommonArea {}

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
    largest_turned(ly, lx, &reach, &|t, y, x| moves.reads(t, y, x, ly, lx), parallel)
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

// ------------------------------------------------------------- the search

/// Whether frame `t` reads the registered point `(y, x)` from inside the
/// frame: where its correction says it comes from, on the part of the raw
/// frame that holds measured samples.
type Reads<'s> = &'s (dyn Fn(usize, f64, f64) -> bool + Sync);

/// A square lattice of points one `step` apart, turned by `angle`: the pixels
/// a turned result would have.
struct Lattice {
    oy: f64,
    ox: f64,
    sin: f64,
    cos: f64,
    angle: f64,
    step: f64,
    rows: usize,
    cols: usize,
}

impl Lattice {
    /// The frame's own pixels.
    fn upright(ly: usize, lx: usize) -> Lattice {
        Lattice {
            oy: 0.0,
            ox: 0.0,
            sin: 0.0,
            cos: 1.0,
            angle: 0.0,
            step: 1.0,
            rows: ly,
            cols: lx,
        }
    }

    /// Turned about the frame's middle pixel, and wide enough to cover the
    /// frame at any angle. Through the middle pixel, so that at angle zero it
    /// is the frame's own pixels.
    fn turned(ly: usize, lx: usize, angle: f64, step: f64) -> Lattice {
        let (sin, cos) = angle.sin_cos();
        let (cy, cx) = (((ly - 1) / 2) as f64, ((lx - 1) / 2) as f64);
        let k = ((ly as f64).hypot(lx as f64) / 2.0 / step).ceil() + 1.0;
        let n = 2 * k as usize + 1;
        // Point (k, k) is the middle pixel.
        Lattice {
            oy: cy - k * step * (sin + cos),
            ox: cx - k * step * (cos - sin),
            sin,
            cos,
            angle,
            step,
            rows: n,
            cols: n,
        }
    }

    fn point(&self, row: usize, col: usize) -> (f64, f64) {
        let (r, c) = (row as f64 * self.step, col as f64 * self.step);
        (
            self.oy + c * self.sin + r * self.cos,
            self.ox + c * self.cos - r * self.sin,
        )
    }

    /// A rectangle of this lattice's points as an area. Only for a lattice
    /// one pixel apart, which is what an area's pixels are.
    fn area(&self, rect: Rect) -> Area {
        let (y, x) = self.point(rect.row, rect.col);
        Area {
            y,
            x,
            height: rect.height,
            width: rect.width,
            angle: self.angle,
        }
    }
}

/// The largest area every frame covers: upright, and at the best of a few
/// promising angles, and whichever of the two wins by more than the lattice
/// can account for.
///
/// The angles are found roughly ([`candidate_angles`]) and each checked
/// exactly: the rough judgement ranks neighbouring angles by a pixel here and
/// there, which is enough to say where to look and not enough to choose.
fn largest_turned(
    ly: usize,
    lx: usize,
    reach: &[f64],
    reads: Reads<'_>,
    parallel: bool,
) -> Option<Area> {
    let upright = Lattice::upright(ly, lx);
    let covered = exact_mask(&upright, ly, lx, reach, reads, parallel);
    if !covered.contains(&true) {
        return None;
    }
    let up = largest_rectangle(&covered, ly, lx).map(|r| upright.area(r));
    let mut turned: Option<Area> = None;
    for angle in candidate_angles(&covered, ly, lx) {
        let lattice = Lattice::turned(ly, lx, angle, 1.0);
        let mask = exact_mask(&lattice, ly, lx, reach, reads, parallel);
        let here = largest_rectangle(&mask, lattice.rows, lattice.cols).map(|r| lattice.area(r));
        if let Some(a) = here {
            if turned.is_none_or(|b| a.height * a.width > b.height * b.width) {
                turned = Some(a);
            }
        }
    }
    match (up, turned) {
        (Some(u), Some(t)) => {
            let margin = u.height.min(u.width);
            Some(if t.height * t.width > u.height * u.width + margin {
                t
            } else {
                u
            })
        }
        (u, t) => u.or(t),
    }
}

/// Which of `lattice`'s points every frame covers: inside the registered
/// frame, and read from inside the raw frame by every frame's correction.
///
/// A point further from the frame's edge than a frame moves anything is read
/// from inside that frame whatever its correction does there, so each frame
/// looks only at the band of points within its own reach of the edge — on a
/// 512-pixel frame that moves by tens of pixels, a few percent of them.
fn exact_mask(
    lattice: &Lattice,
    ly: usize,
    lx: usize,
    reach: &[f64],
    reads: Reads<'_>,
    parallel: bool,
) -> Vec<bool> {
    let cols = lattice.cols;
    let mut mask = vec![false; lattice.rows * cols];
    let deepest = reach.iter().fold(0.0f64, |a, &r| a.max(r));
    let mut band: Vec<(f64, usize)> = Vec::new();
    for (i, cell) in mask.iter_mut().enumerate() {
        let (y, x) = lattice.point(i / cols, i % cols);
        let edge = y.min(ly as f64 - 1.0 - y).min(x).min(lx as f64 - 1.0 - x);
        if edge < -ON_FRAME {
            // Outside the registered frame, so outside the stack.
            continue;
        }
        *cell = true;
        if edge < deepest + 1.0 {
            band.push((edge, i));
        }
    }
    band.sort_by(|a, b| a.0.total_cmp(&b.0));
    let look = |frames: &[usize]| -> Vec<bool> {
        let mut ok = vec![true; band.len()];
        for &t in frames {
            let limit = reach[t] + 1.0;
            for (k, &(edge, i)) in band.iter().enumerate() {
                if edge >= limit {
                    break;
                }
                if ok[k] {
                    let (y, x) = lattice.point(i / cols, i % cols);
                    ok[k] = reads(t, y, x);
                }
            }
        }
        ok
    };
    let frames: Vec<usize> = (0..reach.len()).collect();
    let chunk = frames.len().div_ceil(workers() * 4).max(1);
    let parts: Vec<Vec<bool>> = if parallel {
        frames.par_chunks(chunk).map(look).collect()
    } else {
        frames.chunks(chunk).map(look).collect()
    };
    for part in parts {
        for (k, ok) in part.into_iter().enumerate() {
            if !ok {
                mask[band[k].1] = false;
            }
        }
    }
    mask
}

/// The angles worth checking exactly, best first, judged on the frame-pixel
/// mask of what every frame covers: a turned point counts when the pixel
/// nearest it does. Every degree within a quarter turn, then every tenth of
/// one around the best — the middle of the best run, since neighbouring
/// degrees often tie — on a lattice coarse enough to keep a large frame
/// quick; the six that hold the most of those tenths. Never upright, which is
/// checked anyway.
fn candidate_angles(covered: &[bool], ly: usize, lx: usize) -> Vec<f64> {
    const CHECKED: usize = 6;
    let step = (ly.max(lx) as f64 / 256.0).ceil().max(1.0);
    let held = |degrees: f64| -> usize {
        let lattice = Lattice::turned(ly, lx, degrees.to_radians(), step);
        let mask: Vec<bool> = (0..lattice.rows * lattice.cols)
            .map(|i| {
                let (y, x) = lattice.point(i / lattice.cols, i % lattice.cols);
                roughly_covered(covered, ly, lx, y, x)
            })
            .collect();
        largest_rectangle(&mask, lattice.rows, lattice.cols).map_or(0, |r| r.height * r.width)
    };
    let coarse: Vec<f64> = (-44..=45).filter(|&d| d != 0).map(f64::from).collect();
    let around = middle_of_best(&coarse, &held);
    let mut fine: Vec<(usize, f64)> = (-10..=10)
        .map(|k| around + f64::from(k) / 10.0)
        .filter(|d| d.abs() > 1e-9)
        .map(|d| (held(d), d))
        .collect();
    // Most held first; the nearer the middle of the scan, the earlier among
    // equals.
    fine.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then((a.1 - around).abs().total_cmp(&(b.1 - around).abs()))
    });
    fine.iter()
        .take(CHECKED)
        .map(|&(_, d)| d.to_radians())
        .collect()
}

/// Of `angles`, in order, the middle of the first run that holds the most.
fn middle_of_best(angles: &[f64], held: &dyn Fn(f64) -> usize) -> f64 {
    let scores: Vec<usize> = angles.iter().map(|&a| held(a)).collect();
    let most = scores.iter().copied().max().unwrap_or(0);
    let first = scores.iter().position(|&s| s == most).unwrap_or(0);
    let run = scores[first..].iter().take_while(|&&s| s == most).count();
    angles[first + run / 2]
}

/// Whether a point between pixels is covered, by the pixel nearest it.
fn roughly_covered(covered: &[bool], ly: usize, lx: usize, y: f64, x: f64) -> bool {
    let inside = |v: f64, n: usize| v >= -ON_FRAME && v <= (n - 1) as f64 + ON_FRAME;
    if !(inside(y, ly) && inside(x, lx)) {
        return false;
    }
    let (y, x) = (y.round().clamp(0.0, (ly - 1) as f64), x.round().clamp(0.0, (lx - 1) as f64));
    covered[y as usize * lx + x as usize]
}

/// A rectangle of a mask's cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Rect {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

/// The largest rectangle of `true` in a `rows * cols` row-major mask; the
/// first one found, scanning down, when several tie.
///
/// Row by row, each column's run of `true` ending at that row is a bar, and the
/// largest rectangle resting on that row is the largest rectangle under those
/// bars — found in one pass with a stack of bars still growing. Every row in
/// turn, so `O(rows * cols)` in all.
pub(crate) fn largest_rectangle(mask: &[bool], rows: usize, cols: usize) -> Option<Rect> {
    let mut heights = vec![0usize; cols];
    let mut stack: Vec<usize> = Vec::with_capacity(cols + 1);
    let mut best: Option<Rect> = None;
    let mut best_size = 0;
    for row in 0..rows {
        for (c, h) in heights.iter_mut().enumerate() {
            *h = if mask[row * cols + c] { *h + 1 } else { 0 };
        }
        stack.clear();
        for c in 0..=cols {
            let h = if c < cols { heights[c] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] < h {
                    break;
                }
                stack.pop();
                let height = heights[top];
                let from = stack.last().map_or(0, |&s| s + 1);
                let width = c - from;
                if height * width > best_size {
                    best_size = height * width;
                    best = Some(Rect {
                        row: row + 1 - height,
                        col: from,
                        height,
                        width,
                    });
                }
            }
            stack.push(c);
        }
    }
    best
}

#[cfg(test)]
#[path = "area_tests.rs"]
mod tests;

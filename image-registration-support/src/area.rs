// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The area every frame of a registered recording covers, found the same way
//! whatever moved the frames.
//!
//! Moving frames onto a reference uncovers a strip of each, a different one
//! for every frame. What is left when every frame's strip is taken away is the
//! part of the field the whole recording saw, and a crop keeps the largest
//! rectangle inside it — at whatever angle holds the most, which is upright
//! unless a turned one holds clearly more.
//!
//! The registration crates differ in how a frame moves (a shift, a block
//! warp, a fitted matrix), and supply that as two things: how far each frame
//! can move a pixel (`reach`), and whether it reads a point from inside the
//! frame ([`Reads`]). The search over those, upright ([`largest_upright`]) or
//! at an angle ([`largest_turned`]), is this module.

use crate::par::{map_in_order, workers};

/// How far past the outermost sample a position may land and still count as
/// on the frame: rounding, not motion. A fitted transformation that is the
/// identity to within `1e-12` would otherwise blank the last row and column
/// when the border is a constant.
pub const ON_FRAME: f64 = 1e-6;

/// A rectangle of the registered frame, at any angle: what a crop to the
/// common area keeps.
///
/// The result's pixel at `row`, `col` is the frame point [`Area::at`] gives;
/// an upright area's are whole pixels, a turned one's fall between them.
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
/// that fills: a constant, the edge repeated, or the method's own.
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

// ------------------------------------------------------------- the search

/// Whether frame `t` reads the registered point `(y, x)` from inside the
/// frame: where its correction (a shift, a warp, a fitted correction)
/// sends it, on the part of the raw frame that holds measured samples. Each
/// registration crate supplies its own; the search below is the same for all.
pub type Reads<'s> = &'s (dyn Fn(usize, f64, f64) -> bool + Sync);

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

/// The largest upright area every frame covers: the frame's own pixels,
/// checked exactly, and the largest rectangle of the ones every frame reads
/// from inside itself. For corrections that move frames without turning them
/// (shifts, scalings), where nothing turned can do better.
pub fn largest_upright(
    ly: usize,
    lx: usize,
    reach: &[f64],
    reads: Reads<'_>,
    parallel: bool,
) -> Option<Area> {
    let lattice = Lattice::upright(ly, lx);
    let mask = exact_mask(&lattice, ly, lx, reach, reads, parallel);
    largest_rectangle(&mask, ly, lx).map(|r| lattice.area(r))
}

/// The largest area every frame covers: upright, and at the best of a few
/// promising angles, and whichever of the two wins by more than the lattice
/// can account for.
///
/// The angles are found roughly ([`candidate_angles`]) and each checked
/// exactly: the rough judgement ranks neighbouring angles by a pixel here and
/// there, which is enough to say where to look and not enough to choose.
pub fn largest_turned(
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
/// from inside that frame whatever its correction does there, so each
/// frame looks only at the band of points within its own reach of the edge.
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
    let frames: Vec<usize> = (0..reach.len()).collect();
    let chunk = frames.len().div_ceil(workers() * 4).max(1);
    let pieces: Vec<&[usize]> = frames.chunks(chunk).collect();
    let parts = map_in_order(&pieces, parallel, |frames| {
        let mut ok = vec![true; band.len()];
        for &t in frames.iter() {
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
    });
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
    let (y, x) = (
        y.round().clamp(0.0, (ly - 1) as f64),
        x.round().clamp(0.0, (lx - 1) as f64),
    );
    covered[y as usize * lx + x as usize]
}

/// A rectangle of a mask's cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub row: usize,
    pub col: usize,
    pub height: usize,
    pub width: usize,
}

/// The largest rectangle of `true` in a `rows * cols` row-major mask; the
/// first one found, scanning down, when several tie.
///
/// Row by row, each column's run of `true` ending at that row is a bar, and the
/// largest rectangle resting on that row is the largest rectangle under those
/// bars — found in one pass with a stack of bars still growing. Every row in
/// turn, so `O(rows * cols)` in all.
pub fn largest_rectangle(mask: &[bool], rows: usize, cols: usize) -> Option<Rect> {
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

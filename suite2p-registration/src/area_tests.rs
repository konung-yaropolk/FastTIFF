//! The common area against slow ways of finding it, and cutting to it against
//! a picture whose every point is known.

use super::*;
use crate::nonrigid::{make_blocks, warp};
use crate::rigid::shift_frame_bordered;
use crate::Border;

/// A small deterministic generator, so a failure names its own inputs.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// Uniform in `-r ..= r`.
    fn within(&mut self, r: i64) -> i64 {
        (self.next() % (2 * r as u64 + 1)) as i64 - r
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 10_000) as f32 / 10_000.0
    }
}

fn shift(dy: i32, dx: i32) -> Shift {
    Shift { dy, dx, corr: 1.0 }
}

/// Every rectangle, the slow way: the largest all-`true` one's size.
fn brute_largest(mask: &[bool], rows: usize, cols: usize) -> usize {
    let mut best = 0;
    for y0 in 0..rows {
        for x0 in 0..cols {
            for y1 in y0..rows {
                for x1 in x0..cols {
                    if (y0..=y1).all(|y| (x0..=x1).all(|x| mask[y * cols + x])) {
                        best = best.max((y1 - y0 + 1) * (x1 - x0 + 1));
                    }
                }
            }
        }
    }
    best
}

#[test]
fn the_largest_rectangle_is_the_largest() {
    let mut rng = Lcg(7);
    for case in 0..300 {
        let (rows, cols) = (1 + rng.next() as usize % 8, 1 + rng.next() as usize % 9);
        let mask: Vec<bool> = (0..rows * cols).map(|_| rng.next() % 5 != 0).collect();
        let want = brute_largest(&mask, rows, cols);
        match largest_rectangle(&mask, rows, cols) {
            None => assert_eq!(want, 0, "case {case}"),
            Some(r) => {
                assert_eq!(r.height * r.width, want, "case {case}: {r:?}");
                let all = (r.row..r.row + r.height)
                    .all(|y| (r.col..r.col + r.width).all(|x| mask[y * cols + x]));
                assert!(all, "case {case}: {r:?} is not all covered");
            }
        }
    }
    assert_eq!(largest_rectangle(&[false; 6], 2, 3), None);
}

/// Which pixels a frame of ones still covers once moved: the ones not filled.
fn never_filled(moved: &[Vec<f32>]) -> Vec<bool> {
    let mut all = vec![true; moved[0].len()];
    for frame in moved {
        for (a, &v) in all.iter_mut().zip(frame) {
            *a &= v != 0.0;
        }
    }
    all
}

/// Every pixel of `area` is inside the frame and read from inside it by every
/// frame's correction.
fn all_covered(area: &Area, ly: usize, lx: usize, moves: &Moves<'_>, frames: usize) -> bool {
    (0..area.height).all(|r| {
        (0..area.width).all(|c| {
            let (y, x) = area.at(r, c);
            on_frame(y, x, ly, lx)
                && (0..frames).all(|t| {
                    let (sy, sx) = moves.source(t, y, x);
                    on_frame(sy, sx, ly, lx)
                })
        })
    })
}

/// A rigid run's area is the intersection of rectangles: upright, and exactly
/// the pixels no frame's shift ever fills.
#[test]
fn a_rigid_area_is_what_no_shift_fills() {
    let mut rng = Lcg(11);
    for case in 0..50 {
        let (ly, lx) = (5 + rng.next() as usize % 20, 5 + rng.next() as usize % 20);
        let shifts: Vec<Shift> = (0..1 + rng.next() % 6)
            .map(|_| shift(rng.within(4) as i32, rng.within(4) as i32))
            .collect();
        let ones = vec![1.0f32; ly * lx];
        let moved: Vec<Vec<f32>> = shifts
            .iter()
            .map(|s| shift_frame_bordered(&ones, ly, lx, s.dy, s.dx, Border::Constant))
            .collect();
        let count = never_filled(&moved).iter().filter(|&&c| c).count();
        match common_area(ly, lx, &shifts, None, 0, true) {
            Err(e) => {
                assert_eq!(count, 0, "case {case}");
                assert_eq!(e.frames, shifts.len());
            }
            Ok(a) => {
                assert!(a.is_upright(), "case {case}: {a:?}");
                assert_eq!(a.height * a.width, count, "case {case}: {a:?}");
                assert!(all_covered(&a, ly, lx, &Moves::new(&shifts, None, 0, lx), shifts.len()));
            }
        }
    }
}

/// A warped run's area holds only covered pixels, and when it is upright it is
/// the largest rectangle of what no warp fills.
#[test]
fn a_warped_area_is_covered_and_largest() {
    let mut rng = Lcg(23);
    for case in 0..25 {
        let (ly, lx) = (24 + rng.next() as usize % 40, 24 + rng.next() as usize % 40);
        let blocks = make_blocks(ly, lx, [16, 16], 10);
        let frames = 1 + rng.next() as usize % 5;
        let shifts: Vec<Shift> = (0..frames)
            .map(|_| shift(rng.within(3) as i32, rng.within(3) as i32))
            .collect();
        let fields: Vec<Vec<BlockShift>> = (0..frames)
            .map(|_| {
                (0..blocks.blocks.len())
                    .map(|_| BlockShift {
                        dy: (rng.unit() - 0.5) * 6.0,
                        dx: (rng.unit() - 0.5) * 6.0,
                        corr: 1.0,
                    })
                    .collect()
            })
            .collect();
        let ones = vec![1.0f32; ly * lx];
        let moved: Vec<Vec<f32>> = (0..frames)
            .map(|t| warp(&ones, ly, lx, &blocks, &fields[t], shifts[t], Border::Constant))
            .collect();
        let covered = never_filled(&moved);
        // The fast search, checked against every rectangle above.
        let upright = largest_rectangle(&covered, ly, lx).map_or(0, |r| r.height * r.width);

        let found = common_area(ly, lx, &shifts, Some((&blocks, &fields)), 0, true);
        let moves = Moves::new(&shifts, Some((&blocks, &fields)), 0, lx);
        match found {
            Err(_) => assert_eq!(upright, 0, "case {case}"),
            Ok(a) => {
                assert!(all_covered(&a, ly, lx, &moves, frames), "case {case}: {a:?}");
                if a.is_upright() {
                    assert_eq!(a.height * a.width, upright, "case {case}: {a:?}");
                } else {
                    assert!(a.height * a.width > upright, "case {case}: {a:?} vs {upright}");
                }
            }
        }
        // Single- and multi-threaded give the same area.
        let one = common_area(ly, lx, &shifts, Some((&blocks, &fields)), 0, false);
        assert_eq!(one.ok(), found.ok(), "case {case}");
    }
}

/// A covered region that is a bar at 30 degrees: an upright rectangle fits
/// only a sliver of it, and the search finds the bar's angle and nearly all
/// of its length and width.
#[test]
fn a_turned_region_is_cut_at_its_own_angle() {
    let (ly, lx) = (100usize, 100usize);
    let (sin, cos) = 30f64.to_radians().sin_cos();
    // Frame 0 reads its pixel from along and across the bar, stretched so
    // that only the bar — about 82 by 16 pixels — lands on the frame.
    let source = |_: usize, y: f64, x: f64| {
        let (dy, dx) = (y - 49.0, x - 49.0);
        let along = dx * cos + dy * sin;
        let across = -dx * sin + dy * cos;
        (49.5 + across * 6.0, 49.5 + along * 1.2)
    };
    let reads = |t: usize, y: f64, x: f64| {
        let (sy, sx) = source(t, y, x);
        on_frame(sy, sx, ly, lx)
    };
    let area = largest_turned(ly, lx, &[f64::INFINITY], &reads, true).expect("the bar");
    assert!(
        (area.angle.to_degrees() - 30.0).abs() < 0.5,
        "found {} degrees",
        area.angle.to_degrees()
    );
    assert!(area.height * area.width > 82 * 16 * 3 / 4, "{area:?}");
    // And every pixel of it is on the bar.
    for r in 0..area.height {
        for c in 0..area.width {
            let (y, x) = area.at(r, c);
            let (sy, sx) = source(0, y, x);
            assert!(on_frame(y, x, ly, lx) && on_frame(sy, sx, ly, lx), "({r}, {c})");
        }
    }
}

/// Frames that moved further apart than the frame is wide share nothing, and
/// say so: which frame size, and how many frames.
#[test]
fn frames_that_share_nothing_have_no_area() {
    let err = common_area(10, 10, &[shift(6, 0), shift(-6, 0)], None, 0, true).expect_err("none");
    assert_eq!(
        err,
        NoCommonArea {
            ly: 10,
            lx: 10,
            frames: 2
        }
    );
    assert!(err.to_string().contains("no common area"), "{err}");
    assert!(common_area(0, 5, &[], None, 0, true).is_err());
}

/// A cut reads each result pixel from where its frame's correction says: on a
/// ramp, which bilinear reading reproduces exactly, every value is known.
#[test]
fn a_cut_reads_where_the_correction_says() {
    let (ly, lx) = (48, 64);
    let ramp: Vec<f32> = (0..ly * lx)
        .map(|i| (3 * (i / lx) + 2 * (i % lx)) as f32)
        .collect();
    let blocks = make_blocks(ly, lx, [16, 16], 10);
    // A uniform field, so where a point is read from is plain arithmetic.
    let field = vec![
        vec![
            BlockShift {
                dy: 1.25,
                dx: -0.5,
                corr: 1.0
            };
            blocks.blocks.len()
        ];
        1
    ];
    let shifts = [shift(2, -3)];
    let area = Area {
        y: 10.0,
        x: 20.0,
        height: 12,
        width: 15,
        angle: 20f64.to_radians(),
    };
    let cut = Cut::new(ly, lx, area, &shifts, Some((&blocks, &field)));
    let out = cut.apply(&ramp, 0);
    assert_eq!(out.len(), 12 * 15);
    for r in 0..area.height {
        for c in 0..area.width {
            let (y, x) = area.at(r, c);
            let want = 3.0 * (y + 1.25 + 2.0) + 2.0 * (x - 0.5 - 3.0);
            let got = out[r * area.width + c] as f64;
            assert!((got - want).abs() < 1e-2, "({r}, {c}): {got} vs {want}");
        }
    }
    // And upright on a rigid shift, the frame's own samples, exactly.
    let upright = Area {
        y: 3.0,
        x: 5.0,
        height: 4,
        width: 6,
        angle: 0.0,
    };
    let rigid = Cut::new(ly, lx, upright, &shifts, None).apply(&ramp, 0);
    for r in 0..4 {
        for c in 0..6 {
            let (sy, sx) = (3 + r + 2, 5 + c - 3);
            assert_eq!(rigid[r * 6 + c], ramp[sy * lx + sx], "({r}, {c})");
        }
    }
    assert_eq!(cut.apply_batch(std::slice::from_ref(&ramp), &[0], true), vec![out]);
}

/// The bidirectional-phase correction slides alternate lines and leaves the
/// columns it slid away from unmeasured; a crop keeps clear of them, on
/// whichever side they are.
#[test]
fn a_crop_keeps_clear_of_the_bidiphase_strip() {
    let (ly, lx) = (40, 50);
    let still = [shift(0, 0), shift(0, 0)];
    for (bidi, x0, width) in [(3, 3.0, 47), (-4, 0.0, 46), (0, 0.0, 50), (60, 0.0, 50)] {
        let a = common_area(ly, lx, &still, None, bidi, true).expect("an area");
        assert_eq!((a.x, a.width, a.height), (x0, width, ly), "bidiphase {bidi}");
    }
    // A warped run keeps clear of it too.
    let blocks = make_blocks(ly, lx, [16, 16], 10);
    let fields = vec![
        vec![
            BlockShift {
                dy: 0.0,
                dx: 0.0,
                corr: 1.0
            };
            blocks.blocks.len()
        ];
        2
    ];
    let a = common_area(ly, lx, &still, Some((&blocks, &fields)), 3, true).expect("an area");
    assert!(a.is_upright() && a.x >= 3.0, "{a:?}");
    let moves = Moves::new(&still, Some((&blocks, &fields)), 3, lx);
    for r in 0..a.height {
        for c in 0..a.width {
            let (y, x) = a.at(r, c);
            assert!((0..2).all(|t| moves.reads(t, y, x, ly, lx)), "({r}, {c})");
        }
    }
}

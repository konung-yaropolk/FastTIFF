//! The block grid, the kriging refinement, and the warp.

use super::*;

#[test]
fn a_block_at_least_as_big_as_the_frame_is_one_block() {
    assert_eq!(calculate_nblocks(128, 128), (128, 1));
    assert_eq!(calculate_nblocks(128, 256), (128, 1));
}

/// Blocks overlap rather than tile: `ceil(1.5 * L / block)` of them, which is a
/// third more than would be needed to cover the frame edge to edge.
#[test]
fn blocks_overlap_by_design() {
    let (size, n) = calculate_nblocks(512, 64);
    assert_eq!(size, 64);
    assert_eq!(
        n, 12,
        "512/64 is 8 tiles; 1.5x that is 12 overlapping blocks"
    );
}

#[test]
fn the_grid_covers_the_frame_edge_to_edge() {
    let b = make_blocks(256, 256, [64, 64], 10);
    assert_eq!((b.ny, b.nx), (6, 6));
    assert_eq!(b.blocks.len(), 36);
    // First block starts at the origin, last ends flush with the far corner.
    assert_eq!((b.blocks[0].y0, b.blocks[0].x0), (0, 0));
    let last = b.blocks[b.blocks.len() - 1];
    assert_eq!((last.y1, last.x1), (256, 256));
    // Every block is the requested size.
    for blk in &b.blocks {
        assert_eq!((blk.height(), blk.width()), (64, 64));
    }
}

/// Neighbouring blocks really do overlap — the whole reason the interpolated
/// shift field has no crease at a block boundary.
#[test]
fn neighbouring_blocks_share_pixels() {
    let b = make_blocks(256, 256, [64, 64], 10);
    let first = b.blocks[0];
    let second = b.blocks[1];
    assert!(
        second.x0 < first.x1,
        "blocks {first:?} and {second:?} do not overlap"
    );
}

/// The smoothing matrix is row-stochastic, so borrowing from neighbours cannot
/// change a correlation map's overall scale — only where its mass sits.
#[test]
fn the_block_smoother_preserves_scale() {
    let b = make_blocks(256, 256, [64, 64], 10);
    let nb = b.blocks.len();
    for i in 0..nb {
        let row: f32 = (0..nb).map(|j| b.smoother[i * nb + j]).sum();
        assert!((row - 1.0).abs() < 1e-4, "row {i} sums to {row}");
    }
}

/// A block is most influenced by itself, then by its immediate neighbours.
#[test]
fn the_block_smoother_favours_the_block_itself() {
    let b = make_blocks(256, 256, [64, 64], 10);
    let nb = b.blocks.len();
    let mid = nb / 2;
    let own = b.smoother[mid * nb + mid];
    for j in 0..nb {
        if j != mid {
            assert!(
                b.smoother[mid * nb + j] <= own,
                "block {j} outweighed itself"
            );
        }
    }
}

// --------------------------------------------------------- the warp

fn ramp(ly: usize, lx: usize) -> Vec<f32> {
    (0..ly * lx).map(|i| (i % lx) as f32).collect()
}

/// A field of zero block shifts and no rigid shift leaves the frame alone.
#[test]
fn a_zero_field_is_the_identity() {
    let (ly, lx) = (64, 64);
    let f = ramp(ly, lx);
    let b = make_blocks(ly, lx, [32, 32], 10);
    let zero = vec![
        BlockShift {
            dy: 0.0,
            dx: 0.0,
            corr: 1.0
        };
        b.blocks.len()
    ];
    let out = warp(
        &f,
        ly,
        lx,
        &b,
        &zero,
        Shift {
            dy: 0,
            dx: 0,
            corr: 1.0,
        },
    );
    for (a, c) in out.iter().zip(&f) {
        assert!((a - c).abs() < 1e-4, "{a} vs {c}");
    }
}

/// A uniform field is a plain translation — the non-rigid warp must agree with
/// the rigid one when every block says the same thing.
#[test]
fn a_uniform_field_is_a_translation() {
    let (ly, lx) = (64, 64);
    let f = ramp(ly, lx);
    let b = make_blocks(ly, lx, [32, 32], 10);
    let uniform = vec![
        BlockShift {
            dy: 0.0,
            dx: 2.0,
            corr: 1.0
        };
        b.blocks.len()
    ];
    let out = warp(
        &f,
        ly,
        lx,
        &b,
        &uniform,
        Shift {
            dy: 0,
            dx: 0,
            corr: 1.0,
        },
    );
    // The ramp is the column index, so sampling from x+2 reads 2 higher —
    // away from the right edge, where the clamp holds it.
    for y in 0..ly {
        for x in 0..lx - 4 {
            let got = out[y * lx + x];
            assert!((got - (x as f32 + 2.0)).abs() < 1e-3, "({y},{x}): {got}");
        }
    }
}

/// The edges clamp rather than wrap. A non-rigid warp moves each part of the
/// frame differently, so a wrap would bring the far side into the middle.
#[test]
fn the_warp_clamps_at_the_edge() {
    let (ly, lx) = (32, 32);
    let f = ramp(ly, lx);
    let b = make_blocks(ly, lx, [16, 16], 10);
    let push = vec![
        BlockShift {
            dy: 0.0,
            dx: -8.0,
            corr: 1.0
        };
        b.blocks.len()
    ];
    let out = warp(
        &f,
        ly,
        lx,
        &b,
        &push,
        Shift {
            dy: 0,
            dx: 0,
            corr: 1.0,
        },
    );
    // Sampling from x-8: the left edge clamps to column 0 rather than wrapping
    // round to column 24.
    assert!((out[0] - 0.0).abs() < 1e-3, "left edge wrapped: {}", out[0]);
    assert!(out[..8].iter().all(|&v| v < 1.0), "{:?}", &out[..8]);
}

/// The rigid shift is applied on top of the block field, not instead of it.
#[test]
fn the_rigid_shift_is_added_to_the_field() {
    let (ly, lx) = (64, 64);
    let f = ramp(ly, lx);
    let b = make_blocks(ly, lx, [32, 32], 10);
    let field = vec![
        BlockShift {
            dy: 0.0,
            dx: 1.0,
            corr: 1.0
        };
        b.blocks.len()
    ];
    let out = warp(
        &f,
        ly,
        lx,
        &b,
        &field,
        Shift {
            dy: 0,
            dx: 3,
            corr: 1.0,
        },
    );
    for (x, &got) in out.iter().enumerate().take(lx - 8) {
        assert!((got - (x as f32 + 4.0)).abs() < 1e-3, "x {x}: {got}");
    }
}

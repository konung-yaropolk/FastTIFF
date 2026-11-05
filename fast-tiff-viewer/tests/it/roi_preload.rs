//! Planning under [`LargeImageMode::Preload`]: the two-level scheme.
//!
//! Its whole promise is that there are exactly *two* resolutions — one coarse
//! copy of the frame, kept, and a full-resolution crop — so that zooming
//! crosses a single boundary and the coarse level is decoded once. The tests
//! here pin that promise down, plus the two invariants it shares with the
//! continuous scheme: whatever is planned must fit the device and must cover
//! what is on screen.

use fast_tiff_viewer::roi::LargeImageMode::{Preload, Tiled};
use fast_tiff_viewer::roi::{overview_stride, plan, Budget, Residency, MAX_PRELOAD_BYTES};

/// The file this feature exists for, and a common device limit.
const ANDROMEDA: (u32, u32) = (40_000, 12_788);
const MAX_AXIS: u32 = 16_384;
/// Three 8-bit channels, the Andromeda case.
const RGB8: usize = 3;
/// An ordinary panel, so the display bound is the one that bites.
const PANEL: [f32; 2] = [1920.0, 1080.0];
const WHOLE: ([f32; 2], [f32; 2]) = ([0.0, 0.0], [1.0, 1.0]);

fn preload(w: u32, h: u32, off: [f32; 2], scale: [f32; 2], panel: [f32; 2]) -> Residency {
    plan(
        w,
        h,
        off,
        scale,
        panel,
        Budget::new(MAX_AXIS, RGB8, Preload),
    )
}

/// A view of `px` source pixels wide, centred.
fn view_of(w: u32, h: u32, px: f32) -> ([f32; 2], [f32; 2]) {
    let sx = px / w as f32;
    let sy = (px * 0.5) / h as f32;
    ([0.5 - sx / 2.0, 0.5 - sy / 2.0], [sx, sy])
}

#[test]
fn the_coarse_level_is_the_finest_sampling_that_fits() {
    // Half scale wherever half scale is possible: point-sampling every second
    // pixel aliases far less than every eighth, and that is the entire reason
    // this mode exists.
    assert_eq!(overview_stride(20_000, 20_000, MAX_AXIS, RGB8), 2);
    // Andromeda is 40000 wide, so half scale still overruns a 16384 axis.
    assert_eq!(overview_stride(ANDROMEDA.0, ANDROMEDA.1, MAX_AXIS, RGB8), 4);
    // A bigger device limit lets the same frame keep more detail.
    assert_eq!(overview_stride(ANDROMEDA.0, ANDROMEDA.1, 32_768, RGB8), 2);
    // Always a power of two.
    for w in [17_000u32, 40_000, 100_000, 400_000] {
        let s = overview_stride(w, w, MAX_AXIS, RGB8);
        assert!(s.is_power_of_two(), "{w} gave stride {s}");
    }
}

#[test]
fn the_coarse_level_respects_both_budgets() {
    for &(w, h) in &[(20_000u32, 20_000u32), ANDROMEDA, (60_000, 60_000)] {
        for bpt in [1usize, 3, 8, 24] {
            let s = overview_stride(w, h, MAX_AXIS, bpt);
            let (tw, th) = (w.div_ceil(s), h.div_ceil(s));
            assert!(
                tw <= MAX_AXIS && th <= MAX_AXIS,
                "{w}x{h} bpt {bpt}: {tw}x{th} axis"
            );
            assert!(
                (tw as usize) * (th as usize) * bpt <= MAX_PRELOAD_BYTES,
                "{w}x{h} bpt {bpt}: over the RAM budget"
            );
        }
    }
}

#[test]
fn zoomed_out_the_whole_frame_is_resident_at_the_coarse_level() {
    let (w, h) = ANDROMEDA;
    let r = preload(w, h, WHOLE.0, WHOLE.1, PANEL).resident;
    // Spanning the frame is what lets it serve every view without re-cutting.
    assert_eq!((r.x, r.y), (0, 0));
    assert!(
        r.w >= w && r.h >= h,
        "coarse level does not span the frame: {r:?}"
    );
    assert_eq!(r.stride, overview_stride(w, h, MAX_AXIS, RGB8));
}

#[test]
fn zoomed_in_it_is_the_files_own_resolution() {
    let (w, h) = ANDROMEDA;
    let (off, scale) = view_of(w, h, 1500.0);
    let r = preload(w, h, off, scale, PANEL).resident;
    assert_eq!(r.stride, 1, "a 1500px view on a 1920px panel should be 1:1");
    assert!(
        r.w < w,
        "a full-resolution view should be a crop, not the frame"
    );
}

#[test]
fn nothing_between_the_two_levels_is_ever_planned() {
    // The promise. Sweeping the zoom from fit-to-window down to a few hundred
    // pixels, every plan is either the coarse level or 1:1 — never a 1/6 or a
    // 1/8 that would have to be decoded on the way past.
    let (w, h) = ANDROMEDA;
    let coarse = overview_stride(w, h, MAX_AXIS, RGB8);
    let mut seen = std::collections::BTreeSet::new();
    let mut px = w as f32;
    while px > 200.0 {
        let (off, scale) = view_of(w, h, px);
        seen.insert(preload(w, h, off, scale, PANEL).resident.stride);
        px *= 0.9;
    }
    let levels: Vec<u32> = seen.iter().copied().collect();
    assert!(
        levels.iter().all(|&s| s == 1 || s == coarse),
        "planned {levels:?}, expected only 1 and {coarse}"
    );
    assert!(levels.len() <= 2, "more than two levels: {levels:?}");
}

#[test]
fn the_continuous_scheme_really_does_use_more_levels() {
    // Guards the comparison the mode is sold on: if `Tiled` also produced two
    // levels here, this whole feature would be a no-op and the test above
    // would be passing for the wrong reason.
    let (w, h) = ANDROMEDA;
    let mut seen = std::collections::BTreeSet::new();
    let mut px = w as f32;
    while px > 200.0 {
        let (off, scale) = view_of(w, h, px);
        seen.insert(
            plan(w, h, off, scale, PANEL, Budget::new(MAX_AXIS, RGB8, Tiled))
                .resident
                .stride,
        );
        px *= 0.9;
    }
    assert!(seen.len() > 2, "Tiled produced only {:?}", seen);
}

#[test]
fn every_planned_window_fits_the_device_and_covers_the_view() {
    // The two properties that are fatal to get wrong: a texture the device
    // rejects takes the process with it, and a window short of the view shows
    // the edge of the data.
    let (w, h) = ANDROMEDA;
    for panel in [[800.0, 600.0], PANEL, [3840.0, 2160.0]] {
        let mut px = w as f32;
        while px > 100.0 {
            for cx in [0.0f32, 0.3, 0.5, 0.97] {
                let sx = (px / w as f32).min(1.0);
                let sy = ((px * 0.5) / h as f32).min(1.0);
                let off = [(cx).min(1.0 - sx).max(0.0), 0.5 - (sy / 2.0).min(0.5)];
                let scale = [sx, sy];
                let p = preload(w, h, off, scale, panel);
                let (tw, th) = p.resident.texture_size();
                assert!(
                    tw <= MAX_AXIS && th <= MAX_AXIS,
                    "{tw}x{th} at px {px} panel {panel:?}"
                );

                // Covers the visible region, in source pixels.
                let (vx0, vy0) = (off[0] * w as f32, off[1] * h as f32);
                let (vx1, vy1) = (vx0 + sx * w as f32, vy0 + sy * h as f32);
                let r = p.resident;
                assert!(
                    (r.x as f32) <= vx0 + 1.0
                        && (r.y as f32) <= vy0 + 1.0
                        && (r.x + r.w) as f32 >= vx1 - 1.0
                        && (r.y + r.h) as f32 >= vy1 - 1.0,
                    "{r:?} misses view {vx0}..{vx1} x {vy0}..{vy1}"
                );
            }
            px *= 0.8;
        }
    }
}

#[test]
fn a_frame_that_fits_is_untouched_by_the_mode() {
    // Neither mode should do anything at all to an ordinary image.
    for mode in [Preload, Tiled] {
        let r = plan(
            2048,
            1024,
            WHOLE.0,
            WHOLE.1,
            PANEL,
            Budget::new(MAX_AXIS, RGB8, mode),
        )
        .resident;
        assert_eq!(r.stride, 1);
        assert_eq!((r.x, r.y, r.w, r.h), (0, 0, 2048, 1024));
    }
}

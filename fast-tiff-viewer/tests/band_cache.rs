//! The decoded-band cache, and the grid it depends on.
//!
//! A strip TIFF compresses each strip across the whole image width, so showing
//! a narrow window of a wide mosaic decodes roughly ten times what it displays.
//! That cannot be avoided — it is how the file is stored. What can be avoided is
//! paying it again for rows that were already decompressed a moment ago.
//!
//! Everything rests on one property: bands sit on a grid fixed to the *frame*,
//! not to the window. If they moved with the window, every pan would produce
//! keys nothing had ever been stored under, the cache would return nothing, and
//! all of this would be memory spent to no effect. The first tests here are
//! about that, not about the LRU.

use fast_tiff_viewer::bandcache::{band_range, band_rows, bands_covering, BandCache, MAX_CACHE_BYTES};
use fast_tiff_viewer::prefetch::Decoded;

/// The mosaic this exists for: 40000 wide, three 8-bit channels.
const WIDE: u32 = 40_000;
const RGB8: usize = 3;

#[test]
fn the_grid_does_not_move_with_the_window() {
    let rpb = band_rows(WIDE, RGB8);
    // Two windows at different heights must agree about where bands begin.
    let a = bands_covering(5_000..7_200, rpb);
    let b = bands_covering(5_100..7_300, rpb);
    assert!(a.start <= b.start && a.end <= b.end, "the grid should shift by whole bands, not by rows");
    // Every band index maps to the same rows regardless of who asked.
    for i in [a.start, b.start, a.end - 1] {
        assert_eq!(band_range(i, rpb, 12_788), band_range(i, rpb, 12_788));
    }
}

/// A sideways pan does not change which rows are on screen, so it must reuse
/// every band. This is the case the cache exists for.
#[test]
fn panning_sideways_needs_exactly_the_same_bands() {
    let rpb = band_rows(WIDE, RGB8);
    let before = bands_covering(5_000..7_200, rpb);
    let after = bands_covering(5_000..7_200, rpb); // same rows, different x
    assert_eq!(before, after);
}

/// A vertical pan should reuse most of them — that is the difference between a
/// grid and re-cutting per window.
#[test]
fn panning_vertically_reuses_all_but_the_newly_exposed_edge() {
    let rpb = band_rows(WIDE, RGB8);
    let before = bands_covering(5_000..7_200, rpb);
    // Move down by well under a band.
    let after = bands_covering(5_000 + rpb / 4..7_200 + rpb / 4, rpb);
    let overlap = before.clone().filter(|b| after.contains(b)).count();
    let wanted = after.clone().count();
    assert!(
        overlap * 2 >= wanted,
        "only {overlap} of {wanted} bands reused across a quarter-band pan"
    );
}

/// Zoom changes the stride, which is applied when a band is *sampled*, not when
/// it is decoded — so the same bands serve both zoom levels.
#[test]
fn zooming_over_one_spot_reuses_the_same_bands() {
    let rpb = band_rows(WIDE, RGB8);
    // The same rows viewed at two zoom levels ask for the same grid slots.
    assert_eq!(bands_covering(6_000..6_800, rpb), bands_covering(6_000..6_800, rpb));
}

/// The bands named for a row range must actually cover that range, and none of
/// them may be wasted. This is what ties `bands_covering` to `band_range`: on
/// their own each looks reasonable, and it is only together that they mean
/// "decode these slots and you will have those rows". Get it wrong and the
/// window is assembled out of whatever the grid happened to point at.
#[test]
fn the_bands_named_are_exactly_the_bands_needed() {
    let rpb = band_rows(WIDE, RGB8);
    let h = 12_788;
    for (start, end) in [(0, 1), (1, 2), (5_000, 7_200), (12_000, h), (rpb - 1, rpb + 1)] {
        let bands = bands_covering(start..end, rpb);
        let case = format!("rows {start}..{end} -> bands {bands:?} (band = {rpb} rows)");

        let covered = band_range(bands.start, rpb, h).start..band_range(bands.end - 1, rpb, h).end;
        assert!(covered.start <= start, "leaves the top uncovered: {case}");
        assert!(covered.end >= end.min(h), "leaves the bottom uncovered: {case}");

        // No band on either end that the range does not actually reach.
        assert!(band_range(bands.start, rpb, h).end > start, "first band is wasted: {case}");
        assert!(band_range(bands.end - 1, rpb, h).start < end.max(start + 1), "last band is wasted: {case}");
    }
}

#[test]
fn bands_are_a_sensible_size_for_the_frame() {
    let rpb = band_rows(WIDE, RGB8);
    assert!(rpb.is_power_of_two(), "a power of two keeps the grid stable, got {rpb}");
    let bytes = rpb as usize * WIDE as usize * RGB8;
    assert!(bytes <= MAX_CACHE_BYTES, "a single band must fit the cache, got {bytes} bytes");
    assert!(rpb >= 1);

    // A narrow frame gets more rows per band; a very wide one fewer. Either way
    // the band stays roughly one size in bytes, which is what bounds memory.
    let narrow = band_rows(64, RGB8);
    assert!(narrow >= rpb, "a narrow frame should afford more rows per band");
}

#[test]
fn bands_covering_never_returns_an_empty_range() {
    let rpb = band_rows(WIDE, RGB8);
    for rows in [0..0, 0..1, 5..5, 12_788..12_788] {
        let bands = bands_covering(rows.clone(), rpb);
        assert!(bands.start < bands.end, "{rows:?} produced no bands");
    }
}

#[test]
fn band_ranges_are_clamped_to_the_frame() {
    let rpb = 512;
    let h = 1_000;
    assert_eq!(band_range(0, rpb, h), 0..512);
    assert_eq!(band_range(1, rpb, h), 512..1_000, "the last band stops at the frame");
    assert_eq!(band_range(9, rpb, h), 1_000..1_000, "past the end is empty, not out of bounds");
}

// ---------------------------------------------------------------------------
// The cache itself
// ---------------------------------------------------------------------------

fn plane(n: usize) -> Vec<Decoded> {
    vec![Decoded::U8(vec![7; n])]
}

/// The full width — what a stripped frame always decodes, since a strip cannot
/// be split. Most of these tests are about that case.
const ALL: (u32, u32) = (0, WIDE);

#[test]
fn a_stored_band_comes_back() {
    let mut c = BandCache::default();
    c.put(0, vec![0], 3, ALL, plane(16));
    let got = c.get(0, &[0], 3, ALL).expect("just stored");
    assert!(matches!(&got[0], Decoded::U8(v) if v.len() == 16 && v[0] == 7));
}

/// The key has to include everything that would make the pixels wrong to reuse.
#[test]
fn a_band_is_not_confused_with_a_different_frame_or_channel_set() {
    let mut c = BandCache::default();
    c.put(0, vec![0, 1], 3, ALL, plane(16));
    assert!(c.get(1, &[0, 1], 3, ALL).is_none(), "a different frame is different pixels");
    assert!(c.get(0, &[0], 3, ALL).is_none(), "a different channel set was not decoded");
    assert!(c.get(0, &[0, 1], 4, ALL).is_none(), "a different band is different rows");
    assert!(c.get(0, &[0, 1], 3, ALL).is_some(), "...and the right key still hits");
}

/// A tiled frame narrows the columns, so the same rows over different tile
/// columns are different pixels and must not be confused. On a stripped frame
/// the columns are always the full width, which is what keeps a sideways pan
/// hitting — so the key has to distinguish them without breaking that.
#[test]
fn bands_over_different_columns_are_distinct() {
    let mut c = BandCache::default();
    c.put(0, vec![0], 3, (0, 4_096), plane(16));
    c.put(0, vec![0], 3, (4_096, 8_192), plane(16));
    assert_eq!(c.len(), 2, "different tile columns are different data");
    assert!(c.get(0, &[0], 3, (0, 4_096)).is_some());
    assert!(c.get(0, &[0], 3, (4_096, 8_192)).is_some());
    assert!(c.get(0, &[0], 3, (2_048, 6_144)).is_none(), "a column range never stored");

    // And the stripped case, where every window decodes the full width, still
    // collapses to one entry that a sideways pan re-uses.
    let mut c = BandCache::default();
    c.put(0, vec![0], 3, ALL, plane(16));
    assert!(c.get(0, &[0], 3, ALL).is_some(), "same rows, full width: a hit whatever x was");
}

#[test]
fn storing_the_same_band_twice_does_not_double_count() {
    let mut c = BandCache::default();
    c.put(0, vec![0], 1, ALL, plane(1024));
    let once = c.bytes();
    c.put(0, vec![0], 1, ALL, plane(1024));
    assert_eq!(c.bytes(), once, "re-storing a band should replace it, not stack up");
    assert_eq!(c.len(), 1);
}

/// Eviction is least-recently-*used*, not least-recently-stored: a band being
/// looked at every frame must not be thrown away for one fetched once.
#[test]
fn the_band_being_looked_at_survives_eviction() {
    let mut c = BandCache::default();
    let big = MAX_CACHE_BYTES / 3 + 1; // three will not fit
    c.put(0, vec![0], 0, ALL, plane(big));
    c.put(0, vec![0], 1, ALL, plane(big));
    assert!(c.get(0, &[0], 0, ALL).is_some(), "band 0 is now the most recently used");
    c.put(0, vec![0], 2, ALL, plane(big));

    assert!(c.get(0, &[0], 0, ALL).is_some(), "the recently used band should have survived");
    assert!(c.get(0, &[0], 1, ALL).is_none(), "the stale one should have gone");
}

#[test]
fn the_cache_stays_within_its_budget() {
    let mut c = BandCache::default();
    let each = MAX_CACHE_BYTES / 8;
    for band in 0..40 {
        c.put(0, vec![0], band, ALL, plane(each));
        assert!(
            c.bytes() <= MAX_CACHE_BYTES,
            "over budget after {band} bands: {} bytes",
            c.bytes()
        );
    }
    assert!(!c.is_empty(), "evicting everything would defeat the purpose");
}

/// A band too large to ever fit is not held, rather than emptying the cache to
/// make room for something that cannot stay.
#[test]
fn an_oversized_band_is_declined_without_evicting_anything() {
    let mut c = BandCache::default();
    c.put(0, vec![0], 0, ALL, plane(1024));
    let before = c.bytes();
    c.put(0, vec![0], 1, ALL, plane(MAX_CACHE_BYTES + 1));
    assert_eq!(c.bytes(), before, "the oversized band should have been declined");
    assert!(c.get(0, &[0], 0, ALL).is_some(), "and the existing one kept");
    assert!(c.get(0, &[0], 1, ALL).is_none());
}

#[test]
fn clearing_releases_everything() {
    let mut c = BandCache::default();
    c.put(0, vec![0], 0, ALL, plane(4096));
    c.clear();
    assert_eq!(c.bytes(), 0);
    assert!(c.is_empty());
    assert!(c.get(0, &[0], 0, ALL).is_none());
}

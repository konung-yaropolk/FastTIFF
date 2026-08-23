//! Decoded strip bands, kept so that moving the view re-uses what has already
//! been decompressed.
//!
//! A strip-based TIFF compresses each strip across the **full image width**, so
//! reading any pixel of a strip costs the whole strip. Showing a 3840-pixel-wide
//! window of a 40000-pixel-wide mosaic therefore decodes about ten times what it
//! displays, and that is a property of the file, not of the reader — nothing
//! here can avoid it. (A *tiled* TIFF can, which is why tiles are worth reading
//! natively; see `fast_tiff_lib`.)
//!
//! What can be avoided is paying it *twice*. Pan sideways and the rows on screen
//! do not change at all, yet every strip covering them would be decompressed
//! again. So bands are cached.
//!
//! The one design decision that matters is that bands sit on a **fixed grid** of
//! source rows, not on the window. A band aligned to the window would move with
//! it, so every pan would miss and the cache would never return anything. On a
//! fixed grid a sideways pan hits every band, and a vertical pan hits all but
//! the newly exposed edge. The grid is also independent of zoom — the stride is
//! applied when a band is *sampled*, not when it is decoded — so zooming in and
//! out over one spot re-uses the same bands too.

use crate::prefetch::Decoded;
use std::ops::Range;

/// Bytes of decoded band worth keeping.
///
/// Sized to hold a screen-height window at full resolution plus a little for
/// panning, on the kind of frame this exists for. Larger would buy reuse over
/// longer journeys across the image at a memory cost that is hard to justify
/// for a viewer; smaller and a single window would not fit, which is the one
/// case that has to hit.
pub const MAX_CACHE_BYTES: usize = 384 << 20;

/// Bytes the interface thread's cache keeps once a background worker has taken
/// over window decoding — enough for one band, so a cold build is not made
/// worse, but not a second full cache standing idle.
pub const IDLE_CACHE_BYTES: usize = TARGET_BAND_BYTES;

/// Bytes a single band should come to, before rounding to a power-of-two row
/// count. Small enough that the grid is fine-grained for vertical panning,
/// large enough that per-band overhead stays lost against the decode.
const TARGET_BAND_BYTES: usize = 32 << 20;

/// Rows per band for a frame `width` wide costing `bytes_per_row_sample` per
/// sample across all resident channels.
///
/// A power of two, and derived only from things that are fixed for the open
/// file, so the grid never shifts underfoot — which is the whole basis of the
/// cache returning anything.
pub fn band_rows(width: u32, bytes_per_row_sample: usize) -> u32 {
    let per_row = (width as usize).saturating_mul(bytes_per_row_sample).max(1);
    let rows = (TARGET_BAND_BYTES / per_row).clamp(1, 4096);
    // Round down to a power of two: any two frames of the same file then agree
    // on where band boundaries fall.
    1u32 << (u32::BITS - 1 - (rows as u32).leading_zeros())
}

/// The rows band `index` covers, clamped to the frame.
pub fn band_range(index: u32, rows_per_band: u32, height: u32) -> Range<u32> {
    let start = index.saturating_mul(rows_per_band).min(height);
    let end = start.saturating_add(rows_per_band).min(height);
    start..end
}

/// The bands a row range touches.
pub fn bands_covering(rows: Range<u32>, rows_per_band: u32) -> Range<u32> {
    let rpb = rows_per_band.max(1);
    let first = rows.start / rpb;
    let last = rows.end.max(rows.start + 1).div_ceil(rpb);
    first..last.max(first + 1)
}

struct Entry {
    frame: usize,
    channels: Vec<usize>,
    band: u32,
    cols: (u32, u32),
    planes: Vec<Decoded>,
    bytes: usize,
}

/// A least-recently-used cache of decoded bands.
///
/// Keyed by the *columns actually decoded* as well as the band, because a tiled
/// frame narrows them: two windows over the same rows but different tile
/// columns hold different pixels. On a stripped frame the columns are always
/// the full width, so the key collapses back to the band and a sideways pan
/// still hits — which is the case the cache was built for.
///
/// Deliberately a plain vector: it holds a handful of entries, a linear scan of
/// which is nothing next to decompressing one of them, and the ordering *is* the
/// recency information.
pub struct BandCache {
    entries: Vec<Entry>,
    bytes: usize,
    budget: usize,
}

impl Default for BandCache {
    fn default() -> Self {
        Self { entries: Vec::new(), bytes: 0, budget: MAX_CACHE_BYTES }
    }
}

impl BandCache {
    /// The decoded planes of `band`, if they are still held.
    ///
    /// Takes `&mut self` because a hit is what makes an entry recent; a cache
    /// that did not record that would evict whatever is being looked at.
    pub fn get(
        &mut self,
        frame: usize,
        channels: &[usize],
        band: u32,
        cols: (u32, u32),
    ) -> Option<&[Decoded]> {
        let at = self.entries.iter().position(|e| {
            e.band == band && e.cols == cols && e.frame == frame && e.channels == channels
        })?;
        let entry = self.entries.remove(at);
        self.entries.push(entry);
        self.entries.last().map(|e| e.planes.as_slice())
    }

    /// Hold a decoded band, evicting the least recently used until the total is
    /// back under `MAX_CACHE_BYTES`.
    ///
    /// A band too large to fit on its own is simply not held rather than
    /// emptying the cache for something that cannot stay.
    pub fn put(
        &mut self,
        frame: usize,
        channels: Vec<usize>,
        band: u32,
        cols: (u32, u32),
        planes: Vec<Decoded>,
    ) {
        let bytes: usize = planes.iter().map(Decoded::bytes).sum();
        if bytes > self.budget {
            return;
        }
        self.entries.retain(|e| {
            !(e.band == band && e.cols == cols && e.frame == frame && e.channels == channels)
        });
        self.bytes = self.entries.iter().map(|e| e.bytes).sum();
        self.entries.push(Entry { frame, channels, band, cols, planes, bytes });
        self.bytes += bytes;
        self.evict();
    }

    /// Change how much this cache may hold, evicting down to the new figure.
    ///
    /// Window decoding moves to a background thread once one is running, and
    /// that thread caches its own bands. The copy on the interface thread then
    /// serves only the cold builds — a new file, a new frame, a new channel
    /// set — which are misses whatever it is holding. Rather than pay for two
    /// full caches so one of them can go unread, its budget is cut when the
    /// worker starts.
    pub fn set_budget(&mut self, bytes: usize) {
        self.budget = bytes;
        self.evict();
    }

    /// Drop least-recently-used bands until the total is back within budget.
    fn evict(&mut self) {
        while self.bytes > self.budget && !self.entries.is_empty() {
            let dropped = self.entries.remove(0);
            self.bytes -= dropped.bytes;
        }
    }

    /// Drop everything. Called when what the cache describes stops being true —
    /// a new file, or a re-interpretation of the current one.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    /// Bytes currently held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// How many bands are held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}


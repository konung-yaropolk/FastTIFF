//! Walks the full IFD chain of a TIFF file and builds a per-frame index.
//! Each IFD in the chain is treated as one "frame" (one plane: one channel
//! at one Z/T position), matching how ImageJ writes hyperstacks (one IFD
//! per plane, in `xyczt` order by default). This is a generic multi-page
//! TIFF walker, not an ImageJ-specific format — it doesn't assume anything
//! about how many IFDs there are or what writer produced them.

use crate::ifd::{self, ByteOrder, RawIfdEntry, TiffFlavor};
use crate::metadata::{self, StackMeta};
use anyhow::{anyhow, bail, Result};
#[cfg(feature = "mmap")]
use {memmap2::Mmap, std::fs::File, std::path::Path};

// `pub(crate)`: shared with the write side (`encode`), so reader and writer
// can never disagree on tag numbers.
pub(crate) const TAG_IMAGE_WIDTH: u16 = 256;
pub(crate) const TAG_IMAGE_LENGTH: u16 = 257;
pub(crate) const TAG_BITS_PER_SAMPLE: u16 = 258;
pub(crate) const TAG_COMPRESSION: u16 = 259;
pub(crate) const TAG_PHOTOMETRIC: u16 = 262;
pub(crate) const TAG_IMAGE_DESCRIPTION: u16 = 270;
pub(crate) const TAG_STRIP_OFFSETS: u16 = 273;
pub(crate) const TAG_X_RESOLUTION: u16 = 282;
pub(crate) const TAG_Y_RESOLUTION: u16 = 283;
pub(crate) const TAG_SAMPLES_PER_PIXEL: u16 = 277;
pub(crate) const TAG_ROWS_PER_STRIP: u16 = 278;
pub(crate) const TAG_STRIP_BYTE_COUNTS: u16 = 279;
pub(crate) const TAG_PREDICTOR: u16 = 317;
pub(crate) const TAG_PLANAR_CONFIG: u16 = 284;
pub(crate) const TAG_SAMPLE_FORMAT: u16 = 339;
/// ColorMap (tag 320): the palette for a PhotometricInterpretation=3 (indexed)
/// image — `3 * 2^BitsPerSample` SHORT values, all reds then greens then blues.
const TAG_COLOR_MAP: u16 = 320;
/// InkSet (tag 332): which inks a Separated (PhotometricInterpretation=5) image
/// was separated into. 1 = CMYK in that component order; 2 = anything else,
/// including hi-fi sets like CMYK+orange+green whose order is described only by
/// `InkNames`. Only `1` may be interpreted with the CMYK formula.
pub(crate) const TAG_INK_SET: u16 = 332;
/// TileWidth (322) / TileLength (323): the size of one tile in a tiled image.
/// Both must be multiples of 16 per TIFF6; readers that insist on it reject
/// files libtiff itself will happily open, so this only requires them non-zero.
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
/// TileOffsets (324) / TileByteCounts (325): the tile table, the exact
/// counterpart of StripOffsets/StripByteCounts. Tiles run left to right, top to
/// bottom, and for a planar image the whole grid repeats once per plane.
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;

/// Ceiling on how many frames one file may index.
///
/// A `FrameInfo` costs on the order of 150 bytes once its two strip vectors are
/// counted, and a file can declare far more planes than its own size suggests:
/// a chain of minimal IFDs, or an ImageJ contiguous stack whose frames are a
/// byte apiece, both turn a small file into a large `Vec<FrameInfo>`. This caps
/// that amplification at a few hundred megabytes in the worst case while
/// sitting orders of magnitude above any real stack — a million planes at even
/// a kilobyte each would already be a gigabyte of pixels.
const MAX_FRAMES: usize = 1 << 20;
// Tags 50838/50839 (IJMetadataByteCounts / IJMetadata) carry ImageJ's binary
// per-channel LUT/range block. The format is undocumented and best-effort to
// parse, so it's used only as a supplementary fallback for display info the
// `ImageDescription` (tag 270) didn't provide — see `metadata::imagej`.
const TAG_IJ_METADATA_BYTE_COUNTS: u16 = 50838;
const TAG_IJ_METADATA: u16 = 50839;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    UnsignedInt,
    SignedInt,
    Float,
}

// `non_exhaustive`: codecs have been added before (ZSTD) and may be again;
// downstream matches keep a wildcard arm so that's not a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    Lzw,
    PackBits,
    Deflate,
    /// ZSTD (tag value 50000; a libtiff/GDAL registered extension. The
    /// withdrawn experimental value 34926 is accepted on read too.)
    Zstd,
    Other(u16),
}

/// A frame's strip (or tile) offsets, or its byte counts.
///
/// Nearly every frame in a scientific stack is a single strip, and a
/// `Vec<u64>` holding one element means a heap allocation per frame that lives
/// as long as the stack does — two of them, offsets and counts. A
/// million-frame stack therefore keeps two million tiny allocations alive, and
/// building and freeing them costs more than parsing the directories they came
/// from: 0.117 us per frame against 0.012 for the inline form. Opening such a
/// stack is dominated by heap traffic, not by TIFF.
///
/// Derefs to `[u64]`, so every reader — `len`, indexing, `iter`, `par_iter` —
/// is unchanged.
#[derive(Clone, Debug, Default, Eq)]
pub enum Strips {
    /// No strips declared. Rejected later by `validate_frames`, but
    /// representable so parsing does not have to allocate to say "none".
    #[default]
    None,
    /// The overwhelmingly common case: one strip covering the whole frame.
    One(u64),
    /// A multi-strip or tiled frame.
    Many(Vec<u64>),
}

/// Equal when the *contents* are equal, whatever shape holds them.
///
/// The derived comparison would call `One(5)` and `Many(vec![5])` different,
/// which is a trap for a type whose whole point is that it has two spellings
/// for the same one-element list. Everything the crate builds goes through
/// `From`, which normalises — but `Many` is a public variant, so a caller can
/// spell it the other way, and `tests/row_bands.rs` really does `assert_eq!`
/// two of these against each other.
impl PartialEq for Strips {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Strips {
    /// Shorten to the first `n` entries, as `Vec::truncate` does.
    pub fn truncate(&mut self, n: usize) {
        if n >= self.len() {
            return;
        }
        *self = match self {
            Strips::Many(v) => {
                v.truncate(n);
                std::mem::take(v).into()
            }
            _ => Strips::None,
        };
    }
}

impl std::ops::Deref for Strips {
    type Target = [u64];
    #[inline]
    fn deref(&self) -> &[u64] {
        match self {
            Strips::None => &[],
            Strips::One(v) => std::slice::from_ref(v),
            Strips::Many(v) => v,
        }
    }
}

impl From<&[u64]> for Strips {
    #[inline]
    fn from(v: &[u64]) -> Self {
        match v {
            [] => Strips::None,
            [one] => Strips::One(*one),
            many => Strips::Many(many.to_vec()),
        }
    }
}

impl From<Vec<u64>> for Strips {
    #[inline]
    fn from(v: Vec<u64>) -> Self {
        match v.len() {
            0 => Strips::None,
            1 => Strips::One(v[0]),
            _ => Strips::Many(v),
        }
    }
}

impl FromIterator<u64> for Strips {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

/// Everything needed to locate and decode one plane (one IFD) in the file.
#[derive(Clone, Debug)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u16,
    pub samples_per_pixel: u16,
    pub sample_format: SampleFormat,
    pub compression: Compression,
    pub predictor: u16, // 1 = none, 2 = horizontal differencing
    /// PhotometricInterpretation (tag 262): 2 = RGB, 3 = palette,
    /// 5 = Separated (CMYK and other ink sets — see [`FrameInfo::ink_set`]),
    /// others treated as a single-plane grayscale/whatever. Used to decide
    /// whether a frame's multiple samples are color components to
    /// deinterleave, and how to interpret them once they are.
    pub photometric: u16,
    /// PlanarConfiguration (tag 284): 1 = chunky (samples interleaved per
    /// pixel, the default), 2 = planar (each sample stored as its own whole
    /// plane, one after another). Both are decoded; see `FrameInfo::is_planar`.
    pub planar_config: u16,
    /// Tile size `(width, length)` for a tiled image, `None` for a stripped one.
    ///
    /// Tiles and strips are the same idea — an independently compressed piece of
    /// the image — differing in shape, and the tile table is carried in
    /// [`strip_offsets`](Self::strip_offsets) / [`strip_byte_counts`](Self::strip_byte_counts)
    /// so that everything which only cares about "where are the pieces" needs no
    /// second code path.
    ///
    /// The difference that matters is *what a piece covers*. A strip spans the
    /// full image width, so reading any pixel of it costs the whole width; a
    /// tile is bounded on both axes, so a window of a huge image can be read
    /// without touching the rest of its rows. That is the entire reason to
    /// prefer tiles for large images, and why [`FrameInfo::crop`] can narrow
    /// columns for a tiled frame and not for a stripped one.
    pub tile_size: Option<(u32, u32)>,
    /// InkSet (tag 332), meaningful only for a Separated image
    /// (`photometric == 5`): `1` = the four process inks in C, M, Y, K order,
    /// `2` = some other ink set. Defaults to 1 per TIFF6, which is also the
    /// right default for the overwhelmingly common CMYK case. See
    /// [`FrameInfo::is_cmyk`].
    pub ink_set: u16,
    pub strip_offsets: Strips,
    pub strip_byte_counts: Strips,
    pub rows_per_strip: u32,
}

/// A horizontal band of a frame: the strips covering some rows, as a frame in
/// its own right.
///
/// [`rows`](Self::rows) is the band's position in the original frame, snapped
/// outward to strip boundaries — a caller asking for rows 1000..1100 of a file
/// with 16 rows per strip gets 992..1104, and must offset into the decoded
/// planes accordingly.
#[derive(Clone, Debug)]
pub struct RowBand {
    /// The band, decodable by every reader in this crate.
    pub frame: FrameInfo,
    /// Which rows of the original frame it holds.
    pub rows: std::ops::Range<u32>,
}

/// A frame sampled every `step` pieces — rows that are contiguous in the result
/// but spaced out in the original.
#[derive(Clone, Debug)]
pub struct SampledBand {
    /// The sampled pieces, decodable by every reader in this crate.
    pub frame: FrameInfo,
    /// Source row that decoded row 0 came from.
    pub first_row: u32,
    /// Rows in one piece — decoded rows arrive in runs of this many.
    pub rows_per_piece: u32,
    /// Pieces taken.
    pub pieces: u32,
    /// Pieces skipped between the ones taken; 1 means none were.
    pub step: u32,
}

impl SampledBand {
    /// Where `source_row` landed in the decoded result, if it was sampled.
    ///
    /// Only rows inside a piece that was *taken* are present — the ones in the
    /// skipped pieces are simply not there, which is what makes the decode
    /// cheap.
    ///
    /// Being in a piece that was taken is necessary but not sufficient: the
    /// last piece of a frame is usually short, holding fewer than
    /// `rows_per_piece` rows, and if that is the piece taken then the rows past
    /// its end were never in the file. Answering with an index the result does
    /// not have would send a caller reading off the end of the decoded plane,
    /// so the answer is checked against the frame that was actually built.
    pub fn decoded_row_of(&self, source_row: u32) -> Option<u32> {
        let per = self.rows_per_piece.max(1);
        let offset = source_row.checked_sub(self.first_row)?;
        let piece = offset / per;
        if !piece.is_multiple_of(self.step.max(1)) {
            return None; // fell in a piece that was stepped over
        }
        let taken = piece / self.step.max(1);
        if taken >= self.pieces {
            return None;
        }
        // Checked rather than saturating: `rows_per_piece` comes from the
        // file's own `RowsPerStrip`, so a hostile one can make this overflow,
        // and a saturated index is still an index the result does not have.
        let row = taken.checked_mul(per)?.checked_add(offset % per)?;
        (row < self.frame.height).then_some(row)
    }
}

/// A rectangular piece of a frame: the tiles or strips covering some rows and
/// columns, as a frame in its own right.
///
/// [`rows`](Self::rows) and [`cols`](Self::cols) give its position in the
/// original, snapped outward to whatever the file's compression unit is — tile
/// boundaries for a tiled frame, strip boundaries and the full width for a
/// stripped one, because a strip cannot be narrowed.
#[derive(Clone, Debug)]
pub struct Region {
    /// The piece, decodable by every reader in this crate.
    pub frame: FrameInfo,
    /// Which rows of the original it holds.
    pub rows: std::ops::Range<u32>,
    /// Which columns of the original it holds.
    pub cols: std::ops::Range<u32>,
}

impl RowBand {
    /// How many rows the band holds.
    pub fn len(&self) -> u32 {
        self.rows.end - self.rows.start
    }

    /// Whether the band holds no rows, which `crop_rows` never returns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FrameInfo {
    /// True for an RGB frame whose 3+ samples are color components we can
    /// split into red/green/blue planes. Either interleaving works — the
    /// decoders gather a plane from chunky and planar frames alike.
    pub fn is_rgb(&self) -> bool {
        self.photometric == 2 && self.samples_per_pixel >= 3
    }

    /// True when this frame's samples are stored as separate whole planes
    /// (PlanarConfiguration=2) rather than interleaved per pixel — the layout
    /// `tifffile` writes for a `(3|4, H, W)` array, and libtiff's
    /// `PLANARCONFIG_SEPARATE`. Single-sample frames are identical either way,
    /// so they never count as planar.
    pub fn is_planar(&self) -> bool {
        self.planar_config == 2 && self.samples_per_pixel > 1
    }

    /// True for a palette-color (indexed) frame: PhotometricInterpretation=3,
    /// one 4- or 8-bit sample per pixel that indexes the ColorMap (tag 320). The
    /// pixel value is a lookup index, not a brightness — a consumer should map
    /// it through the palette (exposed as `channel_display[0].lut`) rather than
    /// show it as a gray level. 4-bit indices are unpacked to one byte each on
    /// decode, so both ride the same 8-bit-index display path.
    pub fn is_palette(&self) -> bool {
        self.photometric == 3
            && self.samples_per_pixel == 1
            && matches!(self.bits_per_sample, 4 | 8)
    }

    /// True for a CMYK frame this crate can convert for display: a Separated
    /// image (PhotometricInterpretation=5) separated into the four *process*
    /// inks, at a width the conversion is defined for.
    ///
    /// Deliberately narrow, because every clause rules out a file the CMYK
    /// formula would silently mangle:
    ///
    /// - `ink_set == 1` — TIFF6 §16 lets a Separated image use any ink set. A
    ///   hi-fi separation (CMYK + orange + green, say) is `InkSet = 2` and its
    ///   component order is described only by `InkNames`, so treating sample 0
    ///   as cyan would be a guess. Those fall through to the generic
    ///   multi-sample path, where each ink is simply its own channel.
    /// - `samples_per_pixel >= 4` — fewer than four samples cannot be CMYK, and
    ///   `photometric` and `SamplesPerPixel` are independent untrusted tags, so
    ///   a file may well claim 5-with-1. Extra samples beyond the fourth (alpha,
    ///   or spot colours) are ignored by the conversion.
    /// - unsigned 8- or 16-bit — the only depths separated images occur in.
    ///   Anything else keeps its raw per-ink planes rather than being forced
    ///   through a formula defined on normalized ink coverage.
    ///
    /// Note this says nothing about *photometric* being trustworthy in general:
    /// it is the gate for [`crate::read_planes_rgb_u8`] and friends, which are
    /// the only readers that reinterpret samples rather than returning them.
    pub fn is_cmyk(&self) -> bool {
        self.photometric == 5
            && self.ink_set == 1
            && self.samples_per_pixel >= 4
            && self.sample_format == SampleFormat::UnsignedInt
            && matches!(self.bits_per_sample, 8 | 16)
    }

    /// Whether this frame is stored as tiles rather than strips.
    pub fn is_tiled(&self) -> bool {
        self.tile_size.is_some()
    }

    /// The tile grid: `(across, down)` tiles, and the tile size. `None` for a
    /// stripped frame.
    ///
    /// Edge tiles are *not* trimmed: TIFF6 stores every tile at the full tile
    /// size and pads the ones hanging off the right or bottom edge, so the grid
    /// covers at least the image and usually a little more. Assuming otherwise
    /// is the classic way to read a tiled file with every row shifted.
    pub fn tile_grid(&self) -> Option<(u32, u32, u32, u32)> {
        let (tw, th) = self.tile_size?;
        let (tw, th) = (tw.max(1), th.max(1));
        Some((
            self.width.div_ceil(tw).max(1),
            self.height.div_ceil(th).max(1),
            tw,
            th,
        ))
    }

    /// Pieces of compressed data one plane is divided into — tiles for a tiled
    /// frame, strips otherwise.
    pub(crate) fn pieces_per_plane(&self) -> usize {
        match self.tile_grid() {
            Some((across, down, _, _)) => across as usize * down as usize,
            None => (self.height.div_ceil(self.rows_per_strip.max(1))).max(1) as usize,
        }
    }

    /// Total compression units the frame should have: every piece of every
    /// plane. What the strip or tile table has to be long enough to describe.
    pub(crate) fn piece_count(&self) -> usize {
        self.pieces_per_plane().saturating_mul(self.piece_planes())
    }

    /// How many independent planes the pieces are grouped into: one per sample
    /// for a planar frame, one in total otherwise.
    pub(crate) fn piece_planes(&self) -> usize {
        if self.is_planar() {
            (self.samples_per_pixel as usize).max(1)
        } else {
            1
        }
    }

    /// Every `step`-th piece from the one covering `rows.start`, enough of them
    /// to reach `rows.end`.
    ///
    /// For sampling a frame coarsely. Showing a 40000-pixel mosaic on a
    /// 1900-pixel panel needs roughly every sixteenth row — and with two rows to
    /// a strip, seven strips in every eight contain nothing that will be drawn,
    /// yet a contiguous crop decompresses all of them. Stepping over them makes
    /// the decode proportional to what is sampled rather than to what is
    /// spanned, which on that file is eight times less work.
    ///
    /// The decoded rows are **not** contiguous in the original: row `k` of the
    /// result is source row `first_row + (k / rows_per_piece) * step *
    /// rows_per_piece + (k % rows_per_piece)`. Callers that sample every
    /// `step * rows_per_piece`-th source row can ignore that and simply read
    /// every `rows_per_piece`-th decoded row, which is the case this exists for
    /// — see [`SampledBand::decoded_row_of`].
    ///
    /// `step` of 1 is exactly [`crop_rows`](Self::crop_rows).
    pub fn crop_rows_step(&self, rows: std::ops::Range<u32>, step: u32) -> Result<SampledBand> {
        let step = step.max(1);
        let per_piece = if self.is_tiled() {
            self.tile_size.map_or(1, |(_, h)| h.max(1))
        } else {
            self.rows_per_strip.max(1)
        };
        if step == 1 {
            let band = self.crop_rows(rows)?;
            let first_row = band.rows.start;
            let pieces = band.len().div_ceil(per_piece).max(1);
            return Ok(SampledBand {
                frame: band.frame,
                first_row,
                rows_per_piece: per_piece,
                pieces,
                step,
            });
        }
        // Stepping over a tile grid would skip columns as well as rows, since a
        // tile row is `across` consecutive entries. Not wrong to want, but not
        // what this expresses — a tiled frame narrows by cropping instead.
        if self.is_tiled() {
            bail!("crop_rows_step is for stripped frames; a tiled frame narrows with crop()");
        }

        let per_plane = self.height.div_ceil(per_piece) as usize;
        let planes = self.piece_planes();
        let want = self.piece_count();
        if self.strip_offsets.len() < want || self.strip_byte_counts.len() < want {
            bail!(
                "strip table has {} offsets / {} byte counts, need {want}",
                self.strip_offsets.len(),
                self.strip_byte_counts.len()
            );
        }

        let start = rows.start.min(self.height.saturating_sub(1));
        let end = rows.end.clamp(start + 1, self.height.max(1));
        let first = (start / per_piece) as usize;
        // Pieces from `first`, every `step`, until one starts at or after `end`.
        let last_needed = (end - 1) / per_piece;
        let pieces = ((last_needed as usize).saturating_sub(first) / step as usize) + 1;
        let pieces = pieces.min(per_plane.saturating_sub(first)).max(1);

        let mut offsets = Vec::with_capacity(planes * pieces);
        let mut counts = Vec::with_capacity(planes * pieces);
        for p in 0..planes {
            let base = p * per_plane;
            for i in 0..pieces {
                let idx = base + first + i * step as usize;
                if idx >= base + per_plane {
                    break;
                }
                offsets.push(self.strip_offsets[idx]);
                counts.push(self.strip_byte_counts[idx]);
            }
        }
        let taken = offsets.len() / planes.max(1);

        let mut frame = self.clone();
        // The pieces sit back to back in the result, so its height is however
        // many rows they hold — the gaps between them in the original are not
        // represented, which is the whole point.
        //
        // The last piece of a frame is usually short, and if that one is taken
        // the result is shorter than `taken * per_piece`. Claiming the round
        // number would size the decode for rows the file does not have, and the
        // reader would rightly refuse a frame whose strips fall short of it.
        let last_idx = (first + (taken.saturating_sub(1)) * step as usize) as u32;
        let last_rows = per_piece.min(
            self.height
                .saturating_sub(last_idx.saturating_mul(per_piece)),
        );
        frame.height =
            (taken as u32).saturating_sub(1).saturating_mul(per_piece) + last_rows.max(1);
        frame.strip_offsets = offsets.into();
        frame.strip_byte_counts = counts.into();
        Ok(SampledBand {
            frame,
            first_row: first as u32 * per_piece,
            rows_per_piece: per_piece,
            pieces: taken as u32,
            step,
        })
    }

    /// A view of just the pieces covering `cols` x `rows` — the same frame
    /// cropped to a rectangle.
    ///
    /// The columns are only honoured for a **tiled** frame. A strip spans the
    /// full image width and is compressed as one unit, so there is no way to
    /// read part of one: for a stripped frame the request narrows the rows and
    /// the returned [`Region::cols`] says the full width was kept. That is the
    /// whole practical difference between the two layouts, and the reason a
    /// tiled file is worth preferring for images too large to hold — a window
    /// of one costs the window, where a window of a stripped file costs its
    /// full-width rows.
    ///
    /// As with [`crop_rows`](Self::crop_rows), the result is an ordinary
    /// [`FrameInfo`], so every reader, codec and predictor applies to it
    /// unchanged; and it is snapped *outward*, so the caller must index against
    /// the returned ranges rather than the ones it asked for.
    pub fn crop(&self, cols: std::ops::Range<u32>, rows: std::ops::Range<u32>) -> Result<Region> {
        let Some((across, down, tw, th)) = self.tile_grid() else {
            // Stripped: rows only, full width.
            let band = self.crop_rows(rows)?;
            let _ = cols;
            return Ok(Region {
                rows: band.rows,
                cols: 0..self.width,
                frame: band.frame,
            });
        };

        let span = |req: std::ops::Range<u32>, size: u32, n: u32, count: u32| {
            let start = req.start.min(size.saturating_sub(1));
            let end = req.end.clamp(start + 1, size.max(1));
            let first = (start / n).min(count - 1);
            let last = end.div_ceil(n).clamp(first + 1, count);
            first..last
        };
        let tx = span(cols, self.width, tw, across);
        let ty = span(rows, self.height, th, down);

        let planes = self.piece_planes();
        let want = self.piece_count();
        if self.strip_offsets.len() < want || self.strip_byte_counts.len() < want {
            bail!(
                "tile table has {} offsets / {} byte counts, need {want} for {across}x{down} tiles                  across {planes} plane(s)",
                self.strip_offsets.len(),
                self.strip_byte_counts.len()
            );
        }

        // Take the sub-rectangle of the grid out of each plane's run of tiles.
        let (nx, ny) = ((tx.end - tx.start) as usize, (ty.end - ty.start) as usize);
        let mut offsets = Vec::with_capacity(planes * nx * ny);
        let mut counts = Vec::with_capacity(planes * nx * ny);
        for p in 0..planes {
            let plane_base = p * across as usize * down as usize;
            for row in ty.clone() {
                let row_base = plane_base + row as usize * across as usize;
                let from = row_base + tx.start as usize;
                offsets.extend_from_slice(&self.strip_offsets[from..from + nx]);
                counts.extend_from_slice(&self.strip_byte_counts[from..from + nx]);
            }
        }

        let x0 = tx.start * tw;
        let y0 = ty.start * th;
        let x1 = (tx.end * tw).min(self.width);
        let y1 = (ty.end * th).min(self.height);
        let mut frame = self.clone();
        frame.width = x1 - x0;
        frame.height = y1 - y0;
        frame.strip_offsets = offsets.into();
        frame.strip_byte_counts = counts.into();
        Ok(Region {
            frame,
            rows: y0..y1,
            cols: x0..x1,
        })
    }

    /// A view of just the strips covering `rows` — the same frame cropped to a
    /// horizontal band.
    ///
    /// The point is that *every* reader then works on it unchanged. The band is
    /// an ordinary [`FrameInfo`] with a smaller `height` and a slice of the
    /// strip table, so the codecs, the predictor undo, the chunky/planar
    /// gathers and the CMYK conversion all apply to it exactly as they do to a
    /// whole frame — and decoding costs what the band costs, not what the file
    /// costs. That is what makes a 40000 x 12788 mosaic explorable without
    /// holding it all in memory.
    ///
    /// Strips are the unit of compression, so the band is snapped *outward* to
    /// strip boundaries; the returned [`RowBand::rows`] says which rows it
    /// really covers, which is what a caller must index against. Predictors do
    /// not stand in the way: TIFF differences each row from the one before it
    /// within a row, never across rows, so a strip decodes correctly on its own.
    ///
    /// Errors if the strip table is too short to describe the frame — an
    /// untrusted file may say anything, and a band cut from a table that does
    /// not add up would silently read the wrong pixels.
    pub fn crop_rows(&self, rows: std::ops::Range<u32>) -> Result<RowBand> {
        // A tiled frame keeps a grid rather than a run of strips, so the slice
        // below would take the wrong pieces. Ask for every column instead and
        // let `crop` walk the grid.
        if self.is_tiled() {
            let region = self.crop(0..self.width, rows)?;
            return Ok(RowBand {
                frame: region.frame,
                rows: region.rows,
            });
        }
        let rps = self.rows_per_strip.max(1);
        let start = rows.start.min(self.height);
        let end = rows.end.clamp(start, self.height);
        // An empty request still has to yield a decodable frame; give it the
        // single strip containing `start` rather than a zero-height band.
        let first = start / rps;
        let last = end.max(start + 1).div_ceil(rps).max(first + 1);

        let per_plane = (self.height.div_ceil(rps)) as usize;
        let planes = if self.is_planar() {
            (self.samples_per_pixel as usize).max(1)
        } else {
            1
        };
        let (first, last) = (first as usize, (last as usize).min(per_plane));
        if first >= last {
            bail!(
                "row band {start}..{end} selects no strips of a {}-row frame",
                self.height
            );
        }
        // The strip table has to actually describe the frame before a slice of
        // it can mean anything.
        let want = planes * per_plane;
        if self.strip_offsets.len() < want || self.strip_byte_counts.len() < want {
            bail!(
                "strip table describes {} offsets / {} byte counts, need {want} for {planes} plane(s)                  of {per_plane} strip(s)",
                self.strip_offsets.len(),
                self.strip_byte_counts.len()
            );
        }

        let mut offsets = Vec::with_capacity(planes * (last - first));
        let mut counts = Vec::with_capacity(planes * (last - first));
        for p in 0..planes {
            let base = p * per_plane;
            offsets.extend_from_slice(&self.strip_offsets[base + first..base + last]);
            counts.extend_from_slice(&self.strip_byte_counts[base + first..base + last]);
        }

        let y0 = first as u32 * rps;
        let y1 = ((last as u32) * rps).min(self.height);
        let mut frame = self.clone();
        frame.height = y1 - y0;
        frame.strip_offsets = offsets.into();
        frame.strip_byte_counts = counts.into();
        Ok(RowBand {
            frame,
            rows: y0..y1,
        })
    }

    /// Bytes one sample occupies at this frame's bit depth, or an error for a
    /// depth this crate can't decode. 4-bit is *not* included: its samples are
    /// sub-byte and take a dedicated unpacking path
    /// (`decode::read_frame_u8`), never the byte-oriented one this feeds.
    pub fn sample_bytes(&self) -> Result<usize> {
        match self.bits_per_sample {
            8 => Ok(1),
            16 => Ok(2),
            32 => Ok(4),
            64 => Ok(8),
            other => bail!("unsupported bits_per_sample: {other}"),
        }
    }

    /// Pixels in this frame — `width * height` — with **checked** arithmetic.
    ///
    /// Both factors come from the file. The product can't exceed `usize` on a
    /// 64-bit target, but it can on a 32-bit one, so this is checked for the
    /// same reason [`FrameInfo::decoded_len`] is: a wrapped-small count drives
    /// buffer carving that then reads past the allocation.
    pub fn pixel_count(&self) -> Result<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or_else(|| {
                anyhow!(
                    "frame geometry overflows address space: {}x{}",
                    self.width,
                    self.height
                )
            })
    }

    /// Samples in this frame — `width * height * samples_per_pixel` — with
    /// **checked** arithmetic.
    ///
    /// Unlike [`FrameInfo::pixel_count`] this overflows on 64-bit too: a TIFF
    /// declaring `2147483648 x 2147483648 x 4` lands exactly one past
    /// `usize::MAX`. Reach for this rather than multiplying by hand — the
    /// decoders run it *before* any other validation, so it is the first thing
    /// a malformed frame touches.
    pub fn sample_count(&self) -> Result<usize> {
        self.pixel_count()?
            .checked_mul(self.samples_per_pixel.max(1) as usize)
            .ok_or_else(|| {
                anyhow!(
                    "frame geometry overflows address space: {}x{} x {} sample(s)/px",
                    self.width,
                    self.height,
                    self.samples_per_pixel.max(1)
                )
            })
    }

    /// Byte length of this frame fully decoded — `width * height *
    /// samples_per_pixel * sample_bytes` — computed with **checked** arithmetic.
    ///
    /// The inputs all come straight from the file, so a malformed TIFF can make
    /// this product exceed `usize`. Left unchecked it wraps (release builds
    /// don't trap on overflow), and a wrapped-small length then drives buffer
    /// carving that reads far past the allocation. Every allocation and bounds
    /// check in `decode` derives from this one function so the overflow can only
    /// be got wrong in a single place.
    pub fn decoded_len(&self) -> Result<usize> {
        let sample_bytes = self.sample_bytes()?;
        let too_big = || {
            anyhow!(
                "frame geometry overflows address space: {}x{} x {} sample(s)/px x {} byte(s)/sample",
                self.width,
                self.height,
                self.samples_per_pixel.max(1),
                sample_bytes
            )
        };
        (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|v| v.checked_mul(self.samples_per_pixel.max(1) as usize))
            .and_then(|v| v.checked_mul(sample_bytes))
            .ok_or_else(too_big)
    }

    /// Total bytes the frame's usable strips supply.
    ///
    /// Counts only strips that have **both** an offset and a byte count — the
    /// decoders read them zipped, so a file declaring nine `StripByteCounts`
    /// against one `StripOffsets` supplies exactly one strip's worth, not nine.
    /// (Summing the raw byte-count array instead let a 291-byte fuzz case claim
    /// ~591 KB of input and clear the way for a 4.3 GB allocation.) Saturating:
    /// the values are file-controlled, and `open` has already rejected any
    /// strip that doesn't lie inside the file.
    pub(crate) fn strip_bytes_total(&self) -> u64 {
        self.strip_offsets
            .iter()
            .zip(self.strip_byte_counts.iter())
            .fold(0u64, |acc, (_, &n)| acc.saturating_add(n))
    }

    /// The most decoded bytes this frame's strips could possibly cover.
    ///
    /// A strip holds at most `rows_per_strip` rows, so `strips × rows × row`
    /// bounds the image regardless of codec — no compression-ratio guesswork,
    /// and no false positives, because a well-formed frame always declares
    /// enough strips to cover its own rows (that is what makes it decodable).
    /// Mirrors `decode::strip_dest_lens`'s row geometry, including the planar
    /// case where a "row" is one sample plane's worth.
    pub(crate) fn strip_coverable_bytes(&self) -> Result<u64> {
        let sample_bytes = self.sample_bytes()? as u64;
        let per_row_samples = if self.is_planar() {
            1
        } else {
            self.samples_per_pixel.max(1) as u64
        };
        let row_bytes = (self.width as u64)
            .saturating_mul(per_row_samples)
            .saturating_mul(sample_bytes);
        let rows_per_strip = (self.rows_per_strip as u64).max(1).min(self.height as u64);
        Ok((self.strip_offsets.len() as u64)
            .saturating_mul(rows_per_strip)
            .saturating_mul(row_bytes))
    }
}

/// Where a stack's bytes live. Both variants deref to `&[u8]`, so every decode
/// entry point takes one shape regardless of how the file was opened.
///
/// `Mapped` is the native path — the OS pages data in on demand, so opening a
/// multi-GB stack costs nothing until pixels are touched. `Owned` is for hosts
/// with no filesystem to map (a browser, where the bytes arrive from a file
/// input or `fetch`), and for callers that already hold the whole file.
pub enum Bytes {
    #[cfg(feature = "mmap")]
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(feature = "mmap")]
            Bytes::Mapped(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            #[cfg(feature = "mmap")]
            Bytes::Mapped(_) => "Mapped",
            Bytes::Owned(_) => "Owned",
        };
        write!(f, "Bytes::{kind}({} bytes)", self.len())
    }
}

// `non_exhaustive`: fields have been added before (`description`) and may be
// again; constructing this outside the crate isn't meaningful anyway (it's
// produced by `open` / `from_bytes`).
#[non_exhaustive]
pub struct TiffStack {
    /// The file's bytes — memory-mapped, or owned. Derefs to `&[u8]`, which is
    /// what every `read_*` function takes.
    pub data: Bytes,
    pub byte_order: ByteOrder,
    pub frames: Vec<FrameInfo>,
    pub meta: StackMeta,
    /// The first IFD's raw `ImageDescription` (tag 270) text, verbatim —
    /// full access to whatever the writer put there. `meta` holds the parsed
    /// ImageJ view of it; this is the unparsed original (which may not be
    /// ImageJ-formatted at all).
    pub description: Option<String>,
    /// Classic TIFF (magic 42) or BigTIFF (magic 43). Informational — frames
    /// decode identically either way.
    pub flavor: TiffFlavor,
}

impl TiffStack {
    /// Open and index `path` through a read-only memory map.
    ///
    /// Requires the `mmap` feature (on by default). Where there is no
    /// filesystem to map — wasm, most obviously — use [`TiffStack::from_bytes`].
    #[cfg(feature = "mmap")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| anyhow!("could not open {}: {e}", path.as_ref().display()))?;
        // SAFETY: standard caveat of memmap2 — the file must not be mutated
        // out from under us while mapped. We open read-only and treat the
        // mapping as immutable for the lifetime of the TiffStack.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_backing(Bytes::Mapped(mmap))
    }

    /// Index a TIFF already held in memory — the entry point for a host with no
    /// filesystem (a browser file input, a `fetch`, an embedded asset).
    ///
    /// Identical to [`TiffStack::open`] in every other respect: same parser,
    /// same errors, same frame index. The bytes are kept for the stack's
    /// lifetime, so unlike the mapped path the whole file sits in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Bytes::Owned(bytes))
    }

    /// Walk the IFD chain and build the frame index + metadata. Shared by both
    /// entry points, so a mapped and an in-memory stack can never diverge.
    fn from_backing(data: Bytes) -> Result<Self> {
        let mmap: &[u8] = &data;

        let (order, flavor, first_ifd) = ifd::read_header(mmap)?;

        let mut frames = Vec::new();
        let mut description: Option<String> = None;
        let mut ij_metadata_bytes: Option<Vec<u8>> = None;
        let mut ij_metadata_counts: Option<Vec<u32>> = None;
        // XResolution/YResolution (pixels per unit) → x/y pixel calibration.
        let mut x_resolution: Option<f64> = None;
        let mut y_resolution: Option<f64> = None;
        // ColorMap (tag 320) for a palette image → the channel's display LUT.
        let mut color_map: Option<Vec<u32>> = None;

        let mut offset = usize::try_from(first_ifd)
            .map_err(|_| anyhow!("first IFD offset exceeds address space"))?;
        let mut first = true;
        // Cycle detection, Brent's algorithm: one comparison per IFD and no
        // memory, in place of a `HashSet` of every offset seen. That set cost a
        // hash insert per frame and, on a stack near `MAX_FRAMES`, tens of
        // megabytes of table — for a check that fires only on a malformed file.
        // Brent's finds any cycle from a single forward walk, which is the walk
        // we are already doing.
        //
        // It needs up to ~2x the cycle length to notice, where the set noticed
        // on the first repeat. So a cycle longer than half of `MAX_FRAMES`
        // trips the frame-count bail below instead of this one — a different
        // error message for a file with half a million looping directories,
        // which is bounded either way and not worth a hash table per frame.
        let mut tortoise = offset;
        let mut power = 1usize;
        let mut lam = 0usize;
        // One entry buffer for the whole chain (see `ifd::read_ifd_into`).
        let mut entries: Vec<ifd::RawIfdEntry> = Vec::new();
        let mut strip_scratch: Vec<u64> = Vec::new();

        while offset != 0 {
            let next_offset = ifd::read_ifd_into(mmap, offset, order, flavor, &mut entries)?;
            let frame = frame_info_from_entries(&entries, mmap, order, &mut strip_scratch)?;

            if first {
                for e in &entries {
                    match e.tag {
                        TAG_IMAGE_DESCRIPTION => {
                            description = e.as_ascii(mmap, order).ok();
                        }
                        TAG_IJ_METADATA => {
                            ij_metadata_bytes = e.owned_bytes(mmap, order).ok();
                        }
                        TAG_IJ_METADATA_BYTE_COUNTS => {
                            ij_metadata_counts = e.as_u32_array(mmap, order).ok();
                        }
                        TAG_X_RESOLUTION => {
                            x_resolution = e.as_rational(mmap, order).ok();
                        }
                        TAG_Y_RESOLUTION => {
                            y_resolution = e.as_rational(mmap, order).ok();
                        }
                        TAG_COLOR_MAP => {
                            color_map = e.as_u32_array(mmap, order).ok();
                        }
                        _ => {}
                    }
                }
                first = false;
            }

            frames.push(frame);
            if frames.len() > MAX_FRAMES {
                bail!(
                    "TIFF declares more than {MAX_FRAMES} planes — refusing to index further \
                     (a malformed chain of minimal IFDs can otherwise amplify a small file \
                     into gigabytes of frame index)"
                );
            }
            offset = usize::try_from(next_offset)
                .map_err(|_| anyhow!("next-IFD offset exceeds address space"))?;
            if offset != 0 {
                if offset == tortoise {
                    bail!("malformed TIFF: IFD chain loops back to offset {offset}");
                }
                lam += 1;
                if lam == power {
                    tortoise = offset;
                    power *= 2;
                    lam = 0;
                }
            }
        }

        if frames.is_empty() {
            bail!("TIFF has no image directories");
        }

        // Every frame in the stack must share the first frame's geometry and
        // pixel layout. The viewer uploads every frame into one fixed-size GPU
        // texture (sized to frame 0) and decodes with a single stride, so a
        // differently-shaped frame — e.g. the reduced-resolution levels of a
        // pyramidal TIFF, or an appended thumbnail page — would otherwise be
        // silently mis-rendered. Catch it here with a clear error instead.
        let f0 = &frames[0];
        let f0_shape = (
            f0.width,
            f0.height,
            f0.bits_per_sample,
            f0.samples_per_pixel,
        );
        if let Some((i, f)) = frames
            .iter()
            .enumerate()
            .find(|(_, f)| (f.width, f.height, f.bits_per_sample, f.samples_per_pixel) != f0_shape)
        {
            bail!(
                "TIFF frames are not uniform: frame 0 is {}x{} ({}-bit, {} sample(s)/px) but \
                 frame {} is {}x{} ({}-bit, {} sample(s)/px). This looks like a pyramidal or \
                 mixed-size TIFF, which this stack viewer doesn't support.",
                f0.width,
                f0.height,
                f0.bits_per_sample,
                f0.samples_per_pixel,
                i,
                f.width,
                f.height,
                f.bits_per_sample,
                f.samples_per_pixel,
            );
        }

        // ImageJ's own writer handles >4 GiB stacks not with BigTIFF but with
        // a classic-TIFF hack: ONE IFD, `images=N` in the description, and the
        // remaining N-1 frames appended as raw contiguous pixel data after the
        // first. Without this, such a file opens as a single frame and
        // scrubbing does nothing. Synthesize the virtual frames (tifffile does
        // the same). Only the unambiguous case qualifies: a single
        // uncompressed, predictor-free IFD whose strip data is contiguous.
        if frames.len() == 1 {
            if let Some(n) = description
                .as_deref()
                .and_then(metadata::imagej::images_count)
            {
                if n > 1 {
                    expand_imagej_contiguous(&mut frames, n, mmap.len());
                }
            }
        }

        // Everything above trusted the file's own numbers. Vet them once, here,
        // so every `FrameInfo` a caller can reach carries the invariants the
        // decoders rely on: a non-empty image, geometry that fits in `usize`,
        // and strips that actually lie inside the file.
        validate_frames(&frames, mmap.len())?;

        let mut meta = metadata::parse(
            description.as_deref(),
            ij_metadata_bytes.as_deref(),
            ij_metadata_counts.as_deref(),
            frames.len(),
            x_resolution,
            y_resolution,
        );

        // A ColorMap (tag 320) is the single channel's display LUT. ImageJ
        // attaches one in two situations, and we honor both — the pixels stay
        // as-is (the LUT is applied on display), and the consumer renders the
        // file's real colors instead of a gray level:
        //   - 8-bit **palette** images (photometric=3): the pixel is a direct
        //     index into the map (see `FrameInfo::is_palette`; the viewer then
        //     maps index → entry with an identity contrast window).
        //   - 16-/32-bit **grayscale** images (photometric=1) that ImageJ has
        //     colored with a Fire/Ice-style LUT: the map is applied through the
        //     normal contrast window (`min=`/`max=`), not as a direct index.
        // This is orthogonal to the metadata dialect, so it's applied here after
        // the parse, over the single (grayscale-default) channel it produced.
        if frames[0].samples_per_pixel == 1 {
            if let Some(lut) = color_map.as_deref().and_then(metadata::colormap_to_lut) {
                // A grayscale ramp is a visual no-op — apply it (harmless) but
                // don't mark it "explicit", so the pseudocolor toggle stays
                // available for a genuinely grayscale file.
                let colored = lut.iter().any(|px| px[0] != px[1] || px[1] != px[2]);
                if let Some(cd) = meta.channel_display.first_mut() {
                    cd.lut = lut;
                }
                if colored {
                    // Keep the file's colors (no pseudocolor override); it's a
                    // color image, not grayscale.
                    meta.has_explicit_luts = true;
                    meta.mode = metadata::DisplayMode::Color;
                }
            }
        }

        Ok(TiffStack {
            data,
            byte_order: order,
            frames,
            meta,
            description,
            flavor,
        })
    }
}

/// Reject frames whose declared geometry or strip locations are impossible for
/// a file of this size. Runs once at index time, on the final frame list, so
/// the decoders can treat a `FrameInfo` as already-sane rather than re-deriving
/// the same checks per call.
///
/// Deliberately *not* checked here: an exotic `bits_per_sample` (1-bit bilevel,
/// say). Those have always opened fine and only failed at decode, and metadata
/// readers legitimately open such files without ever decoding pixels — so
/// rejecting them at `open` would be a regression. Their geometry is checked
/// only if the depth is one we can actually decode.
fn validate_frames(frames: &[FrameInfo], file_len: usize) -> Result<()> {
    for (i, f) in frames.iter().enumerate() {
        if f.width == 0 || f.height == 0 {
            bail!(
                "frame {i} declares an empty image ({}x{})",
                f.width,
                f.height
            );
        }
        // Overflow guard. `decoded_len` is what every downstream allocation is
        // sized from, so proving it here means it cannot wrap there.
        if f.sample_bytes().is_ok() {
            f.decoded_len().map_err(|e| anyhow!("frame {i}: {e}"))?;
        }
        // A strip that starts or ends outside the file is unambiguously broken,
        // and refusing it up front bounds `strip_bytes_total` by the file size —
        // which is what makes the decode-side plausibility check meaningful.
        for (&off, &len) in f.strip_offsets.iter().zip(f.strip_byte_counts.iter()) {
            let end = off
                .checked_add(len)
                .ok_or_else(|| anyhow!("frame {i}: strip offset {off} + length {len} overflows"))?;
            if end > file_len as u64 {
                bail!(
                    "frame {i}: strip at offset {off} (+{len} bytes) extends past the end of \
                     the {file_len}-byte file",
                );
            }
        }
    }
    Ok(())
}

/// Expand a single-IFD ImageJ "contiguous" stack (see the call site) into `n`
/// virtual single-strip frames at `base + i * frame_bytes`. Leaves `frames`
/// untouched unless the layout is unambiguously the ImageJ hack; `n` is
/// clamped to the frames that actually fit in the file (ImageJ itself writes
/// the count before the data, so truncated files exist in the wild).
fn expand_imagej_contiguous(frames: &mut Vec<FrameInfo>, n: usize, file_len: usize) {
    let f = &frames[0];
    let sample_bytes = match f.bits_per_sample {
        8 => 1usize,
        16 => 2,
        32 => 4,
        64 => 8,
        _ => return,
    };
    if f.compression != Compression::None || f.predictor != 1 {
        return;
    }
    // Checked: every factor is file-declared, and a wrapped `frame_bytes` would
    // make the `available` division below hand back a bogus frame count.
    let Some(frame_bytes) = (f.width as u64)
        .checked_mul(f.height as u64)
        .and_then(|v| v.checked_mul(f.samples_per_pixel as u64))
        .and_then(|v| v.checked_mul(sample_bytes as u64))
    else {
        return;
    };
    if frame_bytes == 0 || f.strip_offsets.is_empty() {
        return;
    }
    // The IFD's strips must cover frame 0 contiguously from its first offset.
    let base = f.strip_offsets[0];
    let mut cursor = base;
    for (&off, &len) in f.strip_offsets.iter().zip(f.strip_byte_counts.iter()) {
        if off != cursor {
            return; // gap between strips: not the contiguous layout
        }
        cursor = cursor.saturating_add(len);
    }
    if cursor - base < frame_bytes {
        return; // declared strips don't even cover one frame
    }

    let available = (file_len as u64).saturating_sub(base) / frame_bytes;
    let n = (n as u64).min(available).min(MAX_FRAMES as u64).max(1);
    let template = frames[0].clone();
    *frames = (0..n)
        .map(|i| FrameInfo {
            strip_offsets: Strips::One(base + i * frame_bytes),
            strip_byte_counts: Strips::One(frame_bytes),
            rows_per_strip: template.height,
            ..template.clone()
        })
        .collect();
}

impl TiffStack {
    /// Touch every page of `frame`'s strip data so a subsequent decode doesn't
    /// stall on page faults. First access to a memory-mapped page soft-faults,
    /// which is cheap on Linux but costs real time on Windows; calling this
    /// from a background thread (e.g. a decode-ahead worker) for the *next*
    /// frame absorbs those faults off the latency-critical path. Purely a
    /// performance hint — safe to skip, safe to repeat.
    pub fn prefetch_frame(&self, frame: &FrameInfo) {
        const PAGE: usize = 4096;
        for (&off, &len) in frame
            .strip_offsets
            .iter()
            .zip(frame.strip_byte_counts.iter())
        {
            let start = off as usize;
            let end = start.saturating_add(len as usize).min(self.data.len());
            let Some(strip) = self.data.get(start..end) else {
                continue;
            };
            let mut i = 0;
            while i < strip.len() {
                std::hint::black_box(strip[i]);
                i += PAGE;
            }
            if let Some(&last) = strip.last() {
                std::hint::black_box(last);
            }
        }
    }
}

fn frame_info_from_entries(
    entries: &[RawIfdEntry],
    file: &[u8],
    order: ByteOrder,
    // Reused across the whole chain: reading a strip array into a fresh `Vec`
    // and converting it would allocate and free once per frame even for the
    // single-strip case this exists to avoid.
    scratch: &mut Vec<u64>,
) -> Result<FrameInfo> {
    let mut width = None;
    let mut height = None;
    // The TIFF6 default for a missing BitsPerSample is 1 (bilevel), but 1-bit
    // data isn't decodable here anyway; 16 is the pragmatic default for the
    // scientific files this library targets, where the tag is always present.
    let mut bits_per_sample = 16u16;
    let mut samples_per_pixel = 1u16; // default per spec
    let mut sample_format_raw = 1u16; // default: unsigned integer
    let mut compression_raw = 1u16; // default: no compression
    let mut predictor = 1u16;
    let mut photometric = 1u16; // default: BlackIsZero grayscale
    let mut planar_config = 1u16; // default: chunky / interleaved
    let mut ink_set = 1u16; // default per TIFF6 §16: CMYK
    let mut rows_per_strip = u32::MAX; // default: whole image is one strip
    let mut strip_offsets = None;
    let mut strip_byte_counts = None;
    let mut tile_offsets = None;
    let mut tile_byte_counts = None;
    let mut tile_width = None;
    let mut tile_length = None;

    for e in entries {
        match e.tag {
            TAG_IMAGE_WIDTH => width = Some(e.as_u32(file, order)?),
            TAG_IMAGE_LENGTH => height = Some(e.as_u32(file, order)?),
            TAG_BITS_PER_SAMPLE => bits_per_sample = e.as_u32(file, order)? as u16,
            TAG_SAMPLES_PER_PIXEL => samples_per_pixel = e.as_u32(file, order)? as u16,
            TAG_SAMPLE_FORMAT => sample_format_raw = e.as_u32(file, order)? as u16,
            TAG_COMPRESSION => compression_raw = e.as_u32(file, order)? as u16,
            TAG_PREDICTOR => predictor = e.as_u32(file, order)? as u16,
            TAG_ROWS_PER_STRIP => rows_per_strip = e.as_u32(file, order)?,
            // u64 accessors: BigTIFF stores these as LONG8 past 4 GiB.
            TAG_STRIP_OFFSETS => {
                e.read_u64_array(file, order, scratch)?;
                strip_offsets = Some(Strips::from(&scratch[..]));
            }
            TAG_STRIP_BYTE_COUNTS => {
                e.read_u64_array(file, order, scratch)?;
                strip_byte_counts = Some(Strips::from(&scratch[..]));
            }
            TAG_TILE_OFFSETS => {
                e.read_u64_array(file, order, scratch)?;
                tile_offsets = Some(Strips::from(&scratch[..]));
            }
            TAG_TILE_BYTE_COUNTS => {
                e.read_u64_array(file, order, scratch)?;
                tile_byte_counts = Some(Strips::from(&scratch[..]));
            }
            TAG_TILE_WIDTH => tile_width = Some(e.as_u32(file, order)?),
            TAG_TILE_LENGTH => tile_length = Some(e.as_u32(file, order)?),
            TAG_PHOTOMETRIC => photometric = e.as_u32(file, order)? as u16,
            TAG_PLANAR_CONFIG => planar_config = e.as_u32(file, order)? as u16,
            TAG_INK_SET => ink_set = e.as_u32(file, order)? as u16,
            _ => {}
        }
    }

    let width = width.ok_or_else(|| anyhow!("IFD missing ImageWidth"))?;
    let height = height.ok_or_else(|| anyhow!("IFD missing ImageLength"))?;
    // A tiled image carries the same information under different tags. Folding
    // it into the strip fields here is what lets everything downstream — the
    // bounds checks, the size guards, the parallel decompression — apply to
    // tiles unchanged; `tile_size` is what tells the parts that must know
    // apart. Tiles win if both are somehow present, since a file declaring
    // TileOffsets is a tiled file whatever else it says.
    let tile_size = match (tile_width, tile_length) {
        (Some(w), Some(h)) if w > 0 && h > 0 && tile_offsets.is_some() => Some((w, h)),
        _ => None,
    };
    let (strip_offsets, strip_byte_counts) = match tile_size {
        Some(_) => (
            tile_offsets.ok_or_else(|| anyhow!("IFD missing TileOffsets"))?,
            tile_byte_counts.ok_or_else(|| anyhow!("IFD missing TileByteCounts"))?,
        ),
        None => (
            strip_offsets.ok_or_else(|| {
                if tile_offsets.is_some() {
                    anyhow!("IFD has TileOffsets but no usable TileWidth/TileLength")
                } else {
                    anyhow!("IFD missing StripOffsets")
                }
            })?,
            strip_byte_counts.ok_or_else(|| anyhow!("IFD missing StripByteCounts"))?,
        ),
    };

    if rows_per_strip == u32::MAX {
        rows_per_strip = height;
    }
    if let Some((_, tile_h)) = tile_size {
        // A tiled image has no RowsPerStrip. Reporting the tile height keeps
        // anything that reasons in rows-per-unit-of-compression correct.
        rows_per_strip = tile_h;
    }

    let sample_format = match sample_format_raw {
        2 => SampleFormat::SignedInt,
        3 => SampleFormat::Float,
        _ => SampleFormat::UnsignedInt,
    };
    let compression = match compression_raw {
        1 => Compression::None,
        5 => Compression::Lzw,
        32773 => Compression::PackBits,
        8 | 32946 => Compression::Deflate,
        50000 | 34926 => Compression::Zstd,
        other => Compression::Other(other),
    };

    Ok(FrameInfo {
        width,
        height,
        bits_per_sample,
        samples_per_pixel,
        sample_format,
        compression,
        predictor,
        photometric,
        planar_config,
        tile_size,
        ink_set,
        strip_offsets,
        strip_byte_counts,
        rows_per_strip,
    })
}

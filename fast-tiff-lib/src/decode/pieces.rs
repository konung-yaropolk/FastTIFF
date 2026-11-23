//! The pieces of a frame — strips, or a planar frame's strips one sample
//! plane at a time — and the engine that walks them: each piece's bytes,
//! borrowed from the source when they already are the samples or decompressed
//! into a buffer the size of one piece, handed on while still in cache.
//!
//! Also the size guards every decode runs first. The header is untrusted, so a
//! few hundred bytes must not be able to demand gigabytes; see
//! [`guard_frame_size`].

use super::codecs::{decompress_into, undo_predictor_in_place};
use super::core::Src;
use super::should_parallelize;
use crate::error::{bail, err, ErrorKind, OptionContext, Result};
use crate::index::{Compression, FrameInfo};
use rayon::prelude::*;
use std::sync::Mutex;

/// How far a *compressed* frame's decoded size may exceed the compressed bytes
/// its strips actually hold, before the file is treated as malformed.
///
/// This is a heuristic, and the only one in the size guards — everything else is
/// structural. It exists because `rows_per_strip` is file-declared, so the
/// "enough strips to cover the image" check can be satisfied by any frame that
/// simply claims one enormous strip; something has to bound how far a handful of
/// input bytes may expand.
///
/// The value sits comfortably above every real codec and comfortably below the
/// absurd. Deflate tops out near 1032:1 and LZW lower; ZSTD is the outlier,
/// since an RLE block encodes ~128 KiB in a few bytes for a practical ceiling
/// around 2^15 on degenerate all-one-value data. 2^18 leaves that another eight
/// times of headroom, while still refusing the tiny-file-claims-a-terabyte case
/// that motivated these guards.
pub(super) const MAX_DECODED_EXPANSION: u64 = 1 << 18;

/// Bytes per sample for a supported bit depth (8/16/32/64), or an error.
pub(crate) fn bytes_for_bits(bits: u16) -> Result<usize> {
    match bits {
        8 => Ok(1),
        16 => Ok(2),
        32 => Ok(4),
        64 => Ok(8),
        other => bail!(Unsupported, "unsupported bits_per_sample: {other}"),
    }
}

/// Reject a frame whose declared decoded size can't possibly be backed by the
/// bytes its strips hold — **before** anything allocates. Returns the vetted
/// decoded length.
///
/// Everything here comes from the file's header, so without this a few hundred
/// bytes can demand gigabytes. That matters more than an ordinary error path:
/// an oversized `vec![0u8; n]` *aborts* the process, because Rust's
/// allocation-failure path is not catchable — it would take any host
/// application down with it, uncatchably.
///
/// Two bounds, neither of which needs a compression-ratio constant: what the
/// declared strips can structurally *cover*, and — for uncompressed data, where
/// the strips are the pixels — what they actually supply. Callers that then
/// allocate should still do so fallibly (`try_reserve`), so a merely-large
/// frame on a memory-starved machine degrades to an `Err` rather than an abort.
pub(super) fn guard_frame_size(frame: &FrameInfo) -> Result<usize> {
    let total_len = frame.decoded_len()?;
    guard_sizes(
        total_len,
        frame.strip_coverable_bytes()?,
        frame.strip_bytes_total(),
        frame,
        frame.rows_per_strip,
    )?;
    Ok(total_len)
}

/// The two bounds of [`guard_frame_size`], given the numbers: `total_len`
/// decoded bytes against what the strips can cover, `coverable`, and what they
/// hold, `available`, for `frame`'s strips of `rows_per_strip` rows. Its own
/// function so the padded tile rectangle is checked by the same code, without
/// cloning the frame to describe it.
fn guard_sizes(
    total_len: usize,
    coverable: u64,
    available: u64,
    frame: &FrameInfo,
    rows_per_strip: u32,
) -> Result<()> {
    // Structural bound, and the load-bearing one: a strip holds at most
    // `rows_per_strip` rows, so the strips a frame declares can only ever cover
    // so much image. Codec-independent — no compression-ratio constant to guess
    // — and it cannot reject a well-formed file, because declaring enough
    // strips to cover your own rows is what *makes* a frame decodable.
    if total_len as u64 > coverable {
        bail!(
            Malformed,
            "frame declares {total_len} decoded bytes but its {} strip(s) of {} row(s) can cover \
             only {coverable} — refusing to allocate for a malformed or hostile file",
            frame.strip_offsets.len(),
            rows_per_strip,
        );
    }

    // Input-supply bound. The structural check above is necessary but not
    // sufficient, because `rows_per_strip` is itself file-declared: setting it
    // to at least the image height makes "the strips can cover it" trivially
    // true for any height at all. So also bound the decode by the bytes the
    // strips actually hold.
    //
    // Uncompressed is exact — the strips *are* the pixels. Compressed genuinely
    // expands, so it gets [`MAX_DECODED_EXPANSION`] of headroom.
    if frame.compression == Compression::None {
        if total_len as u64 > available {
            bail!(
                Malformed,
                "uncompressed frame declares {total_len} decoded bytes but its strips supply \
                 only {available}"
            );
        }
    } else {
        let ceiling = available.saturating_mul(MAX_DECODED_EXPANSION);
        if total_len as u64 > ceiling {
            bail!(
                Malformed,
                "frame declares {total_len} decoded bytes from only {available} compressed \
                 byte(s) ({:?}) — beyond any real codec's expansion, refusing to allocate",
                frame.compression
            );
        }
    }
    Ok(())
}

/// A tiled frame's grid, vetted: what the tile reader needs, after the checks
/// that have to pass before any tile is touched.
#[derive(Clone, Copy, Debug)]
pub(super) struct TileGrid {
    pub across: usize,
    pub down: usize,
    pub tile_w: usize,
    pub tile_h: usize,
    /// Bytes of one tile row.
    pub tile_row_bytes: usize,
}

/// Vet a tiled frame before decoding it: a geometry with something in it, a
/// tile table long enough to describe the grid, the frame's own size guard,
/// and the same guard over the *padded* rectangle the tiles really cover.
///
/// The image size does not bound that rectangle: tile dimensions are their own
/// file-declared tags, so a 40x36 image may claim 65535x65535 tiles, and a
/// buffer for one of those is twelve gigabytes. The frame guard's input-supply
/// bound (decoded bytes against compressed bytes present) is exactly the check
/// that catches it, so the padded rectangle goes through the same one.
pub(super) fn guard_tiles(frame: &FrameInfo, sample_bytes: usize) -> Result<TileGrid> {
    let (across, down, tile_w, tile_h) = frame.tile_grid().context(
        ErrorKind::Internal,
        "the tile reader was called on a stripped frame",
    )?;
    let spp = (frame.samples_per_pixel as usize).max(1);
    let planar = frame.is_planar();
    let n_planes = if planar { spp } else { 1 };
    // Samples per pixel *within one plane*: a planar plane holds one.
    let per_px = if planar { 1 } else { spp };

    let dst_row_bytes = (frame.width as usize)
        .saturating_mul(per_px)
        .saturating_mul(sample_bytes);
    let tile_row_bytes = (tile_w as usize)
        .saturating_mul(per_px)
        .saturating_mul(sample_bytes);
    let tile_bytes = tile_row_bytes.saturating_mul(tile_h as usize);
    let plane_bytes = dst_row_bytes.saturating_mul(frame.height as usize);
    if tile_bytes == 0 || plane_bytes == 0 {
        bail!(
            Malformed,
            "degenerate tile geometry: {tile_w}x{tile_h} tiles in a {}x{} frame",
            frame.width,
            frame.height
        );
    }

    // The tile table has to describe the grid before any of it can be indexed.
    let want = frame.piece_count();
    if frame.strip_offsets.len() < want || frame.strip_byte_counts.len() < want {
        bail!(
            Malformed,
            "tile table has {} offsets / {} byte counts, need {want} for {across}x{down} tiles              across {n_planes} plane(s)",
            frame.strip_offsets.len(),
            frame.strip_byte_counts.len()
        );
    }

    guard_frame_size(frame)?;
    // The padded rectangle, in the numbers `FrameInfo::decoded_len` and
    // `strip_coverable_bytes` would give a frame of that size in tiles-as-strips.
    let padded = || -> Result<()> {
        let width = across.saturating_mul(tile_w);
        let height = down.saturating_mul(tile_h);
        let too_big = || {
            format!(
                "frame geometry overflows address space: {width}x{height} x {} sample(s)/px x \
                 {sample_bytes} byte(s)/sample",
                spp
            )
        };
        let total_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|v| v.checked_mul(spp))
            .and_then(|v| v.checked_mul(sample_bytes))
            .with_context(ErrorKind::TooLarge, too_big)?;
        let row_bytes = (width as u64)
            .saturating_mul(per_px as u64)
            .saturating_mul(sample_bytes as u64);
        let rows = (tile_h as u64).max(1).min(height as u64);
        let coverable = (frame.strip_offsets.len() as u64)
            .saturating_mul(rows)
            .saturating_mul(row_bytes);
        guard_sizes(
            total_len,
            coverable,
            frame.strip_bytes_total(),
            frame,
            tile_h,
        )
    };
    padded().map_err(|e| {
        crate::error::Error::new(
            e.kind(),
            format!(
                "{tile_w}x{tile_h} tiles over a {}x{} frame: {e}",
                frame.width, frame.height
            ),
        )
    })?;

    Ok(TileGrid {
        across: across as usize,
        down: down as usize,
        tile_w: tile_w as usize,
        tile_h: tile_h as usize,
        tile_row_bytes,
    })
}

/// Where one strip of a piece-plane lies: its index in the strip table, the
/// image rows it holds, and its bytes in the file.
#[derive(Clone, Copy, Debug)]
pub(super) struct Strip {
    pub index: usize,
    pub row: usize,
    pub rows: usize,
    pub offset: u64,
    pub len: u64,
}

/// The strips a stripped frame's piece-planes are cut into.
///
/// A piece-plane is what the strip table is grouped by: a planar frame's
/// sample planes, each split into strips of its own (TIFF6's StripsPerImage
/// strips per plane), or a chunky frame's one interleaved plane. A frame
/// declaring more strips than this has the excess ignored, as it always has;
/// one declaring fewer fails when a missing strip is reached.
///
/// One layout is read as the decoder has always read it rather than as the
/// strip table says: a single uncompressed strip holding the whole frame
/// ([`FrameInfo::whole_strip`]). Whatever its RowsPerStrip, and however many
/// planes it holds, it is one piece per piece-plane, the planes one after
/// another inside it.
#[derive(Clone, Copy, Debug)]
pub(super) struct StripPlan<'f> {
    info: &'f FrameInfo,
    pub per_plane: usize,
    pub rows_per_strip: usize,
    pub height: usize,
    /// Strips with both an offset and a byte count.
    pub available: usize,
    /// The whole-frame strip's offset and the bytes of one piece-plane in it.
    whole: Option<(u64, u64)>,
}

impl<'f> StripPlan<'f> {
    pub fn of(frame: &'f FrameInfo) -> Self {
        let height = frame.height as usize;
        if let Some(offset) = frame.whole_strip() {
            let planes = frame.piece_planes();
            // `whole_strip` vetted `decoded_len`.
            let plane_bytes = frame.decoded_len().unwrap_or(0) / planes.max(1);
            return StripPlan {
                info: frame,
                per_plane: 1,
                rows_per_strip: height.max(1),
                height,
                available: planes,
                whole: Some((offset, plane_bytes as u64)),
            };
        }
        let rows_per_strip = (frame.rows_per_strip as usize).max(1);
        StripPlan {
            info: frame,
            per_plane: height.div_ceil(rows_per_strip),
            rows_per_strip,
            height,
            available: frame.strip_offsets.len().min(frame.strip_byte_counts.len()),
            whole: None,
        }
    }

    /// Strip `k` of piece-plane `plane`, which [`require`](Self::require) has
    /// checked is in the table.
    pub fn strip(&self, plane: usize, k: usize) -> Strip {
        if let Some((offset, plane_bytes)) = self.whole {
            return Strip {
                index: 0,
                row: 0,
                rows: self.height,
                offset: offset + plane as u64 * plane_bytes,
                len: plane_bytes,
            };
        }
        let row = k * self.rows_per_strip;
        let index = plane * self.per_plane + k;
        Strip {
            index,
            row,
            rows: self.rows_per_strip.min(self.height - row),
            offset: self.info.strip_offsets[index],
            len: self.info.strip_byte_counts[index],
        }
    }

    /// Fail unless every strip of `plane` is in the table.
    pub fn require(&self, plane: usize) -> Result<()> {
        let need = (plane + 1).saturating_mul(self.per_plane);
        if need > self.available {
            let have = self.available.saturating_sub(plane * self.per_plane);
            bail!(
                Malformed,
                "strips cover only {} of {} rows — strip data is shorter than the declared \
                 image size",
                (have * self.rows_per_strip).min(self.height),
                self.height
            );
        }
        Ok(())
    }
}

impl FrameInfo {
    /// The offset of the one uncompressed strip that holds this whole stripped
    /// frame, every plane of it, when it is stored that way: exactly one strip,
    /// at least as long as the decoded frame.
    ///
    /// The decoder has always read such a frame as one run, whatever its
    /// RowsPerStrip says and however many planes a planar frame packs into the
    /// strip, and without the strip-table size guard — its bytes are the frame,
    /// and they lie in the file. Writers do produce it, and it stays readable.
    pub(crate) fn whole_strip(&self) -> Option<u64> {
        if self.is_tiled() || self.compression != Compression::None || self.strip_offsets.len() != 1
        {
            return None;
        }
        let total = self.decoded_len().ok()? as u64;
        (self.strip_byte_counts.first().copied().unwrap_or(0) >= total)
            .then(|| self.strip_offsets[0])
    }
}

/// Piece buffers shared by the threads of one decode: taken by a task while it
/// works, given back when it is done, so a warm [`DecodeScratch`] holds one per
/// thread that has needed one, and a steady stream of frames allocates nothing.
///
/// [`DecodeScratch`]: crate::DecodeScratch
pub(super) struct Pool<'s> {
    bufs: Mutex<Vec<Vec<u8>>>,
    home: &'s mut Vec<Vec<u8>>,
}

impl<'s> Pool<'s> {
    pub fn new(home: &'s mut Vec<Vec<u8>>) -> Self {
        Pool {
            bufs: Mutex::new(std::mem::take(home)),
            home,
        }
    }

    pub fn take(&self) -> Vec<u8> {
        self.bufs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default()
    }

    /// Back into the pool, unless it never grew: an empty buffer is worth
    /// nothing to the next read, and pushing it into a cold scratch's empty
    /// list would be the one allocation a read that needs no buffer makes.
    pub fn give(&self, buf: Vec<u8>) {
        if buf.capacity() == 0 {
            return;
        }
        self.bufs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(buf);
    }
}

impl Drop for Pool<'_> {
    fn drop(&mut self) {
        let bufs = std::mem::take(self.bufs.get_mut().unwrap_or_else(|e| e.into_inner()));
        *self.home = bufs;
    }
}

/// The first `n` bytes of `buf`, growing it (zero-filled, fallibly) only when
/// it is shorter. A buffer that is long enough is handed back as it is: what
/// it held before is overwritten by the decode, never read.
pub(super) fn sized(buf: &mut Vec<u8>, n: usize) -> Result<&mut [u8]> {
    if buf.len() < n {
        buf.try_reserve_exact(n - buf.len()).map_err(|_| {
            err!(
                TooLarge,
                "cannot allocate {n} bytes to decode a piece of this frame"
            )
        })?;
        buf.resize(n, 0);
    }
    Ok(&mut buf[..n])
}

/// One strip's bytes into `dest`, which is exactly its rows: decompressed (or
/// copied, uncompressed), checked to have filled `dest`, and only then handed
/// to `finish` — the predictor undo and whatever else follows.
///
/// The check comes first on purpose. A strip that decodes short leaves the end
/// of `dest` holding whatever was there before, and the predictor would run
/// over it as if it were data; an `Err` is the only right answer, and nothing
/// after it should look at those bytes.
pub(super) fn decode_strip(
    src: &Src<'_, '_>,
    strip: &Strip,
    dest: &mut [u8],
    finish: impl FnOnce(&mut [u8]) -> Result<()>,
) -> Result<()> {
    let info = src.info;
    let (index, off, len) = (strip.index, strip.offset, strip.len);
    let n = if info.compression == Compression::None {
        let raw = src.bytes.piece(off, len.min(dest.len() as u64))?;
        dest[..raw.len()].copy_from_slice(raw);
        raw.len()
    } else {
        decompress_into(src.bytes.piece(off, len)?, info.compression, dest)?
    };
    if n != dest.len() {
        bail!(
            Malformed,
            "strip {index} decoded to {n} of the {} bytes its rows need — strip data is shorter \
             than the declared image size",
            dest.len()
        );
    }
    finish(dest)
}

/// The predictor undo for a strip of `src`'s frame, when it has one.
pub(super) fn undo(src: &Src<'_, '_>, sample_bytes: usize, data: &mut [u8]) -> Result<()> {
    if src.info.predictor != 1 {
        undo_predictor_in_place(data, src.info, sample_bytes, src.order)?;
    }
    Ok(())
}

/// A strip's samples, ready for conversion: the source's own bytes when they
/// already are (uncompressed, and either predictor-free or with the predictor
/// left to the caller), otherwise decoded into `buf`.
pub(super) fn strip_native<'b>(
    src: &Src<'b, '_>,
    strip: Strip,
    row_bytes: usize,
    sample_bytes: usize,
    undo_pred: bool,
    buf: &'b mut Vec<u8>,
) -> Result<&'b [u8]> {
    let info = src.info;
    let need = strip.rows * row_bytes;
    if info.compression == Compression::None && (info.predictor == 1 || !undo_pred) {
        let raw = src.bytes.piece(strip.offset, strip.len.min(need as u64))?;
        if raw.len() < need {
            bail!(
                Malformed,
                "strip data is shorter than the declared image size ({} of {need} bytes)",
                raw.len()
            );
        }
        return Ok(raw);
    }
    let dest = sized(buf, need)?;
    decode_strip(src, &strip, dest, |d| {
        if undo_pred {
            undo(src, sample_bytes, d)
        } else {
            Ok(())
        }
    })?;
    Ok(&buf[..need])
}

/// What [`for_strips`] runs per strip: the strip, a piece buffer of its own,
/// the group's part of every destination, and where in those parts the strip's
/// first row starts, in pixels.
pub(super) type EachStrip<'c, T> =
    dyn Fn(Strip, &mut Vec<u8>, &mut [&mut [T]], usize) -> Result<()> + Sync + 'c;

/// Run `each` over the strips of piece-plane `plane`, which land in `outs`
/// (every one a whole plane, `elems` elements per pixel), a group of strips per
/// task when `parallel`.
///
/// `each` gets a strip, a piece buffer of its own, the group's part of every
/// destination, and where in those parts the strip's first row starts, in
/// pixels. Groups are contiguous runs of strips, so a task's part of a
/// destination is one slice, and they are few — a handful per thread — so the
/// split costs a few dozen pointers, not one per strip.
#[allow(clippy::too_many_arguments)]
pub(super) fn for_strips<T: Send>(
    plan: &StripPlan<'_>,
    plane: usize,
    width: usize,
    elems: usize,
    outs: &mut [&mut [T]],
    parallel: bool,
    pool: &Pool<'_>,
    each: &EachStrip<'_, T>,
) -> Result<()> {
    plan.require(plane)?;
    let strips = plan.per_plane;
    let groups = if parallel {
        strips.min(rayon::current_num_threads().max(1) * 4)
    } else {
        1
    };
    if groups <= 1 {
        let mut buf = pool.take();
        let result = (0..strips).try_for_each(|k| {
            let strip = plan.strip(plane, k);
            each(strip, &mut buf, outs, strip.row * width)
        });
        pool.give(buf);
        return result;
    }

    // Strip boundaries of each group, then every destination cut at them.
    let bound = |g: usize| strips * g / groups;
    let row_of = |k: usize| (k * plan.rows_per_strip).min(plan.height);
    let k = outs.len();
    let mut cut: Vec<&mut [T]> = Vec::with_capacity(groups * k);
    {
        let mut columns: Vec<Vec<&mut [T]>> = outs
            .iter_mut()
            .map(|out| {
                let mut rest: &mut [T] = std::mem::take(out);
                let mut parts = Vec::with_capacity(groups);
                for g in 0..groups {
                    let rows = row_of(bound(g + 1)) - row_of(bound(g));
                    let (part, tail) = rest.split_at_mut(rows * width * elems);
                    parts.push(part);
                    rest = tail;
                }
                parts
            })
            .collect();
        for g in 0..groups {
            for column in columns.iter_mut() {
                cut.push(std::mem::take(&mut column[g]));
            }
        }
    }
    cut.par_chunks_mut(k)
        .enumerate()
        .try_for_each(|(g, parts)| {
            let mut buf = pool.take();
            let base_row = row_of(bound(g));
            let result = (bound(g)..bound(g + 1)).try_for_each(|kk| {
                let strip = plan.strip(plane, kk);
                each(strip, &mut buf, parts, (strip.row - base_row) * width)
            });
            pool.give(buf);
            result
        })
}

/// Whether the strips of a frame of `n_pixels` should be decompressed on
/// several cores: compressed, more than one strip, and big enough — the rule
/// the decoder has always followed.
pub(super) fn parallel_strips(frame: &FrameInfo, n_pixels: usize) -> bool {
    frame.compression != Compression::None
        && frame.strip_offsets.len() > 1
        && should_parallelize(n_pixels)
}

/// Copy a stored run into `dest` (the same length), a chunk at a time, `post`
/// applied to each chunk while it is still in cache.
///
/// In chunks rather than in one call because of how a large copy is done: one
/// `memcpy` of a whole 8 MiB frame streams it past the cache with
/// non-temporal stores, where a quarter-megabyte copy stores through it — and
/// the second is about 12% faster for a 2048x2048 `u16` frame on the machine
/// this was measured on (the same file read as 32 strips copied one by one
/// took 550 us against 624 for the single strip), because whatever reads the
/// destination next finds much of it still there.
fn copy_run(run: &[u8], dest: &mut [u8], post: &(dyn Fn(&mut [u8]) + Sync)) {
    const CHUNK: usize = 256 * 1024;
    debug_assert_eq!(run.len(), dest.len());
    for (d, s) in dest.chunks_mut(CHUNK).zip(run.chunks(CHUNK)) {
        d.copy_from_slice(s);
        post(d);
    }
}

/// Decode piece-planes `planes` of a stripped frame straight into `dest`, laid
/// out as the file lays them out, every strip decompressed into its own rows,
/// checked, its predictor undone (when `undo_pred`) and `post` applied while
/// it is still in cache. The route for a read whose output *is* the stored
/// representation, up to a byte swap or a sign flip.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_planes_into(
    src: &Src<'_, '_>,
    planes: std::ops::Range<usize>,
    sample_bytes: usize,
    per_px: usize,
    dest: &mut [u8],
    undo_pred: bool,
    post: &(dyn Fn(&mut [u8]) + Sync),
    pool: &Pool<'_>,
) -> Result<()> {
    let info = src.info;
    if let Some(run) = info.stored_run_planes(planes.clone()) {
        // Uncompressed, predictor-free and back to back: the run already is
        // the destination's bytes, whatever the strip table says about where
        // one strip stops. See `copy_run`.
        let run = src.bytes.piece(run.start, run.end - run.start)?;
        copy_run(run, dest, post);
        return Ok(());
    }
    let plan = StripPlan::of(info);
    let width = info.width as usize;
    let row_bytes = width * per_px * sample_bytes;
    let plane_bytes = plan.height * row_bytes;
    debug_assert_eq!(dest.len(), plane_bytes * planes.len());
    let parallel = parallel_strips(info, info.pixel_count()?);
    for (plane, part) in planes.zip(dest.chunks_mut(plane_bytes.max(1))) {
        for_strips(
            &plan,
            plane,
            width,
            per_px * sample_bytes,
            &mut [part],
            parallel,
            pool,
            &|strip, _buf, parts, at| {
                let at = at * per_px * sample_bytes;
                let d = &mut parts[0][at..at + strip.rows * row_bytes];
                decode_strip(src, &strip, d, |d| {
                    if undo_pred {
                        undo(src, sample_bytes, d)?;
                    }
                    post(d);
                    Ok(())
                })
            },
        )?;
    }
    Ok(())
}

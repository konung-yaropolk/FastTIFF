//! Tiled frames.
//!
//! Strips can be decompressed straight into the output because a strip's rows
//! are contiguous there. A tile's are not — it is a rectangle, so its rows land
//! `width` apart — so each tile is decompressed into a piece buffer and its
//! part of the image converted out of it a row at a time, while it is still in
//! cache. That extra copy is the price of tiles, and it buys the thing strips
//! cannot offer: reading a window of a huge image without touching the rest of
//! its rows.
//!
//! Two details separate a working tiled reader from a subtly broken one:
//!
//! - **edge tiles are stored full-size.** TIFF6 pads the tiles hanging off the
//!   right and bottom edges rather than trimming them, so a tile always
//!   decompresses to `tile_w * tile_h` samples and only part of it is used.
//!   Assume trimming and every row after the first tile column is shifted.
//! - **the predictor runs inside the tile**, over the tile's own width,
//!   including that padding — so it has to be undone per tile, before the
//!   conversion, not on the assembled frame. It is undone here through the
//!   shared routine, pointed at the tile's width, which keeps one
//!   implementation of the per-bit-depth differencing rather than a second one
//!   that can drift.

use super::codecs::{decompress_into, undo_predictor_rows};
use super::core::{Conv, Src};
use super::pieces::{sized, Pool, TileGrid};
use super::should_parallelize;
use crate::error::{bail, Result};
use crate::index::Compression;
use rayon::prelude::*;

/// Run `seg` over every row of every tile of piece-plane `plane` that lies in
/// the image, the tiles of a band (a row of tiles) one after another, bands in
/// parallel when the frame is big and compressed.
///
/// `outs` are whole planes of `elems` elements per pixel. `seg` gets one tile
/// row's part of the image (whole pixels of `px_bytes` each), the band's part
/// of every destination, the first destination plane, and where in the band
/// the row starts and how many pixels it holds.
#[allow(clippy::too_many_arguments)]
pub(super) fn for_tiles<T: Send>(
    src: &Src<'_, '_>,
    grid: &TileGrid,
    plane: usize,
    sample_bytes: usize,
    px_bytes: usize,
    first: usize,
    outs: &mut [&mut [T]],
    elems: usize,
    undo_pred: bool,
    pool: &Pool<'_>,
    seg: &Conv<'_, T>,
) -> Result<()> {
    let info = src.info;
    let width = info.width as usize;
    let height = info.height as usize;
    let spp = (info.samples_per_pixel as usize).max(1);
    let planar = info.is_planar();
    let TileGrid {
        across,
        down,
        tile_w,
        tile_h,
        tile_bytes,
        tile_row_bytes,
    } = *grid;
    let parallel = info.compression != Compression::None
        && info.piece_count() > 1
        && should_parallelize(info.pixel_count()?);

    // `parts` hold the band's rows from image row `base_row` on: the whole
    // planes when bands run one after another, a band each when in parallel.
    let band =
        |ty: usize, buf: &mut Vec<u8>, parts: &mut [&mut [T]], base_row: usize| -> Result<()> {
            let rows = tile_h.min(height - ty * tile_h);
            for tx in 0..across {
                let index = plane * (across * down) + ty * across + tx;
                let (offset, len) = (info.strip_offsets[index], info.strip_byte_counts[index]);
                let raw = src.bytes.piece(offset, len).map_err(|_| {
                    crate::error::err!(
                        Malformed,
                        "tile {index} at offset {offset} (len {len}) out of file bounds"
                    )
                })?;
                let tile = sized(buf, tile_bytes)?;
                // Zero first: a tile whose data runs short leaves the rest of its
                // rectangle blank rather than showing the previous tile.
                tile.fill(0);
                if decompress_into(raw, info.compression, tile)? == 0 {
                    bail!(
                        Malformed,
                        "tile {index} decoded to nothing ({len} bytes at offset {offset})"
                    );
                }
                if undo_pred && info.predictor != 1 {
                    undo_predictor_rows(
                        tile,
                        tile_w,
                        spp,
                        planar,
                        info.predictor,
                        sample_bytes,
                        src.order,
                    )?;
                }
                // The part of the tile that lies inside the image.
                let x0 = tx * tile_w;
                let keep = tile_w.min(width - x0);
                for r in 0..rows {
                    let at = r * tile_row_bytes;
                    seg(
                        &tile[at..at + keep * px_bytes],
                        parts,
                        first,
                        (ty * tile_h + r - base_row) * width + x0,
                        keep,
                    );
                }
            }
            Ok(())
        };

    if !parallel || down <= 1 {
        let mut buf = pool.take();
        let result = (0..down).try_for_each(|ty| band(ty, &mut buf, outs, 0));
        pool.give(buf);
        return result;
    }

    // One task per band: every tile in a band writes into it, at disjoint
    // columns, so bands are filled independently.
    let k = outs.len();
    let mut cut: Vec<&mut [T]> = Vec::with_capacity(down * k);
    {
        let mut columns: Vec<std::slice::ChunksMut<'_, T>> = outs
            .iter_mut()
            .map(|o| o.chunks_mut((tile_h * width * elems).max(1)))
            .collect();
        for _ in 0..down {
            for column in columns.iter_mut() {
                cut.push(column.next().unwrap_or_default());
            }
        }
    }
    cut.par_chunks_mut(k)
        .enumerate()
        .try_for_each(|(ty, parts)| {
            let mut buf = pool.take();
            let result = band(ty, &mut buf, parts, ty * tile_h);
            pool.give(buf);
            result
        })
}

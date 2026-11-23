//! The decode core: one implementation behind every read, choosing per call
//! between four routes.
//!
//! 1. **Borrow.** The plane is a stored run — uncompressed, predictor-free,
//!    stripped, every strip at least as long as its rows and starting where
//!    the last one ended — and the units want exactly what is stored, in the
//!    byte order it is stored in. The read is the source's own bytes.
//! 2. **Direct.** The units want what is stored up to a byte swap or a sign
//!    flip. Strips decompress straight into their rows of the destination, in
//!    parallel for a big compressed frame, and each gets its length checked,
//!    its predictor undone and the swap or flip applied while it is still in
//!    cache. A planar plane decodes only its own strips.
//! 3. **Convert per piece.** Anything that changes representation — 8-bit
//!    widening, integers to `f32`, chunky gathers, wide samples to `u16`
//!    with a range, 4-bit unpacking, CMYK. Each strip or tile is decompressed
//!    into a piece-sized buffer (or read in place, uncompressed) and converted
//!    into the destination rows straight away, while it is still in L2.
//! 4. **Assemble, then convert.** Only wide data to `u16` with no range: the
//!    plane's smallest and largest values have to be known before any sample
//!    can be converted, so the plane is assembled into the scratch first.
//!
//! Whatever the route, a destination is written in full or the read fails, and
//! is never read before it is written.

use super::codecs::decompress_into;
use super::convert::{self, MinMax};
use super::pieces::{
    bytes_for_bits, decode_planes_into, for_strips, guard_frame_size, guard_tiles, parallel_strips,
    sized, strip_native, Pool, StripPlan, TileGrid, MAX_DECODED_EXPANSION,
};
use super::should_parallelize;
use super::tiles::for_tiles;
use crate::display::DisplayMap;
use crate::error::{bail, ErrorKind, OptionContext, Result};
use crate::frame::{cast_run, DisplayMaps, FrameBytes};
use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, SampleFormat};
use crate::DecodeScratch;
use rayon::prelude::*;

/// A frame's bytes, its description and the file's byte order: what every
/// route reads from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Src<'a, 'f> {
    pub bytes: FrameBytes<'a>,
    pub info: &'f FrameInfo,
    pub order: ByteOrder,
}

impl<'a, 'f> Src<'a, 'f> {
    /// A frame of a whole file in memory.
    pub(crate) fn new(file: &'a [u8], info: &'f FrameInfo, order: ByteOrder) -> Self {
        Src {
            bytes: FrameBytes::whole(file),
            info,
            order,
        }
    }
}

/// A conversion from a piece's stored samples into the destinations:
/// `(native, parts, first_plane, at, n)` converts the `n` whole pixels of
/// `native` into elements `at..at + n` (in pixels) of every part, part `i`
/// being plane `first_plane + i` of the read.
pub(super) type Conv<'c, T> = dyn Fn(&[u8], &mut [&mut [T]], usize, usize, usize) + Sync + 'c;

/// The facts about a frame's layout every route needs, worked out once.
#[derive(Clone, Copy)]
struct Layout {
    sample_bytes: usize,
    spp: usize,
    /// Samples per pixel within one piece-plane: one for a planar frame.
    per_px: usize,
    planar: bool,
    width: usize,
    n_pixels: usize,
    signed: bool,
}

impl Layout {
    fn of(info: &FrameInfo, sample_bytes: usize) -> Result<Self> {
        let spp = (info.samples_per_pixel as usize).max(1);
        let planar = info.is_planar();
        Ok(Layout {
            sample_bytes,
            spp,
            per_px: if planar { 1 } else { spp },
            planar,
            width: info.width as usize,
            n_pixels: info.pixel_count()?,
            signed: info.sample_format == SampleFormat::SignedInt,
        })
    }

    /// Bytes one pixel of a piece-plane holds.
    fn px_bytes(&self) -> usize {
        self.per_px * self.sample_bytes
    }

    /// Where sample plane `plane` sits in a piece of native bytes:
    /// `(first sample, samples per pixel)` — interleaved in a chunky piece,
    /// the whole piece in a planar one.
    fn locate(&self, plane: usize) -> (usize, usize) {
        if self.per_px == 1 {
            (0, 1)
        } else {
            (plane, self.spp)
        }
    }

    /// The piece-plane sample plane `plane` lives in.
    fn piece_plane(&self, plane: usize) -> usize {
        if self.planar {
            plane
        } else {
            0
        }
    }
}

/// The size guards a frame of a decodable depth must pass before anything is
/// sized from its header. Returns the vetted tile grid for a tiled frame.
fn guard(info: &FrameInfo, sample_bytes: usize) -> Result<Option<TileGrid>> {
    info.decoded_len()?;
    if info.is_tiled() {
        guard_tiles(info, sample_bytes).map(Some)
    } else if info.whole_strip().is_some() {
        // One strip holding the whole frame: its bytes are the frame and they
        // lie in the file, which is all the guard would establish.
        Ok(None)
    } else {
        guard_frame_size(info).map(|_| None)
    }
}

/// The guards of whatever read `src` is about to get, run before a
/// destination is sized from its header. A depth no read decodes passes here
/// and fails in the read, with the read's own words.
pub(crate) fn vet(src: Src<'_, '_>) -> Result<()> {
    let info = src.info;
    if info.bits_per_sample == 4 {
        return guard_4bit(info).map(drop);
    }
    match bytes_for_bits(info.bits_per_sample) {
        Ok(sb) => guard(info, sb).map(drop),
        Err(_) => Ok(()),
    }
}

/// Convert every strip of piece-plane `piece` into `outs` (sample planes
/// `first..`, `elems` elements per pixel): each strip read in place or
/// decompressed into a piece buffer, its predictor undone when `undo_pred`,
/// then `conv` over its rows. `heavy` conversions are split across cores even
/// when the strips themselves are not.
#[allow(clippy::too_many_arguments)]
fn convert_strips<T: Send>(
    src: &Src<'_, '_>,
    lay: &Layout,
    piece: usize,
    first: usize,
    outs: &mut [&mut [T]],
    elems: usize,
    undo_pred: bool,
    heavy: bool,
    pool: &Pool<'_>,
    conv: &Conv<'_, T>,
) -> Result<()> {
    let info = src.info;
    let plan = StripPlan::of(info);
    let width = lay.width;
    let row_bytes = width * lay.px_bytes();
    let par_strips = parallel_strips(info, lay.n_pixels);
    let par_conv = !par_strips && heavy && should_parallelize(lay.n_pixels);
    let px_bytes = lay.px_bytes();
    for_strips(
        &plan,
        piece,
        width,
        elems,
        outs,
        par_strips,
        pool,
        &|strip, buf, parts, at| {
            let native = strip_native(src, strip, row_bytes, lay.sample_bytes, undo_pred, buf)?;
            let n = strip.rows * width;
            if par_conv {
                convert_parallel(native, parts, first, at, n, elems, px_bytes, conv);
            } else {
                conv(native, parts, first, at, n);
            }
            Ok(())
        },
    )
}

/// One piece's conversion over several cores: in chunks of pixels when it
/// fills one destination, a destination per task when it fills several.
#[allow(clippy::too_many_arguments)]
fn convert_parallel<T: Send>(
    native: &[u8],
    parts: &mut [&mut [T]],
    first: usize,
    at: usize,
    n: usize,
    elems: usize,
    px_bytes: usize,
    conv: &Conv<'_, T>,
) {
    const CHUNK: usize = 1 << 16;
    if let [only] = parts {
        only[at * elems..(at + n) * elems]
            .par_chunks_mut(CHUNK * elems)
            .zip(native.par_chunks(CHUNK * px_bytes))
            .for_each(|(d, s)| {
                let len = d.len() / elems;
                conv(s, &mut [d], first, 0, len)
            });
    } else {
        parts.par_iter_mut().enumerate().for_each(|(i, p)| {
            conv(native, std::slice::from_mut(p), first + i, at, n);
        });
    }
}

/// Every piece of sample planes `first..first + outs.len()`, converted by
/// `conv`: tiles through the tile reader, a chunky frame's strips in one pass
/// for all its planes, a planar frame's plane by plane.
#[allow(clippy::too_many_arguments)]
fn convert_planes<T: Send>(
    src: &Src<'_, '_>,
    lay: &Layout,
    grid: Option<&TileGrid>,
    first: usize,
    outs: &mut [&mut [T]],
    elems: usize,
    undo_pred: bool,
    heavy: bool,
    pool: &Pool<'_>,
    conv: &Conv<'_, T>,
) -> Result<()> {
    let planes: Vec<usize> = if lay.planar {
        (first..first + outs.len()).collect()
    } else {
        vec![first]
    };
    let mut rest: &mut [&mut [T]] = outs;
    for &p in &planes {
        let take = if lay.planar { 1 } else { rest.len() };
        let (these, tail) = std::mem::take(&mut rest).split_at_mut(take);
        rest = tail;
        match grid {
            Some(grid) => for_tiles(
                src,
                grid,
                lay.piece_plane(p),
                lay.sample_bytes,
                lay.px_bytes(),
                p,
                these,
                elems,
                undo_pred,
                pool,
                conv,
            )?,
            None => convert_strips(
                src,
                lay,
                lay.piece_plane(p),
                p,
                these,
                elems,
                undo_pred,
                heavy,
                pool,
                conv,
            )?,
        }
    }
    Ok(())
}

/// Plane `p`'s destination as the bytes a direct decode writes.
fn plane_bytes<T: bytemuck::Pod>(out: &mut [T]) -> &mut [u8] {
    bytemuck::cast_slice_mut(out)
}

/// The direct route for one-sample-per-pixel layouts (a single-sample frame,
/// or a planar plane): every destination is the stored plane, decoded straight
/// into it, `post` turning the stored bytes into the unit's.
fn direct_planes<T: bytemuck::Pod + Send>(
    src: &Src<'_, '_>,
    lay: &Layout,
    grid: Option<&TileGrid>,
    first: usize,
    outs: &mut [&mut [T]],
    pool: &Pool<'_>,
    post: &(dyn Fn(&mut [u8]) + Sync),
) -> Result<()> {
    debug_assert_eq!(lay.per_px, 1);
    let sb = lay.sample_bytes;
    let elems = sb / std::mem::size_of::<T>();
    match grid {
        Some(_) => {
            // Tiles have to be copied out of their piece buffer anyway; the
            // copy does what the post would.
            let copy = |seg: &[u8], parts: &mut [&mut [T]], _first: usize, at: usize, n: usize| {
                for part in parts.iter_mut() {
                    let d = plane_bytes(&mut part[at * elems..(at + n) * elems]);
                    d.copy_from_slice(&seg[..n * sb]);
                    post(d);
                }
            };
            convert_planes(src, lay, grid, first, outs, elems, true, false, pool, &copy)
        }
        None => {
            for (i, out) in outs.iter_mut().enumerate() {
                let piece = lay.piece_plane(first + i);
                decode_planes_into(
                    src,
                    piece..piece + 1,
                    sb,
                    1,
                    plane_bytes(out),
                    true,
                    post,
                    pool,
                )?;
            }
            Ok(())
        }
    }
}

/// The display readers have no widening for 4-bit samples, which are palette
/// indices (or sixteen levels of gray) that [`Stored`](crate::units::Stored)
/// and [`File`](crate::units::File) give as they are. So a single-sample 4-bit
/// frame here is a call to the wrong units, [`ErrorKind::InvalidInput`]; a
/// multi-sample one, which no read takes, the decode refuses as unsupported.
pub(crate) fn refuse_4bit_for_display(frame: &FrameInfo, method: &str) -> Result<()> {
    if frame.bits_per_sample == 4 && frame.samples_per_pixel <= 1 {
        bail!(
            InvalidInput,
            "{method} does not read 4-bit samples; read_plane_u8 and read_plane_f32 give them \
             as they are"
        );
    }
    Ok(())
}

/// Elements a [`Display`](crate::units::Display) plane holds, once the depth
/// is one it reads.
pub(crate) fn display_plane_len(info: &FrameInfo) -> Result<usize> {
    refuse_4bit_for_display(info, "a Display read")?;
    bytes_for_bits(info.bits_per_sample)?;
    info.pixel_count()
}

/// Elements a [`File`](crate::units::File) plane holds, once the depth is one
/// it reads.
pub(crate) fn file_plane_len(info: &FrameInfo) -> Result<usize> {
    if info.bits_per_sample == 4 {
        return guard_4bit(info).map(|(_, n)| n);
    }
    require_f32_samples(info, "a File read")?;
    info.pixel_count()
}

/// Elements a CMYK conversion's plane holds, once the frame is CMYK.
pub(crate) fn cmyk_plane_len(info: &FrameInfo, method: &str) -> Result<usize> {
    super::cmyk::require_cmyk(info, method)?;
    info.pixel_count()
}

/// [`Display`](crate::units::Display) planes `first..`.
pub(crate) fn display(
    src: Src<'_, '_>,
    range: Option<(f32, f32)>,
    first: usize,
    outs: &mut [&mut [u16]],
    scratch: &mut DecodeScratch,
) -> Result<DisplayMaps> {
    let info = src.info;
    refuse_4bit_for_display(info, "a Display read")?;
    let sb = bytes_for_bits(info.bits_per_sample)?;
    let grid = guard(info, sb)?;
    let lay = Layout::of(info, sb)?;
    let order = src.order;
    let signed = lay.signed;
    let DecodeScratch { native, pieces } = scratch;
    let pool = Pool::new(pieces);
    // Chunky multi-sample with Predictor 2: fuse the predictor undo into the
    // per-plane gather (one pass per plane over untouched bytes). Not for a
    // tiled frame: the fusion walks whole image rows, while a tiled frame's
    // predictor runs inside each tile — fusing would difference straight across
    // the seam between tile columns.
    let fuse = lay.per_px > 1 && info.predictor == 2 && grid.is_none() && sb <= 2;
    match sb {
        2 | 1 => {
            let map = DisplayMap::of(info, None).expect("8- and 16-bit frames have a fixed map");
            if sb == 2 && lay.per_px == 1 {
                let post = |d: &mut [u8]| convert::display16_in_place(d, order, signed);
                direct_planes(&src, &lay, grid.as_ref(), first, outs, &pool, &post)?;
            } else if fuse {
                let width = lay.width;
                let spp = lay.spp;
                let conv =
                    |native: &[u8], parts: &mut [&mut [u16]], first: usize, at: usize, n: usize| {
                        for (i, part) in parts.iter_mut().enumerate() {
                            convert::display_fused_pred2(
                                native,
                                sb,
                                first + i,
                                spp,
                                width,
                                order,
                                signed,
                                &mut part[at..at + n],
                            );
                        }
                    };
                convert_planes(&src, &lay, None, first, outs, 1, false, false, &pool, &conv)?;
            } else {
                let conv =
                    |native: &[u8], parts: &mut [&mut [u16]], first: usize, at: usize, n: usize| {
                        for (i, part) in parts.iter_mut().enumerate() {
                            let (base, stride) = lay.locate(first + i);
                            let d = &mut part[at..at + n];
                            if sb == 2 {
                                convert::display16(native, base, stride, order, signed, d);
                            } else {
                                convert::display8(native, base, stride, signed, d);
                            }
                        }
                    };
                convert_planes(
                    &src,
                    &lay,
                    grid.as_ref(),
                    first,
                    outs,
                    1,
                    true,
                    false,
                    &pool,
                    &conv,
                )?;
            }
            Ok(DisplayMaps::One(map))
        }
        _ => {
            let format = info.sample_format;
            if let Some(range) = range {
                let conv =
                    |native: &[u8], parts: &mut [&mut [u16]], first: usize, at: usize, n: usize| {
                        for (i, part) in parts.iter_mut().enumerate() {
                            let (base, stride) = lay.locate(first + i);
                            convert::display_wide(
                                native,
                                base,
                                stride,
                                order,
                                format,
                                sb,
                                range,
                                &mut part[at..at + n],
                            );
                        }
                    };
                convert_planes(
                    &src,
                    &lay,
                    grid.as_ref(),
                    first,
                    outs,
                    1,
                    true,
                    true,
                    &pool,
                    &conv,
                )?;
                return Ok(DisplayMaps::One(DisplayMap::Linear {
                    lo: range.0,
                    hi: range.1,
                }));
            }
            // No range: each plane is ranged to its own values, which have to
            // be known before any of it is converted. A chunky frame is
            // assembled once for all its planes, a planar one plane by plane.
            // One plane, one map: kept out of a `Vec` so a single-plane read
            // allocates nothing.
            let mut one = None;
            let mut maps = Vec::new();
            let mut record = |map: DisplayMap| match one.replace(map) {
                None => {}
                Some(earlier) if maps.is_empty() => maps.extend([earlier, map]),
                Some(_) => maps.push(map),
            };
            let auto = |bytes: &[u8], base: usize, stride: usize, out: &mut [u16]| {
                let mut mm = MinMax::new();
                mm.fold_wide(bytes, base, stride, lay.n_pixels, order, format, sb);
                let range = mm.range();
                // Parallelized across cores for large frames only (input is
                // read-only, writes disjoint).
                if should_parallelize(lay.n_pixels) {
                    const CHUNK: usize = 1 << 16;
                    out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, d)| {
                        let base = base + c * CHUNK * stride;
                        convert::display_wide(bytes, base, stride, order, format, sb, range, d)
                    });
                } else {
                    convert::display_wide(bytes, base, stride, order, format, sb, range, out);
                }
                DisplayMap::Linear {
                    lo: range.0,
                    hi: range.1,
                }
            };
            if lay.planar {
                for (i, out) in outs.iter_mut().enumerate() {
                    let bytes = assemble(&src, &lay, grid.as_ref(), first + i, native, &pool)?;
                    record(auto(bytes, 0, 1, out));
                }
            } else {
                let bytes = assemble(&src, &lay, grid.as_ref(), 0, native, &pool)?;
                for (i, out) in outs.iter_mut().enumerate() {
                    let (base, stride) = lay.locate(first + i);
                    record(auto(bytes, base, stride, out));
                }
            }
            Ok(match (one, maps.is_empty()) {
                (Some(map), true) => DisplayMaps::One(map),
                _ => DisplayMaps::PerPlane(maps),
            })
        }
    }
}

/// Piece-plane `piece`'s stored bytes in the file's byte order and layout,
/// predictor undone: the source's own when they are a stored run, otherwise
/// decoded into `native`, which grows but is never shrunk or zero-filled
/// again.
fn assemble<'n>(
    src: &Src<'n, '_>,
    lay: &Layout,
    grid: Option<&TileGrid>,
    piece: usize,
    native: &'n mut Vec<u8>,
    pool: &Pool<'_>,
) -> Result<&'n [u8]> {
    let info = src.info;
    if let Some(run) = info.stored_run(piece) {
        return src.bytes.piece(run.start, run.end - run.start);
    }
    let plane_bytes = lay
        .n_pixels
        .checked_mul(lay.px_bytes())
        .context(ErrorKind::TooLarge, "frame size overflows")?;
    let dest = sized(native, plane_bytes)?;
    assemble_into(
        src,
        lay,
        grid,
        piece..piece + 1,
        dest,
        pool,
        &|_: &mut [u8]| {},
    )?;
    Ok(&native[..plane_bytes])
}

/// Piece-planes `planes` decoded into `dest` in the file's layout, the
/// predictor undone and `post` applied to every run of bytes written.
fn assemble_into(
    src: &Src<'_, '_>,
    lay: &Layout,
    grid: Option<&TileGrid>,
    planes: std::ops::Range<usize>,
    dest: &mut [u8],
    pool: &Pool<'_>,
    post: &(dyn Fn(&mut [u8]) + Sync),
) -> Result<()> {
    let sb = lay.sample_bytes;
    match grid {
        None => decode_planes_into(src, planes, sb, lay.per_px, dest, true, post, pool),
        Some(grid) => {
            let px = lay.px_bytes();
            let plane_bytes = lay.n_pixels * px;
            for (p, part) in planes.zip(dest.chunks_mut(plane_bytes.max(1))) {
                let copy = |seg: &[u8], parts: &mut [&mut [u8]], _: usize, at: usize, n: usize| {
                    let d = &mut parts[0][at * px..(at + n) * px];
                    d.copy_from_slice(&seg[..n * px]);
                    post(d);
                };
                for_tiles(src, grid, p, sb, px, 0, &mut [part], px, true, pool, &copy)?;
            }
            Ok(())
        }
    }
}

/// Guard the f32 readers' precondition: a depth and format they convert. The
/// bit depth comes from an untrusted file and these are public entry points, so
/// anything else must be an error, never a panic or a misread. 16-bit floats
/// are refused rather than read as integers, which is what the display readers
/// do with them.
pub(crate) fn require_f32_samples(frame: &FrameInfo, method: &str) -> Result<()> {
    match (frame.bits_per_sample, frame.sample_format) {
        (8 | 16, SampleFormat::Float) => bail!(
            Unsupported,
            "{method} does not read {}-bit floating-point samples",
            frame.bits_per_sample
        ),
        (8 | 16 | 32 | 64, _) => Ok(()),
        (other, _) => bail!(Unsupported, "{method} cannot read {other}-bit samples"),
    }
}

/// [`File`](crate::units::File) planes `first..`.
pub(crate) fn file(
    src: Src<'_, '_>,
    first: usize,
    outs: &mut [&mut [f32]],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    let info = src.info;
    if info.bits_per_sample == 4 {
        // Palette indices, packed two to a byte: their own unpacker, one
        // plane only (multi-sample 4-bit is refused there).
        return four_bit(src, outs, scratch, |i| i as f32);
    }
    require_f32_samples(info, "a File read")?;
    let sb = bytes_for_bits(info.bits_per_sample)?;
    let grid = guard(info, sb)?;
    let lay = Layout::of(info, sb)?;
    let pool = Pool::new(&mut scratch.pieces);
    let order = src.order;
    let (bits, format) = (info.bits_per_sample, info.sample_format);
    if lay.per_px == 1 && bits == 32 && format == SampleFormat::Float {
        let swap = order != ByteOrder::host();
        let post = |d: &mut [u8]| {
            if swap {
                convert::swap_samples(d, 4)
            }
        };
        return direct_planes(&src, &lay, grid.as_ref(), first, outs, &pool, &post);
    }
    let conv = |native: &[u8], parts: &mut [&mut [f32]], first: usize, at: usize, n: usize| {
        for (i, part) in parts.iter_mut().enumerate() {
            let (base, stride) = lay.locate(first + i);
            convert::file_f32(
                native,
                base,
                stride,
                bits,
                format,
                order,
                &mut part[at..at + n],
            );
        }
    };
    convert_planes(
        &src,
        &lay,
        grid.as_ref(),
        first,
        outs,
        1,
        true,
        true,
        &pool,
        &conv,
    )
}

/// Bytes a [`Stored`](crate::units::Stored) plane holds.
pub(crate) fn stored_plane_len(info: &FrameInfo) -> Result<usize> {
    if info.bits_per_sample == 4 {
        return guard_4bit(info).map(|(_, n)| n);
    }
    let sb = bytes_for_bits(info.bits_per_sample)?;
    info.pixel_count()?
        .checked_mul(sb)
        .with_context(ErrorKind::TooLarge, || {
            format!(
                "frame geometry overflows address space: {}x{} x {sb} byte(s)/sample",
                info.width, info.height
            )
        })
}

/// [`Stored`](crate::units::Stored)`(want)` planes `first..`.
pub(crate) fn stored(
    src: Src<'_, '_>,
    want: ByteOrder,
    first: usize,
    outs: &mut [&mut [u8]],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    let info = src.info;
    if info.bits_per_sample == 4 {
        return four_bit(src, outs, scratch, |i| i);
    }
    let sb = bytes_for_bits(info.bits_per_sample)?;
    let grid = guard(info, sb)?;
    let lay = Layout::of(info, sb)?;
    let pool = Pool::new(&mut scratch.pieces);
    let swap = sb > 1 && want != src.order;
    if lay.per_px == 1 {
        let post = |d: &mut [u8]| {
            if swap {
                convert::swap_samples(d, sb)
            }
        };
        return direct_planes(&src, &lay, grid.as_ref(), first, outs, &pool, &post);
    }
    // Chunky: a gather per plane, or for 8-bit every plane in one pass.
    let (spp, width) = (lay.spp, lay.width);
    if sb == 1 && info.predictor == 2 && grid.is_none() {
        // The predictor-2 fusion, raw bytes (see `display`).
        let conv = |native: &[u8], parts: &mut [&mut [u8]], first: usize, at: usize, n: usize| {
            for (i, part) in parts.iter_mut().enumerate() {
                convert::u8_fused_pred2(native, first + i, spp, width, &mut part[at..at + n]);
            }
        };
        return convert_planes(&src, &lay, None, first, outs, 1, false, false, &pool, &conv);
    }
    let all = first == 0 && outs.len() == spp;
    let conv = |native: &[u8], parts: &mut [&mut [u8]], first: usize, at: usize, n: usize| {
        if sb == 1 && all && parts.len() == spp {
            // Chunky: split every plane in one pass over the source instead of
            // one strided pass per plane. The run is a whole number of pixels,
            // so it can only fail on a broken invariant.
            let _ = convert::deinterleave_into(native, spp, parts, at, n);
            return;
        }
        for (i, part) in parts.iter_mut().enumerate() {
            let (base, stride) = lay.locate(first + i);
            let _ = convert::stored(
                native,
                base,
                stride,
                sb,
                swap,
                &mut part[at * sb..(at + n) * sb],
            );
        }
    };
    convert_planes(
        &src,
        &lay,
        grid.as_ref(),
        first,
        outs,
        sb,
        true,
        false,
        &pool,
        &conv,
    )
}

/// The checks the 4-bit reader runs before it sizes anything: geometry that
/// fits, and strips that can cover and supply it, in 4-bit's packed units.
/// Returns `(packed bytes per row, pixels)`.
///
/// 4-bit geometry is *not* covered by `FrameInfo::decoded_len` (a 4-bit sample
/// has no whole byte size), so this does its own arithmetic and must guard it
/// too.
pub(crate) fn guard_4bit(frame: &FrameInfo) -> Result<(usize, usize)> {
    if frame.samples_per_pixel > 1 {
        bail!(Unsupported, "4-bit multi-sample images are not supported");
    }
    if frame.is_tiled() {
        bail!(Unsupported, "4-bit tiled images are not supported");
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let packed_row = width.div_ceil(2); // bytes per row (byte-aligned)
    let rows_per_strip = (frame.rows_per_strip as usize).max(1);
    let overflow = || format!("4-bit frame geometry overflows address space ({width}x{height})");
    let packed_len = packed_row
        .checked_mul(height)
        .with_context(ErrorKind::TooLarge, overflow)?;
    let out_len = width
        .checked_mul(height)
        .with_context(ErrorKind::TooLarge, overflow)?;
    // Both bounds from `guard_frame_size`, in 4-bit's packed units: what the
    // strips can structurally cover, and what they actually supply.
    let coverable = (frame.strip_offsets.len() as u64)
        .saturating_mul((rows_per_strip as u64).min(height as u64))
        .saturating_mul(packed_row as u64);
    let available = frame.strip_bytes_total();
    let supply = if frame.compression == Compression::None {
        available
    } else {
        available.saturating_mul(MAX_DECODED_EXPANSION)
    };
    if packed_len as u64 > coverable.min(supply) {
        bail!(
            Malformed,
            "4-bit frame declares {packed_len} packed bytes but its {} strip(s) supply only \
             {available} and cover only {coverable} — refusing to allocate",
            frame.strip_offsets.len()
        );
    }
    Ok((packed_row, out_len))
}

/// A **4-bit single-sample** frame, one value per pixel (each an index
/// 0..=15). 4-bit samples are packed two per byte, high nibble first (TIFF's
/// FillOrder-1 default). Rows are byte-aligned — a byte is never split across
/// rows — so an odd width leaves the last nibble of its final byte unused.
///
/// Used for palette-color TIFFs whose ColorMap is indexed by a 4-bit value.
/// A strip whose data runs short leaves the rest of its rows at index 0, as
/// this reader always has, and a predictor tag is ignored, as it always has
/// been.
fn four_bit<T: Copy>(
    src: Src<'_, '_>,
    outs: &mut [&mut [T]],
    scratch: &mut DecodeScratch,
    conv: impl Fn(u8) -> T,
) -> Result<()> {
    let info = src.info;
    let (packed_row, _) = guard_4bit(info)?;
    let [out] = outs else {
        bail!(InvalidInput, "a 4-bit frame has one plane");
    };
    let width = info.width as usize;
    let height = info.height as usize;
    let rows_per_strip = (info.rows_per_strip as usize).max(1);
    let pool = Pool::new(&mut scratch.pieces);
    let mut buf = pool.take();
    let mut rows_done = 0usize;
    let result = (|| -> Result<()> {
        for (&off, &len) in info.strip_offsets.iter().zip(info.strip_byte_counts.iter()) {
            if rows_done >= height {
                break;
            }
            let rows = rows_per_strip.min(height - rows_done);
            let need = rows * packed_row;
            let raw = src.bytes.piece_clamped(off, len)?;
            let packed: &[u8] = if info.compression == Compression::None && raw.len() >= need {
                &raw[..need]
            } else {
                let dst = sized(&mut buf, need)?;
                dst.fill(0);
                decompress_into(raw, info.compression, dst)?;
                &buf[..need]
            };
            let dst = &mut out[rows_done * width..(rows_done + rows) * width];
            for (prow, orow) in packed
                .chunks_exact(packed_row.max(1))
                .zip(dst.chunks_exact_mut(width.max(1)))
            {
                convert::unpack4(prow, orow, &conv);
            }
            rows_done += rows;
        }
        if rows_done < height {
            bail!(
                Malformed,
                "4-bit strips cover only {rows_done} of {height} rows"
            );
        }
        Ok(())
    })();
    pool.give(buf);
    result
}

/// What [`stored_frame_into`] writes, in bytes.
pub(crate) fn stored_frame_len(info: &FrameInfo) -> Result<usize> {
    if info.bits_per_sample == 4 {
        return guard_4bit(info).map(|(_, n)| n);
    }
    bytes_for_bits(info.bits_per_sample)?;
    info.decoded_len()
}

/// The whole frame as stored, in the file's layout, each sample in `want`'s
/// byte order (see [`Frame::stored_frame_into`](crate::Frame::stored_frame_into)).
/// `out` is [`stored_frame_len`] bytes.
pub(crate) fn stored_frame_into(
    src: Src<'_, '_>,
    want: ByteOrder,
    out: &mut [u8],
    scratch: &mut DecodeScratch,
) -> Result<()> {
    let info = src.info;
    if info.bits_per_sample == 4 {
        return four_bit(src, &mut [out], scratch, |i| i);
    }
    let sb = bytes_for_bits(info.bits_per_sample)?;
    let grid = guard(info, sb)?;
    let lay = Layout::of(info, sb)?;
    let pool = Pool::new(&mut scratch.pieces);
    let swap = sb > 1 && want != src.order;
    let post = |d: &mut [u8]| {
        if swap {
            convert::swap_samples(d, sb)
        }
    };
    assemble_into(
        &src,
        &lay,
        grid.as_ref(),
        0..info.piece_planes(),
        out,
        &pool,
        &post,
    )
}

/// The whole stored frame, borrowed, when it is one stored run in `want`'s
/// byte order.
pub(crate) fn borrow_stored_frame<'a>(src: Src<'a, '_>, want: ByteOrder) -> Option<&'a [u8]> {
    let info = src.info;
    let sb = bytes_for_bits(info.bits_per_sample).ok()?;
    if sb > 1 && want != src.order {
        return None;
    }
    let run = info.stored_run_planes(0..info.piece_planes())?;
    src.bytes.piece(run.start, run.end - run.start).ok()
}

/// The smallest and largest finite values of a 32- or 64-bit frame, over every
/// sample in file order (see [`Frame::value_range`](crate::Frame::value_range)).
pub(crate) fn value_range(
    src: Src<'_, '_>,
    scratch: &mut DecodeScratch,
) -> Result<Option<(f32, f32)>> {
    let info = src.info;
    // Only the wide (32/64-bit) formats auto-range this way; 8/16-bit use their
    // native integer min/max instead.
    if info.bits_per_sample != 32 && info.bits_per_sample != 64 {
        return Ok(None);
    }
    let sb = if info.bits_per_sample == 64 { 8 } else { 4 };
    let (order, format) = (src.order, info.sample_format);
    let mut mm = MinMax::new();
    // Folded in the order the samples are stored, a piece at a time — no
    // whole-frame temporary unless the frame has to be assembled anyway.
    if let Some(run) = info.stored_run_planes(0..info.piece_planes()) {
        let bytes = src.bytes.piece(run.start, run.end - run.start)?;
        let n = info.sample_count()?;
        mm.fold_wide(bytes, 0, 1, n, order, format, sb);
        return Ok(Some(mm.range()));
    }
    let grid = guard(info, sb)?;
    let lay = Layout::of(info, sb)?;
    let DecodeScratch { native, pieces } = scratch;
    let pool = Pool::new(pieces);
    if grid.is_some() || parallel_strips(info, lay.n_pixels) {
        // Tiles land out of order, and strips decompressed on several cores
        // have to land somewhere: assemble, then fold in order.
        let total = info.decoded_len()?;
        let dest = sized(native, total)?;
        assemble_into(
            &src,
            &lay,
            grid.as_ref(),
            0..info.piece_planes(),
            dest,
            &pool,
            &|_: &mut [u8]| {},
        )?;
        mm.fold_wide(&native[..total], 0, 1, total / sb, order, format, sb);
        return Ok(Some(mm.range()));
    }
    let plan = StripPlan::of(info);
    let row_bytes = lay.width * lay.px_bytes();
    let mut buf = pool.take();
    let result = (|| -> Result<()> {
        for piece in 0..info.piece_planes() {
            plan.require(piece)?;
            for k in 0..plan.per_plane {
                let strip = plan.strip(piece, k);
                let native = strip_native(&src, strip, row_bytes, sb, true, &mut buf)?;
                mm.fold_wide(native, 0, 1, native.len() / sb, order, format, sb);
            }
        }
        Ok(())
    })();
    pool.give(buf);
    result.map(|()| Some(mm.range()))
}

/// The run plane `plane` of a read is, when it is one: a single-sample
/// frame's only plane or a planar frame's sample plane.
fn plane_run(info: &FrameInfo, plane: usize) -> Option<std::ops::Range<u64>> {
    let spp = (info.samples_per_pixel as usize).max(1);
    if plane >= spp {
        return None;
    }
    if spp == 1 {
        info.stored_run(0)
    } else if info.is_planar() {
        info.stored_run(plane)
    } else {
        None
    }
}

fn borrow_run<'a, T: bytemuck::Pod>(src: &Src<'a, '_>, plane: usize) -> Option<&'a [T]> {
    let run = plane_run(src.info, plane)?;
    cast_run(src.bytes.piece(run.start, run.end - run.start).ok()?)
}

/// A [`Display`](crate::units::Display) plane borrowed: unsigned 16-bit in
/// the host's byte order.
pub(crate) fn borrow_display<'a>(src: Src<'a, '_>, plane: usize) -> Option<&'a [u16]> {
    let info = src.info;
    // Signed frames are excluded: they need the +32768 offset, so they can't be
    // a zero-copy reinterpret.
    if info.bits_per_sample != 16
        || info.sample_format == SampleFormat::SignedInt
        || src.order != ByteOrder::host()
    {
        return None;
    }
    borrow_run(&src, plane)
}

/// A [`File`](crate::units::File) plane borrowed: 32-bit float in the host's
/// byte order.
pub(crate) fn borrow_file<'a>(src: Src<'a, '_>, plane: usize) -> Option<&'a [f32]> {
    let info = src.info;
    if info.bits_per_sample != 32
        || info.sample_format != SampleFormat::Float
        || src.order != ByteOrder::host()
    {
        return None;
    }
    borrow_run(&src, plane)
}

/// A [`Stored`](crate::units::Stored)`(want)` plane borrowed: any whole-byte
/// depth, in the byte order it is stored in.
pub(crate) fn borrow_stored<'a>(
    src: Src<'a, '_>,
    want: ByteOrder,
    plane: usize,
) -> Option<&'a [u8]> {
    let sb = bytes_for_bits(src.info.bits_per_sample).ok()?;
    if sb > 1 && want != src.order {
        return None;
    }
    borrow_run(&src, plane)
}

/// What the CMYK conversions need from here.
pub(super) struct Inks<'s, 'a, 'f> {
    pub src: Src<'a, 'f>,
    pub scratch: &'s mut DecodeScratch,
}

impl Inks<'_, '_, '_> {
    /// Convert every pixel of the frame's first four samples (C, M, Y, K, each
    /// read through `ink` from its stored bytes) into planes `first..` of
    /// red, green and blue, through `comp(ink, black)`.
    pub fn convert<T: Send + Copy>(
        self,
        first: usize,
        outs: &mut [&mut [T]],
        ink: &(dyn Fn(&[u8]) -> u32 + Sync),
        comp: &(dyn Fn(u32, u32) -> T + Sync),
    ) -> Result<()> {
        let src = self.src;
        let info = src.info;
        let sb = bytes_for_bits(info.bits_per_sample)?;
        let grid = guard(info, sb)?;
        let lay = Layout::of(info, sb)?;
        let DecodeScratch { native, pieces } = self.scratch;
        let pool = Pool::new(pieces);
        if !lay.planar {
            let spp = lay.spp;
            let conv =
                |native: &[u8], parts: &mut [&mut [T]], first: usize, at: usize, n: usize| {
                    for (j, px) in native.chunks_exact(spp * sb).take(n).enumerate() {
                        let k = ink(&px[3 * sb..4 * sb]);
                        for (i, part) in parts.iter_mut().enumerate() {
                            let c = first + i;
                            part[at + j] = comp(ink(&px[c * sb..(c + 1) * sb]), k);
                        }
                    }
                };
            // `convert_planes` takes chunky planes from `first` on; the ink
            // offsets above are absolute, so start the walk at plane 0 and let
            // `conv` see the real first plane through the closure argument.
            return match grid.as_ref() {
                Some(grid) => for_tiles(
                    &src,
                    grid,
                    0,
                    sb,
                    lay.px_bytes(),
                    first,
                    outs,
                    1,
                    true,
                    &pool,
                    &conv,
                ),
                None => convert_strips(&src, &lay, 0, first, outs, 1, true, false, &pool, &conv),
            };
        }
        // Planar: the four ink planes are separate runs. Assemble them (or
        // borrow them, uncompressed), then convert a pixel at a time.
        let plane_bytes = lay.n_pixels * sb;
        let inks: &[u8] = match info.stored_run_planes(0..4) {
            Some(run) => src.bytes.piece(run.start, run.end - run.start)?,
            None => {
                let total = plane_bytes
                    .checked_mul(4)
                    .context(ErrorKind::TooLarge, "frame size overflows")?;
                let dest = sized(native, total)?;
                assemble_into(
                    &src,
                    &lay,
                    grid.as_ref(),
                    0..4,
                    dest,
                    &pool,
                    &|_: &mut [u8]| {},
                )?;
                &native[..total]
            }
        };
        let (c, rest) = inks.split_at(plane_bytes);
        let (m, rest) = rest.split_at(plane_bytes);
        let (y, k) = rest.split_at(plane_bytes);
        let planes = [c, m, y];
        for (i, out) in outs.iter_mut().enumerate() {
            let ink_plane = planes[first + i];
            for (j, o) in out.iter_mut().enumerate() {
                let black = ink(&k[j * sb..(j + 1) * sb]);
                *o = comp(ink(&ink_plane[j * sb..(j + 1) * sb]), black);
            }
        }
        Ok(())
    }
}

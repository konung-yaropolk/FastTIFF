//! Turns a `FrameInfo` into actual pixel data. The fast path — uncompressed
//! strips, native byte order, single strip per frame (ImageJ's default when
//! saving raw stacks) — does zero decoding work at all: it's a direct
//! reinterpret-cast of the memory-mapped file bytes. Everything else
//! (LZW/Deflate/PackBits, multi-strip, predictor, byte-swap, 8-bit upcast,
//! 32-bit float) falls back to an owned `Vec<u16>`.

use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, SampleFormat, TiffStack};
use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

/// Floor on frame size (pixels) below which a decode is never split across
/// cores: too small for rayon's fork-join to pay off, and such frames decode
/// fast enough that they're never the playback bottleneck.
const PARALLEL_MIN_PIXELS: usize = 1024 * 1024;

/// Process-wide hint: split large decodes across cores when `true`, run them
/// serially when `false` (the default). Parallel decode spreads load across
/// cores but uses *more total CPU* (rayon overhead), so it's only worth it when
/// a single core can't keep up. The caller (the viewer) turns it on only while
/// real-time playback is falling behind — see `set_parallel_decode`.
static PARALLEL_DECODE: AtomicBool = AtomicBool::new(false);

/// Enable/disable parallel decoding *and* encoding. A performance hint, not a
/// correctness knob: the pixels are identical either way. The viewer flips this
/// on when playback starts dropping frames (one core saturating on decode) and
/// off otherwise, so steady-state playback that keeps up stays on the cheaper
/// serial path. The same hint governs the writer's per-strip compression (see
/// `encode`), so a host application has one switch for all CPU-heavy pixel
/// work. Off by default.
pub fn set_parallel_decode(enabled: bool) {
    PARALLEL_DECODE.store(enabled, Ordering::Relaxed);
}

/// Whether a CPU-heavy pass should use rayon: only when the caller asked for it
/// *and* the frame is big enough for the speedup to beat the fork-join cost.
/// Shared with `encode` so both directions honor the one hint.
pub(crate) fn should_parallelize(n_pixels: usize) -> bool {
    PARALLEL_DECODE.load(Ordering::Relaxed) && n_pixels >= PARALLEL_MIN_PIXELS
}

/// Decoded pixel data for one plane, always exposed as 16-bit samples:
/// - 8-bit sources are upcast (`v -> (v << 8) | v`, mapping 0..255 evenly
///   onto 0..65535) so the GPU window/level math stays uniform.
/// - 32-bit float sources are linearly rescaled into 0..65535 using
///   `float_range` (the channel's display range, in the float data's own
///   units — matching how ImageJ treats float images: contrast is defined
///   over the data's actual value range, not assumed to already be
///   16-bit-integer-shaped). Pass `None` to auto-range from this frame's
///   own min/max (e.g. for an initial probe before a stable per-channel
///   range has been established — see `frame_float_minmax`). Ignored for
///   non-float sources.
pub fn read_frame_u16<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
) -> Result<Cow<'a, [u16]>> {
    let n_samples = frame.width as usize * frame.height as usize * frame.samples_per_pixel as usize;

    // --- Fast path: uncompressed, single strip, native 16-bit, native byte order ---
    // Signed-int frames are excluded: they need the +32768 offset applied
    // below (see the `16 =>` arm), so they can't be a zero-copy reinterpret.
    // Predictor-differenced frames (legal even without compression) are
    // excluded too: their bytes need the undo pass first.
    if frame.compression == Compression::None
        && frame.bits_per_sample == 16
        && frame.samples_per_pixel == 1
        && frame.sample_format != SampleFormat::SignedInt
        && frame.predictor == 1
        && file_order == ByteOrder::host()
        && frame.strip_offsets.len() == 1
    {
        let offset = frame.strip_offsets[0] as usize;
        let len_bytes = n_samples * 2;
        let slice = mmap
            .get(offset..offset + len_bytes)
            .ok_or_else(|| anyhow!("strip data out of file bounds"))?;
        // bytemuck guarantees this cast is sound: u8 -> u16 reinterpret of
        // an aligned, correctly-sized byte slice. mmap pages are at least
        // 2-byte aligned (page-aligned, in fact), so alignment holds.
        if let Ok(samples) = bytemuck::try_cast_slice::<u8, u16>(slice) {
            return Ok(Cow::Borrowed(samples));
        }
        // Misaligned (rare) — fall through to the owned path.
    }

    // --- General path: decode plane 0 (the whole image for single-sample
    // frames). Multi-sample (RGB) frames are deinterleaved per-plane by
    // `read_plane_u16`, which callers use directly. ---
    Ok(Cow::Owned(read_plane_u16(mmap, frame, file_order, float_range, 0)?))
}

/// Decode an **unsigned 8-bit, single-sample** frame's raw bytes, *without* the
/// `0..255 -> 0..65535` widening `read_frame_u16` does for 8-bit. Zero-copy
/// (a borrow over the memory map) for the uncompressed, single-strip case;
/// otherwise an owned `Vec<u8>` (decompressed, predictor undone). Intended for
/// upload to an `R8Uint` GPU texture so the widening never touches the CPU on
/// the per-frame hot path — the caller scales the window/level into 0..255 units
/// instead (dividing by 257). Only valid for `bits_per_sample == 8`,
/// `samples_per_pixel == 1`, `UnsignedInt`; callers gate on that.
pub fn read_frame_u8<'a>(mmap: &'a [u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Cow<'a, [u8]>> {
    decode_native_bytes(mmap, frame, file_order)
}

/// Decode a single sample plane (`plane`, `< samples_per_pixel`) of a frame
/// into `width * height` u16 values, deinterleaving chunky multi-sample data
/// such as RGB. For single-sample frames `plane` is 0 and this returns the
/// whole image. Handling per bit depth:
/// - 8-bit is upcast (`v -> (v << 8) | v`) onto 0..65535.
/// - signed integers are offset into ImageJ's unsigned display space (sign-bit
///   flip), so signed and unsigned+calibration files render the same.
/// - 32-bit samples (int *or* float) are linearly rescaled into 0..65535 using
///   `float_range` (or this plane's own min/max when `None`).
pub fn read_plane_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    plane: usize,
) -> Result<Vec<u16>> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let n_pixels = frame.width as usize * frame.height as usize;
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let signed = frame.sample_format == SampleFormat::SignedInt;
    let mut out = vec![0u16; n_pixels];

    match frame.bits_per_sample {
        16 => {
            if spp == 1 {
                // Contiguous samples — the per-frame hot path for compressed
                // 16-bit playback. Bulk-convert with the byte-order branch
                // hoisted out of the loop and no strided indexing (the
                // sign-bit flip folds in branchlessly: xor with 0 is a no-op).
                let flip = if signed { 0x8000 } else { 0 };
                let pairs = native[..n_pixels * 2].chunks_exact(2);
                match file_order {
                    ByteOrder::Little => {
                        for (o, c) in out.iter_mut().zip(pairs) {
                            *o = u16::from_le_bytes([c[0], c[1]]) ^ flip;
                        }
                    }
                    ByteOrder::Big => {
                        for (o, c) in out.iter_mut().zip(pairs) {
                            *o = u16::from_be_bytes([c[0], c[1]]) ^ flip;
                        }
                    }
                }
            } else {
                for (i, o) in out.iter_mut().enumerate() {
                    let off = (i * spp + plane) * 2;
                    let v = file_order.u16(&native[off..off + 2]);
                    *o = if signed { v ^ 0x8000 } else { v };
                }
            }
        }
        8 => {
            if spp == 1 {
                let flip = if signed { 0x80 } else { 0 };
                for (o, &b) in out.iter_mut().zip(native[..n_pixels].iter()) {
                    let b = b ^ flip;
                    *o = ((b as u16) << 8) | b as u16;
                }
            } else {
                for (i, o) in out.iter_mut().enumerate() {
                    let b = native[i * spp + plane];
                    let b = if signed { b ^ 0x80 } else { b };
                    *o = ((b as u16) << 8) | b as u16;
                }
            }
        }
        32 => {
            // The heaviest per-pixel path: read each 32-bit sample as f32, then
            // linearly rescale into 0..65535. Parallelized across cores for large
            // frames only (input is read-only, output writes are disjoint).
            let format = frame.sample_format;
            let parallel = should_parallelize(n_pixels);
            let floats: Vec<f32> = if parallel {
                (0..n_pixels)
                    .into_par_iter()
                    .map(|i| sample_f32(&native[(i * spp + plane) * 4..], file_order, format))
                    .collect()
            } else {
                (0..n_pixels)
                    .map(|i| sample_f32(&native[(i * spp + plane) * 4..], file_order, format))
                    .collect()
            };
            let (lo, hi) = float_range.unwrap_or_else(|| minmax_f32(&floats));
            let span = (hi - lo).max(f32::EPSILON);
            if parallel {
                out.par_iter_mut().zip(floats.par_iter()).for_each(|(o, &v)| {
                    let t = ((v - lo) / span).clamp(0.0, 1.0);
                    *o = (t * 65535.0).round() as u16;
                });
            } else {
                for (o, &v) in out.iter_mut().zip(floats.iter()) {
                    let t = ((v - lo) / span).clamp(0.0, 1.0);
                    *o = (t * 65535.0).round() as u16;
                }
            }
        }
        other => bail!("unsupported bits_per_sample: {other}"),
    }
    Ok(out)
}

/// Decode a single sample plane (`plane`, `< samples_per_pixel`) of a frame as
/// **raw unsigned 8-bit bytes**, deinterleaving chunky multi-sample data such as
/// 8-bit RGB. The `u8` sibling of [`read_plane_u16`]: same strided gather, but it
/// keeps the bytes at native 8-bit width (no widening to 0..65535) for callers
/// that upload to an `R8Uint` texture and scale to 0..255 themselves — mirroring
/// [`read_frame_u8`].
///
/// Always allocates: a chunky plane is a strided gather, so unlike
/// [`read_frame_u8`] there's no zero-copy borrow. Only valid for unsigned 8-bit
/// frames (`bits_per_sample == 8`); callers gate on that.
pub fn read_plane_u8(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder, plane: usize) -> Result<Vec<u8>> {
    if frame.bits_per_sample != 8 {
        bail!("read_plane_u8 requires 8-bit samples, got {}", frame.bits_per_sample);
    }
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let n_pixels = frame.width as usize * frame.height as usize;
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let mut out = vec![0u8; n_pixels];
    for (i, o) in out.iter_mut().enumerate() {
        *o = native[i * spp + plane];
    }
    Ok(out)
}

/// Decode a 32-bit-float frame's samples as **raw `f32`**, without the
/// rescale-to-u16 step `read_frame_u16` does. This is for channels uploaded to
/// a float (R32F) GPU texture, where window/level happens on the GPU — so the
/// per-frame CPU cost drops to a borrow (fast path) or a single decode pass.
///
/// Zero-copy (`Cow::Borrowed` over the memory map) when the data is
/// uncompressed, 32-bit float, single-sample, native byte order and a single
/// strip; otherwise an owned `Vec<f32>`. Mirrors `read_frame_u16`'s fast path.
/// Only valid for 32-bit data (4 bytes/sample); callers use it for float
/// channels.
pub fn read_frame_f32<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Cow<'a, [f32]>> {
    let n_pixels = frame.width as usize * frame.height as usize;

    if frame.compression == Compression::None
        && frame.bits_per_sample == 32
        && frame.sample_format == SampleFormat::Float
        && frame.samples_per_pixel == 1
        // Predictor-differenced data (legal even uncompressed) needs its undo
        // pass, so it can't be borrowed raw.
        && frame.predictor == 1
        && file_order == ByteOrder::host()
        && frame.strip_offsets.len() == 1
    {
        let offset = frame.strip_offsets[0] as usize;
        let len_bytes = n_pixels * 4;
        let slice = mmap
            .get(offset..offset + len_bytes)
            .ok_or_else(|| anyhow!("strip data out of file bounds"))?;
        if let Ok(samples) = bytemuck::try_cast_slice::<u8, f32>(slice) {
            return Ok(Cow::Borrowed(samples));
        }
        // Misaligned (rare) — fall through to the owned path.
    }

    Ok(Cow::Owned(read_plane_f32(mmap, frame, file_order, 0)?))
}

/// Decode a single sample plane of a 32-bit frame into raw `f32` values
/// (deinterleaving chunky multi-sample data), with integer 32-bit samples cast
/// to `f32`. The per-pixel read is parallelized across cores.
pub fn read_plane_f32(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
) -> Result<Vec<f32>> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let n_pixels = frame.width as usize * frame.height as usize;
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let format = frame.sample_format;
    Ok(if should_parallelize(n_pixels) {
        (0..n_pixels)
            .into_par_iter()
            .map(|i| sample_f32(&native[(i * spp + plane) * 4..], file_order, format))
            .collect()
    } else {
        (0..n_pixels)
            .map(|i| sample_f32(&native[(i * spp + plane) * 4..], file_order, format))
            .collect()
    })
}

/// Eagerly decode **every** frame to 16-bit samples, in parallel *across*
/// frames (rayon). This is the load-everything-at-once counterpart to the lazy
/// per-frame [`read_frame_u16`]: it returns one owned `Vec<u16>` per frame, all
/// resident in memory. Useful when a consumer genuinely needs the whole stack in
/// RAM (e.g. batch processing) rather than scrubbing it frame by frame.
///
/// Each frame decodes its plane 0 (the whole image for single-sample frames);
/// `float_range` is applied to every frame the same way `read_frame_u16` uses
/// it. Parallelism here is across frames and is independent of the
/// [`set_parallel_decode`] hint (which controls *within*-frame threading) — for
/// a large stack the across-frame split is what scales, so leaving the hint off
/// avoids redundant nested parallelism.
///
/// Note the memory cost: this holds the entire decoded stack at once
/// (`frames * width * height * 2` bytes).
pub fn preload_frames_u16(stack: &TiffStack, float_range: Option<(f32, f32)>) -> Result<Vec<Vec<u16>>> {
    stack
        .frames
        .par_iter()
        .map(|frame| Ok(read_frame_u16(&stack.mmap, frame, stack.byte_order, float_range)?.into_owned()))
        .collect()
}

/// Eagerly decode every frame to raw `f32`, in parallel across frames — the
/// float counterpart to [`preload_frames_u16`] (for 32-bit-float stacks, no
/// rescaling). See that function for the parallelism/memory notes.
pub fn preload_frames_f32(stack: &TiffStack) -> Result<Vec<Vec<f32>>> {
    stack
        .frames
        .par_iter()
        .map(|frame| Ok(read_frame_f32(&stack.mmap, frame, stack.byte_order)?.into_owned()))
        .collect()
}

/// Eagerly decode every frame to raw 8-bit bytes, in parallel across frames —
/// the byte counterpart to [`preload_frames_u16`], at half the memory (1 byte
/// per pixel, no widening). See that function for the parallelism/memory notes.
///
/// Only meaningful for **unsigned single-sample 8-bit** stacks (the same data
/// [`read_frame_u8`] handles); gate on `bits_per_sample == 8` before calling.
/// For other formats it returns the frame's raw native bytes as-is, which won't
/// be the display-ready samples — use [`preload_frames_u16`] instead.
pub fn preload_frames_u8(stack: &TiffStack) -> Result<Vec<Vec<u8>>> {
    stack
        .frames
        .par_iter()
        .map(|frame| Ok(read_frame_u8(&stack.mmap, frame, stack.byte_order)?.into_owned()))
        .collect()
}

/// The actual min/max of a 32-bit frame's raw values (int or float) — used to
/// establish a channel's initial display range, the same way ImageJ
/// auto-ranges a 32-bit image to its own data instead of assuming a fixed
/// scale. `None` for non-32-bit frames (8/16-bit use their native integer
/// min/max instead).
pub fn frame_float_minmax(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Option<(f32, f32)>> {
    if frame.bits_per_sample != 32 {
        return Ok(None);
    }
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let n_samples = frame.width as usize * frame.height as usize * frame.samples_per_pixel as usize;
    // Fold directly over the decoded bytes — no width*height*4 temporary.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for chunk in native[..n_samples * 4].chunks_exact(4) {
        let v = sample_f32(chunk, file_order, frame.sample_format);
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    // Same degenerate-data fallback as `minmax_f32`.
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        Ok(Some((0.0, 1.0)))
    } else {
        Ok(Some((lo, hi)))
    }
}

/// Reads one 32-bit sample as `f32`, interpreting the 4 bytes per the frame's
/// sample format (IEEE float, signed, or unsigned integer) and byte order.
fn sample_f32(chunk: &[u8], file_order: ByteOrder, format: SampleFormat) -> f32 {
    let arr: [u8; 4] = chunk[..4].try_into().unwrap();
    match (format, file_order) {
        (SampleFormat::Float, ByteOrder::Little) => f32::from_le_bytes(arr),
        (SampleFormat::Float, ByteOrder::Big) => f32::from_be_bytes(arr),
        (SampleFormat::SignedInt, ByteOrder::Little) => i32::from_le_bytes(arr) as f32,
        (SampleFormat::SignedInt, ByteOrder::Big) => i32::from_be_bytes(arr) as f32,
        (SampleFormat::UnsignedInt, ByteOrder::Little) => u32::from_le_bytes(arr) as f32,
        (SampleFormat::UnsignedInt, ByteOrder::Big) => u32::from_be_bytes(arr) as f32,
    }
}

fn minmax_f32(values: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        (0.0, 1.0) // empty / all-NaN / constant data -- avoid a zero-width window downstream
    } else {
        (lo, hi)
    }
}

/// Assembles a frame's full pixel data in native sample units (still
/// subject to byte-order interpretation, and with TIFF Predictor=2 already
/// undone). Decompresses **each strip independently** before concatenating.
///
/// TIFF strips are independently-compressed units — each one has its own
/// LZW dictionary / zlib stream, started fresh. Concatenating *compressed*
/// bytes across strips and decompressing once (what an earlier version of
/// this function did) feeds the decoder strip 2's stream header as if it
/// were a continuation of strip 1's stream, which is invalid input; the
/// decoder stops there, silently truncating the result to roughly strip
/// 1's worth of data. For a typical 2-strip image that's exactly "only the
/// top half is shown" — this is what that bug looked like in practice.
fn decode_native_bytes<'a>(mmap: &'a [u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Cow<'a, [u8]>> {
    let sample_bytes = match frame.bits_per_sample {
        16 => 2,
        8 => 1,
        32 => 4,
        other => bail!("unsupported bits_per_sample: {other}"),
    };
    let row_bytes = frame.width as usize * frame.samples_per_pixel as usize * sample_bytes;
    let total_rows = frame.height as usize;
    let total_len = total_rows * row_bytes;
    let rows_per_strip = (frame.rows_per_strip as usize).max(1);

    // Fast path: a single uncompressed strip that already covers the whole
    // image can be read straight out of the memory map — no intermediate copy.
    // (The 16-bit/native-order case is borrowed even more cheaply as `u16` by
    // `read_frame_u16`; this covers 8-bit, 32-bit, RGB and byte-swapped data.)
    if frame.compression == Compression::None
        && frame.strip_offsets.len() == 1
        && frame.strip_byte_counts.first().copied().unwrap_or(0) as usize >= total_len
    {
        let offset = frame.strip_offsets[0] as usize;
        let slice = mmap
            .get(offset..offset + total_len)
            .ok_or_else(|| anyhow!("strip data out of file bounds"))?;
        return if frame.predictor != 1 {
            // Predictor undo mutates in place, so it needs an owned copy.
            Ok(Cow::Owned(undo_predictor(slice.to_vec(), frame, sample_bytes, file_order)?))
        } else {
            Ok(Cow::Borrowed(slice))
        };
    }

    // General path: multi-strip and/or compressed — assemble into an owned
    // buffer, decompressing each strip independently (see the doc comment). The
    // last strip may legitimately have fewer rows than `rows_per_strip` when the
    // image height doesn't divide evenly.
    let compression = frame.compression;
    let n_pixels = frame.width as usize * frame.height as usize;
    let mut native = Vec::with_capacity(total_len);
    // Strips are independent compressed units, so for *large* compressed frames
    // we decompress them in parallel (rayon's ordered `collect` keeps row
    // order; each strip's row span comes from its index, so the map is pure).
    // Small/medium frames stay serial: the fork-join overhead would otherwise
    // cost more total CPU than it saves during fast playback.
    let parallel =
        compression != Compression::None && frame.strip_offsets.len() > 1 && should_parallelize(n_pixels);
    if parallel {
        let strips: Vec<Vec<u8>> = frame
            .strip_offsets
            .par_iter()
            .zip(frame.strip_byte_counts.par_iter())
            .enumerate()
            .map(|(i, (&offset, &len))| -> Result<Vec<u8>> {
                let raw_strip = mmap
                    .get(offset as usize..(offset + len) as usize)
                    .ok_or_else(|| anyhow!("strip at offset {offset} (len {len}) out of file bounds"))?;
                let rows_this_strip = rows_per_strip.min(total_rows.saturating_sub(i * rows_per_strip));
                let expected_len = rows_this_strip * row_bytes;
                decompress(raw_strip, compression, expected_len)
            })
            .collect::<Result<_>>()?;
        for strip in &strips {
            native.extend_from_slice(strip);
        }
    } else {
        let mut rows_done = 0usize;
        for (&offset, &len) in frame.strip_offsets.iter().zip(frame.strip_byte_counts.iter()) {
            let raw_strip = mmap
                .get(offset as usize..(offset + len) as usize)
                .ok_or_else(|| anyhow!("strip at offset {offset} (len {len}) out of file bounds"))?;
            let rows_this_strip = rows_per_strip.min(total_rows.saturating_sub(rows_done));
            let expected_len = rows_this_strip * row_bytes;
            match compression {
                // Same cap as `decompress`: padded strips must not shift rows.
                Compression::None => native.extend_from_slice(&raw_strip[..raw_strip.len().min(expected_len)]),
                _ => native.extend_from_slice(&decompress(raw_strip, compression, expected_len)?),
            }
            rows_done += rows_this_strip;
        }
    }

    if native.len() < total_len {
        bail!(
            "decoded {} bytes but expected {} for a {}x{} frame ({} bytes/sample) — \
             strip data is shorter than the declared image size",
            native.len(),
            total_len,
            frame.width,
            frame.height,
            sample_bytes
        );
    }

    Ok(Cow::Owned(undo_predictor(native, frame, sample_bytes, file_order)?))
}

/// Decompress one strip, returning **at most `expected_len` bytes** (the
/// strip's rows x row bytes). Some writers pad strips — trailing alignment
/// bytes in the raw data, or whole padded rows in the compressed stream —
/// and without this cap the excess would shift every following row of the
/// assembled frame sideways. For the streaming codecs (Deflate/ZSTD) the cap
/// also bounds the memory a corrupt/hostile stream can expand into.
fn decompress(raw: &[u8], compression: Compression, expected_len: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    match compression {
        Compression::None => Ok(raw[..raw.len().min(expected_len)].to_vec()),
        Compression::Lzw => {
            let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let mut out = Vec::with_capacity(expected_len);
            decoder
                .into_stream(&mut out)
                .decode_all(raw)
                .status
                .map_err(|e| anyhow!("LZW decode failed: {e:?}"))?;
            out.truncate(expected_len);
            Ok(out)
        }
        Compression::Deflate => {
            let mut out = Vec::with_capacity(expected_len);
            flate2::read::ZlibDecoder::new(raw)
                .take(expected_len as u64)
                .read_to_end(&mut out)
                .map_err(|e| anyhow!("Deflate decode failed: {e}"))?;
            Ok(out)
        }
        Compression::PackBits => {
            let mut out = packbits_decode(raw, expected_len);
            out.truncate(expected_len);
            Ok(out)
        }
        Compression::Zstd => {
            let mut out = Vec::with_capacity(expected_len);
            zstd::stream::read::Decoder::new(raw)
                .map_err(|e| anyhow!("ZSTD decode failed: {e}"))?
                .take(expected_len as u64)
                .read_to_end(&mut out)
                .map_err(|e| anyhow!("ZSTD decode failed: {e}"))?;
            Ok(out)
        }
        Compression::Other(code) => bail!("unsupported TIFF compression scheme: {code}"),
    }
}

fn packbits_decode(input: &[u8], expected_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(expected_len);
    let mut i = 0;
    while i < input.len() {
        let n = input[i] as i8;
        i += 1;
        if n >= 0 {
            // literal run of (n + 1) bytes
            let count = n as usize + 1;
            let end = (i + count).min(input.len());
            out.extend_from_slice(&input[i..end]);
            i = end;
        } else if n != -128 {
            // replicate next byte (1 - n) times
            if i < input.len() {
                let byte = input[i];
                i += 1;
                let repeat_count = (1 - n as isize) as usize;
                out.extend(std::iter::repeat(byte).take(repeat_count));
            }
        }
        // n == -128 is a documented no-op
    }
    out
}

/// Undo the TIFF predictor pass: Predictor 2 (horizontal differencing, any
/// integer width) or Predictor 3 (TechNote 3 floating-point differencing).
/// Operates per scanline, per sample (so RGB/multi-sample data differences
/// each channel independently), matching the TIFF6 spec / libtiff. Reads and
/// writes in `file_order` since this runs before the final byte-order
/// normalization pass. Strip boundaries are always a whole number of rows, so
/// running this once on the full concatenated buffer (rather than per-strip
/// before concatenating) gives the same result, since differencing resets at
/// every row regardless of which strip it came from.
fn undo_predictor(
    mut data: Vec<u8>,
    frame: &FrameInfo,
    sample_bytes: usize,
    file_order: ByteOrder,
) -> Result<Vec<u8>> {
    let spp = frame.samples_per_pixel as usize;
    let row_samples = frame.width as usize * spp;
    let row_bytes = row_samples * sample_bytes;

    match (frame.predictor, sample_bytes) {
        (1, _) => return Ok(data),
        (2, 1) => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in spp..row_samples {
                    row[i] = row[i].wrapping_add(row[i - spp]);
                }
            }
        }
        (2, 2) => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in spp..row_samples {
                    let prev_off = (i - spp) * 2;
                    let cur_off = i * 2;
                    let prev = file_order.u16(&row[prev_off..prev_off + 2]);
                    let delta = file_order.u16(&row[cur_off..cur_off + 2]);
                    let val = prev.wrapping_add(delta);
                    let bytes = match file_order {
                        ByteOrder::Little => val.to_le_bytes(),
                        ByteOrder::Big => val.to_be_bytes(),
                    };
                    row[cur_off] = bytes[0];
                    row[cur_off + 1] = bytes[1];
                }
            }
        }
        (2, 4) => {
            // 32-bit integer horizontal differencing (libtiff writes these).
            for row in data.chunks_exact_mut(row_bytes) {
                for i in spp..row_samples {
                    let prev_off = (i - spp) * 4;
                    let cur_off = i * 4;
                    let prev = file_order.u32(&row[prev_off..prev_off + 4]);
                    let delta = file_order.u32(&row[cur_off..cur_off + 4]);
                    let val = prev.wrapping_add(delta);
                    let bytes = match file_order {
                        ByteOrder::Little => val.to_le_bytes(),
                        ByteOrder::Big => val.to_be_bytes(),
                    };
                    row[cur_off..cur_off + 4].copy_from_slice(&bytes);
                }
            }
        }
        (3, 4) => undo_float_predictor(&mut data, row_bytes, spp, file_order),
        (2, other) => bail!("predictor 2 undo not implemented for {other}-byte samples"),
        (3, other) => bail!("floating-point predictor requires 32-bit samples, got {other}-byte"),
        (other, _) => bail!("unsupported TIFF predictor: {other}"),
    }
    Ok(std::mem::take(&mut data))
}

/// Undo TIFF Predictor 3 (TIFF TechNote 3 floating-point horizontal
/// differencing), per row: first undo the byte-level differencing (stride =
/// samples per pixel, mirroring libtiff's `fpAcc`), then gather each float's
/// bytes back from the row's four byte-significance planes — the spec stores
/// them MSB-plane-first regardless of the file's byte order — and store the
/// value in `file_order` for the normal downstream reads.
fn undo_float_predictor(data: &mut [u8], row_bytes: usize, spp: usize, file_order: ByteOrder) {
    let wc = row_bytes / 4; // f32 values per row
    let mut scratch = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for i in spp..row_bytes {
            row[i] = row[i].wrapping_add(row[i - spp]);
        }
        for v in 0..wc {
            let be = [row[v], row[wc + v], row[2 * wc + v], row[3 * wc + v]];
            let bytes = match file_order {
                ByteOrder::Little => [be[3], be[2], be[1], be[0]],
                ByteOrder::Big => be,
            };
            scratch[v * 4..v * 4 + 4].copy_from_slice(&bytes);
        }
        row.copy_from_slice(&scratch);
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
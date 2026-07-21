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

/// [`read_frame_u16`] into a caller-provided buffer, reusing its allocation.
/// Uncompressed predictor-free 16-bit frames convert straight from the
/// mapping's strips into `out` (a plain memcpy when the byte order is native
/// and the data unsigned) — no intermediate buffer, no per-frame allocation.
pub fn read_frame_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    out: &mut Vec<u16>,
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    if spp == 1
        && frame.bits_per_sample == 16
        && frame.compression == Compression::None
        && frame.predictor == 1
    {
        ensure_len(out, frame.width as usize * frame.height as usize);
        let signed = frame.sample_format == SampleFormat::SignedInt;
        let flip = if signed { 0x8000u16 } else { 0 };
        let memcpyable = !signed && file_order == ByteOrder::host();
        return for_each_raw_strip(mmap, frame, 2, |src, start, n| {
            let dst = &mut out[start..start + n];
            if memcpyable {
                bytemuck::cast_slice_mut::<u16, u8>(dst).copy_from_slice(src);
            } else {
                match file_order {
                    ByteOrder::Little => {
                        for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                            *o = u16::from_le_bytes([c[0], c[1]]) ^ flip;
                        }
                    }
                    ByteOrder::Big => {
                        for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                            *o = u16::from_be_bytes([c[0], c[1]]) ^ flip;
                        }
                    }
                }
            }
        });
    }
    read_plane_u16_into(mmap, frame, file_order, float_range, 0, out)
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

/// [`read_frame_u8`] into a caller-provided buffer, reusing its allocation.
/// For uncompressed predictor-free frames the strips are copied straight from
/// the mapping into `out` — no intermediate buffer, no per-frame allocation.
pub fn read_frame_u8_into(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder, out: &mut Vec<u8>) -> Result<()> {
    let sample_bytes = sample_bytes(frame)?;
    if frame.compression == Compression::None && frame.predictor == 1 {
        let spp = (frame.samples_per_pixel as usize).max(1);
        ensure_len(out, frame.width as usize * frame.height as usize * spp * sample_bytes);
        return for_each_raw_strip(mmap, frame, sample_bytes, |src, start, n| {
            out[start * sample_bytes..start * sample_bytes + n * sample_bytes].copy_from_slice(src);
        });
    }
    let native = decode_native_bytes(mmap, frame, file_order)?;
    ensure_len(out, native.len());
    out.copy_from_slice(&native);
    Ok(())
}

/// Bytes per sample from the frame's bit depth (the widths this crate decodes).
fn sample_bytes(frame: &FrameInfo) -> Result<usize> {
    bytes_for_bits(frame.bits_per_sample)
}

/// Bytes per sample for a supported bit depth (8/16/32/64), or an error.
fn bytes_for_bits(bits: u16) -> Result<usize> {
    match bits {
        8 => Ok(1),
        16 => Ok(2),
        32 => Ok(4),
        64 => Ok(8),
        other => bail!("unsupported bits_per_sample: {other}"),
    }
}

/// Walk an uncompressed, predictor-free frame's strips, handing each strip's
/// source bytes plus its destination `(sample_start, n_samples)` range to `f`
/// — the engine of the direct read paths, which skip the assembly buffer
/// entirely. Applies the same strip-cap/short-strip policy as the general
/// path (padding dropped; short data is an error).
fn for_each_raw_strip(
    mmap: &[u8],
    frame: &FrameInfo,
    sample_bytes: usize,
    mut f: impl FnMut(&[u8], usize, usize),
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let row_samples = frame.width as usize * spp;
    let total_samples = row_samples * frame.height as usize;
    let rows_per_strip = (frame.rows_per_strip as usize).max(1);
    let mut sample_pos = 0usize;
    for (&offset, &len) in frame.strip_offsets.iter().zip(frame.strip_byte_counts.iter()) {
        if sample_pos >= total_samples {
            break;
        }
        let rows = rows_per_strip.min((total_samples - sample_pos) / row_samples.max(1));
        let n_samples = (rows * row_samples).max(1).min(total_samples - sample_pos);
        let expected = n_samples * sample_bytes;
        let avail = (len as usize).min(expected);
        let src = mmap
            .get(offset as usize..offset as usize + avail)
            .ok_or_else(|| anyhow!("strip at offset {offset} out of file bounds"))?;
        if src.len() < expected {
            bail!(
                "strip data is shorter than the declared image size ({} of {} bytes)",
                src.len(),
                expected
            );
        }
        f(src, sample_pos, n_samples);
        sample_pos += n_samples;
    }
    if sample_pos < total_samples {
        bail!("strips cover only {sample_pos} of {total_samples} samples");
    }
    Ok(())
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
    let mut out = Vec::new();
    read_plane_u16_into(mmap, frame, file_order, float_range, plane, &mut out)?;
    Ok(out)
}

/// [`read_plane_u16`] into a caller-provided buffer, reusing its allocation —
/// for hot per-frame loops this avoids the allocation, the zero-fill, and (on
/// Windows especially) fresh-page faults of a new `Vec` every frame.
pub fn read_plane_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    plane: usize,
    out: &mut Vec<u16>,
) -> Result<()> {
    let native = decode_native_bytes(mmap, frame, file_order)?;
    ensure_len(out, frame.width as usize * frame.height as usize);
    plane_u16_from_native(&native, frame, file_order, float_range, plane, out)
}

/// Size `out` to exactly `n` reusing its allocation; only newly-grown
/// elements pay initialization.
fn ensure_len<T: Copy + Default>(out: &mut Vec<T>, n: usize) {
    if out.len() != n {
        out.resize(n, T::default());
    }
}

/// Decode **all** sample planes to u16 with a single decompression pass — for
/// chunky RGB this is ~3x cheaper than three [`read_plane_u16`] calls on
/// compressed data (each of which decompresses the whole frame again). Returns
/// one plane per sample, in sample order; single-sample frames return one
/// plane. With `float_range = None`, each 32-bit plane auto-ranges to its own
/// min/max, exactly like the per-plane call.
pub fn read_planes_u16(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
) -> Result<Vec<Vec<u16>>> {
    let mut out = Vec::new();
    read_planes_u16_into(mmap, frame, file_order, float_range, &mut out)?;
    Ok(out)
}

/// [`read_planes_u16`] into caller-provided buffers (outer Vec sized to the
/// sample count, inner Vecs reused).
pub fn read_planes_u16_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    out: &mut Vec<Vec<u16>>,
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let n_pixels = frame.width as usize * frame.height as usize;
    // Chunky multi-sample with Predictor 2: fuse the predictor undo into the
    // per-plane gather (each sample channel differences independently, so a
    // plane is just a running sum along each row) — one pass per plane over
    // untouched native bytes, instead of a full read-modify-write undo pass
    // followed by the gathers.
    let fuse = spp > 1 && frame.predictor == 2 && matches!(frame.bits_per_sample, 8 | 16);
    let native = decode_native_bytes_opt(mmap, frame, file_order, !fuse)?;
    out.resize_with(spp, Vec::new);
    for (p, plane_out) in out.iter_mut().enumerate() {
        ensure_len(plane_out, n_pixels);
        if fuse {
            plane_u16_fused_pred2(&native, frame, file_order, p, plane_out);
        } else {
            plane_u16_from_native(&native, frame, file_order, float_range, p, plane_out)?;
        }
    }
    Ok(())
}

/// Fused Predictor-2 undo + deinterleave for one 8/16-bit plane of a chunky
/// frame: accumulate the per-row running sum for this sample channel while
/// gathering it, writing display-space u16 (widened for 8-bit sources).
fn plane_u16_fused_pred2(native: &[u8], frame: &FrameInfo, file_order: ByteOrder, plane: usize, out: &mut [u16]) {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let width = frame.width as usize;
    let row_samples = width * spp;
    let signed = frame.sample_format == SampleFormat::SignedInt;
    match frame.bits_per_sample {
        16 => {
            let flip = if signed { 0x8000u16 } else { 0 };
            for (row_idx, out_row) in out.chunks_mut(width.max(1)).enumerate() {
                let row_base = row_idx * row_samples;
                let mut acc = 0u16;
                for (x, o) in out_row.iter_mut().enumerate() {
                    let off = (row_base + x * spp + plane) * 2;
                    acc = acc.wrapping_add(file_order.u16(&native[off..off + 2]));
                    *o = acc ^ flip;
                }
            }
        }
        _ => {
            // 8-bit: accumulate bytes, then widen into display space.
            let flip = if signed { 0x80u8 } else { 0 };
            for (row_idx, out_row) in out.chunks_mut(width.max(1)).enumerate() {
                let row_base = row_idx * row_samples;
                let mut acc = 0u8;
                for (x, o) in out_row.iter_mut().enumerate() {
                    acc = acc.wrapping_add(native[row_base + x * spp + plane]);
                    let b = acc ^ flip;
                    *o = ((b as u16) << 8) | b as u16;
                }
            }
        }
    }
}

/// The conversion core shared by the u16 plane readers: deinterleave +
/// widen/offset/rescale one plane out of already-decoded native bytes, into
/// `out` (pre-sized to `width * height`).
fn plane_u16_from_native(
    native: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    float_range: Option<(f32, f32)>,
    plane: usize,
    out: &mut [u16],
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let n_pixels = frame.width as usize * frame.height as usize;
    let signed = frame.sample_format == SampleFormat::SignedInt;
    debug_assert_eq!(out.len(), n_pixels);

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
        32 | 64 => {
            // The heaviest per-pixel path: read each wide (32- or 64-bit) sample
            // as f32, then linearly rescale into 0..65535. Parallelized across
            // cores for large frames only (input is read-only, writes disjoint).
            let format = frame.sample_format;
            let sb = if frame.bits_per_sample == 64 { 8 } else { 4 };
            let parallel = should_parallelize(n_pixels);
            let floats: Vec<f32> = if parallel {
                (0..n_pixels)
                    .into_par_iter()
                    .map(|i| wide_sample_f32(&native[(i * spp + plane) * sb..], file_order, format, sb))
                    .collect()
            } else {
                (0..n_pixels)
                    .map(|i| wide_sample_f32(&native[(i * spp + plane) * sb..], file_order, format, sb))
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
    Ok(())
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
    let mut out = Vec::new();
    read_plane_u8_into(mmap, frame, file_order, plane, &mut out)?;
    Ok(out)
}

/// [`read_plane_u8`] into a caller-provided buffer, reusing its allocation.
pub fn read_plane_u8_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let native = decode_native_bytes(mmap, frame, file_order)?;
    ensure_len(out, frame.width as usize * frame.height as usize);
    plane_u8_from_native(&native, frame, plane, out)
}

/// Decode **all** sample planes as raw 8-bit bytes with a single
/// decompression pass — the u8 sibling of [`read_planes_u16`], ~3x cheaper
/// than three [`read_plane_u8`] calls on compressed chunky RGB. Same validity
/// rules as the per-plane call (unsigned 8-bit frames).
pub fn read_planes_u8(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    read_planes_u8_into(mmap, frame, file_order, &mut out)?;
    Ok(out)
}

/// [`read_planes_u8`] into caller-provided buffers.
pub fn read_planes_u8_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let n_pixels = frame.width as usize * frame.height as usize;
    // Same predictor-2 fusion as the u16 planes path (see there), raw bytes.
    let fuse = spp > 1 && frame.predictor == 2 && frame.bits_per_sample == 8;
    let native = decode_native_bytes_opt(mmap, frame, file_order, !fuse)?;
    out.resize_with(spp, Vec::new);
    for (p, plane_out) in out.iter_mut().enumerate() {
        ensure_len(plane_out, n_pixels);
        if fuse {
            let width = frame.width as usize;
            let row_samples = width * spp;
            let plane = p.min(spp - 1);
            for (row_idx, out_row) in plane_out.chunks_mut(width.max(1)).enumerate() {
                let row_base = row_idx * row_samples;
                let mut acc = 0u8;
                for (x, o) in out_row.iter_mut().enumerate() {
                    acc = acc.wrapping_add(native[row_base + x * spp + plane]);
                    *o = acc;
                }
            }
        } else {
            plane_u8_from_native(&native, frame, p, plane_out)?;
        }
    }
    Ok(())
}

/// The gather core shared by the u8 plane readers (`out` pre-sized).
fn plane_u8_from_native(native: &[u8], frame: &FrameInfo, plane: usize, out: &mut [u8]) -> Result<()> {
    if frame.bits_per_sample != 8 {
        bail!("read_plane_u8 requires 8-bit samples, got {}", frame.bits_per_sample);
    }
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    for (i, o) in out.iter_mut().enumerate() {
        *o = native[i * spp + plane];
    }
    Ok(())
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

/// [`read_frame_f32`] into a caller-provided buffer, reusing its allocation.
/// Uncompressed predictor-free float frames convert straight from the
/// mapping's strips into `out` (a plain memcpy in native byte order) — no
/// intermediate buffer, no per-frame allocation.
pub fn read_frame_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<f32>,
) -> Result<()> {
    let spp = (frame.samples_per_pixel as usize).max(1);
    if spp == 1
        && frame.bits_per_sample == 32
        && frame.compression == Compression::None
        && frame.predictor == 1
    {
        ensure_len(out, frame.width as usize * frame.height as usize);
        let format = frame.sample_format;
        let memcpyable = format == SampleFormat::Float && file_order == ByteOrder::host();
        return for_each_raw_strip(mmap, frame, 4, |src, start, n| {
            let dst = &mut out[start..start + n];
            if memcpyable {
                bytemuck::cast_slice_mut::<f32, u8>(dst).copy_from_slice(src);
            } else {
                for (o, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                    *o = sample_f32(c, file_order, format);
                }
            }
        });
    }
    read_plane_f32_into(mmap, frame, file_order, 0, out)
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
    let mut out = Vec::new();
    read_plane_f32_into(mmap, frame, file_order, plane, &mut out)?;
    Ok(out)
}

/// [`read_plane_f32`] into a caller-provided buffer, reusing its allocation.
pub fn read_plane_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    plane: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let native = decode_native_bytes(mmap, frame, file_order)?;
    ensure_len(out, frame.width as usize * frame.height as usize);
    plane_f32_from_native(&native, frame, file_order, plane, out);
    Ok(())
}

/// Decode **all** sample planes as raw `f32` with a single decompression pass
/// — the float sibling of [`read_planes_u16`]. Same validity rules as the
/// per-plane call (32-bit data).
pub fn read_planes_f32(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    read_planes_f32_into(mmap, frame, file_order, &mut out)?;
    Ok(out)
}

/// [`read_planes_f32`] into caller-provided buffers.
pub fn read_planes_f32_into(
    mmap: &[u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    out: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let spp = (frame.samples_per_pixel as usize).max(1);
    let n_pixels = frame.width as usize * frame.height as usize;
    out.resize_with(spp, Vec::new);
    for (p, plane_out) in out.iter_mut().enumerate() {
        ensure_len(plane_out, n_pixels);
        plane_f32_from_native(&native, frame, file_order, p, plane_out);
    }
    Ok(())
}

/// The conversion core shared by the f32 plane readers (`out` pre-sized).
/// 64-bit sources (f64 / i64 / u64) are downcast to f32 here; 32-bit sources
/// take the unchanged 4-byte path.
fn plane_f32_from_native(native: &[u8], frame: &FrameInfo, file_order: ByteOrder, plane: usize, out: &mut [f32]) {
    let spp = (frame.samples_per_pixel as usize).max(1);
    let plane = plane.min(spp - 1);
    let n_pixels = frame.width as usize * frame.height as usize;
    let format = frame.sample_format;
    let sb = if frame.bits_per_sample == 64 { 8 } else { 4 };
    if should_parallelize(n_pixels) {
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| *o = wide_sample_f32(&native[(i * spp + plane) * sb..], file_order, format, sb));
    } else {
        for (i, o) in out.iter_mut().enumerate() {
            *o = wide_sample_f32(&native[(i * spp + plane) * sb..], file_order, format, sb);
        }
    }
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
    // Only the wide (32/64-bit) formats auto-range this way; 8/16-bit use their
    // native integer min/max instead.
    if frame.bits_per_sample != 32 && frame.bits_per_sample != 64 {
        return Ok(None);
    }
    let sb = if frame.bits_per_sample == 64 { 8 } else { 4 };
    let native = decode_native_bytes(mmap, frame, file_order)?;
    let n_samples = frame.width as usize * frame.height as usize * frame.samples_per_pixel as usize;
    // Fold directly over the decoded bytes — no width*height*sb temporary.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for chunk in native[..n_samples * sb].chunks_exact(sb) {
        let v = wide_sample_f32(chunk, file_order, frame.sample_format, sb);
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

/// Reads one 64-bit sample as `f32` — the 8-byte sibling of [`sample_f32`]
/// (f64/i64/u64 per the sample format, downcast to f32). Values outside f32's
/// range become ±inf, which the auto-range fold treats as non-finite; this
/// precision loss is inherent to displaying 64-bit data through an f32 GPU path.
fn sample_f32_64(chunk: &[u8], file_order: ByteOrder, format: SampleFormat) -> f32 {
    let arr: [u8; 8] = chunk[..8].try_into().unwrap();
    match (format, file_order) {
        (SampleFormat::Float, ByteOrder::Little) => f64::from_le_bytes(arr) as f32,
        (SampleFormat::Float, ByteOrder::Big) => f64::from_be_bytes(arr) as f32,
        (SampleFormat::SignedInt, ByteOrder::Little) => i64::from_le_bytes(arr) as f32,
        (SampleFormat::SignedInt, ByteOrder::Big) => i64::from_be_bytes(arr) as f32,
        (SampleFormat::UnsignedInt, ByteOrder::Little) => u64::from_le_bytes(arr) as f32,
        (SampleFormat::UnsignedInt, ByteOrder::Big) => u64::from_be_bytes(arr) as f32,
    }
}

/// Reads one wide (32- or 64-bit) sample as `f32`, dispatching on `sample_bytes`
/// (4 or 8). The width is loop-invariant at every call site, so the branch is
/// free in practice; keeping the two readers separate leaves the 32-bit hot
/// loops byte-for-byte unchanged.
#[inline]
fn wide_sample_f32(chunk: &[u8], file_order: ByteOrder, format: SampleFormat, sample_bytes: usize) -> f32 {
    if sample_bytes == 8 {
        sample_f32_64(chunk, file_order, format)
    } else {
        sample_f32(chunk, file_order, format)
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
    decode_native_bytes_opt(mmap, frame, file_order, true)
}

/// [`decode_native_bytes`] with the predictor undo optional: the fused
/// planes paths skip it (they fold the undo into their per-plane gather) —
/// which also lets a predictor-differenced uncompressed frame stay borrowed.
fn decode_native_bytes_opt<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
    undo_pred: bool,
) -> Result<Cow<'a, [u8]>> {
    let sample_bytes = bytes_for_bits(frame.bits_per_sample)?;
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
        return if undo_pred && frame.predictor != 1 {
            // Predictor undo mutates in place, so it needs an owned copy.
            Ok(Cow::Owned(undo_predictor(slice.to_vec(), frame, sample_bytes, file_order)?))
        } else {
            Ok(Cow::Borrowed(slice))
        };
    }

    // General path: multi-strip and/or compressed — assemble **directly into**
    // one pre-sized buffer, each strip decompressing into its own row range
    // (single pass; the old per-strip Vec + concat copy is gone). The last
    // strip may legitimately have fewer rows than `rows_per_strip` when the
    // image height doesn't divide evenly.
    let compression = frame.compression;
    let n_pixels = frame.width as usize * frame.height as usize;
    let mut native = vec![0u8; total_len];

    // Carve `native` into per-strip destination slices (disjoint row ranges).
    let mut dests: Vec<&mut [u8]> = Vec::with_capacity(frame.strip_offsets.len());
    let mut rest: &mut [u8] = &mut native;
    let mut rows_done = 0usize;
    for _ in 0..frame.strip_offsets.len() {
        let rows_this_strip = rows_per_strip.min(total_rows.saturating_sub(rows_done));
        let (dest, tail) = rest.split_at_mut(rows_this_strip * row_bytes);
        dests.push(dest);
        rest = tail;
        rows_done += rows_this_strip;
    }

    // Strips are independent compressed units, so for *large* compressed frames
    // we decompress them in parallel (disjoint destination slices; each strip's
    // row span comes from its index, so the map is pure). Small/medium frames
    // stay serial: the fork-join overhead would otherwise cost more total CPU
    // than it saves during fast playback.
    let parallel =
        compression != Compression::None && frame.strip_offsets.len() > 1 && should_parallelize(n_pixels);
    let strip_src = |offset: u64, len: u64| -> Result<&[u8]> {
        mmap.get(offset as usize..(offset + len) as usize)
            .ok_or_else(|| anyhow!("strip at offset {offset} (len {len}) out of file bounds"))
    };
    let written: usize = if parallel {
        frame
            .strip_offsets
            .par_iter()
            .zip(frame.strip_byte_counts.par_iter())
            .zip(dests.par_iter_mut())
            .map(|((&offset, &len), dest)| decompress_into(strip_src(offset, len)?, compression, dest))
            .try_reduce(|| 0, |a, b| Ok(a + b))?
    } else {
        let mut total = 0usize;
        for ((&offset, &len), dest) in
            frame.strip_offsets.iter().zip(frame.strip_byte_counts.iter()).zip(dests.iter_mut())
        {
            total += decompress_into(strip_src(offset, len)?, compression, dest)?;
        }
        total
    };

    if written < total_len {
        bail!(
            "decoded {} bytes but expected {} for a {}x{} frame ({} bytes/sample) — \
             strip data is shorter than the declared image size",
            written,
            total_len,
            frame.width,
            frame.height,
            sample_bytes
        );
    }

    if undo_pred {
        Ok(Cow::Owned(undo_predictor(native, frame, sample_bytes, file_order)?))
    } else {
        Ok(Cow::Owned(native))
    }
}

/// Decompress one strip **directly into its destination slice** (the strip's
/// rows x row bytes), returning how many bytes were written. Writing into the
/// caller's pre-carved slice is what makes frame assembly single-pass — no
/// per-strip temporary and no concatenation copy.
///
/// The fixed-size destination also enforces the strip cap: some writers pad
/// strips (trailing alignment bytes, or whole padded rows in the compressed
/// stream), and without the cap the excess would shift every following row
/// sideways. For the streaming codecs it bounds what a corrupt/hostile stream
/// can expand into — nothing is ever allocated beyond `dest`.
fn decompress_into(raw: &[u8], compression: Compression, dest: &mut [u8]) -> Result<usize> {
    match compression {
        Compression::None => {
            let n = raw.len().min(dest.len());
            dest[..n].copy_from_slice(&raw[..n]);
            Ok(n)
        }
        Compression::Lzw => {
            let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let mut in_pos = 0;
            let mut out_pos = 0;
            loop {
                let res = decoder.decode_bytes(&raw[in_pos..], &mut dest[out_pos..]);
                in_pos += res.consumed_in;
                out_pos += res.consumed_out;
                match res.status {
                    Ok(weezl::LzwStatus::Done) => break,
                    Ok(weezl::LzwStatus::Ok) => {
                        // Cap reached, or no forward progress possible.
                        if out_pos == dest.len() || (res.consumed_in == 0 && res.consumed_out == 0) {
                            break;
                        }
                    }
                    Ok(weezl::LzwStatus::NoProgress) => break, // input exhausted early
                    Err(e) => return Err(anyhow!("LZW decode failed: {e:?}")),
                }
            }
            Ok(out_pos)
        }
        Compression::Deflate => {
            read_all_into(flate2::read::ZlibDecoder::new(raw), dest).map_err(|e| anyhow!("Deflate decode failed: {e}"))
        }
        Compression::PackBits => Ok(packbits_decode_into(raw, dest)),
        Compression::Zstd => {
            let dec = zstd::stream::read::Decoder::new(raw).map_err(|e| anyhow!("ZSTD decode failed: {e}"))?;
            read_all_into(dec, dest).map_err(|e| anyhow!("ZSTD decode failed: {e}"))
        }
        Compression::Other(code) => bail!("unsupported TIFF compression scheme: {code}"),
    }
}

/// Read from `r` until `dest` is full or the stream ends; returns bytes read.
fn read_all_into(mut r: impl std::io::Read, dest: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < dest.len() {
        match r.read(&mut dest[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// PackBits-decode into a fixed destination slice; returns bytes written.
/// Output beyond `dest` (padded rows) is dropped, matching the strip cap.
fn packbits_decode_into(input: &[u8], dest: &mut [u8]) -> usize {
    let mut out = 0;
    let mut i = 0;
    while i < input.len() && out < dest.len() {
        let n = input[i] as i8;
        i += 1;
        if n >= 0 {
            // Literal run of (n + 1) bytes: consume what the input actually
            // has; copy what the destination still accepts.
            let take = (n as usize + 1).min(input.len() - i);
            let copy = take.min(dest.len() - out);
            dest[out..out + copy].copy_from_slice(&input[i..i + copy]);
            i += take;
            out += copy;
        } else if n != -128 {
            // replicate next byte (1 - n) times
            if i < input.len() {
                let byte = input[i];
                i += 1;
                let count = ((1 - n as isize) as usize).min(dest.len() - out);
                dest[out..out + count].fill(byte);
                out += count;
            }
        }
        // n == -128 is a documented no-op
    }
    out
}

/// Vec-returning PackBits decode (kept for the unit tests' direct checks).
#[cfg(test)]
fn packbits_decode(input: &[u8], expected_len: usize) -> Vec<u8> {
    let mut out = vec![0u8; expected_len];
    let n = packbits_decode_into(input, &mut out);
    out.truncate(n);
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
        (2, 8) => {
            // 64-bit integer horizontal differencing.
            for row in data.chunks_exact_mut(row_bytes) {
                for i in spp..row_samples {
                    let prev_off = (i - spp) * 8;
                    let cur_off = i * 8;
                    let prev = file_order.u64(&row[prev_off..prev_off + 8]);
                    let delta = file_order.u64(&row[cur_off..cur_off + 8]);
                    let val = prev.wrapping_add(delta);
                    let bytes = match file_order {
                        ByteOrder::Little => val.to_le_bytes(),
                        ByteOrder::Big => val.to_be_bytes(),
                    };
                    row[cur_off..cur_off + 8].copy_from_slice(&bytes);
                }
            }
        }
        (3, 4) | (3, 8) => undo_float_predictor(&mut data, row_bytes, spp, sample_bytes, file_order),
        (2, other) => bail!("predictor 2 undo not implemented for {other}-byte samples"),
        (3, other) => bail!("floating-point predictor requires 32- or 64-bit samples, got {other}-byte"),
        (other, _) => bail!("unsupported TIFF predictor: {other}"),
    }
    Ok(std::mem::take(&mut data))
}

/// Undo TIFF Predictor 3 (TIFF TechNote 3 floating-point horizontal
/// differencing), per row: first undo the byte-level differencing (stride =
/// samples per pixel, mirroring libtiff's `fpAcc`), then gather each float's
/// bytes back from the row's `sample_bytes` byte-significance planes — the spec
/// stores them MSB-plane-first regardless of the file's byte order — and store
/// the value in `file_order` for the normal downstream reads. Handles both f32
/// (`sample_bytes == 4`) and f64 (`sample_bytes == 8`).
fn undo_float_predictor(data: &mut [u8], row_bytes: usize, spp: usize, sample_bytes: usize, file_order: ByteOrder) {
    let wc = row_bytes / sample_bytes; // float values per row
    let mut scratch = vec![0u8; row_bytes];
    for row in data.chunks_exact_mut(row_bytes) {
        for i in spp..row_bytes {
            row[i] = row[i].wrapping_add(row[i - spp]);
        }
        for v in 0..wc {
            // Gather this value's bytes, most-significant plane first.
            for p in 0..sample_bytes {
                // Plane `p` holds significance (sample_bytes-1-p): plane 0 = MSB.
                let byte = row[p * wc + v];
                let dst = match file_order {
                    ByteOrder::Little => sample_bytes - 1 - p, // little-endian: MSB last
                    ByteOrder::Big => p,                       // big-endian: MSB first
                };
                scratch[v * sample_bytes + dst] = byte;
            }
        }
        row.copy_from_slice(&scratch);
    }
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
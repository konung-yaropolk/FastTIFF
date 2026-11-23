//! The conversions from a piece's stored samples to what a read asked for,
//! over a run of whole pixels: a strip's rows, or one row of a tile's part of
//! the image. Each kernel reads `dst.len()` pixels' worth of `src`, plane
//! `base` of `stride` samples per pixel, and writes `dst`.
//!
//! The arithmetic is the old per-plane converters', loop for loop — the same
//! byte-order branch hoisted out, the same sign-bit flips, the same `f32`
//! rescale — so a sample converted a piece at a time is the sample converted
//! a frame at a time, to the bit. What changed is only how much of the frame
//! one call sees: a piece, still in cache from its decompression, rather than
//! a frame assembled in a buffer first.

use crate::error::{ErrorKind, OptionContext, Result};
use crate::ifd::ByteOrder;
use crate::index::SampleFormat;

/// Display 16-bit: the sample, sign-bit flipped when signed, from the file's
/// byte order.
pub(super) fn display16(
    src: &[u8],
    base: usize,
    stride: usize,
    order: ByteOrder,
    signed: bool,
    dst: &mut [u16],
) {
    let n = dst.len();
    if stride == 1 {
        // Contiguous samples — the per-frame hot path for compressed 16-bit
        // playback (and every planar plane). Bulk-convert with the byte-order
        // branch hoisted out of the loop and no strided indexing (the sign-bit
        // flip folds in branchlessly: xor with 0 is a no-op).
        let flip = if signed { 0x8000 } else { 0 };
        let pairs = src[base * 2..(base + n) * 2].as_chunks::<2>().0;
        match order {
            ByteOrder::Little => {
                for (o, c) in dst.iter_mut().zip(pairs) {
                    *o = u16::from_le_bytes([c[0], c[1]]) ^ flip;
                }
            }
            ByteOrder::Big => {
                for (o, c) in dst.iter_mut().zip(pairs) {
                    *o = u16::from_be_bytes([c[0], c[1]]) ^ flip;
                }
            }
        }
    } else {
        for (i, o) in dst.iter_mut().enumerate() {
            let off = (base + i * stride) * 2;
            let v = order.u16(&src[off..off + 2]);
            *o = if signed { v ^ 0x8000 } else { v };
        }
    }
}

/// Display 8-bit: widened onto 0..65535 (`v -> (v << 8) | v`), the sign bit
/// flipped first when signed.
pub(super) fn display8(src: &[u8], base: usize, stride: usize, signed: bool, dst: &mut [u16]) {
    let n = dst.len();
    if stride == 1 {
        let flip = if signed { 0x80 } else { 0 };
        for (o, &b) in dst.iter_mut().zip(src[base..base + n].iter()) {
            let b = b ^ flip;
            *o = ((b as u16) << 8) | b as u16;
        }
    } else {
        for (i, o) in dst.iter_mut().enumerate() {
            let b = src[base + i * stride];
            let b = if signed { b ^ 0x80 } else { b };
            *o = ((b as u16) << 8) | b as u16;
        }
    }
}

/// Fused Predictor-2 undo + deinterleave for one 8/16-bit plane of a chunky
/// strip: accumulate the per-row running sum for this sample channel while
/// gathering it, writing display-space u16 (widened for 8-bit sources). `src`
/// holds whole rows of `width` pixels, still differenced.
///
/// One pass per plane over untouched bytes, instead of a read-modify-write
/// undo pass followed by the gather. Each sample channel differences
/// independently, so a plane is just a running sum along each row.
#[allow(clippy::too_many_arguments)]
pub(super) fn display_fused_pred2(
    src: &[u8],
    sample_bytes: usize,
    plane: usize,
    spp: usize,
    width: usize,
    order: ByteOrder,
    signed: bool,
    dst: &mut [u16],
) {
    let row_samples = width * spp;
    match sample_bytes {
        2 => {
            let flip = if signed { 0x8000u16 } else { 0 };
            for (row_idx, out_row) in dst.chunks_mut(width.max(1)).enumerate() {
                let row_base = row_idx * row_samples;
                let mut acc = 0u16;
                for (x, o) in out_row.iter_mut().enumerate() {
                    let off = (row_base + x * spp + plane) * 2;
                    acc = acc.wrapping_add(order.u16(&src[off..off + 2]));
                    *o = acc ^ flip;
                }
            }
        }
        _ => {
            // 8-bit: accumulate bytes, then widen into display space.
            let flip = if signed { 0x80u8 } else { 0 };
            for (row_idx, out_row) in dst.chunks_mut(width.max(1)).enumerate() {
                let row_base = row_idx * row_samples;
                let mut acc = 0u8;
                for (x, o) in out_row.iter_mut().enumerate() {
                    acc = acc.wrapping_add(src[row_base + x * spp + plane]);
                    let b = acc ^ flip;
                    *o = ((b as u16) << 8) | b as u16;
                }
            }
        }
    }
}

/// [`display_fused_pred2`] for raw 8-bit bytes: the running sum, no widening.
pub(super) fn u8_fused_pred2(src: &[u8], plane: usize, spp: usize, width: usize, dst: &mut [u8]) {
    let row_samples = width * spp;
    for (row_idx, out_row) in dst.chunks_mut(width.max(1)).enumerate() {
        let row_base = row_idx * row_samples;
        let mut acc = 0u8;
        for (x, o) in out_row.iter_mut().enumerate() {
            acc = acc.wrapping_add(src[row_base + x * spp + plane]);
            *o = acc;
        }
    }
}

/// Reads one 32-bit sample as `f32`, interpreting the 4 bytes per the frame's
/// sample format (IEEE float, signed, or unsigned integer) and byte order.
pub(super) fn sample_f32(chunk: &[u8], file_order: ByteOrder, format: SampleFormat) -> f32 {
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
pub(super) fn sample_f32_64(chunk: &[u8], file_order: ByteOrder, format: SampleFormat) -> f32 {
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
pub(super) fn wide_sample_f32(
    chunk: &[u8],
    file_order: ByteOrder,
    format: SampleFormat,
    sample_bytes: usize,
) -> f32 {
    if sample_bytes == 8 {
        sample_f32_64(chunk, file_order, format)
    } else {
        sample_f32(chunk, file_order, format)
    }
}

/// Smallest and largest finite values, folded in order. `(0, 1)` when there
/// is no usable range — empty, all-NaN or constant data — so nothing
/// downstream divides by a zero-width window.
#[derive(Clone, Copy)]
pub(super) struct MinMax {
    lo: f32,
    hi: f32,
}

impl MinMax {
    pub fn new() -> Self {
        MinMax {
            lo: f32::INFINITY,
            hi: f32::NEG_INFINITY,
        }
    }

    #[inline]
    pub fn push(&mut self, v: f32) {
        if v.is_finite() {
            self.lo = self.lo.min(v);
            self.hi = self.hi.max(v);
        }
    }

    /// Every wide sample of `src`, in order.
    #[allow(clippy::too_many_arguments)]
    pub fn fold_wide(
        &mut self,
        src: &[u8],
        base: usize,
        stride: usize,
        n: usize,
        order: ByteOrder,
        format: SampleFormat,
        sample_bytes: usize,
    ) {
        if stride == 1 {
            for chunk in src[base * sample_bytes..(base + n) * sample_bytes].chunks_exact(sample_bytes)
            {
                self.push(wide_sample_f32(chunk, order, format, sample_bytes));
            }
        } else {
            for i in 0..n {
                let off = (base + i * stride) * sample_bytes;
                self.push(wide_sample_f32(&src[off..], order, format, sample_bytes));
            }
        }
    }

    pub fn range(self) -> (f32, f32) {
        if !self.lo.is_finite() || !self.hi.is_finite() || self.hi <= self.lo {
            (0.0, 1.0)
        } else {
            (self.lo, self.hi)
        }
    }
}

/// The range of a run of `f32`s (see [`MinMax`]).
#[cfg(test)]
pub(super) fn minmax_f32(values: &[f32]) -> (f32, f32) {
    let mut m = MinMax::new();
    for &v in values {
        m.push(v);
    }
    m.range()
}

/// Display wide: each 32- or 64-bit sample read as `f32`, then rescaled
/// linearly so `lo` maps to 0 and `hi` to 65535, clamped outside.
#[allow(clippy::too_many_arguments)]
pub(super) fn display_wide(
    src: &[u8],
    base: usize,
    stride: usize,
    order: ByteOrder,
    format: SampleFormat,
    sample_bytes: usize,
    (lo, hi): (f32, f32),
    dst: &mut [u16],
) {
    let span = (hi - lo).max(f32::EPSILON);
    for (i, o) in dst.iter_mut().enumerate() {
        let v = wide_sample_f32(
            &src[(base + i * stride) * sample_bytes..],
            order,
            format,
            sample_bytes,
        );
        let t = ((v - lo) / span).clamp(0.0, 1.0);
        *o = (t * 65535.0).round() as u16;
    }
}

/// Convert a run of contiguous native samples to `f32` file units. `src`
/// holds `out.len()` samples of `bits` bits; the depth and format were
/// checked by `require_f32_samples`.
///
/// One loop per depth, sign and byte order, each a plain map from a fixed-size
/// chunk, so the conversion vectorizes rather than branching per sample.
pub(super) fn contiguous_to_f32(
    src: &[u8],
    bits: u16,
    format: SampleFormat,
    order: ByteOrder,
    out: &mut [f32],
) {
    #[inline(always)]
    fn run<const N: usize>(src: &[u8], out: &mut [f32], f: impl Fn([u8; N]) -> f32) {
        for (o, c) in out.iter_mut().zip(src.as_chunks::<N>().0) {
            *o = f(*c);
        }
    }
    let signed = format == SampleFormat::SignedInt;
    match (bits, signed, order) {
        (8, true, _) => run::<1>(src, out, |[b]| b as i8 as f32),
        (8, false, _) => run::<1>(src, out, |[b]| b as f32),
        (16, true, ByteOrder::Little) => run::<2>(src, out, |c| i16::from_le_bytes(c) as f32),
        (16, true, ByteOrder::Big) => run::<2>(src, out, |c| i16::from_be_bytes(c) as f32),
        (16, false, ByteOrder::Little) => run::<2>(src, out, |c| u16::from_le_bytes(c) as f32),
        (16, false, ByteOrder::Big) => run::<2>(src, out, |c| u16::from_be_bytes(c) as f32),
        (64, _, _) => run::<8>(src, out, |c| sample_f32_64(&c, order, format)),
        _ => run::<4>(src, out, |c| sample_f32(&c, order, format)),
    }
}

/// File units: plane `base` of `stride` samples per pixel, as `f32`.
pub(super) fn file_f32(
    src: &[u8],
    base: usize,
    stride: usize,
    bits: u16,
    format: SampleFormat,
    order: ByteOrder,
    dst: &mut [f32],
) {
    let sb = bits as usize / 8;
    let n = dst.len();
    if stride == 1 {
        // Contiguous: every single-sample frame, and every planar plane.
        contiguous_to_f32(&src[base * sb..(base + n) * sb], bits, format, order, dst);
    } else {
        // A strided gather out of chunky multi-sample data.
        for (i, o) in dst.iter_mut().enumerate() {
            let off = (base + i * stride) * sb;
            contiguous_to_f32(
                &src[off..off + sb],
                bits,
                format,
                order,
                std::slice::from_mut(o),
            );
        }
    }
}

/// Copy every `stride`-th byte from `native[base..]` into `out`.
///
/// Written as a bounded slice plus `step_by` rather than `native[base + i *
/// stride]` per sample. The indexed form leaves a bounds check (and its panic
/// path) inside the loop unless the optimiser can prove the range, which it
/// only manages when the surrounding code happens to inline a certain way — so
/// the same source compiled twice was measured at both 88 and 168 us for one
/// 256x256 RGB frame, flipping on edits elsewhere in the crate. Bounding the
/// source once takes the check out of the loop and pins the fast form.
#[inline]
pub(super) fn gather_u8(native: &[u8], base: usize, stride: usize, out: &mut [u8]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    if stride == 1 {
        let end = base
            .checked_add(out.len())
            .context(ErrorKind::TooLarge, "plane offset overflows")?;
        let src = native
            .get(base..end)
            .context(ErrorKind::Malformed, "plane data out of bounds")?;
        out.copy_from_slice(src);
        return Ok(());
    }
    // Exactly the span the gather reads: the last sample is the only one whose
    // whole pixel need not be present.
    let end = (out.len() - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(base + 1))
        .context(ErrorKind::TooLarge, "plane offset overflows")?;
    let src = native
        .get(base..end)
        .context(ErrorKind::Malformed, "plane data out of bounds")?;
    for (o, &b) in out.iter_mut().zip(src.iter().step_by(stride)) {
        *o = b;
    }
    Ok(())
}

/// Stored samples of plane `base`, `stride` samples per pixel, each
/// `sample_bytes` wide, into `dst` (`sample_bytes` bytes per pixel), byte
/// order reversed when `swap`.
pub(super) fn stored(
    src: &[u8],
    base: usize,
    stride: usize,
    sample_bytes: usize,
    swap: bool,
    dst: &mut [u8],
) -> Result<()> {
    if sample_bytes == 1 {
        return gather_u8(src, base, stride, dst);
    }
    let n = dst.len() / sample_bytes;
    if stride == 1 {
        dst.copy_from_slice(&src[base * sample_bytes..(base + n) * sample_bytes]);
    } else {
        for (i, o) in dst.chunks_exact_mut(sample_bytes).enumerate() {
            let off = (base + i * stride) * sample_bytes;
            o.copy_from_slice(&src[off..off + sample_bytes]);
        }
    }
    if swap {
        swap_samples(dst, sample_bytes);
    }
    Ok(())
}

/// Reverse the bytes of every `sample_bytes`-wide sample in place.
pub(super) fn swap_samples(data: &mut [u8], sample_bytes: usize) {
    match sample_bytes {
        2 => data
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .for_each(|c| c.reverse()),
        4 => data
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .for_each(|c| c.reverse()),
        8 => data
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .for_each(|c| c.reverse()),
        _ => {}
    }
}

/// 16-bit samples in the file's byte order, in place, to the host's `u16`
/// with the sign bit flipped when signed: what a [`Display`] read of 16-bit
/// data is, done over a strip that was decompressed straight into the
/// destination.
///
/// [`Display`]: crate::units::Display
pub(super) fn display16_in_place(data: &mut [u8], order: ByteOrder, signed: bool) {
    let swap = order != ByteOrder::host();
    if !swap && !signed {
        return;
    }
    let flip = if signed { 0x8000u16 } else { 0 };
    match bytemuck::try_cast_slice_mut::<u8, u16>(data) {
        Ok(v) if swap => v.iter_mut().for_each(|x| *x = x.swap_bytes() ^ flip),
        Ok(v) => v.iter_mut().for_each(|x| *x ^= flip),
        // Only a destination carved from a `&mut [u16]` reaches this, so it is
        // aligned; the byte-wise form is here so nothing has to assume it.
        Err(_) => {
            for c in data.as_chunks_mut::<2>().0 {
                *c = (order.u16(c) ^ flip).to_ne_bytes();
            }
        }
    }
}

/// Split a chunky 8-bit run into one plane per sample, reading the source
/// **once**: pixels `at..at + n` of every destination.
///
/// The per-plane gather it replaces walked the whole frame once per sample,
/// each pass touching every cache line to take one byte in `spp` — three reads
/// of a 192 KB RGB frame to produce 192 KB of output, at about 4.6 GB/s. One
/// pass gets 6.7, and on x86 a `pshufb` de-interleave gets 27: for a 256x256
/// RGB8 frame, 85 us down to 14.5.
pub(super) fn deinterleave_into(
    src: &[u8],
    spp: usize,
    outs: &mut [&mut [u8]],
    at: usize,
    n: usize,
) -> Result<()> {
    if n == 0 || spp == 0 {
        return Ok(());
    }
    let need = n
        .checked_mul(spp)
        .context(ErrorKind::TooLarge, "frame size overflows")?;
    let src = src
        .get(..need)
        .context(ErrorKind::Malformed, "plane data out of file bounds")?;

    #[cfg(target_arch = "x86_64")]
    if spp == 3 && outs.len() >= 3 && std::arch::is_x86_feature_detected!("ssse3") {
        let (a, rest) = outs.split_at_mut(1);
        let (b, c) = rest.split_at_mut(1);
        // SAFETY: SSSE3 is confirmed present just above, and the three
        // destinations are distinct slices of `n` bytes each while `src` holds
        // exactly `n * 3` — which is what `deinterleave_rgb8_ssse3` requires.
        unsafe {
            deinterleave_rgb8_ssse3(
                src,
                &mut a[0][at..at + n],
                &mut b[0][at..at + n],
                &mut c[0][at..at + n],
            )
        };
        return Ok(());
    }

    for (i, px) in src.chunks_exact(spp).enumerate() {
        for (o, &v) in outs.iter_mut().zip(px.iter()) {
            o[at + i] = v;
        }
    }
    Ok(())
}

/// [`deinterleave_into`] over whole planes, the shape the unit tests drive.
#[cfg(test)]
pub(super) fn deinterleave_u8<P: AsMut<[u8]>>(
    native: &[u8],
    spp: usize,
    out: &mut [P],
) -> Result<()> {
    let n = out.first_mut().map_or(0, |p| p.as_mut().len());
    let mut planes: Vec<&mut [u8]> = out.iter_mut().map(|p| p.as_mut()).collect();
    deinterleave_into(native, spp, &mut planes, 0, n)
}

/// `pshufb` de-interleave of chunky RGB8: 48 source bytes -> 16 each of R, G, B.
///
/// Each output register is assembled from all three input registers, since a
/// 16-byte window of RGB triples straddles them; `_mm_shuffle_epi8` zeroes the
/// lanes a given input does not supply, so the three shuffles OR together
/// cleanly.
///
/// # Safety
/// The caller must have confirmed SSSE3 at run time. `r`, `g` and `b` must all
/// be `n` bytes and `src` exactly `3 * n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn deinterleave_rgb8_ssse3(src: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
    use std::arch::x86_64::*;
    let n = r.len();
    debug_assert_eq!(src.len(), n * 3);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);

    // A 16-pixel window spans 48 source bytes, i.e. all three input registers,
    // and each channel takes a different count from each (R: 6/5/5, G: 5/6/5,
    // B: 5/5/6) at different lane positions — so the masks are written out
    // rather than derived. A mask byte with its high bit set zeroes that lane,
    // which is what lets the three shuffles OR together; a byte of 16 would
    // *not* zero it, it would select lane 0.
    const Z: i8 = -1;
    #[rustfmt::skip]
    let (mr, mg, mb) = (
        [
            _mm_setr_epi8(0, 3, 6, 9, 12, 15, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, Z, 2, 5, 8, 11, 14, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 1, 4, 7, 10, 13),
        ],
        [
            _mm_setr_epi8(1, 4, 7, 10, 13, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, 0, 3, 6, 9, 12, 15, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 2, 5, 8, 11, 14),
        ],
        [
            _mm_setr_epi8(2, 5, 8, 11, 14, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, 1, 4, 7, 10, 13, Z, Z, Z, Z, Z, Z),
            _mm_setr_epi8(Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 0, 3, 6, 9, 12, 15),
        ],
    );

    let mut i = 0usize;
    while i + 16 <= n {
        let p = src.as_ptr().add(i * 3) as *const __m128i;
        let x0 = _mm_loadu_si128(p);
        let x1 = _mm_loadu_si128(p.add(1));
        let x2 = _mm_loadu_si128(p.add(2));
        let pick = |m: &[__m128i; 3]| {
            _mm_or_si128(
                _mm_or_si128(_mm_shuffle_epi8(x0, m[0]), _mm_shuffle_epi8(x1, m[1])),
                _mm_shuffle_epi8(x2, m[2]),
            )
        };
        _mm_storeu_si128(r.as_mut_ptr().add(i) as *mut __m128i, pick(&mr));
        _mm_storeu_si128(g.as_mut_ptr().add(i) as *mut __m128i, pick(&mg));
        _mm_storeu_si128(b.as_mut_ptr().add(i) as *mut __m128i, pick(&mb));
        i += 16;
    }
    // Tail: fewer than 16 pixels left.
    while i < n {
        r[i] = src[i * 3];
        g[i] = src[i * 3 + 1];
        b[i] = src[i * 3 + 2];
        i += 1;
    }
}

/// One row of 4-bit samples, packed two per byte high nibble first (TIFF's
/// FillOrder-1 default), unpacked to one value per pixel.
pub(super) fn unpack4<T>(packed: &[u8], dst: &mut [T], conv: impl Fn(u8) -> T) {
    for (x, o) in dst.iter_mut().enumerate() {
        let byte = packed[x / 2];
        *o = conv(if x % 2 == 0 { byte >> 4 } else { byte & 0x0f });
    }
}

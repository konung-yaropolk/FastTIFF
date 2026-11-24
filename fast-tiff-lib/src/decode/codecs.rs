//! The per-piece codec stage: one strip or tile's compressed bytes into its
//! destination, and the predictor undone over it. Moved here unchanged from
//! the single-file decoder; everything above this stage decides *where* a
//! piece's bytes go, and everything here only how they are produced.

use crate::error::{bail, err, Result};
use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo};

thread_local! {
    /// One scratch row for the floating-point predictor undo, kept across
    /// calls (see `undo_float_predictor`).
    static FLOAT_PRED_ROW: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Inflate one Deflate strip with a **reused** zlib state.
///
/// Constructing a `ZlibDecoder` per strip allocates and initialises zlib's
/// window and Huffman tables every time, which on a 128 KiB strip costs more
/// than half the decode: 47 us fresh against 20 us reused.
///
/// Its own function, not an arm of `decompress_into`: a `thread_local!` with a
/// non-const initialiser expands to a lazy-init path plus destructor
/// registration, and having two of those inline in the codec `match` showed up
/// as a several-percent slowdown in the *LZW* arm, which touches neither.
fn inflate_into(raw: &[u8], dest: &mut [u8]) -> Result<usize> {
    thread_local! {
        static INFLATE: std::cell::RefCell<flate2::Decompress> =
            std::cell::RefCell::new(flate2::Decompress::new(true));
    }
    INFLATE.with(|cell| {
        let mut d = cell.borrow_mut();
        d.reset(true);
        let mut written = 0usize;
        while written < dest.len() {
            let (before_in, before_out) = (d.total_in(), d.total_out());
            let status = d
                .decompress(
                    &raw[before_in as usize..],
                    &mut dest[written..],
                    flate2::FlushDecompress::Finish,
                )
                .map_err(|e| err!(Malformed, "Deflate decode failed: {e}"))?;
            written = d.total_out() as usize;
            if status == flate2::Status::StreamEnd {
                break;
            }
            // No forward progress: truncated or corrupt input. Stop with what
            // we have, exactly as the streaming reader did on EOF.
            if d.total_in() == before_in && d.total_out() == before_out {
                break;
            }
        }
        Ok(written)
    })
}

/// Decompress one ZSTD strip with a **reused** decompression context, for the
/// same reason as [`inflate_into`] — and it matters more here: a fresh context
/// per strip cost 40% of a 256x256 frame and 17% of a 2048x2048 one.
#[cfg(feature = "codec-zstd")]
fn zstd_into(raw: &[u8], dest: &mut [u8]) -> Result<usize> {
    thread_local! {
        static ZSTD: std::cell::RefCell<zstd::bulk::Decompressor<'static>> =
            std::cell::RefCell::new(zstd::bulk::Decompressor::new().expect("ZSTD context"));
    }
    // One-shot decompression fails rather than truncating when the frame
    // declares more than `dest` holds — which a padded strip legitimately
    // does, and so does every tile hanging off the bottom of the image, whose
    // rows past it are not decompressed. Those go to the streaming decoder,
    // which stops at the cap; a frame header that says so up front costs a
    // parse, not a doomed decode first.
    let too_big = matches!(
        zstd::zstd_safe::get_frame_content_size(raw),
        Ok(Some(size)) if size > dest.len() as u64
    );
    if !too_big {
        if let Ok(n) = ZSTD.with(|cell| cell.borrow_mut().decompress_to_buffer(raw, dest)) {
            return Ok(n);
        }
    }
    zstd_stream_into(raw, dest).map_err(|e| err!(Malformed, "ZSTD decode failed: {e}"))
}

/// Stream-decompress one ZSTD strip into `dest`, stopping when it is full,
/// with a context kept across calls: a fresh `zstd::stream::read::Decoder`
/// per piece was a context and a 128 KiB input buffer allocated for every
/// edge tile of every frame.
///
/// It does exactly what reading that decoder to the cap did — the input fed
/// in the same chunks its buffered reader took, concatenated frames decoded
/// one after another, a frame the input ends inside of an error unless `dest`
/// filled first — so a piece decodes to the same bytes, or fails, as before.
#[cfg(feature = "codec-zstd")]
pub(super) fn zstd_stream_into(raw: &[u8], dest: &mut [u8]) -> std::io::Result<usize> {
    use zstd::stream::raw::{Decoder, InBuffer, Operation, OutBuffer};
    thread_local! {
        static STREAM: std::cell::RefCell<Option<Decoder<'static>>> =
            const { std::cell::RefCell::new(None) };
    }
    STREAM.with(|cell| {
        let mut slot = cell.borrow_mut();
        let d = match &mut *slot {
            Some(d) => d,
            None => slot.insert(Decoder::new()?),
        };
        d.reinit()?;
        let chunk = zstd::zstd_safe::DCtx::in_size().max(1);
        let (mut pos, mut window_end, mut filled) = (0usize, 0usize, 0usize);
        let mut finished_frame = false;
        while filled < dest.len() {
            if pos == window_end {
                window_end = (pos + chunk).min(raw.len());
            }
            let input = &raw[pos..window_end];
            if input.is_empty() {
                if finished_frame {
                    break;
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "incomplete frame",
                ));
            }
            if finished_frame {
                d.reinit()?;
                finished_frame = false;
            }
            let mut src = InBuffer::around(input);
            let mut out = OutBuffer::around(&mut dest[filled..]);
            if d.run(&mut src, &mut out)? == 0 {
                finished_frame = true;
            }
            pos += src.pos();
            filled += out.pos();
        }
        Ok(filled)
    })
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
pub(crate) fn decompress_into(
    raw: &[u8],
    compression: Compression,
    dest: &mut [u8],
) -> Result<usize> {
    match compression {
        Compression::None => {
            let n = raw.len().min(dest.len());
            dest[..n].copy_from_slice(&raw[..n]);
            Ok(n)
        }
        Compression::Lzw => {
            let mut decoder =
                weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
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
                        if out_pos == dest.len() || (res.consumed_in == 0 && res.consumed_out == 0)
                        {
                            break;
                        }
                    }
                    Ok(weezl::LzwStatus::NoProgress) => break, // input exhausted early
                    Err(e) => bail!(Malformed, "LZW decode failed: {e:?}"),
                }
            }
            Ok(out_pos)
        }
        Compression::Deflate => inflate_into(raw, dest),
        Compression::PackBits => Ok(packbits_decode_into(raw, dest)),
        #[cfg(feature = "codec-zstd")]
        Compression::Zstd => zstd_into(raw, dest),
        // Without the codec the tag is still recognized — the file just can't be
        // decoded here. Saying so beats the generic "unsupported scheme" arm,
        // which would leave a wasm user guessing why a file that opens natively
        // fails in the browser.
        #[cfg(not(feature = "codec-zstd"))]
        Compression::Zstd => bail!(
            Unavailable,
            "this frame uses ZSTD compression, but the `codec-zstd` feature is off \
             (it needs a C toolchain, so it is unavailable on wasm)"
        ),
        Compression::Other(code) => {
            bail!(Unsupported, "unsupported TIFF compression scheme: {code}")
        }
    }
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
pub(crate) fn packbits_decode(input: &[u8], expected_len: usize) -> Vec<u8> {
    let mut out = vec![0u8; expected_len];
    let n = packbits_decode_into(input, &mut out);
    out.truncate(n);
    out
}

/// The one operation [`hor_acc_native`] needs from a sample type.
trait WrappingAdd: Copy + bytemuck::Pod {
    fn wadd(self, other: Self) -> Self;
}

macro_rules! impl_wrapping_add {
    ($($t:ty),*) => {$(
        impl WrappingAdd for $t {
            #[inline(always)]
            fn wadd(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
        }
    )*};
}
impl_wrapping_add!(u16, u32, u64);

/// Undo horizontal differencing over samples in their **native machine
/// representation**, for a file whose byte order matches the host's.
///
/// The byte-wise fallback in [`undo_predictor`] loads two bytes, assembles an
/// integer, adds, splits the result and stores two bytes — for every sample.
/// When a little-endian file is read on a little-endian host, which is very
/// nearly every file, the bytes already *are* the sample, so one load, one add
/// and one store will do. This is what libtiff's `horAcc16` does, and it is
/// worth about 3x on frames large enough for the predictor to dominate: a
/// 2048x2048 u16 frame spends more time undoing the predictor than it does in
/// the codec, so this is the difference between winning and losing that run.
///
/// Returns `false` when the fast path does not apply — a foreign byte order, or
/// a buffer not aligned for the wider type. The caller then runs the byte-wise
/// loop, which has no alignment requirement. (In practice the allocator returns
/// blocks aligned well past 8 bytes, but a `Vec<u8>` makes no such promise and
/// a misaligned cast is UB, so it is checked rather than assumed.)
fn hor_acc_native<T: WrappingAdd>(
    data: &mut [u8],
    row_bytes: usize,
    row_samples: usize,
    stride: usize,
    file_order: ByteOrder,
) -> bool {
    if file_order != ByteOrder::host() {
        return false;
    }
    // Whole rows only. A trailing partial row is left untouched, exactly as
    // `chunks_exact_mut` leaves it on the byte-wise path.
    let whole = data.len() - data.len() % row_bytes;
    let Ok(samples) = bytemuck::try_cast_slice_mut::<u8, T>(&mut data[..whole]) else {
        return false;
    };
    for row in samples.chunks_exact_mut(row_samples) {
        // Serial by nature — each sample depends on the one `stride` back — so
        // this cannot vectorize. The gain is in the per-step cost, not in width.
        for i in stride..row_samples {
            row[i] = row[i].wadd(row[i - stride]);
        }
    }
    true
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
#[cfg(test)]
pub(crate) fn undo_predictor(
    mut data: Vec<u8>,
    frame: &FrameInfo,
    sample_bytes: usize,
    file_order: ByteOrder,
) -> Result<Vec<u8>> {
    undo_predictor_in_place(&mut data, frame, sample_bytes, file_order)?;
    Ok(data)
}

/// [`undo_predictor`] over a buffer the caller already owns.
///
/// Split out so the general decode path can undo the predictor **per strip**,
/// immediately after that strip decompresses, while its rows are still in
/// cache. Differencing resets at every row and a strip always holds a whole
/// number of rows, so strip-at-a-time and whole-frame give identical results —
/// but the whole-frame pass re-reads and rewrites the entire buffer from DRAM,
/// which on a 2048x2048 frame costs more than the codec does.
pub(crate) fn undo_predictor_in_place(
    data: &mut [u8],
    frame: &FrameInfo,
    sample_bytes: usize,
    file_order: ByteOrder,
) -> Result<()> {
    undo_predictor_rows(
        data,
        frame.width as usize,
        (frame.samples_per_pixel as usize).max(1),
        frame.is_planar(),
        frame.predictor,
        sample_bytes,
        file_order,
    )
}

/// [`undo_predictor_in_place`] over rows of `width` pixels described by their
/// numbers rather than a frame — what a tile needs, whose rows are the tile's
/// width and not the image's.
pub(crate) fn undo_predictor_rows(
    data: &mut [u8],
    width: usize,
    spp: usize,
    planar: bool,
    predictor: u16,
    sample_bytes: usize,
    file_order: ByteOrder,
) -> Result<()> {
    // Distance (in samples) to the pixel left of this one: `spp` when samples
    // interleave per pixel, 1 when each plane is stored whole. Rows follow the
    // same split — a planar frame has `height * spp` rows of `width` samples,
    // and `chunks_exact_mut` below walks them all.
    let stride = if planar { 1 } else { spp };
    let row_samples = width * stride;
    let row_bytes = row_samples * sample_bytes;
    if row_bytes == 0 {
        return Ok(()); // zero-width frame: nothing to difference
    }

    match (predictor, sample_bytes) {
        (1, _) => return Ok(()),
        (2, 1) => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in stride..row_samples {
                    row[i] = row[i].wrapping_add(row[i - stride]);
                }
            }
        }
        (2, 2) => {
            if !hor_acc_native::<u16>(data, row_bytes, row_samples, stride, file_order) {
                for row in data.chunks_exact_mut(row_bytes) {
                    for i in stride..row_samples {
                        let (prev_off, cur_off) = ((i - stride) * 2, i * 2);
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
        }
        (2, 4) => {
            // 32-bit integer horizontal differencing (libtiff writes these).
            if !hor_acc_native::<u32>(data, row_bytes, row_samples, stride, file_order) {
                for row in data.chunks_exact_mut(row_bytes) {
                    for i in stride..row_samples {
                        let (prev_off, cur_off) = ((i - stride) * 4, i * 4);
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
        }
        (2, 8) => {
            // 64-bit integer horizontal differencing.
            if !hor_acc_native::<u64>(data, row_bytes, row_samples, stride, file_order) {
                for row in data.chunks_exact_mut(row_bytes) {
                    for i in stride..row_samples {
                        let (prev_off, cur_off) = ((i - stride) * 8, i * 8);
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
        }
        (3, 4) | (3, 8) => undo_float_predictor(data, row_bytes, stride, sample_bytes, file_order),
        (2, other) => bail!(
            Unsupported,
            "predictor 2 undo not implemented for {other}-byte samples"
        ),
        (3, other) => bail!(
            Unsupported,
            "floating-point predictor requires 32- or 64-bit samples, got {other}-byte"
        ),
        (other, _) => bail!(Unsupported, "unsupported TIFF predictor: {other}"),
    }
    Ok(())
}

/// Undo TIFF Predictor 3 (TIFF TechNote 3 floating-point horizontal
/// differencing), per row: first undo the byte-level differencing (`stride` =
/// samples per pixel for chunky rows, 1 for planar ones, mirroring libtiff's
/// `fpAcc`), then gather each float's bytes back from the row's `sample_bytes`
/// byte-significance planes — the spec stores them MSB-plane-first
/// regardless of the file's byte order — and store
/// the value in `file_order` for the normal downstream reads. Handles both f32
/// (`sample_bytes == 4`) and f64 (`sample_bytes == 8`).
fn undo_float_predictor(
    data: &mut [u8],
    row_bytes: usize,
    stride: usize,
    sample_bytes: usize,
    file_order: ByteOrder,
) {
    let wc = row_bytes / sample_bytes; // float values per row
                                       // `row_bytes` is `width * stride * sample_bytes` — the one size in this file
                                       // with no `height` factor, so it is *not* bounded by `decoded_len`. It stays
                                       // safe only because `validate_frames` rejects a zero-height frame, which
                                       // makes `row_bytes <= decoded_len` and so already-vetted. Size it from what
                                       // we were actually handed rather than trusting that chain: a scratch buffer
                                       // wider than one row of `data` can never be needed, and the process must not
                                       // die allocating a buffer for rows that do not exist.
    let scratch_len = row_bytes.min(data.len());
    if scratch_len < row_bytes {
        return; // no whole row to transform; `chunks_exact_mut` would yield none
    }
    // One scratch row per thread, not per call. The predictor undo runs once
    // per strip now that it is fused into the decompression loop, so a fresh
    // `vec![0u8; row_bytes]` here would be an allocation per strip per frame —
    // 32 of them on a 2048x2048 frame where there used to be one.
    FLOAT_PRED_ROW.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < scratch_len {
            scratch.resize(scratch_len, 0);
        }
        let scratch = &mut scratch[..scratch_len];
        for row in data.chunks_exact_mut(row_bytes) {
            for i in stride..row_bytes {
                row[i] = row[i].wrapping_add(row[i - stride]);
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
            row.copy_from_slice(scratch);
        }
    });
}

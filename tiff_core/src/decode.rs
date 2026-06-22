//! Turns a `FrameInfo` into actual pixel data. The fast path — uncompressed
//! strips, native byte order, single strip per frame (ImageJ's default when
//! saving raw stacks) — does zero decoding work at all: it's a direct
//! reinterpret-cast of the memory-mapped file bytes. Everything else
//! (LZW/Deflate/PackBits, multi-strip, predictor, byte-swap, 8-bit upcast,
//! 32-bit float) falls back to an owned `Vec<u16>`.

use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo, SampleFormat};
use anyhow::{anyhow, bail, Result};
use std::borrow::Cow;

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
    if frame.compression == Compression::None
        && frame.bits_per_sample == 16
        && frame.sample_format != SampleFormat::SignedInt
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

    // --- General path ---
    let native_bytes = decode_native_bytes(mmap, frame, file_order)?;

    // ImageJ stores signed-integer images by offsetting into unsigned space
    // for display (a signed `v` is shown/windowed as `v + 2^(bits-1)`), and it
    // writes the display window (`min=`/`max=`) in that same offset space. To
    // match — so a signed-int16 file and the equivalent unsigned+calibration
    // file render identically — flip the sign bit, which is exactly that
    // offset modulo the sample range (XOR 0x8000 == +32768 for 16-bit).
    let signed = frame.sample_format == SampleFormat::SignedInt;
    let mut out = vec![0u16; n_samples];
    match frame.bits_per_sample {
        16 => {
            for (i, chunk) in native_bytes.chunks_exact(2).enumerate().take(n_samples) {
                let v = file_order.u16(chunk);
                out[i] = if signed { v ^ 0x8000 } else { v };
            }
        }
        8 => {
            for (i, &b) in native_bytes.iter().enumerate().take(n_samples) {
                let b = if signed { b ^ 0x80 } else { b };
                out[i] = ((b as u16) << 8) | b as u16;
            }
        }
        32 => {
            let floats = bytes_to_f32(&native_bytes, file_order, n_samples);
            let (lo, hi) = float_range.unwrap_or_else(|| minmax_f32(&floats));
            let span = (hi - lo).max(f32::EPSILON);
            for (i, &v) in floats.iter().enumerate() {
                let t = ((v - lo) / span).clamp(0.0, 1.0);
                out[i] = (t * 65535.0).round() as u16;
            }
        }
        _ => unreachable!(), // decode_native_bytes already rejects anything else
    }

    Ok(Cow::Owned(out))
}

/// The actual min/max of a 32-bit float frame's raw values — used to
/// establish a channel's initial display range, the same way ImageJ
/// auto-ranges a float image to its own data instead of assuming a fixed
/// scale. `None` for non-float frames.
pub fn frame_float_minmax(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Option<(f32, f32)>> {
    if frame.sample_format != SampleFormat::Float || frame.bits_per_sample != 32 {
        return Ok(None);
    }
    let native_bytes = decode_native_bytes(mmap, frame, file_order)?;
    let n_samples = frame.width as usize * frame.height as usize * frame.samples_per_pixel as usize;
    Ok(Some(minmax_f32(&bytes_to_f32(&native_bytes, file_order, n_samples))))
}

fn bytes_to_f32(bytes: &[u8], file_order: ByteOrder, n_samples: usize) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .take(n_samples)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            match file_order {
                ByteOrder::Little => f32::from_le_bytes(arr),
                ByteOrder::Big => f32::from_be_bytes(arr),
            }
        })
        .collect()
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
fn decode_native_bytes(mmap: &[u8], frame: &FrameInfo, file_order: ByteOrder) -> Result<Vec<u8>> {
    let sample_bytes = match frame.bits_per_sample {
        16 => 2,
        8 => 1,
        32 => 4,
        other => bail!("unsupported bits_per_sample: {other}"),
    };
    let row_bytes = frame.width as usize * frame.samples_per_pixel as usize * sample_bytes;
    let total_rows = frame.height as usize;
    let rows_per_strip = (frame.rows_per_strip as usize).max(1);

    let mut native = Vec::with_capacity(total_rows * row_bytes);
    let mut rows_done = 0usize;
    for (&offset, &len) in frame.strip_offsets.iter().zip(frame.strip_byte_counts.iter()) {
        let raw_strip = mmap
            .get(offset as usize..(offset + len) as usize)
            .ok_or_else(|| anyhow!("strip at offset {offset} (len {len}) out of file bounds"))?;
        // The last strip may legitimately have fewer rows than
        // `rows_per_strip` if the image height doesn't divide evenly.
        let rows_this_strip = rows_per_strip.min(total_rows.saturating_sub(rows_done));
        let expected_len = rows_this_strip * row_bytes;
        native.extend_from_slice(&decompress(raw_strip, frame.compression, expected_len)?);
        rows_done += rows_this_strip;
    }

    if native.len() < total_rows * row_bytes {
        bail!(
            "decoded {} bytes but expected {} for a {}x{} frame ({} bytes/sample) — \
             strip data is shorter than the declared image size",
            native.len(),
            total_rows * row_bytes,
            frame.width,
            frame.height,
            sample_bytes
        );
    }

    let unpredicted = undo_predictor(native, frame, sample_bytes, file_order)?;
    Ok(unpredicted)
}

fn decompress(raw: &[u8], compression: Compression, expected_len: usize) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(raw.to_vec()),
        Compression::Lzw => {
            let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let mut out = Vec::with_capacity(expected_len);
            decoder
                .into_stream(&mut out)
                .decode_all(raw)
                .status
                .map_err(|e| anyhow!("LZW decode failed: {e:?}"))?;
            Ok(out)
        }
        Compression::Deflate => {
            use std::io::Read;
            let mut decoder = flate2::read::ZlibDecoder::new(raw);
            let mut out = Vec::with_capacity(expected_len);
            decoder
                .read_to_end(&mut out)
                .map_err(|e| anyhow!("Deflate decode failed: {e}"))?;
            Ok(out)
        }
        Compression::PackBits => Ok(packbits_decode(raw, expected_len)),
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

/// Undo TIFF Predictor=2 (horizontal differencing). Operates per scanline,
/// per sample (so RGB/multi-sample data differences each channel
/// independently), matching the TIFF6 spec. Reads/writes in `file_order`
/// since this runs before the final byte-order normalization pass. Strip
/// boundaries are always a whole number of rows, so running this once on
/// the full concatenated buffer (rather than per-strip before
/// concatenating) gives the same result, since differencing resets at
/// every row regardless of which strip it came from.
fn undo_predictor(
    mut data: Vec<u8>,
    frame: &FrameInfo,
    sample_bytes: usize,
    file_order: ByteOrder,
) -> Result<Vec<u8>> {
    if frame.predictor != 2 {
        return Ok(data);
    }
    let spp = frame.samples_per_pixel as usize;
    let row_samples = frame.width as usize * spp;
    let row_bytes = row_samples * sample_bytes;

    match sample_bytes {
        2 => {
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
        1 => {
            for row in data.chunks_exact_mut(row_bytes) {
                for i in spp..row_samples {
                    row[i] = row[i].wrapping_add(row[i - spp]);
                }
            }
        }
        _ => bail!("predictor undo not implemented for {sample_bytes}-byte samples"),
    }
    Ok(std::mem::take(&mut data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packbits_literal_run() {
        // n=2 (literal run of 3 bytes) then n=-2 (repeat next byte 3 times)
        let input = [2u8, 0xAA, 0xBB, 0xCC, (-2i8) as u8, 0xFF];
        let out = packbits_decode(&input, 6);
        assert_eq!(out, vec![0xAA, 0xBB, 0xCC, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn packbits_noop_byte_is_skipped() {
        let input = [0u8, 0x42, (-128i8) as u8, 0u8, 0x99];
        let out = packbits_decode(&input, 3);
        // first record: literal run of 1 byte (0x42)
        // second record: -128 is a documented no-op, skipped entirely
        // third record: literal run of 1 byte (0x99)
        assert_eq!(out, vec![0x42, 0x99]);
    }

    fn make_frame(width: u32, height: u32, predictor: u16) -> FrameInfo {
        FrameInfo {
            width,
            height,
            bits_per_sample: 16,
            samples_per_pixel: 1,
            sample_format: crate::index::SampleFormat::UnsignedInt,
            compression: Compression::None,
            predictor,
            strip_offsets: vec![0],
            strip_byte_counts: vec![(width * height * 2) as u64],
            rows_per_strip: height,
        }
    }

    #[test]
    fn predictor_undo_le_roundtrip() {
        // Two rows of 4 pixels each. Original values:
        let original: [u16; 8] = [100, 105, 90, 200, 1000, 999, 999, 1005];
        let frame = make_frame(4, 2, 2);

        // Encode (horizontal differencing) to build the "compressed" input.
        let mut differenced = [0u16; 8];
        for row in 0..2 {
            let base = row * 4;
            differenced[base] = original[base];
            for i in 1..4 {
                differenced[base + i] = original[base + i].wrapping_sub(original[base + i - 1]);
            }
        }
        let mut bytes = Vec::new();
        for v in differenced {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        let restored = undo_predictor(bytes, &frame, 2, ByteOrder::Little).unwrap();
        let mut out = [0u16; 8];
        for (i, chunk) in restored.chunks_exact(2).enumerate() {
            out[i] = ByteOrder::Little.u16(chunk);
        }
        assert_eq!(out, original);
    }

    #[test]
    fn predictor_undo_be_roundtrip() {
        // Same as above but file byte order is big-endian — this is exactly
        // the bug that was caught and fixed: undo must read/write in the
        // file's declared order, not native order.
        let original: [u16; 4] = [5000, 5010, 4990, 5200];
        let frame = make_frame(4, 1, 2);

        let mut differenced = [0u16; 4];
        differenced[0] = original[0];
        for i in 1..4 {
            differenced[i] = original[i].wrapping_sub(original[i - 1]);
        }
        let mut bytes = Vec::new();
        for v in differenced {
            bytes.extend_from_slice(&v.to_be_bytes());
        }

        let restored = undo_predictor(bytes, &frame, 2, ByteOrder::Big).unwrap();
        let mut out = [0u16; 4];
        for (i, chunk) in restored.chunks_exact(2).enumerate() {
            out[i] = ByteOrder::Big.u16(chunk);
        }
        assert_eq!(out, original);
    }

    #[test]
    fn minmax_f32_handles_normal_nan_and_constant_data() {
        assert_eq!(minmax_f32(&[1.5, -2.0, 3.25, f32::NAN]), (-2.0, 3.25));
        assert_eq!(minmax_f32(&[]), (0.0, 1.0));
        assert_eq!(minmax_f32(&[f32::NAN, f32::NAN]), (0.0, 1.0));
        assert_eq!(minmax_f32(&[4.0, 4.0, 4.0]), (0.0, 1.0)); // constant -- no usable span
    }

    fn float_frame(width: u32, height: u32) -> FrameInfo {
        FrameInfo {
            width,
            height,
            bits_per_sample: 32,
            samples_per_pixel: 1,
            sample_format: SampleFormat::Float,
            compression: Compression::None,
            predictor: 1,
            strip_offsets: vec![0],
            strip_byte_counts: vec![(width * height * 4) as u64],
            rows_per_strip: height,
        }
    }

    #[test]
    fn float_data_rescales_like_imagej_auto_contrast() {
        // Not 0..65535-shaped at all -- a realistic ratiometric range.
        let values: [f32; 4] = [-0.5, 0.0, 1.0, 2.5];
        let mut file = Vec::new();
        for v in values {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let frame = float_frame(2, 2);

        // float_range = None -> auto-ranges to this frame's own min/max,
        // same as ImageJ auto-contrasting a float image to its own data.
        let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, None).unwrap();
        assert_eq!(pixels[0], 0, "min value should map to 0");
        assert_eq!(pixels[3], 65535, "max value should map to 65535");
        // 0.0 is 1/6 of the way from -0.5 to 2.5.
        let expected_mid = (65535.0_f64 * (1.0 / 6.0)).round() as u16;
        assert!(
            pixels[1].abs_diff(expected_mid) <= 1,
            "got {}, expected close to {expected_mid}",
            pixels[1]
        );
    }

    #[test]
    fn float_data_respects_an_explicit_range() {
        // A fixed display range, e.g. one established from the first frame
        // and reused for every subsequent frame in the same channel so
        // contrast doesn't jump around as you scrub.
        let values: [f32; 2] = [0.0, 10.0];
        let mut file = Vec::new();
        for v in values {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let frame = float_frame(2, 1);

        let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, Some((0.0, 100.0))).unwrap();
        assert_eq!(pixels[0], 0);
        // 10.0 out of a 0..100 range is exactly 10%.
        let expected = (65535.0_f64 * 0.10).round() as u16;
        assert!(pixels[1].abs_diff(expected) <= 1);
    }

    #[test]
    fn signed_int16_is_offset_into_unsigned_display_space() {
        // ImageJ stores signed-16 images and windows them in unsigned+32768
        // space. A signed -72 must decode to 32696, signed 0 to 32768, etc. —
        // so a signed file lines up with the equivalent unsigned+calibration
        // file (and the same min=/max= window applies to both).
        let mut frame = make_frame(2, 2, 1);
        frame.sample_format = SampleFormat::SignedInt;
        let values: [i16; 4] = [-72, 0, 878, -32768];
        let mut file = Vec::new();
        for v in values {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, None).unwrap();
        assert_eq!(&*pixels, &[32696u16, 32768, 33646, 0]);
    }

    #[test]
    fn frame_float_minmax_matches_actual_data() {
        let values: [f32; 3] = [-3.0, 0.5, 7.0];
        let mut file = Vec::new();
        for v in values {
            file.extend_from_slice(&v.to_le_bytes());
        }
        let frame = float_frame(3, 1);
        let range = frame_float_minmax(&file, &frame, ByteOrder::Little).unwrap();
        assert_eq!(range, Some((-3.0, 7.0)));
    }

    #[test]
    fn frame_float_minmax_is_none_for_integer_frames() {
        let frame = make_frame(2, 2, 1);
        let file = vec![0u8; 8];
        assert_eq!(frame_float_minmax(&file, &frame, ByteOrder::Little).unwrap(), None);
    }

    /// The actual bug: two strips, each independently LZW-compressed (a
    /// fresh `Encoder` per strip, exactly like a real multi-strip TIFF
    /// writer would produce). Concatenating the *compressed* bytes and
    /// decompressing once — the old behavior — silently stops at roughly
    /// strip 1's data. This proves the per-strip fix actually decodes both.
    #[test]
    fn multi_strip_lzw_decodes_past_the_first_strip() {
        let width = 4usize;
        let height = 4usize; // two 2-row strips
        let rows_per_strip = 2usize;

        let top: Vec<u16> = (0..(width * rows_per_strip) as u16).collect(); // 0..8
        let bottom: Vec<u16> = (100..100 + (width * rows_per_strip) as u16).collect(); // 100..108

        let encode_strip = |pixels: &[u16]| -> Vec<u8> {
            let mut raw_bytes = Vec::new();
            for &v in pixels {
                raw_bytes.extend_from_slice(&v.to_le_bytes());
            }
            weezl::encode::Encoder::new(weezl::BitOrder::Msb, 8)
                .encode(&raw_bytes)
                .expect("LZW encode failed in test setup")
        };

        let top_compressed = encode_strip(&top);
        let bottom_compressed = encode_strip(&bottom);

        // Lay the two compressed strips out in a fake "file" buffer.
        let mut fake_file = Vec::new();
        fake_file.extend_from_slice(&top_compressed);
        let bottom_offset = fake_file.len() as u64;
        fake_file.extend_from_slice(&bottom_compressed);

        let frame = FrameInfo {
            width: width as u32,
            height: height as u32,
            bits_per_sample: 16,
            samples_per_pixel: 1,
            sample_format: SampleFormat::UnsignedInt,
            compression: Compression::Lzw,
            predictor: 1,
            strip_offsets: vec![0, bottom_offset],
            strip_byte_counts: vec![top_compressed.len() as u64, bottom_compressed.len() as u64],
            rows_per_strip: rows_per_strip as u32,
        };

        let pixels = read_frame_u16(&fake_file, &frame, ByteOrder::Little, None).expect("decode failed");
        let mut expected = top.clone();
        expected.extend_from_slice(&bottom);
        assert_eq!(&*pixels, expected.as_slice(), "bottom strip's pixels are missing or wrong");
    }
}
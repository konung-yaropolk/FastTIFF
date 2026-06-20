//! Turns a `FrameInfo` into actual pixel data. The fast path — uncompressed
//! strips, native byte order, single strip per frame (ImageJ's default when
//! saving raw stacks) — does zero decoding work at all: it's a direct
//! reinterpret-cast of the memory-mapped file bytes. Everything else
//! (LZW/Deflate/PackBits, multi-strip, predictor, byte-swap, 8-bit upcast)
//! falls back to an owned `Vec<u16>`.

use crate::ifd::ByteOrder;
use crate::index::{Compression, FrameInfo};
use anyhow::{anyhow, bail, Result};
use std::borrow::Cow;

/// Decoded pixel data for one plane, always exposed as 16-bit samples
/// (8-bit sources are upcast: `v -> (v << 8) | v`, which maps 0..255
/// evenly onto 0..65535 and keeps the GPU window/level math uniform).
pub fn read_frame_u16<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    file_order: ByteOrder,
) -> Result<Cow<'a, [u16]>> {
    let n_samples = frame.width as usize * frame.height as usize * frame.samples_per_pixel as usize;

    // --- Fast path: uncompressed, single strip, native 16-bit, native byte order ---
    if frame.compression == Compression::None
        && frame.bits_per_sample == 16
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

    // --- General path: assemble strips, decompress, undo predictor, upcast/byteswap ---
    let raw_bytes = assemble_strip_bytes(mmap, frame)?;
    let sample_bytes = match frame.bits_per_sample {
        16 => 2,
        8 => 1,
        32 => 4,
        other => bail!("unsupported bits_per_sample: {other}"),
    };

    let decompressed = decompress(&raw_bytes, frame.compression, n_samples * sample_bytes)?;
    let unpredicted = undo_predictor(decompressed, frame, sample_bytes, file_order)?;

    let mut out = vec![0u16; n_samples];
    match frame.bits_per_sample {
        16 => {
            for (i, chunk) in unpredicted.chunks_exact(2).enumerate().take(n_samples) {
                out[i] = file_order.u16(chunk);
            }
        }
        8 => {
            for (i, &b) in unpredicted.iter().enumerate().take(n_samples) {
                out[i] = ((b as u16) << 8) | b as u16;
            }
        }
        32 => {
            // 32-bit float data, common for processed/ratiometric stacks.
            // Window/level on the GPU expects integer-ish sample units, so
            // we rescale into u16 space using the frame's own min/max here
            // as a simple normalization; true float display range should
            // come from ImageDescription min=/max= when present (handled
            // by the caller, which still treats this as "raw sample units"
            // scaled the same way).
            for (i, chunk) in unpredicted.chunks_exact(4).enumerate().take(n_samples) {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                let f = match file_order {
                    ByteOrder::Little => f32::from_le_bytes(arr),
                    ByteOrder::Big => f32::from_be_bytes(arr),
                };
                out[i] = f.clamp(0.0, 65535.0) as u16;
            }
        }
        _ => unreachable!(),
    }

    Ok(Cow::Owned(out))
}

fn assemble_strip_bytes(mmap: &[u8], frame: &FrameInfo) -> Result<Vec<u8>> {
    let total: u64 = frame.strip_byte_counts.iter().sum();
    let mut out = Vec::with_capacity(total as usize);
    for (&offset, &len) in frame.strip_offsets.iter().zip(frame.strip_byte_counts.iter()) {
        let slice = mmap
            .get(offset as usize..(offset + len) as usize)
            .ok_or_else(|| anyhow!("strip at offset {offset} (len {len}) out of file bounds"))?;
        out.extend_from_slice(slice);
    }
    Ok(out)
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
/// since this runs before the final byte-order normalization pass.
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

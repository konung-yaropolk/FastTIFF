//! DICOM's RLE, which is PackBits applied to byte planes.
//!
//! PS3.5 Annex G. Two ideas, and the second is the one that surprises people:
//!
//! 1. Each fragment begins with a 64-byte header of sixteen little-endian
//!    `u32`s — a segment count and up to fifteen offsets.
//! 2. A segment is not a run of *pixels*. It is a run of **one byte position of
//!    one sample**: a 16-bit greyscale frame is two segments, the first holding
//!    the high byte of every pixel and the second the low byte of every pixel.
//!    So the bytes of a single pixel are nowhere near each other, and a decoder
//!    that concatenated the segments would produce an image of plausible noise.
//!
//! The segments themselves are PackBits, byte for byte the same scheme TIFF
//! uses (and which this project's TIFF reader already has its own copy of — not
//! shared, because a plugin may only use `fasttiff_plugin_api`, and that is a
//! boundary worth more than forty lines of duplication).
//!
//! # What comes out
//!
//! The native interleaved little-endian layout, which is to say exactly what an
//! uncompressed file of the same image would have held. That is a deliberate
//! choice over handing back the planes as found: it means the rest of the
//! reader needs no notion of "compressed", and `PlanarConfiguration` — which
//! for RLE describes the *compressed* arrangement — stops being a question.

use super::pixels::Format;
use fasttiff_plugin_api::PluginError;

/// The segment table at the head of every fragment.
const HEADER_BYTES: usize = 64;

/// The most segments the header can name.
const MAX_SEGMENTS: usize = 15;

/// Decode one frame's fragment into native interleaved samples.
pub(super) fn decode(fragment: &[u8], fmt: Format) -> Result<Vec<u8>, PluginError> {
    let bytes = fmt.sample_bytes();
    let n_px = (fmt.rows as usize) * (fmt.columns as usize);
    let want = fmt.samples * bytes;

    if fragment.len() < HEADER_BYTES {
        return Err(PluginError::failed(format!(
            "an RLE fragment is {} bytes, shorter than its own 64-byte segment \
             table",
            fragment.len()
        )));
    }
    let word = |i: usize| -> u32 {
        let at = i * 4;
        u32::from_le_bytes([
            fragment[at],
            fragment[at + 1],
            fragment[at + 2],
            fragment[at + 3],
        ])
    };
    let count = word(0) as usize;
    if count != want {
        // Not a tolerable disagreement. The segment count *is* the layout: a
        // frame claiming two segments for a 16-bit image has one byte position
        // missing, and there is no way to guess which.
        return Err(PluginError::failed(format!(
            "this RLE frame declares {count} segment(s) where its header says \
             {} sample(s) of {bytes} byte(s) each, which needs {want}",
            fmt.samples
        )));
    }
    if count > MAX_SEGMENTS {
        return Err(PluginError::failed(format!(
            "this RLE frame declares {count} segments and the format allows \
             {MAX_SEGMENTS}"
        )));
    }

    let mut out = vec![0u8; n_px * want];
    for seg in 0..count {
        let from = word(seg + 1) as usize;
        // The next segment's offset bounds this one; the last runs to the end.
        // Offsets are from the start of the fragment, so the first is 64.
        let to = if seg + 1 < count {
            word(seg + 2) as usize
        } else {
            fragment.len()
        };
        let slice = fragment
            .get(from..to.min(fragment.len()))
            .filter(|_| from >= HEADER_BYTES && from <= to)
            .ok_or_else(|| {
                PluginError::failed(format!(
                    "this RLE frame's segment {seg} is declared at {from}..{to}, \
                     which is not inside the {}-byte fragment",
                    fragment.len()
                ))
            })?;
        let plane = unpack(slice, n_px, seg)?;

        // Where this segment's bytes belong. Segments run sample by sample, and
        // within a sample from the **most** significant byte down — so segment
        // `seg` is sample `seg / bytes`, and its byte sits `bytes - 1 - (seg %
        // bytes)` from the bottom of the sample once written little-endian.
        //
        // Getting that inversion wrong is the classic RLE bug: the image comes
        // out as recognisable anatomy with the contrast of noise, because every
        // sample's halves are swapped.
        let sample = seg / bytes;
        let within = bytes - 1 - (seg % bytes);
        for (p, &b) in plane.iter().enumerate() {
            out[(p * fmt.samples + sample) * bytes + within] = b;
        }
    }
    Ok(out)
}

/// PackBits one segment into exactly `n` bytes.
///
/// `which` is the segment's index, for the error message only — a truncated
/// segment is worth naming, because it says whether the fragment was cut short
/// or the geometry is wrong.
fn unpack(src: &[u8], n: usize, which: usize) -> Result<Vec<u8>, PluginError> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < src.len() && out.len() < n {
        let control = src[i] as i8;
        i += 1;
        match control {
            // 0..=127: the next `control + 1` bytes are literal.
            0..=127 => {
                let run = control as usize + 1;
                let end = (i + run).min(src.len());
                out.extend_from_slice(&src[i..end]);
                i = end;
            }
            // -128 is a no-op, and is in the scheme only because the byte
            // exists. Skipping it rather than treating it as a run of 129 is
            // the difference between a correct decode and a shifted one.
            -128 => {}
            // -1..=-127: the next byte repeats `1 - control` times.
            _ => {
                let run = 1 - control as isize;
                let Some(&b) = src.get(i) else { break };
                i += 1;
                out.extend(std::iter::repeat_n(b, run as usize));
            }
        }
    }
    // Rows are padded to an even length per segment, so a decode may overrun by
    // a byte; that is the format and not an error.
    out.truncate(n);
    if out.len() < n {
        return Err(PluginError::failed(format!(
            "this RLE frame's segment {which} unpacked to {} bytes where the \
             image needs {n}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "rle_tests.rs"]
mod tests;

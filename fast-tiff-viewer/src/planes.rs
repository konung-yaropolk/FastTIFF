//! Decoding one plane into `f32`, in the file's own units.
//!
//! Split out of [`crate::plugins::StackHost`] so there is one definition of
//! "this plane, as `f32`, in the file's units" rather than one per caller.
//! Getting it subtly differently in two places would be a class of bug nobody
//! would find: two readings of the same pixels disagreeing by a factor of 257
//! on 8-bit data, both looking perfectly plausible.
//!
//! "The file's own units" is the contract, and it is not what the *display*
//! path does. An 8-bit sample arrives here as `0..255`, not widened to
//! `0..65535`; a signed 16-bit sample arrives as the number the file states,
//! not offset into unsigned. Anything measuring pixels has to get the value
//! that is in the file, or the number it reports is about the viewer rather
//! than about the specimen.

use anyhow::{Context, Result};
use fast_tiff_lib::{
    read_frame_u16, read_plane_f32_into, read_plane_u16_into, read_plane_u8_into, FrameInfo,
    SampleFormat, TiffStack,
};

/// Scratch buffers for the paths that cannot decode straight into the output.
///
/// Held by the caller and reused across planes: a run that reads a thousand
/// planes should not allocate a thousand buffers. The common case never touches
/// either of these — see [`plane_f32_into`].
#[derive(Default)]
pub struct Scratch {
    u16s: Vec<u16>,
    u8s: Vec<u8>,
}

/// Decode `(ifd, sample)` of `tiff` into `out` as `f32`, in the file's own
/// units.
///
/// `out` is resized to the plane's pixel count and its allocation reused.
pub fn plane_f32_into(
    tiff: &TiffStack,
    ifd: usize,
    sample: usize,
    scratch: &mut Scratch,
    out: &mut Vec<f32>,
) -> Result<()> {
    let frame: &FrameInfo = tiff
        .frames
        .get(ifd)
        .with_context(|| format!("IFD {ifd} is past the {} this file has", tiff.frames.len()))?;
    let data = &tiff.data;
    let order = tiff.byte_order;

    // `read_plane_f32_into` handles 32- and 64-bit samples only, so the
    // narrower depths are converted here.
    match frame.bits_per_sample {
        32 | 64 => read_plane_f32_into(data, frame, order, sample, out),
        8 => {
            let bytes = &mut scratch.u8s;
            read_plane_u8_into(data, frame, order, sample, bytes)?;
            out.clear();
            out.extend(bytes.iter().map(|&v| v as f32));
            Ok(())
        }
        _ => {
            // 16-bit, and any other narrow depth the u16 reader accepts.
            //
            // The fast path first: for the shape this library's own writer and
            // ImageJ both produce — uncompressed, one strip, native byte order,
            // unsigned — `read_frame_u16` hands back a borrow straight over the
            // memory map, so the samples convert into `out` in one pass with no
            // intermediate buffer at all. That shape is the common case rather
            // than a lucky one; `fast_tiff_lib`'s encoder targets it
            // deliberately.
            //
            // `read_frame_u16` reads plane 0, so it only applies where plane 0
            // is the whole frame: `sample == 0` *and* one sample per pixel.
            // Taking it for any other plane would silently return the wrong
            // channel. Signed data is excluded because the borrow would skip
            // the offset undo below.
            if sample == 0
                && frame.samples_per_pixel <= 1
                && frame.sample_format != SampleFormat::SignedInt
            {
                let borrowed = read_frame_u16(data, frame, order, None)?;
                out.clear();
                out.extend(borrowed.iter().map(|&v| v as f32));
                return Ok(());
            }

            // Otherwise an intermediate is unavoidable — but it is reused
            // between calls rather than allocated per plane.
            //
            // `read_plane_u16_into` offsets a signed sample into unsigned by
            // flipping the sign bit; undo that so the value is the one the file
            // states.
            let signed = frame.sample_format == SampleFormat::SignedInt;
            let raw = &mut scratch.u16s;
            read_plane_u16_into(data, frame, order, None, sample, raw)?;
            out.clear();
            if signed {
                out.extend(raw.iter().map(|&v| v as f32 - 32768.0));
            } else {
                out.extend(raw.iter().map(|&v| v as f32));
            }
            Ok(())
        }
    }
}

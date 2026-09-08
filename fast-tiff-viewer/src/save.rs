//! Writing the open stack back out as a TIFF.
//!
//! The promise is "what is open, in a file" — every plane, and the metadata
//! needed to open it again the same way: the c/z/t interpretation, the display
//! mode and per-channel LUTs, the calibration, and whatever the source said
//! about itself in tag 270.
//!
//! # It is a re-encode, not a copy
//!
//! The planes are decoded and written again rather than the original bytes
//! being copied through. That is the point rather than a cost: what is open may
//! never have been a file — a plugin's result, or an OIR the importer built in
//! memory — and even when it was, the axes may have been reinterpreted since.
//! Writing what the viewer holds is what makes the saved file agree with the
//! window it came from.
//!
//! What it does *not* do is re-interpret the samples. A frame goes out in the
//! type it came in as, which is why the signed case below undoes the decoder's
//! offset instead of writing the offset values.

use crate::stack::Stack;
use anyhow::{bail, Context, Result};
use fast_tiff_lib::metadata::{self, MetadataFormat};
use fast_tiff_lib::{
    read_frame_f32_into, read_frame_u16_into, read_frame_u8_into, FrameInfo, SampleFormat,
    SampleType, StackMetaWrite, TiffWriter, WriterOptions,
};
use std::path::Path;

/// How a frame's samples travel from the file to the file.
///
/// The decoders normalise on the way in — 8-bit is widened, floats are rescaled
/// into the display window, signed integers are offset — so a save has to pick
/// the reader whose normalisation it can undo, rather than the most convenient
/// one. Reading an 8-bit frame as `u16` and writing it as `u16` would double
/// the file and change every value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Samples {
    U8,
    U16,
    /// 16-bit signed. The decoder hands these back offset by +32768 so they sort
    /// as unsigned; writing them without undoing that would shift every pixel by
    /// half the range.
    I16,
    F32,
}

impl Samples {
    fn of(frame: &FrameInfo) -> Result<Samples> {
        Ok(match (frame.bits_per_sample, frame.sample_format) {
            (8, SampleFormat::UnsignedInt) | (8, SampleFormat::SignedInt) => Samples::U8,
            (16, SampleFormat::SignedInt) => Samples::I16,
            // Sub-byte and other narrow depths come out of the decoder as
            // `u16` values, so that is what they are written as: the values
            // survive, the original bit depth does not.
            (b, SampleFormat::UnsignedInt) if b <= 16 => Samples::U16,
            (32, SampleFormat::Float) => Samples::F32,
            (bits, format) => bail!(
                "this stack is {bits}-bit {format:?}, which this writer cannot reproduce \
                 without changing the samples"
            ),
        })
    }

    fn sample_type(self) -> SampleType {
        match self {
            Samples::U8 => SampleType::U8,
            Samples::U16 => SampleType::U16,
            Samples::I16 => SampleType::I16,
            Samples::F32 => SampleType::F32,
        }
    }
}

/// Write `stack` to `path` as a TIFF.
///
/// Streams: one frame is decoded and written at a time, so saving a stack
/// costs one frame of memory rather than a second copy of the whole thing.
pub fn save_stack(stack: &Stack, path: &Path) -> Result<()> {
    let first = stack
        .tiff
        .frames
        .first()
        .context("this stack has no frames to save")?;
    let samples = Samples::of(first)?;
    let spp = first.samples_per_pixel.max(1);

    // One TIFF, one shape. The reader will happily index a file whose frames
    // differ — some scanners write a thumbnail as the first IFD — but a stack
    // of mixed shapes is not something this can write back out, and saying so
    // beats writing a file whose later frames are the wrong size.
    for (i, f) in stack.tiff.frames.iter().enumerate() {
        if (f.width, f.height, f.samples_per_pixel.max(1)) != (first.width, first.height, spp) {
            bail!(
                "frame {i} is {}x{}x{} where the first is {}x{}x{spp}; a stack of mixed \
                 frame shapes cannot be saved as one TIFF",
                f.width,
                f.height,
                f.samples_per_pixel.max(1),
                first.width,
                first.height
            );
        }
        if Samples::of(f)? != samples {
            bail!("frame {i} does not have the same sample type as the first");
        }
    }

    let options = WriterOptions::new(first.width, first.height, samples.sample_type())
        .samples_per_pixel(spp)
        .metadata(metadata_of(stack));
    let mut writer = TiffWriter::create(path, options)
        .with_context(|| format!("creating {}", path.display()))?;

    let data = &stack.tiff.data;
    let order = stack.tiff.byte_order;
    let mut u8s: Vec<u8> = Vec::new();
    let mut u16s: Vec<u16> = Vec::new();
    let mut f32s: Vec<f32> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    for (i, frame) in stack.tiff.frames.iter().enumerate() {
        let wrote = || format!("writing frame {i}");
        match samples {
            Samples::U8 => {
                read_frame_u8_into(data, frame, order, &mut u8s).with_context(wrote)?;
                writer.write_frame_u8(&u8s).with_context(wrote)?;
            }
            Samples::U16 => {
                read_frame_u16_into(data, frame, order, None, &mut u16s).with_context(wrote)?;
                writer.write_frame_u16(&u16s).with_context(wrote)?;
            }
            Samples::I16 => {
                read_frame_u16_into(data, frame, order, None, &mut u16s).with_context(wrote)?;
                // Back to the bit pattern the file had: the decoder XORs the
                // sign bit so signed data sorts as unsigned, and this is that
                // operation run the other way.
                //
                // Written as bytes because the typed calls are keyed to the
                // writer's own sample type, and there is no `write_frame_i16`
                // to hand these to — the values are `i16` in a `u16`'s clothing
                // and only the caller knows it.
                for v in &mut u16s {
                    *v ^= 0x8000;
                }
                bytes.clear();
                bytes.extend(u16s.iter().flat_map(|v| v.to_le_bytes()));
                writer.write_frame_bytes(&bytes).with_context(wrote)?;
            }
            Samples::F32 => {
                read_frame_f32_into(data, frame, order, &mut f32s).with_context(wrote)?;
                writer.write_frame_f32(&f32s).with_context(wrote)?;
            }
        }
    }
    writer
        .finish()
        .with_context(|| format!("finishing {}", path.display()))?;
    Ok(())
}

/// The metadata to write beside the pixels.
///
/// Taken from the *display* rather than from the file's own header wherever the
/// two can differ. A stack whose axes were reinterpreted, or whose channels were
/// recoloured, is saved as it is being looked at — which is the only reading of
/// "save" that does not surprise someone who just changed something.
fn metadata_of(stack: &Stack) -> StackMetaWrite {
    let dims = stack.display.dims;
    let meta = &stack.tiff.meta;
    let mut out = StackMetaWrite::new(dims.channels, dims.slices).mode(stack.display.mode);

    if let Some(unit) = &meta.unit {
        out = out.unit(unit.clone());
    }
    if let Some(fps) = meta.fps {
        out = out.fps(fps);
    }
    if let Some(interval) = meta.frame_interval_s {
        out = out.frame_interval_s(interval);
    }
    if let Some(spacing) = meta.spacing {
        out = out.spacing(spacing);
    }
    if let Some(looped) = meta.loop_playback {
        out = out.loop_playback(looped);
    }
    if let Some((c0, c1)) = meta.calibration {
        out = out.calibration(c0, c1);
    }
    if let (Some(w), Some(h)) = (meta.pixel_width, meta.pixel_height) {
        out = out.pixel_size(w, h);
    }
    // The window on screen. The format carries one for the stack, so the first
    // channel's speaks for it — the alternative is to write none and have the
    // file open on auto-contrast, which is a visible change to something the
    // user did not touch.
    if let Some(first) = stack.display.settings.first() {
        out = out.range(first.min as f64, first.max as f64);
    }
    // Every channel's LUT, always — including plain grayscale ones. They are
    // 768 bytes each, and writing them is what makes a saved file open in the
    // colours it was saved in rather than in whichever ones a reader defaults
    // to for that channel count.
    for lut in &stack.display.luts {
        out = out.channel_lut(*lut);
    }
    if let Some(text) = carried_description(stack) {
        out = out.trailing(text);
    }
    out
}

/// Whatever the source's `ImageDescription` said that this writer is not about
/// to say again.
///
/// A vendor record — the FluoView export an OIR import leaves there, an OME
/// document, free text from another tool — is the part of a file that cannot be
/// reconstructed, so it travels. The ImageJ `key=value` block in front of it
/// does not: every key in it is being regenerated from the values above, and
/// carrying the old copy through would leave two of each in the file.
fn carried_description(stack: &Stack) -> Option<String> {
    let text = stack.tiff.description.as_deref()?;
    let kept = match metadata::detect(Some(text)) {
        MetadataFormat::ImageJ => text
            .lines()
            .skip_while(|l| is_key_value(l))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => text.to_string(),
    };
    (!kept.trim().is_empty()).then_some(kept)
}

/// Whether a line is one of ImageJ's own `key=value` entries.
///
/// Deliberately narrow: a key is a bare identifier. The `"key"\t"value"` lines
/// of an instrument's export are not one, and neither is anything with a space
/// or a quote in it, which is what keeps this from eating the record it is
/// meant to preserve.
fn is_key_value(line: &str) -> bool {
    match line.split_once('=') {
        Some((key, _)) => {
            !key.is_empty()
                && key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        }
        None => false,
    }
}

#[cfg(test)]
#[path = "save_tests.rs"]
mod tests;

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

/// Everything a save reads, owned.
///
/// A save is a decode-and-encode of every frame, which on a big stack is
/// seconds — long enough that it belongs on a worker thread. This is what goes
/// there. It is not a copy of the image: `tiff` is a second handle on the same
/// indexed file, and `display` is the axis/contrast model, which is a few
/// vectors of settings. Taking it costs a refcount bump and one small clone.
///
/// Owned rather than borrowed because the window carries on being used while
/// the write runs, and a `&Stack` held across that would stop it.
pub struct SaveSource {
    tiff: std::sync::Arc<fast_tiff_lib::TiffStack>,
    display: crate::display::Display,
}

impl SaveSource {
    /// Snapshot the stack as it is being looked at *now*.
    ///
    /// The metadata written comes from the display model, so changing the
    /// contrast or the axes after pressing Save must not change the file that
    /// is already being written — hence a snapshot rather than a live borrow.
    pub fn of(stack: &Stack) -> Self {
        SaveSource {
            tiff: std::sync::Arc::clone(&stack.tiff),
            display: stack.display.clone(),
        }
    }

    /// How many frames the write will step through, for a progress readout.
    pub fn frames(&self) -> usize {
        self.tiff.frames.len()
    }
}

/// Write `stack` to `path` as a TIFF.
///
/// The blocking form, for callers with nothing to report progress to — tests,
/// and any future headless use. [`save_source`] is what the app calls.
pub fn save_stack(stack: &Stack, path: &Path) -> Result<()> {
    save_source(&SaveSource::of(stack), path, &mut |_| true)
}

/// Write a snapshot to `path`, reporting progress and stopping when asked.
///
/// Streams: one frame is decoded and written at a time, so saving a stack
/// costs one frame of memory rather than a second copy of the whole thing.
///
/// `on_progress` is called once per frame with the fraction completed and
/// returns `false` to cancel.
///
/// # Nothing is destroyed until the write has succeeded
///
/// The pixels go to a temporary file beside the target and are renamed over it
/// at the end. Two things make that worth the extra step rather than writing
/// straight to `path`:
///
/// * A half-written TIFF has a valid header and a short IFD chain, so it
///   *opens* — as a file with fewer frames than it should have. Nothing
///   downstream can tell it is incomplete. Cancelling a save, or having one
///   fail, must not produce one.
/// * Saving over an existing file truncates it the moment the writer is
///   created. A save that then fails half way would have destroyed the file it
///   was supposed to replace — and "save over the file I am looking at" is a
///   thing people do. Writing beside it means the original survives every
///   failure, including the one where the rename itself is refused because
///   another window has that file memory-mapped (Windows os error 1224).
pub fn save_source(
    source: &SaveSource,
    path: &Path,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<()> {
    let temp = partial_path(path);
    match write_all(source, &temp, on_progress) {
        Ok(()) => std::fs::rename(&temp, path).with_context(|| {
            // Best effort: leaving the part file behind after a failed rename
            // would be a mystery file next to the one the user asked for.
            let _ = std::fs::remove_file(&temp);
            format!("could not put the finished file at {}", path.display())
        }),
        Err(e) => {
            // Best effort: if the part file cannot be removed the error being
            // reported is still the more useful one.
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
}

/// Where the pixels go until the write has succeeded.
///
/// In the same directory as the target, because a rename across filesystems is
/// a copy — and the temp directory is routinely on a different volume from the
/// data drive a microscopy stack is being saved to.
fn partial_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".fasttiff-part");
    path.with_file_name(name)
}

/// The write itself. Split out so [`save_source`] can clean up after it without
/// an early `return` skipping the cleanup.
fn write_all(
    source: &SaveSource,
    path: &Path,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> Result<()> {
    let stack = source;
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
    let total = stack.tiff.frames.len().max(1);
    for (i, frame) in stack.tiff.frames.iter().enumerate() {
        // Before the frame, not after: a stack of one frame should still show
        // that the write started, and the `finish` below is the tail this can
        // never account for.
        if !on_progress(i as f32 / total as f32) {
            bail!("cancelled");
        }
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
fn metadata_of(stack: &SaveSource) -> StackMetaWrite {
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
fn carried_description(stack: &SaveSource) -> Option<String> {
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

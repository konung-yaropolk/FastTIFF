//! Turning what a plugin produced into a document the app can show.
//!
//! A plugin hands back planes in memory; the viewer needs a [`Stack`]. Rather
//! than build one field by field, the result is encoded to a TIFF in memory and
//! opened through the ordinary reader.
//!
//! That round-trip is not free — it encodes and re-decodes every pixel — and it
//! is still the right answer for this first version, because it means a plugin
//! result is a *real* document from the moment it exists. Channel setup,
//! auto-contrast, LUTs, the c/z/t interpretation, the histogram, 3D, "Save
//! as…", opening it in a second window: all of it works with no new code, and
//! none of it can drift from what an opened file does, because it is the same
//! path. Building a `Stack` directly would duplicate that pipeline and then
//! have to be kept in step with it.
//!
//! The cost is one encode plus one decode of the result — which is bounded by
//! the result's size, not the source stack's. If a plugin ever returns
//! something large enough for that to hurt, the fix is a direct constructor for
//! that case, not the loss of the shared path for every other.

use fast_tiff_lib::{
    DisplayMode as LibDisplayMode, SampleType, StackMetaWrite, TiffWriter, WriterOptions,
};
use fasttiff_plugin_api::{DisplayMode, ImageResult, PixelType, PlaneData, StackInfo};
use std::io::Cursor;

/// Encode a plugin's result as a TIFF in memory.
///
/// The planes are written in the order the contract states — channel fastest,
/// then Z, then time — which is the order the reader expects, so the result
/// re-opens with the axes it declared.
pub fn to_tiff_bytes(image: &ImageResult, info: Option<&StackInfo>) -> anyhow::Result<Vec<u8>> {
    image
        .validate()
        .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;

    let sample = match image.pixel_type {
        PixelType::U8 => SampleType::U8,
        PixelType::U16 => SampleType::U16,
        PixelType::I16 => SampleType::I16,
        PixelType::F32 => SampleType::F32,
    };

    let mut meta = StackMetaWrite::new(image.channels.max(1), image.slices.max(1));
    if let Some(info) = info {
        meta = meta.mode(match info.mode {
            DisplayMode::Composite => LibDisplayMode::Composite,
            DisplayMode::Color => LibDisplayMode::Color,
            DisplayMode::Grayscale => LibDisplayMode::Grayscale,
        });
        if let Some(u) = &info.unit {
            meta = meta.unit(u.clone());
        }
        if let Some(s) = info.frame_interval_s {
            meta = meta.frame_interval_s(s);
        }
    } else if image.channels > 1 {
        // A multi-channel result with nothing said about it is far more useful
        // composited than shown one channel at a time.
        meta = meta.mode(LibDisplayMode::Composite);
    }

    let opts = WriterOptions::new(image.width, image.height, sample).metadata(meta);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts)?;
    for plane in &image.planes {
        match plane {
            PlaneData::U8(v) => w.write_frame_bytes(v)?,
            PlaneData::U16(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|s| s.to_le_bytes()).collect();
                w.write_frame_bytes(&bytes)?
            }
            PlaneData::F32(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|s| s.to_le_bytes()).collect();
                w.write_frame_bytes(&bytes)?
            }
        }
    }
    Ok(w.finish()?.into_inner())
}

/// Encode a result and open it as a stack, exactly as a file would be opened.
pub fn to_stack(
    image: &ImageResult,
    info: Option<&StackInfo>,
    apply_pseudocolor: bool,
) -> anyhow::Result<crate::stack::Stack> {
    let bytes = to_tiff_bytes(image, info)?;
    let name = info
        .map(|i| i.name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| image.name.clone());
    crate::stack::Stack::from_bytes(bytes, name.into(), apply_pseudocolor)
}

#[cfg(test)]
#[path = "result_tests.rs"]
mod tests;

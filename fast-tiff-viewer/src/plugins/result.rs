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
    to_tiff_bytes_reporting(image, info, &mut |_| true)
        .transpose()
        .unwrap_or_else(|| Err(anyhow::anyhow!("cancelled")))
}

/// [`to_tiff_bytes`], reporting how far through the planes it is.
///
/// Encoding a plugin's result is not a rounding error on the run that produced
/// it: a stabilised timelapse is every plane rewritten, which on a long
/// recording is gigabytes and takes longer than the registration did. Doing it
/// without saying so is what makes a progress bar reach 100% and then sit
/// there.
///
/// `on_progress` returns `false` to stop, which returns `Ok(None)`.
pub fn to_tiff_bytes_reporting(
    image: &ImageResult,
    info: Option<&StackInfo>,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<Option<Vec<u8>>> {
    image
        .validate()
        .map_err(|e| anyhow::anyhow!("the plugin returned an unusable image: {e}"))?;

    let sample = match image.pixel_type {
        PixelType::U8 => SampleType::U8,
        PixelType::U16 => SampleType::U16,
        PixelType::I16 => SampleType::I16,
        PixelType::F32 => SampleType::F32,
    };

    // An explicit caller override wins; otherwise a result may carry the
    // source metadata its plugin deliberately preserved.
    let info = info.or(image.metadata.as_ref());
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
        // Spacing and calibration are the whole reason an importer bothers to
        // report metadata: without them a measurement made on the result is in
        // pixels and frames rather than microns and seconds.
        if let (Some(w), Some(h)) = (info.spacing.x, info.spacing.y) {
            meta = meta.pixel_size(w, h);
        }
        if let Some(z) = info.spacing.z {
            meta = meta.spacing(z);
        }
        if let Some((c0, c1)) = info.calibration {
            meta = meta.calibration(c0, c1);
        }
        for (i, name) in info.channel_names.iter().enumerate() {
            meta = meta.channel(name.clone(), fast_tiff_lib::metadata::composite_color(i));
        }
        // The source file's own `ImageDescription`, carried into tag 270 of the
        // written one. An importer that read a vendor format puts the vendor's
        // metadata here, and this is the only thing standing between that and
        // it being lost on conversion.
        if let Some(d) = &info.description {
            if !d.trim().is_empty() {
                meta = meta.trailing(d.clone());
            }
        }
    } else if image.channels > 1 {
        // A multi-channel result with nothing said about it is far more useful
        // composited than shown one channel at a time.
        meta = meta.mode(LibDisplayMode::Composite);
    }

    // Per-channel colour, when the plugin asked for one. It has to go in as a
    // full LUT rather than as `channel(name, color)`: the latter feeds the OME
    // dialect, and what an ImageJ-format file (and this viewer) reads is the
    // binary LUT block.
    if !image.channel_colors.is_empty() {
        meta = meta.mode(LibDisplayMode::Composite);
        for c in 0..image.channels.max(1) {
            let color = image
                .channel_colors
                .get(c)
                .copied()
                .unwrap_or_else(|| fast_tiff_lib::metadata::composite_color(c));
            meta = meta.channel_lut(fast_tiff_lib::color_ramp_lut(color));
        }
    }

    let opts = WriterOptions::new(image.width, image.height, sample).metadata(meta);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts)?;
    let total = image.planes.len().max(1);
    for (i, plane) in image.planes.iter().enumerate() {
        if !on_progress(i as f32 / total as f32) {
            return Ok(None);
        }
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
    Ok(Some(w.finish()?.into_inner()))
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

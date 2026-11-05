//! PNG: the frame on screen, as everything else understands it.
//!
//! The first exporter, and the one that shows what an exporter is for. Saving
//! *the stack* is the host's own job and it does that in TIFF, where the axes,
//! the bit depth and the calibration all survive. PNG can hold none of that. So
//! this does the other thing — it writes **what the window is showing**: one
//! frame, contrast applied, channels composited through their LUTs, 8-bit RGB.
//! A figure, not a copy.
//!
//! # Reproducing the display
//!
//! The contrast window the view describes is in the channel's own sample units
//! — the file's values, what `read_plane_f32` returns — so applying it is a
//! matter of reading the plane in those units too. That is what this does for
//! integer data: `read_plane_f32`, and the window over that.
//!
//! A 32- or 64-bit plane ([`PixelType::F32`], which the host reports for wide
//! integers as well as floats) is read through `read_plane_u16` instead, which
//! rescales it through the channel's window onto the 16-bit scale: the window
//! is then already applied and the value is the display value, exactly as the
//! renderer's own float path quantizes it. Which case applies is read from
//! [`ImageInfo::pixel_type`].

/// Reading one in, which is the other half of the format.
mod import;
pub use import::PngImport;

use fasttiff_plugin_api::{
    ExportRequest, Exporter, FileType, HostContext, PixelType, Plane, PluginError, PluginInfo,
};
use std::io::BufWriter;

/// Write the displayed frame as an 8-bit RGB PNG.
#[derive(Default)]
pub struct Png;

impl Exporter for Png {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.png", "PNG")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Write the frame on screen as an 8-bit RGB PNG, as it is displayed.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("PNG image", &["png"])]
    }

    fn export(
        &mut self,
        request: &ExportRequest,
        host: &mut dyn HostContext,
    ) -> Result<(), PluginError> {
        let info = host.image();
        let (w, h) = (info.width as usize, info.height as usize);
        if w == 0 || h == 0 {
            return Err(PluginError::unsupported("this stack has no pixels"));
        }
        let view = host.view().clone();
        let t = view.frame_index.min(info.frames.saturating_sub(1));
        // 2D shows the first slice of a stack that has both axes — the frame
        // slider walks time — so that is the slice this writes. Saying which
        // beats guessing at a "current" slice the view does not have.
        let z = 0;
        // Already applied by the decoder for float samples; still to be applied
        // here for integer ones. See the module docs.
        let windowed_on_read = info.pixel_type == PixelType::F32;

        if info.frames > 1 || info.slices > 1 {
            host.log(&format!(
                "PNG holds one image: writing c*, z{z}, t{t} of a {}x{} stack",
                info.slices, info.frames
            ));
        }

        let mut rgb = vec![0u8; w * h * 3];
        let mut windowed: Vec<u16> = Vec::new();
        let mut plane: Vec<f32> = Vec::new();
        // The displayed channels, which is what `view` describes — the file may
        // have more than the renderer can show, and an export of the picture
        // should contain what the picture contains.
        let shown = view.channels.len().min(info.channels);
        for c in 0..shown {
            let channel = &view.channels[c];
            if !channel.enabled {
                continue;
            }
            if !host.progress(c as f32 / shown.max(1) as f32) {
                return Err(PluginError::unsupported("cancelled"));
            }
            let lut = view.luts.get(c).copied().unwrap_or_else(grayscale_ramp);
            let (lo, hi) = (channel.min, channel.max);
            let span = if (hi - lo).abs() > f32::EPSILON {
                hi - lo
            } else {
                1.0
            };
            // Where each pixel sits in the window, 0..=1.
            if windowed_on_read {
                host.read_plane_u16(Plane::new(c, z, t), &mut windowed)?;
                let scale = f32::from(u16::MAX);
                add_channel(&mut rgb, &lut, windowed.iter().map(|&v| v as f32 / scale));
            } else {
                host.read_plane_f32(Plane::new(c, z, t), &mut plane)?;
                let fractions = plane.iter().map(|&v| ((v - lo) / span).clamp(0.0, 1.0));
                add_channel(&mut rgb, &lut, fractions);
            }
        }

        let file = std::fs::File::create(&request.path).map_err(|e| {
            PluginError::failed(format!("creating {}: {e}", request.path.display()))
        })?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), info.width, info.height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| PluginError::failed(format!("writing the PNG header: {e}")))?;
        writer
            .write_image_data(&rgb)
            .map_err(|e| PluginError::failed(format!("writing the PNG: {e}")))?;
        writer
            .finish()
            .map_err(|e| PluginError::failed(format!("finishing the PNG: {e}")))?;
        Ok(())
    }
}

/// Composite one channel into `rgb`: each pixel's place in the window
/// (`fractions`, `0..=1`) looked up in `lut` and added on.
fn add_channel(rgb: &mut [u8], lut: &[[u8; 3]; 256], fractions: impl Iterator<Item = f32>) {
    for (px, t01) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(fractions) {
        let entry = lut[(t01 * 255.0).round().clamp(0.0, 255.0) as usize];
        // Additive, like the composite on screen: two channels lit at one
        // pixel make the sum, and white where they overlap.
        for (out, add) in px.iter_mut().zip(entry) {
            *out = out.saturating_add(add);
        }
    }
}

/// The LUT for a channel the view did not describe: plain grey, so a missing
/// entry produces the image rather than a black one.
fn grayscale_ramp() -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for (i, e) in lut.iter_mut().enumerate() {
        *e = [i as u8, i as u8, i as u8];
    }
    lut
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

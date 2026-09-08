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
//! The contrast window is in different units depending on the data, and getting
//! that backwards is the whole difficulty here. The host hands a plugin planes
//! through `read_plane_u16`, which:
//!
//! * rescales **float** samples through the channel's window into `0..65535` —
//!   so the window has already been applied and the value is the display value;
//! * widens **integer** samples to `0..65535` untouched — so the window, which
//!   is itself in that same `0..65535` space, still has to be applied here.
//!
//! Treating one as the other produces an image that is uniformly black, or
//! uniformly white, or merely wrong in a way that looks like a contrast
//! setting. Which case applies is read from [`ImageInfo::pixel_type`].
//!
//! One corner is worth naming: the host reports any 32-bit sample as `F32`,
//! including a 32-bit *integer* stack, whose window is nonetheless in the
//! integer space. That inconsistency is in the host's description rather than
//! here, and 32-bit integer microscopy data is rare enough that it has not been
//! worth widening the contract over.

use fasttiff_plugin_api::{
    ExportRequest, Exporter, FileType, HostContext, PixelType, Plane, PluginError, PluginInfo,
};
use std::io::BufWriter;

/// Write the displayed frame as an 8-bit RGB PNG.
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
        let mut plane: Vec<u16> = Vec::new();
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
            host.read_plane_u16(Plane::new(c, z, t), &mut plane)?;
            let lut = view.luts.get(c).copied().unwrap_or_else(grayscale_ramp);
            let (lo, hi) = (channel.min, channel.max);
            let span = if (hi - lo).abs() > f32::EPSILON {
                hi - lo
            } else {
                1.0
            };
            for (px, &v) in rgb.chunks_exact_mut(3).zip(plane.iter()) {
                let t01 = if windowed_on_read {
                    v as f32 / 65535.0
                } else {
                    ((v as f32 - lo) / span).clamp(0.0, 1.0)
                };
                let entry = lut[(t01 * 255.0).round().clamp(0.0, 255.0) as usize];
                // Additive, like the composite on screen: two channels lit at
                // one pixel make the sum, and white where they overlap.
                for (out, add) in px.iter_mut().zip(entry) {
                    *out = out.saturating_add(add);
                }
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

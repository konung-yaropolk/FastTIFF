//! Crop: keep a rectangle of every plane.
//!
//! No dialog. The first run has no selection, and asks the host for one
//! ([`Outcome::SelectRegion`]): the host shows the image in 2D, says in its
//! status line to drag a rectangle and press Enter, and runs this again with
//! the rectangle once Enter is pressed. That second run is the crop.
//!
//! # Only X and Y
//!
//! A crop is a rectangle of the picture, so it changes width and height and
//! nothing else. Every channel, slice and frame is cut the same way and kept:
//! a z-stack or a timelapse comes back as long as it went in, only smaller
//! across. The file's calibration comes too, since a pixel is as wide after a
//! crop as before.
//!
//! # A rectangle, whatever was drawn
//!
//! The host's tool can draw ellipses for a plot, and an image is rectangular,
//! so a crop keeps a selection's bounding box. Only the first selection
//! counts: a crop has one answer.

use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, Params, Plane, PlaneData, Plugin, PluginError,
    PluginInfo, Roi,
};

use crate::plugins::builtin::shared::Store;

/// What the status line says while the rectangle is being drawn.
pub(crate) const PROMPT: &str =
    "Crop: drag a rectangle over the part to keep, then press Enter (Esc cancels)";

/// Keep a rectangle of every plane of whatever is loaded.
#[derive(Default)]
pub struct Crop;

impl Plugin for Crop {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.crop", "Crop")
            .menu_path("Image")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Keep a rectangle of the image: drag it on the picture and press Enter. \
                 Every channel, slice and frame is cropped the same way; only the width \
                 and height change.",
            )
    }

    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        Vec::new()
    }

    fn run(
        &mut self,
        host: &mut dyn HostContext,
        _params: &Params,
    ) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.plane_len() == 0 {
            return Err(PluginError::unsupported("the stack has no pixels"));
        }
        let Some(roi) = host.selection().first().copied() else {
            return Ok(Outcome::SelectRegion {
                prompt: PROMPT.to_string(),
            });
        };
        let (x0, y0, w, h) = bounds(&roi, info.width, info.height).ok_or_else(|| {
            PluginError::unsupported(format!(
                "the rectangle ({} x {} at {}, {}) is not inside the {} x {} image",
                roi.w, roi.h, roi.x, roi.y, info.width, info.height
            ))
        })?;

        let channels = info.channels.max(1);
        let slices = info.slices.max(1);
        let frames = info.frames.max(1);
        let store = Store::of(info.pixel_type);
        let n_planes = channels * slices * frames;
        let width = info.width as usize;

        let mut planes: Vec<PlaneData> = Vec::with_capacity(n_planes);
        let mut buf = Vec::new();
        let mut done = 0usize;
        for t in 0..frames {
            for z in 0..slices {
                for c in 0..channels {
                    if !host.progress(done as f32 / n_planes as f32) {
                        return Ok(Outcome::Cancelled);
                    }
                    host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                    planes.push(store.plane(cut(&buf, width, (x0, y0, w, h))));
                    done += 1;
                }
            }
        }

        host.log(&format!("kept {w} x {h} pixels from ({x0}, {y0})"));
        let image = ImageResult {
            width: w as u32,
            height: h as u32,
            channels,
            slices,
            frames,
            pixel_type: store.pixel_type(),
            planes,
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-crop", host.stack_info().stem()),
        };
        image.validate()?;
        Ok(Outcome::NewDocument(Box::new(image)))
    }
}

/// The rectangle to keep, as `(x, y, width, height)` in pixels: the bounding
/// box of `roi`. `None` when it is empty or does not fit a `width` x `height`
/// image. The host clamps what is drawn to the image, so that is a selection
/// made for a different image.
pub(crate) fn bounds(roi: &Roi, width: u32, height: u32) -> Option<(usize, usize, usize, usize)> {
    let fits = roi.w > 0
        && roi.h > 0
        && roi.x.checked_add(roi.w).is_some_and(|r| r <= width)
        && roi.y.checked_add(roi.h).is_some_and(|b| b <= height);
    fits.then_some((
        roi.x as usize,
        roi.y as usize,
        roi.w as usize,
        roi.h as usize,
    ))
}

/// The rows of `plane` (`width` samples a row) inside the rectangle.
pub(crate) fn cut(
    plane: &[f32],
    width: usize,
    (x0, y0, w, h): (usize, usize, usize, usize),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for row in plane.chunks_exact(width).skip(y0).take(h) {
        out.extend_from_slice(&row[x0..x0 + w]);
    }
    out
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

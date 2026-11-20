//! Split Channels: one channel stack per source channel.
//!
//! A composite image is useful to inspect together but often needs to be
//! processed one label at a time. This makes a separate grayscale stack for
//! every channel, retaining the source's geometry and calibration and writing
//! that channel's actual 256-entry lookup table into its output TIFF.

use crate::shared::Store;
use fasttiff_plugin_api::{
    DisplayMode, DocumentResult, HostContext, ImageResult, Outcome, Params, Plane, Plugin,
    PluginError, PluginInfo,
};

/// Split every source channel into its own document.
#[derive(Default)]
pub struct SplitChannels;

impl Plugin for SplitChannels {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stacktools.split-channels", "Split channels")
            .menu_path("Image")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Open each source channel as a separate stack, keeping its metadata and LUT.",
            )
    }

    // No dialog: there is one unambiguous output per source channel.
    fn run(
        &mut self,
        host: &mut dyn HostContext,
        _params: &Params,
    ) -> Result<Outcome, PluginError> {
        let info = host.image();
        let channels = info.channels.max(1);
        if channels <= 1 {
            return Err(PluginError::unsupported(
                "this image has no channel dimension to split",
            ));
        }

        let store = Store::of(info.pixel_type);
        let slices = info.slices.max(1);
        let frames = info.frames.max(1);
        let total = (channels * slices * frames).max(1);
        let mut done = 0usize;
        let mut documents = Vec::with_capacity(channels);

        for c in 0..channels {
            let mut planes = Vec::with_capacity(slices * frames);
            let mut buf = Vec::new();
            for t in 0..frames {
                for z in 0..slices {
                    if !host.progress(done as f32 / total as f32) {
                        return Ok(Outcome::Cancelled);
                    }
                    host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                    planes.push(store.plane(std::mem::take(&mut buf)));
                    done += 1;
                }
            }

            let mut metadata = host.stack_info().clone();
            // It is now a one-channel image, where the source channel's LUT is
            // its display identity. The rest of its calibration survives.
            metadata.mode = DisplayMode::Color;
            metadata.channel_names = metadata.channel_names.get(c).cloned().into_iter().collect();
            // No table is a legitimate source state. In that case let the
            // output stay neutral rather than inventing a colour convention.
            let lut = host.view().luts.get(c).copied().unwrap_or_else(gray_lut);
            let image = ImageResult {
                width: info.width,
                height: info.height,
                channels: 1,
                slices,
                frames,
                pixel_type: store.pixel_type(),
                planes,
                channel_colors: Vec::new(),
                metadata: Some(metadata),
                name: format!("{}_ch{}", host.stack_info().stem(), c + 1),
            };
            image.validate()?;
            documents.push(DocumentResult {
                image,
                luts: vec![lut],
            });
        }

        host.progress(1.0);
        host.log(&format!(
            "split {channels} channel(s) into separate documents"
        ));
        Ok(Outcome::NewDocuments(documents))
    }
}

fn gray_lut() -> fasttiff_plugin_api::Lut {
    let mut lut = [[0; 3]; 256];
    for (i, rgb) in lut.iter_mut().enumerate() {
        *rgb = [i as u8; 3];
    }
    lut
}

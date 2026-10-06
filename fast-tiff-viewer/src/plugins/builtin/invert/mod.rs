//! Invert: the smallest useful filter, and the boundary's oracle.
//!
//! It inverts **whatever is loaded**. A single-plane file is a stack with one
//! plane in it, so there is one tool rather than two and nothing to choose
//! between: open a photograph and it inverts the photograph, open a timelapse
//! and it inverts the timelapse.
//!
//! # About the type's range, not the plane's
//!
//! `255 - v` for 8-bit, `65535 - v` for 16-bit — which is what ImageJ's
//! `Edit > Invert` does, and it buys two properties that reflecting about each
//! plane's own min and max does not:
//!
//! * **It is exactly reversible.** Running it twice is the identity, because
//!   the range it reflects about does not depend on the data.
//! * **It is comparable across planes.** A per-plane range gives every plane a
//!   different mapping, so a dim frame and a bright frame come back equally
//!   bright — which destroys exactly the signal a timelapse is of.
//!
//! It also keeps the file's bit depth, where a per-plane range has to widen to
//! float: there is no 16-bit answer to "reflect about 41.6".
//!
//! # Float has no range of its own
//!
//! So for a float stack there is nothing to reflect about without looking at
//! the data, and it is measured: one pass for the stack's global minimum and
//! maximum, then `min + max - v`. Global rather than per-plane, for the
//! comparability reason above, and said in the log — unlike the integer cases,
//! the mapping depends on the file.

use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, Params, Plane, PlaneData, Plugin, PluginError,
    PluginInfo,
};

use crate::plugins::builtin::shared::{deliver, in_new_window, Store};

/// Invert every plane of whatever is loaded, about the sample type's range.
pub struct Invert;

impl Plugin for Invert {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.invert", "Invert")
            .menu_path("Filters")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Invert about the sample type's full range — 255-v for 8-bit, 65535-v for \
                 16-bit. Every plane of a stack, or the one plane of a single image. \
                 Exactly reversible, and keeps the file's bit depth.",
            )
    }

    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        vec![in_new_window()]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.plane_len() == 0 {
            return Err(PluginError::unsupported("the stack has no pixels"));
        }
        let channels = info.channels.max(1);
        let slices = info.slices.max(1);
        let frames = info.frames.max(1);
        let store = Store::of(info.pixel_type);
        let n_planes = channels * slices * frames;

        // Integer widths bring their range with them. Float does not, so it is
        // measured — which costs a pass over the stack, and is why the two
        // cases are told apart here rather than inside the loop below.
        let measured = store.range().is_none();
        let (lo, hi) = match store.range() {
            Some(range) => range,
            None => {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                let mut buf = Vec::new();
                let mut done = 0usize;
                for t in 0..frames {
                    for z in 0..slices {
                        for c in 0..channels {
                            // Half the bar for measuring, half for inverting:
                            // the two passes cost the same, and a bar that
                            // reached the end and started again reads as a
                            // stall.
                            if !host.progress(0.5 * done as f32 / n_planes as f32) {
                                return Ok(Outcome::Cancelled);
                            }
                            host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                            for &v in buf.iter().filter(|v| v.is_finite()) {
                                lo = lo.min(v);
                                hi = hi.max(v);
                            }
                            done += 1;
                        }
                    }
                }
                if lo > hi {
                    return Err(PluginError::unsupported(
                        "this stack has no finite samples to invert",
                    ));
                }
                host.log(&format!(
                    "float data: inverted about its own measured range, {lo} to {hi}"
                ));
                (lo, hi)
            }
        };
        let pivot = lo + hi;

        let mut planes: Vec<PlaneData> = Vec::with_capacity(n_planes);
        let mut buf = Vec::new();
        let mut done = 0usize;
        let (base, span) = if measured { (0.5, 0.5) } else { (0.0, 1.0) };
        for t in 0..frames {
            for z in 0..slices {
                for c in 0..channels {
                    if !host.progress(base + span * done as f32 / n_planes as f32) {
                        return Ok(Outcome::Cancelled);
                    }
                    host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                    // A non-finite sample has no reflection; it is left as it
                    // is rather than becoming the bottom of the range, which
                    // would turn a gap into a measurement.
                    for v in buf.iter_mut() {
                        if v.is_finite() {
                            *v = pivot - *v;
                        }
                    }
                    planes.push(store.plane(std::mem::take(&mut buf)));
                    done += 1;
                }
            }
        }

        let image = ImageResult {
            width: info.width,
            height: info.height,
            channels,
            slices,
            frames,
            pixel_type: store.pixel_type(),
            planes,
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-inverted", host.stack_info().name),
        };
        image.validate()?;
        Ok(deliver(image, params))
    }
}

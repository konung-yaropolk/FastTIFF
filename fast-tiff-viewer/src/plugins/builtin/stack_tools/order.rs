//! Slice Order Invert: the same planes, back to front.
//!
//! ImageJ reaches this through `Image > Stacks > Tools > Stack Sorter`'s
//! *Invert*, and it is the operation you want when an acquisition ran the wrong
//! way through a volume — a z-stack collected bottom-up when everything
//! downstream assumes top-down, or a series whose frames were assembled in
//! reverse.
//!
//! It is not a pixel operation at all: every plane comes through untouched, and
//! only their order changes. So it is exactly reversible, it cannot change a
//! sample value, and it keeps the file's bit depth for free.
//!
//! # Not to be confused with Invert
//!
//! `Filters > Invert` reflects the *samples*. This reverses the *order*. The two
//! have nothing in common but the word, which is why this one spells out which
//! it is in its name — "Invert Stack" sitting next to "Invert" was a menu where
//! the difference between light and dark and the difference between first and
//! last read as the same choice.

use super::{axes, deliver, in_new_window, map_planes, Axis, Store};
use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, ParamKind, Params, Plugin, PluginError,
    PluginInfo,
};

/// Reverse the order of the planes along an axis.
pub struct SliceOrderInvert;

impl Plugin for SliceOrderInvert {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stacktools.order", "Slice Order Invert")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Reverse the order of the planes along Z or T. The pixels are untouched \
                 — this reorders, it does not invert samples.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let offered = axes(&info);
        vec![
            ParamDecl::new(
                "axis",
                "Axis",
                ParamKind::Choice {
                    default: 0,
                    options: offered.iter().map(|a| a.label().to_string()).collect(),
                },
            )
            .help("Which axis to reverse. Channels keep their order either way."),
            in_new_window(),
        ]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        let offered = axes(&info);
        let axis = offered
            .get(params.choice("axis", 0))
            .copied()
            .unwrap_or(Axis::Z);
        let depth = axis.depth(&info);
        if depth <= 1 {
            return Err(PluginError::unsupported(format!(
                "this stack is one plane deep along {} — there is no order to reverse",
                axis.label()
            )));
        }

        // The other axis keeps its order; only the chosen one runs backwards.
        let (out_slices, out_frames) = (info.slices.max(1), info.frames.max(1));
        let mut pairs = Vec::with_capacity(out_slices * out_frames);
        for t in 0..out_frames {
            for z in 0..out_slices {
                pairs.push(match axis {
                    Axis::Z => (depth - 1 - z, t),
                    Axis::T => (z, depth - 1 - t),
                });
            }
        }

        let store = Store::of(info.pixel_type);
        let Some(planes) = map_planes(host, &pairs, store, |_| {})? else {
            return Ok(Outcome::Cancelled);
        };

        host.log(&format!(
            "reversed the order of {depth} plane(s) along {}",
            axis.label()
        ));

        let image = ImageResult {
            width: info.width,
            height: info.height,
            channels: info.channels.max(1),
            slices: out_slices,
            frames: out_frames,
            pixel_type: store.pixel_type(),
            planes,
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-reversed", host.stack_info().name),
        };
        image.validate()?;
        Ok(deliver(image, params))
    }
}

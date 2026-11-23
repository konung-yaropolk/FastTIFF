//! Split a stack before a one-based frame or slice, keeping whole channel sets.

use super::{result_spec, Axis};
use crate::shared::{Store, Stored};
use fasttiff_plugin_api::{
    HostContext, ImageInfo, ImageStream, Outcome, ParamDecl, ParamKind, Params, Plane, Plugin,
    PluginError, PluginInfo, StreamTarget,
};

#[derive(Default)]
pub struct SplitStack;

fn cut_axis(info: &ImageInfo, params: &Params) -> Axis {
    if info.frames <= 1 || (info.slices > 1 && params.choice("axis", default_axis(info)) == 1) {
        Axis::Z
    } else {
        Axis::T
    }
}

// Prefer time, unless only Z has a legal cut.
fn default_axis(info: &ImageInfo) -> usize {
    usize::from(info.frames < 3 && info.slices >= 3)
}

impl Plugin for SplitStack {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stacktools.split-stack", "Split Stack…")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Split along time or Z and open both parts in new windows.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let mut decls = Vec::new();
        if info.slices > 1 && info.frames > 1 {
            decls.push(ParamDecl::new(
                "axis",
                "Split along",
                ParamKind::Choice {
                    options: vec![Axis::T.label().into(), Axis::Z.label().into()],
                    default: default_axis(&info),
                },
            ));
        }
        let axis = cut_axis(&info, host.pending_params());
        let n = axis.depth(&info) as i64;
        if n >= 3 {
            let label = if axis == Axis::Z || host.frame_axis_is_z() {
                "First slice in second stack"
            } else {
                "First frame in second stack"
            };
            decls.push(ParamDecl::new(
                "first",
                label,
                ParamKind::Int {
                    min: 2,
                    max: n - 1,
                    default: n / 2 + 1,
                },
            ));
        } else if !decls.is_empty() {
            decls.push(ParamDecl::new(
                "too_short",
                "Splitting requires at least three positions on this axis.",
                ParamKind::Label,
            ));
        }
        decls
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        let axis = cut_axis(&info, params);
        let n = axis.depth(&info);
        if n < 3 {
            return Err(PluginError::unsupported(format!(
                "Split Stack needs at least three positions along {}.",
                axis.label()
            )));
        }
        let first = params.int("first", (n / 2 + 1) as i64);
        if !(2..n as i64).contains(&first) {
            return Err(PluginError::failed(format!(
                "The first position of the second stack must be between 2 and {}.",
                n - 1
            )));
        }
        let cut = first as usize - 1;
        let channels = info.channels.max(1);
        let mut luts = host
            .view()
            .luts
            .iter()
            .take(channels)
            .copied()
            .collect::<Vec<_>>();
        if !luts.is_empty() {
            luts.resize_with(channels, super::split::gray_lut);
        }
        let store = Store::of(info.pixel_type);
        let mut samples = Vec::new();
        let mut stored = Stored::default();
        let mut documents = Vec::new();
        let total = info.plane_count();
        let mut done = 0;
        for (suffix, range) in [("part1", 0..cut), ("part2", cut..n)] {
            let (times, slices) = match axis {
                Axis::T => (range, 0..info.slices.max(1)),
                Axis::Z => (0..info.frames.max(1), range),
            };
            let (depth, frames) = if host.frame_axis_is_z() {
                (times.len(), 1)
            } else {
                (slices.len(), times.len())
            };
            let spec = result_spec(host, store, depth, frames, suffix);
            let mut out = ImageStream::begin(host, spec)?;
            for t in times {
                for z in slices.clone() {
                    for c in 0..channels {
                        if !host.progress(done as f32 / total as f32) {
                            return Ok(Outcome::Cancelled);
                        }
                        host.read_plane_f32(Plane::new(c, z, t), &mut samples)?;
                        out.push(host, store.encode(&samples, &mut stored))?;
                        done += 1;
                    }
                }
            }
            if let Some(document) = out.finish_document(host, luts.clone())? {
                documents.push(document);
            }
        }
        if !host.progress(1.0) {
            return Ok(Outcome::Cancelled);
        }
        Ok(if host.streams_results() {
            Outcome::Streamed {
                target: StreamTarget::NewDocuments,
            }
        } else {
            Outcome::NewDocuments(documents)
        })
    }
}

//! Invert: the smallest useful filter, and the boundary's oracle.

use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType, Plane, PlaneData,
    Plugin, PluginError, PluginInfo,
};

/// Invert the frame on screen, about its own range.
pub struct Invert;

impl Plugin for Invert {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.invert", "Invert")
            .menu_path("Filters")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Invert the current frame about its own min/max.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let mut decls = vec![ParamDecl::new(
            "all_channels",
            "All channels",
            ParamKind::Bool { default: false },
        )
        .help("Invert every channel rather than only the first.")];
        // Only offer the choice when there is one — a checkbox that cannot
        // change anything is worse than no checkbox.
        if info.channels <= 1 {
            decls.clear();
        }
        decls
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.plane_len() == 0 {
            return Err(PluginError::unsupported("the stack has no pixels"));
        }
        let t = host.view().frame_index.min(info.frames.saturating_sub(1));
        let all = params.bool("all_channels", false);
        let n = if all { info.channels.max(1) } else { 1 };

        let mut planes = Vec::with_capacity(n);
        let mut buf = Vec::new();
        for c in 0..n {
            if !host.progress(c as f32 / n as f32) {
                return Ok(Outcome::Cancelled);
            }
            host.read_plane_f32(Plane::new(c, 0, t), &mut buf)?;
            // Invert about the plane's own range, which is what makes this
            // meaningful for float data with no natural maximum.
            let (lo, hi) = buf
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    if v.is_finite() {
                        (l.min(v), h.max(v))
                    } else {
                        (l, h)
                    }
                });
            let (lo, hi) = if lo.is_finite() && hi.is_finite() {
                (lo, hi)
            } else {
                (0.0, 1.0)
            };
            planes.push(PlaneData::F32(
                buf.iter()
                    .map(|&v| if v.is_finite() { hi - (v - lo) } else { v })
                    .collect(),
            ));
        }

        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels: n,
            slices: 1,
            frames: 1,
            pixel_type: PixelType::F32,
            planes,
            name: format!("{}-inverted", host.stack_info().name),
        })))
    }
}

//! Plugins compiled into the application.
//!
//! These exist for three reasons beyond being useful. They give the browser
//! build a working Plugins menu, where no shared library can ever be loaded.
//! They force the contract to have a real consumer before any loading mechanism
//! exists, which is how a plugin API avoids being designed in the abstract and
//! then failing on the first real plugin. And once the `.dll` lane lands they
//! are the oracle for it: the same filter, run both ways, must produce
//! byte-identical output.

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

/// Flatten the Z axis with a per-pixel statistic — ImageJ's Z Project.
pub struct ZProject;

impl Plugin for ZProject {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.zproject", "Z Project…")
            .menu_path("Stacks")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Project the Z axis of the current timepoint to one plane.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        vec![
            ParamDecl::new(
                "method",
                "Projection",
                ParamKind::Choice {
                    default: 0,
                    options: vec![
                        "Maximum".into(),
                        "Mean".into(),
                        "Minimum".into(),
                        "Sum".into(),
                    ],
                },
            ),
            ParamDecl::new(
                "first",
                "First slice",
                ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: info.slices.max(1) as i64,
                },
            ),
            ParamDecl::new(
                "last",
                "Last slice",
                ParamKind::Int {
                    default: info.slices.max(1) as i64,
                    min: 1,
                    max: info.slices.max(1) as i64,
                },
            ),
            ParamDecl::new(
                "all_channels",
                "All channels",
                ParamKind::Bool { default: true },
            ),
        ]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.slices <= 1 {
            return Err(PluginError::unsupported(
                "this stack has a single Z slice — there is nothing to project",
            ));
        }
        let n_px = info.plane_len();
        let t = host.view().frame_index.min(info.frames.saturating_sub(1));

        // The dialog is 1-based, as ImageJ's is; convert once, here.
        let first = (params.int("first", 1).max(1) as usize) - 1;
        let last = (params.int("last", info.slices as i64).max(1) as usize) - 1;
        let (first, last) = if first <= last {
            (first, last)
        } else {
            (last, first)
        };
        let last = last.min(info.slices - 1);
        let depth = last - first + 1;

        let method = params.choice("method", 0);
        let channels = if params.bool("all_channels", true) {
            info.channels.max(1)
        } else {
            1
        };

        let mut planes = Vec::with_capacity(channels);
        let mut buf = Vec::new();
        let total = (channels * depth).max(1);
        let mut done = 0usize;

        for c in 0..channels {
            let mut acc = vec![
                match method {
                    0 => f32::NEG_INFINITY, // Maximum
                    2 => f32::INFINITY,     // Minimum
                    _ => 0.0,
                };
                n_px
            ];
            for z in first..=last {
                if !host.progress(done as f32 / total as f32) {
                    return Ok(Outcome::Cancelled);
                }
                host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                for (a, &v) in acc.iter_mut().zip(buf.iter()) {
                    match method {
                        0 => *a = a.max(v),
                        2 => *a = a.min(v),
                        _ => *a += v,
                    }
                }
                done += 1;
            }
            // Mean is Sum scaled; doing it here keeps one accumulation loop.
            if method == 1 && depth > 0 {
                let inv = 1.0 / depth as f32;
                for a in &mut acc {
                    *a *= inv;
                }
            }
            planes.push(PlaneData::F32(acc));
        }

        let label = ["max", "mean", "min", "sum"][method.min(3)];
        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels,
            slices: 1,
            frames: 1,
            pixel_type: PixelType::F32,
            planes,
            name: format!("{}-{label}", host.stack_info().name),
        })))
    }
}

/// The plugins compiled into this build.
pub fn all() -> Vec<Box<dyn Plugin>> {
    vec![Box::new(Invert), Box::new(ZProject)]
}

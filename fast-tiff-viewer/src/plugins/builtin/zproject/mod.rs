//! Z Project: flatten one of a stack's axes with a per-pixel statistic.

use fasttiff_plugin_api::{
    HostContext, ImageInfo, ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType, Plane,
    PlaneData, Plugin, PluginError, PluginInfo,
};

/// An axis a projection can run along.
///
/// Not every stack has both, and one with a single plane along an axis has
/// nothing to project there — flattening it would be a copy. So the dialog is
/// offered only the axes that exist, which is why this is a list rather than a
/// fixed pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axis {
    Z,
    T,
}

impl Axis {
    fn label(self) -> &'static str {
        match self {
            Axis::Z => "Z (slices)",
            Axis::T => "T (frames)",
        }
    }

    /// The suffix on the result's name, so two projections of one stack are
    /// told apart by what they are rather than by the order they were made in.
    fn tag(self) -> &'static str {
        match self {
            Axis::Z => "z",
            Axis::T => "t",
        }
    }

    /// How many planes this stack has along the axis.
    fn depth(self, info: &ImageInfo) -> usize {
        match self {
            Axis::Z => info.slices,
            Axis::T => info.frames,
        }
    }
}

/// The axes worth offering, in dialog order.
///
/// Never empty: a stack that is a single plane still gets a selector, showing
/// the axis it would have projected, and [`ZProject::run`] refuses it with a
/// reason. An empty dropdown would be a control with nothing in it at all,
/// which reads as broken rather than as inapplicable.
fn axes(info: &ImageInfo) -> Vec<Axis> {
    let found: Vec<Axis> = [Axis::Z, Axis::T]
        .into_iter()
        .filter(|a| a.depth(info) > 1)
        .collect();
    if found.is_empty() {
        vec![Axis::Z]
    } else {
        found
    }
}

/// Flatten an axis with a per-pixel statistic — ImageJ's Z Project.
pub struct ZProject;

impl Plugin for ZProject {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.zproject", "Z Project…")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Flatten a stack's Z or T axis to one plane per channel.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let axes = axes(&info);
        // The range has to cover whichever axis is picked, and which one that
        // is cannot be known until the dialog is answered — the declarations
        // are made once, before it opens. So it spans the longest axis on
        // offer, and `run` clamps to the one actually chosen.
        let longest = axes
            .iter()
            .map(|a| a.depth(&info))
            .max()
            .unwrap_or(1)
            .max(1) as i64;
        vec![
            ParamDecl::new(
                "axis",
                "Axis",
                ParamKind::Choice {
                    default: 0,
                    options: axes.iter().map(|a| a.label().to_string()).collect(),
                },
            )
            .help("Which axis to flatten. A stack with only one of them has no choice to make."),
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
                "First",
                ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: longest,
                },
            )
            .help("Counted along the axis above, from 1."),
            ParamDecl::new(
                "last",
                "Last",
                ParamKind::Int {
                    default: longest,
                    min: 1,
                    max: longest,
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
        let offered = axes(&info);
        let axis = offered
            .get(params.choice("axis", 0))
            .copied()
            .unwrap_or(Axis::Z);
        let available = axis.depth(&info);
        if available <= 1 {
            return Err(PluginError::unsupported(format!(
                "this stack is one plane deep along {} — there is nothing to project",
                axis.label()
            )));
        }
        let n_px = info.plane_len();
        let t = host.view().frame_index.min(info.frames.saturating_sub(1));

        // The dialog is 1-based, as ImageJ's is; convert once, here.
        let first = (params.int("first", 1).max(1) as usize) - 1;
        let last = (params.int("last", available as i64).max(1) as usize) - 1;
        let (first, last) = if first <= last {
            (first, last)
        } else {
            (last, first)
        };
        // Clamped to the axis actually chosen, which the declared range could
        // not be: it had to cover the longer of the two.
        let first = first.min(available - 1);
        let last = last.min(available - 1);
        let depth = last - first + 1;

        let method = params.choice("method", 0);
        let channels = if params.bool("all_channels", true) {
            info.channels.max(1)
        } else {
            1
        };

        // Projecting Z flattens the slices of the timepoint on screen: one
        // plane per channel. Projecting T flattens the timepoints *at every
        // slice*, because the view says which timepoint is showing but not
        // which slice — there is no current one to pick, and picking the first
        // would quietly throw the rest of a 4D stack away. A timelapse, which
        // is the ordinary case, has one slice and so gets one plane either way.
        let out_slices = match axis {
            Axis::Z => 1,
            Axis::T => info.slices.max(1),
        };

        let mut planes = Vec::with_capacity(channels * out_slices);
        let mut buf = Vec::new();
        let total = (out_slices * channels * depth).max(1);
        let mut done = 0usize;

        // Channel fastest, then Z: the plane order the contract asks for.
        for z in 0..out_slices {
            for c in 0..channels {
                let mut acc = vec![
                    match method {
                        0 => f32::NEG_INFINITY, // Maximum
                        2 => f32::INFINITY,     // Minimum
                        _ => 0.0,
                    };
                    n_px
                ];
                for k in first..=last {
                    if !host.progress(done as f32 / total as f32) {
                        return Ok(Outcome::Cancelled);
                    }
                    let plane = match axis {
                        Axis::Z => Plane::new(c, k, t),
                        Axis::T => Plane::new(c, z, k),
                    };
                    host.read_plane_f32(plane, &mut buf)?;
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
        }

        let label = ["max", "mean", "min", "sum"][method.min(3)];
        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels,
            slices: out_slices,
            frames: 1,
            pixel_type: PixelType::F32,
            planes,
            channel_colors: Vec::new(),
            name: format!("{}-{}{label}", host.stack_info().name, axis.tag()),
        })))
    }
}

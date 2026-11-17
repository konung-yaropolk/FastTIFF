//! Plot third axis: mean pixel value along Z or T, whole frame or per region.
//!
//! What a fluorescence timelapse is usually *for*: how bright something is, as
//! it changes. The whole frame answers "did anything happen"; a region answers
//! "did it happen *there*", which is the question a cell or a process is.
//!
//! # Everything here is the plugin's
//!
//! The host draws the chart and turns drags into regions, and that is all it
//! does. Which axes exist, what a mean is, what the axes are called, why a
//! single-plane stack is refused — all of it lives in this file. The host is
//! handed a [`Plot`] and never learns what it is a plot of.
//!
//! # Units
//!
//! The file's own sample values: 0..255 for an 8-bit file, 0..65535 for a
//! 16-bit one, the stated number for float or signed data. That is what
//! [`HostContext::read_plane_f32`] promises, and it is deliberately *not* the
//! display's units — a trace rescaled by the contrast slider would move when
//! the slider did while the specimen sat still.
//!
//! # What is measured, and what is not
//!
//! A plain mean over the pixels a region covers. No background subtraction, no
//! bleach correction, no dF/F. Those are analysis choices with more than one
//! defensible answer, and a viewer that quietly picked one would be reporting
//! its own opinion as a measurement.

use fasttiff_plugin_api::{
    HostContext, ImageInfo, Outcome, ParamDecl, ParamKind, Params, Plane, Plot, Plugin,
    PluginError, PluginInfo, Roi, SelectionKind, Series,
};

/// An axis a trace can run along.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Axis {
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

    fn depth(self, info: &ImageInfo) -> usize {
        match self {
            Axis::Z => info.slices,
            Axis::T => info.frames,
        }
    }

    /// The plane at index `i` along this axis.
    fn plane(self, c: usize, i: usize) -> Plane {
        match self {
            Axis::Z => Plane::new(c, i, 0),
            Axis::T => Plane::new(c, 0, i),
        }
    }
}

/// The axes with more than one plane on them, longest first.
///
/// Empty for a single-plane stack, which is exactly the case this plugin
/// refuses — and an empty list says so more clearly than a list with a useless
/// entry in it.
///
/// Worth knowing: `fast_tiff_lib::resolve_dimensions` folds Z into T unless a
/// file genuinely has all three axes, so for most stacks the answer here is T
/// alone. That is the same interpretation the rest of the viewer uses, which is
/// the point — a plot along "Z" that disagreed with the Z slider would be
/// measuring something the user cannot see.
pub(crate) fn axes(info: &ImageInfo) -> Vec<Axis> {
    let mut found: Vec<Axis> = [Axis::Z, Axis::T]
        .into_iter()
        .filter(|a| a.depth(info) > 1)
        .collect();
    found.sort_by_key(|a| std::cmp::Reverse(a.depth(info)));
    found
}

/// The mean of the pixels `roi` covers, or of the whole plane when it is `None`.
///
/// `f64` for the running sum: a 2048x2048 plane of 16-bit samples adds up to
/// ~2.7e11, which `f32` cannot hold to the nearest integer, and the error grows
/// with every pixel added. The mean of a region is a number people subtract
/// from another number, so a few counts of drift is a few counts of nonsense.
pub(crate) fn mean_of(plane: &[f32], indices: Option<&[usize]>) -> f32 {
    let (sum, n) = match indices {
        None => (plane.iter().map(|&v| v as f64).sum::<f64>(), plane.len()),
        Some(idx) => {
            let mut sum = 0f64;
            let mut n = 0usize;
            for &i in idx {
                if let Some(v) = plane.get(i) {
                    sum += *v as f64;
                    n += 1;
                }
            }
            (sum, n)
        }
    };
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64) as f32
    }
}

/// Mean pixel value along a stack's third axis.
#[derive(Default)]
pub struct PlotAxis;

impl Plugin for PlotAxis {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.plot-axis", "Plot third axis")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Graph the mean pixel value along Z or T. Drag on the image to \
                 measure regions instead of the whole frame.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let found = axes(&info);
        let mut decls = Vec::new();

        // A stack with one axis still gets the selector, showing what will be
        // plotted — the value says what is about to happen even when it is not
        // a decision. A stack with none gets nothing, and `run` refuses it.
        if !found.is_empty() {
            decls.push(
                ParamDecl::new(
                    "axis",
                    "Axis",
                    ParamKind::Choice {
                        default: 0,
                        options: found.iter().map(|a| a.label().to_string()).collect(),
                    },
                )
                .help("Which axis to plot along."),
            );
        }
        if info.channels > 1 {
            decls.push(ParamDecl::new(
                "channel",
                "Channel",
                ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: info.channels as i64,
                },
            ));
        }
        decls
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        let found = axes(&info);
        if found.is_empty() {
            // The refusal the request asked for, in the plugin that knows why.
            return Err(PluginError::unsupported(
                "there is no third dimension to measure: this stack is a single \
                 plane in both Z and T",
            ));
        }
        let axis = *found.get(params.choice("axis", 0)).unwrap_or(&found[0]);
        let depth = axis.depth(&info);
        // Counted from 1 in the dialog, because that is how a microscope counts
        // channels; from 0 here, because that is how a plane is addressed.
        let channel = (params.int("channel", 1).max(1) as usize - 1).min(info.channels.max(1) - 1);

        // The regions the user drew, resolved to plane indices once rather than
        // per point: a thousand-frame trace would otherwise re-run the ellipse
        // test a thousand times over the same coordinates.
        let rois: Vec<Roi> = host.selection().to_vec();
        let masks: Vec<Vec<usize>> = rois
            .iter()
            .map(|r| r.indices(info.width, info.height))
            .collect();

        let mut series: Vec<Vec<f32>> = vec![Vec::with_capacity(depth); masks.len().max(1)];
        let mut plane = Vec::new();
        for i in 0..depth {
            if !host.progress(i as f32 / depth as f32) {
                return Ok(Outcome::Cancelled);
            }
            host.read_plane_f32(axis.plane(channel, i), &mut plane)?;
            if masks.is_empty() {
                series[0].push(mean_of(&plane, None));
            } else {
                for (s, mask) in series.iter_mut().zip(&masks) {
                    s.push(mean_of(&plane, Some(mask)));
                }
            }
        }

        let mut plot = Plot::new("Plot third axis")
            .labels(axis.label(), "Mean pixel value")
            // The tool is asked for every time: it is what makes the next drag
            // recompute this plot.
            .wants(SelectionKind::Regions);

        // Seconds instead of a frame index when the file says how long a frame
        // took — the number a reader of the plot actually wants.
        if axis == Axis::T {
            if let Some(dt) = host.stack_info().frame_interval_s.filter(|d| *d > 0.0) {
                plot = plot.scale(0.0, dt).labels("Time (s)", "Mean pixel value");
            }
        } else if let Some(dz) = host.stack_info().spacing.z.filter(|d| *d > 0.0) {
            let unit = host.stack_info().unit.clone().unwrap_or_default();
            let label = if unit.is_empty() {
                "Z".to_string()
            } else {
                format!("Z ({unit})")
            };
            plot = plot.scale(0.0, dz).labels(label, "Mean pixel value");
        }

        for (i, values) in series.into_iter().enumerate() {
            let label = if masks.is_empty() {
                "Whole frame".to_string()
            } else {
                format!("ROI {}", i + 1)
            };
            plot = plot.push(Series::new(label, values));
        }

        host.log(&format!(
            "{} points along {}, {}",
            depth,
            axis.label(),
            if rois.is_empty() {
                "whole frame".to_string()
            } else {
                format!("{} region(s)", rois.len())
            }
        ));
        Ok(Outcome::Plot(Box::new(plot)))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

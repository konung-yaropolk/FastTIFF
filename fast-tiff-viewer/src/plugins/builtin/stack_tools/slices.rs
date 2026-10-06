//! Slice Keeper and Slice Remover: taking planes out along an axis.
//!
//! ImageJ's `Image > Stacks > Tools > Slice Keeper` and `Slice Remover`, which
//! are the same arithmetic read two ways: both take a first, a last and an
//! increment, both count from 1 and include both ends, and the selection they
//! describe is `first, first+increment, first+2*increment, …` up to and
//! including `last`. Keeper keeps exactly that selection; Remover keeps exactly
//! its complement.
//!
//! That the increment applies to *both* is the part worth stating, because the
//! common case hides it: with an increment of 1 the selection is the whole range
//! and Remover deletes a contiguous block, which is what people picture. With an
//! increment of 2 Remover deletes every other plane in the range and leaves the
//! ones between — it is not "remove the range in steps".
//!
//! # 1-based, like the dialog
//!
//! The numbers in the dialog are the numbers on the slice slider, so they are
//! 1-based and converted exactly once, on the way in. Everything past that
//! point is an index.

use super::super::shared::{deliver, in_new_window, Store};
use super::{axes, map_planes, Axis};
use fasttiff_plugin_api::{
    HostContext, ImageInfo, ImageResult, Outcome, ParamDecl, ParamKind, Params, Plugin,
    PluginError, PluginInfo,
};

/// Keep the selected planes and discard the rest.
pub struct SliceKeeper;

/// Discard the selected planes and keep the rest.
pub struct SliceRemover;

/// Which way round a tool reads the selection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sense {
    Keep,
    Remove,
}

impl Plugin for SliceKeeper {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stacktools.keeper", "Slice Keeper…")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Keep a range of planes along Z or T, and discard the rest.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        declare(&host.image())
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        run(host, params, Sense::Keep)
    }
}

impl Plugin for SliceRemover {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stacktools.remover", "Slice Remover…")
            .menu_path("Stack")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Remove a range of planes along Z or T, and keep the rest.")
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        declare(&host.image())
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        run(host, params, Sense::Remove)
    }
}

/// The dialog both tools share.
fn declare(info: &ImageInfo) -> Vec<ParamDecl> {
    let offered = axes(info);
    // As in `ZProject`: the declared range has to cover whichever axis is
    // chosen, and that is not known until the dialog is answered, so it spans
    // the longest on offer and `run` clamps to the one actually picked.
    let longest = offered
        .iter()
        .map(|a| a.depth(info))
        .max()
        .unwrap_or(1)
        .max(1) as i64;
    vec![
        ParamDecl::new(
            "axis",
            "Axis",
            ParamKind::Choice {
                default: 0,
                options: offered.iter().map(|a| a.label().to_string()).collect(),
            },
        )
        .help("Which axis the numbers below count along. Channels are always kept whole."),
        ParamDecl::new(
            "first",
            "First",
            ParamKind::Int {
                default: 1,
                min: 1,
                max: longest,
            },
        )
        .help("Counted from 1, as the slider is. Included."),
        ParamDecl::new(
            "last",
            "Last",
            ParamKind::Int {
                default: longest,
                min: 1,
                max: longest,
            },
        )
        .help("Included."),
        ParamDecl::new(
            "increment",
            "Increment",
            ParamKind::Int {
                default: 1,
                min: 1,
                max: longest,
            },
        )
        .help(
            "1 selects every plane in the range. 2 selects every other one, counting \
             from the first.",
        ),
        in_new_window(),
    ]
}

/// The planes the dialog selected, as indices along `axis`.
///
/// Separated from the reading so the arithmetic can be tested without a stack:
/// the off-by-one lives here and nowhere else.
fn selected(first: usize, last: usize, increment: usize, depth: usize) -> Vec<usize> {
    if depth == 0 {
        return Vec::new();
    }
    // A dialog can be answered the wrong way round, and a range is a range.
    let (lo, hi) = if first <= last {
        (first, last)
    } else {
        (last, first)
    };
    let lo = lo.min(depth - 1);
    let hi = hi.min(depth - 1);
    let step = increment.max(1);
    (lo..=hi).step_by(step).collect()
}

fn run(host: &mut dyn HostContext, params: &Params, sense: Sense) -> Result<Outcome, PluginError> {
    let info = host.image();
    let offered = axes(&info);
    let axis = offered
        .get(params.choice("axis", 0))
        .copied()
        .unwrap_or(Axis::Z);
    let depth = axis.depth(&info);
    if depth <= 1 {
        return Err(PluginError::unsupported(format!(
            "this stack is one plane deep along {} — there is nothing to take out",
            axis.label()
        )));
    }

    // The dialog is 1-based; converted once, here.
    let first = (params.int("first", 1).max(1) as usize) - 1;
    let last = (params.int("last", depth as i64).max(1) as usize) - 1;
    let increment = params.int("increment", 1).max(1) as usize;
    let chosen = selected(first, last, increment, depth);

    let keep_along: Vec<usize> = match sense {
        Sense::Keep => chosen,
        Sense::Remove => (0..depth).filter(|i| !chosen.contains(i)).collect(),
    };
    if keep_along.is_empty() {
        return Err(PluginError::unsupported(
            "that would leave no planes at all. A stack has to have one.",
        ));
    }

    // Every (z, t) pair to read, in `xyczt` order. Taking planes out along one
    // axis leaves the other whole, so this is the product of the two — which is
    // also what makes the result's declared shape right by construction.
    let (out_slices, out_frames) = match axis {
        Axis::Z => (keep_along.len(), info.frames.max(1)),
        Axis::T => (info.slices.max(1), keep_along.len()),
    };
    let mut pairs = Vec::with_capacity(out_slices * out_frames);
    // z varies fastest within t, and `map_planes` walks the channels inside
    // each pair, which together is the `xyczt` order the contract asks for.
    for t in 0..out_frames {
        for z in 0..out_slices {
            pairs.push(match axis {
                Axis::Z => (keep_along[z], t),
                Axis::T => (z, keep_along[t]),
            });
        }
    }
    let store = Store::of(info.pixel_type);
    let Some(planes) = map_planes(host, &pairs, store, |_| {})? else {
        return Ok(Outcome::Cancelled);
    };

    let verb = match sense {
        Sense::Keep => "kept",
        Sense::Remove => "removed",
    };
    host.log(&format!(
        "{verb} {} of {depth} plane(s) along {}",
        match sense {
            Sense::Keep => keep_along.len(),
            Sense::Remove => depth - keep_along.len(),
        },
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
        name: format!("{}-{verb}", host.stack_info().name),
    };
    image.validate()?;
    Ok(deliver(image, params))
}

#[cfg(test)]
#[path = "slices_tests.rs"]
mod tests;

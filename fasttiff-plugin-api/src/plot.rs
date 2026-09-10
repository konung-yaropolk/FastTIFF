//! A chart a plugin computed, for the host to draw.
//!
//! The same split as [`crate::params`], for the same reason: a plugin cannot be
//! handed an `&mut egui::Ui`, so it does not draw the chart — it *declares* one
//! and the host renders it in whatever toolkit it happens to have. What the
//! plugin keeps is everything that makes the chart mean anything: what was
//! measured, what the axes are, what the traces are called.
//!
//! # A plot is a function of the selection
//!
//! A [`Plot`] can ask, through [`Plot::wants`], for a tool on the canvas. The
//! host arms it, and whenever the set of regions changes it calls
//! [`Plugin::run`](crate::Plugin::run) **again** — same plugin, same
//! [`Params`](crate::Params), new [`HostContext::selection`](crate::HostContext::selection).
//!
//! There is no session and no second entry point. A run is still one call that
//! returns and is done, the vtable stays stateless, and nothing acquires a
//! lifetime the two sides have to agree on. The only new thing is that the host
//! may start a run the user did not pick from the menu.

use crate::selection::Roi;

/// What the host should offer on the canvas while this plot is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectionKind {
    /// No tool. The plot is what it is, and the host will not call again.
    #[default]
    None,
    /// Regions, snapped to whole pixels. The host picks the gesture — on the
    /// desktop a drag, with Shift to add another — and which shapes to offer.
    Regions,
}

/// One curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    /// What to call it in the legend.
    pub label: String,
    /// One y per point.
    ///
    /// A non-finite value is a **gap**: the host breaks the line there rather
    /// than drawing through it, because a line across a hole asserts a
    /// measurement nobody made.
    pub values: Vec<f32>,
    /// What to draw it in, or `None` to take the next colour from the host's
    /// palette.
    ///
    /// The reason a plugin gets a say at all: when a series describes a region
    /// the user drew, echoing that region's colour back makes the legend
    /// readable off the picture without the two sides having to agree an
    /// ordering.
    pub color: Option<[u8; 3]>,
}

impl Series {
    pub fn new(label: impl Into<String>, values: Vec<f32>) -> Self {
        Series {
            label: label.into(),
            values,
            color: None,
        }
    }

    pub fn color(mut self, rgb: [u8; 3]) -> Self {
        self.color = Some(rgb);
        self
    }

    /// The smallest and largest finite value, or `None` if there are none.
    ///
    /// `None` rather than a made-up `0..1`: a plot with no finite point has no
    /// range, and inventing one draws a flat line that reads as data.
    pub fn range(&self) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in self.values.iter().copied().filter(|v| v.is_finite()) {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// A chart.
///
/// Every series shares one x axis. That is what makes them comparable by eye,
/// which is the whole reason to draw them together.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    /// The window's title.
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    /// The x value of point 0.
    pub x_start: f64,
    /// The x distance between consecutive points.
    ///
    /// Two numbers rather than a parallel array of x values: it cannot be
    /// ragged, it cannot disagree with a series' length, and it covers what a
    /// calibrated axis actually is — seconds from a frame interval, microns
    /// from a z step. A plot whose x values are genuinely irregular is a
    /// scatter, which is a different picture and not this one.
    pub x_step: f64,
    pub series: Vec<Series>,
    /// What the user should be able to draw so this plot can be recomputed for
    /// it.
    pub wants: SelectionKind,
}

impl Plot {
    /// A plot with an index x-axis and no selection tool — the simple case.
    pub fn new(title: impl Into<String>) -> Self {
        Plot {
            title: title.into(),
            x_label: String::new(),
            y_label: String::new(),
            x_start: 0.0,
            x_step: 1.0,
            series: Vec::new(),
            wants: SelectionKind::None,
        }
    }

    pub fn labels(mut self, x: impl Into<String>, y: impl Into<String>) -> Self {
        self.x_label = x.into();
        self.y_label = y.into();
        self
    }

    /// Scale the x axis: the first point's value, and the step between points.
    ///
    /// Use it to plot against seconds or microns instead of an index, when the
    /// file says what those are.
    pub fn scale(mut self, start: f64, step: f64) -> Self {
        self.x_start = start;
        self.x_step = step;
        self
    }

    /// Ask the host for a canvas tool, and to call again when it is used.
    pub fn wants(mut self, kind: SelectionKind) -> Self {
        self.wants = kind;
        self
    }

    pub fn push(mut self, series: Series) -> Self {
        self.series.push(series);
        self
    }

    /// The longest series, which is how many points the x axis spans.
    pub fn points(&self) -> usize {
        self.series
            .iter()
            .map(|s| s.values.len())
            .max()
            .unwrap_or(0)
    }

    /// The x value at point `i`.
    pub fn x_at(&self, i: usize) -> f64 {
        self.x_start + i as f64 * self.x_step
    }

    /// The value range across every series, so they share one scale.
    ///
    /// `None` when nothing finite was measured.
    pub fn range(&self) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for (a, b) in self.series.iter().filter_map(|s| s.range()) {
            lo = lo.min(a);
            hi = hi.max(b);
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// The regions a plot was computed for, as the host hands them over.
///
/// A borrowed slice on [`HostContext`](crate::HostContext), empty when the user
/// has drawn nothing — which a plugin should read as "the whole frame", since
/// that is the question it was asked before anything was selected.
pub type Selection<'a> = &'a [Roi];

#[cfg(test)]
#[path = "plot_tests.rs"]
mod tests;

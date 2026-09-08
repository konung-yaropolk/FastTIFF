//! The file's own metadata, restated so this crate needs no dependencies.
//!
//! These mirror `fast_tiff_lib::StackMeta` rather than re-exporting it. That
//! looks like duplication and is deliberate: a plugin author should not have to
//! depend on `fast-tiff-lib` — and pin its version — merely to read a channel
//! name. Keeping the contract dependency-free is what allows the app's own
//! libraries to move without every plugin needing a rebuild.

/// How the stack is meant to be shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisplayMode {
    /// One channel at a time.
    #[default]
    Grayscale,
    /// Channels composited in colour.
    Composite,
    /// A single channel carrying its own colour table.
    Color,
}

/// Physical pixel spacing, when the file states it.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Spacing {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
}

/// The file, as the host understands it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StackInfo {
    /// The file's path, when it has one. A stack opened from bytes — dropped
    /// onto the window, or picked in the browser build — has `None` here even
    /// though the viewer shows a name for it. A plugin that wants to re-read
    /// the file itself must handle that; the host's own workers do.
    pub path: Option<String>,
    /// What to call this document in a message or a new window's title.
    pub name: String,
    pub mode: DisplayMode,
    /// Unit for the spacing values, e.g. `"micron"`.
    pub unit: Option<String>,
    pub spacing: Spacing,
    /// Seconds between timepoints, when known.
    pub frame_interval_s: Option<f64>,
    /// Channel names, when the file names them. May be shorter than the
    /// channel count, or empty.
    pub channel_names: Vec<String>,
    /// Linear calibration `(c0, c1)`: a raw sample `r` means `c0 + c1 * r`.
    pub calibration: Option<(f64, f64)>,
    /// The file's `ImageDescription`, verbatim, for a plugin that wants to read
    /// a dialect the host does not parse.
    pub description: Option<String>,
}

//! Format-neutral stack metadata: the normalized model every dialect parses
//! *into* and serializes *out of*, plus the dispatch that picks a dialect.
//!
//! A TIFF's `ImageDescription` (tag 270) can carry metadata in several
//! incompatible conventions — ImageJ's `key=value` block, OME-XML, and others
//! to come. The rest of the crate (and the viewer) never wants to care which:
//! it wants channel/slice/frame counts, a display mode, per-channel LUTs and
//! ranges, calibration, and pixel size. That normalized view is [`StackMeta`],
//! and each dialect lives in its own submodule ([`imagej`], [`ome`]) as a pair
//! of `parse`/`serialize` functions over this shared model.
//!
//! - **Read**: [`parse`] sniffs the description text, dispatches to the matching
//!   dialect, and records which one ran in [`StackMeta::source_format`].
//! - **Write**: callers fill a neutral [`StackMetaWrite`] builder and pick a
//!   [`MetadataFormat`]; [`serialize_description`] renders it in that dialect.

pub mod imagej;
pub mod ome;

use anyhow::Result;

/// Which metadata dialect a file's `ImageDescription` was written in (read
/// side), or should be written in (write side). `None` means no recognized
/// metadata — a plain TIFF whose dimensions are inferred from the IFD count.
///
/// `non_exhaustive`: more dialects will be added; downstream matches keep a
/// wildcard arm so that's not a breaking change.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataFormat {
    None,
    ImageJ,
    Ome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayMode {
    Grayscale,
    Composite,
    Color,
}

#[derive(Clone, Debug)]
pub struct ChannelDisplay {
    /// 256-entry RGB lookup table. Defaults to identity grayscale, or a
    /// cycling set of standard channel colors for composite stacks when no
    /// explicit LUT/color could be parsed.
    pub lut: [[u8; 3]; 256],
    /// Display window (min, max) in raw sample units. `None` means
    /// "compute auto-contrast from the data" — the caller decides how.
    pub range: Option<(f64, f64)>,
}

// `non_exhaustive`: fields have been added before (`spacing`, `loop_playback`,
// `source_format`) and may be again; it's produced by parsing, not constructed
// downstream.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct StackMeta {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub mode: DisplayMode,
    pub unit: Option<String>,
    pub frame_interval_s: Option<f64>,
    pub channel_display: Vec<ChannelDisplay>,
    /// Linear pixel calibration `(c0, c1)`: a raw sample `r` represents the
    /// real value `c0 + c1 * r`. `None` when the file carries no usable linear
    /// calibration, in which case raw sample values are shown directly.
    pub calibration: Option<(f64, f64)>,
    /// Playback rate in frames/second. `None` when the file doesn't specify
    /// one — the viewer falls back to a default.
    pub fps: Option<f64>,
    /// Z-step between slices (in `unit`s).
    pub spacing: Option<f64>,
    /// Whether playback should loop.
    pub loop_playback: Option<bool>,
    /// Physical pixel width/height (in `unit`s). From a dialect's own field
    /// (OME `PhysicalSize`) when it has one, else the TIFF XResolution /
    /// YResolution tags (`pixel = 1 / resolution`). `None` when uncalibrated.
    pub pixel_width: Option<f64>,
    pub pixel_height: Option<f64>,
    /// True when the file supplied real per-channel colors (an ImageJ IJMetadata
    /// LUT block, or OME channel `Color`s), so the viewer should keep them
    /// rather than override with its grayscale/pseudocolor default.
    pub has_explicit_luts: bool,
    /// Which dialect produced this metadata (`None` if the description carried
    /// no recognized format and dimensions were inferred from the IFD count).
    pub source_format: MetadataFormat,
}

impl StackMeta {
    /// A blank metadata record: one grayscale channel, no calibration, `frames`
    /// inferred from the IFD count. The starting point every dialect refines,
    /// and the whole result for a plain TIFF with no recognized description.
    pub(crate) fn inferred(total_ifds: usize, source_format: MetadataFormat) -> Self {
        StackMeta {
            channels: 1,
            slices: 1,
            frames: total_ifds.max(1),
            mode: DisplayMode::Grayscale,
            unit: None,
            frame_interval_s: None,
            channel_display: vec![ChannelDisplay { lut: grayscale_lut(), range: None }],
            calibration: None,
            fps: None,
            spacing: None,
            loop_playback: None,
            pixel_width: None,
            pixel_height: None,
            has_explicit_luts: false,
            source_format,
        }
    }

    /// Maps a raw sample value to its calibrated real value (`c0 + c1 * raw`),
    /// or returns it unchanged when the file has no linear calibration.
    pub fn calibrate(&self, raw: f64) -> f64 {
        match self.calibration {
            Some((c0, c1)) => c0 + c1 * raw,
            None => raw,
        }
    }

    /// Physical voxel scale `(x, y, z)` for 3D display, from the pixel
    /// calibration (pixel width/height) and the Z `spacing`, all in the same
    /// `unit` — the raw calibrated numbers, *not* normalized (the 3D viewer
    /// normalizes them for display on its own). Missing values default to 1.0
    /// (an uncalibrated stack is 1:1:1; one with only `spacing=` keeps its
    /// 1:1:z ratio).
    pub fn voxel_scale(&self) -> [f32; 3] {
        let pos = |v: Option<f64>| v.filter(|x| x.is_finite() && *x > 0.0);
        let px = pos(self.pixel_width).unwrap_or(1.0);
        let py = pos(self.pixel_height).unwrap_or(px); // assume square pixels
        let pz = pos(self.spacing).unwrap_or(1.0);
        let clamp = |v: f64| (v as f32).clamp(1e-4, 1e6);
        [clamp(px), clamp(py), clamp(pz)]
    }
}

/// Read-side dispatch: build a normalized [`StackMeta`] from a file's raw
/// metadata inputs, choosing the dialect by inspecting the `ImageDescription`
/// text. `x/y_resolution` are the TIFF resolution tags (pixels per unit); a
/// dialect that carries its own pixel size (OME) prefers that and treats these
/// as a fallback. `ij_metadata`/`ij_metadata_counts` are the binary IJMetadata
/// block, only consulted on the ImageJ path.
pub fn parse(
    description: Option<&str>,
    ij_metadata: Option<&[u8]>,
    ij_metadata_counts: Option<&[u32]>,
    total_ifds: usize,
    x_resolution: Option<f64>,
    y_resolution: Option<f64>,
) -> StackMeta {
    match detect(description) {
        MetadataFormat::Ome => {
            // OME-XML: fall back to the neutral inferred meta if the XML turns
            // out to be malformed, rather than failing the whole open.
            ome::parse(description.unwrap_or_default(), total_ifds, x_resolution, y_resolution)
                .unwrap_or_else(|| StackMeta::inferred(total_ifds, MetadataFormat::None))
        }
        MetadataFormat::ImageJ => imagej::parse(
            description,
            ij_metadata,
            ij_metadata_counts,
            total_ifds,
            x_resolution,
            y_resolution,
        ),
        // No recognized dialect: neutral inferred metadata, but still honor the
        // TIFF resolution tags so a plain calibrated TIFF reports pixel size.
        _ => {
            let mut meta = StackMeta::inferred(total_ifds, MetadataFormat::None);
            meta.pixel_width = resolution_to_pixel(x_resolution);
            meta.pixel_height = resolution_to_pixel(y_resolution);
            meta
        }
    }
}

/// Classify a description string by dialect. OME-XML is checked first: an
/// OME-TIFF's description is XML in the OME namespace (ImageJ never emits XML),
/// so the check is unambiguous.
pub fn detect(description: Option<&str>) -> MetadataFormat {
    let Some(desc) = description else { return MetadataFormat::None };
    let trimmed = desc.trim_start();
    let looks_ome = trimmed.contains("<OME")
        || (trimmed.starts_with("<?xml") && desc.contains("openmicroscopy.org"));
    if looks_ome {
        MetadataFormat::Ome
    } else if desc.contains("ImageJ=") {
        MetadataFormat::ImageJ
    } else {
        MetadataFormat::None
    }
}

/// TIFF XResolution/YResolution (pixels per unit) → pixel size (`1 / res`),
/// filtering out the useless (non-finite / non-positive) values.
pub(crate) fn resolution_to_pixel(resolution: Option<f64>) -> Option<f64> {
    resolution.filter(|r| r.is_finite() && *r > 0.0).map(|r| 1.0 / r)
}

// ---- write side ----

/// One channel's write-side display info: an optional name and RGB color.
/// Consumed by dialects that carry per-channel color (OME); the ImageJ dialect
/// takes channel colors from [`StackMetaWrite::mode`] instead (it doesn't write
/// the binary LUT block).
#[derive(Clone, Debug, Default)]
pub(crate) struct ChannelWrite {
    pub name: Option<String>,
    pub color: Option<[u8; 3]>,
}

/// Format-neutral metadata to embed when writing a stack — the single input
/// every dialect serializes from (see [`serialize_description`]). Built with
/// [`StackMetaWrite::new`] plus chained setters; the dialect is chosen
/// separately via [`WriterOptions::metadata_format`](crate::WriterOptions::metadata_format).
///
/// The time-frame count isn't stored here — it's derived at write time from the
/// number of planes actually written (`planes / (channels * slices)`, which
/// must divide evenly). `channels`/`slices` describe the plane order, expected
/// in `xyczt` order (channel fastest, then Z, then time).
#[derive(Clone, Debug)]
pub struct StackMetaWrite {
    pub(crate) channels: usize,
    pub(crate) slices: usize,
    pub(crate) mode: DisplayMode,
    pub(crate) unit: Option<String>,
    pub(crate) fps: Option<f64>,
    pub(crate) frame_interval_s: Option<f64>,
    pub(crate) spacing: Option<f64>,
    pub(crate) loop_playback: Option<bool>,
    pub(crate) calibration: Option<(f64, f64)>,
    pub(crate) range: Option<(f64, f64)>,
    pub(crate) pixel_width: Option<f64>,
    pub(crate) pixel_height: Option<f64>,
    pub(crate) channel_info: Vec<ChannelWrite>,
    pub(crate) extra: Vec<(String, String)>,
}

impl StackMetaWrite {
    /// `channels` and `slices` describe the plane order; both are clamped to a
    /// minimum of 1. Display mode defaults to grayscale.
    pub fn new(channels: usize, slices: usize) -> Self {
        Self {
            channels: channels.max(1),
            slices: slices.max(1),
            mode: DisplayMode::Grayscale,
            unit: None,
            fps: None,
            frame_interval_s: None,
            spacing: None,
            loop_playback: None,
            calibration: None,
            range: None,
            pixel_width: None,
            pixel_height: None,
            channel_info: Vec::new(),
            extra: Vec::new(),
        }
    }

    /// Display mode (grayscale / composite / color).
    pub fn mode(mut self, mode: DisplayMode) -> Self {
        self.mode = mode;
        self
    }

    /// Spatial unit label (e.g. `"um"`).
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Playback rate in frames/second.
    pub fn fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Seconds between time frames.
    pub fn frame_interval_s(mut self, seconds: f64) -> Self {
        self.frame_interval_s = Some(seconds);
        self
    }

    /// Z-step between slices (in `unit`s).
    pub fn spacing(mut self, spacing: f64) -> Self {
        self.spacing = Some(spacing);
        self
    }

    /// Whether playback should loop.
    pub fn loop_playback(mut self, looped: bool) -> Self {
        self.loop_playback = Some(looped);
        self
    }

    /// Linear pixel calibration: real value = `c0 + c1 * raw`.
    pub fn calibration(mut self, c0: f64, c1: f64) -> Self {
        self.calibration = Some((c0, c1));
        self
    }

    /// Display window (min, max), applied to all channels.
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Physical pixel size (width, height, in `unit`s) — written to the TIFF
    /// XResolution/YResolution tags (as `1 / size`) and, for OME, to
    /// `PhysicalSizeX/Y`.
    pub fn pixel_size(mut self, width: f64, height: f64) -> Self {
        self.pixel_width = Some(width);
        self.pixel_height = Some(height);
        self
    }

    /// Add one channel's name and RGB color, in channel order. Consumed by
    /// dialects that carry per-channel color (OME); the ImageJ dialect ignores
    /// it (its colors follow [`mode`](Self::mode)).
    pub fn channel(mut self, name: impl Into<String>, color: [u8; 3]) -> Self {
        self.channel_info.push(ChannelWrite { name: Some(name.into()), color: Some(color) });
        self
    }

    /// Append a verbatim `key=value` line — the escape hatch for ImageJ
    /// description keys without a dedicated setter (`vunit`, `tunit`, …).
    /// Ignored by dialects other than ImageJ.
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.push((key.into(), value.into()));
        self
    }

    /// The time-frame count implied by `planes` under this stack's
    /// channels×slices layout, or an error if they don't divide evenly.
    pub(crate) fn derive_frames(&self, planes: usize) -> Result<usize> {
        let per_frame = self.channels * self.slices;
        if per_frame == 0 || planes % per_frame != 0 {
            anyhow::bail!(
                "{planes} plane(s) written, which doesn't divide evenly into {} channel(s) x {} \
                 slice(s) — a hyperstack needs channels x slices planes per time frame",
                self.channels,
                self.slices
            );
        }
        Ok(planes / per_frame)
    }

    /// Validate the text fields that land in the ASCII `ImageDescription` tag:
    /// NUL bytes truncate it on read-back regardless of dialect. `strict_lines`
    /// (ImageJ) additionally forbids newlines / `=`-in-keys, which would corrupt
    /// its line-oriented `key=value` block.
    pub(crate) fn validate_text(&self, strict_lines: bool) -> Result<()> {
        let has_nul = |s: &str| s.contains('\0');
        if self.unit.as_deref().is_some_and(has_nul) {
            anyhow::bail!("unit must not contain NUL bytes (TIFF ASCII fields are NUL-terminated)");
        }
        for ci in &self.channel_info {
            if ci.name.as_deref().is_some_and(has_nul) {
                anyhow::bail!("channel name must not contain NUL bytes");
            }
        }
        if strict_lines {
            let clean = |s: &str| !s.contains('\0') && !s.contains('\n');
            if !self.unit.as_deref().map_or(true, clean) {
                anyhow::bail!("ImageJ unit must not contain NUL or newline characters");
            }
            for (key, value) in &self.extra {
                if !clean(key) || key.contains('=') || !clean(value) {
                    anyhow::bail!("ImageJ extra key/value {key:?} must not contain NUL, newline, or '=' in the key");
                }
            }
        }
        Ok(())
    }
}

/// Geometry a dialect needs to render (OME's `Pixels` element wants the frame
/// size and pixel type). Supplied by the writer, which owns the sample type.
pub(crate) struct WriteGeometry {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: usize,
    /// OME `Pixels/@Type` string (`uint16`, `float`, …) for the sample type.
    pub ome_pixel_type: &'static str,
}

/// Write-side dispatch: render `meta` as `ImageDescription` text in `format`.
/// `planes` is the total plane count (for the derived frame count); `geom` is
/// only used by dialects that need frame geometry (OME).
pub(crate) fn serialize_description(
    format: MetadataFormat,
    meta: &StackMetaWrite,
    planes: usize,
    geom: &WriteGeometry,
) -> Result<String> {
    match format {
        MetadataFormat::ImageJ => imagej::serialize(planes, meta),
        MetadataFormat::Ome => ome::serialize(planes, meta, geom),
        other => anyhow::bail!("cannot serialize metadata as {other:?}"),
    }
}

// ---- dimension resolution ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedDimensions {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub triple_axis_warning: bool,
}

/// A dimension this size or smaller is assumed to be channels; anything
/// larger is assumed to be time. Matches the renderer's `MAX_CHANNELS` (6) so a
/// genuine 5- or 6-channel hyperstack isn't mistaken for a short time series.
/// Hardcoded rather than configurable: there's no size that's correct for every
/// dataset, but a per-file manual channels/frames swap (exposed in the UI)
/// covers the cases this misses, without the complexity of a global setting.
const CHANNEL_SIZE_CUTOFF: usize = 6;

/// Decides the *effective* (channels, slices, frames) to use, classifying
/// by size rather than trusting the file's own axis labels: a dimension of
/// `CHANNEL_SIZE_CUTOFF` or fewer is assumed to be channels, anything
/// larger is assumed to be time — this holds even when the file's
/// metadata claims the opposite (a "channels" value that's actually a
/// mislabeled frame count, or vice versa), or is missing entirely.
///
/// Z is never its own navigable axis: it's always folded into "frames" —
/// just another time-like step — *unless* the file genuinely has
/// channels, Z-slices, AND time-frames all greater than 1 at once. In that
/// case there's no lossless way to collapse to two axes, so the three raw
/// values are kept as-is and `triple_axis_warning` is set; the caller
/// should show a warning and always read the first Z-slice.
///
/// `channels * slices * frames` on the output always equals `c * z * f` on
/// the input — this never invents or drops planes, only reclassifies which
/// axis they belong to.
pub fn resolve_dimensions(c: usize, z: usize, f: usize) -> ResolvedDimensions {
    if c > 1 && z > 1 && f > 1 {
        return ResolvedDimensions {
            channels: c,
            slices: z,
            frames: f,
            triple_axis_warning: true,
        };
    }

    let time_only = (z * f).max(1); // Z always folds into time, unconditionally
    let c_is_channel_sized = c > 1 && c <= CHANNEL_SIZE_CUTOFF;
    let time_is_channel_sized = time_only > 1 && time_only <= CHANNEL_SIZE_CUTOFF;

    let (channels, frames) = if c_is_channel_sized {
        (c, time_only)
    } else if c > CHANNEL_SIZE_CUTOFF && time_is_channel_sized {
        // Roles look swapped: what's labeled "channels" is too big to
        // really be channels, but the combined time axis is small enough
        // to plausibly be the real channel count.
        (time_only, c)
    } else {
        // Nothing looks channel-sized; treat the whole stack as one
        // unsplit time series.
        (1, c.max(1) * time_only)
    };

    ResolvedDimensions {
        channels,
        slices: 1,
        frames,
        triple_axis_warning: false,
    }
}

// ---- LUT helpers ----

pub fn grayscale_lut() -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = [i as u8, i as u8, i as u8];
    }
    lut
}

/// A 256-entry ramp from black up to `color` at full intensity — the general
/// form of a single-hue channel LUT. Used for both the standard composite
/// palette and arbitrary per-channel colors (OME `Color`).
pub fn color_ramp_lut(color: [u8; 3]) -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let t = i as f32 / 255.0;
        *entry = [
            (color[0] as f32 * t) as u8,
            (color[1] as f32 * t) as u8,
            (color[2] as f32 * t) as u8,
        ];
    }
    lut
}

/// The base color for channel `index` in the standard composite-mode palette
/// (the full-intensity hue its LUT ramps up to).
pub fn composite_color(index: usize) -> [u8; 3] {
    const COLORS: [[u8; 3]; 7] = [
        [255, 0, 0],     // red
        [0, 255, 0],     // green
        [0, 0, 255],     // blue
        [255, 255, 0],   // yellow
        [255, 0, 255],   // magenta
        [0, 255, 255],   // cyan
        [255, 255, 255], // gray/white
    ];
    COLORS[index % COLORS.len()]
}

/// Standard composite-mode channel color cycle (channel LUTs assigned when a
/// hyperstack is opened without explicit per-channel colors).
pub fn default_composite_lut(channel_index: usize) -> [[u8; 3]; 256] {
    color_ramp_lut(composite_color(channel_index))
}

/// The default LUT for channel `c` under display `mode`: a flat grayscale
/// ramp in grayscale mode, otherwise the cycling composite-color palette.
/// The single source of truth so parsers and `resize_channel_display` (in the
/// app) can't drift — a grayscale stack stays grayscale even after a
/// channel-count change.
pub fn default_lut_for(mode: DisplayMode, c: usize) -> [[u8; 3]; 256] {
    match mode {
        DisplayMode::Grayscale => grayscale_lut(),
        DisplayMode::Composite | DisplayMode::Color => default_composite_lut(c),
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

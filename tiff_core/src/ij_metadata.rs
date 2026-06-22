//! ImageJ writes two kinds of metadata into the first IFD of a hyperstack:
//!
//! 1. `ImageDescription` (tag 270): a plain `key=value\n`-separated ASCII
//!    block (`ImageJ=1.x`, `channels=`, `slices=`, `frames=`, `mode=`,
//!    `min=`, `max=`, ...). This is well documented and stable; parsing it
//!    is straightforward.
//!
//! 2. `IJMetadata` / `IJMetadataByteCounts` (tags 50839 / 50838): a binary
//!    blob containing per-channel LUTs, display ranges, slice labels, ROIs,
//!    etc. **This format is not officially documented by ImageJ**, and reading
//!    it caused otherwise-identical files to render differently depending on
//!    inconsistencies in this block. It is therefore **no longer read** — all
//!    display metadata now comes from the `ImageDescription` text above.
//!    Composite-channel colors fall back to a standard cycling palette and
//!    contrast falls back to the file's `min=`/`max=` (or the data's own
//!    min/max). The former best-effort binary parser was removed; see git
//!    history if it ever needs to be revived.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayMode {
    Grayscale,
    Composite,
    Color,
}

#[derive(Clone, Debug)]
pub struct ChannelDisplay {
    /// 256-entry RGB lookup table. Defaults to identity grayscale, or a
    /// cycling set of standard ImageJ channel colors for composite stacks
    /// when no explicit LUT could be parsed.
    pub lut: [[u8; 3]; 256],
    /// Display window (min, max) in raw sample units. `None` means
    /// "compute auto-contrast from the data" — the caller decides how.
    pub range: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct StackMeta {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub mode: DisplayMode,
    pub unit: Option<String>,
    pub frame_interval_s: Option<f64>,
    pub channel_display: Vec<ChannelDisplay>,
    /// Linear pixel calibration `(c0, c1)` from ImageJ's `c0=`/`c1=` (with a
    /// straight-line `cf=0`, or no `cf=`): a raw sample `r` represents the
    /// real value `c0 + c1 * r`. `None` when the file carries no usable linear
    /// calibration, in which case raw sample values are shown directly.
    /// Non-linear calibration functions (`cf` > 0) are not supported and are
    /// treated as uncalibrated.
    pub calibration: Option<(f64, f64)>,
    /// Playback rate in frames/second from ImageJ's `fps=`. `None` when the
    /// file doesn't specify one — the viewer falls back to a default.
    pub fps: Option<f64>,
}

impl StackMeta {
    /// Maps a raw sample value to its calibrated real value (`c0 + c1 * raw`),
    /// or returns it unchanged when the file has no linear calibration.
    pub fn calibrate(&self, raw: f64) -> f64 {
        match self.calibration {
            Some((c0, c1)) => c0 + c1 * raw,
            None => raw,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedDimensions {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub triple_axis_warning: bool,
}

/// A dimension this size or smaller is assumed to be channels; anything
/// larger is assumed to be time. Hardcoded rather than configurable: there's
/// no size that's correct for every dataset, but a per-file manual
/// channels/frames swap (exposed in the UI) covers the cases this misses,
/// without the complexity of a global adjustable setting.
const CHANNEL_SIZE_CUTOFF: usize = 4;

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

#[cfg(test)]
mod dimension_tests {
    use super::*;

    fn check(c: usize, z: usize, f: usize, expected: ResolvedDimensions, label: &str) {
        let got = resolve_dimensions(c, z, f);
        assert_eq!(got, expected, "{label}: resolve_dimensions({c}, {z}, {f})");
        // Invariant: reclassifying axes must never invent or drop planes.
        assert_eq!(
            got.channels * got.slices * got.frames,
            c * z * f,
            "{label}: product changed"
        );
    }

    #[test]
    fn fixes_mislabeled_large_channel_count() {
        // The actual reported bug: metadata says channels=100 (z=1, f=1).
        check(
            100,
            1,
            1,
            ResolvedDimensions { channels: 1, slices: 1, frames: 100, triple_axis_warning: false },
            "mislabeled channels=100",
        );
        check(
            100,
            1,
            7,
            ResolvedDimensions { channels: 1, slices: 1, frames: 700, triple_axis_warning: false },
            "mislabeled channels=100, real frames also present",
        );
    }

    #[test]
    fn leaves_normal_stacks_untouched() {
        check(
            2,
            1,
            350,
            ResolvedDimensions { channels: 2, slices: 1, frames: 350, triple_axis_warning: false },
            "normal 2-channel timelapse",
        );
        check(
            1,
            1,
            500,
            ResolvedDimensions { channels: 1, slices: 1, frames: 500, triple_axis_warning: false },
            "normal single-channel timelapse",
        );
        check(
            1,
            1,
            1,
            ResolvedDimensions { channels: 1, slices: 1, frames: 1, triple_axis_warning: false },
            "single image",
        );
    }

    #[test]
    fn folds_z_into_time() {
        check(
            1,
            50,
            1,
            ResolvedDimensions { channels: 1, slices: 1, frames: 50, triple_axis_warning: false },
            "pure z-stack becomes a 50-frame series",
        );
        check(
            2,
            3,
            1,
            ResolvedDimensions { channels: 2, slices: 1, frames: 3, triple_axis_warning: false },
            "2-channel z-stack: z folds into frames",
        );
    }

    #[test]
    fn detects_swapped_channel_and_time_roles() {
        // Small value mislabeled as frames, large value mislabeled as channels.
        check(
            500,
            1,
            2,
            ResolvedDimensions { channels: 2, slices: 1, frames: 500, triple_axis_warning: false },
            "swapped roles recovered",
        );
    }

    #[test]
    fn warns_on_genuine_triple_axis_stack() {
        check(
            3,
            10,
            20,
            ResolvedDimensions { channels: 3, slices: 10, frames: 20, triple_axis_warning: true },
            "channels + Z + time all present",
        );
    }

    #[test]
    fn channel_size_boundary_is_inclusive_at_cutoff() {
        check(
            4,
            1,
            100,
            ResolvedDimensions { channels: 4, slices: 1, frames: 100, triple_axis_warning: false },
            "exactly at the cutoff (4) counts as channel-sized",
        );
        check(
            5,
            1,
            100,
            ResolvedDimensions { channels: 1, slices: 1, frames: 500, triple_axis_warning: false },
            "one past the cutoff does not count as channel-sized",
        );
    }
}

pub fn grayscale_lut() -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = [i as u8, i as u8, i as u8];
    }
    lut
}

/// Standard ImageJ composite-mode channel color cycle (channel LUTs assigned
/// when a hyperstack is opened without explicit per-channel LUTs).
pub fn default_composite_lut(channel_index: usize) -> [[u8; 3]; 256] {
    let colors: [[u8; 3]; 7] = [
        [255, 0, 0],   // red
        [0, 255, 0],   // green
        [0, 0, 255],   // blue
        [255, 255, 0], // yellow
        [255, 0, 255], // magenta
        [0, 255, 255], // cyan
        [255, 255, 255], // gray/white
    ];
    let base = colors[channel_index % colors.len()];
    let mut lut = [[0u8; 3]; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let t = i as f32 / 255.0;
        *entry = [
            (base[0] as f32 * t) as u8,
            (base[1] as f32 * t) as u8,
            (base[2] as f32 * t) as u8,
        ];
    }
    lut
}

/// The default LUT for channel `c` under display `mode`: a flat grayscale
/// ramp in grayscale mode, otherwise the cycling composite-color palette.
/// The single source of truth so `build_stack_meta` and `resize_channel_display`
/// (in the app) can't drift — a grayscale stack stays grayscale even after a
/// channel-count change.
pub fn default_lut_for(mode: DisplayMode, c: usize) -> [[u8; 3]; 256] {
    match mode {
        DisplayMode::Grayscale => grayscale_lut(),
        DisplayMode::Composite | DisplayMode::Color => default_composite_lut(c),
    }
}

/// Parse the `key=value` lines of an ImageDescription tag.
fn parse_description(desc: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in desc.lines() {
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    map
}

pub fn build_stack_meta(description: Option<&str>, total_ifds: usize) -> StackMeta {
    // Only parse ImageDescription as ImageJ's key=value format if it
    // actually carries ImageJ's own signature. A TIFF written by something
    // else entirely (a different microscopy package, a generic image
    // tool, ...) can have arbitrary free-form text here, and that text
    // might coincidentally contain lines that look like "channels=" or
    // "min=" — silently treating those as real values would be worse than
    // not parsing anything. With no usable description, everything below
    // falls back cleanly: dimensions come from `resolve_dimensions`'s
    // size-based guess (see app.rs), and contrast falls back to the
    // image's own min/max.
    let kv = description
        .filter(|d| d.contains("ImageJ="))
        .map(parse_description)
        .unwrap_or_default();

    let get_usize = |key: &str| kv.get(key).and_then(|s| s.parse::<usize>().ok());
    let get_f64 = |key: &str| kv.get(key).and_then(|s| s.parse::<f64>().ok());

    let channels = get_usize("channels").unwrap_or(1).max(1);
    let slices = get_usize("slices").unwrap_or(1).max(1);
    // `frames=` may be absent for a plain single-channel time series saved
    // without hyperstack dimension tags; fall back to inferring it from the
    // total IFD count so a bare "N-page TIFF" still scrubs correctly.
    let frames = get_usize("frames").unwrap_or_else(|| {
        (total_ifds / (channels * slices)).max(1)
    });

    // Default to grayscale when no `mode=` is present. A missing mode used to
    // be inferred as composite whenever channels>1, but a mislabeled
    // `channels=N` (really a frame count — see `resolve_dimensions`) would
    // then wrongly tint the image. With no explicit mode the safe assumption
    // is plain grayscale; composite/color colors only apply when the file
    // actually says so.
    let mode = match kv.get("mode").map(|s| s.as_str()) {
        Some("composite") => DisplayMode::Composite,
        Some("color") => DisplayMode::Color,
        _ => DisplayMode::Grayscale,
    };

    let global_range = match (get_f64("min"), get_f64("max")) {
        (Some(lo), Some(hi)) if hi > lo => Some((lo, hi)),
        _ => None,
    };

    // Linear calibration only: `cf=0` is ImageJ's straight-line function. A
    // missing `cf=` (older files that just wrote c0/c1) is treated as linear
    // too; any other `cf` is a non-linear function we don't model, so we skip
    // calibration entirely rather than apply the wrong transform.
    let cf_linear = kv.get("cf").map(|s| s.trim() == "0").unwrap_or(true);
    let calibration = match (get_f64("c0"), get_f64("c1")) {
        (Some(c0), Some(c1)) if cf_linear && c1 != 0.0 => Some((c0, c1)),
        _ => None,
    };

    let fps = get_f64("fps").filter(|f| *f > 0.0);

    let channel_display: Vec<ChannelDisplay> = (0..channels)
        .map(|c| ChannelDisplay {
            lut: default_lut_for(mode, c),
            range: global_range,
        })
        .collect();

    StackMeta {
        channels,
        slices,
        frames,
        mode,
        unit: kv.get("unit").cloned(),
        frame_interval_s: get_f64("finterval"),
        channel_display,
        calibration,
        fps,
    }
}

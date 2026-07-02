//! ImageJ writes two kinds of metadata into the first IFD of a hyperstack:
//!
//! 1. `ImageDescription` (tag 270): a plain `key=value\n`-separated ASCII
//!    block (`ImageJ=1.x`, `channels=`, `slices=`, `frames=`, `mode=`,
//!    `min=`, `max=`, ...). This is well documented and stable; parsing it
//!    is straightforward.
//!
//! 2. `IJMetadata` / `IJMetadataByteCounts` (tags 50839 / 50838): a binary
//!    blob containing per-channel LUTs, display ranges, slice labels, ROIs,
//!    etc. **This format is not officially documented by ImageJ** — the parser
//!    below is reconstructed from known open-source readers and is best-effort.
//!    It is read only as a **supplementary fallback**: it fills in display
//!    information the `ImageDescription` text didn't provide (a per-channel
//!    display range when there's no `min=`/`max=`, and per-channel composite
//!    LUTs, which `ImageDescription` never carries). It never overrides a value
//!    that tag 270 already specified — so it can't make two files with the same
//!    `ImageDescription` render differently in any axis that text controls.
//!    Every step is defensive and requires an exact channel-count match before
//!    trusting the block, falling back to defaults on any inconsistency.

use crate::ifd::ByteOrder;
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
    /// Z-step between slices (in `unit`s) from ImageJ's `spacing=`.
    pub spacing: Option<f64>,
    /// Whether playback should loop, from ImageJ's `loop=`.
    pub loop_playback: Option<bool>,
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

#[cfg(test)]
#[path = "ij_metadata_tests.rs"]
mod dimension_tests;

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

pub fn build_stack_meta(
    description: Option<&str>,
    ij_metadata: Option<&[u8]>,
    ij_metadata_counts: Option<&[u32]>,
    total_ifds: usize,
) -> StackMeta {
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

    let mut channel_display: Vec<ChannelDisplay> = (0..channels)
        .map(|c| ChannelDisplay {
            lut: default_lut_for(mode, c),
            range: global_range,
        })
        .collect();

    // Supplement (never override) the above from the binary IJMetadata block,
    // filling only what ImageDescription didn't provide. A correctly-formed
    // block has exactly one range-pair and one LUT per channel; we require that
    // exact count match before trusting it (a mismatched block is stale — e.g.
    // left over from before the file was reduced to fewer channels — so it's
    // ignored entirely rather than misattributed).
    if let (Some(data), Some(counts)) = (ij_metadata, ij_metadata_counts) {
        let blocks = try_parse_ij_blocks(data, counts, ByteOrder::Big)
            .or_else(|| try_parse_ij_blocks(data, counts, ByteOrder::Little));
        if let Some(blocks) = blocks {
            // Display range: only as a fallback when ImageDescription gave no
            // `min=`/`max=` window at all.
            if global_range.is_none() {
                if let Some(ranges) = &blocks.ranges {
                    if ranges.len() == channels {
                        for (disp, &r) in channel_display.iter_mut().zip(ranges.iter()) {
                            if r.1 > r.0 && r.0.is_finite() && r.1.is_finite() {
                                disp.range = Some(r);
                            }
                        }
                    }
                }
            }
            // Per-channel LUTs: ImageDescription only carries `mode=`, never the
            // actual colors, so a count-matched LUT block is the genuine source
            // of custom composite colors. Skip it for grayscale mode, where the
            // color isn't "missing" — the file explicitly asked for grayscale.
            if mode != DisplayMode::Grayscale && blocks.luts.len() == channels {
                for (disp, lut) in channel_display.iter_mut().zip(blocks.luts.iter()) {
                    disp.lut = *lut;
                }
            }
        }
    }

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
        spacing: get_f64("spacing"),
        loop_playback: kv.get("loop").map(|s| s == "true"),
    }
}

struct IjBlocks {
    ranges: Option<Vec<(f64, f64)>>,
    luts: Vec<[[u8; 3]; 256]>,
}

/// Best-effort parse of the IJMetadata directory format. Returns `None` on
/// any structural inconsistency rather than guessing — callers fall back to
/// defaults in that case. See module docs for the honesty caveat here.
fn try_parse_ij_blocks(data: &[u8], byte_counts: &[u32], header_order: ByteOrder) -> Option<IjBlocks> {
    if byte_counts.is_empty() {
        return None;
    }
    let header_len = *byte_counts.first()? as usize;
    let header = data.get(..header_len)?;

    // Header is a sequence of 8-byte records: 4-byte ASCII type code + a
    // 4-byte count. The on-disk endianness of this internal directory isn't
    // officially documented; we try big-endian first (consistent with this
    // block being written by Java's DataOutputStream) and fall back to
    // little-endian if that doesn't produce a structurally consistent plan.
    if header_len % 8 != 0 {
        return None;
    }
    let mut plan: Vec<([u8; 4], usize)> = Vec::new();
    for chunk in header.chunks_exact(8) {
        let mut code = [0u8; 4];
        code.copy_from_slice(&chunk[0..4]);
        if !code.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            return None; // doesn't look like a real type code; bail out
        }
        let count = header_order.u32(&chunk[4..8]) as usize;
        plan.push((code, count));
    }

    let expanded: usize = plan.iter().map(|(_, n)| n).sum();
    if byte_counts.len() != 1 + expanded {
        return None; // doesn't match our assumed layout
    }

    let mut cursor = header_len;
    let mut block_idx = 1; // byte_counts[0] was the header itself
    let mut ranges: Option<Vec<(f64, f64)>> = None;
    let mut luts: Vec<[[u8; 3]; 256]> = Vec::new();

    for (code, n) in plan {
        for _ in 0..n {
            let len = *byte_counts.get(block_idx)? as usize;
            let block = data.get(cursor..cursor + len)?;
            cursor += len;
            block_idx += 1;

            match &code {
                b"rang" => {
                    // n channel pairs of f64 (min, max), same endianness as the directory.
                    let mut parsed = Vec::new();
                    for pair in block.chunks_exact(16) {
                        let lo = header_order.f64_from(&pair[0..8]);
                        let hi = header_order.f64_from(&pair[8..16]);
                        parsed.push((lo, hi));
                    }
                    if !parsed.is_empty() {
                        ranges = Some(parsed);
                    }
                }
                b"luts" => {
                    // 768 bytes: 256 R, then 256 G, then 256 B (planar, not interleaved).
                    if block.len() == 768 {
                        let mut lut = [[0u8; 3]; 256];
                        for i in 0..256 {
                            lut[i] = [block[i], block[256 + i], block[512 + i]];
                        }
                        luts.push(lut);
                    }
                }
                _ => { /* info / labl / roi / over / plot: not needed for rendering */ }
            }
        }
    }

    if ranges.is_none() && luts.is_empty() {
        return None;
    }
    if cursor != data.len() {
        // A correct parse should land exactly on the end of the buffer.
        // Landing short or overrunning means the directory was
        // misinterpreted somewhere — possibly correctly counting total
        // bytes by coincidence while misattributing which bytes belong to
        // which block type. Don't trust a result we can't fully account for.
        return None;
    }
    Some(IjBlocks { ranges, luts })
}

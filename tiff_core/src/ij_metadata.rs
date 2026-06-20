//! ImageJ writes two kinds of metadata into the first IFD of a hyperstack:
//!
//! 1. `ImageDescription` (tag 270): a plain `key=value\n`-separated ASCII
//!    block (`ImageJ=1.x`, `channels=`, `slices=`, `frames=`, `mode=`,
//!    `min=`, `max=`, ...). This is well documented and stable; parsing it
//!    is straightforward.
//!
//! 2. `IJMetadata` / `IJMetadataByteCounts` (tags 50839 / 50838): a binary
//!    blob containing per-channel LUTs, display ranges, slice labels, ROIs,
//!    etc. **This format is not officially documented by ImageJ** — the
//!    parser below is reconstructed from known open-source reader
//!    implementations (e.g. tifffile's `imagej_metadata`). It's best-effort:
//!    every step is defensive and falls back to sane defaults (grayscale,
//!    auto-contrast) rather than producing garbage if a layout assumption
//!    turns out to be wrong for a particular file. If composite-mode colors
//!    look off, this is the module to revisit.

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
    /// True if IJMetadata LUT/range parsing succeeded for at least one
    /// channel. If false, everything in `channel_display` is a default.
    pub ij_metadata_parsed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedDimensions {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub triple_axis_warning: bool,
}

/// Decides the *effective* (channels, slices, frames) to use, classifying
/// by size rather than trusting the file's own axis labels: a dimension of
/// `channel_size_cutoff` or fewer is assumed to be channels, anything
/// larger is assumed to be time — this holds even when the file's
/// metadata claims the opposite (a "channels" value that's actually a
/// mislabeled frame count, or vice versa), or is missing entirely.
///
/// `channel_size_cutoff` is caller-supplied rather than fixed because a
/// real acquisition can genuinely have more than a handful of channels
/// (spectral imaging, for instance) — there's no size threshold that's
/// correct for every dataset, so the caller (the UI) exposes it as an
/// adjustable setting instead of hardcoding a guess.
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
pub fn resolve_dimensions(c: usize, z: usize, f: usize, channel_size_cutoff: usize) -> ResolvedDimensions {
    if c > 1 && z > 1 && f > 1 {
        return ResolvedDimensions {
            channels: c,
            slices: z,
            frames: f,
            triple_axis_warning: true,
        };
    }

    let time_only = (z * f).max(1); // Z always folds into time, unconditionally
    let c_is_channel_sized = c > 1 && c <= channel_size_cutoff;
    let time_is_channel_sized = time_only > 1 && time_only <= channel_size_cutoff;

    let (channels, frames) = if c_is_channel_sized {
        (c, time_only)
    } else if c > channel_size_cutoff && time_is_channel_sized {
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

    const DEFAULT_CUTOFF: usize = 4;

    fn check(c: usize, z: usize, f: usize, cutoff: usize, expected: ResolvedDimensions, label: &str) {
        let got = resolve_dimensions(c, z, f, cutoff);
        assert_eq!(got, expected, "{label}: resolve_dimensions({c}, {z}, {f}, cutoff={cutoff})");
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
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 1, slices: 1, frames: 100, triple_axis_warning: false },
            "mislabeled channels=100",
        );
        check(
            100,
            1,
            7,
            DEFAULT_CUTOFF,
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
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 2, slices: 1, frames: 350, triple_axis_warning: false },
            "normal 2-channel timelapse",
        );
        check(
            1,
            1,
            500,
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 1, slices: 1, frames: 500, triple_axis_warning: false },
            "normal single-channel timelapse",
        );
        check(
            1,
            1,
            1,
            DEFAULT_CUTOFF,
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
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 1, slices: 1, frames: 50, triple_axis_warning: false },
            "pure z-stack becomes a 50-frame series",
        );
        check(
            2,
            3,
            1,
            DEFAULT_CUTOFF,
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
            DEFAULT_CUTOFF,
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
            DEFAULT_CUTOFF,
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
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 4, slices: 1, frames: 100, triple_axis_warning: false },
            "exactly at the default cutoff (4) counts as channel-sized",
        );
        check(
            5,
            1,
            100,
            DEFAULT_CUTOFF,
            ResolvedDimensions { channels: 1, slices: 1, frames: 500, triple_axis_warning: false },
            "one past the default cutoff does not count as channel-sized",
        );
    }

    #[test]
    fn respects_custom_channel_size_cutoff() {
        // A real 6-channel acquisition: the default cutoff of 4 misreads
        // it as a 6-frame time series, but raising the cutoff to 8 (e.g.
        // for spectral imaging with more channels than the default
        // assumes) correctly recovers it as 6 channels.
        check(
            6,
            1,
            100,
            4,
            ResolvedDimensions { channels: 1, slices: 1, frames: 600, triple_axis_warning: false },
            "6 channels misread at the default cutoff of 4",
        );
        check(
            6,
            1,
            100,
            8,
            ResolvedDimensions { channels: 6, slices: 1, frames: 100, triple_axis_warning: false },
            "6 channels correctly recognized at cutoff=8",
        );
        // Lowering the cutoff can also flip a value that used to qualify:
        // 3 channels is channel-sized at the default cutoff of 4, but not
        // once the cutoff is dropped to 2.
        check(
            3,
            1,
            100,
            4,
            ResolvedDimensions { channels: 3, slices: 1, frames: 100, triple_axis_warning: false },
            "3 channels qualifies at the default cutoff",
        );
        check(
            3,
            1,
            100,
            2,
            ResolvedDimensions { channels: 1, slices: 1, frames: 300, triple_axis_warning: false },
            "3 channels no longer qualifies once cutoff is lowered to 2",
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
    let kv = description.map(parse_description).unwrap_or_default();

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

    let mode = match kv.get("mode").map(|s| s.as_str()) {
        Some("composite") => DisplayMode::Composite,
        Some("color") => DisplayMode::Color,
        _ if channels > 1 => DisplayMode::Composite,
        _ => DisplayMode::Grayscale,
    };

    let global_range = match (get_f64("min"), get_f64("max")) {
        (Some(lo), Some(hi)) if hi > lo => Some((lo, hi)),
        _ => None,
    };

    let mut channel_display: Vec<ChannelDisplay> = (0..channels)
        .map(|c| ChannelDisplay {
            lut: if channels == 1 {
                grayscale_lut()
            } else {
                default_composite_lut(c)
            },
            range: global_range,
        })
        .collect();

    let mut ij_metadata_parsed = false;
    if let (Some(data), Some(counts)) = (ij_metadata, ij_metadata_counts) {
        let blocks = try_parse_ij_blocks(data, counts, ByteOrder::Big)
            .or_else(|| try_parse_ij_blocks(data, counts, ByteOrder::Little));
        if let Some(blocks) = blocks {
            if let Some(ranges) = &blocks.ranges {
                for (c, disp) in channel_display.iter_mut().enumerate() {
                    if let Some(&r) = ranges.get(c) {
                        if r.1 > r.0 && r.0.is_finite() && r.1.is_finite() {
                            disp.range = Some(r);
                            ij_metadata_parsed = true;
                        }
                    }
                }
            }
            for (c, disp) in channel_display.iter_mut().enumerate() {
                if let Some(lut) = blocks.luts.get(c) {
                    disp.lut = *lut;
                    ij_metadata_parsed = true;
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
        ij_metadata_parsed,
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
    Some(IjBlocks { ranges, luts })
}
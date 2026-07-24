//! The ImageJ metadata dialect. ImageJ writes two things into the first IFD of
//! a hyperstack:
//!
//! 1. `ImageDescription` (tag 270): a plain `key=value\n`-separated ASCII block
//!    (`ImageJ=1.x`, `channels=`, `slices=`, `frames=`, `mode=`, `min=`,
//!    `max=`, ...). Well documented and stable; parsing it is straightforward.
//!
//! 2. `IJMetadata` / `IJMetadataByteCounts` (tags 50839 / 50838): a binary blob
//!    of per-channel LUTs, display ranges, slice labels, ROIs, etc. **This
//!    format is not officially documented by ImageJ** — the parser below is
//!    reconstructed from known open-source readers and is best-effort. It fills
//!    in display info the `ImageDescription` text can't carry: a per-channel
//!    display range when there's no `min=`/`max=`, and the actual per-channel
//!    LUTs. A count-matched *colored* LUT block is the genuine source of channel
//!    colors and takes priority over the mode-derived default (including over
//!    `mode=grayscale`); it never overrides the numeric values tag 270
//!    specifies. Every step is defensive and requires an exact channel-count
//!    match before trusting the block.
//!
//! The write side ([`serialize`]) emits only the `key=value` description block,
//! never the binary LUT block (ranges go in the description; colors follow the
//! display mode).

use super::{
    default_lut_for, resolution_to_pixel, ChannelDisplay, DisplayMode, MetadataFormat, StackMeta,
    StackMetaWrite,
};
use crate::ifd::ByteOrder;
use anyhow::Result;
use std::collections::HashMap;

/// Decode Java-style `\uXXXX` escapes that ImageJ writes into text fields for
/// non-ASCII values — most commonly `unit=µm`, which should read as `µm`.
/// Invalid escapes are left verbatim.
fn decode_ij_escapes(s: &str) -> String {
    if !s.contains("\\u") {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'u') {
            chars.next(); // consume 'u'
            let hex: String = (0..4).map_while(|_| chars.next_if(|d| d.is_ascii_hexdigit())).collect();
            match u32::from_str_radix(&hex, 16).ok().filter(|_| hex.len() == 4).and_then(char::from_u32) {
                Some(ch) => out.push(ch),
                None => {
                    out.push('\\');
                    out.push('u');
                    out.push_str(&hex);
                }
            }
        } else {
            out.push(c);
        }
    }
    out
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

/// The `images=N` plane count from an ImageJ description, if present. Used by
/// `index` to detect ImageJ's single-IFD contiguous big-stack layout (where
/// the IFD chain has one entry but the file carries N contiguous frames).
pub(crate) fn images_count(description: &str) -> Option<usize> {
    if !description.contains("ImageJ=") {
        return None;
    }
    parse_description(description).get("images").and_then(|s| s.parse().ok())
}

/// Parse ImageJ metadata (the `key=value` description + the optional binary
/// IJMetadata block) into the normalized [`StackMeta`].
pub fn parse(
    description: Option<&str>,
    ij_metadata: Option<&[u8]>,
    ij_metadata_counts: Option<&[u32]>,
    total_ifds: usize,
    x_resolution: Option<f64>,
    y_resolution: Option<f64>,
) -> StackMeta {
    // Only parse ImageDescription as ImageJ's key=value format if it actually
    // carries ImageJ's own signature. A TIFF written by something else can have
    // arbitrary free-form text here that might coincidentally contain lines
    // that look like "channels=" — silently treating those as real values would
    // be worse than not parsing anything.
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
    let frames = get_usize("frames").unwrap_or_else(|| (total_ifds / (channels * slices)).max(1));

    // Default to grayscale when no `mode=` is present. A missing mode used to be
    // inferred as composite whenever channels>1, but a mislabeled `channels=N`
    // (really a frame count — see `resolve_dimensions`) would then wrongly tint
    // the image. Composite/color colors only apply when the file says so.
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
    // too; any other `cf` is a non-linear function we don't model.
    let cf_linear = kv.get("cf").map(|s| s.trim() == "0").unwrap_or(true);
    let calibration = match (get_f64("c0"), get_f64("c1")) {
        (Some(c0), Some(c1)) if cf_linear && c1 != 0.0 => Some((c0, c1)),
        _ => None,
    };

    let fps = get_f64("fps").filter(|f| *f > 0.0);

    let mut channel_display: Vec<ChannelDisplay> = (0..channels)
        .map(|c| ChannelDisplay { lut: default_lut_for(mode, c), range: global_range })
        .collect();

    // Supplement the above from the binary IJMetadata block. A correctly-formed
    // block has exactly one range-pair and one LUT per channel; require that
    // exact count match before trusting it (a mismatched block is stale — e.g.
    // left over from before the file was reduced to fewer channels).
    let mut has_explicit_luts = false;
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
            // Per-channel LUTs: a count-matched block is the genuine source of
            // channel colors and takes priority over the mode-derived default.
            // Only *colored* LUTs count as "explicit": a block of plain grayscale
            // ramps is applied (a visual no-op) but leaves the pseudocolor toggle
            // available, so a truly grayscale stack can still be tinted on demand.
            if blocks.luts.len() == channels {
                let colored = blocks
                    .luts
                    .iter()
                    .any(|lut| lut.iter().any(|px| px[0] != px[1] || px[1] != px[2]));
                for (disp, lut) in channel_display.iter_mut().zip(blocks.luts.iter()) {
                    disp.lut = *lut;
                }
                has_explicit_luts = colored;
            }
        }
    }

    StackMeta {
        channels,
        slices,
        frames,
        mode,
        unit: kv.get("unit").map(|u| decode_ij_escapes(u)),
        frame_interval_s: get_f64("finterval"),
        channel_display,
        calibration,
        fps,
        spacing: get_f64("spacing"),
        loop_playback: kv.get("loop").map(|s| s == "true"),
        pixel_width: resolution_to_pixel(x_resolution),
        pixel_height: resolution_to_pixel(y_resolution),
        has_explicit_luts,
        source_format: MetadataFormat::ImageJ,
    }
}

/// Serialize `meta` as an ImageJ `key=value` ImageDescription block. `planes`
/// is the total plane count (the derived time-frame count is `planes /
/// (channels * slices)`, which must divide evenly). Emits the description text
/// only — never the binary LUT block.
pub(crate) fn serialize(planes: usize, meta: &StackMetaWrite) -> Result<String> {
    let frames = meta.derive_frames(planes)?;

    let mut s = String::from("ImageJ=1.54f\n");
    s += &format!("images={planes}\n");
    if meta.channels > 1 {
        s += &format!("channels={}\n", meta.channels);
    }
    if meta.slices > 1 {
        s += &format!("slices={}\n", meta.slices);
    }
    if frames > 1 {
        s += &format!("frames={frames}\n");
    }
    if [meta.channels > 1, meta.slices > 1, frames > 1].iter().filter(|&&b| b).count() >= 2 {
        s += "hyperstack=true\n";
    }
    if meta.channels > 1 || meta.mode != DisplayMode::Grayscale {
        let mode = match meta.mode {
            DisplayMode::Grayscale => "grayscale",
            DisplayMode::Composite => "composite",
            DisplayMode::Color => "color",
        };
        s += &format!("mode={mode}\n");
    }
    if let Some(unit) = &meta.unit {
        s += &format!("unit={unit}\n");
    }
    if let Some(fi) = meta.frame_interval_s {
        s += &format!("finterval={fi}\n");
    }
    if let Some(fps) = meta.fps {
        s += &format!("fps={fps}\n");
    }
    if let Some(spacing) = meta.spacing {
        s += &format!("spacing={spacing}\n");
    }
    if let Some(looped) = meta.loop_playback {
        s += &format!("loop={looped}\n");
    }
    if let Some((lo, hi)) = meta.range {
        s += &format!("min={lo}\nmax={hi}\n");
    }
    if let Some((c0, c1)) = meta.calibration {
        s += &format!("cf=0\nc0={c0}\nc1={c1}\n");
    }
    for (key, value) in &meta.extra {
        s += &format!("{key}={value}\n");
    }
    Ok(s)
}

struct IjBlocks {
    ranges: Option<Vec<(f64, f64)>>,
    luts: Vec<[[u8; 3]; 256]>,
}

/// Best-effort parse of the IJMetadata directory format. Returns `None` on any
/// structural inconsistency rather than guessing — callers fall back to
/// defaults in that case. See module docs for the honesty caveat here.
fn try_parse_ij_blocks(data: &[u8], byte_counts: &[u32], header_order: ByteOrder) -> Option<IjBlocks> {
    if byte_counts.is_empty() {
        return None;
    }
    let header_len = *byte_counts.first()? as usize;
    let header = data.get(..header_len)?;

    // Header is a sequence of 8-byte records: 4-byte ASCII type code + a 4-byte
    // count. The on-disk endianness of this internal directory isn't officially
    // documented; try big-endian first (consistent with Java's
    // DataOutputStream) and fall back to little-endian.
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
        // Landing short or overrunning means the directory was misinterpreted.
        return None;
    }
    Some(IjBlocks { ranges, luts })
}

#[cfg(test)]
#[path = "imagej_tests.rs"]
mod tests;

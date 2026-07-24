//! The OME-TIFF metadata dialect. An OME-TIFF stores an OME-XML document in the
//! first IFD's `ImageDescription` (tag 270) describing the pixel geometry and
//! channels. This module reads/writes a **minimal but useful subset** — the
//! `Pixels` core and per-`Channel` name/color — which is everything the
//! normalized [`StackMeta`] needs; ROIs, instruments, and per-plane
//! position/timestamp records are ignored on read and not written.
//!
//! Parsing uses `quick-xml` (namespaces and entity escaping make OME-XML
//! genuinely fiddly to hand-roll); the writer emits a fixed minimal template.

use super::{
    color_ramp_lut, composite_color, default_lut_for, resolution_to_pixel, ChannelDisplay,
    ChannelWrite, DisplayMode, MetadataFormat, StackMeta, StackMetaWrite, WriteGeometry,
};
use anyhow::Result;
use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// OME's default physical-size unit (the schema default when `*Unit` is absent).
const DEFAULT_UNIT: &str = "µm";

/// Parse an OME-XML `ImageDescription` into the normalized [`StackMeta`].
/// Returns `None` when the XML is malformed or carries no `<Pixels>` element,
/// so the dispatcher can fall back to neutral inferred metadata rather than
/// failing the whole file open. `x/y_resolution` (TIFF resolution tags) are a
/// fallback for pixel size when OME's `PhysicalSize` is absent.
pub fn parse(
    xml: &str,
    total_ifds: usize,
    x_resolution: Option<f64>,
    y_resolution: Option<f64>,
) -> Option<StackMeta> {
    let (pixels, channels) = read_pixels(xml)?;

    let get = |k: &str| pixels.get(k);
    let get_usize = |k: &str| get(k).and_then(|v| v.parse::<usize>().ok());
    let get_f64 = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());

    let size_c = get_usize("SizeC").unwrap_or(1).max(1);
    let size_z = get_usize("SizeZ").unwrap_or(1).max(1);
    let size_t = get_usize("SizeT").unwrap_or_else(|| (total_ifds / (size_c * size_z)).max(1)).max(1);

    // Per-channel colors: OME `Color` is a signed int32 RGBA (R in the high
    // byte). A non-gray color makes the stack composite.
    let colors: Vec<Option<[u8; 3]>> = channels.iter().map(|a| a.get("Color").and_then(|s| parse_color(s))).collect();
    let any_colored = colors.iter().flatten().any(|c| c[0] != c[1] || c[1] != c[2]);
    let mode = if any_colored { DisplayMode::Composite } else { DisplayMode::Grayscale };

    let channel_display: Vec<ChannelDisplay> = (0..size_c)
        .map(|c| {
            let lut = colors.get(c).copied().flatten().map(color_ramp_lut).unwrap_or_else(|| default_lut_for(mode, c));
            ChannelDisplay { lut, range: None } // OME core carries no display window
        })
        .collect();

    // Physical sizes: OME's own field wins; the TIFF resolution tags are the
    // fallback for a file that omitted PhysicalSize.
    let pixel_width = get_f64("PhysicalSizeX").or_else(|| resolution_to_pixel(x_resolution));
    let pixel_height = get_f64("PhysicalSizeY").or_else(|| resolution_to_pixel(y_resolution));
    let unit = get("PhysicalSizeXUnit").cloned();

    Some(StackMeta {
        channels: size_c,
        slices: size_z,
        frames: size_t,
        mode,
        unit,
        frame_interval_s: get_f64("TimeIncrement"),
        channel_display,
        calibration: None, // OME expresses calibration through PhysicalSize, not a linear map
        fps: None,
        spacing: get_f64("PhysicalSizeZ"),
        loop_playback: None,
        pixel_width,
        pixel_height,
        has_explicit_luts: any_colored,
        source_format: MetadataFormat::Ome,
    })
}

/// Pull the first `<Pixels>` element's attributes and its `<Channel>` children's
/// attributes out of an OME-XML document. `None` if no `<Pixels>` is found or
/// the XML doesn't parse.
fn read_pixels(xml: &str) -> Option<(HashMap<String, String>, Vec<HashMap<String, String>>)> {
    let mut reader = Reader::from_str(xml);
    let mut pixels: Option<HashMap<String, String>> = None;
    let mut channels: Vec<HashMap<String, String>> = Vec::new();
    let mut in_pixels = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => match e.local_name().as_ref() {
                b"Pixels" => {
                    // Only the first Image's Pixels (the viewer treats the file
                    // as one stack); ignore any later ones.
                    if pixels.is_none() {
                        pixels = Some(attrs_map(&e));
                        in_pixels = true;
                    }
                }
                b"Channel" if in_pixels => channels.push(attrs_map(&e)),
                _ => {}
            },
            Ok(Event::End(e)) if e.local_name().as_ref() == b"Pixels" => in_pixels = false,
            Ok(Event::Eof) => break,
            Err(_) => return None,
            _ => {}
        }
    }
    pixels.map(|p| (p, channels))
}

/// An element's attributes as a `local-name -> value` map (namespace prefixes
/// dropped, entity escapes decoded).
fn attrs_map(e: &BytesStart) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for attr in e.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.local_name().as_ref()).into_owned();
        if let Ok(value) = attr.unescape_value() {
            m.insert(key, value.into_owned());
        }
    }
    m
}

/// Decode an OME `Color` (signed int32 RGBA, R in the high byte) to RGB.
fn parse_color(s: &str) -> Option<[u8; 3]> {
    let signed: i64 = s.trim().parse().ok()?;
    let bits = signed as u32; // wrap the 32-bit RGBA back out
    Some([(bits >> 24) as u8, (bits >> 16) as u8, (bits >> 8) as u8])
}

/// Encode RGB as an OME `Color` (signed int32 RGBA with full alpha).
fn ome_color(rgb: [u8; 3]) -> i32 {
    let bits = (rgb[0] as u32) << 24 | (rgb[1] as u32) << 16 | (rgb[2] as u32) << 8 | 0xff;
    bits as i32
}

/// Serialize `meta` as a minimal OME-XML `ImageDescription`. `planes` gives the
/// derived `SizeT`; `geom` supplies frame size + pixel type for `<Pixels>`.
pub(crate) fn serialize(planes: usize, meta: &StackMetaWrite, geom: &WriteGeometry) -> Result<String> {
    let size_t = meta.derive_frames(planes)?;
    let size_c = meta.channels;
    let size_z = meta.slices;

    // Samples per OME channel: an RGB frame is one OME channel carrying all the
    // samples; a grayscale multichannel stack is one sample per channel.
    let channel_spp = if size_c <= 1 { geom.samples_per_pixel } else { (geom.samples_per_pixel / size_c).max(1) };

    let mut s = String::new();
    s += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    s += &format!(
        "<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\" \
         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" \
         xsi:schemaLocation=\"http://www.openmicroscopy.org/Schemas/OME/2016-06 \
         http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd\" UUID=\"urn:uuid:{}\">\n",
        generate_uuid()
    );
    s += "  <Image ID=\"Image:0\">\n";
    s += &format!(
        "    <Pixels ID=\"Pixels:0\" DimensionOrder=\"XYCZT\" Type=\"{}\" \
         SizeX=\"{}\" SizeY=\"{}\" SizeC=\"{}\" SizeZ=\"{}\" SizeT=\"{}\"",
        geom.ome_pixel_type, geom.width, geom.height, size_c, size_z, size_t
    );

    let unit = meta.unit.as_deref().unwrap_or(DEFAULT_UNIT);
    if let Some(pw) = meta.pixel_width {
        s += &format!(" PhysicalSizeX=\"{pw}\" PhysicalSizeXUnit=\"{}\"", xml_escape(unit));
    }
    if let Some(ph) = meta.pixel_height {
        s += &format!(" PhysicalSizeY=\"{ph}\" PhysicalSizeYUnit=\"{}\"", xml_escape(unit));
    }
    if let Some(pz) = meta.spacing {
        s += &format!(" PhysicalSizeZ=\"{pz}\" PhysicalSizeZUnit=\"{}\"", xml_escape(unit));
    }
    if let Some(ti) = meta.frame_interval_s {
        s += &format!(" TimeIncrement=\"{ti}\" TimeIncrementUnit=\"s\"");
    }
    s += ">\n";

    let colored = meta.mode != DisplayMode::Grayscale;
    for c in 0..size_c {
        let info: Option<&ChannelWrite> = meta.channel_info.get(c);
        s += &format!("      <Channel ID=\"Channel:0:{c}\" SamplesPerPixel=\"{channel_spp}\"");
        if let Some(name) = info.and_then(|i| i.name.as_deref()) {
            s += &format!(" Name=\"{}\"", xml_escape(name));
        }
        // Explicit per-channel color, else the composite palette when the stack
        // is colored; grayscale stacks write no Color.
        let color = info.and_then(|i| i.color).or_else(|| colored.then(|| composite_color(c)));
        if let Some(rgb) = color {
            s += &format!(" Color=\"{}\"", ome_color(rgb));
        }
        s += "/>\n";
    }
    s += "      <TiffData/>\n";
    s += "    </Pixels>\n";
    s += "  </Image>\n";
    s += "</OME>\n";
    Ok(s)
}

/// Escape the five XML metacharacters for use inside a double-quoted attribute.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// A v4-shaped UUID string, uniqueness sourced from the clock plus a
/// process-lifetime counter — enough for OME's `UUID` attribute (readers don't
/// enforce global uniqueness), without pulling in a `uuid` dependency.
fn generate_uuid() -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ COUNTER.fetch_add(1, Ordering::Relaxed).rotate_left(32);

    // Two hasher draws → 128 bits.
    let mut h = RandomState::new().build_hasher();
    h.write_u64(seed);
    let hi = h.finish();
    h.write_u64(!seed);
    let lo = h.finish();
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&hi.to_be_bytes());
    b[8..].copy_from_slice(&lo.to_be_bytes());
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant 1
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

#[cfg(test)]
#[path = "ome_tests.rs"]
mod tests;

//! Olympus/Evident OIR.
//!
//! OIR is proprietary and has no published specification. What follows was
//! determined from a real FluoView file and **checked against an oracle**: the
//! same acquisition exported to TIFF by the acquisition software itself. Frame
//! zero reassembled from the container is byte-identical to the exported one,
//! which is the only kind of evidence worth having about a format nobody
//! documents.
//!
//! # The container
//!
//! ```text
//!   0x00  "OLYMPUSRAWFORMAT"
//!   0x20  u64  total file size
//!   0x28  u64  offset of the block index
//!   0x48  "FLUOVIEW"
//!   ...   blocks
//!   index: u32 0xFFFFFFFF, u32, u32, then one u64 file offset per block
//! ```
//!
//! Every block is `u32 length, u32 type, length bytes`. They come in pairs: a
//! *descriptor* (type 3) naming a plane and the byte range of it that follows,
//! then the block carrying those bytes. A 512x512 16-bit plane arrives as 34
//! chunks of 15,360 bytes and one of 2,048 — so a plane is scattered, and
//! reassembling it is the whole job.
//!
//! Descriptor payload: `u32 offset-within-plane, u32 length, u32 name length,
//! name`. Names look like `t001_0_1_<uid>` — the timepoint, then two more axis
//! indices.
//!
//! # What it does not do
//!
//! A multi-file acquisition (`name.oir` plus `name_00001.oir`) is not stitched;
//! only the part named is read. Every plane is held in memory, because that is
//! what the importer contract asks for — fine for the gigabyte-scale files this
//! was built against, not for a forty-gigabyte timelapse.

use fasttiff_plugin_api::{
    Confidence, FileType, ImageResult, ImportHost, ImportRequest, ImportResult, Importer,
    PixelType, PlaneData, PluginError, PluginInfo, Spacing, StackInfo,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// The signature every OIR file starts with.
const MAGIC: &[u8] = b"OLYMPUSRAWFORMAT";
/// Where the header states the block index begins.
const INDEX_OFFSET_AT: u64 = 0x28;
/// The index's own leading marker, checked before anything is read from it.
const INDEX_MARKER: u32 = 0xFFFF_FFFF;
/// Bytes of index header before the offsets begin.
const INDEX_HEADER: usize = 12;
/// A descriptor block, naming the data block that follows it.
const TYPE_DESCRIPTOR: u32 = 3;

/// Refusals for structures that cannot be real, so a corrupt or unfamiliar file
/// fails on a bound rather than on an allocation.
const MAX_BLOCKS: u64 = 8_000_000;
const MAX_BLOCK_BYTES: u32 = 64 << 20;
const MAX_PLANE_BYTES: usize = 1 << 30;
const MAX_PLANES: usize = 200_000;

pub struct Oir;

impl Importer for Oir {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.oir", "Olympus OIR")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Read an Olympus/Evident FluoView OIR acquisition.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Olympus OIR", &["oir"])]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        // The signature is the whole answer when it is there. A file named
        // `.oir` without it is not one, whatever it is called, so it goes to
        // another importer rather than failing in this one.
        if head.starts_with(MAGIC) {
            return Confidence::Certain;
        }
        if head.is_empty() && self.file_types().iter().any(|t| t.matches(path)) {
            // An unreadable head — a file on a slow share — is no evidence
            // either way, so the name is all there is.
            return Confidence::Maybe;
        }
        Confidence::No
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let mut file = File::open(&request.path)
            .map_err(|e| PluginError::failed(format!("could not open the file: {e}")))?;

        let index_at = read_header(&mut file)?;
        let offsets = read_index(&mut file, index_at)?;
        host.progress(0.05);

        let planes = read_plane_map(&mut file, &offsets)?;
        if planes.is_empty() {
            return Err(PluginError::unsupported(
                "this OIR carries no image planes this reader recognises — it may be a \
                 multi-file acquisition, whose other parts are named `<name>_00001.oir`",
            ));
        }

        // The sidecar the acquisition software writes beside every OIR. It is
        // the readable form of everything below, and the thing worth keeping.
        let sidecar = read_sidecar(&request.path);
        let summary = sidecar.as_deref().map(Sidecar::parse).unwrap_or_default();

        let (width, height) = dimensions(&mut file, &offsets, &summary, &planes)?;
        let mut planes = planes;
        let (shape, dropped) = Shape::derive(&mut planes, &summary, width, height)?;
        host.log(&format!(
            "{width}x{height}, {} channel(s), {} slice(s), {} frame(s), {}-bit",
            shape.channels,
            shape.slices,
            shape.frames,
            shape.bytes_per_sample * 8
        ));
        if dropped > 0 {
            // Worth saying rather than silently shortening the stack: the user
            // may have expected the frame that the acquisition did not finish.
            host.log(&format!(
                "skipped {dropped} incomplete plane(s) — the acquisition was stopped part-way"
            ));
        }

        let data = read_planes(&mut file, &planes, &shape, host)?;

        let description = sidecar.or_else(|| {
            // No sidecar: fall back to the file's own XML so the acquisition
            // record is not simply lost.
            let docs = embedded_xml(&mut file, &offsets);
            (!docs.is_empty()).then(|| docs.join("\n"))
        });

        let name = request
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "oir".into());

        Ok(ImportResult {
            image: ImageResult {
                width,
                height,
                channels: shape.channels,
                slices: shape.slices,
                frames: shape.frames,
                pixel_type: if shape.bytes_per_sample == 1 {
                    PixelType::U8
                } else {
                    PixelType::U16
                },
                planes: data,
                channel_colors: Vec::new(),
                name: name.clone(),
            },
            info: Some(StackInfo {
                name,
                mode: if shape.channels > 1 {
                    fasttiff_plugin_api::DisplayMode::Composite
                } else {
                    fasttiff_plugin_api::DisplayMode::Grayscale
                },
                unit: summary.pixel_size.map(|_| "micron".to_string()),
                spacing: Spacing {
                    x: summary.pixel_size,
                    y: summary.pixel_size,
                    z: summary.z_step,
                },
                frame_interval_s: summary.frame_interval_s,
                channel_names: summary.channel_names.clone(),
                description,
                ..Default::default()
            }),
        })
    }
}

// ---------------------------------------------------------------- container

fn read_at(file: &mut File, at: u64, len: usize) -> Result<Vec<u8>, PluginError> {
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(at))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|e| PluginError::failed(format!("could not read at {at:#x}: {e}")))?;
    Ok(buf)
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    let s = b.get(at..at + 8)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// Check the signature and return where the block index begins.
fn read_header(file: &mut File) -> Result<u64, PluginError> {
    let len = file
        .metadata()
        .map_err(|e| PluginError::failed(format!("could not stat the file: {e}")))?
        .len();
    let head = read_at(file, 0, 0x50.min(len as usize))?;
    if !head.starts_with(MAGIC) {
        return Err(PluginError::unsupported(
            "not an OIR file: the OLYMPUSRAWFORMAT signature is missing",
        ));
    }
    let at = u64_at(&head, INDEX_OFFSET_AT as usize)
        .ok_or_else(|| PluginError::failed("the OIR header is truncated"))?;
    // Validated rather than trusted: a bad offset here would otherwise become a
    // wild seek and an allocation sized from whatever bytes were there.
    if at < MAGIC.len() as u64 || at.checked_add(INDEX_HEADER as u64).is_none_or(|e| e > len) {
        return Err(PluginError::failed(format!(
            "the OIR block index offset ({at:#x}) is outside the file"
        )));
    }
    Ok(at)
}

/// The block index: one file offset per block.
fn read_index(file: &mut File, at: u64) -> Result<Vec<u64>, PluginError> {
    let len = file
        .metadata()
        .map_err(|e| PluginError::failed(format!("could not stat the file: {e}")))?
        .len();
    let head = read_at(file, at, INDEX_HEADER)?;
    if u32_at(&head, 0) != Some(INDEX_MARKER) {
        return Err(PluginError::unsupported(
            "this OIR's block index is not in a layout this reader knows",
        ));
    }
    let bytes = len - at - INDEX_HEADER as u64;
    let count = bytes / 8;
    if count == 0 || count > MAX_BLOCKS {
        return Err(PluginError::failed(format!(
            "the OIR block index claims {count} blocks, which cannot be right"
        )));
    }
    let raw = read_at(file, at + INDEX_HEADER as u64, (count * 8) as usize)?;
    let mut offsets = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let o = u64_at(&raw, i * 8).unwrap_or(u64::MAX);
        // Blocks past the end are the shape a truncated or foreign file takes.
        if o + 8 > at {
            continue;
        }
        offsets.push(o);
    }
    Ok(offsets)
}

/// One run of bytes belonging to a plane.
#[derive(Clone, Copy)]
struct Chunk {
    /// Where in the reassembled plane these bytes go.
    at: usize,
    /// Where in the file they are.
    file_at: u64,
    len: usize,
}

/// Every plane in the file, keyed by its name, with the chunks that make it up.
///
/// A `BTreeMap` because the key ordering *is* the plane ordering: names are
/// `t001_0_1`, `t002_0_1`, … so sorting them sorts by timepoint and then by the
/// remaining axes, which is the order the stack wants.
type PlaneMap = BTreeMap<String, Vec<Chunk>>;

fn read_plane_map(file: &mut File, offsets: &[u64]) -> Result<PlaneMap, PluginError> {
    let mut planes: PlaneMap = BTreeMap::new();
    let mut i = 0usize;
    while i + 1 < offsets.len() {
        let head = match read_at(file, offsets[i], 16) {
            Ok(h) => h,
            Err(_) => {
                i += 1;
                continue;
            }
        };
        let (Some(len), Some(ty)) = (u32_at(&head, 0), u32_at(&head, 4)) else {
            i += 1;
            continue;
        };
        if ty != TYPE_DESCRIPTOR || !(12..=MAX_BLOCK_BYTES).contains(&len) {
            i += 1;
            continue;
        }
        // Descriptor payload: offset-within-plane, length, name length, name.
        let body = read_at(file, offsets[i] + 8, len as usize)?;
        let (Some(at), Some(run), Some(nlen)) =
            (u32_at(&body, 0), u32_at(&body, 4), u32_at(&body, 8))
        else {
            i += 1;
            continue;
        };
        let Some(raw_name) = body.get(12..12 + nlen as usize) else {
            i += 1;
            continue;
        };
        let Ok(name) = std::str::from_utf8(raw_name) else {
            i += 1;
            continue;
        };

        // The block that follows carries the bytes this one describes.
        let data_head = read_at(file, offsets[i + 1], 8)?;
        let data_len = u32_at(&data_head, 0).unwrap_or(0);
        if data_len as usize != run as usize || data_len > MAX_BLOCK_BYTES {
            i += 1;
            continue;
        }

        if let Some(key) = plane_key(name) {
            let end = (at as usize).saturating_add(run as usize);
            if end <= MAX_PLANE_BYTES {
                planes.entry(key).or_default().push(Chunk {
                    at: at as usize,
                    file_at: offsets[i + 1] + 8,
                    len: run as usize,
                });
            }
        }
        i += 2;
    }
    if planes.len() > MAX_PLANES {
        return Err(PluginError::failed(format!(
            "this OIR declares {} planes, which cannot be right",
            planes.len()
        )));
    }
    Ok(planes)
}

/// The axis part of a block name (`t001_0_1_<uid>` → `t001_0_1`), or `None` for
/// anything that is not an image plane — the reference and thumbnail blocks are
/// named `REF_LSM0_…` and must not become frames.
fn plane_key(name: &str) -> Option<String> {
    if !name.starts_with('t') {
        return None;
    }
    let mut fields = name.split('_');
    let t = fields.next()?;
    if t.len() < 2 || !t[1..].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // Every numeric field is zero-padded, including the timepoint: the files
    // seen so far write `t001`, but one writing `t1`, `t2`, `t10` would
    // otherwise order the tenth frame second.
    let mut key = format!("t{:06}", t[1..].parse::<u64>().ok()?);
    for f in fields {
        if f.bytes().all(|b| b.is_ascii_digit()) && !f.is_empty() {
            key.push('_');
            // Zero-padded so `_10` sorts after `_9` rather than before it.
            key.push_str(&format!("{:06}", f.parse::<u64>().ok()?));
        } else {
            break;
        }
    }
    Some(key)
}

// ------------------------------------------------------------------- shape

struct Shape {
    channels: usize,
    slices: usize,
    frames: usize,
    bytes_per_sample: usize,
    plane_bytes: usize,
}

impl Shape {
    /// Also drops incomplete planes, and says how many.
    ///
    /// An acquisition stopped mid-frame leaves a short last plane — the sample
    /// this was built against ends with 92,160 bytes of a 524,288-byte frame,
    /// and the acquisition software's own export omits it. Keeping it would add
    /// a mostly-black frame to the end of every interrupted recording, which is
    /// the kind of thing that quietly ruins an average.
    fn derive(
        planes: &mut PlaneMap,
        summary: &Sidecar,
        width: u32,
        height: u32,
    ) -> Result<(Shape, usize), PluginError> {
        let px = (width as usize)
            .checked_mul(height as usize)
            .filter(|v| *v > 0)
            .ok_or_else(|| PluginError::failed("this OIR states an impossible frame size"))?;

        // Sample width is measured, not assumed: the chunks of a plane add up
        // to its byte count, and that over the pixel count is the answer.
        // A chunk cannot begin beyond the largest plane these dimensions
        // allow, so one that claims to is excluded here rather than being left
        // to decide the sample width for the whole file.
        let ceiling = px.saturating_mul(2);
        let plane_bytes = planes
            .values()
            .map(|cs| {
                cs.iter()
                    .filter(|c| c.at < ceiling)
                    .map(|c| c.at + c.len)
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let bytes_per_sample = match plane_bytes.checked_div(px) {
            Some(1) => 1,
            Some(2) => 2,
            _ => {
                return Err(PluginError::failed(format!(
                    "a {width}x{height} plane of {plane_bytes} bytes is neither 8- nor 16-bit"
                )))
            }
        };

        let full = px * bytes_per_sample;
        let before = planes.len();
        planes.retain(|_, cs| {
            let covered: usize = cs.iter().filter(|c| c.at < ceiling).map(|c| c.len).sum();
            covered >= full
        });
        let dropped = before - planes.len();
        let n = planes.len();
        if n == 0 {
            return Err(PluginError::failed(format!(
                "every plane in this OIR is incomplete ({before} found)"
            )));
        }

        // The sidecar states the axes; the plane count has to agree with it, or
        // it describes a different file and is not used.
        let (c, z) = match (summary.channels, summary.slices) {
            (Some(c), Some(z)) if c * z > 0 && n.is_multiple_of(c * z) => (c, z),
            (Some(c), None) if c > 0 && n.is_multiple_of(c) => (c, 1),
            (None, Some(z)) if z > 0 && n.is_multiple_of(z) => (1, z),
            _ => (1, 1),
        };
        Ok((
            Shape {
                channels: c,
                slices: z,
                frames: n / (c * z),
                bytes_per_sample,
                plane_bytes: full,
            },
            dropped,
        ))
    }
}

/// The frame size, from the sidecar if it says, otherwise from the file's XML.
fn dimensions(
    file: &mut File,
    offsets: &[u64],
    summary: &Sidecar,
    planes: &PlaneMap,
) -> Result<(u32, u32), PluginError> {
    if let (Some(w), Some(h)) = (summary.width, summary.height) {
        return Ok((w, h));
    }
    let docs = embedded_xml(file, offsets);
    let w = element_number(&docs, "commonimage:width");
    let h = element_number(&docs, "commonimage:height");
    match (w, h) {
        (Some(w), Some(h)) if w >= 1.0 && h >= 1.0 && w <= 1e6 && h <= 1e6 => {
            Ok((w as u32, h as u32))
        }
        _ => Err(PluginError::failed(format!(
            "could not determine the frame size: neither the sidecar `.txt` nor the \
             file's own metadata states it (found {} plane(s))",
            planes.len()
        ))),
    }
}

fn read_planes(
    file: &mut File,
    planes: &PlaneMap,
    shape: &Shape,
    host: &mut dyn ImportHost,
) -> Result<Vec<PlaneData>, PluginError> {
    let total = planes.len();
    let mut out = Vec::with_capacity(total);
    for (i, chunks) in planes.values().enumerate() {
        if i % 16 == 0 && !host.progress(0.1 + 0.85 * (i as f32 / total.max(1) as f32)) {
            return Err(PluginError::unsupported("cancelled"));
        }
        let mut buf = vec![0u8; shape.plane_bytes];
        for c in chunks {
            let end = c.at.saturating_add(c.len);
            // A chunk claiming to run past the plane it belongs to is a
            // corrupt descriptor; the plane is short rather than the read wild.
            if end > buf.len() {
                continue;
            }
            file.seek(SeekFrom::Start(c.file_at))
                .and_then(|_| file.read_exact(&mut buf[c.at..end]))
                .map_err(|e| PluginError::failed(format!("could not read plane data: {e}")))?;
        }
        out.push(match shape.bytes_per_sample {
            1 => PlaneData::U8(buf),
            _ => PlaneData::U16(
                buf.chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect(),
            ),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------- metadata

/// The `<name>.txt` the acquisition software writes beside every OIR.
///
/// This is the file's metadata in the form a person can read, and it is what
/// goes into the written TIFF's `ImageDescription`. Converting an OIR without
/// it throws away the acquisition record — the objective, the laser, the PMT
/// voltage, the event timings — which is the part that cannot be recovered from
/// the pixels afterwards.
fn read_sidecar(path: &Path) -> Option<String> {
    let txt = path.with_extension("txt");
    let bytes = std::fs::read(&txt).ok()?;
    // FluoView writes these as plain ASCII; anything else is not the sidecar.
    let text = String::from_utf8(bytes).ok()?;
    (!text.trim().is_empty()).then_some(text)
}

/// The values worth acting on from the sidecar, as opposed to merely carrying.
#[derive(Default)]
struct Sidecar {
    width: Option<u32>,
    height: Option<u32>,
    channels: Option<usize>,
    slices: Option<usize>,
    /// Microns per pixel.
    pixel_size: Option<f64>,
    /// Microns between slices.
    z_step: Option<f64>,
    frame_interval_s: Option<f64>,
    channel_names: Vec<String>,
}

impl Sidecar {
    /// Parse FluoView's `"key"\t"value"` export.
    ///
    /// Deliberately forgiving: a key that is missing, renamed by a later
    /// software version, or formatted differently leaves its field `None`
    /// rather than failing the import. The pixels do not depend on any of it.
    fn parse(text: &str) -> Sidecar {
        let mut s = Sidecar::default();
        for line in text.lines() {
            let mut parts = line.splitn(2, '\t');
            let (Some(k), Some(v)) = (parts.next(), parts.next()) else {
                continue;
            };
            let key = k.trim().trim_matches('"');
            let value = v.trim().trim_matches('"');
            match key {
                // `512, 0.0 - 318.198 [um], 0.621 [um/pixel]`
                "X Dimension" => {
                    s.width = first_number(value).map(|v| v as u32);
                    s.pixel_size = unit_number(value, "[um/pixel]");
                }
                "Y Dimension" => s.height = first_number(value).map(|v| v as u32),
                // `1 [Ch]`
                "Channel Dimension" => s.channels = first_number(value).map(|v| v as usize),
                // `10, 0.0 - 9.0 [um], 1.0 [um/slice]`
                "Z Dimension" => {
                    s.slices = first_number(value).map(|v| v as usize);
                    s.z_step = unit_number(value, "[um/slice]");
                }
                // `298, 0.000 - 322.701 [s], Interval FreeRun`
                "T Dimension" => {
                    let n = first_number(value);
                    // The span is start-to-end, so the interval is over one
                    // fewer gaps than there are frames.
                    let span = value
                        .split_once('-')
                        .and_then(|(_, rest)| first_number(rest));
                    s.frame_interval_s = match (n, span) {
                        (Some(n), Some(span)) if n > 1.0 => Some(span / (n - 1.0)),
                        _ => None,
                    };
                }
                "Channel Name" => s.channel_names.push(value.to_string()),
                _ => {}
            }
        }
        s
    }
}

/// The first number in `s`, ignoring anything around it.
fn first_number(s: &str) -> Option<f64> {
    let mut start = None;
    for (i, c) in s.char_indices() {
        let numeric = c.is_ascii_digit() || (c == '-' && start.is_none()) || c == '.';
        match (numeric, start) {
            (true, None) => start = Some(i),
            (false, Some(b)) => return s[b..i].parse().ok(),
            _ => {}
        }
    }
    start.and_then(|b| s[b..].parse().ok())
}

/// The number immediately before `unit`, as in `0.621 [um/pixel]`.
fn unit_number(s: &str, unit: &str) -> Option<f64> {
    let at = s.find(unit)?;
    let before = s[..at].trim_end();
    let start = before
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .map(|i| i + 1)
        .unwrap_or(0);
    before[start..].parse().ok().filter(|v: &f64| v.is_finite())
}

/// Every XML document in the file's metadata blocks.
///
/// Only the blocks the index points at are examined, so this never scans the
/// pixel data — which in a real acquisition is 95% of the file.
fn embedded_xml(file: &mut File, offsets: &[u64]) -> Vec<String> {
    let mut out = Vec::new();
    for &o in offsets {
        let Ok(head) = read_at(file, o, 8) else {
            continue;
        };
        let Some(len) = u32_at(&head, 0) else {
            continue;
        };
        // Metadata blocks are XML documents of a few tens of kilobytes; pixel
        // blocks are a fixed small size and never start with `<?xml`.
        if !(16..=(4 << 20)).contains(&len) {
            continue;
        }
        let Ok(body) = read_at(file, o + 8, len as usize) else {
            continue;
        };
        let start = body
            .windows(5)
            .position(|w| w == b"<?xml")
            .unwrap_or(usize::MAX);
        if start == usize::MAX {
            continue;
        }
        if let Ok(text) = std::str::from_utf8(&body[start..]) {
            let trimmed = text.trim_end_matches('\0').trim();
            if trimmed.ends_with('>') {
                out.push(trimmed.to_string());
            }
        }
    }
    out
}

/// The number inside `<tag>…</tag>`, across all documents.
fn element_number(docs: &[String], tag: &str) -> Option<f64> {
    let open = format!("<{tag}>");
    for d in docs {
        if let Some(i) = d.find(&open) {
            let rest = &d[i + open.len()..];
            let end = rest.find('<')?;
            if let Ok(v) = rest[..end].trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "oir_tests.rs"]
mod tests;

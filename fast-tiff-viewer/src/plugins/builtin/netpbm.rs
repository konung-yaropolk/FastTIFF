//! A built-in importer for binary Netpbm images (PGM `P5`, PPM `P6`).
//!
//! Netpbm earns its place as the first importer for the same reasons it earns
//! its place in a hundred other tools: the format is small enough to implement
//! correctly in one sitting, it is genuinely used in scientific pipelines, and
//! it has a real magic number — so it exercises [`Importer::probe`] rather than
//! leaning on the extension the way a format without one has to.
//!
//! It also demonstrates the shape a vendor-format importer takes, which is the
//! point of the exercise: read the file, fill in an [`ImportResult`], and the
//! document that appears is indistinguishable from an opened TIFF.

use fasttiff_plugin_api::{
    Confidence, FileType, ImageResult, ImportHost, ImportRequest, ImportResult, Importer,
    PixelType, PlaneData, PluginError, PluginInfo, StackInfo,
};
use std::path::Path;

pub struct Netpbm;

/// The header of a binary Netpbm file.
struct Header {
    /// 5 = PGM (one channel), 6 = PPM (three).
    kind: u8,
    width: u32,
    height: u32,
    max: u32,
    /// Offset of the first pixel byte.
    data_at: usize,
}

/// Parse the header: `P5`/`P6`, then width, height and maxval as whitespace-
/// separated tokens, with `#` comments allowed anywhere between them.
///
/// Exactly one whitespace byte follows the maxval and the pixels start after
/// it — that single byte is part of the format, not optional padding, and
/// getting it wrong shifts every pixel.
fn parse_header(b: &[u8]) -> Result<Header, PluginError> {
    if b.len() < 2 || b[0] != b'P' || !matches!(b[1], b'5' | b'6') {
        return Err(PluginError::unsupported(
            "not a binary Netpbm file (expected a P5 or P6 signature)",
        ));
    }
    let kind = b[1] - b'0';
    let mut i = 2usize;
    let mut nums = [0u32; 3];

    for slot in &mut nums {
        // Skip whitespace and whole comment lines.
        loop {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && b[i] == b'#' {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            } else {
                break;
            }
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            return Err(PluginError::failed("truncated Netpbm header"));
        }
        *slot = std::str::from_utf8(&b[start..i])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| PluginError::failed("unreadable number in the Netpbm header"))?;
    }

    // Exactly one whitespace byte separates the header from the pixels.
    if i >= b.len() || !b[i].is_ascii_whitespace() {
        return Err(PluginError::failed(
            "the Netpbm header is not followed by the single whitespace byte the format requires",
        ));
    }
    let header = Header {
        kind,
        width: nums[0],
        height: nums[1],
        max: nums[2],
        data_at: i + 1,
    };

    if header.width == 0 || header.height == 0 {
        return Err(PluginError::failed(
            "the Netpbm header declares an empty image",
        ));
    }
    if header.max == 0 || header.max > 65535 {
        return Err(PluginError::failed(format!(
            "the Netpbm maxval is {}, which the format does not allow",
            header.max
        )));
    }
    Ok(header)
}

impl Importer for Netpbm {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.netpbm", "Netpbm (PGM/PPM)")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Read binary Netpbm images: P5 greyscale and P6 colour, 8- or 16-bit.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![
            FileType::new("Netpbm greyscale", &["pgm"]),
            FileType::new("Netpbm colour", &["ppm"]),
            FileType::new("Netpbm", &["pnm"]),
        ]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        // The signature is authoritative, so a mislabelled file is still read
        // and a `.pgm` that is really something else is declined.
        match head.first_chunk::<2>() {
            Some(b"P5") | Some(b"P6") => Confidence::Certain,
            Some(_) if !head.is_empty() => Confidence::No,
            // Nothing to look at (an empty read): fall back to the extension.
            _ => {
                if self.file_types().iter().any(|t| t.matches(path)) {
                    Confidence::Maybe
                } else {
                    Confidence::No
                }
            }
        }
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let bytes = std::fs::read(&request.path)
            .map_err(|e| PluginError::failed(format!("reading {}: {e}", request.path.display())))?;
        let h = parse_header(&bytes)?;

        let channels = if h.kind == 6 { 3usize } else { 1 };
        let sample_bytes = if h.max > 255 { 2usize } else { 1 };
        let n_px = h.width as usize * h.height as usize;
        let need = n_px
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(sample_bytes))
            .ok_or_else(|| PluginError::failed("the Netpbm header declares an impossible size"))?;

        let data = bytes.get(h.data_at..).unwrap_or(&[]);
        if data.len() < need {
            return Err(PluginError::failed(format!(
                "the file holds {} pixel bytes but its header declares {}x{} with {channels} channel(s) at {sample_bytes} byte(s) = {need}",
                data.len(),
                h.width,
                h.height
            )));
        }

        if !host.progress(0.5) {
            return Err(PluginError::failed("cancelled"));
        }

        // Netpbm is interleaved and big-endian; split into planes, which is
        // what the contract asks for and what the viewer wants anyway.
        let mut planes = Vec::with_capacity(channels);
        for c in 0..channels {
            let plane: Vec<u16> = (0..n_px)
                .map(|i| {
                    let at = (i * channels + c) * sample_bytes;
                    if sample_bytes == 1 {
                        data[at] as u16
                    } else {
                        u16::from_be_bytes([data[at], data[at + 1]])
                    }
                })
                .collect();
            planes.push(if sample_bytes == 1 {
                PlaneData::U8(plane.into_iter().map(|v| v as u8).collect())
            } else {
                PlaneData::U16(plane)
            });
        }

        host.log(&format!(
            "Netpbm P{}: {}x{}, {channels} channel(s), maxval {}",
            h.kind, h.width, h.height, h.max
        ));
        host.progress(1.0);

        let name = request
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "netpbm".into());

        Ok(ImportResult {
            image: ImageResult {
                width: h.width,
                height: h.height,
                channels,
                slices: 1,
                frames: 1,
                pixel_type: if sample_bytes == 1 {
                    PixelType::U8
                } else {
                    PixelType::U16
                },
                planes,
                name: name.clone(),
            },
            info: Some(StackInfo {
                name,
                path: Some(request.path.display().to_string()),
                // P6 is colour; the three planes are its components.
                mode: if channels == 3 {
                    fasttiff_plugin_api::DisplayMode::Composite
                } else {
                    fasttiff_plugin_api::DisplayMode::Grayscale
                },
                ..Default::default()
            }),
        })
    }
}

#[cfg(test)]
#[path = "netpbm_tests.rs"]
mod tests;

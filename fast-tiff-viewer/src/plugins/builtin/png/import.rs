//! Reading a PNG in.
//!
//! The other direction from [the exporter](super::Png), and a different job.
//! The exporter writes *what the window is showing* — one frame, contrast
//! applied, 8-bit RGB, a figure. This reads a PNG as a **document**: the
//! samples as the file stores them, at the depth it stores them, so a 16-bit
//! greyscale PNG opens as a 16-bit greyscale stack and can be measured, not
//! just looked at.
//!
//! # What it does with each kind of PNG
//!
//! * **Greyscale** becomes one channel; **colour** becomes three, composited,
//!   which is how the viewer shows a colour image.
//! * **8-bit stays 8-bit and 16-bit stays 16-bit.** A file's depth is a
//!   property of the measurement; widening it would invent precision and
//!   narrowing it would throw some away.
//! * **Palette images are expanded** to the colours they stand for. A palette
//!   index is not a measurement — sample 3 is not three times sample 1 — so
//!   carrying the indices through as if they were samples would produce a
//!   picture that is wrong everywhere the palette is not a ramp.
//! * **Low bit depths** (1, 2 and 4-bit) are expanded to 8-bit, which is the
//!   narrowest a stack can be.
//!
//! # Alpha is dropped, and said so
//!
//! Transparency is a compositing instruction, not a channel of data: carried
//! through as a fourth channel it would be added to the picture as if it were
//! light. It is dropped, and the log line says so — a viewer that quietly
//! showed three of a file's four channels would be worse.
//!
//! # What comes with it
//!
//! A PNG can state its pixel size (`pHYs`) and carry text. Both are read: the
//! pixel size becomes the stack's calibration, so a measurement on an imported
//! PNG is in microns when the file knew them, and the text becomes the
//! document's description, which is where anything written by whatever produced
//! the file — ImageJ writes its own record there — stays readable.

use fasttiff_plugin_api::{
    Confidence, DisplayMode, FileType, ImageResult, ImportHost, ImportRequest, ImportResult,
    Importer, PixelType, PlaneData, PluginError, PluginInfo, Spacing, StackInfo,
};
use std::path::Path;

/// The eight bytes every PNG starts with.
const MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";

/// How much text a file may contribute to the description.
///
/// A generous bound on a record and a mean one on a payload: text chunks are
/// meant for provenance, and a PNG carrying half a megabyte of them is using
/// them as storage. What is kept is still the beginning of it.
///
/// A budget for the *file*, not for each chunk. Per chunk it would bound
/// nothing: a PNG may carry as many as it likes, and twenty thousand of them
/// inflating to 64 KiB each is a gigabyte of work to keep 64 KiB of it.
const MAX_TEXT: usize = 64 * 1024;

/// Read a PNG as a document.
pub struct PngImport;

impl Importer for PngImport {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.png.import", "PNG")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Read a PNG: 8- or 16-bit, greyscale, colour or palette, with its \
                 pixel size and text if it states them.",
            )
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("PNG", &["png"])]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        if head.starts_with(MAGIC) {
            return Confidence::Certain;
        }
        // Too little to decide on: the head is empty because the file could
        // not be read, or shorter than the signature and still consistent with
        // it. The host reads the head with one `read`, which is allowed to
        // come up short, so "fewer than eight bytes" is not the same as "not a
        // PNG" — and answering `No` there would leave a real PNG with no
        // importer at all. The extension is then the only evidence there is.
        if MAGIC.starts_with(head) && self.file_types().iter().any(|t| t.matches(path)) {
            return Confidence::Maybe;
        }
        // A head long enough to decide on, that says it is something else.
        Confidence::No
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let file = std::fs::File::open(&request.path)
            .map_err(|e| PluginError::failed(format!("reading {}: {e}", request.path.display())))?;
        let mut decoder = ::png::Decoder::new(std::io::BufReader::new(file));
        // Palette entries become the colours they stand for, narrow depths
        // become 8-bit, and a `tRNS` chunk becomes an alpha channel — which is
        // then dropped below. Without this a palette image would arrive as a
        // plane of indices.
        decoder.set_transformations(::png::Transformations::EXPAND);

        let mut reader = decoder.read_info().map_err(|e| {
            PluginError::unsupported(format!("this is not a PNG this reader can take: {e}"))
        })?;

        // Read before anything is decoded, because decoding moves it on: an
        // `fcTL` *before* the image data means the image data is animation
        // frame 1, and its absence means the image data is a separate still.
        // Asked after `next_frame`, this would be the *next* frame's control
        // chunk and would say `Some` either way.
        let animation_frame = reader.info().frame_control.is_some();

        let (color, depth) = reader.output_color_type();
        let samples = match color {
            ::png::ColorType::Grayscale => 1usize,
            ::png::ColorType::GrayscaleAlpha => 2,
            ::png::ColorType::Rgb => 3,
            ::png::ColorType::Rgba => 4,
            // `EXPAND` turns a palette into colours, so reaching here means the
            // decoder did something this reader does not know about.
            ::png::ColorType::Indexed => {
                return Err(PluginError::unsupported(
                    "this PNG's palette could not be expanded",
                ))
            }
        };
        let sample_bytes = match depth {
            ::png::BitDepth::Sixteen => 2usize,
            // Everything narrower has been expanded to eight bits.
            _ => 1,
        };
        // Alpha is a compositing instruction, not a measurement.
        let channels = if samples >= 3 { 3 } else { 1 };
        let has_alpha = samples == 2 || samples == 4;

        let size = reader
            .output_buffer_size()
            .ok_or_else(|| PluginError::failed("this PNG declares a size too large to read"))?;
        let mut buf = vec![0u8; size];
        if !host.progress(0.1) {
            return Err(cancelled());
        }
        let frame = reader
            .next_frame(&mut buf)
            .map_err(|e| PluginError::failed(format!("decoding this PNG: {e}")))?;
        if !host.progress(0.6) {
            return Err(cancelled());
        }

        let (width, height) = (frame.width, frame.height);
        if width == 0 || height == 0 {
            return Err(PluginError::failed("this PNG has no pixels"));
        }

        // One plane per kept channel, read out of the interleaved rows. The
        // row length comes from the decoder rather than being recomputed from
        // the width: `line_size` is what it actually filled.
        let n_px = width as usize * height as usize;
        let stride = frame.line_size;
        let mut planes = Vec::with_capacity(channels);
        for c in 0..channels {
            let mut plane = Vec::with_capacity(n_px);
            for y in 0..height as usize {
                let row = y * stride;
                for x in 0..width as usize {
                    let at = row + (x * samples + c) * sample_bytes;
                    // A row the decoder did not fill would be a bug in it
                    // rather than in the file; treat it as black instead of
                    // panicking half way through an import.
                    let v = match sample_bytes {
                        1 => buf.get(at).map(|&b| b as u16),
                        _ => buf
                            .get(at..at + 2)
                            .map(|b| u16::from_be_bytes([b[0], b[1]])),
                    };
                    plane.push(v.unwrap_or(0));
                }
            }
            planes.push(match sample_bytes {
                1 => PlaneData::U8(plane.into_iter().map(|v| v as u8).collect()),
                _ => PlaneData::U16(plane),
            });
        }
        if !host.progress(0.9) {
            return Err(cancelled());
        }

        // Read to the end of the file before asking what it said. `next_frame`
        // stops at the end of the image data, and a PNG may carry its text
        // *after* that — the spec allows it either side, and a writer that
        // only learns the text once the pixels are out has no choice, which is
        // the only layout the `png` crate's own writer can produce. Without
        // this the description is silently empty for every one of them.
        //
        // The pixels are already decoded, so a file that is damaged past them
        // costs the text and nothing else.
        let _ = reader.finish();
        let info = reader.info();
        // `pHYs` in pixels per metre, which is the only unit it can be in.
        let micron_per_px = info.pixel_dims.as_ref().and_then(|d| {
            (d.unit == ::png::Unit::Meter && d.xppu > 0 && d.yppu > 0)
                .then(|| (1e6 / d.xppu as f64, 1e6 / d.yppu as f64))
        });
        let description = text_of(info);

        host.log(&format!(
            "PNG {}x{}, {:?} {}-bit -> {channels} channel(s){}{}",
            width,
            height,
            color,
            sample_bytes * 8,
            if has_alpha { ", alpha dropped" } else { "" },
            match micron_per_px {
                Some((x, _)) => format!(", {x:.4} micron/pixel"),
                None => String::new(),
            }
        ));
        if info.animation_control.is_some() {
            // Which picture that was depends on where the first `fcTL` sits.
            // With one before the image data, the image data *is* animation
            // frame 1. With none, it is a separate still — the default image,
            // which is what a viewer that does not animate should show, and
            // which is not any of the frames.
            host.log(if animation_frame {
                "this is an animated PNG; only its first frame was read"
            } else {
                "this is an animated PNG with a separate default image; \
                 that still image was read, not any animation frame"
            });
        }
        host.progress(1.0);

        let name = request
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "png".into());

        Ok(ImportResult {
            image: ImageResult {
                width,
                height,
                channels,
                slices: 1,
                frames: 1,
                pixel_type: if sample_bytes == 1 {
                    PixelType::U8
                } else {
                    PixelType::U16
                },
                planes,
                // Left to the host, which colours the first three channels red,
                // green and blue — which is what these three are.
                channel_colors: Vec::new(),
                metadata: None,
                name: name.clone(),
            },
            info: Some(StackInfo {
                name,
                path: Some(request.path.display().to_string()),
                mode: if channels == 3 {
                    DisplayMode::Composite
                } else {
                    DisplayMode::Grayscale
                },
                unit: micron_per_px.map(|_| "micron".to_string()),
                spacing: Spacing {
                    x: micron_per_px.map(|(x, _)| x),
                    y: micron_per_px.map(|(_, y)| y),
                    z: None,
                },
                description,
                ..Default::default()
            }),
        })
    }
}

/// The file's text chunks as one record, or `None` when it carries none.
///
/// Every flavour of text chunk, in the order PNG defines them, as
/// `keyword: text` lines — which is what the file itself says, laid out so a
/// person can read it. A compressed chunk that will not decompress is skipped
/// rather than failing the import: the pixels are the point, and a damaged
/// comment is not a reason to refuse a file.
fn text_of(info: &::png::Info<'static>) -> Option<String> {
    let mut out = String::new();
    collect_text(&mut out, info);
    (!out.is_empty()).then_some(out)
}

/// Append chunks until [`MAX_TEXT`] is spent, then stop.
///
/// Stopping is the point. Inflating every compressed chunk and throwing away
/// the ones that no longer fit costs a file's whole payload to keep 64 KiB of
/// it, which is the denial of service the bound was meant to prevent.
fn collect_text(out: &mut String, info: &::png::Info<'static>) {
    for c in &info.uncompressed_latin1_text {
        if !push_text(out, &c.keyword, &c.text) {
            return;
        }
    }
    // Compressed text is inflated against whatever budget is left rather than
    // on trust. A few hundred bytes of zTXt decompress to gigabytes, and the
    // crate's plain `get_text` has no bound at all — so a reader that used it
    // would let a file's *comment* exhaust memory before it had looked at a
    // single pixel.
    for c in &info.compressed_latin1_text {
        let mut c = c.clone();
        // Bounded by what is left rather than by the whole budget, so the
        // chunk that runs out of room costs only the room it had.
        if c.decompress_text_with_limit(room_in(out)).is_err() {
            continue;
        }
        let Ok(text) = c.get_text() else { continue };
        if !push_text(out, &c.keyword, &text) {
            return;
        }
    }
    for c in &info.utf8_text {
        let mut c = c.clone();
        // Bounded by what is left rather than by the whole budget, so the
        // chunk that runs out of room costs only the room it had.
        if c.decompress_text_with_limit(room_in(out)).is_err() {
            continue;
        }
        let Ok(text) = c.get_text() else { continue };
        if !push_text(out, &c.keyword, &text) {
            return;
        }
    }
}

/// What is left of the budget.
fn room_in(out: &str) -> usize {
    MAX_TEXT.saturating_sub(out.len())
}

/// Append one `keyword: text` line, clipped to the room left. `false` once the
/// budget is spent and the caller should stop.
///
/// Clipped rather than dropped: a file using text chunks as storage should lose
/// its tail, not its whole record — and dropping whole chunks would also let a
/// short one land after a long one was skipped, so what survived would depend
/// on lengths rather than on order.
///
/// The keyword and its separator are appended without being counted, so `out`
/// may overrun by a keyword's length. PNG bounds a keyword at 79 bytes and the
/// decoder enforces it, which makes the overrun at most about eighty bytes on
/// a 64 KiB budget — lenient in the direction that cannot hurt.
fn push_text(out: &mut String, keyword: &str, text: &str) -> bool {
    if out.len() >= MAX_TEXT {
        return false;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(keyword);
    out.push_str(": ");
    out.push_str(clip(text.trim_end(), room_in(out)));
    out.len() < MAX_TEXT
}

/// The longest prefix of `s` that fits in `n` bytes without splitting a
/// character.
fn clip(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The user pressed stop.
fn cancelled() -> PluginError {
    PluginError::failed("cancelled")
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;

//! An example FastTIFF plugin library.
//!
//! Two plugins, chosen to cover the two halves of the contract:
//!
//! * [`Invert`] is a byte-for-byte copy of the viewer's built-in `Invert`. It
//!   is the **oracle** for the whole C boundary: the same arithmetic, reached
//!   through `dlopen`, a function pointer table and two rounds of copying,
//!   must produce exactly the bits the built-in produces. Anything wrong with
//!   plane addressing, sample marshalling or result assembly shows up as a
//!   difference, in a test that needs no fixture and no judgement call.
//!
//! * [`RawImport`] is an importer with a dialog, for headerless binary files —
//!   the format every instrument eventually emits and no library can guess.
//!   It exercises the part of the boundary `Invert` cannot: `probe`, a dialog
//!   declared from a *path* rather than an open stack, and a result produced
//!   with no host to read pixels from.
//!
//! * [`CsvExport`] is an exporter, which is the boundary run backwards: the
//!   host hands over a path and the dialog values, the plugin reads pixels back
//!   out of the host, and what crosses in return is a file on disk rather than
//!   an image.
//!
//! # Building one of these yourself
//!
//! ```toml
//! [lib]
//! crate-type = ["cdylib"]
//!
//! [dependencies]
//! fasttiff-plugin = "0.18"
//! ```
//!
//! Implement `Plugin` or `Importer`, call `export_plugin!`, and drop the
//! resulting `.dll`/`.so`/`.dylib` in the folder that **Plugins > Open plugin
//! folder...** opens.

use fasttiff_plugin::api::{
    Confidence, ExportRequest, Exporter, FileType, HostContext, HostContextExt, ImageResult,
    ImportHost, ImportRequest, ImportResult, Importer, Outcome, ParamDecl, ParamKind, Params,
    PixelType, Plane, PlaneData, Plugin, PluginError, PluginInfo,
};
use std::path::Path;

// --------------------------------------------------------------------- filter

/// Invert the frame on screen, about its own range.
///
/// Deliberately identical to `fast_tiff_viewer::plugins::builtin::Invert`,
/// down to the NaN handling and the result's name. The test that compares them
/// is only meaningful while that stays true — if you change one, change both.
#[derive(Default)]
pub struct Invert;

impl Plugin for Invert {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.example.invert", "Invert (from library)")
            .menu_path("Examples")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Invert the current frame about its own min/max. Loaded from a shared library.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        // Only offer the choice when there is one.
        if info.channels <= 1 {
            return Vec::new();
        }
        vec![ParamDecl::new(
            "all_channels",
            "All channels",
            ParamKind::Bool { default: false },
        )
        .help("Invert every channel rather than only the first.")]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.plane_len() == 0 {
            return Err(PluginError::unsupported("the stack has no pixels"));
        }
        let t = host.view().frame_index.min(info.frames.saturating_sub(1));
        let all = params.bool("all_channels", false);
        let n = if all { info.channels.max(1) } else { 1 };

        let mut planes = Vec::with_capacity(n);
        let mut buf = Vec::new();
        for c in 0..n {
            if !host.progress(c as f32 / n as f32) {
                return Ok(Outcome::Cancelled);
            }
            host.read_plane_f32(Plane::new(c, 0, t), &mut buf)?;
            let (lo, hi) = buf
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    if v.is_finite() {
                        (l.min(v), h.max(v))
                    } else {
                        (l, h)
                    }
                });
            let (lo, hi) = if lo.is_finite() && hi.is_finite() {
                (lo, hi)
            } else {
                (0.0, 1.0)
            };
            planes.push(PlaneData::F32(
                buf.iter()
                    .map(|&v| if v.is_finite() { hi - (v - lo) } else { v })
                    .collect(),
            ));
        }

        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels: n,
            slices: 1,
            frames: 1,
            pixel_type: PixelType::F32,
            planes,
            channel_colors: Vec::new(),
            metadata: None,
            name: format!("{}-inverted", host.stack_info().name),
        })))
    }
}

// ------------------------------------------------------------------- importer

/// Read a headerless binary file, given its shape.
///
/// There is nothing in such a file to detect, so [`probe`](Importer::probe)
/// never answers `Certain` — the extension is the only evidence there is, which
/// is exactly the case the confidence ranking exists to handle: a real `.raw`
/// importer for a specific instrument would answer `Certain` on its own magic
/// and take precedence over this one.
#[derive(Default)]
pub struct RawImport;

/// The sample types the dialog offers, in the order it offers them.
const RAW_TYPES: [(&str, PixelType); 4] = [
    ("8-bit unsigned", PixelType::U8),
    ("16-bit unsigned", PixelType::U16),
    ("16-bit signed", PixelType::I16),
    ("32-bit float", PixelType::F32),
];

impl Importer for RawImport {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.example.raw", "Raw binary")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Read a headerless binary image, given its dimensions.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Raw binary image", &["raw", "bin"])]
    }

    fn probe(&self, path: &Path, _head: &[u8]) -> Confidence {
        if self.file_types().iter().any(|t| t.matches(path)) {
            Confidence::Maybe
        } else {
            Confidence::No
        }
    }

    fn params(&self, path: &Path) -> Vec<ParamDecl> {
        // The file's size is the one fact available without reading it, and it
        // makes the dialog far less painful: a square guess that divides the
        // file exactly is usually right, and is always a better starting point
        // than 512x512.
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let guess = square_guess(bytes, 2);
        vec![
            ParamDecl::new(
                "width",
                "Width",
                ParamKind::Int {
                    default: guess as i64,
                    min: 1,
                    max: 1 << 20,
                },
            ),
            ParamDecl::new(
                "height",
                "Height",
                ParamKind::Int {
                    default: guess as i64,
                    min: 1,
                    max: 1 << 20,
                },
            ),
            ParamDecl::new(
                "type",
                "Sample type",
                ParamKind::Choice {
                    default: 1,
                    options: RAW_TYPES.iter().map(|(n, _)| (*n).to_string()).collect(),
                },
            ),
            ParamDecl::new(
                "images",
                "Number of images",
                ParamKind::Int {
                    default: 1,
                    min: 1,
                    max: 1 << 20,
                },
            ),
            ParamDecl::new(
                "offset",
                "Header bytes to skip",
                ParamKind::Int {
                    default: 0,
                    min: 0,
                    max: i64::MAX,
                },
            )
            .help("Bytes at the start of the file that are not pixels."),
            ParamDecl::new(
                "little_endian",
                "Little-endian",
                ParamKind::Bool { default: true },
            ),
            ParamDecl::new(
                "pixel_size",
                "Pixel size (micron)",
                ParamKind::Float {
                    default: 0.0,
                    min: 0.0,
                    max: 1e6,
                },
            )
            .help("0 to leave the result uncalibrated."),
        ]
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let width = request.params.int("width", 0).max(0) as usize;
        let height = request.params.int("height", 0).max(0) as usize;
        let frames = request.params.int("images", 1).max(1) as usize;
        let offset = request.params.int("offset", 0).max(0) as usize;
        let le = request.params.bool("little_endian", true);
        // 0 means "not stated", which is different from 0 microns.
        let pixel_size = Some(request.params.float("pixel_size", 0.0)).filter(|v| *v > 0.0);
        let ty = RAW_TYPES
            .get(request.params.choice("type", 1))
            .map(|(_, t)| *t)
            .unwrap_or(PixelType::U16);

        if width == 0 || height == 0 {
            return Err(PluginError::failed("width and height are required"));
        }
        let n_px = width
            .checked_mul(height)
            .ok_or_else(|| PluginError::failed("those dimensions overflow"))?;
        let plane_bytes = n_px
            .checked_mul(ty.bytes())
            .ok_or_else(|| PluginError::failed("those dimensions overflow"))?;

        let data = std::fs::read(&request.path)
            .map_err(|e| PluginError::failed(format!("could not read the file: {e}")))?;
        let body = data
            .get(offset..)
            .ok_or_else(|| PluginError::failed("the header offset is past the end of the file"))?;

        // Say what is actually wrong rather than reading a short plane: a
        // mistyped width is the overwhelmingly likely cause, and the numbers
        // are what the user needs to see to fix it.
        let need = plane_bytes.saturating_mul(frames);
        if body.len() < need {
            return Err(PluginError::failed(format!(
                "{width}x{height} {} x{frames} needs {need} bytes but only {} follow the header",
                RAW_TYPES
                    .iter()
                    .find(|(_, t)| *t == ty)
                    .map(|(n, _)| *n)
                    .unwrap_or("?"),
                body.len()
            )));
        }

        let mut planes = Vec::with_capacity(frames);
        for f in 0..frames {
            // A raw file can be enormous, so this is not ceremony: it is the
            // only way a user gets out of a wrong width they typed by mistake.
            if !host.progress(f as f32 / frames as f32) {
                return Err(PluginError::unsupported("cancelled"));
            }
            let chunk = &body[f * plane_bytes..(f + 1) * plane_bytes];
            planes.push(decode_plane(chunk, ty, le));
        }
        host.log(&format!("read {frames} {width}x{height} plane(s)"));

        Ok(ImportResult {
            image: ImageResult {
                width: width as u32,
                height: height as u32,
                channels: 1,
                slices: 1,
                frames,
                // I16 has no `PlaneData` of its own; it is carried in the u16
                // lane, offset into unsigned, exactly as the viewer does.
                pixel_type: if ty == PixelType::I16 {
                    PixelType::U16
                } else {
                    ty
                },
                planes,
                channel_colors: Vec::new(),
                metadata: None,
                name: name_of(&request.path),
            },
            // A headerless file states nothing about itself, so the dialog is
            // the only source of scale — but reporting it matters anyway: it is
            // what makes the resulting document measure in the units the user
            // said rather than in pixels.
            info: Some(fasttiff_plugin::api::StackInfo {
                name: name_of(&request.path),
                spacing: fasttiff_plugin::api::Spacing {
                    x: pixel_size,
                    y: pixel_size,
                    z: None,
                },
                unit: pixel_size.map(|_| "micron".to_string()),
                description: Some(format!(
                    "raw import: {width}x{height} {}, {} images",
                    RAW_TYPES
                        .iter()
                        .find(|(_, t)| *t == ty)
                        .map(|(n, _)| *n)
                        .unwrap_or("?"),
                    frames
                )),
                ..Default::default()
            }),
        })
    }
}

/// A file's stem, or a fallback when it has none.
fn name_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "raw".into())
}

/// The largest square side whose plane would fill `bytes` exactly.
///
/// Returns 512 when nothing divides, which is the conventional default and no
/// worse than any other guess.
fn square_guess(bytes: u64, sample_bytes: u64) -> u32 {
    if bytes == 0 || sample_bytes == 0 {
        return 512;
    }
    let px = bytes / sample_bytes;
    let side = (px as f64).sqrt() as u64;
    if side > 0 && side * side * sample_bytes == bytes {
        side as u32
    } else {
        512
    }
}

fn decode_plane(chunk: &[u8], ty: PixelType, le: bool) -> PlaneData {
    match ty {
        PixelType::U8 => PlaneData::U8(chunk.to_vec()),
        PixelType::F32 => PlaneData::F32(
            chunk
                .chunks_exact(4)
                .map(|b| {
                    let a = [b[0], b[1], b[2], b[3]];
                    if le {
                        f32::from_le_bytes(a)
                    } else {
                        f32::from_be_bytes(a)
                    }
                })
                .collect(),
        ),
        PixelType::I16 => PlaneData::U16(
            chunk
                .chunks_exact(2)
                .map(|b| {
                    let v = if le {
                        i16::from_le_bytes([b[0], b[1]])
                    } else {
                        i16::from_be_bytes([b[0], b[1]])
                    };
                    // Offset into unsigned rather than casting: the viewer
                    // shows u16, and a bare `as u16` would put the negative
                    // half above the positive one.
                    (v as i32 + 32768) as u16
                })
                .collect(),
        ),
        PixelType::U16 => PlaneData::U16(
            chunk
                .chunks_exact(2)
                .map(|b| {
                    if le {
                        u16::from_le_bytes([b[0], b[1]])
                    } else {
                        u16::from_be_bytes([b[0], b[1]])
                    }
                })
                .collect(),
        ),
    }
}

// ------------------------------------------------------------------ metadata

/// Report what the host says about the open file — ImageJ's **Image ▸ Show
/// Info**, and the smallest plugin that is actually useful.
///
/// It is also the only way to see the metadata half of the boundary working. A
/// filter that reads pixels proves the pixels crossed; nothing proves the
/// *calibration* crossed except a plugin that reads it and says so, and a
/// plugin that quietly assumes microns when the file said nothing is exactly
/// the failure this exists to make visible.
#[derive(Default)]
pub struct ShowInfo;

impl Plugin for ShowInfo {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.example.showinfo", "Show Info")
            .menu_path("Examples")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Report the file's dimensions, spacing and calibration.")
    }

    fn run(&mut self, host: &mut dyn HostContext, _p: &Params) -> Result<Outcome, PluginError> {
        let image = host.image();
        let info = host.stack_info();
        let unit = info.unit.clone().unwrap_or_else(|| "pixels".into());

        let mut out = String::new();
        out.push_str(&format!("name: {}\n", info.name));
        out.push_str(&format!(
            "size: {}x{}, {} channel(s), {} slice(s), {} frame(s)\n",
            image.width, image.height, image.channels, image.slices, image.frames
        ));
        out.push_str(&format!("type: {:?}\n", image.pixel_type));
        out.push_str(&format!("mode: {:?}\n", info.mode));
        // Every one of these is `None` until the metadata actually crosses, so
        // "unknown" here is a real answer and not a placeholder.
        out.push_str(&format!(
            "pixel: {} x {} {unit}\n",
            fmt(info.spacing.x),
            fmt(info.spacing.y)
        ));
        out.push_str(&format!("z step: {} {unit}\n", fmt(info.spacing.z)));
        out.push_str(&format!(
            "frame interval: {} s\n",
            fmt(info.frame_interval_s)
        ));
        out.push_str(&match info.calibration {
            Some((c0, c1)) => format!("calibration: {c0} + {c1} * raw\n"),
            None => "calibration: none\n".to_string(),
        });
        if !info.channel_names.is_empty() {
            out.push_str(&format!("channels: {}\n", info.channel_names.join(", ")));
        }
        if let Some(d) = &info.description {
            out.push_str(&format!("description: {} bytes\n", d.len()));
        }

        host.log("read the file's metadata");
        Ok(Outcome::Message(out))
    }
}

fn fmt(v: Option<f64>) -> String {
    match v {
        Some(v) => format!("{v}"),
        None => "unknown".into(),
    }
}

// ------------------------------------------------------------------- exporter

/// Write the frame on screen as a grid of numbers.
///
/// Deliberately the dullest possible format. What an exporter in this crate has
/// to demonstrate is the *boundary* — that a shared library can be handed a
/// path and a set of dialog values, call back into the host for pixels, and be
/// judged by the bytes it leaves on disk. A format with a real encoder would
/// demonstrate the encoder.
///
/// The numbers are the file's own, read through `read_plane_f32` rather than
/// the windowed ones on screen: a value written into a text file is a
/// measurement, and rescaling it to wherever the contrast slider happens to sit
/// would quietly change what it says.
#[derive(Default)]
pub struct CsvExport;

impl Exporter for CsvExport {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.example.csv", "CSV (from library)")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Write the frame on screen as comma-separated values.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Comma-separated values", &["csv"])]
    }

    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        vec![
            ParamDecl::new("header", "Size header", ParamKind::Bool { default: true })
                .help("Write a leading `# widthxheight` comment line."),
            ParamDecl::new(
                "decimals",
                "Decimals",
                ParamKind::Int {
                    default: 3,
                    min: 0,
                    max: 9,
                },
            ),
        ]
    }

    fn export(
        &mut self,
        request: &ExportRequest,
        host: &mut dyn HostContext,
    ) -> Result<(), PluginError> {
        let info = host.image();
        let (w, h) = (info.width as usize, info.height as usize);
        if w == 0 || h == 0 {
            return Err(PluginError::unsupported("there is no image to write"));
        }
        let decimals = request.params.int("decimals", 3).clamp(0, 9) as usize;
        let header = request.params.bool("header", true);

        let mut plane = Vec::new();
        host.read_current_plane_f32(&mut plane)?;

        let mut out = String::new();
        if header {
            out.push_str(&format!("# {w}x{h}\n"));
        }
        for (y, row) in plane.chunks(w).enumerate() {
            for (x, v) in row.iter().enumerate() {
                if x > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{v:.decimals$}"));
            }
            out.push('\n');
            // Cheap enough to check every row: `h` is a picture's height, not a
            // frame count, so this is thousands of calls at worst.
            if !host.progress((y + 1) as f32 / h as f32) {
                return Err(PluginError::unsupported("cancelled"));
            }
        }

        // Written in one go. A half-written CSV that stops mid-row looks like a
        // real file to whatever reads it next, and the host cannot tell the
        // difference either — the status it got back was `Ok`.
        std::fs::write(&request.path, out).map_err(|e| {
            PluginError::failed(format!("could not write {}: {e}", request.path.display()))
        })?;
        host.log("wrote the displayed frame as CSV");
        Ok(())
    }
}

// ------------------------------------------------------------ panic fixture

/// A plugin that panics, on purpose.
///
/// Nothing else can test the one guarantee the whole boundary rests on. Since
/// Rust 1.81 every `extern "C"` function carries an abort shim, so a panic
/// escaping a plugin's entry point kills the process **inside the plugin**,
/// before any host frame runs — no `catch_unwind` on the host side can save it.
/// The only thing that can is the guard `export_plugin!` wraps around every
/// entry point, and a guard nobody has ever fired is a guard nobody knows
/// works.
///
/// It is in the shipped example rather than hidden behind a `cfg` because the
/// test must exercise the *same* generated entry points as everything else; a
/// separately configured build would be testing different code. This crate is
/// `publish = false` and its library is a test fixture, not something a user
/// installs.
#[derive(Default)]
pub struct Panics;

impl Plugin for Panics {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.example.panics", "Panic (test fixture)")
            .menu_path("Examples")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description("Panics when run. Exists so the host's containment can be tested.")
    }

    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        vec![ParamDecl::new(
            "where",
            "Panic in",
            ParamKind::Choice {
                default: 0,
                options: vec!["run".into(), "the middle of a result".into()],
            },
        )]
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        if params.choice("where", 0) == 1 {
            // Worse than panicking straight away: the host has already been
            // handed part of a result and must discard it rather than open a
            // half-written document.
            let info = host.image();
            let mut buf = Vec::new();
            host.read_plane_f32(Plane::new(0, 0, 0), &mut buf)?;
            let _ = ImageResult {
                width: info.width,
                height: info.height,
                channels: 1,
                slices: 1,
                frames: 1,
                pixel_type: PixelType::F32,
                planes: vec![PlaneData::F32(buf)],
                channel_colors: Vec::new(),
                metadata: None,
                name: "never".into(),
            };
        }
        panic!("this plugin panics on purpose");
    }
}

fasttiff_plugin::export_plugin! {
    plugins: [Invert, ShowInfo, Panics],
    importers: [RawImport],
    exporters: [CsvExport],
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

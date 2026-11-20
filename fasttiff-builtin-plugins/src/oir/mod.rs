//! Olympus/Evident FluoView OIR, through the `oir-reader` crate.
//!
//! The container reader and the acquisition-record translator are the
//! `oir-reader` crate (MPL-2.0, published on its own), and everything about the
//! format — how a plane is scattered, how a split recording is found, how an
//! unfinished one is recovered, why the vendor's `.txt` is never read — is
//! written down there. This module is the importer: it asks the crate for the
//! acquisition, says in the log what the crate reports about the files, and
//! maps the result onto the plugin contract.
//!
//! Each part is memory-mapped where this build (`mmap`) and the platform allow,
//! and read through a bounded buffer otherwise — decided when this is compiled,
//! and again for each file as it is opened, if mapping it fails.

use fasttiff_plugin_api::{
    Confidence, DisplayMode, FileType, ImageResult, ImportHost, ImportRequest, ImportResult,
    Importer, PixelType, PlaneData, PluginError, PluginInfo, Spacing, StackInfo,
};
use oir_reader::{Access, BlockSource, ErrorKind, Options, Planes};
use std::path::Path;

#[derive(Default)]
pub struct Oir;

const IMPORTER_ID: &str = "dev.fasttiff.oir";

impl Importer for Oir {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(IMPORTER_ID, "Olympus OIR")
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
        if oir_reader::is_oir(head) {
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
        self.import_with(request, host, Access::Auto)
    }
}

impl Oir {
    /// [`Importer::import`], with how the parts are read chosen. The tests read
    /// every fixture both ways and require the same import from each.
    fn import_with(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
        access: Access,
    ) -> Result<ImportResult, PluginError> {
        // Every part of the acquisition, the named one first, each read once.
        // The bar follows bytes, and stops short for what is left after.
        let gathered = Options::new()
            .access(access)
            .gather(&request.path, |f| host.progress(0.9 * f))
            .map_err(plugin_error)?;

        let parts = gathered.parts();
        for (n, part) in parts.iter().enumerate() {
            if let Some(why) = part.map_error() {
                // Nothing the user has to do about it, but a part that was
                // read a slower way than usual should say so, in case it is
                // the reason an import took longer than the last one.
                host.log(&format!(
                    "part {} of {} could not be memory-mapped ({why}); it was read \
                     through a buffer instead",
                    n + 1,
                    parts.len()
                ));
            }
        }
        let walked = parts
            .iter()
            .filter(|p| p.block_source() == BlockSource::Walk)
            .count();
        if walked > 0 {
            // Said plainly, because the frames that are missing are the ones
            // the file does not have yet — not ones this reader declined to
            // read. Someone who expected the whole recording should know to
            // open it again when the acquisition has finished.
            host.log(&format!(
                "unfinished acquisition: {walked} of {} part(s) have no block index, so \
                 their blocks were recovered by walking the file. This is what was on \
                 disk when it was opened; open it again later for the rest.",
                parts.len()
            ));
        }
        if parts.len() > 1 {
            host.log(&format!(
                "{} part(s) of this acquisition, {} planes in total",
                parts.len(),
                gathered.planes_found()
            ));
        }

        let acquisition = gathered.assemble().map_err(plugin_error)?;
        let shape = acquisition.shape();
        host.log(&format!(
            "{}x{}, {} channel(s), {} slice(s), {} frame(s), {}-bit",
            shape.width,
            shape.height,
            shape.channels,
            shape.slices,
            shape.frames,
            shape.sample_type.bytes() * 8
        ));
        let dropped = acquisition.dropped_planes();
        if dropped > 0 {
            // Worth saying rather than silently shortening the stack: the user
            // may have expected the frame that the acquisition did not finish.
            host.log(&format!(
                "skipped {dropped} incomplete plane(s) — the acquisition was stopped part-way"
            ));
        }
        host.progress(0.95);

        let name = request
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "oir".into());

        // What tag 270 of the converted file will carry: the acquisition
        // record, written the way the acquisition software exports it. Never
        // the file's own XML — that is four megabytes in one real acquisition,
        // 3.1 MB of it three 65,536-entry lookup tables, and nothing can read
        // it as a description.
        let file_name = request
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("{name}.oir"));
        let description = acquisition.fluoview_text(&file_name);
        let record = acquisition.record();
        host.log(&match description {
            Some(_) => format!(
                "acquisition record: {} channel(s), {} event marker(s)",
                record.channels().len(),
                record.events().len()
            ),
            None => "this OIR carries nothing this reader recognises as an acquisition record"
                .to_string(),
        });

        // One pixel size for both axes, x's, as this importer has always
        // reported it. The record states y separately (`pixel_y_um`);
        // reporting it here would change every converted file with
        // non-square pixels, so it is a change to make on its own.
        let pixel = record.pixel_x_um();
        let info = StackInfo {
            name: name.clone(),
            mode: if shape.channels > 1 {
                DisplayMode::Composite
            } else {
                DisplayMode::Grayscale
            },
            unit: pixel.map(|_| "micron".to_string()),
            spacing: Spacing {
                x: pixel,
                y: pixel,
                z: record.z_step_um(),
            },
            // From the file's own frame timestamps, over the frames actually
            // imported.
            frame_interval_s: acquisition.frame_interval_s(),
            channel_names: record
                .channels()
                .iter()
                .filter_map(|c| c.label())
                .map(str::to_owned)
                .collect(),
            description: description.map(|text| format!("\nOIR metadata\n\n{text}")),
            ..Default::default()
        };

        // Moved, not copied: the crate's planes become the contract's.
        let (pixel_type, planes): (PixelType, Vec<PlaneData>) = match acquisition.into_planes() {
            Planes::U8(p) => (PixelType::U8, p.into_iter().map(PlaneData::U8).collect()),
            Planes::U16(p) => (PixelType::U16, p.into_iter().map(PlaneData::U16).collect()),
            _ => {
                return Err(PluginError::failed(
                    "this OIR's samples are of a type this importer cannot pass on",
                ))
            }
        };
        Ok(ImportResult {
            image: ImageResult {
                width: shape.width,
                height: shape.height,
                channels: shape.channels,
                slices: shape.slices,
                frames: shape.frames,
                pixel_type,
                planes,
                channel_colors: Vec::new(),
                metadata: None,
                name,
            },
            info: Some(info),
        })
    }
}

/// The crate's error as the plugin contract's.
///
/// The crate keeps the operating system's reason in `source()` rather than in
/// its message, and the user should see both, as they always have: "could not
/// open the file: <why>". A file this importer cannot read at all — not an OIR,
/// or one with no planes in it — is `unsupported`, as is a cancelled import;
/// everything else is a failure.
fn plugin_error(e: oir_reader::Error) -> PluginError {
    let text = match std::error::Error::source(&e) {
        Some(cause) => format!("{e}: {cause}"),
        None => e.to_string(),
    };
    match e.kind() {
        ErrorKind::NotOir | ErrorKind::NoPlanes | ErrorKind::Cancelled => {
            PluginError::unsupported(text)
        }
        _ => PluginError::failed(text),
    }
}

/// Export the acquisition record carried in the open document's ImageDescription.
#[derive(Default)]
pub struct OirMetadata;

impl OirMetadata {
    /// Stable identifier used by the desktop export command.
    pub const ID: &str = "dev.fasttiff.oir.metadata";

    /// Check before prompting for a destination.
    pub fn check_source(
        importer: Option<&str>,
        description: Option<&str>,
    ) -> Result<(), &'static str> {
        if importer != Some(IMPORTER_ID) {
            return Err("No OIR file is open. There is nothing to export.");
        }
        if !description.is_some_and(|d| d.lines().any(|l| l == "OIR metadata")) {
            return Err("This OIR file contains no acquisition metadata to export.");
        }
        Ok(())
    }
}

impl fasttiff_plugin_api::Exporter for OirMetadata {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(Self::ID, "Export OIR metadata")
            .description("Save the FluoView acquisition record as text.")
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("OIR metadata", &["txt"])]
    }

    fn export(
        &mut self,
        request: &fasttiff_plugin_api::ExportRequest,
        host: &mut dyn fasttiff_plugin_api::HostContext,
    ) -> Result<(), PluginError> {
        let record = acquisition_text(host.stack_info().description.as_deref().unwrap_or_default())
            .ok_or_else(|| PluginError::unsupported("This document has no OIR metadata."))?;
        std::fs::write(&request.path, record)
            .map_err(|e| PluginError::failed(format!("writing {}: {e}", request.path.display())))
    }
}

fn acquisition_text(description: &str) -> Option<String> {
    let normalized = description.replace("\r\n", "\n");
    let record = if let Some((_, record)) = normalized.split_once("\nOIR metadata\n\n") {
        record
    } else {
        // Imports written before the heading have the same FluoView preamble.
        let start = normalized.find("\"[General]\"\t\"\"\n\"Name\"\t\"")?;
        let record = &normalized[start..];
        let filename = record
            .lines()
            .nth(1)?
            .strip_prefix("\"Name\"\t\"")?
            .strip_suffix('"')?;
        if !filename.to_ascii_lowercase().ends_with(".oir") {
            return None;
        }
        record
    };
    let record = record.trim_end_matches(['\0', '\n']);
    if record.is_empty() {
        return None;
    }
    let text = format!("{record}\n");
    Some(if cfg!(windows) {
        text.replace('\n', "\r\n")
    } else {
        text
    })
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

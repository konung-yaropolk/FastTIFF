//! Importers: plugins that turn a file the app cannot read into one it can.
//!
//! This is the plugin type with the least ambiguity about when it runs — a file
//! extension decides — and the most obvious value for a microscopy tool, where
//! every instrument vendor has its own container. It is also the one type that
//! runs when **nothing is open**, which is why it does not take a
//! [`HostContext`](crate::HostContext): there is no stack to give it.

use crate::meta::StackInfo;
use crate::params::{ParamDecl, Params};
use crate::plugin::{ImageResult, PluginError, PluginInfo};
use std::path::Path;

/// A family of files an importer handles, as the Open dialog shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileType {
    /// What to call it in the dialog: `"Olympus OIR"`.
    pub description: String,
    /// Extensions, lowercase and without the dot: `["oir"]`.
    pub extensions: Vec<String>,
}

impl FileType {
    pub fn new(description: impl Into<String>, extensions: &[&str]) -> Self {
        FileType {
            description: description.into(),
            extensions: extensions
                .iter()
                .map(|e| e.trim_start_matches('.').to_lowercase())
                .collect(),
        }
    }

    pub fn matches(&self, path: &Path) -> bool {
        path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            let e = e.to_lowercase();
            self.extensions.contains(&e)
        })
    }
}

/// How sure an importer is that it can read a file.
///
/// Extensions collide — `.tif` most of all — so an extension match alone cannot
/// decide which importer runs. Each candidate is shown the file's opening bytes
/// and says how confident it is; the host picks the most confident, and asks the
/// user only when two are equally sure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Definitely not this importer's format. It will not be offered.
    No,
    /// The extension matches but the content was not checked, or was
    /// inconclusive. The usual answer for a format with no magic number.
    Maybe,
    /// A magic number (or equivalent) matched. Beats `Maybe`.
    Certain,
}

/// What the host asks an importer to read.
#[derive(Clone, Debug)]
pub struct ImportRequest {
    /// The file. Always a real path — an importer is the one plugin type that
    /// is handed a file rather than an open stack.
    pub path: std::path::PathBuf,
    /// The dialog values, already clamped to what [`Importer::params`]
    /// declared. Empty when the importer declared no dialog.
    pub params: Params,
}

/// Progress reporting for an import. Deliberately smaller than
/// [`HostContext`](crate::HostContext): there is no stack to read.
pub trait ImportHost {
    /// Report progress, `0.0..=1.0`. Returns `false` if the user cancelled.
    fn progress(&mut self, fraction: f32) -> bool;
    /// A line for the host to show.
    fn log(&mut self, message: &str);
}

/// What an importer produced.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportResult {
    pub image: ImageResult,
    /// Metadata read from the file: channel names, spacing, calibration, the
    /// display mode the instrument intended. `None` lets the host derive what
    /// it can from the filename and the pixels.
    pub info: Option<StackInfo>,
}

/// A plugin that reads a file format the application does not know.
///
/// The host indexes every installed importer at startup — from its manifest,
/// without loading any code — so the extensions it handles can be added to the
/// Open dialog and to drag-and-drop before the plugin has ever run.
pub trait Importer: Send {
    fn info(&self) -> PluginInfo;

    /// The file types this importer offers to read.
    fn file_types(&self) -> Vec<FileType>;

    /// How sure this importer is, given the file's first bytes.
    ///
    /// `head` is the first few kilobytes, or fewer for a short file. The
    /// default answers `Maybe` on an extension match, which is right for a
    /// format with no magic number; override it when there is one.
    fn probe(&self, path: &Path, _head: &[u8]) -> Confidence {
        if self.file_types().iter().any(|t| t.matches(path)) {
            Confidence::Maybe
        } else {
            Confidence::No
        }
    }

    /// A dialog, when the format cannot be read without one — raw binary needs
    /// its dimensions, for instance. Return an empty list to import with no
    /// dialog, which is what most formats should do: a dialog on every open of
    /// a known format is an irritation, not a feature.
    fn params(&self, _path: &Path) -> Vec<ParamDecl> {
        Vec::new()
    }

    /// Read the file.
    ///
    /// Runs on a worker thread. Report progress through `host` and stop when it
    /// returns `false`.
    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError>;
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;

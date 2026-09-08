//! Exporters: plugins that write the open stack in a format the app cannot.
//!
//! The mirror of an [`Importer`](crate::Importer), and deliberately the simpler
//! half. An importer has to hand back an image the host then owns, so its
//! result crosses the boundary as pixels and metadata. An exporter is given the
//! open stack and a path, and writes the file itself — nothing comes back but
//! success or a reason.
//!
//! That difference is why an exporter takes a [`HostContext`]: it reads planes,
//! it wants the contrast windows and LUTs the user is looking at, and unlike an
//! importer it only ever runs when something is open.
//!
//! # What it is for
//!
//! Not "save this stack faithfully" — the host does that itself, in TIFF, and a
//! format that cannot hold a 4D 16-bit hyperstack has no business pretending
//! to. An exporter is for handing the data to something else: a figure, a
//! movie, a colleague's software. Losing the axes is usually the point.

use crate::import::FileType;
use crate::params::{ParamDecl, Params};
use crate::plugin::{PluginError, PluginInfo};
use crate::HostContext;
use std::path::PathBuf;

/// What the host asks an exporter to write.
#[derive(Clone, Debug)]
pub struct ExportRequest {
    /// Where to write. The host has already applied the chosen format's
    /// extension, and has *not* checked whether the file exists — the save
    /// dialog asked that question, in the platform's own words.
    pub path: PathBuf,
    /// The dialog values, already clamped to what [`Exporter::params`]
    /// declared. Empty when the exporter declared no dialog.
    pub params: Params,
}

/// A plugin that writes a format the application does not know.
///
/// The formats it declares become rows in the Save-as dialog, alongside the
/// host's own TIFF. Which exporter runs is decided by the extension the user
/// ends up with — unlike opening, where the content has a say, because a file
/// that does not exist yet cannot be probed.
pub trait Exporter: Send {
    fn info(&self) -> PluginInfo;

    /// The file types this exporter offers to write.
    ///
    /// An exporter that declares none can never be reached, and the host says
    /// so rather than installing it.
    fn file_types(&self) -> Vec<FileType>;

    /// A dialog, when the format has a real choice to offer — a quality, a bit
    /// depth. Return an empty list to export without one, which is what most
    /// formats should do.
    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        Vec::new()
    }

    /// Write the file.
    ///
    /// Report progress through `host` and stop when it returns `false`; a
    /// cancelled export should leave nothing behind that looks finished.
    fn export(
        &mut self,
        request: &ExportRequest,
        host: &mut dyn HostContext,
    ) -> Result<(), PluginError>;
}

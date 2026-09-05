//! The plugin host: what is installed, and what running one means.
//!
//! Lives in the viewer core rather than the app because it needs the stack, not
//! the window — and because the browser build gets the built-in plugins for
//! free that way. The app supplies only the dialog and the menu.
//!
//! The `.dll`/`.so` lane is not here yet. When it lands it adds entries to the
//! same [`Registry`] and implements the same
//! [`Plugin`](fasttiff_plugin_api::Plugin) trait, so nothing above this module
//! learns that a plugin came from a file.

pub mod builtin;
pub mod discover;
pub mod host;
pub mod netpbm;
pub mod result;

pub use discover::{install_dir, is_library, search_paths, user_plugin_dir, LIBRARY_EXT};
pub use host::{describe_image, describe_stack, describe_view, describe_volume, StackHost};
pub use result::{to_stack, to_tiff_bytes};

use fasttiff_plugin_api::{Confidence, FileType, Importer, Plugin, PluginInfo};
use std::path::Path;

/// Where a plugin came from. Shown in the menu so a user can tell a built-in
/// from something they installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Compiled into the application.
    BuiltIn,
    /// Loaded from a shared library in the plugins folder.
    Library,
}

/// One installed importer.
pub struct ImporterEntry {
    pub info: PluginInfo,
    pub origin: Origin,
    pub file_types: Vec<FileType>,
    pub importer: Box<dyn Importer>,
}

/// One installed plugin.
pub struct Entry {
    pub info: PluginInfo,
    pub origin: Origin,
    pub plugin: Box<dyn Plugin>,
}

/// Everything installed, in menu order.
pub struct Registry {
    entries: Vec<Entry>,
    importers: Vec<ImporterEntry>,
    /// Problems found while indexing — a library that would not load, two
    /// plugins claiming one id. Surfaced rather than swallowed: a plugin the
    /// user installed and cannot find is worse than one that says why.
    pub problems: Vec<String>,
}

impl Default for Registry {
    fn default() -> Self {
        Registry::new()
    }
}

impl Registry {
    /// The built-ins only. External lanes add to this.
    pub fn new() -> Self {
        let mut reg = Registry {
            entries: Vec::new(),
            importers: Vec::new(),
            problems: Vec::new(),
        };
        for p in builtin::all() {
            reg.add(p, Origin::BuiltIn);
        }
        reg.add_importer(Box::new(netpbm::Netpbm), Origin::BuiltIn);
        reg.sort();
        reg
    }

    /// Install one importer, refusing a duplicate id for the reason
    /// [`add`](Self::add) gives.
    pub fn add_importer(&mut self, importer: Box<dyn Importer>, origin: Origin) -> bool {
        let info = importer.info();
        if info.id.trim().is_empty() {
            self.problems
                .push(format!("importer \"{}\" has no id; ignored", info.name));
            return false;
        }
        if let Some(e) = self.importers.iter().find(|e| e.info.id == info.id) {
            self.problems.push(format!(
                "two importers claim the id \"{}\": \"{}\" and \"{}\"; keeping the first",
                info.id, e.info.name, info.name
            ));
            return false;
        }
        let file_types = importer.file_types();
        if file_types.is_empty() {
            self.problems.push(format!(
                "importer \"{}\" declares no file types, so nothing could ever reach it; ignored",
                info.name
            ));
            return false;
        }
        self.importers.push(ImporterEntry {
            info,
            origin,
            file_types,
            importer,
        });
        true
    }

    pub fn importers(&self) -> &[ImporterEntry] {
        &self.importers
    }

    pub fn importer_mut(&mut self, index: usize) -> Option<&mut ImporterEntry> {
        self.importers.get_mut(index)
    }

    /// Every file type any importer offers, for the Open dialog's filter list.
    ///
    /// Deduplicated by extension set: two importers for the same vendor format
    /// should not produce two identical rows in a file dialog.
    pub fn open_file_types(&self) -> Vec<FileType> {
        let mut out: Vec<FileType> = Vec::new();
        for e in &self.importers {
            for t in &e.file_types {
                if !out.iter().any(|o| o.extensions == t.extensions) {
                    out.push(t.clone());
                }
            }
        }
        out
    }

    /// Whether any importer claims this path by extension. Opens no file, so it
    /// is safe to call while handling a drag-and-drop.
    pub fn claims_extension(&self, path: &Path) -> bool {
        self.importers
            .iter()
            .any(|e| e.file_types.iter().any(|t| t.matches(path)))
    }

    /// Which importer should read `path`, best first.
    ///
    /// Extensions collide — `.tif` most of all — so the choice cannot be made
    /// from the name alone. Each candidate is shown the file's opening bytes and
    /// ranked by how sure it is; equal confidence keeps installation order,
    /// which puts a user's own plugin ahead of a system-wide one because the
    /// search path is ordered that way.
    pub fn importers_for(&self, path: &Path, head: &[u8]) -> Vec<(usize, Confidence)> {
        let mut ranked: Vec<(usize, Confidence)> = self
            .importers
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.importer.probe(path, head)))
            .filter(|(_, c)| *c > Confidence::No)
            .collect();
        // Stable and reversed, so equal confidence preserves installation
        // order while the most confident importer comes first.
        ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        ranked
    }

    /// The first few kilobytes of a file, for [`importers_for`](Self::importers_for).
    ///
    /// An unreadable file yields an empty slice, which the probes treat as "no
    /// evidence" and answer from the extension instead — the right behaviour
    /// for a file on a slow share that a dialog must not block on.
    pub fn read_head(path: &Path) -> Vec<u8> {
        use std::io::Read;
        let mut buf = vec![0u8; 4096];
        match std::fs::File::open(path).and_then(|mut f| f.read(&mut buf)) {
            Ok(n) => {
                buf.truncate(n);
                buf
            }
            Err(_) => Vec::new(),
        }
    }

    /// Install one plugin, refusing a duplicate id.
    ///
    /// Two plugins with the same id is an installation error the host reports
    /// rather than resolves: picking one silently would make which of them runs
    /// depend on directory order, which is the kind of thing that wastes an
    /// afternoon.
    pub fn add(&mut self, plugin: Box<dyn Plugin>, origin: Origin) -> bool {
        let info = plugin.info();
        if info.id.trim().is_empty() {
            self.problems
                .push(format!("plugin \"{}\" has no id; ignored", info.name));
            return false;
        }
        if let Some(existing) = self.entries.iter().find(|e| e.info.id == info.id) {
            self.problems.push(format!(
                "two plugins claim the id \"{}\": \"{}\" ({:?}) and \"{}\" ({:?}); keeping the first",
                info.id, existing.info.name, existing.origin, info.name, origin
            ));
            return false;
        }
        self.entries.push(Entry {
            info,
            origin,
            plugin,
        });
        true
    }

    fn sort(&mut self) {
        self.entries.sort_by(|a, b| {
            (&a.info.menu_path, &a.info.name).cmp(&(&b.info.menu_path, &b.info.name))
        });
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Entry> {
        self.entries.get_mut(index)
    }

    pub fn find(&self, id: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.info.id == id)
    }

    /// Menu entries grouped by their `menu_path`, in display order. An empty
    /// path means the top level.
    pub fn grouped(&self) -> Vec<(&str, Vec<(usize, &PluginInfo)>)> {
        let mut groups: Vec<(&str, Vec<(usize, &PluginInfo)>)> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let path = e.info.menu_path.as_str();
            match groups.iter_mut().find(|(p, _)| *p == path) {
                Some((_, v)) => v.push((i, &e.info)),
                None => groups.push((path, vec![(i, &e.info)])),
            }
        }
        groups
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

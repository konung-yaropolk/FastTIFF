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
pub mod host;

pub use host::{describe_image, describe_stack, describe_view, describe_volume, StackHost};

use fasttiff_plugin_api::{Plugin, PluginInfo};

/// Where a plugin came from. Shown in the menu so a user can tell a built-in
/// from something they installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Compiled into the application.
    BuiltIn,
    /// Loaded from a shared library in the plugins folder.
    Library,
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
            problems: Vec::new(),
        };
        for p in builtin::all() {
            reg.add(p, Origin::BuiltIn);
        }
        reg.sort();
        reg
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

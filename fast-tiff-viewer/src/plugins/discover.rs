//! Where plugins live.
//!
//! A single `plugins/` folder beside the executable is the Windows habit, and
//! it is the wrong answer everywhere else: on macOS the app is a signed bundle
//! and writing into it breaks the signature, and on Linux a packaged binary
//! lands in `/usr/bin` with no writable directory anywhere near it. So there is
//! no single location — there is a search path, and every entry on it earns its
//! place:
//!
//! | | why |
//! |---|---|
//! | `$FASTTIFF_PLUGIN_PATH` | developers, CI, and a lab pointing every workstation at one shared folder |
//! | the user's data directory | the only place a normal user can always write, and it survives reinstalling the app |
//! | beside the executable | portable installs — an unzipped folder on a USB stick, which is how a lot of instrument PCs run software |
//! | the system directory | what a `.deb`/`.rpm` can write to, shared by every user on the machine |
//!
//! Earlier entries win: a plugin in the user's own directory shadows a
//! system-wide one of the same id, so a user can override an
//! administrator-installed plugin without needing an administrator.
//!
//! Nothing here touches the filesystem beyond asking whether a directory
//! exists, and nothing is loaded: indexing reads manifests, and a shared
//! library is opened only when its plugin is actually run.

use std::path::PathBuf;

/// The environment variable that prepends to the search path. Entries are
/// separated the way `PATH` is on the platform: `;` on Windows, `:` elsewhere.
pub const PLUGIN_PATH_VAR: &str = "FASTTIFF_PLUGIN_PATH";

/// The directory name used under the user and system data directories.
const APP_DIR: &str = "FastTIFF";

/// Every directory to look in, highest priority first, deduplicated.
///
/// Directories that do not exist are kept rather than filtered out: the caller
/// shows this list in the UI so a user can see *where to put a plugin*, and a
/// list that hid the empty ones would be useless for exactly the person who has
/// not installed one yet.
pub fn search_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !p.as_os_str().is_empty() && !out.contains(&p) {
            out.push(p);
        }
    };

    // 1. Explicit override.
    if let Some(val) = std::env::var_os(PLUGIN_PATH_VAR) {
        for p in std::env::split_paths(&val) {
            push(p);
        }
    }

    // 2. The user's own directory — writable without privileges, and it
    //    outlives reinstalling the application.
    if let Some(p) = user_plugin_dir() {
        push(p);
    }

    // 3. Beside the executable, for a portable install.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            push(dir.join("plugins"));
            // A macOS bundle puts the binary in `Contents/MacOS`; the
            // conventional place for loadable code is `Contents/PlugIns`
            // alongside it. Harmless on other platforms, where that path
            // simply will not exist.
            if let Some(contents) = dir.parent() {
                push(contents.join("PlugIns"));
            }
        }
    }

    // 4. System-wide, for a distribution package.
    for p in system_plugin_dirs() {
        push(p);
    }

    out
}

/// The per-user plugin directory, following each platform's convention.
pub fn user_plugin_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        // Roaming, so a plugin follows the user across machines on a domain —
        // which is common in an institutional lab.
        std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join(APP_DIR).join("plugins"))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(|h| {
            PathBuf::from(h)
                .join("Library")
                .join("Application Support")
                .join(APP_DIR)
                .join("PlugIns")
        })
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        // XDG: `$XDG_DATA_HOME` if set, else `~/.local/share`. Lowercase
        // directory name, as is conventional on Linux.
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
            })
            .map(|d| d.join("fasttiff").join("plugins"))
    }
}

/// System-wide plugin directories, for packaged installs.
pub fn system_plugin_dirs() -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("PROGRAMDATA")
            .map(|a| vec![PathBuf::from(a).join(APP_DIR).join("plugins")])
            .unwrap_or_default()
    }
    #[cfg(target_os = "macos")]
    {
        vec![PathBuf::from("/Library/Application Support")
            .join(APP_DIR)
            .join("PlugIns")]
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        // `$XDG_DATA_DIRS` when set, else the two standard prefixes.
        match std::env::var_os("XDG_DATA_DIRS") {
            Some(dirs) => std::env::split_paths(&dirs)
                .filter(|p| p.is_absolute())
                .map(|d| d.join("fasttiff").join("plugins"))
                .collect(),
            None => vec![
                PathBuf::from("/usr/local/share/fasttiff/plugins"),
                PathBuf::from("/usr/share/fasttiff/plugins"),
            ],
        }
    }
}

/// The directory the UI should offer to open when the user asks where plugins
/// go: the first writable candidate, created if it does not exist.
pub fn install_dir() -> Option<PathBuf> {
    let dir = user_plugin_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The file extension a loadable plugin has on this platform.
pub const LIBRARY_EXT: &str = if cfg!(target_os = "windows") {
    "dll"
} else if cfg!(target_os = "macos") {
    "dylib"
} else {
    "so"
};

/// Whether a filename looks like a plugin library for this platform.
pub fn is_library(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(LIBRARY_EXT))
}

#[cfg(test)]
#[path = "discover_tests.rs"]
mod tests;

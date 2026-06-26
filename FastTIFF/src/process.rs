//! Opening several files at once. FastTIFF is one-stack-per-window, so when
//! multiple files arrive together — passed on the command line (e.g. selecting
//! several and choosing "Open with"), or dropped onto the window in one go — the
//! first opens in the current process and each of the rest is launched as its
//! own independent viewer process, so they all appear side by side.

use std::path::Path;
use std::process::Command;

/// Launch a new instance of this executable to view `path`. The child is a fully
/// independent process (we don't wait on it), so it keeps running if this one
/// exits. Failures are logged and otherwise ignored — one extra file that won't
/// open shouldn't disturb the files that did.
///
/// Each child receives exactly one path, so it never spawns further processes
/// of its own — the fan-out is one level deep.
pub fn open_in_new_process(path: &Path) {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            log::error!("can't locate current executable to open {}: {e}", path.display());
            return;
        }
    };
    match Command::new(exe).arg(path).spawn() {
        Ok(_child) => {} // dropped: not awaited, runs independently
        Err(e) => log::error!("failed to open {} in a new process: {e}", path.display()),
    }
}

/// Open every file in `paths` at once: each entry past the first is launched in
/// its own process here, and the first (if any) is returned for the caller to
/// open in the current process.
pub fn open_all(paths: &[std::path::PathBuf]) -> Option<&std::path::PathBuf> {
    let (first, rest) = paths.split_first()?;
    for extra in rest {
        open_in_new_process(extra);
    }
    Some(first)
}

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
    if let Err(e) = try_open_in_new_process(path) {
        log::error!("failed to open {} in a new process: {e}", path.display());
    }
}

/// [`open_in_new_process`], for a caller that has something to do about a
/// failure — a plugin result has nowhere else to go, and showing it in this
/// window beats losing it.
pub fn try_open_in_new_process(path: &Path) -> std::io::Result<()> {
    let exe = std::env::current_exe()?;
    Command::new(exe).arg(path).spawn().map(|_child| ())
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

/// The flag a viewer is launched with when its stack arrives down a pipe rather
/// than from a file. Followed by the name to show for it.
///
/// Deliberately not a path: a file called `--from-stdin` would have to be passed
/// as `.\--from-stdin` to be opened, and the alternative — a flag that could be
/// mistaken for one — is worse.
pub const FROM_STDIN: &str = "--from-stdin";

/// Hand `bytes` to a new viewer process without going through a file.
///
/// The child reads its whole stack from standard input, so a plugin result
/// never touches the disk: a stabilised timelapse is the same size as the
/// recording it came from, and writing a gigabyte to a temporary file only to
/// read it straight back is the slowest part of a run that had nothing else
/// left to do.
///
/// The write is chunked so `on_progress` has something to report — it is the
/// child's read as much as this process's write, the pipe having no room to
/// buffer a stack — and returning `false` from it kills the half-fed child
/// rather than leaving a window that would open onto a truncated file.
///
/// Errors are the caller's cue to fall back to a file. Any of them can happen
/// for reasons that have nothing to do with the result: the child could fail to
/// start, or die before it has read everything, which shows up here as a broken
/// pipe.
pub fn open_bytes_in_new_process(
    bytes: &[u8],
    name: &str,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> std::io::Result<()> {
    let exe = std::env::current_exe()?;
    let mut child = Command::new(exe)
        .arg(FROM_STDIN)
        .arg(name)
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    feed(&mut child, bytes, on_progress)
}

/// Write `bytes` into `child`'s standard input and close it.
///
/// Split from the spawn so it can be tested against a child that is not a
/// viewer: what has to be right here is that every byte arrives, in order, and
/// that the pipe is then closed — a stack that is one chunk short is a file
/// that does not parse.
fn feed(
    child: &mut std::process::Child,
    bytes: &[u8],
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> std::io::Result<()> {
    use std::io::Write;

    let mut stdin = match child.stdin.take() {
        Some(pipe) => pipe,
        None => {
            let _ = child.kill();
            return Err(std::io::Error::other(
                "the new process has no standard input",
            ));
        }
    };

    // Big enough that the syscall overhead disappears, small enough that the
    // bar moves on a stack of any size.
    const CHUNK: usize = 8 << 20;
    let total = bytes.len().max(1) as f32;
    let mut written = 0usize;
    for chunk in bytes.chunks(CHUNK) {
        if let Err(e) = stdin.write_all(chunk) {
            drop(stdin);
            let _ = child.kill();
            return Err(e);
        }
        written += chunk.len();
        if !on_progress(written as f32 / total) {
            // Cancelled with the child part-fed. Closing the pipe would leave it
            // opening a truncated stack, so it goes with the work it was for.
            drop(stdin);
            let _ = child.kill();
            return Err(std::io::Error::other("cancelled"));
        }
    }
    // The end of the pipe is the end of the file: the child reads to EOF, and
    // without this it would wait for the rest of a stack it already has.
    drop(stdin);
    // An empty result still has to report something, or a bar that was moving
    // stops wherever it was.
    on_progress(1.0);
    Ok(())
}

/// Whether a result of `size` bytes should be handed over in memory.
///
/// Three copies' worth is asked for, not one. While the handover runs this
/// process still holds the encoded result and the child is building its own,
/// which is two; the third is headroom, so that opening a window does not take
/// the machine to the edge of its memory and make everything else on it swap.
///
/// A result too big for that goes through a file, which is not a consolation
/// prize: the new window memory-maps a file and lets the operating system page
/// it, and for a stack that large that is the better way to hold it anyway.
pub fn fits_in_memory(size: usize) -> bool {
    match available_memory() {
        Some(available) => (size as u64).saturating_mul(3) <= available,
        // Nothing was willing to say. Small results still avoid the disk;
        // anything substantial takes the path that does not have to guess.
        None => size <= 512 << 20,
    }
}

/// Physical memory that could be allocated right now, in bytes.
///
/// `None` where this platform is not asked rather than where the question
/// failed — see [`fits_in_memory`], which treats not knowing as a reason to be
/// careful rather than as an error.
fn available_memory() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: `GlobalMemoryStatusEx` fills a `MEMORYSTATUSEX` whose
        // `dwLength` says how big it is, which is the whole of the contract.
        // The struct is plain data and lives on this stack for the call.
        unsafe {
            let mut status: MEMORYSTATUSEX = std::mem::zeroed();
            status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut status) != 0 {
                return Some(status.ullAvailPhys);
            }
        }
        None
    }
    // `MemAvailable` rather than `MemFree`: the kernel's own estimate of what a
    // new process could get, which counts reclaimable cache. `MemFree` on a
    // machine that has been up a while reads as almost nothing.
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib.saturating_mul(1024))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

#[cfg(test)]
#[path = "process_tests.rs"]
mod tests;

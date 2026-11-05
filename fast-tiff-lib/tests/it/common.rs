//! Shared helpers for the integration suites.
//!
//! Not a test binary: Cargo only builds top-level `tests/*.rs` as test targets,
//! so a subdirectory module is the way to share code between them.

#![allow(dead_code)]

use fast_tiff_lib::TiffStack;
use std::path::Path;

/// Open a TIFF from a path, whichever way this build can.
///
/// The suites that use this write a temp file (or read a fixture) and then open
/// it. They were gated behind `#![cfg(feature = "mmap")]` purely because
/// [`TiffStack::open`] is — which conflated "no memory mapping" with "no
/// filesystem". The `--no-default-features` build still has `std::fs`; it just
/// has no `Mmap`. The gate therefore cost that configuration — the shape the
/// browser ships — every one of those tests, and it did so silently, because a
/// test binary containing zero tests still prints `ok`.
///
/// Reading the bytes and going in through `TiffStack::from_bytes` exercises the
/// same index and the same decoders. Where `mmap` is available the mapped path
/// is still the one taken, so nothing is given up.
///
/// Takes `impl AsRef<Path>` so it is a drop-in for the `TiffStack::open` calls
/// it replaced, which accepted both `&Path` and `PathBuf`. The error is
/// flattened to a `String` — through anyhow's `{:#}`, so the cause chain
/// survives — to keep the suites free of a dev-dependency on anyhow.
pub fn open_tiff(path: impl AsRef<Path>) -> Result<TiffStack, String> {
    let path = path.as_ref();
    #[cfg(feature = "mmap")]
    {
        TiffStack::open(path).map_err(|e| format!("{e:#}"))
    }
    #[cfg(not(feature = "mmap"))]
    {
        let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        TiffStack::from_bytes(bytes).map_err(|e| format!("{e:#}"))
    }
}

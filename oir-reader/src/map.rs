//! Memory-mapping a part. One of the crate's two places allowed `unsafe`; the
//! other is the C interface, `capi.rs`, when the `capi` feature builds it.

use std::fs::File;
use std::io;

/// Map `file` whole, as it stands now.
#[allow(unsafe_code)]
pub(crate) fn map(file: &File) -> io::Result<memmap2::Mmap> {
    #[cfg(test)]
    if FAIL_NEXT_MAP.with(|f| f.replace(false)) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "mapping refused by the test",
        ));
    }
    // SAFETY: the standard caveat of a mapping — the bytes under it must not
    // change. They do not: a mapping's length is fixed when it is made, and an
    // acquisition only ever *appends*, so a file still being written is read as
    // the snapshot it was at this moment and every byte in that snapshot is one
    // already written and never rewritten. That is what lets this reader open
    // an unfinished recording at all.
    //
    // The bargain would be broken by the file being *truncated* while mapped,
    // which is not something an acquisition does — and a reader that must
    // survive it anyway asks for `Access::Buffered`. It is the same bargain the
    // TIFF reader in `fast-tiff-lib` makes.
    unsafe { memmap2::Mmap::map(file) }
}

#[cfg(test)]
thread_local! {
    /// Set to make this thread's next [`map`] fail, as a mapping on a
    /// filesystem that does not support one would.
    pub(crate) static FAIL_NEXT_MAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

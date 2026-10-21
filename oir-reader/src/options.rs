//! How a read is done.

use crate::acquisition::Gathered;
use crate::error::Error;
use crate::read::{acquisition_parts, gather_inputs, PartInput};
use std::path::Path;

/// How to read an acquisition.
///
/// ```no_run
/// use oir_reader::{Access, Options};
///
/// // A recording on a network share: no mapping, so a dropped connection is
/// // an error rather than a crash.
/// let acquisition = Options::new()
///     .access(Access::Buffered)
///     .gather("recording.oir", |done| {
///         eprintln!("{:.0}%", done * 100.0);
///         true
///     })?
///     .assemble()?;
/// # Ok::<(), oir_reader::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    access: Access,
}

/// How the parts of an acquisition named by a path are read.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Access {
    /// Memory-map each part where this build can — see
    /// [`MMAP_SUPPORTED`](crate::MMAP_SUPPORTED) — and read it through a
    /// bounded buffer where it cannot. That is decided when the crate is
    /// compiled, and again for each part as it is opened: a part that cannot be
    /// mapped is read through the buffer instead, and
    /// [`Part::map_error`](crate::Part::map_error) says why.
    #[default]
    Auto,
    /// Never map; read every part through a bounded buffer.
    ///
    /// For a file on a network share, or one another process may truncate:
    /// under a mapping, either turns an I/O error into a crash.
    Buffered,
}

impl Options {
    /// The defaults: [`Access::Auto`].
    pub fn new() -> Self {
        Options::default()
    }

    /// How parts named by a path are read.
    #[must_use]
    pub fn access(mut self, access: Access) -> Self {
        self.access = access;
        self
    }

    /// The first phase of a read, which is all of its I/O: every part of the
    /// acquisition `first` begins (see [`acquisition_parts`]) read once, front
    /// to back, its planes reassembled and its record translated.
    ///
    /// `progress` is told the fraction of the acquisition's bytes done, from 0
    /// to 1, every thousand or so blocks and at the end of each part; it may
    /// step back while a part is read a second time. Returning `false` stops
    /// the read with [`ErrorKind::Cancelled`](crate::ErrorKind::Cancelled).
    ///
    /// [`acquisition_parts`]: crate::acquisition_parts
    pub fn gather(
        &self,
        first: impl AsRef<Path>,
        mut progress: impl FnMut(f32) -> bool,
    ) -> Result<Gathered, Error> {
        let inputs = acquisition_parts(first)
            .into_iter()
            .map(PartInput::Path)
            .collect();
        gather_inputs(inputs, self.access, true, &mut progress)
    }

    /// [`gather`](Options::gather), over parts already in memory, in order.
    ///
    /// Every byte is read in place. [`Access`] does not apply: each part's
    /// [`ReadPath`](crate::ReadPath) is `Memory`.
    pub fn gather_bytes(
        &self,
        parts: &[&[u8]],
        mut progress: impl FnMut(f32) -> bool,
    ) -> Result<Gathered, Error> {
        let inputs = parts.iter().map(|b| PartInput::Bytes(b)).collect();
        gather_inputs(inputs, self.access, false, &mut progress)
    }
}

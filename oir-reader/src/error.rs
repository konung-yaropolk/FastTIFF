//! Why a read failed.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// Why an acquisition could not be read.
///
/// The message says what is wrong with *the file*, in words a person holding
/// it can act on — these end up in front of whoever opened it. [`kind`] is for
/// a program deciding what to do about it. An I/O failure carries the operating
/// system's error as its [`source`](std::error::Error::source), and the message
/// does not repeat it.
///
/// [`kind`]: Error::kind
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    path: Option<PathBuf>,
    source: Option<io::Error>,
}

/// What kind of failure an [`Error`] is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// A part could not be opened or read.
    Io,
    /// Not an OIR file at all: the signature is missing.
    NotOir,
    /// Structure or metadata that is damaged, truncated, contradictory or
    /// unrecognisable. The message says which.
    Malformed,
    /// Nothing in the acquisition is an image plane this reader recognises —
    /// usually because the parts that hold the planes are not there to read.
    NoPlanes,
    /// The progress callback asked to stop.
    Cancelled,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
            path: None,
            source: None,
        }
    }

    pub(crate) fn malformed(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Malformed, message)
    }

    pub(crate) fn io(message: impl Into<String>, source: io::Error) -> Self {
        Error {
            source: Some(source),
            ..Error::new(ErrorKind::Io, message)
        }
    }

    pub(crate) fn cancelled() -> Self {
        Error::new(ErrorKind::Cancelled, "cancelled")
    }

    /// Name the part being read, unless the error already names one.
    pub(crate) fn at_path(mut self, path: &Path) -> Self {
        if self.path.is_none() {
            self.path = Some(path.to_path_buf());
        }
        self
    }

    /// What kind of failure this is.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The part that was being read when it happened, when the acquisition
    /// was read from files.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// A read failed part-way.
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::io("could not read the file", e)
    }
}

/// The file ends before bytes it points to.
pub(crate) fn truncated(at: u64, len: usize) -> Error {
    Error::malformed(format!(
        "this OIR ends before the {len} byte(s) it points to at {at:#x}"
    ))
}

//! Why a call failed: [`Error`], and the [`ErrorKind`] a program decides on.
//!
//! The crate-internal half is how the rest of the crate makes them: the
//! [`bail!`] and [`err!`] macros for an error of its own, and the [`Context`]
//! and [`OptionContext`] traits for saying what a step that failed was doing.

use std::borrow::Cow;
use std::fmt;
use std::io;

/// [`std::result::Result`] with this crate's [`Error`] as the error unless
/// another is named.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a read or a write failed.
///
/// The message says what is wrong in words a person can act on; it is what
/// ends up in front of whoever opened the file. [`kind`](Error::kind) is for a
/// program deciding what to do about it — handing a file that is
/// [`NotTiff`](ErrorKind::NotTiff) to another reader, say — so that nothing
/// has to match on the words, which are free to change.
///
/// An error can wrap another: the outer one says what was being done, the
/// inner one why it failed, and every link has the kind of the innermost.
/// [`source`](std::error::Error::source) walks the links outermost first.
/// Where the crate says what it was doing when the operating system failed it
/// (`could not open …`), the [`io::Error`] it reported is the next link; a
/// read or a write that fails mid-stream reads as that `io::Error` does.
/// Either way [`io_error`](Error::io_error) hands it back.
/// `{}` prints the outermost link, `{:#}` the whole chain joined by `": "` —
/// the same text `anyhow`'s `{:#}` prints for an error converted into one.
///
/// One pointer wide, so a `Result` of it costs a word more than its value at
/// most, and nothing more at all for `Result<()>`. Unwind-safe, as `anyhow`'s
/// is: nothing in it changes once it is made.
pub struct Error(Box<Inner>);

// An `io::Error` can hold a `dyn Error`, which the compiler cannot see is free
// of interior mutability, so these are not derived. An `Error` is never
// changed once built: a panic cannot leave one half-updated.
impl std::panic::UnwindSafe for Error {}
impl std::panic::RefUnwindSafe for Error {}

struct Inner {
    kind: ErrorKind,
    /// What this link says. `None` for an I/O error converted as it stands,
    /// which says only what the operating system said: its words are this
    /// link's, so that converting one does not print the reason twice.
    message: Option<Cow<'static, str>>,
    source: Source,
}

/// The link after this one.
enum Source {
    None,
    Error(Error),
    Io(io::Error),
}

/// What kind of failure an [`Error`] is.
///
/// More kinds may be added; a `match` on one needs a `_` arm.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The operating system failed an open, a read or a write: no such file,
    /// no permission, a full disk, a mapping refused. The message ends with
    /// what it said, and [`Error::io_error`] gives the [`io::Error`] itself,
    /// to tell those apart by its [`kind`](io::Error::kind). An open or a
    /// create also keeps it as the next link of the chain.
    Io,
    /// The bytes are not a TIFF at all: no `II` or `MM` byte-order mark, or a
    /// magic number other than 42 (classic) or 43 (BigTIFF). Whatever the
    /// file is, another reader may know it.
    NotTiff,
    /// A TIFF that uses what this library cannot read or write: a compression
    /// scheme, bit depth, sample format or predictor it has no code for, or a
    /// pyramid of differently sized pages.
    Unsupported,
    /// What it needs is behind a cargo feature this build was made without:
    /// ZSTD, without `codec-zstd`. The same file or request works in a build
    /// that has the feature.
    Unavailable,
    /// The file is damaged: truncated, pointing outside itself, missing a
    /// required tag, contradicting itself (strips that cannot hold the image
    /// it declares, a directory chain that loops), or holding compressed data
    /// that does not decode.
    Malformed,
    /// Larger than this library or this target can hold: a size past the
    /// address space, an allocation the system refused, more frames than the
    /// index takes, more ROIs than an ROI set holds.
    TooLarge,
    /// The caller asked for what cannot be done: an option out of range, a
    /// reader for a depth or a layout the frame does not have (a 16-bit frame
    /// through [`read_plane_u8`](crate::read_plane_u8), an RGB frame through
    /// the CMYK conversion), text the format cannot hold, metadata that does
    /// not fit the frames written, or a writer finished with nothing in it.
    InvalidInput,
    /// This library, or a codec inside it, failed where no input should make
    /// it fail: an encoder refusing bytes it takes whatever they are, a
    /// decoder handing back fewer planes than it was built to. A bug, worth
    /// reporting with the file or the call that caused it.
    Internal,
}

impl Error {
    /// An error of `kind` that says `message`.
    pub(crate) fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Error(Box::new(Inner {
            kind,
            message: Some(message.into()),
            source: Source::None,
        }))
    }

    /// [`new`](Self::new) from `format_args!`, which is what [`err!`] hands
    /// it: a message with nothing to format stays a `&'static str`, so the
    /// error path allocates only the box.
    pub(crate) fn from_args(kind: ErrorKind, message: fmt::Arguments<'_>) -> Self {
        match message.as_str() {
            Some(text) => Error::new(kind, text),
            None => Error::new(kind, message.to_string()),
        }
    }

    /// An I/O failure, `message` saying what was being done and `source` what
    /// the operating system said.
    fn io(message: Cow<'static, str>, source: io::Error) -> Self {
        Error(Box::new(Inner {
            kind: ErrorKind::Io,
            message: Some(message),
            source: Source::Io(source),
        }))
    }

    /// This error, wrapped in one that says `message` and has its kind.
    fn wrap(self, message: Cow<'static, str>) -> Self {
        let kind = self.0.kind;
        let source = if self.0.message.is_none() {
            // An I/O error converted as it stands says nothing of its own, so
            // the operating system's error becomes the next link directly —
            // the chain anyhow builds when it wraps one.
            self.0.source
        } else {
            Source::Error(self)
        };
        Error(Box::new(Inner {
            kind,
            message: Some(message),
            source,
        }))
    }

    /// What kind of failure this is.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// The [`io::Error`] the operating system reported, for an
    /// [`ErrorKind::Io`] failure: whether the crate said what it was doing
    /// (`could not open …`) or passed a failed read or write on as it stood.
    /// Its [`kind`](io::Error::kind) tells a full disk
    /// ([`StorageFull`](io::ErrorKind::StorageFull)) from a refused
    /// permission and the rest. `None` for every other kind.
    ///
    /// [`source`](std::error::Error::source) reaches it too where the crate
    /// added words of its own; a write that failed mid-stream has none, and
    /// reads as the `io::Error` itself, so this is the way to it.
    pub fn io_error(&self) -> Option<&io::Error> {
        let mut link = self;
        loop {
            match &link.0.source {
                Source::Io(e) => return Some(e),
                Source::Error(next) => link = next,
                Source::None => return None,
            }
        }
    }

    /// This link's own words, without the links after it.
    fn write_own(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.0.message, &self.0.source) {
            (Some(message), _) => f.write_str(message),
            (None, Source::Io(e)) => write!(f, "{e}"),
            (None, Source::Error(e)) => write!(f, "{e}"),
            (None, Source::None) => Ok(()),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_own(f)?;
        if f.alternate() {
            let mut next = std::error::Error::source(self);
            while let Some(e) = next {
                write!(f, ": {e}")?;
                next = e.source();
            }
        }
        Ok(())
    }
}

/// `Kind: the whole chain`, which is what an `unwrap` on one prints; `{:#?}`
/// shows the links as a structure.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let mut s = f.debug_struct("Error");
            s.field("kind", &self.0.kind);
            if let Some(message) = &self.0.message {
                s.field("message", message);
            }
            match &self.0.source {
                Source::None => {}
                Source::Error(e) => {
                    s.field("source", e);
                }
                Source::Io(e) => {
                    s.field("source", e);
                }
            }
            s.finish()
        } else {
            write!(f, "{:?}: {self:#}", self.0.kind)
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match (&self.0.message, &self.0.source) {
            (_, Source::None) => None,
            (Some(_), Source::Error(e)) => Some(e),
            (Some(_), Source::Io(e)) => Some(e),
            // This link is its source's words (see `Inner::message`), so the
            // chain carries on from what that source wraps.
            (None, Source::Error(e)) => e.source(),
            (None, Source::Io(e)) => e.source(),
        }
    }
}

/// An [`ErrorKind::Io`] error that reads as the I/O error does. Wrapped in a
/// context, the I/O error becomes the next link of the chain.
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error(Box::new(Inner {
            kind: ErrorKind::Io,
            message: None,
            source: Source::Io(e),
        }))
    }
}

/// `return Err(err!(Kind, "format", args...))`.
macro_rules! bail {
    ($kind:ident, $($message:tt)+) => {
        return ::core::result::Result::Err($crate::error::err!($kind, $($message)+))
    };
}

/// An [`Error`] of `ErrorKind::Kind` with a `format!`-style message:
/// `err!(Malformed, "IFD at {offset} truncated")`.
macro_rules! err {
    ($kind:ident, $($message:tt)+) => {
        $crate::error::Error::from_args(
            $crate::error::ErrorKind::$kind,
            ::core::format_args!($($message)+),
        )
    };
}

pub(crate) use {bail, err};

/// What a step that failed was doing, added to its error as the outer link.
///
/// On `Result<T, Error>` the kind stays the inner error's; on
/// `Result<T, io::Error>` it is [`ErrorKind::Io`], with the I/O error as the
/// source. A message that has to be formatted goes in `with_context`, so the
/// success path formats nothing.
pub(crate) trait Context<T> {
    // Only the C interface has a fixed message to add so far.
    #[cfg_attr(not(feature = "capi"), allow(dead_code))]
    fn context(self, message: &'static str) -> Result<T>;
    fn with_context<M: Into<Cow<'static, str>>>(self, message: impl FnOnce() -> M) -> Result<T>;
}

impl<T> Context<T> for Result<T, Error> {
    fn context(self, message: &'static str) -> Result<T> {
        self.map_err(|e| e.wrap(Cow::Borrowed(message)))
    }

    fn with_context<M: Into<Cow<'static, str>>>(self, message: impl FnOnce() -> M) -> Result<T> {
        self.map_err(|e| e.wrap(message().into()))
    }
}

impl<T> Context<T> for Result<T, io::Error> {
    fn context(self, message: &'static str) -> Result<T> {
        self.map_err(|e| Error::io(Cow::Borrowed(message), e))
    }

    fn with_context<M: Into<Cow<'static, str>>>(self, message: impl FnOnce() -> M) -> Result<T> {
        self.map_err(|e| Error::io(message().into(), e))
    }
}

/// [`Context`] for a value that may be missing, which has no error to take a
/// kind from: the caller names one.
pub(crate) trait OptionContext<T> {
    fn context(self, kind: ErrorKind, message: &'static str) -> Result<T>;
    fn with_context<M: Into<Cow<'static, str>>>(
        self,
        kind: ErrorKind,
        message: impl FnOnce() -> M,
    ) -> Result<T>;
}

impl<T> OptionContext<T> for Option<T> {
    fn context(self, kind: ErrorKind, message: &'static str) -> Result<T> {
        self.ok_or_else(|| Error::new(kind, message))
    }

    fn with_context<M: Into<Cow<'static, str>>>(
        self,
        kind: ErrorKind,
        message: impl FnOnce() -> M,
    ) -> Result<T> {
        self.ok_or_else(|| Error::new(kind, message()))
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod tests;

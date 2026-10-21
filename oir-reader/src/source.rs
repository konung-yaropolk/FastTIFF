//! Where a part's bytes come from.
//!
//! The reader is written once, against [`Source`], and runs over two kinds of
//! them:
//!
//! * [`Memory`] — bytes already in memory: a memory-mapped part, or bytes the
//!   caller handed over. Every read is a borrow, so nothing is copied but the
//!   pixels themselves, into the planes they belong to.
//! * [`Buffered`] — anything `Read + Seek`, through a window that slides
//!   forward as the part is read. The ordinary-I/O path: where mapping is not
//!   available, not wanted, or refused for one particular file.
//!
//! One copy of the algorithm over both is what keeps the two paths from ever
//! disagreeing about a file.
//!
//! # Why a window, and not a `BufReader` or a read per request
//!
//! A part is read front to back, and nearly every request is small and lands at
//! or just after the last one: an 8- or 16-byte block header, a descriptor of
//! fifty-odd bytes, a 15 KB chunk of pixels, a 512-byte peek for XML. Reading
//! each with its own seek and read is the quarter of a million syscalls a long
//! recording cost before this reader mapped its files. A `BufReader` would
//! batch them, but it drops its buffer on every seek and cannot hand back one
//! contiguous slice that spans a refill. So this keeps its own window: a
//! request inside it is a borrow, and one past its end slides it forward with
//! one large read.

use std::io::{self, Read, Seek, SeekFrom};

/// A part's bytes, read by offset.
///
/// The two layers of the answer are the point. `Ok(None)` means the range is
/// not wholly inside the part — the case the reader has always treated as part
/// of the file's shape, skipping a block or refusing the file as it sees fit.
/// `Err` means the input failed, and every caller passes it on: a bad sector
/// read as "past the end" would skip blocks and produce a different picture
/// rather than an error.
pub(crate) trait Source {
    /// The part's length: the snapshot taken when it was opened. Nothing past
    /// it is ever returned, even from a file that has grown since.
    fn len(&self) -> u64;

    /// The `n` bytes at `at`, or `None` when they are not all in the part.
    /// Borrowed until the next call.
    fn get(&mut self, at: u64, n: usize) -> io::Result<Option<&[u8]>>;
}

/// Bytes already in memory: a mapping, or the caller's own.
pub(crate) struct Memory<'a>(pub(crate) &'a [u8]);

impl Source for Memory<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn get(&mut self, at: u64, n: usize) -> io::Result<Option<&[u8]>> {
        Ok(usize::try_from(at)
            .ok()
            .and_then(|start| self.0.get(start..start.checked_add(n)?)))
    }
}

/// The window's usual size.
pub(crate) const WINDOW: usize = 4 << 20;

/// How much to read on the first fill and after a jump — backwards, or
/// further forward than a window. A whole window there would make an index
/// that lists its blocks out of order cost four megabytes per entry; this is
/// more than any real block, and the window grows back to full size as soon as
/// reading carries on forward.
const FIRST: usize = 64 << 10;

/// Any `Read + Seek`, through a window that slides forward.
pub(crate) struct Buffered<R> {
    inner: R,
    len: u64,
    /// The file offset of `buf[0]`.
    start: u64,
    /// File bytes `[start, start + buf.len())`.
    buf: Vec<u8>,
    /// Where `inner` is positioned, when that is known — so reading on from the
    /// end of the window needs no seek.
    pos: Option<u64>,
    window: usize,
}

impl<R: Read + Seek> Buffered<R> {
    /// `len` is the part's length as it stands now.
    pub(crate) fn new(inner: R, len: u64, window: usize) -> Self {
        Buffered {
            inner,
            len,
            start: 0,
            buf: Vec::new(),
            pos: None,
            window: window.max(1),
        }
    }

    /// Make the window cover `[at, end)`, which is inside the part.
    fn cover(&mut self, at: u64, end: u64) -> io::Result<()> {
        let wend = self.start + self.buf.len() as u64;
        let forward = !self.buf.is_empty()
            && at >= self.start
            && at <= wend.saturating_add(self.window as u64);
        let target = if forward {
            // Carrying on forward: keep what is still wanted and read on from
            // the end without a seek. A small gap is read through rather than
            // sought over, and trimmed below.
            if at < wend {
                let keep = (at - self.start) as usize;
                self.buf.drain(..keep);
                self.start = at;
            } else {
                self.buf.clear();
                self.start = wend;
            }
            end.max(at.saturating_add(self.window as u64))
        } else {
            // A jump, or the first read.
            self.buf.clear();
            self.start = at;
            end.max(at.saturating_add(FIRST as u64))
        };
        let target = target.min(self.len);
        let need = (target - self.start) as usize;
        // A window grown for one oversized request — a hostile block can ask
        // for 64 MiB — goes back to its usual size as soon as a request fits
        // in it again, rather than staying that large for the rest of the part.
        if self.buf.capacity() > 2 * self.window && need <= self.window {
            self.buf.shrink_to(self.window);
        }
        let from = self.start + self.buf.len() as u64;
        if self.pos != Some(from) {
            self.pos = None;
            self.inner.seek(SeekFrom::Start(from))?;
            self.pos = Some(from);
        }
        let mut have = self.buf.len();
        self.buf.resize(need, 0);
        let mut failed: Option<io::Error> = None;
        while have < need {
            match self.inner.read(&mut self.buf[have..need]) {
                Ok(0) => break,
                Ok(k) => {
                    have += k;
                    self.pos = self.pos.map(|p| p + k as u64);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    self.pos = None;
                    failed = Some(e);
                    break;
                }
            }
        }
        self.buf.truncate(have);
        // Trim a gap that was read through, so the window starts at the
        // request.
        if self.start < at {
            let skip = ((at - self.start) as usize).min(self.buf.len());
            self.buf.drain(..skip);
            self.start += skip as u64;
        }
        if self.start + self.buf.len() as u64 >= end {
            // What was asked for is here. Reading further was only ahead of
            // need, so a failure there is left for whoever asks for those bytes.
            return Ok(());
        }
        Err(failed.unwrap_or_else(|| {
            // The file ended before the length it had when it was opened. A
            // recording only ever grows, so this is a file cut short while it
            // was being read — which under a mapping would have been a fault.
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the file became shorter while it was being read",
            )
        }))
    }
}

impl<R: Read + Seek> Source for Buffered<R> {
    fn len(&self) -> u64 {
        self.len
    }

    fn get(&mut self, at: u64, n: usize) -> io::Result<Option<&[u8]>> {
        let Some(end) = at.checked_add(n as u64).filter(|e| *e <= self.len) else {
            return Ok(None);
        };
        if n == 0 {
            return Ok(Some(&[]));
        }
        let wend = self.start + self.buf.len() as u64;
        if !(at >= self.start && end <= wend) {
            self.cover(at, end)?;
        }
        let from = (at - self.start) as usize;
        Ok(Some(&self.buf[from..from + n]))
    }
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;

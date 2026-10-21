//! The buffered window against the plain slice: the same answer to every
//! request, whatever the window's size, and the right failure when the stream
//! under it misbehaves.

use super::*;
use std::io::Cursor;

/// A small deterministic generator, so a failure names a seed that reproduces
/// it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next() % n
        }
    }
}

fn data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + i / 251) as u8).collect()
}

/// Requests shaped like the reader's: mostly ascending and small, with the
/// occasional jump back, overlap, empty request, request at or past the end,
/// and offset near the top of the range.
fn requests(rng: &mut Rng, len: u64, count: usize) -> Vec<(u64, usize)> {
    let mut at = 0u64;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let r = rng.below(100);
        at = match r {
            0..=59 => at + rng.below(64),
            60..=69 => at.saturating_sub(rng.below(256)),
            70..=74 => rng.below(len + 1),
            75..=77 => len.saturating_sub(rng.below(8)),
            78 => len + rng.below(8),
            79 => u64::MAX - rng.below(16),
            _ => at,
        };
        let n = match rng.below(20) {
            0 => 0,
            1 => rng.below(4096) as usize,
            2 => usize::MAX - rng.below(4) as usize,
            _ => rng.below(64) as usize,
        };
        out.push((at, n));
        if at > len {
            at = rng.below(len + 1);
        }
    }
    out
}

#[test]
fn every_window_gives_the_slices_answer_to_every_request() {
    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let len = rng.below(20_000) as usize;
        let bytes = data(len);
        let reqs = requests(&mut rng, len as u64, 2_000);
        for window in [1usize, 3, 8, 64, 4096, WINDOW] {
            let mut slice = Memory(&bytes);
            let mut buffered = Buffered::new(Cursor::new(&bytes[..]), len as u64, window);
            for &(at, n) in &reqs {
                let want = slice.get(at, n).unwrap().map(<[u8]>::to_vec);
                let got = buffered
                    .get(at, n)
                    .unwrap_or_else(|e| panic!("seed {seed} window {window}: {e}"))
                    .map(<[u8]>::to_vec);
                assert_eq!(
                    got, want,
                    "seed {seed}, window {window}, {n} byte(s) at {at:#x} of {len}"
                );
            }
        }
    }
}

/// Bytes a file gained after it was opened are never served: a recording that
/// is still growing reads as the snapshot a mapping made at that moment would.
#[test]
fn nothing_past_the_snapshot_is_ever_returned() {
    let bytes = data(1000);
    for window in [1, 7, 4096] {
        let mut src = Buffered::new(Cursor::new(&bytes[..]), 600, window);
        assert_eq!(src.len(), 600);
        assert_eq!(src.get(590, 10).unwrap(), Some(&bytes[590..600]));
        assert_eq!(src.get(590, 11).unwrap(), None);
        assert_eq!(src.get(600, 1).unwrap(), None);
        assert_eq!(src.get(600, 0).unwrap(), Some(&[][..]));
        // The read-ahead stops at the snapshot too.
        assert!(src.start + src.buf.len() as u64 <= 600);
    }
}

/// A file cut shorter than it was when it was opened is an error — the case a
/// mapping would have turned into a crash.
#[test]
fn a_file_that_shrank_while_open_is_an_error_not_a_short_read() {
    let bytes = data(500);
    for window in [1, 61, 4096] {
        let mut src = Buffered::new(Cursor::new(&bytes[..]), 800, window);
        assert_eq!(src.get(10, 20).unwrap(), Some(&bytes[10..30]));
        let e = src.get(490, 20).expect_err("the bytes are not there");
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof, "{e}");
    }
}

/// A reader handing over one byte at a time, and interrupted every other call.
struct Trickle<'a> {
    inner: Cursor<&'a [u8]>,
    calls: usize,
}

impl Read for Trickle<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.calls.is_multiple_of(2) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "again"));
        }
        let n = buf.len().min(1);
        self.inner.read(&mut buf[..n])
    }
}

impl Seek for Trickle<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[test]
fn short_and_interrupted_reads_are_carried_on_from() {
    let bytes = data(5000);
    let mut rng = Rng(77);
    let reqs = requests(&mut rng, 5000, 500);
    for window in [1, 61, 4096] {
        let mut slice = Memory(&bytes);
        let mut src = Buffered::new(
            Trickle {
                inner: Cursor::new(&bytes[..]),
                calls: 0,
            },
            5000,
            window,
        );
        for &(at, n) in &reqs {
            let want = slice.get(at, n).unwrap().map(<[u8]>::to_vec);
            let got = src.get(at, n).unwrap().map(<[u8]>::to_vec);
            assert_eq!(got, want, "window {window}, {n} at {at}");
        }
    }
}

/// A reader whose medium fails at one offset.
struct BadSector<'a> {
    inner: Cursor<&'a [u8]>,
    bad: u64,
}

impl Read for BadSector<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let pos = self.inner.position();
        if pos >= self.bad {
            return Err(io::Error::other("bad sector"));
        }
        let n = buf.len().min((self.bad - pos) as usize);
        self.inner.read(&mut buf[..n])
    }
}

impl Seek for BadSector<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

/// A failure is the error of whoever needed the bytes that could not be read,
/// and nobody else's: reading ahead of a request is best effort, and must not
/// fail a request it was not needed for — nor ever answer "not in the file"
/// for bytes that are.
#[test]
fn a_read_failure_fails_only_the_requests_that_needed_it() {
    let bytes = data(10_000);
    for window in [1, 61, 4096, WINDOW] {
        let mut src = Buffered::new(
            BadSector {
                inner: Cursor::new(&bytes[..]),
                bad: 6_000,
            },
            10_000,
            window,
        );
        // Satisfied before the bad sector, though the read-ahead runs into it.
        assert_eq!(src.get(5_900, 50).unwrap(), Some(&bytes[5_900..5_950]));
        assert_eq!(src.get(5_990, 10).unwrap(), Some(&bytes[5_990..6_000]));
        // Needing it.
        let e = src.get(5_995, 10).expect_err("byte 6000 cannot be read");
        assert_eq!(e.to_string(), "bad sector");
        let e = src.get(7_000, 1).expect_err("nor anything after it");
        assert_eq!(e.to_string(), "bad sector");
        // Earlier bytes are still there to be had.
        assert_eq!(src.get(100, 10).unwrap(), Some(&bytes[100..110]));
        // And a request past the snapshot is still "not in the file", not an
        // I/O error: no I/O is done for it.
        assert_eq!(src.get(9_999, 2).unwrap(), None);
    }
}

/// One oversized request grows the window; the next one that fits shrinks it
/// back, so a hostile block cannot keep memory for the rest of the part.
#[test]
fn a_window_grown_for_one_request_goes_back_to_its_size() {
    let bytes = data(3 << 20);
    let window = 4096;
    let mut src = Buffered::new(Cursor::new(&bytes[..]), bytes.len() as u64, window);
    assert_eq!(
        src.get(10, 2 << 20).unwrap().map(<[u8]>::len),
        Some(2 << 20)
    );
    assert!(src.buf.capacity() >= 2 << 20);
    // Carrying on forward, from just inside the end of what was read.
    let at = (2 << 20) + 5;
    assert_eq!(src.get(at as u64, 100).unwrap(), Some(&bytes[at..at + 100]));
    assert!(
        src.buf.capacity() <= 2 * window,
        "capacity {} after a small request",
        src.buf.capacity()
    );
    // And after a jump.
    src.get(0, 2 << 20).unwrap();
    let at = (3 << 20) - 10;
    assert_eq!(src.get(at as u64, 8).unwrap(), Some(&bytes[at..at + 8]));
    assert!(
        src.buf.capacity() <= 2 * window,
        "capacity {} after a jump",
        src.buf.capacity()
    );
}

/// Counts what the window asks of the stream under it.
struct Counting<'a> {
    inner: Cursor<&'a [u8]>,
    seeks: usize,
    reads: usize,
    bytes: u64,
}

impl Read for Counting<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reads += 1;
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

impl Seek for Counting<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.seeks += 1;
        self.inner.seek(pos)
    }
}

/// Reading on from the end of the window needs no seek, and a forward gap
/// shorter than a window is read through rather than sought over.
#[test]
fn reading_forward_is_one_stream() {
    let bytes = data(1 << 20);
    let mut src = Buffered::new(
        Counting {
            inner: Cursor::new(&bytes[..]),
            seeks: 0,
            reads: 0,
            bytes: 0,
        },
        bytes.len() as u64,
        4096,
    );
    let mut at = 0u64;
    while at + 16 < bytes.len() as u64 {
        assert_eq!(
            src.get(at, 16).unwrap(),
            Some(&bytes[at as usize..at as usize + 16])
        );
        at += 16 + 1000;
    }
    assert_eq!(src.inner.seeks, 1, "only the first fill should seek");
    assert!(src.inner.bytes <= bytes.len() as u64);
}

//! Who is being measured.
//!
//! Every reader has three names — one for tables, one for narrow columns, one
//! for the CSV and the plots — and before this they were free-text strings
//! repeated at the measurement site, in a `short()` match, in a `csv_id()`
//! match, and again in every `n/s` reason. Six readers times four spellings is
//! twenty-four chances to disagree, and they did: the CSV wrote
//! `fast-tiff-preload` while the plotting script looked for
//! `fast-tiff-lib (preload)`, so that series quietly lost its colour and its
//! place in the reader order for as long as anyone had been reading the charts.
//!
//! So the names live here, once, on an enum. A new reader cannot be added
//! without giving it all three, and a renamed one cannot be half-renamed.

use std::fmt;

/// A contender. Ordering is the order they are reported in: the floor first,
/// then this crate, then the others.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reader {
    /// Plain sequential `read` of the decoded-size bytes. Not a TIFF reader —
    /// the no-decode floor everything else is measured against.
    Raw,
    /// This crate, one frame at a time. The viewer's scrubbing path.
    FastTiff,
    /// This crate again, whole stack in one rayon-parallel call. The *same
    /// library*, a different API — which is the entire reason the two names
    /// have to make their kinship obvious.
    FastTiffPreload,
    /// The pure-Rust `tiff` crate from crates.io.
    TiffRs,
    /// Vendored TinyTIFF (C), via FFI. Uncompressed classic TIFF only.
    TinyTiff,
    /// System libtiff (C), via FFI. Optional, behind `--features libtiff`.
    LibTiff,
}

impl Reader {
    /// Every reader, in report order. `libtiff` is included whether or not the
    /// feature is on: it is simply absent from the results when it is off, and
    /// a plot reading an old CSV still knows where it belongs.
    pub const ALL: [Reader; 6] = [
        Reader::Raw,
        Reader::FastTiff,
        Reader::FastTiffPreload,
        Reader::TiffRs,
        Reader::TinyTiff,
        Reader::LibTiff,
    ];

    /// Full name, for tables and chart legends.
    ///
    /// The two `fast-tiff-lib` entries deliberately share a stem: they are one
    /// library in two modes, and naming them as though they were separate
    /// projects is how the benchmark came to read as a five-way comparison
    /// between libraries when it is really four libraries and two APIs.
    pub fn label(self) -> &'static str {
        match self {
            Reader::Raw => "RAW fread",
            Reader::FastTiff => "fast-tiff-lib",
            Reader::FastTiffPreload => "fast-tiff-lib (preload)",
            Reader::TiffRs => "tiff-rs",
            Reader::TinyTiff => "TinyTIFF (C)",
            Reader::LibTiff => "libtiff (C)",
        }
    }

    /// Stable machine name: CSV column, plot lookup key, filename fragment.
    /// Kebab-case, no spaces or parentheses, so it survives a shell and a
    /// filename unquoted.
    pub fn id(self) -> &'static str {
        match self {
            Reader::Raw => "raw",
            Reader::FastTiff => "fast-tiff",
            Reader::FastTiffPreload => "fast-tiff-preload",
            Reader::TiffRs => "tiff-rs",
            Reader::TinyTiff => "tinytiff",
            Reader::LibTiff => "libtiff",
        }
    }

    /// Compact name for narrow table columns.
    pub fn short(self) -> &'static str {
        match self {
            Reader::Raw => "RAW",
            Reader::FastTiff => "fast",
            Reader::FastTiffPreload => "preload",
            Reader::TiffRs => "tiff-rs",
            Reader::TinyTiff => "tiny",
            Reader::LibTiff => "libtiff",
        }
    }

    /// One line on how it reads a frame, printed with the run header so the
    /// numbers below it can be read without opening the source.
    pub fn how(self) -> &'static str {
        match self {
            Reader::Raw => "sequential read of decoded-size bytes (no decode)",
            Reader::FastTiff => "read_frame_*_into / read_planes_*_into, mmap, one frame at a time",
            Reader::FastTiffPreload => "preload_frames_*, whole stack in one rayon-parallel call",
            Reader::TiffRs => "tiff::Decoder::read_image per IFD",
            Reader::TinyTiff => "TinyTIFFReader_readNext + getSampleData",
            Reader::LibTiff => "TIFFReadEncodedStrip per strip",
        }
    }

    /// Whether this is the no-decode floor rather than a competitor. Relative
    /// speeds are computed against the fastest *TIFF reader*, so the floor is
    /// excluded from that comparison and usually lands below 1.
    pub fn is_floor(self) -> bool {
        self == Reader::Raw
    }
}

impl fmt::Display for Reader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names have to be distinct, or two readers collapse into one row of
    /// the CSV and one bar of the chart.
    #[test]
    fn every_name_is_unique() {
        for pick in [Reader::label, Reader::id, Reader::short] {
            let mut seen: Vec<&str> = Reader::ALL.iter().map(|r| pick(*r)).collect();
            let before = seen.len();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), before, "duplicate name among {seen:?}");
        }
    }

    /// The id goes into a CSV field, a filename and a plot lookup, so it may
    /// not carry a comma, a space, or anything a path would object to.
    #[test]
    fn ids_are_safe_for_files_and_csv() {
        for r in Reader::ALL {
            let id = r.id();
            assert!(!id.is_empty(), "{r:?} has no id");
            assert!(
                id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "{r:?} id {id:?} is not plain kebab-case"
            );
        }
    }

    /// The pair that started this: one library, two APIs, and the labels have
    /// to say so.
    #[test]
    fn the_two_fast_tiff_entries_read_as_one_library() {
        let a = Reader::FastTiff.label();
        let b = Reader::FastTiffPreload.label();
        assert!(b.starts_with(a), "{b:?} should extend {a:?}, not stand apart from it");
        assert!(
            Reader::FastTiffPreload.id().starts_with(Reader::FastTiff.id()),
            "the ids should share a stem too, so a plot can group them"
        );
    }

    /// Exactly one floor, and it is not a TIFF reader.
    #[test]
    fn there_is_exactly_one_floor() {
        let floors: Vec<_> = Reader::ALL.iter().filter(|r| r.is_floor()).collect();
        assert_eq!(floors.len(), 1, "{floors:?}");
        assert_eq!(*floors[0], Reader::Raw);
    }
}

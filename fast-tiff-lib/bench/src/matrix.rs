//! What gets measured: the stack shapes, and the grid they are crossed into.
//!
//! Two axes, and keeping them apart is what makes the results readable.
//!
//! A **family** is one functional configuration — a frame size, a sample
//! format, a codec, a strip layout. It answers "how fast is this reader on
//! *this kind of file*".
//!
//! A **frame count** is how many frames that configuration is written out to.
//! It answers "how does this reader *scale*" — per-frame overhead against pixel
//! throughput, and how much of the cost is paid once at open.
//!
//! Every family is crossed with every frame count that fits the size budget, so
//! a single run produces both readings from the same measurements. There used
//! to be a separate `sweep` mode that re-ran one family (16x16 u16) across the
//! frame counts and wrote its own CSV in its own schema; it measured nothing
//! the matrix was not already measuring, and having two of everything — two
//! modes, two files, two plotting scripts — is most of why this was hard to
//! follow. The sweep is now a *view* of these rows, not a second run.

use fast_tiff_lib::{Compression, SampleType};

/// Frame counts every family is crossed with: seven decades, so a reader's
/// scaling from one frame to a million is visible in one line.
pub const FRAME_COUNTS: [usize; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];

/// The counts a `--quick` smoke run uses. Two, not one, so the scaling view
/// still has a line to draw rather than a dot.
pub const QUICK_FRAME_COUNTS: [usize; 2] = [10, 100];

/// Per-stack pixel-data budget.
///
/// A count that would push a family past this is skipped rather than written: a
/// million-frame 256x256 u16 stack would be 128 GB. So the 256x256 families
/// reach 10k frames, 2048x2048 reaches 100, and only the 16x16 family — where
/// pixel volume is negligible and per-frame overhead is the whole measurement —
/// covers all seven decades.
pub const MAX_STACK_BYTES: u64 = 4 << 30;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    U8,
    U16,
    F32,
    RgbU8,
    RgbU16,
}

impl PixelFormat {
    pub fn sample_type(self) -> SampleType {
        match self {
            PixelFormat::U8 | PixelFormat::RgbU8 => SampleType::U8,
            PixelFormat::U16 | PixelFormat::RgbU16 => SampleType::U16,
            PixelFormat::F32 => SampleType::F32,
        }
    }
    /// Samples per pixel: 3 for the chunky RGB formats, 1 otherwise.
    pub fn spp(self) -> usize {
        match self {
            PixelFormat::RgbU8 | PixelFormat::RgbU16 => 3,
            _ => 1,
        }
    }
    pub fn bytes_per_sample(self) -> usize {
        self.sample_type().bytes()
    }
    pub fn label(self) -> &'static str {
        match self {
            PixelFormat::U8 => "u8",
            PixelFormat::U16 => "u16",
            PixelFormat::F32 => "f32",
            PixelFormat::RgbU8 => "rgb8",
            PixelFormat::RgbU16 => "rgb16",
        }
    }
}

/// One functional configuration, without a frame count. The unit a reader is
/// compared *within*; crossing it with [`FRAME_COUNTS`] gives the runs.
#[derive(Clone, Copy)]
pub struct Family {
    pub width: usize,
    pub height: usize,
    pub format: PixelFormat,
    pub compression: Compression,
    pub predictor: bool,
    /// `None` = one strip per frame, the writer's uncompressed default and the
    /// layout the zero-copy read path needs.
    pub rows_per_strip: Option<u32>,
    pub bigtiff: bool,
    /// What this family is here to show. Printed above its table and used as
    /// the panel title in the scaling grid, so every configuration says why it
    /// is in the matrix rather than leaving the reader to infer it.
    pub asks: &'static str,
}

impl Family {
    pub fn compression_label(&self) -> &'static str {
        match self.compression {
            Compression::None => "none",
            Compression::Lzw => "lzw",
            Compression::Deflate => "deflate",
            Compression::PackBits => "packbits",
            Compression::Zstd => "zstd",
            _ => "other",
        }
    }

    /// Human name, frame count excluded — the key the scaling view groups on.
    pub fn label(&self) -> String {
        format!(
            "{}x{} {} {}{}{}{}",
            self.width,
            self.height,
            self.format.label(),
            self.compression_label(),
            if self.predictor { "+pred" } else { "" },
            match self.rows_per_strip {
                Some(r) => format!(" rps{r}"),
                None => String::new(),
            },
            if self.bigtiff { " bigtiff" } else { "" },
        )
    }

    /// Filename-safe form of [`label`](Self::label).
    pub fn slug(&self) -> String {
        self.label().replace([' ', '/', '+'], "_")
    }

    pub fn at(&self, frames: usize) -> Run {
        Run { family: *self, frames }
    }
}

/// One family written out to a particular number of frames: a single stack, and
/// one row of the results per reader.
#[derive(Clone, Copy)]
pub struct Run {
    pub family: Family,
    pub frames: usize,
}

impl Run {
    pub fn bytes_per_frame(&self) -> usize {
        let f = &self.family;
        f.width * f.height * f.format.spp() * f.format.bytes_per_sample()
    }

    pub fn pixel_bytes(&self) -> u64 {
        self.bytes_per_frame() as u64 * self.frames as u64
    }

    /// Whether this stack is small enough to be worth writing to disk.
    pub fn fits_budget(&self) -> bool {
        self.pixel_bytes() <= MAX_STACK_BYTES
    }

    pub fn label(&self) -> String {
        format!("{} / {} frames", self.family.label(), self.frames)
    }

    pub fn slug(&self) -> String {
        format!("{}_{}_frames", self.family.slug(), self.frames)
    }
}

/// The coverage matrix: every functional axis of the library, each crossed with
/// every frame count that fits the budget.
///
/// Ordered so the tiny-frame family comes first — it is the one that answers
/// "what does a frame cost before any pixels are involved", and reading it
/// first makes the rest legible.
pub fn families() -> Vec<Family> {
    use Compression::*;
    let f = |width, height, format, compression, predictor, rows_per_strip, bigtiff, asks| Family {
        width,
        height,
        format,
        compression,
        predictor,
        rows_per_strip,
        bigtiff,
        asks,
    };
    vec![
        // Pixel volume is negligible here, so what is left is the per-frame
        // overhead: opening, indexing the IFD chain, stepping to the next
        // directory. The only family that reaches a million frames.
        f(16, 16, PixelFormat::U16, None, false, Option::None, false,
          "per-frame overhead, with pixels out of the way"),
        // Format coverage at a middling frame size.
        f(256, 256, PixelFormat::U8, None, false, Option::None, false, "8-bit, the cheapest sample"),
        f(256, 256, PixelFormat::U16, None, false, Option::None, false, "16-bit single-strip: the zero-copy path"),
        f(256, 256, PixelFormat::U16, None, false, Some(32), false, "the same, split into strips"),
        f(256, 256, PixelFormat::F32, None, false, Option::None, false, "32-bit float, zero-copy"),
        // Codecs, on one format so the codec is the only variable.
        f(256, 256, PixelFormat::U16, Lzw, false, Option::None, false, "LZW"),
        f(256, 256, PixelFormat::U16, Lzw, true, Option::None, false, "LZW with a horizontal predictor"),
        f(256, 256, PixelFormat::U16, Deflate, true, Option::None, false, "Deflate + predictor"),
        f(256, 256, PixelFormat::U16, Zstd, true, Option::None, false, "Zstd + predictor"),
        f(256, 256, PixelFormat::U16, PackBits, false, Option::None, false, "PackBits"),
        f(256, 256, PixelFormat::F32, Zstd, true, Option::None, false, "float predictor (3) under Zstd"),
        // Layout coverage.
        f(256, 256, PixelFormat::RgbU8, None, false, Option::None, false, "chunky RGB, deinterleaved on read"),
        f(256, 256, PixelFormat::RgbU16, Deflate, true, Option::None, false, "16-bit RGB, compressed"),
        f(256, 256, PixelFormat::U16, None, false, Option::None, true, "BigTIFF (64-bit offsets)"),
        // Big frames: pixel throughput dominates and per-frame overhead vanishes.
        f(2048, 2048, PixelFormat::U16, None, false, Option::None, false, "large frames, throughput-bound"),
        f(2048, 2048, PixelFormat::U16, None, false, Some(64), false, "large frames, many strips"),
        f(2048, 2048, PixelFormat::U16, Zstd, true, Option::None, false, "large frames, compressed"),
    ]
}

/// Every run the matrix calls for: families x frame counts, minus what will not
/// fit the size budget.
pub fn runs(quick: bool) -> Vec<Run> {
    let counts: &[usize] = if quick { &QUICK_FRAME_COUNTS } else { &FRAME_COUNTS };
    families()
        .into_iter()
        .flat_map(|fam| counts.iter().map(move |&n| fam.at(n)))
        .filter(Run::fits_budget)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Families are the grouping key for the scaling view, so two families that
    /// shared a label would be drawn on top of each other.
    #[test]
    fn family_labels_are_unique() {
        let mut labels: Vec<String> = families().iter().map(Family::label).collect();
        let before = labels.len();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), before, "two families share a label");
    }

    /// The budget is the only reason a run is dropped, and it must not drop so
    /// much that a family has nothing left to plot.
    #[test]
    fn every_family_survives_at_some_frame_count() {
        for fam in families() {
            let kept = FRAME_COUNTS.iter().filter(|&&n| fam.at(n).fits_budget()).count();
            assert!(kept >= 2, "{} keeps only {kept} frame counts", fam.label());
        }
    }

    /// The tiny family is the one that has to reach the far end of the sweep —
    /// it is the whole reason there is a 16x16 family at all.
    #[test]
    fn the_tiny_family_covers_every_decade() {
        let tiny = families().into_iter().next().expect("the tiny family comes first");
        assert_eq!((tiny.width, tiny.height), (16, 16));
        for n in FRAME_COUNTS {
            assert!(tiny.at(n).fits_budget(), "{n} frames of 16x16 should fit the budget");
        }
    }

    /// No run may exceed the budget, or the bench fills the disk.
    #[test]
    fn no_run_exceeds_the_budget() {
        for r in runs(false) {
            assert!(r.pixel_bytes() <= MAX_STACK_BYTES, "{} is over budget", r.label());
        }
    }

    /// A quick run still has to produce a line per family, not a point.
    #[test]
    fn a_quick_run_still_has_two_points_per_family() {
        let quick = runs(true);
        for fam in families() {
            let n = quick.iter().filter(|r| r.family.label() == fam.label()).count();
            assert!(n >= 2, "{} has {n} quick runs; a scaling line needs two", fam.label());
        }
    }

    /// Slugs become filenames, so they may not carry anything a path minds.
    #[test]
    fn slugs_are_filename_safe() {
        for fam in families() {
            let slug = fam.at(1_000).slug();
            assert!(
                slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "{slug:?} is not filename-safe"
            );
        }
    }
}

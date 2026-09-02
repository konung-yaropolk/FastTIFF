//! How a measurement is taken, and what one is.
//!
//! Method follows jkriege2/TinyTIFF's `tinytiffwriter_speedtest`: time each
//! frame with `Instant`, drop the slowest tenth before averaging, warm the page
//! cache first, and accumulate a checksum so nothing can be optimised away.
//!
//! Two rules keep the comparison honest and are worth stating where they are
//! enforced rather than in a README nobody has open:
//!
//! - **Every reader decodes into an owned host buffer.** A zero-copy mmap
//!   borrow would otherwise "win" by not doing the work the others do.
//! - **Open is timed separately from reads.** A reader that indexes the whole
//!   IFD chain up front pays at open and reads quickly afterwards; one that
//!   walks lazily pays per frame. Summing them into a single number hides the
//!   trade, and the trade is the interesting part.

use anyhow::Result;
use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::matrix::Run;
use crate::reader::Reader;

/// Fraction of the slowest per-frame times dropped before averaging. Absorbs
/// scheduler noise and page-fault outliers without flattering any one reader —
/// they all get the same treatment.
pub const TRIM_SLOWEST: f64 = 0.10;

/// One reader's timings for one run.
pub struct Measured {
    pub reader: Reader,
    /// Microseconds per frame, in read order.
    pub per_frame_us: Vec<f64>,
    pub bytes_per_frame: usize,
    /// Guards against the optimiser eliding the decode. Readers within the
    /// same checksum domain must agree — see the README on why the domains
    /// differ between raw bytes and decoded samples.
    pub checksum: u64,
    /// Open + index cost, kept out of the per-frame numbers on purpose.
    pub open_us: f64,
}

impl Measured {
    /// Mean of the fastest `1 - TRIM_SLOWEST` of the frames.
    pub fn mean_us(&self) -> f64 {
        let mut v = self.per_frame_us.clone();
        v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
        let keep = ((v.len() as f64) * (1.0 - TRIM_SLOWEST)).ceil() as usize;
        let keep = keep.clamp(1, v.len());
        v[..keep].iter().sum::<f64>() / keep as f64
    }

    pub fn min_us(&self) -> f64 {
        self.per_frame_us.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Decoded MB/s at the trimmed mean.
    pub fn throughput_mb_s(&self) -> f64 {
        let secs = self.mean_us() / 1e6;
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes_per_frame as f64 / (1024.0 * 1024.0)) / secs
    }

    /// Wall time for the whole stack, reads only.
    pub fn total_read_ms(&self) -> f64 {
        self.per_frame_us.iter().sum::<f64>() / 1000.0
    }
}

/// What came of asking a reader to read a run.
pub enum Outcome {
    Measured(Measured),
    /// The reader cannot handle this configuration. Reported with its reason
    /// rather than silently dropped, so a gap in a chart is explained.
    Unsupported { reader: Reader, reason: String },
}

impl Outcome {
    pub fn reader(&self) -> Reader {
        match self {
            Outcome::Measured(m) => m.reader,
            Outcome::Unsupported { reader, .. } => *reader,
        }
    }
    pub fn measured(&self) -> Option<&Measured> {
        match self {
            Outcome::Measured(m) => Some(m),
            Outcome::Unsupported { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

/// FNV-1a over bytes.
pub fn checksum_bytes(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// FNV-1a over decoded 16-bit samples.
pub fn checksum_u16(b: &[u16]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// FNV-1a over decoded floats, by bit pattern.
pub fn checksum_f32(b: &[f32]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3))
}

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

/// One frame of deterministic, non-uniform sample bytes.
///
/// Non-uniform on purpose: a constant frame compresses to almost nothing and
/// would make every codec look identical and instant.
pub fn frame_bytes(run: &Run) -> Vec<u8> {
    let f = &run.family;
    let (w, spp) = (f.width, f.format.spp());
    let n = f.width * f.height * spp;
    let mut out = Vec::with_capacity(run.bytes_per_frame());
    match f.format.sample_type() {
        SampleType::U8 => {
            for i in 0..n {
                let (x, y) = ((i / spp) % w, (i / spp) / w);
                out.push(
                    (((x / 12 + y / 12) as u8).wrapping_mul(31))
                        .wrapping_add((x ^ y) as u8 ^ (i % spp) as u8),
                );
            }
        }
        SampleType::U16 => {
            for i in 0..n {
                let (x, y) = ((i / spp) % w, (i / spp) / w);
                let v = (((x / 12 + y / 12) as u16).wrapping_mul(1031))
                    .wrapping_add((x ^ y) as u16 ^ (i % spp) as u16);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        SampleType::F32 => {
            for i in 0..n {
                let v = ((i % 2000) as f32) * 0.25 - 250.0;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        other => unreachable!("the matrix only uses U8/U16/F32, not {other:?}"),
    }
    out
}

/// The pair of files one run reads: the TIFF stack, and the same pixels
/// concatenated raw for the floor to read.
pub struct Stacks {
    pub tiff: PathBuf,
    pub raw: PathBuf,
    /// Seconds fast-tiff-lib's writer took, which is the write benchmark.
    pub write_secs: f64,
}

impl Stacks {
    pub fn remove(&self) {
        let _ = std::fs::remove_file(&self.tiff);
        let _ = std::fs::remove_file(&self.raw);
    }
}

/// Write the stack with fast-tiff-lib's own writer — timed, so the write
/// benchmark comes free — plus the raw baseline file.
pub fn write_stacks(dir: &Path, run: &Run) -> Result<Stacks> {
    let f = &run.family;
    let tiff = dir.join(format!("bench_{}.tif", run.slug()));
    let raw = tiff.with_extension("raw");
    let frame = frame_bytes(run);

    let mut opts = WriterOptions::new(f.width as u32, f.height as u32, f.format.sample_type())
        .samples_per_pixel(f.format.spp() as u16)
        .compression(f.compression)
        .predictor(f.predictor)
        .bigtiff(f.bigtiff);
    if let Some(rps) = f.rows_per_strip {
        opts = opts.rows_per_strip(rps);
    }

    let t = Instant::now();
    let mut w = TiffWriter::create(&tiff, opts)?;
    for _ in 0..run.frames {
        w.write_frame_bytes(&frame)?;
    }
    w.finish()?;
    let write_secs = t.elapsed().as_secs_f64();

    let mut out = File::create(&raw)?;
    for _ in 0..run.frames {
        out.write_all(&frame)?;
    }
    Ok(Stacks { tiff, raw, write_secs })
}

/// Where generated stacks live: `TIFF_BENCH_DIR` if set, else the system temp
/// dir. The biggest configurations peak around 7.5 GB on disk (a 4 GiB stack
/// plus its raw sibling), which can overflow a tight system drive.
pub fn scratch_dir() -> PathBuf {
    let sub = "fast_tiff_bench";
    match std::env::var_os("TIFF_BENCH_DIR") {
        Some(dir) => PathBuf::from(dir).join(sub),
        None => std::env::temp_dir().join(sub),
    }
}

/// Read the file once so the first timed read is not also the first page
/// fault. Every reader gets the same warm cache.
pub fn warm_cache(path: &Path) -> Result<()> {
    let mut f = File::open(path)?;
    let mut sink = vec![0u8; 1 << 20];
    while f.read(&mut sink)? != 0 {}
    Ok(())
}

/// The trimmed mean of the fastest TIFF reader in a run — the denominator for
/// relative speed. The floor is excluded: it does no decode work, so measuring
/// decoders against it would report the cost of decompression as a defeat.
pub fn best_tiff_mean(outcomes: &[Outcome]) -> f64 {
    outcomes
        .iter()
        .filter_map(Outcome::measured)
        .filter(|m| !m.reader.is_floor())
        .map(Measured::mean_us)
        .fold(f64::INFINITY, f64::min)
}

/// Geometric mean — the right average for ratios, which multiply.
pub fn geomean(vals: &[f64]) -> f64 {
    let usable: Vec<f64> = vals.iter().copied().filter(|v| *v > 0.0).collect();
    if usable.is_empty() {
        return 0.0;
    }
    (usable.iter().map(|v| v.ln()).sum::<f64>() / usable.len() as f64).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{Family, PixelFormat};
    use fast_tiff_lib::Compression;

    fn run(format: PixelFormat, w: usize, h: usize) -> Run {
        Family {
            width: w,
            height: h,
            format,
            compression: Compression::None,
            predictor: false,
            rows_per_strip: None,
            bigtiff: false,
            asks: "test",
        }
        .at(3)
    }

    fn measured(reader: Reader, times: &[f64]) -> Measured {
        Measured {
            reader,
            per_frame_us: times.to_vec(),
            bytes_per_frame: 1024,
            checksum: 0,
            open_us: 0.0,
        }
    }

    /// The trim drops the slow tail, which is the point: one descheduled frame
    /// should not decide the number.
    #[test]
    fn the_mean_drops_the_slowest_tenth() {
        // Ten frames, one of them wildly slow.
        let mut times = vec![10.0; 9];
        times.push(1000.0);
        let m = measured(Reader::FastTiff, &times);
        assert!((m.mean_us() - 10.0).abs() < 1e-9, "outlier survived: {}", m.mean_us());
        // The minimum still reports the best single frame.
        assert!((m.min_us() - 10.0).abs() < 1e-9);
    }

    /// With too few frames to trim, it still has to produce a number rather
    /// than dividing by zero.
    #[test]
    fn a_single_frame_still_averages() {
        let m = measured(Reader::FastTiff, &[42.0]);
        assert!((m.mean_us() - 42.0).abs() < 1e-9);
        assert!(m.throughput_mb_s() > 0.0);
    }

    /// Relative speed is measured against the fastest *decoder*, never the
    /// floor — otherwise every compressed run reports decompression as a loss.
    #[test]
    fn the_floor_is_not_the_benchmark() {
        let outcomes = vec![
            Outcome::Measured(measured(Reader::Raw, &[1.0])),
            Outcome::Measured(measured(Reader::FastTiff, &[10.0])),
            Outcome::Measured(measured(Reader::TiffRs, &[40.0])),
        ];
        assert!((best_tiff_mean(&outcomes) - 10.0).abs() < 1e-9, "the floor was used as the best");
    }

    /// Ratios multiply, so their average is geometric. A 4x win and a 4x loss
    /// average to parity, not to 2.125.
    #[test]
    fn the_average_of_ratios_is_geometric() {
        assert!((geomean(&[4.0, 0.25]) - 1.0).abs() < 1e-9);
        assert_eq!(geomean(&[]), 0.0, "nothing to average");
        assert!((geomean(&[0.0, 4.0, 0.25]) - 1.0).abs() < 1e-9, "zeroes are not data");
    }

    /// Constant frames would compress to nothing and make every codec look
    /// alike, so the generated pattern has to actually vary.
    #[test]
    fn generated_frames_are_not_uniform() {
        for fmt in [PixelFormat::U8, PixelFormat::U16, PixelFormat::F32, PixelFormat::RgbU8] {
            let r = run(fmt, 32, 32);
            let bytes = frame_bytes(&r);
            assert_eq!(bytes.len(), r.bytes_per_frame(), "{fmt:?}: wrong length");
            let first = bytes[0];
            assert!(bytes.iter().any(|&b| b != first), "{fmt:?}: frame is uniform");
        }
    }

    /// Same input, same bytes: a run has to be repeatable or two runs cannot
    /// be compared.
    #[test]
    fn generated_frames_are_deterministic() {
        let r = run(PixelFormat::U16, 16, 16);
        assert_eq!(frame_bytes(&r), frame_bytes(&r));
    }

    /// The checksum exists to make the decode observable. Different data must
    /// give a different answer, or the optimiser is free to skip the work.
    #[test]
    fn checksums_distinguish_their_input() {
        assert_ne!(checksum_bytes(&[1, 2, 3]), checksum_bytes(&[1, 2, 4]));
        assert_ne!(checksum_u16(&[1, 2, 3]), checksum_u16(&[1, 2, 4]));
        assert_ne!(checksum_f32(&[1.0, 2.0]), checksum_f32(&[1.0, 2.5]));
        // Order matters too: a sum would not catch a transposed image.
        assert_ne!(checksum_bytes(&[1, 2]), checksum_bytes(&[2, 1]));
    }
}

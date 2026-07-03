//! TIFF reader/writer speed benchmark.
//!
//! Spirit and reporting follow jkriege2/TinyTIFF's `tinytiffwriter_speedtest`
//! (per-frame timing, drop the slowest X%), extended to cover the whole
//! fast-tiff-lib feature envelope: every sample format, every codec,
//! predictor, strip layouts, RGB, BigTIFF — each crossed with several FRAME
//! COUNTS, so per-frame overhead and pixel throughput are both visible.
//!
//! Contenders, all reading the *same* stacks (written by fast-tiff-lib's own
//! writer, which is cross-validated against libtiff/tifffile in the crate's
//! test suite):
//!   - fast-tiff-lib    (Rust, this repo, mmap)      -- read_frame_*/read_planes_*
//!   - fast-tiff preload (same, batch, rayon)        -- preload_frames_*
//!   - tiff             (Rust, crates.io, pure Rust) -- Decoder::read_image
//!   - TinyTIFF         (C, vendored, FFI)           -- uncompressed only
//!   - libtiff          (C, system, FFI)             -- optional `--features libtiff`
//!   - RAW fread        (lower bound)                -- plain sequential read
//!
//! Each frame is decoded into an owned host buffer so every library does
//! comparable work (a zero-copy mmap borrow is forced to `.into_owned()`).
//! Readers that don't support a configuration are reported as `n/s` rather
//! than silently skipped. The decode-parallelism hint is OFF (steady-state
//! single-frame latency, the viewer's scrubbing workload).
//!
//! Modes:
//!   cargo run --release                 # matrix: formats x codecs x frames
//!   cargo run --release -- sweep        # frame-count sweep on tiny frames
//!   add `--quick` to either for a fast smoke run

mod ffi;

use anyhow::{anyhow, Result};
use fast_tiff_lib::{Compression, SampleType, TiffStack, TiffWriter, WriterOptions};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Fraction of the slowest per-frame samples to discard before averaging.
const REMOVE_SLOWEST_PERCENT: f64 = 0.10;

// ----------------------------- configuration -------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum PixelFormat {
    U8,
    U16,
    F32,
    RgbU8,
    RgbU16,
}

impl PixelFormat {
    fn sample_type(self) -> SampleType {
        match self {
            PixelFormat::U8 | PixelFormat::RgbU8 => SampleType::U8,
            PixelFormat::U16 | PixelFormat::RgbU16 => SampleType::U16,
            PixelFormat::F32 => SampleType::F32,
        }
    }
    fn spp(self) -> usize {
        match self {
            PixelFormat::RgbU8 | PixelFormat::RgbU16 => 3,
            _ => 1,
        }
    }
    fn bytes_per_sample(self) -> usize {
        self.sample_type().bytes()
    }
    fn label(self) -> &'static str {
        match self {
            PixelFormat::U8 => "u8",
            PixelFormat::U16 => "u16",
            PixelFormat::F32 => "f32",
            PixelFormat::RgbU8 => "rgb8",
            PixelFormat::RgbU16 => "rgb16",
        }
    }
}

#[derive(Clone, Copy)]
struct TestConfig {
    width: usize,
    height: usize,
    frames: usize,
    format: PixelFormat,
    compression: Compression,
    predictor: bool,
    /// None = one strip per frame (the writer's uncompressed default).
    rows_per_strip: Option<u32>,
    bigtiff: bool,
}

impl TestConfig {
    fn bytes_per_frame(&self) -> usize {
        self.width * self.height * self.format.spp() * self.format.bytes_per_sample()
    }
    fn compression_label(&self) -> &'static str {
        match self.compression {
            Compression::None => "none",
            Compression::Lzw => "lzw",
            Compression::Deflate => "deflate",
            Compression::PackBits => "packbits",
            Compression::Zstd => "zstd",
            _ => "other",
        }
    }
    fn describe(&self) -> String {
        format!(
            "{}x{} {} {}{}{}{} / {} frames",
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
            self.frames,
        )
    }
    fn slug(&self) -> String {
        self.describe().replace([' ', '/'], "_")
    }
}

/// The coverage matrix: every functional axis of the library, each crossed
/// with several frame counts so per-frame overhead scales are visible.
fn configs(quick: bool) -> Vec<TestConfig> {
    use Compression::*;
    // (format, compression, predictor, rows_per_strip, bigtiff)
    type Fmt = (PixelFormat, Compression, bool, Option<u32>, bool);
    let coverage: &[Fmt] = &[
        (PixelFormat::U8, None, false, Option::None, false),
        (PixelFormat::U16, None, false, Option::None, false), // zero-copy fast path
        (PixelFormat::U16, None, false, Some(32), false),     // multi-strip uncompressed
        (PixelFormat::F32, None, false, Option::None, false), // zero-copy f32 fast path
        (PixelFormat::U16, Lzw, false, Option::None, false),
        (PixelFormat::U16, Lzw, true, Option::None, false),
        (PixelFormat::U16, Deflate, true, Option::None, false),
        (PixelFormat::U16, Zstd, true, Option::None, false),
        (PixelFormat::U16, PackBits, false, Option::None, false),
        (PixelFormat::F32, Zstd, true, Option::None, false), // fp predictor 3
        (PixelFormat::RgbU8, None, false, Option::None, false),
        (PixelFormat::RgbU16, Deflate, true, Option::None, false),
        (PixelFormat::U16, None, false, Option::None, true), // BigTIFF
    ];
    let big_frames: &[Fmt] = &[
        (PixelFormat::U16, None, false, Option::None, false),
        (PixelFormat::U16, None, false, Some(64), false),
        (PixelFormat::U16, Zstd, true, Option::None, false),
    ];

    let coverage_counts: &[usize] = if quick { &[24] } else { &[40, 160, 640] };
    let big_counts: &[usize] = if quick { &[6] } else { &[8, 24] };

    let mut v = Vec::new();
    for &(format, compression, predictor, rows_per_strip, bigtiff) in coverage {
        for &frames in coverage_counts {
            v.push(TestConfig {
                width: 256,
                height: 256,
                frames,
                format,
                compression,
                predictor,
                rows_per_strip,
                bigtiff,
            });
        }
    }
    for &(format, compression, predictor, rows_per_strip, bigtiff) in big_frames {
        for &frames in big_counts {
            v.push(TestConfig {
                width: 2048,
                height: 2048,
                frames,
                format,
                compression,
                predictor,
                rows_per_strip,
                bigtiff,
            });
        }
    }
    v
}

// ----------------------------- result plumbing -----------------------------

struct FrameResult {
    name: &'static str,
    per_frame_us: Vec<f64>,
    bytes_per_frame: usize,
    checksum: u64, // guards against the optimizer eliding the decode
    open_us: f64,  // open + index cost, separate from reads
}

impl FrameResult {
    fn trimmed_mean_us(&self) -> f64 {
        let mut v = self.per_frame_us.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let keep = ((v.len() as f64) * (1.0 - REMOVE_SLOWEST_PERCENT)).ceil() as usize;
        let keep = keep.max(1).min(v.len());
        v[..keep].iter().sum::<f64>() / keep as f64
    }
    fn min_us(&self) -> f64 {
        self.per_frame_us.iter().copied().fold(f64::INFINITY, f64::min)
    }
    fn throughput_mb_s(&self) -> f64 {
        let secs = self.trimmed_mean_us() / 1e6;
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes_per_frame as f64 / (1024.0 * 1024.0)) / secs
    }
}

enum Outcome {
    Done(FrameResult),
    Skipped { name: &'static str, reason: String },
}

/// One reader x one run, flattened for the overall summary + CSV.
struct SummaryRow {
    config: String,
    format: &'static str,
    compression: &'static str,
    predictor: bool,
    strips: usize,
    bigtiff: bool,
    width: usize,
    frames: usize,
    reader: &'static str,
    ok: bool,
    reason: String,
    open_us: f64,
    mean_us: f64,
    min_us: f64,
    mb_s: f64,
    rel: f64, // mean / best-mean of this run (1.0 = fastest)
    write_mb_s: f64,
}

// ------------------------------ test data ----------------------------------

/// Deterministic non-zero pattern (same idea as TinyTIFF's test data), as raw
/// little-endian interleaved sample bytes for one frame.
fn make_frame_bytes(cfg: &TestConfig) -> Vec<u8> {
    let (w, h, spp) = (cfg.width, cfg.height, cfg.format.spp());
    let n = w * h * spp;
    let mut out = Vec::with_capacity(cfg.bytes_per_frame());
    match cfg.format.sample_type() {
        SampleType::U8 => {
            for i in 0..n {
                let (x, y) = ((i / spp) % w, (i / spp) / w);
                out.push((((x / 12 + y / 12) as u8).wrapping_mul(31)).wrapping_add((x ^ y) as u8 ^ (i % spp) as u8));
            }
        }
        SampleType::U16 => {
            for i in 0..n {
                let (x, y) = ((i / spp) % w, (i / spp) / w);
                let v = (((x / 12 + y / 12) as u16).wrapping_mul(1031)).wrapping_add((x ^ y) as u16 ^ (i % spp) as u16);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        SampleType::F32 => {
            for i in 0..n {
                let v = ((i % 2000) as f32) * 0.25 - 250.0;
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        _ => unreachable!("bench uses U8/U16/F32 sample types"),
    }
    out
}

fn checksum_bytes(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn checksum_u16(b: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn checksum_f32(b: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ------------------------------- writer ------------------------------------

/// Write the stack with fast-tiff-lib's own writer (timed — this doubles as
/// the write benchmark), plus the concatenated-raw-frames baseline file.
fn write_stack(dir: &Path, cfg: &TestConfig) -> Result<(PathBuf, PathBuf, f64)> {
    let path = dir.join(format!("bench_{}.tif", cfg.slug()));
    let raw_path = path.with_extension("raw");
    let frame = make_frame_bytes(cfg);

    let mut opts = WriterOptions::new(cfg.width as u32, cfg.height as u32, cfg.format.sample_type())
        .samples_per_pixel(cfg.format.spp() as u16)
        .compression(cfg.compression)
        .predictor(cfg.predictor)
        .bigtiff(cfg.bigtiff);
    if let Some(rps) = cfg.rows_per_strip {
        opts = opts.rows_per_strip(rps);
    }

    let t = Instant::now();
    let mut w = TiffWriter::create(&path, opts)?;
    for _ in 0..cfg.frames {
        w.write_frame_bytes(&frame)?;
    }
    w.finish()?;
    let write_secs = t.elapsed().as_secs_f64();

    let mut f = File::create(&raw_path)?;
    for _ in 0..cfg.frames {
        f.write_all(&frame)?;
    }
    Ok((path, raw_path, write_secs))
}

fn warm_cache(path: &Path) -> Result<()> {
    let mut f = File::open(path)?;
    let mut sink = vec![0u8; 1 << 20];
    while f.read(&mut sink)? != 0 {}
    Ok(())
}

// ------------------------------- readers -----------------------------------

/// RAW: sequential `read` of each frame's decoded-size bytes. The floor.
fn bench_raw(raw_path: &Path, cfg: &TestConfig) -> Result<FrameResult> {
    let bpf = cfg.bytes_per_frame();
    let t_open = Instant::now();
    let mut f = File::open(raw_path)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;
    let mut buf = vec![0u8; bpf];
    let mut per_frame = Vec::with_capacity(cfg.frames);
    let mut sum: u64 = 0;
    for _ in 0..cfg.frames {
        let t = Instant::now();
        f.read_exact(&mut buf)?;
        per_frame.push(t.elapsed().as_secs_f64() * 1e6);
        sum = sum.wrapping_add(checksum_bytes(&buf));
    }
    Ok(FrameResult { name: "RAW fread", per_frame_us: per_frame, bytes_per_frame: bpf, checksum: sum, open_us })
}

/// fast-tiff-lib per-frame reads via the `*_into` API: each frame decodes
/// into a **reused host buffer** — the same model as the C readers (and the
/// RAW baseline), so per-frame allocation isn't part of the measurement. RGB
/// uses read_planes_*_into (one decompression pass, all planes).
fn bench_fast_tiff(path: &Path, cfg: &TestConfig) -> Result<FrameResult> {
    use fast_tiff_lib::{
        read_frame_f32_into, read_frame_u16_into, read_frame_u8_into, read_planes_u16_into, read_planes_u8_into,
    };

    let bpf = cfg.bytes_per_frame();
    let t_open = Instant::now();
    let stack = TiffStack::open(path)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;
    if stack.frames.len() != cfg.frames {
        return Err(anyhow!("fast-tiff-lib saw {} frames, expected {}", stack.frames.len(), cfg.frames));
    }
    let order = stack.byte_order;
    let mut per_frame = Vec::with_capacity(cfg.frames);
    let mut sum: u64 = 0;
    let mut buf8: Vec<u8> = Vec::new();
    let mut buf16: Vec<u16> = Vec::new();
    let mut buf32: Vec<f32> = Vec::new();
    let mut planes8: Vec<Vec<u8>> = Vec::new();
    let mut planes16: Vec<Vec<u16>> = Vec::new();

    for frame in &stack.frames {
        let t = Instant::now();
        match cfg.format {
            PixelFormat::U8 => {
                read_frame_u8_into(&stack.mmap, frame, order, &mut buf8)?;
                per_frame.push(t.elapsed().as_secs_f64() * 1e6);
                sum = sum.wrapping_add(checksum_bytes(&buf8));
            }
            PixelFormat::U16 => {
                read_frame_u16_into(&stack.mmap, frame, order, None, &mut buf16)?;
                per_frame.push(t.elapsed().as_secs_f64() * 1e6);
                sum = sum.wrapping_add(checksum_u16(&buf16));
            }
            PixelFormat::F32 => {
                read_frame_f32_into(&stack.mmap, frame, order, &mut buf32)?;
                per_frame.push(t.elapsed().as_secs_f64() * 1e6);
                sum = sum.wrapping_add(checksum_f32(&buf32));
            }
            PixelFormat::RgbU8 => {
                read_planes_u8_into(&stack.mmap, frame, order, &mut planes8)?;
                per_frame.push(t.elapsed().as_secs_f64() * 1e6);
                for p in &planes8 {
                    sum = sum.wrapping_add(checksum_bytes(p));
                }
            }
            PixelFormat::RgbU16 => {
                read_planes_u16_into(&stack.mmap, frame, order, None, &mut planes16)?;
                per_frame.push(t.elapsed().as_secs_f64() * 1e6);
                for p in &planes16 {
                    sum = sum.wrapping_add(checksum_u16(p));
                }
            }
        }
    }
    Ok(FrameResult { name: "fast-tiff-lib", per_frame_us: per_frame, bytes_per_frame: bpf, checksum: sum, open_us })
}

/// fast-tiff-lib batch preload: all frames in one rayon-parallel call. The
/// per-frame figure is total/frames — throughput, not single-frame latency.
fn bench_fast_preload(path: &Path, cfg: &TestConfig) -> Result<Outcome> {
    use fast_tiff_lib::{preload_frames_f32, preload_frames_u16, preload_frames_u8};

    if cfg.format.spp() > 1 {
        return Ok(Outcome::Skipped { name: "fast-tiff preload", reason: "preload_frames_* is single-plane".into() });
    }
    let bpf = cfg.bytes_per_frame();
    let t_open = Instant::now();
    let stack = TiffStack::open(path)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let sum = match cfg.format {
        PixelFormat::U8 => preload_frames_u8(&stack)?.iter().fold(0u64, |a, f| a.wrapping_add(checksum_bytes(f))),
        PixelFormat::U16 => preload_frames_u16(&stack, None)?.iter().fold(0u64, |a, f| a.wrapping_add(checksum_u16(f))),
        PixelFormat::F32 => preload_frames_f32(&stack)?.iter().fold(0u64, |a, f| a.wrapping_add(checksum_f32(f))),
        _ => unreachable!(),
    };
    let per_frame = t.elapsed().as_secs_f64() * 1e6 / cfg.frames as f64;
    Ok(Outcome::Done(FrameResult {
        name: "fast-tiff preload",
        per_frame_us: vec![per_frame; cfg.frames],
        bytes_per_frame: bpf,
        checksum: sum,
        open_us,
    }))
}

/// The pure-Rust `tiff` crate decoder. Unsupported layouts surface as `n/s`.
fn bench_tiff_crate(path: &Path, cfg: &TestConfig) -> Outcome {
    match bench_tiff_crate_inner(path, cfg) {
        Ok(r) => Outcome::Done(r),
        Err(e) => Outcome::Skipped { name: "tiff-rs", reason: format!("{e}") },
    }
}

fn bench_tiff_crate_inner(path: &Path, cfg: &TestConfig) -> Result<FrameResult> {
    use tiff::decoder::{Decoder, DecodingResult};

    let bpf = cfg.bytes_per_frame();
    let t_open = Instant::now();
    let file = File::open(path)?;
    let mut dec = Decoder::new(file)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;
    let mut per_frame = Vec::with_capacity(cfg.frames);
    let mut sum: u64 = 0;

    for i in 0..cfg.frames {
        let t = Instant::now();
        let img = dec.read_image()?;
        per_frame.push(t.elapsed().as_secs_f64() * 1e6);
        match img {
            DecodingResult::U8(v) => sum = sum.wrapping_add(checksum_bytes(&v)),
            DecodingResult::U16(v) => sum = sum.wrapping_add(checksum_u16(&v)),
            DecodingResult::F32(v) => sum = sum.wrapping_add(checksum_f32(&v)),
            _ => return Err(anyhow!("unexpected sample type from tiff crate")),
        }
        if i + 1 < cfg.frames {
            dec.next_image()?;
        }
    }
    Ok(FrameResult { name: "tiff-rs", per_frame_us: per_frame, bytes_per_frame: bpf, checksum: sum, open_us })
}

/// TinyTIFF (vendored C): uncompressed, single-sample, classic TIFF only.
fn bench_tinytiff(path: &Path, cfg: &TestConfig) -> Result<Outcome> {
    use ffi::tinytiff::*;
    use std::ffi::CString;
    use std::os::raw::c_void;

    if cfg.compression != Compression::None || cfg.format.spp() > 1 || cfg.bigtiff {
        return Ok(Outcome::Skipped {
            name: "TinyTIFF (C)",
            reason: "uncompressed single-sample classic TIFF only".into(),
        });
    }

    let bpf = cfg.bytes_per_frame();
    let cpath = CString::new(path.to_string_lossy().as_bytes())?;
    let per_frame;
    let mut sum: u64 = 0;
    let open_us;

    unsafe {
        let t_open = Instant::now();
        let tiff = TinyTIFFReader_open(cpath.as_ptr());
        open_us = t_open.elapsed().as_secs_f64() * 1e6;
        if tiff.is_null() || TinyTIFFReader_wasError(tiff) != 0 {
            if !tiff.is_null() {
                TinyTIFFReader_close(tiff);
            }
            return Ok(Outcome::Skipped { name: "TinyTIFF (C)", reason: "open failed".into() });
        }

        let mut pf = Vec::with_capacity(cfg.frames);
        let mut frame_buf = vec![0u8; bpf];
        let mut f = 0usize;
        loop {
            let t = Instant::now();
            let ok = TinyTIFFReader_getSampleData(tiff, frame_buf.as_mut_ptr() as *mut c_void, 0);
            let dt = t.elapsed().as_secs_f64() * 1e6;
            if ok == 0 {
                TinyTIFFReader_close(tiff);
                return Ok(Outcome::Skipped { name: "TinyTIFF (C)", reason: format!("read failed at frame {f}") });
            }
            pf.push(dt);
            sum = sum.wrapping_add(checksum_bytes(&frame_buf));
            f += 1;
            if f >= cfg.frames || TinyTIFFReader_hasNext(tiff) == 0 || TinyTIFFReader_readNext(tiff) == 0 {
                break;
            }
        }
        TinyTIFFReader_close(tiff);
        if f != cfg.frames {
            return Ok(Outcome::Skipped { name: "TinyTIFF (C)", reason: format!("read {f}/{} frames", cfg.frames) });
        }
        per_frame = pf;
    }
    Ok(Outcome::Done(FrameResult { name: "TinyTIFF (C)", per_frame_us: per_frame, bytes_per_frame: bpf, checksum: sum, open_us }))
}

/// libtiff via FFI (optional feature): read every strip of each directory.
#[cfg(feature = "libtiff")]
fn bench_libtiff(path: &Path, cfg: &TestConfig) -> Outcome {
    use ffi::libtiff::*;
    use std::ffi::CString;
    use std::os::raw::c_void;

    let bpf = cfg.bytes_per_frame();
    let cpath = match CString::new(path.to_string_lossy().as_bytes()) {
        Ok(c) => c,
        Err(e) => return Outcome::Skipped { name: "libtiff (C)", reason: e.to_string() },
    };
    let mode = CString::new("r").unwrap();
    let per_frame;
    let mut sum: u64 = 0;
    let open_us;

    unsafe {
        let t_open = Instant::now();
        let tif = TIFFOpen(cpath.as_ptr(), mode.as_ptr());
        open_us = t_open.elapsed().as_secs_f64() * 1e6;
        if tif.is_null() {
            return Outcome::Skipped { name: "libtiff (C)", reason: "TIFFOpen failed".into() };
        }
        let mut pf = Vec::with_capacity(cfg.frames);
        let mut frame_buf = vec![0u8; bpf];
        for f in 0..cfg.frames {
            let t = Instant::now();
            let nstrips = TIFFNumberOfStrips(tif);
            let stripsz = TIFFStripSize(tif);
            let mut off = 0usize;
            for s in 0..nstrips {
                let dst = frame_buf.as_mut_ptr().add(off) as *mut c_void;
                let got = TIFFReadEncodedStrip(tif, s, dst, stripsz);
                if got < 0 {
                    TIFFClose(tif);
                    return Outcome::Skipped { name: "libtiff (C)", reason: format!("strip read failed at frame {f}") };
                }
                off += got as usize;
            }
            pf.push(t.elapsed().as_secs_f64() * 1e6);
            sum = sum.wrapping_add(checksum_bytes(&frame_buf[..off.min(bpf)]));
            if f + 1 < cfg.frames && TIFFReadDirectory(tif) == 0 {
                TIFFClose(tif);
                return Outcome::Skipped { name: "libtiff (C)", reason: format!("out of directories at frame {f}") };
            }
        }
        TIFFClose(tif);
        per_frame = pf;
    }
    Outcome::Done(FrameResult { name: "libtiff (C)", per_frame_us: per_frame, bytes_per_frame: bpf, checksum: sum, open_us })
}

// ------------------------------- reporting ---------------------------------

fn print_table(cfg: &TestConfig, outcomes: &[Outcome], strips: usize, write_mb_s: f64) {
    println!("\n========================================================================");
    println!("  {}", cfg.describe());
    let total_mb = (cfg.bytes_per_frame() * cfg.frames) as f64 / (1024.0 * 1024.0);
    println!(
        "  {:.1} MB decoded | {} strip(s)/frame | written by fast-tiff-lib at {:.0} MB/s",
        total_mb, strips, write_mb_s
    );
    println!("------------------------------------------------------------------------");
    println!("  {:<18} {:>12} {:>12} {:>13}", "reader", "mean us/fr", "min us/fr", "MB/s (mean)");
    println!("------------------------------------------------------------------------");

    let mut done: Vec<&FrameResult> = outcomes
        .iter()
        .filter_map(|o| match o {
            Outcome::Done(r) => Some(r),
            _ => None,
        })
        .collect();
    done.sort_by(|a, b| a.trimmed_mean_us().partial_cmp(&b.trimmed_mean_us()).unwrap());
    // Relative speeds are vs the fastest *TIFF reader*: RAW fread does no
    // decode work, so it's reported as the physical floor, not a competitor.
    let best = best_tiff_mean(&done);
    for r in &done {
        let rel = r.trimmed_mean_us() / best;
        let tag = if r.name == "RAW fread" {
            format!("  {rel:.2}x (no-decode floor)")
        } else if (rel - 1.0).abs() < 1e-9 {
            "  <-- fastest reader".to_string()
        } else {
            format!("  {rel:.2}x")
        };
        println!(
            "  {:<18} {:>12.2} {:>12.2} {:>13.1}{}",
            r.name,
            r.trimmed_mean_us(),
            r.min_us(),
            r.throughput_mb_s(),
            tag
        );
    }
    for o in outcomes {
        if let Outcome::Skipped { name, reason } = o {
            println!("  {name:<18} {:>12} ({reason})", "n/s");
        }
    }
    print!("  checksums: ");
    for r in &done {
        print!("{}={:#010x}  ", short(r.name), r.checksum as u32);
    }
    println!();
}

/// The fastest trimmed mean among actual TIFF readers (RAW excluded).
fn best_tiff_mean(done: &[&FrameResult]) -> f64 {
    done.iter()
        .filter(|r| r.name != "RAW fread")
        .map(|r| r.trimmed_mean_us())
        .fold(f64::INFINITY, f64::min)
        .min(f64::MAX)
}

fn short(name: &str) -> &'static str {
    match name {
        "fast-tiff-lib" => "fast",
        "fast-tiff preload" => "preload",
        "libtiff (C)" => "libtiff",
        "TinyTIFF (C)" => "tiny",
        "tiff-rs" => "tiffrs",
        "RAW fread" => "raw",
        _ => "?",
    }
}

fn csv_id(name: &str) -> &'static str {
    match name {
        "fast-tiff-lib" => "fast-tiff-lib",
        "fast-tiff preload" => "fast-tiff-preload",
        "libtiff (C)" => "libtiff",
        "TinyTIFF (C)" => "TinyTIFF",
        "tiff-rs" => "tiff-rs",
        "RAW fread" => "RAW",
        _ => "?",
    }
}

/// Geometric mean (relative speeds multiply, so this is the right average).
fn geomean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    (vals.iter().map(|v| v.ln()).sum::<f64>() / vals.len() as f64).exp()
}

fn print_overall_summary(rows: &[SummaryRow]) {
    let readers = ["RAW fread", "fast-tiff-lib", "fast-tiff preload", "tiff-rs", "TinyTIFF (C)", "libtiff (C)"];
    let runs: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.config.as_str()).collect();

    println!("\n########################################################################");
    println!("  OVERALL SUMMARY  ({} runs, formats x codecs x frame counts)", runs.len());
    println!("########################################################################");
    println!(
        "  {:<18} {:>5} {:>6} {:>9} {:>9} {:>9} {:>11}",
        "reader", "runs", "wins", "geomean", "median", "worst", "mean MB/s"
    );
    println!("  ----------------------------------------------------------------------");

    struct Agg {
        name: &'static str,
        geo: f64,
        rels: Vec<f64>,
    }
    let mut aggs: Vec<Agg> = Vec::new();
    for name in readers {
        let mut rels: Vec<f64> = rows.iter().filter(|r| r.reader == name && r.ok).map(|r| r.rel).collect();
        if rels.is_empty() {
            continue;
        }
        let wins = rels.iter().filter(|&&r| (r - 1.0).abs() < 1e-9).count();
        rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = rels[rels.len() / 2];
        let worst = *rels.last().unwrap();
        let geo = geomean(&rels);
        let mbs: Vec<f64> = rows.iter().filter(|r| r.reader == name && r.ok).map(|r| r.mb_s).collect();
        let mean_mbs = mbs.iter().sum::<f64>() / mbs.len() as f64;
        println!(
            "  {:<18} {:>5} {:>6} {:>8.2}x {:>8.2}x {:>8.2}x {:>11.0}",
            name,
            rels.len(),
            wins,
            geo,
            median,
            worst,
            mean_mbs
        );
        aggs.push(Agg { name, geo, rels });
    }

    // ASCII infographic: relative speed, lower = better = shorter bar.
    println!("\n  Relative speed (geomean; 1.00x = fastest TIFF reader, RAW fread = no-decode floor)");
    aggs.sort_by(|a, b| a.geo.partial_cmp(&b.geo).unwrap());
    let scale = 46.0 / aggs.iter().map(|a| a.geo).fold(1.0f64, f64::max).min(8.0);
    for a in &aggs {
        let len = ((a.geo.min(8.0)) * scale).round() as usize;
        println!("  {:<18} {} {:.2}x  ({} runs)", a.name, "#".repeat(len.max(1)), a.geo, a.rels.len());
    }

    // Writer throughput per codec (each stack was written by fast-tiff-lib).
    println!("\n  fast-tiff-lib WRITE throughput by codec (mean over its runs):");
    let mut codecs: Vec<&str> = rows.iter().map(|r| r.compression).collect();
    codecs.sort_unstable();
    codecs.dedup();
    for codec in codecs {
        let v: Vec<f64> = rows
            .iter()
            .filter(|r| r.compression == codec && r.reader == "fast-tiff-lib")
            .map(|r| r.write_mb_s)
            .collect();
        if !v.is_empty() {
            println!("    {:<9} {:>8.0} MB/s", codec, v.iter().sum::<f64>() / v.len() as f64);
        }
    }
    println!("\n  Full per-run data: bench_results.csv  ->  python plot_results.py");
}

// ------------------------------ system info --------------------------------

fn system_info_lines() -> Vec<String> {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_all();
    let cpu = sys.cpus().first();
    // `mut` is only used when the libtiff feature appends its version line.
    #[allow(unused_mut)]
    let mut lines = vec![
        format!(
            "OS:        {} {} ({})",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default(),
            System::cpu_arch().unwrap_or_default()
        ),
        format!(
            "CPU:       {} ({} physical / {} logical cores, {} MHz)",
            cpu.map(|c| c.brand().trim().to_string()).unwrap_or_default(),
            sys.physical_core_count().unwrap_or(0),
            sys.cpus().len(),
            cpu.map(|c| c.frequency()).unwrap_or(0)
        ),
        format!("RAM:       {:.1} GiB", sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)),
        format!("toolchain: {}", env!("BENCH_RUSTC_VERSION")),
        format!("fast-tiff-lib: {} (path dependency)", env!("FAST_TIFF_LIB_VERSION")),
        format!("tiff (rust): 0.11 | TinyTIFF: vendored | libtiff: {}", if cfg!(feature = "libtiff") { "linked" } else { "not built (--features libtiff)" }),
    ];
    #[cfg(feature = "libtiff")]
    unsafe {
        let lt = ffi::libtiff::TIFFGetVersion();
        if !lt.is_null() {
            let s = std::ffi::CStr::from_ptr(lt).to_string_lossy();
            lines.push(format!("libtiff:   {}", s.lines().next().unwrap_or("").trim()));
        }
    }
    lines
}

// --------------------------------- main ------------------------------------

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick" || a == "quick");
    let mode = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--") && a.as_str() != "quick")
        .map(String::as_str)
        .unwrap_or("matrix");

    println!("TIFF reader/writer speed benchmark{}", if quick { "  [--quick]" } else { "" });
    println!("------------------------------------------------------------------------");
    for line in system_info_lines() {
        println!("  {line}");
    }
    println!("------------------------------------------------------------------------");

    // Steady-state single-frame latency: the viewer's scrubbing regime.
    fast_tiff_lib::set_parallel_decode(false);

    match mode {
        "matrix" => run_matrix(quick),
        "sweep" => run_sweep(quick),
        other => Err(anyhow!("unknown mode '{other}' (expected 'matrix' or 'sweep')")),
    }
}

fn run_one(path: &Path, raw_path: &Path, cfg: &TestConfig) -> Result<Vec<Outcome>> {
    let mut outcomes = Vec::new();
    outcomes.push(Outcome::Done(bench_raw(raw_path, cfg)?));
    outcomes.push(match bench_fast_tiff(path, cfg) {
        Ok(r) => Outcome::Done(r),
        Err(e) => Outcome::Skipped { name: "fast-tiff-lib", reason: e.to_string() },
    });
    outcomes.push(bench_fast_preload(path, cfg)?);
    outcomes.push(bench_tiff_crate(path, cfg));
    outcomes.push(bench_tinytiff(path, cfg)?);
    #[cfg(feature = "libtiff")]
    outcomes.push(bench_libtiff(path, cfg));
    Ok(outcomes)
}

fn run_matrix(quick: bool) -> Result<()> {
    let tmp = std::env::temp_dir().join("tiff_read_bench_data");
    std::fs::create_dir_all(&tmp)?;
    println!("mode: matrix   scratch dir: {}", tmp.display());

    let mut summary: Vec<SummaryRow> = Vec::new();

    for cfg in configs(quick) {
        let (path, raw_path, write_secs) = write_stack(&tmp, &cfg)?;
        let write_mb_s = (cfg.bytes_per_frame() * cfg.frames) as f64 / (1024.0 * 1024.0) / write_secs;
        warm_cache(&path)?;
        warm_cache(&raw_path)?;

        let strips = TiffStack::open(&path)?.frames[0].strip_offsets.len();
        let outcomes = run_one(&path, &raw_path, &cfg)?;
        print_table(&cfg, &outcomes, strips, write_mb_s);

        // Fold this run into the summary rows. Relative speeds are vs the
        // fastest TIFF reader (RAW is the no-decode floor, rel can be < 1).
        let done_refs: Vec<&FrameResult> = outcomes
            .iter()
            .filter_map(|o| match o {
                Outcome::Done(r) => Some(r),
                _ => None,
            })
            .collect();
        let best = best_tiff_mean(&done_refs);
        for o in &outcomes {
            let (reader, ok, reason, open_us, mean_us, min_us, mb_s, rel) = match o {
                Outcome::Done(r) => (
                    r.name,
                    true,
                    String::new(),
                    r.open_us,
                    r.trimmed_mean_us(),
                    r.min_us(),
                    r.throughput_mb_s(),
                    r.trimmed_mean_us() / best,
                ),
                Outcome::Skipped { name, reason } => (*name, false, reason.clone(), 0.0, 0.0, 0.0, 0.0, 0.0),
            };
            summary.push(SummaryRow {
                config: cfg.describe(),
                format: cfg.format.label(),
                compression: cfg.compression_label(),
                predictor: cfg.predictor,
                strips,
                bigtiff: cfg.bigtiff,
                width: cfg.width,
                frames: cfg.frames,
                reader,
                ok,
                reason,
                open_us,
                mean_us,
                min_us,
                mb_s,
                rel,
                write_mb_s,
            });
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&raw_path);
    }

    print_overall_summary(&summary);
    write_csv(Path::new("bench_results.csv"), &summary)?;
    println!("\nwrote bench_results.csv");
    Ok(())
}

fn write_csv(path: &Path, rows: &[SummaryRow]) -> Result<()> {
    let mut csv = String::new();
    for line in system_info_lines() {
        csv.push_str(&format!("# {line}\n"));
    }
    csv.push_str(
        "config,format,compression,predictor,strips,bigtiff,width,frames,reader,ok,reason,\
         open_us,mean_us,min_us,mb_s,rel,write_mb_s\n",
    );
    for r in rows {
        csv.push_str(&format!(
            "\"{}\",{},{},{},{},{},{},{},{},{},\"{}\",{:.3},{:.4},{:.4},{:.3},{:.4},{:.1}\n",
            r.config,
            r.format,
            r.compression,
            r.predictor,
            r.strips,
            r.bigtiff,
            r.width,
            r.frames,
            csv_id(r.reader),
            r.ok,
            r.reason.replace('"', "'"),
            r.open_us,
            r.mean_us,
            r.min_us,
            r.mb_s,
            r.rel,
            r.write_mb_s,
        ));
    }
    std::fs::write(path, csv)?;
    Ok(())
}

/// Frame-count sweep on a fixed tiny frame (16x16 u16, single strip): isolates
/// per-frame overhead — open + IFD indexing vs per-frame stepping/decode.
fn run_sweep(quick: bool) -> Result<()> {
    const W: usize = 16;
    const H: usize = 16;
    let frame_counts: &[usize] = if quick {
        &[10, 100, 1_000, 10_000]
    } else {
        &[10, 100, 1_000, 10_000, 100_000, 1_000_000]
    };

    println!("mode: sweep  (frame={W}x{H}, u16, single-strip)");
    let tmp = std::env::temp_dir().join("tiff_read_bench_sweep");
    std::fs::create_dir_all(&tmp)?;

    let mut csv = String::new();
    for line in system_info_lines() {
        csv.push_str(&format!("# {line}\n"));
    }
    csv.push_str("reader,frames,frame_w,frame_h,bits,open_us,mean_read_us,min_read_us,total_read_ms,read_throughput_mb_s\n");

    for &frames in frame_counts {
        let cfg = TestConfig {
            width: W,
            height: H,
            frames,
            format: PixelFormat::U16,
            compression: Compression::None,
            predictor: false,
            rows_per_strip: None,
            bigtiff: false,
        };
        println!(
            "\n--- {frames} frames ({:.1} MB pixel data) ---",
            (cfg.bytes_per_frame() * frames) as f64 / (1024.0 * 1024.0)
        );
        let (path, raw_path, _) = write_stack(&tmp, &cfg)?;
        warm_cache(&path)?;
        warm_cache(&raw_path)?;

        for o in run_one(&path, &raw_path, &cfg)? {
            if let Outcome::Done(r) = o {
                let total_read_ms: f64 = r.per_frame_us.iter().sum::<f64>() / 1000.0;
                println!(
                    "  {:<18} open={:>10.1} us   read mean={:>8.3} us/fr   total={:>9.1} ms   {:>7.1} MB/s",
                    r.name,
                    r.open_us,
                    r.trimmed_mean_us(),
                    total_read_ms,
                    r.throughput_mb_s()
                );
                csv.push_str(&format!(
                    "{},{},{},{},16,{:.3},{:.4},{:.4},{:.4},{:.3}\n",
                    csv_id(r.name),
                    frames,
                    W,
                    H,
                    r.open_us,
                    r.trimmed_mean_us(),
                    r.min_us(),
                    total_read_ms,
                    r.throughput_mb_s(),
                ));
            }
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&raw_path);
    }

    std::fs::write("sweep_results.csv", csv)?;
    println!("\nwrote sweep_results.csv  ->  python plot_sweep.py sweep_results.csv graphs/");
    Ok(())
}

//! The contenders, one function each.
//!
//! Each returns an [`Outcome`]: timings, or a stated reason it cannot read this
//! configuration. Nothing is skipped silently — an absent bar in a chart should
//! always be explainable from the CSV.
//!
//! They all obey the same two rules (see [`crate::measure`]): decode into an
//! owned host buffer, and time the open separately from the reads.

use anyhow::{anyhow, Result};
use fast_tiff_lib::{Compression, TiffStack};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use crate::ffi;
use crate::matrix::{PixelFormat, Run};
use crate::measure::{checksum_bytes, checksum_f32, checksum_u16, Measured, Outcome};
use crate::reader::Reader;

/// Time every reader against one already-written, cache-warm run.
pub fn run_all(tiff: &Path, raw: &Path, run: &Run) -> Result<Vec<Outcome>> {
    // `mut` only when the libtiff feature has something to push.
    #[allow(unused_mut)]
    let mut out = vec![
        Outcome::Measured(raw_fread(raw, run)?),
        match fast_tiff(tiff, run) {
            Ok(m) => Outcome::Measured(m),
            Err(e) => unsupported(Reader::FastTiff, e.to_string()),
        },
        fast_tiff_preload(tiff, run)?,
        tiff_rs(tiff, run),
        tinytiff(tiff, run)?,
    ];
    #[cfg(libtiff)]
    out.push(libtiff(tiff, run));
    Ok(out)
}

fn unsupported(reader: Reader, reason: impl Into<String>) -> Outcome {
    Outcome::Unsupported { reader, reason: reason.into() }
}

/// The floor: sequential `read` of each frame's decoded-size bytes, no decode.
fn raw_fread(raw: &Path, run: &Run) -> Result<Measured> {
    let bpf = run.bytes_per_frame();
    let t_open = Instant::now();
    let mut f = File::open(raw)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;

    let mut buf = vec![0u8; bpf];
    let mut per_frame_us = Vec::with_capacity(run.frames);
    let mut checksum: u64 = 0;
    for _ in 0..run.frames {
        let t = Instant::now();
        f.read_exact(&mut buf)?;
        per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
        checksum = checksum.wrapping_add(checksum_bytes(&buf));
    }
    Ok(Measured { reader: Reader::Raw, per_frame_us, bytes_per_frame: bpf, checksum, open_us })
}

/// This crate, one frame at a time, through the `*_into` API so each frame
/// decodes into a reused buffer — the same model as the C readers and the
/// floor, keeping per-frame allocation out of the measurement.
fn fast_tiff(path: &Path, run: &Run) -> Result<Measured> {
    use fast_tiff_lib::{
        read_frame_f32_into, read_frame_u16_into, read_frame_u8_into, read_planes_u16_into,
        read_planes_u8_into,
    };

    let bpf = run.bytes_per_frame();
    let t_open = Instant::now();
    let stack = TiffStack::open(path)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;
    if stack.frames.len() != run.frames {
        return Err(anyhow!("indexed {} frames, expected {}", stack.frames.len(), run.frames));
    }

    let order = stack.byte_order;
    let mut per_frame_us = Vec::with_capacity(run.frames);
    let mut checksum: u64 = 0;
    let (mut b8, mut b16, mut b32) = (Vec::new(), Vec::new(), Vec::new());
    let (mut p8, mut p16): (Vec<Vec<u8>>, Vec<Vec<u16>>) = (Vec::new(), Vec::new());

    for frame in &stack.frames {
        let t = Instant::now();
        match run.family.format {
            PixelFormat::U8 => {
                read_frame_u8_into(&stack.data, frame, order, &mut b8)?;
                per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
                checksum = checksum.wrapping_add(checksum_bytes(&b8));
            }
            PixelFormat::U16 => {
                read_frame_u16_into(&stack.data, frame, order, None, &mut b16)?;
                per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
                checksum = checksum.wrapping_add(checksum_u16(&b16));
            }
            PixelFormat::F32 => {
                read_frame_f32_into(&stack.data, frame, order, &mut b32)?;
                per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
                checksum = checksum.wrapping_add(checksum_f32(&b32));
            }
            PixelFormat::RgbU8 => {
                read_planes_u8_into(&stack.data, frame, order, &mut p8)?;
                per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
                checksum = p8.iter().fold(checksum, |a, p| a.wrapping_add(checksum_bytes(p)));
            }
            PixelFormat::RgbU16 => {
                read_planes_u16_into(&stack.data, frame, order, None, &mut p16)?;
                per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
                checksum = p16.iter().fold(checksum, |a, p| a.wrapping_add(checksum_u16(p)));
            }
        }
    }
    Ok(Measured { reader: Reader::FastTiff, per_frame_us, bytes_per_frame: bpf, checksum, open_us })
}

/// The same crate, whole stack in one rayon-parallel call.
///
/// Its per-frame figure is total / frames — **throughput, not latency**. It is
/// in the matrix to show what the batch API buys on compressed stacks, not to
/// be read as a faster way to fetch one frame.
fn fast_tiff_preload(path: &Path, run: &Run) -> Result<Outcome> {
    use fast_tiff_lib::{preload_frames_f32, preload_frames_u16, preload_frames_u8};

    if run.family.format.spp() > 1 {
        return Ok(unsupported(Reader::FastTiffPreload, "preload_frames_* is single-plane"));
    }
    let bpf = run.bytes_per_frame();
    let t_open = Instant::now();
    let stack = TiffStack::open(path)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;

    let t = Instant::now();
    let checksum = match run.family.format {
        PixelFormat::U8 => preload_frames_u8(&stack)?
            .iter()
            .fold(0u64, |a, f| a.wrapping_add(checksum_bytes(f))),
        PixelFormat::U16 => preload_frames_u16(&stack, None)?
            .iter()
            .fold(0u64, |a, f| a.wrapping_add(checksum_u16(f))),
        PixelFormat::F32 => preload_frames_f32(&stack)?
            .iter()
            .fold(0u64, |a, f| a.wrapping_add(checksum_f32(f))),
        other => unreachable!("multi-plane {other:?} was declined above"),
    };
    let per_frame = t.elapsed().as_secs_f64() * 1e6 / run.frames as f64;

    Ok(Outcome::Measured(Measured {
        reader: Reader::FastTiffPreload,
        per_frame_us: vec![per_frame; run.frames],
        bytes_per_frame: bpf,
        checksum,
        open_us,
    }))
}

/// The pure-Rust `tiff` crate.
fn tiff_rs(path: &Path, run: &Run) -> Outcome {
    match tiff_rs_inner(path, run) {
        Ok(m) => Outcome::Measured(m),
        Err(e) => unsupported(Reader::TiffRs, e.to_string()),
    }
}

fn tiff_rs_inner(path: &Path, run: &Run) -> Result<Measured> {
    use tiff::decoder::{Decoder, DecodingResult};

    let bpf = run.bytes_per_frame();
    let t_open = Instant::now();
    let mut dec = Decoder::new(File::open(path)?)?;
    let open_us = t_open.elapsed().as_secs_f64() * 1e6;

    let mut per_frame_us = Vec::with_capacity(run.frames);
    let mut checksum: u64 = 0;
    for i in 0..run.frames {
        let t = Instant::now();
        let img = dec.read_image()?;
        per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
        checksum = match img {
            DecodingResult::U8(v) => checksum.wrapping_add(checksum_bytes(&v)),
            DecodingResult::U16(v) => checksum.wrapping_add(checksum_u16(&v)),
            DecodingResult::F32(v) => checksum.wrapping_add(checksum_f32(&v)),
            _ => return Err(anyhow!("unexpected sample type from the tiff crate")),
        };
        if i + 1 < run.frames {
            dec.next_image()?;
        }
    }
    Ok(Measured { reader: Reader::TiffRs, per_frame_us, bytes_per_frame: bpf, checksum, open_us })
}

/// Vendored TinyTIFF (C), via FFI. Uncompressed single-sample classic TIFF only.
fn tinytiff(path: &Path, run: &Run) -> Result<Outcome> {
    use ffi::tinytiff::*;
    use std::ffi::CString;
    use std::os::raw::c_void;

    let f = &run.family;
    if f.compression != Compression::None || f.format.spp() > 1 || f.bigtiff {
        return Ok(unsupported(Reader::TinyTiff, "uncompressed single-sample classic TIFF only"));
    }

    let bpf = run.bytes_per_frame();
    let cpath = CString::new(path.to_string_lossy().as_bytes())?;
    let mut checksum: u64 = 0;

    unsafe {
        let t_open = Instant::now();
        let tiff = TinyTIFFReader_open(cpath.as_ptr());
        let open_us = t_open.elapsed().as_secs_f64() * 1e6;
        if tiff.is_null() || TinyTIFFReader_wasError(tiff) != 0 {
            if !tiff.is_null() {
                TinyTIFFReader_close(tiff);
            }
            return Ok(unsupported(Reader::TinyTiff, "open failed"));
        }

        let mut per_frame_us = Vec::with_capacity(run.frames);
        let mut buf = vec![0u8; bpf];
        let mut seen = 0usize;
        loop {
            let t = Instant::now();
            let ok = TinyTIFFReader_getSampleData(tiff, buf.as_mut_ptr() as *mut c_void, 0);
            let dt = t.elapsed().as_secs_f64() * 1e6;
            if ok == 0 {
                TinyTIFFReader_close(tiff);
                return Ok(unsupported(Reader::TinyTiff, format!("read failed at frame {seen}")));
            }
            per_frame_us.push(dt);
            checksum = checksum.wrapping_add(checksum_bytes(&buf));
            seen += 1;
            if seen >= run.frames
                || TinyTIFFReader_hasNext(tiff) == 0
                || TinyTIFFReader_readNext(tiff) == 0
            {
                break;
            }
        }
        TinyTIFFReader_close(tiff);
        if seen != run.frames {
            return Ok(unsupported(Reader::TinyTiff, format!("read {seen}/{} frames", run.frames)));
        }
        Ok(Outcome::Measured(Measured {
            reader: Reader::TinyTiff,
            per_frame_us,
            bytes_per_frame: bpf,
            checksum,
            open_us,
        }))
    }
}

/// System libtiff (C), via FFI. Reads every strip of each directory.
#[cfg(libtiff)]
fn libtiff(path: &Path, run: &Run) -> Outcome {
    use ffi::libtiff::*;
    use std::ffi::CString;
    use std::os::raw::c_void;

    let bpf = run.bytes_per_frame();
    let cpath = match CString::new(path.to_string_lossy().as_bytes()) {
        Ok(c) => c,
        Err(e) => return unsupported(Reader::LibTiff, e.to_string()),
    };
    let mode = CString::new("r").expect("static");
    let mut checksum: u64 = 0;

    unsafe {
        let t_open = Instant::now();
        let tif = TIFFOpen(cpath.as_ptr(), mode.as_ptr());
        let open_us = t_open.elapsed().as_secs_f64() * 1e6;
        if tif.is_null() {
            return unsupported(Reader::LibTiff, "TIFFOpen failed");
        }

        let mut per_frame_us = Vec::with_capacity(run.frames);
        let mut buf = vec![0u8; bpf];
        for f in 0..run.frames {
            let t = Instant::now();
            let nstrips = TIFFNumberOfStrips(tif);
            let stripsz = TIFFStripSize(tif);
            let mut off = 0usize;
            for s in 0..nstrips {
                let dst = buf.as_mut_ptr().add(off) as *mut c_void;
                let got = TIFFReadEncodedStrip(tif, s, dst, stripsz);
                if got < 0 {
                    TIFFClose(tif);
                    return unsupported(Reader::LibTiff, format!("strip read failed at frame {f}"));
                }
                off += got as usize;
            }
            per_frame_us.push(t.elapsed().as_secs_f64() * 1e6);
            checksum = checksum.wrapping_add(checksum_bytes(&buf[..off.min(bpf)]));
            if f + 1 < run.frames && TIFFReadDirectory(tif) == 0 {
                TIFFClose(tif);
                return unsupported(Reader::LibTiff, format!("out of directories at frame {f}"));
            }
        }
        TIFFClose(tif);
        Outcome::Measured(Measured {
            reader: Reader::LibTiff,
            per_frame_us,
            bytes_per_frame: bpf,
            checksum,
            open_us,
        })
    }
}

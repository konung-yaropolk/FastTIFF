//! The [`Frame`] read API: what it borrows, what it decodes, and what it
//! refuses.
//!
//! The decode core is held to the old readers bit for bit by the oracle in
//! `src/decode/oracle_tests.rs`. These are the promises that are new: a
//! multi-strip uncompressed frame borrowed straight from the mapping, a planar
//! plane read without touching the others, a short strip failing before its
//! predictor runs, the plane index checked rather than clamped, and the
//! destination contract.

use crate::common::open_tiff;
use fast_tiff_lib::units::{Display, File, Stored};
use fast_tiff_lib::{
    ByteOrder, Compression, Crop, DecodeScratch, DisplayMap, DisplayMaps, ErrorKind, Pixels,
    SampleType, TiffStack, TiffWriter, WriterOptions,
};
use std::io::Cursor;

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ftl-frame-{}-{name}", std::process::id()))
}

fn ramp(n: usize) -> Vec<u16> {
    (0..n).map(|i| (i * 7 + 3) as u16).collect()
}

fn write_u16(opts: WriterOptions, frames: &[&[u16]]) -> Vec<u8> {
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in frames {
        w.write_frame_u16(f).unwrap();
    }
    w.finish().unwrap().into_inner()
}

/// A libtiff-style uncompressed frame, sixteen strips back to back, is read
/// as the mapping's own bytes: no decode, no copy, `out` untouched. The old
/// readers borrowed only a single-strip frame and copied this one.
#[test]
fn a_contiguous_multi_strip_frame_is_borrowed_from_the_source() {
    let (w, h) = (64u32, 64u32);
    let pixels = ramp((w * h) as usize);
    let opts = WriterOptions::new(w, h, SampleType::U16).rows_per_strip(4);
    let path = temp("multi-strip.tif");
    std::fs::write(&path, write_u16(opts, &[&pixels])).unwrap();
    let stack = open_tiff(&path).unwrap();
    assert_eq!(stack.frames[0].strip_offsets.len(), 16);

    let frame = stack.frame(0).unwrap();
    let units = Display { range: None };
    assert!(frame.borrows(units, 0));
    let mut out = Vec::new();
    let mut scratch = DecodeScratch::new();
    let (px, maps) = frame.read(units, 0, &mut out, &mut scratch).unwrap();
    assert!(matches!(px, Pixels::Borrowed(_)) && px.is_source());
    let file = stack.data.as_ptr_range();
    assert!(file.contains(&(px.as_ptr() as *const u8)), "inside the mapping");
    assert_eq!(&*px, &pixels[..]);
    assert_eq!(maps, DisplayMaps::One(DisplayMap::Identity));
    assert!(out.is_empty(), "a borrowed read leaves `out` as it was");
    assert_eq!(scratch.held_bytes(), 0, "and needs no scratch");
    // The stored bytes borrow too, in the file's byte order.
    assert!(frame.borrows(Stored(stack.byte_order), 0));
    drop(stack);
    let _ = std::fs::remove_file(path);
}

/// A planar frame's plane decodes from its own strips only: a plane whose
/// strips are garbage cannot stop another from being read. The old readers
/// decompressed every plane for any one of them.
#[test]
fn a_planar_plane_reads_without_the_other_planes() {
    let (w, h) = (48u32, 32u32);
    let n = (w * h) as usize;
    let mut chunky = Vec::with_capacity(n * 3);
    for i in 0..n {
        for p in 0..3 {
            chunky.push(((i * (p + 2) + p * 1000) % 65536) as u16);
        }
    }
    // The writer takes planar frames plane after plane.
    let planar: Vec<u16> = (0..3)
        .flat_map(|p| (0..n).map(move |i| ((i * (p + 2) + p * 1000) % 65536) as u16))
        .collect();
    let opts = WriterOptions::new(w, h, SampleType::U16)
        .samples_per_pixel(3)
        .planar(true)
        .compression(Compression::Lzw)
        .rows_per_strip(8);
    let mut bytes = write_u16(opts, &[&planar]);
    let stack = TiffStack::from_bytes(bytes.clone()).unwrap();
    let info = &stack.frames[0];
    assert!(info.is_planar());
    let per_plane = info.strip_offsets.len() / 3;
    // Corrupt every strip of plane 2: LZW's first code becomes the end code.
    for j in 2 * per_plane..3 * per_plane {
        let off = info.strip_offsets[j] as usize;
        let len = info.strip_byte_counts[j] as usize;
        bytes[off..off + len].fill(0xFF);
    }
    let stack = TiffStack::from_bytes(bytes).unwrap();
    let frame = stack.frame(0).unwrap();
    let mut scratch = DecodeScratch::new();
    for p in 0..2 {
        let mut out = Vec::new();
        let (px, _) = frame
            .read(Display { range: None }, p, &mut out, &mut scratch)
            .unwrap_or_else(|e| panic!("plane {p}: {e}"));
        let want: Vec<u16> = (0..n).map(|i| chunky[i * 3 + p]).collect();
        assert_eq!(&*px, &want[..], "plane {p}");
    }
    let mut out = Vec::new();
    assert!(frame
        .read(Display { range: None }, 2, &mut out, &mut scratch)
        .is_err());
}

/// A strip that decodes short is an error, whichever strip it is — even the
/// middle one of a frame decoded on several cores — and never a frame with a
/// hole in it.
#[test]
fn a_frame_whose_middle_strip_is_short_is_malformed() {
    let (w, h) = (1024u32, 1024u32);
    let pixels = ramp((w * h) as usize);
    for compression in [Compression::Lzw, Compression::Deflate, Compression::None] {
        let opts = WriterOptions::new(w, h, SampleType::U16)
            .compression(compression)
            .predictor(true)
            .rows_per_strip(64);
        let stack = TiffStack::from_bytes(write_u16(opts, &[&pixels])).unwrap();
        // The middle strip's data cut to half: it decodes to fewer bytes
        // than its rows hold.
        let mut info = stack.frames[0].clone();
        let mut counts: Vec<u64> = info.strip_byte_counts.to_vec();
        let mid = counts.len() / 2;
        counts[mid] /= 2;
        info.strip_byte_counts = counts.into();
        for parallel in [false, true] {
            let _p = fast_tiff_lib::parallel_decode_scope(parallel);
            let frame = fast_tiff_lib::Frame::new(&stack.data, &info, stack.byte_order);
            let mut out = vec![0u16; (w * h) as usize];
            let e = frame
                .read_into(
                    Display { range: None },
                    0,
                    &mut [&mut out[..]],
                    &mut DecodeScratch::new(),
                )
                .unwrap_err();
            assert_eq!(
                e.kind(),
                ErrorKind::Malformed,
                "{compression:?} parallel {parallel}: {e}"
            );
        }
    }
}

/// The core refuses a plane past the last, and a destination of the wrong
/// size, rather than clamping or truncating; the old readers kept clamping.
#[test]
fn a_plane_past_the_last_or_a_wrong_destination_is_refused() {
    let pixels = ramp(16 * 8);
    let bytes = write_u16(WriterOptions::new(16, 8, SampleType::U16), &[&pixels]);
    let stack = TiffStack::from_bytes(bytes).unwrap();
    let frame = stack.frame(0).unwrap();
    let mut scratch = DecodeScratch::new();
    let mut out = Vec::new();
    let e = frame.read(File, 1, &mut out, &mut scratch).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    let mut short = vec![0f32; 16 * 8 - 1];
    let e = frame
        .read_into(File, 0, &mut [&mut short[..]], &mut scratch)
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert!(short.iter().all(|&v| v == 0.0), "nothing written");
    assert_eq!(
        stack.frame(1).unwrap_err().kind(),
        ErrorKind::InvalidInput,
        "a frame past the end"
    );
    // The old reader clamps, as it always has.
    assert_eq!(
        fast_tiff_lib::read_plane_f32(&stack.data, &stack.frames[0], stack.byte_order, 5).unwrap(),
        pixels.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
}

/// Every element of a destination is written: what it held before never
/// shows, whatever the route — borrowed source, direct decode, per-piece
/// conversion, assembled wide data.
#[test]
fn a_dirty_destination_comes_back_fully_written() {
    let pixels = ramp(40 * 30);
    for (compression, rps) in [(Compression::None, 30), (Compression::Lzw, 7)] {
        let opts = WriterOptions::new(40, 30, SampleType::U16)
            .compression(compression)
            .rows_per_strip(rps);
        let stack = TiffStack::from_bytes(write_u16(opts, &[&pixels])).unwrap();
        let frame = stack.frame(0).unwrap();
        let mut scratch = DecodeScratch::new();
        for fill in [0u16, 0xFFFF, 0xA5A5] {
            let mut out = vec![fill; pixels.len()];
            frame
                .read_into(Display { range: None }, 0, &mut [&mut out[..]], &mut scratch)
                .unwrap();
            assert_eq!(out, pixels, "{compression:?}, filled with {fill:#x}");
            let mut f = vec![f32::NAN; pixels.len()];
            frame
                .read_into(File, 0, &mut [&mut f[..]], &mut scratch)
                .unwrap();
            assert!(f.iter().zip(&pixels).all(|(&a, &b)| a == b as f32));
        }
    }
}

/// A crop reads what the full frame holds in the rows it covers, and its
/// coverage says which rows those are.
#[test]
fn a_cropped_frame_holds_the_rows_its_coverage_names() {
    let (w, h) = (32u32, 40u32);
    let pixels = ramp((w * h) as usize);
    let opts = WriterOptions::new(w, h, SampleType::U16)
        .compression(Compression::Deflate)
        .rows_per_strip(6);
    let stack = TiffStack::from_bytes(write_u16(opts, &[&pixels])).unwrap();
    let frame = stack.frame(0).unwrap();
    let mut scratch = DecodeScratch::new();
    for crop in [
        Crop::Rows(10..17),
        Crop::Region {
            cols: 3..9,
            rows: 20..40,
        },
        Crop::Sampled { rows: 0..40, step: 3 },
    ] {
        let (band, cov) = frame.crop(crop.clone()).unwrap();
        let mut out = Vec::new();
        let (px, _) = band
            .read(Display { range: None }, 0, &mut out, &mut scratch)
            .unwrap();
        assert_eq!(cov.cols, 0..w, "{crop:?}: strips keep the full width");
        for source_row in 0..h {
            if let Some(row) = cov.decoded_row_of(source_row) {
                let (a, b) = (row as usize * w as usize, source_row as usize * w as usize);
                assert_eq!(
                    &px[a..a + w as usize],
                    &pixels[b..b + w as usize],
                    "{crop:?} row {source_row}"
                );
            }
        }
    }
}

/// The stored frame in either byte order is the file's samples, swapped or
/// not; read as the host's order it is the samples themselves.
#[test]
fn the_stored_frame_is_the_samples_in_the_order_asked_for() {
    let pixels = ramp(24 * 10);
    let opts = WriterOptions::new(24, 10, SampleType::U16)
        .compression(Compression::PackBits)
        .predictor(true)
        .rows_per_strip(3);
    let stack = TiffStack::from_bytes(write_u16(opts, &[&pixels])).unwrap();
    let frame = stack.frame(0).unwrap();
    let mut scratch = DecodeScratch::new();
    let n = frame.stored_frame_len().unwrap();
    let mut host = vec![0u8; n];
    frame
        .stored_frame_into(ByteOrder::host(), &mut host, &mut scratch)
        .unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u16>(&host), &pixels[..]);
    let mut big = vec![0u8; n];
    frame
        .stored_frame_into(ByteOrder::Big, &mut big, &mut scratch)
        .unwrap();
    let want: Vec<u8> = pixels.iter().flat_map(|v| v.to_be_bytes()).collect();
    assert_eq!(big, want);
}

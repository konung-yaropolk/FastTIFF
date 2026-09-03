//! The fused per-strip predictor undo must equal the whole-frame pass it
//! replaced.
//!
//! `decode_native_bytes_opt` used to decompress every strip, concatenate, and
//! then run one predictor pass over the assembled frame. It now undoes the
//! predictor inside the per-strip loop, while the strip is still in cache — and
//! on the rayon path, on the worker that decompressed it. That is worth about
//! 2x on a large compressed frame, and it rests entirely on one claim:
//! differencing resets at every row, and a strip holds a whole number of rows,
//! so per-strip and whole-frame are the same computation.
//!
//! These tests attack that claim from four directions: a differential sweep
//! comparing a single-strip frame against a byte-identical multi-strip one
//! across every geometry, depth, layout and byte order; real
//! writer-produced stacks decoded serially and in parallel; the floating-point
//! predictor, whose undo is a different algorithm; and adversarial strip tables
//! that must not panic.
//!
//! Deliberately built on `TiffStack::from_bytes` and hand-made `FrameInfo`s, so
//! this suite runs in the `--no-default-features` (wasm-shaped) configuration
//! too, where the `mmap`-gated integration tests do not.

use fast_tiff_lib::index::{Compression as IndexCompression, FrameInfo, SampleFormat, Strips};
use fast_tiff_lib::{ByteOrder, Compression, SampleType, TiffStack, TiffWriter, WriterOptions};
use std::io::Cursor;

fn frame(
    w: u32,
    h: u32,
    bits: u16,
    spp: u16,
    planar: bool,
    pred: u16,
    fmt: SampleFormat,
) -> FrameInfo {
    FrameInfo {
        width: w,
        height: h,
        bits_per_sample: bits,
        samples_per_pixel: spp,
        sample_format: fmt,
        compression: IndexCompression::None,
        predictor: pred,
        photometric: if spp >= 3 { 2 } else { 1 },
        planar_config: if planar { 2 } else { 1 },
        tile_size: None,
        ink_set: 1,
        strip_offsets: vec![0u64].into(),
        strip_byte_counts: vec![0u64].into(),
        rows_per_strip: h,
    }
}

/// Same geometry, but split into strips of `rps` rows (planar: per plane).
fn stripify(f: &FrameInfo, rps: u32, sample_bytes: usize) -> FrameInfo {
    let mut out = f.clone();
    out.rows_per_strip = rps;
    let spp = (f.samples_per_pixel as usize).max(1);
    let (n_planes, row_bytes) = if f.planar_config == 2 && f.samples_per_pixel > 1 {
        (spp, f.width as usize * sample_bytes)
    } else {
        (1, f.width as usize * spp * sample_bytes)
    };
    let mut offs = Vec::new();
    let mut cnts = Vec::new();
    let mut pos = 0u64;
    for _ in 0..n_planes {
        let mut done = 0u32;
        while done < f.height {
            let rows = rps.min(f.height - done) as usize;
            offs.push(pos);
            cnts.push((rows * row_bytes) as u64);
            pos += (rows * row_bytes) as u64;
            done += rows as u32;
        }
    }
    out.strip_offsets = offs.into();
    out.strip_byte_counts = cnts.into();
    out
}

fn data(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i * 37 + seed * 101 + i / 5) % 256) as u8)
        .collect()
}

fn check(tag: &str, one: &FrameInfo, many: &FrameInfo, file: &[u8], order: ByteOrder) {
    let spp = (one.samples_per_pixel as usize).max(1);
    for p in 0..spp {
        let a = fast_tiff_lib::decode::read_plane_u16(file, one, order, Some((0.0, 1.0)), p);
        let b = fast_tiff_lib::decode::read_plane_u16(file, many, order, Some((0.0, 1.0)), p);
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "{tag}: plane {p} u16 differs"),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{tag}: errors differ"),
            (a, b) => panic!("{tag}: one Ok one Err: {:?} vs {:?}", a.is_ok(), b.is_ok()),
        }
    }
    // planes API too (exercises the fused-planes gather / deinterleave)
    let a = fast_tiff_lib::decode::read_planes_u16(file, one, order, Some((0.0, 1.0)));
    let b = fast_tiff_lib::decode::read_planes_u16(file, many, order, Some((0.0, 1.0)));
    if let (Ok(a), Ok(b)) = (a, b) {
        assert_eq!(a, b, "{tag}: planes u16 differ");
    }
    if one.bits_per_sample == 8 {
        for p in 0..spp {
            let a = fast_tiff_lib::decode::read_plane_u8(file, one, order, p).unwrap();
            let b = fast_tiff_lib::decode::read_plane_u8(file, many, order, p).unwrap();
            assert_eq!(a, b, "{tag}: plane {p} u8 differs");
        }
    }
    if one.bits_per_sample == 32 || one.bits_per_sample == 64 {
        for p in 0..spp {
            let a = fast_tiff_lib::decode::read_plane_f32(file, one, order, p).unwrap();
            let b = fast_tiff_lib::decode::read_plane_f32(file, many, order, p).unwrap();
            let differs = a
                .iter()
                .zip(b.iter())
                .position(|(x, y)| x.to_bits() != y.to_bits());
            assert!(
                differs.is_none(),
                "{tag}: plane {p} f32 differs at {:?}: {:?} vs {:?}",
                differs,
                differs.map(|i| a[i]),
                differs.map(|i| b[i])
            );
        }
    }
}

#[test]
fn fused_per_strip_matches_whole_frame() {
    let geoms = [
        (4u32, 8u32, 4u32),
        (4, 8, 3), // rps does not divide height
        (5, 7, 2),
        (3, 9, 1),
        (17, 5, 2),
        (1, 6, 4),
    ];
    for &(w, h, rps) in &geoms {
        for &bits in &[8u16, 16, 32, 64] {
            let sb = (bits / 8) as usize;
            for &spp in &[1u16, 2, 3, 4] {
                for &planar in &[false, true] {
                    for &(pred, fmt) in &[
                        (2u16, SampleFormat::UnsignedInt),
                        (2, SampleFormat::SignedInt),
                        (3, SampleFormat::Float),
                    ] {
                        if pred == 3 && bits != 32 && bits != 64 {
                            continue;
                        }
                        if pred == 3 && fmt != SampleFormat::Float {
                            continue;
                        }
                        for &order in &[ByteOrder::Little, ByteOrder::Big] {
                            let total = w as usize * h as usize * spp as usize * sb;
                            let file = data(total, (w + h + rps + bits as u32) as usize);
                            let mut one = frame(w, h, bits, spp, planar, pred, fmt);
                            one.strip_offsets = vec![0u64].into();
                            one.strip_byte_counts = vec![total as u64].into();
                            one.rows_per_strip = h;
                            let many = stripify(&one, rps, sb);
                            if many.strip_offsets.len() == 1 {
                                continue; // no split happened
                            }
                            let tag = format!(
                                "{w}x{h} rps{rps} bits{bits} spp{spp} planar{planar} pred{pred} {order:?}"
                            );
                            check(&tag, &one, &many, &file, order);
                        }
                    }
                }
            }
        }
    }
}

fn write_u16(w: u32, h: u32, rps: u32, comp: Compression, pred: bool, src: &[u16]) -> TiffStack {
    let opts = WriterOptions::new(w, h, SampleType::U16)
        .compression(comp)
        .predictor(pred)
        .rows_per_strip(rps);
    let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    wr.write_frame_u16(src).unwrap();
    let bytes = wr.finish().unwrap().into_inner();
    TiffStack::from_bytes(bytes).unwrap()
}

fn roundtrip(w: u32, h: u32, rps: u32, comp: Compression, pred: bool) {
    let n = (w * h) as usize;
    let src: Vec<u16> = (0..n)
        .map(|i| ((i * 7919 + i / 3) % 60000) as u16)
        .collect();
    let stack = write_u16(w, h, rps, comp, pred, &src);
    let frame = &stack.frames[0];
    assert!(
        frame.strip_offsets.len() > 1,
        "{w}x{h} rps{rps}: wanted multiple strips, got {}",
        frame.strip_offsets.len()
    );
    for &par in &[false, true] {
        fast_tiff_lib::set_parallel_decode(par);
        let got = fast_tiff_lib::read_frame_u16(&stack.data, frame, stack.byte_order, None)
            .unwrap()
            .into_owned();
        fast_tiff_lib::set_parallel_decode(false);
        let bad = got.iter().zip(src.iter()).position(|(a, b)| a != b);
        assert!(
            bad.is_none(),
            "{w}x{h} rps{rps} {comp:?} pred{pred} par{par}: first mismatch at {bad:?} ({:?} vs {:?})",
            bad.map(|i| got[i]),
            bad.map(|i| src[i])
        );
    }
}

#[test]
fn compressed_multi_strip_predictor_serial_and_parallel() {
    for &comp in &[
        Compression::None,
        Compression::Lzw,
        Compression::Deflate,
        Compression::PackBits,
    ] {
        for &pred in &[false, true] {
            roundtrip(64, 40, 7, comp, pred);
            roundtrip(1024, 1024, 32, comp, pred);
            roundtrip(1024, 1030, 32, comp, pred); // height % rps != 0
            roundtrip(1024, 1030, 1, comp, pred); // one row per strip
        }
    }
}

#[test]
fn float_predictor_multi_strip() {
    let (w, h) = (1024u32, 1030u32);
    let n = (w * h) as usize;
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1000.0).collect();
    for &comp in &[Compression::None, Compression::Deflate] {
        let opts = WriterOptions::new(w, h, SampleType::F32)
            .compression(comp)
            .predictor(true)
            .rows_per_strip(32);
        let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        wr.write_frame_f32(&src).unwrap();
        let bytes = wr.finish().unwrap().into_inner();
        let stack = TiffStack::from_bytes(bytes).unwrap();
        let frame = &stack.frames[0];
        assert_eq!(frame.predictor, 3);
        assert!(frame.strip_offsets.len() > 1);
        for &par in &[false, true] {
            fast_tiff_lib::set_parallel_decode(par);
            let got =
                fast_tiff_lib::read_plane_f32(&stack.data, frame, stack.byte_order, 0).unwrap();
            fast_tiff_lib::set_parallel_decode(false);
            let bad = got
                .iter()
                .zip(src.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                bad.is_none(),
                "{comp:?} par={par}: first mismatch at {bad:?}"
            );
        }
    }
}

/// Planar multi-strip with the integer predictor, chunky RGB too.
#[test]
fn planar_and_chunky_multi_strip_predictor() {
    for &planar in &[false, true] {
        for &comp in &[Compression::None, Compression::Deflate, Compression::Lzw] {
            let (w, h, spp) = (600u32, 43u32, 3u16);
            let n = (w * h) as usize * spp as usize;
            let src: Vec<u16> = (0..n)
                .map(|i| ((i * 4099 + i / 11) % 65000) as u16)
                .collect();
            let opts = WriterOptions::new(w, h, SampleType::U16)
                .samples_per_pixel(spp)
                .planar(planar)
                .compression(comp)
                .predictor(true)
                .rows_per_strip(5);
            let mut wr = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
            wr.write_frame_u16(&src).unwrap();
            let bytes = wr.finish().unwrap().into_inner();
            let stack = TiffStack::from_bytes(bytes).unwrap();
            let frame = &stack.frames[0];
            assert!(frame.strip_offsets.len() > 1);
            let planes =
                fast_tiff_lib::read_planes_u16(&stack.data, frame, stack.byte_order, None).unwrap();
            let npx = (w * h) as usize;
            for p in 0..spp as usize {
                let want: Vec<u16> = if planar {
                    src[p * npx..(p + 1) * npx].to_vec()
                } else {
                    (0..npx).map(|i| src[i * spp as usize + p]).collect()
                };
                let bad = planes[p].iter().zip(want.iter()).position(|(a, b)| a != b);
                assert!(
                    bad.is_none(),
                    "planar{planar} {comp:?} plane{p}: mismatch at {bad:?} ({:?} vs {:?})",
                    bad.map(|i| planes[p][i]),
                    bad.map(|i| want[i])
                );
            }
        }
    }
}

fn bare_frame(w: u32, h: u32, bits: u16, spp: u16, planar: bool, pred: u16) -> FrameInfo {
    FrameInfo {
        width: w,
        height: h,
        bits_per_sample: bits,
        samples_per_pixel: spp,
        sample_format: SampleFormat::UnsignedInt,
        compression: IndexCompression::None,
        predictor: pred,
        photometric: if spp >= 3 { 2 } else { 1 },
        planar_config: if planar { 2 } else { 1 },
        tile_size: None,
        ink_set: 1,
        strip_offsets: Strips::None,
        strip_byte_counts: Strips::None,
        rows_per_strip: h,
    }
}

#[test]
fn adversarial_strip_tables_do_not_panic() {
    let file = vec![0x5Au8; 1 << 16];
    let mut cases = 0usize;
    for &(w, h) in &[(1u32, 1u32), (3, 7), (8, 8), (5, 13), (64, 3)] {
        for &bits in &[8u16, 16, 32, 64] {
            for &spp in &[1u16, 3, 4] {
                for &planar in &[false, true] {
                    for &pred in &[1u16, 2, 3, 5] {
                        for &rps in &[0u32, 1, 2, 3, h, h + 5, u32::MAX] {
                            for &n_strips in &[0usize, 1, 2, 3, 7, 40] {
                                for &bc in &[0u64, 1, 7, 4096] {
                                    let mut f = bare_frame(w, h, bits, spp, planar, pred);
                                    f.rows_per_strip = rps;
                                    f.strip_offsets =
                                        (0..n_strips).map(|i| (i as u64) * 13).collect();
                                    f.strip_byte_counts = (0..n_strips).map(|_| bc).collect();
                                    for &order in &[ByteOrder::Little, ByteOrder::Big] {
                                        let _ =
                                            fast_tiff_lib::read_frame_u16(&file, &f, order, None);
                                        let _ =
                                            fast_tiff_lib::read_planes_u16(&file, &f, order, None);
                                        let _ = fast_tiff_lib::read_frame_u8(&file, &f, order);
                                        let _ = fast_tiff_lib::read_planes_u8(&file, &f, order);
                                        if bits == 32 || bits == 64 {
                                            let _ =
                                                fast_tiff_lib::read_planes_f32(&file, &f, order);
                                        }
                                        cases += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("{cases} adversarial cases");
}

/// Byte-count larger than the destination (padded strips) and shorter than it.
#[test]
fn padded_and_short_strips() {
    let file = vec![0x11u8; 1 << 20];
    for &pred in &[1u16, 2, 3] {
        for &planar in &[false, true] {
            for &(w, h, rps) in &[(4u32, 9u32, 2u32), (7, 5, 3), (16, 16, 4)] {
                let mut f = bare_frame(w, h, 32, 3, planar, pred);
                f.sample_format = if pred == 3 {
                    SampleFormat::Float
                } else {
                    SampleFormat::UnsignedInt
                };
                f.rows_per_strip = rps;
                let n_planes = if planar { 3usize } else { 1 };
                let per_plane = (h.div_ceil(rps)) as usize;
                let n = n_planes * per_plane;
                for &mult in &[0.5f64, 1.0, 2.0, 4.0] {
                    let row_bytes = if planar {
                        w as usize * 4
                    } else {
                        w as usize * 12
                    };
                    let bc = ((rps as usize * row_bytes) as f64 * mult) as u64;
                    f.strip_offsets = (0..n).map(|i| (i * 1024) as u64).collect();
                    f.strip_byte_counts = (0..n).map(|_| bc).collect();
                    let _ = fast_tiff_lib::read_planes_f32(&file, &f, ByteOrder::Little);
                    let _ = fast_tiff_lib::read_planes_u16(&file, &f, ByteOrder::Little, None);
                }
            }
        }
    }
}

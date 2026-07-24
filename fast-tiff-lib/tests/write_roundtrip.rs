//! End-to-end writer→reader round-trips through real files: everything a
//! consumer does — `TiffWriter::create` → `TiffStack::open` → decode — with
//! nothing stubbed. (In-memory codec/structure tests live in
//! `src/encode_tests.rs`.)

use fast_tiff_lib::{
    frame_float_minmax, read_frame_f32, read_frame_u16, read_frame_u8, Compression, DisplayMode,
    MetadataFormat, SampleType, StackMetaWrite, TiffStack, TiffWriter, WriterOptions,
};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A collision-free scratch path; removed by `Cleanup` even if the test panics.
fn unique_temp_path(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("fast_tiff_write_{}_{}_{}", std::process::id(), n, name))
}

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn u16_stack_roundtrips_and_scrubs_zero_copy() {
    let path = unique_temp_path("u16.tif");
    let _cleanup = Cleanup(path.clone());

    let frames: Vec<Vec<u16>> = (0..3)
        .map(|f| (0..4 * 3).map(|i| (f * 1000 + i * 17) as u16).collect())
        .collect();

    let mut w = TiffWriter::create(&path, WriterOptions::new(4, 3, SampleType::U16)).unwrap();
    for f in &frames {
        w.write_frame_u16(f).unwrap();
    }
    assert_eq!(w.frames_written(), 3);
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 3);
    for (i, expected) in frames.iter().enumerate() {
        // Prefetch is a pure performance hint: must be safe before any read.
        stack.prefetch_frame(&stack.frames[i]);
        let got = read_frame_u16(&stack.mmap, &stack.frames[i], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &expected[..], "frame {i}");
        // The writer's default layout IS the reader's zero-copy fast path.
        #[cfg(target_endian = "little")]
        assert!(matches!(got, Cow::Borrowed(_)), "frame {i} should decode zero-copy");
    }
}

#[test]
fn u8_and_f32_typed_frames_roundtrip() {
    let path8 = unique_temp_path("u8.tif");
    let _c8 = Cleanup(path8.clone());
    let pixels8: Vec<u8> = (0..25).map(|i| (i * 11 % 256) as u8).collect();
    let mut w = TiffWriter::create(&path8, WriterOptions::new(5, 5, SampleType::U8)).unwrap();
    w.write_frame_u8(&pixels8).unwrap();
    w.finish().unwrap();
    let stack = TiffStack::open(&path8).unwrap();
    let got = read_frame_u8(&stack.mmap, &stack.frames[0], stack.byte_order).unwrap();
    assert_eq!(got.as_ref(), &pixels8[..]);

    let pathf = unique_temp_path("f32.tif");
    let _cf = Cleanup(pathf.clone());
    let pixelsf: Vec<f32> = vec![-1.5, 0.0, 0.25, 100.0, f32::MIN_POSITIVE, 6.5e12];
    let mut w = TiffWriter::create(&pathf, WriterOptions::new(3, 2, SampleType::F32)).unwrap();
    w.write_frame_f32(&pixelsf).unwrap();
    w.finish().unwrap();
    let stack = TiffStack::open(&pathf).unwrap();
    let got = read_frame_f32(&stack.mmap, &stack.frames[0], stack.byte_order).unwrap();
    assert_eq!(got.as_ref(), &pixelsf[..]);
}

#[test]
fn compressed_multistrip_stack_roundtrips() {
    // LZW + predictor, 2-row strips over an odd height, several frames: the
    // most decode-hostile layout the writer can produce.
    let path = unique_temp_path("lzw.tif");
    let _cleanup = Cleanup(path.clone());

    let frames: Vec<Vec<u16>> = (0..2)
        .map(|f| (0..6 * 5).map(|i| (f * 7 + i * 31 % 900) as u16).collect())
        .collect();
    let opts = WriterOptions::new(6, 5, SampleType::U16)
        .compression(Compression::Lzw)
        .predictor(true)
        .rows_per_strip(2);
    let mut w = TiffWriter::create(&path, opts).unwrap();
    for f in &frames {
        w.write_frame_u16(f).unwrap();
    }
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames[0].compression, Compression::Lzw);
    assert_eq!(stack.frames[0].strip_offsets.len(), 3);
    for (i, expected) in frames.iter().enumerate() {
        let got = read_frame_u16(&stack.mmap, &stack.frames[i], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &expected[..], "frame {i}");
    }
}

#[test]
fn rgb8_stack_opens_as_rgb() {
    let path = unique_temp_path("rgb.tif");
    let _cleanup = Cleanup(path.clone());
    // 2x2 chunky RGB.
    let frame: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30];
    let mut w =
        TiffWriter::create(&path, WriterOptions::new(2, 2, SampleType::U8).samples_per_pixel(3)).unwrap();
    w.write_frame_bytes(&frame).unwrap();
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert!(stack.frames[0].is_rgb());
    let red = fast_tiff_lib::read_plane_u8(&stack.mmap, &stack.frames[0], stack.byte_order, 0).unwrap();
    assert_eq!(red, vec![255, 0, 0, 10]);
}

/// Writing planar must produce a file whose *decoded planes* are identical to
/// the chunky file holding the same image — same pixels, different byte layout
/// on disk. Run across the strip/compression/predictor combinations, since
/// planar changes how strips are split (per plane) and how the predictor
/// strides (by 1, not by `spp`).
#[test]
fn planar_rgb_roundtrips_and_matches_chunky() {
    // 4x3 RGB16: distinct values per sample so a mixed-up plane can't pass.
    let (w, h, spp) = (4usize, 3usize, 3usize);
    let chunky: Vec<u16> =
        (0..w * h).flat_map(|i| [i as u16, 1000 + i as u16, 2000 + i as u16]).collect();
    let planar: Vec<u16> =
        (0..spp).flat_map(|p| (0..w * h).map(|i| chunky[i * spp + p]).collect::<Vec<_>>()).collect();

    for (label, compression, predictor, rows_per_strip) in [
        ("plain", Compression::None, false, None),
        ("multistrip", Compression::None, false, Some(2)),
        ("deflate+pred", Compression::Deflate, true, Some(2)),
        ("lzw", Compression::Lzw, false, None),
    ] {
        let base = || {
            let mut o = WriterOptions::new(w as u32, h as u32, SampleType::U16)
                .samples_per_pixel(spp as u16)
                .compression(compression)
                .predictor(predictor);
            if let Some(r) = rows_per_strip {
                o = o.rows_per_strip(r);
            }
            o
        };

        let p_path = unique_temp_path(&format!("planar_{label}.tif"));
        let _pc = Cleanup(p_path.clone());
        let mut pw = TiffWriter::create(&p_path, base().planar(true)).unwrap();
        pw.write_frame_u16(&planar).unwrap();
        pw.finish().unwrap();

        let c_path = unique_temp_path(&format!("chunky_{label}.tif"));
        let _cc = Cleanup(c_path.clone());
        let mut cw = TiffWriter::create(&c_path, base()).unwrap();
        cw.write_frame_u16(&chunky).unwrap();
        cw.finish().unwrap();

        let ps = TiffStack::open(&p_path).unwrap();
        let cs = TiffStack::open(&c_path).unwrap();
        let (pf, cf) = (&ps.frames[0], &cs.frames[0]);

        assert_eq!(pf.planar_config, 2, "{label}: PlanarConfiguration tag");
        assert!(pf.is_planar(), "{label}: should read back as planar");
        assert!(pf.is_rgb(), "{label}: 3 samples is still RGB");
        assert_eq!(cf.planar_config, 1, "{label}: chunky writes no tag (default 1)");
        // Planar splits strips per plane, so it has `spp` times as many.
        assert_eq!(
            pf.strip_offsets.len(),
            cf.strip_offsets.len() * spp,
            "{label}: strips per frame"
        );

        for p in 0..spp {
            let got = fast_tiff_lib::read_plane_u16(&ps.mmap, pf, ps.byte_order, None, p).unwrap();
            let want = fast_tiff_lib::read_plane_u16(&cs.mmap, cf, cs.byte_order, None, p).unwrap();
            assert_eq!(got, want, "{label}: plane {p}");
            // ...and against the source data, so a symmetric bug in both paths
            // can't make this vacuously true.
            let expect: Vec<u16> = (0..w * h).map(|i| chunky[i * spp + p]).collect();
            assert_eq!(got, expect, "{label}: plane {p} vs source");
        }
    }
}

/// `planar(true)` on single-sample data is a no-op, not an error: TIFF6 calls
/// PlanarConfiguration irrelevant there, and the two layouts are byte-identical.
#[test]
fn planar_is_ignored_for_single_sample_frames() {
    let path = unique_temp_path("planar_gray.tif");
    let _cleanup = Cleanup(path.clone());
    let frame: Vec<u16> = (0..8).collect();
    let opts = WriterOptions::new(4, 2, SampleType::U16).planar(true);
    let mut w = TiffWriter::create(&path, opts).unwrap();
    w.write_frame_u16(&frame).unwrap();
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames[0].planar_config, 1, "no tag written -> chunky default");
    assert!(!stack.frames[0].is_planar());
    let got = read_frame_u16(&stack.mmap, &stack.frames[0], stack.byte_order, None).unwrap();
    assert_eq!(got.as_ref(), &frame[..]);
}

#[test]
fn imagej_hyperstack_metadata_roundtrips_through_stack_meta() {
    let path = unique_temp_path("ij.tif");
    let _cleanup = Cleanup(path.clone());

    // 2 channels x 3 time frames = 6 planes.
    let opts = WriterOptions::new(2, 2, SampleType::U16).metadata(
        StackMetaWrite::new(2, 1)
            .mode(DisplayMode::Composite)
            .fps(12.5)
            .frame_interval_s(0.08)
            .unit("um")
            .range(100.0, 4000.0)
            .calibration(2.0, 0.5)
            .spacing(0.3)
            .loop_playback(false)
            .extra("vunit", "V"),
    );
    let mut w = TiffWriter::create(&path, opts).unwrap();
    for i in 0..6u16 {
        w.write_frame_u16(&[i, i + 1, i + 2, i + 3]).unwrap();
    }
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    let meta = &stack.meta;
    assert_eq!(meta.source_format, MetadataFormat::ImageJ);
    assert_eq!(meta.channels, 2);
    assert_eq!(meta.slices, 1);
    assert_eq!(meta.frames, 3);
    assert_eq!(meta.mode, DisplayMode::Composite);
    assert_eq!(meta.fps, Some(12.5));
    assert_eq!(meta.frame_interval_s, Some(0.08));
    assert_eq!(meta.unit.as_deref(), Some("um"));
    assert_eq!(meta.calibration, Some((2.0, 0.5)));
    assert_eq!(meta.spacing, Some(0.3));
    assert_eq!(meta.loop_playback, Some(false));
    assert_eq!(meta.channel_display.len(), 2);
    assert_eq!(meta.channel_display[0].range, Some((100.0, 4000.0)));
    // The raw tag-270 text is exposed verbatim alongside the parsed view,
    // including keys without a parsed field (the extra() escape hatch).
    let desc = stack.description.as_deref().expect("raw description exposed");
    assert!(desc.contains("vunit=V"), "raw description keeps extra keys:\n{desc}");
}

/// The same neutral metadata builder, written as OME instead of ImageJ, must
/// produce a file that opens as OME with the dimensions, pixel size, and pixels
/// intact — the end-to-end proof of OME write+read through a real file.
#[test]
fn ome_metadata_roundtrips_through_stack_meta() {
    let path = unique_temp_path("ome.tif");
    let _cleanup = Cleanup(path.clone());

    // 2 channels x 3 time frames = 6 planes.
    let opts = WriterOptions::new(2, 2, SampleType::U16)
        .metadata_format(MetadataFormat::Ome)
        .metadata(
            StackMetaWrite::new(2, 1)
                .mode(DisplayMode::Composite)
                .unit("µm")
                .pixel_size(0.1, 0.1)
                .frame_interval_s(1.5)
                .channel("DAPI", [0, 0, 255])
                .channel("GFP", [0, 255, 0]),
        );
    let mut w = TiffWriter::create(&path, opts).unwrap();
    let frames: Vec<[u16; 4]> = (0..6).map(|i| [i, i + 1, i + 2, i + 3]).collect();
    for f in &frames {
        w.write_frame_u16(f).unwrap();
    }
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    let meta = &stack.meta;
    assert_eq!(meta.source_format, MetadataFormat::Ome);
    assert_eq!((meta.channels, meta.slices, meta.frames), (2, 1, 3));
    assert_eq!(meta.mode, DisplayMode::Composite);
    assert_eq!(meta.unit.as_deref(), Some("µm"));
    assert_eq!(meta.pixel_width, Some(0.1));
    assert_eq!(meta.pixel_height, Some(0.1));
    assert_eq!(meta.frame_interval_s, Some(1.5));
    // The raw description is genuinely OME-XML, not ImageJ key=value.
    let desc = stack.description.as_deref().expect("raw description");
    assert!(desc.contains("<OME"), "description should be OME-XML:\n{desc}");
    // Pixels survive: each written frame decodes back to what went in.
    for (i, f) in frames.iter().enumerate() {
        let got = read_frame_u16(&stack.mmap, &stack.frames[i], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &f[..], "frame {i}");
    }
}

#[test]
fn forced_bigtiff_file_roundtrips_through_stack_open() {
    let path = unique_temp_path("big.tif");
    let _cleanup = Cleanup(path.clone());

    let frames: Vec<Vec<u16>> = (0..3)
        .map(|f| (0..4 * 3).map(|i| (f * 2000 + i * 13) as u16).collect())
        .collect();
    let opts = WriterOptions::new(4, 3, SampleType::U16).bigtiff(true);
    let mut w = TiffWriter::create(&path, opts).unwrap();
    for f in &frames {
        w.write_frame_u16(f).unwrap();
    }
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.flavor, fast_tiff_lib::TiffFlavor::Big);
    assert_eq!(stack.frames.len(), 3);
    for (i, expected) in frames.iter().enumerate() {
        let got = read_frame_u16(&stack.mmap, &stack.frames[i], stack.byte_order, None).unwrap();
        assert_eq!(got.as_ref(), &expected[..], "frame {i}");
        // BigTIFF changes only the directory layout — the zero-copy pixel
        // fast path is identical.
        #[cfg(target_endian = "little")]
        assert!(matches!(got, Cow::Borrowed(_)), "frame {i} should decode zero-copy");
    }
}

#[test]
fn verbatim_description_roundtrips_raw() {
    let path = unique_temp_path("desc.tif");
    let _cleanup = Cleanup(path.clone());

    let text = "acquired with rig 3\nlaser=488nm\noperator=YA";
    let opts = WriterOptions::new(1, 1, SampleType::U8).description(text);
    let mut w = TiffWriter::create(&path, opts).unwrap();
    w.write_frame_u8(&[42]).unwrap();
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.description.as_deref(), Some(text));
    // Non-ImageJ text must not be mistaken for hyperstack metadata.
    assert_eq!(stack.meta.channels, 1);
}

#[test]
fn f64_stack_opens_and_downcasts_to_f32() {
    let path = unique_temp_path("f64.tif");
    let _cleanup = Cleanup(path.clone());

    let frames: Vec<Vec<f64>> = (0..2)
        .map(|f| (0..4 * 3).map(|i| f as f64 * 10.0 + i as f64 / 3.0).collect())
        .collect();

    let mut w = TiffWriter::create(&path, WriterOptions::new(4, 3, SampleType::F64)).unwrap();
    for f in &frames {
        w.write_frame_f64(f).unwrap();
    }
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 2);
    assert_eq!(stack.frames[0].bits_per_sample, 64);
    for (i, expected) in frames.iter().enumerate() {
        let got = read_frame_f32(&stack.mmap, &stack.frames[i], stack.byte_order).unwrap();
        let want: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
        assert_eq!(got.as_ref(), &want[..], "frame {i}");
    }
    // 64-bit float auto-ranges to its own data min/max (like 32-bit float).
    assert!(frame_float_minmax(&stack.mmap, &stack.frames[0], stack.byte_order).unwrap().is_some());
}

#[test]
fn u64_stack_opens_and_rescales_to_display_space() {
    let path = unique_temp_path("u64.tif");
    let _cleanup = Cleanup(path.clone());

    // A 2x2 frame whose min (0) and max (2^40) bracket the display range; the
    // two interior values sit at 1/4 and 1/2 of the span.
    let vals: [u64; 4] = [0, 1u64 << 38, 1u64 << 39, 1u64 << 40];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut w = TiffWriter::create(&path, WriterOptions::new(2, 2, SampleType::U64)).unwrap();
    w.write_frame_bytes(&data).unwrap(); // no typed u64 writer (like U32/I32)
    w.finish().unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames[0].bits_per_sample, 64);
    let got = read_frame_u16(&stack.mmap, &stack.frames[0], stack.byte_order, None).unwrap();
    // 2^38 / 2^40 = 0.25 -> 16384, 2^39 / 2^40 = 0.5 -> 32768, 2^40 -> 65535.
    assert_eq!(got.as_ref(), &[0, 16384, 32768, 65535]);
}

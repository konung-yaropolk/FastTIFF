//! Inputs that once made a public reader panic, kept as a permanent guard.
//!
//! The library's contract for malformed data is `Ok` or `Err` — **never** a
//! panic, and never a wrapped length that drives an out-of-bounds read. The
//! fuzz targets in `fuzz/` enforce that on nightly + libFuzzer; this replays
//! the specific inputs that have broken it, on stable, on every platform, in
//! the ordinary `cargo test` run.
//!
//! Drop any new crash artifact into `tests/fuzz-regressions/` and it is picked
//! up automatically — no code change needed.
//!
//! **Reproducing a fuzz crash locally:** build with overflow checks, the way
//! `cargo fuzz` does. A plain `cargo test --release` silently *wraps* on
//! overflow and will happily report success on an input that crashes CI:
//!
//! ```text
//! RUSTFLAGS="-C debug-assertions -C overflow-checks" cargo test -p fast-tiff-lib
//! ```

use std::panic::{catch_unwind, AssertUnwindSafe};

/// The `tiff_decode` fuzz target's call sequence, verbatim. Keep the two in
/// step: a reader added there should be added here.
fn exercise(data: &[u8]) {
    let Ok(stack) = fast_tiff_lib::TiffStack::from_bytes(data.to_vec()) else {
        return;
    };
    let Some(frame) = stack.frames.first() else {
        return;
    };
    let order = stack.byte_order;
    let b = &stack.data;

    let _ = fast_tiff_lib::read_frame_u16(b, frame, order, None);
    let _ = fast_tiff_lib::read_frame_u8(b, frame, order);
    let _ = fast_tiff_lib::read_frame_f32(b, frame, order);
    let _ = fast_tiff_lib::frame_float_minmax(b, frame, order);

    for plane in [0usize, 1, 3, 7] {
        let _ = fast_tiff_lib::read_plane_u16(b, frame, order, None, plane);
        let _ = fast_tiff_lib::read_plane_u8(b, frame, order, plane);
        let _ = fast_tiff_lib::read_plane_f32(b, frame, order, plane);
    }

    let _ = fast_tiff_lib::read_planes_u16(b, frame, order, None);
    let _ = fast_tiff_lib::read_planes_u8(b, frame, order);
    let _ = fast_tiff_lib::read_planes_f32(b, frame, order);

    let mut v16 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u16_into(b, frame, order, None, &mut v16);
    let mut v8 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u8_into(b, frame, order, &mut v8);
    let mut vf = Vec::new();
    let _ = fast_tiff_lib::read_frame_f32_into(b, frame, order, &mut vf);

    // Crops build a `FrameInfo` the indexer never produced — a spliced strip or
    // tile table over a shrunken frame — and hand it to the readers above, so
    // they get the same treatment.
    // The last of each set is backwards on purpose — built rather than written
    // as a literal, which the lints reject as a mistake. Here it is the input a
    // caller produces from an inverted drag, and it must not panic.
    let bad_cols = std::ops::Range {
        start: 9u32,
        end: 2,
    };
    let bad_rows = std::ops::Range {
        start: 7u32,
        end: 3,
    };
    let backwards_band = std::ops::Range {
        start: 7u32,
        end: 2,
    };
    for (cols, rows) in [
        (0u32..1, 0u32..1),
        (0..u32::MAX, 0..u32::MAX),
        (2..9, 3..7),
        (bad_cols, bad_rows),
    ] {
        if let Ok(region) = frame.crop(cols, rows) {
            let r = &region.frame;
            let _ = fast_tiff_lib::read_planes_u8(b, r, order);
            let _ = fast_tiff_lib::read_planes_u16(b, r, order, None);
            let _ = fast_tiff_lib::read_frame_f32(b, r, order);
        }
    }
    for rows in [
        0u32..1,
        0..u32::MAX,
        3..9,
        u32::MAX - 1..u32::MAX,
        backwards_band,
    ] {
        if let Ok(band) = frame.crop_rows(rows) {
            let f = &band.frame;
            let _ = fast_tiff_lib::read_planes_u8(b, f, order);
            let _ = fast_tiff_lib::read_planes_u16(b, f, order, None);
            let _ = fast_tiff_lib::read_planes_f32(b, f, order);
            let _ = fast_tiff_lib::read_planes_rgb_u8(b, f, order);
            if let Ok(inner) = f.crop_rows(0..2) {
                let _ = fast_tiff_lib::read_planes_u8(b, &inner.frame, order);
            }
        }
    }
}

#[test]
fn known_crashes_no_longer_panic() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fuzz-regressions");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("tests/fuzz-regressions") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "tif") {
            continue;
        }
        let data = std::fs::read(&path).expect("read artifact");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let started = std::time::Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| exercise(&data)));
        let took = started.elapsed();
        assert!(
            r.is_ok(),
            "{name} panicked — the Ok-or-Err contract is broken again"
        );
        // Some of these are *slow* units rather than crashes: a kilobyte of
        // input that once kept a reader busy for minutes, which for a library
        // taking untrusted bytes is a denial of service whether or not it ever
        // returns. A generous ceiling, so the bound catches a regression to
        // minutes without turning a slow CI machine into a false failure.
        assert!(
            took < std::time::Duration::from_secs(20),
            "{name} took {took:?} — a small input must not be able to occupy a reader indefinitely"
        );
        checked += 1;
    }
    assert!(checked > 0, "no artifacts found in {}", dir.display());
    println!("{checked} known-bad input(s) replayed cleanly");
}

/// Tile dimensions are their own tags, independent of the image size, so a
/// 40x36 image is free to declare 65535x65535 tiles. One scratch buffer for a
/// tile that size is twelve gigabytes — and `vec![0; n]` *aborts* when the
/// allocator refuses, which an embedder cannot catch. So the geometry has to be
/// refused before anything is allocated for it.
#[test]
fn absurd_tile_dimensions_are_refused_before_allocating() {
    use fast_tiff_lib::{ByteOrder, Compression, FrameInfo, SampleFormat};
    let frame = FrameInfo {
        width: 40,
        height: 36,
        bits_per_sample: 8,
        samples_per_pixel: 3,
        sample_format: SampleFormat::UnsignedInt,
        compression: Compression::Lzw,
        predictor: 1,
        photometric: 2,
        planar_config: 1,
        tile_size: Some((65_535, 65_535)),
        ink_set: 1,
        strip_offsets: vec![8].into(),
        strip_byte_counts: vec![16].into(),
        rows_per_strip: 65_535,
    };
    let data = vec![0u8; 256];
    let err = fast_tiff_lib::read_planes_u8(&data, &frame, ByteOrder::Little)
        .expect_err("a twelve-gigabyte tile must not be attempted");
    assert!(
        err.to_string().contains("tiles over a"),
        "should name the tile geometry as the problem, got: {err}"
    );
}

/// The geometry that produced the first artifact, asserted directly so the
/// intent survives even if the file is ever lost.
#[test]
fn frame_geometry_overflow_is_an_error_not_a_panic() {
    use fast_tiff_lib::{FrameInfo, SampleFormat};
    // 2147483648 x 2147483648 x 4 samples = usize::MAX + 1 exactly, on 64-bit.
    let frame = FrameInfo {
        width: 2_147_483_648,
        height: 2_147_483_648,
        bits_per_sample: 16,
        samples_per_pixel: 4,
        sample_format: SampleFormat::UnsignedInt,
        compression: fast_tiff_lib::Compression::None,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        tile_size: None,
        ink_set: 1,
        strip_offsets: vec![8].into(),
        strip_byte_counts: vec![16].into(),
        rows_per_strip: 1,
    };
    assert!(
        frame.pixel_count().is_ok(),
        "w*h alone still fits on 64-bit"
    );
    let err = frame.sample_count().expect_err("w*h*spp must overflow");
    assert!(
        err.to_string().contains("overflows address space"),
        "got: {err}"
    );
    assert!(frame.decoded_len().is_err(), "and so must the byte length");

    // The reader has to surface that as an error, not trap on the multiply.
    let data = vec![0u8; 64];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fast_tiff_lib::read_frame_u16(&data, &frame, fast_tiff_lib::ByteOrder::Little, None)
            .is_err()
    }));
    assert_eq!(
        r.ok(),
        Some(true),
        "read_frame_u16 must return Err, not panic"
    );
}

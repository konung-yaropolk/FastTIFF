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
    let Ok(stack) = fast_tiff_lib::TiffStack::from_bytes(data.to_vec()) else { return };
    let Some(frame) = stack.frames.first() else { return };
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
        let r = catch_unwind(AssertUnwindSafe(|| exercise(&data)));
        assert!(r.is_ok(), "{name} panicked — the Ok-or-Err contract is broken again");
        checked += 1;
    }
    assert!(checked > 0, "no artifacts found in {}", dir.display());
    println!("{checked} known-crash input(s) replayed cleanly");
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
        strip_offsets: vec![8],
        strip_byte_counts: vec![16],
        rows_per_strip: 1,
    };
    assert!(frame.pixel_count().is_ok(), "w*h alone still fits on 64-bit");
    let err = frame.sample_count().expect_err("w*h*spp must overflow");
    assert!(err.to_string().contains("overflows address space"), "got: {err}");
    assert!(frame.decoded_len().is_err(), "and so must the byte length");

    // The reader has to surface that as an error, not trap on the multiply.
    let data = vec![0u8; 64];
    let r = catch_unwind(AssertUnwindSafe(|| {
        fast_tiff_lib::read_frame_u16(&data, &frame, fast_tiff_lib::ByteOrder::Little, None).is_err()
    }));
    assert_eq!(r.ok(), Some(true), "read_frame_u16 must return Err, not panic");
}

//! Fuzz the *decoding* half: everything `tiff_open` does, plus actually pulling
//! pixels through every public reader.
//!
//! This is the target that exercises the size arithmetic — `width * height *
//! samples_per_pixel * bytes_per_sample` from file-declared values — and the
//! per-codec strip paths (LZW / Deflate / PackBits), the predictor undo, and
//! the chunky/planar plane gathers. Same contract as `tiff_open`: `Ok` or
//! `Err`, never a panic or abort.
//!
//! Run with:  cargo +nightly fuzz run tiff_decode -- -rss_limit_mb=2048
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(stack) = fast_tiff_lib::TiffStack::from_bytes(data.to_vec()) else {
        return;
    };
    let Some(frame) = stack.frames.first() else {
        return;
    };
    let order = stack.byte_order;
    let bytes = &stack.data;

    // Every public reader, on frame 0. Each must return Result, never unwind.
    let _ = fast_tiff_lib::read_frame_u16(bytes, frame, order, None);
    let _ = fast_tiff_lib::read_frame_u8(bytes, frame, order);
    let _ = fast_tiff_lib::read_frame_f32(bytes, frame, order);
    let _ = fast_tiff_lib::frame_float_minmax(bytes, frame, order);

    // Plane gathers: exercise the chunky/planar deinterleave, including an
    // out-of-range plane index (readers clamp rather than panic).
    for plane in [0usize, 1, 3, 7] {
        let _ = fast_tiff_lib::read_plane_u16(bytes, frame, order, None, plane);
        let _ = fast_tiff_lib::read_plane_u8(bytes, frame, order, plane);
        let _ = fast_tiff_lib::read_plane_f32(bytes, frame, order, plane);
    }

    // Single-pass all-planes readers (a different assembly path).
    let _ = fast_tiff_lib::read_planes_u16(bytes, frame, order, None);
    let _ = fast_tiff_lib::read_planes_u8(bytes, frame, order);
    let _ = fast_tiff_lib::read_planes_f32(bytes, frame, order);

    // Buffer-reusing variants take a different branch than the allocating ones.
    let mut buf16 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u16_into(bytes, frame, order, None, &mut buf16);
    let mut buf8 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u8_into(bytes, frame, order, &mut buf8);
    let mut buff = Vec::new();
    let _ = fast_tiff_lib::read_frame_f32_into(bytes, frame, order, &mut buff);
});

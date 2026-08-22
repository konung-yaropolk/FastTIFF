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

    // CMYK converting readers. Called unconditionally: the point is that a
    // frame the mutator has turned into a *near*-CMYK one (photometric 5 with
    // an odd sample count, a truncated plate, a bogus InkSet) comes back as an
    // error rather than a panic. They are cheap to call on a non-CMYK frame,
    // since the gate rejects before any decoding happens.
    let _ = fast_tiff_lib::read_planes_rgb_u8(bytes, frame, order);
    let _ = fast_tiff_lib::read_planes_rgb_u16(bytes, frame, order);
    for plane in [0usize, 2, 5] {
        let _ = fast_tiff_lib::read_plane_rgb_u8(bytes, frame, order, plane);
        let _ = fast_tiff_lib::read_plane_rgb_u16(bytes, frame, order, plane);
    }

    // Row bands. `crop_rows` builds a `FrameInfo` the indexer never produced —
    // a spliced strip table over a shrunken height — and then hands it to every
    // reader above. That makes it a way to reach the decoders with geometry the
    // open path would have rejected, so the bands get the same treatment as the
    // frames they came from.
    // Two-axis crops, which on a tiled frame splice a *grid* rather than a run
    // — a different way to build a strip table the open path never saw.
    // The last of each set is backwards on purpose — built rather than written
    // as a literal, which the lints reject as a mistake. Here it is the input a
    // caller produces from an inverted drag, and it must not panic.
    let bad_cols = std::ops::Range { start: 9u32, end: 2 };
    let bad_rows = std::ops::Range { start: 7u32, end: 3 };
    let backwards_band = std::ops::Range { start: 7u32, end: 2 };
    for (cols, rows) in [(0u32..1, 0u32..1), (0..u32::MAX, 0..u32::MAX), (2..9, 3..7), (bad_cols, bad_rows)] {
        if let Ok(region) = frame.crop(cols, rows) {
            let r = &region.frame;
            let _ = fast_tiff_lib::read_planes_u8(bytes, r, order);
            let _ = fast_tiff_lib::read_planes_u16(bytes, r, order, None);
            let _ = fast_tiff_lib::read_frame_f32(bytes, r, order);
        }
    }

    for rows in [0u32..1, 0..u32::MAX, 3..9, u32::MAX - 1..u32::MAX, backwards_band] {
        if let Ok(band) = frame.crop_rows(rows) {
            let b = &band.frame;
            let _ = fast_tiff_lib::read_planes_u8(bytes, b, order);
            let _ = fast_tiff_lib::read_planes_u16(bytes, b, order, None);
            let _ = fast_tiff_lib::read_planes_f32(bytes, b, order);
            let _ = fast_tiff_lib::read_frame_u16(bytes, b, order, None);
            let _ = fast_tiff_lib::read_planes_rgb_u8(bytes, b, order);
            // A band of a band: nothing forbids it, and the second crop works
            // from a table this crate wrote rather than one a file did.
            if let Ok(inner) = b.crop_rows(0..2) {
                let _ = fast_tiff_lib::read_planes_u8(bytes, &inner.frame, order);
            }
        }
    }

    // Buffer-reusing variants take a different branch than the allocating ones.
    let mut buf16 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u16_into(bytes, frame, order, None, &mut buf16);
    let mut buf8 = Vec::new();
    let _ = fast_tiff_lib::read_frame_u8_into(bytes, frame, order, &mut buf8);
    let mut buff = Vec::new();
    let _ = fast_tiff_lib::read_frame_f32_into(bytes, frame, order, &mut buff);
});

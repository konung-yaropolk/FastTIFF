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
    // Starting *at* `u32::MAX`, which is a different case than one below it:
    // `crop_rows` clamps `start` to the frame height, so a file claiming
    // `u32::MAX` rows makes `start` saturate there too and the `start + 1` that
    // guarantees a non-empty band overflows. `u32::MAX - 1` does not reach it.
    let at_the_top = std::ops::Range { start: u32::MAX, end: 0 };
    let top_to_top = std::ops::Range { start: u32::MAX, end: u32::MAX };
    for (cols, rows) in [
        (0u32..1, 0u32..1),
        (0..u32::MAX, 0..u32::MAX),
        (2..9, 3..7),
        (bad_cols, bad_rows),
        (0..u32::MAX, at_the_top.clone()),
        (0..1, top_to_top.clone()),
    ] {
        if let Ok(region) = frame.crop(cols, rows) {
            let r = &region.frame;
            let _ = fast_tiff_lib::read_planes_u8(bytes, r, order);
            let _ = fast_tiff_lib::read_planes_u16(bytes, r, order, None);
            let _ = fast_tiff_lib::read_frame_f32(bytes, r, order);
        }
    }

    for rows in [
        0u32..1,
        0..u32::MAX,
        3..9,
        u32::MAX - 1..u32::MAX,
        backwards_band,
        at_the_top,
        top_to_top,
    ] {
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

    // Sampled bands: the same spliced table, but keeping only every `step`th
    // strip. The pieces it keeps are chosen from the file's own `RowsPerStrip`,
    // so a hostile one decides both how many pieces there are and how tall the
    // last is — and `decoded_row_of` has to stay inside the frame it built
    // whatever they say.
    let backwards_step = std::ops::Range { start: 6u32, end: 1 };
    for rows in [0u32..1, 0..u32::MAX, 3..9, u32::MAX - 1..u32::MAX, backwards_step] {
        for step in [0u32, 1, 2, 7, u32::MAX] {
            let Ok(band) = frame.crop_rows_step(rows.clone(), step) else { continue };
            let b = &band.frame;
            let _ = fast_tiff_lib::read_planes_u8(bytes, b, order);
            let _ = fast_tiff_lib::read_planes_u16(bytes, b, order, None);
            let _ = fast_tiff_lib::read_planes_f32(bytes, b, order);
            let _ = fast_tiff_lib::read_planes_rgb_u8(bytes, b, order);
            // Every row it claims to hold must land inside the frame it built.
            for probe in [0u32, 1, band.first_row, band.first_row.wrapping_add(1), u32::MAX] {
                if let Some(row) = band.decoded_row_of(probe) {
                    assert!(row < b.height, "decoded_row_of({probe}) = {row} of {}", b.height);
                }
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

    // The Frame API over the same frame: every unit, plane and crop, through
    // ONE scratch kept dirty across all of them and into destinations that
    // already hold something, so a decode that read a buffer before writing
    // it, or trusted what a warm scratch held, would show up here.
    frame_reads(&stack);
});

/// Every [`fast_tiff_lib::Frame`] read of frame 0 and of its crops.
fn frame_reads(stack: &fast_tiff_lib::TiffStack) {
    use fast_tiff_lib::units::{CmykRgb16, CmykRgb8, Display, File, Stored};
    use fast_tiff_lib::{ByteOrder, Crop, DecodeScratch, Frame};

    fn all<U: fast_tiff_lib::units::Units>(f: &Frame<'_>, units: U, s: &mut DecodeScratch) {
        for plane in [0usize, 1, 3, 7] {
            let mut out = vec![U::Sample::default(); 3];
            let _ = f.read(units, plane, &mut out, s);
        }
        let Ok(n) = f.plane_len(units) else { return };
        let planes = f.planes(units).min(4);
        let mut bufs = vec![vec![U::Sample::default(); n]; planes];
        let mut slices: Vec<&mut [U::Sample]> = bufs.iter_mut().map(|v| &mut v[..]).collect();
        let _ = f.read_into(units, 0, &mut slices, s);
        if planes > 1 {
            let _ = f.read_into(units, 1, &mut slices[1..], s);
        }
    }
    fn every_unit(f: &Frame<'_>, s: &mut DecodeScratch) {
        all(f, Display { range: None }, s);
        all(
            f,
            Display {
                range: Some((-3.0, 70000.0)),
            },
            s,
        );
        all(f, File, s);
        all(f, Stored(ByteOrder::Little), s);
        all(f, Stored(ByteOrder::Big), s);
        all(f, CmykRgb8, s);
        all(f, CmykRgb16, s);
        let _ = f.value_range(s);
        if let Ok(n) = f.stored_frame_len() {
            let mut out = vec![0xA5u8; n];
            let _ = f.stored_frame_into(ByteOrder::Big, &mut out, s);
        }
        // Borrowed or decoded, the whole frame is the bytes the copying read
        // writes.
        let mut out = vec![0x5Au8; 3];
        if let Ok(px) = f.stored_frame(ByteOrder::Little, &mut out, s) {
            let px = px.to_vec();
            let mut again = vec![0xC3u8; px.len()];
            f.stored_frame_into(ByteOrder::Little, &mut again, s)
                .expect("a frame stored_frame read is one stored_frame_into reads");
            assert_eq!(px, again, "stored_frame against stored_frame_into");
        }
        f.touch();
    }

    let Ok(frame) = stack.frame(0) else { return };
    let mut scratch = DecodeScratch::new();
    every_unit(&frame, &mut scratch);
    let max = u32::MAX;
    let backwards = std::ops::Range {
        start: 7u32,
        end: 2,
    };
    for crop in [
        Crop::Rows(0..1),
        Crop::Rows(3..9),
        Crop::Rows(max - 1..max),
        Crop::Rows(backwards.clone()),
        Crop::Region {
            cols: 2..9,
            rows: 3..7,
        },
        Crop::Region {
            cols: 0..max,
            rows: backwards.clone(),
        },
        Crop::Sampled {
            rows: 0..max,
            step: 2,
        },
        Crop::Sampled {
            rows: backwards,
            step: max,
        },
    ] {
        if let Ok((band, coverage)) = frame.crop(crop) {
            every_unit(&band, &mut scratch);
            for probe in [0u32, 1, coverage.first_row, max] {
                if let Some(row) = coverage.decoded_row_of(probe) {
                    assert!(row < band.info().height, "decoded_row_of({probe}) = {row}");
                }
            }
        }
    }
}


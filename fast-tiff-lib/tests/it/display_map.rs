//! The two decoded views of a sample, and the function between them.
//!
//! The `*_u16` readers decode into display space and the `*_f32` readers into
//! the file's own units; [`DisplayMap`] is meant to be exactly the map from the
//! second to the first. If it were not, every caller that converts a window or
//! a measurement through it would be off by the difference, plausibly and
//! silently. So this checks it sample for sample: on every committed fixture
//! (big-endian, LZW, PackBits, tiled, palette among them), and on files written
//! here for every sample type the writer has, single-sample and RGB, chunky and
//! planar, with and without compression and the predictor.

use crate::common;
use fast_tiff_lib::{
    read_frame_f32, read_frame_u16, read_plane_f32, read_plane_u16_into, read_planes_f32,
    read_planes_u16, Compression, DisplayMap, FrameInfo, SampleType, TiffStack, TiffWriter,
    WriterOptions,
};
use std::path::Path;

fn wide(f: &FrameInfo) -> bool {
    matches!(f.bits_per_sample, 32 | 64)
}

/// Every plane of every frame of `stack`, through both readers.
fn check_stack(stack: &TiffStack, what: &str) {
    let (data, order) = (&stack.data[..], stack.byte_order);
    for (i, f) in stack.frames.iter().enumerate() {
        if !matches!(f.bits_per_sample, 8 | 16 | 32 | 64) {
            continue;
        }
        let what = format!(
            "{what}, frame {i} ({}-bit {:?})",
            f.bits_per_sample, f.sample_format
        );
        let spp = (f.samples_per_pixel as usize).max(1);
        for p in 0..spp {
            let file = read_plane_f32(data, f, order, p).unwrap();
            // A fixed range inside the data as well as the auto one, so the
            // clamp at both ends is exercised for wide samples.
            let mid = file
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0f32, f32::max)
                / 2.0;
            for range in [None, Some((-mid.abs(), mid.abs().max(1.0)))] {
                let mut display = Vec::new();
                let map = read_plane_u16_into(data, f, order, range, p, &mut display).unwrap();
                if range.is_some() || !wide(f) {
                    assert_eq!(
                        Some(map),
                        DisplayMap::of(f, range),
                        "{what}: the map reported"
                    );
                }
                assert_eq!(display.len(), file.len(), "{what}");
                for (k, (&d, &v)) in display.iter().zip(&file).enumerate() {
                    assert_eq!(
                        d,
                        map.display_sample(v as f64),
                        "{what}, plane {p}, sample {k}: file value {v}, range {range:?}"
                    );
                    if !wide(f) {
                        assert_eq!(
                            map.to_file(d as f64),
                            v as f64,
                            "{what}, sample {k}: back to the file"
                        );
                        assert_eq!(
                            map.to_display(v as f64),
                            d as f64,
                            "{what}, sample {k}: exact"
                        );
                    }
                }
            }
        }
        if !wide(f) {
            let map = DisplayMap::of(f, None).unwrap();
            // The all-planes readers (the fused predictor path among them).
            let all16 = read_planes_u16(data, f, order, None).unwrap();
            let all32 = read_planes_f32(data, f, order).unwrap();
            for (p, (a, b)) in all16.iter().zip(&all32).enumerate() {
                let want: Vec<u16> = b.iter().map(|&v| map.display_sample(v as f64)).collect();
                assert_eq!(a, &want, "{what}: plane {p} of the all-planes readers");
            }
            // And the frame readers, whose fast paths skip the plane code.
            if spp == 1 {
                let d = read_frame_u16(data, f, order, None).unwrap();
                let v = read_frame_f32(data, f, order).unwrap();
                let want: Vec<u16> = v.iter().map(|&v| map.display_sample(v as f64)).collect();
                assert_eq!(&d[..], &want[..], "{what}: the frame readers");
            }
        }
    }
}

#[test]
fn the_map_is_the_function_between_the_two_readers_on_every_fixture() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("tif") {
            continue;
        }
        let stack = common::open_tiff(&path).unwrap();
        check_stack(&stack, &path.file_name().unwrap().to_string_lossy());
        checked += 1;
    }
    assert!(
        checked > 20,
        "only {checked} fixtures found in {}",
        dir.display()
    );
}

/// `n` values of type `ty` that reach both ends of its range, zero and its
/// neighbours, and (for floats) the values a float can hold that an integer
/// cannot, as little-endian bytes.
fn samples(ty: SampleType, n: usize) -> Vec<u8> {
    let ints = |lo: i128, hi: i128| {
        let special = [lo, hi, 0, 1, -1, lo + 1, hi - 1, 127, 128, 255, 256];
        (0..n).map(move |k| {
            let v = if k < special.len() {
                special[k]
            } else {
                lo + (k as i128 * 7919 * (hi - lo) / n as i128 / 13) % (hi - lo)
            };
            v.clamp(lo, hi)
        })
    };
    let floats = (0..n).map(|k| match k {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => -1.5e30,
        4 => 0.0,
        5 => -0.0,
        _ => (k as f64 - n as f64 / 3.0) * 12.375,
    });
    let mut out = Vec::new();
    match ty {
        SampleType::U8 => ints(0, 255).for_each(|v| out.push(v as u8)),
        SampleType::I8 => ints(-128, 127).for_each(|v| out.push(v as i8 as u8)),
        SampleType::U16 => ints(0, 65535).for_each(|v| out.extend((v as u16).to_le_bytes())),
        SampleType::I16 => ints(-32768, 32767).for_each(|v| out.extend((v as i16).to_le_bytes())),
        SampleType::U32 => {
            ints(0, u32::MAX as i128).for_each(|v| out.extend((v as u32).to_le_bytes()))
        }
        SampleType::I32 => ints(i32::MIN as i128, i32::MAX as i128)
            .for_each(|v| out.extend((v as i32).to_le_bytes())),
        SampleType::U64 => {
            ints(0, u64::MAX as i128).for_each(|v| out.extend((v as u64).to_le_bytes()))
        }
        SampleType::I64 => ints(i64::MIN as i128, i64::MAX as i128)
            .for_each(|v| out.extend((v as i64).to_le_bytes())),
        SampleType::F32 => floats.for_each(|v| out.extend((v as f32).to_le_bytes())),
        SampleType::F64 => floats.for_each(|v| out.extend(v.to_le_bytes())),
    }
    out
}

#[test]
fn the_map_is_the_function_between_the_two_readers_for_every_sample_type() {
    // The writer writes host byte order; the fixtures above carry big-endian.
    if cfg!(target_endian = "big") {
        return;
    }
    let (w, h) = (13u32, 5u32);
    let (mut checked, mut skipped) = (0, Vec::new());
    use SampleType::*;
    for ty in [U8, I8, U16, I16, U32, I32, U64, I64, F32, F64] {
        for (spp, planar) in [(1u16, false), (3, false), (3, true)] {
            for (compression, predictor) in [
                (Compression::None, false),
                (Compression::Deflate, false),
                (Compression::Deflate, true),
            ] {
                let what = format!(
                    "{ty:?}, {spp} sample(s){}, {compression:?}{}",
                    if planar { " planar" } else { "" },
                    if predictor { " + predictor" } else { "" }
                );
                let options = WriterOptions::new(w, h, ty)
                    .samples_per_pixel(spp)
                    .planar(planar)
                    .compression(compression)
                    .predictor(predictor);
                let mut writer = match TiffWriter::new(std::io::Cursor::new(Vec::new()), options) {
                    Ok(writer) => writer,
                    // A combination the writer does not offer (a predictor for
                    // 64-bit samples, say) is not this test's concern.
                    Err(e) => {
                        skipped.push(format!("{what}: {e}"));
                        continue;
                    }
                };
                let n = (w * h) as usize * spp as usize;
                let bytes = samples(ty, n);
                if let Err(e) = writer.write_frame_bytes(&bytes) {
                    skipped.push(format!("{what}: {e}"));
                    continue;
                }
                let file = writer.finish().unwrap().into_inner();
                let stack = TiffStack::from_bytes(file).unwrap();
                check_stack(&stack, &what);
                checked += 1;
            }
        }
    }
    assert!(checked >= 80, "{checked} checked; skipped: {skipped:#?}");
    assert!(skipped.len() <= 10, "skipped: {skipped:#?}");
}

//! The window conversions, checked against the decoder they describe.
//!
//! A conversion that is merely self-consistent proves little: a window and a
//! texture can agree with each other and both be off from the pixels by the
//! same factor. So besides the round trips, these decode real frames of every
//! sample type through the reader each channel kind uses and check that
//! [`SampleUnits`] reads a texture sample back as the value `read_plane_f32`
//! gives for the same pixel.

use super::*;
use crate::document::Document;
use fast_tiff_lib::{
    read_frame_f32, read_frame_u16, read_frame_u8, SampleType, StackMetaWrite, TiffStack,
    TiffWriter, WriterOptions,
};
use std::io::Cursor;

/// Every kind of channel there is, with what it is called in a failure.
fn every_unit() -> Vec<(&'static str, SampleUnits)> {
    vec![
        ("raw 8-bit", SampleUnits::RAW8),
        ("float", SampleUnits::float(Window::new(0.0, 1.0))),
        (
            "float, wide span",
            SampleUnits::float(Window::new(-3.0, 70_000.0)),
        ),
        ("16-bit", SampleUnits::display(DisplayMap::Identity)),
        ("widened 8-bit", SampleUnits::display(DisplayMap::Widen8)),
        (
            "signed 8-bit",
            SampleUnits::display(DisplayMap::SignedWiden8),
        ),
        ("signed 16-bit", SampleUnits::display(DisplayMap::Signed16)),
        (
            "rescaled wide",
            SampleUnits::display(DisplayMap::Linear {
                lo: -1000.0,
                hi: 250_000.0,
            }),
        ),
    ]
}

/// Whether `v` survives a trip through a half float without becoming infinite.
fn half_float_holds(v: f32) -> bool {
    v.is_finite() && v.abs() <= 65504.0
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn a_window_survives_the_trip_to_the_texture_and_back() {
    for (what, units) in every_unit() {
        let full = units.full_range();
        // Inside the range, at its ends, and past them: a window may lie
        // beyond the data, and its conversion must not clamp.
        for w in [
            full,
            Window::new(full.lo + full.width() * 0.25, full.hi - full.width() * 0.4),
            Window::new(full.lo - 10.0, full.hi + 10.0),
        ] {
            let (lo, hi) = units.to_texture(w);
            let back = Window::new(units.texture_to_file(lo), units.texture_to_file(hi));
            assert!(
                close(back.lo, w.lo) && close(back.hi, w.hi),
                "{what}: {w:?} came back as {back:?}"
            );
        }
    }
}

#[test]
fn the_whole_range_fills_the_texture() {
    let u16_max = f32::from(u16::MAX);
    let cases = [
        (SampleUnits::RAW8, Window::new(0.0, 255.0), (0.0, 255.0)),
        (
            SampleUnits::display(DisplayMap::Identity),
            Window::new(0.0, u16_max),
            (0.0, u16_max),
        ),
        (
            SampleUnits::display(DisplayMap::Widen8),
            Window::new(0.0, 255.0),
            (0.0, u16_max),
        ),
        (
            SampleUnits::display(DisplayMap::SignedWiden8),
            Window::new(-128.0, 127.0),
            (0.0, u16_max),
        ),
        (
            SampleUnits::display(DisplayMap::Signed16),
            Window::new(f32::from(i16::MIN), f32::from(i16::MAX)),
            (0.0, u16_max),
        ),
    ];
    for (units, full, texture) in cases {
        assert_eq!(units.full_range(), full, "{units:?}");
        assert_eq!(units.to_texture(full), texture, "{units:?}");
    }
    // A float texture has no whole range; it opens on the unit one.
    assert_eq!(
        SampleUnits::float(Window::new(-3.0, 70_000.0)).full_range(),
        Window::new(0.0, 1.0)
    );
}

/// The 3D view's volumes are normalized; the window has to be on the scale the
/// voxels were normalized to. For an integer channel "the whole range is
/// 0..1"; for a float one its span is.
#[test]
fn the_volume_window_is_normalized_as_the_voxels_are() {
    for (what, units) in every_unit() {
        let (lo, hi) = if units.kind() == ChannelKind::Float {
            units.to_unorm(match what {
                "float" => Window::new(0.0, 1.0),
                _ => Window::new(-3.0, 70_000.0),
            })
        } else {
            units.to_unorm(units.full_range())
        };
        assert!(close(lo, 0.0) && close(hi, 1.0), "{what}: {lo}..{hi}");
        // The window and the voxels go through one map.
        let (scale, offset) = units.volume_affine();
        let w = Window::new(7.0, 11.0);
        let (t_lo, t_hi) = units.to_texture(w);
        assert_eq!(
            units.to_unorm(w),
            (t_lo * scale + offset, t_hi * scale + offset),
            "{what}"
        );
    }
    // And the divisor is the one the volume builder encodes with.
    let (lo, hi) = SampleUnits::RAW8.to_unorm(Window::new(51.0, 102.0));
    assert!(close(lo, 51.0 / UNORM8_MAX) && close(hi, 102.0 / UNORM8_MAX));
    let (lo, hi) = SampleUnits::display(DisplayMap::Identity).to_unorm(Window::new(100.0, 200.0));
    assert!(close(lo, 100.0 / UNORM16_MAX) && close(hi, 200.0 / UNORM16_MAX));
    // A float window is normalized over the span, so a value no half float
    // can hold raw (past 65504) still lands inside 0..1.
    let (lo, hi) =
        SampleUnits::float(Window::new(-3.0, 70_000.0)).to_unorm(Window::new(0.0, 66_000.0));
    assert!(
        close(lo, 3.0 / 70_003.0) && close(hi, 66_003.0 / 70_003.0),
        "{lo}..{hi}"
    );
    assert!(half_float_holds(hi), "{hi}");
}

/// ImageJ keeps a signed 16-bit image offset into unsigned and states its
/// display range in that space; every other type's range is in the samples'
/// own values. Both directions use the one definition, so a range read from a
/// file is written back unchanged.
#[test]
fn imagej_ranges_are_offset_for_signed_16_bit_only() {
    let signed16 = SampleUnits::display(DisplayMap::Signed16);
    assert_eq!(
        signed16.from_imagej((32_668.0, 33_268.0)),
        Window::new(-100.0, 500.0)
    );
    assert_eq!(
        signed16.to_imagej(Window::new(-100.0, 500.0)),
        (32_668.0, 33_268.0)
    );

    for (what, units) in every_unit() {
        let declared = (-12.0, 200.0);
        let window = units.from_imagej(declared);
        assert_eq!(units.to_imagej(window), declared, "{what}: round trip");
        if units != signed16 {
            assert_eq!(window, Window::new(-12.0, 200.0), "{what}: no offset");
        }
    }
}

#[test]
fn each_unit_names_its_texture_kind() {
    assert_eq!(SampleUnits::RAW8.kind(), ChannelKind::Int8);
    assert_eq!(
        SampleUnits::float(Window::new(0.0, 1.0)).kind(),
        ChannelKind::Float
    );
    assert_eq!(
        SampleUnits::display(DisplayMap::Signed16).kind(),
        ChannelKind::Int16
    );
    for kind in [ChannelKind::Int8, ChannelKind::Int16, ChannelKind::Float] {
        assert_eq!(SampleUnits::of(kind, None, None).kind(), kind);
    }
    // With nothing to say what the 16-bit scale is of, it is its own unit.
    assert_eq!(
        SampleUnits::of(ChannelKind::Int16, None, None).display_map(),
        Some(DisplayMap::Identity)
    );
}

/// A TIFF of one plane of `sample` holding `values`.
fn one_plane_bytes(sample: SampleType, values: &[f64]) -> Vec<u8> {
    let n = values.len() as u32;
    let opts = WriterOptions::new(n, 1, sample).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| -> Vec<u8> {
            match sample {
                SampleType::U8 => vec![v as u8],
                SampleType::I8 => vec![v as i8 as u8],
                SampleType::U16 => (v as u16).to_le_bytes().to_vec(),
                SampleType::I16 => (v as i16).to_le_bytes().to_vec(),
                SampleType::U32 => (v as u32).to_le_bytes().to_vec(),
                SampleType::I32 => (v as i32).to_le_bytes().to_vec(),
                SampleType::F32 => (v as f32).to_le_bytes().to_vec(),
                other => unreachable!("{other:?}"),
            }
        })
        .collect();
    w.write_frame_bytes(&bytes).unwrap();
    w.finish().unwrap().into_inner()
}

/// What an opened stack's first channel decodes to on its own texture, read
/// back through its units, against what the file says each pixel is.
fn check_against_the_decoder(sample: SampleType, values: &[f64]) {
    let bytes = one_plane_bytes(sample, values);
    let tiff = TiffStack::from_bytes(bytes.clone()).unwrap();
    let file = read_frame_f32(&tiff.data, &tiff.frames[0], tiff.byte_order).unwrap();
    let stack = Document::from_bytes(bytes, "units.tif".into(), false).unwrap();
    let s = stack.display.settings()[0];
    let (data, order, frame) = (&tiff.data, tiff.byte_order, &tiff.frames[0]);
    let texture: Vec<f32> = match s.kind() {
        ChannelKind::Int8 => read_frame_u8(data, frame, order)
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        ChannelKind::Int16 => read_frame_u16(data, frame, order, None)
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect(),
        ChannelKind::Float => read_frame_f32(data, frame, order).unwrap().to_vec(),
    };
    // A rescaled texture is quantized to steps of its range over the 16-bit
    // scale; every other one holds the file's values exactly.
    let step = match s.units.display_map() {
        Some(DisplayMap::Linear { lo, hi }) => (hi - lo) / f32::from(u16::MAX),
        _ => 0.0,
    };
    for (i, (&t, &f)) in texture.iter().zip(file.iter()).enumerate() {
        let back = s.units.texture_to_file(t);
        assert!(
            close(back, f) || (back - f).abs() <= step,
            "{sample:?} pixel {i}: texture {t} reads back as {back}, the file says {f}"
        );
    }
    // And the window the channel opened with is the data's range, in the
    // file's units.
    let (lo, hi) = file
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(
        close(s.window.lo, lo) && close(s.window.hi, hi),
        "{sample:?}: opened on {:?}, the data spans {lo}..{hi}",
        s.window
    );
}

#[test]
fn every_depth_reads_its_texture_back_as_the_files_values() {
    check_against_the_decoder(SampleType::U8, &[0.0, 1.0, 17.0, 128.0, 254.0, 255.0]);
    check_against_the_decoder(SampleType::I8, &[-128.0, -1.0, 0.0, 1.0, 100.0, 127.0]);
    check_against_the_decoder(SampleType::U16, &[0.0, 1.0, 300.0, 40_000.0, 65_535.0]);
    check_against_the_decoder(
        SampleType::I16,
        &[-32_768.0, -1.0, 0.0, 1.0, 12_345.0, 32_767.0],
    );
    check_against_the_decoder(SampleType::F32, &[-2.5, 0.0, 0.125, 1e6]);
    // A wide integer plane is rescaled through its own range, and the channel
    // records that range: its ends read back exactly, the rest to within a
    // step of the 16-bit scale.
    check_against_the_decoder(SampleType::I32, &[-70_000.0, 0.0, 65_535.0, 1_000_000.0]);
    check_against_the_decoder(SampleType::U32, &[5.0, 70_000.0, 4_000_000.0]);
}

/// A 32-bit integer channel's window reaches the shader exactly as it did when
/// the window was held on the rescaled 16-bit scale: the whole of the data is
/// the whole of the texture.
#[test]
fn a_wide_integer_window_is_the_rescaled_range() {
    let bytes = one_plane_bytes(SampleType::I32, &[-70_000.0, 0.0, 1_000_000.0]);
    let stack = Document::from_bytes(bytes, "wide.tif".into(), false).unwrap();
    let s = stack.display.settings()[0];
    assert_eq!(s.kind(), ChannelKind::Int16);
    assert_eq!(
        s.units.display_map(),
        Some(DisplayMap::Linear {
            lo: -70_000.0,
            hi: 1_000_000.0
        })
    );
    let (lo, hi) = s.units.to_texture(s.window);
    assert!(
        close(lo, 0.0) && close(hi, f32::from(u16::MAX)),
        "{lo}..{hi}"
    );
}

/// An 8-bit file's declared range is the window it opens with, in its own
/// units, and the range a save declares again.
#[test]
fn an_8_bit_files_declared_range_opens_and_saves_unchanged() {
    let opts = WriterOptions::new(4, 1, SampleType::U8)
        .metadata(StackMetaWrite::new(1, 1).range(12.0, 99.0));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_u8(&[0, 50, 100, 255]).unwrap();
    let stack =
        Document::from_bytes(w.finish().unwrap().into_inner(), "8.tif".into(), false).unwrap();
    let s = stack.display.settings()[0];
    assert_eq!(s.units, SampleUnits::RAW8);
    assert_eq!(s.window, Window::new(12.0, 99.0));
    assert_eq!(s.initial, s.window);
    assert_eq!(s.units.to_texture(s.window), (12.0, 99.0));
    assert_eq!(s.units.to_imagej(s.window), (12.0, 99.0));
}

#[test]
fn window_helpers() {
    let w = Window::new(10.0, 20.0);
    assert_eq!(w.width(), 10.0);
    assert!(w.contains(10.0) && w.contains(20.0) && !w.contains(20.5));
    assert_eq!(
        Window::new(-5.0, 30.0).clamp_to(w),
        Window::new(10.0, 20.0),
        "each end clamped"
    );
    assert_eq!(
        Window::new(25.0, 30.0).clamp_to(w),
        Window::new(20.0, 20.0),
        "past one end, both land on it"
    );
    assert_eq!(w.union(Window::new(15.0, 40.0)), Window::new(10.0, 40.0));
    assert_eq!(Window::from((1.0, 2.0)), Window::new(1.0, 2.0));
    assert_eq!(<(f32, f32)>::from(w), (10.0, 20.0));
}

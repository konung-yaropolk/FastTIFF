//! Running a plugin end to end against a real stack.
//!
//! The registry's own bookkeeping is unit-tested next to it; this is the part
//! that matters — that a plugin reaches the right pixels through
//! [`StackHost`] and hands back a result the host can use. When the `.dll` lane
//! lands these same assertions become its oracle: the identical filter, run
//! through the C boundary, must produce byte-identical output.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::plugins::{builtin, describe_view, StackHost};
use fast_tiff_viewer::Stack;
use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamValue, Params, PixelType, Plane, PlaneData, Plugin,
    PluginError, VolumeMode, VolumeView,
};
use std::io::Cursor;

const W: u32 = 8;
const H: u32 = 4;

/// A stack with `channels x slices x frames` planes whose every pixel encodes
/// which plane it came from, so a mis-addressed read is visible rather than
/// merely wrong-looking.
///
/// To keep a real Z axis, every one of the three counts must exceed 1:
/// `resolve_dimensions` folds Z into time unconditionally otherwise, because a
/// single-timepoint z-stack is indistinguishable from a movie and the movie
/// reading is right far more often. The assertion below holds callers to that.
fn stack(channels: usize, slices: usize, frames: usize) -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::F32)
        // Frames are derived from the plane count when writing, so the
        // metadata states only channels and slices.
        .metadata(StackMetaWrite::new(channels, slices));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    // xyczt order: channel fastest, then z, then t — the order the reader uses.
    for t in 0..frames.max(1) {
        for z in 0..slices.max(1) {
            for c in 0..channels.max(1) {
                let tag = plane_tag(c, z, t);
                let px: Vec<f32> = (0..W * H).map(|i| tag + i as f32 / 1000.0).collect();
                let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_frame_bytes(&bytes).unwrap();
            }
        }
    }
    let bytes = w.finish().unwrap().into_inner();
    let s = Stack::from_bytes(bytes, "probe.tif".into(), false).expect("stack should open");
    // `resolve_dimensions` reclassifies a mislabelled axis on purpose, and a
    // single-timepoint z-stack reads as a movie. Assert the shape here so a
    // test never silently exercises a different one than its name claims.
    assert_eq!(
        (
            s.display.dims.channels,
            s.display.dims.slices,
            s.display.dims.frames
        ),
        (channels.max(1), slices.max(1), frames.max(1)),
        "the stack did not resolve to the requested shape"
    );
    s
}

/// A value unique to each (c, z, t).
fn plane_tag(c: usize, z: usize, t: usize) -> f32 {
    (c * 100 + z * 10 + t) as f32
}

fn view() -> VolumeView {
    VolumeView {
        mode: VolumeMode::Mip,
        density: 1.0,
        iso: 0.5,
        eye: [0.0; 3],
        forward: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
    }
}

fn host(s: &Stack, frame_index: usize) -> StackHost<'_> {
    StackHost::new(s, describe_view(s, frame_index, false, view()))
}

fn planes_f32(r: &ImageResult) -> Vec<&Vec<f32>> {
    r.planes
        .iter()
        .map(|p| match p {
            PlaneData::F32(v) => v,
            other => panic!("expected f32 planes, got {:?}", other.pixel_type()),
        })
        .collect()
}

/// The whole point of file-plane addressing: every (c, z, t) must reach its own
/// plane, including the channels past the renderer's six-slot display cap.
#[test]
fn every_plane_is_reachable_and_distinct() {
    let s = stack(3, 4, 2);
    let mut h = host(&s, 0);
    let info = h.image();
    assert_eq!((info.channels, info.slices, info.frames), (3, 4, 2));

    let mut buf = Vec::new();
    for t in 0..info.frames {
        for z in 0..info.slices {
            for c in 0..info.channels {
                h.read_plane_f32(Plane::new(c, z, t), &mut buf).unwrap();
                assert_eq!(buf.len(), info.plane_len(), "short read at c{c} z{z} t{t}");
                assert_eq!(
                    buf[0],
                    plane_tag(c, z, t),
                    "c{c} z{z} t{t} decoded some other plane"
                );
            }
        }
    }
}

/// A plugin asking for a plane that does not exist gets a stated error, not a
/// panic and not silently plane zero.
#[test]
fn an_out_of_range_plane_is_an_error() {
    let s = stack(2, 2, 2);
    let mut h = host(&s, 0);
    let mut buf = Vec::new();
    for bad in [
        Plane::new(9, 0, 0),
        Plane::new(0, 9, 0),
        Plane::new(0, 0, 9),
    ] {
        match h.read_plane_f32(bad, &mut buf) {
            Err(PluginError::OutOfRange(_)) => {}
            other => panic!("{bad:?} should be out of range, got {other:?}"),
        }
    }
}

#[test]
fn invert_reflects_the_plane_about_its_own_range() {
    let s = stack(1, 1, 1);
    let mut h = host(&s, 0);
    let mut original = Vec::new();
    h.read_plane_f32(Plane::new(0, 0, 0), &mut original)
        .unwrap();

    let mut p = builtin::Invert;
    let decls = p.params(&h);
    let out = p.run(&mut h, &Params::defaults(&decls)).unwrap();

    let Outcome::NewDocument(img) = out else {
        panic!("Invert should open a document")
    };
    img.validate()
        .expect("a plugin result must describe itself correctly");
    assert_eq!((img.width, img.height), (W, H));
    assert_eq!(img.pixel_type, PixelType::F32);

    let (lo, hi) = original
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    let got = planes_f32(&img);
    for (i, (&o, &g)) in original.iter().zip(got[0].iter()).enumerate() {
        assert!((g - (hi - (o - lo))).abs() < 1e-4, "pixel {i}: {o} -> {g}");
    }
    // Inverting twice is the identity, which is the property rather than the
    // arithmetic.
    let re: Vec<f32> = got[0].iter().map(|&v| hi - (v - lo)).collect();
    for (&o, &r) in original.iter().zip(re.iter()) {
        assert!((o - r).abs() < 1e-3);
    }
}

/// A stack with one Z slice has nothing to project; saying so is different from
/// failing, and the host shows it differently.
#[test]
fn z_project_refuses_a_stack_with_one_slice() {
    let s = stack(1, 1, 1);
    let mut h = host(&s, 0);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);
    match p.run(&mut h, &Params::defaults(&decls)) {
        Err(PluginError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn z_project_computes_each_statistic_over_the_chosen_slices() {
    let s = stack(2, 4, 2);
    // Timepoint 1, so a plugin that ignored frame_index would be caught.
    let mut h = host(&s, 1);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);

    for (method, name) in [(0usize, "max"), (1, "mean"), (2, "min"), (3, "sum")] {
        let mut params = Params::defaults(&decls);
        params.set("method", ParamValue::Choice(method));
        let Outcome::NewDocument(img) = p.run(&mut h, &params.clamp_to(&decls)).unwrap() else {
            panic!("{name}: expected a document")
        };
        img.validate().unwrap();
        assert_eq!(img.channels, 2, "{name}: all channels by default");
        assert_eq!((img.slices, img.frames), (1, 1), "{name}: Z is flattened");

        let got = planes_f32(&img);
        for (c, plane) in got.iter().enumerate() {
            // The reference, straight from the tags: slices 0..=3 at t = 1.
            let vals: Vec<f32> = (0..4).map(|z| plane_tag(c, z, 1)).collect();
            let want = match method {
                0 => vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                1 => vals.iter().sum::<f32>() / vals.len() as f32,
                2 => vals.iter().copied().fold(f32::INFINITY, f32::min),
                _ => vals.iter().sum::<f32>(),
            };
            // Pixel 0 of each plane carries the tag exactly.
            assert!(
                (plane[0] - want).abs() < 1e-3,
                "{name} c{c}: got {} want {want}",
                plane[0]
            );
        }
    }
}

/// The slice range is 1-based in the dialog, as ImageJ's is, and inclusive.
#[test]
fn z_project_honours_the_slice_range() {
    let s = stack(2, 4, 2);
    let mut h = host(&s, 0);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);

    let mut params = Params::defaults(&decls);
    params.set("method", ParamValue::Choice(3)); // Sum
    params.set("first", ParamValue::Int(2));
    params.set("last", ParamValue::Int(3));
    let Outcome::NewDocument(img) = p.run(&mut h, &params.clamp_to(&decls)).unwrap() else {
        panic!("expected a document")
    };
    // 1-based 2..=3 is z = 1 and z = 2.
    let want = plane_tag(0, 1, 0) + plane_tag(0, 2, 0);
    assert!((planes_f32(&img)[0][0] - want).abs() < 1e-3);
}

/// A reversed range is the user's slip, not a reason to fail or to read out of
/// bounds.
#[test]
fn z_project_tolerates_a_reversed_range() {
    let s = stack(2, 4, 2);
    let mut h = host(&s, 0);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);
    let mut params = Params::defaults(&decls);
    params.set("method", ParamValue::Choice(3));
    params.set("first", ParamValue::Int(3));
    params.set("last", ParamValue::Int(2));
    let Outcome::NewDocument(img) = p.run(&mut h, &params.clamp_to(&decls)).unwrap() else {
        panic!("expected a document")
    };
    let want = plane_tag(0, 1, 0) + plane_tag(0, 2, 0);
    assert!((planes_f32(&img)[0][0] - want).abs() < 1e-3);
}

/// Cancelling stops the run and yields nothing to apply.
#[test]
fn a_cancelled_run_returns_cancelled() {
    use std::sync::atomic::{AtomicBool, AtomicU32};
    use std::sync::Arc;

    let s = stack(2, 4, 2);
    let flag = Arc::new(AtomicBool::new(true)); // already cancelled
    let progress = Arc::new(AtomicU32::new(0));
    let mut h = StackHost::new(&s, describe_view(&s, 0, false, view())).with_cancel(flag, progress);

    let mut p = builtin::ZProject;
    let decls = p.params(&h);
    assert_eq!(
        p.run(&mut h, &Params::defaults(&decls)).unwrap(),
        Outcome::Cancelled
    );
}

/// A result whose planes do not match its declared shape must be caught before
/// the host tries to build a document out of it.
#[test]
fn a_malformed_result_is_rejected() {
    let base = ImageResult {
        width: 2,
        height: 2,
        channels: 1,
        slices: 1,
        frames: 1,
        pixel_type: PixelType::F32,
        planes: vec![PlaneData::F32(vec![0.0; 4])],
        name: "ok".into(),
    };
    base.validate().expect("the well-formed case must pass");

    let mut short = base.clone();
    short.planes = vec![PlaneData::F32(vec![0.0; 3])];
    assert!(short.validate().is_err(), "a short plane must be caught");

    let mut miscounted = base.clone();
    miscounted.slices = 4;
    assert!(
        miscounted.validate().is_err(),
        "too few planes must be caught"
    );

    let mut mistyped = base.clone();
    mistyped.planes = vec![PlaneData::U16(vec![0; 4])];
    assert!(
        mistyped.validate().is_err(),
        "a plane of the wrong type must be caught"
    );

    let mut empty = base.clone();
    empty.width = 0;
    assert!(empty.validate().is_err(), "a zero dimension must be caught");
}

/// Every bit depth must reach a plugin through `read_plane_f32`, in the file's
/// own units.
///
/// The whole existing suite used f32 stacks, so it never noticed that
/// `read_plane_f32_into` accepts 32- and 64-bit samples only — running Invert
/// on an imported 8-bit PPM in the actual application is what surfaced it.
/// An 8-bit sample must arrive as 0..255, *not* widened to 16-bit the way the
/// display path widens it: a plugin computing a mean has to get the number that
/// is in the file.
#[test]
fn every_bit_depth_reaches_a_plugin_as_f32() {
    use fast_tiff_lib::SampleType;

    for (ty, raw, want) in [
        (
            SampleType::U8,
            vec![0u8, 1, 128, 255],
            vec![0.0f32, 1.0, 128.0, 255.0],
        ),
        (
            SampleType::U16,
            vec![0u8, 0, 1, 0, 0, 1, 255, 255],
            vec![0.0f32, 1.0, 256.0, 65535.0],
        ),
    ] {
        let opts = WriterOptions::new(2, 2, ty).metadata(StackMetaWrite::new(1, 1));
        let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        w.write_frame_bytes(&raw).unwrap();
        let bytes = w.finish().unwrap().into_inner();
        let s = Stack::from_bytes(bytes, "depth.tif".into(), false).unwrap();

        let mut h = host(&s, 0);
        let mut got = Vec::new();
        h.read_plane_f32(Plane::new(0, 0, 0), &mut got)
            .unwrap_or_else(|e| panic!("{ty:?}: {e}"));
        assert_eq!(got, want, "{ty:?} did not arrive in the file's own units");
    }
}

/// And the plugins themselves must run on those depths, not only on float.
#[test]
fn invert_runs_on_an_8_bit_stack() {
    let opts =
        WriterOptions::new(2, 2, fast_tiff_lib::SampleType::U8).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_bytes(&[0u8, 85, 170, 255]).unwrap();
    let bytes = w.finish().unwrap().into_inner();
    let s = Stack::from_bytes(bytes, "eight.tif".into(), false).unwrap();

    let mut h = host(&s, 0);
    let mut p = builtin::Invert;
    let decls = p.params(&h);
    let Outcome::NewDocument(img) = p.run(&mut h, &Params::defaults(&decls)).unwrap() else {
        panic!("Invert should open a document")
    };
    img.validate().unwrap();
    // Inverted about its own 0..255 range.
    assert_eq!(planes_f32(&img)[0], &vec![255.0, 170.0, 85.0, 0.0]);
}

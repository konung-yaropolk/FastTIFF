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
    HostContext, ImageResult, Outcome, ParamKind, ParamValue, Params, PixelType, Plane, PlaneData,
    Plugin, PluginError, VolumeMode, VolumeView,
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

fn host(s: &Stack, frame_index: usize) -> StackHost {
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

/// The axis selector offers only the axes the stack actually has.
///
/// An axis one plane deep has nothing to project — flattening it is a copy —
/// and offering it would be offering a mistake. When only one is left the
/// dialog draws the selector inactive rather than hiding it, so the projection
/// still says which axis it is about to work on.
#[test]
fn z_project_offers_the_axes_the_stack_has() {
    let axis_options = |s: &Stack| -> Vec<String> {
        let h = host(s, 0);
        let decls = builtin::ZProject.params(&h);
        let d = decls
            .iter()
            .find(|d| d.key == "axis")
            .expect("an axis selector");
        match &d.kind {
            ParamKind::Choice { options, .. } => options.clone(),
            other => panic!("expected a choice, got {other:?}"),
        }
    };

    // There is no "Z only" case to test: `resolve_dimensions` folds a
    // single-timepoint z-stack into a movie, so a stack the viewer presents
    // with slices > 1 always has frames > 1 as well. The code offers Z alone if
    // it ever sees one; nothing here can build one to show it.
    //
    // A timelapse of one slice: nothing to project along Z.
    assert_eq!(axis_options(&stack(2, 1, 3)), vec!["T (frames)"]);
    // Both, Z first — the axis this plugin has always projected.
    assert_eq!(
        axis_options(&stack(2, 4, 3)),
        vec!["Z (slices)", "T (frames)"]
    );
    // Neither: the selector still shows what it would have done, and the run
    // refuses with a reason rather than the dialog being empty.
    assert_eq!(axis_options(&stack(1, 1, 1)), vec!["Z (slices)"]);
}

/// Projecting T averages across time, not across slices — and it does it at
/// every slice, because the view says which timepoint is on screen but not
/// which slice, so there is no current one to pick and dropping the rest would
/// be a choice made on the user's behalf.
#[test]
fn z_project_along_t_flattens_time_at_every_slice() {
    let s = stack(2, 2, 3);
    let mut h = host(&s, 0);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);
    let mut params = Params::defaults(&decls);
    // Both axes are on offer here, so index 1 is T.
    params.set("axis", ParamValue::Choice(1));
    params.set("method", ParamValue::Choice(3)); // Sum

    let Outcome::NewDocument(img) = p.run(&mut h, &params.clamp_to(&decls)).unwrap() else {
        panic!("expected a document")
    };
    // One plane per channel per slice, time flattened away.
    assert_eq!((img.channels, img.slices, img.frames), (2, 2, 1));
    let planes = planes_f32(&img);
    assert_eq!(planes.len(), 4);
    // Channel fastest, then Z — the plane order the contract asks for, and the
    // one thing a nested loop gets backwards without anything looking wrong.
    for z in 0..2 {
        for c in 0..2 {
            let want: f32 = (0..3).map(|t| plane_tag(c, z, t)).sum();
            let got = planes[z * 2 + c][0];
            assert!(
                (got - want).abs() < 1e-3,
                "c{c} z{z} summed the wrong planes: {got} vs {want}"
            );
        }
    }
}

/// The declared range has to span the longer axis, because the declarations are
/// made before the dialog is answered. Choosing the shorter one then has to
/// clamp, or a Z projection would ask for slices a timelapse's frame count made
/// look available.
#[test]
fn z_project_clamps_the_range_to_the_axis_that_was_chosen() {
    let s = stack(2, 2, 5);
    let mut h = host(&s, 0);
    let mut p = builtin::ZProject;
    let decls = p.params(&h);
    let mut params = Params::defaults(&decls);
    params.set("axis", ParamValue::Choice(0)); // Z, which is 2 deep
    params.set("method", ParamValue::Choice(3)); // Sum
    params.set("first", ParamValue::Int(1));
    params.set("last", ParamValue::Int(5)); // past the end of Z

    let Outcome::NewDocument(img) = p.run(&mut h, &params.clamp_to(&decls)).unwrap() else {
        panic!("expected a document")
    };
    // Slices 1..=2 of the timepoint on screen, not five of anything.
    let want: f32 = (0..2).map(|z| plane_tag(0, z, 0)).sum();
    assert!((planes_f32(&img)[0][0] - want).abs() < 1e-3);
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
        channel_colors: Vec::new(),
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

/// The borrow shortcut in `read_plane_f32` must produce the same numbers as the
/// decode it skips.
///
/// For the common shape — uncompressed, one strip, native order, unsigned
/// 16-bit — the host now converts straight out of the memory map instead of
/// decoding into a scratch buffer first. That is a second code path for the
/// most common kind of file there is, and the only thing that makes it safe is
/// that both paths agree exactly. Compression forces the slow path, so writing
/// the same pixels twice gives an oracle for free.
#[test]
fn the_borrowed_and_decoded_plane_paths_agree() {
    use fast_tiff_lib::{Compression, SampleType};

    // Values chosen to catch a sign-extension or byte-order slip: 0, 1, the
    // byte boundary, and the top of the range.
    let px: Vec<u16> = vec![0, 1, 255, 256, 32767, 32768, 65534, 65535];
    let raw: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut planes = Vec::new();
    for compression in [Compression::None, Compression::Deflate] {
        let opts = WriterOptions::new(4, 2, SampleType::U16)
            .compression(compression)
            .metadata(StackMetaWrite::new(1, 1));
        let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        w.write_frame_bytes(&raw).unwrap();
        let bytes = w.finish().unwrap().into_inner();
        let s = Stack::from_bytes(bytes, "cmp.tif".into(), false).unwrap();

        // Not vacuous: the two files must genuinely differ in how they are
        // stored, or both would be taking the same path.
        assert_eq!(s.tiff.frames[0].compression, compression);

        let mut h = host(&s, 0);
        let mut got = Vec::new();
        h.read_plane_f32(Plane::new(0, 0, 0), &mut got).unwrap();
        planes.push(got);
    }

    let want: Vec<f32> = px.iter().map(|&v| v as f32).collect();
    assert_eq!(planes[0], want, "the borrowed path");
    assert_eq!(planes[1], want, "the decoded path");
    assert_eq!(planes[0], planes[1]);
}

/// Reading many planes must not allocate a buffer per plane. The scratch the
/// slow path uses lives on the host and is reused; the fast path uses none at
/// all. Checked by capacity rather than by a counter: after N reads the host
/// holds at most one buffer's worth.
#[test]
fn repeated_plane_reads_reuse_one_buffer() {
    use fast_tiff_lib::{Compression, SampleType};

    // Compressed, so the scratch path is the one under test.
    let opts = WriterOptions::new(4, 2, SampleType::U16)
        .compression(Compression::Deflate)
        .metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for _ in 0..8 {
        w.write_frame_bytes(&[7u8; 16]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    let s = Stack::from_bytes(bytes, "many.tif".into(), false).unwrap();

    let mut h = host(&s, 0);
    let mut out = Vec::new();
    for t in 0..8 {
        h.read_plane_f32(Plane::new(0, 0, t), &mut out).unwrap();
        assert_eq!(out.len(), 8);
    }
    // The plugin's own buffer was reused too — `read_plane_f32` must not leave
    // it growing by a plane each call.
    assert!(
        out.capacity() < 64,
        "capacity {} after 8 reads",
        out.capacity()
    );
}

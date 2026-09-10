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

// ------------------------------------------------- Plot third axis, end to end
//
// The fixture's pixel `i` of plane (c, z, t) is `c*100 + z*10 + t + i/1000`,
// over an 8x4 plane. So a whole-frame mean is the tag plus the mean of
// `i/1000` for i in 0..32, which is 31/2/1000 = 0.0155.

/// The mean of the fixture's plane `(c, z, t)`, worked out from the fixture's
/// own definition rather than from the code under test.
fn expected_mean(c: usize, z: usize, t: usize) -> f32 {
    let n = (W * H) as usize;
    let frac: f32 = (0..n).map(|i| i as f32 / 1000.0).sum::<f32>() / n as f32;
    plane_tag(c, z, t) + frac
}

#[test]
fn plot_third_axis_is_installed() {
    let reg = fast_tiff_viewer::plugins::Registry::new();
    let i = reg
        .find("dev.fasttiff.plot-axis")
        .expect("Plot third axis should be a built-in");
    assert_eq!(reg.entries()[i].info.name, "Plot third axis");
}

/// A stack that is one plane in both Z and T has nothing to plot against, and
/// the refusal has to say so in those words.
#[test]
fn plot_third_axis_refuses_a_stack_with_no_third_dimension() {
    let s = stack(1, 1, 1);
    assert_eq!(s.display.dims.slices, 1);
    assert_eq!(s.display.dims.frames, 1);

    let mut h = host(&s, 0);
    let err = builtin::PlotAxis
        .run(&mut h, &Params::new())
        .expect_err("a single plane has no third axis");
    let msg = err.to_string();
    assert!(
        msg.contains("no third dimension"),
        "the reason must name what is missing: {msg}"
    );
}

/// Nothing selected: one trace over the whole frame, one point per frame.
#[test]
fn plot_third_axis_measures_the_whole_frame_by_default() {
    let s = stack(1, 1, 4);
    let mut h = host(&s, 0);
    let Outcome::Plot(plot) = builtin::PlotAxis.run(&mut h, &Params::new()).expect("run") else {
        panic!("Plot third axis should return a plot");
    };

    assert_eq!(plot.series.len(), 1);
    assert_eq!(plot.series[0].label, "Whole frame");
    assert_eq!(plot.y_label, "Mean pixel value");
    assert_eq!(plot.series[0].values.len(), 4);
    for (t, v) in plot.series[0].values.iter().enumerate() {
        assert!(
            (v - expected_mean(0, 0, t)).abs() < 1e-3,
            "frame {t}: {v} vs {}",
            expected_mean(0, 0, t)
        );
    }
    // And it asks for the tool that lets the user replace this with regions.
    assert_eq!(plot.wants, fasttiff_plugin_api::SelectionKind::Regions);
}

/// With regions selected the whole-frame trace is replaced by one per region —
/// the behaviour the feature was asked for.
#[test]
fn plot_third_axis_replaces_the_whole_frame_with_the_regions() {
    let s = stack(1, 1, 2);
    // Two single-pixel regions whose values are known exactly: pixel (0,0) is
    // index 0, and pixel (7,3) is index 31 of an 8-wide plane.
    let rois = vec![
        fasttiff_plugin_api::Roi {
            shape: fasttiff_plugin_api::Shape::Rect,
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        },
        fasttiff_plugin_api::Roi {
            shape: fasttiff_plugin_api::Shape::Rect,
            x: 7,
            y: 3,
            w: 1,
            h: 1,
        },
    ];
    let mut h = StackHost::new(&s, describe_view(&s, 0, false, view())).with_selection(rois);

    let Outcome::Plot(plot) = builtin::PlotAxis.run(&mut h, &Params::new()).expect("run") else {
        panic!("expected a plot");
    };
    assert_eq!(plot.series.len(), 2, "one trace per region");
    assert_eq!(plot.series[0].label, "ROI 1");
    assert_eq!(plot.series[1].label, "ROI 2");
    assert!(
        !plot.series.iter().any(|s| s.label == "Whole frame"),
        "a selection replaces the whole-frame trace rather than joining it"
    );

    for t in 0..2 {
        let tag = plane_tag(0, 0, t);
        assert!((plot.series[0].values[t] - tag).abs() < 1e-4);
        assert!((plot.series[1].values[t] - (tag + 31.0 / 1000.0)).abs() < 1e-4);
    }
}

/// The axis is calibrated when the file says how long a frame took, so the plot
/// reads in seconds rather than in frame numbers.
#[test]
fn plot_third_axis_uses_the_files_frame_interval() {
    let s = stack(1, 1, 4);
    let info = fasttiff_plugin_api::StackInfo {
        frame_interval_s: Some(0.25),
        ..fast_tiff_viewer::plugins::describe_stack(&s)
    };
    let mut h = StackHost::new(&s, describe_view(&s, 0, false, view())).with_info(info);

    let Outcome::Plot(plot) = builtin::PlotAxis.run(&mut h, &Params::new()).expect("run") else {
        panic!("expected a plot");
    };
    assert_eq!(plot.x_step, 0.25);
    assert_eq!(plot.x_at(2), 0.5);
    assert_eq!(plot.x_label, "Time (s)");
}

/// Without a stated interval it plots against the index rather than inventing a
/// calibration.
#[test]
fn plot_third_axis_falls_back_to_frame_numbers() {
    let s = stack(1, 1, 3);
    let mut h = host(&s, 0);
    let Outcome::Plot(plot) = builtin::PlotAxis.run(&mut h, &Params::new()).expect("run") else {
        panic!("expected a plot");
    };
    assert_eq!(plot.x_step, 1.0);
    assert_eq!(plot.x_label, "T (frames)");
}

/// A 4D stack really can be walked along Z, and that is a different set of
/// planes from walking along T.
#[test]
fn plot_third_axis_can_walk_z_on_a_4d_stack() {
    let s = stack(2, 3, 2);
    assert_eq!((s.display.dims.slices, s.display.dims.frames), (3, 2));

    // `axes` offers the longer first, so Z (3) precedes T (2) and is index 0.
    let mut h = host(&s, 0);
    let Outcome::Plot(z) = builtin::PlotAxis.run(&mut h, &Params::new()).expect("run") else {
        panic!("expected a plot");
    };
    assert_eq!(z.x_label, "Z (slices)");
    assert_eq!(z.series[0].values.len(), 3);
    for (i, v) in z.series[0].values.iter().enumerate() {
        assert!((v - expected_mean(0, i, 0)).abs() < 1e-3, "slice {i}");
    }

    // And T, chosen explicitly, walks timepoints instead.
    let mut params = Params::new();
    params.set("axis", ParamValue::Choice(1));
    let mut h = host(&s, 0);
    let Outcome::Plot(t) = builtin::PlotAxis.run(&mut h, &params).expect("run") else {
        panic!("expected a plot");
    };
    assert_eq!(t.x_label, "T (frames)");
    assert_eq!(t.series[0].values.len(), 2);
    for (i, v) in t.series[0].values.iter().enumerate() {
        assert!((v - expected_mean(0, 0, i)).abs() < 1e-3, "frame {i}");
    }
}

// ------------------------------------------------ suite2p stabilization

/// A movie of one field wandering along a known path, written as a real TIFF
/// and opened as a real stack — so this exercises the plugin exactly as the
/// menu does.
fn wandering_stack(ly: u32, lx: u32, path: &[(i32, i32)], channels: usize) -> Stack {
    // Blobs, so there is something to lock onto. Flat noise registers nowhere.
    let base: Vec<f32> = {
        let mut f = vec![10.0f32; (ly * lx) as usize];
        for (cy, cx, amp) in [
            (20.0f32, 24.0f32, 200.0f32),
            (40.0, 44.0, 150.0),
            (28.0, 12.0, 120.0),
        ] {
            for y in 0..ly {
                for x in 0..lx {
                    let d2 = ((y as f32 - cy).powi(2) + (x as f32 - cx).powi(2)) / 8.0;
                    f[(y * lx + x) as usize] += amp * (-d2).exp();
                }
            }
        }
        f
    };
    let at = |dy: i32, dx: i32, y: u32, x: u32| -> f32 {
        let sy = (y as i32 - dy).rem_euclid(ly as i32) as u32;
        let sx = (x as i32 - dx).rem_euclid(lx as i32) as u32;
        base[(sy * lx + sx) as usize]
    };

    let opts =
        WriterOptions::new(lx, ly, SampleType::F32).metadata(StackMetaWrite::new(channels, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for &(dy, dx) in path {
        for c in 0..channels {
            // Every channel moves together, as a real two-channel recording
            // does — one field photographed twice at once.
            let px: Vec<f32> = (0..ly * lx)
                .map(|i| at(dy, dx, i / lx, i % lx) + c as f32 * 1000.0)
                .collect();
            w.write_frame_f32(&px).unwrap();
        }
    }
    let bytes = w.finish().unwrap().into_inner();
    Stack::from_bytes(bytes, "moving.tif".into(), false).expect("open")
}

#[test]
fn stabilize_is_installed_in_its_own_submenu() {
    let reg = fast_tiff_viewer::plugins::Registry::new();
    let i = reg
        .find("dev.fasttiff.stabilize")
        .expect("suite2p stabilization should be a built-in");
    assert_eq!(reg.entries()[i].info.name, "suite2p stabilization");
    assert_eq!(reg.entries()[i].info.menu_path, "Stabilization");
}

/// A stack with one timepoint has no motion to correct, and saying so beats
/// handing back a copy that looks registered.
#[test]
fn stabilize_refuses_a_stack_with_no_time_axis() {
    let s = stack(1, 1, 1);
    let mut h = host(&s, 0);
    let err = builtin::Stabilize
        .run(&mut h, &Params::new())
        .expect_err("one timepoint is not a time series");
    assert!(err.to_string().contains("time series"), "{err}");
}

/// The headline: a movie that wanders comes back still.
#[test]
fn stabilize_removes_a_known_motion() {
    let (ly, lx) = (64u32, 64u32);
    let path = [(0i32, 0i32), (3, -2), (-4, 1), (2, 3), (-1, -3), (4, 2)];
    let s = wandering_stack(ly, lx, &path, 1);
    let mut h = host(&s, 0);

    let mut params = Params::new();
    // A taper sized for a 64-pixel test frame; the default 40 is for a
    // 512-pixel two-photon frame and would fade this one away entirely.
    params.set("spatial_taper", ParamValue::Float(5.0));
    // And headroom on the search. The reference lands wherever the
    // best-correlated frames sit — with six frames that can be a few pixels off
    // centre — so the budget has to cover the motion *plus* that offset. The
    // default 0.1 of a 64-pixel frame is 6 pixels, and this path needs 7.
    params.set("maxregshift", ParamValue::Float(0.3));
    let Outcome::NewDocument(out) = builtin::Stabilize.run(&mut h, &params).expect("run") else {
        panic!("stabilization should produce a new document");
    };
    assert_eq!(out.frames, path.len());
    assert_eq!((out.width, out.height), (lx, ly));
    out.validate().expect("the result's shape must be valid");

    // Every registered frame should agree with the first, away from the border
    // the wrap brings the far side into.
    let planes = planes_f32(&out);
    for y in 12..(ly - 12) as usize {
        for x in 12..(lx - 12) as usize {
            let first = planes[0][y * lx as usize + x];
            for (t, p) in planes.iter().enumerate().skip(1) {
                let v = p[y * lx as usize + x];
                assert!(
                    (v - first).abs() < 1.0,
                    "frame {t} at ({y},{x}) is {v}, frame 0 is {first} — the movie \
                     is still moving after stabilization"
                );
            }
        }
    }
}

/// The channels must move together. Measuring on one and applying to all is
/// what keeps a two-channel recording in register with itself.
#[test]
fn stabilize_moves_every_channel_by_the_same_amount() {
    let (ly, lx) = (64u32, 64u32);
    let path = [(0i32, 0i32), (3, -2), (-4, 1)];
    let s = wandering_stack(ly, lx, &path, 2);
    assert_eq!(s.display.dims.channels, 2);

    let mut h = host(&s, 0);
    let mut params = Params::new();
    params.set("spatial_taper", ParamValue::Float(5.0));
    params.set("maxregshift", ParamValue::Float(0.3));
    let Outcome::NewDocument(out) = builtin::Stabilize.run(&mut h, &params).expect("run") else {
        panic!("expected a document");
    };
    assert_eq!(out.channels, 2);
    let planes = planes_f32(&out);

    // Channel 1 is channel 0 plus 1000 everywhere, by construction. If the two
    // had been registered independently they would have drifted apart.
    for t in 0..path.len() {
        let (c0, c1) = (planes[t * 2], planes[t * 2 + 1]);
        for y in 12..(ly - 12) as usize {
            for x in 12..(lx - 12) as usize {
                let i = y * lx as usize + x;
                assert!(
                    (c1[i] - c0[i] - 1000.0).abs() < 1.0,
                    "frame {t} at ({y},{x}): the channels moved apart"
                );
            }
        }
    }
}

/// A still movie must not be "corrected" into moving.
#[test]
fn stabilize_leaves_a_still_movie_alone() {
    let (ly, lx) = (64u32, 64u32);
    let s = wandering_stack(ly, lx, &[(0, 0), (0, 0), (0, 0), (0, 0)], 1);
    let mut h = host(&s, 0);
    let mut params = Params::new();
    params.set("spatial_taper", ParamValue::Float(5.0));
    params.set("maxregshift", ParamValue::Float(0.3));
    let Outcome::NewDocument(out) = builtin::Stabilize.run(&mut h, &params).expect("run") else {
        panic!("expected a document");
    };
    let planes = planes_f32(&out);
    for (t, p) in planes.iter().enumerate() {
        for (i, v) in p.iter().enumerate() {
            assert!(
                (v - planes[0][i]).abs() < 1e-3,
                "frame {t} was moved for no reason"
            );
        }
    }
}

/// Non-rigid produces a different result from rigid — if it did not, the switch
/// would be doing nothing.
#[test]
fn stabilize_non_rigid_differs_from_rigid() {
    let (ly, lx) = (128u32, 128u32);
    let path = [(0i32, 0i32), (3, -2), (-2, 1), (1, 2)];
    let s = wandering_stack(ly, lx, &path, 1);

    let run = |nonrigid: bool| {
        let mut h = host(&s, 0);
        let mut params = Params::new();
        params.set("spatial_taper", ParamValue::Float(10.0));
        params.set("maxregshift", ParamValue::Float(0.3));
        params.set("nonrigid", ParamValue::Bool(nonrigid));
        params.set("block_size", ParamValue::Int(32));
        params.set("snr_thresh", ParamValue::Float(1.0));
        match builtin::Stabilize.run(&mut h, &params).expect("run") {
            Outcome::NewDocument(out) => out,
            other => panic!("expected a document, got {other:?}"),
        }
    };

    let rigid = run(false);
    let warped = run(true);
    assert_eq!(rigid.frames, warped.frames);
    assert_eq!((rigid.width, rigid.height), (warped.width, warped.height));
    warped
        .validate()
        .expect("the warped result's shape must be valid");

    // The warp is a resample, so it cannot be bit-identical to a whole-pixel
    // roll. If it were, the block field was empty and the switch did nothing.
    let a = planes_f32(&rigid);
    let b = planes_f32(&warped);
    let differing = a
        .iter()
        .zip(&b)
        .flat_map(|(p, q)| p.iter().zip(q.iter()))
        .filter(|(x, y)| (*x - *y).abs() > 1e-4)
        .count();
    assert!(
        differing > 0,
        "non-rigid produced exactly the rigid result; the block field did nothing"
    );
}

/// A GPU backend that cannot take this stack is refused with a reason, not
/// quietly swapped for the CPU — the two are indistinguishable to the user
/// otherwise, one just being slower.
///
/// A frame that is not a power of two in both axes cannot go through the
/// device's radix-2 FFT. That holds whether or not the `gpu` feature is
/// compiled in, which is what makes this test say the same thing in every
/// build.
#[test]
fn stabilize_refuses_a_gpu_run_it_cannot_do() {
    let s = wandering_stack(48, 48, &[(0, 0), (1, 1)], 1);
    let mut h = host(&s, 0);
    let mut params = Params::new();
    // Index 2 is GPU in the selector's order.
    params.set("backend", ParamValue::Choice(2));
    let err = builtin::Stabilize
        .run(&mut h, &params)
        .expect_err("a GPU run it cannot do must be refused, not silently swapped");
    let msg = err.to_string();
    assert!(
        msg.contains("GPU"),
        "the reason must name the backend: {msg}"
    );
}

/// And the CPU backends take that same stack without complaint — so the refusal
/// above is about the GPU, not about the stack being unregisterable.
#[test]
fn stabilize_runs_the_same_stack_on_the_cpu() {
    let s = wandering_stack(48, 48, &[(0, 0), (1, 1)], 1);
    for backend in [0, 1] {
        let mut h = host(&s, 0);
        let mut params = Params::new();
        params.set("backend", ParamValue::Choice(backend));
        params.set("spatial_taper", ParamValue::Float(5.0));
        assert!(
            builtin::Stabilize.run(&mut h, &params).is_ok(),
            "backend {backend} refused a stack it should take"
        );
    }
}

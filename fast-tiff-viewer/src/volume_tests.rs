//! The background volume builder's failure, which is not a rare case.
//!
//! The worker opens its own map of the file — which a stack that came from a
//! plugin does not have. An imported OIR is held in memory and its `path` is
//! the name it is shown under, so the open fails every time, and what the
//! caller does about that decides whether 3D works at all for imported data.
//!
//! It used to sit on "Loading 3D…" forever. The caller's check was "did
//! `request` succeed", and it succeeded: the request was sent while the worker
//! was still starting, before the open had been tried. Every frame after that
//! polled for a reply from a thread that had already exited.

use super::*;

/// Wait for `f`, up to a second. The worker has to be scheduled and try to open
/// a file first, so there is nothing to assert synchronously.
fn within_a_second(what: &str, mut f: impl FnMut() -> bool) {
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(1) {
        if f() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("{what} did not happen within a second");
}

/// A file that is not there is the ordinary case for an imported stack, and the
/// builder has to *say* it gave up rather than leaving the caller to infer it.
#[test]
fn a_builder_that_cannot_open_its_file_reports_it() {
    let path = std::env::temp_dir().join("fasttiff-no-such-stack-9f3a.tif");
    let _ = std::fs::remove_file(&path);
    let builder = VolumeBuilder::new(path).expect("the thread should spawn");

    within_a_second("the builder should report that it gave up", || {
        builder.failed()
    });
}

/// Why the flag exists rather than a failed `request`.
///
/// A request sent before the worker has finished failing is *accepted* — the
/// channel is open until the thread exits and drops the receiver. A caller
/// that reads that as "the worker is fine" waits for a reply forever, which is
/// exactly what the loading screen was doing.
#[test]
fn a_request_can_succeed_against_a_worker_that_is_already_doomed() {
    let path = std::env::temp_dir().join("fasttiff-no-such-stack-4b21.tif");
    let _ = std::fs::remove_file(&path);
    let builder = VolumeBuilder::new(path).expect("the thread should spawn");

    // Whether this particular send wins the race is a scheduling detail, so it
    // is not asserted either way. What is asserted is that the flag settles on
    // "gave up" regardless — the caller has something reliable to read.
    let _accepted = builder.request(0, plan());
    within_a_second("the builder should report that it gave up", || {
        builder.failed()
    });
    // And nothing is ever going to arrive for it.
    assert!(builder.take_matching(0, 0).is_none());
}

/// A builder over a real file does not claim to have failed.
#[test]
fn a_builder_over_a_readable_file_does_not_report_failure() {
    let path = std::env::temp_dir().join("fasttiff-volume-builder-ok.tif");
    std::fs::write(&path, tiny_tiff()).expect("write a stack to open");
    let builder = VolumeBuilder::new(path.clone()).expect("the thread should spawn");

    // Give the worker the same chance to fail that the tests above give it.
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        !builder.failed(),
        "a file that opens should not be reported as unopenable"
    );
    drop(builder);
    let _ = std::fs::remove_file(&path);
}

fn plan() -> VolumePlan {
    VolumePlan {
        kinds: vec![ChannelKind::Int8],
        rgb: false,
        channels: 1,
        slices: 1,
        frames: 1,
        time: 0,
        max_dim: 256,
        format: scivis_render::pack::TexelFormat::Unorm8,
        budget: DEFAULT_VOLUME_BUDGET,
        scale: [1.0; 3],
        detail_cap: u32::MAX,
    }
}

/// The smallest stack `TiffStack::open` will accept.
fn tiny_tiff() -> Vec<u8> {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let mut w = TiffWriter::new(
        std::io::Cursor::new(Vec::new()),
        WriterOptions::new(2, 2, SampleType::U8),
    )
    .expect("writer");
    w.write_frame_bytes(&[0u8; 4]).expect("frame");
    w.finish().expect("finish").into_inner()
}

// ------------------------------------------------------------- making it fit

/// An in-memory stack of `frames` planes of `w` x `h` 16-bit samples, sample
/// `i` of frame `f` holding `value(f, i)`.
fn u16_stack(w: u32, h: u32, frames: u32, value: impl Fn(u32, u32) -> u16) -> TiffStack {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let mut wr = TiffWriter::new(
        std::io::Cursor::new(Vec::new()),
        WriterOptions::new(w, h, SampleType::U16),
    )
    .expect("writer");
    for f in 0..frames {
        let bytes: Vec<u8> = (0..w * h).flat_map(|i| value(f, i).to_ne_bytes()).collect();
        wr.write_frame_bytes(&bytes).expect("frame");
    }
    let bytes = wr.finish().expect("finish").into_inner();
    TiffStack::from_bytes(bytes).expect("parse")
}

fn u16_plan(frames: usize, budget: usize) -> VolumePlan {
    VolumePlan {
        kinds: vec![ChannelKind::Int16],
        frames,
        format: scivis_render::pack::TexelFormat::Unorm16,
        budget,
        ..plan()
    }
}

/// Every level-0 voxel of channel 0, in texture units.
fn level0(p: &PackedVolume) -> Vec<f32> {
    let f = p.layout.format;
    let texel = p.layout.texel_bytes(0);
    p.levels[0][0]
        .chunks_exact(texel)
        .map(|t| f.decode(t))
        .collect()
}

#[test]
fn the_crashing_stack_is_reduced_in_plane_and_keeps_its_depth() {
    // 4096 x 4096 x 69, two 16-bit channels, 0.124 um pixels and 2.5 um
    // slices: the in-plane axes are 20x finer and give way first.
    let s = choose_strides(
        (4096, 4096, 69),
        2,
        scivis_render::pack::TexelFormat::Unorm16,
        768 << 20,
        [0.124, 0.124, 2.5],
        2048,
        u32::MAX,
    );
    assert_eq!(s.2, 1, "depth is the coarse axis: kept whole");
    assert_eq!(s.0, s.1);
    let voxels = 4096u32.div_ceil(s.0) as f64 * 4096u32.div_ceil(s.1) as f64 * 69.0;
    let bytes = voxels * pack::bytes_per_voxel(2, scivis_render::pack::TexelFormat::Unorm16);
    assert!(bytes <= (768u64 << 20) as f64, "{bytes} over budget");
}

#[test]
fn the_finest_axis_is_reduced_first() {
    // Fine in z (0.1 against 1): z gives way until its voxels catch up.
    let s = choose_strides(
        (256, 256, 256),
        1,
        scivis_render::pack::TexelFormat::Unorm8,
        8 << 20,
        [1.0, 1.0, 0.1],
        2048,
        u32::MAX,
    );
    assert!(s.2 > 1 && s.0 == 1 && s.1 == 1, "{s:?}");
    // Equal voxels: the axis with the most of them goes first.
    let s = choose_strides(
        (100, 100, 2000),
        1,
        scivis_render::pack::TexelFormat::Unorm8,
        12 << 20,
        [1.0; 3],
        4096,
        u32::MAX,
    );
    assert!(s.2 > 1 && s.0 == 1, "{s:?}");
}

#[test]
fn no_axis_keeps_more_than_the_detail_cap_or_the_texture_limit() {
    let fmt = scivis_render::pack::TexelFormat::Unorm8;
    let s = choose_strides((8000, 100, 10), 1, fmt, usize::MAX, [1.0; 3], 16384, 2048);
    assert_eq!(s, (4, 1, 1));
    let s = choose_strides(
        (8000, 100, 10),
        1,
        fmt,
        usize::MAX,
        [1.0; 3],
        1000,
        u32::MAX,
    );
    assert_eq!(s, (8, 1, 1));
    // A budget too small for anything still ends, at one voxel.
    let s = choose_strides((50, 40, 30), 1, fmt, 0, [1.0; 3], 4096, u32::MAX);
    assert_eq!(s, (50, 40, 30));
}

/// At full size every sample arrives exactly.
#[test]
fn a_volume_that_fits_is_packed_sample_for_sample() {
    let tiff = u16_stack(5, 3, 4, |f, i| (f * 1000 + i * 37) as u16);
    let p = build_volume(&tiff, &u16_plan(4, usize::MAX)).expect("built");
    assert_eq!(p.layout.dims, (5, 3, 4));
    let got: Vec<u16> = p.levels[0][0]
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    let want: Vec<u16> = (0..4)
        .flat_map(|f| (0..15).map(move |i| (f * 1000 + i * 37) as u16))
        .collect();
    assert_eq!(got, want);
    assert_eq!(p.levels[0].len() as u32, p.layout.levels, "mips built");
    assert_eq!(p.bricks.max.len(), 1, "brick maxima built");
}

/// Reduced, each voxel is the mean of the box of source voxels behind it,
/// the boxes at the far edges clipped to what is there.
#[test]
fn a_reduced_volume_averages_each_box() {
    let (w, h, d) = (7u32, 5u32, 5u32);
    let value = |f: u32, i: u32| ((f * 3001 + i * 401) % 60_000) as u16;
    let tiff = u16_stack(w, h, d, value);
    // Small enough a budget to force a reduction on every axis.
    let budget = 40;
    let plan = u16_plan(d as usize, budget);
    let (sx, sy, sz) = choose_strides(
        (w, h, d),
        1,
        plan.format,
        budget,
        plan.scale,
        plan.max_dim,
        plan.detail_cap,
    );
    assert!(sx > 1 || sy > 1 || sz > 1, "the test must reduce something");
    let p = build_volume(&tiff, &plan).expect("built");
    let (ow, oh, od) = p.layout.dims;
    assert_eq!(
        (ow, oh, od),
        (w.div_ceil(sx), h.div_ceil(sy), d.div_ceil(sz))
    );
    let got = level0(&p);
    for oz in 0..od {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut sum = 0.0f64;
                let mut n = 0.0f64;
                for z in oz * sz..((oz + 1) * sz).min(d) {
                    for y in oy * sy..((oy + 1) * sy).min(h) {
                        for x in ox * sx..((ox + 1) * sx).min(w) {
                            sum += value(z, y * w + x) as f64 / 65535.0;
                            n += 1.0;
                        }
                    }
                }
                let want = (sum / n) as f32;
                let at = ((oz * oh + oy) * ow + ox) as usize;
                assert!(
                    (got[at] - want).abs() <= 1.0 / 65535.0,
                    "voxel ({ox}, {oy}, {oz}): {} vs {want}",
                    got[at]
                );
            }
        }
    }
}

/// A stack masked with NaN (ImageJ's "NaN background") keeps its edges: a
/// NaN is left out of a box's average instead of turning all of it NaN.
#[test]
fn a_nan_is_left_out_of_an_average() {
    let format = TexelFormat::Float32;
    let layout = VolumeLayout::new((2, 1, 1), 1, format);
    // Two slices of 4 x 2, into 2 x 1 x 1: the left box has three numbers
    // among its NaNs, the right one nothing but NaN.
    let nan = f32::NAN;
    let mut boxes = SliceBoxes::new(1, (4, 2), (2, 2), (2, 1));
    boxes.add(
        0,
        &Decoded::F32(vec![1.0, nan, nan, nan, 3.0, 5.0, nan, nan]),
    );
    boxes.add(
        0,
        &Decoded::F32(vec![nan, nan, nan, nan, nan, nan, nan, nan]),
    );
    let mut slab = vec![0u8; layout.slice_bytes(0)];
    boxes.write(&layout, &mut [&mut slab[..]], 2);
    let got: Vec<f32> = slab.chunks_exact(4).map(|b| format.decode(b)).collect();
    assert_eq!(got[0], 3.0, "(1 + 3 + 5) / 3");
    assert!(got[1].is_nan(), "a box of nothing but NaN stays NaN");

    // Infinities (the log of a zero, a ratio over one) are left out of an
    // average too, and a box of nothing else keeps the one it saw.
    let (inf, ninf) = (f32::INFINITY, f32::NEG_INFINITY);
    let layout = VolumeLayout::new((3, 1, 1), 1, format);
    let mut boxes = SliceBoxes::new(1, (6, 1), (2, 1), (3, 1));
    boxes.add(0, &Decoded::F32(vec![2.0, ninf, inf, ninf, ninf, nan]));
    let mut slab = vec![0u8; layout.slice_bytes(0)];
    boxes.write(&layout, &mut [&mut slab[..]], 1);
    let got: Vec<f32> = slab.chunks_exact(4).map(|b| format.decode(b)).collect();
    assert_eq!(
        got,
        vec![2.0, inf, ninf],
        "a finite mean, then +inf over -inf"
    );

    // Integer channels are counted as before: every voxel of the box.
    let layout = VolumeLayout::new((1, 1, 1), 1, TexelFormat::Unorm16);
    let mut boxes = SliceBoxes::new(1, (2, 1), (2, 1), (1, 1));
    boxes.add(0, &Decoded::U16(vec![0, 65535]));
    let mut slab = vec![0u8; layout.slice_bytes(0)];
    boxes.write(&layout, &mut [&mut slab[..]], 1);
    assert_eq!(TexelFormat::Unorm16.decode(&slab), 32768.0 / 65535.0);
}

/// A smaller rebuild after the GPU ran out of memory is the same volume: a
/// build in flight is dropped, but what a frontend compares to tell that a
/// different stack is on show (an animation stops for that) does not move.
#[test]
fn a_rebuild_is_not_a_different_stack() {
    let mut v = crate::viewer::VolumeView {
        built_frame: Some(0),
        requested: Some((0, 0)),
        planned_budget: Some(1 << 20),
        ..Default::default()
    };
    v.rebuild();
    assert_eq!((v.generation, v.stack_generation), (1, 0));
    assert_eq!(
        (v.built_frame, v.requested, v.planned_budget),
        (None, None, None)
    );
    v.invalidate();
    assert_eq!(
        (v.generation, v.stack_generation),
        (2, 1),
        "a new stack moves both"
    );
}

#[test]
fn the_detail_cap_follows_the_screen_with_a_floor() {
    assert_eq!(detail_cap(0), u32::MAX, "unknown screen: no cap");
    assert_eq!(detail_cap(1920), 3840);
    assert_eq!(detail_cap(800), 2048);
}

#[test]
fn a_reduced_view_takes_as_many_steps_per_ray_as_a_sharp_one() {
    let mut v = crate::viewer::VolumeView {
        viewport_px: [1921, 1081],
        ..Default::default()
    };
    let sharp = v.max_steps(2);
    assert_eq!(
        sharp,
        scivis_render::max_steps(1921.0 * 1081.0, 2, v.interp),
        "the whole viewport's"
    );
    v.downscale = 4;
    assert_eq!(
        v.max_steps(2),
        sharp,
        "a reduced frame's rays take the steps a sharp frame's do, so it does a sixteenth of the work"
    );
}

/// The budget's bounds are ordered on every target: a ceiling that wrapped
/// (4 GB does not fit a 32-bit `usize`) made `clamp` panic in the browser.
#[test]
fn the_budget_bounds_are_ordered() {
    const { assert!(MIN_VOLUME_BUDGET < DEFAULT_VOLUME_BUDGET) };
    const { assert!(DEFAULT_VOLUME_BUDGET < MAX_VOLUME_BUDGET) };
}

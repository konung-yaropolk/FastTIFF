//! The background volume builder, and the stacks it has to serve.
//!
//! The worker shares the window's own stack rather than opening the file again,
//! which matters most for a stack that has no file: an imported OIR or a
//! plugin's result is held in memory, and its `path` is only the name it is
//! shown under. A worker that re-opened by path failed on every one of them,
//! and 3D for imported data was built on the interface thread — or, before
//! that was noticed, sat on "Loading 3D…" for ever.

use super::*;

/// Wait for `f`, up to five seconds: the worker has to be scheduled and build
/// the volume first, so there is nothing to assert synchronously.
#[cfg(feature = "threads")]
fn within_seconds(what: &str, mut f: impl FnMut() -> bool) {
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(5) {
        if f() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("{what} did not happen within five seconds");
}

/// A stack held only in memory — the imported case — gets its volume built in
/// the background and answered under the request's key.
#[cfg(feature = "threads")]
#[test]
fn a_stack_held_in_memory_is_built_in_the_background() {
    let stack = Arc::new(TiffStack::from_bytes(tiny_tiff()).expect("parse"));
    let builder = VolumeBuilder::new(stack).expect("the thread should spawn");
    assert!(builder.request(7, plan()));
    let mut answer = None;
    within_seconds("the builder should answer", || {
        answer = builder.take_matching(7, 0);
        answer.is_some()
    });
    // Another key gets nothing: the reply was taken, and it was not theirs.
    assert!(builder.take_matching(7, 0).is_none());
    assert!(builder.take_matching(8, 0).is_none());
}

// ------------------------------------------------------------- cancelling

/// A build told to stop after its first plane stops there: it reports
/// `Cancelled`, not a volume, and reads none of the other 63 planes.
///
/// The token says "go on" exactly once and the build asks it before every
/// plane it reads, so exactly one plane was decoded. A build that never asked,
/// or asked only once at the start, would come back with a volume.
#[test]
fn a_cancelled_build_stops_at_the_next_plane() {
    const DEPTH: u32 = 64;
    let tiff = u16_stack(8, 8, DEPTH, |f, i| (f * 64 + i) as u16);
    let plan = u16_plan(DEPTH as usize, usize::MAX);

    let (token, asked) = crate::worker::CancelToken::after(1);
    assert_eq!(
        build_volume(&tiff, &plan, &token).map(|_| ()),
        Err(Cancelled)
    );
    let asked = asked.load(std::sync::atomic::Ordering::Relaxed);
    assert!(asked >= 2, "asked again after the one plane it was allowed");
    // One slice at a time, it stops at the very next one.
    #[cfg(not(feature = "threads"))]
    assert_eq!(asked, 2);

    // Never cancelled, the same build completes.
    let built = build_volume(&tiff, &plan, &CancelToken::never()).expect("not cancelled");
    assert_eq!(built.expect("built").layout.dims, (8, 8, DEPTH));
}

/// The mip chain is a pass over the whole volume, so a build cancelled after
/// its last plane does not go on to build one.
#[test]
fn a_build_cancelled_after_its_planes_skips_the_mip_chain() {
    const DEPTH: u32 = 4;
    let tiff = u16_stack(4, 4, DEPTH, |f, i| (f + i) as u16);
    let (token, asked) = crate::worker::CancelToken::after(DEPTH as usize);
    let got = build_volume(&tiff, &u16_plan(DEPTH as usize, usize::MAX), &token);
    assert_eq!(got.map(|_| ()), Err(Cancelled));
    assert_eq!(
        asked.load(std::sync::atomic::Ordering::Relaxed),
        DEPTH as usize + 1,
        "once per plane, and once before the mip chain"
    );
}

/// Requests for different keys in a row: the newest is answered, and nothing
/// for the older ones is left to take.
///
/// Whether an older one was replaced before it started or stopped part-way is
/// up to the scheduler here, so this proves only which answer wins. The
/// stopping is forced where it can be: the worker's tests overtake a job they
/// hold part-way, and `a_cancelled_build_stops_at_the_next_plane` cancels a
/// build at a known plane.
#[cfg(feature = "threads")]
#[test]
fn the_newest_of_several_requests_is_the_one_answered() {
    let stack = Arc::new(u16_stack(64, 64, 64, |f, i| (f * 7 + i) as u16));
    let builder = VolumeBuilder::new(stack).expect("the thread should spawn");
    for generation in 1..=3 {
        assert!(builder.request(generation, u16_plan(64, usize::MAX)));
    }
    let mut answer = None;
    within_seconds("the newest build", || {
        answer = builder.take_matching(3, 0);
        answer.is_some()
    });
    assert!(answer.flatten().is_some(), "and it built");
    assert!(builder.take_matching(1, 0).is_none());
    assert!(builder.take_matching(2, 0).is_none());
}

/// Dropping the builder — closing the file — with a build asked for lets go
/// of the stack (the file's mapping, for a mapped file): its thread exits and
/// keeps nothing.
///
/// This build is over long before the bound, cancelled or not, so the test
/// says nothing about *when* the stack goes. That a dropped worker stops its job at
/// the next check is forced in the worker's tests
/// (`dropping_a_worker_mid_job_lets_go_of_what_it_holds`), and that a build
/// checks at every plane in `a_cancelled_build_stops_at_the_next_plane`.
#[cfg(feature = "threads")]
#[test]
fn a_dropped_builder_lets_go_of_its_stack() {
    let stack = Arc::new(u16_stack(256, 256, 128, |f, i| (f + i) as u16));
    let weak = Arc::downgrade(&stack);
    let builder = VolumeBuilder::new(stack).expect("the thread should spawn");
    assert!(builder.request(1, u16_plan(128, usize::MAX)));
    drop(builder);
    within_seconds("the dropped builder letting go of its stack", || {
        weak.upgrade().is_none()
    });
}

fn plan() -> VolumePlan {
    VolumePlan {
        units: vec![SampleUnits::RAW8],
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
#[cfg(feature = "threads")]
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
        units: vec![SampleUnits::display(fast_tiff_lib::DisplayMap::Identity)],
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
    let p = build_volume(&tiff, &u16_plan(4, usize::MAX), &CancelToken::never())
        .expect("not cancelled")
        .expect("built");
    assert_eq!(p.layout.dims, (5, 3, 4));
    let got: Vec<u16> = p.levels[0][0]
        .as_chunks::<2>()
        .0
        .iter()
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
    let p = build_volume(&tiff, &plan, &CancelToken::never())
        .expect("not cancelled")
        .expect("built");
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
    boxes.write(&layout, &mut [&mut slab[..]], 2, &[(1.0, 0.0)]);
    let got: Vec<f32> = slab
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| format.decode(b))
        .collect();
    assert_eq!(got[0], 3.0, "(1 + 3 + 5) / 3");
    assert!(got[1].is_nan(), "a box of nothing but NaN stays NaN");

    // Infinities (the log of a zero, a ratio over one) are left out of an
    // average too, and a box of nothing else keeps the one it saw.
    let (inf, ninf) = (f32::INFINITY, f32::NEG_INFINITY);
    let layout = VolumeLayout::new((3, 1, 1), 1, format);
    let mut boxes = SliceBoxes::new(1, (6, 1), (2, 1), (3, 1));
    boxes.add(0, &Decoded::F32(vec![2.0, ninf, inf, ninf, ninf, nan]));
    let mut slab = vec![0u8; layout.slice_bytes(0)];
    boxes.write(&layout, &mut [&mut slab[..]], 1, &[(1.0, 0.0)]);
    let got: Vec<f32> = slab
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| format.decode(b))
        .collect();
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
    let identity16 = SampleUnits::display(fast_tiff_lib::DisplayMap::Identity);
    boxes.write(
        &layout,
        &mut [&mut slab[..]],
        1,
        &[identity16.volume_affine()],
    );
    assert_eq!(TexelFormat::Unorm16.decode(&slab), 32768.0 / 65535.0);
}

/// A float volume is stored as half floats on wgpu, which hold nothing past
/// 65504. Normalized over the channel's span first, a stack of photon counts
/// in the hundreds of thousands stays finite and keeps its order, and the
/// window goes through the same map; stored raw, all of it was infinity, which
/// rendered white whatever the window.
#[test]
fn a_float_volume_past_the_half_float_range_stays_finite() {
    let format = TexelFormat::Float16;
    let layout = VolumeLayout::new((4, 1, 1), 1, format);
    let span = crate::units::Window::new(0.0, 400_000.0);
    let units = SampleUnits::float(span);
    let values = vec![0.0f32, 100_000.0, 250_000.0, 400_000.0];
    let mut slab = vec![0u8; layout.slice_bytes(0)];
    encode_slice(
        &layout,
        &mut [&mut slab[..]],
        0,
        &Decoded::F32(values.clone()),
        units.volume_affine(),
    );
    let got: Vec<f32> = slab
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| format.decode(b))
        .collect();
    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    assert!(got.windows(2).all(|w| w[0] < w[1]), "order kept: {got:?}");
    for (v, g) in values.iter().zip(&got) {
        assert!((g - v / 400_000.0).abs() < 1e-3, "{v} stored as {g}");
    }
    // The window lands on the voxels' scale: 100k..250k selects the middle two.
    let (lo, hi) = units.to_unorm(crate::units::Window::new(100_000.0, 250_000.0));
    assert!(
        (lo - got[1]).abs() < 1e-3 && (hi - got[2]).abs() < 1e-3,
        "{lo}..{hi} vs {got:?}"
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

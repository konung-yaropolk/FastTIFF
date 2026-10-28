//! Crate-level properties: the things a caller can rely on across modules.
//!
//! The algorithm's own arithmetic is tested beside it, module by module. What
//! is here is the contract the crate advertises.

/// The licence note in the manifest is load-bearing, not decoration: this crate
/// may be distributed because EPFL relicensed TurboReg/StackReg to GPL-3+, and
/// it may *not* be a transcription of pystackreg, whose own terms are not
/// GPL-compatible. A build that quietly became something else — an added
/// dependency under different terms, say — is worth catching here rather than
/// in a licence audit.
#[test]
fn the_crate_declares_the_licence_it_relies_on() {
    assert_eq!(
        env!("CARGO_PKG_LICENSE"),
        "GPL-3.0-only",
        "the algorithm's licence travels with it, whoever retypes it"
    );
}

use super::*;

/// A drifting stack, each frame shifted a little further than the last.
fn drifting(w: usize, h: usize, n: usize, per_frame: f64) -> Vec<Vec<f32>> {
    let base: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            128.0
                + (x * 0.23).sin() * 50.0
                + (y * 0.19).cos() * 40.0
                + (x * 0.7 + y * 0.5).sin() * 15.0
        })
        .collect();
    (0..n)
        .map(|t| {
            let d = per_frame * t as f64;
            transform::warp(
                &base,
                w,
                h,
                &Matrix {
                    c: [d, 1.0, 0.0, 0.0, d * 0.5, 0.0, 1.0, 0.0],
                },
            )
        })
        .collect()
}

fn stack_settings(reference: Reference) -> Settings {
    Settings {
        transformation: Transformation::Translation,
        reference,
        backend: Backend::SingleThread,
        ..Settings::default()
    }
}

/// Every reference mode takes the drift out: after registering, the frames sit
/// on top of each other. Which target they used differs; that they end up
/// aligned does not.
#[test]
fn every_reference_mode_removes_a_drift() {
    let (w, h, n) = (72, 64, 6);
    let frames = drifting(w, h, n, 1.5);
    for mode in Reference::all() {
        let movie = Frames {
            ly: h,
            lx: w,
            frames: &frames,
        };
        let out = register_stack(&movie, &stack_settings(mode), &mut |_| true)
            .unwrap_or_else(|| panic!("{mode:?} should register"));
        assert_eq!(out.matrices.len(), n);
        assert!(out.failed.is_empty(), "{mode:?} failed on {:?}", out.failed);

        // Register each frame and compare against the first registered frame.
        let aligned: Vec<Vec<f32>> = frames
            .iter()
            .enumerate()
            .map(|(t, f)| out.apply(f, h, w, t).expect("no crop asked for"))
            .collect();
        let mut worst = 0.0f32;
        for t in 1..n {
            for y in 12..h - 12 {
                for x in 12..w - 12 {
                    worst = worst.max((aligned[t][y * w + x] - aligned[0][y * w + x]).abs());
                }
            }
        }
        assert!(
            worst < 12.0,
            "{mode:?}: frames still differ by {worst} after registering"
        );
    }
}

/// Under `First` the first frame is the target, so it keeps the identity rather
/// than being fitted against itself.
#[test]
fn the_first_frame_is_untouched_when_it_is_the_reference() {
    let frames = drifting(48, 48, 4, 1.0);
    let movie = Frames {
        ly: 48,
        lx: 48,
        frames: &frames,
    };
    let out =
        register_stack(&movie, &stack_settings(Reference::First), &mut |_| true).expect("fit");
    assert_eq!(
        out.matrices[0],
        Matrix::identity(),
        "the reference frame should not be moved"
    );
}

/// `Previous` composes down the stack, so a steady drift gives steadily growing
/// transformations rather than the same small one repeated.
#[test]
fn the_chained_mode_accumulates() {
    let frames = drifting(64, 56, 5, 2.0);
    let movie = Frames {
        ly: 56,
        lx: 64,
        frames: &frames,
    };
    let out =
        register_stack(&movie, &stack_settings(Reference::Previous), &mut |_| true).expect("fit");
    let shifts: Vec<f64> = out.matrices.iter().map(|m| m.c[0]).collect();
    for t in 2..shifts.len() {
        assert!(
            shifts[t].abs() > shifts[t - 1].abs(),
            "chained shifts should grow: {shifts:?}"
        );
    }
    assert!(
        out.reference.is_none(),
        "a chained run has no single target"
    );
}

/// Bilinear cannot be chained, and says so rather than producing a matrix that
/// is not in the family.
#[test]
fn bilinear_refuses_the_chained_reference() {
    let frames = drifting(48, 48, 3, 1.0);
    let movie = Frames {
        ly: 48,
        lx: 48,
        frames: &frames,
    };
    let s = Settings {
        transformation: Transformation::Bilinear,
        reference: Reference::Previous,
        ..Settings::default()
    };
    assert!(register_stack(&movie, &s, &mut |_| true).is_none());
}

/// Progress is reported and cancellation is honoured.
#[test]
fn a_run_can_be_stopped() {
    let frames = drifting(48, 48, 5, 1.0);
    let movie = Frames {
        ly: 48,
        lx: 48,
        frames: &frames,
    };
    let mut seen: Vec<f32> = Vec::new();
    let out = register_stack(&movie, &stack_settings(Reference::Mean), &mut |f| {
        seen.push(f);
        true
    });
    assert!(out.is_some());
    assert!(!seen.is_empty(), "progress should be reported");
    assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)), "{seen:?}");
    // And refusing stops it.
    let stopped = register_stack(&movie, &stack_settings(Reference::Mean), &mut |_| false);
    assert!(stopped.is_none(), "returning false should stop the run");
}

/// An empty stack is not an error.
#[test]
fn an_empty_stack_registers_to_nothing() {
    let movie = Frames {
        ly: 8,
        lx: 8,
        frames: &[],
    };
    let out = register_stack(&movie, &Settings::default(), &mut |_| true).expect("ok");
    assert!(out.matrices.is_empty());
}

/// The temporal smoothing window is centred and odd, so it raises signal
/// without shifting it. An even window would be off-centre by half a frame,
/// which is a drift the registration would then faithfully remove.
#[test]
fn the_moving_average_is_centred() {
    let frames: Vec<Vec<f32>> = (0..5).map(|t| vec![t as f32 * 10.0; 4]).collect();
    let smoothed = moving_average(&frames, 3, false);
    assert_eq!(smoothed.len(), 5);
    // The middle frame of a linear ramp is unchanged by a centred window.
    assert!((smoothed[2][0] - 20.0).abs() < 1e-5, "{:?}", smoothed[2]);
    // An even window is treated as the next odd one rather than being
    // off-centre.
    assert_eq!(moving_average(&frames, 4, false)[2][0], smoothed[2][0]);
    // A window of one changes nothing.
    assert_eq!(moving_average(&frames, 1, false), frames);
}

/// The stack-wide averages are the same numbers on every core: each pixel's
/// sum still runs over the frames in order, whichever thread takes its piece.
#[test]
fn the_stack_averages_do_not_depend_on_threads() {
    // Enough pixels to span several pieces, and values that make float
    // addition order-sensitive, so a reordered sum would show.
    let len = 3 * PIXELS_PER_CHUNK + 17;
    let frames: Vec<Vec<f32>> = (0..7)
        .map(|t| {
            (0..len)
                .map(|i| ((i * 31 + t * 7) % 1000) as f32 * 0.1 + 1e6 * (t % 2) as f32)
                .collect()
        })
        .collect();
    for take in [1, 3, 7] {
        assert_eq!(
            mean_of(&frames, take, false),
            mean_of(&frames, take, true),
            "the mean of {take} frames should not depend on threads"
        );
    }
    assert_eq!(
        moving_average(&frames, 3, false),
        moving_average(&frames, 3, true),
        "the moving average should not depend on threads"
    );
}

/// A whole stack registers to bit-identical matrices on either backend, in
/// every reference mode — including the chained one, which is the mode that
/// used to run on one core whatever was chosen.
#[test]
fn both_backends_register_a_stack_identically() {
    let frames = drifting(64, 56, 4, 1.5);
    let movie = Frames {
        ly: 56,
        lx: 64,
        frames: &frames,
    };
    for mode in Reference::all() {
        let single = register_stack(&movie, &stack_settings(mode), &mut |_| true).expect("single");
        let multi_settings = Settings {
            backend: Backend::MultiThread,
            ..stack_settings(mode)
        };
        let multi = register_stack(&movie, &multi_settings, &mut |_| true).expect("multi");
        assert_eq!(
            single.matrices, multi.matrices,
            "{mode:?}: the backends should agree bit for bit"
        );
        assert_eq!(single.failed, multi.failed);
        // And the result carries the backend into `apply`.
        assert!(!single.threaded);
        assert_eq!(multi.threaded, Backend::MultiThread.available());
    }
}

/// Under `Border::Crop` every frame comes back cut to the common area — the
/// zero-filled frame with its strip cut off — and frames with nothing in
/// common are an error rather than something else.
#[test]
fn a_cropped_stack_hands_back_the_common_area() {
    let (w, h) = (48, 40);
    let frames = drifting(w, h, 5, 0.8);
    let movie = Frames {
        ly: h,
        lx: w,
        frames: &frames,
    };
    let mut out =
        register_stack(&movie, &stack_settings(Reference::First), &mut |_| true).expect("ran");
    out.border = Border::Crop;
    let area = out.common_area(h, w).expect("an area");
    assert!(area.is_upright(), "a translation leaves an upright area: {area:?}");
    assert!(area.height < h && area.width < w, "a drift uncovers both edges: {area:?}");
    let (y0, x0) = (area.y as usize, area.x as usize);
    assert_eq!(out.output_size(h, w), Ok((area.height, area.width)));
    for (t, frame) in frames.iter().enumerate() {
        let cut = out.apply(frame, h, w, t).expect("an area");
        let filled = transform::warp_bordered(
            frame,
            w,
            h,
            &out.matrices[t],
            Border::Constant,
            out.threaded,
        );
        let want: Vec<f32> = (y0..y0 + area.height)
            .flat_map(|y| filled[y * w + x0..y * w + x0 + area.width].to_vec())
            .collect();
        assert_eq!(cut, want, "frame {t}");
    }

    let mut apart = register_stack(&movie, &stack_settings(Reference::First), &mut |_| true)
        .expect("ran");
    apart.border = Border::Crop;
    apart.matrices[1].c[0] = 100.0;
    let err = apart.common_area(h, w).expect_err("nothing in common");
    assert_eq!(err.frames, 5);
    assert_eq!(apart.apply(&frames[0], h, w, 0), Err(err));
    apart.border = Border::Edge;
    assert!(apart.apply(&frames[0], h, w, 0).is_ok());
}

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
            .unwrap_or_else(|e| panic!("{mode:?} should register: {e}"));
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
    assert_eq!(
        register_stack(&movie, &s, &mut |_| true).err(),
        Some(Error::Unchainable)
    );
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
    assert!(out.is_ok());
    assert!(!seen.is_empty(), "progress should be reported");
    assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)), "{seen:?}");
    // And refusing stops it, there and then: the first call is the last.
    let mut calls = 0;
    let stopped = register_stack(&movie, &stack_settings(Reference::Mean), &mut |_| {
        calls += 1;
        false
    });
    assert_eq!(
        stopped.err(),
        Some(Error::Cancelled),
        "returning false should stop the run"
    );
    assert_eq!(calls, 1, "the run went on after being told to stop");
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
    // Without the `threads` feature the multi-thread backend is refused, not
    // quietly run on one thread.
    if !Backend::MultiThread.available() {
        return;
    }
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
        assert!(multi.threaded);
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
    assert!(
        area.is_upright(),
        "a translation leaves an upright area: {area:?}"
    );
    assert!(
        area.height < h && area.width < w,
        "a drift uncovers both edges: {area:?}"
    );
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

    let mut apart =
        register_stack(&movie, &stack_settings(Reference::First), &mut |_| true).expect("ran");
    apart.border = Border::Crop;
    apart.matrices[1].c[0] = 100.0;
    let err = apart.common_area(h, w).expect_err("nothing in common");
    assert_eq!(err.frames, 5);
    assert_eq!(apart.apply(&frames[0], h, w, 0), Err(err));
    apart.border = Border::Edge;
    assert!(apart.apply(&frames[0], h, w, 0).is_ok());
}

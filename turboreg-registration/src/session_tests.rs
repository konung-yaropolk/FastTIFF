//! What a session asks for, what it keeps, and what it refuses.
//!
//! That its answers are the old driver's is `session_oracle_tests.rs`; these
//! are the properties of the feeding itself: the passes and their order, the
//! bound on what a session holds between batches, and a batch that is not
//! what was asked for coming back as an error rather than a wrong answer.

use super::*;
use crate::{register_stack, transform, Frames};

const W: usize = 24;
const H: usize = 20;

fn stack(n: usize) -> Vec<Vec<f32>> {
    let base: Vec<f32> = (0..W * H)
        .map(|i| {
            let (x, y) = ((i % W) as f32, (i / W) as f32);
            100.0 + (x * 0.4).sin() * 30.0 + (y * 0.3).cos() * 20.0
        })
        .collect();
    (0..n)
        .map(|t| {
            let d = 0.5 * t as f64;
            transform::warp(
                &base,
                W,
                H,
                &Matrix {
                    c: [d, 1.0, 0.0, 0.0, -d, 0.0, 1.0, 0.0],
                },
            )
        })
        .collect()
}

fn on_one_thread(reference: Reference, window: usize, n_frames: usize) -> Settings {
    Settings {
        transformation: crate::Transformation::Translation,
        reference,
        moving_average: window,
        n_frames,
        backend: Backend::SingleThread,
        ..Settings::default()
    }
}

/// Everything a session asks for, in order.
fn asked(mut session: Session, frames: &[Vec<f32>]) -> Vec<Wanted> {
    let mut all = Vec::new();
    while let Some(wanted) = session.wanted() {
        session
            .feed(Batch::Split(&frames[wanted.frames.clone()]))
            .unwrap();
        all.push(wanted);
    }
    session.finish().expect("done");
    all
}

/// The target's frames come first, in a pass of their own, then the stack —
/// from where the first frame fitted needs it: its own window under the
/// first-frame target, the one before it under the chain.
#[test]
fn the_target_is_asked_for_before_the_stack() {
    let n = 7;
    let plan = |reference, window, n_frames| {
        Session::new(H, W, n, &on_one_thread(reference, window, n_frames))
            .unwrap()
            .plan()
    };
    // The first frame alone is the target, and frame 0 is not fitted.
    assert_eq!(
        plan(Reference::First, 1, 1),
        Plan {
            reference: 0..1,
            fit: 1..n
        }
    );
    // Three frames averaged, smoothed over five: the target needs frames up
    // to 2 + 2, and frame 1's window reaches back to frame 0.
    assert_eq!(
        plan(Reference::First, 5, 3),
        Plan {
            reference: 0..5,
            fit: 0..n
        }
    );
    // A target longer than the stack is the whole stack.
    assert_eq!(plan(Reference::First, 1, 50).reference, 0..n);
    assert_eq!(
        plan(Reference::Mean, 3, 1),
        Plan {
            reference: 0..n,
            fit: 0..n
        }
    );
    // The chain builds no target; frame 0 is read to fit frame 1 against.
    assert_eq!(
        plan(Reference::Previous, 3, 1),
        Plan {
            reference: 0..0,
            fit: 0..n
        }
    );

    // And it asks for them in that order, a batch at a time.
    let frames = stack(n);
    let s = on_one_thread(Reference::Mean, 1, 1);
    let session = Session::new(H, W, n, &s).unwrap();
    assert_eq!(session.batch_len(), 1, "one thread, one frame");
    let wanted = asked(session, &frames);
    let expect: Vec<Wanted> = (0..n)
        .map(|t| Wanted {
            frames: t..t + 1,
            pass: Pass::Reference,
        })
        .chain((0..n).map(|t| Wanted {
            frames: t..t + 1,
            pass: Pass::Fit,
        }))
        .collect();
    assert_eq!(wanted, expect);
}

/// What a session keeps between batches is bounded by the batch and the
/// window, never by the stack: at most a batch of smoothed frames, the ring
/// of the frames before the batch, and the one frame the chain is fitted
/// against. Checked after every batch of a long stack, and with a whole pass
/// fed at once, which is still fitted a batch at a time.
#[test]
fn a_session_holds_a_batch_not_the_stack() {
    let n = 16;
    let frames = stack(n);
    let pixels = W * H;
    for reference in Reference::all() {
        for window in [1, 5] {
            let s = on_one_thread(reference, window, 3);
            let mut session = Session::new(H, W, n, &s).unwrap();
            let per = session.batch_len();
            let half = window / 2;
            // The pair list keeps its allocation from batch to batch: what
            // it holds between them is nothing, but its capacity carries on.
            let mut kept = 0;
            while let Some(wanted) = session.wanted() {
                session.feed(Batch::Split(&frames[wanted.frames])).unwrap();
                assert!(session.smoothed.len() <= per, "{s:?}");
                assert!(session.ring.len() <= 2 * half, "{s:?}");
                assert!(session.previous.len() <= pixels, "{s:?}");
                assert!(session.sum.len() <= pixels, "{s:?}");
                assert!(session.spare_pairs.is_empty(), "{s:?}");
                if !session.stages.is_empty() {
                    kept = kept.max(session.spare_pairs.capacity());
                    if kept > 0 {
                        assert_eq!(
                            session.spare_pairs.capacity(),
                            kept,
                            "{s:?}: the pair list was allocated afresh"
                        );
                    }
                }
                if window == 1 {
                    assert!(
                        session.smoothed.is_empty() && session.ring.is_empty(),
                        "{s:?}"
                    );
                }
            }
            assert!(kept >= per, "{s:?}: the pair list was not kept");
            session.finish().unwrap();

            // A whole pass at a time: the pieces are still batches.
            let mut session = Session::new(H, W, n, &s).unwrap();
            while let Some(wanted) = session.wanted() {
                let pass = session.stages[0].reads.clone();
                assert_eq!(wanted.frames.start, pass.start);
                session.feed(Batch::Split(&frames[pass])).unwrap();
                assert!(session.smoothed.len() <= per, "{s:?}, a pass at once");
            }
            let whole = session.finish().unwrap();
            let fed = register_stack(
                &Frames {
                    ly: H,
                    lx: W,
                    frames: &frames,
                },
                &s,
                &mut |_| true,
            )
            .unwrap();
            assert_eq!(whole.matrices, fed.matrices, "{s:?}");
        }
    }
}

/// A batch that is not what the session asked for is an error that says
/// what is wrong, and leaves the session as it was.
#[test]
fn a_batch_it_did_not_ask_for_is_refused() {
    let frames = stack(4);
    let s = on_one_thread(Reference::Previous, 1, 1);
    let mut session = Session::new(H, W, 4, &s).unwrap();
    let short = vec![vec![0.0; W * H - 1]];
    let five = stack(5);
    for wrong in [
        Batch::Split(&short),
        Batch::Packed(&frames[0][..W * H - 3]),
        Batch::Split(&[]),
        // More than the pass has left.
        Batch::Split(&five),
    ] {
        assert!(
            matches!(session.feed(wrong), Err(Error::Batch(_))),
            "{wrong:?}"
        );
    }
    // Nothing moved: the next frames are still the first ones.
    assert_eq!(session.wanted().unwrap().frames, 0..1);
    let early = Session::new(H, W, 4, &s).unwrap();
    assert_eq!(early.finish().err(), Some(Error::Unfinished));

    let done = asked(Session::new(H, W, 4, &s).unwrap(), &frames);
    assert_eq!(done.len(), 4);
    let mut over = Session::new(H, W, 1, &s).unwrap();
    assert!(over.wanted().is_none(), "one frame has nothing to chain");
    assert!(matches!(
        over.feed(Batch::Split(&frames[..1])),
        Err(Error::Batch(_))
    ));
    // `run` wants the stack the session was made for.
    let three = Session::new(H, W, 3, &s).unwrap();
    assert!(matches!(
        three.run(Batch::Split(&frames), &mut |_| true),
        Err(Error::Batch(_))
    ));
    assert!(Error::Batch("x".into()).to_string().contains('x'));
}

/// What the method cannot run, and what this build or machine cannot, is
/// refused before a frame is asked for — the chain first, since it is wrong
/// whatever the stack.
#[test]
fn what_cannot_run_is_refused_up_front() {
    let s = Settings {
        transformation: crate::Transformation::Bilinear,
        reference: Reference::Previous,
        ..Settings::default()
    };
    assert_eq!(Session::new(H, W, 5, &s).err(), Some(Error::Unchainable));
    assert_eq!(Session::new(H, W, 0, &s).err(), Some(Error::Unchainable));
    assert!(Error::Unchainable.to_string().contains("bilinear"));

    for backend in Backend::all() {
        let s = Settings {
            backend,
            ..Settings::default()
        };
        match backend.unavailable_reason(H, W) {
            Some(why) => {
                assert_eq!(
                    Session::new(H, W, 5, &s).err(),
                    Some(Error::Unavailable(why)),
                    "{backend:?}"
                );
                // Nothing to register needs no backend.
                assert!(Session::new(H, W, 0, &s).is_ok(), "{backend:?}");
            }
            None => assert!(Session::new(H, W, 5, &s).is_ok(), "{backend:?}"),
        }
        // And asks nothing of it, not even its batch rule, which on the GPU
        // opens the device: frames that are not there need no batches.
        for (ly, lx, n) in [(H, W, 0), (0, W, 5), (H, 0, 5)] {
            let empty = Session::new(ly, lx, n, &s).unwrap();
            assert_eq!(empty.batch_len(), 1, "{backend:?}, {n} of {lx} x {ly}");
            assert!(empty.wanted().is_none());
            assert!(empty.finish().unwrap().matrices.is_empty());
        }
    }
}

/// A batch the GPU fails — a card with no room for its level — fails the run
/// as `Error::Device`, saying why, rather than handing back its frames
/// unmoved as though they needed no correction. The reason is the run's own:
/// one an earlier caller left in the slot is not, and the slot is left empty.
/// Skipped without a usable adapter.
#[cfg(feature = "gpu")]
#[test]
fn frames_the_gpu_fails_fail_the_run() {
    use crate::gpu;
    if !gpu::available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    // Large enough that the full-resolution level goes to the device.
    let (lx, ly, n) = (192, 160, 4);
    assert!(lx * ly >= gpu::MIN_PIXELS);
    let base: Vec<f32> = (0..lx * ly)
        .map(|i| {
            let (x, y) = ((i % lx) as f32, (i / lx) as f32);
            100.0 + (x * 0.2).sin() * 30.0 + (y * 0.15).cos() * 20.0
        })
        .collect();
    let frames: Vec<Vec<f32>> = (0..n)
        .map(|t| {
            let d = 0.5 * t as f64;
            transform::warp(
                &base,
                lx,
                ly,
                &Matrix {
                    c: [d, 1.0, 0.0, 0.0, -d, 0.0, 1.0, 0.0],
                },
            )
        })
        .collect();
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let s = Settings {
        transformation: crate::Transformation::Translation,
        backend: Backend::Gpu,
        ..Settings::default()
    };

    // An earlier caller's reason, left behind, is not this run's — even in a
    // run with frames the fit gave up on: an affine fit of frames one row
    // high, which are too small to go to the device at all.
    {
        let _turn = gpu::TURN.lock().unwrap_or_else(|e| e.into_inner());
        gpu::note_trouble("an earlier caller's".to_string());
    }
    let thin: Vec<Vec<f32>> = (0..n)
        .map(|t| (0..3).map(|i| ((i * 7 + t * 3) % 5) as f32).collect())
        .collect();
    let affine = Settings {
        transformation: crate::Transformation::Affine,
        ..s
    };
    let given_up = register_stack(
        &Frames {
            ly: 1,
            lx: 3,
            frames: &thin,
        },
        &affine,
        &mut |_| true,
    )
    .expect("frames the fit gave up on are not the device's");
    assert_eq!(given_up.failed, (1..n).collect::<Vec<_>>());
    let fine = register_stack(&movie, &s, &mut |_| true).expect("the device fits them");
    assert!(fine.failed.is_empty());

    // A full card, on this thread only.
    gpu::NO_ROOM.with(|no_room| no_room.set(true));
    let out = register_stack(&movie, &s, &mut |_| true);
    gpu::NO_ROOM.with(|no_room| no_room.set(false));
    match out {
        Err(Error::Device {
            failed,
            frames,
            why,
        }) => {
            assert_eq!((failed, frames), (n - 1, n));
            assert!(why.contains("no room"), "{why}");
        }
        Err(e) => panic!("{e}"),
        Ok(r) => panic!("the run came back with {:?} unfitted", r.failed),
    }
    let e = Error::Device {
        failed: 3,
        frames: 4,
        why: "why".into(),
    };
    assert_eq!(
        e.to_string(),
        "3 of 4 frames could not be registered on the GPU: why"
    );
}

/// Progress runs from 0 to 1 over the frames fitted, and returning `false`
/// stops the run with `Cancelled` — at once, or between any two batches: the
/// call that said stop is the last. Ending in `Cancelled` is not enough to
/// show it, since the last call, at 1.0, could say it after a run that went
/// on regardless.
#[test]
fn a_run_reports_progress_and_can_be_stopped() {
    let frames = stack(6);
    let s = on_one_thread(Reference::Mean, 3, 1);
    let mut seen = Vec::new();
    Session::new(H, W, 6, &s)
        .unwrap()
        .run(Batch::Split(&frames), &mut |f| {
            seen.push(f);
            true
        })
        .unwrap();
    assert_eq!(seen.first(), Some(&0.0));
    assert_eq!(seen.last(), Some(&1.0));
    assert!(seen.windows(2).all(|p| p[0] <= p[1]), "{seen:?}");
    for stop_at in 0..seen.len() {
        let mut calls = 0;
        let out = Session::new(H, W, 6, &s)
            .unwrap()
            .run(Batch::Split(&frames), &mut |_| {
                calls += 1;
                calls <= stop_at
            });
        assert_eq!(
            out.err(),
            Some(Error::Cancelled),
            "stopped at call {stop_at}"
        );
        assert_eq!(
            calls,
            stop_at + 1,
            "the run went on after call {stop_at} said stop"
        );
    }
}

/// A window wider than the stack averages every frame over the whole stack,
/// which a window of `2n - 1` already does — so wider ones give its answers,
/// and ask for no more memory: the half-window stops at `n - 1`. A window of
/// 2^40 used to size the ring at 2^41 slots, an allocation that aborts the
/// process, and nothing upstream refuses such a window.
#[test]
fn a_window_wider_than_the_stack_is_the_whole_stack() {
    let n = 5;
    let frames = stack(n);
    let packed = frames.concat();
    let movie = Frames {
        ly: H,
        lx: W,
        frames: &frames,
    };
    for reference in Reference::all() {
        let whole = on_one_thread(reference, 2 * n - 1, 2);
        let want = register_stack(&movie, &whole, &mut |_| true).unwrap();
        for window in [usize::MAX, usize::MAX / 2 + 1, 1 << 30, 2 * n + 1] {
            let s = on_one_thread(reference, window, 2);
            let session = Session::new(H, W, n, &s).unwrap();
            assert_eq!(session.half, n - 1, "{s:?}");
            let got = session.run(Batch::Packed(&packed), &mut |_| true).unwrap();
            assert_eq!(got.matrices, want.matrices, "{s:?}, packed");
            let got = register_stack(&movie, &s, &mut |_| true).unwrap();
            assert_eq!(got.matrices, want.matrices, "{s:?}");
            assert_eq!(got.reference, want.reference, "{s:?}");
        }
    }
}

/// The batch kinds are three ways of lending the same frames: a batch of
/// borrowed slices is read as the same frames as one of buffers or one buffer
/// of them all, and a slice of the wrong length is refused as the others are.
#[test]
fn every_kind_of_batch_is_the_same_frames() {
    let n = 6;
    let frames = stack(n);
    let slices: Vec<&[f32]> = frames.iter().map(|f| &f[..]).collect();
    let packed = frames.concat();
    for reference in Reference::all() {
        let s = on_one_thread(reference, 3, 2);
        let run = |batch: Batch<'_>| {
            Session::new(H, W, n, &s)
                .unwrap()
                .run(batch, &mut |_| true)
                .unwrap()
                .matrices
        };
        let want = run(Batch::Split(&frames));
        assert_eq!(run(Batch::Slices(&slices)), want, "{s:?}");
        assert_eq!(run(Batch::Packed(&packed)), want, "{s:?}");
    }
    let short: Vec<&[f32]> = vec![&frames[0][..], &frames[1][1..]];
    let mut session = Session::new(H, W, n, &on_one_thread(Reference::First, 1, 1)).unwrap();
    assert!(matches!(
        session.feed(Batch::Slices(&short)),
        Err(Error::Batch(why)) if why.contains("frame 1")
    ));
}

/// The smoothing window is centred and odd, so it raises the signal without
/// shifting it. An even window would be off-centre by half a frame, which is
/// a drift the registration would then faithfully remove.
#[test]
fn the_moving_average_is_centred() {
    let frames: Vec<Vec<f32>> = (0..5).map(|t| vec![t as f32 * 10.0; 4]).collect();
    let smoothed = |t: usize, half: usize| {
        let mut out = vec![0.0f32; 4];
        average_into(&mut out, window(t, half, 5), &|i| &frames[i][..], false);
        out[0]
    };
    // The middle frame of a linear ramp is unchanged by a centred window, and
    // the ends are cut off at the stack's ends rather than padded.
    assert_eq!(smoothed(2, 1), 20.0);
    assert_eq!(smoothed(0, 1), 5.0);
    assert_eq!(smoothed(4, 2), 30.0);
    // A window of one changes nothing.
    assert_eq!(smoothed(3, 0), 30.0);
    // An even window is the next odd one.
    for (asked, half) in [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2)] {
        let s = on_one_thread(Reference::First, asked, 1);
        assert_eq!(Session::new(H, W, 5, &s).unwrap().half, half, "{asked}");
    }
    // And with one frame there is nothing to average.
    let s = on_one_thread(Reference::First, 5, 1);
    assert_eq!(Session::new(H, W, 1, &s).unwrap().half, 0);
}

/// The stack-wide sums are the same numbers on every core: each pixel's sum
/// still runs over the frames in order, whichever thread takes its piece.
#[test]
fn the_stack_sums_do_not_depend_on_threads() {
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
    let raw = |i: usize| &frames[i][..];
    for t in 0..7 {
        let (mut one, mut many) = (vec![0.0f32; len], vec![0.0f32; len]);
        average_into(&mut one, window(t, 2, 7), &raw, false);
        average_into(&mut many, window(t, 2, 7), &raw, true);
        assert_eq!(one, many, "frame {t} smoothed");
    }
    let (mut one, mut many) = (vec![0.0f64; len], vec![0.0f64; len]);
    for f in &frames {
        add_into(&mut one, f, false);
        add_into(&mut many, f, true);
    }
    assert_eq!(one, many, "the target's sum");
}

/// The batch rule: one frame on one thread; on every core a few frames per
/// core, but no more than a batch's memory of frames of this size; on the
/// GPU what the device holds.
#[test]
fn a_batch_is_bounded_by_memory() {
    let s = |backend| Settings {
        backend,
        ..Settings::default()
    };
    assert_eq!(batch_len(&s(Backend::SingleThread), 4096, 4096), 1);
    let small = batch_len(&s(Backend::MultiThread), 64, 64);
    assert_eq!(small, (4 * par::workers()).max(8));
    // 64 MiB a frame: four of them make the 256 MiB.
    assert_eq!(
        batch_len(&s(Backend::MultiThread), 4096, 4096),
        4.min(small)
    );
    // Never no frames at all, however large one is.
    assert_eq!(batch_len(&s(Backend::MultiThread), 16384, 8192), 1);
    #[cfg(feature = "gpu")]
    assert_eq!(
        batch_len(&s(Backend::Gpu), 192, 160),
        crate::gpu::lanes_for(192, 160).max(1)
    );
}

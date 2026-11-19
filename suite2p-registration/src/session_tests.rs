//! The session: which frames it asks for, what it does to them, and that a
//! feeder cannot get the order wrong.

use super::*;
use crate::{register, shift_frame, shift_frame_bordered, Border, Frames};

/// A field of many small features, moved along `path`, with a comb on frame
/// `t` that a `bidiphase` of `comb[t]` undoes.
fn movie(ly: usize, lx: usize, path: &[(i32, i32)], comb: &[i32]) -> Vec<Vec<f32>> {
    let mut base = vec![10.0f32; ly * lx];
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for _ in 0..(ly * lx / 120).max(8) {
        let (cy, cx) = (next() % ly, next() % lx);
        let amp = 80.0 + (next() % 120) as f32;
        for y in cy.saturating_sub(4)..(cy + 5).min(ly) {
            for x in cx.saturating_sub(4)..(cx + 5).min(lx) {
                let d2 = ((y as f32 - cy as f32).powi(2) + (x as f32 - cx as f32).powi(2)) / 4.0;
                base[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    path.iter()
        .zip(comb)
        .map(|(&(dy, dx), &c)| {
            let mut f = shift_frame(&base, ly, lx, -dy, -dx);
            bidiphase::shift(&mut f, ly, lx, -c);
            f
        })
        .collect()
}

fn rigid() -> Settings {
    Settings {
        nonrigid: false,
        spatial_taper: 5.0,
        maxregshift: 0.3,
        backend: Backend::SingleThread,
        ..Settings::default()
    }
}

/// What a streaming feeder saw of the last pass, batch by batch.
#[derive(Default)]
struct Seen {
    shifts: Vec<Shift>,
    fields: Vec<Vec<BlockShift>>,
}

/// Feed a session the way a streaming host does — reading each frame as it is
/// asked for — and keep every final batch's shifts and fields as they came.
fn streamed(frames: &[Vec<f32>], ly: usize, lx: usize, settings: &Settings) -> (Registered, Seen) {
    let mut session = Session::new(ly, lx, frames.len(), settings);
    let mut seen = Seen::default();
    while session.needs_reference() {
        let sample = session
            .reference_frames()
            .iter()
            .map(|&t| frames[t].clone())
            .collect();
        session
            .build_reference(sample, &mut |_| true)
            .expect("not cancelled");
        while let Some(batch) = session.next_batch() {
            let read: Vec<Vec<f32>> = frames[batch].to_vec();
            let measured = session.measure(&read);
            if measured.is_final() {
                seen.shifts.extend_from_slice(measured.shifts());
                if let Some(fields) = measured.fields() {
                    seen.fields.extend_from_slice(fields);
                }
            }
        }
    }
    (session.finish(), seen)
}

/// Read and build every reference the session wants, from `frames`.
fn build(session: &mut Session, frames: &[Vec<f32>]) {
    let sample = session
        .reference_frames()
        .iter()
        .map(|&t| frames[t].clone())
        .collect();
    session
        .build_reference(sample, &mut |_| true)
        .expect("not cancelled");
}

/// The reference is built from suite2p's sample —
/// `np.linspace(0, n, 1 + min(nimg_init, n), dtype=int)[:-1]` — and the
/// session asks for exactly those frames. The values are numpy's own, for
/// recordings shorter than, about, and well over `nimg_init`.
///
/// The crate and the viewer used to step by `n / nimg_init` instead, which for
/// a recording of 599 frames and the default 300 sampled all 599.
#[test]
fn the_reference_sample_is_suite2ps() {
    assert_eq!(reference_sample(10, 4), [0, 2, 5, 7]);
    assert_eq!(reference_sample(5, 300), [0, 1, 2, 3, 4]);
    assert_eq!(reference_sample(0, 300), Vec::<usize>::new());
    // `nimg_init` of 0 is taken as 1, as the crate always has.
    assert_eq!(reference_sample(7, 0), [0]);
    let long = reference_sample(1000, 300);
    assert_eq!(long.len(), 300);
    assert_eq!(long[..12], [0, 3, 6, 10, 13, 16, 20, 23, 26, 30, 33, 36]);
    let just_short = reference_sample(599, 300);
    assert_eq!(just_short.len(), 300);
    assert_eq!(just_short[295..], [589, 591, 593, 595, 597]);
    // Where numpy's floating point lands a frame below the exact quotient:
    // 50 * (114 / 100) is 56.99999999999999, and numpy truncates it.
    let odd = reference_sample(114, 100);
    assert_eq!((odd.len(), odd[50]), (100, 56));

    let s = Session::new(16, 16, 599, &Settings::default());
    assert_eq!(s.reference_frames(), &just_short[..]);
}

/// The batch is bounded by memory in the crate, for every feeder: 100 frames
/// of 1024 square are 400 MiB, so a batch is 64 of them — and since that
/// shortens the window `smooth_sigma_time` smooths across, the session says
/// so. With the smoothing off it is a scheduling detail and says nothing; at
/// 512 square the budget is not reached at all.
#[test]
fn the_batch_is_bounded_by_memory_and_says_when_that_matters() {
    let smoothing = Settings {
        smooth_sigma_time: 1.0,
        ..Settings::default()
    };
    let big = Session::new(1024, 1024, 1000, &smoothing);
    assert_eq!(big.batch_len(), 64);
    assert_eq!(pipeline::batch_len(1024, 1024, &smoothing), 64);
    let note = big.shortened_window().expect("the window is shorter");
    assert!(note.contains("100") && note.contains("64"), "{note}");

    let plain = Session::new(1024, 1024, 1000, &Settings::default());
    assert_eq!(plain.batch_len(), 64);
    assert_eq!(plain.shortened_window(), None);

    let usual = Session::new(512, 512, 1000, &smoothing);
    assert_eq!(usual.batch_len(), 100);
    assert_eq!(usual.shortened_window(), None);
    // Never none at all, however large the frame.
    assert_eq!(pipeline::batch_len(100_000, 100_000, &smoothing), 1);

    // A recording no longer than the shorter batch is one batch either way,
    // and its answer is the same: nothing to say. One frame longer, and the
    // bounded batch splits it where the asked-for one would not.
    let short = Session::new(1024, 1024, 50, &smoothing);
    assert_eq!(short.batch_len(), 64);
    assert_eq!(short.shortened_window(), None);
    assert_eq!(
        Session::new(1024, 1024, 64, &smoothing).shortened_window(),
        None
    );
    assert!(Session::new(1024, 1024, 65, &smoothing)
        .shortened_window()
        .is_some());
}

/// The batches are the session's: consecutive, `batch_len` long but the last,
/// and covering the recording once per pass.
#[test]
fn the_batches_cover_the_recording_once_per_pass() {
    let (ly, lx) = (32, 32);
    let path = [(0, 0); 7];
    let frames = movie(ly, lx, &path, &[0; 7]);
    let settings = Settings {
        batch_size: 3,
        two_step_registration: true,
        ..rigid()
    };
    let mut session = Session::new(ly, lx, frames.len(), &settings);
    assert_eq!(session.passes(), 2);
    assert_eq!(session.next_batch(), None, "no batch before a reference");
    let mut asked = Vec::new();
    while session.needs_reference() {
        let sample = session
            .reference_frames()
            .iter()
            .map(|&t| frames[t].clone())
            .collect();
        session.build_reference(sample, &mut |_| true).unwrap();
        while let Some(batch) = session.next_batch() {
            asked.push((session.pass(), batch.clone()));
            let final_pass = session.measure(&frames[batch]).is_final();
            assert_eq!(final_pass, session.pass() == 1);
        }
    }
    assert!(session.is_done());
    assert_eq!(
        asked,
        [
            (0, 0..3),
            (0, 3..6),
            (0, 6..7),
            (1, 0..3),
            (1, 3..6),
            (1, 6..7)
        ]
    );
    assert_eq!(session.finish().shifts.len(), 7);
}

/// The comb is measured on the reference sample, as suite2p measures it —
/// not on the whole recording, which is what the crate's `register` did while
/// the viewer measured the sample. Here the two disagree: the sample's frames
/// are combed by 2 and the rest, which outnumber them, by -3.
#[test]
fn the_comb_is_measured_on_the_reference_sample() {
    let (ly, lx) = (64, 64);
    let path = [(0, 0); 10];
    let mut comb = [-3; 10];
    let settings = Settings {
        nimg_init: 2,
        do_bidiphase: true,
        ..rigid()
    };
    for &t in &reference_sample(10, settings.nimg_init) {
        comb[t] = 2;
    }
    let frames = movie(ly, lx, &path, &comb);
    assert_eq!(
        bidiphase::compute(&frames, ly, lx),
        -3,
        "the fixture: the whole recording reads -3"
    );
    let out = register(
        &Frames {
            ly,
            lx,
            frames: &frames,
        },
        &settings,
        &mut |_| true,
    )
    .unwrap();
    assert_eq!(out.bidiphase, 2);
}

/// A batch is borrowed and never written, whatever the session does to it:
/// a comb to undo and a second pass to move it for.
#[test]
fn a_batch_is_never_written() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3), (-2, 0)];
    let frames = movie(ly, lx, &path, &[2; 5]);
    let settings = Settings {
        bidiphase: 2,
        two_step_registration: true,
        batch_size: 2,
        ..rigid()
    };
    let mut session = Session::new(ly, lx, frames.len(), &settings);
    while session.needs_reference() {
        let sample = session
            .reference_frames()
            .iter()
            .map(|&t| frames[t].clone())
            .collect();
        session.build_reference(sample, &mut |_| true).unwrap();
        while let Some(batch) = session.next_batch() {
            let read: Vec<Vec<f32>> = frames[batch.clone()].to_vec();
            session.measure(&read);
            assert_eq!(read, frames[batch], "measure wrote to the caller's frames");
        }
    }
    // And what it corrects into is one batch long, however many batches ran.
    assert_eq!(session.scratch.len(), 2);
}

/// Feeding a session as a streaming host does gives `register`'s answer,
/// batch by batch: the final shifts as each batch is measured are the run's.
/// Two-step, a comb, the temporal smoothing and a short batch, so every one
/// of the steps that used to be sequenced twice is in it.
#[test]
fn a_streamed_session_is_register() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3), (-2, 0), (3, 3), (0, -2)];
    let frames = movie(ly, lx, &path, &[2; 7]);
    let settings = Settings {
        do_bidiphase: true,
        two_step_registration: true,
        smooth_sigma_time: 1.0,
        batch_size: 3,
        nonrigid: true,
        block_size: [24, 24],
        ..rigid()
    };
    let whole = register(
        &Frames {
            ly,
            lx,
            frames: &frames,
        },
        &settings,
        &mut |_| true,
    )
    .unwrap();
    let (fed, seen) = streamed(&frames, ly, lx, &settings);
    assert_eq!(whole.bidiphase, 2);
    assert_eq!(seen.shifts, whole.shifts);
    assert_eq!(fed.shifts, whole.shifts);
    assert_eq!(fed.reference, whole.reference);
    assert_eq!(fed.bad_frames, whole.bad_frames);
    // The block field too, a batch at a time: the streamed session held each
    // batch's while it was moved, and `register`, which moves frames after,
    // kept every one.
    let (a, b) = (whole.nonrigid.unwrap(), fed.nonrigid.unwrap());
    assert_eq!(seen.fields, a.shifts);
    assert_eq!(a.blocks.blocks.len(), b.blocks.blocks.len());
}

/// Planes go in as recorded, everywhere: the batch's `apply`, the result's
/// `apply` and its `apply_batch` all undo the comb first and agree with each
/// other, cropped or not.
#[test]
fn every_apply_takes_planes_as_recorded() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3)];
    let frames = movie(ly, lx, &path, &[3; 4]);
    for border in [Border::Native, Border::Constant, Border::Crop] {
        for nonrigid in [false, true] {
            let settings = Settings {
                bidiphase: 3,
                nonrigid,
                block_size: [24, 24],
                border,
                ..rigid()
            };
            let mut session = Session::new(ly, lx, frames.len(), &settings);
            // The result moves frames after `finish` too, so it needs every
            // frame's field (a crop keeps them regardless).
            session.keep_fields();
            build(&mut session, &frames);
            let batch = session.next_batch().unwrap();
            assert_eq!(batch, 0..4, "one batch");
            let mut during = frames.clone();
            let at: Vec<usize> = (0..4).collect();
            session.measure(&frames).apply(&mut during, &at);
            let out = session.finish();

            let mut after = frames.clone();
            out.apply_batch(&mut after, ly, lx, &at).unwrap();
            for t in 0..4 {
                let one = out.apply(&frames[t], ly, lx, t).unwrap();
                assert_eq!(after[t], one, "{border:?} non-rigid {nonrigid}: frame {t}");
                if border != Border::Crop {
                    assert_eq!(during[t], one, "{border:?} non-rigid {nonrigid}: frame {t}");
                }
            }
            if !nonrigid && border != Border::Crop {
                // And what that is: the comb undone, then the shift.
                let mut combless = frames[1].clone();
                bidiphase::shift(&mut combless, ly, lx, 3);
                let s = out.shifts[1];
                assert_eq!(
                    after[1],
                    shift_frame_bordered(&combless, ly, lx, s.dy, s.dx, border)
                );
            }
        }
    }
}

/// A pass before the last has no shifts to apply, and moving planes by them
/// is refused rather than done.
#[test]
#[should_panic(expected = "no shifts to apply")]
fn the_first_of_two_passes_cannot_be_applied() {
    let (ly, lx) = (32, 32);
    let frames = movie(ly, lx, &[(0, 0), (1, 1)], &[0, 0]);
    let settings = Settings {
        two_step_registration: true,
        ..rigid()
    };
    let mut session = Session::new(ly, lx, 2, &settings);
    let sample = frames.clone();
    session.build_reference(sample, &mut |_| true).unwrap();
    let mut planes = frames.clone();
    session.measure(&frames).apply(&mut planes, &[0, 1]);
}

/// The feeder does not choose the batch: one of a different length is
/// refused, since where the batches fall is part of the answer. (Which frames
/// they are is the feeder's to get right: nothing in a frame says which it
/// is, so the right count of the wrong frames is not something a session can
/// see.)
#[test]
#[should_panic(expected = "as many frames as the batch next_batch names")]
fn a_batch_of_the_wrong_length_is_refused() {
    let (ly, lx) = (32, 32);
    let frames = movie(ly, lx, &[(0, 0); 5], &[0; 5]);
    let settings = Settings {
        batch_size: 2,
        ..rigid()
    };
    let mut session = Session::new(ly, lx, 5, &settings);
    session
        .build_reference(frames.clone(), &mut |_| true)
        .unwrap();
    session.measure(&frames[..3]);
}

/// A recording with no frames is a registration of none, not a panic.
#[test]
fn no_frames_is_an_empty_registration() {
    let settings = Settings {
        two_step_registration: true,
        ..rigid()
    };
    let mut session = Session::new(16, 16, 0, &settings);
    let mut references = 0;
    while session.needs_reference() {
        assert!(session.reference_frames().is_empty());
        session.build_reference(Vec::new(), &mut |_| true).unwrap();
        references += 1;
        assert_eq!(session.next_batch(), None);
    }
    assert_eq!(references, 2);
    let out = session.finish();
    assert!(out.shifts.is_empty() && out.bad_frames.is_empty());
}

/// Stopped while the reference is built, the build says so — and the session
/// then wants nothing more, so a feeder's `while needs_reference()` loop ends
/// even when it dropped the `None`, rather than reading the sample and
/// building again for as long as its callback keeps saying stop.
#[test]
fn a_stopped_reference_is_none() {
    let (ly, lx) = (32, 32);
    let frames = movie(ly, lx, &[(0, 0); 4], &[0; 4]);
    let mut session = Session::new(ly, lx, 4, &rigid());
    let mut builds = 0;
    while session.needs_reference() {
        builds += 1;
        assert!(
            builds < 3,
            "a stopped session asked for its reference again"
        );
        let stopped = session.build_reference(frames.clone(), &mut |_| false);
        assert!(stopped.is_none());
    }
    assert_eq!(builds, 1);
    assert_eq!(session.next_batch(), None);
    assert!(!session.is_done());
}

/// And there is no registration to finish.
#[test]
#[should_panic(expected = "stopped while its reference was built")]
fn a_stopped_session_cannot_finish() {
    let (ly, lx) = (32, 32);
    let frames = movie(ly, lx, &[(0, 0); 4], &[0; 4]);
    let mut session = Session::new(ly, lx, 4, &rigid());
    let _ = session.build_reference(frames, &mut |_| false);
    session.finish();
}

/// A non-rigid session holds one batch's block fields — the ones its feeder
/// is moving that batch's planes by — unless asked to keep them all. A field
/// per frame is memory that grows with the recording, and a streaming feeder
/// that moves each batch as it goes has no use for the rest. The result
/// still names the grid, so the block count can be reported.
///
/// It used to keep every frame's field to the end, kept or not.
#[test]
fn a_session_holds_one_batch_of_block_fields() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3), (-2, 0), (3, 3), (0, -2)];
    let frames = movie(ly, lx, &path, &[0; 7]);
    let settings = Settings {
        nonrigid: true,
        block_size: [24, 24],
        batch_size: 2,
        ..rigid()
    };
    let grid = nonrigid::make_blocks(ly, lx, settings.block_size, settings.subpixel)
        .blocks
        .len();
    for keep in [false, true] {
        let mut session = Session::new(ly, lx, frames.len(), &settings);
        if keep {
            session.keep_fields();
        }
        build(&mut session, &frames);
        let mut seen = Vec::new();
        while let Some(batch) = session.next_batch() {
            let fields = session
                .measure(&frames[batch.clone()])
                .fields()
                .unwrap()
                .to_vec();
            assert_eq!(fields.len(), batch.len());
            seen.extend_from_slice(&fields);
            let held = if keep { batch.end } else { batch.len() };
            assert_eq!(session.fields.len(), held, "kept {keep}, batch {batch:?}");
            for t in 0..batch.end {
                assert_eq!(
                    session.field(t).is_some(),
                    keep || batch.contains(&t),
                    "kept {keep}: frame {t}'s field after batch {batch:?}"
                );
            }
        }
        let nr = session.finish().nonrigid.expect("a non-rigid run");
        assert_eq!(nr.blocks.blocks.len(), grid, "the grid, named either way");
        if keep {
            assert_eq!(nr.shifts, seen);
        } else {
            assert!(nr.shifts.is_empty());
        }
    }

    // A crop's area is found from every frame's field, so a crop keeps them.
    let crop = Settings {
        border: Border::Crop,
        ..settings
    };
    let mut session = Session::new(ly, lx, frames.len(), &crop);
    build(&mut session, &frames);
    while let Some(batch) = session.next_batch() {
        session.measure(&frames[batch]);
    }
    assert_eq!(
        session.finish().nonrigid.unwrap().shifts.len(),
        frames.len()
    );
}

/// A registration that kept no block field will not move a frame afterwards:
/// moved rigidly where it should have been warped, the frame would look
/// registered and not be.
#[test]
#[should_panic(expected = "kept no block field")]
fn a_registration_without_its_fields_will_not_move_a_frame() {
    let (ly, lx) = (48, 48);
    let frames = movie(ly, lx, &[(0, 0), (2, -1), (-3, 2)], &[0; 3]);
    let settings = Settings {
        nonrigid: true,
        block_size: [24, 24],
        ..rigid()
    };
    let mut session = Session::new(ly, lx, frames.len(), &settings);
    build(&mut session, &frames);
    session.measure(&frames);
    let _ = session.finish().apply(&frames[0], ly, lx, 0);
}

/// A feeder that owns its batch measures it in place: the same shifts and
/// fields as `measure`, the frames handed back bit for bit as recorded, and
/// no batch-sized copy — the comb undone where the frames are, keeping the
/// few columns it slides over, and a second pass's roll reversed after.
///
/// `measure` borrows, so a comb or a second pass has it copy the batch, and
/// a feeder that had read the batch into buffers of its own held it twice.
#[test]
fn measuring_in_place_is_measuring() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3), (-2, 0), (3, 3), (0, -2)];
    for (comb, two_step, nonrigid) in [(2, false, false), (-3, true, true), (0, true, false)] {
        let frames = movie(ly, lx, &path, &[comb; 7]);
        let settings = Settings {
            bidiphase: comb,
            two_step_registration: two_step,
            nonrigid,
            block_size: [24, 24],
            batch_size: 3,
            smooth_sigma_time: 1.0,
            ..rigid()
        };
        let case = format!("comb {comb}, two-step {two_step}, non-rigid {nonrigid}");
        let mut borrowed = Session::new(ly, lx, frames.len(), &settings);
        let mut owned = Session::new(ly, lx, frames.len(), &settings);
        while borrowed.needs_reference() {
            assert!(owned.needs_reference(), "{case}");
            build(&mut borrowed, &frames);
            build(&mut owned, &frames);
            while let Some(batch) = borrowed.next_batch() {
                assert_eq!(owned.next_batch(), Some(batch.clone()), "{case}");
                let a = borrowed.measure(&frames[batch.clone()]);
                let (shifts, fields) = (a.shifts().to_vec(), a.fields().map(<[_]>::to_vec));
                let mut read = frames[batch.clone()].to_vec();
                let b = owned.measure_in_place(&mut read);
                assert_eq!(b.shifts(), &shifts[..], "{case}: batch {batch:?}");
                assert_eq!(
                    b.fields().map(<[_]>::to_vec),
                    fields,
                    "{case}: batch {batch:?}"
                );
                assert!(
                    read == frames[batch.clone()],
                    "{case}: batch {batch:?} came back changed"
                );
                assert_eq!(owned.scratch.capacity(), 0, "{case}: a copy of the batch");
            }
        }
        assert!(owned.is_done(), "{case}");
        // The borrowing session did copy, for the same batches: this is the
        // case the in-place path is for.
        assert_eq!(borrowed.scratch.len(), 3, "{case}");
        let (a, b) = (borrowed.finish(), owned.finish());
        assert_eq!(a.shifts, b.shifts, "{case}");
        assert_eq!(a.reference, b.reference, "{case}");
        assert_eq!(a.bad_frames, b.bad_frames, "{case}");
    }
}

/// A two-step run's second reference is built from the sample with the comb
/// undone and each frame moved by the first pass's shift — the frames as the
/// first pass left them, which is what makes a second pass worth running.
/// Built here from the public stages, with a sample smaller than the
/// recording: with every frame in the sample the refinement re-aligns them to
/// the same reference either way, and a sample left where it was would go
/// unnoticed.
#[test]
fn the_second_reference_is_the_sample_moved_by_the_first_pass() {
    let (ly, lx) = (64, 64);
    let path: Vec<(i32, i32)> = (0..40)
        .map(|t: i32| ((t * 7 % 11) - 5, (t * 5 % 9) - 4))
        .collect();
    // Windows onto a larger field rather than rolls of one frame: the motion
    // brings new tissue in at the edges, as a real one does. Rolls of a
    // single frame refine to the same reference wherever they start, so
    // they could not tell a moved sample from one left where it was.
    let (my, mx) = (ly + 12, lx + 12);
    let field = &movie(my, mx, &[(0, 0)], &[0])[0];
    let frames: Vec<Vec<f32>> = path
        .iter()
        .map(|&(dy, dx)| {
            let (y0, x0) = ((6 + dy) as usize, (6 + dx) as usize);
            let mut f: Vec<f32> = (0..ly)
                .flat_map(|y| {
                    field[(y0 + y) * mx + x0..(y0 + y) * mx + x0 + lx]
                        .iter()
                        .copied()
                })
                .collect();
            bidiphase::shift(&mut f, ly, lx, -2);
            f
        })
        .collect();
    let settings = Settings {
        bidiphase: 2,
        two_step_registration: true,
        nimg_init: 13,
        batch_size: 7,
        ..rigid()
    };
    let mut session = Session::new(ly, lx, frames.len(), &settings);
    let mut first = None;
    while session.needs_reference() {
        if session.pass() == 0 && !session.shifts().is_empty() {
            // The first pass is measured; the next build starts the second.
            first = Some(session.shifts().to_vec());
        }
        build(&mut session, &frames);
        while let Some(batch) = session.next_batch() {
            session.measure(&frames[batch]);
        }
    }
    let first = first.expect("a first pass");
    let sample: Vec<Vec<f32>> = session
        .reference_frames()
        .iter()
        .map(|&t| {
            let mut f = frames[t].clone();
            bidiphase::shift(&mut f, ly, lx, 2);
            f
        })
        .collect();
    let moved: Vec<Vec<f32>> = sample
        .iter()
        .zip(session.reference_frames())
        .map(|(f, &t)| shift_frame(f, ly, lx, first[t].dy, first[t].dx))
        .collect();
    let reference = |frames: &[Vec<f32>]| {
        crate::compute_reference(&Frames { ly, lx, frames }, &settings, &mut |_| true).unwrap()
    };
    let want = reference(&moved);
    assert_ne!(
        want,
        reference(&sample),
        "the fixture: moving the sample changes its reference"
    );
    assert_eq!(session.finish().reference, want);
}

/// Everything `Measured` says, the session says too, with nothing borrowed
/// from one call to the next — what a caller across a foreign function
/// interface needs, which cannot hold a `Measured` until it has the planes.
#[test]
fn a_feeder_needs_no_measured() {
    let (ly, lx) = (48, 48);
    let path = [(0, 0), (2, -1), (-3, 2), (1, 3), (-2, 0)];
    let frames = movie(ly, lx, &path, &[3; 5]);
    let settings = Settings {
        bidiphase: 3,
        nonrigid: true,
        block_size: [24, 24],
        batch_size: 2,
        ..rigid()
    };
    let mut with = Session::new(ly, lx, frames.len(), &settings);
    let mut without = Session::new(ly, lx, frames.len(), &settings);
    build(&mut with, &frames);
    build(&mut without, &frames);
    while let Some(batch) = with.next_batch() {
        let at: Vec<usize> = batch.clone().collect();
        let measured = with.measure(&frames[batch.clone()]);
        let mut a = frames[batch.clone()].to_vec();
        measured.apply(&mut a, &at);

        let _ = without.measure(&frames[batch.clone()]);
        assert_eq!(&without.shifts()[batch.clone()], measured.shifts());
        for (k, t) in batch.clone().enumerate() {
            assert_eq!(without.field(t), Some(&measured.fields().unwrap()[k][..]));
        }
        let mut b = frames[batch.clone()].to_vec();
        without.apply(&mut b, &at);
        assert!(a == b, "batch {batch:?}");
    }
    assert!(without.blocks().unwrap().blocks.len() > 1);
}

/// Every apply refuses a frame it has no correction for, and a count of
/// planes that does not match the frames named, before it moves a plane. One
/// used to undo the comb and move by nothing while another left the plane
/// combed, and a zip dropped the planes past the shorter list.
#[test]
fn every_apply_refuses_what_it_has_no_correction_for() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let (ly, lx) = (32, 32);
    let frames = movie(ly, lx, &[(0, 0), (1, -1), (-1, 2)], &[2; 3]);
    let settings = Settings {
        bidiphase: 2,
        batch_size: 2,
        ..rigid()
    };
    let refused = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f)).is_err();
    let mut session = Session::new(ly, lx, 3, &settings);
    build(&mut session, &frames);
    let first = session.next_batch().unwrap();
    session.measure(&frames[first]);
    let planes = vec![frames[0].clone(), frames[2].clone()];
    // Frame 2 is in the next batch: not measured yet.
    assert!(refused(&|| session.apply(&mut planes.clone(), &[0, 2])));
    assert!(refused(&|| session.apply(&mut planes.clone(), &[0])));
    let rest = session.next_batch().unwrap();
    session.measure(&frames[rest]);
    let out = session.finish();
    assert!(refused(&|| drop(out.apply(&frames[0], ly, lx, 3))));
    assert!(refused(&|| {
        let _ = out.apply_batch(&mut planes.clone(), ly, lx, &[0, 3]);
    }));
    assert!(refused(&|| {
        let _ = out.apply_batch(&mut planes.clone(), ly, lx, &[0]);
    }));
    // What is refused is left as it was.
    let mut kept = planes.clone();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        out.apply_batch(&mut kept, ly, lx, &[0, 3])
    }));
    assert!(kept == planes);
}

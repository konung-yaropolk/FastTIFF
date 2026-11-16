//! The session against the driver it replaced.
//!
//! `register_stack` used to take the whole movie, smooth all of it into a
//! second movie, build the target from that, and fit the pairs in batches.
//! That code is kept below verbatim, as scaffolding — it is the oracle, not
//! something to maintain — and the session is held to it **bit for bit** on
//! synthetic stacks: every reference mode, windows from none to wider than the
//! stack, several target lengths, both CPU backends, fed whole and fed in
//! batches of every awkward size, and on the GPU where there is one. The
//! session is free to batch differently; what it may not do is answer
//! differently.

use super::*;
use crate::{optimize, register_stack, transform, Frames};

// ------------------------------------------------- the old driver, verbatim

const OLD_PIXELS_PER_CHUNK: usize = 4096;

fn old_mean_of(frames: &[Vec<f32>], len: usize, parallel: bool) -> Vec<f32> {
    let take = len.clamp(1, frames.len().max(1));
    let n = frames.first().map_or(0, |f| f.len());
    let mut out = vec![0.0f32; n];
    par::for_each_chunk_mut(&mut out, OLD_PIXELS_PER_CHUNK, parallel, |i, piece| {
        let base = i * OLD_PIXELS_PER_CHUNK;
        let mut acc = vec![0.0f64; piece.len()];
        for f in frames.iter().take(take) {
            for (a, &v) in acc.iter_mut().zip(&f[base..base + piece.len()]) {
                *a += v as f64;
            }
        }
        for (o, a) in piece.iter_mut().zip(acc) {
            *o = (a / take as f64) as f32;
        }
    });
    out
}

fn old_moving_average(frames: &[Vec<f32>], window: usize, parallel: bool) -> Vec<Vec<f32>> {
    let w = window.max(1) | 1;
    if w == 1 || frames.len() < 2 {
        return frames.to_vec();
    }
    let half = (w / 2) as isize;
    let n = frames.len() as isize;
    let times: Vec<usize> = (0..frames.len()).collect();
    par::map_in_order(&times, parallel, |&t| {
        let t = t as isize;
        let lo = (t - half).max(0);
        let hi = (t + half).min(n - 1);
        let count = (hi - lo + 1) as f64;
        let len = frames[0].len();
        let mut acc = vec![0.0f64; len];
        for f in frames.iter().take((hi + 1) as usize).skip(lo as usize) {
            for (a, &v) in acc.iter_mut().zip(f.iter()) {
                *a += v as f64;
            }
        }
        acc.iter().map(|&v| (v / count) as f32).collect()
    })
}

fn old_batch_size(settings: &Settings, lx: usize, ly: usize) -> usize {
    match settings.backend {
        #[cfg(feature = "gpu")]
        Backend::Gpu => crate::gpu::lanes_for(lx, ly).max(1),
        Backend::SingleThread => 1,
        _ => {
            let _ = (lx, ly);
            (4 * par::workers()).max(8)
        }
    }
}

/// `register_stack` as it was, `Frames` spelled out.
fn old_register_stack(
    ly: usize,
    lx: usize,
    frames: &[Vec<f32>],
    settings: &Settings,
) -> Option<Registered> {
    let n = frames.len();
    let threaded = settings.backend.threaded();
    if n == 0 || ly == 0 || lx == 0 {
        return Some(Registered {
            matrices: Vec::new(),
            reference: None,
            failed: Vec::new(),
            threaded,
            border: Border::default(),
            area: Default::default(),
        });
    }
    if settings.reference == Reference::Previous
        && settings.transformation == Transformation::Bilinear
    {
        return None;
    }

    let measured = old_moving_average(frames, settings.moving_average, threaded);
    let mut matrices = vec![Matrix::identity(); n];
    let mut failed = Vec::new();

    let reference = match settings.reference {
        Reference::First => Some(old_mean_of(&measured, settings.n_frames.max(1), threaded)),
        Reference::Mean => Some(old_mean_of(&measured, n, threaded)),
        Reference::Previous => None,
    };
    let start = usize::from(settings.reference != Reference::Mean);
    let targets: Vec<usize> = (start..n).collect();
    let pairs: Vec<(&[f32], &[f32])> = targets
        .iter()
        .map(|&t| {
            let against = match &reference {
                Some(r) => &r[..],
                None => &measured[t - 1][..],
            };
            (against, &measured[t][..])
        })
        .collect();

    let per = old_batch_size(settings, lx, ly);
    let parallel = settings.backend.parallel_for(settings.reference);
    let mut fits: Vec<Option<optimize::Fit>> = Vec::with_capacity(pairs.len());
    for chunk in pairs.chunks(per) {
        fits.extend(optimize::register_pairs(chunk, lx, ly, settings, parallel));
    }

    match settings.reference {
        Reference::Previous => {
            for (&t, fit) in targets.iter().zip(fits) {
                match fit {
                    Some(f) => {
                        matrices[t] = matrices[t - 1].compose(&f.matrix).unwrap_or(f.matrix);
                    }
                    None => {
                        failed.push(t);
                        matrices[t] = matrices[t - 1];
                    }
                }
            }
        }
        Reference::First | Reference::Mean => {
            for (&t, fit) in targets.iter().zip(fits) {
                match fit {
                    Some(f) => matrices[t] = f.matrix,
                    None => failed.push(t),
                }
            }
        }
    }
    Some(Registered {
        matrices,
        reference,
        failed,
        threaded,
        border: Border::default(),
        area: Default::default(),
    })
}

// --------------------------------------------------------------- the stacks

/// A textured stack that drifts and turns a little more each frame, with a
/// frame-to-frame flicker so a moving average has something to change.
fn wandering(w: usize, h: usize, n: usize) -> Vec<Vec<f32>> {
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
            let d = 0.9 * t as f64 - 0.07 * (t * t) as f64;
            let a = 0.004 * t as f64;
            let (s, c) = (a.sin(), a.cos());
            let m = Matrix {
                c: [d, c, -s, 0.0, 0.6 * d, s, c, 0.0],
            };
            let flicker = 1.0 + 0.03 * ((t * 7 % 5) as f32 - 2.0);
            transform::warp(&base, w, h, &m)
                .into_iter()
                .map(|v| v * flicker)
                .collect()
        })
        .collect()
}

fn settings(
    transformation: Transformation,
    reference: Reference,
    window: usize,
    n_frames: usize,
    backend: Backend,
) -> Settings {
    Settings {
        transformation,
        reference,
        moving_average: window,
        n_frames,
        backend,
        ..Settings::default()
    }
}

/// Feed `frames` to a session in batches of the sizes `sizes` cycles through,
/// each capped at what is left of its pass, and each batch in the next of the
/// three shapes a batch can take.
fn fed_in(ly: usize, lx: usize, frames: &[Vec<f32>], s: &Settings, sizes: &[usize]) -> Registered {
    let mut session = Session::new(ly, lx, frames.len(), s).expect("a session");
    let mut k = 0;
    while let Some(wanted) = session.wanted() {
        let left = session.stages[0].reads.end - wanted.frames.start;
        let size = sizes[k % sizes.len()].clamp(1, left);
        let from = wanted.frames.start;
        let these = &frames[from..from + size];
        let packed: Vec<f32>;
        let slices: Vec<&[f32]>;
        let batch = match k % 3 {
            0 => Batch::Split(these),
            1 => {
                slices = these.iter().map(|f| &f[..]).collect();
                Batch::Slices(&slices)
            }
            _ => {
                packed = these.concat();
                Batch::Packed(&packed)
            }
        };
        k += 1;
        session.feed(batch).expect("the frames it asked for");
    }
    session.finish().expect("every frame fed")
}

/// The two results, field by field, to the bit.
fn same(want: &Registered, got: &Registered, what: &str) {
    assert_eq!(got.matrices, want.matrices, "{what}: matrices");
    assert_eq!(got.failed, want.failed, "{what}: failed frames");
    assert_eq!(got.threaded, want.threaded, "{what}: threaded");
    assert_eq!(got.border, want.border, "{what}: border");
    match (&want.reference, &got.reference) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(b), bits(a), "{what}: reference");
        }
        _ => panic!(
            "{what}: reference {:?} vs {:?}",
            want.reference.is_some(),
            got.reference.is_some()
        ),
    }
}

/// The session against the old driver: fed from memory as `register_stack`
/// feeds it — on one thread that is a frame at a time — and in batches of
/// each of `cuts`, which never line up with a window or a pass; and, when
/// `packed`, from one buffer as the C interface feeds it.
fn agrees(
    ly: usize,
    lx: usize,
    frames: &[Vec<f32>],
    s: &Settings,
    cuts: &[&[usize]],
    packed: bool,
) {
    let what = format!("{s:?}, {} frames", frames.len());
    let want = old_register_stack(ly, lx, frames, s).expect("the old driver ran");
    let split =
        register_stack(&Frames { ly, lx, frames }, s, &mut |_| true).expect("register_stack");
    same(&want, &split, &format!("{what}, register_stack"));
    if packed {
        let all: Vec<f32> = frames.concat();
        let got = Session::new(ly, lx, frames.len(), s)
            .unwrap()
            .run(Batch::Packed(&all), &mut |_| true)
            .expect("packed");
        same(&want, &got, &format!("{what}, packed"));
    }
    for sizes in cuts {
        same(
            &want,
            &fed_in(ly, lx, frames, s, sizes),
            &format!("{what}, fed in {sizes:?}"),
        );
    }
}

/// Batch sizes that cut windows and passes everywhere.
const CUTS: &[&[usize]] = &[&[2, 3], &[5, 1, 4]];

const W: usize = 32;
const H: usize = 28;

/// Every reference mode, every window from none to wider than the stack,
/// every target length — translation, on one thread.
#[test]
fn every_mode_and_window_matches_the_old_driver() {
    let frames = wandering(W, H, 7);
    for reference in Reference::all() {
        for window in [1, 2, 5, 31] {
            let lengths: &[usize] = if reference == Reference::First {
                &[1, 3, 50]
            } else {
                &[1]
            };
            for &n_frames in lengths {
                let s = settings(
                    Transformation::Translation,
                    reference,
                    window,
                    n_frames,
                    Backend::SingleThread,
                );
                agrees(H, W, &frames, &s, CUTS, window == 5);
            }
        }
    }
}

/// The models with more freedom, which take other paths through the fit, and
/// bilinear where it is allowed.
#[test]
fn every_model_matches_the_old_driver() {
    let frames = wandering(W, H, 5);
    for model in Transformation::all() {
        for reference in Reference::all() {
            if model == Transformation::Bilinear && reference == Reference::Previous {
                continue;
            }
            let s = settings(model, reference, 3, 2, Backend::SingleThread);
            agrees(H, W, &frames, &s, &CUTS[1..], false);
        }
    }
}

/// On every core, where a batch is many frames: smoothed and fitted side by
/// side, each frame's sums spread over the threads, and cut where the
/// batches fall.
#[test]
fn every_mode_matches_the_old_driver_on_every_core() {
    if !Backend::MultiThread.available() {
        return;
    }
    let frames = wandering(W, H, 12);
    for reference in Reference::all() {
        for window in [1, 5] {
            let s = settings(
                Transformation::Translation,
                reference,
                window,
                4,
                Backend::MultiThread,
            );
            agrees(H, W, &frames, &s, &CUTS[1..], window == 5);
        }
    }
}

/// The edges of the method: one frame, two frames, a stack with no pixels,
/// no frames at all.
#[test]
fn short_and_empty_stacks_match_the_old_driver() {
    for n in [1, 2] {
        let frames = wandering(W, H, n);
        for reference in Reference::all() {
            for window in [1, 3] {
                let s = settings(
                    Transformation::RigidBody,
                    reference,
                    window,
                    3,
                    Backend::SingleThread,
                );
                agrees(H, W, &frames, &s, CUTS, true);
            }
        }
    }
    let s = Settings {
        backend: Backend::SingleThread,
        ..Settings::default()
    };
    let none = register_stack(
        &Frames {
            ly: H,
            lx: W,
            frames: &[],
        },
        &s,
        &mut |_| true,
    )
    .unwrap();
    same(
        &old_register_stack(H, W, &[], &s).unwrap(),
        &none,
        "no frames",
    );
    let flat = vec![Vec::new(); 3];
    let empty = register_stack(
        &Frames {
            ly: 0,
            lx: W,
            frames: &flat,
        },
        &s,
        &mut |_| true,
    )
    .unwrap();
    same(
        &old_register_stack(0, W, &flat, &s).unwrap(),
        &empty,
        "no rows",
    );
}

/// On the graphics card, where the old driver batched pairs by the device's
/// lanes and the session batches frames: each lane is scored on its own, so
/// the answers are the old ones to the bit — in every mode, smoothed or not,
/// and across more than one dispatch's worth of frames. Skipped without a
/// usable adapter.
#[cfg(feature = "gpu")]
#[test]
fn the_gpu_path_matches_the_old_driver() {
    if !crate::gpu::available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    // Large enough that the full-resolution level goes to the device.
    let (w, h) = (192, 160);
    assert!(w * h >= crate::gpu::MIN_PIXELS);
    let frames = wandering(w, h, 7);
    for reference in Reference::all() {
        for window in [1, 3] {
            let s = settings(
                Transformation::RigidBody,
                reference,
                window,
                2,
                Backend::Gpu,
            );
            let want = old_register_stack(h, w, &frames, &s).unwrap();
            assert!(want.failed.is_empty(), "{s:?}: the device failed frames");
            let got = register_stack(
                &Frames {
                    ly: h,
                    lx: w,
                    frames: &frames,
                },
                &s,
                &mut |_| true,
            )
            .unwrap();
            same(&want, &got, &format!("{s:?}"));
            same(
                &want,
                &fed_in(h, w, &frames, &s, &[2, 3]),
                &format!("{s:?}, fed in [2, 3]"),
            );
        }
    }
    // More frames than one dispatch takes, so the old driver and the session
    // both split the stack, at different places.
    let lanes = crate::gpu::lanes_for(w, h);
    let frames = wandering(w, h, lanes + 5);
    let s = settings(
        Transformation::Translation,
        Reference::Previous,
        1,
        1,
        Backend::Gpu,
    );
    let want = old_register_stack(h, w, &frames, &s).unwrap();
    let got = register_stack(
        &Frames {
            ly: h,
            lx: w,
            frames: &frames,
        },
        &s,
        &mut |_| true,
    )
    .unwrap();
    same(&want, &got, "previous, past one dispatch");
}

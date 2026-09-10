//! The whole pipeline, on a movie with a known motion path.

use super::*;

fn blobs(ly: usize, lx: usize) -> Vec<f32> {
    let mut f = vec![10.0f32; ly * lx];
    for (cy, cx, amp) in [
        (20usize, 24usize, 200.0f32),
        (40, 44, 150.0),
        (28, 12, 120.0),
    ] {
        for y in 0..ly {
            for x in 0..lx {
                let d2 = ((y as f32 - cy as f32).powi(2) + (x as f32 - cx as f32).powi(2)) / 8.0;
                f[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    f
}

/// A movie of one field wandering along a known path.
fn wandering(ly: usize, lx: usize, path: &[(i32, i32)]) -> Vec<Vec<f32>> {
    let base = blobs(ly, lx);
    path.iter()
        .map(|&(dy, dx)| shift_frame(&base, ly, lx, -dy, -dx))
        .collect()
}

#[test]
fn suite2ps_defaults_are_the_defaults() {
    // Read off suite2p's `default_ops`; a drift here is a drift away from the
    // numbers published results were produced with.
    let s = Settings::default();
    assert!(!s.align_by_chan2);
    assert_eq!(s.nimg_init, 300);
    assert_eq!(s.maxregshift, 0.1);
    assert!(!s.do_bidiphase);
    assert_eq!(s.bidiphase, 0);
    assert_eq!(s.batch_size, 100);
    assert!(!s.nonrigid);
    assert_eq!(s.maxregshift_nr, 10.0);
    assert_eq!(s.block_size, [64, 64]);
    assert_eq!(s.smooth_sigma_time, 0.0);
    assert_eq!(s.smooth_sigma, 1.15);
    assert_eq!(s.spatial_taper, 50.0);
    assert_eq!(s.th_badframes, 1.0);
    assert!(s.norm_frames);
    assert_eq!(s.snr_thresh, 1.25);
    assert_eq!(s.subpixel, 10);
    assert!(!s.two_step_registration);
    // Not a suite2p option: where it runs. Multi-thread by default, because a
    // single thread is a diagnostic rather than a choice anyone wants.
    assert_eq!(s.backend, Backend::MultiThread);
}

/// The initial reference is an average of the frames that agree with each
/// other, so on a still movie it looks like the movie.
#[test]
fn the_initial_reference_resembles_a_still_movie() {
    let (ly, lx) = (48, 48);
    let base = blobs(ly, lx);
    let frames: Vec<Vec<f32>> = (0..12).map(|_| base.clone()).collect();
    let r = pick_initial_reference(&frames, ly, lx);

    // Mean-subtracted, so compare shape rather than level: the brightest pixel
    // of the reference must be where the brightest blob is.
    let arg_max = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(arg_max(&r), arg_max(&base));
}

#[test]
fn a_single_frame_is_its_own_reference() {
    let (ly, lx) = (16, 16);
    let f = blobs(ly, lx);
    assert_eq!(pick_initial_reference(std::slice::from_ref(&f), ly, lx), f);
}

/// The headline: a movie with a known motion path has that path measured back.
#[test]
fn a_known_motion_path_is_recovered() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (2, -1), (-3, 2), (1, 4), (-2, -2), (0, 3)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(&movie, &Settings::default(), &mut |_| true).expect("not cancelled");

    assert_eq!(out.shifts.len(), path.len());
    // The reference lands wherever the recentring puts it, so what is fixed is
    // the motion *between* frames, not its absolute origin. Compare each shift
    // against the first.
    let base = (out.shifts[0].dy - path[0].0, out.shifts[0].dx - path[0].1);
    for (i, (s, want)) in out.shifts.iter().zip(&path).enumerate() {
        assert_eq!(
            (s.dy - base.0, s.dx - base.1),
            *want,
            "frame {i}: measured {s:?}, path said {want:?}"
        );
    }
}

/// And applying those shifts genuinely stabilises the movie: every registered
/// frame agrees with every other, away from the wrapped border.
#[test]
fn applying_the_shifts_stabilises_the_movie() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (3, -2), (-4, 1), (2, 3)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(&movie, &Settings::default(), &mut |_| true).expect("not cancelled");

    let fixed: Vec<Vec<f32>> = frames
        .iter()
        .zip(&out.shifts)
        .map(|(f, s)| shift_frame(f, ly, lx, s.dy, s.dx))
        .collect();

    for y in 12..ly - 12 {
        for x in 12..lx - 12 {
            let first = fixed[0][y * lx + x];
            for (i, f) in fixed.iter().enumerate().skip(1) {
                assert!(
                    (f[y * lx + x] - first).abs() < 1e-2,
                    "frame {i} at ({y},{x}): {} vs {first}",
                    f[y * lx + x]
                );
            }
        }
    }
}

/// A still movie needs no correction, and must not invent one.
#[test]
fn a_still_movie_gets_no_shifts() {
    let (ly, lx) = (48, 48);
    let base = blobs(ly, lx);
    let frames: Vec<Vec<f32>> = (0..8).map(|_| base.clone()).collect();
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(&movie, &Settings::default(), &mut |_| true).expect("not cancelled");
    for (i, s) in out.shifts.iter().enumerate() {
        assert_eq!((s.dy, s.dx), (0, 0), "frame {i} was moved for no reason");
    }
}

#[test]
fn a_stopped_registration_returns_nothing_rather_than_half_an_answer() {
    let (ly, lx) = (48, 48);
    let frames = wandering(ly, lx, &[(0, 0), (1, 1), (2, 2), (3, 3)]);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let mut calls = 0;
    let out = register(&movie, &Settings::default(), &mut |_| {
        calls += 1;
        calls < 3
    });
    assert!(
        out.is_none(),
        "a stopped run must not hand back shifts that look complete"
    );
}

/// Progress rises and stays in range — the plugin draws a bar from it.
#[test]
fn progress_is_reported_in_order() {
    let (ly, lx) = (32, 32);
    let frames = wandering(ly, lx, &[(0, 0), (1, 0), (0, 1)]);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let mut seen = Vec::new();
    register(&movie, &Settings::default(), &mut |f| {
        seen.push(f);
        true
    })
    .expect("not cancelled");
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|&f| (0.0..=1.0).contains(&f)), "{seen:?}");
    assert!(
        seen.windows(2).all(|w| w[1] >= w[0]),
        "progress went backwards: {seen:?}"
    );
}

/// Bidiphase is off by default, and asking for a fixed one applies it.
#[test]
fn a_fixed_bidiphase_is_applied_without_measuring() {
    let (ly, lx) = (32, 32);
    let frames = wandering(ly, lx, &[(0, 0), (1, 0)]);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let settings = Settings {
        bidiphase: 2,
        ..Settings::default()
    };
    let out = register(&movie, &settings, &mut |_| true).expect("not cancelled");
    assert_eq!(out.bidiphase, 2);

    let off = register(&movie, &Settings::default(), &mut |_| true).expect("not cancelled");
    assert_eq!(off.bidiphase, 0);
}

// ------------------------------------------------------------- the backends

/// The claim the selector rests on: which backend runs the arithmetic changes
/// how many frames are in flight and nothing else. If these ever disagree, the
/// selector stops being a free choice and becomes a correctness setting.
#[test]
fn every_backend_measures_the_same_shifts() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (2, -1), (-3, 2), (1, 3), (-2, -2), (3, 1)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let base = Settings {
        spatial_taper: 5.0,
        maxregshift: 0.3,
        ..Settings::default()
    };

    let single = register(
        &movie,
        &Settings {
            backend: Backend::SingleThread,
            ..base
        },
        &mut |_| true,
    )
    .expect("single");
    let multi = register(
        &movie,
        &Settings {
            backend: Backend::MultiThread,
            ..base
        },
        &mut |_| true,
    )
    .expect("multi");

    assert_eq!(single.shifts.len(), multi.shifts.len());
    for (i, (a, b)) in single.shifts.iter().zip(&multi.shifts).enumerate() {
        assert_eq!((a.dy, a.dx), (b.dy, b.dx), "frame {i} differed by backend");
    }
    assert_eq!(single.reference, multi.reference, "the references differed");
}

/// `batch_size` is a memory knob, not a result knob — except through temporal
/// smoothing, which smooths *within* a batch. With it off, batching must not
/// change a single shift.
#[test]
fn batch_size_alone_does_not_change_the_answer() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (2, -1), (-3, 2), (1, 3), (-2, -2), (3, 1)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let base = Settings {
        spatial_taper: 5.0,
        maxregshift: 0.3,
        ..Settings::default()
    };

    let whole = register(
        &movie,
        &Settings {
            batch_size: 100,
            ..base
        },
        &mut |_| true,
    )
    .unwrap();
    let split = register(
        &movie,
        &Settings {
            batch_size: 2,
            ..base
        },
        &mut |_| true,
    )
    .unwrap();
    for (i, (a, b)) in whole.shifts.iter().zip(&split.shifts).enumerate() {
        assert_eq!(
            (a.dy, a.dx),
            (b.dy, b.dx),
            "frame {i} moved with the batch size"
        );
    }
}

/// Normalisation clips to the reference's 1st and 99th percentile, so a
/// saturated speck cannot drag the correlation peak onto itself.
///
/// It does change the picture being correlated — on a field that is mostly dark
/// background with a few bright cells, the 99th percentile cuts into the cells
/// themselves. So the thing to hold it to is not "identical numbers" but "still
/// finds the motion", which is what it is for.
#[test]
fn normalizing_still_recovers_the_motion() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (2, -1), (-3, 2), (1, 3)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let base = Settings {
        spatial_taper: 5.0,
        maxregshift: 0.3,
        ..Settings::default()
    };
    let out = register(
        &movie,
        &Settings {
            norm_frames: true,
            ..base
        },
        &mut |_| true,
    )
    .unwrap();

    // Relative to frame 0, because the reference sits wherever the
    // best-correlated frames put it.
    let offset = (out.shifts[0].dy - path[0].0, out.shifts[0].dx - path[0].1);
    for (i, (s, want)) in out.shifts.iter().zip(&path).enumerate() {
        assert_eq!(
            (s.dy - offset.0, s.dx - offset.1),
            *want,
            "frame {i} with norm_frames on"
        );
    }
}

/// Two-step reports the *total* shift, not the residual of the second pass —
/// otherwise applying it would leave the movie still moving.
#[test]
fn two_step_reports_the_total_shift() {
    let (ly, lx) = (64, 64);
    let path = [(0i32, 0i32), (3, -2), (-4, 1), (2, 3)];
    let frames = wandering(ly, lx, &path);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let base = Settings {
        spatial_taper: 5.0,
        maxregshift: 0.3,
        ..Settings::default()
    };
    let out = register(
        &movie,
        &Settings {
            two_step_registration: true,
            ..base
        },
        &mut |_| true,
    )
    .expect("two-step");

    let fixed: Vec<Vec<f32>> = frames
        .iter()
        .zip(&out.shifts)
        .map(|(f, s)| shift_frame(f, ly, lx, s.dy, s.dx))
        .collect();
    for y in 14..ly - 14 {
        for x in 14..lx - 14 {
            let first = fixed[0][y * lx + x];
            for (i, f) in fixed.iter().enumerate().skip(1) {
                assert!(
                    (f[y * lx + x] - first).abs() < 1e-2,
                    "frame {i} still moves after two-step"
                );
            }
        }
    }
}

/// Bad frames are reported, and a clean recording has none.
#[test]
fn a_clean_recording_reports_no_bad_frames() {
    let (ly, lx) = (64, 64);
    let frames = wandering(ly, lx, &[(0, 0), (1, 0), (0, 1), (1, 1)]);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(
        &movie,
        &Settings {
            spatial_taper: 5.0,
            ..Settings::default()
        },
        &mut |_| true,
    )
    .unwrap();
    assert_eq!(out.bad_frames.len(), 4);
    assert!(!out.bad_frames.iter().any(|&b| b), "{:?}", out.bad_frames);
}

// ----------------------------------------------------------- non-rigid

/// A field of many small features, so every block has something to lock onto.
/// Three big blobs would leave most blocks looking at empty background.
fn textured(ly: usize, lx: usize) -> Vec<f32> {
    let mut f = vec![10.0f32; ly * lx];
    let mut seed = 12345u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for _ in 0..120 {
        let (cy, cx) = (next() % ly, next() % lx);
        let amp = 80.0 + (next() % 120) as f32;
        for y in cy.saturating_sub(4)..(cy + 5).min(ly) {
            for x in cx.saturating_sub(4)..(cx + 5).min(lx) {
                let d2 = ((y as f32 - cy as f32).powi(2) + (x as f32 - cx as f32).powi(2)) / 4.0;
                f[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    f
}

/// Squash a frame: the top of the field moves one way and the bottom the other,
/// which is what breathing tissue does and what a single rigid shift cannot
/// express.
fn sheared(base: &[f32], ly: usize, lx: usize, amount: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; ly * lx];
    for y in 0..ly {
        // -amount at the top, +amount at the bottom.
        let dx = amount * (2.0 * y as f32 / (ly - 1) as f32 - 1.0);
        for x in 0..lx {
            let sx = (x as f32 - dx).clamp(0.0, (lx - 1) as f32);
            let (x0, t) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(lx - 1);
            out[y * lx + x] = base[y * lx + x0] * (1.0 - t) + base[y * lx + x1] * t;
        }
    }
    out
}

/// The point of non-rigid: a deformation a rigid shift cannot undo, undone.
#[test]
fn non_rigid_corrects_a_shear_that_rigid_cannot() {
    let (ly, lx) = (128, 128);
    let base = textured(ly, lx);
    // Half the frames are sheared, half are not, so the reference is the
    // undeformed field and the shear is what has to be found.
    let frames: Vec<Vec<f32>> = (0..8)
        .map(|t| {
            if t % 2 == 0 {
                base.clone()
            } else {
                sheared(&base, ly, lx, 3.0)
            }
        })
        .collect();
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let settings = Settings {
        spatial_taper: 10.0,
        block_size: [32, 32],
        nonrigid: true,
        snr_thresh: 1.0,
        ..Settings::default()
    };
    let out = register(&movie, &settings, &mut |_| true).expect("registered");
    let nr = out.nonrigid.as_ref().expect("a block field");
    assert_eq!(nr.shifts.len(), frames.len());
    assert_eq!(nr.shifts[0].len(), nr.blocks.blocks.len());

    // A sheared frame's blocks must disagree with each other along y — that is
    // the deformation. An undeformed frame's blocks should broadly agree.
    let spread = |t: usize| -> f32 {
        let dxs: Vec<f32> = nr.shifts[t].iter().map(|s| s.dx).collect();
        dxs.iter().cloned().fold(f32::MIN, f32::max) - dxs.iter().cloned().fold(f32::MAX, f32::min)
    };
    assert!(
        spread(1) > 2.0,
        "a sheared frame's blocks should disagree by several pixels, got {}",
        spread(1)
    );

    // And applying the field brings a sheared frame back onto the reference
    // better than the rigid shift alone could.
    let rigid_only = shift_frame(&frames[1], ly, lx, out.shifts[1].dy, out.shifts[1].dx);
    let warped = out.apply(&frames[1], ly, lx, 1);
    let err = |v: &[f32]| -> f64 {
        let mut e = 0.0;
        let mut n = 0;
        for y in 20..ly - 20 {
            for x in 20..lx - 20 {
                let d = (v[y * lx + x] - frames[0][y * lx + x]) as f64;
                e += d * d;
                n += 1;
            }
        }
        (e / n as f64).sqrt()
    };
    let (before, after) = (err(&rigid_only), err(&warped));
    assert!(
        after < before * 0.9,
        "the warp did not improve on the rigid shift: {after:.2} vs {before:.2}"
    );
}

/// With `nonrigid` off there is no field, and `apply` is the rigid shift.
#[test]
fn no_block_field_without_asking_for_one() {
    let (ly, lx) = (64, 64);
    let frames = wandering(ly, lx, &[(0, 0), (2, -1), (-1, 2)]);
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(
        &movie,
        &Settings {
            spatial_taper: 5.0,
            ..Settings::default()
        },
        &mut |_| true,
    )
    .unwrap();
    assert!(out.nonrigid.is_none());
    // `apply` then agrees with `shift_frame` exactly.
    let a = out.apply(&frames[1], ly, lx, 1);
    let b = shift_frame(&frames[1], ly, lx, out.shifts[1].dy, out.shifts[1].dx);
    assert_eq!(a, b);
}

/// Sub-pixel refinement really is sub-pixel: block shifts are not whole numbers.
#[test]
fn block_shifts_are_resolved_below_a_pixel() {
    let (ly, lx) = (128, 128);
    let base = textured(ly, lx);
    let frames = vec![base.clone(), sheared(&base, ly, lx, 2.5)];
    let movie = Frames {
        ly,
        lx,
        frames: &frames,
    };
    let out = register(
        &movie,
        &Settings {
            spatial_taper: 10.0,
            block_size: [32, 32],
            nonrigid: true,
            snr_thresh: 1.0,
            subpixel: 10,
            ..Settings::default()
        },
        &mut |_| true,
    )
    .unwrap();
    let nr = out.nonrigid.as_ref().unwrap();
    let fractional = nr.shifts[1]
        .iter()
        .filter(|s| (s.dx - s.dx.round()).abs() > 1e-3 || (s.dy - s.dy.round()).abs() > 1e-3)
        .count();
    assert!(
        fractional > 0,
        "every block shift landed on a whole pixel; the kriging refinement did nothing"
    );
}

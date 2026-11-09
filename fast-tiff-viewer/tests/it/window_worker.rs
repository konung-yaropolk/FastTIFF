//! Which window is drawn, and who builds it.
//!
//! A frame too large for one texture is shown through a window, and the window
//! has to be re-cut whenever the view moves far enough. Cutting it costs tenths
//! of a second on a gigapixel mosaic and seconds when the view spans most of the
//! file — long enough that doing it on the interface thread stops the program,
//! and it stops it precisely when the user is moving fastest.
//!
//! Two things prevent that, and both are tested here:
//!
//! - [`roi::serve`] decides whether the resident window can carry the view for a
//!   moment longer. When it can, the new one is built in the background and the
//!   old one keeps being drawn, scaled — soft for a moment rather than dead for
//!   a moment.
//! - [`fast_tiff_viewer::window::WindowWorker`] does that building, off-thread,
//!   and must produce exactly what the inline path would have produced. A
//!   background worker that quietly returned something different would be far
//!   worse than the stall it replaces.

use fast_tiff_viewer::roi::{self, Residency, Roi};

// A view of the NASA Andromeda mosaic — the frame this machinery exists for.
const W: u32 = 40_000;
const H: u32 = 12_788;

fn roi(x: u32, y: u32, w: u32, h: u32, stride: u32) -> Roi {
    Roi { x, y, w, h, stride }
}

/// `resident` is what would be uploaded, `required` the least that serves the
/// view — the pair `roi::plan` returns.
fn plan(resident: Roi, required: Roi) -> Residency {
    Residency { resident, required }
}

// ---------------------------------------------------------------------------
// What to draw, and what to prepare
// ---------------------------------------------------------------------------

/// The first view of a file: there is no picture to keep showing, so this one
/// is waited for. Drawing *something* matters more than drawing it instantly.
#[test]
fn the_first_window_is_built_synchronously() {
    let want = roi(0, 0, W, H, 8);
    let s = roi::serve(None, None, &plan(want, roi(0, 0, W, H, 8)));
    assert_eq!(s.show, want);
    assert_eq!(s.want, want, "nothing better to prepare");
    assert!(
        s.ready,
        "with nothing on screen there is nothing to fall back on"
    );
}

/// A small pan inside the margin: the resident window still serves, so the GPU
/// is not touched at all. This is the case that keeps ordinary panning free.
#[test]
fn a_pan_inside_the_resident_window_asks_for_nothing() {
    let current = roi(4_000, 2_000, 8_000, 4_000, 1);
    let required = roi(4_500, 2_200, 3_840, 2_160, 1);
    // A fresh plan would centre a new window on the view — irrelevant, because
    // what is resident already covers it at the right resolution.
    let s = roi::serve(
        Some(current),
        None,
        &plan(roi(3_000, 1_000, 8_000, 4_000, 1), required),
    );
    assert_eq!(s.show, current);
    assert_eq!(
        s.want, current,
        "wanting the freshly planned window instead would re-cut on every pan, \
         which is exactly what the margin exists to avoid"
    );
    assert!(s.ready);
}

/// Zooming in past the resident sampling level. The old window covers the same
/// ground at the wrong resolution, so it is kept on screen while the sharper one
/// is built — this is the freeze that used to happen when zoom crossed 1:1.
#[test]
fn zooming_in_keeps_the_coarse_window_up_while_a_finer_one_is_built() {
    let current = roi(0, 0, W, H, 8); // the whole frame, coarse
    let finer = roi(10_000, 4_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(current),
        None,
        &plan(finer, roi(11_000, 4_500, 3_840, 2_160, 1)),
    );

    assert_eq!(
        s.show, current,
        "the picture must not be blanked while the finer one is cut"
    );
    assert_eq!(
        s.want, finer,
        "and the finer one must actually be asked for"
    );
    assert!(
        !s.ready,
        "not ready means: draw `show`, build `want` off-thread"
    );
    assert_ne!(s.show, s.want, "the caller only builds when these differ");
}

/// The same view from the caller's side: `ready == false` is the only state that
/// starts background work, and it must always name something to do. A plan that
/// was not ready but wanted what it already showed would leave the image soft
/// for ever, with nothing queued to sharpen it.
#[test]
fn not_ready_always_names_something_to_build() {
    let cases = [
        (
            Some(roi(0, 0, W, H, 8)),
            None,
            roi(0, 0, W, H, 1),
            roi(0, 0, W, H, 1),
        ),
        (
            Some(roi(0, 0, W, H, 4)),
            None,
            roi(2_000, 1_000, 9_000, 5_000, 1),
            roi(3_000, 2_000, 3_840, 2_160, 1),
        ),
        (
            Some(roi(0, 0, W, H, 1)),
            None,
            roi(0, 0, W, H, 4),
            roi(0, 0, W, H, 4),
        ),
        // The overview path: one where it is what was planned (so it must not
        // be offered) and one where it is not.
        (
            Some(roi(10_000, 4_000, 8_000, 4_000, 1)),
            Some(roi(0, 0, W, H, 16)),
            roi(0, 0, W, H, 16),
            roi(0, 0, W, H, 16),
        ),
        (
            Some(roi(0, 0, 8_000, 4_000, 1)),
            Some(roi(0, 0, W, H, 16)),
            roi(30_000, 8_000, 8_000, 4_000, 1),
            roi(31_000, 8_500, 3_840, 2_160, 1),
        ),
    ];
    for (current, overview, resident, required) in cases {
        let s = roi::serve(current, overview, &plan(resident, required));
        if s.ready {
            assert_eq!(s.show, s.want, "a ready plan has nothing outstanding");
        } else {
            assert_ne!(
                s.show, s.want,
                "an unready plan must name the window to build"
            );
            assert_eq!(
                s.want, resident,
                "and it is the planned window, margin and all"
            );
        }
    }
}

/// A jump clear across the image with *no* overview held. Keeping the old window
/// would draw the edge of the data — blank where the picture should be — so this
/// one is waited for even though something is on screen.
///
/// With an overview this becomes case 3 instead; see
/// [`a_jump_across_the_image_falls_back_to_the_overview`]. What this pins is
/// that case 4 still fires when there is nothing to fall back on, which is what
/// an overview wrongly believed to be present would break.
#[test]
fn a_jump_outside_the_resident_window_is_not_covered_up() {
    let current = roi(0, 0, 8_000, 4_000, 1);
    let elsewhere = roi(30_000, 8_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(current),
        None,
        &plan(elsewhere, roi(31_000, 8_500, 3_840, 2_160, 1)),
    );
    assert_eq!(
        s.show, elsewhere,
        "showing `current` would display blank pixels"
    );
    assert!(s.ready);
}

/// Zooming *out* past the resident window with no overview held. It is finer
/// than needed but does not cover the new view, so it cannot be stretched over
/// it and there is nothing else to show.
///
/// This is the case the overview exists to remove — see
/// [`a_jump_across_the_image_falls_back_to_the_overview`] — but without one the
/// answer must still be to build it now rather than show something wrong.
#[test]
fn zooming_out_beyond_the_resident_window_is_built_now() {
    let current = roi(10_000, 4_000, 8_000, 4_000, 1);
    let whole = roi(0, 0, W, H, 8);
    let s = roi::serve(Some(current), None, &plan(whole, whole));
    assert_eq!(s.show, whole);
    assert!(s.ready);
}

/// A resident window covering the view at a *finer* stride than required —
/// zooming out a little, still inside the same window. It covers the ground, so
/// it is kept up; a sharper picture than asked for is not a reason to blank.
#[test]
fn a_sharper_window_than_needed_is_kept_on_screen() {
    let current = roi(0, 0, W, H, 2);
    let s = roi::serve(
        Some(current),
        None,
        &plan(roi(0, 0, W, H, 4), roi(0, 0, W, H, 4)),
    );
    assert_eq!(
        s.show, current,
        "what is up is already better than asked for"
    );
    assert!(!s.ready, "but the cheaper window is still worth having");
}

/// Zooming out, or jumping across the image, with a coarse overview held. The
/// resident window covers too little ground to be stretched over the new view,
/// but the overview covers everything — so the picture stays up, correct if
/// coarse, while the properly-sampled window is built off-thread.
///
/// This is the 0.75-second zoom-out stall, removed.
#[test]
fn a_jump_across_the_image_falls_back_to_the_overview() {
    let current = roi(0, 0, 8_000, 4_000, 1);
    let overview = roi(0, 0, W, H, 16);
    let elsewhere = roi(30_000, 8_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(current),
        Some(overview),
        &plan(elsewhere, roi(31_000, 8_500, 3_840, 2_160, 1)),
    );
    assert_eq!(
        s.show, overview,
        "the overview covers the new view; the fine window does not"
    );
    assert_eq!(
        s.want, elsewhere,
        "and the properly-sampled window is still asked for"
    );
    assert!(
        !s.ready,
        "showing the overview costs an upload, not a decode"
    );
}

/// Zoomed fully out, the planned window *is* the overview — there is nothing
/// better to prepare. Offering it as a fallback here would return `show ==
/// want` with `ready: false`: the caller queues no build, requests no repaint,
/// and the image stays coarse for ever with nothing on its way.
///
/// So this must fall through to the last case and be reported ready, leaving
/// the upload path to notice it already holds those pixels.
#[test]
fn the_overview_is_not_offered_when_it_is_what_was_planned() {
    let overview = roi(0, 0, W, H, 16);
    let s = roi::serve(
        Some(roi(10_000, 4_000, 8_000, 4_000, 1)),
        Some(overview),
        &plan(overview, overview),
    );
    assert_eq!(s.show, overview);
    assert_eq!(s.want, overview);
    assert!(
        s.ready,
        "nothing to build means ready, or nothing ever gets built"
    );
    assert_eq!(s.show, s.want, "a ready plan has nothing outstanding");
}

/// The overview is the fallback of last resort, not a preference. A resident
/// window that can serve the view — or that covers it at a finer sampling —
/// must win, or every pan and every zoom would drop the picture to the
/// overview's coarse sampling first and climb back up.
#[test]
fn an_overview_never_preempts_a_resident_window() {
    let overview = roi(0, 0, W, H, 16);

    // Case 1: resident serves outright.
    let current = roi(4_000, 2_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(current),
        Some(overview),
        &plan(
            roi(3_000, 1_000, 8_000, 4_000, 1),
            roi(4_500, 2_200, 3_840, 2_160, 1),
        ),
    );
    assert_eq!(
        s.show, current,
        "an overview must not displace a window that already serves"
    );
    assert!(s.ready);

    // Case 2: resident covers the ground, at the wrong sampling.
    let coarse = roi(0, 0, W, H, 8);
    let finer = roi(10_000, 4_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(coarse),
        Some(overview),
        &plan(finer, roi(11_000, 4_500, 3_840, 2_160, 1)),
    );
    assert_eq!(
        s.show, coarse,
        "stride 8 covers this view; dropping to the overview's 16 is worse"
    );
    assert_eq!(s.want, finer);
    assert!(!s.ready);
}

/// An overview that does not cover the view is no use — the same rule the
/// resident window is held to. It always does cover, being the whole frame, but
/// nothing in `serve` may assume that.
#[test]
fn an_overview_that_does_not_cover_the_view_is_not_offered() {
    let partial = roi(0, 0, W / 2, H, 16);
    let elsewhere = roi(30_000, 8_000, 8_000, 4_000, 1);
    let s = roi::serve(
        Some(roi(0, 0, 8_000, 4_000, 1)),
        Some(partial),
        &plan(elsewhere, roi(31_000, 8_500, 3_840, 2_160, 1)),
    );
    assert_eq!(
        s.show, elsewhere,
        "half the frame cannot stand in for the other half"
    );
    assert!(s.ready);
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

#[cfg(all(feature = "threads", feature = "mmap"))]
mod worker {
    use super::{roi, Roi};
    use fast_tiff_lib::TiffStack;
    use fast_tiff_viewer::bandcache::BandCache;
    use fast_tiff_viewer::prefetch::{ChannelJob, Decoded};
    use fast_tiff_viewer::window::{self, Built, WindowWorker};
    use fast_tiff_viewer::worker::CancelToken;
    use fast_tiff_viewer::Revision;
    use scivis_render::ChannelKind;
    use std::path::PathBuf;

    /// Any one display revision, for a worker driven without a display.
    const REV: Revision = Revision {
        layout: 1,
        enabled: 1,
    };

    fn fixture(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("fast-tiff-lib")
            .join("tests")
            .join("fixtures")
            .join(name);
        p.exists().then_some(p)
    }

    fn jobs_for(tiff: &TiffStack) -> Vec<ChannelJob> {
        let f = &tiff.frames[0];
        vec![ChannelJob {
            channel: 0,
            ifd_idx: 0,
            plane: 0,
            kind: ChannelKind::Int16,
            rgb: false,
            width: f.width,
            height: f.height,
        }]
    }

    /// Wait for the worker's answer, but never for ever: a worker that silently
    /// produced nothing would otherwise hang the suite instead of failing it.
    fn wait_for(w: &WindowWorker, frame: usize, want: &Roi, rev: Revision) -> Option<Built> {
        for _ in 0..600 {
            if let Some(b) = w.take_matching(frame, want, rev) {
                return Some(b);
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        None
    }

    fn same(a: &Decoded, b: &Decoded) -> bool {
        match (a, b) {
            (Decoded::U8(x), Decoded::U8(y)) => x == y,
            (Decoded::U16(x), Decoded::U16(y)) => x == y,
            (Decoded::F32(x), Decoded::F32(y)) => x == y,
            _ => false,
        }
    }

    /// The point of the whole exercise: what arrives from the worker must be
    /// what the inline path would have built. Anything else trades a visible
    /// stall for an invisible wrong picture.
    #[test]
    fn the_worker_builds_what_the_inline_path_would_have() {
        let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else {
            return;
        };
        let tiff = TiffStack::open(&path).unwrap();
        let jobs = jobs_for(&tiff);
        // A window off the tile grid and subsampled, so the assembly has to do
        // real work rather than hand back the frame.
        let want = roi(20, 20, 60, 40, 2);

        let mut bands = BandCache::default();
        let inline = window::assemble(
            &tiff,
            &mut bands,
            REV.layout,
            &jobs,
            &want,
            0,
            &CancelToken::never(),
        )
        .unwrap();

        let worker = WindowWorker::new(std::sync::Arc::new(TiffStack::open(&path).unwrap()))
            .expect("worker should start");
        worker.request(0, want, REV, jobs.clone());
        let built = wait_for(&worker, 0, &want, REV).expect("worker produced nothing");

        assert_eq!(built.roi, want);
        assert_eq!(built.channels, vec![0]);
        assert_eq!(built.planes.len(), inline.len());
        let (tw, th) = want.texture_size();
        for (i, (got, expect)) in built.planes.iter().zip(&inline).enumerate() {
            assert!(same(got, expect), "plane {i} differs from the inline build");
            match got {
                Decoded::U16(v) => assert_eq!(v.len(), tw as usize * th as usize),
                _ => panic!("plane {i}: expected a u16 plane"),
            }
        }
    }

    /// A finished window is identified by everything that would make it the
    /// wrong picture — which frame, which rectangle, and which display revision
    /// (the channel mapping and the channels on) — and a mismatch on any of
    /// them must not be uploaded. It must also not be thrown away: the view may
    /// be about to want it, and discarding it means building it again.
    #[test]
    fn a_window_survives_being_asked_for_with_the_wrong_key() {
        let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else {
            return;
        };
        let tiff = TiffStack::open(&path).unwrap();
        let jobs = jobs_for(&tiff);
        let want = roi(16, 16, 48, 32, 1);

        let worker = WindowWorker::new(std::sync::Arc::new(TiffStack::open(&path).unwrap()))
            .expect("worker should start");
        worker.request(0, want, REV, jobs);

        for _ in 0..600 {
            assert!(worker.take_matching(1, &want, REV).is_none(), "wrong frame");
            assert!(
                worker
                    .take_matching(0, &roi(0, 0, 48, 32, 1), REV)
                    .is_none(),
                "wrong rect"
            );
            let toggled = Revision {
                enabled: REV.enabled + 1,
                ..REV
            };
            assert!(
                worker.take_matching(0, &want, toggled).is_none(),
                "a channel was switched since"
            );
            let reassigned = Revision {
                layout: REV.layout + 1,
                ..REV
            };
            assert!(
                worker.take_matching(0, &want, reassigned).is_none(),
                "the axes were reassigned since"
            );
            // Every wrong key above has been refused; the right one must still
            // find the window there.
            if worker.take_matching(0, &want, REV).is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the window never arrived, so nothing was proved about keeping it");
    }

    /// The point of asking "can this serve the view?" rather than "is this the
    /// window I asked for?".
    ///
    /// A gesture re-plans every frame, so the rectangle wanted drifts
    /// continuously and no window built against an earlier frame is ever equal
    /// to the current one. Under equality the worker's output was thrown away
    /// almost every time: measured over one zoom in and out of a 40000x12788
    /// mosaic, 15 windows were built and 4 ever reached the screen, and every
    /// one of the other 11 covered the view that discarded it.
    #[test]
    fn a_window_larger_than_needed_still_serves() {
        let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else {
            return;
        };
        let tiff = TiffStack::open(&path).unwrap();
        let jobs = jobs_for(&tiff);
        let built_for = roi(0, 0, 96, 64, 1);

        let worker = WindowWorker::new(std::sync::Arc::new(TiffStack::open(&path).unwrap()))
            .expect("worker should start");
        worker.request(0, built_for, REV, jobs);

        // The view has moved on to something smaller and inside it — which is
        // what a zoom does between the request and its answer.
        let now_needs = roi(16, 16, 48, 32, 1);
        assert!(
            built_for.serves(&now_needs),
            "the test is only meaningful if it covers"
        );
        let got = wait_for(&worker, 0, &now_needs, REV).expect("a covering window should serve");
        assert_eq!(
            got.roi, built_for,
            "and it arrives as the window that was actually built"
        );
    }

    /// Still refused when it genuinely cannot serve: a window that does not
    /// cover the ground, or covers it at the wrong sampling, would draw the
    /// edge of the data or the wrong resolution.
    #[test]
    fn a_window_that_cannot_serve_is_still_refused() {
        let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else {
            return;
        };
        let tiff = TiffStack::open(&path).unwrap();
        let jobs = jobs_for(&tiff);
        let built_for = roi(0, 0, 48, 32, 1);

        let worker = WindowWorker::new(std::sync::Arc::new(TiffStack::open(&path).unwrap()))
            .expect("worker should start");
        worker.request(0, built_for, REV, jobs);

        for _ in 0..600 {
            // Off to one side: not covered.
            assert!(
                worker
                    .take_matching(0, &roi(52, 0, 40, 32, 1), REV)
                    .is_none(),
                "not covered"
            );
            // Same ground, finer sampling than the built window holds.
            assert!(
                worker
                    .take_matching(0, &roi(0, 0, 48, 32, 2), REV)
                    .is_none(),
                "a different stride is a different picture"
            );
            if worker.take_matching(0, &built_for, REV).is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the window never arrived, so nothing was proved");
    }

    /// The worker takes the newest request and drops what it has been overtaken
    /// by. A zoom produces a request per frame; answering all of them would put
    /// the worker further behind the longer the user keeps moving.
    #[test]
    fn a_burst_of_requests_settles_on_the_last_one() {
        let Some(path) = fixture("tld_u16_spp1_p1_grid.tif") else {
            return;
        };
        let tiff = TiffStack::open(&path).unwrap();
        let jobs = jobs_for(&tiff);

        let worker = WindowWorker::new(std::sync::Arc::new(TiffStack::open(&path).unwrap()))
            .expect("worker should start");
        let last = roi(0, 0, 100, 70, 1);
        for stride in [8u32, 4, 2] {
            worker.request(0, roi(0, 0, 100, 70, stride), REV, jobs.clone());
        }
        worker.request(0, last, REV, jobs.clone());

        let built = wait_for(&worker, 0, &last, REV).expect("the newest request was never served");
        assert_eq!(built.roi, last);
    }

    /// Two channels by three timepoints of 64 x 64 u16, plane `i` filled with
    /// `i * 1000 + 7`, so which IFD a window came from can be read off any of
    /// its samples.
    fn two_channel_movie() -> fast_tiff_viewer::Stack {
        use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
        let opts = WriterOptions::new(64, 64, SampleType::U16).metadata(StackMetaWrite::new(2, 1));
        let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
        for plane in 0..6u16 {
            w.write_frame_u16(&vec![plane * 1000 + 7; 64 * 64]).unwrap();
        }
        let bytes = w.finish().unwrap().into_inner();
        fast_tiff_viewer::Stack::from_bytes(bytes, "movie.tif".into(), false).unwrap()
    }

    fn first_sample(plane: &Decoded) -> u16 {
        match plane {
            Decoded::U16(v) => v[0],
            _ => panic!("16-bit samples should come back as U16"),
        }
    }

    /// The gap the display revision closes. A window asked for before the axes
    /// were reassigned — still being built, or finished and not yet taken —
    /// holds the old mapping's planes, and once the override lands it must not
    /// be shown: frame 1's channel 0 was IFD 2 and is now IFD 3. Before the
    /// revision was part of the key, it was taken (same frame, same rectangle,
    /// same channel list), and a repeat request for that key was a no-op, so
    /// nothing even rebuilt it.
    ///
    /// Nor may the window asked for under the new mapping be cut from the bands
    /// the worker decoded for the old one. They are keyed by display channel,
    /// so the same channel list over the same rows looks like a hit; nothing
    /// outside the worker can reach its cache to clear it.
    #[test]
    fn a_window_from_before_a_dimension_override_is_not_taken_after_it() {
        use fast_tiff_viewer::prefetch::build_jobs;
        let mut stack = two_channel_movie();
        let worker =
            WindowWorker::new(std::sync::Arc::clone(&stack.tiff)).expect("worker should start");
        let whole = Roi::full(64, 64);
        let before = stack.display.revision();
        worker.request(1, whole, before, build_jobs(&stack, 1, &[true, true]));

        fast_tiff_viewer::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
        let after = stack.display.revision();
        assert_ne!(after.layout, before.layout, "an override moves the layout");

        let mut old = None;
        for _ in 0..600 {
            assert!(
                worker.take_matching(1, &whole, after).is_none(),
                "a window of the old channel mapping was taken after the override"
            );
            if let Some(b) = worker.take_matching(1, &whole, before) {
                old = Some(b);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let old = old.expect("the window never arrived, so nothing was proved");
        assert_eq!(first_sample(&old.planes[0]), 2007, "channel 0 read IFD 2");

        // The same channel list as before, so the worker's bands for the old
        // mapping would be a hit if they were still held.
        let jobs = build_jobs(&stack, 1, &[true, true, false]);
        assert_eq!(jobs.iter().map(|j| j.channel).collect::<Vec<_>>(), [0, 1]);
        worker.request(1, whole, after, jobs);
        let new = wait_for(&worker, 1, &whole, after).expect("the new window was never built");
        assert_eq!(
            first_sample(&new.planes[0]),
            3007,
            "channel 0 reads IFD 3 now, not a band decoded from IFD 2"
        );
        assert_eq!(first_sample(&new.planes[1]), 4007, "and channel 1 IFD 4");
    }
}

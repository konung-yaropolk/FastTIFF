//! What the status bar shows for work running on a worker.
//!
//! The readout has one job that is easy to get wrong in a way nothing catches:
//! it has to tell "no estimate yet" apart from "nought per cent". They look
//! identical in an `AtomicU32` and completely different on screen — a bar
//! parked at 0% for thirty seconds reads as a hang, and a plugin is under no
//! obligation to call `progress` at all.

use super::{Job, PROGRESS_UNKNOWN};
use std::sync::atomic::Ordering;

#[test]
fn a_job_that_has_reported_nothing_has_no_fraction() {
    let job = Job::new("Importing");
    assert_eq!(
        job.fraction(),
        None,
        "a fresh job must read as unknown, not as zero — the readout draws a \
         spinner for one and a stalled bar for the other"
    );
    assert_eq!(job.progress.load(Ordering::Relaxed), PROGRESS_UNKNOWN);
}

#[test]
fn the_first_report_turns_the_spinner_into_a_bar() {
    let job = Job::new("Saving");
    // Nought really reported is a bar at nought, not a spinner. This is the
    // whole reason the sentinel is not simply 0.
    Job::report(&job.progress, 0.0);
    assert_eq!(job.fraction(), Some(0.0));

    Job::report(&job.progress, 0.5);
    assert_eq!(job.fraction(), Some(0.5));
}

#[test]
fn a_fraction_outside_the_range_is_brought_back_into_it() {
    let job = Job::new("Exporting");
    // A plugin's arithmetic is its own; the bar's is not. `ProgressBar` with a
    // fraction above 1.0 draws past its own rectangle.
    Job::report(&job.progress, 4.0);
    assert_eq!(job.fraction(), Some(1.0));

    Job::report(&job.progress, -1.0);
    assert_eq!(job.fraction(), Some(0.0));

    // And NaN, which `clamp` would panic on if it reached it — it does not,
    // because the cast to `u32` saturates at zero first.
    Job::report(&job.progress, f32::NAN);
    let f = job.fraction().expect("still a number");
    assert!((0.0..=1.0).contains(&f), "{f}");
}

#[test]
fn a_job_starts_uncancelled() {
    let job = Job::new("Importing");
    assert!(!job.cancel.load(Ordering::Relaxed));
    job.cancel.store(true, Ordering::Relaxed);
    assert!(job.cancel.load(Ordering::Relaxed));
}

/// The label is what is printed over the bar, so it is the label — not the
/// file name — that has to survive into the job.
#[test]
fn the_label_is_what_the_bar_will_say() {
    assert_eq!(Job::new("Importing").label(), "Importing");
    assert_eq!(
        Job::new(format!("Exporting ({})", "PNG")).label(),
        "Exporting (PNG)"
    );
}

/// A long job is several pieces of work, and the bar says which one it is on.
///
/// This is the whole of the fix for a bar that reached 100% and then sat there:
/// the run was not over, it had moved on to encoding and handing over a result,
/// and nothing said so. A phase renames the bar *and* puts it back to unknown,
/// so it cannot inherit the finished look of the phase before it.
#[test]
fn a_new_phase_renames_the_bar_and_starts_it_again() {
    let job = Job::new("Derivatives");
    Job::report(&job.progress, 1.0);
    assert_eq!(job.fraction(), Some(1.0));

    Job::begin_phase(&job.label, &job.progress, "Encoding result");
    assert_eq!(job.label(), "Encoding result");
    assert_eq!(
        job.fraction(),
        None,
        "a new phase starts as a spinner, not at the last one's 100%"
    );

    Job::report(&job.progress, 0.25);
    assert_eq!(job.fraction(), Some(0.25));
}

/// The handles a worker gets are the job's own, not copies of its state.
///
/// The phase is set from the worker thread and read by the interface thread; if
/// `begin_phase` wrote to anything but the shared label, the bar would keep
/// saying what the run started as.
#[test]
fn a_worker_renames_the_bar_the_interface_is_reading() {
    let job = Job::new("Stabilizing");
    let (label, progress) = (job.label.clone(), job.progress.clone());
    std::thread::spawn(move || Job::begin_phase(&label, &progress, "Writing result"))
        .join()
        .expect("the worker panicked");
    assert_eq!(job.label(), "Writing result");
}

// ---------------------------------------------------- what the bar says

use super::{info_key, progress_text};

#[test]
fn the_bar_says_the_percentage_then_what_is_running() {
    assert_eq!(progress_text("Importing", Some(0.0)), "0%  Importing");
    assert_eq!(progress_text("Importing", Some(0.42)), "42%  Importing");
    assert_eq!(progress_text("Importing", Some(0.999)), "100%  Importing");
    assert_eq!(progress_text("Saving", Some(1.0)), "100%  Saving");

    // Whole numbers only: a bar that reported "42.4%" would be claiming a
    // precision the permille it came from does not have, and the digit would
    // flicker on every frame.
    assert!(!progress_text("Saving", Some(0.4237)).contains('.'));
}

/// No fraction, no percentage — the label alone, over an animated bar. Printing
/// "0%" there would be a claim the host cannot support.
#[test]
fn an_unknown_fraction_reads_as_the_label_alone() {
    assert_eq!(progress_text("Opening", None), "Opening");
    assert_eq!(progress_text("Exporting (PNG)", None), "Exporting (PNG)");
}

/// The file name used to be shown beside the label. It is not in the readout
/// any more — it is in the window title — so nothing here should be carrying
/// one around.
#[test]
fn the_readout_says_the_process_not_the_file() {
    let text = progress_text("Importing", Some(0.5));
    assert!(!text.contains(".oir"), "{text}");
    assert_eq!(text, "50%  Importing");
}

/// The bug this guards: the key is compared frame to frame to decide whether
/// the bar changed height and the window should grow. A percentage in it
/// changes on nearly every frame, so the window would try to grow on nearly
/// every frame for the whole length of a load.
#[test]
fn the_height_key_does_not_move_with_the_percentage() {
    let a = info_key(None, Some("Importing"));
    let b = info_key(None, Some("Importing"));
    assert_eq!(a, b);

    // Whereas the things that genuinely reflow the row do change it.
    assert_ne!(info_key(None, Some("Importing")), info_key(None, None));
    assert_ne!(
        info_key(Some("Saved a.tif"), Some("Importing")),
        info_key(None, Some("Importing"))
    );
    assert_ne!(
        info_key(None, Some("Importing")),
        info_key(None, Some("Exporting (PNG)"))
    );
}

/// A status and a label must not be able to collide into one key — otherwise a
/// row that gained a label and lost the tail of a status would look unchanged.
#[test]
fn the_two_parts_of_the_key_stay_separate() {
    assert_ne!(
        info_key(Some("ab"), Some("c")),
        info_key(Some("a"), Some("bc"))
    );
    assert_eq!(info_key(None, None), "");
}

// ---------------------------------------------- what the status line says

use super::{stand_aside, status_label, status_lines, StatusLine};
use fast_tiff_viewer::{Notice, ShapeNote, VolumeNote};

fn texts(lines: &[StatusLine]) -> Vec<&str> {
    lines.iter().map(|l| l.text.as_str()).collect()
}

/// The app's message alone reads as it always did: amber, unless it is the
/// one recorded as done.
#[test]
fn a_message_is_green_only_when_it_is_the_one_that_finished() {
    let note = status_lines(Some("Open an image first"), None, [], &[]);
    assert_eq!(
        note,
        [StatusLine {
            text: "Open an image first".into(),
            done: false
        }]
    );
    let done = status_lines(Some("Saved a.tif"), Some("Saved a.tif"), [], &[]);
    assert!(done[0].done);
    let other = status_lines(Some("Could not save a.tif"), Some("Saved a.tif"), [], &[]);
    assert!(!other[0].done, "another message under the green one's text");
    assert!(status_lines(None, Some("Saved a.tif"), [], &[]).is_empty());
}

/// The bug the notices exist for, on the line itself: a save, or a refusal,
/// used to replace the note about the file's shape for good, and a decode
/// failure to replace it until the next open. Now each says its own piece,
/// the app's message first, in amber except a success.
#[test]
fn the_message_and_every_notice_share_the_line() {
    let shape = Notice::Shape(ShapeNote::TooManyChannels(9));
    let failed = Notice::DecodeFailed("LZW: bad code".into());
    let lines = status_lines(
        Some("Saved a.tif"),
        Some("Saved a.tif"),
        [&failed, &shape],
        &[],
    );
    assert_eq!(
        texts(&lines),
        [
            "Saved a.tif",
            "Failed to decode frame: LZW: bad code",
            "Note: stack has 9 channels; showing the first 6."
        ]
    );
    assert_eq!(
        lines.iter().map(|l| l.done).collect::<Vec<_>>(),
        [true, false, false]
    );
    // With no message of the app's, the core's alone.
    assert_eq!(
        texts(&status_lines(None, None, [&shape], &[])),
        [shape.to_string()]
    );
}

/// While a load or a job runs, the row is the progress bar's: what was said
/// of the document on show before it appeared stands aside rather than
/// squeezing the bar or reading as a comment on it, and the app's message
/// (said since the work began, since starting it cleared the last) stays
/// beside the bar as it did. What the core finds while the work runs shows
/// beside the bar, as it did when it was written over the one message: a 3D
/// view that runs out of GPU memory during a long save says so then.
#[test]
fn the_notices_stand_aside_while_something_is_in_progress() {
    let shape = Notice::Shape(ShapeNote::TooManyChannels(9));
    let reduced = Notice::Volume(VolumeNote::Reduced);
    let no_room = Notice::Volume(VolumeNote::NoRoom("out of memory".into()));
    let mut aside = None;
    let mut held = vec![shape.clone(), reduced.clone()];
    let line = |held: &[Notice], aside: &Option<Vec<Notice>>| {
        status_lines(
            Some("Busy"),
            None,
            held,
            aside.as_deref().unwrap_or_default(),
        )
        .into_iter()
        .map(|l| l.text)
        .collect::<Vec<_>>()
    };

    // No bar: all of it.
    stand_aside(&mut aside, false, || held.iter());
    assert_eq!(line(&held, &aside).len(), 3);

    // The bar appears: what was up stands aside, the message stays.
    stand_aside(&mut aside, true, || held.iter());
    assert_eq!(line(&held, &aside), ["Busy"]);
    stand_aside(&mut aside, true, || held.iter());
    assert_eq!(line(&held, &aside), ["Busy"], "and stays aside");

    // Found while it runs: news, beside the bar. The volume's note is of a
    // kind already aside, but not the one that was.
    held[1] = no_room.clone();
    stand_aside(&mut aside, true, || held.iter());
    assert_eq!(
        line(&held, &aside),
        ["Busy".to_string(), no_room.to_string()]
    );

    // A note that went away and came back is news too.
    held.remove(0);
    stand_aside(&mut aside, true, || held.iter());
    held.push(shape.clone());
    stand_aside(&mut aside, true, || held.iter());
    assert_eq!(
        line(&held, &aside),
        ["Busy".to_string(), no_room.to_string(), shape.to_string()]
    );

    // Done: all of it again, and the next bar starts from what is up then.
    stand_aside(&mut aside, false, || held.iter());
    assert_eq!(aside, None);
    assert_eq!(line(&held, &aside).len(), 3);
    stand_aside(&mut aside, true, || held.iter());
    assert_eq!(line(&held, &aside), ["Busy"]);
}

/// One line is laid out exactly as the single amber label the status line
/// used to be, so the common case looks as it did; more are a line each, in
/// the same row.
#[test]
fn one_line_is_the_label_it_always_was() {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(400.0, 300.0),
        )),
        ..Default::default()
    };
    let text = "Note: stack has 9 channels; showing the first 6.";
    let one = [StatusLine {
        text: text.into(),
        done: false,
    }];
    let two = [one[0].clone(), one[0].clone()];
    let mut sizes = None;
    // Two passes: the first loads fonts.
    for _ in 0..2 {
        let _ = ctx.run_ui(input.clone(), |ui| {
            let mut row = |add: &dyn Fn(&mut egui::Ui) -> egui::Response| {
                ui.horizontal(|ui| add(ui).rect.size()).inner
            };
            let old =
                row(&|ui| ui.label(egui::RichText::new(text).color(super::STATUS_NOTE).small()));
            let new = row(&|ui| ui.label(status_label(ui, one.to_vec())));
            let both = row(&|ui| ui.label(status_label(ui, two.to_vec())));
            // The label's text is the row's height key: the lines, a line
            // each, as joining them gave.
            let joined = status_label(ui, two.to_vec()).text;
            assert_eq!(joined, format!("{text}\n{text}"));
            sizes = Some((old, new, both));
        });
    }
    let (old, new, both) = sizes.expect("laid out");
    assert_eq!(old, new);
    assert_eq!(both.x, old.x, "{both:?} vs {old:?}");
    assert!(both.y > old.y * 1.5, "two lines {both:?} vs one {old:?}");
}

// ------------------------------------------------- the bar really is wide

/// Lay the info row out in a real (headless) egui context and report the
/// progress bar's rectangle, plus the row's full width.
///
/// Worth doing against the actual widget rather than reasoning about it: the
/// bar spreading across the row is not something the call site states, it is
/// what `egui::ProgressBar` does when no `desired_width` is given — and that is
/// a default in a dependency, which can change under us.
fn lay_out_row(width: f32, status: Option<&str>, with_stop: bool) -> (egui::Rect, f32) {
    let ctx = egui::Context::default();
    let mut bar = None;
    let mut avail = 0.0;
    // Two passes: the first loads fonts, so the second lays text out for real.
    // The window's own size, not `ui.set_width`: the available width a widget
    // sees comes from the screen rect, and a headless context defaults to one
    // far larger than any window.
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, 200.0),
        )),
        ..Default::default()
    };
    for _ in 0..2 {
        let _ = ctx.run_ui(input.clone(), |ui| {
            ui.horizontal(|ui| {
                avail = ui.available_size_before_wrap().x;
                if let Some(s) = status {
                    ui.label(egui::RichText::new(s).small());
                }
                if with_stop {
                    let _ = ui.small_button("✕");
                }
                // The app's own bar, not one built here: what is being
                // checked is that *this call site* spreads.
                bar = Some(ui.add(super::progress_bar("Importing", Some(0.5))).rect);
            });
        });
    }
    (bar.expect("the bar was laid out"), avail)
}

#[test]
fn the_bar_spreads_across_whatever_is_left_of_the_row() {
    let (bar, avail) = lay_out_row(600.0, None, false);
    assert!(
        bar.width() > avail * 0.9,
        "the bar should take the row, not a fixed 120px: {} of {avail}",
        bar.width()
    );

    // A wider row gives a wider bar — the old fixed width did not.
    let (wide, _) = lay_out_row(900.0, None, false);
    assert!(
        wide.width() > bar.width() + 200.0,
        "{} vs {}",
        wide.width(),
        bar.width()
    );
}

/// The status and the stop button are placed first precisely so they survive.
/// If they went after the bar it would have eaten the row and pushed them out.
#[test]
fn the_status_and_stop_button_keep_their_room() {
    let (plain, _) = lay_out_row(600.0, None, false);
    let (crowded, _) = lay_out_row(600.0, Some("Saved something.tif"), true);
    assert!(
        crowded.width() < plain.width(),
        "the bar should yield width to what is beside it: {} vs {}",
        crowded.width(),
        plain.width()
    );
    // But still be the largest thing in the row by a distance.
    assert!(crowded.width() > 250.0, "{}", crowded.width());
    // And start after them rather than under them.
    assert!(crowded.left() > plain.left(), "{crowded:?} {plain:?}");
}

// -------------------------------------------------------- worker containment

/// The worker thread is the only thing that will ever give the registry back
/// and clear the job. If a panic escapes it, the window is left saying
/// "running" for ever with its buttons disabled — strictly worse than the
/// synchronous version this replaced, where a panic at least reached the top.
#[test]
fn a_panicking_worker_becomes_a_message() {
    let err = super::contained("Invert", || panic!("this plugin panics on purpose"))
        .expect_err("a panic must not escape");
    assert!(err.contains("Invert"), "{err}");
    assert!(err.contains("panicked"), "{err}");
    assert!(
        err.contains("this plugin panics on purpose"),
        "the reason must survive: {err}"
    );
}

/// A `String` payload — `panic!("{}", x)` — reads as well as a `&str` one.
#[test]
fn a_formatted_panic_keeps_its_message() {
    let err = super::contained("Saving", || panic!("frame {} is wrong", 3))
        .expect_err("a panic must not escape");
    assert!(err.contains("frame 3 is wrong"), "{err}");
}

/// A payload of some other type still produces something, rather than being
/// silently dropped into an empty reason.
#[test]
fn an_unprintable_panic_still_says_something() {
    let err = super::contained("Exporting", || std::panic::panic_any(7u32))
        .expect_err("a panic must not escape");
    assert!(
        err.contains("Exporting") && err.contains("panicked"),
        "{err}"
    );
}

#[test]
fn work_that_does_not_panic_is_returned_untouched() {
    assert_eq!(super::contained("x", || 41 + 1), Ok(42));
}

// ------------------------------------------------- one door in, one door out

/// `begin_job` clears the status line, and its doc comment claims that cannot
/// be forgotten "at one of the four call sites" — because every job goes
/// through it. That claim is only true while nothing else assigns `self.job`,
/// which no ordinary test can check: it is a property of the source, not of a
/// value. So the source is what this reads.
///
/// There is precedent for this in the repo (`boundary.rs` reads the example
/// plugin's own source to check it stays the oracle it claims to be).
#[test]
fn every_job_goes_through_begin_job() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/app.rs"))
        .expect("app.rs should be readable from its own crate");

    let assignments = src.matches("self.job = Some(").count();
    assert_eq!(
        assignments, 1,
        "`self.job` is assigned in {assignments} places; it must only be set inside \
         `begin_job`, which is what clears the stale status message before the bar \
         appears. Route the new job through `begin_job` instead."
    );

    // And that the one assignment really is the one inside `begin_job`, rather
    // than somewhere else with `begin_job` deleted.
    let begin = src
        .find("fn begin_job")
        .expect("begin_job should still exist");
    let at = src.find("self.job = Some(").expect("the assignment");
    assert!(
        at > begin && at - begin < 600,
        "the assignment is not inside `begin_job`"
    );
    assert!(
        src[begin..at].contains("self.clear_status()"),
        "`begin_job` must clear the status before taking the job"
    );
}

/// The same for the plain-TIFF load, which shows the readout without a `Job`.
#[test]
fn a_plain_load_clears_the_status_too() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/app.rs"))
        .expect("app.rs should be readable");
    let open = src.find("self.core.begin_open(").expect("begin_open call");
    let before = &src[open.saturating_sub(300)..open];
    assert!(
        before.contains("self.clear_status()"),
        "a load shows the progress readout, so it must clear the last message first"
    );
}

/// Whatever this app says next drops the report of a failed open, as it did
/// when the two shared one string: open A, drop a corrupt B, press Ctrl+C,
/// and the line reads "Copied 512 x 512 image" alone, not with "Failed to
/// open file" under it for as long as nothing else starts.
///
/// Checked on the source, as `begin_job` is: every message goes through
/// `say` (or `report_done`, which calls it), and `say` dismisses the
/// failure. A new `self.status = Some(..)` anywhere else is the regression.
#[test]
fn every_message_drops_the_report_of_a_failed_open() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut sources = vec![std::path::PathBuf::from(format!("{dir}/app.rs"))];
    let mut stack = vec![std::path::PathBuf::from(format!("{dir}/app"))];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("the app's sources") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.to_string_lossy().ends_with("_tests.rs")
            {
                sources.push(path);
            }
        }
    }
    let mut writes = Vec::new();
    for path in &sources {
        let src = std::fs::read_to_string(path).expect("a source file");
        for (i, line) in src.lines().enumerate() {
            if line.contains("self.status = Some(") {
                writes.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert_eq!(
        writes.len(),
        1,
        "the app's message is written outside `say` at {writes:?}; call `self.say(..)`, \
         which also drops the report of a failed open"
    );

    let src = std::fs::read_to_string(format!("{dir}/app.rs")).expect("app.rs");
    let say = src.find("fn say(").expect("`say` should still exist");
    let body = &src[say..say + src[say..].find("\n    }").expect("the end of `say`")];
    assert!(
        body.contains("self.status = Some("),
        "`say` is the one write"
    );
    assert!(
        body.contains("self.core.dismiss(NoticeKind::OpenFailed)"),
        "`say` must drop the report of a failed open"
    );
    let done = src.find("fn report_done(").expect("`report_done`");
    let body = &src[done..done + src[done..].find("\n    }").expect("the end")];
    assert!(
        body.contains("self.say("),
        "`report_done` must say through `say`"
    );
}

// ------------------------------------------ what a plugin reports, drawn

/// A plugin that has done a tenth of its work draws a tenth of a bar.
///
/// Tested across the real seam — a `StackHost` wired to a `Job` exactly as
/// `run_plugin` wires them — because each end was right on its own. The host
/// stored basis points and the bar read permille, so every plugin's bar was
/// full by the time it was a tenth done: stabilization reached "100%" in a
/// fraction of a second and sat there for the rest of the run.
#[test]
fn a_plugins_progress_reaches_the_bar_at_the_same_scale() {
    use fasttiff_plugin_api::{HostContext, VolumeMode, VolumeView};

    let opts = fast_tiff_lib::WriterOptions::new(4, 4, fast_tiff_lib::SampleType::U16);
    let mut w = fast_tiff_lib::TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for _ in 0..2 {
        w.write_frame_bytes(&[0u8; 32]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    let stack = fast_tiff_viewer::Document::from_bytes(bytes, "seam.tif".into(), false).unwrap();
    let view = fast_tiff_viewer::plugins::describe_view(
        &stack,
        0,
        false,
        VolumeView {
            mode: VolumeMode::Mip,
            density: 1.0,
            iso: 0.5,
            eye: [0.0; 3],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        },
    );

    let job = Job::new("suite2p stabilization");
    let mut host = fast_tiff_viewer::plugins::StackHost::new(&stack, view)
        .with_cancel(job.cancel.clone(), job.progress.clone());

    assert_eq!(job.fraction(), None, "nothing reported yet is a spinner");
    for f in [0.1f32, 0.3, 0.75, 1.0] {
        assert!(host.progress(f), "an uncancelled run is told to go on");
        let drawn = job
            .fraction()
            .expect("a reported fraction is drawn as a bar");
        assert!(
            (drawn - f).abs() < 0.001,
            "the plugin said {f}, the bar drew {drawn}"
        );
    }
}

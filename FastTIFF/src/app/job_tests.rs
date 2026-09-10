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

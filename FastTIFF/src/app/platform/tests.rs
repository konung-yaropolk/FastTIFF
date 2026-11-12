//! What each host decides, tested for both on whichever machine runs the
//! tests: the platform is a value here ([`Target`]), not the build's.

use super::*;

const IMAGE: egui::Vec2 = egui::vec2(1000.0, 800.0);

// ---------------------------------------------------------------- the build

#[test]
fn this_build_is_the_target_it_was_compiled_for() {
    let expected = if cfg!(target_arch = "wasm32") {
        Target::Browser
    } else {
        Target::Desktop
    };
    assert_eq!(TARGET, expected);
    assert_eq!(WEB, TARGET.is_web());
}

// ------------------------------------------------- how a fresh picture fits

#[test]
fn each_target_fits_a_new_picture_its_own_way() {
    assert_eq!(Target::Desktop.opening(), Opening::SizeWindow);
    assert_eq!(Target::Browser.opening(), Opening::FitCanvas);
}

/// The desktop opens at a fixed ladder level, the largest at most 100% at
/// which the picture and the chrome fit the monitor's work area.
#[test]
fn the_desktop_opens_at_the_largest_level_that_fits_the_monitor() {
    let monitor = Some(egui::vec2(1800.0, 1000.0));
    // Fits at 1:1 with room to spare: 100%, never more.
    assert_eq!(Opening::SizeWindow.zoom(IMAGE, monitor, 60.0), Some(1.0));
    // 800 + 250 of chrome is taller than the monitor; 75% is the next rung.
    assert_eq!(Opening::SizeWindow.zoom(IMAGE, monitor, 250.0), Some(0.75));
    // A mosaic: the first rung at which it fits, on the ladder.
    let big = egui::vec2(40_000.0, 30_000.0);
    let z = Opening::SizeWindow.zoom(big, monitor, 60.0).unwrap();
    assert!(ZOOM_LEVELS.contains(&z), "{z} is not a ladder level");
    assert!(big.x * z <= 1800.0 && big.y * z + 60.0 <= 1000.0);
}

#[test]
fn the_desktop_opens_too_large_a_picture_at_the_smallest_level() {
    let monitor = Some(egui::vec2(800.0, 600.0));
    let huge = egui::vec2(1.0e7, 1.0e7);
    assert_eq!(
        Opening::SizeWindow.zoom(huge, monitor, 60.0),
        Some(ZOOM_LEVELS[0])
    );
}

/// The browser fits the canvas exactly, off the ladder, and never magnifies.
#[test]
fn the_browser_fits_the_canvas_exactly() {
    let canvas = Some(egui::vec2(1600.0, 400.0));
    assert_eq!(Opening::FitCanvas.zoom(IMAGE, canvas, 0.0), Some(0.5));
    let roomy = Some(egui::vec2(4000.0, 4000.0));
    assert_eq!(Opening::FitCanvas.zoom(IMAGE, roomy, 0.0), Some(1.0));
    // The chrome is the canvas's business, not the fit's.
    assert_eq!(Opening::FitCanvas.zoom(IMAGE, canvas, 300.0), Some(0.5));
}

/// Neither fits to a room it does not know the size of yet: the caller waits.
#[test]
fn neither_fits_before_it_knows_the_room() {
    for opening in [Opening::SizeWindow, Opening::FitCanvas] {
        assert_eq!(opening.zoom(IMAGE, None, 60.0), None, "{opening:?}");
    }
    // A canvas laid out at nothing is not a size either.
    assert_eq!(
        Opening::FitCanvas.zoom(IMAGE, Some(egui::Vec2::ZERO), 0.0),
        None
    );
}

// -------------------------------------------------------- the plugin folder

#[test]
fn only_the_desktop_has_a_plugin_folder_to_offer() {
    assert!(Target::Desktop.has_plugin_folder());
    assert!(!Target::Browser.has_plugin_folder());
}

#[test]
fn only_the_desktop_browses_for_a_plugins_path() {
    assert!(Target::Desktop.asks_for_paths());
    assert!(!Target::Browser.asks_for_paths());
}

// ------------------------------------------------- a result's own window

/// A result written to a file is opened from it, in a new process that maps
/// it, and whether the machine could hold a copy is not even asked.
#[test]
fn the_desktop_maps_a_result_in_a_file() {
    let how = Target::Desktop.delivery(true, None, |_| panic!("a file is not copied"));
    assert_eq!(how, Delivery::MapFile);
}

#[test]
fn the_desktop_pipes_a_result_in_memory_when_a_copy_fits() {
    let asked = std::cell::Cell::new(None);
    let how = Target::Desktop.delivery(false, Some(4096), |n| {
        asked.set(Some(n));
        true
    });
    assert_eq!(how, Delivery::SendBytes);
    assert_eq!(
        asked.get(),
        Some(4096),
        "the size asked about is the result's"
    );
}

#[test]
fn the_desktop_refuses_a_result_too_large_to_copy() {
    assert_eq!(
        Target::Desktop.delivery(false, Some(usize::MAX), |_| false),
        Delivery::Refuse(TOO_LARGE)
    );
}

/// A tab is all a browser has, and its memory is the only place a result can
/// be; there is no second process to make room for.
#[test]
fn the_browser_sends_a_result_to_a_new_tab() {
    let how = Target::Browser.delivery(false, Some(usize::MAX), |_| {
        panic!("a tab is not refused for its size")
    });
    assert_eq!(how, Delivery::SendBytes);
}

#[test]
fn a_result_held_nowhere_is_refused_on_either() {
    for target in [Target::Desktop, Target::Browser] {
        assert_eq!(
            target.delivery(false, None, |_| true),
            Delivery::Refuse(NOT_IN_MEMORY),
            "{target:?}"
        );
    }
    // And a browser cannot open a file, whatever it is told.
    assert_eq!(
        Target::Browser.delivery(true, None, |_| true),
        Delivery::Refuse(NOT_IN_MEMORY)
    );
}

// ------------------------------------------------------ where a file goes

#[test]
fn the_desktop_writes_where_a_plugin_said_and_asks_otherwise() {
    assert_eq!(
        Target::Desktop.place("/data/out/turn.tif", "stack-animation.tif".into()),
        Placement::At(Destination::File("/data/out/turn.tif".into()))
    );
    assert_eq!(
        Target::Desktop.place("", "stack-animation.tif".into()),
        Placement::Ask("stack-animation.tif".into())
    );
}

/// A browser is never asked where: it downloads, under the plugin's name
/// without its directories, or under the suggestion.
#[test]
fn the_browser_downloads_and_never_asks() {
    assert_eq!(
        Target::Browser.place("/data/out/turn.tif", "stack-animation.tif".into()),
        Placement::At(Destination::Download("turn.tif".into()))
    );
    assert_eq!(
        Target::Browser.place("", "stack-animation.tif".into()),
        Placement::At(Destination::Download("stack-animation.tif".into()))
    );
}

/// What a plugin asked the host to write is never a question: the run is over
/// and on a worker by then.
#[test]
fn a_plugins_own_file_is_never_asked_about() {
    assert_eq!(
        Target::Desktop.plugin_file(r"C:\work\derivative.tif"),
        Destination::File(r"C:\work\derivative.tif".into())
    );
    assert_eq!(
        Target::Browser.plugin_file(r"C:\work\derivative.tif"),
        Destination::Download("derivative.tif".into())
    );
}

/// The desktop offers the file's own stem back, which its dialog lets the
/// user change; a download cannot be renamed first, so the browser takes the
/// stem at the first dot.
#[test]
fn the_name_offered_for_a_save() {
    let shown = Path::new("scan.ome.tif");
    assert_eq!(Target::Desktop.save_name(shown), "scan.ome.tif");
    assert_eq!(Target::Browser.save_name(shown), "scan.tif");
    assert_eq!(Target::Desktop.save_name(Path::new("")), "stack.tif");
    assert_eq!(Target::Browser.save_name(Path::new("")), "stack.tif");
}

#[test]
fn a_destination_is_reported_by_its_name() {
    assert_eq!(Destination::Download("a.tif".into()).name(), "a.tif");
    assert_eq!(
        Destination::File(Path::new("dir").join("b.tif")).name(),
        Path::new("dir").join("b.tif").display().to_string()
    );
}

// ------------------------------------------------------------ the wording

/// A dialog is promised only where there is one.
#[test]
fn a_save_button_says_what_the_host_will_do() {
    assert_eq!(Target::Desktop.save_label("as TIFF"), "Save as TIFF…");
    assert_eq!(Target::Browser.save_label("as TIFF"), "Download as TIFF");
    assert_eq!(
        Target::Browser.save_label("ImageJ overlay"),
        "Download ImageJ overlay"
    );
}

/// The browser's prompt adds that nothing is uploaded, and the lines the two
/// share are the same lines.
#[test]
fn only_the_browser_promises_nothing_is_uploaded() {
    let desktop = Target::Desktop.welcome_text();
    let browser = Target::Browser.welcome_text();
    assert!(!desktop.contains("uploaded"));
    assert!(browser.contains("no file is ever uploaded"));
    for shared in ["Drag and drop a file here", "Ctrl + Scroll — zoom"] {
        assert!(
            desktop.contains(shared) && browser.contains(shared),
            "{shared}"
        );
    }
}

// ----------------------------------------------------- the rest of the chrome

#[test]
fn the_browser_draws_its_chrome_larger() {
    assert_eq!(Target::Desktop.ui_scale(), 1.0);
    assert!(Target::Browser.ui_scale() > 1.0);
}

#[test]
fn pop_ups_open_where_each_host_can_put_them() {
    assert_eq!(Target::Desktop.popups(), Popups::OwnWindow);
    assert_eq!(Target::Browser.popups(), Popups::InCanvas);
}

/// Only a job on a worker can be watched and stopped while it runs.
#[test]
fn only_the_desktop_runs_a_job_behind_a_live_interface() {
    assert!(Target::Desktop.jobs_run_behind());
    assert!(!Target::Browser.jobs_run_behind());
}

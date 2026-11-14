//! The viewer's `eframe::App`: window chrome, widgets, and input.
//!
//! All non-GUI state and logic lives in `fast_tiff_viewer::Viewer` — the loaded
//! stack, channel settings, playback clock, 3D camera, and the per-frame
//! decode→GPU sync. `ViewerApp` holds one of those plus the things only a
//! window has: its sizing, which panels are open, what the pointer is doing.
//! The rule for where a field belongs is "would a browser UI need this to show
//! the right pixels?" — if yes, it's in `core`. The 2D view's zoom and pan are
//! the core crate's too ([`fast_tiff_viewer::View2d`]), held here beside what
//! the window adds to them (`CanvasView`); the 3D navigation preferences
//! are in the core's volume settings, beside the camera they steer.
//!
//! The GPU is reached only through `crate::render`, the eframe adapter over
//! `scivis-render`, so nothing here mentions glow or wgpu.
//!
//! A frame is `ViewerApp::draw`: the toolbar, the bottom bar, the windows and
//! the canvas, each laid out by a method of its own. What their widgets ask for
//! is queued as a `command::UiCommand` rather than done from inside egui's
//! closures, and applied at the points of the frame `command` documents.
//!
//! Supporting clusters live in child modules (which share this module's
//! privacy, so the split adds no `pub` surface beyond `pub(super)`):
//!   * `animation` — a plugin's animation of the 3D view, played or recorded
//!   * `camera`  — egui input → the core camera
//!   * `canvas`  — the central panel: the picture, in 2D or 3D
//!   * `canvas_tool` — the tools that take the left button on the picture,
//!     and which of them has it
//!   * `command` — the interface's requests, and the order they take effect in
//!   * `minimap` — the 2D navigator: where the view sits in the whole frame
//!   * `overlay` — the 3D coordinate-box overlay, drawn with the egui painter
//!   * `platform` — what the desktop and the browser do differently: the
//!     one place that knows which of the two this is
//!   * `point_counter` — clicking points onto the picture, counted in groups
//!   * `scale`   — how large the chrome is drawn (the web build runs bigger)
//!   * `screen`  — egui's positions and rects as the core's 2D view's, and back
//!   * `scrub_bar` — the bottom bar: playback, channels, contrast, status
//!   * `toolbar` — the top bar: open, save, copy, plugins, 2D/3D
//!   * `widgets` — the contrast range slider + value formatting
//!   * `windows` — the render-settings, metadata and histogram pop-ups

use crate::render::{self, Render};
use egui::{Color32, RichText};
use fast_tiff_viewer::channels::{
    gray_lut_applicable, gray_lut_count, gray_lut_sel_name, gray_lut_sel_tint,
    pseudocolor_applicable,
};
use fast_tiff_viewer::roi::LargeImageMode;
use fast_tiff_viewer::view2d::ZoomStep;
use fast_tiff_viewer::{
    DecodeMode, Document, FrameInputs, Notice, NoticeKind, View2d, ViewMode, Viewer,
};
// The registry's entries are the C-boundary types themselves rather than
// trait objects, so the traits whose methods the app calls are named here.
use fasttiff_plugin_api::{Exporter as _, Importer as _, Plugin as _};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

mod animation;
mod camera;
mod canvas;
mod canvas_tool;
mod clipboard;
mod command;
mod copy_view;
mod dialog;
mod minimap;
mod overlay;
mod platform;
// Drawing a plot a plugin declared, and the canvas selection tool that feeds
// it. On both platforms: the browser build has the Plugins menu too, and a
// plot-producing plugin among its built-ins.
mod plot;
mod plugins_ui;
mod point_counter;
mod region_pick;
mod scale;
mod screen;
mod scroll;
mod scrub_bar;
mod toolbar;
mod widgets;
mod windows;

use canvas_tool::{CanvasGesture, ImageXform, Tools};
use command::{Applying, Commands, Step, UiCommand, Via, FRAME};
// Named in the docs of this module and of its children, which link to the
// phases; the code goes by `FRAME` and never names one.
#[cfg(doc)]
use command::Phase;
use platform::{Current, Destination, Placement, Platform as _, Purpose, TARGET, WEB};
use scale::UI_SCALE;
use screen::{Egui as _, Logical as _};
use widgets::{contrast_controls, histogram_button, ContrastLayout};
use windows::{WindowId, Windows};

/// Install the chrome defaults both hosts share, before the first frame.
///
/// Small enough to inline at each entry point, but the two would then be free to
/// drift — and "the desktop app and the web app look different" is exactly the
/// bug this crate exists to prevent. Anything that styles the app as a whole
/// belongs here.
pub fn install_chrome(ctx: &egui::Context) {
    // Dark by default rather than following the system theme: this is an image
    // viewer, and a light chrome throws stray light onto the canvas, skewing how
    // dim structures in a microscopy stack read. The user can still switch it in
    // egui's own settings.
    ctx.set_theme(egui::ThemePreference::Dark);
    // Bigger on the web than on the desktop — see `scale`.
    ctx.set_zoom_factor(UI_SCALE);
}

/// Turn a core LUT tint (raw RGB, or `None` for "plain grayscale — use the
/// default color") into an egui color. The *decision* is display logic and
/// lives in `fast_tiff_viewer::channels`; only this conversion is egui's.
fn tint_color(tint: Option<[u8; 3]>) -> Option<Color32> {
    tint.map(|[r, g, b]| Color32::from_rgb(r, g, b))
}

/// Fast-scroll rate is a fraction of movie total frames number to be skipped
/// per mouse wheel notch or arrow press when Shift is held. (0.1 means 10% of the stack)
/// Fast-scroll glide speed in *steps per second* (one step is FAST_SCROLL_RATE of the stack).
/// while the Shift+wheel glide decays after a notch, the frame position advances
/// at this rate. Scaling by the real per-frame delta-time — not a flat per-frame
/// amount — makes one notch's jump depend only on the glide's (frame-rate
/// independent) real-time duration, so single- and multi-channel stacks, which
/// render at different speeds, scroll the SAME distance. ~3.75/s reproduces the
/// previous 1/16-per-frame feel at 60 fps; raise/lower it to taste.
const FAST_SCROLL_RATE: f64 = 0.1;
const FAST_SCROLL_GLIDE_RATE: f64 = 5.5;

/// The frame's size in real units, e.g. `"512×512 µm"`, or `None` when the
/// file carries no pixel calibration.
///
/// Both axes are shown even when they are equal: a reader cannot otherwise
/// tell a square field from a rounded one, and anisotropic pixels are common
/// enough in microscopy that collapsing them would be a lie half the time.
///
/// Only rendered when both axes and a unit are present. A number without its
/// unit is not information — it is a number that will be read as whichever
/// unit the reader expects, which for this field is exactly the mistake worth
/// preventing.
///
/// Takes the three metadata values rather than the record they come from:
/// `StackMeta` is `#[non_exhaustive]` and cannot be built outside its own
/// crate, so a signature that asked for one would put this beyond reach of a
/// test — and formatting a measurement for a reader is exactly the kind of
/// thing that should have one.
fn physical_size(
    pixel_width: Option<f64>,
    pixel_height: Option<f64>,
    unit: Option<&str>,
    px_w: u32,
    px_h: u32,
) -> Option<String> {
    let (pw, ph) = (pixel_width?, pixel_height?);
    let unit = unit.filter(|u| !u.is_empty())?;
    if !(pw.is_finite() && ph.is_finite() && pw > 0.0 && ph > 0.0) {
        return None;
    }
    let (w, h) = (pw * px_w as f64, ph * px_h as f64);
    Some(format!("{} × {} {unit}", trim_num(w), trim_num(h)))
}

/// A measurement rounded for reading: enough places to distinguish neighbouring
/// values, with trailing zeros dropped so a round number looks round.
fn trim_num(v: f64) -> String {
    let places = if v >= 100.0 {
        0
    } else if v >= 10.0 {
        1
    } else if v >= 1.0 {
        2
    } else {
        3
    };
    let s = format!("{v:.places$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Read `path` with whichever registered importer claims it, reporting how far
/// it has got into `progress` as a permille.
///
/// A free function rather than a method, and the reason is the whole point of
/// the change around it: this runs on a worker thread, which cannot hold a
/// `&mut ViewerApp`. Everything it needs is what it is given — the registry it
/// borrows for the duration, the path, and somewhere to leave progress — and
/// everything it has to say comes back in the return value.
///
/// The `Err` is a message for the status line. Saying which plugin declined and
/// why beats falling through to the TIFF reader and reporting "not a TIFF".
///
/// An importer reads a path, so only a host with files ever gets here — see
/// [`ViewerApp::start_import`].
fn import_with_plugin(
    registry: &mut fast_tiff_viewer::plugins::Registry,
    path: &std::path::Path,
    progress: &std::sync::atomic::AtomicU32,
    cancel: &std::sync::atomic::AtomicBool,
) -> Result<Opened, String> {
    let head = fast_tiff_viewer::plugins::Registry::read_head(path);
    let (index, _confidence) = *registry
        .importers_for(path, &head)
        .first()
        .ok_or_else(|| format!("nothing here can read {}", path.display()))?;

    let request = fasttiff_plugin_api::ImportRequest {
        path: path.to_path_buf(),
        params: fasttiff_plugin_api::Params::new(),
    };

    struct Host<'a>(
        &'a std::sync::atomic::AtomicU32,
        &'a std::sync::atomic::AtomicBool,
    );
    impl fasttiff_plugin_api::ImportHost for Host<'_> {
        fn progress(&mut self, f: f32) -> bool {
            Job::report(self.0, f);
            !self.1.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn log(&mut self, m: &str) {
            log::info!("import: {m}");
        }
    }

    let entry = registry
        .importer_mut(index)
        .ok_or_else(|| "the importer went away".to_string())?;
    let name = entry.info.name.clone();
    let imported = entry
        .importer
        .import(&request, &mut Host(progress, cancel))
        .map_err(|e| format!("{name}: {e}"))?;

    let fallback = imported
        .info
        .as_ref()
        .map(|i| i.name.clone())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| imported.image.name.clone());
    // The planes go into the encode, and each is freed once written: the
    // import is then held about once, not twice.
    let bytes = fast_tiff_viewer::plugins::into_tiff_bytes(imported.image, imported.info.as_ref())
        .map_err(|e| format!("{name}: {e:#}"))?;
    Ok(Opened::Bytes(bytes.into(), imported_label(path, &fallback)))
}

/// The status line's two voices: amber for something to be aware of, green for
/// something that finished. Both are chosen to stay legible against the light
/// and dark themes rather than to match either one.
const STATUS_NOTE: Color32 = Color32::from_rgb(230, 170, 60);
const STATUS_DONE: Color32 = Color32::from_rgb(120, 195, 120);

/// The toolbar's icons, and the size they are drawn at.
///
/// Every one of these is checked against the bundled fonts by
/// `icon_glyph_tests`. A character the font does not have is not a compile
/// error and not a runtime error: it is a tofu box in the toolbar, which
/// nothing but a person looking at the window would ever notice. `≡` and `🗁`
/// are two that look obvious and are not there.
const ICON_OPEN: &str = "🗁";
const ICON_SAVE: &str = "💾";
const ICON_COPY: &str = "🗐";
const ICON_PLUGINS: &str = "☰";
const ICON_SETTINGS: &str = "⚙";
const ICON_SIZE: f32 = 14.0;

/// What a job sends back once it is done: everything that arrives from
/// somewhere other than this frame's user input, but for files, which come in
/// through the platform ([`platform::Platform::take_incoming_files`]).
enum Incoming {
    /// An import finished. The registry travels back with the result, because
    /// the worker was holding it — see [`ViewerApp::plugins`].
    Imported(
        Box<fast_tiff_viewer::plugins::Registry>,
        Result<Opened, String>,
    ),
    /// A filter plugin finished. The registry travels back for the same reason
    /// an import's does.
    Ran(Box<fast_tiff_viewer::plugins::Registry>, Box<PluginRun>),
    /// A save or an export finished: the file's name, and how it went.
    ///
    /// The registry is `Some` only for an export, which had to borrow it; a
    /// plain TIFF save is the app's own code and never takes it.
    Wrote(
        Option<Box<fast_tiff_viewer::plugins::Registry>>,
        String,
        WriteOutcome,
    ),
}

/// How a save or an export ended.
///
/// Three cases rather than a `Result`, because stopping is not failing and the
/// window should not say "could not save" about a file the user chose not to
/// write. The distinction is drawn from the cancel flag rather than from the
/// error text: an exporter reports being stopped through whatever error its
/// author picked, and matching on strings to find out would be guesswork.
enum WriteOutcome {
    Done,
    Cancelled,
    Failed(String),
}

impl WriteOutcome {
    fn of(result: Result<(), String>, cancel: &std::sync::atomic::AtomicBool) -> Self {
        match result {
            Ok(()) => WriteOutcome::Done,
            Err(_) if cancel.load(std::sync::atomic::Ordering::Relaxed) => WriteOutcome::Cancelled,
            Err(e) => WriteOutcome::Failed(e),
        }
    }
}

/// What a filter plugin produced, carried back from its worker.
struct PluginRun {
    /// Which plugin, so a plot it returned can be recomputed for a new
    /// selection without asking the user again.
    index: usize,
    /// Which document the run was on, as [`ViewerApp::document`] counted it
    /// when the run started; see [`outlived`](Self::outlived).
    run_on: u64,
    /// What its dialog was answered with, reused verbatim on a re-run.
    params: fasttiff_plugin_api::Params,
    name: String,
    /// What the plugin logged, drained from its host when the run ended.
    messages: Vec<String>,
    outcome: Result<fast_tiff_viewer::plugins::Product, String>,
    /// The file the document the run was on stands for, so a result shown
    /// in this window is said to be made from it — whatever is on show by
    /// the time it lands.
    made_from: Option<PathBuf>,
}

impl PluginRun {
    /// Whether the run returned something drawn on the picture it read — a
    /// plot of its regions, or a rectangle for the user to draw — and another
    /// document has landed since it started, `now` being the window's count.
    ///
    /// A new document puts every canvas tool away ([`Tools::reset`]), and a
    /// result that lands later must not bring one back: it would put one
    /// picture's chart, or its prompt, over another. On the desktop a run can
    /// still be going when the next file opens — a file no importer claims
    /// does not wait for the registry the run holds. (In a browser the run
    /// ends inside the frame that started it.)
    ///
    /// Anything else was either decided on the worker already (a replacement
    /// goes to a new window once the document has changed:
    /// `Deliver::same_document`) or is news whatever is on show.
    fn outlived(&self, now: u64) -> bool {
        use fast_tiff_viewer::plugins::Product;
        self.run_on != now
            && matches!(
                self.outcome,
                Ok(Product::Plot(_) | Product::SelectRegion(_))
            )
    }
}

/// The app's half of delivering a plugin's result: the job's bar, and the
/// platform's way of opening a window or saving a file. Everything before
/// that last step is `fast_tiff_viewer::plugins::deliver`, shared with the
/// browser build.
struct AppDelivery<'a> {
    label: &'a std::sync::Mutex<String>,
    progress: &'a std::sync::atomic::AtomicU32,
    cancel: &'a std::sync::atomic::AtomicBool,
    same_document: &'a dyn Fn() -> bool,
}

impl fast_tiff_viewer::plugins::Deliver for AppDelivery<'_> {
    fn phase(&mut self, label: &str) {
        Job::begin_phase(self.label, self.progress, label);
    }

    fn report(&mut self, fraction: f32) -> bool {
        Job::report(self.progress, fraction);
        !self.cancel.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn same_document(&self) -> bool {
        (self.same_document)()
    }

    /// A process on the desktop, a tab in a browser: see
    /// [`Platform::deliver_new_document`](platform::Platform::deliver_new_document).
    fn open_in_new_window(
        &mut self,
        result: fast_tiff_viewer::plugins::ResultTiff,
    ) -> fast_tiff_viewer::plugins::Handover {
        // Named after the image, so the new window's title says what it is
        // showing rather than naming a scratch file.
        let name = format!(
            "{}.tif",
            fast_tiff_viewer::plugins::file_stem(result.name())
        );
        Current::deliver_new_document(result, name, self.progress, self.cancel)
    }

    /// Where the plugin said, on the desktop. In a browser the download
    /// manager is the filesystem, and the plugin's path still names the file —
    /// the last component of it — because the desktop honours that name and
    /// the two should not disagree about what the result is called; what a tab
    /// cannot honour is the directory, and it does not pretend to.
    fn save(
        &mut self,
        result: fast_tiff_viewer::plugins::ResultTiff,
        path: &str,
    ) -> Result<String, String> {
        match TARGET.plugin_file(path) {
            Destination::File(file) => {
                Current::write_result(result, &file).map(|()| path.to_string())
            }
            Destination::Download(name) => {
                let bytes = result.into_bytes().map_err(|e| e.to_string())?;
                Current::download(&bytes, &name, "image/tiff").map(|()| name)
            }
        }
    }
}

/// Progress that has not been reported yet, as distinct from nought reported.
///
/// A plugin is not obliged to call `progress` at all — most short ones do not —
/// and a bar sitting at 0% for the whole run reads as "stuck" rather than as
/// "no estimate". This sentinel is what makes the readout a spinner instead.
///
/// Defined next to the encoding plugin hosts write with, so the two cannot
/// disagree — see `fast_tiff_viewer::plugins::progress`.
const PROGRESS_UNKNOWN: u32 = fast_tiff_viewer::plugins::progress::UNKNOWN;

/// Run a worker's work, turning a panic into a message instead of a dead thread.
///
/// The thread is the only thing holding the registry, and the only thing that
/// will ever clear [`ViewerApp::job`]. If it unwinds out from under us the
/// window is left saying "running" for ever, with the Plugins and Save buttons
/// disabled and no way back short of a restart — strictly worse than the
/// synchronous version this replaced, where a panic at least reached the top.
///
/// A plugin's own panics are already caught at the C boundary — every plugin's,
/// the built-ins included, since they are registered through it too — and come
/// back as an error naming the plugin. This is for the app's own code around
/// the call: encoding the result, handing it to a window.
fn contained<T>(what: &str, work: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).map_err(|e| {
        // The payload is `&str` for `panic!("...")` and `String` for a
        // formatted one; anything else is a custom type nobody can print.
        let why = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_string());
        format!("{what} panicked: {why}")
    })
}

/// What is written over the progress bar: how far along, then what is running.
///
/// One line, over the bar rather than beside it, so the bar can spread across
/// the row instead of sharing it. A stage with no knowable length has no
/// percentage to give and reads as the label alone.
fn progress_text(label: &str, fraction: Option<f32>) -> String {
    match fraction {
        Some(f) => format!("{:.0}%  {label}", (f.clamp(0.0, 1.0)) * 100.0),
        None => label.to_string(),
    }
}

/// The status bar's progress bar.
///
/// No `desired_width`, which is what makes it spread: `egui::ProgressBar` then
/// takes `available_size_before_wrap().x`, so it fills whatever is left of the
/// row. Everything that has to stay visible — the status message, the stop
/// button — is therefore placed *before* it.
///
/// `animate` when there is no fraction: it says "working" without inventing a
/// number, and the repaints its animation asks for are already being requested
/// while a job is running.
fn progress_bar(label: &str, fraction: Option<f32>) -> egui::ProgressBar {
    egui::ProgressBar::new(fraction.unwrap_or(0.0))
        .animate(fraction.is_none())
        .text(progress_text(label, fraction))
}

/// What the info row's *height* depends on, as a string to compare frames by.
///
/// The percentage is deliberately absent. This key is what
/// [`PanelLayout::note_info_row`] compares one frame against the next to decide
/// whether the row changed height and the window should grow — and a string
/// that changed every frame would arm that on every frame, giving a window that
/// creeps taller for the whole length of a load. Only the things that can
/// actually reflow the row belong here: whether each part is present, and the
/// text whose length decides how many lines it wraps to.
fn info_key(status: Option<&str>, label: Option<&str>) -> String {
    let mut key = String::new();
    if let Some(status) = status {
        key.push_str(status);
    }
    if let Some(label) = label {
        key.push('\u{1}');
        key.push_str(label);
    }
    key
}

/// One line of the status row, and whether it reads as *done* (green) rather
/// than as something to be aware of (amber).
#[derive(Clone, Debug, PartialEq)]
struct StatusLine {
    text: String,
    done: bool,
}

/// What the status row says: this app's last message (`message`, green when
/// it is the one recorded as `good`), then what the core has to say of the
/// document and of the view on screen (`notices`, see [`Viewer::notices`]),
/// one line each, leaving out those standing `aside` for a progress bar (see
/// [`stand_aside`]).
///
/// The core's notes are its own kinds and none replaces another or this
/// app's message, so a refusal or a save no longer wipes the note about the
/// file's shape.
fn status_lines<'a>(
    message: Option<&str>,
    good: Option<&str>,
    notices: impl IntoIterator<Item = &'a Notice>,
    aside: &[Notice],
) -> Vec<StatusLine> {
    let mut lines: Vec<StatusLine> = message
        .map(|m| StatusLine {
            text: m.to_string(),
            done: good == Some(m),
        })
        .into_iter()
        .collect();
    lines.extend(
        notices
            .into_iter()
            .filter(|n| !aside.contains(n))
            .map(|n| StatusLine {
                text: n.to_string(),
                done: false,
            }),
    );
    lines
}

/// Keep `aside` for this frame: while something is `working`, the core's
/// notices that were already up when its bar appeared (the first frame it is
/// `working`), and none once it is done.
///
/// The row is then the progress bar's, and a note about the document on
/// show, said before the load of another or a save began, would read as a
/// comment on it, as an old message would (which is why starting one clears
/// this app's, see [`ViewerApp::begin_job`]). Those stand aside, still held,
/// and come back when the work is done. What the core finds while the work
/// runs is news, and shows beside the bar as it always did: a volume that
/// ran out of GPU memory during a long save says so then, not after it. So
/// does a note that went away and came back: one is kept aside only while it
/// is still `shown` (the notices for the view on screen, as
/// [`Viewer::notices`] lists them).
fn stand_aside<'a, I: Iterator<Item = &'a Notice>>(
    aside: &mut Option<Vec<Notice>>,
    working: bool,
    shown: impl Fn() -> I,
) {
    if !working {
        *aside = None;
    } else if let Some(aside) = aside {
        aside.retain(|n| shown().any(|m| m == n));
    } else {
        *aside = Some(shown().cloned().collect());
    }
}

/// The status row's text: a line each, each in its own voice, as the one
/// label the row has room for before the progress bar — one line laid out
/// exactly as a single `RichText` label of it is. Its `text` is the lines
/// joined by line breaks.
fn status_label(ui: &egui::Ui, lines: Vec<StatusLine>) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    for line in lines {
        if let Some(last) = job.sections.last_mut() {
            // The break ends the line before, in its voice. Every line is in
            // the same small font, so which line it belongs to lays nothing
            // out differently.
            job.text.push('\n');
            last.byte_range.end = job.text.len();
        }
        let color = if line.done { STATUS_DONE } else { STATUS_NOTE };
        RichText::new(line.text).color(color).small().append_to(
            &mut job,
            ui.style(),
            egui::FontSelection::Default,
            ui.text_valign(),
        );
    }
    job
}

/// A slow thing running on a worker: what to call it, how far it has got, and
/// how to ask it to stop.
///
/// One type for all four — importing, running a filter, saving, exporting —
/// because the window shows them identically and only ever runs one at a time.
/// The alternative was four near-identical structs and four branches in the
/// status bar.
///
/// Progress is a permille in an atomic rather than an `f32` because it crosses
/// threads and `f32` has no atomic form. A thousandth of a file is finer than
/// any progress bar can draw.
///
/// **No new plugin API carries it.** `HostContext::progress` and its cancel
/// return already exist and are already what a plugin calls; this is just the
/// other end of them, which used to be a local nobody could see.
struct Job {
    /// What to print over the bar: "Importing", "Saving", the plugin's name.
    ///
    /// Shared and mutable, because a job has phases worth telling apart. A
    /// plugin run measures and then *writes* the result — and on a long
    /// recording the write is the longer half — so one label with one bar would
    /// reach 100% at the end of the measuring and sit there until the file
    /// landed. Which is exactly what it did.
    label: std::sync::Arc<std::sync::Mutex<String>>,
    progress: std::sync::Arc<std::sync::atomic::AtomicU32>,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Job {
    /// Start one, with nothing reported yet.
    fn new(label: impl Into<String>) -> Self {
        Job {
            label: std::sync::Arc::new(std::sync::Mutex::new(label.into())),
            progress: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(PROGRESS_UNKNOWN)),
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// What the bar should say now.
    fn label(&self) -> String {
        self.label
            .lock()
            .map(|l| l.clone())
            .unwrap_or_else(|e| e.into_inner().clone())
    }

    /// Move to a named phase, with its progress starting again from nothing.
    ///
    /// A free function over the shared handles, so a worker holding only those
    /// can call it.
    fn begin_phase(
        label: &std::sync::Mutex<String>,
        progress: &std::sync::atomic::AtomicU32,
        name: impl Into<String>,
    ) {
        match label.lock() {
            Ok(mut l) => *l = name.into(),
            Err(e) => *e.into_inner() = name.into(),
        }
        fast_tiff_viewer::plugins::progress::clear(progress);
    }

    /// How far along, or `None` when the worker has not said — which is a
    /// spinner rather than a bar.
    fn fraction(&self) -> Option<f32> {
        fast_tiff_viewer::plugins::progress::load(&self.progress)
    }

    /// Record a fraction reported by a worker. Free function shape, so a worker
    /// that only has the `Arc` can call it.
    fn report(progress: &std::sync::atomic::AtomicU32, fraction: f32) {
        fast_tiff_viewer::plugins::progress::store(progress, fraction);
        Current::wake_for_progress();
    }
}

/// The interface's egui context, for a worker to wake it with.
///
/// A worker cannot draw, and nothing makes the interface look at a job but a
/// frame. Before this, a frame was asked for on every vsync for as long as any
/// job was alive: a ten-minute registration was some 36,000 full redraws, each
/// the whole of `draw`, to move a bar a percent at a time — a core taken from
/// the registration, and a GPU kept out of idle. Now a frame comes when there
/// is something to show: a result arriving ([`wake_ui`]) or the bar moving
/// ([`Platform::wake_for_progress`](platform::Platform::wake_for_progress)),
/// with a slow poll as a safety net.
///
/// Recorded on the first frame, before any worker can exist.
static UI: std::sync::OnceLock<egui::Context> = std::sync::OnceLock::new();

/// Ask the interface for a frame, from any thread: a result has arrived.
fn wake_ui() {
    if let Some(ctx) = UI.get() {
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    }
}

/// A file the user asked to open, however it arrived.
///
/// Native and web get files by different routes — a path from a dialog, argv or
/// an Apple Event, versus bytes from an async browser picker or a drop event —
/// so they meet here and everything downstream is shared.
pub enum Opened {
    /// A path on disk. Only the desktop has any to give, and only the
    /// desktop's platform makes a source of one (`Platform::path_source`).
    Path(PathBuf),
    /// The file's bytes plus its name, for display.
    ///
    /// The browser build's only route, there being no filesystem there.
    /// Natively it is how a plugin result reaches a new window when no file
    /// could be written for it and it was handed over in memory instead.
    ///
    /// Owned or shared, and never copied on the way to the reader: a browser
    /// drop arrives as egui's `Arc<[u8]>`, and the document keeps that buffer
    /// (see [`fast_tiff_lib::Bytes`]).
    Bytes(fast_tiff_lib::Bytes, String),
    /// A plugin's result, to be opened as it was written: straight to the
    /// reader, past the importers — it is a TIFF this app wrote, and an
    /// importer claiming `.tif` has no business reading it back.
    Source(fast_tiff_viewer::LoadSource),
}

/// The 2D view, and what this app adds to it: a window sized around the
/// picture, and the navigator's hold on a drag.
///
/// The view itself — the zoom and pan, the opening fit, the glides, the
/// autoscroll anchor, wheel travel, and where all that puts the picture — is
/// the core's [`View2d`], arithmetic in logical points that the canvas feeds
/// this frame's input and turns into the sync's
/// [`ImageInputs`](fast_tiff_viewer::ImageInputs). What is here is the
/// window's.
#[derive(Default)]
struct CanvasView {
    /// The view: see [`View2d`].
    model: View2d,
    /// Set when something (the opening fit, or a zoom step) wants the window
    /// resized to match the zoom this frame. Applied once, then cleared. The
    /// window is only resized in response to such an event — never every
    /// frame — so the user can freely resize or maximize it; between those
    /// events the image scales to fit, aspect locked.
    resize_to_zoom: bool,
    /// A zoom step the window has still to follow, keeping the point under
    /// the cursor put once the new layout is known.
    zoom_reposition: Option<ZoomStep>,
    /// Whether the drag in flight began on the navigator, and is therefore
    /// moving the view rather than dragging the picture.
    ///
    /// Sticky for the life of the gesture rather than re-tested each frame, and
    /// it has to be, in both directions. The pointer leaves the map the moment
    /// the view box is dragged to an edge, and the drag must carry on. And a
    /// drag begun on the picture that happens to pass over the map must not
    /// start teleporting the view half way through.
    minimap_drag: bool,
}

impl CanvasView {
    /// Forget the picture that was open, now that another has landed.
    ///
    /// The view resets itself ([`View2d::reset`]: it keeps the last layout and
    /// wheel travel short of a frame, and waits for the opening fit). Of the
    /// window's part, a resize goes with the picture it was for, and so does a
    /// navigator drag, which would carry on moving a view of an image that is
    /// no longer open. A zoom reposition is kept: a zoom step sets it and the
    /// same frame's window sizing takes it (natively — the browser never reads
    /// it).
    fn reset(&mut self) {
        self.model.reset();
        self.resize_to_zoom = false;
        self.minimap_drag = false;
    }
}

/// How many frames to keep looking for the bar's height to change, once a
/// grow has been armed. Two would do — the change lands on the frame after the
/// one that armed it — and the third is slack for a frame that egui decides to
/// lay out differently.
const GROW_WAIT: u8 = 3;

/// Layout bookkeeping for the collapsible channels/contrast panel, which grows
/// and shrinks the window by its own height delta when toggled.
#[derive(Default)]
struct PanelLayout {
    /// Channel buttons + contrast sliders are tucked under a small triangle
    /// toggle to keep the bar minimal by default.
    expanded: bool,
    /// Frames left to wait for the bar's height to actually change, after
    /// something asked the window to grow by the difference.
    ///
    /// A countdown rather than a flag, because the wait is not guaranteed to
    /// end. The toggle needs one: the frame the click arrives still draws the
    /// bar in its old state, so the delta only becomes visible on the next. But
    /// a status message replaced by another of the same height changes nothing
    /// to measure, and a flag would sit armed for a delta that never comes —
    /// repainting every frame while it waited, which costs a core for as long
    /// as the window is open.
    grow_armed: u8,
    /// The panel's height *before* the toggle, for that delta.
    old_h: f32,
    /// The bar's height as it was last drawn.
    ///
    /// The toggle can use the height in front of it as "before", because the
    /// bar is still drawn in its old state on the frame the click arrives. A
    /// status message is not: by the time this frame's height can be measured
    /// the message is already in it, so the height to compare against is the
    /// one from the frame before.
    last_h: f32,
    /// What the bar's info row was last laid out with — the progress label and
    /// the status message, or `None` when it drew neither and the row was not
    /// there at all.
    ///
    /// Compared as text rather than as a flag because a longer message wraps to
    /// more lines, and that is a height change too.
    last_info: Option<String>,
}

impl PanelLayout {
    /// Start over for a new document. The panel is sized and populated for the
    /// previous file's channel count, so it is rebuilt wholesale. That also
    /// disarms a toggle caught in flight, whose remembered height belongs to a
    /// layout that no longer exists and would resize the window by a stale
    /// delta.
    fn reset(&mut self) {
        *self = PanelLayout::default();
    }

    /// Note what the bottom bar's info row just drew, and arm a window grow if
    /// it changed the bar's height.
    ///
    /// Called every frame, after the bar has been laid out. The height compared
    /// against is the previous frame's, because unlike the panel toggle — which
    /// is clicked on a frame the bar is still drawn in its old state — the row
    /// is already in the height being measured by the time it can be noticed.
    fn note_info_row(&mut self, drawn: Option<String>, height: f32) {
        if self.last_info != drawn {
            self.last_info = drawn;
            // Not over a toggle still waiting for its own delta, which measured
            // a height this frame has already left behind; and not before the
            // bar has been drawn once, when there is no previous height to have
            // grown from.
            if self.grow_armed == 0 && self.last_h > 0.0 {
                self.grow_armed = GROW_WAIT;
                self.old_h = self.last_h;
            }
        }
        self.last_h = height;
    }

    /// Arm a grow against the height in front of us, for a change that has not
    /// happened yet — the panel toggle, whose new size only appears next frame.
    ///
    /// What is armed is read by the half of this bookkeeping that only a host
    /// sizing its own window has (`grow_delta` and `waiting`, in
    /// `app/platform/native.rs`): in a browser the window is the page's.
    fn arm_grow(&mut self, height: f32) {
        self.grow_armed = GROW_WAIT;
        self.old_h = height;
    }
}

pub struct ViewerApp {
    /// Everything that isn't GUI: the loaded document, channel settings, playback
    /// clock, 3D camera, and the decode→GPU sync. See `fast_tiff_viewer`.
    core: Viewer,
    /// GPU textures/shader for compositing the image, shared with the paint
    /// callback. Created once at startup (see `crate::render::init`).
    render: Render,
    /// Where the image sits in the viewport (pan/zoom and the geometry derived
    /// from it), and the window's part in it.
    view: CanvasView,
    /// The collapsible channels panel's layout bookkeeping.
    panel: PanelLayout,
    /// The plugins, importers and exporters, unless a job currently has them.
    /// On the desktop, the built-ins and the libraries installed on disk; in a
    /// browser, the built-ins alone, there being no disk to search and no way
    /// to load a library. See `fast_tiff_viewer::plugins`.
    ///
    /// Moved to the worker rather than shared behind a lock. A lock would have
    /// to be taken by the menu on every frame, and taken *while the import
    /// holds it* — which is the whole duration of the slow thing this exists to
    /// get off the interface thread. Handing the registry over and taking it
    /// back makes "busy" a state the code can see rather than a wait it cannot.
    plugins: Option<fast_tiff_viewer::plugins::Registry>,
    /// What [`plugins`](Self::plugins) claims to import, by extension, kept for
    /// when the registry is with a job: a file none of them wants is the TIFF
    /// reader's either way, and opening it need not wait for the job to end.
    claimed_extensions: Vec<String>,
    /// Bumped each time a document lands in this window. A plugin run notes it
    /// when it starts, so a result meant to replace the document it ran on can
    /// tell that the user has opened another one since — and go to a window of
    /// its own instead of over the one the user is now looking at.
    document: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The plugin whose dialog is open, and the values entered so far.
    plugin_dialog: Option<PluginDialog>,
    /// Where each job's result is sent once it is done ([`Incoming`]).
    /// Drained once a frame by [`drain_pending_open`](Self::drain_pending_open).
    /// Files come in through the platform instead, the browser's asynchronous
    /// picker included.
    open_tx: Sender<Incoming>,
    open_rx: Receiver<Incoming>,
    /// The slow thing running on a worker thread, while one is: an import, a
    /// filter plugin, a save or an export.
    ///
    /// One at a time, which is also the answer to "block the data from other
    /// plugins while one is processing": there is no second worker to race, and
    /// while this is `Some` the Plugins and Save buttons are disabled. For an
    /// import, a plugin run or an export its presence is what makes
    /// [`plugins`](Self::plugins) `None` — the worker has the registry and
    /// gives it back with the result.
    job: Option<Job>,
    /// The canvas tools: a plot's regions, a rectangle a plugin asked for, the
    /// point counter.
    tools: Tools,
    /// The rectangle a plugin asked for, drawn and confirmed, waiting for the
    /// run it is the selection of: which plugin, and the regions.
    ///
    /// A run's input rather than one of the [`Tools`], and not reset with
    /// them: it is set only as it is handed to [`run_plugin`](Self::run_plugin),
    /// which takes it.
    picked: Option<(usize, Vec<fasttiff_plugin_api::Roi>)>,

    // --- window chrome ------------------------------------------------------
    /// The desktop or the browser: what this app does that the two hosts do
    /// their own way. See [`platform`].
    platform: Current,
    /// What the last thing this app did has to say: a refusal, a plugin's
    /// outcome, a save. Shown on the status line first, above what the core
    /// has to say of the document ([`Viewer::notices`]), which this neither
    /// replaces nor is replaced by. Written only through
    /// [`say`](Self::say), which also drops the report of a failed open.
    status: Option<String>,
    /// The status message that should read as *done*, not as *careful*.
    ///
    /// The status line is amber, which is right for what it usually carries: a
    /// refusal, a plugin's error, the core's notes. It is wrong for "opened it
    /// in a new window", where amber reads as a warning about work that in fact
    /// succeeded.
    ///
    /// Held as the message itself rather than as a flag beside it.
    /// [`status`](Self::status) is written from a dozen places — every plugin
    /// outcome, every refusal — and a flag set here would still be set once one
    /// of them replaced the text underneath it, colouring somebody else's
    /// message green. Comparing the text means the colour can only ever apply
    /// to the message it was set for.
    good_status: Option<String>,
    /// The core's notices standing aside while a progress bar is up: those
    /// that were up when it appeared. See [`stand_aside`].
    notices_aside: Option<Vec<Notice>>,
    /// What this app decides each frame for the core's
    /// [`sync`](Viewer::sync): the part of the 2D image on screen, the size of
    /// the 3D view and how finely to render it, an animation's pose, whether
    /// the view is moving and the histograms are open.
    ///
    /// Written where each is decided, as the frame is laid out, and handed to
    /// the sync once it has been. Kept from one frame to the next, and not
    /// reset with a document, as the core kept them before they were inputs:
    /// a frame that does not lay a view out leaves what the last one that did
    /// decided.
    sync_inputs: FrameInputs,
    /// The pop-up windows — metadata, histogram, render settings — and which
    /// are open.
    windows: Windows,
    /// What the frames between need to remember about the 3D view.
    volume_ui: VolumeUi,
    /// A plugin's animation of the 3D view, playing or being recorded. See
    /// [`animation`].
    animation: animation::Animations,
}

/// The 3D view's side of the frontend: what one frame leaves for the next
/// about it. The user's preferences for it are the core's, beside the camera
/// ([`VolumeSettings::navigation`](fast_tiff_viewer::VolumeSettings::navigation)).
///
/// Nothing here is reset when a document lands: it describes the panel and
/// the camera, which are still there.
#[derive(Default)]
struct VolumeUi {
    /// The 3-D panel's rect, recorded as it is painted.
    ///
    /// Copying the 3-D view is a screenshot of the window cropped to this, and
    /// the crop happens a frame or more after the request — so the rect has to
    /// be remembered rather than asked for at crop time, when the panel may not
    /// be being laid out at all.
    rect: Option<egui::Rect>,
    /// Set when a screenshot has been asked for in order to copy the 3-D view,
    /// holding the rect to cut out of it. `None` when no copy is outstanding,
    /// which is also what keeps an unrelated screenshot from being copied.
    pending_copy: Option<egui::Rect>,
    /// Set when the pointer is pressed on the 3D view, and taken by the next
    /// frame's animation step: grabbing a turning volume is how people stop
    /// it, whether or not the grab goes on to move anything.
    pressed: bool,
    /// The 3D camera as it was in the last 3D frame, and that frame's number
    /// (`cumulative_frame_nr`), to tell a frame that moved it.
    last_camera: Option<(animation::CameraMark, u64)>,
    /// When the user last moved the 3D view (the frontend clock, seconds).
    /// Until [`VolumeNavPrefs::SETTLE_S`] after it, the view renders at
    /// reduced size.
    ///
    /// [`VolumeNavPrefs::SETTLE_S`]: fast_tiff_viewer::camera::VolumeNavPrefs::SETTLE_S
    moved_at: Option<f64>,
}

/// A plugin's dialog, while it is open.
struct PluginDialog {
    /// Index into the registry's plugin list.
    index: usize,
    title: String,
    decls: Vec<fasttiff_plugin_api::ParamDecl>,
    values: fasttiff_plugin_api::Params,
}

/// The window title for a file an importer converted.
///
/// The source file's own name, extension and all, marked as converted. What
/// reaches the viewer is TIFF bytes, so without saying so the title would claim
/// a `.tif` that exists nowhere — and the extension is worth keeping, since
/// which vendor format this came from is the first thing anyone asks.
/// `fallback` — the plugin's own name for the image — covers an importer that
/// synthesises rather than reads, where the path names no file.
fn imported_label(path: &std::path::Path, fallback: &str) -> String {
    match path.file_name() {
        Some(n) => format!("{} - imported", n.to_string_lossy()),
        None => fallback.to_string(),
    }
}

impl ViewerApp {
    pub fn new(initial: Option<Opened>, render: Render) -> Self {
        let (open_tx, open_rx) = channel();
        let mut app = Self {
            core: Viewer::new(),
            render,
            view: CanvasView::default(),
            panel: PanelLayout::default(),
            plugins: Some(Current::registry()),
            plugin_dialog: None,
            claimed_extensions: Vec::new(),
            document: Default::default(),
            open_tx,
            open_rx,
            job: None,
            tools: Tools::default(),
            picked: None,
            platform: Current::new(),
            status: None,
            good_status: None,
            notices_aside: None,
            sync_inputs: FrameInputs::default(),
            windows: Windows::new(),
            volume_ui: VolumeUi::default(),
            animation: Default::default(),
        };
        app.claimed_extensions = app
            .plugins
            .as_ref()
            .map(|r| r.claimed_extensions())
            .unwrap_or_default();
        Current::sweep_old_results();
        if let Some(opened) = initial {
            app.apply_opened(opened);
        }
        app
    }

    /// Hand an opened file to the core, then reset the chrome that described
    /// the previous one. The core records any failure as a notice, which the
    /// bottom bar shows, and keeps the old stack loaded.
    fn apply_opened(&mut self, opened: Opened) {
        // Started, not done: opening walks the whole IFD chain and then decodes
        // a frame per channel to find its display range, which on a large stack
        // is seconds. Doing that here would be seconds of a window that does not
        // repaint or respond. The core runs it on a thread and this returns at
        // once; `finish_open` picks the result up when it lands.
        //
        // The chrome is deliberately *not* reset here. It still describes the
        // file on screen, and that file stays on screen and usable for as long
        // as the new one takes to arrive.
        // An importer plugin claims some extensions; a file it recognises is
        // read by the plugin and handed on as TIFF bytes, so everything
        // downstream — and every later save — sees an ordinary document.
        //
        // On a worker, and not because importing is merely slow: an OIR is
        // gigabytes of scattered chunks to reassemble, which is tens of seconds
        // of a window that does not repaint, does not respond, and that the
        // desktop offers to kill. The core already opens files this way and
        // draws a progress readout for it; an import is the same promise.
        let opened = match opened {
            Opened::Path(path) => match self.start_import(path) {
                Ok(()) => return,
                Err(path) => Opened::Path(path),
            },
            other => other,
        };

        let source = match opened {
            Opened::Path(path) => match Current::path_source(path) {
                Ok(source) => source,
                Err(why) => {
                    self.say(why);
                    return;
                }
            },
            Opened::Bytes(bytes, name) => {
                fast_tiff_viewer::LoadSource::Bytes(bytes, PathBuf::from(name))
            }
            Opened::Source(source) => source,
        };
        // Same rule as a worker job: the readout is about to appear, and what
        // the last action said is not about this one. See `begin_job`.
        self.clear_status();
        self.core.begin_open(source);
    }

    /// Start showing a job, clearing what the last one left behind.
    ///
    /// The status line and the progress bar share a row, so a message from
    /// earlier would sit beside the new bar looking like a comment on it —
    /// "Saved field_1.tif" next to an import that has nothing to do with it,
    /// or worse, an error from the previous run beside a bar that is going
    /// fine. The old message is spent the moment new work starts.
    ///
    /// Every job goes through here so that cannot be forgotten at one of the
    /// four call sites.
    fn begin_job(
        &mut self,
        job: Job,
    ) -> (
        std::sync::Arc<std::sync::atomic::AtomicU32>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
        std::sync::Arc<std::sync::Mutex<String>>,
    ) {
        self.clear_status();
        let handles = (job.progress.clone(), job.cancel.clone(), job.label.clone());
        self.job = Some(job);
        handles
    }

    /// Drop the last thing said, message and green-ness together, and the
    /// core's report of the last open: what the last attempt said, rather
    /// than what is true of the document on show, which stays.
    ///
    /// Both, always: `good_status` is what makes the text green, and leaving it
    /// set while replacing the text would paint the next message green whether
    /// or not it reported success.
    fn clear_status(&mut self) {
        self.clear_message();
        self.core.dismiss(NoticeKind::OpenFailed);
    }

    /// Drop this app's own message, and its green-ness with it, leaving what
    /// the core has to say.
    fn clear_message(&mut self) {
        self.status = None;
        self.good_status = None;
    }

    /// Report something that finished, in the voice for it.
    ///
    /// The status line is amber, which is right for a dimension note or a
    /// refusal and wrong for work that succeeded. Green is asked for by
    /// recording the message itself rather than setting a flag beside it:
    /// `status` is written from a dozen places, and a flag would still be set
    /// once one of them replaced the text underneath it.
    fn report_done(&mut self, message: String) {
        self.good_status = Some(message.clone());
        self.say(message);
    }

    /// Say `message` on the status line, in place of this app's last one.
    ///
    /// Every message of this app's goes through here or
    /// [`report_done`](Self::report_done), because saying anything is work
    /// the user has moved on to: it also drops the core's report of the last
    /// open ([`NoticeKind::OpenFailed`], see [`Viewer::dismiss`]), which
    /// beside "Copied 512 x 512 image" would read as a comment on the copy.
    /// The core's notes about the document on show stay.
    fn say(&mut self, message: String) {
        self.status = Some(message);
        self.core.dismiss(NoticeKind::OpenFailed);
    }

    /// Reset everything that described the previous file, once a load has
    /// landed. The other half of [`apply_opened`](Self::apply_opened).
    ///
    /// Each part of the chrome resets itself: what it keeps and what it drops
    /// is its own business, written down where its fields are.
    fn finish_open(&mut self) {
        self.document
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The landing is the news now, said by the core (the new document's
        // notes, or why it did not open): whatever this app said while it was
        // on its way is not about it.
        self.clear_message();
        self.view.reset();
        self.tools.reset();
        self.panel.reset();
        // The pop-ups mostly stay up on the desktop, repointed at the new
        // document; each says for itself (`AppWindow::outlives_document`).
        self.windows.reset(&self.core, WEB);
    }

    /// Apply whatever arrived since the last frame ([`Incoming`]): the result
    /// of a job that finished.
    fn drain_pending_open(&mut self) {
        while let Ok(message) = self.open_rx.try_recv() {
            match message {
                Incoming::Imported(registry, result) => {
                    // The registry comes home first, so the app is usable again
                    // whether the import worked or not.
                    self.plugins = Some(*registry);
                    self.job = None;
                    match result {
                        Ok(opened) => self.apply_opened(opened),
                        Err(message) => self.say(message),
                    }
                }
                Incoming::Ran(registry, run) => {
                    self.plugins = Some(*registry);
                    self.job = None;
                    self.apply_plugin_run(*run);
                }
                Incoming::Wrote(registry, name, result) => {
                    if let Some(r) = registry {
                        self.plugins = Some(*r);
                    }
                    self.job = None;
                    match result {
                        WriteOutcome::Done => self.report_done(format!("Saved {name}")),
                        WriteOutcome::Cancelled => self.say(format!("Stopped saving {name}")),
                        WriteOutcome::Failed(e) => self.say(format!("Could not save {name}: {e}")),
                    }
                }
            }
        }
    }

    /// Begin a plugin: show its dialog, or run it at once if it declared none.
    fn start_plugin(&mut self, index: usize) {
        let Some(loaded) = self.core.document.as_ref() else {
            self.say("Open an image first".into());
            return;
        };
        // A plain TIFF save does not take the registry, so the menu being
        // reachable is not on its own proof that nothing is running.
        if self.job.is_some() {
            self.say(self.busy_message());
            return;
        }
        // The dialog may depend on what is open — how many slices there are,
        // whether there is more than one channel — so it is asked for now
        // rather than cached at load.
        let host = fast_tiff_viewer::plugins::StackHost::new(loaded, self.plugin_view(loaded))
            .with_animating(self.animating_for(index));
        let Some(entry) = self.plugins.as_ref().and_then(|p| p.entries().get(index)) else {
            return;
        };
        let title = entry.info.name.clone();
        // The fallible form: a dialog that failed is reported, and the plugin
        // is not run. Asking `params` instead would read the failure as "no
        // controls" and start the run at once, with none of the settings the
        // plugin meant to ask for.
        let decls = match entry.plugin.dialog(&host) {
            Ok(d) => d,
            Err(e) => {
                self.say(format!("{title}: {e}"));
                return;
            }
        };

        if decls.is_empty() {
            self.run_plugin(index, fasttiff_plugin_api::Params::new());
        } else {
            let values = fasttiff_plugin_api::Params::defaults(&decls);
            self.plugin_dialog = Some(PluginDialog {
                index,
                title,
                decls,
                values,
            });
        }
    }

    /// The viewer state a plugin sees, snapshotted for one run.
    fn plugin_view(&self, loaded: &fast_tiff_viewer::Document) -> fasttiff_plugin_api::ViewParams {
        let v = &self.core.volume;
        let (forward, right, up) = v.cam.basis();
        fast_tiff_viewer::plugins::describe_view(
            loaded,
            loaded.frame(),
            self.core.view_mode == ViewMode::Volume,
            fast_tiff_viewer::plugins::describe_volume(
                match v.render {
                    fast_tiff_viewer::VolumeRender::Alpha => 1,
                    fast_tiff_viewer::VolumeRender::Surface => 2,
                    _ => 0,
                },
                v.density,
                v.iso,
                v.cam.eye(forward),
                forward,
                up,
                right,
            ),
        )
    }

    /// Draw the open plot, and act on what it asks for.
    ///
    /// A re-run goes through the ordinary plugin path — same worker, same
    /// progress readout, same stop button — so a measurement over a long
    /// timelapse behaves like any other slow plugin rather than freezing the
    /// window every time a region is drawn.
    fn poll_plot(&mut self, ctx: &egui::Context) {
        let Some(win) = self.tools.plot.as_mut() else {
            return;
        };
        let busy = self.job.is_some();
        match plot::plot_window(ctx, win, busy) {
            plot::PlotAction::Close => self.tools.plot = None,
            plot::PlotAction::Rerun => win.stale = true,
            plot::PlotAction::None => {}
        }
        // Deferred to here rather than done inside the window: a run needs the
        // registry, which the window's borrow of `self.tools.plot` would still hold.
        let Some(win) = self.tools.plot.as_ref() else {
            return;
        };
        if !win.stale {
            return;
        }
        match &self.job {
            // A measurement is already running and its answer is now the wrong
            // one — the user has drawn another region since. Stop it rather
            // than queue behind it: a five-thousand-frame trace takes seconds,
            // and waiting out a result nobody wants any more is the difference
            // between a tool that follows the pointer and one that lags a
            // gesture behind.
            //
            // The flag is all this does. The worker still owns the registry and
            // still has to hand it back; the fresh run starts on the frame
            // after it does.
            Some(job) => job.cancel.store(true, std::sync::atomic::Ordering::Relaxed),
            None if self.plugins.is_some() => {
                let (index, params) = (win.plugin, win.params.clone());
                self.tools.plot.as_mut().expect("still open").stale = false;
                self.run_plugin(index, params);
            }
            None => {}
        }
    }

    /// Draw the open plugin dialog, and run the plugin when it is accepted.
    fn poll_plugin_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.plugin_dialog.take() else {
            return;
        };
        let before = dialog.values.clone();
        // A panel's buttons wait while something else has the plugins: a press
        // now could only be refused as busy.
        let busy = self.job.is_some() || self.plugins.is_none();
        match plugins_ui::dialog(ctx, &dialog.title, &dialog.decls, &mut dialog.values, busy) {
            Some(plugins_ui::DialogAnswer::Run) => {
                // Clamp to what the plugin declared, so `run` never sees a
                // value it already ruled out.
                let values = dialog.values.clamp_to(&dialog.decls);
                self.run_plugin(dialog.index, values);
            }
            Some(plugins_ui::DialogAnswer::Cancel) => {}
            // A button: a run with the values as they stand and that button's
            // key set, and the panel stays where it is for the next press.
            Some(plugins_ui::DialogAnswer::Press(key)) => {
                let mut values = dialog.values.clamp_to(&dialog.decls);
                values.set(key, fasttiff_plugin_api::ParamValue::Bool(true));
                let index = dialog.index;
                self.plugin_dialog = Some(dialog);
                self.run_plugin(index, values);
            }
            None => {
                if dialog.values != before {
                    self.requery_dialog(&mut dialog);
                }
                self.plugin_dialog = Some(dialog);
            }
        }
    }

    /// Ask the plugin for its dialog again, now that a value has changed.
    ///
    /// This is what makes a dialog follow the choices made in it: a plugin
    /// sees what has been chosen through `HostContext::pending_params` and
    /// may declare a different set of controls for it — the iteration count
    /// only for the methods that iterate, the coverslip only for the model
    /// that uses it.
    ///
    /// Only on a change, not every frame. The cost is one `params` call, but
    /// a plugin is entitled to do real work in it (`Generate PSF` reads the
    /// stack's calibration) and sixty of those a second for a dialog nobody
    /// is touching would be waste.
    ///
    /// What the user has already set is kept: `clamp_to` takes the existing
    /// value for every key still declared and the plugin's default for any
    /// that has just appeared. Values for controls that went away stay in
    /// `values` untouched, so going back to a method restores what was set
    /// for it rather than resetting it.
    fn requery_dialog(&mut self, dialog: &mut PluginDialog) {
        let Some(loaded) = self.core.document.as_ref() else {
            return;
        };
        let host = fast_tiff_viewer::plugins::StackHost::new(loaded, self.plugin_view(loaded))
            .with_pending_params(dialog.values.clone())
            .with_animating(self.animating_for(dialog.index));
        let Some(entry) = self
            .plugins
            .as_ref()
            .and_then(|p| p.entries().get(dialog.index))
        else {
            return;
        };
        // A dialog that fails to re-declare keeps the controls it had, and says
        // why: dropping them would leave the user an empty dialog whose OK
        // runs the plugin with nothing set.
        let decls = match entry.plugin.dialog(&host) {
            Ok(d) => d,
            Err(e) => {
                self.say(format!("{}: {e}", dialog.title));
                return;
            }
        };
        if decls == dialog.decls {
            return;
        }
        // Seed anything newly declared, without disturbing what is set. The
        // merge is onto `values` rather than onto the clamped copy, so a
        // control that disappears and comes back comes back as it was.
        for (k, v) in fasttiff_plugin_api::Params::defaults(&decls).iter() {
            if dialog.values.get(k).is_none() {
                dialog.values.set(k, v.clone());
            }
        }
        dialog.decls = decls;
    }

    /// What to say when something slow is already running.
    fn busy_message(&self) -> String {
        match &self.job {
            Some(job) => format!("Still busy: {}", job.label().to_lowercase()),
            // The registry is away but no job holds it, which should not happen
            // and is worth saying plainly rather than silently ignoring.
            None => "Busy".to_string(),
        }
    }

    /// Run a plugin on a worker thread.
    ///
    /// The host is built here, on the interface thread, because that is where
    /// the stack is — and then moved. It can be moved because it *owns* what it
    /// reads: a second handle on the same memory-mapped file, not a copy of it.
    /// See `fast_tiff_viewer::plugins::StackHost`.
    ///
    /// The registry goes with it, exactly as an import's does, which is what
    /// stops a second plugin starting while this one has the data.
    fn run_plugin(&mut self, index: usize, values: fasttiff_plugin_api::Params) {
        let Some(loaded) = self.core.document.as_ref() else {
            self.say("Open an image first".into());
            return;
        };
        // Reachable with a job already running: the dialog may have been left
        // open while a save was started behind it. Taking the registry is not
        // enough of a check on its own — a plain save never takes it.
        if self.job.is_some() {
            self.say(self.busy_message());
            return;
        }
        let Some(registry) = self.plugins.take() else {
            self.say(self.busy_message());
            return;
        };
        let Some(name) = registry.entries().get(index).map(|e| e.info.name.clone()) else {
            self.plugins = Some(registry);
            return;
        };

        let made_from = loaded.original();
        let view = self.plugin_view(loaded);
        // The regions belong to the plot this plugin is feeding, if it has one.
        // A different plugin gets none — a selection drawn for one measurement
        // is not an input to an unrelated one.
        //
        // Or the rectangle the plugin asked to have drawn, which is its alone.
        let picked = self
            .picked
            .take()
            .filter(|(plugin, _)| *plugin == index)
            .map(|(_, rois)| rois);
        let selection = picked.unwrap_or_else(|| {
            self.tools
                .plot
                .as_ref()
                .filter(|p| p.plugin == index)
                .map(|p| p.rois.clone())
                .unwrap_or_default()
        });
        let job = Job::new(name.clone());
        // A result is written as it arrives — to a file of its own on the
        // desktop, to memory in a browser — so a registration's result is
        // never whole in memory on its way to a window. It is written with the
        // source's own tables, so a result that keeps its channels opens
        // looking like the document it came from: those of `view`, taken now
        // with the rest of the run's inputs rather than when the result lands
        // — the user may change a LUT while a long run is going, and the
        // result belongs to the state the run started in.
        let host = fast_tiff_viewer::plugins::StackHost::new(loaded, view)
            .with_selection(selection)
            .with_animating(self.animating_for(index))
            .with_cancel(job.cancel.clone(), job.progress.clone())
            .streaming_to(fast_tiff_viewer::plugins::ResultDest::platform_default());
        let (progress, cancel, label) = self.begin_job(job);

        let tx = self.open_tx.clone();
        // Which document this ran on, to tell if it is still the one on show
        // when the result arrives (see `Deliver::same_document`).
        let document = std::sync::Arc::clone(&self.document);
        let run_on = document.load(std::sync::atomic::Ordering::Relaxed);
        // The run itself, as one closure, so that the only difference between
        // the platforms is whether a thread carries it.
        let work = move || {
            let mut registry = registry;
            let mut host = host;
            let outcome = contained(&name, || match registry.get_mut(index) {
                Some(entry) => entry.plugin.run(&mut host, &values),
                // Cannot happen — the index was checked before the spawn and
                // the registry has been on this thread since.
                None => Err(fasttiff_plugin_api::PluginError::failed(
                    "the plugin went away",
                )),
            })
            .unwrap_or_else(|e| Err(fasttiff_plugin_api::PluginError::failed(e)));
            // Finishing the result and handing it to a window happen here, on
            // the worker, under their own label. They used to happen on the
            // interface thread once the result came back — which is why the bar
            // reached 100% and the window then froze until it was done.
            let outcome = match outcome {
                Ok(o) => fast_tiff_viewer::plugins::deliver::finish(
                    o,
                    &mut host,
                    &mut AppDelivery {
                        label: &label,
                        progress: &progress,
                        cancel: &cancel,
                        same_document: &|| {
                            document.load(std::sync::atomic::Ordering::Relaxed) == run_on
                        },
                    },
                ),
                Err(e) => {
                    host.discard_result();
                    Err(e.to_string())
                }
            };
            let run = PluginRun {
                index,
                run_on,
                params: values,
                name,
                messages: std::mem::take(&mut host.messages),
                outcome,
                made_from,
            };
            let _ = tx.send(Incoming::Ran(Box::new(registry), Box::new(run)));
            wake_ui();
        };
        // A thread on the desktop; in a browser, here and now, on the frame
        // that asked — which is also what lets a result open in a new tab. See
        // `Platform::spawn_job` for why, and what it costs.
        Current::spawn_job(work);
    }

    /// Apply what a plugin returned, once its worker has handed it back.
    fn apply_plugin_run(&mut self, run: PluginRun) {
        for m in &run.messages {
            log::info!("{}: {m}", run.name);
        }
        if run.outlived(self.document.load(std::sync::atomic::Ordering::Relaxed)) {
            self.say(format!(
                "{}: not shown, as another image was opened while it ran",
                run.name
            ));
            return;
        }
        let PluginRun {
            index,
            run_on: _,
            params,
            name,
            messages: _,
            outcome,
            made_from,
        } = run;

        use fast_tiff_viewer::plugins::Product;
        match outcome {
            Ok(Product::Plot(plot)) => {
                match self.tools.plot.as_mut().filter(|p| p.plugin == index) {
                    // A re-run for a changed selection: keep the window, the
                    // tool and the regions exactly where they are and swap the
                    // chart. Rebuilding it would move a window the user placed.
                    Some(open) => {
                        open.plot = *plot;
                        open.stale = false;
                    }
                    None => self.tools.plot = Some(plot::PlotWindow::new(index, params, *plot)),
                }
            }
            Ok(Product::Nothing) => {}
            // A run stopped to make way for a newer selection is not news:
            // the user drew a region, they did not cancel anything. Saying
            // "cancelled" for their own edit would be the tool complaining
            // about being used.
            Ok(Product::Cancelled)
                if self
                    .tools
                    .plot
                    .as_ref()
                    .is_some_and(|p| p.plugin == index && p.stale) => {}
            Ok(Product::Cancelled) => self.say(format!("{name}: cancelled")),
            Ok(Product::Message(m)) => self.say(format!("{name}: {m}")),
            // The window is already on its way; there is nothing left to do
            // here but say so.
            Ok(Product::NewWindow(what)) => {
                self.report_done(format!("{name}: opened {what} in a new window"))
            }
            Ok(Product::Inline { result, why }) => {
                self.say(format!(
                    "{name}: could not open a new window ({why}); showing it here"
                ));
                self.open_result(&name, result, made_from);
            }
            Ok(Product::Replace(result)) => {
                self.report_done(format!("{name}: replaced this window's image"));
                self.open_result(&name, result, made_from);
            }
            Ok(Product::Saved(path)) => self.report_done(format!("{name}: wrote {path}")),
            Ok(Product::Animate(a)) => self.start_animation(index, &name, *a),
            Ok(Product::StopAnimation) => self.stop_animation_of(index),
            Ok(Product::SelectRegion(prompt)) => {
                self.start_region_pick(index, params, &name, prompt)
            }
            Err(e) => self.say(format!("{name}: {e}")),
        }
    }

    /// Show a plugin's result in this window: the file it was written to,
    /// mapped as any opened file is, or its bytes where it was held in memory.
    ///
    /// The document it opens as is the result's own: named after it, owning
    /// its scratch file (deleted when the document is closed), and made from
    /// `made_from`, the file of the document the run was on — so "Save as…"
    /// starts in that one's folder, and a plugin run on the result is not
    /// told the scratch file is its source.
    fn open_result(
        &mut self,
        plugin: &str,
        result: fast_tiff_viewer::plugins::ResultTiff,
        made_from: Option<PathBuf>,
    ) {
        match result.into_document(made_from) {
            Ok(source) => self.apply_opened(Opened::Source(source)),
            Err(e) => self.say(format!("{plugin}: could not open its result: {e}")),
        }
    }

    /// Ask the open dialog's plugin for its dialog again, if it is `plugin`'s —
    /// because something it reads has changed underneath it: an animation of
    /// its started or stopped, and its Play/Stop button has to say so.
    fn requery_dialog_of(&mut self, plugin: usize) {
        let Some(mut dialog) = self.plugin_dialog.take() else {
            return;
        };
        if dialog.index == plugin {
            self.requery_dialog(&mut dialog);
        }
        self.plugin_dialog = Some(dialog);
    }

    /// Hand `path` to an importer plugin on a worker thread, if one claims it.
    ///
    /// `Ok(())` means an import is running and the caller is done; the result
    /// arrives through [`drain_pending_open`](Self::drain_pending_open). `Err`
    /// hands the path back, unclaimed, for the ordinary TIFF reader.
    ///
    /// Reached only with a path, which only the desktop has: a browser's
    /// files arrive as bytes and go straight to the reader.
    fn start_import(&mut self, path: std::path::PathBuf) -> Result<(), std::path::PathBuf> {
        // One import at a time. The registry is with the worker, and a second
        // import would have nothing to read the file with — better to say so
        // than to hand an OIR to the TIFF reader and report that it is not a
        // TIFF. A file no importer claims needs no importer, though: it is the
        // TIFF reader's whether a job is running or not, and refusing it made
        // every file unopenable for the length of an hour-long filter.
        let Some(registry) = self.plugins.take() else {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_lowercase);
            if ext.is_none_or(|e| !self.claimed_extensions.contains(&e)) {
                return Err(path);
            }
            self.say(self.busy_message());
            return Ok(());
        };
        self.claimed_extensions = registry.claimed_extensions();
        // Cheap rejection first: no file is opened unless an extension matches,
        // so the common case of a TIFF costs one string compare — and costs no
        // thread.
        if !registry.claims_extension(&path) {
            self.plugins = Some(registry);
            return Err(path);
        }

        let (progress, cancel, _label) = self.begin_job(Job::new("Importing"));
        let tx = self.open_tx.clone();
        Current::spawn_job(move || {
            let mut registry = registry;
            let result = contained("the importer", || {
                import_with_plugin(&mut registry, &path, &progress, &cancel)
            })
            .unwrap_or_else(Err);
            // If the send fails the window is gone, and so is anyone who wanted
            // the result.
            let _ = tx.send(Incoming::Imported(Box::new(registry), result));
            wake_ui();
        });
        Ok(())
    }

    /// Write the open stack where the user wants it: asked where on the
    /// desktop, handed to the downloads in a browser ([`Target::place`]).
    ///
    /// The desktop's dialog blocks the interface, which is right — it is a
    /// modal, and nothing else should be happening while it is up. The *write*
    /// does not: it is a decode-and-encode of every plane, seconds on a large
    /// stack, and it runs on a worker against its own handle on the same
    /// memory-mapped file. See `fast_tiff_viewer::save::SaveSource`.
    ///
    /// [`Target::place`]: platform::Target::place
    fn save_as(&mut self) {
        let Some(stack) = self.core.document.as_ref() else {
            self.say("Open an image first".into());
            return;
        };
        if self.job.is_some() {
            self.say(self.busy_message());
            return;
        }
        // What the window is titled with: the file's own name, the name an
        // imported stack is shown under, the name a browser document arrived
        // under — with a `.tif` on it.
        let suggested = TARGET.save_name(&stack.path);
        let destination = match TARGET.place("", suggested) {
            Placement::At(destination) => destination,
            Placement::Ask(name) => {
                // Every installed exporter's formats, beside the host's own
                // TIFF. TIFF is first because it is the only one that keeps the
                // stack whole; the rest are for handing a picture to something
                // else.
                let mut filters =
                    vec![fasttiff_plugin_api::FileType::new("TIFF", &["tif", "tiff"])];
                filters.extend(
                    self.plugins
                        .as_ref()
                        .map(|p| p.save_file_types())
                        .unwrap_or_default(),
                );
                let dir = stack.save_dir();
                match Current::ask_path(Purpose::Save, Some(&name), dir.as_deref(), &filters) {
                    Some(path) => Destination::File(path),
                    // Cancelled: not a failure, and not worth a status line
                    // either.
                    None => return,
                }
            }
        };
        match destination {
            Destination::File(path) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| path.display().to_string());
                // The extension decides, not the filter row that was selected:
                // the dialogs disagree about whether choosing a filter rewrites
                // the name, and the name is the thing the user can see. An
                // extension no exporter claims is a TIFF, which is also what an
                // extension-less name gets.
                match self.exporter_for(&path) {
                    Some(index) => self.start_export(index, path, name),
                    None => self.start_save(path, name),
                }
            }
            Destination::Download(name) => self.download_tiff(name),
        }
    }

    /// Write the open stack as a TIFF, on a worker.
    fn start_save(&mut self, path: std::path::PathBuf, name: String) {
        let Some(stack) = self.core.document.as_ref() else {
            return;
        };
        let source = fast_tiff_viewer::save::SaveSource::of(stack);
        let (progress, cancel, _label) = self.begin_job(Job::new("Saving"));

        let tx = self.open_tx.clone();
        Current::spawn_job(move || {
            let result = contained("the writer", || {
                Current::write_tiff(&source, &path, &mut |f| {
                    Job::report(&progress, f);
                    !cancel.load(std::sync::atomic::Ordering::Relaxed)
                })
            })
            .unwrap_or_else(Err);
            let outcome = WriteOutcome::of(result, &cancel);
            // The registry was never taken: a plain save is the app's own code.
            let _ = tx.send(Incoming::Wrote(None, name, outcome));
            wake_ui();
        });
    }

    /// Encode the open stack and hand it to the browser's downloads as `name`.
    ///
    /// Shorter than the desktop's save because both of the things that make
    /// that one long are gone. There is no dialog to put up — a tab cannot ask
    /// where a file should go, and the download manager asks instead, if the
    /// user has told it to — and there is no worker to put the write on.
    ///
    /// # It runs on the frame that asked for it
    ///
    /// Same as a job on this target, for the same reason (see
    /// `Platform::spawn_job`). So the interface does not repaint while the
    /// encode runs, which means a progress bar would never be seen and Cancel
    /// could never be clicked — hence no [`Job`] at all here, rather than one
    /// that reports to nothing. The status line afterwards is the whole of what
    /// this says.
    ///
    /// A large stack will therefore hold the tab for as long as the encode
    /// takes. That is a real limit of this target and not a small one; it is
    /// also why the desktop path was written to stream in the first place.
    ///
    /// # Only TIFF
    ///
    /// The exporter lane does not reach here. An [`Exporter`] is handed a path
    /// and writes the file itself — that is its contract — and a browser has no
    /// path to hand it. Offering PNG in a menu that could only fail would be
    /// worse than not offering it, so the web build registers the PNG plugin as
    /// an importer and this button writes TIFF.
    ///
    /// [`Exporter`]: fasttiff_plugin_api::Exporter
    fn download_tiff(&mut self, name: String) {
        let Some(stack) = self.core.document.as_ref() else {
            return;
        };
        let source = fast_tiff_viewer::save::SaveSource::of(stack);
        // Nothing to report progress to and nothing that could set the cancel
        // flag, so the callback is a constant rather than a pretence.
        let wrote = fast_tiff_viewer::save::save_to_bytes(&source, &mut |_| true)
            .map_err(|e| format!("{e:#}"))
            .and_then(|bytes| Current::download(&bytes, &name, "image/tiff"));
        match wrote {
            // "Saved" rather than "Downloaded": where it went is the browser's
            // business, and the user asked to save their image.
            Ok(()) => self.report_done(format!("Saved {name}")),
            Err(e) => self.say(format!("Could not save {name}: {e}")),
        }
    }

    /// Put the view on the clipboard as an image.
    ///
    /// Which picture that is depends on the mode, and the two are genuinely
    /// different requests rather than one with a switch in it:
    ///
    /// * **2-D** copies the *image* — the current frame at its own size, every
    ///   enabled channel through its own window and LUT, composed on the CPU.
    ///   Not what is on screen: that is at some zoom and pan, and on a stack too
    ///   large to upload whole it is only a window of the frame.
    /// * **3-D** copies the *view*, because for a ray-march there is nothing
    ///   else to copy. There is no original size, and the camera and projection
    ///   are the picture. So it screenshots the panel, which cannot finish this
    ///   frame — the backend reads a screenshot back after painting — and
    ///   completes in [`poll_copy`](Self::poll_copy).
    ///
    /// Shared by the toolbar button and the hotkey, so the two cannot drift.
    fn copy_view(&mut self, ctx: &egui::Context) {
        let Some(stack) = self.core.document.as_ref() else {
            self.say("Open an image first".into());
            return;
        };
        if self.job.is_some() {
            self.say(self.busy_message());
            return;
        }
        if self.core.view_mode == ViewMode::Volume {
            // Nothing has been painted yet on the frame a file opens in, so
            // there is no view to grab. Saying so beats copying a blank.
            let Some(rect) = self.volume_ui.rect else {
                self.say("The 3D view has not been drawn yet".into());
                return;
            };
            copy_view::request_volume(ctx);
            self.volume_ui.pending_copy = Some(rect);
            return;
        }
        match fast_tiff_viewer::snapshot::of_frame(stack) {
            Ok(snap) => {
                let (w, h) = (snap.width, snap.height);
                copy_view::put(ctx, copy_view::image_of(&snap));
                self.report_done(format!("Copied {w} x {h} image"));
            }
            Err(e) => self.say(format!("Could not copy the view: {e:#}")),
        }
    }

    /// Finish a 3-D copy once the screenshot lands.
    ///
    /// Only acts while a copy is outstanding: a screenshot taken for any other
    /// reason is none of this function's business, and copying it would put a
    /// picture on the clipboard that nobody asked for.
    fn poll_copy(&mut self, ctx: &egui::Context) {
        let Some(rect) = self.volume_ui.pending_copy else {
            return;
        };
        let Some(shot) = copy_view::take_screenshot(ctx) else {
            return;
        };
        // Taken, whatever happens next: a screenshot that cannot be cropped is
        // not going to become croppable on the following frame, and leaving the
        // request standing would copy the next unrelated screenshot instead.
        self.volume_ui.pending_copy = None;
        match copy_view::crop(&shot, rect, ctx.pixels_per_point()) {
            Some(image) => {
                let (w, h) = (image.width(), image.height());
                copy_view::put(ctx, image);
                self.report_done(format!("Copied {w} x {h} view"));
            }
            None => self.say("The 3D view is too small to copy".into()),
        }
    }

    /// Hand the open stack to an exporter, on a worker.
    ///
    /// The registry goes with it, as an import's and a filter's do, which is
    /// what keeps a second job from starting while this one holds the data.
    ///
    /// An exporter writes to a path, so this is reached only with a file to
    /// write: the desktop's.
    fn start_export(&mut self, index: usize, path: std::path::PathBuf, name: String) {
        let Some(loaded) = self.core.document.as_ref() else {
            return;
        };
        let Some(registry) = self.plugins.take() else {
            self.say(self.busy_message());
            return;
        };
        let Some(plugin_name) = registry.exporters().get(index).map(|e| e.info.name.clone()) else {
            self.plugins = Some(registry);
            return;
        };

        let view = self.plugin_view(loaded);
        let job = Job::new(format!("Exporting ({plugin_name})"));
        let host = fast_tiff_viewer::plugins::StackHost::new(loaded, view)
            .with_cancel(job.cancel.clone(), job.progress.clone());
        let (_, cancel, _label) = self.begin_job(job);

        let tx = self.open_tx.clone();
        Current::spawn_job(move || {
            let mut registry = registry;
            let mut host = host;
            let result = contained(&plugin_name, || match registry.exporter_mut(index) {
                Some(entry) => {
                    // The declared dialog is not shown — the values are the
                    // exporter's own defaults. An export runs from a save dialog
                    // that has already been answered, and a second modal on top
                    // of it needs the cross-frame state machine the filter
                    // plugins use. Importers are in the same position today;
                    // whichever gets a dialog first should give both one.
                    // Failing to declare it fails the export, rather than
                    // writing a file with none of the exporter's defaults.
                    let decls = entry
                        .exporter
                        .dialog(&host)
                        .map_err(|e| format!("{plugin_name}: {e}"))?;
                    let request = fasttiff_plugin_api::ExportRequest {
                        path,
                        params: fasttiff_plugin_api::Params::defaults(&decls).clamp_to(&decls),
                    };
                    entry
                        .exporter
                        .export(&request, &mut host)
                        .map_err(|e| format!("{plugin_name}: {e}"))
                }
                None => Err("that exporter is no longer installed".to_string()),
            })
            .unwrap_or_else(Err);
            for m in &host.messages {
                log::info!("{plugin_name}: {m}");
            }
            let outcome = WriteOutcome::of(result, &cancel);
            let _ = tx.send(Incoming::Wrote(Some(Box::new(registry)), name, outcome));
            wake_ui();
        });
    }

    /// Which exporter, if any, claims this name.
    fn exporter_for(&self, path: &std::path::Path) -> Option<usize> {
        self.plugins.as_ref()?.exporter_for(path)
    }

    /// Show the platform's file picker, offering what the importers read
    /// beside TIFF, and open what it gives back now. A picker that answers
    /// later — the browser's — hands its file in through the platform's
    /// incoming files instead (see [`intake`](Self::intake)).
    fn show_open_dialog(&mut self, ctx: &egui::Context) {
        let readable = self
            .plugins
            .as_ref()
            .map(|p| p.open_file_types())
            .unwrap_or_default();
        if let Some(opened) = self.platform.pick_file(ctx, &readable) {
            self.apply_opened(opened);
        }
    }
}

impl ViewerApp {
    /// The whole of one frame. Kept apart from [`eframe::App::ui`] so that
    /// what must happen after every frame happens however this one ends.
    ///
    /// The frame is [`FRAME`], taken a step at a time. The panels and windows
    /// do not act on what is clicked: they queue it as a [`UiCommand`], and
    /// each [`Step::Apply`] carries out what is due at that point. [`Phase`]
    /// says why each of those points is where it is, and [`UiCommand::slot`]
    /// which command lands at which.
    fn draw(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let cmds = Commands::default();
        // What a step leaves for the steps after it.
        let mut pinching = false;
        let mut toolbar = None;
        let mut bar: Option<scrub_bar::Bar> = None;
        for step in FRAME {
            match step {
                Step::Intake => self.intake(&ctx),
                Step::ZoomInput => pinching = self.zoom_input(ui),
                Step::Toolbar => toolbar = Some(self.toolbar(ui, &cmds)),
                Step::ToolWindows => self.tool_windows(&ctx),
                Step::Hotkeys => self.hotkeys(&ctx, &cmds),
                Step::PollCopy => self.poll_copy(&ctx),
                Step::ScrubBar => bar = Some(self.scrub_bar(ui, &cmds)),
                Step::Popups => self.popups(&ctx),
                Step::NoteInfoRow => {
                    // A status message lands in the same bar as the panel, and
                    // pushes the image up in exactly the same way — so it gets
                    // the same answer: the window grows by the message's height
                    // instead of the canvas shrinking under it. Losing thirty
                    // pixels of canvas is enough to put an image that was
                    // fitted to the window into pan mode, which is a strange
                    // thing for "saved it" to do.
                    //
                    // Armed here rather than where the status is set, because a
                    // status can be set from anywhere — including a worker
                    // thread — and what matters is not that it changed but that
                    // the bar in front of the user is now a different height.
                    let bar = bar.as_mut().expect("FRAME lays the bar out first");
                    let height = bar.height();
                    self.panel.note_info_row(bar.drawn_info.take(), height);
                }
                Step::RunPlayback => {
                    let bar = bar.as_ref().expect("FRAME lays the bar out first");
                    self.run_playback(&ctx, bar.scrubber_held);
                }
                Step::SettleView => self.settle_view(&ctx),
                Step::Canvas => self.canvas(ui, pinching, &cmds),
                Step::EndFrame => {
                    let toolbar = toolbar.as_ref().expect("FRAME lays the toolbar out first");
                    let bar = bar.as_ref().expect("FRAME lays the bar out first");
                    self.end_frame(ui, toolbar, &bar.response);
                }
                Step::Apply(phase) => {
                    // The bar's height, from the phase after it on.
                    let at = Applying {
                        ctx: &ctx,
                        bar_height: bar.as_ref().map(scrub_bar::Bar::height),
                    };
                    self.apply_due(phase, &cmds, &at);
                }
            }
        }
        cmds.finish();
    }

    /// Take in what arrived since the last frame: files dropped on the window
    /// or handed over by the system or a picker, what the workers sent, and a
    /// load that finished — and keep frames coming while one is still on its
    /// way.
    fn intake(&mut self, ctx: &egui::Context) {
        // Paths on the desktop, several of which open the first here and the
        // rest in processes of their own; bytes in a browser. See
        // `Platform::take_incoming_files`.
        for opened in self.platform.take_incoming_files(ctx) {
            self.apply_opened(opened);
        }
        self.drain_pending_open();
        // Take delivery of a load that finished since the last frame, and keep
        // frames coming while one is still running — the worker cannot wake the
        // interface by itself, and without this the progress readout would only
        // move when the mouse did.
        if self.core.poll_open() {
            self.finish_open();
        }
        // A worker — an import, a plugin, a save or an export — wakes the
        // interface itself when its result is sent and as its progress moves
        // (`wake_ui`). The slow poll here is only a safety net, for a wake
        // that raced the frame. A load cannot wake anything (the core knows no
        // toolkit), so it is polled, at a rate a readout needs rather than at
        // the display's. A job that runs on the frame that started it — the
        // browser's — has nothing to poll for.
        let working = TARGET.jobs_run_behind() && self.job.is_some();
        if self.core.load_stage().is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if working {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }

    /// This frame's zoom input — the wheel and keys, and a pinch — applied to
    /// the 2D view before any panel is laid out. Returns whether a pinch is in
    /// progress, which the canvas needs to know too.
    fn zoom_input(&mut self, ui: &egui::Ui) -> bool {
        // Collect zoom input before panels consume events.
        // `zoom_delta()` is the correct API: egui routes Ctrl+scroll into
        // `zoom_factor_delta` rather than `smooth_scroll_delta`, so checking
        // smooth_delta while Ctrl is held would always be zero.
        let zoom_step: i32 = ui.input(|i| {
            let d = i.zoom_delta();
            let from_scroll = if d > 1.05 {
                1
            } else if d < 0.95 {
                -1
            } else {
                0
            };
            let from_keys = if i.modifiers.ctrl
                && (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals))
            {
                1
            } else if i.modifiers.ctrl && i.key_pressed(egui::Key::Minus) {
                -1
            } else {
                0
            };
            // If both trigger in the same frame, clamp to ±1.
            (from_scroll + from_keys).clamp(-1, 1)
        });

        // Pinch and two-finger pan, taken before the stepped path and in place
        // of it.
        //
        // A pinch is continuous, and the fixed ladder cannot express one: at
        // whole levels a slow pinch does nothing at all until it suddenly jumps
        // a step. So a gesture zooms freely between the ends of the ladder
        // instead, and the wheel keeps the rungs.
        //
        // The stepped path has to stand down while a gesture is live, because
        // `zoom_delta()` folds pinch in — left alone, one pinch would drive
        // both and the zoom would run away.
        let touch = ui.input(|i| i.multi_touch());
        let pinching = touch.is_some();
        if let Some(t) = touch {
            if self.core.document.is_some() && self.core.view_mode == ViewMode::Movie {
                // Anchored between the fingers, so the picture stays under them
                // — the touch equivalent of keeping the point under the cursor
                // put — and translated by however far they slid. It ends any
                // glide, which would be two things steering the same value.
                self.view.model.pinch(
                    t.center_pos.logical(),
                    t.zoom_delta,
                    t.translation_delta.logical(),
                );
                // Deliberately no `resize_to_zoom`: a gesture runs for many
                // frames, and resizing the window under the fingers on each of
                // them makes the thing being touched move away from the touch.
            }
        }

        // Apply the zoom value *before* the panels are drawn, so the image
        // redraws at the new zoom in this very frame. (Doing it after the
        // central panel meant the change only showed once a window resize
        // happened to drive an extra frame — so zooming past the monitor cap,
        // where the window no longer resizes, appeared frozen.) The window
        // resize and optional reposition are handled later, once the chrome
        // height is known. Cursor-centering uses last frame's cached geometry.
        if zoom_step != 0
            && !pinching
            && self.core.document.is_some()
            && self.core.view_mode == ViewMode::Movie
        {
            // Stepped off wherever the picture is *heading*, so a second notch
            // during a glide advances a second rung, and glided to about the
            // cursor (see `View2d::step_zoom`).
            let cursor = ui.ctx().input(|i| i.pointer.latest_pos());
            if let Some(step) = self
                .view
                .model
                .step_zoom(zoom_step, cursor.map(|p| p.logical()))
            {
                // The window is told about the destination now, so it still
                // resizes and repositions once per step rather than once per
                // frame.
                self.view.resize_to_zoom = true;
                self.view.zoom_reposition = Some(step);
            }
        }

        // Advance the glide, in this same frame as the step that started it, so
        // the picture is already moving on the frame the wheel turned. A pinch
        // has already ended it (it drives the zoom directly); with no picture,
        // or a 3D view, which has no zoom, it has nothing left to say.
        if self.core.document.is_none() || self.core.view_mode != ViewMode::Movie {
            self.view.model.stop_zoom_glide();
        }
        let dt = ui.input(|i| i.stable_dt);
        if self.view.model.advance_zoom_glide(dt) {
            // Nothing else would ask for the next frame once the wheel stops
            // turning.
            ui.ctx().request_repaint();
        }

        // Tell the residency planner whether the view is still moving, a
        // glide or a pinch, so it does not start decoding a window for a zoom
        // about to be left behind. Noted by the advance, *after* it, so the
        // frame a glide lands on plans at once rather than waiting one more.
        self.sync_inputs.moving = self.view.model.is_moving();
        pinching
    }

    /// The tool windows: a plugin's dialog, its plot, the point counter. Each
    /// acts on its own buttons as it is drawn.
    fn tool_windows(&mut self, ctx: &egui::Context) {
        self.poll_plugin_dialog(ctx);
        self.poll_plot(ctx);
        self.poll_point_counter(ctx);
    }

    /// The main window's chords, read once the tool windows have been drawn
    /// (see [`Phase::ToolWindows`]).
    fn hotkeys(&self, ctx: &egui::Context, cmds: &Commands) {
        // Ctrl/Cmd+C.
        if copy_view::hotkey(ctx) {
            cmds.push(UiCommand::Copy(Via::Hotkey));
        }
        // Ctrl/Cmd+Z, the canvas tools' undo (the point counter's), while the
        // main window has the keyboard — and only while a tool that keeps an
        // undo history is open, which is when the chord is taken. Applied
        // before the picture handles this frame's pointer, so an undo that
        // ends a drag in flight is not followed by one more step of it.
        if self.tools.takes_undo() && canvas_tool::undo_hotkey(ctx) {
            cmds.push(UiCommand::UndoTool);
        }
    }

    /// The pop-up windows that are open, in the order they were registered.
    fn popups(&mut self, ctx: &egui::Context) {
        let mut host = windows::Host {
            core: &mut self.core,
        };
        self.windows.show(ctx, &mut host);
    }

    /// Run the playback clock for this frame.
    fn run_playback(&mut self, ctx: &egui::Context, scrubber_held: bool) {
        // Looped playback: the core advances by real elapsed time so the movie
        // runs at the file's `fps` (or the default) regardless of render
        // cadence. All we add is the repaint scheduling.
        //
        // A held scrubber suspends the advance without clearing `playing`, so
        // the frame being scrubbed to is the only one written and the movie
        // picks up again on release. Without it the clock and the scrubber
        // write the frame index in turn and the picture flickers between them.
        self.core.playback.scrubbing = scrubber_held;
        self.core.tick_playback(ctx.input(|i| i.time));
        if self.core.playback.playing {
            // Ask for the next repaint at the playback rate rather than
            // immediately: no point re-running egui faster than frames actually
            // change. If a frame takes longer than this to produce, egui
            // repaints as soon as it's ready, so we still render as fast as we
            // can when behind (and the core's demand estimate still detects it).
            let fps = self.core.playback.fps.max(0.1);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(1.0 / fps));
        }
    }

    /// Settle the view before the canvas is drawn: back to 2D when there is no
    /// volume left to show, then the animation's step and the canvas tools'
    /// tick (a rectangle's Enter and Escape), both of which read the settled
    /// mode.
    fn settle_view(&mut self, ctx: &egui::Context) {
        // A dimension swap can collapse the stack to a single frame (e.g.
        // 1 ch × N frames -> N ch × 1 frame), which can't build a volume — the
        // stale volume (first channel only) would keep showing, and with the
        // 2D/3D toggle disabled below two frames the user would be stranded in
        // 3D. Drop back to 2D; the toggle stays disabled until a swap restores
        // a multi-frame layout. (Runs before the central panel so the click
        // frame already renders 2D.)
        if self.core.view_mode == ViewMode::Volume
            && self
                .core
                .document
                .as_ref()
                .is_some_and(|l| l.display.dims().frames < 2)
        {
            self.core.view_mode = ViewMode::Movie;
            self.core.playback.playing = false;
            self.core.playback.last_time = None;
        }

        // The animation's step for this frame, now that every way of leaving
        // 3D or moving the camera this frame has been applied, and before the
        // 3D view is drawn with the pose it decides.
        self.tick_animation(ctx);
        self.tick_tools(ctx);
    }

    /// Size the window, then bring the GPU up to date with this frame.
    fn end_frame(
        &mut self,
        ui: &egui::Ui,
        toolbar_response: &egui::InnerResponse<()>,
        scrub_bar_response: &egui::InnerResponse<()>,
    ) {
        // Window chrome (sizing, position, title): the desktop's alone, and
        // nothing in a browser, whose canvas the page sizes.
        self.platform.manage_window(
            ui.ctx(),
            &mut self.view,
            &mut self.panel,
            self.core.document.as_ref(),
            toolbar_response.response.rect.height(),
            scrub_bar_response.response.rect.height(),
        );

        // Bring the GPU up to date with the core state, then honor a pending
        // background volume build by scheduling another frame. The histograms
        // are binned from what the sync decodes, while their window is open.
        self.sync_inputs.histograms = self.windows.is_open(WindowId::HISTOGRAM);
        let render = std::sync::Arc::clone(&self.render);
        let outcome = match render.lock() {
            Ok(mut resources) => {
                let outcome = self.core.sync(&mut resources, &self.sync_inputs);
                // A recording's frame, now that the GPU has this frame's pose.
                self.capture_animation_frame(&mut resources);
                outcome
            }
            Err(_) => Default::default(),
        };
        if outcome.needs_repaint {
            ui.ctx().request_repaint();
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        UI.get_or_init(|| ui.ctx().clone());
        // The graphics device is gone, and egui would panic drawing this frame
        // without it. Ending here instead puts the reason in words; the panic
        // hook shows it (see `render::explain_gpu_failure`).
        if let Some(why) = Current::gpu_lost() {
            panic!("{why}");
        }
        self.draw(ui, frame);
        // This window's copies, in the system's line endings. Every pop-out
        // dialog does the same for its own — see `clipboard`.
        clipboard::to_system_line_endings(ui.ctx());
    }
}

#[cfg(test)]
#[path = "app/view2d_oracle_tests.rs"]
mod view2d_oracle_tests;

#[cfg(test)]
#[path = "app/readout_tests.rs"]
mod readout_tests;

#[cfg(test)]
#[path = "app/import_label_tests.rs"]
mod import_label_tests;

#[cfg(test)]
#[path = "app/panel_grow_tests.rs"]
mod panel_grow_tests;

#[cfg(test)]
#[path = "app/reset_tests.rs"]
mod reset_tests;

#[cfg(test)]
#[path = "app/job_tests.rs"]
mod job_tests;

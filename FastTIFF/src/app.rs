//! The viewer's `eframe::App`: window chrome, widgets, and input.
//!
//! All non-GUI state and logic lives in `fast_tiff_viewer::Viewer` — the loaded
//! stack, channel settings, playback clock, 3D camera, and the per-frame
//! decode→GPU sync. `ViewerApp` holds one of those plus the things only a
//! desktop window has: zoom, pan, window sizing, which panels are open. The
//! rule for where a field belongs is "would a browser UI need this to show the
//! right pixels?" — if yes, it's in `core`.
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
//!   * `command` — the interface's requests, and the order they take effect in
//!   * `minimap` — the 2D navigator: where the view sits in the whole frame
//!   * `overlay` — the 3D coordinate-box overlay, drawn with the egui painter
//!   * `point_counter` — clicking points onto the picture, counted in groups
//!   * `scale`   — how large the chrome is drawn (the web build runs bigger)
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
use fast_tiff_viewer::{DecodeMode, Document, ViewMode, Viewer};
// The registry's entries are the C-boundary types themselves rather than
// trait objects, so the traits whose methods the app calls are named here —
// the importer's and exporter's only where the app imports and exports with
// plugins, which a browser does not.
use fasttiff_plugin_api::Plugin as _;
#[cfg(not(target_arch = "wasm32"))]
use fasttiff_plugin_api::{Exporter as _, Importer as _};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

mod animation;
mod autoscroll;
mod camera;
mod canvas;
mod clipboard;
mod command;
mod copy_view;
mod dialog;
mod kinetic;
mod minimap;
mod overlay;
// Drawing a plot a plugin declared, and the canvas selection tool that feeds
// it. Native only, like `plugins_ui`: both are reached from the Plugins menu,
// which the browser build does not have.
mod plot;
mod plugins_ui;
mod point_counter;
mod region_pick;
mod scale;
mod scroll;
mod scrub_bar;
mod toolbar;
mod widgets;
mod windows;

use command::{Applying, Commands, Step, UiCommand, Via, FRAME};
// Named in the docs of this module and of its children, which link to the
// phases; the code goes by `FRAME` and never names one.
#[cfg(doc)]
use command::Phase;
use scale::UI_SCALE;
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

/// Whether this is the browser build. Where an answer differs there, the
/// platform is handed to whatever gives it as this value rather than tested
/// with `cfg!` in place, so that both answers are tested on either build.
const WEB: bool = cfg!(target_arch = "wasm32");

/// Discrete zoom levels the viewer snaps to (3.1% … 3200%). Zooming in/out
/// steps between adjacent levels. Above 100% the levels are mostly whole-number
/// magnifications (200%, 300%, 400%, …), where one source pixel maps to an exact
/// NxN block of screen pixels — crisp and uniform under our nearest sampling —
/// with 150% as the one fractional step for finer control. The stored values
/// are rounded to the percentages shown in the UI (e.g. 0.333 reads as 33.3%).
const ZOOM_LEVELS: [f32; 21] = [
    0.031, 0.042, 0.063, 0.083, 0.125, 0.167, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    8.0, 12.0, 16.0, 24.0, 32.0,
];

/// Smallest the window is ever sized to (inner size, points). Zooming out past
/// this keeps the window here and just letterboxes the shrinking image.
/// Native only — the canvas is sized by the page.
#[cfg(not(target_arch = "wasm32"))]
const MIN_WINDOW: f32 = 256.0;

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

/// How close a fit factor must be to a fixed level before it counts as that
/// level rather than being inserted next to it.
///
/// Two steps a percent apart are one step as far as the eye and the wheel are
/// concerned, and the pair would cost a wheel notch to cross for no visible
/// change. 2% is below what anyone can see in a zoom and well above the
/// rounding in the table.
const FIT_SNAP: f32 = 0.02;

/// Whether the window's size is the window manager's to decide, not ours.
///
/// True when maximized *or* full screen. Both were meant to be covered by the
/// "leave the window alone" rule, but only `maximized` was ever asked about —
/// and on macOS the green button goes full screen rather than maximizing, so
/// every zoom step out shrank a full-screen window out from under the picture.
///
/// `unwrap_or(false)` on each: a backend that does not report one of them
/// leaves the window resizable, which is the behaviour it had before either
/// flag existed.
///
/// **Known gap, and it is egui's, not ours.** On macOS `maximized` is read only
/// when the window is created and never refreshed — `egui-winit` skips the
/// runtime query on that platform to avoid a deadlock (its `lib.rs`, guarded by
/// `is_init || !cfg!(target_os = "macos")`; egui issue #3494). So a macOS window
/// *zoomed* rather than made full screen — Option-green, a double-clicked title
/// bar, Window > Zoom, or macOS 15 edge tiling — reports neither flag, and a
/// zoom step out will still resize it. `fullscreen` has no such gate and is
/// refreshed every frame on every platform, which is why the green button (full
/// screen, the usual case) is covered. Detecting the zoomed case would mean
/// guessing from the window's size against the monitor's, and the app's own
/// sizing cap (`monitor_work_area`, 95% x 90%) sits too close to what a zoomed
/// window actually measures to tell the two apart reliably.
///
/// Native only, like [`monitor_work_area`]: the browser build has no window of
/// its own to resize.
#[cfg(not(target_arch = "wasm32"))]
fn window_size_is_not_ours(ctx: &egui::Context) -> bool {
    let (maximized, fullscreen) = ctx.input(|i| (i.viewport().maximized, i.viewport().fullscreen));
    maximized.unwrap_or(false) || fullscreen.unwrap_or(false)
}

/// Time constant of the glide a zoom step takes to its new level.
///
/// The approach is exponential, so a step is about 63% done after this long.
/// With [`ZOOM_GLIDE_SNAP`] ending it, one rung of the ladder takes about
/// 130 ms and the longest run across the whole ladder about 190 ms — long
/// enough to read as movement rather than a jump, short enough that a second
/// notch never feels held up. Slower starts to feel like the program is
/// deciding whether to obey.
const ZOOM_GLIDE_TAU: f32 = 0.035;

/// How close to the target ends the glide. A ratio, not a difference, because
/// zoom is geometric.
///
/// One percent, which is well under [`FIT_SNAP`]'s two — already described
/// there as below what anyone can see in a zoom. Chasing the last percent is
/// pure latency: an exponential approach spends as long covering it as it did
/// covering the first three quarters.
const ZOOM_GLIDE_SNAP: f32 = 0.01;

/// Longest frame the glide will integrate. A hitch — a slow decode, a window
/// drag — must not be turned into one enormous zoom jump on the frame after.
const ZOOM_GLIDE_MAX_DT: f32 = 1.0 / 20.0;

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
#[cfg(not(target_arch = "wasm32"))]
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
    Ok(Opened::Bytes(bytes, imported_label(path, &fallback)))
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

/// A zoom step in flight: the level being glided to, and the point it turns
/// about.
///
/// Only the *drawn* zoom is animated. Everything the window does — resizing to
/// the image, keeping the cursor's point on the same desktop spot — is decided
/// once from the target, exactly as it was when the zoom snapped, so the window
/// still moves once per step rather than sixty times.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ZoomGlide {
    /// The ladder rung being approached.
    target: f32,
    /// Screen point held fixed while the picture scales about it.
    anchor: egui::Pos2,
}

/// The zoom levels in effect: the fixed ladder, plus this image's exact
/// fit-to-window factor when it has one and it is not already on the ladder.
///
/// Inserted in order rather than appended, so stepping through it is monotonic.
fn zoom_ladder(fit: Option<f32>) -> Vec<f32> {
    let mut levels = ZOOM_LEVELS.to_vec();
    if let Some(f) = fit.filter(|f| f.is_finite() && *f > 0.0) {
        if !levels.iter().any(|z| (z - f).abs() <= f * FIT_SNAP) {
            levels.insert(levels.partition_point(|z| *z < f), f);
        }
    }
    levels
}

/// The next zoom level in `dir` (+1 = in, −1 = out) from whichever level is
/// nearest `current`, clamped to the ends of the ladder.
///
/// `fit` is the open image's fit-to-window factor, which joins the ladder for
/// as long as that image is open — so zooming back out lands on the fitted view
/// instead of stepping past it to the nearest fixed level.
fn stepped_zoom(current: f32, dir: i32, fit: Option<f32>) -> f32 {
    let levels = zoom_ladder(fit);
    let nearest = levels
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - current)
                .abs()
                .partial_cmp(&(**b - current).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let next = (nearest as i32 + dir).clamp(0, levels.len() as i32 - 1) as usize;
    levels[next]
}

/// The range a continuous zoom — a pinch — may take: the ends of this image's
/// ladder. The stepped path cannot leave the ladder, so this is what keeps a
/// pinch from running off to a scale no wheel notch could ever return from.
fn zoom_bounds(fit: Option<f32>) -> (f32, f32) {
    let ladder = zoom_ladder(fit);
    (ladder[0], ladder[ladder.len() - 1])
}

/// The zoom at which an `img_w` x `img_h` image exactly fits `avail`, capped at
/// 1:1 so a small image is not magnified into a blur.
///
/// `None` while the area is not yet known — a canvas has no size on the frame
/// before it is first laid out — so the caller waits rather than fitting to
/// nothing.
///
/// Web-only, as the desktop build sizes its window to the image instead
/// ([`initial_fit_zoom`] is that path's opposite number), but compiled under
/// `test` on every target so the guards above stay covered.
#[cfg(any(target_arch = "wasm32", test))]
fn fit_to_panel(img_w: f32, img_h: f32, avail: egui::Vec2) -> Option<f32> {
    if !(img_w >= 1.0 && img_h >= 1.0 && avail.x >= 1.0 && avail.y >= 1.0) {
        return None;
    }
    let z = (avail.x / img_w).min(avail.y / img_h).min(1.0);
    z.is_finite().then_some(z).filter(|z| *z > 0.0)
}

/// The usable desktop area for the window, i.e. the monitor size minus headroom
/// for the title bar, taskbar, and window borders. `None` until the monitor
/// size is reported. This is the cap on how large the window may grow — beyond
/// it the image overflows the window and becomes pannable.
#[cfg(not(target_arch = "wasm32"))]
fn monitor_work_area(ctx: &egui::Context) -> Option<egui::Vec2> {
    ctx.input(|i| i.viewport().monitor_size)
        .map(|m| egui::vec2((m.x * 0.95).max(1.0), (m.y * 0.90).max(1.0)))
}

/// The opening zoom for a freshly loaded image: the largest zoom level ≤ 100%
/// at which the image plus chrome still fits the monitor's work area (so a
/// normal image opens at 100%, a big one at 50% or 25%). Returns `None` when the
/// monitor size isn't reported yet (caller should keep waiting rather than
/// guess) so a huge image never briefly opens oversized.
#[cfg(not(target_arch = "wasm32"))]
fn initial_fit_zoom(ctx: &egui::Context, img_w: f32, img_h: f32, chrome_h: f32) -> Option<f32> {
    let avail = monitor_work_area(ctx)?;
    // Largest zoom level at most 100% that still fits the work area.
    for &z in ZOOM_LEVELS.iter().rev().filter(|&&z| z <= 1.0) {
        if img_w * z <= avail.x && img_h * z + chrome_h <= avail.y {
            return Some(z);
        }
    }
    Some(ZOOM_LEVELS[0]) // even the smallest level overflows — open there and pan
}

/// The prompt shown while nothing is loaded.
///
/// The browser build adds a line the desktop has no need of. A page that
/// accepts a dropped file is indistinguishable, from the outside, from one
/// that uploads it — and for microscopy data, which is routinely unpublished
/// and sometimes clinical, that ambiguity is a real reason not to try the
/// tool at all. The drop target is the one place the reassurance arrives
/// before the question does; the README says as much, but nobody reads it
/// first.
///
/// `cfg!` rather than `#[cfg]` so both wordings compile and are checked on
/// every target: the shared lines are written once and cannot drift apart.
fn welcome_text() -> String {
    let mut text = String::from("Drag and drop a file here, \nor click \"Open File\" above.\n");
    if cfg!(target_arch = "wasm32") {
        text.push_str(
            "\n\nEverything is processed locally in your browser — \nno file is ever uploaded to a server.\n",
        );
    }
    text.push_str(
        "\n\n\nScroll — navigate frames\nShift + Scroll — fast navigate\nCtrl + Scroll — zoom",
    );
    text
}

/// A file the user asked to open, however it arrived.
///
/// Native and web get files by different routes — a path from a dialog, argv or
/// an Apple Event, versus bytes from an async browser picker or a drop event —
/// so they meet here and everything downstream is shared.
/// Something that arrived from somewhere other than this frame's user input.
enum Incoming {
    /// A file chosen by the asynchronous picker. Only the web build has one:
    /// the native dialog blocks and applies its result directly.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Opened(Opened),
    /// An import finished. The registry travels back with the result, because
    /// the worker was holding it — see [`ViewerApp::plugins`].
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
enum WriteOutcome {
    Done,
    Cancelled,
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
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

    /// A window per document is a process per document here: that is already
    /// how the app opens several files at once, and it needs no window manager
    /// of its own inside egui. In a browser it is a tab.
    fn open_in_new_window(
        &mut self,
        result: fast_tiff_viewer::plugins::ResultTiff,
    ) -> fast_tiff_viewer::plugins::Handover {
        use fast_tiff_viewer::plugins::Handover;
        // Named after the image, so the new window's title says what it is
        // showing rather than naming a scratch file.
        let name = format!(
            "{}.tif",
            fast_tiff_viewer::plugins::file_stem(result.name())
        );

        // The browser's handover. A tab, not a process — see `crate::web_open`
        // for why a blob and not a message, and why the plugin had to run
        // synchronously for this call to be allowed. A refused popup is the
        // expected failure, not an exceptional one: the result comes back and
        // is shown in this tab, with the reason in the status line.
        #[cfg(target_arch = "wasm32")]
        {
            let opened = match result.bytes() {
                Some(bytes) => crate::web_open::open_bytes_in_new_tab(bytes, &name),
                None => Err("the result is not in memory".to_string()),
            };
            match opened {
                Ok(()) => Handover::Opened(name),
                Err(why) => Handover::Refused(Box::new(result), why),
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // The file the result was written to as it arrived. The new window
            // maps it like any file it opens, so the result is never copied
            // into either process: the pages it needs are read in, and the OS
            // may let go of the rest. It is left behind deliberately — the
            // process that would delete it is the one still reading it, and it
            // cannot say when it is done — and `sweep_old_results` removes it
            // at a later start, once it is a day old. Its file is named after
            // the result, so that is what the new window is titled.
            if let Some(path) = result.path() {
                let shown = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(name);
                return match crate::process::try_open_in_new_process(path) {
                    Ok(()) => {
                        let _ = result.into_kept_path();
                        Handover::Opened(shown)
                    }
                    Err(e) => Handover::Refused(Box::new(result), e.to_string()),
                };
            }
            // In memory, because no file could be made for it. Through a pipe
            // when the machine has the room for the new window to hold it too.
            let bytes = result.bytes().unwrap_or_default();
            if !crate::process::fits_in_memory(bytes.len()) {
                return Handover::Refused(
                    Box::new(result),
                    "no file could be written for it, and it is too large to hand over in \
                     memory"
                        .into(),
                );
            }
            let cancel = self.cancel;
            let progress = self.progress;
            match crate::process::open_bytes_in_new_process(bytes, &name, &mut |f| {
                Job::report(progress, f);
                !cancel.load(std::sync::atomic::Ordering::Relaxed)
            }) {
                Ok(()) => Handover::Opened(name),
                Err(_) if cancel.load(std::sync::atomic::Ordering::Relaxed) => Handover::Cancelled,
                Err(e) => Handover::Refused(Box::new(result), e.to_string()),
            }
        }
    }

    fn save(
        &mut self,
        result: fast_tiff_viewer::plugins::ResultTiff,
        path: &str,
    ) -> Result<String, String> {
        // The browser's filesystem is the download manager. The plugin's path
        // still names the file — the last component of it — because the
        // desktop honours that name and the two should not disagree about what
        // the result is called; what a tab cannot honour is the directory, and
        // it does not pretend to.
        #[cfg(target_arch = "wasm32")]
        {
            let name = crate::web_save::name_from_path(path);
            let bytes = result.into_bytes().map_err(|e| e.to_string())?;
            crate::web_save::download(&bytes, &name).map(|()| name)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            result
                .save_as(std::path::Path::new(path))
                .map(|()| path.to_string())
                .map_err(|e| e.to_string())
        }
    }
}

/// Clear out plugin results more than a day old, on a thread of its own so
/// the window does not wait for it.
///
/// What this removes is mostly results handed to new windows: those are
/// left behind on purpose, the window's process reading them for as long as
/// it is open, and nothing else ever deletes them. A result still on show in
/// another window survives it on Windows, where a mapped file cannot be
/// deleted; see `fast_tiff_viewer::plugins::sweep`. Failures are ignored: the
/// next start tries again.
fn sweep_old_results() {
    // Not from a test that builds an app: that is the user's own folder.
    #[cfg(not(any(target_arch = "wasm32", test)))]
    if let fast_tiff_viewer::plugins::ResultDest::Dir(root) =
        fast_tiff_viewer::plugins::ResultDest::platform_default()
    {
        const DAY: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
        let _ = std::thread::Builder::new()
            .name("fasttiff-sweep".to_owned())
            .spawn(move || {
                let removed = fast_tiff_viewer::plugins::sweep(&root, DAY);
                if removed > 0 {
                    log::info!(
                        "removed {removed} old plugin result(s) from {}",
                        root.display()
                    );
                }
            });
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
        wake_ui_for_progress();
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
/// ([`wake_ui_for_progress`]), with a slow poll as a safety net.
///
/// Recorded on the first frame, before any worker can exist.
static UI: std::sync::OnceLock<egui::Context> = std::sync::OnceLock::new();

/// Ask the interface for a frame, from any thread: a result has arrived.
fn wake_ui() {
    if let Some(ctx) = UI.get() {
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    }
}

/// [`wake_ui`] for a progress report, at most every 100 ms: a bar shows no
/// finer steps than that, and a plugin that reports once a row would otherwise
/// redraw the window thousands of times a second.
///
/// Not in the browser, where a job runs on the interface thread itself and
/// there is no clock to throttle by (`Instant` has none on wasm32).
fn wake_ui_for_progress() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis() as u64;
        if due(&LAST_MS, now, 100) {
            wake_ui();
        }
    }
}

/// Whether at least `every` ms have passed since the time in `last`, claiming
/// `now` as the new last time when they have — once, however many threads ask
/// at the same moment. `last` starts at 0, so the first ask after `every` ms
/// of running is due.
#[cfg(not(target_arch = "wasm32"))]
fn due(last: &std::sync::atomic::AtomicU64, now: u64, every: u64) -> bool {
    use std::sync::atomic::Ordering;
    let then = last.load(Ordering::Relaxed);
    now.saturating_sub(then) >= every
        && last
            .compare_exchange(then, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}

pub enum Opened {
    /// A path on disk. Native only; the browser has no filesystem to name.
    #[cfg(not(target_arch = "wasm32"))]
    Path(PathBuf),
    /// The file's bytes plus its name, for display.
    ///
    /// The browser build's only route, there being no filesystem there.
    /// Natively it is how a plugin result reaches a new window when no file
    /// could be written for it and it was handed over in memory instead.
    Bytes(Vec<u8>, String),
    /// A plugin's result, to be opened as it was written: straight to the
    /// reader, past the importers — it is a TIFF this app wrote, and an
    /// importer claiming `.tif` has no business reading it back.
    Source(fast_tiff_viewer::LoadSource),
}

/// Where the image sits in the viewport: the 2D pan/zoom transform and the
/// geometry derived from it each frame.
///
/// Pan and zoom are *not* applied by moving the viewport — a GPU backend would
/// clamp an oversized one to the framebuffer and squash the picture. They are
/// turned into the UV sub-rect the shader samples (`Viewer::uv_offset` /
/// `uv_scale`); these fields are the frontend's side of that conversion.
struct View2d {
    /// Zoom factor: 1.0 = one window pixel per image pixel. The window is only
    /// resized in response to an explicit event (opening a file, a zoom step) —
    /// never every frame — so the user can freely resize or maximize it.
    /// Between those events the image scales to fit, aspect locked.
    zoom: f32,
    pan: egui::Vec2,
    /// Set when a file opens: the next frame computes a fit-to-screen zoom and
    /// sizes the window once. Deferred because the chrome height and monitor
    /// size aren't known at open time.
    pending_initial_fit: bool,
    /// The exact zoom at which the open image fits the view, once computed, and
    /// an extra rung on this image's zoom ladder (see [`zoom_ladder`]).
    ///
    /// Per image, so it is cleared on open: it is a property of this picture in
    /// this window, and carrying a previous file's factor over would leave a
    /// rung that fits nothing. `None` on the desktop, where the window is
    /// resized to the image and the opening zoom is a fixed level already.
    fit_zoom: Option<f32>,
    /// Set when something (initial fit, or a zoom step) wants the window resized
    /// to match `zoom` this frame. Applied once, then cleared.
    resize_to_zoom: bool,
    /// A pending (zoom, anchor) so a zoom step can keep the point under the
    /// cursor put, applied once the new layout is known.
    zoom_reposition: Option<(f32, egui::Pos2)>,
    /// A zoom step being glided out, if one is in flight. See [`ZoomGlide`].
    zoom_glide: Option<ZoomGlide>,
    /// Sub-notch wheel deltas accumulated until they add up to a frame step.
    scroll_accum: f32,
    /// Top-left of the drawn image in screen space, from the last frame.
    image_origin: egui::Pos2,
    /// The central panel's rect, from the last frame.
    panel_rect: egui::Rect,
    /// Inertia for swipe-panning, so the view keeps moving for a moment after
    /// the fingers leave the trackpad. See [`kinetic`].
    pan_glide: kinetic::Glide,
    /// Whether the image overflowed the panel on the *previous* frame.
    ///
    /// A zoom step changes the image's size immediately but the window that
    /// holds it only on the frame after, so between the two the image genuinely
    /// overflows a panel that is about to grow to fit it. Requiring the
    /// overflow to survive a frame is what stops the navigator blinking on
    /// during a zoom between two views that both fit.
    was_pannable: bool,
    /// Whether the drag in flight began on the navigator, and is therefore
    /// moving the view rather than dragging the picture.
    ///
    /// Sticky for the life of the gesture rather than re-tested each frame, and
    /// it has to be, in both directions. The pointer leaves the map the moment
    /// the view box is dragged to an edge, and the drag must carry on. And a
    /// drag begun on the picture that happens to pass over the map must not
    /// start teleporting the view half way through.
    minimap_drag: bool,
    /// Where an autoscroll gesture is anchored, when that mode is on.
    ///
    /// The anchor is the whole of the mode's state: the velocity is a function
    /// of where the pointer is *relative to it* (see [`autoscroll`]), so there
    /// is no accumulated speed to keep and nothing to decay. Holding the point
    /// rather than a flag is also what lets the marker be drawn where the
    /// gesture started, which is the only thing telling the user where neutral
    /// is.
    ///
    /// Cleared on open along with the rest of the transform: an anchor is a
    /// position in *this* window over *this* picture.
    autoscroll: Option<egui::Pos2>,
}

impl View2d {
    /// Forget the picture that was open, now that another has landed.
    ///
    /// Everything goes back to [`Default`] but the fields named here, so a
    /// field added to the view is reset unless someone decides it should not
    /// be. What that drops, and why it has to:
    ///
    /// * the zoom and pan: the new picture starts at 1:1, and on native the
    ///   next frame computes a fit-to-screen zoom and sizes the window once (on
    ///   the web `pending_initial_fit` is consumed by the canvas layout
    ///   instead); the fit factor belongs to the last picture;
    /// * `was_pannable`: a new file starts with no history of overflowing, or
    ///   the navigator would draw on its first frame from the previous file's
    ///   answer;
    /// * `pan_glide`: momentum from the last picture means nothing in this one;
    /// * `autoscroll` and `minimap_drag`: positions on screen tied to a
    ///   transform that has just been reset. An autoscroll anchor would park
    ///   the new picture's neutral point wherever the last click happened to
    ///   land, and a navigator drag would carry on moving a view of an image
    ///   that is no longer open;
    /// * `zoom_glide`: a glide belongs to the picture that was on screen, and
    ///   carrying one into a new file would slide the fresh image away from the
    ///   zoom it just opened at; `resize_to_zoom` with it.
    fn reset(&mut self) {
        *self = View2d {
            // The opening fit, still to come.
            pending_initial_fit: true,
            // Kept: the last layout's geometry, which the next frame lays out
            // afresh; a zoom reposition, which a zoom step sets and the same
            // frame's window sizing takes (natively — the browser never reads
            // it); and wheel travel short of one frame.
            image_origin: self.image_origin,
            panel_rect: self.panel_rect,
            zoom_reposition: self.zoom_reposition,
            scroll_accum: self.scroll_accum,
            ..View2d::default()
        };
    }

    /// The zoom the picture is heading for: the glide's target while one is in
    /// flight, otherwise what is on screen.
    ///
    /// Everything outside the drawing itself asks this rather than [`Self::zoom`]
    /// — the window size, the reposition, the next step off the ladder — so a
    /// glide changes when the picture arrives, never where it is going.
    fn zoom_settled(&self) -> f32 {
        self.zoom_glide.map_or(self.zoom, |g| g.target)
    }

    /// Scale the picture about `anchor`, keeping whatever sits under that point
    /// under it. The pan is expressed against the previous frame's cached
    /// geometry, which is what the stepped path used before it animated.
    fn zoom_about(&mut self, anchor: egui::Pos2, new_zoom: f32) {
        let p = (anchor - self.image_origin) / self.zoom;
        self.pan = self.panel_rect.min - (anchor - p * new_zoom);
        self.zoom = new_zoom;
    }

    /// Advance a glide by `dt`, returning whether one is still running.
    ///
    /// Exponential approach, integrated in the logarithm so the picture changes
    /// by the same *ratio* every frame: zoom is geometric, and interpolating it
    /// linearly would crawl at the bottom of a step and race at the top.
    fn advance_zoom_glide(&mut self, dt: f32) -> bool {
        let Some(glide) = self.zoom_glide else {
            return false;
        };
        let dt = dt.clamp(0.0, ZOOM_GLIDE_MAX_DT);
        let from = self.zoom;
        if !(from > 0.0 && glide.target > 0.0) {
            self.zoom_glide = None;
            return false;
        }
        let alpha = 1.0 - (-dt / ZOOM_GLIDE_TAU).exp();
        let next = (from.ln() + (glide.target.ln() - from.ln()) * alpha).exp();
        let arrived = !next.is_finite() || (next / glide.target - 1.0).abs() <= ZOOM_GLIDE_SNAP;
        let new_zoom = if arrived { glide.target } else { next };
        self.zoom_about(glide.anchor, new_zoom);
        if arrived {
            self.zoom_glide = None;
        }
        !arrived
    }
}

impl Default for View2d {
    fn default() -> Self {
        View2d {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            pending_initial_fit: false,
            fit_zoom: None,
            resize_to_zoom: false,
            zoom_reposition: None,
            zoom_glide: None,
            scroll_accum: 0.0,
            image_origin: egui::Pos2::ZERO,
            panel_rect: egui::Rect::ZERO,
            was_pannable: false,
            minimap_drag: false,
            autoscroll: None,
            pan_glide: kinetic::Glide::default(),
        }
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
    fn arm_grow(&mut self, height: f32) {
        self.grow_armed = GROW_WAIT;
        self.old_h = height;
    }

    /// How much taller the window should get, if that is settled this frame.
    ///
    /// `Some` disarms. `None` means either nothing was armed or the change has
    /// not shown up yet — [`waiting`](Self::waiting) tells those apart, and the
    /// wait is bounded so a change that never comes stops asking for frames.
    ///
    /// Native only, like its caller: in a browser the window is the page's,
    /// and nothing here resizes it.
    #[cfg(not(target_arch = "wasm32"))]
    fn grow_delta(&mut self, height: f32) -> Option<f32> {
        if self.grow_armed == 0 {
            return None;
        }
        let delta = height - self.old_h;
        if delta.abs() > 0.5 {
            self.grow_armed = 0;
            return Some(delta);
        }
        self.grow_armed -= 1;
        None
    }

    /// Whether a grow is still expecting the bar's height to change, and the
    /// caller should keep frames coming so it can see it.
    #[cfg(not(target_arch = "wasm32"))]
    fn waiting(&self) -> bool {
        self.grow_armed > 0
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
    /// from it).
    view: View2d,
    /// The collapsible channels panel's layout bookkeeping.
    panel: PanelLayout,
    /// Installed plugins and importers. Native only — see
    /// `fast_tiff_viewer::plugins`.
    /// The plugins, unless an import worker currently has them.
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
    /// Files opened asynchronously arrive here. Only the web picker uses it —
    /// the native dialog blocks and applies its result directly — but the
    /// channel exists on both so the drain path is shared.
    /// Only the async web picker sends; natively the dialog blocks and applies
    /// its result directly, so the sender is unused there.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
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
    /// The window title last sent via `ViewportCommand::Title`. Native only —
    /// a canvas has no title bar; the page's `<title>` is the host's business.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    last_title: Option<String>,
    /// The status message that should read as *done*, not as *careful*.
    ///
    /// The status line is amber, which is right for what it usually carries: a
    /// dimension note, a refusal, a plugin's error. It is wrong for "opened it
    /// in a new window", where amber reads as a warning about work that in fact
    /// succeeded.
    ///
    /// Held as the message itself rather than as a flag beside it. `core.status`
    /// is written from a dozen places — the loader, the dimension notes, every
    /// plugin outcome — and a flag set here would still be set once one of them
    /// replaced the text underneath it, colouring somebody else's message green.
    /// Comparing the text means the colour can only ever apply to the message it
    /// was set for.
    good_status: Option<String>,
    /// The pop-up windows — metadata, histogram, render settings — and which
    /// are open.
    windows: Windows,
    /// The 3D view's preferences, and what the frames between need to
    /// remember about it.
    volume_ui: VolumeUi,
    /// A plugin's animation of the 3D view, playing or being recorded. See
    /// [`animation`].
    animation: animation::Animations,
}

/// The canvas tools, each while it is in use.
///
/// Each one's marks are positions in the picture it was used on, so a new
/// document takes them away (see [`reset`](Self::reset)).
#[derive(Default)]
struct Tools {
    /// The plot a plugin returned, while one is on screen.
    ///
    /// It owns the canvas selection too: the regions belong to the plot they
    /// feed, so closing the window takes the tool away with it rather than
    /// leaving a selection behind that nothing is measuring.
    plot: Option<plot::PlotWindow>,
    /// A rectangle being drawn because a plugin asked for one. See
    /// [`region_pick`].
    region_pick: Option<region_pick::RegionPick>,
    /// The point counter, while its window is open. Like the plot's regions,
    /// the points belong to the window: closing it takes them, and their
    /// marks, away. See [`point_counter`].
    point_counter: Option<point_counter::Session>,
}

impl Tools {
    /// Put the tools away, now that another document has landed.
    fn reset(&mut self) {
        // A plot of the last picture. The regions are in pixels of the picture
        // they were drawn on; carrying them to a different image would put them
        // somewhere arbitrary and go on measuring, which is worse than losing
        // them.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.plot = None;
        }
        // A rectangle being drawn for a plugin was drawn on the last picture.
        self.region_pick = None;
        // And so were the counted points: they are positions in that picture,
        // and counting on into this one would mix two images in one table.
        self.point_counter = None;
    }
}

/// The user's 3D input preferences. Edited in the render-settings window, and
/// kept across files.
struct VolumePrefs {
    /// Overlay: draw the volume's bounding box with x/y/z coordinate ticks.
    show_coord_box: bool,
    /// User-adjustable 3D navigation speeds (multipliers on the built-in base
    /// rates). `move_speed` scales WASD / Space / Shift translation;
    /// `scroll_speed` scales the mouse-wheel fly.
    move_speed: f32,
    scroll_speed: f32,
    /// How many times smaller, per axis, the 3D view renders while the user
    /// moves it: 1 (full), 2 or 4.
    moving_downscale: u32,
}

impl Default for VolumePrefs {
    fn default() -> Self {
        VolumePrefs {
            show_coord_box: false,
            move_speed: 1.0,
            scroll_speed: 1.0,
            moving_downscale: render::MOVING_DOWNSCALE,
        }
    }
}

/// The 3D view's side of the frontend: the user's preferences for it, and
/// what one frame leaves for the next about it.
///
/// Nothing here is reset when a document lands: the preferences are the
/// user's, and the rest describes the panel and the camera, which are still
/// there.
#[derive(Default)]
struct VolumeUi {
    prefs: VolumePrefs,
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
    /// Until [`VOLUME_SETTLE_S`] after it, the view renders at reduced size.
    moved_at: Option<f64>,
}

/// How long the 3D view stays at reduced resolution after the user last moved
/// it, in seconds: long enough that a drag with pauses in it, or a run of wheel
/// clicks, does not flick between sharp and coarse frames; short enough that
/// the sharp frame arrives as soon as the hand stops.
const VOLUME_SETTLE_S: f64 = 0.15;

/// Reveal a directory in the platform's file manager.
///
/// Best effort by design: the plugin folder's path is shown in the status bar
/// regardless, and a desktop with no file manager — a bare window manager, a
/// remote session — should not turn "where do plugins go?" into an error.
#[cfg(not(target_arch = "wasm32"))]
fn open_in_file_manager(dir: &std::path::Path) -> std::io::Result<()> {
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = std::process::Command::new("explorer");
        c.arg(dir);
        c
    };
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("open");
        c.arg(dir);
        c
    };
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let mut cmd = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(dir);
        c
    };
    cmd.spawn().map(|_| ())
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
#[cfg(not(target_arch = "wasm32"))]
fn imported_label(path: &std::path::Path, fallback: &str) -> String {
    match path.file_name() {
        Some(n) => format!("{} - imported", n.to_string_lossy()),
        None => fallback.to_string(),
    }
}

impl ViewerApp {
    pub fn new(initial: Option<Opened>, render: Render) -> Self {
        let (open_tx, open_rx) = channel();
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut app = Self {
            core: Viewer::new(),
            render,
            view: View2d::default(),
            panel: PanelLayout::default(),
            // `with_installed` scans the plugin folders and loads every
            // shared library it finds; `new` is the built-ins alone. The
            // browser gets the latter because there is nothing to scan and no
            // `dlopen` to do it with — which is the only difference between
            // the two platforms' plugin menus.
            #[cfg(not(target_arch = "wasm32"))]
            plugins: Some(fast_tiff_viewer::plugins::Registry::with_installed()),
            #[cfg(target_arch = "wasm32")]
            plugins: Some(fast_tiff_viewer::plugins::Registry::new()),
            plugin_dialog: None,
            claimed_extensions: Vec::new(),
            document: Default::default(),
            open_tx,
            open_rx,
            job: None,
            tools: Tools::default(),
            picked: None,
            last_title: None,
            good_status: None,
            windows: Windows::new(),
            volume_ui: VolumeUi::default(),
            animation: Default::default(),
        };
        app.claimed_extensions = app
            .plugins
            .as_ref()
            .map(|r| r.claimed_extensions())
            .unwrap_or_default();
        sweep_old_results();
        if let Some(opened) = initial {
            app.apply_opened(opened);
        }
        app
    }

    /// Hand an opened file to the core, then reset the chrome that described
    /// the previous one. The core records any failure in `core.status`, which
    /// the bottom bar shows, and keeps the old stack loaded.
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
        #[cfg(not(target_arch = "wasm32"))]
        let opened = match opened {
            Opened::Path(path) => match self.start_import(path) {
                Ok(()) => return,
                Err(path) => Opened::Path(path),
            },
            other => other,
        };

        // Same rule as a worker job: the readout is about to appear, and what
        // the last action said is not about this one. See `begin_job`.
        self.clear_status();
        self.core.begin_open(match opened {
            #[cfg(not(target_arch = "wasm32"))]
            Opened::Path(path) => fast_tiff_viewer::LoadSource::Path(path),
            Opened::Bytes(bytes, name) => {
                fast_tiff_viewer::LoadSource::Bytes(bytes, PathBuf::from(name))
            }
            Opened::Source(source) => source,
        });
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

    /// Drop the last thing said, message and green-ness together.
    ///
    /// Both, always: `good_status` is what makes the text green, and leaving it
    /// set while replacing the text would paint the next message green whether
    /// or not it reported success.
    fn clear_status(&mut self) {
        self.core.status = None;
        self.good_status = None;
    }

    /// Report something that finished, in the voice for it.
    ///
    /// The status line is amber, which is right for a dimension note or a
    /// refusal and wrong for work that succeeded. Green is asked for by
    /// recording the message itself rather than setting a flag beside it:
    /// `core.status` is written from a dozen places, and a flag would still be
    /// set once one of them replaced the text underneath it.
    fn report_done(&mut self, message: String) {
        self.good_status = Some(message.clone());
        self.core.status = Some(message);
    }

    /// Reset everything that described the previous file, once a load has
    /// landed. The other half of [`apply_opened`](Self::apply_opened).
    ///
    /// Each part of the chrome resets itself: what it keeps and what it drops
    /// is its own business, written down where its fields are.
    fn finish_open(&mut self) {
        self.document
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.view.reset();
        self.tools.reset();
        self.panel.reset();
        // The pop-ups mostly stay up on the desktop, repointed at the new
        // document; each says for itself (`AppWindow::outlives_document`).
        self.windows.reset(&self.core, WEB);
    }

    /// Drain anything the async picker produced since the last frame. A no-op
    /// natively, where the dialog is synchronous.
    fn drain_pending_open(&mut self) {
        while let Ok(message) = self.open_rx.try_recv() {
            match message {
                Incoming::Opened(opened) => self.apply_opened(opened),
                #[cfg(not(target_arch = "wasm32"))]
                Incoming::Imported(registry, result) => {
                    // The registry comes home first, so the app is usable again
                    // whether the import worked or not.
                    self.plugins = Some(*registry);
                    self.job = None;
                    match result {
                        Ok(opened) => self.apply_opened(opened),
                        Err(message) => self.core.status = Some(message),
                    }
                }
                Incoming::Ran(registry, run) => {
                    self.plugins = Some(*registry);
                    self.job = None;
                    self.apply_plugin_run(*run);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Incoming::Wrote(registry, name, result) => {
                    if let Some(r) = registry {
                        self.plugins = Some(*r);
                    }
                    self.job = None;
                    match result {
                        WriteOutcome::Done => self.report_done(format!("Saved {name}")),
                        WriteOutcome::Cancelled => {
                            self.core.status = Some(format!("Stopped saving {name}"))
                        }
                        WriteOutcome::Failed(e) => {
                            self.core.status = Some(format!("Could not save {name}: {e}"))
                        }
                    }
                }
            }
        }
    }

    /// Begin a plugin: show its dialog, or run it at once if it declared none.
    fn start_plugin(&mut self, index: usize) {
        let Some(loaded) = self.core.document.as_ref() else {
            self.core.status = Some("Open an image first".into());
            return;
        };
        // A plain TIFF save does not take the registry, so the menu being
        // reachable is not on its own proof that nothing is running.
        if self.job.is_some() {
            self.core.status = Some(self.busy_message());
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
                self.core.status = Some(format!("{title}: {e}"));
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
                self.core.status = Some(format!("{}: {e}", dialog.title));
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
            self.core.status = Some("Open an image first".into());
            return;
        };
        // Reachable with a job already running: the dialog may have been left
        // open while a save was started behind it. Taking the registry is not
        // enough of a check on its own — a plain save never takes it.
        if self.job.is_some() {
            self.core.status = Some(self.busy_message());
            return;
        }
        let Some(registry) = self.plugins.take() else {
            self.core.status = Some(self.busy_message());
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
                params: values,
                name,
                messages: std::mem::take(&mut host.messages),
                outcome,
                made_from,
            };
            let _ = tx.send(Incoming::Ran(Box::new(registry), Box::new(run)));
            wake_ui();
        };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(work);

        // No thread to spawn: `std::thread::spawn` compiles on
        // wasm32-unknown-unknown and panics when called, and a browser without
        // shared memory has no worker that could hold this anyway. So it runs
        // here, on the frame that asked for it. Two things follow, and both
        // are deliberate.
        //
        // The interface does not repaint while a plugin runs, so the progress
        // bar it reports to is never seen and Cancel cannot be clicked. That
        // is a real limitation of the web build, not an oversight — the
        // alternative is a worker, shared memory, and a second wasm instance.
        //
        // It is also what makes `web_open` work: the result is handed to a new
        // tab from inside the same task as the click that started the run, so
        // the browser still counts the user as having asked for it and does
        // not refuse the popup.
        #[cfg(target_arch = "wasm32")]
        work();
    }

    /// Apply what a plugin returned, once its worker has handed it back.
    fn apply_plugin_run(&mut self, run: PluginRun) {
        let PluginRun {
            index,
            params,
            name,
            messages,
            outcome,
            made_from,
        } = run;
        for m in &messages {
            log::info!("{name}: {m}");
        }

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
            Ok(Product::Cancelled) => self.core.status = Some(format!("{name}: cancelled")),
            Ok(Product::Message(m)) => self.core.status = Some(format!("{name}: {m}")),
            // The window is already on its way; there is nothing left to do
            // here but say so.
            Ok(Product::NewWindow(what)) => {
                self.report_done(format!("{name}: opened {what} in a new window"))
            }
            Ok(Product::Inline { result, why }) => {
                self.core.status = Some(format!(
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
            Err(e) => self.core.status = Some(format!("{name}: {e}")),
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
            Err(e) => self.core.status = Some(format!("{plugin}: could not open its result: {e}")),
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
    #[cfg(not(target_arch = "wasm32"))]
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
            self.core.status = Some(self.busy_message());
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
        std::thread::spawn(move || {
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

    /// Ask where to put the open stack, and write it there.
    ///
    /// The dialog blocks the interface, which is right — it is a modal, and
    /// nothing else should be happening while it is up. The *write* does not:
    /// it is a decode-and-encode of every plane, seconds on a large stack, and
    /// it runs on a worker against its own handle on the same memory-mapped
    /// file. See `fast_tiff_viewer::save::SaveSource`.
    #[cfg(not(target_arch = "wasm32"))]
    fn save_as(&mut self) {
        let Some(stack) = self.core.document.as_ref() else {
            self.core.status = Some("Open an image first".into());
            return;
        };
        if self.job.is_some() {
            self.core.status = Some(self.busy_message());
            return;
        }
        // Suggest the file's own name with a `.tif` on it. An imported stack
        // has no file, and its `path` is the name it is shown under — which is
        // exactly what to suggest saving it as.
        let stem = stack
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "stack".to_string());
        let mut dialog = rfd::FileDialog::new()
            .add_filter("TIFF", &["tif", "tiff"])
            .set_file_name(format!("{stem}.tif"));
        // Every installed exporter's formats, beside the host's own TIFF. TIFF
        // is first because it is the only one that keeps the stack whole; the
        // rest are for handing a picture to something else.
        let offered = self
            .plugins
            .as_ref()
            .map(|p| p.save_file_types())
            .unwrap_or_default();
        for t in &offered {
            let exts: Vec<&str> = t.extensions.iter().map(|s| s.as_str()).collect();
            dialog = dialog.add_filter(&t.description, &exts);
        }
        if let Some(dir) = stack.save_dir() {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else {
            // Cancelled: not a failure, and not worth a status line either.
            return;
        };
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());
        // The extension decides, not the filter row that was selected: the
        // dialogs disagree about whether choosing a filter rewrites the name,
        // and the name is the thing the user can see. An extension no exporter
        // claims is a TIFF, which is also what an extension-less name gets.
        match self.exporter_for(&path) {
            Some(index) => self.start_export(index, path, name),
            None => self.start_save(path, name),
        }
    }

    /// Write the open stack as a TIFF, on a worker.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_save(&mut self, path: std::path::PathBuf, name: String) {
        let Some(stack) = self.core.document.as_ref() else {
            return;
        };
        let source = fast_tiff_viewer::save::SaveSource::of(stack);
        let (progress, cancel, _label) = self.begin_job(Job::new("Saving"));

        let tx = self.open_tx.clone();
        std::thread::spawn(move || {
            let result = contained("the writer", || {
                fast_tiff_viewer::save::save_source(&source, &path, &mut |f| {
                    Job::report(&progress, f);
                    !cancel.load(std::sync::atomic::Ordering::Relaxed)
                })
                .map_err(|e| format!("{e:#}"))
            })
            .unwrap_or_else(Err);
            let outcome = WriteOutcome::of(result, &cancel);
            // The registry was never taken: a plain save is the app's own code.
            let _ = tx.send(Incoming::Wrote(None, name, outcome));
            wake_ui();
        });
    }

    /// Encode the open stack and hand it to the browser as a download.
    ///
    /// The browser's `save_as`, and shorter than the desktop's because both of
    /// the things that make that one long are gone. There is no dialog to put
    /// up — a tab cannot ask where a file should go, and the download manager
    /// asks instead, if the user has told it to — and there is no worker to put
    /// the write on.
    ///
    /// # It runs on the frame that asked for it
    ///
    /// Same as a plugin on this target, for the same reason: `std::thread::spawn`
    /// compiles on `wasm32-unknown-unknown` and panics when called. So the
    /// interface does not repaint while the encode runs, which means the
    /// progress bar would never be seen and Cancel could never be clicked —
    /// hence no [`Job`] at all here, rather than one that reports to nothing.
    /// The status line afterwards is the whole of what this says.
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
    #[cfg(target_arch = "wasm32")]
    fn save_as(&mut self) {
        let Some(stack) = self.core.document.as_ref() else {
            self.core.status = Some("Open an image first".into());
            return;
        };
        if self.job.is_some() {
            self.core.status = Some(self.busy_message());
            return;
        }
        // What the window is titled with, which for a document that arrived as
        // bytes is the name it arrived under.
        let name = crate::web_save::suggested_name(&stack.path.to_string_lossy());
        let source = fast_tiff_viewer::save::SaveSource::of(stack);
        // Nothing to report progress to and nothing that could set the cancel
        // flag, so the callback is a constant rather than a pretence.
        let wrote = fast_tiff_viewer::save::save_to_bytes(&source, &mut |_| true)
            .map_err(|e| format!("{e:#}"))
            .and_then(|bytes| crate::web_save::download(&bytes, &name));
        match wrote {
            // "Saved" rather than "Downloaded": where it went is the browser's
            // business, and the user asked to save their image.
            Ok(()) => self.report_done(format!("Saved {name}")),
            Err(e) => self.core.status = Some(format!("Could not save {name}: {e}")),
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
            self.core.status = Some("Open an image first".into());
            return;
        };
        if self.job.is_some() {
            self.core.status = Some(self.busy_message());
            return;
        }
        if self.core.view_mode == ViewMode::Volume {
            // Nothing has been painted yet on the frame a file opens in, so
            // there is no view to grab. Saying so beats copying a blank.
            let Some(rect) = self.volume_ui.rect else {
                self.core.status = Some("The 3D view has not been drawn yet".into());
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
            Err(e) => self.core.status = Some(format!("Could not copy the view: {e:#}")),
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
            None => self.core.status = Some("The 3D view is too small to copy".into()),
        }
    }

    /// Hand the open stack to an exporter, on a worker.
    ///
    /// The registry goes with it, as an import's and a filter's do, which is
    /// what keeps a second job from starting while this one holds the data.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_export(&mut self, index: usize, path: std::path::PathBuf, name: String) {
        let Some(loaded) = self.core.document.as_ref() else {
            return;
        };
        let Some(registry) = self.plugins.take() else {
            self.core.status = Some(self.busy_message());
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
        std::thread::spawn(move || {
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
    #[cfg(not(target_arch = "wasm32"))]
    fn exporter_for(&self, path: &std::path::Path) -> Option<usize> {
        self.plugins.as_ref()?.exporter_for(path)
    }

    /// Show the platform's file picker.
    ///
    /// Native: a blocking dialog that can select several files — the first
    /// opens here and the rest are launched in their own processes, matching
    /// the command line and drag-drop. Web: an async picker that posts its
    /// result back through `open_rx`.
    fn show_open_dialog(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            // Importers add their formats to the picker, so a file an installed
            // plugin can read is selectable without the user changing the
            // filter to "all files" and hoping.
            let mut dialog = rfd::FileDialog::new().add_filter("TIFF", &["tif", "tiff"]);
            let types = self
                .plugins
                .as_ref()
                .map(|p| p.open_file_types())
                .unwrap_or_default();
            if !types.is_empty() {
                let mut every: Vec<&str> = vec!["tif", "tiff"];
                for t in &types {
                    let exts: Vec<&str> = t.extensions.iter().map(|s| s.as_str()).collect();
                    dialog = dialog.add_filter(&t.description, &exts);
                    every.extend(exts);
                }
                // A combined entry first, so "open whatever this app can read"
                // is one click rather than a hunt through the format list.
                dialog = rfd::FileDialog::new()
                    .add_filter("Images FastTIFF can read", &every)
                    .add_filter("TIFF", &["tif", "tiff"]);
                for t in &types {
                    let exts: Vec<&str> = t.extensions.iter().map(|s| s.as_str()).collect();
                    dialog = dialog.add_filter(&t.description, &exts);
                }
            }
            if let Some(paths) = dialog.pick_files() {
                if let Some(first) = crate::process::open_all(&paths) {
                    self.apply_opened(Opened::Path(first.clone()));
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let tx = self.open_tx.clone();
            let ctx = ctx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(handle) = rfd::AsyncFileDialog::new()
                    .add_filter("TIFF", &["tif", "tiff"])
                    .pick_file()
                    .await
                {
                    let name = handle.file_name();
                    let _ = tx.send(Incoming::Opened(Opened::Bytes(handle.read().await, name)));
                }
                // Wake the UI whether or not a file was chosen.
                ctx.request_repaint();
            });
        }
    }
}

impl ViewerApp {
    /// Size, position and title the OS window in response to this frame's
    /// events. Native only — a browser canvas is sized by CSS, and none of
    /// `ViewportCommand` means anything there.
    #[cfg(not(target_arch = "wasm32"))]
    fn manage_window(
        &mut self,
        ui: &egui::Ui,
        toolbar_response: &egui::InnerResponse<()>,
        scrub_bar_response: &egui::InnerResponse<()>,
    ) {
        // Window sizing happens ONLY in response to explicit events — a freshly
        // opened file, or a zoom change (handled below) — never every frame.
        // That's what lets the window be freely resized and maximized without
        // shaking or being snapped back.
        let toolbar_height = toolbar_response.response.rect.height();
        let bottom_bar_height = scrub_bar_response.response.rect.height();
        let chrome_height = toolbar_height + bottom_bar_height;

        // Panel expand/collapse: grow (or shrink) the window height by the
        // panel's own height change, so the image and toolbar above stay put
        // and the panel "drops down" from its position. One-shot, triggered
        // only by the toggle. Skipped when the window manager owns the size
        // (maximized or full screen) — there the
        // image just letterboxes into the space the panel takes. We stay armed
        // until the height actually changes (the toggle frame still reports the
        // old height), repainting meanwhile so the next frame lands.
        if let Some(delta) = self.panel.grow_delta(bottom_bar_height) {
            if !window_size_is_not_ours(ui.ctx()) {
                let cur = ui.ctx().content_rect().size();
                let h = (cur.y + delta).round().max(200.0);
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        cur.x.round(),
                        h,
                    )));
            }
        } else if self.panel.waiting() {
            // The change has not shown up yet. Keep the frames coming so the
            // next one can see it.
            ui.ctx().request_repaint();
        }

        let img_dims = self
            .core
            .document
            .as_ref()
            .and_then(|l| l.tiff.frames.first())
            .map(|f| (f.width as f32, f.height as f32));

        if let Some((img_w, img_h)) = img_dims {
            // On open: pick the largest zoom level ≤ 100% at which the image +
            // chrome still fits the monitor (a huge image opens scaled down, a
            // normal one at 100%). Deferred to here because the chrome height
            // and monitor size aren't known at open time.
            if self.view.pending_initial_fit {
                if let Some(z) = initial_fit_zoom(ui.ctx(), img_w, img_h, chrome_height) {
                    self.view.zoom = z;
                    self.view.pan = egui::Vec2::ZERO;
                    self.view.pending_initial_fit = false;
                    self.view.resize_to_zoom = true;
                } else {
                    // Monitor size not reported yet (can stay unknown until the
                    // window first gets focus/input). Poll a few times a second
                    // rather than spinning `request_repaint` every frame, which
                    // would peg a CPU core while the app sits idle on load.
                    ui.ctx()
                        .request_repaint_after(std::time::Duration::from_millis(100));
                }
            }

            // When the window manager owns the size — maximized or full screen —
            // the window is left completely alone on zoom, and the image just
            // zooms/pans/letterboxes inside it (handled by the central panel's
            // UV transform).
            let size_is_fixed = window_size_is_not_ours(ui.ctx());

            // The target window inner size for the current zoom: the image scaled
            // uniformly, clamped to fit the monitor and to the minimum size. Once
            // it hits the minimum the window stops shrinking and the image just
            // letterboxes. Computed once so the reposition decision and the
            // actual resize agree. `None` when the window manager owns the size
            // (window left alone).
            let target_window = if size_is_fixed {
                None
            } else {
                // The zoom being glided to, so the window is the right size for
                // the destination while the picture is still on its way there.
                // Sizing to the animated value instead would resize the window
                // on every frame of every step.
                let settled = self.view.zoom_settled();
                let window_scale = match monitor_work_area(ui.ctx()) {
                    Some(m) => {
                        let fit = (m.x / img_w).min((m.y - chrome_height).max(1.0) / img_h);
                        settled.min(fit)
                    }
                    None => settled,
                };
                let w = (img_w * window_scale).round().max(MIN_WINDOW);
                let h = (img_h * window_scale + chrome_height)
                    .round()
                    .max(MIN_WINDOW);
                Some(egui::vec2(w, h))
            };

            // The zoom value + pan were already applied early (above), so the
            // image is redrawing at the new zoom this frame. Here we only decide
            // whether to move the window so the cursor's point stays on the same
            // desktop spot.
            let mut reposition: Option<egui::Pos2> = None;
            if let Some((old_zoom, cursor)) = self.view.zoom_reposition.take() {
                let new_zoom = self.view.zoom_settled();
                let fits = monitor_work_area(ui.ctx())
                    .map(|m| img_w * new_zoom <= m.x && img_h * new_zoom + chrome_height <= m.y)
                    .unwrap_or(true);
                // Whether the window grew vs. the previous frame (zoom-in case),
                // and whether the image is now letterboxed inside the window
                // (smaller than the content on either axis).
                let cur_inner = ui
                    .ctx()
                    .input(|i| i.viewport().inner_rect.map(|r| r.size()));
                let grew = match (target_window, cur_inner) {
                    (Some(t), Some(c)) => t.x > c.x + 0.5 || t.y > c.y + 0.5,
                    _ => true,
                };
                let letterboxing = match target_window {
                    Some(t) => {
                        img_w * new_zoom < t.x - 0.5
                            || img_h * new_zoom < (t.y - chrome_height) - 0.5
                    }
                    None => false,
                };
                // Whether the image was letterboxed *before* this zoom step. In
                // that state the cursor can sit in the empty margin, off the
                // image, so the cursor-anchor math would jump the window — skip
                // the one reposition on the letterboxed → first-fitted step.
                let was_letterboxing = match cur_inner {
                    Some(c) => {
                        img_w * old_zoom < c.x - 0.5
                            || img_h * old_zoom < (c.y - chrome_height) - 0.5
                    }
                    None => false,
                };
                // Follow the cursor when zooming *in* and the window grows (but
                // not on the first step out of a letterboxed state), or when
                // zooming *out* while the image still fills the window. Once it's
                // letterboxing at the minimum size, or the window manager owns
                // the size, it stays put.
                let follow = !size_is_fixed
                    && fits
                    && ((new_zoom > old_zoom && grew && !was_letterboxing)
                        || (new_zoom < old_zoom && !letterboxing));
                if follow {
                    if let Some(outer) = ui.ctx().input(|i| i.viewport().outer_rect.map(|r| r.min))
                    {
                        let ratio = new_zoom / old_zoom;
                        reposition =
                            Some(outer + (cursor - self.view.panel_rect.min) * (1.0 - ratio));
                    }
                }
            }

            // Apply a pending resize (one-shot), unless the window manager owns
            // the size.
            if self.view.resize_to_zoom {
                if let Some(size) = target_window {
                    let (w, h) = (size.x, size.y);
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::InnerSize(size));

                    // Keep the window fully on the desktop. The target position is
                    // the cursor-zoom move (or the current position when none).
                    // Horizontally it's clamped to the monitor width. Vertically,
                    // if the (grown) window's bottom would drop below the usable
                    // area, it's *centered* between the top and bottom of the
                    // monitor — symmetric margins, so it's least likely to be
                    // covered by a taskbar whether that's docked at the top or
                    // the bottom (egui doesn't report which).
                    let info = ui.ctx().input(|i| {
                        (
                            i.viewport().outer_rect,
                            i.viewport().inner_rect,
                            i.viewport().monitor_size,
                        )
                    });
                    if let (Some(outer), Some(inner), Some(monitor)) = info {
                        let decoration = outer.size() - inner.size();
                        let new_outer = egui::vec2(w, h) + decoration;
                        let target = reposition.unwrap_or(outer.min);
                        let max_x = (monitor.x - new_outer.x).max(0.0);
                        let usable_bottom = monitor_work_area(ui.ctx())
                            .map(|wa| wa.y)
                            .unwrap_or(monitor.y);
                        let y = if target.y + new_outer.y > usable_bottom {
                            ((monitor.y - new_outer.y) * 0.5).max(0.0)
                        } else {
                            target.y.max(0.0)
                        };
                        let clamped = egui::pos2(target.x.clamp(0.0, max_x), y);
                        if (clamped - outer.min).length() > 0.5 {
                            ui.ctx()
                                .send_viewport_cmd(egui::ViewportCommand::OuterPosition(clamped));
                        }
                    } else if let Some(pos) = reposition {
                        ui.ctx()
                            .send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                    }
                }
                self.view.resize_to_zoom = false;
            }
        }

        // Window title: loaded filename, or the app name when nothing is open.
        let desired_title = match &self.core.document {
            Some(loaded) => {
                let name = loaded
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                format!("{name} • FastTIFF")
            }
            None => "FastTIFF".to_string(),
        };
        if self.last_title.as_deref() != Some(desired_title.as_str()) {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Title(desired_title.clone()));
            self.last_title = Some(desired_title);
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

    /// Take in what arrived since the last frame: files dropped on the window,
    /// what the asynchronous picker and the workers sent, and a load that
    /// finished — and keep frames coming while one is still on its way.
    fn intake(&mut self, ctx: &egui::Context) {
        // Files dropped onto the window. Natively these carry a path, and
        // dropping several opens the first here and launches the rest in their
        // own processes; in a browser the drop event carries the bytes instead.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dropped: Vec<PathBuf> = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.clone())
                    .collect()
            });
            if let Some(first) = crate::process::open_all(&dropped) {
                self.apply_opened(Opened::Path(first.clone()));
            }

            // macOS "Open With" / double-click delivers files via an Apple Event
            // (not argv); drain whatever `macos_open`'s handler has queued and
            // open them the same way as drag-drop.
            #[cfg(target_os = "macos")]
            {
                let opened = crate::macos_open::take_opened_files();
                if let Some(first) = crate::process::open_all(&opened) {
                    self.apply_opened(Opened::Path(first.clone()));
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let dropped: Vec<Opened> = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| {
                        f.bytes
                            .as_ref()
                            .map(|b| Opened::Bytes(b.to_vec(), f.name.clone()))
                    })
                    .collect()
            });
            for d in dropped {
                self.apply_opened(d);
            }
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
        // the display's.
        #[cfg(not(target_arch = "wasm32"))]
        let working = self.job.is_some();
        #[cfg(target_arch = "wasm32")]
        let working = false;
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
                let old_zoom = self.view.zoom;
                let (lo, hi) = zoom_bounds(self.view.fit_zoom);
                let new_zoom = (old_zoom * t.zoom_delta).clamp(lo, hi);
                // Anchored between the fingers, so the picture stays under them
                // — the touch equivalent of keeping the point under the cursor
                // put — and translated by however far they slid, so a pinch can
                // pan and zoom at once the way it does everywhere else.
                let p = (t.center_pos - self.view.image_origin) / old_zoom;
                self.view.pan =
                    self.view.panel_rect.min - (t.center_pos - p * new_zoom) - t.translation_delta;
                self.view.zoom = new_zoom;
                // A pinch is already continuous; a glide underneath it would be
                // two things steering the same value.
                self.view.zoom_glide = None;
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
            // Stepped off wherever the picture is *heading*, not where it has
            // got to, so a second notch during a glide advances a second rung
            // instead of re-deciding the first one.
            let old_zoom = self.view.zoom;
            let from = self.view.zoom_settled();
            let new_zoom = stepped_zoom(from, zoom_step, self.view.fit_zoom);
            if (new_zoom - from).abs() > f32::EPSILON {
                let cursor = ui
                    .ctx()
                    .input(|i| i.pointer.latest_pos())
                    .filter(|p| self.view.panel_rect.contains(*p))
                    .unwrap_or_else(|| self.view.panel_rect.center());
                // The picture glides to the new rung about the cursor; the
                // window is told about the destination now, so it still resizes
                // and repositions once per step rather than once per frame.
                self.view.zoom_glide = Some(ZoomGlide {
                    target: new_zoom,
                    anchor: cursor,
                });
                self.view.resize_to_zoom = true;
                self.view.zoom_reposition = Some((old_zoom, cursor));
            }
        }

        // Advance the glide, in this same frame as the step that started it, so
        // the picture is already moving on the frame the wheel turned.
        if self.view.zoom_glide.is_some() {
            if pinching || self.core.document.is_none() || self.core.view_mode != ViewMode::Movie {
                // A pinch drives the zoom directly and a 3D view has none;
                // either way the glide has nothing left to say.
                self.view.zoom_glide = None;
            } else {
                let dt = ui.input(|i| i.stable_dt);
                if self.view.advance_zoom_glide(dt) {
                    // Nothing else would ask for the next frame once the wheel
                    // stops turning.
                    ui.ctx().request_repaint();
                }
            }
        }

        // Tell the residency planner whether the view is still moving, so it
        // does not start decoding a window for a zoom about to be left behind.
        // Read *after* the advance, so the frame a glide lands on plans at once
        // rather than waiting one more.
        self.core.view_moving = self.view.zoom_glide.is_some() || pinching;
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
        // Ctrl/Cmd+Z, the point counter's undo, while the main window has the
        // keyboard — and only while the counter is open, which is when the
        // chord is taken. Applied before the picture handles this frame's
        // pointer, so an undo that ends a drag in flight is not followed by
        // one more step of it.
        if self.tools.point_counter.is_some() && point_counter::undo_hotkey(ctx) {
            cmds.push(UiCommand::UndoPoint);
        }
    }

    /// The pop-up windows that are open, in the order they were registered.
    fn popups(&mut self, ctx: &egui::Context) {
        let mut host = windows::Host {
            core: &mut self.core,
            prefs: &mut self.volume_ui.prefs,
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
    /// volume left to show, then the animation's step and the region pick's
    /// keys, both of which read the settled mode.
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
        self.tick_region_pick(ctx);
    }

    /// Size the window, then bring the GPU up to date with this frame.
    fn end_frame(
        &mut self,
        ui: &egui::Ui,
        toolbar_response: &egui::InnerResponse<()>,
        scrub_bar_response: &egui::InnerResponse<()>,
    ) {
        // Window chrome (sizing, position, title) is a desktop-only concern.
        #[cfg(not(target_arch = "wasm32"))]
        self.manage_window(ui, toolbar_response, scrub_bar_response);
        #[cfg(target_arch = "wasm32")]
        let _ = (toolbar_response, scrub_bar_response);

        // Bring the GPU up to date with the core state, then honor a pending
        // background volume build by scheduling another frame. The histograms
        // are binned from what the sync decodes, while their window is open.
        self.core.histograms_wanted = self.windows.is_open(WindowId::HISTOGRAM);
        let render = std::sync::Arc::clone(&self.render);
        let outcome = match render.lock() {
            Ok(mut resources) => {
                let outcome = self.core.sync(&mut resources);
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(why) = crate::render::gpu_lost() {
            panic!("{why}");
        }
        self.draw(ui, frame);
        // This window's copies, in the system's line endings. Every pop-out
        // dialog does the same for its own — see `clipboard`.
        clipboard::to_system_line_endings(ui.ctx());
    }
}

#[cfg(test)]
#[path = "app/zoom_tests.rs"]
mod zoom_tests;

#[cfg(test)]
#[path = "app/readout_tests.rs"]
mod readout_tests;

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
#[path = "app/import_label_tests.rs"]
mod import_label_tests;

#[cfg(test)]
#[path = "app/panel_grow_tests.rs"]
mod panel_grow_tests;

#[cfg(test)]
#[path = "app/reset_tests.rs"]
mod reset_tests;

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
#[path = "app/job_tests.rs"]
mod job_tests;

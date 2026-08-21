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
//! Supporting clusters live in child modules (which share this module's
//! privacy, so the split adds no `pub` surface beyond `pub(super)`):
//!   * `camera`  — egui input → the core camera
//!   * `overlay` — the 3D coordinate-box overlay, drawn with the egui painter
//!   * `scale`   — how large the chrome is drawn (the web build runs bigger)
//!   * `widgets` — the contrast range slider + value formatting
//!   * `windows` — the render-settings, metadata and histogram pop-ups

use crate::render::{self, Render};
use egui::{Color32, RichText};
use fast_tiff_viewer::channels::{
    gray_lut_applicable, gray_lut_count, gray_lut_sel_name, gray_lut_sel_tint,
    pseudocolor_applicable,
};
use fast_tiff_viewer::{DecodeMode, Stack, ViewMode, Viewer};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

mod camera;
mod dialog;
mod overlay;
mod scale;
mod widgets;
mod windows;

use scale::UI_SCALE;
use widgets::{contrast_controls, histogram_button, ContrastLayout};
use windows::{histogram_window, metadata_window, render_settings_window};

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

/// Discrete zoom levels the viewer snaps to (3.1% … 3200%). Zooming in/out
/// steps between adjacent levels. Above 100% the levels are mostly whole-number
/// magnifications (200%, 300%, 400%, …), where one source pixel maps to an exact
/// NxN block of screen pixels — crisp and uniform under our nearest sampling —
/// with 150% as the one fractional step for finer control. The stored values
/// are rounded to the percentages shown in the UI (e.g. 0.333 reads as 33.3%).
const ZOOM_LEVELS: [f32; 21] = [
    0.031, 0.042, 0.063, 0.083, 0.125, 0.167, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
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

/// The next zoom level in `dir` (+1 = in, −1 = out) from whichever level is
/// nearest `current`, clamped to the ends of `ZOOM_LEVELS`.
fn stepped_zoom(current: f32, dir: i32) -> f32 {
    let nearest = ZOOM_LEVELS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - current).abs().partial_cmp(&(**b - current).abs()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let next = (nearest as i32 + dir).clamp(0, ZOOM_LEVELS.len() as i32 - 1) as usize;
    ZOOM_LEVELS[next]
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

/// Longest the histogram may lag the frame on screen while the movie plays.
///
/// Rebuilding decodes the frame a second time — `sync` has already decoded it
/// for the GPU, but into buffers it does not keep — so an open histogram window
/// would otherwise double the per-frame decode cost of playback, which is
/// plainly visible on a large compressed stack. Nobody reads a distribution
/// redrawn thirty times a second anyway; a few updates per second still looks
/// live and costs a bounded amount no matter how fast the movie runs.
const HIST_PLAYBACK_INTERVAL: f64 = 0.25;

/// The channel histograms on display, and the state they were computed from.
///
/// Building these decodes every channel of the frame, which is far too much to
/// redo per repaint, so they are held until something the plot is a *function
/// of* moves: the frame, the plane interpretation (`prefetch_gen` bumps on a
/// dimension-order change), or a channel's track — the axis the bins are laid
/// out on. Contrast is deliberately not in the key: the handles slide along the
/// distribution rather than changing it, which is the whole reason for showing
/// the two together.
struct HistCache {
    frame: usize,
    generation: u64,
    bounds: Vec<(f32, f32)>,
    hists: Vec<fast_tiff_viewer::Histogram>,
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
    let mut text = String::from("Drag and drop a TIFF here, \nor click \"Open TIFF...\" above.\n");
    if cfg!(target_arch = "wasm32") {
        text.push_str(
            "\n\nEverything is processed locally in your browser — \nno file is ever uploaded to a server.\n",
        );
    }
    text.push_str("\n\n\nScroll — navigate frames\nShift + Scroll — fast navigate\nCtrl + Scroll — zoom");
    text
}

/// A file the user asked to open, however it arrived.
///
/// Native and web get files by different routes — a path from a dialog, argv or
/// an Apple Event, versus bytes from an async browser picker or a drop event —
/// so they meet here and everything downstream is shared.
enum Opened {
    /// A path on disk. Native only; the browser has no filesystem to name.
    #[cfg(not(target_arch = "wasm32"))]
    Path(PathBuf),
    /// The file's bytes plus its name, for display. Only the web build
    /// constructs this — natively every route to a file yields a path.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Bytes(Vec<u8>, String),
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
    /// Set when something (initial fit, or a zoom step) wants the window resized
    /// to match `zoom` this frame. Applied once, then cleared.
    resize_to_zoom: bool,
    /// A pending (zoom, anchor) so a zoom step can keep the point under the
    /// cursor put, applied once the new layout is known.
    zoom_reposition: Option<(f32, egui::Pos2)>,
    /// Sub-notch wheel deltas accumulated until they add up to a frame step.
    scroll_accum: f32,
    /// Top-left of the drawn image in screen space, from the last frame.
    image_origin: egui::Pos2,
    /// The central panel's rect, from the last frame.
    panel_rect: egui::Rect,
}

impl Default for View2d {
    fn default() -> Self {
        View2d {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            pending_initial_fit: false,
            resize_to_zoom: false,
            zoom_reposition: None,
            scroll_accum: 0.0,
            image_origin: egui::Pos2::ZERO,
            panel_rect: egui::Rect::ZERO,
        }
    }
}

/// Layout bookkeeping for the collapsible channels/contrast panel, which grows
/// and shrinks the window by its own height delta when toggled.
#[derive(Default)]
struct PanelLayout {
    /// Channel buttons + contrast sliders are tucked under a small triangle
    /// toggle to keep the bar minimal by default.
    expanded: bool,
    /// Set on the frame the toggle is clicked; the next frame (once the panel
    /// has been redrawn at its new size) grows or shrinks the window by the
    /// difference.
    grow_armed: bool,
    /// The panel's height *before* the toggle, for that delta.
    old_h: f32,
}

pub struct ViewerApp {
    /// Everything that isn't GUI: the loaded stack, channel settings, playback
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
    /// Files opened asynchronously arrive here. Only the web picker uses it —
    /// the native dialog blocks and applies its result directly — but the
    /// channel exists on both so the drain path is shared.
    /// Only the async web picker sends; natively the dialog blocks and applies
    /// its result directly, so the sender is unused there.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    open_tx: Sender<Opened>,
    open_rx: Receiver<Opened>,

    // --- window chrome ------------------------------------------------------
    /// The window title last sent via `ViewportCommand::Title`. Native only —
    /// a canvas has no title bar; the page's `<title>` is the host's business.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    last_title: Option<String>,
    /// Whether the file-metadata pop-up window is open.
    show_metadata: bool,
    /// Whether the 3D render-settings pop-up is open.
    show_render_settings: bool,
    /// Whether the channel-histogram pop-up is open.
    show_histogram: bool,
    /// Plot log(1 + count) in the histogram. A preference, so it outlives both
    /// the window being closed and the file being changed.
    ///
    /// On by default. A 16-bit microscopy frame is mostly background sitting in
    /// the bottom percent of a 0..65535 track, so a linear plot is one spike at
    /// the left edge and a flat line — which reads as a broken widget rather
    /// than as data. Log shows the distribution that is actually there; the
    /// checkbox is right under the plot for anyone who wants true counts.
    hist_log: bool,
    /// The histograms currently plotted, and what they describe. `None` until
    /// the window is first opened.
    hist: Option<HistCache>,
    /// `input.time` when `hist` was last rebuilt, for the playback throttle.
    hist_built_at: f64,

    // --- 3D input preferences (persist across files) ------------------------
    /// Overlay: draw the volume's bounding box with x/y/z coordinate ticks.
    show_coord_box: bool,
    /// User-adjustable 3D navigation speeds (multipliers on the built-in base
    /// rates), edited in the render-settings window. `move_speed` scales WASD /
    /// Space / Shift translation; `scroll_speed` scales the mouse-wheel fly.
    move_speed: f32,
    scroll_speed: f32,
}

impl ViewerApp {
    pub fn new(initial_path: Option<PathBuf>, render: Render) -> Self {
        let (open_tx, open_rx) = channel();
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut app = Self {
            core: Viewer::new(),
            render,
            view: View2d::default(),
            panel: PanelLayout::default(),
            open_tx,
            open_rx,
            last_title: None,
            show_metadata: false,
            show_render_settings: false,
            show_histogram: false,
            hist_log: true,
            hist: None,
            hist_built_at: f64::NEG_INFINITY,
            show_coord_box: false,
            move_speed: 1.0,
            scroll_speed: 1.0,
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = initial_path {
            app.apply_opened(Opened::Path(path));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = initial_path;
        app
    }

    /// Hand an opened file to the core, then reset the chrome that described
    /// the previous one. The core records any failure in `core.status`, which
    /// the bottom bar shows, and keeps the old stack loaded.
    fn apply_opened(&mut self, opened: Opened) {
        let _ = match opened {
            #[cfg(not(target_arch = "wasm32"))]
            Opened::Path(path) => self.core.open(path),
            Opened::Bytes(bytes, name) => self.core.load_bytes(bytes, PathBuf::from(name)),
        };
        // Start at 1:1; on native the next frame computes a fit-to-screen zoom
        // and sizes the window once. On the web `pending_initial_fit` is
        // consumed by the canvas layout instead.
        self.view.zoom = 1.0;
        self.view.pan = egui::Vec2::ZERO;
        self.view.pending_initial_fit = true;
        self.view.resize_to_zoom = false;
        // The channels panel is sized and populated for the previous file's
        // channel count, so it is rebuilt wholesale. That also disarms a toggle
        // caught in flight, whose remembered height belongs to a layout that no
        // longer exists and would resize the window by a stale delta.
        self.panel = PanelLayout::default();
        // The pop-ups, though, stay up on the desktop. Each is its own window
        // there, and what it shows is "the open file" rather than one
        // particular file — every one of them is rebuilt from the live stack
        // each frame, so leaving them open simply repoints them at the new one.
        // That is the behaviour worth having when comparing files: park the
        // histogram on a second monitor and step through a folder, rather than
        // reopening it after every load.
        //
        // In the browser they are windows *inside* the canvas, laid over the
        // image and sized for the channel count that is going away, so there
        // they still close.
        #[cfg(target_arch = "wasm32")]
        {
            self.show_metadata = false;
            self.show_render_settings = false;
            self.show_histogram = false;
        }
        // Except the 3D settings, which are reachable only through a button the
        // toolbar hides for a stack with no volume. Leaving that window up over
        // a single-frame file would strand it: closed, it cannot be reopened.
        #[cfg(not(target_arch = "wasm32"))]
        if !self.core.can_show_volume() {
            self.show_render_settings = false;
        }
        // The cached histograms describe the *previous* stack, and a new one
        // starts at frame 0 generation 0 — exactly the key the old cache holds,
        // so staleness alone would not catch it.
        self.hist = None;
    }

    /// Rebuild the cached histograms if what they describe has moved. Called
    /// only while the window is open, so a closed window costs nothing.
    ///
    /// `now` is `input.time`, for the playback throttle (see
    /// [`HIST_PLAYBACK_INTERVAL`]). Scrubbing by hand is never throttled: those
    /// are one frame at a time, and a histogram that ignored the frame you just
    /// moved to would be worse than useless.
    fn refresh_histograms(&mut self, now: f64) {
        let playing = self.core.playback.playing;
        let Some(loaded) = &self.core.stack else {
            self.hist = None;
            return;
        };
        let bounds: Vec<(f32, f32)> = loaded.display.settings.iter().map(|s| s.bounds).collect();
        let fresh = self.hist.as_ref().is_some_and(|c| {
            c.frame == loaded.frame_index && c.generation == loaded.prefetch_gen && c.bounds == bounds
        });
        if fresh {
            return;
        }
        // Playback repaints continuously, so a skipped rebuild is picked up on
        // its own a moment later — no repaint needs scheduling here.
        if playing && self.hist.is_some() && now - self.hist_built_at < HIST_PLAYBACK_INTERVAL {
            return;
        }
        self.hist = Some(HistCache {
            frame: loaded.frame_index,
            generation: loaded.prefetch_gen,
            bounds,
            hists: fast_tiff_viewer::histogram::frame_histograms(loaded),
        });
        self.hist_built_at = now;
    }

    /// Drain anything the async picker produced since the last frame. A no-op
    /// natively, where the dialog is synchronous.
    fn drain_pending_open(&mut self) {
        while let Ok(opened) = self.open_rx.try_recv() {
            self.apply_opened(opened);
        }
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
            if let Some(paths) = rfd::FileDialog::new().add_filter("TIFF", &["tif", "tiff"]).pick_files() {
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
                if let Some(handle) =
                    rfd::AsyncFileDialog::new().add_filter("TIFF", &["tif", "tiff"]).pick_file().await
                {
                    let name = handle.file_name();
                    let _ = tx.send(Opened::Bytes(handle.read().await, name));
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
        // only by the toggle. Skipped when the window is maximized — there the
        // image just letterboxes into the space the panel takes. We stay armed
        // until the height actually changes (the toggle frame still reports the
        // old height), repainting meanwhile so the next frame lands.
        if self.panel.grow_armed {
            let delta = bottom_bar_height - self.panel.old_h;
            if delta.abs() > 0.5 {
                self.panel.grow_armed = false;
                let maximized = ui.ctx().input(|i| i.viewport().maximized).unwrap_or(false);
                if !maximized {
                    let cur = ui.ctx().content_rect().size();
                    let h = (cur.y + delta).round().max(200.0);
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(cur.x.round(), h)));
                }
            } else {
                ui.ctx().request_repaint();
            }
        }

        let img_dims = self
            .core
            .stack
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
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                }
            }

            // When maximized, the window is left completely alone on zoom — the
            // image just zooms/pans/letterboxes inside it (handled by the
            // central panel's UV transform).
            let maximized = ui.ctx().input(|i| i.viewport().maximized).unwrap_or(false);

            // The target window inner size for the current zoom: the image scaled
            // uniformly, clamped to fit the monitor and to the minimum size. Once
            // it hits the minimum the window stops shrinking and the image just
            // letterboxes. Computed once so the reposition decision and the
            // actual resize agree. `None` when maximized (window left alone).
            let target_window = if maximized {
                None
            } else {
                let window_scale = match monitor_work_area(ui.ctx()) {
                    Some(m) => {
                        let fit = (m.x / img_w).min((m.y - chrome_height).max(1.0) / img_h);
                        self.view.zoom.min(fit)
                    }
                    None => self.view.zoom,
                };
                let w = (img_w * window_scale).round().max(MIN_WINDOW);
                let h = (img_h * window_scale + chrome_height).round().max(MIN_WINDOW);
                Some(egui::vec2(w, h))
            };

            // The zoom value + pan were already applied early (above), so the
            // image is redrawing at the new zoom this frame. Here we only decide
            // whether to move the window so the cursor's point stays on the same
            // desktop spot.
            let mut reposition: Option<egui::Pos2> = None;
            if let Some((old_zoom, cursor)) = self.view.zoom_reposition.take() {
                let new_zoom = self.view.zoom;
                let fits = monitor_work_area(ui.ctx())
                    .map(|m| img_w * new_zoom <= m.x && img_h * new_zoom + chrome_height <= m.y)
                    .unwrap_or(true);
                // Whether the window grew vs. the previous frame (zoom-in case),
                // and whether the image is now letterboxed inside the window
                // (smaller than the content on either axis).
                let cur_inner = ui.ctx().input(|i| i.viewport().inner_rect.map(|r| r.size()));
                let grew = match (target_window, cur_inner) {
                    (Some(t), Some(c)) => t.x > c.x + 0.5 || t.y > c.y + 0.5,
                    _ => true,
                };
                let letterboxing = match target_window {
                    Some(t) => {
                        img_w * new_zoom < t.x - 0.5 || img_h * new_zoom < (t.y - chrome_height) - 0.5
                    }
                    None => false,
                };
                // Whether the image was letterboxed *before* this zoom step. In
                // that state the cursor can sit in the empty margin, off the
                // image, so the cursor-anchor math would jump the window — skip
                // the one reposition on the letterboxed → first-fitted step.
                let was_letterboxing = match cur_inner {
                    Some(c) => {
                        img_w * old_zoom < c.x - 0.5 || img_h * old_zoom < (c.y - chrome_height) - 0.5
                    }
                    None => false,
                };
                // Follow the cursor when zooming *in* and the window grows (but
                // not on the first step out of a letterboxed state), or when
                // zooming *out* while the image still fills the window. Once it's
                // letterboxing at the minimum size, or maximized, it stays put.
                let follow = !maximized
                    && fits
                    && ((new_zoom > old_zoom && grew && !was_letterboxing)
                        || (new_zoom < old_zoom && !letterboxing));
                if follow {
                    if let Some(outer) = ui.ctx().input(|i| i.viewport().outer_rect.map(|r| r.min)) {
                        let ratio = new_zoom / old_zoom;
                        reposition = Some(outer + (cursor - self.view.panel_rect.min) * (1.0 - ratio));
                    }
                }
            }

            // Apply a pending resize (one-shot), unless maximized.
            if self.view.resize_to_zoom {
                if let Some(size) = target_window {
                    let (w, h) = (size.x, size.y);
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(size));

                    // Keep the window fully on the desktop. The target position is
                    // the cursor-zoom move (or the current position when none).
                    // Horizontally it's clamped to the monitor width. Vertically,
                    // if the (grown) window's bottom would drop below the usable
                    // area, it's *centered* between the top and bottom of the
                    // monitor — symmetric margins, so it's least likely to be
                    // covered by a taskbar whether that's docked at the top or
                    // the bottom (egui doesn't report which).
                    let info = ui.ctx().input(|i| {
                        (i.viewport().outer_rect, i.viewport().inner_rect, i.viewport().monitor_size)
                    });
                    if let (Some(outer), Some(inner), Some(monitor)) = info {
                        let decoration = outer.size() - inner.size();
                        let new_outer = egui::vec2(w, h) + decoration;
                        let target = reposition.unwrap_or(outer.min);
                        let max_x = (monitor.x - new_outer.x).max(0.0);
                        let usable_bottom = monitor_work_area(ui.ctx()).map(|wa| wa.y).unwrap_or(monitor.y);
                        let y = if target.y + new_outer.y > usable_bottom {
                            ((monitor.y - new_outer.y) * 0.5).max(0.0)
                        } else {
                            target.y.max(0.0)
                        };
                        let clamped = egui::pos2(target.x.clamp(0.0, max_x), y);
                        if (clamped - outer.min).length() > 0.5 {
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::OuterPosition(clamped));
                        }
                    } else if let Some(pos) = reposition {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                    }
                }
                self.view.resize_to_zoom = false;
            }
        }

        // Window title: loaded filename, or the app name when nothing is open.
        let desired_title = match &self.core.stack {
            Some(loaded) => {
                let name = loaded.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                format!("{name} — FastTIFF")
            }
            None => "FastTIFF".to_string(),
        };
        if self.last_title.as_deref() != Some(desired_title.as_str()) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Title(desired_title.clone()));
            self.last_title = Some(desired_title);
        }

    }

}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Files dropped onto the window. Natively these carry a path, and
        // dropping several opens the first here and launches the rest in their
        // own processes; in a browser the drop event carries the bytes instead.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dropped: Vec<PathBuf> =
                ui.ctx().input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
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
            let dropped: Vec<Opened> = ui.ctx().input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.bytes.as_ref().map(|b| Opened::Bytes(b.to_vec(), f.name.clone())))
                    .collect()
            });
            for d in dropped {
                self.apply_opened(d);
            }
        }
        self.drain_pending_open();

        // Collect zoom input before panels consume events.
        // `zoom_delta()` is the correct API: egui routes Ctrl+scroll into
        // `zoom_factor_delta` rather than `smooth_scroll_delta`, so checking
        // smooth_delta while Ctrl is held would always be zero.
        let zoom_step: i32 = ui.input(|i| {
            let d = i.zoom_delta();
            let from_scroll = if d > 1.05 { 1 } else if d < 0.95 { -1 } else { 0 };
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

        // Apply the zoom value *before* the panels are drawn, so the image
        // redraws at the new zoom in this very frame. (Doing it after the
        // central panel meant the change only showed once a window resize
        // happened to drive an extra frame — so zooming past the monitor cap,
        // where the window no longer resizes, appeared frozen.) The window
        // resize and optional reposition are handled later, once the chrome
        // height is known. Cursor-centering uses last frame's cached geometry.
        if zoom_step != 0 && self.core.stack.is_some() && self.core.view_mode == ViewMode::Movie {
            let old_zoom = self.view.zoom;
            let new_zoom = stepped_zoom(old_zoom, zoom_step);
            if (new_zoom - old_zoom).abs() > f32::EPSILON {
                let cursor = ui
                    .ctx()
                    .input(|i| i.pointer.latest_pos())
                    .filter(|p| self.view.panel_rect.contains(*p))
                    .unwrap_or_else(|| self.view.panel_rect.center());
                // The native-pixel point under the cursor, kept fixed by pan
                // (used when the image overflows; re-clamped to 0 when it fits,
                // where the window move below handles the centering instead).
                let p = (cursor - self.view.image_origin) / old_zoom;
                self.view.pan = self.view.panel_rect.min - (cursor - p * new_zoom);
                self.view.zoom = new_zoom;
                self.view.resize_to_zoom = true;
                self.view.zoom_reposition = Some((old_zoom, cursor));
            }
        }

        // 2D/3D view toggle + the 3D-settings button are set inside the toolbar
        // closure via these locals (applied after) so the closure never needs a
        // second borrow of `self`.
        let current_view_mode = self.core.view_mode;
        // Read before the toolbar closure, which would otherwise need a second
        // borrow of `self` to ask.
        let can_show_volume = self.core.can_show_volume();
        let mut mode_request: Option<ViewMode> = None;
        let mut open_requested = false;
        let mut render_settings_toggle = false;

        let toolbar_response = egui::Panel::top("toolbar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open TIFF...").clicked() {
                    open_requested = true;
                }
                // 2D/3D switch, right next to Open, and the render-settings
                // button beside it. Both need a volume to act on, which needs at
                // least two frames to stack into one.
                //
                // Hidden rather than greyed out when there is none. A disabled
                // control is a promise that it will work under some condition
                // the user can reach, and offers a tooltip to say which; these
                // cannot become available for the open file no matter what is
                // clicked, so a permanently dead 2D/3D pair next to Open reads
                // as breakage. The toolbar is also short enough that the gap is
                // no loss. The file-info group below supplies the trailing
                // separator, so this block owns only its own leading ones.
                if can_show_volume {
                    ui.separator();
                    if ui
                        .selectable_label(current_view_mode == ViewMode::Movie, "2D")
                        .on_hover_text("Movie (2D) view")
                        .clicked()
                    {
                        mode_request = Some(ViewMode::Movie);
                    }
                    if ui
                        .selectable_label(current_view_mode == ViewMode::Volume, "3D")
                        .on_hover_text("Volume (3D) view — drag to rotate, scroll to zoom")
                        .clicked()
                    {
                        mode_request = Some(ViewMode::Volume);
                    }
                    ui.separator();
                    if ui
                        .button(RichText::new("⚙").size(16.0))
                        .on_hover_text("3D render settings")
                        .clicked()
                    {
                        render_settings_toggle = true;
                    }
                }
                if self.core.stack.is_none() {
                    // Nothing open yet: show the version + active render backend
                    // in the space the file info will later occupy.
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "FastTIFF v{}, Renderer: {}",
                            env!("CARGO_PKG_VERSION"),
                            render::BACKEND
                        ))
                        .weak(),
                    );
                }
                if let Some(loaded) = &self.core.stack {
                    let meta = &loaded.tiff.meta;
                    // What the viewer is actually showing, which is not the
                    // file's raw claim whenever an axis was reclassified.
                    let dims = loaded.display.dims;
                    // Reflect a toggle click made earlier in this same toolbar
                    // (the 2D/3D buttons run before this), so the layout updates
                    // on the click frame rather than one frame later.
                    let in_3d = mode_request.unwrap_or(current_view_mode) == ViewMode::Volume;
                    // In 3D the frame axis becomes the volume's depth, so the
                    // frame counter/time are only meaningful when the stack also
                    // has a separate time axis (the channels+Z+time case, where
                    // `slices > 1`); otherwise hide them.
                    let hide_frame_info = in_3d && dims.slices <= 1;

                    // Zoom is a 2D-only concept — hidden entirely in 3D.
                    if !in_3d {
                        ui.separator();
                        // Up to 2 decimals, trailing zeros trimmed: 3.1%, 33.3%,
                        // 100%, 3200% — so the fractional small zooms read correctly.
                        let pct = format!("{:.2}", self.view.zoom * 100.0);
                        let pct = pct.trim_end_matches('0').trim_end_matches('.');
                        ui.label(RichText::new(format!("{pct}%")).monospace())
                            .on_hover_text("Zoom (Ctrl+scroll to change)");
                    }
                    ui.separator();
                    let channels_desc = if loaded.display.cmyk {
                        // Say what was read *and* what is shown: the file holds
                        // four ink plates, the viewer shows three converted
                        // components, and a printing user will want to know the
                        // K plate is folded in rather than dropped.
                        //
                        // ASCII arrow on purpose: the bundled font has
                        // no U+2192, so a real arrow renders as a tofu
                        // box. The multiplication sign already on this
                        // status line IS in the font, which makes that
                        // gap easy to assume away.
                        "CMYK->RGB".to_string()
                    } else if loaded.display.rgb {
                        "RGB".to_string()
                    } else {
                        format!("{} channel(s)", dims.channels)
                    };
                    let bits = loaded.tiff.frames.first().map(|f| f.bits_per_sample).unwrap_or(0);
                    ui.label(format!(
                        "{}×{} px, {}-bit, {}",
                        loaded.tiff.frames.first().map(|f| f.width).unwrap_or(0),
                        loaded.tiff.frames.first().map(|f| f.height).unwrap_or(0),
                        bits,
                        channels_desc,
                    ));

                    // Say so when the picture on screen is not the picture in
                    // the file. This only appears for frames past the GPU
                    // texture limit, which is rare and enormous — but silently
                    // showing a reduced image as if it were the data would be
                    // the kind of wrong a viewer must never be.
                    if loaded.gpu_stride > 1 {
                        ui.separator();
                        ui.label(
                            RichText::new(format!("1/{} scale", loaded.gpu_stride))
                                .color(Color32::from_rgb(230, 170, 60)),
                        )
                        .on_hover_text(format!(
                            "This frame is larger than one GPU texture can hold, so it is                              shown subsampled {}x on each axis. Pixel values and measurements                              read from it are sampled, not exact.",
                            loaded.gpu_stride
                        ));
                    }

                    if !hide_frame_info {
                        ui.separator();
                        let frame_digits = dims.frames.to_string().len();
                        ui.label(
                            RichText::new(format!("Frame {:>frame_digits$} / {}", loaded.frame_index + 1, dims.frames))
                                .monospace(),
                        );
                        if let Some(interval) = meta.frame_interval_s {
                            ui.separator();
                            let max_time = dims.frames.saturating_sub(1) as f64 * interval;
                            let time_width = format!("{max_time:.3}").len();
                            let current_time = loaded.frame_index as f64 * interval;
                            ui.label(RichText::new(format!("t = {current_time:>time_width$.3}s")).monospace());
                        }
                    }
                }
            });
        });
        if open_requested {
            self.show_open_dialog(ui.ctx());
        }
        if let Some(mode) = mode_request {
            self.core.view_mode = mode;
            // Entering 3D stops movie playback — unless the stack is 4D (a
            // separate time axis, `slices > 1`), where playing animates the
            // volume through time.
            if mode == ViewMode::Volume {
                let is_4d = self.core.stack.as_ref().is_some_and(|l| l.display.dims.slices > 1);
                if !is_4d {
                    self.core.playback.playing = false;
                    self.core.playback.last_time = None;
                }
            }
        }
        // In 3D the arrow keys rotate the volume (handled in the central panel),
        // so the movie's arrow-scrub and wheel-scrub paths must stand down.
        let view_is_volume = self.core.view_mode == ViewMode::Volume;

        let panel_expanded = self.panel.expanded;
        let is_playing = self.core.playback.playing;
        let pseudocolor_on = self.core.apply_pseudocolor;
        let mut toggle_requested = false;
        let mut play_toggle_requested = false;
        // A requested dimension-role reassignment: (channels, slices, frames).
        let mut dimension_override: Option<(usize, usize, usize)> = None;
        let mut pseudocolor_toggle: Option<bool> = None;
        // New selection for the single-channel grayscale color/colormap selector.
        let mut gray_lut_change: Option<usize> = None;
        let mut scroll_step: i32 = 0;
        let mut playback_fps = self.core.playback.fps;
        let mut decode_mode = self.core.decode_mode;
        let mut metadata_toggle = false;
        let mut histogram_toggle = false;
        let current_status = self.core.status.clone();

        let scrub_bar_response = egui::Panel::bottom("scrub_bar").show_inside(ui, |ui| {
            let Some(loaded) = &mut self.core.stack else {
                ui.label("Open a TIFF stack to begin.");
                return;
            };
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                let max_frame = loaded.display.dims.frames.saturating_sub(1);
                let has_multiple_frames = loaded.display.dims.frames > 1;
                // In 3D the frame axis is the volume's depth, so play/step/scrub
                // are meaningless unless the stack has a separate time axis
                // (`slices > 1`). Grey them out otherwise.
                let frame_nav_enabled =
                    has_multiple_frames && !(view_is_volume && loaded.display.dims.slices <= 1);

                let toggle_size = egui::vec2(20.0, 20.0);
                let toggle_response = ui
                    .add_sized(toggle_size, egui::Button::new(""))
                    .on_hover_text("Show/hide channel & contrast settings");
                if toggle_response.clicked() {
                    toggle_requested = true;
                }
                let icon_color = ui.style().interact(&toggle_response).fg_stroke.color;
                let r = toggle_response.rect.shrink(6.0);
                let triangle = if panel_expanded {
                    vec![r.left_bottom(), r.right_bottom(), r.center_top()]
                } else {
                    vec![r.left_top(), r.right_top(), r.center_bottom()]
                };
                ui.painter().add(egui::Shape::convex_polygon(triangle, icon_color, egui::Stroke::NONE));

                // Play/pause looped movie. Painted (triangle / two bars) rather
                // than using glyphs, since the default font may not carry the
                // ▶/⏸ characters.
                ui.add_enabled_ui(frame_nav_enabled, |ui| {
                    let play_resp = ui
                        .add_sized(egui::vec2(22.0, 20.0), egui::Button::new(""))
                        .on_hover_text("Play/pause looped movie");
                    if play_resp.clicked() {
                        play_toggle_requested = true;
                    }
                    let color = ui.style().interact(&play_resp).fg_stroke.color;
                    let r = play_resp.rect.shrink(5.0);
                    if is_playing {
                        let bar = r.width() * 0.32;
                        let left = egui::Rect::from_min_max(r.left_top(), egui::pos2(r.left() + bar, r.bottom()));
                        let right = egui::Rect::from_min_max(egui::pos2(r.right() - bar, r.top()), r.right_bottom());
                        ui.painter().rect_filled(left, 0.0, color);
                        ui.painter().rect_filled(right, 0.0, color);
                    } else {
                        let tri = vec![r.left_top(), r.left_bottom(), egui::pos2(r.right(), r.center().y)];
                        ui.painter().add(egui::Shape::convex_polygon(tri, color, egui::Stroke::NONE));
                    }
                });

                ui.add_enabled_ui(frame_nav_enabled, |ui| {
                    if ui.button("|<").on_hover_text("First frame").clicked() {
                        loaded.frame_index = 0;
                    }
                    if ui.button("<").on_hover_text("Previous frame (←)").clicked() {
                        loaded.frame_index = loaded.frame_index.saturating_sub(1);
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button(">|").on_hover_text("Last frame").clicked() {
                            loaded.frame_index = max_frame;
                        }
                        if ui.button(">").on_hover_text("Next frame (→)").clicked() {
                            loaded.frame_index = (loaded.frame_index + 1).min(max_frame);
                        }

                        let remaining = ui.available_width();
                        if has_multiple_frames {
                            ui.spacing_mut().slider_width = remaining.max(40.0);
                            ui.add(
                                egui::Slider::new(&mut loaded.frame_index, 0..=max_frame)
                                    .show_value(false)
                                    .trailing_fill(true),
                            );
                        } else {
                            // Single-frame stack: there's nothing to scrub, so draw
                            // a flat, handleless track instead of a slider parked at
                            // zero (the whole row is already disabled above).
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(remaining.max(40.0), 18.0),
                                egui::Sense::hover(),
                            );
                            let y = rect.center().y;
                            let track = egui::Rect::from_min_max(
                                egui::pos2(rect.left(), y - 2.0),
                                egui::pos2(rect.right(), y + 2.0),
                            );
                            ui.painter().rect_filled(track, 2.0, ui.visuals().widgets.inactive.bg_fill);
                        }
                    });
                });
            });

            if !view_is_volume {
                ui.input(|i| {
                    // Shift jumps ~5% of the stack at a time (min 1 frame) instead
                    // of 1, matching the Shift+wheel fast-scroll step.
                    let step = if i.modifiers.shift {
                        ((loaded.display.dims.frames as f64 * FAST_SCROLL_RATE).round() as usize).max(1)
                    } else {
                        1
                    };
                    let max_frame = loaded.display.dims.frames.saturating_sub(1);
                    if i.key_pressed(egui::Key::ArrowRight) {
                        loaded.frame_index = (loaded.frame_index + step).min(max_frame);
                    }
                    if i.key_pressed(egui::Key::ArrowLeft) {
                        loaded.frame_index = loaded.frame_index.saturating_sub(step);
                    }
                });
            }

            if panel_expanded {
                ui.separator();
                // Wrapping, not clipping: this row holds up to six independent
                // control groups, and which of them are present depends on the
                // file. Narrow the window — or open the web build, whose chrome
                // is drawn half again as large — and the tail of the row used to
                // simply vanish past the edge with nothing to indicate it was
                // there. Wrapping spends panel height, which the layout has, to
                // buy back reachability, which it did not.
                ui.horizontal_wrapped(|ui| {
                    // Whether a control group already sits in the row — each
                    // optional group below draws its leading separator only if
                    // so, so hiding a group never leaves an orphaned separator.
                    let mut row_has_items = false;

                    // The channels-vs-time guess (and its override) is
                    // meaningless for RGB, where the "channels" are fixed color
                    // planes — so the dropdown and pseudocolor toggle are hidden
                    // there.
                    if !loaded.display.rgb {
                        ui.label("Dimension order:");
                        let c = loaded.display.dims.channels;
                        let z = loaded.display.dims.slices;
                        let f = loaded.display.dims.frames;
                        // When the file has a real Z axis (as loaded — see
                        // `has_z_axis`), offer every assignment of the three
                        // counts to the three roles; otherwise just the
                        // channels/time swap (Z passes through untouched).
                        // sort+dedup collapses duplicates when counts are equal
                        // and keeps the list order stable across
                        // reinterpretations.
                        let show_z = loaded.display.has_z_axis;
                        let mut options: Vec<(usize, usize, usize)> = if show_z {
                            vec![(c, z, f), (c, f, z), (z, c, f), (z, f, c), (f, c, z), (f, z, c)]
                        } else {
                            vec![(c, z, f), (f, z, c)]
                        };
                        options.sort_unstable();
                        options.dedup();
                        let dim_label = |oc: usize, oz: usize, of: usize| {
                            if show_z {
                                format!("c: {oc}  z: {oz}  t: {of}")
                            } else if view_is_volume {
                                // Without a separate Z axis, 3D uses the frame
                                // axis as the volume's depth — so what reads as
                                // time in 2D is genuinely Z here.
                                format!("c: {oc}  z: {of}")
                            } else {
                                format!("c: {oc}  t: {of}")
                            }
                        };
                        egui::ComboBox::from_id_salt("dim_override")
                            .selected_text(dim_label(c, z, f))
                            .show_ui(ui, |ui| {
                                for (oc, oz, of) in options {
                                    if ui
                                        .selectable_label((oc, oz, of) == (c, z, f), dim_label(oc, oz, of))
                                        .clicked()
                                    {
                                        dimension_override = Some((oc, oz, of));
                                    }
                                }
                            });

                        // Optional channel palette — only for multi-channel
                        // grayscale stacks that carry no colors of their own.
                        if pseudocolor_applicable(loaded) {
                            ui.separator();
                            let mut on = pseudocolor_on;
                            if ui
                                .checkbox(&mut on, "Apply pseudocolor")
                                .on_hover_text("Tint channels ch1 = red, ch2 = green, ch3 = blue, …")
                                .changed()
                            {
                                pseudocolor_toggle = Some(on);
                            }
                        }
                        row_has_items = true;
                    }

                    // Editable playback rate (seeded from metadata `fps=`, else
                    // 30). Only shown when there's a playable time axis: several
                    // frames in 2D; in 3D the frame axis is the volume's depth,
                    // so time only exists for 4D stacks (`slices > 1`) — matches
                    // the play/scrub controls' enable logic above.
                    let fps_playable = loaded.display.dims.frames > 1
                        && !(view_is_volume && loaded.display.dims.slices <= 1);
                    if fps_playable {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.add(
                            egui::DragValue::new(&mut playback_fps)
                                .speed(0.5)
                                .range(0.1..=1000.0)
                                .max_decimals(2)
                                .suffix(" fps"),
                        )
                        .on_hover_text("Playback speed (frames per second)");
                        row_has_items = true;
                    }

                    // LUT selector for a single channel: show it through a
                    // channel color or a perceptual colormap
                    // (magma/plasma/inferno/viridis/turbo). When the file carries
                    // its own LUT the list leads with "Built-in LUT" (the
                    // default). Hidden for RGB, composite, and multi-channel
                    // stacks — there the per-channel colors / pseudocolor toggle
                    // already handle coloring.
                    if gray_lut_applicable(loaded) {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("LUT:");
                        let sel = loaded.display.gray_lut_sel;
                        egui::ComboBox::from_id_salt("gray_lut")
                            .selected_text(gray_lut_sel_name(&loaded.display, sel))
                            .show_ui(ui, |ui| {
                                for opt in 0..gray_lut_count(&loaded.display) {
                                    // Tint each entry with its LUT's low (dark) end
                                    // — the color the darkest samples map to. A
                                    // grayscale/black low end (grayscale + the pure
                                    // channel-color ramps) keeps the default text color.
                                    let name = gray_lut_sel_name(&loaded.display, opt);
                                    let text = match tint_color(gray_lut_sel_tint(&loaded.display, opt)) {
                                        Some(c) => RichText::new(name).color(c),
                                        None => RichText::new(name),
                                    };
                                    if ui.selectable_label(opt == sel, text).clicked() {
                                        gray_lut_change = Some(opt);
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Display this channel through a color LUT or colormap");
                        row_has_items = true;
                    }

                    // CPU decode parallelism: Auto threads only when playback
                    // can't keep up; Serial/Threaded force it off/on. Threaded
                    // decode only ever kicks in for compressed frames (parallel
                    // strip decompression) or wide 32-/64-bit frames (parallel
                    // per-pixel rescale/cast). 8- and 16-bit uncompressed frames
                    // decode zero-copy or with an unthreaded copy, so the control
                    // has no effect and is hidden for them.
                    let threadable = loaded.tiff.frames.first().is_some_and(|f| {
                        f.compression != fast_tiff_lib::Compression::None || f.bits_per_sample >= 32
                    });
                    if threadable {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("Decode:");
                        egui::ComboBox::from_id_salt("decode_mode")
                            .selected_text(decode_mode.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut decode_mode, DecodeMode::Auto, "Auto")
                                    .on_hover_text("Single-threaded until playback drops frames, then multi-threaded");
                                ui.selectable_value(&mut decode_mode, DecodeMode::Serial, "Serial")
                                    .on_hover_text("Always single-threaded (lowest total CPU)");
                                ui.selectable_value(&mut decode_mode, DecodeMode::Threaded, "Threaded")
                                    .on_hover_text("Always multi-threaded for large frames (spreads across cores)");
                            });
                        row_has_items = true;
                    }

                    // The two pop-up toggles. They flow with the rest of the
                    // row rather than being pinned to its right edge: once the
                    // row can wrap there is no single right edge to pin to, and
                    // a button that jumps to the end of whichever line happens
                    // to be last is harder to find than one that simply follows
                    // the controls.
                    if row_has_items {
                        ui.separator();
                    }
                    if histogram_button(ui).clicked() {
                        histogram_toggle = true;
                    }
                    if ui
                        .button(RichText::new("( i )").size(12.0))
                        .on_hover_text("See metadata")
                        .clicked()
                    {
                        metadata_toggle = true;
                    }
                });
                if !loaded.display.rgb {
                    ui.label(
                        RichText::new(
                            "Channels are guessed automatically (6 or fewer = channels, more = time); \
                             use Dimension order if that guess is wrong for this file.",
                        )
                        .small()
                        .weak(),
                    );
                }

                contrast_controls(ui, loaded, ContrastLayout::Inline, None);
            }
            if let Some(status) = &current_status {
                // The triple-axis note explains that 2D freezes Z at its first
                // slice — but the 3D view *does* use Z (as the volume depth), so
                // showing it there would be wrong. When `triple_axis_warning` is
                // set, the status IS that note (`compute_status` short-circuits
                // on it), so this suppresses exactly the right message.
                if !(view_is_volume && loaded.display.triple_axis_warning) {
                    ui.separator();
                    ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)).small());
                }
            }
            ui.add_space(4.0);
        });

        self.core.playback.fps = playback_fps;
        self.core.decode_mode = decode_mode;
        if metadata_toggle {
            self.show_metadata = !self.show_metadata;
        }
        if self.show_metadata {
            match &self.core.stack {
                Some(loaded) => metadata_window(ui.ctx(), &mut self.show_metadata, loaded),
                None => self.show_metadata = false,
            }
        }
        if histogram_toggle {
            self.show_histogram = !self.show_histogram;
        }
        if self.show_histogram {
            self.refresh_histograms(ui.input(|i| i.time));
            // Split the borrows by field: the window edits the stack's contrast
            // while reading the cached plots computed from it.
            let mut open = true;
            if let (Some(loaded), Some(cache)) = (self.core.stack.as_mut(), self.hist.as_ref()) {
                histogram_window(ui.ctx(), &mut open, loaded, &cache.hists, &mut self.hist_log);
            } else {
                open = false;
            }
            self.show_histogram = open;
        }
        if render_settings_toggle {
            self.show_render_settings = !self.show_render_settings;
        }
        if self.show_render_settings {
            let prev_nav = self.core.volume.cam.nav;
            let mut reset_position = false;
            render_settings_window(
                ui.ctx(),
                &mut self.show_render_settings,
                &mut self.core.volume.scale,
                &mut self.core.volume.interp,
                &mut self.core.volume.cam.nav,
                &mut self.core.volume.cam.orbit_point,
                &mut self.move_speed,
                &mut self.scroll_speed,
                &mut self.core.volume.render,
                &mut self.core.volume.density,
                &mut self.core.volume.iso,
                &mut self.show_coord_box,
                &mut reset_position,
                self.core.stack.as_ref(),
            );
            // Keep the view continuous across a fly⇄orbit switch (the two use
            // different eye representations) instead of snapping to a default.
            if self.core.volume.cam.nav != prev_nav {
                self.core.volume.cam.sync_for_nav(prev_nav.is_fly());
            }
            if reset_position {
                self.core.volume.cam.reset();
            }
        }

        if toggle_requested {
            self.panel.expanded = !self.panel.expanded;
            // Remember the panel's height *before* it expands/collapses; the
            // next frame (once it's redrawn in the new state) grows or shrinks
            // the window by the difference. This frame still shows the old
            // height, so the actual delta only becomes known next frame.
            self.panel.grow_armed = true;
            self.panel.old_h = scrub_bar_response.response.rect.height();
        }

        if play_toggle_requested {
            self.core.playback.playing = !self.core.playback.playing;
            // Start each play/pause from a clean clock, so the first tick after
            // resuming doesn't jump by however long we were paused and the
            // keeping-up estimate starts neutral. (`decode_parallel` stays
            // latched across a pause — if this stack needed it, it still does.)
            self.core.playback.restart();
        }

        if let Some(on) = pseudocolor_toggle {
            self.core.set_pseudocolor(on);
        }

        if let Some(sel) = gray_lut_change {
            self.core.set_gray_lut(sel);
        }

        // Looped playback: the core advances by real elapsed time so the movie
        // runs at the file's `fps` (or the default) regardless of render
        // cadence. All we add is the repaint scheduling.
        self.core.tick_playback(ui.input(|i| i.time));
        if self.core.playback.playing {
            // Ask for the next repaint at the playback rate rather than
            // immediately: no point re-running egui faster than frames actually
            // change. If a frame takes longer than this to produce, egui
            // repaints as soon as it's ready, so we still render as fast as we
            // can when behind (and the core's demand estimate still detects it).
            let fps = self.core.playback.fps.max(0.1);
            ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(1.0 / fps));
        }

        // Reassigning the axes also refreshes LUTs + status and invalidates the
        // volume (the frame axis, i.e. the volume depth, just changed).
        if let Some((c, z, f)) = dimension_override {
            self.core.set_dimension_order(c, z, f);
        }

        // A dimension swap can collapse the stack to a single frame (e.g.
        // 1 ch × N frames -> N ch × 1 frame), which can't build a volume — the
        // stale volume (first channel only) would keep showing, and with the
        // 2D/3D toggle disabled below two frames the user would be stranded in
        // 3D. Drop back to 2D; the toggle stays disabled until a swap restores
        // a multi-frame layout. (Runs before the central panel so the click
        // frame already renders 2D.)
        if self.core.view_mode == ViewMode::Volume
            && self.core.stack.as_ref().is_some_and(|l| l.display.dims.frames < 2)
        {
            self.core.view_mode = ViewMode::Movie;
            self.core.playback.playing = false;
            self.core.playback.last_time = None;
        }

        // Central panel: the image is drawn at exactly `image_size × zoom`. When
        // that fits the panel it's centered (letterboxed); when it's larger
        // (zoomed past what the monitor-capped window can show) it overflows and
        // is pannable by dragging. Aspect ratio is always preserved.
        // Zero inner margin: the window is sized to exactly the image, so any
        // panel padding would make the available area smaller than the image
        // and produce a small spurious pan/overflow even when it should fit.
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(egui::Margin::ZERO))
            .show_inside(ui, |ui| {
            if self.core.stack.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label(welcome_text());
                });
                return;
            }

            let available = ui.available_size();
            let (panel_rect, response) =
                ui.allocate_exact_size(available, egui::Sense::click_and_drag());
            self.view.panel_rect = panel_rect;

            // 3D volume view: drive the camera per the active nav mode and paint
            // the GPU ray-march. The 2D pan/UV/scrub path below is bypassed. This
            // runs before the `loaded` borrow so it can call `&mut self` methods.
            if self.core.view_mode == ViewMode::Volume {
                self.core.volume.aspect = (panel_rect.width() / panel_rect.height().max(1.0)).clamp(0.1, 10.0);

                // Until the first volume is built, show a black loading screen so
                // the heavy decode doesn't freeze on the previous view (`sync_gpu`
                // defers the initial build until after this frame paints).
                if self.core.volume.built_frame.is_none() {
                    ui.painter().rect_filled(panel_rect, 0.0, egui::Color32::BLACK);
                    ui.painter().text(
                        panel_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Loading 3D…",
                        egui::FontId::proportional(16.0),
                        egui::Color32::from_gray(150),
                    );
                    ui.ctx().request_repaint();
                    return;
                }

                let animating = self.drive_volume_camera(ui, &response, panel_rect);
                response.on_hover_cursor(egui::CursorIcon::Crosshair);
                ui.painter()
                    .with_clip_rect(panel_rect)
                    .add(render::paint_volume_callback(&self.render, panel_rect));
                // Coordinate-box overlay, drawn on top of the ray-march with the
                // 2D painter (aligned via the same camera; see `overlay`).
                if self.show_coord_box {
                    let painter = ui.painter().with_clip_rect(panel_rect);
                    self.draw_coord_box(&painter, panel_rect);
                }
                // Keep repainting while a drag or held key keeps the camera moving.
                if animating {
                    ui.ctx().request_repaint();
                }
                return;
            }

            let Some(loaded) = &self.core.stack else { return };
            let (Some(w), Some(h)) = (
                loaded.tiff.frames.first().map(|f| f.width),
                loaded.tiff.frames.first().map(|f| f.height),
            ) else {
                return;
            };

            let img_px = egui::vec2(w as f32 * self.view.zoom, h as f32 * self.view.zoom);
            // A 1px tolerance so sub-pixel rounding between the window size and
            // the panel's available area doesn't register as a pannable overflow.
            let overflow = egui::vec2(
                (img_px.x - available.x - 1.0).max(0.0),
                (img_px.y - available.y - 1.0).max(0.0),
            );
            let pannable = overflow.x > 0.0 || overflow.y > 0.0;

            // Drag to pan when the image overflows the panel.
            if pannable && response.dragged() {
                self.view.pan -= response.drag_delta();
            }
            self.view.pan.x = self.view.pan.x.clamp(0.0, overflow.x);
            self.view.pan.y = self.view.pan.y.clamp(0.0, overflow.y);

            // Where the image's top-left *would* be if drawn full-size: scrolled
            // by `pan` on an overflowing axis, centered on an axis that fits.
            // (Cached for cursor-centered zoom; may lie outside the panel.)
            let origin = egui::pos2(
                if overflow.x > 0.0 { panel_rect.min.x - self.view.pan.x } else { panel_rect.min.x + (available.x - img_px.x) * 0.5 },
                if overflow.y > 0.0 { panel_rect.min.y - self.view.pan.y } else { panel_rect.min.y + (available.y - img_px.y) * 0.5 },
            );
            self.view.image_origin = origin;

            // Render into the on-screen *visible* rectangle only, and pan/zoom
            // via UVs. Drawing into an oversized rect doesn't work: the callback
            // viewport is clamped to the framebuffer, which would just squash the
            // whole image back to fit instead of zooming.
            let full_rect = egui::Rect::from_min_size(origin, img_px);
            let visible = full_rect.intersect(panel_rect);
            if visible.is_positive() {
                let inv = egui::vec2(1.0 / img_px.x.max(1.0), 1.0 / img_px.y.max(1.0));
                self.core.uv_offset = ((visible.min - origin) * inv).into();
                self.core.uv_scale = (visible.size() * inv).into();
                ui.painter()
                    .with_clip_rect(panel_rect)
                    .add(render::paint_callback(&self.render, visible));
            }

            response.on_hover_cursor(if pannable {
                egui::CursorIcon::Grab
            } else {
                egui::CursorIcon::Crosshair
            });

            // Scrub frames by scrolling over the image (Ctrl+scroll is zoom, so
            // it's excluded). Two modes:
            //   • normal — discrete wheel *events*, so one mouse notch is exactly
            //     one frame (touchpad pixels accumulate to ~one notch);
            //   • Shift (fast-scroll) — ride the smoothed glide, advancing a
            //     ~10%-of-stack step at `FAST_SCROLL_GLIDE_RATE` per second (time-
            //     scaled, so single- and multi-channel stacks scroll the same),
            //     so one notch sums to ~10% while keeping the smooth glide feel.
            // egui remaps Shift+wheel to horizontal scrolling, so the smoothed
            // delta lands on `.x` with the same sign — `x + y` recovers it.
            if ui.rect_contains_pointer(panel_rect) {
                let shift = ui.input(|i| i.modifiers.shift);
                if shift {
                    let (glide, dt) = ui.input(|i| {
                        let s = i.smooth_scroll_delta;
                        (s.x + s.y, i.stable_dt)
                    });
                    if glide != 0.0 {
                        // ~10% of the stack per notch, spread across the glide.
                        let n_frames = self.core.stack.as_ref().map(|l| l.display.dims.frames).unwrap_or(1);
                        let fast_step = (n_frames as f64 * FAST_SCROLL_RATE).max(1.0);
                        // glide < 0 is scroll-down → advance frames. Advance at a
                        // fixed rate *per second* (scaled by the frame time), so
                        // the jump depends only on the glide's real-time duration
                        // — identical for single- and multi-channel stacks despite
                        // their different render speeds. Fractions accumulate so
                        // short stacks still move.
                        let dir = if glide < 0.0 { 1.0 } else { -1.0 };
                        self.view.scroll_accum += (dir * fast_step * FAST_SCROLL_GLIDE_RATE * dt as f64) as f32;
                        let steps = self.view.scroll_accum.trunc();
                        self.view.scroll_accum -= steps;
                        scroll_step = steps as i32;
                    }
                } else {
                    // Pixels of touchpad scroll that count as one frame step.
                    const POINTS_PER_FRAME: f32 = 50.0;
                    let notches = ui.input(|i| {
                        i.events.iter().fold(0.0_f32, |acc, e| match e {
                            egui::Event::MouseWheel { unit, delta, modifiers, .. } if !modifiers.ctrl => {
                                acc + match unit {
                                    egui::MouseWheelUnit::Point => delta.y / POINTS_PER_FRAME,
                                    _ => delta.y, // Line / Page: one frame per unit
                                }
                            }
                            _ => acc,
                        })
                    });
                    // egui scroll is +y up; we scrub the next frame on scroll-down.
                    self.view.scroll_accum -= notches;
                    let steps = self.view.scroll_accum.trunc();
                    self.view.scroll_accum -= steps;
                    scroll_step = steps as i32;
                }
            } else {
                self.view.scroll_accum = 0.0;
            }
        });

        if scroll_step != 0 {
            if let Some(loaded) = &mut self.core.stack {
                let max_frame = loaded.display.dims.frames.saturating_sub(1) as i64;
                let target = (loaded.frame_index as i64 + scroll_step as i64).clamp(0, max_frame);
                loaded.frame_index = target as usize;
            }
        }

        // Window chrome (sizing, position, title) is a desktop-only concern.
        #[cfg(not(target_arch = "wasm32"))]
        self.manage_window(ui, &toolbar_response, &scrub_bar_response);
        #[cfg(target_arch = "wasm32")]
        let _ = (&toolbar_response, &scrub_bar_response);

        // Bring the GPU up to date with the core state, then honor a pending
        // background volume build by scheduling another frame.
        let Self { core, render, .. } = self;
        let outcome = match render.lock() {
            Ok(mut resources) => core.sync(&mut resources),
            Err(_) => Default::default(),
        };
        if outcome.needs_repaint {
            ui.ctx().request_repaint();
        }
    }
}


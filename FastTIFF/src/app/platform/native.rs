//! The desktop: a window of its own, a filesystem, threads and processes.
//!
//! What [`Platform`] asks for, done the desktop's way. The decisions are not
//! here — they are [`Target`]'s, and this applies `Target::Desktop`'s.

use super::{Delivery, Opening, Platform, Purpose, Target};
use crate::app::{wake_ui, Document, Job, Opened, PanelLayout, View2d};
use fast_tiff_viewer::plugins::{Handover, Registry, ResultTiff};
use fast_tiff_viewer::save::SaveSource;
use fast_tiff_viewer::LoadSource;
use fasttiff_plugin_api::FileType;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Smallest the window is ever sized to (inner size, points). Zooming out past
/// this keeps the window here and just letterboxes the shrinking image.
///
/// Kept in step with `main.rs`, which gives the same floor to the window
/// builder for the user's own resizing.
const MIN_WINDOW: f32 = 256.0;

/// The desktop host.
pub(crate) struct Desktop {
    /// The window title last sent via `ViewportCommand::Title`, so it is sent
    /// again only when it changes.
    last_title: Option<String>,
}

impl Platform for Desktop {
    const WEB: bool = false;

    fn new() -> Self {
        Desktop { last_title: None }
    }

    fn registry() -> Registry {
        // Scans the plugin folders and loads every shared library it finds.
        Registry::with_installed()
    }

    /// On a thread of its own, so the window does not wait for it.
    ///
    /// What this removes is mostly results handed to new windows: those are
    /// left behind on purpose, the window's process reading them for as long
    /// as it is open, and nothing else ever deletes them. A result still on
    /// show in another window survives it on Windows, where a mapped file
    /// cannot be deleted; see `fast_tiff_viewer::plugins::sweep`. Failures are
    /// ignored: the next start tries again.
    fn sweep_old_results() {
        // Not from a test: that is the user's own folder.
        #[cfg(not(test))]
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

    fn take_incoming_files(&mut self, ctx: &egui::Context) -> Vec<Opened> {
        // Dropped onto the window: these carry a path, and dropping several
        // opens the first here and launches the rest in their own processes.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        let mut opened = Vec::new();
        opened.extend(
            crate::process::open_all(&dropped)
                .cloned()
                .map(Opened::Path),
        );

        // macOS "Open With" / double-click delivers files via an Apple Event
        // (not argv); drain whatever `macos_open`'s handler has queued and
        // open them the same way as drag-drop.
        #[cfg(target_os = "macos")]
        opened.extend(
            crate::process::open_all(&crate::macos_open::take_opened_files())
                .cloned()
                .map(Opened::Path),
        );
        opened
    }

    /// A blocking dialog that can select several files — the first opens here
    /// and the rest are launched in their own processes, matching the command
    /// line and drag-drop.
    fn pick_file(&self, _ctx: &egui::Context, readable: &[FileType]) -> Option<Opened> {
        let tiff = ["tif", "tiff"];
        // Importers add their formats to the picker, so a file an installed
        // plugin can read is selectable without the user changing the filter
        // to "all files" and hoping.
        let mut dialog = rfd::FileDialog::new();
        if !readable.is_empty() {
            // A combined entry first, so "open whatever this app can read" is
            // one click rather than a hunt through the format list.
            let mut every: Vec<&str> = tiff.to_vec();
            for t in readable {
                every.extend(t.extensions.iter().map(String::as_str));
            }
            dialog = dialog.add_filter("Images FastTIFF can read", &every);
        }
        dialog = dialog.add_filter("TIFF", &tiff);
        for t in readable {
            let exts: Vec<&str> = t.extensions.iter().map(String::as_str).collect();
            dialog = dialog.add_filter(&t.description, &exts);
        }
        let paths = dialog.pick_files()?;
        crate::process::open_all(&paths).cloned().map(Opened::Path)
    }

    fn path_source(path: PathBuf) -> Result<LoadSource, String> {
        Ok(LoadSource::Path(path))
    }

    /// The dialog blocks the interface, which is right — it is a modal, and
    /// nothing else should be happening while it is up.
    fn ask_path(
        purpose: Purpose,
        name: Option<&str>,
        dir: Option<&Path>,
        filters: &[FileType],
    ) -> Option<PathBuf> {
        let mut dialog = rfd::FileDialog::new();
        for t in filters {
            let exts: Vec<&str> = t.extensions.iter().map(String::as_str).collect();
            dialog = dialog.add_filter(&t.description, &exts);
        }
        if let Some(name) = name {
            dialog = dialog.set_file_name(name);
        }
        if let Some(dir) = dir {
            dialog = dialog.set_directory(dir);
        }
        match purpose {
            Purpose::Open => dialog.pick_file(),
            Purpose::Save => dialog.save_file(),
        }
    }

    fn download(_bytes: &[u8], _name: &str, _mime: &str) -> Result<(), String> {
        Err("there is nowhere to download to outside a browser".into())
    }

    fn write_tiff(
        source: &SaveSource,
        path: &Path,
        report: &mut dyn FnMut(f32) -> bool,
    ) -> Result<(), String> {
        fast_tiff_viewer::save::save_source(source, path, report).map_err(|e| format!("{e:#}"))
    }

    fn write_result(result: ResultTiff, path: &Path) -> Result<(), String> {
        result.save_as(path).map_err(|e| e.to_string())
    }

    fn spawn_job(work: impl FnOnce() + Send + 'static) {
        std::thread::spawn(work);
    }

    fn wake_for_progress() {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        static LAST_MS: AtomicU64 = AtomicU64::new(0);
        let now = START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_millis() as u64;
        if due(&LAST_MS, now, 100) {
            wake_ui();
        }
    }

    /// A window per document is a process per document here: that is already
    /// how the app opens several files at once, and it needs no window manager
    /// of its own inside egui.
    fn deliver_new_document(
        result: ResultTiff,
        name: String,
        progress: &AtomicU32,
        cancel: &AtomicBool,
    ) -> Handover {
        let how = Target::Desktop.delivery(
            result.path().is_some(),
            result.bytes().map(<[u8]>::len),
            crate::process::fits_in_memory,
        );
        match how {
            // The file the result was written to as it arrived. The new window
            // maps it like any file it opens, so the result is never copied
            // into either process: the pages it needs are read in, and the OS
            // may let go of the rest. It is left behind deliberately — the
            // process that would delete it is the one still reading it, and it
            // cannot say when it is done — and the sweep removes it at a later
            // start, once it is a day old. Its file is named after the result,
            // so that is what the new window is titled.
            Delivery::MapFile => {
                let Some(path) = result.path().map(Path::to_path_buf) else {
                    return Handover::Refused(Box::new(result), "the result has no file".into());
                };
                let shown = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(name);
                match crate::process::try_open_in_new_process(&path) {
                    Ok(()) => {
                        let _ = result.into_kept_path();
                        Handover::Opened(shown)
                    }
                    Err(e) => Handover::Refused(Box::new(result), e.to_string()),
                }
            }
            // In memory, because no file could be made for it: through a pipe,
            // the machine having the room for the new window to hold it too.
            Delivery::SendBytes => {
                let bytes = result.bytes().unwrap_or_default();
                match crate::process::open_bytes_in_new_process(bytes, &name, &mut |f| {
                    Job::report(progress, f);
                    !cancel.load(Ordering::Relaxed)
                }) {
                    Ok(()) => Handover::Opened(name),
                    Err(_) if cancel.load(Ordering::Relaxed) => Handover::Cancelled,
                    Err(e) => Handover::Refused(Box::new(result), e.to_string()),
                }
            }
            Delivery::Refuse(why) => Handover::Refused(Box::new(result), why.into()),
        }
    }

    /// Window sizing happens ONLY in response to explicit events — a freshly
    /// opened file, or a zoom change — never every frame. That's what lets the
    /// window be freely resized and maximized without shaking or being
    /// snapped back.
    fn manage_window(
        &mut self,
        ctx: &egui::Context,
        view: &mut View2d,
        panel: &mut PanelLayout,
        document: Option<&Document>,
        toolbar_h: f32,
        bar_h: f32,
    ) {
        let chrome_height = toolbar_h + bar_h;

        // Panel expand/collapse: grow (or shrink) the window height by the
        // panel's own height change, so the image and toolbar above stay put
        // and the panel "drops down" from its position. One-shot, triggered
        // only by the toggle. Skipped when the window manager owns the size
        // (maximized or full screen) — there the
        // image just letterboxes into the space the panel takes. We stay armed
        // until the height actually changes (the toggle frame still reports the
        // old height), repainting meanwhile so the next frame lands.
        if let Some(delta) = panel.grow_delta(bar_h) {
            if !window_size_is_not_ours(ctx) {
                let cur = ctx.content_rect().size();
                let h = (cur.y + delta).round().max(200.0);
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                    cur.x.round(),
                    h,
                )));
            }
        } else if panel.waiting() {
            // The change has not shown up yet. Keep the frames coming so the
            // next one can see it.
            ctx.request_repaint();
        }

        let img_dims = document
            .and_then(|l| l.tiff.frames.first())
            .map(|f| (f.width as f32, f.height as f32));

        if let Some((img_w, img_h)) = img_dims {
            // On open: pick the largest zoom level ≤ 100% at which the image +
            // chrome still fits the monitor (a huge image opens scaled down, a
            // normal one at 100%). Deferred to here because the chrome height
            // and monitor size aren't known at open time.
            if view.pending_initial_fit {
                let image = egui::vec2(img_w, img_h);
                if let Some(z) =
                    Opening::SizeWindow.zoom(image, monitor_work_area(ctx), chrome_height)
                {
                    view.zoom = z;
                    view.pan = egui::Vec2::ZERO;
                    view.pending_initial_fit = false;
                    view.resize_to_zoom = true;
                } else {
                    // Monitor size not reported yet (can stay unknown until the
                    // window first gets focus/input). Poll a few times a second
                    // rather than spinning `request_repaint` every frame, which
                    // would peg a CPU core while the app sits idle on load.
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
            }

            // When the window manager owns the size — maximized or full screen —
            // the window is left completely alone on zoom, and the image just
            // zooms/pans/letterboxes inside it (handled by the central panel's
            // UV transform).
            let size_is_fixed = window_size_is_not_ours(ctx);

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
                let settled = view.zoom_settled();
                let window_scale = match monitor_work_area(ctx) {
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
            if let Some((old_zoom, cursor)) = view.zoom_reposition.take() {
                let new_zoom = view.zoom_settled();
                let fits = monitor_work_area(ctx)
                    .map(|m| img_w * new_zoom <= m.x && img_h * new_zoom + chrome_height <= m.y)
                    .unwrap_or(true);
                // Whether the window grew vs. the previous frame (zoom-in case),
                // and whether the image is now letterboxed inside the window
                // (smaller than the content on either axis).
                let cur_inner = ctx.input(|i| i.viewport().inner_rect.map(|r| r.size()));
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
                    if let Some(outer) = ctx.input(|i| i.viewport().outer_rect.map(|r| r.min)) {
                        let ratio = new_zoom / old_zoom;
                        reposition = Some(outer + (cursor - view.panel_rect.min) * (1.0 - ratio));
                    }
                }
            }

            // Apply a pending resize (one-shot), unless the window manager owns
            // the size.
            if view.resize_to_zoom {
                if let Some(size) = target_window {
                    let (w, h) = (size.x, size.y);
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size));

                    // Keep the window fully on the desktop. The target position is
                    // the cursor-zoom move (or the current position when none).
                    // Horizontally it's clamped to the monitor width. Vertically,
                    // if the (grown) window's bottom would drop below the usable
                    // area, it's *centered* between the top and bottom of the
                    // monitor — symmetric margins, so it's least likely to be
                    // covered by a taskbar whether that's docked at the top or
                    // the bottom (egui doesn't report which).
                    let info = ctx.input(|i| {
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
                        let usable_bottom =
                            monitor_work_area(ctx).map(|wa| wa.y).unwrap_or(monitor.y);
                        let y = if target.y + new_outer.y > usable_bottom {
                            ((monitor.y - new_outer.y) * 0.5).max(0.0)
                        } else {
                            target.y.max(0.0)
                        };
                        let clamped = egui::pos2(target.x.clamp(0.0, max_x), y);
                        if (clamped - outer.min).length() > 0.5 {
                            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(clamped));
                        }
                    } else if let Some(pos) = reposition {
                        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                    }
                }
                view.resize_to_zoom = false;
            }
        }

        // Window title: loaded filename, or the app name when nothing is open.
        let desired_title = match document {
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
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(desired_title.clone()));
            self.last_title = Some(desired_title);
        }
    }

    fn gpu_lost() -> Option<String> {
        crate::render::gpu_lost()
    }

    fn plugin_folder() -> Option<PathBuf> {
        fast_tiff_viewer::plugins::install_dir()
    }

    /// Best effort by design: the plugin folder's path is shown in the status
    /// bar regardless, and a desktop with no file manager — a bare window
    /// manager, a remote session — should not turn "where do plugins go?" into
    /// an error.
    fn reveal_folder(dir: &Path) -> std::io::Result<()> {
        #[cfg(target_os = "windows")]
        let program = "explorer";
        #[cfg(target_os = "macos")]
        let program = "open";
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let program = "xdg-open";
        std::process::Command::new(program)
            .arg(dir)
            .spawn()
            .map(|_| ())
    }
}

/// The window-sizing half of the panel's bookkeeping: what only a host that
/// sizes its own window reads. The rest of [`PanelLayout`] is shared, and is
/// in `app.rs` with its fields.
impl PanelLayout {
    /// How much taller the window should get, if that is settled this frame.
    ///
    /// `Some` disarms. `None` means either nothing was armed or the change has
    /// not shown up yet — [`waiting`](Self::waiting) tells those apart, and the
    /// wait is bounded so a change that never comes stops asking for frames.
    pub(in crate::app) fn grow_delta(&mut self, height: f32) -> Option<f32> {
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
    pub(in crate::app) fn waiting(&self) -> bool {
        self.grow_armed > 0
    }
}

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
fn window_size_is_not_ours(ctx: &egui::Context) -> bool {
    let (maximized, fullscreen) = ctx.input(|i| (i.viewport().maximized, i.viewport().fullscreen));
    maximized.unwrap_or(false) || fullscreen.unwrap_or(false)
}

/// The usable desktop area for the window, i.e. the monitor size minus headroom
/// for the title bar, taskbar, and window borders. `None` until the monitor
/// size is reported. This is the cap on how large the window may grow — beyond
/// it the image overflows the window and becomes pannable.
fn monitor_work_area(ctx: &egui::Context) -> Option<egui::Vec2> {
    ctx.input(|i| i.viewport().monitor_size)
        .map(|m| egui::vec2((m.x * 0.95).max(1.0), (m.y * 0.90).max(1.0)))
}

/// Whether at least `every` ms have passed since the time in `last`, claiming
/// `now` as the new last time when they have — once, however many threads ask
/// at the same moment. `last` starts at 0, so the first ask after `every` ms
/// of running is due.
fn due(last: &AtomicU64, now: u64, every: u64) -> bool {
    let then = last.load(Ordering::Relaxed);
    now.saturating_sub(then) >= every
        && last
            .compare_exchange(then, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;

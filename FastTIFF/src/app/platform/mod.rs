//! The desktop and the browser: everything the two hosts do differently, in
//! one place.
//!
//! The interface is written once, as [`ViewerApp`], and so is everything below
//! it. What the two hosts do differently is of two kinds, and this module has
//! a half for each.
//!
//! * **What each one decides** is a method on [`Target`], a plain value: how a
//!   picture that has just opened is fitted ([`Target::opening`]), where a
//!   file the app writes goes ([`Target::place`]), how a result reaches a
//!   window of its own ([`Target::delivery`]), how large the chrome is drawn,
//!   where a pop-up opens, what the save buttons are called. Being values, both
//!   answers are tested on whichever machine runs the tests — the way
//!   `AppWindow::outlives_document` takes the platform as an argument.
//! * **What each one does** is the [`Platform`] trait, implemented once per
//!   target: `native.rs` for the desktop, `web.rs` for the browser. Taking in
//!   files, the picker, asking where, downloads, running a job, a new process
//!   or a new tab, the OS window. [`ViewerApp`] holds one and calls through it.
//!
//! So nothing outside this module tests which target it was compiled for:
//! what differs is asked of [`TARGET`] (or of [`WEB`], where the pop-up
//! windows take the platform as a `bool`) and done through [`Current`].
//!
//! The `cfg` pair below picks the implementation, and is the only one that
//! does. Elsewhere the crate gates only what cannot compile on the other
//! target:
//!
//! * the host modules declared in `lib.rs`: `process` and `macos_open` on the
//!   desktop (the latter is AppKit's, so macOS's alone), and `web_open` and
//!   `web_save` in the browser, which are written against `web-sys` and
//!   `js-sys` — dependencies that exist only on wasm32;
//! * in `render.rs`, eframe's per-target option hooks, which take
//!   `NativeOptions` on the desktop and `WebOptions` in the browser — each
//!   type exists on its own target only — and the device hand-over to the GPU
//!   registrations (`fast_tiff_viewer::share_gpu_device`), whose GPU paths the
//!   viewer has on the desktop alone;
//! * in `native.rs`, the operating system's own: macOS's Apple Events, and the
//!   file manager each system has.
//!
//! The viewer core gates what needs a filesystem or a thread, and that is why
//! [`Platform::path_source`], [`Platform::write_tiff`] and
//! [`Platform::write_result`] exist: the paths that use them are shared, and
//! reached only where those exist.
//!
//! # Values both targets compile
//!
//! What the shared code hands a platform is plain arguments rather than a
//! struct made to carry them. A field of one that only one implementation
//! reads would be a field never read on the other target, and the lint would
//! say so there — which is the kind of one-sided `allow` this seam replaced.
//!
//! [`Platform::manage_window`] is the exception. The desktop's window sizer is
//! app policy over the view's and the panel's own state — the opening fit, a
//! resize a zoom step asked for, the point that step repositions around, a
//! panel grow in flight — so it is handed [`View2d`] and [`PanelLayout`]
//! themselves and works on them, the panel's grow half (`grow_delta`,
//! `waiting`) being in `native.rs`. In the browser that state is kept up and
//! never acted on: `resize_to_zoom`, `zoom_reposition`, and the panel's
//! `grow_armed` and `old_h`. No lint says so, since the app's own code writes
//! those fields too.
//!
//! [`ViewerApp`]: super::ViewerApp

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub(super) use native::Desktop as Current;
#[cfg(target_arch = "wasm32")]
pub(super) use web::Browser as Current;

mod naming;
use naming::{name_from_path, suggested_name};

use super::{Document, Opened, PanelLayout, View2d, ZOOM_LEVELS};
use fast_tiff_viewer::plugins::{Handover, Registry, ResultTiff};
use fast_tiff_viewer::save::SaveSource;
use fast_tiff_viewer::LoadSource;
use fasttiff_plugin_api::FileType;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32};

/// Whether this is the browser build, for what takes the platform as a `bool`.
pub(super) const WEB: bool = <Current as Platform>::WEB;

/// The target this build is for.
pub(super) const TARGET: Target = if WEB {
    Target::Browser
} else {
    Target::Desktop
};

/// What the app does that each host does its own way.
///
/// Implemented by `native::Desktop` and `web::Browser`; [`Current`] is
/// whichever this build is for. Associated functions where nothing is kept
/// between calls — a worker runs some of them, and holds no `&self` — and
/// methods where the host keeps something: the browser's picker answers
/// through a channel of its own, and the desktop remembers the title it set.
pub(super) trait Platform: Sized {
    /// Whether this is the browser.
    const WEB: bool;

    fn new() -> Self;

    /// The plugins, importers and exporters this host can run: on the
    /// desktop the built-ins and every library installed on disk, in a
    /// browser the built-ins alone — there is no disk to search and no
    /// `dlopen` to load a library with.
    fn registry() -> Registry;

    /// Clear out plugin results more than a day old, without the window
    /// waiting for it. Only the desktop keeps results in files.
    fn sweep_old_results();

    /// What arrived as files since the last frame: dropped on the window,
    /// handed over by the system (macOS's Apple Events), or chosen in a
    /// picker that answers later (the browser's).
    ///
    /// Several dropped on the desktop open the first here and the rest in
    /// processes of their own, as the command line does. A browser drop
    /// carries the bytes, and they are kept as egui's shared buffer rather
    /// than copied — see [`fast_tiff_lib::Bytes`].
    fn take_incoming_files(&mut self, ctx: &egui::Context) -> Vec<Opened>;

    /// Show the file picker. `readable` is what the importers add to TIFF.
    ///
    /// A file chosen there and then is returned. The browser's picker is
    /// asynchronous: it answers `None`, and what is chosen arrives through
    /// [`take_incoming_files`](Self::take_incoming_files) on a later frame.
    fn pick_file(&self, ctx: &egui::Context, readable: &[FileType]) -> Option<Opened>;

    /// The reader's source for a file on disk, or why there is none. Only the
    /// desktop has a disk; nothing in a browser names a path to open.
    fn path_source(path: PathBuf) -> Result<LoadSource, String>;

    /// Ask the user for a path: where to save (`purpose` is
    /// [`Purpose::Save`]), or which file. `name` is the file name to offer,
    /// `dir` the folder to start in, `filters` the formats to list.
    ///
    /// `None` when the user cancelled — and always in a browser, which has no
    /// way to ask. See [`Target::place`], which decides when asking is the
    /// answer at all.
    fn ask_path(
        purpose: Purpose,
        name: Option<&str>,
        dir: Option<&Path>,
        filters: &[FileType],
    ) -> Option<PathBuf>;

    /// Hand `bytes` to the user as a download called `name`, of media type
    /// `mime`: the browser's filesystem. A desktop has nowhere to download to.
    fn download(bytes: &[u8], name: &str, mime: &str) -> Result<(), String>;

    /// Write the stack `source` describes as a TIFF at `path`, reporting
    /// progress to `report`, which answers `false` to stop. Only where there
    /// is a filesystem to write it into.
    fn write_tiff(
        source: &SaveSource,
        path: &Path,
        report: &mut dyn FnMut(f32) -> bool,
    ) -> Result<(), String>;

    /// Put a plugin's result at `path`, moved there when it is already in a
    /// file. Only where there is a filesystem.
    fn write_result(result: ResultTiff, path: &Path) -> Result<(), String>;

    /// Run a job — an import, a plugin, a save, an export — off the frame
    /// that started it, and tell that frame nothing: the job reports through
    /// the channel and the shared progress it was given.
    ///
    /// On the desktop a thread of its own. In the browser, **here and now**,
    /// on the frame that asked, and deliberately: `std::thread::spawn` compiles
    /// on `wasm32-unknown-unknown` and panics when called, and a browser
    /// without shared memory has no worker that could hold the job anyway. Two
    /// things follow, and both are meant. The interface does not repaint while
    /// the job runs, so its progress bar is never seen and Stop cannot be
    /// pressed ([`Target::jobs_run_behind`] says so to the bar). And a result
    /// handed to a new tab is handed over inside the same task as the click
    /// that started the run, which is what makes the browser count the popup
    /// as asked for (see `crate::web_open`).
    fn spawn_job(work: impl FnOnce() + Send + 'static);

    /// A job's progress moved: wake the interface, if that is worth doing.
    /// At most every 100 ms on the desktop; never in the browser, where a job
    /// runs on the interface's own thread and there is no clock to throttle
    /// by (`Instant` has none on wasm32).
    fn wake_for_progress();

    /// Open a plugin's result in a window of its own, called `name`: a new
    /// process on the desktop, a new tab in a browser — whichever
    /// [`Target::delivery`] says for where the result is held. Reports
    /// progress into `progress`; `cancel` is the job's stop flag.
    fn deliver_new_document(
        result: ResultTiff,
        name: String,
        progress: &AtomicU32,
        cancel: &AtomicBool,
    ) -> Handover;

    /// Size, place and title the OS window in response to this frame's
    /// events: `toolbar_h` and `bar_h` are the heights of the chrome above
    /// and below the picture, just laid out. A browser canvas is sized by the
    /// page, and none of it means anything there.
    fn manage_window(
        &mut self,
        ctx: &egui::Context,
        view: &mut View2d,
        panel: &mut PanelLayout,
        document: Option<&Document>,
        toolbar_h: f32,
        bar_h: f32,
    );

    /// Why the graphics device is gone, once it is: what the frame loop
    /// closes with rather than let egui draw without one. The renderer
    /// records the loss on every target (`render::gpu_lost`); only the
    /// desktop closes over it, its panic hook putting the reason in words.
    fn gpu_lost() -> Option<String>;

    /// The folder plugins are installed into, made if it is not there yet.
    /// `None` when there is no writable one — and always in a browser, which
    /// can load no plugin library (see [`Target::has_plugin_folder`]).
    fn plugin_folder() -> Option<PathBuf>;

    /// Show `dir` in the system's file manager, best effort.
    fn reveal_folder(dir: &Path) -> std::io::Result<()>;
}

// ------------------------------------------------------------- the decisions

/// Which host this is, as a value: what each one decides, apart from how it
/// does it. [`TARGET`] is this build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    /// A window of its own, a filesystem, threads and processes.
    Desktop,
    /// A canvas in a page: no filesystem, no threads, downloads and tabs.
    Browser,
}

/// How a picture that has just opened is first fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Opening {
    /// Open at the largest ladder level at most 100% at which the picture and
    /// the chrome still fit the monitor's work area — so a normal image opens
    /// at 100% and a big one at 50% or 25% — and size the window around it.
    /// Decided once the frame's chrome has been laid out, since its height is
    /// part of the sum. The desktop.
    SizeWindow,
    /// Fit the picture to the canvas *exactly*, before it is first drawn, and
    /// make the fit a rung of the picture's zoom ladder so that scrolling back
    /// out returns to it. The browser, whose canvas the page sizes: without it
    /// a gigapixel mosaic opens at 1:1 and shows one corner of itself.
    FitCanvas,
}

/// Where a pop-up window is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Popups {
    /// A top-level window of its own, which can be dragged to another monitor
    /// and left beside the picture. The desktop.
    OwnWindow,
    /// An egui window floating over the canvas: a page cannot lay out a
    /// second document. The browser.
    InCanvas,
}

/// Where a file the app writes goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Destination {
    /// A file, at this path.
    File(PathBuf),
    /// A download, under this name: the browser's only filesystem.
    Download(String),
}

impl Destination {
    /// The name to report it under.
    pub(super) fn name(&self) -> String {
        match self {
            Destination::File(p) => p.display().to_string(),
            Destination::Download(n) => n.clone(),
        }
    }
}

/// Where a file goes, before anyone is asked: decided, or the user's to say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Placement {
    /// Decided: a path a plugin named, or a browser's download.
    At(Destination),
    /// Ask where ([`Platform::ask_path`]), offering this name.
    Ask(String),
}

/// What the save dialog is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Purpose {
    /// Choosing a file to read.
    Open,
    /// Choosing where to write one.
    Save,
}

/// How a plugin's result reaches a window of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Delivery {
    /// Open the file it was written to as it arrived. A new process maps it,
    /// so the result is never copied into either: the desktop, for a result
    /// in a file.
    MapFile,
    /// Hand the bytes over: down a pipe to a new process on the desktop, in a
    /// blob to a new tab in a browser.
    SendBytes,
    /// Do not: show it in this window instead, saying why.
    Refuse(&'static str),
}

/// Why a result the browser cannot hand over was not.
const NOT_IN_MEMORY: &str = "the result is not in memory";

/// Why a result too large to copy was not handed over in memory.
const TOO_LARGE: &str =
    "no file could be written for it, and it is too large to hand over in memory";

impl Target {
    /// Whether this is the browser.
    pub(super) const fn is_web(self) -> bool {
        matches!(self, Target::Browser)
    }

    /// How large the chrome is drawn, as egui's zoom factor. Larger in a
    /// browser: see `app::scale`.
    pub(super) const fn ui_scale(self) -> f32 {
        match self {
            Target::Desktop => 1.0,
            Target::Browser => 1.25,
        }
    }

    /// How a picture that has just opened is fitted.
    pub(super) const fn opening(self) -> Opening {
        match self {
            Target::Desktop => Opening::SizeWindow,
            Target::Browser => Opening::FitCanvas,
        }
    }

    /// Where a pop-up is drawn.
    pub(super) const fn popups(self) -> Popups {
        match self {
            Target::Desktop => Popups::OwnWindow,
            Target::Browser => Popups::InCanvas,
        }
    }

    /// Whether a job runs behind an interface that keeps drawing — a worker
    /// thread — so that its progress can be watched and Stop pressed. Not in
    /// a browser: see [`Platform::spawn_job`].
    pub(super) const fn jobs_run_behind(self) -> bool {
        matches!(self, Target::Desktop)
    }

    /// Whether there is a folder to install plugins into, for the Plugins
    /// menu to offer. Not in a browser, which runs the built-in plugins and
    /// can load no library.
    pub(super) const fn has_plugin_folder(self) -> bool {
        matches!(self, Target::Desktop)
    }

    /// Whether a plugin's path field can offer a browse button. A browser has
    /// no path to produce — its picker hands over bytes, and a plugin that
    /// asked for a path means to open it itself, which it cannot do there
    /// either — so the field is shown with no button.
    pub(super) const fn asks_for_paths(self) -> bool {
        matches!(self, Target::Desktop)
    }

    /// What a button that writes `what` is called: "Save as TIFF…" where the
    /// user is asked where, "Download as TIFF" where the file goes to the
    /// download manager instead, and promising a dialog would be a lie.
    pub(super) fn save_label(self, what: &str) -> String {
        match self {
            Target::Desktop => format!("Save {what}…"),
            Target::Browser => format!("Download {what}"),
        }
    }

    /// The prompt shown while nothing is loaded.
    ///
    /// The browser adds a line the desktop has no need of. A page that
    /// accepts a dropped file is indistinguishable, from the outside, from one
    /// that uploads it — and for microscopy data, which is routinely
    /// unpublished and sometimes clinical, that ambiguity is a real reason not
    /// to try the tool at all. The drop target is the one place the
    /// reassurance arrives before the question does; the README says as much,
    /// but nobody reads it first.
    pub(super) fn welcome_text(self) -> String {
        let mut text = String::from("Drag and drop a file here, \nor click \"Open File\" above.\n");
        if self.is_web() {
            text.push_str(
                "\n\nEverything is processed locally in your browser — \nno file is ever uploaded to a server.\n",
            );
        }
        text.push_str(
            "\n\n\nScroll — navigate frames\nShift + Scroll — fast navigate\nCtrl + Scroll — zoom",
        );
        text
    }

    /// The name to offer for saving the document shown as `shown_as`.
    ///
    /// The file's own stem with a `.tif` on it on the desktop — an imported
    /// stack has no file, and its name is what to suggest saving it as. In a
    /// browser, [`suggested_name`], which also takes the stem at the first dot:
    /// a download cannot be renamed in a dialog first, so `scan.ome.tif` is
    /// offered as `scan.tif` rather than as itself.
    pub(super) fn save_name(self, shown_as: &Path) -> String {
        match self {
            Target::Desktop => {
                let stem = shown_as
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "stack".to_string());
                format!("{stem}.tif")
            }
            Target::Browser => suggested_name(&shown_as.to_string_lossy()),
        }
    }

    /// Where a file goes: `given` is a path a plugin chose, empty when it
    /// chose none, and `suggested` the name to offer when there is a choice.
    ///
    /// The desktop writes where it was told, or asks. A browser cannot be
    /// asked for a place, only given a name — the plugin's, without its
    /// directories ([`name_from_path`]), or the suggestion.
    pub(super) fn place(self, given: &str, suggested: String) -> Placement {
        if !given.is_empty() {
            return Placement::At(self.plugin_file(given));
        }
        match self {
            Target::Desktop => Placement::Ask(suggested),
            Target::Browser => Placement::At(Destination::Download(suggested)),
        }
    }

    /// Where a file a plugin asked the host to write at `path` goes, never
    /// asking: the run is over by then, on a worker, and a dialog is no
    /// business of a worker's. The desktop honours the path; a browser
    /// downloads under its last component, the one part of it a tab can
    /// honour — the same name on both, so one plugin does not make two
    /// differently named results.
    pub(super) fn plugin_file(self, path: &str) -> Destination {
        match self {
            Target::Desktop => Destination::File(path.into()),
            Target::Browser => Destination::Download(name_from_path(path)),
        }
    }

    /// How a result reaches a window of its own, given where it is held: in a
    /// file (`in_file`), or in memory, `in_memory` bytes of it.
    /// `fits_in_memory` says whether that many bytes may be copied to a new
    /// process, and is asked only on the desktop.
    ///
    /// The desktop maps a result's file in a new process, so it is copied
    /// into neither; one in memory goes down a pipe when the machine has room
    /// for the copy, and is refused otherwise. A browser has no files: what it
    /// holds in memory goes to a new tab.
    pub(super) fn delivery(
        self,
        in_file: bool,
        in_memory: Option<usize>,
        fits_in_memory: impl FnOnce(usize) -> bool,
    ) -> Delivery {
        match (self, in_file, in_memory) {
            (Target::Desktop, true, _) => Delivery::MapFile,
            (Target::Desktop, false, Some(n)) if fits_in_memory(n) => Delivery::SendBytes,
            (Target::Desktop, false, Some(_)) => Delivery::Refuse(TOO_LARGE),
            (Target::Browser, _, Some(_)) => Delivery::SendBytes,
            (_, _, None) => Delivery::Refuse(NOT_IN_MEMORY),
        }
    }
}

impl Opening {
    /// The zoom an `image` (width, height) that has just opened is shown at.
    ///
    /// `room` is the monitor's work area for [`Opening::SizeWindow`] and the
    /// canvas for [`Opening::FitCanvas`], and `chrome_h` the height of the
    /// bars above and below the picture, which only a window sized around the
    /// picture has to make room for. `None` while the room is not known yet —
    /// the monitor's size can be unreported until the window first gets
    /// input, and a canvas has no size before it is first laid out — so the
    /// caller waits rather than fitting to nothing.
    pub(super) fn zoom(
        self,
        image: egui::Vec2,
        room: Option<egui::Vec2>,
        chrome_h: f32,
    ) -> Option<f32> {
        let room = room?;
        match self {
            Opening::SizeWindow => {
                // Largest zoom level at most 100% that still fits the work area.
                for &z in ZOOM_LEVELS.iter().rev().filter(|&&z| z <= 1.0) {
                    if image.x * z <= room.x && image.y * z + chrome_h <= room.y {
                        return Some(z);
                    }
                }
                // Even the smallest level overflows: open there and pan.
                Some(ZOOM_LEVELS[0])
            }
            Opening::FitCanvas => fit_to_panel(image.x, image.y, room),
        }
    }
}

/// The zoom at which an `img_w` x `img_h` image exactly fits `avail`, capped at
/// 1:1 so a small image is not magnified into a blur. `None` for a degenerate
/// size on either side.
pub(super) fn fit_to_panel(img_w: f32, img_h: f32, avail: egui::Vec2) -> Option<f32> {
    if !(img_w >= 1.0 && img_h >= 1.0 && avail.x >= 1.0 && avail.y >= 1.0) {
        return None;
    }
    let z = (avail.x / img_w).min(avail.y / img_h).min(1.0);
    z.is_finite().then_some(z).filter(|z| *z > 0.0)
}

#[cfg(test)]
mod tests;

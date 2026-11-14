//! A [`ViewerApp`] with no window and no GPU, run a whole frame at a time.
//!
//! What eframe does for a real window, done by hand: a [`egui::RawInput`] per
//! frame — the window's size and scale, the monitor it is on, the pointer and
//! the keys, files dropped on it — and the app's [`eframe::App`] methods called
//! inside [`egui::Context::run_ui`] in the order eframe calls them (its
//! `EpiIntegration::update`, and the web runner's twin of it), with the empty
//! [`eframe::Frame`] eframe has for testing an app (`Frame::_new_kittest`,
//! hidden from its docs; `egui_kittest` is built on it). What the app asks of
//! its window comes back as viewport commands, and the window does with a
//! resize what a real one does: the next frame is laid out at the new size.
//!
//! No GPU. The app is built without a renderer ([`ViewerApp::render`] is
//! `None`), so a frame is decided and laid out as in a real run and only the
//! sync that would bring a device up to date with it is left out; what the
//! frame decided for the sync is in `sync_inputs`. That is the viewer core's
//! own split — it is a CPU-side model with or without a backend — rather than
//! a stand-in GPU.
//!
//! The host's decisions follow a [`Target`], so the browser's are run here
//! too. What the host *does* is still this build's [`Current`], the desktop's
//! on a test machine. A browser run reports its window maximized, which is how
//! the desktop's window sizing is told that the size is not its own to set —
//! as a browser canvas's is the page's.
//!
//! Pop-ups are embedded, on either target. eframe gives a desktop pop-up a
//! viewport of its own (`set_embed_viewports(false)`, with a renderer for
//! immediate viewports); here the context keeps egui's default and has no such
//! renderer, so `dialog`'s window of its own falls back to an `egui::Window`
//! in the root viewport, where its texts and clicks are found. What a pop-up
//! shows and does is the same either way; how the desktop sizes, focuses and
//! closes its window is not tested here.

use super::*;
use egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect, Vec2, ViewportCommand, ViewportId};

/// The frame time: a 60 Hz display. Also the `predicted_dt` every frame is
/// given, which egui takes off each delay a frame asks for its next one after
/// ([`Headless::repaint_after`]).
pub(super) const DT: f64 = 1.0 / 60.0;

/// The monitor every window here is on, in points.
const MONITOR: Vec2 = Vec2::new(1920.0, 1080.0);

/// Where the window's inside sits on the monitor, and the frame around it.
const INNER_AT: Pos2 = Pos2::new(100.0, 80.0);
const TITLE_BAR: f32 = 30.0;

/// How long a test waits for a load on another thread to land.
const PATIENCE: std::time::Duration = std::time::Duration::from_secs(20);

/// Ctrl held, as egui-winit reports it on Windows and Linux: `command` is
/// set with `ctrl` there, which is what a chord written for Cmd/Ctrl
/// (`Modifiers::COMMAND`) is matched against.
pub(super) const CTRL: Modifiers = Modifiers {
    ctrl: true,
    command: true,
    ..Modifiers::NONE
};

/// A window, its app, and what the last frame put out.
pub(super) struct Headless {
    pub(super) app: ViewerApp,
    pub(super) ctx: egui::Context,
    frame: eframe::Frame,
    /// The window's inside, in points: what the next frame is laid out in.
    size: Vec2,
    pixels_per_point: f32,
    /// The window manager has the window's size: maximized.
    maximized: bool,
    /// The frontend clock (`RawInput::time`), seconds.
    time: f64,
    modifiers: Modifiers,
    /// What the next frame takes in.
    events: Vec<Event>,
    dropped: Vec<egui::DroppedFile>,
    /// What the last frame put out.
    pub(super) output: egui::FullOutput,
    /// Every command the app sent its window, oldest first.
    pub(super) sent: Vec<ViewportCommand>,
}

impl Headless {
    /// An empty window of 800 x 600 points at one pixel per point, following
    /// `target`'s decisions and offering the built-in plugins.
    pub(super) fn new(target: Target) -> Self {
        // A worker wakes the interface through one process-wide context
        // (`UI`), recorded by the first frame drawn. A test process draws
        // many windows at once, on threads of their own, and a job in one of
        // them must not ask for another's frames: so the context woken is one
        // that no test draws, and is set before any of them can be recorded.
        let _ = UI.set(egui::Context::default());
        let ctx = egui::Context::default();
        // As both hosts do before the first frame.
        install_chrome(&ctx);
        Headless {
            app: ViewerApp::assemble(None, target, fast_tiff_viewer::plugins::Registry::new()),
            ctx,
            frame: eframe::Frame::_new_kittest(),
            size: Vec2::new(800.0, 600.0),
            pixels_per_point: 1.0,
            maximized: target.is_web(),
            time: 0.0,
            modifiers: Modifiers::NONE,
            events: Vec::new(),
            dropped: Vec::new(),
            output: egui::FullOutput::default(),
            sent: Vec::new(),
        }
    }

    /// The window `size` points across, `pixels_per_point` device pixels to a
    /// point.
    pub(super) fn sized(mut self, size: Vec2, pixels_per_point: f32) -> Self {
        self.size = size;
        self.pixels_per_point = pixels_per_point;
        self
    }

    /// Maximized (`true`): its size is the window manager's, not the app's.
    pub(super) fn maximized(mut self, maximized: bool) -> Self {
        self.maximized = maximized;
        self
    }

    /// The window's inside, in points.
    pub(super) fn size(&self) -> Vec2 {
        self.size
    }

    /// Open `opened` as a host opens the file it was started with.
    pub(super) fn open(&mut self, opened: Opened) {
        self.app.apply_opened(opened);
    }

    /// Open `bytes` as a file called `name`, and run frames until it is on
    /// show.
    pub(super) fn open_bytes(&mut self, bytes: Vec<u8>, name: &str) {
        self.open(Opened::Bytes(bytes.into(), name.to_string()));
        let before = self.app.document_generation();
        self.run_until("the file to land", |app| {
            app.document_generation() != before
        });
        assert!(
            self.app.core.document().is_some(),
            "{name} did not open: {:?}",
            self.app.core.notices().collect::<Vec<_>>()
        );
    }

    // ------------------------------------------------------------- frames

    /// Run one frame: take in what was queued, lay the window out, and do
    /// with its commands what a window would.
    pub(super) fn frame(&mut self) {
        self.time += DT;
        let mut raw = self.raw_input();
        let Headless {
            app, ctx, frame, ..
        } = self;
        eframe::App::raw_input_hook(app, ctx, &mut raw);
        let output = ctx.run_ui(raw, |ui| {
            eframe::App::logic(app, ui.ctx(), frame);
            #[allow(deprecated)]
            eframe::App::update(app, ui.ctx(), frame);
            eframe::App::ui(app, ui, frame);
        });
        if let Some(root) = output.viewport_output.get(&ViewportId::ROOT) {
            for command in &root.commands {
                if let ViewportCommand::InnerSize(size) = command {
                    self.size = *size;
                }
                self.sent.push(command.clone());
            }
        }
        self.output = output;
    }

    /// Run `n` frames.
    pub(super) fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.frame();
        }
    }

    /// Run frames until `done` says so, waiting on work running on other
    /// threads (a load, a plugin) between them.
    pub(super) fn run_until(&mut self, what: &str, mut done: impl FnMut(&ViewerApp) -> bool) {
        let start = std::time::Instant::now();
        loop {
            self.frame();
            if done(&self.app) {
                return;
            }
            assert!(start.elapsed() < PATIENCE, "gave up waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Run frames until the window stops asking for them, as an idle one
    /// does; panics, saying what asked, if it never does.
    pub(super) fn settle(&mut self) {
        for _ in 0..30 {
            self.frame();
            if self.repaint_after() == std::time::Duration::MAX {
                return;
            }
        }
        // What asked, on the frame before this one.
        self.frame();
        panic!(
            "the window kept asking for frames: {:?}",
            self.ctx.repaint_causes()
        );
    }

    /// How soon the last frame asked for the next: `Duration::MAX` for not at
    /// all, until something happens. As eframe is told it, the soonest delay
    /// asked for less the frame time ([`DT`]), so that the frame drawn then
    /// is ready when it is due.
    pub(super) fn repaint_after(&self) -> std::time::Duration {
        self.output
            .viewport_output
            .get(&ViewportId::ROOT)
            .map_or(std::time::Duration::MAX, |root| root.repaint_delay)
    }

    fn raw_input(&mut self) -> egui::RawInput {
        let inner = Rect::from_min_size(INNER_AT, self.size);
        let outer = Rect::from_min_max(inner.min - Vec2::new(0.0, TITLE_BAR), inner.max);
        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, self.size)),
            time: Some(self.time),
            predicted_dt: DT as f32,
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            dropped_files: std::mem::take(&mut self.dropped),
            focused: true,
            ..Default::default()
        };
        let root = raw.viewports.entry(ViewportId::ROOT).or_default();
        root.native_pixels_per_point = Some(self.pixels_per_point);
        root.monitor_size = Some(MONITOR);
        root.inner_rect = Some(inner);
        root.outer_rect = Some(outer);
        root.maximized = Some(self.maximized);
        root.fullscreen = Some(false);
        root.focused = Some(true);
        raw
    }

    // -------------------------------------------------------------- input

    /// Drop the file at `path` on the window.
    pub(super) fn drop_file(&mut self, path: &std::path::Path) {
        self.dropped.push(egui::DroppedFile {
            path: Some(path.to_path_buf()),
            name: path.display().to_string(),
            ..Default::default()
        });
    }

    /// Move the pointer to `at` (one frame).
    pub(super) fn hover(&mut self, at: Pos2) {
        self.events.push(Event::PointerMoved(at));
        self.frame();
    }

    /// Click `button` at `at`: pressed on one frame, released on the next,
    /// which is the frame the click is seen on.
    pub(super) fn click_with(&mut self, button: PointerButton, at: Pos2) {
        self.events.push(Event::PointerMoved(at));
        self.events.push(self.button(button, at, true));
        self.frame();
        self.events.push(self.button(button, at, false));
        self.frame();
    }

    /// Click the primary button at `at`.
    pub(super) fn click(&mut self, at: Pos2) {
        self.click_with(PointerButton::Primary, at);
    }

    /// Click the middle of the text `text` where the last frame drew it.
    pub(super) fn click_text(&mut self, text: &str) {
        let at = self.text_rect(text).center();
        self.click(at);
    }

    /// Drag with `button` from `from` to `to`, over `steps` frames of
    /// movement between the press and the release.
    pub(super) fn drag_with(&mut self, button: PointerButton, from: Pos2, to: Pos2, steps: usize) {
        self.events.push(Event::PointerMoved(from));
        self.events.push(self.button(button, from, true));
        self.frame();
        for i in 1..=steps {
            let at = from + (to - from) * (i as f32 / steps as f32);
            self.events.push(Event::PointerMoved(at));
            self.frame();
        }
        self.events.push(self.button(button, to, false));
        self.frame();
    }

    /// Drag with the primary button.
    pub(super) fn drag(&mut self, from: Pos2, to: Pos2) {
        self.drag_with(PointerButton::Primary, from, to, 5);
    }

    /// Press `key` with `modifiers` held: down on one frame, up on the next.
    pub(super) fn key(&mut self, key: Key, modifiers: Modifiers) {
        self.modifiers = modifiers;
        self.events.push(self.key_event(key, true));
        self.frame();
        self.events.push(self.key_event(key, false));
        self.modifiers = Modifiers::NONE;
        self.frame();
    }

    /// Turn the mouse wheel by `notches` over `at`, positive away from the
    /// user (egui's up), in one frame.
    pub(super) fn wheel(&mut self, at: Pos2, notches: f32) {
        self.events.push(Event::PointerMoved(at));
        self.events.push(Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: Vec2::new(0.0, notches),
            phase: egui::TouchPhase::Move,
            modifiers: self.modifiers,
        });
        self.frame();
    }

    fn button(&self, button: PointerButton, pos: Pos2, pressed: bool) -> Event {
        Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: self.modifiers,
        }
    }

    fn key_event(&self, key: Key, pressed: bool) -> Event {
        Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: self.modifiers,
        }
    }

    // ----------------------------------------------------- what was drawn

    /// Every piece of text the last frame drew, and where.
    pub(super) fn texts(&self) -> Vec<(String, Rect)> {
        let mut out = Vec::new();
        for clipped in &self.output.shapes {
            walk(&clipped.shape, &mut |shape| {
                if let egui::Shape::Text(text) = shape {
                    out.push((
                        text.galley.text().to_string(),
                        text.galley.rect.translate(text.pos.to_vec2()),
                    ));
                }
            });
        }
        out
    }

    /// Whether the last frame drew text containing `text`.
    pub(super) fn shows(&self, text: &str) -> bool {
        self.texts().iter().any(|(t, _)| t.contains(text))
    }

    /// Whether the last frame drew `text` as a piece of text of its own.
    pub(super) fn drew(&self, text: &str) -> bool {
        self.texts().iter().any(|(t, _)| t == text)
    }

    /// Where the last frame drew the text `text`, exactly; panics if it did
    /// not.
    pub(super) fn text_rect(&self, text: &str) -> Rect {
        self.texts()
            .into_iter()
            .find_map(|(t, rect)| (t == text).then_some(rect))
            .unwrap_or_else(|| panic!("no {text:?} on screen: {:?}", self.texts()))
    }

    /// The filled triangles the last frame drew — the bottom bar's painted
    /// icons among them — as their three corners, in the order drawn.
    pub(super) fn triangles(&self) -> Vec<[Pos2; 3]> {
        let mut out = Vec::new();
        for clipped in &self.output.shapes {
            walk(&clipped.shape, &mut |shape| {
                if let egui::Shape::Path(path) = shape {
                    if let [a, b, c] = path.points[..] {
                        out.push([a, b, c]);
                    }
                }
            });
        }
        out
    }

    /// The titles the app gave its window, oldest first.
    pub(super) fn titles(&self) -> Vec<&str> {
        self.sent
            .iter()
            .filter_map(|c| match c {
                ViewportCommand::Title(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The sizes the app asked its window to take, oldest first.
    pub(super) fn resizes(&self) -> Vec<Vec2> {
        self.sent
            .iter()
            .filter_map(|c| match c {
                ViewportCommand::InnerSize(size) => Some(*size),
                _ => None,
            })
            .collect()
    }
}

/// Call `f` on `shape` and on every shape inside it.
fn walk(shape: &egui::Shape, f: &mut impl FnMut(&egui::Shape)) {
    match shape {
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                walk(shape, f);
            }
        }
        shape => f(shape),
    }
}

/// A `frames`-frame 8-bit stack `w` x `h`, each frame a flat grey of its own,
/// as TIFF bytes.
pub(super) fn stack(w: u32, h: u32, frames: u8) -> Vec<u8> {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(w, h, SampleType::U8);
    let mut writer = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames {
        writer
            .write_frame_u8(&vec![f.wrapping_mul(40); (w * h) as usize])
            .unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// A file in the temporary directory, named for this process, removed again
/// (best effort: a document may still have it mapped) when dropped.
pub(super) struct TempFile(std::path::PathBuf);

impl TempFile {
    pub(super) fn new(name: &str, bytes: &[u8]) -> Self {
        let path =
            std::env::temp_dir().join(format!("fasttiff-headless-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        TempFile(path)
    }

    pub(super) fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

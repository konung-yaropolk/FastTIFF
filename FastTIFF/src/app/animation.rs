//! Carrying out a plugin's animation of the 3D view: playing it, and recording
//! one cycle of it to a TIFF.
//!
//! A plugin declares the motion — one pose of the volume per frame — and this
//! is the host's half: show the poses in turn, stop the moment the user takes
//! the view back, and for a save, render each pose once and keep the picture.
//!
//! # The camera is not touched
//!
//! A pose turns the *volume* (see `VolumeView::pose` in `fast-tiff-viewer`),
//! and the user's camera stays exactly as they left it. That is what makes the
//! stop rule cheap and exact: every way of moving the view — a drag, the wheel,
//! the arrow and WASD keys, Reset, switching the navigation style — changes the
//! camera, so a camera that is no longer the one the animation started from
//! *is* the user taking the view back, whatever gesture did it. A press on the
//! view that moves nothing counts too, because grabbing a spinning thing is
//! how people stop it. And because the camera never moved, stopping puts the
//! view back where it was by simply dropping the pose.
//!
//! # Recording renders each pose off screen
//!
//! The 3D view is a ray-march with no resolution of its own (every pixel is a
//! ray), so the movie is not tied to the window: each pose is rendered again,
//! off screen, at the movie's size, and read back. The view on screen shows
//! the same pose meanwhile, so the user sees what is being recorded, and
//! nothing drawn over the view (the coordinate box, a dialog in the browser)
//! is in the movie.
//!
//! The size is chosen once, when the first frame is rendered: the shape of the
//! view on screen, made as large as it takes for every pixel of the stack in
//! view to be at least a pixel of the movie, and never smaller than the view
//! is on screen. See [`movie_size`]. Or, when the plugin asks for
//! `MovieResolution::Window`, the view's own size.
//!
//! The renderer draws a frame in tiles, a tile a frame of the interface, so a
//! large frame takes several before it comes back. See
//! `scivis_render::CAPTURE_TILE` for why.
//!
//! A readback lands a frame or more after it is asked for, and only then is
//! the next pose rendered, so every pose is in the movie exactly once.
//!
//! On the desktop the frames go to a writer thread as they arrive, so a long
//! recording is never held in memory whole; in the browser, which has no
//! threads and no files, they are kept and the movie is downloaded at the end.

use fast_tiff_lib::{Compression, SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::camera::{CameraState, NavMode, OrbitPoint};
use fasttiff_plugin_api::{Animation, AnimationOutput, MovieResolution};
use scivis_render::VolumeImage;
use std::io::{Seek, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Frames to let pass once the volume is ready, before the first render: the
/// first frame after a switch to 3D lays the view out, and the one after it is
/// the first drawn at its final size, which is the shape the movie takes.
const WARMUP_FRAMES: u32 = 2;

/// The longest side a movie frame may have, whatever the GPU allows: a frame
/// this size is already 200 MB of RGB to compress.
pub(super) const MOVIE_MAX_SIDE: u32 = 8192;

/// Frames the writer thread may have waiting before the next pose is held
/// back. A large movie compresses more slowly than it renders, and frames
/// queued without a limit would pile up in memory.
#[cfg(not(target_arch = "wasm32"))]
const WRITER_QUEUE: usize = 3;

/// How long to wait for the writer to catch up before looking again, in
/// seconds.
const WRITER_WAIT: f64 = 0.01;

/// The user's camera, as it was when the animation started.
///
/// Everything the user can change by moving the view, and nothing the host
/// changes behind them: `box_he` is a cache the renderer refreshes every frame
/// and is deliberately left out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CameraMark {
    yaw: f32,
    pitch: f32,
    dist: f32,
    target: [f32; 3],
    fly_pos: [f32; 3],
    nav: NavMode,
    orbit_point: OrbitPoint,
}

impl CameraMark {
    pub(super) fn of(cam: &CameraState) -> CameraMark {
        CameraMark {
            yaw: cam.yaw,
            pitch: cam.pitch,
            dist: cam.dist,
            target: cam.target,
            fly_pos: cam.fly_pos,
            nav: cam.nav,
            orbit_point: cam.orbit_point,
        }
    }

    /// Whether the user has moved the view since this was taken.
    pub(super) fn moved(&self, cam: &CameraState) -> bool {
        *self != CameraMark::of(cam)
    }

    /// Whether the two see the same picture: the pose alone, without the
    /// navigation style and orbiting point, which change how the next gesture
    /// moves the view and not the view.
    pub(super) fn same_pose(&self, other: &CameraMark) -> bool {
        (self.yaw, self.pitch, self.dist, self.target, self.fly_pos)
            == (
                other.yaw,
                other.pitch,
                other.dist,
                other.target,
                other.fly_pos,
            )
    }
}

/// Why an animation ended.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Ended {
    /// The user took the view back: moved it, or pressed on it.
    Interacted,
    /// The view went back to 2D.
    LeftVolume,
    /// A different stack, or the same one re-laid out, is in the view now.
    StackChanged,
    /// The plugin, or the user through it, asked.
    Asked,
    /// Another animation took its place.
    Replaced,
    /// A recording could not go on.
    Failed(String),
}

impl Ended {
    /// What to tell the user, for a recording that ended this way. A playing
    /// animation ending is not news — the user is the one who stopped it — but
    /// a save that did not happen is.
    pub(super) fn recording_message(&self) -> String {
        let why = match self {
            Ended::Interacted => "the view was moved",
            Ended::LeftVolume => "the view went back to 2D",
            Ended::StackChanged => "a different stack was opened",
            Ended::Asked => "Stop was pressed",
            Ended::Replaced => "another animation was started",
            Ended::Failed(why) => why.as_str(),
        };
        format!("Recording stopped: {why}. Nothing was saved.")
    }
}

/// Where a recorded movie goes.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Destination {
    /// A file, written as the frames arrive. Desktop only.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    File(std::path::PathBuf),
    /// A download, under this name. The browser's only filesystem.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Download(String),
}

impl Destination {
    /// The name to report the movie under.
    pub(super) fn name(&self) -> String {
        match self {
            Destination::File(p) => p.display().to_string(),
            Destination::Download(n) => n.clone(),
        }
    }
}

/// A recording in progress.
pub(super) struct Recording {
    /// The pose whose render has been asked for and has not come back.
    waiting: Option<usize>,
    warmup: u32,
    /// The size of every frame, in pixels, fixed when the first is asked for:
    /// every frame of a TIFF stack is the same size.
    size: Option<(u32, u32)>,
    frames: FrameSink,
    /// Frames rendered so far.
    pub(super) done: usize,
    pub(super) dest: Destination,
    /// How large the plugin asked the frames to be.
    pub(super) resolution: MovieResolution,
}

/// An animation being carried out.
pub(super) struct Running {
    /// The plugin that started it, by its registry index — whose dialog's
    /// button reads Stop, and whose Stop stops it.
    pub(super) plugin: usize,
    poses: Vec<[[f32; 3]; 3]>,
    fps: f32,
    mark: CameraMark,
    /// The volume's stack generation when it started: a different number is a
    /// different stack, or the same one with its axes reassigned. (Not the
    /// builder's generation, which also moves when the same volume is rebuilt
    /// smaller for want of GPU memory: that is waited out, like a first build.)
    generation: u64,
    /// The frontend clock when pose 0 was first shown. `None` until the volume
    /// is ready to be drawn, so a long first build does not eat into the cycle.
    started: Option<f64>,
    pub(super) recording: Option<Recording>,
}

/// What one frame should look like, according to the animation.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Step {
    /// Show this pose. If `capture` is set, also render it off screen at that
    /// size, for the movie. Ask for the next frame no later than
    /// `repaint_after` seconds from now.
    Show {
        pose: [[f32; 3]; 3],
        capture: Option<(u32, u32)>,
        repaint_after: f64,
    },
    /// The animation is over.
    End(Ended),
    /// A recording has every frame; the movie is being finished.
    Recorded,
}

/// What the frame the animation is stepping through looks like.
pub(super) struct Frame<'a> {
    /// The frontend clock, in seconds.
    pub(super) now: f64,
    pub(super) in_volume: bool,
    /// The volume's stack generation (`VolumeView::stack_generation`).
    pub(super) generation: u64,
    pub(super) camera: &'a CameraState,
    /// Whether the pointer was pressed on the 3D view since the last frame.
    pub(super) pressed: bool,
    /// Whether the volume is built and the view has been drawn at least once.
    pub(super) ready: bool,
    /// The size a movie of the view would be now (see [`movie_size`]).
    pub(super) movie_size: Option<(u32, u32)>,
    /// Whether an off-screen render is still on its way back, so another
    /// cannot be asked for yet.
    pub(super) capture_busy: bool,
    /// The off-screen render that came back this frame, if one did.
    pub(super) shot: Option<Result<VolumeImage, String>>,
}

impl Running {
    /// Start carrying out `animation` for `plugin`, from the camera `cam` and
    /// the volume generation `generation`. `recording` is where the frames go,
    /// for a save.
    pub(super) fn new(
        plugin: usize,
        animation: Animation,
        cam: &CameraState,
        generation: u64,
        recording: Option<(Destination, FrameSink)>,
    ) -> Running {
        let resolution = animation.resolution;
        Running {
            plugin,
            poses: animation.poses.into_iter().map(|p| p.rotation).collect(),
            fps: animation.fps,
            mark: CameraMark::of(cam),
            generation,
            started: None,
            recording: recording.map(|(dest, frames)| Recording {
                waiting: None,
                warmup: WARMUP_FRAMES,
                size: None,
                frames,
                done: 0,
                dest,
                resolution,
            }),
        }
    }

    /// How many frames one cycle is.
    pub(super) fn total(&self) -> usize {
        self.poses.len()
    }

    /// Decide this frame.
    pub(super) fn step(&mut self, f: Frame<'_>) -> Step {
        // The stop rules first, and for a recording as much as for a play: a
        // movie of a view the user was moving is not the movie they asked for.
        if !f.in_volume {
            return Step::End(Ended::LeftVolume);
        }
        if f.generation != self.generation {
            return Step::End(Ended::StackChanged);
        }
        if f.pressed || self.mark.moved(f.camera) {
            return Step::End(Ended::Interacted);
        }

        let period = 1.0 / self.fps as f64;
        let Some(rec) = self.recording.as_mut() else {
            // Playing. The clock starts when there is something to see — and
            // only then is readiness asked about: a 4D stack whose movie plays
            // on rebuilds its volume every timepoint, and snapping back to the
            // first pose for each of those builds would make it judder.
            let started = match self.started {
                Some(s) => s,
                None if f.ready => *self.started.insert(f.now),
                None => {
                    return Step::Show {
                        pose: self.poses[0],
                        capture: None,
                        repaint_after: period,
                    }
                }
            };
            let i = pose_index(f.now - started, self.fps, self.poses.len());
            return Step::Show {
                pose: self.poses[i],
                capture: None,
                repaint_after: period,
            };
        };

        // Recording: not paced by the clock but by the renders, one pose per
        // render, each asked for as soon as the last is in.
        let asap = 0.0;
        let show = |pose: usize, capture: Option<(u32, u32)>, repaint_after: f64| Step::Show {
            pose: self.poses[pose],
            capture,
            repaint_after,
        };
        if !f.ready {
            return show(0, None, asap);
        }
        if rec.warmup > 0 {
            rec.warmup -= 1;
            return show(0, None, asap);
        }
        if let Some(i) = rec.waiting {
            let Some(shot) = f.shot else {
                return show(i, None, asap);
            };
            rec.waiting = None;
            let image = match shot {
                Ok(image) => image,
                Err(why) => return Step::End(Ended::Failed(why)),
            };
            if Some((image.width, image.height)) != rec.size {
                return Step::End(Ended::Failed(format!(
                    "a frame came back {} x {} instead of the movie's size",
                    image.width, image.height
                )));
            }
            let size = (image.width as usize, image.height as usize);
            if let Err(why) = rec.frames.push(rgb_of(&image), size) {
                return Step::End(Ended::Failed(why));
            }
            rec.done = i + 1;
            if rec.done == self.poses.len() {
                return Step::Recorded;
            }
        }
        let next = rec.done;
        let Some(size) = rec.size.or(f.movie_size) else {
            return show(next, None, asap);
        };
        if f.capture_busy || rec.frames.busy() {
            return show(next, None, WRITER_WAIT);
        }
        rec.size = Some(size);
        rec.waiting = Some(next);
        show(next, Some(size), asap)
    }
}

/// The size of a movie of the 3D view, in pixels: the shape of the view as it
/// is on screen (`panel`, in pixels), as large as it takes for a pixel of the
/// stack in view to be at least a pixel of the movie (`detail_height` — see
/// `VolumeView::detail_height`), and never smaller than the view on screen.
///
/// Scaled down, keeping the shape, to `max_side`. `None` for a view with no
/// size to copy.
pub(super) fn movie_size(
    panel: (f32, f32),
    detail_height: f32,
    max_side: u32,
) -> Option<(u32, u32)> {
    let (w, h) = panel;
    if !(w >= 1.0 && h >= 1.0 && w.is_finite() && h.is_finite()) {
        return None;
    }
    let grow = if detail_height.is_finite() {
        (detail_height / h).max(1.0)
    } else {
        1.0
    };
    let (w, h) = (w * grow, h * grow);
    let fit = (max_side.max(1) as f32 / w.max(h)).min(1.0);
    let side = |v: f32| ((v * fit).round() as u32).clamp(1, max_side.max(1));
    Some((side(w), side(h)))
}

/// Which of `n` poses to show `elapsed` seconds in, at `fps` a second, looping.
pub(super) fn pose_index(elapsed: f64, fps: f32, n: usize) -> usize {
    if n == 0 || !elapsed.is_finite() || elapsed <= 0.0 {
        return 0;
    }
    ((elapsed * fps as f64).floor() as u64 % n as u64) as usize
}

/// A rendered view's pixels as packed RGB, which is what the movie stores.
///
/// The ray-march is opaque, so the alpha carries nothing.
pub(super) fn rgb_of(image: &VolumeImage) -> Vec<u8> {
    let mut out = Vec::with_capacity(image.rgba.len() / 4 * 3);
    for p in image.rgba.chunks_exact(4) {
        out.extend_from_slice(&p[..3]);
    }
    out
}

/// How the movie is written: 8-bit RGB, one frame per pose, at the animation's
/// rate, marked to loop.
///
/// Deflate at its fastest setting. A render is mostly black around the volume,
/// which squeezes to a fraction of its size at almost no cost, and a writer
/// that fell far behind the captures would hold frames in memory waiting.
pub(super) fn movie_options(width: usize, height: usize, fps: f32) -> WriterOptions {
    let meta = StackMetaWrite::new(1, 1)
        .fps(fps as f64)
        .frame_interval_s(1.0 / fps as f64)
        .loop_playback(true);
    WriterOptions::new(width as u32, height as u32, SampleType::U8)
        .samples_per_pixel(3)
        .compression(Compression::Deflate)
        .compression_level(1)
        .metadata(meta)
}

/// Writes a movie a frame at a time, opening the writer on the first frame,
/// which is the one that says how large they are.
pub(super) struct MovieWriter<W: Write + Seek> {
    open: Option<W>,
    fps: f32,
    writer: Option<TiffWriter<W>>,
    pub(super) frames: usize,
}

impl<W: Write + Seek> MovieWriter<W> {
    pub(super) fn new(out: W, fps: f32) -> Self {
        MovieWriter {
            open: Some(out),
            fps,
            writer: None,
            frames: 0,
        }
    }

    pub(super) fn push(&mut self, rgb: &[u8], (w, h): (usize, usize)) -> Result<(), String> {
        if self.writer.is_none() {
            let out = self
                .open
                .take()
                .ok_or("the movie writer has already failed")?;
            self.writer = Some(
                TiffWriter::new(out, movie_options(w, h, self.fps))
                    .map_err(|e| format!("could not start the movie: {e:#}"))?,
            );
        }
        let w = self.writer.as_mut().expect("opened above");
        w.write_frame_bytes(rgb)
            .map_err(|e| format!("could not write a frame of the movie: {e:#}"))?;
        self.frames += 1;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<W, String> {
        match self.writer {
            Some(w) => w
                .finish()
                .map_err(|e| format!("could not finish the movie: {e:#}")),
            None => Err("the movie has no frames".into()),
        }
    }
}

// ------------------------------------------------------------ where frames go

/// The far end of a recording: a writer thread on the desktop, memory in the
/// browser.
pub(super) struct FrameSink(Sink);

enum Sink {
    #[cfg(not(target_arch = "wasm32"))]
    Thread {
        tx: Option<std::sync::mpsc::Sender<Msg>>,
        done: std::sync::mpsc::Receiver<Result<usize, String>>,
        /// Frames sent and not yet written.
        queued: Arc<AtomicUsize>,
        /// Joined only where a caller has to know the abandoned file is gone
        /// — the tests. The app lets the thread finish on its own.
        #[cfg_attr(not(test), allow(dead_code))]
        writer: Option<std::thread::JoinHandle<()>>,
    },
    // Boxed: the writer is large, and the other two are a few pointers.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Memory(Option<Box<MovieWriter<std::io::Cursor<Vec<u8>>>>>),
    /// The frames, handed straight to a test.
    #[cfg(test)]
    Test(std::sync::mpsc::Sender<Vec<u8>>, usize),
}

#[cfg(not(target_arch = "wasm32"))]
enum Msg {
    Frame(Vec<u8>, (usize, usize)),
    Finish,
}

/// A recording whose frames are all in, being written out.
pub(super) struct Saving {
    pub(super) dest: Destination,
    done: Done,
}

enum Done {
    #[cfg(not(target_arch = "wasm32"))]
    Thread(std::sync::mpsc::Receiver<Result<usize, String>>),
    #[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
    Ready(Option<Result<usize, String>>),
}

impl Saving {
    /// How the save went, once it is over: the frame count, or why not.
    pub(super) fn poll(&mut self) -> Option<Result<usize, String>> {
        match &mut self.done {
            #[cfg(not(target_arch = "wasm32"))]
            Done::Thread(rx) => match rx.try_recv() {
                Ok(r) => Some(r),
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    Some(Err("the movie writer stopped without finishing".into()))
                }
            },
            Done::Ready(r) => r.take(),
        }
    }
}

impl FrameSink {
    /// Frames written to `path` by a thread of their own.
    ///
    /// The file is created when the first frame arrives and removed again if
    /// the recording stops before the last: a half movie is not a movie, and
    /// one left behind under the name the user chose would be mistaken for one.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn to_file(path: std::path::PathBuf, fps: f32) -> FrameSink {
        let (tx, rx) = std::sync::mpsc::channel::<Msg>();
        let (done_tx, done) = std::sync::mpsc::channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let written = Arc::clone(&queued);
        let writer = std::thread::spawn(move || {
            let mut movie: Option<MovieWriter<std::io::BufWriter<std::fs::File>>> = None;
            let fail = |why: String, created: bool| {
                if created {
                    let _ = std::fs::remove_file(&path);
                }
                let _ = done_tx.send(Err(why));
            };
            loop {
                match rx.recv() {
                    Ok(Msg::Frame(rgb, size)) => {
                        if movie.is_none() {
                            match std::fs::File::create(&path) {
                                Ok(f) => {
                                    movie = Some(MovieWriter::new(std::io::BufWriter::new(f), fps))
                                }
                                Err(e) => {
                                    return fail(format!("could not create the file: {e}"), false)
                                }
                            }
                        }
                        let pushed = movie.as_mut().expect("created above").push(&rgb, size);
                        written.fetch_sub(1, Ordering::Relaxed);
                        if let Err(why) = pushed {
                            return fail(why, true);
                        }
                    }
                    Ok(Msg::Finish) => {
                        let Some(m) = movie.take() else {
                            return fail("the movie has no frames".into(), false);
                        };
                        let n = m.frames;
                        let written = m
                            .finish()
                            .and_then(|mut w| w.flush().map_err(|e| e.to_string()));
                        match written {
                            Ok(()) => {
                                let _ = done_tx.send(Ok(n));
                            }
                            Err(why) => fail(why, true),
                        }
                        return;
                    }
                    // The recording was abandoned: nothing more is coming, and
                    // what was written is not a movie.
                    Err(_) => {
                        if let Some(m) = movie.take() {
                            drop(m);
                            let _ = std::fs::remove_file(&path);
                        }
                        return;
                    }
                }
            }
        });
        FrameSink(Sink::Thread {
            tx: Some(tx),
            done,
            queued,
            writer: Some(writer),
        })
    }

    /// Frames kept in memory, for a download at the end.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(super) fn in_memory(fps: f32) -> FrameSink {
        FrameSink(Sink::Memory(Some(Box::new(MovieWriter::new(
            std::io::Cursor::new(Vec::new()),
            fps,
        )))))
    }

    /// A sink whose frames arrive on the returned channel.
    #[cfg(test)]
    pub(super) fn for_test() -> (FrameSink, std::sync::mpsc::Receiver<Vec<u8>>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (FrameSink(Sink::Test(tx, 0)), rx)
    }

    /// Give up on the recording, and wait until whatever it wrote is gone.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(super) fn abandon(self) {
        if let Sink::Thread { tx, writer, .. } = self.0 {
            drop(tx);
            if let Some(w) = writer {
                let _ = w.join();
            }
        }
    }

    /// Whether the far end has as many frames waiting as it should, and the
    /// next had better wait. Only a writer thread ever falls behind: the
    /// other sinks are done with a frame by the time `push` returns.
    pub(super) fn busy(&self) -> bool {
        match &self.0 {
            #[cfg(not(target_arch = "wasm32"))]
            Sink::Thread { queued, .. } => queued.load(Ordering::Relaxed) >= WRITER_QUEUE,
            _ => false,
        }
    }

    pub(super) fn push(&mut self, rgb: Vec<u8>, size: (usize, usize)) -> Result<(), String> {
        match &mut self.0 {
            #[cfg(not(target_arch = "wasm32"))]
            Sink::Thread { tx, queued, .. } => match tx {
                Some(tx) => {
                    queued.fetch_add(1, Ordering::Relaxed);
                    tx.send(Msg::Frame(rgb, size))
                        .map_err(|_| "the movie writer stopped".to_string())
                }
                None => Err("the movie has already been finished".into()),
            },
            Sink::Memory(m) => match m.as_mut() {
                Some(m) => m.push(&rgb, size),
                None => Err("the movie has already been finished".into()),
            },
            #[cfg(test)]
            Sink::Test(tx, n) => {
                *n += 1;
                tx.send(rgb)
                    .map_err(|_| "the test stopped listening".to_string())
            }
        }
    }

    /// Every frame is in: finish the movie, wherever it is going.
    pub(super) fn finish(self, dest: Destination) -> Saving {
        let done = match self.0 {
            #[cfg(not(target_arch = "wasm32"))]
            Sink::Thread { tx, done, .. } => {
                if let Some(tx) = tx {
                    let _ = tx.send(Msg::Finish);
                }
                Done::Thread(done)
            }
            Sink::Memory(m) => Done::Ready(Some(match m {
                Some(m) => {
                    let n = m.frames;
                    m.finish()
                        .and_then(|cursor| deliver(&cursor.into_inner(), &dest).map(|()| n))
                }
                None => Err("the movie has already been finished".into()),
            })),
            #[cfg(test)]
            Sink::Test(_, n) => Done::Ready(Some(Ok(n))),
        };
        Saving { dest, done }
    }
}

/// Hand a movie built in memory to where it is going: the browser's downloads.
fn deliver(bytes: &[u8], dest: &Destination) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        crate::web_save::download(bytes, &dest.name())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match dest {
            Destination::File(p) => std::fs::write(p, bytes).map_err(|e| e.to_string()),
            Destination::Download(_) => {
                let _ = bytes;
                Err("there is nowhere to download to outside a browser".into())
            }
        }
    }
}

// ----------------------------------------------------------- the app's half

impl super::ViewerApp {
    /// Whether the plugin at `index` has an animation running — what its
    /// dialog's Play/Stop button reads.
    pub(super) fn animating_for(&self, index: usize) -> bool {
        self.animation.as_ref().is_some_and(|a| a.plugin == index)
    }

    /// Carry out an animation the plugin at `plugin` (called `name`) returned.
    pub(super) fn start_animation(&mut self, plugin: usize, name: &str, animation: Animation) {
        let Some(stack) = self.core.stack.as_ref() else {
            self.core.status = Some("Open an image first".into());
            return;
        };
        // The plugin checks this too, but the rule is the host's: the 3D view
        // is what it is, and a plugin built against another idea of it should
        // hear why nothing happened.
        if !self.core.can_show_volume() {
            self.core.status = Some(format!(
                "{name}: this stack has no 3D view to animate — it needs at least two planes"
            ));
            return;
        }
        // Playing needs nothing the writer has; a second recording would
        // need the slot the first one's file is still being finished in.
        if self.animation_saving.is_some() && !matches!(animation.output, AnimationOutput::Play) {
            self.core.status = Some(format!("{name}: the last animation is still being written"));
            return;
        }
        let recording = match &animation.output {
            AnimationOutput::Play => None,
            AnimationOutput::SaveTiff { path } => {
                let stem = stack
                    .path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "stack".to_string());
                match destination(path, &stem) {
                    Some(dest) => Some(dest),
                    // The user closed the save dialog: they changed their mind,
                    // which is not a failure and not worth more than a word.
                    None => {
                        self.core.status = Some(format!("{name}: not saved"));
                        return;
                    }
                }
            }
            other => {
                self.core.status = Some(format!(
                    "{name}: asked for its animation to go somewhere this version of \
                     FastTIFF does not know ({other:?})"
                ));
                return;
            }
        };
        // One view, one motion: whatever was running gives way, and puts the
        // view back first so the new one starts from what the user set up.
        if let Some(previous) = self.animation.take() {
            self.end_animation(previous, Ended::Replaced);
        }
        let saving = recording.is_some();
        self.core.view_mode = fast_tiff_viewer::ViewMode::Volume;
        // As the 2D/3D switch does: a movie stops in 3D unless the stack has a
        // time axis of its own. A recording stops it either way — one turn of
        // a volume that is changing underneath it is not one turn.
        if saving || !self.core.is_4d() {
            self.core.playback.playing = false;
            self.core.playback.last_time = None;
        }
        let fps = animation.fps;
        let sink = recording.map(|dest| {
            let sink = match &dest {
                #[cfg(not(target_arch = "wasm32"))]
                Destination::File(p) => FrameSink::to_file(p.clone(), fps),
                _ => FrameSink::in_memory(fps),
            };
            (dest, sink)
        });
        self.animation = Some(Running::new(
            plugin,
            animation,
            &self.core.volume.cam,
            self.core.volume.stack_generation,
            sink,
        ));
        if saving {
            self.core.status = Some(format!("{name}: recording…"));
        }
        self.requery_dialog_of(plugin);
    }

    /// The plugin at `plugin` asked to stop. Only its own animation: another
    /// plugin's is not its to stop.
    pub(super) fn stop_animation_of(&mut self, plugin: usize) {
        if self.animating_for(plugin) {
            let run = self.animation.take().expect("checked above");
            self.end_animation(run, Ended::Asked);
        }
    }

    /// Put the view back as it was, and say why a recording did not finish.
    fn end_animation(&mut self, run: Running, why: Ended) {
        self.core.volume.pose = None;
        if run.recording.is_some() {
            // Dropping the recording drops its sink, and the writer removes
            // whatever part of the file it had written.
            self.core.status = Some(why.recording_message());
        }
        self.requery_dialog_of(run.plugin);
    }

    /// One frame of the animation, if one is running: decide the pose, ask for
    /// a capture, notice the user taking the view back. Called before the 3D
    /// view is drawn, so the pose it sets is the one this frame shows.
    pub(super) fn tick_animation(&mut self, ctx: &egui::Context) {
        // A save whose frames are all in, finishing.
        if let Some(saving) = self.animation_saving.as_mut() {
            match saving.poll() {
                Some(result) => {
                    let dest = saving.dest.name();
                    self.animation_saving = None;
                    match result {
                        Ok(n) => self.report_done(format!("Saved a {n}-frame animation: {dest}")),
                        Err(why) => {
                            self.core.status = Some(format!("Could not save the animation: {why}"))
                        }
                    }
                }
                None => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            }
        }
        // An off-screen render that came back, or could not be made. Collected
        // whether or not a recording still wants it, so one that ended while
        // its last frame was on the way does not hold up the next.
        let (shot, max_side) = {
            let mut shot = self.volume_capture.failed.take().map(Err);
            let mut max_side = MOVIE_MAX_SIDE;
            if let Ok(mut resources) = self.render.lock() {
                if self.volume_capture.in_flight && shot.is_none() {
                    shot = resources.take_volume_capture();
                }
                max_side = max_side.min(resources.max_2d_texture_size());
            }
            if shot.is_some() {
                self.volume_capture.in_flight = false;
            }
            (shot, max_side)
        };

        let pressed = std::mem::take(&mut self.volume_pressed);
        let Some(run) = self.animation.as_mut() else {
            return;
        };
        let ready = self.core.volume.built_frame.is_some()
            && self.core.volume.requested.is_none()
            && self.volume_rect.is_some();
        let ppp = ctx.pixels_per_point();
        let movie = match (self.volume_rect, self.core.stack.as_ref()) {
            (Some(rect), Some(stack)) => {
                let (w, h) = stack.dimensions().unwrap_or((1, 1));
                // At window resolution the movie is the view's own size,
                // which is what `movie_size` makes of no detail to keep.
                let window = run
                    .recording
                    .as_ref()
                    .is_some_and(|r| r.resolution == MovieResolution::Window);
                let detail = if window {
                    0.0
                } else {
                    self.core.volume.detail_height((w, h, stack.volume_depth()))
                };
                movie_size((rect.width() * ppp, rect.height() * ppp), detail, max_side)
            }
            _ => None,
        };
        let frame = Frame {
            now: ctx.input(|i| i.time),
            in_volume: self.core.view_mode == fast_tiff_viewer::ViewMode::Volume,
            generation: self.core.volume.stack_generation,
            camera: &self.core.volume.cam,
            pressed,
            ready,
            movie_size: movie,
            capture_busy: self.volume_capture.in_flight,
            shot,
        };
        match run.step(frame) {
            Step::Show {
                pose,
                capture,
                repaint_after,
            } => {
                self.core.volume.pose = Some(pose);
                // Rendered once this frame's pose has reached the GPU: see
                // `capture_animation_frame`.
                self.volume_capture.request = capture;
                if let Some(rec) = &run.recording {
                    let size = rec
                        .size
                        .map(|(w, h)| format!(", {w} x {h}"))
                        .unwrap_or_default();
                    self.core.status = Some(format!(
                        "Recording animation: {} of {} frames{size}",
                        rec.done,
                        run.total()
                    ));
                }
                if repaint_after > 0.0 {
                    ctx.request_repaint_after(std::time::Duration::from_secs_f64(repaint_after));
                } else {
                    ctx.request_repaint();
                }
            }
            Step::End(why) => {
                let run = self.animation.take().expect("running");
                self.end_animation(run, why);
            }
            Step::Recorded => {
                let run = self.animation.take().expect("running");
                self.core.volume.pose = None;
                let rec = run.recording.expect("a recording");
                let dest = rec.dest.clone();
                self.core.status = Some(format!("Writing animation: {}", dest.name()));
                self.animation_saving = Some(rec.frames.finish(dest));
                self.requery_dialog_of(run.plugin);
                ctx.request_repaint();
            }
        }
    }
}

/// The off-screen render a recording has asked for, and the one it is
/// waiting on.
#[derive(Default)]
pub(super) struct VolumeCapture {
    /// The size to render this frame's pose at, once the GPU has it.
    request: Option<(u32, u32)>,
    /// A render has been asked for and has not come back.
    in_flight: bool,
    /// Why the last render could not even be asked for, to hand to the
    /// recording on the next frame like any other answer.
    failed: Option<String>,
}

impl super::ViewerApp {
    /// Render the pose a recording asked for, off screen. Called with the
    /// renderer just brought up to date with this frame, so the pose rendered
    /// is the one the view shows.
    pub(super) fn capture_animation_frame(&mut self, resources: &mut fast_tiff_viewer::Renderer) {
        let Some((w, h)) = self.volume_capture.request.take() else {
            return;
        };
        match resources.capture_volume(w, h) {
            Ok(()) => self.volume_capture.in_flight = true,
            Err(why) => self.volume_capture.failed = Some(why),
        }
    }
}

/// Where a recorded animation goes: where the plugin said, or where the user
/// says when asked. `None` when the user would not say.
fn destination(path: &str, stem: &str) -> Option<Destination> {
    let suggested = format!("{stem}-animation.tif");
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !path.is_empty() {
            return Some(Destination::File(path.into()));
        }
        rfd::FileDialog::new()
            .add_filter("TIFF", &["tif", "tiff"])
            .set_file_name(&suggested)
            .save_file()
            .map(Destination::File)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // A browser cannot be asked for a place, only given a name.
        Some(Destination::Download(if path.is_empty() {
            suggested
        } else {
            crate::web_save::name_from_path(path)
        }))
    }
}

#[cfg(test)]
#[path = "animation_tests.rs"]
mod tests;

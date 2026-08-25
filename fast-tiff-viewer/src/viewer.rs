//! [`Viewer`] — the frontend-agnostic viewer state: which stack is open, what
//! it's showing, and everything the per-frame GPU sync needs.
//!
//! This is the state a *second* frontend would otherwise have to reinvent. It
//! deliberately holds no window, panel, zoom or pointer state: those are the
//! frontend's, and the eframe app keeps them in its own `ViewerApp` alongside a
//! `Viewer`. The dividing line is "would a browser UI need this to show the
//! right pixels?" — if yes it lives here.

use crate::camera::CameraState;
use crate::channels::refresh_pseudocolor;
use crate::dimensions::{apply_dimension_override, compute_status};
use crate::stack::Stack;
use scivis_render::{VolumeInterp, VolumeRender};
use std::path::PathBuf;

/// Playback rate used when the file's metadata doesn't specify `fps=`.
pub const DEFAULT_FPS: f64 = 30.0;

/// Which view the frontend should show: the 2D movie (scrub/play through
/// frames) or the GPU-ray-marched 3D volume built from the whole stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ViewMode {
    #[default]
    Movie,
    Volume,
}

/// How per-frame decoding is split across CPU cores. The choice maps onto
/// `fast-tiff-lib`'s parallel-decode hint each frame (see [`crate::sync`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DecodeMode {
    /// Serial by default; switch to threaded automatically when real-time
    /// playback starts dropping frames (a core saturating on decode).
    #[default]
    Auto,
    /// Always single-threaded (lowest total CPU; one core).
    Serial,
    /// Always multi-threaded for large frames (spreads load across cores).
    Threaded,
}

impl DecodeMode {
    pub fn label(self) -> &'static str {
        match self {
            DecodeMode::Auto => "Auto",
            DecodeMode::Serial => "Serial",
            DecodeMode::Threaded => "Threaded",
        }
    }

    /// The parallel-decode flag this mode feeds to `fast-tiff-lib`. `latched` is
    /// whether `Auto`'s adaptive trigger has fired (playback fell behind); it
    /// only matters in `Auto`.
    pub fn parallel(self, latched: bool) -> bool {
        match self {
            DecodeMode::Serial => false,
            DecodeMode::Threaded => true,
            DecodeMode::Auto => latched,
        }
    }
}

/// Looped-playback clock. Advances by real elapsed time so the movie runs at the
/// file's fps regardless of render cadence.
#[derive(Clone, Copy, Debug)]
pub struct Playback {
    pub playing: bool,
    /// Playback rate (frames/second). Seeded from the file's `fps=` metadata (or
    /// [`DEFAULT_FPS`]) on every load.
    pub fps: f64,
    /// Frontend clock time (seconds) at the previous frame while playing.
    /// `None` when not playing.
    pub last_time: Option<f64>,
    /// Fractional-frame carry so a non-integer frames-per-tick advance doesn't
    /// lose or gain time over a long playback.
    pub accumulator: f64,
    /// Smoothed "frames demanded per render" while playing: how many frames the
    /// elapsed real time wanted us to advance each tick. ~1 when keeping up; >1
    /// means renders are slower than the target fps (frames dropping — one core
    /// is saturated). Drives the `Auto` decode latch.
    pub demand_ema: f32,
}

impl Default for Playback {
    fn default() -> Self {
        Playback { playing: false, fps: DEFAULT_FPS, last_time: None, accumulator: 0.0, demand_ema: 1.0 }
    }
}

impl Playback {
    /// Restart the clock, so the first tick after resuming doesn't jump by
    /// however long we were paused.
    pub fn restart(&mut self) {
        self.last_time = None;
        self.accumulator = 0.0;
        self.demand_ema = 1.0;
    }
}

/// The 3D volume view's camera, appearance, and build tracking.
#[derive(Clone, Copy, Debug)]
pub struct VolumeView {
    pub cam: CameraState,
    /// Per-axis voxel scale (x, y, z) for the volume box. Seeded from the
    /// stack's pixel calibration + Z spacing on load (else 1:1:1).
    pub scale: [f32; 3],
    pub interp: VolumeInterp,
    pub render: VolumeRender,
    /// Alpha-DVR opacity scale (only used by the `Alpha` render mode).
    pub density: f32,
    /// Isosurface threshold in windowed units 0..1 (only used by `Surface`).
    pub iso: f32,
    /// The volume viewport aspect ratio (width/height), set by the frontend each
    /// frame so the ray-march projection matches the on-screen rect.
    pub aspect: f32,
    /// Which timepoint the currently-uploaded volume holds (`Some(frame_index)`
    /// in the 4D case, else `Some(0)`); `None` means no volume is uploaded yet,
    /// so a frontend should show a loading state until the first build lands. A
    /// mismatch with the current timepoint queues a rebuild — this is what makes
    /// 4D playback advance the volume through time.
    pub built_frame: Option<usize>,
    /// Generation for background builds: bumped when the plan changes shape (new
    /// file, dimension-order swap) so an in-flight build for the old layout is
    /// recognized as stale and ignored.
    pub generation: u64,
    /// The `(generation, time)` currently queued on the background builder, so
    /// the request isn't re-sent on every polling frame.
    pub requested: Option<(u64, usize)>,
}

impl Default for VolumeView {
    fn default() -> Self {
        VolumeView {
            cam: CameraState::default(),
            scale: [1.0, 1.0, 1.0],
            interp: VolumeInterp::Linear,
            render: VolumeRender::Mip,
            density: 100.0,
            iso: 0.1,
            aspect: 1.0,
            built_frame: None,
            generation: 0,
            requested: None,
        }
    }
}

impl VolumeView {
    /// Invalidate the uploaded volume and any in-flight background build, so the
    /// next 3D draw rebuilds. Call after anything that changes the volume's
    /// shape (new file, dimension-order swap).
    pub fn invalidate(&mut self) {
        self.built_frame = None;
        self.generation = self.generation.wrapping_add(1);
        self.requested = None;
    }
}

/// The viewer's frontend-agnostic state.
#[derive(Default)]
pub struct Viewer {
    pub stack: Option<Stack>,
    /// A note about the stack's shape worth surfacing to the user (mislabeled
    /// channel counts, the frozen-Z warning). `None` when there's nothing to say.
    pub status: Option<String>,
    pub view_mode: ViewMode,
    /// User preference (persists across files): tint a multi-channel *grayscale*
    /// stack with the standard channel palette (ch1 red, ch2 green, ch3 blue, …).
    /// Has no effect on stacks that already carry colors.
    pub apply_pseudocolor: bool,
    /// User's decode-parallelism preference (persists across files).
    pub decode_mode: DecodeMode,
    /// A file being opened on a worker thread, if one is. The previous stack
    /// stays on screen and usable until it lands — an open replaces a picture
    /// rather than removing one and then providing another.
    #[cfg(feature = "threads")]
    pub loading: Option<crate::loader::Loading>,
    /// Set when a load lands, cleared by [`poll_open`](Self::poll_open).
    ///
    /// The signal is the same whether the load ran on a worker or here on this
    /// thread, which is the point: a frontend resets the chrome for the new
    /// file on one event, and does not have to know which way the file arrived.
    /// Getting that wrong is what once left the browser build — which has no
    /// threads, so it always loads inline — opening every image at 100%,
    /// because the "a file arrived" signal only ever came from the worker path.
    load_landed: bool,
    /// How frames past the GPU's texture limit are kept on screen (persists
    /// across files). Only consulted for such frames; see
    /// [`crate::roi::LargeImageMode`].
    pub large_image_mode: crate::roi::LargeImageMode,
    /// Latched once playback falls behind in `Auto` mode. Reset per stack.
    pub decode_parallel: bool,
    /// The visible UV sub-rect of the 2D image (`offset`, `scale`), computed by
    /// the frontend from its zoom + pan and uploaded to the shader. The image is
    /// always drawn into the on-screen visible rect with pan/zoom done via these
    /// UVs — never via an oversized viewport, which a GPU backend would clamp to
    /// the framebuffer (squashing the image instead of zooming).
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
    /// Size in physical pixels of the area the image is drawn into.
    ///
    /// Set alongside [`uv_offset`](Self::uv_offset)/[`uv_scale`](Self::uv_scale)
    /// — it is the other half of the same question. Those say *which* part of
    /// the image is on screen; this says how much room there is to draw it, and
    /// together they decide how finely the image needs to be sampled. Without
    /// it a huge image is sampled as finely as the texture budget allows, which
    /// can be a hundred times more detail than the display can show, paid for
    /// on every zoom step.
    ///
    /// Zero means "unknown", and residency falls back to the texture budget.
    pub viewport_px: [f32; 2],
    /// Whether the view is still moving under an animation or gesture, so any
    /// window started now would be planned against a zoom that has already
    /// changed by the time it is decoded.
    ///
    /// Set by the frontend while it is gliding a zoom step out or a pinch is
    /// live. Windows already finished are still taken — a better picture is
    /// always worth having — but no new one is started until the view settles.
    /// Without this an animated zoom decodes at every sampling level it passes
    /// through on the way, which is work the user never sees: the view is only
    /// at each of them for a few frames, and the level it is heading for is the
    /// only one that will still be wanted.
    pub view_moving: bool,
    pub playback: Playback,
    pub volume: VolumeView,
}

impl Viewer {
    pub fn new() -> Self {
        Viewer { uv_scale: [1.0, 1.0], ..Default::default() }
    }

    /// Open `path`, replacing whatever was loaded. On success every derived
    /// per-stack default is re-seeded: playback rate from the file's `fps=`,
    /// voxel scale from its calibration, a fresh 3D camera, and the 2D movie
    /// view. On failure the previous stack is dropped and `status` carries the
    /// error, matching what the user sees.
    ///
    /// On failure the **previously loaded stack stays open** and only `status`
    /// changes — dropping a corrupt file onto a window must not close the good
    /// image the user was looking at.
    ///
    /// Frontend-side chrome (window title, zoom, open pop-ups) is *not* touched
    /// — that's the frontend's to reset.
    #[cfg(feature = "mmap")]
    pub fn open(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let opened = Stack::open(path, self.apply_pseudocolor);
        self.adopt(opened)
    }

    /// Start opening `source`, off this thread where that is possible.
    ///
    /// Returns immediately. [`poll_open`](Self::poll_open) reports when it
    /// lands; until then [`load_stage`](Self::load_stage) says how far it has
    /// got, and the previously open stack is still there to look at.
    ///
    /// Without threads to spawn — a browser, or a failed spawn — this falls
    /// back to loading here and now, which blocks exactly as it always did.
    /// Slower to respond is a great deal better than not opening the file.
    pub fn begin_open(&mut self, source: crate::loader::LoadSource) {
        #[cfg(feature = "threads")]
        let source = match crate::loader::Loading::spawn(source, self.apply_pseudocolor) {
            Ok(loading) => {
                self.loading = Some(loading);
                return;
            }
            // No thread to be had. Load it here — slower to respond is far
            // better than not opening the file.
            Err(Some(source)) => source,
            Err(None) => {
                self.status = Some("Could not start the loader".to_owned());
                self.load_landed = true;
                return;
            }
        };
        let opened = source.load(self.apply_pseudocolor, &mut |_| {});
        let _ = self.adopt(opened);
        self.load_landed = true;
    }

    /// Take delivery of a finished load, if one has finished this frame.
    ///
    /// Returns `true` when a stack was installed or an error recorded — the
    /// signal for a frontend to reset whatever it had arranged around the
    /// previous file.
    pub fn poll_open(&mut self) -> bool {
        #[cfg(feature = "threads")]
        if let Some(loading) = self.loading.as_mut() {
            if let Some(result) = loading.take() {
                self.loading = None;
                let _ = self.adopt(result);
                self.load_landed = true;
            }
        }
        std::mem::take(&mut self.load_landed)
    }

    /// How far the in-flight open has got, or `None` when nothing is loading.
    pub fn load_stage(&self) -> Option<crate::loader::LoadStage> {
        #[cfg(feature = "threads")]
        {
            self.loading.as_ref().map(|l| l.stage())
        }
        #[cfg(not(feature = "threads"))]
        None
    }

    /// The name of the file being opened, while one is.
    pub fn loading_name(&self) -> Option<String> {
        #[cfg(feature = "threads")]
        {
            self.loading.as_ref().map(|l| {
                l.name().file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
            })
        }
        #[cfg(not(feature = "threads"))]
        None
    }

    /// Load a stack from bytes already in memory — the browser's entry point,
    /// where `name` is just the picked file's name for display. Otherwise
    /// identical to [`Viewer::open`], including every per-stack default it
    /// re-seeds.
    pub fn load_bytes(&mut self, bytes: Vec<u8>, name: PathBuf) -> anyhow::Result<()> {
        let opened = Stack::from_bytes(bytes, name, self.apply_pseudocolor);
        self.adopt(opened)
    }

    /// Install a freshly opened stack (or record why it failed), re-seeding
    /// every per-stack default. Shared so the two entry points can't drift.
    fn adopt(&mut self, opened: anyhow::Result<Stack>) -> anyhow::Result<()> {
        match opened {
            Ok(stack) => {
                self.playback = Playback { fps: stack.tiff.meta.fps.unwrap_or(DEFAULT_FPS), ..Default::default() };
                // 3D defaults for the new stack: per-axis voxel scale from the
                // pixel calibration (X/YResolution) + Z spacing (else 1:1:1), a
                // fresh orbit, and an invalidated volume so entering 3D uploads
                // this stack's data.
                self.volume.scale = stack.tiff.meta.voxel_scale();
                self.volume.cam.reset();
                self.volume.invalidate();
                // Always start a newly-opened file in the 2D movie view.
                self.view_mode = ViewMode::Movie;
                self.status = compute_status(stack.display.dims, stack.display.triple_axis_warning);
                // New stack: re-evaluate decode parallelism from scratch (its
                // per-frame decode cost is different).
                self.decode_parallel = false;
                self.stack = Some(stack);
                Ok(())
            }
            Err(e) => {
                self.status = Some(format!("Failed to open file: {e:#}"));
                Err(e)
            }
        }
    }

    /// Reassign which axis counts mean channels / Z / time, then refresh
    /// everything derived from that: LUTs, the status note, and the volume.
    pub fn set_dimension_order(&mut self, channels: usize, slices: usize, frames: usize) {
        if let Some(stack) = &mut self.stack {
            apply_dimension_override(stack, channels, slices, frames);
            // The swap rebuilds channel_display from `mode`, so re-apply the
            // pseudocolor preference on top of the fresh LUTs.
            refresh_pseudocolor(stack, self.apply_pseudocolor);
            self.status = compute_status(stack.display.dims, stack.display.triple_axis_warning);
        }
        // The frame axis (volume depth) just changed.
        self.volume.invalidate();
    }

    /// Apply the pseudocolor preference to the current stack (and remember it
    /// for stacks opened later).
    pub fn set_pseudocolor(&mut self, on: bool) {
        self.apply_pseudocolor = on;
        if let Some(stack) = &mut self.stack {
            refresh_pseudocolor(stack, on);
        }
    }

    /// Whether the stack has enough depth to build a volume from.
    /// Pick the single-channel LUT by selector index — index 0 is the file's
    /// own "Built-in" map when it has one, then grayscale, the named channel
    /// colours, and the perceptual colormaps (see
    /// [`crate::channels::gray_lut_sel_lut`]).
    ///
    /// Restoring "Built-in" works because opening a file never overwrites
    /// `tiff.meta`: the file's colours are still there to go back to. No-op
    /// when no stack is open or the selector doesn't apply to it.
    pub fn set_gray_lut(&mut self, sel: usize) {
        let Some(stack) = &mut self.stack else { return };
        if !crate::channels::gray_lut_applicable(stack) {
            return;
        }
        stack.display.gray_lut_sel = sel;
        let lut = crate::channels::gray_lut_sel_lut(&stack.display, sel);
        if stack.display.set_lut(0, lut) {
            stack.luts_uploaded = false; // force a LUT re-upload on the next sync
        }
    }

    pub fn can_show_volume(&self) -> bool {
        self.stack.as_ref().is_some_and(|s| s.display.dims.frames >= 2)
    }

    /// Whether the stack has a time axis *separate* from the volume's depth —
    /// i.e. playback still means something in the 3D view. False for an ordinary
    /// 3D stack, where the frame axis *is* the depth.
    pub fn is_4d(&self) -> bool {
        self.stack.as_ref().is_some_and(|s| s.display.dims.slices > 1)
    }

    /// Advance looped playback to match `now` (the frontend's monotonic clock, in
    /// seconds). Returns the number of frames stepped. Also maintains the
    /// keeping-up estimate that latches parallel decoding in `Auto` mode.
    ///
    /// Call once per rendered frame while playing; the frontend is responsible
    /// for scheduling the next repaint (at `1.0 / fps`).
    pub fn tick_playback(&mut self, now: f64) -> usize {
        let Some(stack) = &mut self.stack else {
            self.playback.last_time = None;
            self.playback.accumulator = 0.0;
            return 0;
        };
        let n = stack.frame_count();
        if !self.playback.playing || n <= 1 {
            self.playback.playing = self.playback.playing && n > 1;
            self.playback.last_time = None;
            self.playback.accumulator = 0.0;
            return 0;
        }
        let fps = self.playback.fps.max(0.1);
        let mut stepped = 0;
        if let Some(last) = self.playback.last_time {
            // `demand` = frames the elapsed real time wanted this render to
            // cover. ~1 when keeping up; >1 means renders are slower than the
            // fps target (frames dropping → a core is saturated). Once the
            // smoothed demand crosses the threshold, latch parallel decoding.
            let demand = (now - last) * fps;
            self.playback.demand_ema = self.playback.demand_ema * 0.9 + demand as f32 * 0.1;
            if self.decode_mode == DecodeMode::Auto && self.playback.demand_ema > 1.3 {
                self.decode_parallel = true;
            }
            self.playback.accumulator += demand;
            if self.playback.accumulator >= 1.0 {
                stepped = self.playback.accumulator.floor() as usize;
                self.playback.accumulator -= stepped as f64;
                stack.frame_index = (stack.frame_index + stepped) % n;
            }
        }
        self.playback.last_time = Some(now);
        stepped
    }
}

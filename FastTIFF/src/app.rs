//! The viewer's egui::App. Holds the loaded stack (if any), per-channel
//! display settings, and the current scrub position. Drives GPU texture
//! uploads directly from UI code (not from inside the paint callback) so a
//! frame change is just: mmap read -> texture upload -> (next frame) draw call.
//! The GPU backend (glow or wgpu) is reached only through `crate::render`'s
//! backend-agnostic surface, so nothing here mentions either by name.

use crate::prefetch::{decode_jobs, ChannelJob, Decoded, PrefetchResult};
use crate::render::{self, ChannelKind, ChannelUniform, Render, MAX_CHANNELS};
use egui::{Color32, RichText};
use std::path::PathBuf;
use fast_tiff_lib::TiffStack;

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
fn monitor_work_area(ctx: &egui::Context) -> Option<egui::Vec2> {
    ctx.input(|i| i.viewport().monitor_size)
        .map(|m| egui::vec2((m.x * 0.95).max(1.0), (m.y * 0.90).max(1.0)))
}

/// The opening zoom for a freshly loaded image: the largest zoom level ≤ 100%
/// at which the image plus chrome still fits the monitor's work area (so a
/// normal image opens at 100%, a big one at 50% or 25%). Returns `None` when the
/// monitor size isn't reported yet (caller should keep waiting rather than
/// guess) so a huge image never briefly opens oversized.
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

/// How per-frame decoding is split across CPU cores. The choice maps onto
/// `fast-tiff-lib`'s parallel-decode hint each frame (see `sync_gpu`).
#[derive(Clone, Copy, PartialEq)]
enum DecodeMode {
    /// Serial by default; switch to threaded automatically when real-time
    /// playback starts dropping frames (a core saturating on decode).
    Auto,
    /// Always single-threaded (lowest total CPU; one core).
    Serial,
    /// Always multi-threaded for large frames (spreads load across cores).
    Threaded,
}

impl DecodeMode {
    fn label(self) -> &'static str {
        match self {
            DecodeMode::Auto => "Auto",
            DecodeMode::Serial => "Serial",
            DecodeMode::Threaded => "Threaded",
        }
    }

    /// The parallel-decode flag this mode feeds to `fast-tiff-lib`. `latched` is
    /// whether `Auto`'s adaptive trigger has fired (playback fell behind); it
    /// only matters in `Auto`.
    fn parallel(self, latched: bool) -> bool {
        match self {
            DecodeMode::Serial => false,
            DecodeMode::Threaded => true,
            DecodeMode::Auto => latched,
        }
    }
}

/// Which view the central panel shows: the 2D movie (scrub/play through frames)
/// or the GPU-ray-marched 3D volume built from the whole stack.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Movie,
    Volume,
}

/// How mouse/keyboard drive the 3D camera, modeled on familiar 3D apps. The
/// first three orbit a pivot (differing only in which button/modifier does what);
/// `WasdFly` is a first-person free-fly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NavMode {
    Cad,
    Blender,
    Maya,
    WasdFly,
}

impl NavMode {
    fn label(self) -> &'static str {
        match self {
            NavMode::Cad => "CAD",
            NavMode::Blender => "Blender",
            NavMode::Maya => "Maya",
            NavMode::WasdFly => "Minecraft Spectator",
        }
    }

    /// One-line control hint shown under the selector.
    fn help(self) -> &'static str {
        match self {
            NavMode::Cad => "Left-drag: orbit · Middle-drag: pan · Scroll: zoom",
            NavMode::Blender => "Middle-drag: orbit · Shift+Middle: pan · Scroll: zoom",
            NavMode::Maya => "Alt+Left: orbit · Alt+Middle: pan · Alt+Right / Scroll: zoom",
            NavMode::WasdFly => "Left-drag: look · WASD: move · Space/Shift: up/down · Scroll: fly",
        }
    }

    /// Whether this mode is a first-person free-fly (vs. orbiting a pivot).
    fn is_fly(self) -> bool {
        matches!(self, NavMode::WasdFly)
    }
}

#[derive(Clone, Copy)]
struct ChannelSettings {
    min: f32,
    max: f32,
    enabled: bool,
    /// The full track range `(lo, hi)` the contrast range-slider spans, in raw
    /// sample units. Derived from the channel's data range (and widened to
    /// include any metadata window) at load time so the two handles always sit
    /// somewhere on the track.
    bounds: (f32, f32),
    /// Which GPU texture format this channel uploads to (picked from the source
    /// pixel format): `Int8` (R8Uint, raw 8-bit — zero-copy), `Float` (R32F, raw
    /// float, window/level on the GPU), or `Int16` (R16Uint — the default, incl.
    /// RGB planes and any data the CPU widens/rescales into 0..65535). Drives
    /// both texture allocation and the decode path in `sync_gpu`. For float
    /// channels `min`/`max` are the contrast window in the data's own float
    /// units (matching how ImageJ shows float-image contrast).
    kind: render::ChannelKind,
}

struct LoadedStack {
    tiff: TiffStack,
    path: PathBuf,
    channel_settings: Vec<ChannelSettings>,
    frame_index: usize,
    last_uploaded: Option<usize>,
    /// The per-channel `enabled` flags as of the last GPU upload. A disabled
    /// channel is skipped during upload (the shader multiplies it out anyway),
    /// so re-enabling one must re-upload it even when the frame index is
    /// unchanged — a difference here forces that.
    last_enabled: Vec<bool>,
    luts_uploaded: bool,
    /// Set once at load time when the file genuinely has channels, Z, and
    /// time all present simultaneously — Z then stays permanently frozen
    /// at its first slice (see `resolve_dimensions`). Kept around so the
    /// warning note is still shown correctly after a manual channels/frames
    /// swap via the dimension-order dropdown, which never touches Z.
    triple_axis_warning: bool,
    /// True when each IFD is a chunky RGB image: the "channels" are then the
    /// red/green/blue sample planes of a *single* IFD per frame (deinterleaved
    /// on upload), not separate IFDs. Flips how `ifd_index`/`sync_gpu` map a
    /// display channel to file data.
    rgb: bool,
    /// Background read-ahead worker (own mmap): decode-ahead for compressed
    /// stacks, page-touch for uncompressed ones (absorbs the next frame's mmap
    /// soft faults off the UI thread while its inline decode stays zero-copy).
    /// `None` only if the worker failed to start. See `crate::prefetch`.
    prefetch: Option<crate::prefetch::Prefetcher>,
    /// Bumped whenever the decode plan changes (dimension-order swap, enabled-set
    /// change) so an in-flight prefetch decoded under the old plan is recognized
    /// as stale and ignored rather than uploaded.
    prefetch_gen: u64,
    /// Background 3D-volume builder (own mmap), spawned lazily on the first 3D
    /// use. `None` after a failed spawn (`volume_builder_tried` set) — volume
    /// builds then fall back to running synchronously on the UI thread.
    volume_builder: Option<crate::volume::VolumeBuilder>,
    volume_builder_tried: bool,
    /// Which LUT the single-channel grayscale color selector currently shows
    /// (index into `GRAY_LUT_OPTIONS`; 0 = plain grayscale). Only meaningful
    /// while `gray_lut_applicable` holds. Lives on the stack — not the app — so
    /// it resets to grayscale for each newly opened file (the "default" state).
    gray_lut_sel: usize,
}

pub struct ViewerApp {
    /// GPU textures/shader for compositing the image, shared with the paint
    /// callback. Created once at startup (see `crate::render::init`).
    render: Render,
    stack: Option<LoadedStack>,
    status: Option<String>,
    /// Channel buttons + contrast sliders are tucked under a small
    /// triangle toggle to keep the bar minimal by default.
    channels_panel_expanded: bool,
    /// User preference (persists across files): tint a multi-channel *grayscale*
    /// stack with the standard channel palette (ch1 red, ch2 green, ch3 blue,
    /// …). Has no effect on stacks that already carry colors (composite mode,
    /// or RGB) — those keep their own LUTs.
    apply_pseudocolor: bool,
    /// Zoom factor used the last time we sized the window: 1.0 = one window
    /// pixel per image pixel. The window is only ever resized in response to an
    /// explicit event (opening a file, or a zoom in/out) — never every frame —
    /// so the user can freely resize or maximize the window. Between those
    /// events the image just scales to fit the window with its aspect ratio
    /// locked (letterboxed), handled entirely in the central panel.
    zoom: f32,
    /// Set when a file is opened: the next frame computes an initial fit-to-
    /// screen zoom (largest level ≤ 100% that fits the monitor) and sizes the
    /// window once. Deferred to `ui()` because the chrome height and monitor
    /// size aren't known yet at open time.
    pending_initial_fit: bool,
    /// Set when something (initial fit or a zoom step) wants the window resized
    /// to match `zoom` on this frame. Applied once, then cleared.
    resize_to_zoom: bool,
    /// The window title last sent via `ViewportCommand::Title`.
    last_title: Option<String>,
    /// Whether the stack is auto-advancing (looped playback).
    playing: bool,
    /// `egui` input time (seconds) at the previous frame while playing, used
    /// to advance by real elapsed time regardless of render rate. `None` when
    /// not playing.
    last_play_time: Option<f64>,
    /// Fractional-frame carry so a non-integer frames-per-render-tick advance
    /// doesn't lose or gain time over a long playback.
    play_accumulator: f64,
    /// Smoothed "frames demanded per render" while playing: how many frames the
    /// elapsed real time wanted us to advance each render tick. ~1 when we're
    /// keeping up; >1 means renders are slower than the target fps (we're
    /// dropping frames — one core is saturated). Drives `decode_parallel`.
    play_demand_ema: f32,
    /// Latched once playback falls behind (in `Auto` mode): ask `fast-tiff-lib` to
    /// split decoding across cores (worth the extra total CPU only when a single
    /// core can't keep up). Reset per stack — see `set_parallel_decode`.
    decode_parallel: bool,
    /// User's decode-parallelism preference (persists across files). `Auto` uses
    /// `decode_parallel`; `Serial`/`Threaded` force it off/on.
    decode_mode: DecodeMode,
    /// When the channels panel was just toggled: the bottom-bar height *before*
    /// the toggle took visual effect, so the next frame can grow/shrink the
    /// window by exactly the panel's height change. `false` when idle.
    panel_grow_armed: bool,
    panel_old_h: f32,
    /// Playback rate (frames/second) the user can edit. Seeded from the file's
    /// `fps=` metadata (or `DEFAULT_FPS`) on every load.
    playback_fps: f64,
    /// Whether the file-metadata pop-up window is open (toggled by the 🗎
    /// button in the expanded bottom panel).
    show_metadata: bool,
    /// Scroll offset of the image inside the central panel, in screen points:
    /// how far the image's top-left is pushed up/left past the panel's. Only
    /// meaningful when the image is larger than the panel (zoomed past what the
    /// monitor-capped window can show); 0 otherwise. Drag to pan.
    pan: egui::Vec2,
    /// The central panel's rect and the image's painted top-left from the last
    /// frame, cached so the early zoom step (which runs before the panel is
    /// redrawn) can keep the point under the cursor fixed while zooming.
    panel_rect: egui::Rect,
    image_origin: egui::Pos2,
    /// Set by the early zoom step when zoom changed this frame: `(old_zoom,
    /// cursor)`, consumed later by the window-sizing code to decide whether to
    /// reposition the window. Separate from the zoom value itself, which is
    /// applied early so the image redraws this same frame.
    zoom_reposition: Option<(f32, egui::Pos2)>,
    /// The visible UV sub-rect of the image (`uv_offset`, `uv_scale`), computed
    /// from zoom + pan in the central panel and uploaded to the shader. The
    /// image is always rendered into the on-screen visible rect with the
    /// pan/zoom done via these UVs — never via an oversized viewport, which
    /// egui-wgpu would clamp to the framebuffer (squashing the image instead of
    /// zooming).
    uv_offset: egui::Vec2,
    uv_scale: egui::Vec2,
    /// Accumulated mouse-wheel scroll not yet turned into a frame step, for the
    /// precise (no-Shift) scrubbing mode. One wheel notch is a Line event of ±1
    /// → exactly one frame; touchpad pixel scrolls accumulate here until they
    /// cross a whole frame, so fine scrolling isn't lost or jumpy.
    scroll_accum: f32,

    // --- 3D volume view -----------------------------------------------------
    /// Movie (2D) vs. Volume (3D). Reset to `Movie` when a single-frame stack
    /// is opened (nothing to build a volume from).
    view_mode: ViewMode,
    /// Orbit camera angles (radians) and the eye→pivot distance for the volume
    /// view. Yaw spins around the vertical axis, pitch tilts up/down; `vol_dist`
    /// is the orbit radius (0 = rotate in place around the eye).
    vol_yaw: f32,
    vol_pitch: f32,
    vol_dist: f32,
    /// Orbit pivot (world space): pan translates this so orbit modes can slide
    /// the volume off-center. Origin by default. Also re-set to the focal-axis
    /// box-entry point when an orbit drag begins.
    vol_target: [f32; 3],
    /// The volume box half-extents from the last `sync_gpu` (mirrors the shader's
    /// `box_he`), cached so the orbit re-pivot can ray-cast against the box
    /// without the stack dimensions on hand.
    vol_box_he: [f32; 3],
    /// Free-fly eye position (world space) for the `Minecraft` nav mode.
    vol_fly_pos: [f32; 3],
    /// How mouse/keyboard drive the 3D camera (CAD/Blender/Maya/Minecraft).
    nav_mode: NavMode,
    /// Per-axis voxel scale (x, y, z) for the volume box. Seeded from the
    /// stack's Z spacing metadata on load (else 1:1:1); editable in the render
    /// settings window.
    vol_scale: [f32; 3],
    /// Volume texture filtering: nearest (no interpolation) or trilinear.
    vol_interp: render::VolumeInterp,
    /// Ray-march compositing: MIP (default) or ImageJ-3D-Viewer-style alpha DVR.
    vol_render: render::VolumeRender,
    /// Alpha-DVR opacity scale (only used by the `Alpha` render mode).
    vol_density: f32,
    /// The volume viewport aspect ratio (width/height), captured each frame so
    /// the ray-march projection matches the on-screen rect.
    vol_aspect: f32,
    /// Which timepoint the currently-uploaded volume holds (`Some(frame_index)`
    /// in the 4D case, else `Some(0)`); `None` means no volume is uploaded yet,
    /// so the loading screen shows until the first background build lands. A
    /// mismatch with the current timepoint queues a rebuild — this is what makes
    /// 4D playback advance the volume through time.
    volume_built_frame: Option<usize>,
    /// Generation for background volume builds: bumped when the plan changes
    /// shape (new file, dimension-order swap) so an in-flight build for the old
    /// layout is recognized as stale and ignored.
    volume_gen: u64,
    /// The `(generation, time)` currently queued on the background builder, so
    /// the request isn't re-sent on every polling frame.
    volume_requested: Option<(u64, usize)>,
    /// Whether the render-settings pop-up (voxel scale + interpolation) is open,
    /// toggled by the ⚙ button in the expanded bottom panel.
    show_render_settings: bool,
    /// User-adjustable 3D navigation speeds (multipliers on the built-in base
    /// rates), edited in the render-settings window. `move_speed` scales WASD /
    /// Space / Shift translation; `scroll_speed` scales the mouse-wheel fly.
    /// Persist across files (they're input preferences, not per-stack state).
    move_speed: f32,
    scroll_speed: f32,
}

/// Playback rate used when the file's metadata doesn't specify `fps=`.
const DEFAULT_FPS: f64 = 30.0;

impl ViewerApp {
    pub fn new(initial_path: Option<PathBuf>, render: Render) -> Self {
        let mut app = Self {
            render,
            stack: None,
            status: None,
            channels_panel_expanded: false,
            apply_pseudocolor: false,
            zoom: 1.0,
            pending_initial_fit: false,
            resize_to_zoom: false,
            last_title: None,
            playing: false,
            last_play_time: None,
            play_accumulator: 0.0,
            play_demand_ema: 1.0,
            decode_parallel: false,
            decode_mode: DecodeMode::Auto,
            panel_grow_armed: false,
            panel_old_h: 0.0,
            playback_fps: DEFAULT_FPS,
            show_metadata: false,
            pan: egui::Vec2::ZERO,
            panel_rect: egui::Rect::ZERO,
            image_origin: egui::Pos2::ZERO,
            zoom_reposition: None,
            uv_offset: egui::Vec2::ZERO,
            uv_scale: egui::Vec2::splat(1.0),
            scroll_accum: 0.0,
            view_mode: ViewMode::Movie,
            vol_yaw: 0.7,
            vol_pitch: 0.5,
            vol_dist: 3.0,
            vol_target: [0.0, 0.0, 0.0],
            vol_box_he: [0.5, 0.5, 0.5],
            vol_fly_pos: [0.0, 0.0, 3.0],
            nav_mode: NavMode::Cad,
            vol_scale: [1.0, 1.0, 1.0],
            vol_interp: render::VolumeInterp::Linear,
            vol_render: render::VolumeRender::Mip,
            vol_density: 100.0,
            vol_aspect: 1.0,
            volume_built_frame: None,
            volume_gen: 0,
            volume_requested: None,
            show_render_settings: false,
            move_speed: 1.0,
            scroll_speed: 1.0,
        };
        if let Some(path) = initial_path {
            app.open_file(path);
        }
        app
    }

    fn open_file(&mut self, path: PathBuf) {
        match TiffStack::open(&path) {
            Ok(tiff) => {
                // Spin up the read-ahead worker: decode-ahead for compressed
                // stacks, page-touch for uncompressed (see `LoadedStack::prefetch`).
                let compressed = tiff
                    .frames
                    .first()
                    .is_some_and(|f| f.compression != fast_tiff_lib::Compression::None);
                let prefetch = crate::prefetch::Prefetcher::new(path.clone(), !compressed);
                let mut loaded = LoadedStack {
                    tiff,
                    path,
                    channel_settings: Vec::new(),
                    frame_index: 0,
                    last_uploaded: None,
                    last_enabled: Vec::new(),
                    luts_uploaded: false,
                    triple_axis_warning: false,
                    rgb: false,
                    prefetch,
                    prefetch_gen: 0,
                    volume_builder: None,
                    volume_builder_tried: false,
                    gray_lut_sel: 0,
                };
                let (c, z, f) = (
                    loaded.tiff.meta.channels,
                    loaded.tiff.meta.slices,
                    loaded.tiff.meta.frames,
                );
                let resolved = fast_tiff_lib::resolve_dimensions(c, z, f);
                apply_resolved_dimensions(&mut loaded, resolved);
                // Chunky RGB overrides the channel layout: the sample planes of
                // each IFD become red/green/blue display channels.
                if loaded.tiff.frames.first().is_some_and(|f| f.is_rgb()) {
                    setup_rgb(&mut loaded);
                }
                // Carry the pseudocolor preference onto the new stack.
                refresh_pseudocolor(&mut loaded, self.apply_pseudocolor);

                // Seed the editable playback rate from the file (or default).
                self.playback_fps = loaded.tiff.meta.fps.unwrap_or(DEFAULT_FPS);

                // 3D volume defaults for the new stack: per-axis voxel scale from
                // the pixel calibration (X/YResolution) + Z spacing (else 1:1:1),
                // a fresh orbit, and a rebuild flag so entering 3D uploads this
                // stack's volume. A single-frame stack has no depth to ray-march
                // — force movie view.
                self.vol_scale = loaded.tiff.meta.voxel_scale();
                self.reset_volume_camera();
                self.volume_built_frame = None;
                self.volume_gen = self.volume_gen.wrapping_add(1);
                self.volume_requested = None;
                // Always start a newly-opened file in the 2D movie view.
                self.view_mode = ViewMode::Movie;

                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
                self.stack = Some(loaded);
                // Start at 1:1; the next frame computes a fit-to-screen zoom
                // (the largest level ≤ 100% that fits) and sizes the window once.
                self.zoom = 1.0;
                self.pan = egui::Vec2::ZERO;
                self.pending_initial_fit = true;
                self.resize_to_zoom = false;
                self.playing = false;
                self.last_play_time = None;
                self.play_accumulator = 0.0;
                // New stack: re-evaluate decode parallelism from scratch (its
                // per-frame decode cost is different).
                self.play_demand_ema = 1.0;
                self.decode_parallel = false;
            }
            Err(e) => {
                self.status = Some(format!("Failed to open file: {e:#}"));
            }
        }
    }

    /// Reset the 3D camera to a default three-quarter view looking at the origin.
    /// Used on load and by the Reset-position button.
    fn reset_volume_camera(&mut self) {
        self.vol_yaw = 0.7;
        self.vol_pitch = 0.5;
        self.vol_dist = 3.0;
        self.vol_target = [0.0, 0.0, 0.0];
        // Free-fly eye starts where the orbit eye would be, looking at the origin.
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        self.vol_fly_pos = [-forward[0] * self.vol_dist, -forward[1] * self.vol_dist, -forward[2] * self.vol_dist];
    }

    /// Rotate the orbit/look by a pointer delta (screen pixels).
    fn vol_orbit(&mut self, delta: egui::Vec2) {
        self.vol_yaw -= delta.x * 0.01;
        self.vol_pitch = (self.vol_pitch + delta.y * 0.01).clamp(-1.54, 1.54);
    }

    /// Pan the orbit pivot in the camera's screen plane by a pointer delta
    /// (grab-and-drag: the scene follows the cursor).
    fn vol_pan(&mut self, delta: egui::Vec2, right: [f32; 3], up: [f32; 3], pan_speed: f32) {
        let (dx, dy) = (delta.x * pan_speed, delta.y * pan_speed);
        let t = self.vol_target;
        self.vol_target = [
            t[0] + up[0] * dy - right[0] * dx,
            t[1] + up[1] * dy - right[1] * dx,
            t[2] + up[2] * dy - right[2] * dx,
        ];
    }

    /// The eye's world position for the given look direction: `fly_pos` in the
    /// free-fly mode, else `target - forward*dist` (the orbit eye).
    fn current_eye(&self, forward: [f32; 3]) -> [f32; 3] {
        if self.nav_mode.is_fly() {
            self.vol_fly_pos
        } else {
            let dist = vol_dist_clamped(self.vol_dist);
            [
                self.vol_target[0] - forward[0] * dist,
                self.vol_target[1] - forward[1] * dist,
                self.vol_target[2] - forward[2] * dist,
            ]
        }
    }

    /// Set the orbit pivot to where the camera's focal axis first enters the
    /// volume box, keeping the eye where it is (the orbit radius becomes that
    /// entry distance). Called when an orbit drag begins, so the rotation centers
    /// on what's under the view. When the eye is inside the box the entry distance
    /// is 0, so the pivot lands on the eye itself — the camera rotates in place.
    /// If the focal ray misses the box, the pivot falls to the focal-axis point
    /// nearest the box center (still on the axis, so the eye never jumps).
    fn repivot_to_focal(&mut self) {
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let eye = self.current_eye(forward);
        let t = focal_box_entry(eye, forward, self.vol_box_he).unwrap_or_else(|| {
            (-(eye[0] * forward[0] + eye[1] * forward[1] + eye[2] * forward[2])).max(0.0)
        });
        self.vol_target = [eye[0] + forward[0] * t, eye[1] + forward[1] * t, eye[2] + forward[2] * t];
        // Radius = eye->pivot distance, so the eye doesn't move (t = 0 inside).
        self.vol_dist = vol_dist_clamped(t);
    }

    /// Rotate the view while keeping the eye fixed (first-person "mouse look"):
    /// the pivot swings to stay `dist` ahead of the eye along the new direction.
    fn vol_look_in_place(&mut self, delta: egui::Vec2) {
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let eye = self.current_eye(forward);
        self.vol_orbit(delta);
        let (fwd, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        self.vol_target = [eye[0] + fwd[0] * dist, eye[1] + fwd[1] * dist, eye[2] + fwd[2] * dist];
    }

    /// Orbit the free-fly eye around the current pivot (used by the free-fly
    /// mode's right-drag): rotate, then place `fly_pos` on the orbit sphere.
    fn vol_orbit_fly(&mut self, delta: egui::Vec2) {
        self.vol_orbit(delta);
        let (fwd, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        self.vol_fly_pos = [
            self.vol_target[0] - fwd[0] * dist,
            self.vol_target[1] - fwd[1] * dist,
            self.vol_target[2] - fwd[2] * dist,
        ];
    }

    /// Keep the view continuous when switching between a free-fly and an orbit
    /// mode: the two store the eye differently, so re-derive one from the other
    /// (same eye position + look direction, so nothing on screen jumps).
    fn sync_camera_for_nav(&mut self, was_fly: bool) {
        let now_fly = self.nav_mode.is_fly();
        if was_fly == now_fly {
            return;
        }
        let (forward, _, _) = volume_basis(self.vol_yaw, self.vol_pitch);
        let dist = vol_dist_clamped(self.vol_dist);
        if now_fly {
            // orbit -> fly: put the free eye where the orbit eye is.
            self.vol_fly_pos = [
                self.vol_target[0] - forward[0] * dist,
                self.vol_target[1] - forward[1] * dist,
                self.vol_target[2] - forward[2] * dist,
            ];
        } else {
            // fly -> orbit: pivot sits `dist` ahead of the eye along the look dir.
            self.vol_target = [
                self.vol_fly_pos[0] + forward[0] * dist,
                self.vol_fly_pos[1] + forward[1] * dist,
                self.vol_fly_pos[2] + forward[2] * dist,
            ];
        }
    }

    /// Apply this frame's mouse/keyboard to the 3D camera per the active nav mode.
    /// Returns whether the camera is actively moving (so the caller keeps
    /// repainting while a drag or a held key continues).
    fn drive_volume_camera(&mut self, ui: &egui::Ui, response: &egui::Response, panel_rect: egui::Rect) -> bool {
        const KEY_ROT: f32 = 0.04;
        // Fly speed: cross the volume's longest axis (length 1.0 in box space) in
        // ~5 s. Time-based (× dt) so it's frame-rate independent.
        const FLY_UNITS_PER_SEC: f32 = 0.2;
        const FLY_WHEEL: f32 = 0.15;

        let mut animating = false;
        let (forward, right, up) = volume_basis(self.vol_yaw, self.vol_pitch);
        let panel_h = panel_rect.height().max(1.0);
        let dist = vol_dist_clamped(self.vol_dist);
        let tan = (45.0f32.to_radians() * 0.5).tan();
        // Pan speed floors the radius so panning still works when rotating in place.
        let pan_speed = 2.0 * dist.max(0.1) * tan / panel_h;
        let hovered = ui.rect_contains_pointer(panel_rect);
        // Clamp the frame time so a long stall (or the first frame) can't teleport.
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);

        // Keyboard + wheel (wheel only while the pointer is over the canvas).
        let (alt, shift, wheel, wasd, space, arrows) = ui.input(|i| {
            let wheel = if hovered {
                i.events.iter().fold(0.0_f32, |a, e| match e {
                    egui::Event::MouseWheel { unit, delta, .. } => {
                        a + match unit {
                            egui::MouseWheelUnit::Point => delta.y / 50.0,
                            _ => delta.y,
                        }
                    }
                    _ => a,
                })
            } else {
                0.0
            };
            let wasd = [
                i.key_down(egui::Key::A),
                i.key_down(egui::Key::D),
                i.key_down(egui::Key::W),
                i.key_down(egui::Key::S),
            ];
            let arrows = [
                i.key_down(egui::Key::ArrowLeft),
                i.key_down(egui::Key::ArrowRight),
                i.key_down(egui::Key::ArrowUp),
                i.key_down(egui::Key::ArrowDown),
            ];
            (i.modifiers.alt, i.modifiers.shift, wheel, wasd, i.key_down(egui::Key::Space), arrows)
        });

        let d = response.drag_delta();
        let moved = d != egui::Vec2::ZERO;
        let drag_l = response.dragged_by(egui::PointerButton::Primary);
        let drag_m = response.dragged_by(egui::PointerButton::Middle);
        let drag_r = response.dragged_by(egui::PointerButton::Secondary);
        let start_l = response.drag_started_by(egui::PointerButton::Primary);
        let start_m = response.drag_started_by(egui::PointerButton::Middle);
        let start_r = response.drag_started_by(egui::PointerButton::Secondary);

        // Mouse drag → orbit / pan / dolly, mapped per navigation style. Orbit
        // modes re-pivot to where the focal axis enters the volume when the orbit
        // drag begins, so you rotate around what's centered in view.
        match self.nav_mode {
            NavMode::Cad => {
                if start_l {
                    self.repivot_to_focal();
                }
                if drag_l && moved {
                    self.vol_orbit(d);
                    animating = true;
                }
                if drag_m && moved {
                    self.vol_pan(d, right, up, pan_speed);
                    animating = true;
                }
                if drag_r && moved {
                    // Right-drag looks around from a fixed eye (first-person).
                    self.vol_look_in_place(d);
                    animating = true;
                }
            }
            NavMode::Blender => {
                if start_m && !shift {
                    self.repivot_to_focal();
                }
                if drag_m && moved {
                    if shift {
                        self.vol_pan(d, right, up, pan_speed);
                    } else {
                        self.vol_orbit(d);
                    }
                    animating = true;
                }
            }
            NavMode::Maya => {
                if alt && start_l {
                    self.repivot_to_focal();
                }
                if alt && moved {
                    if drag_l {
                        self.vol_orbit(d);
                        animating = true;
                    } else if drag_m {
                        self.vol_pan(d, right, up, pan_speed);
                        animating = true;
                    } else if drag_r {
                        // Alt+Right vertical drag dollies (down = out). Floors the
                        // radius so it can back out of a radius-0 (in-place) orbit.
                        self.vol_dist =
                            vol_dist_clamped(self.vol_dist.max(VOL_DIST_UNSTICK) * (1.0 + d.y * 0.005));
                        animating = true;
                    }
                }
            }
            NavMode::WasdFly => {
                if drag_l && moved {
                    self.vol_orbit(d); // mouse-look (first-person)
                    animating = true;
                }
                if start_r {
                    // Right-drag orbits: pivot on where the view enters the box.
                    self.repivot_to_focal();
                }
                if drag_r && moved {
                    self.vol_orbit_fly(d);
                    animating = true;
                }
            }
        }

        // WASD translation, in every mode: fly moves the eye, orbit modes move
        // the pivot. Space/Shift add vertical movement in the fly, CAD and Maya
        // modes — not Blender, where Shift is the pan modifier.
        if hovered {
            let mut mv = [0.0f32; 3];
            if wasd[0] {
                mv[0] -= 1.0;
            }
            if wasd[1] {
                mv[0] += 1.0;
            }
            if wasd[2] {
                mv[2] += 1.0;
            }
            if wasd[3] {
                mv[2] -= 1.0;
            }
            let vertical_keys = self.nav_mode.is_fly() || matches!(self.nav_mode, NavMode::Cad | NavMode::Maya);
            if vertical_keys {
                if space {
                    mv[1] += 1.0;
                }
                if shift {
                    mv[1] -= 1.0;
                }
            }
            if mv != [0.0; 3] {
                let speed = FLY_UNITS_PER_SEC * dt * self.move_speed;
                if self.nav_mode.is_fly() {
                    self.vol_fly_pos = translate3(self.vol_fly_pos, forward, right, mv, speed);
                } else {
                    self.vol_target = translate3(self.vol_target, forward, right, mv, speed);
                }
                animating = true;
            }
        }

        // Arrow keys orbit/look in every mode (a keyboard fallback).
        if hovered {
            let mut arot = egui::Vec2::ZERO;
            if arrows[0] {
                arot.x -= KEY_ROT;
            }
            if arrows[1] {
                arot.x += KEY_ROT;
            }
            if arrows[2] {
                arot.y -= KEY_ROT;
            }
            if arrows[3] {
                arot.y += KEY_ROT;
            }
            if arot != egui::Vec2::ZERO {
                // Apply `arot` like a mouse drag delta so the keys match the
                // pointer's sense of rotation (see `vol_orbit`): yaw is negated,
                // pitch is not. Without the negation the left/right keys spin the
                // camera the wrong way (vertical, which isn't negated, was fine).
                self.vol_yaw -= arot.x;
                self.vol_pitch = (self.vol_pitch + arot.y).clamp(-1.54, 1.54);
                animating = true;
            }
        }

        // Wheel: a linear fly along the focal axis (not a zoom). In fly mode it
        // moves the eye; in orbit modes it moves the whole camera (eye + pivot).
        // Speed is spectator-slow inside the box and grows with the eye's
        // distance from the box, so far views approach fast and near ones creep.
        if wheel.abs() > 0.01 {
            if self.nav_mode.is_fly() {
                for (p, f) in self.vol_fly_pos.iter_mut().zip(forward) {
                    *p += f * wheel * FLY_WHEEL * self.scroll_speed;
                }
            } else {
                let eye = self.current_eye(forward);
                let to_box = focal_box_entry(eye, forward, self.vol_box_he)
                    .unwrap_or_else(|| (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt());
                let m = wheel * (to_box * 0.15).max(FLY_WHEEL) * self.scroll_speed;
                self.vol_target = [
                    self.vol_target[0] + forward[0] * m,
                    self.vol_target[1] + forward[1] * m,
                    self.vol_target[2] + forward[2] * m,
                ];
            }
        }

        animating
    }

    fn sync_gpu(&mut self, egui_ctx: &egui::Context, frame: &eframe::Frame) {
        // The per-frame upload handle (GL context, or device+queue) for whatever
        // backend is compiled in. `None` only before the backend is initialized.
        let Some(ctx) = render::upload_ctx(frame) else { return };
        let Some(loaded) = &mut self.stack else { return };
        let mut resources = self.render.lock().unwrap();

        let n_channels = loaded.channel_settings.len();
        if n_channels == 0 {
            return;
        }

        // Per-channel GPU texture kind (R8Uint / R16Uint / R32F), picked from the
        // source format at load time — drives both texture allocation and the
        // decode path below.
        let kinds: Vec<ChannelKind> = loaded.channel_settings.iter().map(|s| s.kind).collect();

        if let Some(first) = loaded.tiff.frames.first() {
            resources.ensure_size(&ctx, first.width, first.height, &kinds);
        }

        if !loaded.luts_uploaded {
            for c in 0..n_channels {
                resources.upload_lut(&ctx, c, &loaded.tiff.meta.channel_display[c].lut);
            }
            loaded.luts_uploaded = true;
        }

        // 3D volume view: make sure the volume textures hold the current
        // timepoint, then push the camera + per-channel window params. The 2D
        // per-frame decode/upload path below is skipped — the volume holds
        // every slice.
        //
        // The build itself runs on a background thread (`volume::VolumeBuilder`)
        // so neither the initial build nor a 4D timepoint step blocks the UI:
        // until the result lands, the loading screen (initial) or the previous
        // timepoint's volume (4D) stays on screen, and we poll each frame. In
        // the 4D case (`slices > 1`) the volume depth is Z at the current
        // frame_index (time), so playback animates the volume through time; in
        // the ordinary case the frame axis *is* the depth and `time` stays 0.
        if self.view_mode == ViewMode::Volume {
            let is_4d = loaded.tiff.meta.slices > 1;
            let time = if is_4d { loaded.frame_index } else { 0 };
            if self.volume_built_frame != Some(time) {
                // Lazily spawn the background builder on first 3D use (it opens
                // its own mmap of the file, like the prefetch worker).
                if loaded.volume_builder.is_none() && !loaded.volume_builder_tried {
                    loaded.volume_builder = crate::volume::VolumeBuilder::new(loaded.path.clone());
                    loaded.volume_builder_tried = true;
                }
                let max_dim = resources.max_3d_texture_size(&ctx);
                let plan = plan_volume(loaded, max_dim, time);
                let mut handled = false;
                if let Some(builder) = &loaded.volume_builder {
                    if let Some(built) = builder.take_matching(self.volume_gen, time) {
                        if let Some((vw, vh, vd, chans)) = built {
                            resources.upload_volumes(&ctx, vw, vh, vd, &chans);
                        }
                        // Mark built even on failure so we don't retry every
                        // frame (the canvas just stays black).
                        self.volume_built_frame = Some(time);
                        self.volume_requested = None;
                        handled = true;
                    } else {
                        let queued = self.volume_requested == Some((self.volume_gen, time))
                            || builder.request(self.volume_gen, plan.clone());
                        if queued {
                            self.volume_requested = Some((self.volume_gen, time));
                            // In flight: poll again next frame (the previous
                            // volume / loading screen stays up meanwhile).
                            egui_ctx.request_repaint();
                            handled = true;
                        }
                        // queued == false: the worker died (its file open
                        // failed) — fall through to the synchronous build.
                    }
                }
                if !handled {
                    if let Some((vw, vh, vd, chans)) = crate::volume::build_volume(&loaded.tiff, &plan) {
                        resources.upload_volumes(&ctx, vw, vh, vd, &chans);
                    }
                    self.volume_built_frame = Some(time);
                }
            }
            resources.set_volume_interp(&ctx, self.vol_interp);
            let params = build_volume_params(
                loaded,
                VolumeCam {
                    yaw: self.vol_yaw,
                    pitch: self.vol_pitch,
                    dist: self.vol_dist,
                    target: self.vol_target,
                    fly_pos: self.vol_fly_pos,
                    nav: self.nav_mode,
                    scale: self.vol_scale,
                    aspect: self.vol_aspect,
                    render: self.vol_render,
                    density: self.vol_density,
                },
            );
            // Cache the box extents so the orbit re-pivot can ray-cast the box.
            self.vol_box_he = params.box_he;
            resources.set_volume_params(params);
            return;
        }

        // Push the decode-parallelism choice to fast-tiff-lib: Auto follows the
        // playback-keeping-up latch, Serial/Threaded force it off/on.
        fast_tiff_lib::set_parallel_decode(self.decode_mode.parallel(self.decode_parallel));

        // Skip disabled channels (the shader multiplies them out). Re-upload when
        // the frame moves *or* the enabled set changes; an enabled-set change also
        // bumps the prefetch generation so an in-flight prefetch under the old set
        // is recognized as stale.
        let enabled: Vec<bool> = loaded.channel_settings.iter().map(|s| s.enabled).collect();
        if loaded.last_enabled != enabled {
            loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
        }
        if loaded.last_uploaded != Some(loaded.frame_index) || loaded.last_enabled != enabled {
            let frame_index = loaded.frame_index;
            let want_gen = loaded.prefetch_gen;
            let jobs = build_jobs(loaded, frame_index, &enabled, &kinds);

            // Use a prefetched frame if one is ready and matches exactly
            // (generation, frame index, and channel layout); otherwise decode
            // inline. A mismatch only costs a little redundant work — it can
            // never upload the wrong frame.
            let mut used_prefetch = false;
            if let Some(p) = &loaded.prefetch {
                if let Some(result) = p.take_matching(want_gen, frame_index) {
                    if prefetch_matches(&result, &jobs) {
                        for ch in &result.channels {
                            match &ch.data {
                                Decoded::U8(v) => resources.upload_channel_u8(&ctx, ch.channel, ch.width, ch.height, v),
                                Decoded::U16(v) => resources.upload_channel(&ctx, ch.channel, ch.width, ch.height, v),
                                Decoded::F32(v) => resources.upload_channel_f32(&ctx, ch.channel, ch.width, ch.height, v),
                            }
                        }
                        used_prefetch = true;
                    }
                }
            }
            if !used_prefetch {
                // One call decodes every enabled channel; RGB planes share a
                // single decompression pass inside `decode_jobs`.
                match decode_jobs(&loaded.tiff.mmap, &loaded.tiff.frames, loaded.tiff.byte_order, &jobs) {
                    Ok(decoded) => {
                        for (job, data) in jobs.iter().zip(decoded) {
                            match data {
                                Decoded::U8(v) => resources.upload_channel_u8(&ctx, job.channel, job.width, job.height, &v),
                                Decoded::U16(v) => resources.upload_channel(&ctx, job.channel, job.width, job.height, &v),
                                Decoded::F32(v) => resources.upload_channel_f32(&ctx, job.channel, job.width, job.height, &v),
                            }
                        }
                    }
                    Err(e) => self.status = Some(format!("Failed to decode frame: {e:#}")),
                }
            }
            loaded.last_uploaded = Some(frame_index);
        }

        // Read-ahead: while playing and keeping up (serial regime), ask the
        // worker to prepare the next frame — decode it (compressed) or touch
        // its pages (uncompressed) — so reaching it costs only the upload.
        // Skipped when behind (parallel decode handles that).
        if self.playing && !self.decode_parallel {
            if let Some(p) = &loaded.prefetch {
                let n = loaded.tiff.meta.frames.max(1);
                if n > 1 {
                    let next = (loaded.frame_index + 1) % n;
                    let next_jobs = build_jobs(loaded, next, &enabled, &kinds);
                    p.request(loaded.prefetch_gen, next, next_jobs);
                }
            }
        }
        loaded.last_enabled = enabled;

        // Window/level goes to the shader in the units its texture actually
        // holds: 16-bit ints in raw 0..65535, floats in their own units (R32F
        // holds raw samples), and 8-bit ints in 0..255 — the slider keeps the
        // window in 0..65535, so an 8-bit channel's bounds are rescaled by 257
        // (the widening factor) here. `is_float` tells the shader which texture
        // to sample; the two integer formats share one sampler.
        const SCALE_8BIT: f32 = 257.0;
        let uniforms: Vec<ChannelUniform> = loaded
            .channel_settings
            .iter()
            .map(|s| {
                let scale = if s.kind == ChannelKind::Int8 { SCALE_8BIT } else { 1.0 };
                ChannelUniform {
                    min: s.min / scale,
                    max: s.max / scale,
                    enabled: s.enabled,
                    is_float: s.kind == ChannelKind::Float,
                }
            })
            .collect();
        resources.set_params(&ctx, &uniforms, n_channels as u32, self.uv_offset.into(), self.uv_scale.into());
    }
}

/// The per-channel decode jobs for `frame_index`'s enabled channels, used both
/// to decode inline and to ask the prefetch worker for the next frame. Maps each
/// display channel to its IFD/plane: for RGB, all channels are sample planes of
/// one IFD per frame; otherwise each channel is its own IFD in ImageJ's default
/// `xyczt` plane order (channel fastest, then Z — frozen at slice 0 — then time).
fn build_jobs(loaded: &LoadedStack, frame_index: usize, enabled: &[bool], kinds: &[ChannelKind]) -> Vec<ChannelJob> {
    let (width, height) = match loaded.tiff.frames.first() {
        Some(f) => (f.width, f.height),
        None => return Vec::new(),
    };
    let meta = &loaded.tiff.meta;
    (0..loaded.channel_settings.len())
        .filter(|&c| enabled.get(c).copied().unwrap_or(false))
        .map(|c| {
            let (ifd_idx, plane) = if loaded.rgb {
                (frame_index * meta.slices, c)
            } else {
                (frame_index * meta.slices * meta.channels + c, 0)
            };
            ChannelJob { channel: c, ifd_idx, plane, kind: kinds[c], rgb: loaded.rgb, width, height }
        })
        .collect()
}

/// Snapshot everything the volume builder needs (see `volume::VolumePlan`):
/// the dimensions come from the app's (possibly manually overridden) metadata
/// so a channels/frames swap is honored, `time` is the 4D timepoint to build.
fn plan_volume(loaded: &LoadedStack, max_dim: u32, time: usize) -> crate::volume::VolumePlan {
    crate::volume::VolumePlan {
        kinds: loaded.channel_settings.iter().map(|s| s.kind).collect(),
        rgb: loaded.rgb,
        channels: loaded.tiff.meta.channels,
        slices: loaded.tiff.meta.slices,
        frames: loaded.tiff.meta.frames,
        time,
        max_dim,
    }
}

/// The 3D camera control state, snapshotted from `ViewerApp` each frame and fed
/// to the params builder. Bundled so the plumbing stays a single argument.
#[derive(Clone, Copy)]
struct VolumeCam {
    yaw: f32,
    pitch: f32,
    dist: f32,
    target: [f32; 3],
    fly_pos: [f32; 3],
    nav: NavMode,
    scale: [f32; 3],
    aspect: f32,
    render: render::VolumeRender,
    density: f32,
}

/// Assemble the ray-march uniforms for the current camera + window. The volume's
/// depth axis is Z in the 4D case (else the frame axis); the box half-extents
/// fold in the per-axis scale so anisotropic voxels render with correct
/// proportions regardless of the (subsampled) texture size.
fn build_volume_params(loaded: &LoadedStack, view: VolumeCam) -> render::VolumeParams {
    let f0 = loaded.tiff.frames.first();
    let w = f0.map(|f| f.width).unwrap_or(1);
    let h = f0.map(|f| f.height).unwrap_or(1);
    let slices = loaded.tiff.meta.slices.max(1);
    let d = if slices > 1 { slices as u32 } else { loaded.tiff.meta.frames.max(1) as u32 };
    let cam = volume_camera(view, (w, h, d));

    // Per-channel window/level, in the sampled texture's units: raw for float,
    // else the 0..65535 display window divided by 65535 (both U8 and U16 volumes
    // are unorm-normalized — see render::VolumeKind).
    let n = loaded.channel_settings.len().min(MAX_CHANNELS);
    let mut windows = [0.0f32; MAX_CHANNELS * 2];
    let mut enabled = [0.0f32; MAX_CHANNELS];
    let mut is_float = [0.0f32; MAX_CHANNELS];
    for (c, s) in loaded.channel_settings.iter().take(MAX_CHANNELS).enumerate() {
        let (mut lo, mut hi) = (s.min, s.max);
        let float = s.kind == ChannelKind::Float;
        if !float {
            lo /= 65535.0;
            hi /= 65535.0;
        }
        windows[c * 2] = lo;
        windows[c * 2 + 1] = hi;
        enabled[c] = if s.enabled { 1.0 } else { 0.0 };
        is_float[c] = if float { 1.0 } else { 0.0 };
    }

    render::VolumeParams {
        num_channels: n as i32,
        windows,
        enabled,
        is_float,
        render_mode: view.render.shader_mode(),
        density: view.density,
        eye: cam.eye,
        forward: cam.forward,
        right: cam.right,
        up: cam.up,
        tan_half_fov: cam.tan_half_fov,
        aspect: view.aspect,
        box_he: cam.box_he,
    }
}

/// The camera basis (eye + orthonormal forward/right/up) and volume-box
/// half-extents the ray-march shader consumes.
struct VolumeCamera {
    eye: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    tan_half_fov: f32,
    box_he: [f32; 3],
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Translate `base` by motion `mv` = (strafe, up, forward) relative to the look
/// basis (`forward`/`right`, with world-Y as up).
fn translate3(base: [f32; 3], forward: [f32; 3], right: [f32; 3], mv: [f32; 3], speed: f32) -> [f32; 3] {
    [
        base[0] + (forward[0] * mv[2] + right[0] * mv[0]) * speed,
        base[1] + (forward[1] * mv[2] + right[1] * mv[0]) * speed + mv[1] * speed,
        base[2] + (forward[2] * mv[2] + right[2] * mv[0]) * speed,
    ]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-6 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Orbit camera distance (eye→pivot) bounds. `MIN = 0` lets the re-pivot put the
/// pivot right at the eye (rotate in place) when the eye is inside the volume;
/// wheel/dolly keep a small floor (`UNSTICK`) so they never sit exactly on the
/// pivot and can back out of a radius-0 orbit.
const VOL_DIST_MIN: f32 = 0.0;
const VOL_DIST_MAX: f32 = 300.0;
const VOL_DIST_UNSTICK: f32 = 0.02;

fn vol_dist_clamped(dist: f32) -> f32 {
    dist.clamp(VOL_DIST_MIN, VOL_DIST_MAX)
}

/// Near intersection distance of the ray `ro + t*rd` with the axis-aligned box
/// `[-he, he]` (a slab test). `None` if the ray misses the box ahead of the eye;
/// clamped to ≥ 0, so it's 0 when the eye is already inside the box.
fn focal_box_entry(ro: [f32; 3], rd: [f32; 3], he: [f32; 3]) -> Option<f32> {
    let mut t0 = f32::NEG_INFINITY;
    let mut t1 = f32::INFINITY;
    for i in 0..3 {
        if rd[i].abs() < 1e-9 {
            // Ray parallel to this slab: a miss unless the eye is between its faces.
            if ro[i] < -he[i] || ro[i] > he[i] {
                return None;
            }
        } else {
            let inv = 1.0 / rd[i];
            let mut ta = (-he[i] - ro[i]) * inv;
            let mut tb = (he[i] - ro[i]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
        }
    }
    if t1 < t0.max(0.0) {
        return None;
    }
    Some(t0.max(0.0))
}

/// Orthonormal camera basis (`forward`, `right`, `up`) for an orientation. At
/// `yaw = pitch = 0` the camera looks along -Z with +Y up; yaw spins around the
/// world vertical, pitch tilts. Shared by `volume_camera` and the pan/fly input
/// math so both agree on which way "right"/"up"/"forward" point.
fn volume_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let pitch = pitch.clamp(-1.54, 1.54); // ~±88°, avoid the pole singularity
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let sph = [cp * sy, sp, cp * cy]; // origin -> orbit eye
    let forward = norm3([-sph[0], -sph[1], -sph[2]]);
    let right = norm3(cross(forward, [0.0, 1.0, 0.0]));
    let up = norm3(cross(right, forward));
    (forward, right, up)
}

/// Camera basis + eye + volume-box half-extents for the ray-marcher. Orbit modes
/// place the eye at `target - forward*dist` (looking at the pivot); the free-fly
/// mode uses `fly_pos` directly. The box's largest scaled axis is 0.5.
fn volume_camera(view: VolumeCam, dims: (u32, u32, u32)) -> VolumeCamera {
    let (forward, right, up) = volume_basis(view.yaw, view.pitch);
    let dist = vol_dist_clamped(view.dist);
    let eye = if view.nav.is_fly() {
        view.fly_pos
    } else {
        [
            view.target[0] - forward[0] * dist,
            view.target[1] - forward[1] * dist,
            view.target[2] - forward[2] * dist,
        ]
    };

    let scale = view.scale;
    let phys = [dims.0 as f32 * scale[0], dims.1 as f32 * scale[1], dims.2 as f32 * scale[2]];
    let m = phys[0].max(phys[1]).max(phys[2]).max(1e-6);
    let box_he = [
        (0.5 * phys[0] / m).max(1e-3),
        (0.5 * phys[1] / m).max(1e-3),
        (0.5 * phys[2] / m).max(1e-3),
    ];
    let tan_half_fov = (45.0f32.to_radians() * 0.5).tan();
    VolumeCamera { eye, forward, right, up, tan_half_fov, box_he }
}

/// Whether a prefetched result still matches the wanted jobs (same channels, in
/// order, with matching kind + dimensions). The generation/frame check happens
/// first; this guards against any residual layout mismatch before upload.
fn prefetch_matches(result: &PrefetchResult, jobs: &[ChannelJob]) -> bool {
    result.channels.len() == jobs.len()
        && result.channels.iter().zip(jobs).all(|(ch, job)| {
            ch.channel == job.channel && ch.kind == job.kind && ch.width == job.width && ch.height == job.height
        })
}

/// Actual pixel min/max of channel `c`'s first frame, for integer-format
/// data. Used as the auto-contrast fallback when no display range came
/// from the file's metadata.
fn first_frame_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    let pixels = fast_tiff_lib::read_frame_u16(&tiff.mmap, frame, tiff.byte_order, None).ok()?;
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    for &p in pixels.iter() {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if hi <= lo {
        hi = lo.saturating_add(1);
    }
    Some((lo as f32, hi as f32))
}

/// Actual float min/max of channel `c`'s first frame, for 32-bit float
/// data — matches ImageJ auto-ranging a float image to its own values
/// rather than assuming a fixed integer-shaped scale.
fn first_frame_float_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    fast_tiff_lib::frame_float_minmax(&tiff.mmap, frame, tiff.byte_order).ok()?
}

/// Resizes `meta.channel_display` to `new_channels` entries, preserving the
/// per-channel display range. When the channel count is *unchanged* (the usual
/// case after `resolve_dimensions`), the existing LUTs are kept — including any
/// custom per-channel colors supplied by the IJMetadata block. When the count
/// *changes* (a mislabeled `channels=N` collapsing to a single channel, or a
/// manual channels/frames swap), the old LUTs no longer correspond to the new
/// channels, so they're regenerated from `mode` — which also avoids leaving a
/// collapsed grayscale stack wearing a stale composite (e.g. red) LUT.
fn resize_channel_display(meta: &mut fast_tiff_lib::StackMeta, new_channels: usize) {
    let old = std::mem::take(&mut meta.channel_display);
    let mode = meta.mode;
    let keep_luts = new_channels == old.len();
    meta.channel_display = (0..new_channels)
        .map(|c| fast_tiff_lib::ChannelDisplay {
            lut: if keep_luts {
                old[c].lut
            } else {
                fast_tiff_lib::default_lut_for(mode, c)
            },
            range: old.get(c).and_then(|d| d.range),
        })
        .collect();
}

/// The contrast range-slider's track bounds: the channel's data min/max
/// (when known) unioned with the current display `window`, so both handles
/// always land on the track and the user has a little headroom to widen the
/// window past the metadata defaults. Falls back to the window alone when the
/// data range is unknown, and pads a degenerate (zero-width) range so the
/// slider stays usable.
fn slider_bounds(window: (f32, f32), data: Option<(f32, f32)>) -> (f32, f32) {
    let (mut lo, mut hi) = window;
    if let Some((dlo, dhi)) = data {
        lo = lo.min(dlo);
        hi = hi.max(dhi);
    }
    if !(hi > lo) {
        let pad = lo.abs().max(1.0);
        lo -= pad;
        hi += pad;
    }
    (lo, hi)
}

/// Formats a raw sample value for display, applying the stack's linear
/// calibration (`c0 + c1 * raw`) when present so the user sees real values;
/// otherwise shows the raw value. Picks a coarse/fine precision by magnitude.
fn format_calibrated(calibration: Option<(f64, f64)>, raw: f32) -> String {
    let v = match calibration {
        Some((c0, c1)) => c0 + c1 * raw as f64,
        None => raw as f64,
    };
    if v.abs() >= 100.0 || v.fract().abs() < 1e-6 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// The contrast range sliders never draw narrower than this, no matter how
/// small the window gets — below it the two handles collide and the slider
/// stops being usable. The value text to the right clips first.
const MIN_CONTRAST_SLIDER_W: f32 = 80.0;

/// A two-handle horizontal range slider editing `(min, max)` within the
/// inclusive track `[lo, hi]` (all in raw sample units). The handles can't
/// cross. `salt` disambiguates the interaction ids when several sliders share
/// a parent (e.g. one per channel). `tint`, when set, colors the selected span
/// with the channel's display color (composite/RGB or pseudocolor); otherwise
/// the default selection color is used.
fn range_slider(
    ui: &mut egui::Ui,
    salt: u64,
    min: &mut f32,
    max: &mut f32,
    lo: f32,
    hi: f32,
    width: f32,
    tint: Option<Color32>,
) {
    // Defensive: keep the handles inside the track and ordered, even if the
    // values were pushed out of range elsewhere (e.g. by the shift-sync).
    *min = (*min).clamp(lo, hi);
    *max = (*max).clamp(lo, hi).max(*min);
    let height = 18.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let span = (hi - lo).max(f32::EPSILON);
    let x_of = |v: f32| rect.left() + ((v - lo) / span).clamp(0.0, 1.0) * rect.width();
    let v_of = |x: f32| lo + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) * span;
    let track_y = rect.center().y;
    let visuals = ui.visuals().clone();

    // Track + the selected span between the two handles.
    let track = egui::Rect::from_min_max(
        egui::pos2(rect.left(), track_y - 2.0),
        egui::pos2(rect.right(), track_y + 2.0),
    );
    ui.painter().rect_filled(track, 2.0, visuals.widgets.inactive.bg_fill);
    let sel = egui::Rect::from_min_max(
        egui::pos2(x_of(*min), track_y - 2.0),
        egui::pos2(x_of(*max), track_y + 2.0),
    );
    ui.painter().rect_filled(sel, 2.0, tint.unwrap_or(visuals.selection.bg_fill));

    let radius = 6.0;
    // min handle.
    {
        let id = ui.id().with((salt, "range_min"));
        let hit = egui::Rect::from_center_size(egui::pos2(x_of(*min), track_y), egui::vec2(radius * 2.5, height));
        let resp = ui.interact(hit, id, egui::Sense::drag());
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                *min = v_of(p.x).min(*max);
            }
        }
        let col = handle_color(&visuals, resp.dragged() || resp.hovered());
        ui.painter().circle_filled(egui::pos2(x_of(*min), track_y), radius, col);
    }
    // max handle.
    {
        let id = ui.id().with((salt, "range_max"));
        let hit = egui::Rect::from_center_size(egui::pos2(x_of(*max), track_y), egui::vec2(radius * 2.5, height));
        let resp = ui.interact(hit, id, egui::Sense::drag());
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                *max = v_of(p.x).max(*min);
            }
        }
        let col = handle_color(&visuals, resp.dragged() || resp.hovered());
        ui.painter().circle_filled(egui::pos2(x_of(*max), track_y), radius, col);
    }
}

fn handle_color(visuals: &egui::Visuals, active: bool) -> Color32 {
    if active {
        visuals.widgets.active.fg_stroke.color
    } else {
        visuals.widgets.inactive.fg_stroke.color
    }
}

/// The channel's display color for tinting its contrast slider, taken from the
/// top (full-intensity) entry of its LUT. Returns `None` for a plain grayscale
/// LUT (`r == g == b`), so only genuinely colored channels — composite/RGB, or a
/// pseudocolored grayscale stack — get a tinted slider; grayscale ones keep the
/// default selection color.
fn channel_tint(lut: &[[u8; 3]; 256]) -> Option<Color32> {
    let [r, g, b] = lut[255];
    if r == g && g == b {
        None
    } else {
        Some(Color32::from_rgb(r, g, b))
    }
}

/// Like `channel_tint`, but taken from the LUT's *low* (index 0) entry — the
/// color the darkest samples map to. Used by the single-channel grayscale color
/// selector, where the low end better identifies a perceptual colormap (whose
/// high end washes out to near-white) than the top does. `None` for a
/// grayscale/black low entry (`r == g == b`), so plain grayscale and the pure
/// single-hue channel ramps — whose low end is black — keep the default color.
fn ui_tint(lut: &[[u8; 3]; 256]) -> Option<Color32> {
    let [r, g, b] = lut[127];
    (!(r == g && g == b)).then(|| Color32::from_rgb(r, g, b))
}

/// The named channel colors the single-channel grayscale color selector offers
/// (after plain grayscale, before the perceptual colormaps). Order matches the
/// palette `fast_tiff_lib::default_composite_lut` cycles through, so color option
/// *i* maps to `default_composite_lut(i)`. White is omitted — its ramp is
/// identical to grayscale.
const GRAY_LUT_COLOR_NAMES: [&str; 6] = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"];

/// Number of options in the grayscale color selector: plain grayscale, then the
/// named channel colors, then the perceptual colormaps (`crate::colormap`).
fn gray_lut_option_count() -> usize {
    1 + GRAY_LUT_COLOR_NAMES.len() + crate::colormap::NAMES.len()
}

/// Display name for grayscale-color option `sel` (see `gray_lut_for` for the
/// index layout).
fn gray_lut_name(sel: usize) -> &'static str {
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        "Grayscale"
    } else if sel <= colors {
        GRAY_LUT_COLOR_NAMES[sel - 1]
    } else {
        crate::colormap::NAMES[sel - 1 - colors]
    }
}

/// The 256-entry LUT for grayscale-color option `sel`: `0` = plain grayscale,
/// `1..=6` = the named channel colors, the rest = the perceptual colormaps —
/// the same order the selector lists them in.
fn gray_lut_for(sel: usize) -> [[u8; 3]; 256] {
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        fast_tiff_lib::grayscale_lut()
    } else if sel <= colors {
        fast_tiff_lib::default_composite_lut(sel - 1)
    } else {
        crate::colormap::LUTS[sel - 1 - colors]
    }
}

/// Builds the UI-level per-channel settings (window/level, enabled,
/// float-encoding range) from `tiff.meta`'s current channel count and
/// display info.
fn build_channel_settings(tiff: &TiffStack) -> Vec<ChannelSettings> {
    (0..tiff.meta.channels.min(MAX_CHANNELS))
        .map(|c| {
            let disp = &tiff.meta.channel_display[c];
            let frame = tiff.frames.get(c);
            let is_float = frame
                .is_some_and(|f| f.sample_format == fast_tiff_lib::SampleFormat::Float && f.bits_per_sample == 32);
            // Unsigned single-sample 8-bit can upload raw (R8Uint) instead of
            // being widened to 16-bit on the CPU each frame.
            let is_u8 = frame.is_some_and(|f| {
                f.bits_per_sample == 8
                    && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt
                    && f.samples_per_pixel == 1
            });

            if is_float {
                let data = first_frame_float_minmax(tiff, c);
                let (lo, hi) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .or(data)
                    .unwrap_or((0.0, 1.0));
                let bounds = slider_bounds((lo, hi), data);
                ChannelSettings { min: lo, max: hi, enabled: true, bounds, kind: ChannelKind::Float }
            } else {
                let data = first_frame_minmax(tiff, c);
                let (min, max) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    // No display range in metadata at all (not even
                    // ImageDescription min=/max=) — fall back to the actual
                    // min/max of channel c's first frame.
                    .or(data)
                    .unwrap_or((0.0, 65535.0));
                let bounds = slider_bounds((min, max), data);
                // min/max stay in the widened 0..65535 space (slider unchanged);
                // for an 8-bit (R8Uint) channel `sync_gpu` rescales them to the
                // 0..255 the texture actually holds.
                let kind = if is_u8 { ChannelKind::Int8 } else { ChannelKind::Int16 };
                ChannelSettings { min, max, enabled: true, bounds, kind }
            }
        })
        .collect()
}

/// The status note shown at the top of the window, derived from the
/// stack's current (resolved) dimensions. Shared between the initial load
/// and the manual dimension-order override so the two can't drift out of
/// sync with each other.
fn compute_status(meta: &fast_tiff_lib::StackMeta, triple_axis_warning: bool) -> Option<String> {
    if triple_axis_warning {
        Some(format!(
            "Warning: this file has channels, Z-slices, and time frames all present at once \
             ({} channel(s) × {} Z-slice(s) × {} frame(s)). Z isn't shown as a separate axis here \
             — only the first Z-slice is used; scrubbing covers channels × time only.",
            meta.channels, meta.slices, meta.frames
        ))
    } else if meta.channels > MAX_CHANNELS {
        Some(format!(
            "Note: stack has {} channels; showing the first {MAX_CHANNELS}.",
            meta.channels
        ))
    } else {
        None
    }
}

/// The 3D render-settings pop-up: rendering method (+ alpha density), per-axis
/// voxel scale (x:y:z), interpolation, navigation style, and a camera-reset
/// button. Scale defaults to the stack's pixel calibration (with a button to
/// re-seed it). `reset_position` is set true when the user clicks Reset position.
#[allow(clippy::too_many_arguments)]
fn render_settings_window(
    ctx: &egui::Context,
    open: &mut bool,
    scale: &mut [f32; 3],
    interp: &mut render::VolumeInterp,
    nav: &mut NavMode,
    move_speed: &mut f32,
    scroll_speed: &mut f32,
    render_mode: &mut render::VolumeRender,
    density: &mut f32,
    reset_position: &mut bool,
    loaded: Option<&LoadedStack>,
) {
    egui::Window::new("3D render settings")
        .open(open)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.label(RichText::new("Rendering").strong());
            ui.horizontal(|ui| {
                ui.selectable_value(render_mode, render::VolumeRender::Mip, "Max intensity")
                    .on_hover_text("Maximum-intensity projection: brightest sample along each ray");
                ui.selectable_value(render_mode, render::VolumeRender::Alpha, "Volume (ImageJ)")
                    .on_hover_text("ImageJ 3D Viewer style: translucent alpha-blended volume");
            });
            // Density only affects the alpha DVR — disabled for MIP.
            ui.add_enabled_ui(*render_mode == render::VolumeRender::Alpha, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Density");
                    ui.add(egui::Slider::new(density, 1.0..=1000.0).logarithmic(true))
                        .on_hover_text("Opacity of the alpha volume (higher = brighter/more solid)");
                });
            });

            ui.separator();
            ui.label(RichText::new("Voxel scale (x : y : z)").strong());
            ui.horizontal(|ui| {
                for (i, axis) in ["x", "y", "z"].iter().enumerate() {
                    ui.label(*axis);
                    ui.add(
                        egui::DragValue::new(&mut scale[i])
                            .speed(0.01)
                            .range(0.0001..=100_000.0)
                            .max_decimals(4),
                    );
                }
            });
            if let Some(loaded) = loaded {
                ui.horizontal(|ui| {
                    if ui
                        .button("Reset from metadata")
                        .on_hover_text("Re-seed x:y:z from the file's pixel calibration + spacing (else 1:1:1)")
                        .clicked()
                    {
                        *scale = loaded.tiff.meta.voxel_scale();
                    }
                    if let Some(unit) = loaded.tiff.meta.unit.as_deref().filter(|u| !u.is_empty()) {
                        ui.label(RichText::new(format!("unit: {unit}")).weak());
                    }
                });
            }

            ui.separator();
            ui.label(RichText::new("Interpolation").strong());
            ui.horizontal(|ui| {
                ui.selectable_value(interp, render::VolumeInterp::Nearest, "None (nearest)")
                    .on_hover_text("Crisp voxels, no smoothing");
                ui.selectable_value(interp, render::VolumeInterp::Linear, "Trilinear")
                    .on_hover_text("Smoothly interpolated samples");
                ui.selectable_value(interp, render::VolumeInterp::Cubic, "Cubic")
                    .on_hover_text("Tricubic B-spline: smoothest, but slower (8 taps/sample)");
            });

            ui.separator();
            ui.label(RichText::new("Navigation").strong());
            ui.horizontal_wrapped(|ui| {
                for mode in [NavMode::Cad, NavMode::Blender, NavMode::Maya, NavMode::WasdFly] {
                    ui.selectable_value(nav, mode, mode.label()).on_hover_text(mode.help());
                }
            });
            // Controls hint for the selected mode.
            ui.label(RichText::new(nav.help()).small().weak());
            // Speed multipliers on the built-in WASD / wheel base rates.
            // Logarithmic so 1.0 (the default) sits mid-track with symmetric
            // slower/faster range; the reset button restores both to 1×.
            ui.horizontal(|ui| {
                ui.label("Move speed");
                ui.add(egui::Slider::new(move_speed, 0.1..=10.0).logarithmic(true))
                    .on_hover_text("WASD / Space / Shift movement speed (× the default)");
            });
            ui.horizontal(|ui| {
                ui.label("Scroll speed");
                ui.add(egui::Slider::new(scroll_speed, 0.1..=10.0).logarithmic(true))
                    .on_hover_text("Mouse-wheel fly speed (× the default)");
                if ui.small_button("Reset").on_hover_text("Restore both speeds to 1×").clicked() {
                    *move_speed = 1.0;
                    *scroll_speed = 1.0;
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(RichText::new("Position").strong());
                if ui
                    .button("Reset position")
                    .on_hover_text("Recenter the camera to the default three-quarter view")
                    .clicked()
                {
                    *reset_position = true;
                }
            });
        });
}

fn metadata_window(ctx: &egui::Context, open: &mut bool, loaded: &LoadedStack) {
    let tiff = &loaded.tiff;
    egui::Window::new("File metadata")
        .open(open)
        .resizable(true)
        .default_width(256.0)
        .vscroll(true)
        .show(ctx, |ui| {
            fn kv(ui: &mut egui::Ui, k: &str, v: impl Into<String>) {
                ui.label(RichText::new(k).strong());
                ui.label(v.into());
                ui.end_row();
            }

            ui.heading("File");
            egui::Grid::new("meta_file").num_columns(2).striped(true).show(ui, |ui| {
                kv(ui, "Size", human_bytes(tiff.mmap.len() as u64));
                let container = match tiff.flavor {
                    fast_tiff_lib::TiffFlavor::Classic => "classic TIFF",
                    fast_tiff_lib::TiffFlavor::Big => "BigTIFF",
                };
                kv(ui, "Container", container);
                let order = match tiff.byte_order {
                    fast_tiff_lib::ByteOrder::Little => "little-endian (II)",
                    fast_tiff_lib::ByteOrder::Big => "big-endian (MM)",
                };
                kv(ui, "Byte order", order);
                kv(ui, "Planes (IFDs)", tiff.frames.len().to_string());
            });

            if let Some(f) = tiff.frames.first() {
                ui.add_space(12.0);
                ui.heading("Frame format");
                egui::Grid::new("meta_frame").num_columns(2).striped(true).show(ui, |ui| {
                    kv(ui, "Dimensions", format!("{} x {} px", f.width, f.height));
                    let format = match f.sample_format {
                        fast_tiff_lib::SampleFormat::UnsignedInt => "unsigned integer",
                        fast_tiff_lib::SampleFormat::SignedInt => "signed integer",
                        fast_tiff_lib::SampleFormat::Float => "IEEE float",
                    };
                    kv(ui, "Pixel type", format!("{}-bit {format}", f.bits_per_sample));
                    kv(
                        ui,
                        "Samples/pixel",
                        if f.is_rgb() {
                            format!("{} (chunky RGB)", f.samples_per_pixel)
                        } else {
                            f.samples_per_pixel.to_string()
                        },
                    );
                    let photometric = match f.photometric {
                        0 => "0 (WhiteIsZero)".into(),
                        1 => "1 (BlackIsZero)".into(),
                        2 => "2 (RGB)".into(),
                        3 => "3 (palette)".into(),
                        other => format!("{other}"),
                    };
                    kv(ui, "Photometric", photometric);
                    let compression = match f.compression {
                        fast_tiff_lib::Compression::None => "uncompressed".into(),
                        fast_tiff_lib::Compression::Lzw => "LZW".into(),
                        fast_tiff_lib::Compression::PackBits => "PackBits".into(),
                        fast_tiff_lib::Compression::Deflate => "Deflate (zip)".into(),
                        fast_tiff_lib::Compression::Zstd => "ZSTD".into(),
                        other => format!("{other:?}"),
                    };
                    kv(ui, "Compression", compression);
                    let predictor = match f.predictor {
                        1 => "none".into(),
                        2 => "2 (horizontal differencing)".into(),
                        3 => "3 (floating-point)".into(),
                        other => format!("{other}"),
                    };
                    kv(ui, "Predictor", predictor);
                    kv(
                        ui,
                        "Strips/frame",
                        format!("{} ({} rows/strip)", f.strip_offsets.len(), f.rows_per_strip),
                    );
                    let bpf = f.width as u64
                        * f.height as u64
                        * f.samples_per_pixel as u64
                        * (f.bits_per_sample as u64 / 8);
                    kv(ui, "Decoded frame", human_bytes(bpf));
                });
            }

            ui.add_space(12.0);
            egui::CollapsingHeader::new("ImageDescription (tag 270)")
                .default_open(true)
                .show(ui, |ui| match &tiff.description {
                    Some(desc) => {
                        // Read-only TextEdit: selectable + copyable.
                        let mut text = desc.as_str();
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(f32::INFINITY)
                                .desired_rows(desc.lines().count().clamp(2, 16)),
                        );
                    }
                    None => {
                        ui.label(RichText::new("(this file carries no ImageDescription)").weak());
                    }
                });

            ui.add_space(12.0);
            ui.heading("ImageJ metadata");
            let meta = &tiff.meta;
            egui::Grid::new("meta_ij").num_columns(2).striped(true).show(ui, |ui| {
                kv(
                    ui,
                    "Dimensions",
                    format!(
                        "{} channel(s) x {} slice(s) x {} frame(s)",
                        meta.channels, meta.slices, meta.frames
                    ),
                );
                let mode = match meta.mode {
                    fast_tiff_lib::DisplayMode::Grayscale => "grayscale",
                    fast_tiff_lib::DisplayMode::Composite => "composite",
                    fast_tiff_lib::DisplayMode::Color => "color",
                };
                kv(ui, "Display mode", mode);
                if let Some(unit) = &meta.unit {
                    kv(ui, "Unit", unit.clone());
                }
                if let Some(fi) = meta.frame_interval_s {
                    kv(ui, "Frame interval", format!("{fi} s"));
                }
                if let Some(fps) = meta.fps {
                    kv(ui, "Playback fps", fps.to_string());
                }
                if let Some(spacing) = meta.spacing {
                    kv(ui, "Z spacing", spacing.to_string());
                }
                if let Some(looped) = meta.loop_playback {
                    kv(ui, "Loop playback", looped.to_string());
                }
                if let Some((c0, c1)) = meta.calibration {
                    kv(ui, "Calibration", format!("value = {c0} + {c1} x raw"));
                }
                for (i, cd) in meta.channel_display.iter().enumerate() {
                    let range = match cd.range {
                        Some((lo, hi)) => format!("{lo} .. {hi}"),
                        None => "auto-contrast".into(),
                    };
                    kv(ui, &format!("Ch {} display range", i + 1), range);
                }
            });
        });
}

/// `1234567` -> `"1.2 MiB (1234567 bytes)"`.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{:.1} {} ({n} bytes)", v, UNITS[u])
    }
}

/// Applies a (possibly newly resolved) channel/slice/frame interpretation
/// to a stack: updates the metadata, rebuilds channel_display +
/// channel_settings to match the new channel count, and resets the scrub
/// position. The one place that does this, so the manual channels/frames
/// swap can't drift out of sync with `open_file` the way `self.status`
/// previously did.
fn apply_resolved_dimensions(loaded: &mut LoadedStack, resolved: fast_tiff_lib::ResolvedDimensions) {
    loaded.tiff.meta.channels = resolved.channels;
    loaded.tiff.meta.slices = resolved.slices;
    loaded.tiff.meta.frames = resolved.frames;
    loaded.triple_axis_warning = resolved.triple_axis_warning;
    resize_channel_display(&mut loaded.tiff.meta, resolved.channels);
    loaded.channel_settings = build_channel_settings(&loaded.tiff);
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Reconfigures a freshly-loaded chunky-RGB stack: the first `min(spp, 3)`
/// sample planes become red/green/blue display channels with identity
/// full-range windows (so true colors show without any contrast tweaking).
/// Additively blending the three color ramps in the composite shader
/// reconstructs the original RGB pixel. Frame navigation still walks IFDs (one
/// full-color image per IFD) — see `LoadedStack::rgb`.
fn setup_rgb(loaded: &mut LoadedStack) {
    let spp = loaded.tiff.frames.first().map(|f| f.samples_per_pixel as usize).unwrap_or(3);
    let planes = spp.min(3).min(MAX_CHANNELS); // RGB only; ignore any alpha/extra samples
    loaded.rgb = true;
    loaded.tiff.meta.mode = fast_tiff_lib::DisplayMode::Color;
    loaded.tiff.meta.channel_display = (0..planes)
        .map(|c| fast_tiff_lib::ChannelDisplay {
            lut: fast_tiff_lib::default_composite_lut(c), // 0 = red, 1 = green, 2 = blue
            range: None,
        })
        .collect();
    // Unsigned 8-bit RGB deinterleaves into raw u8 planes (`read_plane_u8`) and
    // rides the R8Uint path — half the texture memory + upload of widening each
    // plane to u16. Deeper or signed RGB still widens to u16 via `read_plane_u16`.
    // The window stays in 0..65535 either way; `sync_gpu` rescales it to 0..255
    // for an 8-bit (Int8) channel.
    let kind = if loaded
        .tiff
        .frames
        .first()
        .is_some_and(|f| f.bits_per_sample == 8 && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt)
    {
        ChannelKind::Int8
    } else {
        ChannelKind::Int16
    };
    loaded.channel_settings = (0..planes)
        .map(|_| ChannelSettings {
            min: 0.0,
            max: 65535.0,
            enabled: true,
            bounds: (0.0, 65535.0),
            kind,
        })
        .collect();
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Whether the "apply pseudocolor" option is meaningful for this stack: only
/// multi-channel grayscale stacks (composite files already carry colors; RGB is
/// handled separately) can be optionally tinted with the channel palette.
fn pseudocolor_applicable(loaded: &LoadedStack) -> bool {
    !loaded.rgb
        && loaded.channel_settings.len() > 1
        && loaded.tiff.meta.mode == fast_tiff_lib::DisplayMode::Grayscale
        // The file's own per-channel LUTs win: don't override them with the
        // grayscale/pseudocolor default.
        && !loaded.tiff.meta.has_explicit_luts
}

/// Whether the single-channel grayscale color/colormap selector applies: exactly
/// one channel, genuinely grayscale (not RGB or composite, no file-supplied
/// LUTs). The multi-channel case is covered by the pseudocolor toggle instead.
fn gray_lut_applicable(loaded: &LoadedStack) -> bool {
    !loaded.rgb
        && loaded.channel_settings.len() == 1
        && loaded.tiff.meta.mode == fast_tiff_lib::DisplayMode::Grayscale
        && !loaded.tiff.meta.has_explicit_luts
}

/// Sets the per-channel LUTs of an applicable (multi-channel grayscale) stack:
/// the standard channel palette (ch1 red, ch2 green, …) when `apply` is true,
/// plain grayscale otherwise. No-op for stacks that carry their own colors.
fn refresh_pseudocolor(loaded: &mut LoadedStack, apply: bool) {
    if !pseudocolor_applicable(loaded) {
        return;
    }
    for (c, disp) in loaded.tiff.meta.channel_display.iter_mut().enumerate() {
        disp.lut = if apply {
            fast_tiff_lib::default_composite_lut(c)
        } else {
            fast_tiff_lib::grayscale_lut()
        };
    }
    loaded.luts_uploaded = false; // force re-upload on the next sync
}

/// Applies a manual dimension-order change from the dropdown: reassigns the
/// channels / Z-slices / time-frames roles to the given counts (the product
/// stays the stack's plane count — the selector only offers permutations).
/// For stacks without a real Z axis the selector passes `slices` through
/// unchanged, so it stays a plain channels/time swap. The triple-axis
/// warning flag is carried over — it describes the file, not the current
/// role assignment.
fn apply_dimension_override(loaded: &mut LoadedStack, channels: usize, slices: usize, frames: usize) {
    let resolved = fast_tiff_lib::ResolvedDimensions {
        channels,
        slices,
        frames,
        triple_axis_warning: loaded.triple_axis_warning,
    };
    apply_resolved_dimensions(loaded, resolved);
    // The channel->IFD mapping just changed, so invalidate any in-flight prefetch
    // decoded under the old mapping.
    loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Drag-and-drop files onto the window: open the first in this window and
        // launch each of the rest in its own process, so dropping several at once
        // opens them all side by side (mirrors the command-line behavior).
        let dropped: Vec<PathBuf> =
            ui.ctx().input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if let Some(first) = crate::process::open_all(&dropped) {
            self.open_file(first.clone());
        }

        // macOS "Open With" / double-click delivers files via an Apple Event
        // (not argv); drain whatever `macos_open`'s handler has queued and open
        // them the same way as drag-drop.
        #[cfg(target_os = "macos")]
        {
            let opened = crate::macos_open::take_opened_files();
            if let Some(first) = crate::process::open_all(&opened) {
                self.open_file(first.clone());
            }
        }

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
        if zoom_step != 0 && self.stack.is_some() && self.view_mode == ViewMode::Movie {
            let old_zoom = self.zoom;
            let new_zoom = stepped_zoom(old_zoom, zoom_step);
            if (new_zoom - old_zoom).abs() > f32::EPSILON {
                let cursor = ui
                    .ctx()
                    .input(|i| i.pointer.latest_pos())
                    .filter(|p| self.panel_rect.contains(*p))
                    .unwrap_or_else(|| self.panel_rect.center());
                // The native-pixel point under the cursor, kept fixed by pan
                // (used when the image overflows; re-clamped to 0 when it fits,
                // where the window move below handles the centering instead).
                let p = (cursor - self.image_origin) / old_zoom;
                self.pan = self.panel_rect.min - (cursor - p * new_zoom);
                self.zoom = new_zoom;
                self.resize_to_zoom = true;
                self.zoom_reposition = Some((old_zoom, cursor));
            }
        }

        // 2D/3D view toggle + the 3D-settings button are set inside the toolbar
        // closure via these locals (applied after) so the closure never needs a
        // second borrow of `self`.
        let current_view_mode = self.view_mode;
        let mut mode_request: Option<ViewMode> = None;
        let mut render_settings_toggle = false;

        let toolbar_response = egui::Panel::top("toolbar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open TIFF...").clicked() {
                    // Allow selecting several files at once: open the first in
                    // this window and launch each of the rest in its own process
                    // (same fan-out as drag-drop / the command line).
                    if let Some(paths) = rfd::FileDialog::new()
                        .add_filter("TIFF", &["tif", "tiff"])
                        .pick_files()
                    {
                        if let Some(first) = crate::process::open_all(&paths) {
                            self.open_file(first.clone());
                        }
                    }
                }
                // 2D/3D switch, right next to Open. 3D needs at least two frames
                // to build a volume; disabled otherwise.
                if let Some(loaded) = &self.stack {
                    let can_3d = loaded.tiff.meta.frames >= 2;
                    ui.separator();
                    ui.add_enabled_ui(can_3d, |ui| {
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
                    });
                    // 3D render-settings button, next to the toggle. The file-info
                    // group below adds the trailing separator.
                    ui.separator();
                    if ui
                        .button(RichText::new("⚙").size(16.0))
                        .on_hover_text("3D render settings")
                        .clicked()
                    {
                        render_settings_toggle = true;
                    }
                }
                if self.stack.is_none() {
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
                if let Some(loaded) = &self.stack {
                    let meta = &loaded.tiff.meta;
                    // Reflect a toggle click made earlier in this same toolbar
                    // (the 2D/3D buttons run before this), so the layout updates
                    // on the click frame rather than one frame later.
                    let in_3d = mode_request.unwrap_or(current_view_mode) == ViewMode::Volume;
                    // In 3D the frame axis becomes the volume's depth, so the
                    // frame counter/time are only meaningful when the stack also
                    // has a separate time axis (the channels+Z+time case, where
                    // `slices > 1`); otherwise hide them.
                    let hide_frame_info = in_3d && meta.slices <= 1;

                    // Zoom is a 2D-only concept — hidden entirely in 3D.
                    if !in_3d {
                        ui.separator();
                        // Up to 2 decimals, trailing zeros trimmed: 3.1%, 33.3%,
                        // 100%, 3200% — so the fractional small zooms read correctly.
                        let pct = format!("{:.2}", self.zoom * 100.0);
                        let pct = pct.trim_end_matches('0').trim_end_matches('.');
                        ui.label(RichText::new(format!("{pct}%")).monospace())
                            .on_hover_text("Zoom (Ctrl+scroll to change)");
                    }
                    ui.separator();
                    let channels_desc = if loaded.rgb {
                        "RGB".to_string()
                    } else {
                        format!("{} channel(s)", meta.channels)
                    };
                    let bits = loaded.tiff.frames.first().map(|f| f.bits_per_sample).unwrap_or(0);
                    ui.label(format!(
                        "{}×{} px, {}-bit, {}",
                        loaded.tiff.frames.first().map(|f| f.width).unwrap_or(0),
                        loaded.tiff.frames.first().map(|f| f.height).unwrap_or(0),
                        bits,
                        channels_desc,
                    ));

                    if !hide_frame_info {
                        ui.separator();
                        let frame_digits = meta.frames.to_string().len();
                        ui.label(
                            RichText::new(format!("Frame {:>frame_digits$} / {}", loaded.frame_index + 1, meta.frames))
                                .monospace(),
                        );
                        if let Some(interval) = meta.frame_interval_s {
                            let max_time = meta.frames.saturating_sub(1) as f64 * interval;
                            let time_width = format!("{max_time:.3}").len();
                            let current_time = loaded.frame_index as f64 * interval;
                            ui.label(RichText::new(format!("t = {current_time:>time_width$.3}s")).monospace());
                        }
                    }
                }
            });
        });
        if let Some(mode) = mode_request {
            self.view_mode = mode;
            // Entering 3D stops movie playback — unless the stack is 4D (a
            // separate time axis, `slices > 1`), where playing animates the
            // volume through time.
            if mode == ViewMode::Volume {
                let is_4d = self.stack.as_ref().is_some_and(|l| l.tiff.meta.slices > 1);
                if !is_4d {
                    self.playing = false;
                    self.last_play_time = None;
                }
            }
        }
        // In 3D the arrow keys rotate the volume (handled in the central panel),
        // so the movie's arrow-scrub and wheel-scrub paths must stand down.
        let view_is_volume = self.view_mode == ViewMode::Volume;

        let panel_expanded = self.channels_panel_expanded;
        let is_playing = self.playing;
        let pseudocolor_on = self.apply_pseudocolor;
        let mut toggle_requested = false;
        let mut play_toggle_requested = false;
        // A requested dimension-role reassignment: (channels, slices, frames).
        let mut dimension_override: Option<(usize, usize, usize)> = None;
        let mut pseudocolor_toggle: Option<bool> = None;
        // New selection for the single-channel grayscale color/colormap selector.
        let mut gray_lut_change: Option<usize> = None;
        let mut scroll_step: i32 = 0;
        let mut playback_fps = self.playback_fps;
        let mut decode_mode = self.decode_mode;
        let mut metadata_toggle = false;
        let current_status = self.status.clone();

        let scrub_bar_response = egui::Panel::bottom("scrub_bar").show_inside(ui, |ui| {
            let Some(loaded) = &mut self.stack else {
                ui.label("Open a TIFF stack to begin.");
                return;
            };
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                let max_frame = loaded.tiff.meta.frames.saturating_sub(1);
                let has_multiple_frames = loaded.tiff.meta.frames > 1;
                // In 3D the frame axis is the volume's depth, so play/step/scrub
                // are meaningless unless the stack has a separate time axis
                // (`slices > 1`). Grey them out otherwise.
                let frame_nav_enabled =
                    has_multiple_frames && !(view_is_volume && loaded.tiff.meta.slices <= 1);

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
                        ((loaded.tiff.meta.frames as f64 * FAST_SCROLL_RATE).round() as usize).max(1)
                    } else {
                        1
                    };
                    let max_frame = loaded.tiff.meta.frames.saturating_sub(1);
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
                ui.horizontal(|ui| {
                    // Whether a control group already sits in this row — each
                    // optional group below draws its leading separator only if
                    // so, so hiding a group never leaves an orphaned separator.
                    let mut row_has_items = false;

                    // The channels-vs-time guess (and its override) is
                    // meaningless for RGB, where the "channels" are fixed color
                    // planes — so the dropdown and pseudocolor toggle are hidden
                    // there.
                    if !loaded.rgb {
                        ui.label("Dimension order:");
                        let c = loaded.tiff.meta.channels;
                        let z = loaded.tiff.meta.slices;
                        let f = loaded.tiff.meta.frames;
                        // With a real Z axis alongside channels and time, offer
                        // every assignment of the three counts to the three
                        // roles; otherwise just the channels/time swap (Z passes
                        // through untouched). sort+dedup collapses duplicates
                        // when counts are equal and keeps the list order stable
                        // across reinterpretations.
                        let all_three = c > 1 && z > 1 && f > 1;
                        let mut options: Vec<(usize, usize, usize)> = if all_three {
                            vec![(c, z, f), (c, f, z), (z, c, f), (z, f, c), (f, c, z), (f, z, c)]
                        } else {
                            vec![(c, z, f), (f, z, c)]
                        };
                        options.sort_unstable();
                        options.dedup();
                        let dim_label = |oc: usize, oz: usize, of: usize| {
                            if all_three {
                                format!("c: {oc}  z: {oz}  t: {of}")
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
                    let fps_playable = loaded.tiff.meta.frames > 1
                        && !(view_is_volume && loaded.tiff.meta.slices <= 1);
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

                    // Grayscale color LUT: show a single grayscale channel
                    // through a channel color or a perceptual colormap
                    // (magma/plasma/inferno/viridis/turbo). Hidden for RGB,
                    // composite, and multi-channel stacks — there the per-channel
                    // colors / pseudocolor toggle already handle coloring.
                    if gray_lut_applicable(loaded) {
                        if row_has_items {
                            ui.separator();
                        }
                        ui.label("LUT:");
                        let sel = loaded.gray_lut_sel;
                        egui::ComboBox::from_id_salt("gray_lut")
                            .selected_text(gray_lut_name(sel))
                            .show_ui(ui, |ui| {
                                for opt in 0..gray_lut_option_count() {
                                    // Tint each entry with its LUT's low (dark) end
                                    // — the color the darkest samples map to. A
                                    // grayscale/black low end (grayscale + the pure
                                    // channel-color ramps) keeps the default text color.
                                    let text = match ui_tint(&gray_lut_for(opt)) {
                                        Some(c) => RichText::new(gray_lut_name(opt)).color(c),
                                        None => RichText::new(gray_lut_name(opt)),
                                    };
                                    if ui.selectable_label(opt == sel, text).clicked() {
                                        gray_lut_change = Some(opt);
                                    }
                                }
                            })
                            .response
                            .on_hover_text("Display this grayscale channel through a color LUT or colormap");
                        row_has_items = true;
                    }

                    // CPU decode parallelism: Auto threads only when playback
                    // can't keep up; Serial/Threaded force it off/on. Threaded
                    // decode only ever kicks in for compressed frames (parallel
                    // strip decompression) or 32-bit frames (parallel per-pixel
                    // rescale/cast). 8- and 16-bit uncompressed frames decode
                    // zero-copy or with an unthreaded copy, so the control has no
                    // effect and is hidden for them.
                    let threadable = loaded.tiff.frames.first().is_some_and(|f| {
                        f.compression != fast_tiff_lib::Compression::None || f.bits_per_sample == 32
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
                    }

                    // File-metadata pop-up toggle, pushed to the row's right edge.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(RichText::new("</>").size(16.0))
                            .on_hover_text("See metadata")
                            .clicked()
                        {
                            metadata_toggle = true;
                        }
                    });
                });
                if !loaded.rgb {
                    ui.label(
                        RichText::new(
                            "Channels are guessed automatically (6 or fewer = channels, more = time); \
                             use this if that guess is wrong for this file.",
                        )
                        .small()
                        .weak(),
                    );
                }

                let calibration = loaded.tiff.meta.calibration;
                let rgb = loaded.rgb;
                // Tint for the single-channel contrast slider: the low (dark) end
                // of its chosen color LUT (grayscale/black → None → the default
                // selection color). Snapshot here, before the mutable borrow of
                // `channel_settings` below.
                let single_tint = (loaded.channel_settings.len() == 1)
                    .then(|| ui_tint(&loaded.tiff.meta.channel_display[0].lut))
                    .flatten();
                if loaded.channel_settings.len() > 1 {
                    ui.separator();
                    // Hold Shift while dragging one channel's slider to move every
                    // channel's window by the same amount. Snapshot the values
                    // first so we can detect which one moved and by how much.
                    let shift = ui.input(|i| i.modifiers.shift);
                    let before: Vec<(f32, f32)> =
                        loaded.channel_settings.iter().map(|s| (s.min, s.max)).collect();
                    // Per-channel slider tints from each channel's display LUT —
                    // colored only for composite/RGB or pseudocolor stacks, `None`
                    // (default color) for plain grayscale.
                    let tints: Vec<Option<Color32>> = loaded
                        .tiff
                        .meta
                        .channel_display
                        .iter()
                        .map(|cd| channel_tint(&cd.lut))
                        .collect();
                    // One row per channel — checkbox in line with its slider —
                    // stacked vertically.
                    for (c, settings) in loaded.channel_settings.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let label = if rgb {
                                ["R", "G", "B", "A"].get(c).copied().unwrap_or("Ch").to_string()
                            } else {
                                format!("Ch {}", c + 1)
                            };
                            // Fixed-width checkbox so every slider starts at the
                            // same x regardless of label length.
                            ui.add_sized(egui::vec2(48.0, 18.0), egui::Checkbox::new(&mut settings.enabled, label));
                            let value = format!(
                                "{} – {}",
                                format_calibrated(calibration, settings.min),
                                format_calibrated(calibration, settings.max),
                            );
                            // Reserve room for the value text on the right; the
                            // slider fills what's left of the row.
                            let slider_w = (ui.available_width() - 120.0).max(MIN_CONTRAST_SLIDER_W);
                            let (lo, hi) = settings.bounds;
                            let tint = tints.get(c).copied().flatten();
                            range_slider(ui, c as u64, &mut settings.min, &mut settings.max, lo, hi, slider_w, tint);
                            ui.label(RichText::new(value).small());
                        });
                    }
                    // Shift-sync: if a slider moved this frame, apply the same
                    // delta to every other channel (clamped to its own bounds).
                    if shift {
                        let moved = loaded.channel_settings.iter().enumerate().find_map(|(c, s)| {
                            let (bmin, bmax) = before[c];
                            let (dmin, dmax) = (s.min - bmin, s.max - bmax);
                            if dmin != 0.0 || dmax != 0.0 {
                                Some((c, dmin, dmax))
                            } else {
                                None
                            }
                        });
                        if let Some((src, dmin, dmax)) = moved {
                            for (i, s) in loaded.channel_settings.iter_mut().enumerate() {
                                if i == src {
                                    continue;
                                }
                                s.min = (s.min + dmin).clamp(s.bounds.0, s.bounds.1);
                                s.max = (s.max + dmax).clamp(s.bounds.0, s.bounds.1);
                                if s.min > s.max {
                                    s.min = s.max;
                                }
                            }
                        }
                    }
                    ui.label(
                        RichText::new("Hold Shift while dragging to adjust all channels together.")
                            .small()
                            .weak(),
                    );
                } else if let Some(settings) = loaded.channel_settings.first_mut() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Contrast:");
                        let value = format!(
                            "{} – {}",
                            format_calibrated(calibration, settings.min),
                            format_calibrated(calibration, settings.max),
                        );
                        // Reserve room for the value text on the right; the
                        // slider fills what's left of the row.
                        let slider_w = (ui.available_width() - 120.0).max(MIN_CONTRAST_SLIDER_W);
                        let (lo, hi) = settings.bounds;
                        // Tinted when a color LUT/colormap is chosen, else `None`
                        // (plain grayscale → the default selection color).
                        range_slider(ui, 0, &mut settings.min, &mut settings.max, lo, hi, slider_w, single_tint);
                        ui.label(RichText::new(value).small());
                    });
                }
            }
            if let Some(status) = &current_status {
                // The triple-axis note explains that 2D freezes Z at its first
                // slice — but the 3D view *does* use Z (as the volume depth), so
                // showing it there would be wrong. When `triple_axis_warning` is
                // set, the status IS that note (`compute_status` short-circuits
                // on it), so this suppresses exactly the right message.
                if !(view_is_volume && loaded.triple_axis_warning) {
                    ui.separator();
                    ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)).small());
                }
            }
            ui.add_space(4.0);
        });

        self.playback_fps = playback_fps;
        self.decode_mode = decode_mode;
        if metadata_toggle {
            self.show_metadata = !self.show_metadata;
        }
        if self.show_metadata {
            match &self.stack {
                Some(loaded) => metadata_window(ui.ctx(), &mut self.show_metadata, loaded),
                None => self.show_metadata = false,
            }
        }
        if render_settings_toggle {
            self.show_render_settings = !self.show_render_settings;
        }
        if self.show_render_settings {
            let prev_nav = self.nav_mode;
            let mut reset_position = false;
            render_settings_window(
                ui.ctx(),
                &mut self.show_render_settings,
                &mut self.vol_scale,
                &mut self.vol_interp,
                &mut self.nav_mode,
                &mut self.move_speed,
                &mut self.scroll_speed,
                &mut self.vol_render,
                &mut self.vol_density,
                &mut reset_position,
                self.stack.as_ref(),
            );
            // Keep the view continuous across a fly⇄orbit switch (the two use
            // different eye representations) instead of snapping to a default.
            if self.nav_mode != prev_nav {
                self.sync_camera_for_nav(prev_nav.is_fly());
            }
            if reset_position {
                self.reset_volume_camera();
            }
        }

        if toggle_requested {
            self.channels_panel_expanded = !self.channels_panel_expanded;
            // Remember the panel's height *before* it expands/collapses; the
            // next frame (once it's redrawn in the new state) grows or shrinks
            // the window by the difference. This frame still shows the old
            // height, so the actual delta only becomes known next frame.
            self.panel_grow_armed = true;
            self.panel_old_h = scrub_bar_response.response.rect.height();
        }

        if play_toggle_requested {
            self.playing = !self.playing;
            // Start each play/pause from a clean clock so the first tick after
            // resuming doesn't jump by however long we were paused.
            self.last_play_time = None;
            self.play_accumulator = 0.0;
            // Start the keeping-up estimate neutral (decode_parallel stays
            // latched across a pause — if this stack needed it, it still does).
            self.play_demand_ema = 1.0;
        }

        if let Some(on) = pseudocolor_toggle {
            self.apply_pseudocolor = on;
            if let Some(loaded) = &mut self.stack {
                refresh_pseudocolor(loaded, on);
            }
        }

        if let Some(sel) = gray_lut_change {
            if let Some(loaded) = &mut self.stack {
                loaded.gray_lut_sel = sel;
                if let Some(disp) = loaded.tiff.meta.channel_display.first_mut() {
                    disp.lut = gray_lut_for(sel);
                }
                loaded.luts_uploaded = false; // force LUT re-upload next sync
            }
        }

        // Looped playback: advance by real elapsed time so the movie runs at
        // the file's `fps` (or the default) regardless of render cadence, and
        // request continuous repaints while it's running.
        if self.playing {
            if let Some(loaded) = &mut self.stack {
                let n = loaded.tiff.meta.frames.max(1);
                if n > 1 {
                    let fps = self.playback_fps.max(0.1);
                    let now = ui.input(|i| i.time);
                    if let Some(last) = self.last_play_time {
                        // `demand` = frames the elapsed real time wanted this
                        // render to cover. ~1 when keeping up; >1 means renders
                        // are slower than the fps target (frames dropping → a
                        // core is saturated). Once the smoothed demand crosses
                        // the threshold, latch parallel decoding for this stack.
                        let demand = (now - last) * fps;
                        self.play_demand_ema = self.play_demand_ema * 0.9 + demand as f32 * 0.1;
                        if self.decode_mode == DecodeMode::Auto && self.play_demand_ema > 1.3 {
                            self.decode_parallel = true;
                        }
                        self.play_accumulator += demand;
                        if self.play_accumulator >= 1.0 {
                            let steps = self.play_accumulator.floor() as usize;
                            self.play_accumulator -= steps as f64;
                            loaded.frame_index = (loaded.frame_index + steps) % n;
                        }
                    }
                    self.last_play_time = Some(now);
                    // Ask for the next repaint at the playback rate rather than
                    // immediately: no point re-running egui faster than frames
                    // actually change. If a frame takes longer than this to
                    // produce, egui repaints as soon as it's ready, so we still
                    // render as fast as we can when behind (and `demand` above
                    // still detects it).
                    ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(1.0 / fps));
                } else {
                    self.playing = false;
                }
            }
        } else {
            self.last_play_time = None;
            self.play_accumulator = 0.0;
        }

        if let Some((c, z, f)) = dimension_override {
            if let Some(loaded) = &mut self.stack {
                apply_dimension_override(loaded, c, z, f);
                // The swap rebuilds channel_display from `mode`, so re-apply the
                // pseudocolor preference on top of the fresh LUTs.
                refresh_pseudocolor(loaded, self.apply_pseudocolor);
                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
            }
            // The frame axis (volume depth) just changed — rebuild on next 3D
            // draw, and invalidate any in-flight build under the old layout.
            self.volume_built_frame = None;
            self.volume_gen = self.volume_gen.wrapping_add(1);
            self.volume_requested = None;
        }

        // A dimension swap can collapse the stack to a single frame (e.g.
        // 1 ch × N frames -> N ch × 1 frame), which can't build a volume — the
        // stale volume (first channel only) would keep showing, and with the
        // 2D/3D toggle disabled below two frames the user would be stranded in
        // 3D. Drop back to 2D; the toggle stays disabled until a swap restores
        // a multi-frame layout. (Runs before the central panel so the click
        // frame already renders 2D.)
        if self.view_mode == ViewMode::Volume
            && self.stack.as_ref().is_some_and(|l| l.tiff.meta.frames < 2)
        {
            self.view_mode = ViewMode::Movie;
            self.playing = false;
            self.last_play_time = None;
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
            if self.stack.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Drag and drop a TIFF here, \nor click \"Open TIFF...\" above.\n\n\n\nScroll — navigate frames\nShift + Scroll — fast navigate\nCtrl + Scroll — zoom");
                });
                return;
            }

            let available = ui.available_size();
            let (panel_rect, response) =
                ui.allocate_exact_size(available, egui::Sense::click_and_drag());
            self.panel_rect = panel_rect;

            // 3D volume view: drive the camera per the active nav mode and paint
            // the GPU ray-march. The 2D pan/UV/scrub path below is bypassed. This
            // runs before the `loaded` borrow so it can call `&mut self` methods.
            if self.view_mode == ViewMode::Volume {
                self.vol_aspect = (panel_rect.width() / panel_rect.height().max(1.0)).clamp(0.1, 10.0);

                // Until the first volume is built, show a black loading screen so
                // the heavy decode doesn't freeze on the previous view (`sync_gpu`
                // defers the initial build until after this frame paints).
                if self.volume_built_frame.is_none() {
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
                // Keep repainting while a drag or held key keeps the camera moving.
                if animating {
                    ui.ctx().request_repaint();
                }
                return;
            }

            let Some(loaded) = &self.stack else { return };
            let (Some(w), Some(h)) = (
                loaded.tiff.frames.first().map(|f| f.width),
                loaded.tiff.frames.first().map(|f| f.height),
            ) else {
                return;
            };

            let img_px = egui::vec2(w as f32 * self.zoom, h as f32 * self.zoom);
            // A 1px tolerance so sub-pixel rounding between the window size and
            // the panel's available area doesn't register as a pannable overflow.
            let overflow = egui::vec2(
                (img_px.x - available.x - 1.0).max(0.0),
                (img_px.y - available.y - 1.0).max(0.0),
            );
            let pannable = overflow.x > 0.0 || overflow.y > 0.0;

            // Drag to pan when the image overflows the panel.
            if pannable && response.dragged() {
                self.pan -= response.drag_delta();
            }
            self.pan.x = self.pan.x.clamp(0.0, overflow.x);
            self.pan.y = self.pan.y.clamp(0.0, overflow.y);

            // Where the image's top-left *would* be if drawn full-size: scrolled
            // by `pan` on an overflowing axis, centered on an axis that fits.
            // (Cached for cursor-centered zoom; may lie outside the panel.)
            let origin = egui::pos2(
                if overflow.x > 0.0 { panel_rect.min.x - self.pan.x } else { panel_rect.min.x + (available.x - img_px.x) * 0.5 },
                if overflow.y > 0.0 { panel_rect.min.y - self.pan.y } else { panel_rect.min.y + (available.y - img_px.y) * 0.5 },
            );
            self.image_origin = origin;

            // Render into the on-screen *visible* rectangle only, and pan/zoom
            // via UVs. Drawing into an oversized rect doesn't work: the callback
            // viewport is clamped to the framebuffer, which would just squash the
            // whole image back to fit instead of zooming.
            let full_rect = egui::Rect::from_min_size(origin, img_px);
            let visible = full_rect.intersect(panel_rect);
            if visible.is_positive() {
                let inv = egui::vec2(1.0 / img_px.x.max(1.0), 1.0 / img_px.y.max(1.0));
                self.uv_offset = (visible.min - origin) * inv;
                self.uv_scale = visible.size() * inv;
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
                        let n_frames = self.stack.as_ref().map(|l| l.tiff.meta.frames).unwrap_or(1);
                        let fast_step = (n_frames as f64 * FAST_SCROLL_RATE).max(1.0);
                        // glide < 0 is scroll-down → advance frames. Advance at a
                        // fixed rate *per second* (scaled by the frame time), so
                        // the jump depends only on the glide's real-time duration
                        // — identical for single- and multi-channel stacks despite
                        // their different render speeds. Fractions accumulate so
                        // short stacks still move.
                        let dir = if glide < 0.0 { 1.0 } else { -1.0 };
                        self.scroll_accum += (dir * fast_step * FAST_SCROLL_GLIDE_RATE * dt as f64) as f32;
                        let steps = self.scroll_accum.trunc();
                        self.scroll_accum -= steps;
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
                    self.scroll_accum -= notches;
                    let steps = self.scroll_accum.trunc();
                    self.scroll_accum -= steps;
                    scroll_step = steps as i32;
                }
            } else {
                self.scroll_accum = 0.0;
            }
        });

        if scroll_step != 0 {
            if let Some(loaded) = &mut self.stack {
                let max_frame = loaded.tiff.meta.frames.saturating_sub(1) as i64;
                let target = (loaded.frame_index as i64 + scroll_step as i64).clamp(0, max_frame);
                loaded.frame_index = target as usize;
            }
        }

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
        if self.panel_grow_armed {
            let delta = bottom_bar_height - self.panel_old_h;
            if delta.abs() > 0.5 {
                self.panel_grow_armed = false;
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
            .stack
            .as_ref()
            .and_then(|l| l.tiff.frames.first())
            .map(|f| (f.width as f32, f.height as f32));

        if let Some((img_w, img_h)) = img_dims {
            // On open: pick the largest zoom level ≤ 100% at which the image +
            // chrome still fits the monitor (a huge image opens scaled down, a
            // normal one at 100%). Deferred to here because the chrome height
            // and monitor size aren't known at open time.
            if self.pending_initial_fit {
                if let Some(z) = initial_fit_zoom(ui.ctx(), img_w, img_h, chrome_height) {
                    self.zoom = z;
                    self.pan = egui::Vec2::ZERO;
                    self.pending_initial_fit = false;
                    self.resize_to_zoom = true;
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
                        self.zoom.min(fit)
                    }
                    None => self.zoom,
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
            if let Some((old_zoom, cursor)) = self.zoom_reposition.take() {
                let new_zoom = self.zoom;
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
                        reposition = Some(outer + (cursor - self.panel_rect.min) * (1.0 - ratio));
                    }
                }
            }

            // Apply a pending resize (one-shot), unless maximized.
            if self.resize_to_zoom {
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
                self.resize_to_zoom = false;
            }
        }

        // Window title: loaded filename, or the app name when nothing is open.
        let desired_title = match &self.stack {
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

        self.sync_gpu(ui.ctx(), frame);
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeMode;

    #[test]
    fn decode_mode_drives_parallel_flag() {
        // Serial is always off and Threaded always on, regardless of whether
        // Auto's "falling behind" latch happens to be set.
        assert!(!DecodeMode::Serial.parallel(false));
        assert!(!DecodeMode::Serial.parallel(true));
        assert!(DecodeMode::Threaded.parallel(false));
        assert!(DecodeMode::Threaded.parallel(true));
        // Auto follows the latch: serial until playback falls behind, then parallel.
        assert!(!DecodeMode::Auto.parallel(false));
        assert!(DecodeMode::Auto.parallel(true));
    }
}
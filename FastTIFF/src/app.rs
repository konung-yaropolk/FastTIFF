//! The viewer's egui::App. Holds the loaded stack (if any), per-channel
//! display settings, and the current scrub position. Drives GPU texture
//! uploads directly from UI code (not from inside the paint callback) so a
//! frame change is just: mmap read -> texture upload -> (next frame) draw call.
//! The GPU backend (glow or wgpu) is reached only through `crate::render`'s
//! backend-agnostic surface, so nothing here mentions either by name.
//!
//! This file keeps the state types (`ViewerApp`, `LoadedStack`, …) and the
//! per-frame `ui()` orchestration; the supporting clusters live in child
//! modules (which share this module's privacy, so the split adds no `pub`
//! surface beyond `pub(super)`):
//!   * `camera`     — 3D navigation modes + camera math and input driving
//!   * `gpu_sync`   — per-frame decode/upload + volume plan/uniforms
//!   * `channels`   — per-channel settings, LUT/color + pseudocolor helpers
//!   * `dimensions` — stack-shape (c/z/t) interpretation + status line
//!   * `widgets`    — the contrast range slider + value formatting
//!   * `windows`    — the render-settings and file-metadata pop-ups

use crate::render::{self, Render};
use egui::{Color32, RichText};
use std::path::PathBuf;
use fast_tiff_lib::TiffStack;

mod camera;
mod channels;
mod dimensions;
mod gpu_sync;
mod widgets;
mod windows;
#[cfg(test)]
mod tests;

use camera::NavMode;
use channels::{
    channel_tint, gray_lut_applicable, gray_lut_for, gray_lut_name, gray_lut_option_count,
    pseudocolor_applicable, refresh_pseudocolor, ui_tint,
};
use dimensions::{apply_dimension_override, apply_resolved_dimensions, compute_status, setup_rgb};
use widgets::{format_calibrated, range_slider, MIN_CONTRAST_SLIDER_W};
use windows::{metadata_window, render_settings_window};

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
    /// Whether the file had a real Z axis (`slices > 1`) as loaded. Gates the
    /// three-way (c/z/t) dimension-order selector, and deliberately snapshots
    /// the *load-time* shape rather than the current one: a permutation that
    /// assigns 1 to Z must not collapse the selector to the two-way c/t swap
    /// and strand Z there.
    has_z_axis: bool,
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
                    has_z_axis: false,
                };
                let (c, z, f) = (
                    loaded.tiff.meta.channels,
                    loaded.tiff.meta.slices,
                    loaded.tiff.meta.frames,
                );
                let resolved = fast_tiff_lib::resolve_dimensions(c, z, f);
                apply_resolved_dimensions(&mut loaded, resolved);
                loaded.has_z_axis = loaded.tiff.meta.slices > 1;
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
                    // group below adds the trailing separator. Disabled together
                    // with the 2D/3D toggle: without a third dimension to show,
                    // the 3D render settings have nothing to apply to.
                    ui.separator();
                    ui.add_enabled_ui(can_3d, |ui| {
                        if ui
                            .button(RichText::new("⚙").size(16.0))
                            .on_hover_text("3D render settings")
                            .clicked()
                        {
                            render_settings_toggle = true;
                        }
                    });
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
                        // When the file has a real Z axis (as loaded — see
                        // `has_z_axis`), offer every assignment of the three
                        // counts to the three roles; otherwise just the
                        // channels/time swap (Z passes through untouched).
                        // sort+dedup collapses duplicates when counts are equal
                        // and keeps the list order stable across
                        // reinterpretations.
                        let show_z = loaded.has_z_axis;
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


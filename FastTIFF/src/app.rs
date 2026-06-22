//! The viewer's egui::App. Holds the loaded stack (if any), per-channel
//! display settings, and the current scrub position. Drives GPU texture
//! uploads directly from UI code (not from inside the paint callback —
//! see render/callback.rs for why) so a frame change is just:
//!   mmap read -> queue.write_texture -> (next egui frame) draw call.

use crate::render::pipeline::{ChannelUniform, ImageRenderResources, MAX_CHANNELS};
use eframe::egui_wgpu;
use egui::{Color32, RichText};
use std::path::PathBuf;
use tiff_core::TiffStack;

const ZOOM_STEP: f32 = 0.1;
const MIN_ZOOM: f32 = 0.2;
const MAX_ZOOM: f32 = 8.0;

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
    /// For 32-bit float channels only: the fixed range used to encode raw
    /// float samples into the GPU's 16-bit texture space (see
    /// `tiff_core::read_frame_u16`'s `float_range` parameter), established
    /// once from the channel's first frame and then reused for every
    /// subsequent frame so the texture encoding — and therefore contrast —
    /// doesn't jump around as you scrub. `min`/`max` above are the
    /// user-facing contrast window in the data's own float units (matching
    /// how ImageJ shows float image contrast); they get remapped through
    /// this fixed range into texture-space when building the GPU uniform
    /// (see `sync_gpu`). `None` for integer-format channels, which don't
    /// need this indirection — their texture already holds native values.
    encoding_range: Option<(f32, f32)>,
}

struct LoadedStack {
    tiff: TiffStack,
    path: PathBuf,
    channel_settings: Vec<ChannelSettings>,
    frame_index: usize,
    last_uploaded: Option<usize>,
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
}

pub struct ViewerApp {
    stack: Option<LoadedStack>,
    status: Option<String>,
    /// Channel buttons + contrast sliders are tucked under a small
    /// triangle toggle to keep the bar minimal by default.
    channels_panel_expanded: bool,
    /// Zoom factor: 1.0 = window sized to native image pixels.
    /// Controls the desired *window size* only — the image always fits the
    /// available canvas with aspect ratio preserved regardless of this
    /// value (whether from zooming or from the user manually resizing).
    /// Resets to 1.0 on every file load.
    zoom: f32,
    /// The last `(window_width, chrome_height)` we resized for, to avoid
    /// sending a resize command every frame when nothing has changed. Tracking
    /// chrome height too means expanding/collapsing the bottom panel grows or
    /// shrinks the window by that panel's height — keeping the image area
    /// (and everything above it) put instead of squeezing it.
    last_enforced: Option<(f32, f32)>,
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
}

/// Playback rate used when the file's metadata doesn't specify `fps=`.
const DEFAULT_FPS: f64 = 30.0;

impl ViewerApp {
    pub fn new(initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            stack: None,
            status: None,
            channels_panel_expanded: false,
            zoom: 1.0,
            last_enforced: None,
            last_title: None,
            playing: false,
            last_play_time: None,
            play_accumulator: 0.0,
        };
        if let Some(path) = initial_path {
            app.open_file(path);
        }
        app
    }

    fn open_file(&mut self, path: PathBuf) {
        match TiffStack::open(&path) {
            Ok(tiff) => {
                let mut loaded = LoadedStack {
                    tiff,
                    path,
                    channel_settings: Vec::new(),
                    frame_index: 0,
                    last_uploaded: None,
                    luts_uploaded: false,
                    triple_axis_warning: false,
                    rgb: false,
                };
                let (c, z, f) = (
                    loaded.tiff.meta.channels,
                    loaded.tiff.meta.slices,
                    loaded.tiff.meta.frames,
                );
                let resolved = tiff_core::resolve_dimensions(c, z, f);
                apply_resolved_dimensions(&mut loaded, resolved);
                // Chunky RGB overrides the channel layout: the sample planes of
                // each IFD become red/green/blue display channels.
                if loaded.tiff.frames.first().is_some_and(|f| f.is_rgb()) {
                    setup_rgb(&mut loaded);
                }

                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
                self.stack = Some(loaded);
                // Reset to native 1:1 on every fresh load.
                self.zoom = 1.0;
                self.last_enforced = None;
                self.playing = false;
                self.last_play_time = None;
                self.play_accumulator = 0.0;
            }
            Err(e) => {
                self.status = Some(format!("Failed to open file: {e:#}"));
            }
        }
    }

    /// Index into `tiff.frames` for (current frame, channel). Z is never
    /// separately navigable (see `resolve_dimensions`) — this always reads
    /// the first Z-slice within each frame's stride, which is correct
    /// whether `meta.slices` is 1 (the common case, Z folded away
    /// entirely) or >1 (the rare channels+Z+time case, where Z stays
    /// frozen at index 0 by construction). Assumes ImageJ's default
    /// `xyczt` plane order (channel varies fastest, then z, then t) — if
    /// scrubbing shows the wrong plane on a particular file, this is the
    /// formula to revisit.
    fn ifd_index(loaded: &LoadedStack, channel: usize) -> usize {
        let meta = &loaded.tiff.meta;
        loaded.frame_index * meta.slices * meta.channels + channel
    }

    fn sync_gpu(&mut self, render_state: &egui_wgpu::RenderState) {
        let Some(loaded) = &mut self.stack else { return };
        let mut renderer = render_state.renderer.write();
        let Some(resources) = renderer.callback_resources.get_mut::<ImageRenderResources>() else {
            return;
        };

        let n_channels = loaded.channel_settings.len();
        if n_channels == 0 {
            return;
        }

        if let Some(first) = loaded.tiff.frames.first() {
            resources.ensure_size(&render_state.device, first.width, first.height);
        }

        if !loaded.luts_uploaded {
            for c in 0..n_channels {
                resources.upload_lut(&render_state.queue, c, &loaded.tiff.meta.channel_display[c].lut);
            }
            loaded.luts_uploaded = true;
        }

        if loaded.last_uploaded != Some(loaded.frame_index) {
            for c in 0..n_channels {
                // RGB: every display channel is a sample plane of the *same*
                // IFD (one full-color image per frame). Otherwise each channel
                // is its own IFD (the hyperstack plane layout).
                let (ifd_idx, plane) = if loaded.rgb {
                    (loaded.frame_index * loaded.tiff.meta.slices, c)
                } else {
                    (Self::ifd_index(loaded, c), 0)
                };
                let encoding_range = loaded.channel_settings.get(c).and_then(|s| s.encoding_range);
                if let Some(frame_info) = loaded.tiff.frames.get(ifd_idx) {
                    let decoded = if loaded.rgb {
                        tiff_core::read_plane_u16(&loaded.tiff.mmap, frame_info, loaded.tiff.byte_order, encoding_range, plane)
                            .map(std::borrow::Cow::Owned)
                    } else {
                        tiff_core::read_frame_u16(&loaded.tiff.mmap, frame_info, loaded.tiff.byte_order, encoding_range)
                    };
                    match decoded {
                        Ok(pixels) => {
                            resources.upload_channel(
                                &render_state.queue,
                                c,
                                frame_info.width,
                                frame_info.height,
                                &pixels,
                            );
                        }
                        Err(e) => {
                            self.status = Some(format!("Failed to decode frame: {e:#}"));
                        }
                    }
                }
            }
            loaded.last_uploaded = Some(loaded.frame_index);
        }

        // For float channels the texture holds samples already rescaled
        // through `encoding_range` into 0..65535, so the user's contrast
        // window (in real float units) needs the same remap before it's a
        // meaningful window/level pair for the shader. Integer channels
        // pass their min/max straight through, unchanged.
        let uniforms: Vec<ChannelUniform> = loaded
            .channel_settings
            .iter()
            .map(|s| {
                let (min, max) = match s.encoding_range {
                    Some((lo, hi)) => {
                        let span = (hi - lo).max(f32::EPSILON);
                        let to_texture_space = |v: f32| ((v - lo) / span * 65535.0).clamp(0.0, 65535.0);
                        (to_texture_space(s.min), to_texture_space(s.max))
                    }
                    None => (s.min, s.max),
                };
                ChannelUniform { min, max, enabled: s.enabled }
            })
            .collect();
        resources.update_params(&render_state.queue, &uniforms, n_channels as u32);
    }
}

/// Actual pixel min/max of channel `c`'s first frame, for integer-format
/// data. Used as the auto-contrast fallback when no display range came
/// from the file's metadata.
fn first_frame_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    let pixels = tiff_core::read_frame_u16(&tiff.mmap, frame, tiff.byte_order, None).ok()?;
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
    tiff_core::frame_float_minmax(&tiff.mmap, frame, tiff.byte_order).ok()?
}

/// Resizes `meta.channel_display` to `new_channels` entries. The display
/// range is preserved per-channel where the index still exists, but the LUT
/// is always regenerated from the stack's `mode` — so collapsing a
/// mislabeled `channels=N` stack down to a single grayscale channel (see
/// `resolve_dimensions`) doesn't leave channel 0 wearing a stale composite
/// (e.g. red) LUT.
fn resize_channel_display(meta: &mut tiff_core::StackMeta, new_channels: usize) {
    let old = std::mem::take(&mut meta.channel_display);
    let mode = meta.mode;
    meta.channel_display = (0..new_channels)
        .map(|c| tiff_core::ChannelDisplay {
            lut: tiff_core::default_lut_for(mode, c),
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

/// A two-handle horizontal range slider editing `(min, max)` within the
/// inclusive track `[lo, hi]` (all in raw sample units). The handles can't
/// cross. `salt` disambiguates the interaction ids when several sliders share
/// a parent (e.g. one per channel).
fn range_slider(ui: &mut egui::Ui, salt: u64, min: &mut f32, max: &mut f32, lo: f32, hi: f32, width: f32) {
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
    ui.painter().rect_filled(sel, 2.0, visuals.selection.bg_fill);

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

/// Builds the UI-level per-channel settings (window/level, enabled,
/// float-encoding range) from `tiff.meta`'s current channel count and
/// display info.
fn build_channel_settings(tiff: &TiffStack) -> Vec<ChannelSettings> {
    (0..tiff.meta.channels.min(MAX_CHANNELS))
        .map(|c| {
            let disp = &tiff.meta.channel_display[c];
            let is_float = tiff
                .frames
                .get(c)
                .is_some_and(|f| f.sample_format == tiff_core::SampleFormat::Float && f.bits_per_sample == 32);

            if is_float {
                let data = first_frame_float_minmax(tiff, c);
                let (lo, hi) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .or(data)
                    .unwrap_or((0.0, 1.0));
                let bounds = slider_bounds((lo, hi), data);
                ChannelSettings { min: lo, max: hi, enabled: true, encoding_range: Some((lo, hi)), bounds }
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
                ChannelSettings { min, max, enabled: true, encoding_range: None, bounds }
            }
        })
        .collect()
}

/// The status note shown at the top of the window, derived from the
/// stack's current (resolved) dimensions. Shared between the initial load
/// and the manual dimension-order override so the two can't drift out of
/// sync with each other.
fn compute_status(meta: &tiff_core::StackMeta, triple_axis_warning: bool) -> Option<String> {
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

/// Applies a (possibly newly resolved) channel/slice/frame interpretation
/// to a stack: updates the metadata, rebuilds channel_display +
/// channel_settings to match the new channel count, and resets the scrub
/// position. The one place that does this, so the manual channels/frames
/// swap can't drift out of sync with `open_file` the way `self.status`
/// previously did.
fn apply_resolved_dimensions(loaded: &mut LoadedStack, resolved: tiff_core::ResolvedDimensions) {
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
    loaded.tiff.meta.mode = tiff_core::DisplayMode::Color;
    loaded.tiff.meta.channel_display = (0..planes)
        .map(|c| tiff_core::ChannelDisplay {
            lut: tiff_core::default_composite_lut(c), // 0 = red, 1 = green, 2 = blue
            range: None,
        })
        .collect();
    loaded.channel_settings = (0..planes)
        .map(|_| ChannelSettings {
            min: 0.0,
            max: 65535.0,
            enabled: true,
            bounds: (0.0, 65535.0),
            encoding_range: None,
        })
        .collect();
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Applies a manual channels/frames swap from the dimension-order
/// dropdown. Z (if any) and the triple-axis warning are carried over
/// unchanged — the swap only concerns the channels/frames roles.
fn apply_dimension_override(loaded: &mut LoadedStack, channels: usize, frames: usize) {
    let resolved = tiff_core::ResolvedDimensions {
        channels,
        slices: loaded.tiff.meta.slices,
        frames,
        triple_axis_warning: loaded.triple_axis_warning,
    };
    apply_resolved_dimensions(loaded, resolved);
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Drag-and-drop a file onto the window.
        let dropped = ui.ctx().input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.open_file(path);
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

        let toolbar_response = egui::Panel::top("toolbar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open TIFF...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("TIFF", &["tif", "tiff"])
                        .pick_file()
                    {
                        self.open_file(path);
                    }
                }
                if let Some(loaded) = &self.stack {
                    let meta = &loaded.tiff.meta;
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
            });
        });

        let panel_expanded = self.channels_panel_expanded;
        let is_playing = self.playing;
        let mut toggle_requested = false;
        let mut play_toggle_requested = false;
        let mut dimension_override: Option<(usize, usize)> = None;
        let mut scroll_step: i32 = 0;
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
                ui.add_enabled_ui(has_multiple_frames, |ui| {
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

                ui.add_enabled_ui(has_multiple_frames, |ui| {
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
                        ui.spacing_mut().slider_width = remaining.max(40.0);
                        ui.add(
                            egui::Slider::new(&mut loaded.frame_index, 0..=max_frame)
                                .show_value(false)
                                .trailing_fill(true),
                        );
                    });
                });
            });

            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowRight) {
                    loaded.frame_index = (loaded.frame_index + 1).min(loaded.tiff.meta.frames.saturating_sub(1));
                }
                if i.key_pressed(egui::Key::ArrowLeft) {
                    loaded.frame_index = loaded.frame_index.saturating_sub(1);
                }
            });

            if panel_expanded {
                // The channels-vs-time guess (and its override) is meaningless
                // for RGB, where the "channels" are fixed color planes.
                if !loaded.rgb {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Dimension order:");
                        let c = loaded.tiff.meta.channels;
                        let f = loaded.tiff.meta.frames;
                        let mut options = vec![(c, f), (f, c)];
                        options.sort_unstable();
                        options.dedup();
                        egui::ComboBox::from_id_salt("dim_override")
                            .selected_text(format!("c: {c}  f: {f}"))
                            .show_ui(ui, |ui| {
                                for (oc, of) in options {
                                    let label = format!("c: {oc}  f: {of}");
                                    if ui.selectable_label((oc, of) == (c, f), label).clicked() {
                                        dimension_override = Some((oc, of));
                                    }
                                }
                            });
                    });
                    ui.label(
                        RichText::new(
                            "Channels are guessed automatically (4 or fewer = channels, more = time); \
                             use this if that guess is wrong for this file.",
                        )
                        .small()
                        .weak(),
                    );
                }

                let calibration = loaded.tiff.meta.calibration;
                let rgb = loaded.rgb;
                if loaded.channel_settings.len() > 1 {
                    ui.separator();
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
                            let slider_w = (ui.available_width() - 120.0).max(80.0);
                            let (lo, hi) = settings.bounds;
                            range_slider(ui, c as u64, &mut settings.min, &mut settings.max, lo, hi, slider_w);
                            ui.label(RichText::new(value).small());
                        });
                    }
                } else if let Some(settings) = loaded.channel_settings.first_mut() {
                    ui.horizontal(|ui| {
                        ui.label("Contrast:");
                        let (lo, hi) = settings.bounds;
                        let width = ui.available_width().max(120.0);
                        range_slider(ui, 0, &mut settings.min, &mut settings.max, lo, hi, width);
                    });
                    ui.label(
                        RichText::new(format!(
                            "{} – {}{}",
                            format_calibrated(calibration, settings.min),
                            format_calibrated(calibration, settings.max),
                            calibration.map(|_| " (calibrated)").unwrap_or(""),
                        ))
                        .small()
                        .weak(),
                    );
                }
            }
            if let Some(status) = &current_status {
                ui.separator();
                ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)).small());
            }
            ui.add_space(4.0);
        });

        if toggle_requested {
            self.channels_panel_expanded = !self.channels_panel_expanded;
        }

        if play_toggle_requested {
            self.playing = !self.playing;
            // Start each play/pause from a clean clock so the first tick after
            // resuming doesn't jump by however long we were paused.
            self.last_play_time = None;
            self.play_accumulator = 0.0;
        }

        // Looped playback: advance by real elapsed time so the movie runs at
        // the file's `fps` (or the default) regardless of render cadence, and
        // request continuous repaints while it's running.
        if self.playing {
            if let Some(loaded) = &mut self.stack {
                let n = loaded.tiff.meta.frames.max(1);
                if n > 1 {
                    let fps = loaded.tiff.meta.fps.unwrap_or(DEFAULT_FPS).max(0.1);
                    let now = ui.input(|i| i.time);
                    if let Some(last) = self.last_play_time {
                        self.play_accumulator += (now - last) * fps;
                        if self.play_accumulator >= 1.0 {
                            let steps = self.play_accumulator.floor() as usize;
                            self.play_accumulator -= steps as f64;
                            loaded.frame_index = (loaded.frame_index + steps) % n;
                        }
                    }
                    self.last_play_time = Some(now);
                    ui.ctx().request_repaint();
                } else {
                    self.playing = false;
                }
            }
        } else {
            self.last_play_time = None;
            self.play_accumulator = 0.0;
        }

        if let Some((c, f)) = dimension_override {
            if let Some(loaded) = &mut self.stack {
                apply_dimension_override(loaded, c, f);
                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
            }
        }

        // Central panel: image always fills the available space with correct
        // aspect ratio. No overflow, no panning. The user can resize the
        // window freely — the image adapts. Zoom only controls the *window
        // size* (handled below), not the rendering here.
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(loaded) = &self.stack else {
                ui.centered_and_justified(|ui| {
                    ui.label("Drag and drop a TIFF stack here, or click \"Open TIFF...\" above.");
                });
                return;
            };
            let (Some(w), Some(h)) = (
                loaded.tiff.frames.first().map(|f| f.width),
                loaded.tiff.frames.first().map(|f| f.height),
            ) else {
                return;
            };

            let available = ui.available_size();
            let aspect = w as f32 / h as f32;
            // Fit the image inside the available area, preserving aspect.
            let fitted = if available.x / available.y.max(1.0) > aspect {
                egui::vec2(available.y * aspect, available.y)
            } else {
                egui::vec2(available.x, available.x / aspect.max(0.0001))
            };
            let padding = (available - fitted) * 0.5;

            let (panel_rect, response) = ui.allocate_exact_size(available, egui::Sense::hover());
            let response = response.on_hover_cursor(egui::CursorIcon::Crosshair);

            let image_rect = egui::Rect::from_min_size(panel_rect.min + padding, fitted);
            ui.painter().with_clip_rect(panel_rect).add(egui_wgpu::Callback::new_paint_callback(
                image_rect,
                crate::render::callback::ImagePaintCallback,
            ));

            // Plain scroll (no Ctrl) scrubs frames when hovering the image.
            // Ctrl+scroll is consumed by egui's zoom_delta() above, so
            // smooth_scroll_delta here is always zero when Ctrl is held.
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll < 0.0 {
                    scroll_step = 1;
                } else if scroll > 0.0 {
                    scroll_step = -1;
                }
            }
        });

        if scroll_step != 0 {
            if let Some(loaded) = &mut self.stack {
                let max_frame = loaded.tiff.meta.frames.saturating_sub(1);
                if scroll_step > 0 {
                    loaded.frame_index = (loaded.frame_index + 1).min(max_frame);
                } else {
                    loaded.frame_index = loaded.frame_index.saturating_sub(1);
                }
            }
        }

        if zoom_step != 0 {
            self.zoom = (self.zoom + zoom_step as f32 * ZOOM_STEP).clamp(MIN_ZOOM, MAX_ZOOM);
        }

        // Enforce aspect ratio every frame: height always follows width so the
        // window can only be resized proportionally (as if dragging diagonally),
        // plus whatever chrome (toolbar + bottom panel) currently needs. Zoom
        // overrides the width directly; manual resizing the width is fine and
        // height will snap to match. Sending InnerSize is skipped when neither
        // the width nor the chrome height changed, to avoid a resize command
        // every frame. Because the chrome height is added on top of a constant
        // image height, expanding the bottom panel makes the window taller by
        // exactly the panel's height rather than eating into the image.
        let toolbar_height = toolbar_response.response.rect.height();
        let bottom_bar_height = scrub_bar_response.response.rect.height();
        let chrome_height = toolbar_height + bottom_bar_height;
        if let Some(loaded) = &self.stack {
            if let Some(first) = loaded.tiff.frames.first() {
                let img_w = first.width as f32;
                let img_h = first.height as f32;
                let aspect = img_w / img_h.max(1.0);

                let screen_w = ui.ctx().content_rect().width();
                let target_w = if zoom_step != 0 {
                    (img_w * self.zoom).max(200.0)
                } else {
                    screen_w.max(200.0)
                };
                // Round to whole pixels so floating-point jitter in content_rect
                // doesn't cause an endless resize→repaint→resize feedback loop.
                let target_w_px = target_w.round();
                let chrome_px = chrome_height.round();
                let target_h_px = (target_w_px / aspect + chrome_px).max(200.0);

                let key = (target_w_px, chrome_px);
                if self.last_enforced != Some(key) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(target_w_px, target_h_px)));
                    self.last_enforced = Some(key);
                }
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

        if let Some(render_state) = frame.wgpu_render_state().cloned() {
            self.sync_gpu(&render_state);
        }

    }
}
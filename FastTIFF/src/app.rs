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
    /// The last window width we computed height for, to avoid sending a resize
    /// command every frame when nothing has changed.
    last_enforced_w: Option<f32>,
    /// The window title last sent via `ViewportCommand::Title`.
    last_title: Option<String>,
}

impl ViewerApp {
    pub fn new(initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            stack: None,
            status: None,
            channels_panel_expanded: false,
            zoom: 1.0,
            last_enforced_w: None,
            last_title: None,
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
                };
                let (c, z, f) = (
                    loaded.tiff.meta.channels,
                    loaded.tiff.meta.slices,
                    loaded.tiff.meta.frames,
                );
                let resolved = tiff_core::resolve_dimensions(c, z, f);
                apply_resolved_dimensions(&mut loaded, resolved);

                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
                self.stack = Some(loaded);
                // Reset to native 1:1 on every fresh load.
                self.zoom = 1.0;
                self.last_enforced_w = None;
            }
            Err(e) => {
                self.status = Some(format!("Failed to open {}: {e:#}", path.display()));
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
                let ifd_idx = Self::ifd_index(loaded, c);
                let encoding_range = loaded.channel_settings.get(c).and_then(|s| s.encoding_range);
                if let Some(frame_info) = loaded.tiff.frames.get(ifd_idx) {
                    match tiff_core::read_frame_u16(&loaded.tiff.mmap, frame_info, loaded.tiff.byte_order, encoding_range) {
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

/// Resizes `meta.channel_display` to `new_channels` entries, keeping
/// existing LUT/range entries where the index still exists and
/// synthesizing defaults for any new ones.
fn resize_channel_display(meta: &mut tiff_core::StackMeta, new_channels: usize) {
    let old = std::mem::take(&mut meta.channel_display);
    meta.channel_display = (0..new_channels)
        .map(|c| {
            old.get(c).cloned().unwrap_or_else(|| tiff_core::ChannelDisplay {
                lut: tiff_core::default_composite_lut(c),
                range: None,
            })
        })
        .collect();
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
                let (lo, hi) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .or_else(|| first_frame_float_minmax(tiff, c))
                    .unwrap_or((0.0, 1.0));
                ChannelSettings { min: lo, max: hi, enabled: true, encoding_range: Some((lo, hi)) }
            } else {
                let (min, max) = disp.range.map(|(lo, hi)| (lo as f32, hi as f32)).unwrap_or_else(|| {
                    // No display range in metadata at all (not even
                    // ImageDescription min=/max=) — fall back to the actual
                    // min/max of channel c's first frame.
                    first_frame_minmax(tiff, c).unwrap_or((0.0, 65535.0))
                });
                ChannelSettings { min, max, enabled: true, encoding_range: None }
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
    } else if !meta.ij_metadata_parsed && meta.channels > 1 {
        Some(
            "Note: couldn't parse per-channel LUTs/ranges from this file's IJMetadata block — using default colors and auto-contrast. Adjust manually below if needed.".to_string(),
        )
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
                    ui.label(format!(
                        "{}×{} px, {} channel(s)",
                        loaded.tiff.frames.first().map(|f| f.width).unwrap_or(0),
                        loaded.tiff.frames.first().map(|f| f.height).unwrap_or(0),
                        meta.channels,
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
        let mut toggle_requested = false;
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

                if loaded.channel_settings.len() > 1 {
                    ui.separator();
                    ui.horizontal(|ui| {
                        for (c, settings) in loaded.channel_settings.iter_mut().enumerate() {
                            let speed = settings
                                .encoding_range
                                .map(|(lo, hi)| (((hi - lo) as f64) / 1000.0).max(0.0001))
                                .unwrap_or(10.0);
                            ui.vertical(|ui| {
                                ui.checkbox(&mut settings.enabled, format!("Ch {}", c + 1));
                                ui.add(egui::DragValue::new(&mut settings.min).prefix("min ").speed(speed));
                                ui.add(egui::DragValue::new(&mut settings.max).prefix("max ").speed(speed));
                            });
                        }
                    });
                } else if let Some(settings) = loaded.channel_settings.first_mut() {
                    let speed = settings
                        .encoding_range
                        .map(|(lo, hi)| (((hi - lo) as f64) / 1000.0).max(0.0001))
                        .unwrap_or(10.0);
                    ui.horizontal(|ui| {
                        ui.label("Contrast:");
                        ui.add(egui::DragValue::new(&mut settings.min).prefix("min ").speed(speed));
                        ui.add(egui::DragValue::new(&mut settings.max).prefix("max ").speed(speed));
                    });
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
        // window can only be resized proportionally (as if dragging diagonally).
        // Zoom overrides the width directly; manual resizing the width is fine
        // and height will snap to match. Sending InnerSize is skipped when the
        // current width hasn't changed to avoid a resize command every frame.
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
                let target_h = (target_w / aspect + chrome_height).max(200.0);

                if self.last_enforced_w != Some(target_w) || zoom_step != 0 {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(target_w, target_h)));
                    self.last_enforced_w = Some(target_w);
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

        if self.stack.is_some() {
            ui.ctx().request_repaint();
        }
    }
}
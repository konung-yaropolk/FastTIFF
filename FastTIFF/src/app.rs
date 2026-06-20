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

#[derive(Clone, Copy)]
struct ChannelSettings {
    min: f32,
    max: f32,
    enabled: bool,
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
    /// The (channels, slices, frames) as originally parsed from the file's
    /// own metadata, before `resolve_dimensions` reinterprets them. Kept so
    /// that changing the channel-size cutoff can re-run the heuristic from
    /// scratch rather than from an already-reinterpreted (or manually
    /// swapped) state.
    raw_dimensions: (usize, usize, usize),
}

pub struct ViewerApp {
    stack: Option<LoadedStack>,
    status: Option<String>,
    /// Channel buttons + contrast sliders are tucked under a small
    /// triangle toggle to keep the bar minimal by default.
    channels_panel_expanded: bool,
    /// Bottom bar's height as of the last frame. Used to grow/shrink the
    /// native window by exactly the delta when this changes (panel
    /// toggled, a stack loading/unloading, ...) so the image canvas above
    /// it keeps a constant size instead of being squeezed or stretched by
    /// the bar's own size changes.
    last_bottom_bar_height: Option<f32>,
    /// A dimension this size or smaller is guessed to be channels; larger
    /// is guessed to be time (see `tiff_core::resolve_dimensions`). User
    /// adjustable since there's no size that's correct for every dataset —
    /// a real acquisition can genuinely have more than a handful of
    /// channels. Applies to whatever stack is currently loaded, and to any
    /// loaded afterward.
    channel_size_cutoff: usize,
}

impl ViewerApp {
    pub fn new(initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            stack: None,
            status: None,
            channels_panel_expanded: false,
            last_bottom_bar_height: None,
            channel_size_cutoff: 4,
        };
        if let Some(path) = initial_path {
            app.open_file(path);
        }
        app
    }

    fn open_file(&mut self, path: PathBuf) {
        match TiffStack::open(&path) {
            Ok(tiff) => {
                let raw_dimensions = (tiff.meta.channels, tiff.meta.slices, tiff.meta.frames);
                let mut loaded = LoadedStack {
                    tiff,
                    path,
                    channel_settings: Vec::new(),
                    frame_index: 0,
                    last_uploaded: None,
                    luts_uploaded: false,
                    triple_axis_warning: false,
                    raw_dimensions,
                };
                let (c, z, f) = raw_dimensions;
                let resolved = tiff_core::resolve_dimensions(c, z, f, self.channel_size_cutoff);
                apply_resolved_dimensions(&mut loaded, resolved);

                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
                self.stack = Some(loaded);
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
                if let Some(frame_info) = loaded.tiff.frames.get(ifd_idx) {
                    match tiff_core::read_frame_u16(&loaded.tiff.mmap, frame_info, loaded.tiff.byte_order) {
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

        let uniforms: Vec<ChannelUniform> = loaded
            .channel_settings
            .iter()
            .map(|s| ChannelUniform {
                min: s.min,
                max: s.max,
                enabled: s.enabled,
            })
            .collect();
        resources.update_params(&render_state.queue, &uniforms, n_channels as u32);
    }
}

fn first_frame_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    let pixels = tiff_core::read_frame_u16(&tiff.mmap, frame, tiff.byte_order).ok()?;
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

/// Builds the UI-level per-channel settings (window/level, enabled) from
/// `tiff.meta`'s current channel count and display info.
fn build_channel_settings(tiff: &TiffStack) -> Vec<ChannelSettings> {
    (0..tiff.meta.channels.min(MAX_CHANNELS))
        .map(|c| {
            let disp = &tiff.meta.channel_display[c];
            let (min, max) = disp.range.map(|(lo, hi)| (lo as f32, hi as f32)).unwrap_or_else(|| {
                // No display range in metadata at all (not even
                // ImageDescription min=/max=) — fall back to the actual
                // min/max of channel c's first frame.
                first_frame_minmax(tiff, c).unwrap_or((0.0, 65535.0))
            });
            ChannelSettings { min, max, enabled: true }
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
/// swap and a cutoff change can't drift out of sync with each other (or
/// with `open_file`) the way `self.status` previously did.
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

        egui::TopBottomPanel::top("toolbar").show_inside(ui, |ui| {
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
                    ui.separator();
                    let name = loaded
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let meta = &loaded.tiff.meta;
                    ui.label(format!(
                        "{name}  —  {}×{} px, {} frames, {} channel(s)",
                        loaded.tiff.frames.first().map(|f| f.width).unwrap_or(0),
                        loaded.tiff.frames.first().map(|f| f.height).unwrap_or(0),
                        meta.frames,
                        meta.channels,
                    ));
                }
            });
            if let Some(status) = &self.status {
                ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)));
            }
        });

        // Read-only snapshot for use inside the panel closure below, plus a
        // deferred-write flag set on click — same pattern as `scroll_step`
        // further down, and for the same reason: keeps the mutation of
        // `self.channels_panel_expanded` outside any closure that's also
        // borrowing `self.stack`, so there's no ambiguity about disjoint
        // field captures across nested closures.
        let panel_expanded = self.channels_panel_expanded;
        let mut toggle_requested = false;
        let mut dimension_override: Option<(usize, usize)> = None; // (channels, frames)
        let channel_size_cutoff = self.channel_size_cutoff;
        let mut cutoff_override: Option<usize> = None;

        let scrub_bar_response = egui::TopBottomPanel::bottom("scrub_bar").show_inside(ui, |ui| {
            let Some(loaded) = &mut self.stack else {
                ui.label("Open a TIFF stack to begin.");
                return;
            };
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                let max_frame = loaded.tiff.meta.frames.saturating_sub(1);

                // A blank square button with a small triangle painted on top
                // — same technique egui's own CollapsingHeader arrow uses
                // (see `egui::containers::collapsing_header::paint_default_icon`).
                // This sidesteps font glyph coverage entirely: no character
                // is drawn at all, just a filled polygon, so it renders
                // identically regardless of what fonts are installed.
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
                    vec![r.left_bottom(), r.right_bottom(), r.center_top()] // pointing up
                } else {
                    vec![r.left_top(), r.right_top(), r.center_bottom()] // pointing down
                };
                ui.painter().add(egui::Shape::convex_polygon(triangle, icon_color, egui::Stroke::NONE));

                if ui.button("|<").on_hover_text("First frame").clicked() {
                    loaded.frame_index = 0;
                }
                if ui.button("<").on_hover_text("Previous frame (←)").clicked() {
                    loaded.frame_index = loaded.frame_index.saturating_sub(1);
                }

                // Next/last buttons + frame counter are anchored to the
                // right edge of the window; the scrollbar fills whatever
                // horizontal space remains between them and the prev/first
                // buttons above. Widgets here are added right-to-left, so
                // in reverse of their final left-to-right visual order.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(interval) = loaded.tiff.meta.frame_interval_s {
                        ui.label(format!("t = {:.3}s", loaded.frame_index as f64 * interval));
                    }
                    ui.label(format!("{} / {}", loaded.frame_index + 1, loaded.tiff.meta.frames));
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
                    ); // the horizontal scrubber: drag, or click-to-jump
                });
            });

            // Keyboard scrubbing.
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
                    ui.label("Channel size cutoff:");
                    let mut cutoff = channel_size_cutoff;
                    if ui.add(egui::DragValue::new(&mut cutoff).range(1..=64).speed(0.1)).changed() {
                        cutoff_override = Some(cutoff);
                    }
                });
                ui.label(
                    RichText::new(
                        "A dimension this size or smaller is guessed to be channels; larger is guessed \
                         to be time. Raise this if a real multi-channel image is being misread as a \
                         time series — the dropdown below can also fix it for this file specifically.",
                    )
                    .small()
                    .weak(),
                );

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

                if loaded.channel_settings.len() > 1 {
                    ui.separator();
                    ui.horizontal(|ui| {
                        for (c, settings) in loaded.channel_settings.iter_mut().enumerate() {
                            ui.vertical(|ui| {
                                ui.checkbox(&mut settings.enabled, format!("Ch {}", c + 1));
                                ui.add(
                                    egui::DragValue::new(&mut settings.min)
                                        .prefix("min ")
                                        .speed(10.0),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut settings.max)
                                        .prefix("max ")
                                        .speed(10.0),
                                );
                            });
                        }
                    });
                } else if let Some(settings) = loaded.channel_settings.first_mut() {
                    ui.horizontal(|ui| {
                        ui.label("Contrast:");
                        ui.add(egui::DragValue::new(&mut settings.min).prefix("min ").speed(10.0));
                        ui.add(egui::DragValue::new(&mut settings.max).prefix("max ").speed(10.0));
                    });
                }
            }
            ui.add_space(4.0);
        });

        if toggle_requested {
            self.channels_panel_expanded = !self.channels_panel_expanded;
        }

        if let Some(new_cutoff) = cutoff_override {
            self.channel_size_cutoff = new_cutoff;
            if let Some(loaded) = &mut self.stack {
                let (c, z, f) = loaded.raw_dimensions;
                let resolved = tiff_core::resolve_dimensions(c, z, f, new_cutoff);
                apply_resolved_dimensions(loaded, resolved);
                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
            }
        }

        if let Some((c, f)) = dimension_override {
            if let Some(loaded) = &mut self.stack {
                apply_dimension_override(loaded, c, f);
                self.status = compute_status(&loaded.tiff.meta, loaded.triple_axis_warning);
            }
        }

        // Grow/shrink the window by exactly however much the bottom bar's
        // height just changed, so the rest of the layout (the image
        // canvas) keeps a constant size regardless of why the bar resized
        // (the toggle above, a stack loading/unloading). Skipped on the
        // very first frame, since there's no prior height yet to diff
        // against.
        let bottom_bar_height = scrub_bar_response.response.rect.height();
        if let Some(prev_height) = self.last_bottom_bar_height {
            let delta = bottom_bar_height - prev_height;
            if delta.abs() > 0.5 {
                if let Some(inner_rect) = ui.ctx().input(|i| i.viewport().inner_rect) {
                    let new_size = egui::vec2(inner_rect.width(), (inner_rect.height() + delta).max(200.0));
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(new_size));
                }
            }
        }
        self.last_bottom_bar_height = Some(bottom_bar_height);

        let mut scroll_step: i32 = 0;

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
            let mut fitted = available;
            if available.x / available.y.max(1.0) > aspect {
                fitted.x = available.y * aspect;
            } else {
                fitted.y = available.x / aspect.max(0.0001);
            }
            let padding_x = ((available.x - fitted.x).max(0.0)) * 0.5;
            let padding_y = ((available.y - fitted.y).max(0.0)) * 0.5;

            ui.add_space(padding_y);
            ui.horizontal(|ui| {
                ui.add_space(padding_x);
                let (rect, response) = ui.allocate_exact_size(fitted, egui::Sense::hover());
                let response = response.on_hover_cursor(egui::CursorIcon::Crosshair);
                ui.painter()
                    .add(egui_wgpu::Callback::new_paint_callback(rect, crate::render::callback::ImagePaintCallback));

                // Scroll wheel over the image scrubs frames — the fast path
                // for "speed through a huge movie" rather than dragging the
                // bottom slider pixel by pixel. We only *record* the intent
                // here; self.stack is borrowed immutably as `loaded` for
                // this whole closure, so the actual mutation happens after
                // this panel block returns.
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll < 0.0 {
                        scroll_step = 1;
                    } else if scroll > 0.0 {
                        scroll_step = -1;
                    }
                }
            });
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

        if let Some(render_state) = frame.wgpu_render_state().cloned() {
            self.sync_gpu(&render_state);
        }

        // Keep redrawing while a stack is open so drag/scroll scrubbing
        // feels immediate rather than waiting for the next input event.
        if self.stack.is_some() {
            ui.ctx().request_repaint();
        }
    }
}
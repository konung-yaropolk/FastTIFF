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
    slice_index: usize,
    last_uploaded: Option<(usize, usize)>,
    luts_uploaded: bool,
}

pub struct ViewerApp {
    stack: Option<LoadedStack>,
    status: Option<String>,
}

impl ViewerApp {
    pub fn new() -> Self {
        Self {
            stack: None,
            status: None,
        }
    }

    fn open_file(&mut self, path: PathBuf) {
        match TiffStack::open(&path) {
            Ok(tiff) => {
                let channel_settings = (0..tiff.meta.channels.min(MAX_CHANNELS))
                    .map(|c| {
                        let disp = &tiff.meta.channel_display[c];
                        let (min, max) = disp.range.map(|(lo, hi)| (lo as f32, hi as f32)).unwrap_or_else(|| {
                            // No display range in metadata at all (not even
                            // ImageDescription min=/max=) — fall back to the
                            // actual min/max of channel c's first frame.
                            first_frame_minmax(&tiff, c).unwrap_or((0.0, 65535.0))
                        });
                        ChannelSettings {
                            min,
                            max,
                            enabled: true,
                        }
                    })
                    .collect();

                if tiff.meta.channels > MAX_CHANNELS {
                    self.status = Some(format!(
                        "Note: stack has {} channels; showing the first {MAX_CHANNELS}.",
                        tiff.meta.channels
                    ));
                } else if !tiff.meta.ij_metadata_parsed && tiff.meta.channels > 1 {
                    self.status = Some(
                        "Note: couldn't parse per-channel LUTs/ranges from this file's IJMetadata block — using default colors and auto-contrast. Adjust manually below if needed.".to_string(),
                    );
                } else {
                    self.status = None;
                }

                self.stack = Some(LoadedStack {
                    tiff,
                    path,
                    channel_settings,
                    frame_index: 0,
                    slice_index: 0,
                    last_uploaded: None,
                    luts_uploaded: false,
                });
            }
            Err(e) => {
                self.status = Some(format!("Failed to open {}: {e:#}", path.display()));
            }
        }
    }

    /// Index into `tiff.frames` for (current frame, current slice, channel),
    /// assuming ImageJ's default `xyczt` plane order (channel varies
    /// fastest, then z, then t). This matches how ImageJ writes hyperstacks
    /// unless the file was produced by something that explicitly reorders
    /// planes — if scrubbing shows the wrong plane on a particular file,
    /// this is the formula to revisit.
    fn ifd_index(loaded: &LoadedStack, channel: usize) -> usize {
        let meta = &loaded.tiff.meta;
        (loaded.frame_index * meta.slices + loaded.slice_index) * meta.channels + channel
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

        let want = (loaded.frame_index, loaded.slice_index);
        if loaded.last_uploaded != Some(want) {
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
            loaded.last_uploaded = Some(want);
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
                        "{name}  —  {}×{} px, {} frames, {} channel(s), {} slice(s)",
                        loaded.tiff.frames.first().map(|f| f.width).unwrap_or(0),
                        loaded.tiff.frames.first().map(|f| f.height).unwrap_or(0),
                        meta.frames,
                        meta.channels,
                        meta.slices,
                    ));
                }
            });
            if let Some(status) = &self.status {
                ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)));
            }
        });

        egui::TopBottomPanel::bottom("scrub_bar").show_inside(ui, |ui| {
            let Some(loaded) = &mut self.stack else {
                ui.label("Open a TIFF stack to begin.");
                return;
            };
            ui.add_space(4.0);

            // Z-slice selector, only shown for true Z-stacks-over-time.
            if loaded.tiff.meta.slices > 1 {
                ui.horizontal(|ui| {
                    ui.label("Z slice:");
                    let max = loaded.tiff.meta.slices - 1;
                    ui.add(egui::Slider::new(&mut loaded.slice_index, 0..=max));
                });
            }

            ui.horizontal(|ui| {
                let max_frame = loaded.tiff.meta.frames.saturating_sub(1);
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
            ui.add_space(4.0);
        });

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
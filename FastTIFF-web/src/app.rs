//! The egui interface, mirroring the desktop app's layout without any of its
//! window management (there's no window to size on a canvas).
//!
//! Everything below the UI — the loaded stack, channel settings, contrast,
//! dimension order, playback clock, 3D camera, decode→GPU sync — comes from
//! `fast_tiff_viewer`, exactly as it does for the desktop app and the React
//! build. This file is the frontend and nothing else.

use crate::render::{self, Render};
use eframe::egui;
use egui::{Color32, RichText};
use fast_tiff_viewer::camera::{NavMode, OrbitPoint};
use fast_tiff_viewer::channels::{
    channel_tint, gray_lut_applicable, gray_lut_count, gray_lut_sel_lut, gray_lut_sel_name,
    pseudocolor_applicable, ui_tint,
};
use fast_tiff_viewer::{DecodeMode, ViewMode, Viewer};
use scivis_render::{VolumeInterp, VolumeRender};
use std::sync::mpsc::{channel, Receiver, Sender};

/// A file picked in the browser: its bytes and its name.
type Picked = (Vec<u8>, String);

pub struct WebApp {
    core: Viewer,
    render: Render,

    /// Files arrive from an async browser dialog, so the picker hands them back
    /// through a channel that `update` drains.
    tx: Sender<Picked>,
    rx: Receiver<Picked>,
    picking: bool,

    // --- view chrome --------------------------------------------------------
    channels_open: bool,
    show_settings: bool,
    show_metadata: bool,
    /// 2D zoom (image pixels per point) and pan, in points.
    zoom: f32,
    pan: egui::Vec2,
    /// Set after a load so the next frame fits the image to the canvas, once
    /// the canvas size is actually known.
    pending_fit: bool,
    scroll_accum: f32,
    error: Option<String>,

    // --- 3D input preferences ----------------------------------------------
    move_speed: f32,
    scroll_speed: f32,
}

/// Discrete zoom levels, matching the desktop app's ladder.
const ZOOM_LEVELS: [f32; 21] = [
    0.031, 0.042, 0.063, 0.083, 0.125, 0.167, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
];

/// The next zoom level in `dir` (+1 in, −1 out) from whichever is nearest.
fn stepped_zoom(current: f32, dir: i32) -> f32 {
    let nearest = ZOOM_LEVELS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (**a - current).abs().partial_cmp(&(**b - current).abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    ZOOM_LEVELS[(nearest as i32 + dir).clamp(0, ZOOM_LEVELS.len() as i32 - 1) as usize]
}

fn tint_color(t: Option<[u8; 3]>) -> Option<Color32> {
    t.map(|[r, g, b]| Color32::from_rgb(r, g, b))
}

impl WebApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Dark by default, like the desktop app: a light chrome throws stray
        // light onto the canvas and skews how dim structures read.
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        let (tx, rx) = channel();
        WebApp {
            core: Viewer::new(),
            render: render::init(cc),
            tx,
            rx,
            picking: false,
            channels_open: true,
            show_settings: false,
            show_metadata: false,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            pending_fit: false,
            scroll_accum: 0.0,
            error: None,
            move_speed: 1.0,
            scroll_speed: 1.0,
        }
    }

    /// Open the browser's file picker. It resolves on another task, so the
    /// result comes back through the channel rather than being awaited here.
    fn pick_file(&mut self, ctx: &egui::Context) {
        if self.picking {
            return;
        }
        self.picking = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(handle) = rfd::AsyncFileDialog::new()
                .add_filter("TIFF", &["tif", "tiff"])
                .pick_file()
                .await
            {
                let name = handle.file_name();
                let bytes = handle.read().await;
                let _ = tx.send((bytes, name));
            }
            // Wake the UI whether or not a file was chosen, so `picking` clears.
            ctx.request_repaint();
        });
    }

    /// Take whatever the picker produced since the last frame.
    fn drain_picked(&mut self) {
        while let Ok((bytes, name)) = self.rx.try_recv() {
            self.picking = false;
            match self.core.load_bytes(bytes, std::path::PathBuf::from(&name)) {
                Ok(()) => {
                    self.error = None;
                    self.zoom = 1.0;
                    self.pan = egui::Vec2::ZERO;
                    self.pending_fit = true;
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            }
        }
    }

    /// Drive the 3D camera from this frame's pointer/keyboard.
    fn drive_camera(&mut self, ui: &egui::Ui, response: &egui::Response, rect: egui::Rect) -> bool {
        let mut animating = false;
        let hovered = ui.rect_contains_pointer(rect);
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let pan_speed = self.core.volume.cam.pan_speed(rect.height());
        let (_, right, up) = self.core.volume.cam.basis();
        let (move_speed, scroll_speed) = (self.move_speed, self.scroll_speed);

        let (wheel, wasd, space, shift, arrows) = ui.input(|i| {
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
            (
                wheel,
                [
                    i.key_down(egui::Key::A),
                    i.key_down(egui::Key::D),
                    i.key_down(egui::Key::W),
                    i.key_down(egui::Key::S),
                ],
                i.key_down(egui::Key::Space),
                i.modifiers.shift,
                [
                    i.key_down(egui::Key::ArrowLeft),
                    i.key_down(egui::Key::ArrowRight),
                    i.key_down(egui::Key::ArrowUp),
                    i.key_down(egui::Key::ArrowDown),
                ],
            )
        });

        let d = response.drag_delta();
        let moved = d != egui::Vec2::ZERO;
        let cam = &mut self.core.volume.cam;

        if response.drag_started_by(egui::PointerButton::Primary) {
            cam.repivot();
        }
        if moved {
            if response.dragged_by(egui::PointerButton::Primary) {
                cam.orbit_drag(d.x, d.y);
                animating = true;
            } else if response.dragged_by(egui::PointerButton::Middle)
                || response.dragged_by(egui::PointerButton::Secondary)
            {
                cam.pan(d.x, d.y, right, up, pan_speed);
                animating = true;
            }
        }

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
            if cam.nav.has_vertical_keys() {
                if space {
                    mv[1] += 1.0;
                }
                if shift {
                    mv[1] -= 1.0;
                }
            }
            if mv != [0.0; 3] {
                cam.translate(mv, fast_tiff_viewer::camera::FLY_UNITS_PER_SEC * dt * move_speed);
                animating = true;
            }

            let k = fast_tiff_viewer::camera::KEY_ROT;
            let mut rot = egui::Vec2::ZERO;
            if arrows[0] {
                rot.x -= k;
            }
            if arrows[1] {
                rot.x += k;
            }
            if arrows[2] {
                rot.y -= k;
            }
            if arrows[3] {
                rot.y += k;
            }
            if rot != egui::Vec2::ZERO {
                cam.key_rotate(rot.x, rot.y);
                animating = true;
            }
        }

        if wheel.abs() > 0.01 {
            cam.wheel_fly(wheel, scroll_speed);
            animating = true;
        }
        animating
    }
}

impl eframe::App for WebApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain_picked();

        // Dropping a file onto the canvas: the browser hands eframe the bytes
        // directly, since there's no path to read.
        let dropped: Vec<Picked> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.bytes.as_ref().map(|b| (b.to_vec(), f.name.clone())))
                .collect()
        });
        for (bytes, name) in dropped {
            let _ = self.tx.send((bytes, name));
        }
        self.drain_picked();

        let has_stack = self.core.stack.is_some();
        let mut open_requested = false;

        // --- toolbar --------------------------------------------------------
        egui::Panel::top("toolbar").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open TIFF…").clicked() {
                    open_requested = true;
                }
                let Some(stack) = &self.core.stack else {
                    ui.separator();
                    ui.label(RichText::new("FastTIFF for the web — egui + WebGPU").weak());
                    return;
                };
                let meta = &stack.tiff.meta;
                let can_3d = self.core.can_show_volume();
                let is_4d = self.core.is_4d();
                let in_3d = self.core.view_mode == ViewMode::Volume;

                ui.separator();
                ui.add_enabled_ui(can_3d, |ui| {
                    if ui.selectable_label(!in_3d, "2D").clicked() {
                        self.core.view_mode = ViewMode::Movie;
                    }
                    if ui.selectable_label(in_3d, "3D").clicked() {
                        self.core.view_mode = ViewMode::Volume;
                        if !is_4d {
                            self.core.playback.playing = false;
                        }
                    }
                });
                ui.separator();
                ui.add_enabled_ui(can_3d, |ui| {
                    if ui.button(RichText::new("⚙").size(16.0)).on_hover_text("3D render settings").clicked() {
                        self.show_settings = !self.show_settings;
                    }
                });

                if !in_3d {
                    ui.separator();
                    let pct = format!("{:.2}", self.zoom * 100.0);
                    let pct = pct.trim_end_matches('0').trim_end_matches('.');
                    if ui
                        .button(RichText::new(format!("{pct}%")).monospace())
                        .on_hover_text("Fit to window (Ctrl+scroll to zoom)")
                        .clicked()
                    {
                        self.pending_fit = true;
                    }
                }

                ui.separator();
                let (w, h) = stack.dimensions().unwrap_or((0, 0));
                let bits = stack.tiff.frames.first().map(|f| f.bits_per_sample).unwrap_or(0);
                let chans = if stack.rgb { "RGB".into() } else { format!("{} channel(s)", meta.channels) };
                ui.label(format!("{w}×{h} px, {bits}-bit, {chans}"));

                // In 3D the frame axis is the volume's depth, so the counter
                // only means something for a stack with a separate time axis.
                if !(in_3d && !is_4d) {
                    ui.separator();
                    ui.label(
                        RichText::new(format!("Frame {} / {}", stack.frame_index + 1, meta.frames)).monospace(),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("( i )").on_hover_text("See metadata").clicked() {
                        self.show_metadata = !self.show_metadata;
                    }
                });
            });
        });

        // --- bottom bar -----------------------------------------------------
        egui::Panel::bottom("scrub").show_inside(ui, |ui| {
            let in_3d = self.core.view_mode == ViewMode::Volume;
            let is_4d = self.core.is_4d();
            let Some(stack) = &mut self.core.stack else {
                ui.label("Open a TIFF stack to begin.");
                return;
            };
            let n = stack.frame_count();
            let max_frame = n.saturating_sub(1);
            let nav = n > 1 && !(in_3d && !is_4d);

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let arrow = if self.channels_open { "⏷" } else { "⏵" };
                if ui.button(arrow).on_hover_text("Show/hide channel & contrast settings").clicked() {
                    self.channels_open = !self.channels_open;
                }
                ui.add_enabled_ui(nav, |ui| {
                    let label = if self.core.playback.playing { "❚❚" } else { "▶" };
                    if ui.button(label).on_hover_text("Play/pause").clicked() {
                        self.core.playback.playing = !self.core.playback.playing;
                        self.core.playback.restart();
                    }
                    if ui.button("|◀").clicked() {
                        stack.frame_index = 0;
                    }
                    if ui.button("◀").clicked() {
                        stack.frame_index = stack.frame_index.saturating_sub(1);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("▶|").clicked() {
                            stack.frame_index = max_frame;
                        }
                        if ui.button("▶").clicked() {
                            stack.frame_index = (stack.frame_index + 1).min(max_frame);
                        }
                        let mut fps = self.core.playback.fps;
                        if ui
                            .add(egui::DragValue::new(&mut fps).speed(0.5).range(0.1..=1000.0).suffix(" fps"))
                            .changed()
                        {
                            self.core.playback.fps = fps;
                        }
                        let remaining = ui.available_width();
                        ui.spacing_mut().slider_width = remaining.max(40.0);
                        if max_frame > 0 {
                            ui.add(
                                egui::Slider::new(&mut stack.frame_index, 0..=max_frame)
                                    .show_value(false)
                                    .trailing_fill(true),
                            );
                        }
                    });
                });
            });

            if self.channels_open {
                ui.separator();
                channel_controls(ui, &mut self.core);
            }
            if let Some(status) = &self.core.status {
                ui.separator();
                ui.label(RichText::new(status).color(Color32::from_rgb(230, 170, 60)).small());
            }
            if let Some(err) = &self.error {
                ui.separator();
                ui.label(RichText::new(err).color(Color32::from_rgb(230, 110, 110)).small());
            }
            ui.add_space(4.0);
        });

        // --- pop-ups ---------------------------------------------------------
        if self.show_settings {
            volume_settings(
                &ctx,
                &mut self.show_settings,
                &mut self.core,
                &mut self.move_speed,
                &mut self.scroll_speed,
            );
        }
        if self.show_metadata {
            metadata_window(&ctx, &mut self.show_metadata, &self.core);
        }

        // --- canvas -----------------------------------------------------------
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(egui::Margin::ZERO))
            .show_inside(ui, |ui| {
                if !has_stack {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            "Drop a TIFF here, or click \"Open TIFF…\" above.\n\n\
                             Scroll — frames · Shift+scroll — fast · Ctrl+scroll — zoom\n\
                             Files are decoded in your browser and never uploaded.",
                        );
                    });
                    return;
                }
                let avail = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

                if self.core.view_mode == ViewMode::Volume {
                    self.core.volume.aspect = (rect.width() / rect.height().max(1.0)).clamp(0.1, 10.0);
                    if self.core.volume.built_frame.is_none() {
                        ui.painter().rect_filled(rect, 0.0, Color32::BLACK);
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "Building the volume…",
                            egui::FontId::proportional(16.0),
                            Color32::from_gray(150),
                        );
                        ctx.request_repaint();
                        return;
                    }
                    let animating = self.drive_camera(ui, &response, rect);
                    ui.painter()
                        .with_clip_rect(rect)
                        .add(render::paint_volume_callback(&self.render, rect));
                    if animating {
                        ctx.request_repaint();
                    }
                    return;
                }

                self.paint_2d(ui, &response, rect);
            });

        // Playback advances on the core's clock; ask for the next frame at the
        // playback rate rather than spinning.
        if self.core.playback.playing {
            self.core.tick_playback(ctx.input(|i| i.time));
            let fps = self.core.playback.fps.max(0.1);
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(1.0 / fps));
        }

        // Push everything to the GPU, then repaint again if a volume build is
        // still in flight.
        let outcome = match self.render.lock() {
            Ok(mut r) => self.core.sync(&mut r),
            Err(_) => Default::default(),
        };
        if outcome.needs_repaint {
            ctx.request_repaint();
        }

        if open_requested {
            self.pick_file(&ctx);
        }
    }
}

impl WebApp {
    /// Lay out and draw the 2D image: fit-to-canvas on load, then zoom/pan via
    /// the shader's UV sub-rect, letterboxed so the aspect ratio is preserved.
    fn paint_2d(&mut self, ui: &egui::Ui, response: &egui::Response, rect: egui::Rect) {
        let Some(stack) = &self.core.stack else { return };
        let Some((w, h)) = stack.dimensions() else { return };
        let (fw, fh) = (w as f32, h as f32);

        if self.pending_fit {
            let fit = (rect.width() / fw).min(rect.height() / fh);
            // Never open above 1:1 — matches the desktop's initial fit.
            self.zoom = fit.min(1.0);
            self.pan = egui::Vec2::ZERO;
            self.pending_fit = false;
        }

        // Ctrl+scroll zooms about the cursor; plain scroll scrubs frames.
        if ui.rect_contains_pointer(rect) {
            let (zoom_delta, scroll, shift) = ui.input(|i| {
                (i.zoom_delta(), i.smooth_scroll_delta.y, i.modifiers.shift)
            });
            let step = if zoom_delta > 1.05 { 1 } else if zoom_delta < 0.95 { -1 } else { 0 };
            if step != 0 {
                self.zoom = stepped_zoom(self.zoom, step);
            } else if scroll != 0.0 {
                let frames = stack.frame_count();
                let jump = if shift { (frames as f32 * 0.1).round().max(1.0) as i64 } else { 1 };
                self.scroll_accum -= scroll / 50.0;
                let whole = self.scroll_accum.trunc();
                self.scroll_accum -= whole;
                if whole != 0.0 {
                    if let Some(s) = &mut self.core.stack {
                        let max = s.frame_count().saturating_sub(1) as i64;
                        s.frame_index =
                            (s.frame_index as i64 + whole as i64 * jump).clamp(0, max) as usize;
                    }
                }
            }
        }

        let img = egui::vec2(fw * self.zoom, fh * self.zoom);
        let overflow = egui::vec2(
            (img.x - rect.width()).max(0.0),
            (img.y - rect.height()).max(0.0),
        );
        if (overflow.x > 0.0 || overflow.y > 0.0) && response.dragged() {
            self.pan -= response.drag_delta();
        }
        self.pan.x = self.pan.x.clamp(0.0, overflow.x);
        self.pan.y = self.pan.y.clamp(0.0, overflow.y);

        let origin = egui::pos2(
            if overflow.x > 0.0 { rect.min.x - self.pan.x } else { rect.min.x + (rect.width() - img.x) * 0.5 },
            if overflow.y > 0.0 { rect.min.y - self.pan.y } else { rect.min.y + (rect.height() - img.y) * 0.5 },
        );
        let visible = egui::Rect::from_min_size(origin, img).intersect(rect);
        if visible.is_positive() {
            let inv = egui::vec2(1.0 / img.x.max(1.0), 1.0 / img.y.max(1.0));
            self.core.uv_offset = ((visible.min - origin) * inv).into();
            self.core.uv_scale = (visible.size() * inv).into();
            ui.painter()
                .with_clip_rect(rect)
                .add(render::paint_callback(&self.render, visible));
        }
    }
}

/// Per-channel contrast plus the stack-wide display controls.
fn channel_controls(ui: &mut egui::Ui, core: &mut Viewer) {
    let apply_pseudocolor = core.apply_pseudocolor;
    let mut pseudocolor_toggle = None;
    let mut lut_change = None;
    let mut dim_change = None;
    let mut decode_mode = core.decode_mode;

    let Some(stack) = &core.stack else { return };
    let (c, z, f) = (stack.tiff.meta.channels, stack.tiff.meta.slices, stack.tiff.meta.frames);

    ui.horizontal_wrapped(|ui| {
        if !stack.rgb {
            let show_z = stack.has_z_axis;
            let mut options: Vec<(usize, usize, usize)> = if show_z {
                vec![(c, z, f), (c, f, z), (z, c, f), (z, f, c), (f, c, z), (f, z, c)]
            } else {
                vec![(c, z, f), (f, z, c)]
            };
            options.sort_unstable();
            options.dedup();
            let label = |oc: usize, oz: usize, of: usize| {
                if show_z { format!("c: {oc}  z: {oz}  t: {of}") } else { format!("c: {oc}  t: {of}") }
            };
            ui.label("Dimension order:");
            egui::ComboBox::from_id_salt("dims")
                .selected_text(label(c, z, f))
                .show_ui(ui, |ui| {
                    for (oc, oz, of) in options {
                        if ui.selectable_label((oc, oz, of) == (c, z, f), label(oc, oz, of)).clicked() {
                            dim_change = Some((oc, oz, of));
                        }
                    }
                });

            if pseudocolor_applicable(stack) {
                ui.separator();
                let mut on = apply_pseudocolor;
                if ui.checkbox(&mut on, "Apply pseudocolor").changed() {
                    pseudocolor_toggle = Some(on);
                }
            }
        }

        if gray_lut_applicable(stack) {
            ui.separator();
            ui.label("LUT:");
            let sel = stack.gray_lut_sel;
            egui::ComboBox::from_id_salt("lut")
                .selected_text(gray_lut_sel_name(stack, sel))
                .show_ui(ui, |ui| {
                    for opt in 0..gray_lut_count(stack) {
                        let name = gray_lut_sel_name(stack, opt);
                        let text = match tint_color(ui_tint(&gray_lut_sel_lut(stack, opt))) {
                            Some(col) => RichText::new(name).color(col),
                            None => RichText::new(name),
                        };
                        if ui.selectable_label(opt == sel, text).clicked() {
                            lut_change = Some(opt);
                        }
                    }
                });
        }

        ui.separator();
        ui.label("Decode:");
        egui::ComboBox::from_id_salt("decode")
            .selected_text(decode_mode.label())
            .show_ui(ui, |ui| {
                for m in [DecodeMode::Auto, DecodeMode::Serial, DecodeMode::Threaded] {
                    ui.selectable_value(&mut decode_mode, m, m.label());
                }
            });
    });

    // Contrast sliders. A palette channel's window is a fixed index→LUT
    // identity, so it has nothing to adjust.
    let palette = stack.palette;
    let rgb = stack.rgb;
    let tints: Vec<Option<Color32>> = stack
        .tiff
        .meta
        .channel_display
        .iter()
        .map(|cd| tint_color(channel_tint(&cd.lut)))
        .collect();

    if !palette {
        let Some(stack) = &mut core.stack else { return };
        ui.separator();
        for (i, s) in stack.channel_settings.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let label = if rgb {
                    ["R", "G", "B", "A"].get(i).map(|s| s.to_string()).unwrap_or_else(|| format!("S{}", i + 1))
                } else {
                    format!("Ch {}", i + 1)
                };
                ui.add_sized(egui::vec2(52.0, 18.0), egui::Checkbox::new(&mut s.enabled, label));
                let (lo, hi) = s.bounds;
                if let Some(col) = tints.get(i).copied().flatten() {
                    ui.visuals_mut().selection.bg_fill = col;
                }
                let w = (ui.available_width() - 40.0).max(80.0) * 0.5;
                ui.spacing_mut().slider_width = w;
                ui.add(egui::Slider::new(&mut s.min, lo..=hi).show_value(false));
                ui.add(egui::Slider::new(&mut s.max, lo..=hi).show_value(false));
                if s.min > s.max {
                    s.min = s.max;
                }
                ui.label(RichText::new(format!("{:.0} – {:.0}", s.min, s.max)).small());
            });
        }
    }

    if let Some(on) = pseudocolor_toggle {
        core.set_pseudocolor(on);
    }
    if let Some((oc, oz, of)) = dim_change {
        core.set_dimension_order(oc, oz, of);
    }
    if let Some(sel) = lut_change {
        if let Some(stack) = &mut core.stack {
            stack.gray_lut_sel = sel;
            let lut = gray_lut_sel_lut(stack, sel);
            if let Some(disp) = stack.tiff.meta.channel_display.first_mut() {
                disp.lut = lut;
            }
            stack.luts_uploaded = false;
        }
    }
    core.decode_mode = decode_mode;
}

/// The 3D render-settings pop-up.
fn volume_settings(
    ctx: &egui::Context,
    open: &mut bool,
    core: &mut Viewer,
    move_speed: &mut f32,
    scroll_speed: &mut f32,
) {
    egui::Window::new("3D render settings").open(open).resizable(false).show(ctx, |ui| {
        let v = &mut core.volume;
        ui.horizontal(|ui| {
            ui.label("Mode:");
            for (m, name) in [
                (VolumeRender::Mip, "MIP"),
                (VolumeRender::Alpha, "Alpha"),
                (VolumeRender::Surface, "Surface"),
            ] {
                ui.selectable_value(&mut v.render, m, name);
            }
        });
        // Density only drives alpha DVR, iso only the isosurface — hiding the
        // inapplicable one keeps a dead slider off the panel.
        match v.render {
            VolumeRender::Alpha => {
                ui.add(egui::Slider::new(&mut v.density, 1.0..=400.0).text("Density"));
            }
            VolumeRender::Surface => {
                ui.add(egui::Slider::new(&mut v.iso, 0.0..=1.0).text("Threshold"));
            }
            VolumeRender::Mip => {}
        }
        ui.horizontal(|ui| {
            ui.label("Interpolation:");
            for (m, name) in [
                (VolumeInterp::Nearest, "Nearest"),
                (VolumeInterp::Linear, "Linear"),
                (VolumeInterp::Cubic, "Cubic"),
            ] {
                ui.selectable_value(&mut v.interp, m, name);
            }
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Navigation:");
            let prev_fly = v.cam.nav.is_fly();
            egui::ComboBox::from_id_salt("nav")
                .selected_text(v.cam.nav.label())
                .show_ui(ui, |ui| {
                    for m in [NavMode::Cad, NavMode::Blender, NavMode::Maya, NavMode::WasdFly] {
                        ui.selectable_value(&mut v.cam.nav, m, m.label());
                    }
                });
            if v.cam.nav.is_fly() != prev_fly {
                v.cam.sync_for_nav(prev_fly);
            }
        });
        ui.label(RichText::new(v.cam.nav.help()).small().weak());
        ui.horizontal(|ui| {
            ui.label("Orbit around:");
            ui.selectable_value(&mut v.cam.orbit_point, OrbitPoint::VolumeCenter, "Volume center");
            ui.selectable_value(&mut v.cam.orbit_point, OrbitPoint::ScreenCenter, "Screen center");
        });
        ui.add(egui::Slider::new(move_speed, 0.1..=10.0).text("Move speed").logarithmic(true));
        ui.add(egui::Slider::new(scroll_speed, 0.1..=10.0).text("Scroll speed").logarithmic(true));
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Voxel scale:");
            for (i, axis) in ["x", "y", "z"].iter().enumerate() {
                ui.add(egui::DragValue::new(&mut v.scale[i]).speed(0.01).range(0.01..=100.0).prefix(*axis));
            }
        });
        if ui.button("Reset position").clicked() {
            v.cam.reset();
        }
    });
}

/// The file-metadata pop-up.
fn metadata_window(ctx: &egui::Context, open: &mut bool, core: &Viewer) {
    let Some(stack) = &core.stack else {
        *open = false;
        return;
    };
    egui::Window::new("File metadata").open(open).default_width(420.0).show(ctx, |ui| {
        let meta = &stack.tiff.meta;
        let (w, h) = stack.dimensions().unwrap_or((0, 0));
        egui::Grid::new("kv").num_columns(2).striped(true).show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(RichText::new(k).weak());
                ui.label(v);
                ui.end_row();
            };
            row("File", stack.path.display().to_string());
            row("Dimensions", format!("{w} × {h} px"));
            row("Channels", meta.channels.to_string());
            row("Z-slices", meta.slices.to_string());
            row("Frames", meta.frames.to_string());
            if let Some(fps) = meta.fps {
                row("fps", fps.to_string());
            }
        });
        if let Some(desc) = &stack.tiff.description {
            ui.separator();
            ui.label(RichText::new("ImageDescription").strong());
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                ui.label(RichText::new(desc).small().monospace());
            });
        }
    });
}

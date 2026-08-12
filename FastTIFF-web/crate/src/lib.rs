//! wasm-bindgen bindings that put [`fast_tiff_viewer`] behind a browser canvas.
//!
//! This is the browser's counterpart to `FastTIFF/src/render.rs` — the desktop
//! app's ~200-line eframe adapter. Both do the same three jobs (get GPU handles
//! from the host, own the render pass, translate input), over the identical
//! core. Nothing about stacks, channels, contrast, dimension order, playback or
//! the 3D camera is reimplemented here; all of it comes from the shared crates.
//!
//! The React layer never touches Rust types directly. It calls the methods
//! below and reads plain JS objects, so the UI can be rewritten without
//! touching this file.

use fast_tiff_viewer::camera::NavMode;
use fast_tiff_viewer::channels::{
    channel_tint, gray_lut_applicable, gray_lut_count, gray_lut_sel_lut, gray_lut_sel_name,
    pseudocolor_applicable,
};
use fast_tiff_viewer::{DecodeMode, Renderer, ViewMode, Viewer as CoreViewer};
use scivis_render::{VolumeInterp, VolumeRender};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// Install panic + log forwarding so a Rust panic shows up in the browser
/// console instead of an opaque `unreachable executed`.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);
}

// ---------------------------------------------------------------- JS shapes

/// Everything the UI needs to lay itself out after a stack loads. Mirrors what
/// the desktop toolbar and channel panel read off the core.
#[derive(Serialize)]
pub struct StackInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub bits: u16,
    pub frames: usize,
    pub channels: usize,
    pub slices: usize,
    pub rgb: bool,
    pub palette: bool,
    /// True when the stack has enough depth for the 3D view (≥ 2 frames).
    pub can_volume: bool,
    /// True when there's a time axis *separate* from the volume's depth, so
    /// playback still means something in 3D.
    pub is_4d: bool,
    pub fps: f64,
    /// A note about the stack's shape worth surfacing (mislabeled channel
    /// counts, the frozen-Z warning), or `null`.
    pub status: Option<String>,
    /// Physical frame interval in seconds, when the file records one.
    pub frame_interval_s: Option<f64>,
    pub channel_settings: Vec<ChannelInfo>,
    /// Whether the pseudocolor toggle applies to this stack.
    pub pseudocolor_applicable: bool,
    /// Whether the single-channel LUT selector applies, and its options.
    pub lut_selector: Option<LutSelector>,
    /// The dimension-order options to offer, as `[channels, slices, frames]`.
    pub dimension_options: Vec<[usize; 3]>,
    pub has_z_axis: bool,
}

/// One channel's contrast window and slider track.
#[derive(Serialize)]
pub struct ChannelInfo {
    pub index: usize,
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub lo: f32,
    pub hi: f32,
    pub enabled: bool,
    /// `#rrggbb` tint for the slider, or `null` for a plain grayscale channel.
    pub tint: Option<String>,
}

#[derive(Serialize)]
pub struct LutSelector {
    pub selected: usize,
    pub options: Vec<String>,
}

fn hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

// ------------------------------------------------------------------ the app

/// The viewer, bound to one canvas.
#[wasm_bindgen]
pub struct FastTiffViewer {
    core: CoreViewer,
    renderer: Renderer,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    name: String,
    /// Whether the last `render` wants another frame soon (a volume build is in
    /// flight). The JS animation loop reads this.
    needs_repaint: bool,
    /// 2D zoom (1.0 = one image pixel per device pixel) and pan, in device
    /// pixels. `render` turns these into a letterbox viewport plus a UV
    /// sub-rect, exactly as the desktop app does.
    zoom: f32,
    pan: [f32; 2],
}

#[wasm_bindgen]
impl FastTiffViewer {
    /// Create a device + swapchain for `canvas` and build the renderer.
    ///
    /// Async because WebGPU adapter/device requests are promises. Falls back to
    /// WebGL2 automatically: the `webgl` feature is on, so wgpu picks WebGPU
    /// when the browser has it and WebGL2 otherwise.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<FastTiffViewer, JsError> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsError::new(&format!("could not create a GPU surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| JsError::new(&format!("no suitable GPU adapter: {e}")))?;

        // Ask for exactly the adapter's limits, as the desktop app does — WebGL2
        // reports far lower ceilings than wgpu's defaults assume, and requesting
        // the defaults would fail outright on that backend.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fasttiff device"),
                required_features: scivis_render::wgpu_backend::optional_features(&adapter),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| JsError::new(&format!("could not create a GPU device: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let renderer = Renderer::new(device.clone(), queue.clone(), format);

        Ok(FastTiffViewer {
            core: CoreViewer::new(),
            renderer,
            surface,
            device,
            queue,
            config,
            name: String::new(),
            needs_repaint: false,
            zoom: 1.0,
            pan: [0.0, 0.0],
        })
    }

    /// Load a stack from bytes (a `File` read in JS). Returns the `StackInfo`
    /// the UI lays itself out from.
    pub fn load(&mut self, bytes: Vec<u8>, name: String) -> Result<JsValue, JsError> {
        self.core
            .load_bytes(bytes, std::path::PathBuf::from(&name))
            .map_err(|e| JsError::new(&format!("{e:#}")))?;
        self.name = name;
        self.info()
    }

    /// The current `StackInfo`, or `null` when nothing is loaded.
    pub fn info(&self) -> Result<JsValue, JsError> {
        let Some(stack) = &self.core.stack else {
            return Ok(JsValue::NULL);
        };
        let (width, height) = stack.dimensions().unwrap_or((0, 0));
        let meta = &stack.tiff.meta;
        let f0 = stack.tiff.frames.first();

        let channel_settings = stack
            .channel_settings
            .iter()
            .enumerate()
            .map(|(i, s)| ChannelInfo {
                index: i,
                label: if stack.rgb {
                    ["R", "G", "B", "A"]
                        .get(i)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("S{}", i + 1))
                } else {
                    format!("Ch {}", i + 1)
                },
                min: s.min,
                max: s.max,
                lo: s.bounds.0,
                hi: s.bounds.1,
                enabled: s.enabled,
                tint: meta
                    .channel_display
                    .get(i)
                    .and_then(|cd| channel_tint(&cd.lut))
                    .map(hex),
            })
            .collect();

        let lut_selector = gray_lut_applicable(stack).then(|| LutSelector {
            selected: stack.gray_lut_sel,
            options: (0..gray_lut_count(stack))
                .map(|o| gray_lut_sel_name(stack, o).to_string())
                .collect(),
        });

        // Same option set the desktop dropdown offers: every assignment of the
        // three counts to the three roles when the file has a real Z axis, else
        // just the channels/time swap.
        let (c, z, f) = (meta.channels, meta.slices, meta.frames);
        let mut dimension_options: Vec<[usize; 3]> = if stack.has_z_axis {
            vec![[c, z, f], [c, f, z], [z, c, f], [z, f, c], [f, c, z], [f, z, c]]
        } else {
            vec![[c, z, f], [f, z, c]]
        };
        dimension_options.sort_unstable();
        dimension_options.dedup();

        let info = StackInfo {
            name: self.name.clone(),
            width,
            height,
            bits: f0.map(|f| f.bits_per_sample).unwrap_or(0),
            frames: meta.frames,
            channels: meta.channels,
            slices: meta.slices,
            rgb: stack.rgb,
            palette: stack.palette,
            can_volume: self.core.can_show_volume(),
            is_4d: self.core.is_4d(),
            fps: self.core.playback.fps,
            status: self.core.status.clone(),
            frame_interval_s: meta.frame_interval_s,
            channel_settings,
            pseudocolor_applicable: pseudocolor_applicable(stack),
            lut_selector,
            dimension_options,
            has_z_axis: stack.has_z_axis,
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// The file's raw `ImageDescription`, for a metadata panel.
    pub fn description(&self) -> Option<String> {
        self.core.stack.as_ref().and_then(|s| s.tiff.description.clone())
    }

    // -------------------------------------------------------------- 2D view

    #[wasm_bindgen(js_name = frameIndex)]
    pub fn frame_index(&self) -> usize {
        self.core.stack.as_ref().map(|s| s.frame_index).unwrap_or(0)
    }

    #[wasm_bindgen(js_name = setFrame)]
    pub fn set_frame(&mut self, index: usize) {
        if let Some(stack) = &mut self.core.stack {
            stack.frame_index = index.min(stack.frame_count().saturating_sub(1));
        }
    }

    /// Step by `delta` frames, clamped (the wheel-scrub path).
    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, delta: i32) {
        if let Some(stack) = &mut self.core.stack {
            let max = stack.frame_count().saturating_sub(1) as i64;
            stack.frame_index = (stack.frame_index as i64 + delta as i64).clamp(0, max) as usize;
        }
    }

    /// Set one channel's contrast window and on/off state.
    #[wasm_bindgen(js_name = setChannel)]
    pub fn set_channel(&mut self, index: usize, min: f32, max: f32, enabled: bool) {
        if let Some(stack) = &mut self.core.stack {
            if let Some(s) = stack.channel_settings.get_mut(index) {
                s.min = min.clamp(s.bounds.0, s.bounds.1);
                s.max = max.clamp(s.bounds.0, s.bounds.1);
                if s.min > s.max {
                    s.min = s.max;
                }
                s.enabled = enabled;
            }
        }
    }

    /// 2D zoom and pan. `zoom` is image pixels per device pixel (1.0 = 1:1);
    /// `pan` is the scroll offset in device pixels, only meaningful once the
    /// zoomed image overflows the canvas. `render` clamps the pan and derives
    /// the viewport + UVs from these.
    #[wasm_bindgen(js_name = setZoomPan)]
    pub fn set_zoom_pan(&mut self, zoom: f32, pan_x: f32, pan_y: f32) {
        self.zoom = zoom.clamp(0.02, 64.0);
        self.pan = [pan_x, pan_y];
    }

    /// The zoom actually in effect (after clamping), for the UI's readout.
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// The zoom at which the whole image just fits the canvas — what the UI
    /// should start at, and what a "fit" button applies.
    #[wasm_bindgen(js_name = fitZoom)]
    pub fn fit_zoom(&self) -> f32 {
        match self.core.stack.as_ref().and_then(|s| s.dimensions()) {
            Some((w, h)) if w > 0 && h > 0 => {
                let sx = self.config.width as f32 / w as f32;
                let sy = self.config.height as f32 / h as f32;
                sx.min(sy)
            }
            _ => 1.0,
        }
    }

    /// The pan range in device pixels, `[max_x, max_y]` — zero on an axis where
    /// the image fits. The UI uses it to clamp its own drag state.
    #[wasm_bindgen(js_name = panRange)]
    pub fn pan_range(&self) -> Vec<f32> {
        let (ox, oy) = self.overflow();
        vec![ox, oy]
    }

    #[wasm_bindgen(js_name = setPseudocolor)]
    pub fn set_pseudocolor(&mut self, on: bool) {
        self.core.set_pseudocolor(on);
    }

    /// Pick the single-channel LUT by selector index (see `lut_selector`).
    #[wasm_bindgen(js_name = setLut)]
    pub fn set_lut(&mut self, sel: usize) {
        if let Some(stack) = &mut self.core.stack {
            stack.gray_lut_sel = sel;
            let lut = gray_lut_sel_lut(stack, sel);
            if let Some(disp) = stack.tiff.meta.channel_display.first_mut() {
                disp.lut = lut;
            }
            stack.luts_uploaded = false; // force a re-upload on the next sync
        }
    }

    /// Reassign which axis counts mean channels / Z / time.
    #[wasm_bindgen(js_name = setDimensionOrder)]
    pub fn set_dimension_order(&mut self, channels: usize, slices: usize, frames: usize) {
        self.core.set_dimension_order(channels, slices, frames);
    }

    // ------------------------------------------------------------- playback

    #[wasm_bindgen(js_name = setPlaying)]
    pub fn set_playing(&mut self, playing: bool) {
        self.core.playback.playing = playing;
        self.core.playback.restart();
    }

    #[wasm_bindgen(js_name = isPlaying)]
    pub fn is_playing(&self) -> bool {
        self.core.playback.playing
    }

    #[wasm_bindgen(js_name = setFps)]
    pub fn set_fps(&mut self, fps: f64) {
        self.core.playback.fps = fps.max(0.1);
    }

    /// Advance playback to `now_seconds` (pass `performance.now() / 1000`).
    /// Returns how many frames it stepped.
    #[wasm_bindgen(js_name = tickPlayback)]
    pub fn tick_playback(&mut self, now_seconds: f64) -> usize {
        self.core.tick_playback(now_seconds)
    }

    // -------------------------------------------------------------- 3D view

    #[wasm_bindgen(js_name = setViewMode)]
    pub fn set_view_mode(&mut self, volume: bool) {
        self.core.view_mode = if volume { ViewMode::Volume } else { ViewMode::Movie };
    }

    #[wasm_bindgen(js_name = isVolume)]
    pub fn is_volume(&self) -> bool {
        self.core.view_mode == ViewMode::Volume
    }

    /// True once the first volume has been built — until then a frontend should
    /// show a loading state.
    #[wasm_bindgen(js_name = volumeReady)]
    pub fn volume_ready(&self) -> bool {
        self.core.volume.built_frame.is_some()
    }

    #[wasm_bindgen(js_name = orbitDrag)]
    pub fn orbit_drag(&mut self, dx: f32, dy: f32) {
        self.core.volume.cam.orbit_drag(dx, dy);
    }

    #[wasm_bindgen(js_name = panDrag)]
    pub fn pan_drag(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let speed = self.core.volume.cam.pan_speed(viewport_height);
        let (_, right, up) = self.core.volume.cam.basis();
        self.core.volume.cam.pan(dx, dy, right, up, speed);
    }

    /// Begin an orbit drag — re-pivots to what's centered in view, matching the
    /// desktop behavior.
    #[wasm_bindgen(js_name = beginOrbit)]
    pub fn begin_orbit(&mut self) {
        self.core.volume.cam.repivot();
    }

    #[wasm_bindgen(js_name = wheelFly)]
    pub fn wheel_fly(&mut self, notches: f32) {
        self.core.volume.cam.wheel_fly(notches, 1.0);
    }

    #[wasm_bindgen(js_name = resetCamera)]
    pub fn reset_camera(&mut self) {
        self.core.volume.cam.reset();
    }

    /// `0` = CAD, `1` = Blender, `2` = Maya, `3` = free-fly.
    #[wasm_bindgen(js_name = setNavMode)]
    pub fn set_nav_mode(&mut self, mode: u8) {
        let was_fly = self.core.volume.cam.nav.is_fly();
        self.core.volume.cam.nav = match mode {
            1 => NavMode::Blender,
            2 => NavMode::Maya,
            3 => NavMode::WasdFly,
            _ => NavMode::Cad,
        };
        self.core.volume.cam.sync_for_nav(was_fly);
    }

    /// `0` = MIP, `1` = alpha DVR, `2` = isosurface.
    #[wasm_bindgen(js_name = setVolumeRender)]
    pub fn set_volume_render(&mut self, mode: u8) {
        self.core.volume.render = match mode {
            1 => VolumeRender::Alpha,
            2 => VolumeRender::Surface,
            _ => VolumeRender::Mip,
        };
    }

    /// `0` = nearest, `1` = linear, `2` = cubic.
    #[wasm_bindgen(js_name = setVolumeInterp)]
    pub fn set_volume_interp(&mut self, mode: u8) {
        self.core.volume.interp = match mode {
            0 => VolumeInterp::Nearest,
            2 => VolumeInterp::Cubic,
            _ => VolumeInterp::Linear,
        };
    }

    #[wasm_bindgen(js_name = setDensity)]
    pub fn set_density(&mut self, density: f32) {
        self.core.volume.density = density;
    }

    #[wasm_bindgen(js_name = setIso)]
    pub fn set_iso(&mut self, iso: f32) {
        self.core.volume.iso = iso;
    }

    /// Per-axis voxel scale for the volume box (anisotropic Z spacing).
    #[wasm_bindgen(js_name = setVoxelScale)]
    pub fn set_voxel_scale(&mut self, x: f32, y: f32, z: f32) {
        self.core.volume.scale = [x, y, z];
    }

    #[wasm_bindgen(js_name = voxelScale)]
    pub fn voxel_scale(&self) -> Vec<f32> {
        self.core.volume.scale.to_vec()
    }

    /// `0` = Auto, `1` = Serial, `2` = Threaded. On wasm there are no worker
    /// threads, so this only affects the hint the decoder sees — kept so the UI
    /// can mirror the desktop control.
    #[wasm_bindgen(js_name = setDecodeMode)]
    pub fn set_decode_mode(&mut self, mode: u8) {
        self.core.decode_mode = match mode {
            1 => DecodeMode::Serial,
            2 => DecodeMode::Threaded,
            _ => DecodeMode::Auto,
        };
    }

    // -------------------------------------------------------------- drawing

    /// Resize the swapchain. Pass the canvas's backing-store size in device
    /// pixels (`clientWidth * devicePixelRatio`), not CSS pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if (self.config.width, self.config.height) == (w, h) {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.core.volume.aspect = (w as f32 / h as f32).clamp(0.1, 10.0);
    }

    /// How far the zoomed image overflows the canvas on each axis, in device
    /// pixels (0 where it fits and should be letterboxed instead).
    fn overflow(&self) -> (f32, f32) {
        let Some((w, h)) = self.core.stack.as_ref().and_then(|s| s.dimensions()) else {
            return (0.0, 0.0);
        };
        let (iw, ih) = (w as f32 * self.zoom, h as f32 * self.zoom);
        (
            (iw - self.config.width as f32).max(0.0),
            (ih - self.config.height as f32).max(0.0),
        )
    }

    /// The 2D layout for this frame: the on-screen rect to draw into
    /// (letterboxed when the image is smaller than the canvas) and the UV
    /// sub-rect to sample (when it's larger and therefore pannable).
    ///
    /// Same split the desktop app uses: never draw into an oversized viewport,
    /// because the backend clamps it to the framebuffer and squashes the image
    /// instead of zooming.
    fn layout_2d(&mut self) -> (f32, f32, f32, f32) {
        let (cw, ch) = (self.config.width as f32, self.config.height as f32);
        let Some((w, h)) = self.core.stack.as_ref().and_then(|s| s.dimensions()) else {
            self.core.uv_offset = [0.0, 0.0];
            self.core.uv_scale = [1.0, 1.0];
            return (0.0, 0.0, cw, ch);
        };
        let (iw, ih) = (w as f32 * self.zoom, h as f32 * self.zoom);
        let (ox, oy) = ((iw - cw).max(0.0), (ih - ch).max(0.0));
        self.pan = [self.pan[0].clamp(0.0, ox), self.pan[1].clamp(0.0, oy)];

        // Visible size on each axis, and where it sits on the canvas.
        let vw = iw.min(cw);
        let vh = ih.min(ch);
        let vx = if ox > 0.0 { 0.0 } else { (cw - iw) * 0.5 };
        let vy = if oy > 0.0 { 0.0 } else { (ch - ih) * 0.5 };

        self.core.uv_offset = [
            if ox > 0.0 { self.pan[0] / iw } else { 0.0 },
            if oy > 0.0 { self.pan[1] / ih } else { 0.0 },
        ];
        self.core.uv_scale = [vw / iw, vh / ih];
        (vx, vy, vw.max(1.0), vh.max(1.0))
    }

    /// Sync the core to the GPU and draw one frame.
    ///
    /// Returns `true` when another frame is wanted soon — a background volume
    /// build is in flight, or playback is running — so the JS loop knows
    /// whether to keep going.
    pub fn render(&mut self) -> bool {
        // Derive the 2D layout *before* syncing: `sync` uploads whatever UVs are
        // set, so computing them afterwards would show the previous frame's.
        let volume_now = self.core.view_mode == ViewMode::Volume;
        let vp_2d = if volume_now { None } else { Some(self.layout_2d()) };

        let outcome = self.core.sync(&mut self.renderer);
        self.needs_repaint = outcome.needs_repaint;

        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match self.surface.get_current_texture() {
            Acquired::Success(f) => f,
            // Suboptimal still hands back a usable texture; reconfigure so the
            // *next* frame is right, and draw this one anyway.
            Acquired::Suboptimal(f) => {
                self.surface.configure(&self.device, &self.config);
                f
            }
            // The swapchain went stale (a resize, a tab restore, a lost device).
            // Reconfigure and skip this frame rather than tearing down, and ask
            // for another so the canvas isn't left blank.
            Acquired::Outdated | Acquired::Lost => {
                self.surface.configure(&self.device, &self.config);
                return true;
            }
            // Nothing to draw into right now, and reconfiguring wouldn't help.
            Acquired::Timeout | Acquired::Occluded | Acquired::Validation => return true,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fasttiff frame") });

        // The volume path stages its uniform separately from the draw (the wgpu
        // backend's one ordering requirement); the 2D path has already uploaded
        // everything during `sync`.
        let volume = self.core.view_mode == ViewMode::Volume;
        if volume {
            self.renderer.write_volume_uniform();
        }
        // 3D fills the canvas (the shader takes the aspect ratio); 2D draws
        // into a letterboxed rect so the image keeps its proportions.
        let vp = vp_2d.unwrap_or((0.0, 0.0, self.config.width as f32, self.config.height as f32));

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fasttiff pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Nothing to draw until a stack is open (and, in 3D, until the first
            // volume has landed) — the clear above leaves a black canvas.
            if self.core.stack.is_some() {
                pass.set_viewport(vp.0, vp.1, vp.2, vp.3, 0.0, 1.0);
                if volume {
                    if self.core.volume.built_frame.is_some() {
                        self.renderer.paint_volume(&mut pass);
                    }
                } else {
                    self.renderer.paint(&mut pass);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        self.needs_repaint || self.core.playback.playing
    }

    /// The status note for the current stack, if any.
    pub fn status(&self) -> Option<String> {
        self.core.status.clone()
    }
}

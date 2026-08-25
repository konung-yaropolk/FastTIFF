//! Per-frame GPU synchronization: decoding the current frame's channels
//! (inline or via the prefetcher), uploading textures/LUTs, and assembling the
//! 3D volume plan + ray-march uniforms.
//!
//! This is the heart of the viewer and the main reason a `core` crate exists:
//! everything here — which IFD a display channel maps to, when a prefetch is
//! still valid, how the contrast window is rescaled per texture format, when
//! the volume needs rebuilding — is domain logic a second frontend would
//! otherwise have to duplicate.
//!
//! Only compiled with a GPU backend selected; without one the rest of the crate
//! still provides the full CPU-side model.

use crate::bandcache;
use crate::camera::{build_volume_params, volume_camera, VolumeChannel};
use crate::prefetch::{build_jobs, decode_jobs, ChannelJob, Decoded, PrefetchResult};
use crate::roi::{self, Roi, Serving};
use crate::stack::Stack;
use crate::viewer::{DecodeMode, ViewMode, Viewer};
use crate::window;
use crate::volume::VolumePlan;
use crate::Renderer;
use scivis_render::{ChannelKind, ChannelUniform, MAX_CHANNELS};

/// What the frontend must do after a [`Viewer::sync`] call, beyond drawing.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyncOutcome {
    /// A background volume build is in flight: schedule another frame so the
    /// result is picked up when it lands.
    pub needs_repaint: bool,
}

impl Viewer {
    /// Bring `renderer` up to date with the current state: allocate textures for
    /// the frame size and channel layout, upload LUTs and this frame's pixels
    /// (2D) or the volume (3D), and push the window/level + camera uniforms.
    ///
    /// Call once per rendered frame, *before* the draw. In the 3D view the 2D
    /// per-frame decode is skipped entirely — the volume already holds every
    /// slice.
    pub fn sync(&mut self, renderer: &mut Renderer) -> SyncOutcome {
        let mut outcome = SyncOutcome::default();
        // Read before the stack is borrowed mutably: residency depends on which
        // part of the image is on screen, which is the frontend's business.
        let (uv_offset, uv_scale) = (self.uv_offset, self.uv_scale);
        let viewport = self.viewport_px;
        let Some(loaded) = &mut self.stack else { return outcome };

        let n_channels = loaded.display.settings.len();
        if n_channels == 0 {
            return outcome;
        }

        // Per-channel GPU texture kind (R8Uint / R16Uint / R32F), picked from the
        // source format at load time — drives both texture allocation and the
        // decode path below.
        let kinds: Vec<ChannelKind> = loaded.display.settings.iter().map(|s| s.kind).collect();

        // Work out what should be resident for this view, and size the textures
        // to it. For everything that fits a texture this is the whole frame at
        // full resolution, exactly as before; only a frame past the device limit
        // gets a window.
        let (frame_w, frame_h) = loaded.tiff.frames.first().map_or((0, 0), |f| (f.width, f.height));
        let large_mode = self.large_image_mode;
        let target =
            plan_residency(loaded, renderer, &kinds, uv_offset, uv_scale, viewport, large_mode);
        // Size the textures to what is being *shown*, not to what is being
        // prepared: resizing now would blank the picture for as long as the new
        // window takes to build, which is the stall this avoids.
        if let Some(r) = target {
            let (tw, th) = r.show.texture_size();
            renderer.ensure_size(tw, th, &kinds);
        }

        if !loaded.luts_uploaded {
            for c in 0..n_channels {
                renderer.upload_lut(c, &loaded.display.luts[c]);
            }
            loaded.luts_uploaded = true;
        }

        if self.view_mode == ViewMode::Volume {
            // Nothing will consult the overview until the 2D view comes back,
            // and a volume build wants every byte it can get. Same argument as
            // cutting the band-cache budget once the window worker starts.
            loaded.overview = None;
            outcome.needs_repaint = sync_volume(loaded, &mut self.volume, renderer);
            return outcome;
        }

        // Push the decode-parallelism choice to fast-tiff-lib: Auto follows the
        // playback-keeping-up latch, Serial/Threaded force it off/on.
        // A frame too large for one texture is decoded a band at a time, with
        // the user waiting on each one — a second of wall clock that threads cut
        // to a fraction. `Auto` is otherwise driven by the playback latch, which
        // never fires here because nobody plays back a gigapixel stack; an
        // explicit Serial is still honoured, since that is a deliberate choice
        // about CPU use.
        let windowed = target.is_some_and(|r| !window::covers_whole_frame(&r.show, frame_w, frame_h));
        let parallel = self.decode_mode.parallel(self.decode_parallel)
            || (windowed && self.decode_mode == DecodeMode::Auto);
        fast_tiff_lib::set_parallel_decode(parallel);

        if let Some(r) = target {
            let read_ahead = self.playback.playing && !self.decode_parallel;
            let moving = self.view_moving;
            match sync_movie(loaded, renderer, &kinds, read_ahead, r, large_mode, moving) {
                Ok(pending) => outcome.needs_repaint |= pending,
                Err(e) => self.status = Some(format!("Failed to decode frame: {e:#}")),
            }
        }

        // Window/level goes to the shader in the units its texture actually
        // holds: 16-bit ints in raw 0..65535, floats in their own units (R32F
        // holds raw samples), and 8-bit ints in 0..255 — the slider keeps the
        // window in 0..65535, so an 8-bit channel's bounds are rescaled by 257
        // (the widening factor) here. `is_float` tells the shader which texture
        // to sample; the two integer formats share one sampler.
        const SCALE_8BIT: f32 = 257.0;
        let uniforms: Vec<ChannelUniform> = loaded
            .display
            .settings
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
        // The frontend asked in units of the whole image; the shader has to be
        // told where that lands on whatever is actually resident.
        //
        // `loaded.roi` rather than `target.show`, because a finer window may
        // have landed from the worker during the call just made — every path
        // that uploads writes `loaded.roi` in the same breath, so it is what the
        // textures hold, and mapping from the window that was merely *planned*
        // would draw one frame with the wrong corner of the image.
        let shown = loaded.roi.or_else(|| target.map(|r| r.show));
        let (off, scale) = match (shown, loaded.tiff.frames.first()) {
            (Some(r), Some(f)) => r.map_uv(f.width, f.height, uv_offset, uv_scale),
            _ => (uv_offset, uv_scale),
        };
        renderer.set_params(&uniforms, n_channels as u32, off, scale);
        outcome
    }
}

/// Decide which window of the current frame should be on the GPU, recording it
/// on the stack. Returns the window now being targeted, or `None` when there is
/// no frame to show.
///
/// Three outcomes, in decreasing order of how good the picture is:
///
/// - the frame fits a texture, so all of it is resident at full resolution;
/// - it does not fit, but the decoded frame fits in memory, so a *window* of it
///   is resident — full resolution when zoomed in, re-cut as the view moves;
/// - it does not fit and cannot be held decoded either, so a coarse view of the
///   whole frame is resident. Planning it as the whole frame is what makes this
///   work without a cache: a window covering everything always serves the next
///   view, so nothing is ever re-cut and nothing has to be decoded twice.
fn plan_residency(
    loaded: &mut Stack,
    renderer: &Renderer,
    kinds: &[ChannelKind],
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
    viewport: [f32; 2],
    mode: roi::LargeImageMode,
) -> Option<Serving> {
    let first = loaded.tiff.frames.first()?;
    let (w, h) = (first.width, first.height);
    let max_axis = renderer.max_2d_texture_size();

    // Every display channel gets a full-size texture, disabled or not, so the
    // per-texel cost is the sum over all of them.
    let bytes_per_texel: usize = kinds.iter().map(|k| gpu_bytes_per_texel(*k)).sum();

    // The ordinary case: it fits, so keep doing exactly what was done before.
    // Nothing below this line runs for an image of ordinary size — no planning,
    // no cropping, no extra copy — which is what keeps the machinery for huge
    // frames from costing anything on the frames that do not need it.
    loaded.over_texture_limit = roi::needs_window(w, h, max_axis);
    if !loaded.over_texture_limit {
        loaded.gpu_stride = 1;
        let full = Roi::full(w, h);
        return Some(Serving { show: full, want: full, required: full, ready: true });
    }

    let budget = roi::Budget::new(max_axis, bytes_per_texel, mode);
    let plan = roi::plan(w, h, uv_offset, uv_scale, viewport, budget);

    // Check the overview here rather than where it is uploaded. If a stale one
    // were allowed through, it would steer `show` at a window whose pixels
    // nobody holds — and the upload site, finding nothing to upload, would fall
    // through to assembling that frame-spanning window synchronously, which is
    // slower than the case this exists to avoid. Dropped rather than merely
    // skipped, since a stale overview will never become fresh again and the
    // memory is better returned.
    let frame_index = loaded.frame_index;
    let gen = loaded.prefetch_gen;
    let enabled: Vec<bool> = loaded.display.settings.iter().map(|s| s.enabled).collect();
    let overview = match &loaded.overview {
        Some(o) if o.matches(frame_index, gen, &enabled, kinds) => Some(o.built.roi),
        Some(_) => {
            loaded.overview = None;
            None
        }
        None => None,
    };

    // Which of the four cases applies is decided in `roi`, away from the GPU,
    // so it can be tested without one.
    let serving = roi::serve(loaded.roi, overview, &plan);
    loaded.gpu_stride = serving.show.stride;
    Some(serving)
}

/// Bytes one texel of a channel occupies in VRAM.
fn gpu_bytes_per_texel(kind: ChannelKind) -> usize {
    match kind {
        ChannelKind::Int8 => 1,
        ChannelKind::Int16 => 2,
        ChannelKind::Float => 4,
    }
}



/// 2D path: decode and upload this frame's enabled channels (from the prefetch
/// worker when its result matches, else inline), then queue the read-ahead for
/// the next frame when `read_ahead` is set.
fn sync_movie(
    loaded: &mut Stack,
    renderer: &mut Renderer,
    kinds: &[ChannelKind],
    read_ahead: bool,
    plan: Serving,
    mode: roi::LargeImageMode,
    view_moving: bool,
) -> anyhow::Result<bool> {
    let roi = plan.show;
    // Skip disabled channels (the shader multiplies them out). Re-upload when
    // the frame moves *or* the enabled set changes; an enabled-set change also
    // bumps the prefetch generation so an in-flight prefetch under the old set
    // is recognized as stale.
    let enabled: Vec<bool> = loaded.display.settings.iter().map(|s| s.enabled).collect();
    if loaded.last_enabled != enabled {
        loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
    }
    let mut result = Ok(());
    let frame_index = loaded.frame_index;
    let (fw, fh) = loaded.tiff.frames.first().map_or((0, 0), |f| (f.width, f.height));
    // A window smaller than the frame has to be re-cut when the view leaves it,
    // as well as when the frame or the channel set changes.
    let window_moved = loaded.roi != Some(roi);
    let partial = !window::covers_whole_frame(&roi, fw, fh);
    if loaded.last_uploaded != Some(frame_index) || loaded.last_enabled != enabled || window_moved {
        let want_gen = loaded.prefetch_gen;
        let jobs = build_jobs(loaded, frame_index, &enabled, kinds);

        let channels: Vec<usize> = jobs.iter().map(|j| j.channel).collect();
        // The window asked for is the overview already held, cut to exactly
        // this texture size. Upload it and skip the decode — this is the arm
        // that makes zooming out cost an upload instead of a rebuild, and
        // without it `show` being the coarse whole-frame window would send the
        // interface thread off to assemble that window synchronously, which is
        // more work than not having an overview at all.
        let from_overview = loaded
            .overview
            .as_ref()
            .is_some_and(|o| o.can_upload(&roi, frame_index, want_gen, &enabled, kinds, &channels));
        if from_overview {
            let (tw, th) = roi.texture_size();
            let o = loaded.overview.as_ref().expect("checked");
            for (channel, plane) in o.built.channels.iter().zip(&o.built.planes) {
                upload_texture(renderer, *channel, tw, th, plane);
            }
        } else if partial {
            // Decode only the strips the window covers, a band at a time. This
            // is what keeps a gigapixel frame affordable: the cost follows what
            // is on screen, and neither the frame nor the window has to be held
            // whole in memory.
            //
            // The prefetcher is bypassed here — it decodes whole frames, which
            // is exactly the work being avoided, and a stack of frames this
            // large is not something anyone plays back anyway.
            let assembled = {
                let Stack { tiff, bands, .. } = &mut *loaded;
                window::assemble(tiff, bands, &jobs, kinds, &roi, frame_index)
            };
            match assembled {
                Ok(planes) => {
                    let (tw, th) = roi.texture_size();
                    for (job, plane) in jobs.iter().zip(&planes) {
                        upload_texture(renderer, job.channel, tw, th, plane);
                    }
                    // If that was the whole frame, keep it: it is what the next
                    // view leaving its window will fall back on.
                    let built = window::Built { frame_index, roi, channels, planes };
                    capture_overview(loaded, built, fw, fh, &enabled, kinds, mode);
                }
                Err(e) => result = Err(e),
            }
        } else {
            // The whole frame is wanted, so take the ordinary path — including
            // the prefetched frame when one is ready and matches exactly
            // (generation, frame index, and channel layout). A mismatch only
            // costs a little redundant work; it can never upload the wrong
            // frame.
            let mut used_prefetch = false;
            if let Some(p) = &loaded.prefetch {
                if let Some(ready) = p.take_matching(want_gen, frame_index) {
                    if prefetch_matches(&ready, &jobs) {
                        for ch in &ready.channels {
                            upload(renderer, ch.channel, ch.width, ch.height, &roi, &ch.data);
                        }
                        used_prefetch = true;
                    }
                }
            }
            if !used_prefetch {
                // One call decodes every enabled channel; RGB planes share a
                // single decompression pass inside `decode_jobs`.
                match decode_jobs(&loaded.tiff.data, &loaded.tiff.frames, loaded.tiff.byte_order, &jobs) {
                    Ok(decoded) => {
                        for (job, data) in jobs.iter().zip(&decoded) {
                            upload(renderer, job.channel, job.width, job.height, &roi, data);
                        }
                    }
                    Err(e) => result = Err(e),
                }
            }
        }
        loaded.roi = Some(roi);
        loaded.last_uploaded = Some(frame_index);
    }

    // Read-ahead: while playing and keeping up (serial regime), ask the worker
    // to prepare the next frame — decode it (compressed) or touch its pages
    // (uncompressed) — so reaching it costs only the upload. Skipped when
    // behind (parallel decode handles that).
    if read_ahead {
        if let Some(p) = &loaded.prefetch {
            let n = loaded.frame_count();
            if n > 1 {
                let next = (loaded.frame_index + 1) % n;
                let next_jobs = build_jobs(loaded, next, &enabled, kinds);
                p.request(loaded.prefetch_gen, next, next_jobs);
            }
        }
    }
    loaded.last_enabled.clone_from(&enabled);

    // A finer window was asked for and what is on screen is the older, coarser
    // one. Take the finished article if it has landed, otherwise ask for it and
    // keep drawing until it does.
    let mut pending = false;
    if !plan.ready && plan.want != plan.show {
        let frame_index = loaded.frame_index;
        let jobs = build_jobs(loaded, frame_index, &enabled, kinds);
        let channels: Vec<usize> = jobs.iter().map(|j| j.channel).collect();

        // Spawn on first need rather than at load: a thread and a second map of
        // the file are only worth it for a frame too large to upload whole,
        // which is a small minority of what gets opened.
        if loaded.window_worker.is_none() {
            loaded.window_worker = window::WindowWorker::new(loaded.path.clone());
            if loaded.window_worker.is_some() {
                // The worker keeps its own bands, and from here on it is the
                // one decoding windows. Holding a second full cache here would
                // be memory for a thread that has stopped asking.
                loaded.bands.set_budget(bandcache::IDLE_CACHE_BYTES);
            }
        }

        // Asked against what the view *needs*, not against the rectangle that
        // was requested: the request moves with every frame of a gesture, so
        // insisting the answer match it threw away most of what was built.
        let ready = loaded
            .window_worker
            .as_ref()
            .and_then(|w| w.take_matching(frame_index, &plan.required, &channels));
        match ready {
            Some(built) => {
                // The built window's own geometry, which may be a little larger
                // than what is now wanted — that is what made it usable.
                let roi = built.roi;
                let (tw, th) = roi.texture_size();
                renderer.ensure_size(tw, th, kinds);
                for (channel, plane) in built.channels.iter().zip(&built.planes) {
                    upload_texture(renderer, *channel, tw, th, plane);
                }
                loaded.roi = Some(roi);
                loaded.last_uploaded = Some(frame_index);
                loaded.pending_window = None;
                capture_overview(loaded, built, fw, fh, &enabled, kinds, mode);
            }
            None => match &loaded.window_worker {
                Some(worker) => {
                    // Only ask when what is already being built would not do.
                    // A gesture re-plans every frame, so without this the queue
                    // fills with near-identical requests — 240 of them for 19
                    // distinct windows, measured — and the worker spends the
                    // whole gesture starting over.
                    let in_flight = loaded.pending_window.as_ref().is_some_and(|(f, r, ch)| {
                        *f == frame_index && ch == &channels && r.serves(&plan.required)
                    });
                    // While the view is still moving, whatever were started now
                    // would be planned against a zoom that has moved on before
                    // it finished. An animated step passes through every
                    // sampling level between the two rungs, and is at each of
                    // them for a few frames; only the one it lands on is worth
                    // decoding.
                    if !in_flight && !view_moving {
                        worker.request(frame_index, plan.want, jobs, kinds.to_vec());
                        loaded.pending_window = Some((frame_index, plan.want, channels.clone()));
                    }
                    // Keep the frames coming until it lands; nothing else would
                    // wake the interface once the user stops moving.
                    pending = true;
                }
                None => {
                    // No worker — no threads, or the file cannot be reopened.
                    // Take the stall rather than never sharpening.
                    let assembled = {
                        let Stack { tiff, bands, .. } = &mut *loaded;
                        window::assemble(tiff, bands, &jobs, kinds, &plan.want, frame_index)
                    };
                    if let Ok(planes) = assembled {
                        let (tw, th) = plan.want.texture_size();
                        renderer.ensure_size(tw, th, kinds);
                        for (job, plane) in jobs.iter().zip(&planes) {
                            upload_texture(renderer, job.channel, tw, th, plane);
                        }
                        loaded.roi = Some(plan.want);
                        // Its twin above sets this; leaving it out here means
                        // the next sync re-decodes a window already uploaded.
                        loaded.last_uploaded = Some(frame_index);
                        let built = window::Built {
                            frame_index,
                            roi: plan.want,
                            channels: channels.clone(),
                            planes,
                        };
                        capture_overview(loaded, built, fw, fh, &enabled, kinds, mode);
                    }
                }
            },
        }
    }
    result.map(|()| pending)
}

/// Keep a frame-spanning build as the overview, so a view leaving the resident
/// window has correct pixels to show at once.
///
/// Moves the planes rather than copying them: [`Decoded`] is deliberately not
/// `Clone`, and copying tens of megabytes on the very path this exists to speed
/// up would give back most of what it saves.
///
/// A build that does not span the frame, or is too large to hold, leaves
/// whatever is already there — an older, coarser overview is still usable, and
/// it is checked against the current frame and channel set on every read
/// anyway.
fn capture_overview(
    loaded: &mut Stack,
    built: window::Built,
    fw: u32,
    fh: u32,
    enabled: &[bool],
    kinds: &[ChannelKind],
    mode: roi::LargeImageMode,
) {
    let gen = loaded.prefetch_gen;
    let budget = window::overview_budget(mode);
    if let Some(o) = window::Overview::capture(built, fw, fh, gen, enabled, kinds, budget) {
        let (tw, th) = o.built.roi.texture_size();
        log::debug!(
            "overview retained: {}x{} at 1/{}, {} MB",
            tw,
            th,
            o.built.roi.stride,
            o.bytes() / (1 << 20)
        );
        loaded.overview = Some(o);
    }
}

/// Send one channel's window to whichever texture its format lives in.
///
/// The common case — the whole frame, full resolution — hands the decoded plane
/// straight to the GPU with no copy, which is what it did before any of this
/// existed. Only a real window pays for the cut.
fn upload(renderer: &mut Renderer, channel: usize, width: u32, height: u32, roi: &Roi, data: &Decoded) {
    if window::covers_whole_frame(roi, width, height) {
        match data {
            Decoded::U8(v) => renderer.upload_channel_u8(channel, width, height, v),
            Decoded::U16(v) => renderer.upload_channel_u16(channel, width, height, v),
            Decoded::F32(v) => renderer.upload_channel_f32(channel, width, height, v),
        }
        return;
    }
    let (tw, th) = roi.texture_size();
    upload_texture(renderer, channel, tw, th, &window::cut(data, width, height, roi));
}

/// Write an already-sized plane to its texture.
fn upload_texture(renderer: &mut Renderer, channel: usize, w: u32, h: u32, data: &Decoded) {
    match data {
        Decoded::U8(v) => renderer.upload_channel_u8(channel, w, h, v),
        Decoded::U16(v) => renderer.upload_channel_u16(channel, w, h, v),
        Decoded::F32(v) => renderer.upload_channel_f32(channel, w, h, v),
    }
}

/// 3D path: make sure the volume textures hold the current timepoint, then push
/// the camera + per-channel window params. Returns whether a build is in flight
/// (so the frontend keeps polling).
///
/// The build itself runs on a background thread (`volume::VolumeBuilder`) so
/// neither the initial build nor a 4D timepoint step blocks the UI: until the
/// result lands, the frontend's loading state (initial) or the previous
/// timepoint's volume (4D) stays on screen, and we poll each frame. In the 4D
/// case (`slices > 1`) the volume depth is Z at the current frame_index (time),
/// so playback animates the volume through time; otherwise the frame axis *is*
/// the depth and `time` stays 0.
fn sync_volume(loaded: &mut Stack, view: &mut crate::viewer::VolumeView, renderer: &mut Renderer) -> bool {
    let mut needs_repaint = false;
    let is_4d = loaded.display.dims.slices > 1;
    let time = if is_4d { loaded.frame_index } else { 0 };

    if view.built_frame != Some(time) {
        // Lazily spawn the background builder on first 3D use (it opens its own
        // mmap of the file, like the prefetch worker).
        if loaded.volume_builder.is_none() && !loaded.volume_builder_tried {
            loaded.volume_builder = crate::volume::VolumeBuilder::new(loaded.path.clone());
            loaded.volume_builder_tried = true;
        }
        let plan = plan_volume(loaded, renderer.max_3d_texture_size(), time);
        let mut handled = false;
        if let Some(builder) = &loaded.volume_builder {
            if let Some(built) = builder.take_matching(view.generation, time) {
                if let Some((vw, vh, vd, chans)) = built {
                    renderer.upload_volumes(vw, vh, vd, &chans);
                }
                // Mark built even on failure so we don't retry every frame (the
                // canvas just stays black).
                view.built_frame = Some(time);
                view.requested = None;
                handled = true;
            } else {
                let queued = view.requested == Some((view.generation, time))
                    || builder.request(view.generation, plan.clone());
                if queued {
                    view.requested = Some((view.generation, time));
                    // In flight: poll again next frame (the previous volume /
                    // loading screen stays up meanwhile).
                    needs_repaint = true;
                    handled = true;
                }
                // queued == false: the worker died (its file open failed) — fall
                // through to the synchronous build.
            }
        }
        if !handled {
            if let Some((vw, vh, vd, chans)) = crate::volume::build_volume(&loaded.tiff, &plan) {
                renderer.upload_volumes(vw, vh, vd, &chans);
            }
            view.built_frame = Some(time);
        }
    }

    renderer.set_volume_interp(view.interp);

    // Per-channel window/level, in the sampled texture's units: raw for float,
    // else the 0..65535 display window divided by 65535 (both U8 and U16 volumes
    // are unorm-normalized — see scivis_render::VolumeKind).
    // Bounded by MAX_CHANNELS, so it lives on the stack — this runs every 3D
    // frame and has no business calling the allocator.
    let blank = VolumeChannel { min: 0.0, max: 0.0, is_float: false, enabled: false, albedo_t: 1.0 };
    let mut windows = [blank; MAX_CHANNELS];
    let n = loaded.display.settings.len().min(MAX_CHANNELS);
    for (c, (slot, s)) in windows.iter_mut().zip(&loaded.display.settings).enumerate() {
        let is_float = s.kind == ChannelKind::Float;
        let (min, max) = if is_float { (s.min, s.max) } else { (s.min / 65535.0, s.max / 65535.0) };
        // The isosurface paints one fixed colour, taken from the LUT's
        // brightest point rather than its top entry — a contrast-stretched
        // palette ends black, and sampling the top would render nothing at all.
        let albedo_t = loaded
            .display
            .luts
            .get(c)
            .map_or(1.0, scivis_render::brightest_lut_t);
        *slot = VolumeChannel { min, max, is_float, enabled: s.enabled, albedo_t };
    }
    let windows = &windows[..n];

    let (w, h) = loaded.dimensions().unwrap_or((1, 1));
    let cam = volume_camera(&view.cam, view.scale, (w, h, loaded.volume_depth()));
    // Cache the box extents so the orbit re-pivot can ray-cast the box.
    view.cam.box_he = cam.box_he;
    let params = build_volume_params(&cam, windows, view.aspect, view.render, view.density, view.iso);
    renderer.set_volume_params(params);
    needs_repaint
}

/// Snapshot everything the volume builder needs (see [`VolumePlan`]): the
/// dimensions come from the stack's (possibly manually overridden) metadata so a
/// channels/frames swap is honored, `time` is the 4D timepoint to build.
pub fn plan_volume(loaded: &Stack, max_dim: u32, time: usize) -> VolumePlan {
    VolumePlan {
        kinds: loaded.display.settings.iter().map(|s| s.kind).collect(),
        rgb: loaded.display.rgb,
        channels: loaded.display.dims.channels,
        slices: loaded.display.dims.slices,
        frames: loaded.display.dims.frames,
        time,
        max_dim,
    }
}

/// Whether a prefetched result still matches the wanted jobs (same channels, in
/// order, with matching kind + dimensions). The generation/frame check happens
/// first; this guards against any residual layout mismatch before upload.
pub fn prefetch_matches(result: &PrefetchResult, jobs: &[ChannelJob]) -> bool {
    result.channels.len() == jobs.len()
        && result.channels.iter().zip(jobs).all(|(ch, job)| {
            ch.channel == job.channel && ch.kind == job.kind && ch.width == job.width && ch.height == job.height
        })
}


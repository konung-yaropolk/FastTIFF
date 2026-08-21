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

use crate::camera::{build_volume_params, volume_camera, VolumeChannel};
use crate::prefetch::{build_jobs, decode_jobs, ChannelJob, Decoded, PrefetchResult};
use crate::roi::{self, Roi, MAX_SOURCE_BYTES};
use crate::stack::{FrameSource, Stack};
use crate::viewer::{ViewMode, Viewer};
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
        let target = plan_residency(loaded, renderer, &kinds, uv_offset, uv_scale);
        if let Some(roi) = target {
            let (tw, th) = roi.texture_size();
            renderer.ensure_size(tw, th, &kinds);
        }

        if !loaded.luts_uploaded {
            for c in 0..n_channels {
                renderer.upload_lut(c, &loaded.display.luts[c]);
            }
            loaded.luts_uploaded = true;
        }

        if self.view_mode == ViewMode::Volume {
            outcome.needs_repaint = sync_volume(loaded, &mut self.volume, renderer);
            return outcome;
        }

        // Push the decode-parallelism choice to fast-tiff-lib: Auto follows the
        // playback-keeping-up latch, Serial/Threaded force it off/on.
        fast_tiff_lib::set_parallel_decode(self.decode_mode.parallel(self.decode_parallel));

        if let Some(roi) = target {
            let read_ahead = self.playback.playing && !self.decode_parallel;
            if let Err(e) = sync_movie(loaded, renderer, &kinds, read_ahead, roi) {
                self.status = Some(format!("Failed to decode frame: {e:#}"));
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
        let (off, scale) = match (target, loaded.tiff.frames.first()) {
            (Some(roi), Some(f)) => roi.map_uv(f.width, f.height, uv_offset, uv_scale),
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
) -> Option<Roi> {
    let first = loaded.tiff.frames.first()?;
    let (w, h) = (first.width, first.height);
    let max_axis = renderer.max_2d_texture_size();

    // Every display channel gets a full-size texture, disabled or not, so the
    // per-texel cost is the sum over all of them.
    let bytes_per_texel: usize = kinds.iter().map(|k| gpu_bytes_per_texel(*k)).sum();

    // The ordinary case: it fits, so keep doing exactly what was done before.
    if w <= max_axis && h <= max_axis {
        loaded.roi_source = None;
        loaded.gpu_stride = 1;
        return Some(Roi::full(w, h));
    }

    // Can the decoded frame be held? Windows are re-cut from it on every pan,
    // and re-decoding instead would cost what opening the file did.
    let source_bytes = (w as usize)
        .saturating_mul(h as usize)
        .saturating_mul(kinds.iter().map(cpu_bytes_per_sample).sum::<usize>());
    let can_hold = source_bytes <= MAX_SOURCE_BYTES;

    let plan = if can_hold {
        roi::plan(w, h, uv_offset, uv_scale, max_axis, bytes_per_texel)
    } else {
        // No cache, so plan for the whole frame: it serves every view, and is
        // therefore uploaded once.
        loaded.roi_source = None;
        roi::plan(w, h, [0.0, 0.0], [1.0, 1.0], max_axis, bytes_per_texel)
    };

    // Keep what is already resident if it still covers the view — that is what
    // stops an ordinary pan from touching the GPU at all.
    let target = match loaded.roi {
        Some(current) if current.serves(&plan.required) => current,
        _ => plan.resident,
    };
    loaded.gpu_stride = target.stride;
    Some(target)
}

/// Bytes one texel of a channel occupies in VRAM.
fn gpu_bytes_per_texel(kind: ChannelKind) -> usize {
    match kind {
        ChannelKind::Int8 => 1,
        ChannelKind::Int16 => 2,
        ChannelKind::Float => 4,
    }
}

/// Bytes one sample of a channel occupies in the decoded plane held in memory.
fn cpu_bytes_per_sample(kind: &ChannelKind) -> usize {
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
    roi: Roi,
) -> anyhow::Result<()> {
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
    let partial = !covers_whole_frame(&roi, fw, fh);
    if loaded.last_uploaded != Some(frame_index) || loaded.last_enabled != enabled || window_moved {
        let want_gen = loaded.prefetch_gen;
        let jobs = build_jobs(loaded, frame_index, &enabled, kinds);

        // Re-cut from the frame already in memory when it is the right one.
        // This is the whole point of holding it: panning a gigapixel mosaic
        // becomes a strided copy rather than another decode.
        let reusable = loaded
            .roi_source
            .as_ref()
            .is_some_and(|c| c.frame == frame_index && c.enabled == enabled);

        if reusable {
            let source = loaded.roi_source.as_ref().expect("checked just above");
            for job in &jobs {
                if let Some(data) = source.plane(job.channel) {
                    upload(renderer, job.channel, fw, fh, &roi, data);
                }
            }
        } else {
            // Use a prefetched frame if one is ready and matches exactly
            // (generation, frame index, and channel layout); otherwise decode
            // inline. A mismatch only costs a little redundant work — it can
            // never upload the wrong frame.
            let mut fresh: Option<Vec<(usize, Decoded)>> = None;
            if let Some(p) = &loaded.prefetch {
                if let Some(ready) = p.take_matching(want_gen, frame_index) {
                    if prefetch_matches(&ready, &jobs) {
                        fresh = Some(ready.channels.into_iter().map(|c| (c.channel, c.data)).collect());
                    }
                }
            }
            if fresh.is_none() {
                // One call decodes every enabled channel; RGB planes share a
                // single decompression pass inside `decode_jobs`.
                match decode_jobs(&loaded.tiff.data, &loaded.tiff.frames, loaded.tiff.byte_order, &jobs) {
                    Ok(decoded) => {
                        fresh = Some(jobs.iter().map(|j| j.channel).zip(decoded).collect());
                    }
                    Err(e) => result = Err(e),
                }
            }
            if let Some(planes) = fresh {
                for (channel, data) in &planes {
                    upload(renderer, *channel, fw, fh, &roi, data);
                }
                // Hold the decoded frame only when windows are being cut from
                // it. For a frame that fits whole there is nothing to re-cut,
                // and holding it would be pure memory for no gain.
                loaded.roi_source = partial.then(|| FrameSource { frame: frame_index, enabled: enabled.clone(), planes });
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
    loaded.last_enabled = enabled;
    result
}

/// Whether a window is the entire frame, in which case no view can ever leave
/// it and nothing needs re-cutting.
fn covers_whole_frame(roi: &Roi, width: u32, height: u32) -> bool {
    roi.x == 0 && roi.y == 0 && roi.w >= width && roi.h >= height && roi.stride == 1
}

/// Send one channel's window to whichever texture its format lives in.
///
/// The common case — the whole frame, full resolution — hands the decoded plane
/// straight to the GPU with no copy, which is what it did before any of this
/// existed. Only a real window pays for the cut.
fn upload(renderer: &mut Renderer, channel: usize, width: u32, height: u32, roi: &Roi, data: &Decoded) {
    if covers_whole_frame(roi, width, height) {
        match data {
            Decoded::U8(v) => renderer.upload_channel_u8(channel, width, height, v),
            Decoded::U16(v) => renderer.upload_channel_u16(channel, width, height, v),
            Decoded::F32(v) => renderer.upload_channel_f32(channel, width, height, v),
        }
        return;
    }
    let (tw, th) = roi.texture_size();
    match data {
        Decoded::U8(v) => renderer.upload_channel_u8(channel, tw, th, &roi::extract(v, width, height, roi)),
        Decoded::U16(v) => renderer.upload_channel_u16(channel, tw, th, &roi::extract(v, width, height, roi)),
        Decoded::F32(v) => renderer.upload_channel_f32(channel, tw, th, &roi::extract(v, width, height, roi)),
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

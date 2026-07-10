//! Per-frame GPU synchronization: decoding the current frame's channels
//! (inline or via the prefetcher), uploading textures/LUTs, and assembling
//! the 3D volume plan + ray-march uniforms. Split from `app.rs`.

use super::*;

use super::camera::{volume_camera, VolumeCam};
use crate::prefetch::{decode_jobs, ChannelJob, Decoded, PrefetchResult};
use crate::render::{self, ChannelKind, ChannelUniform, MAX_CHANNELS};

impl ViewerApp {
    pub(super) fn sync_gpu(&mut self, egui_ctx: &egui::Context, frame: &eframe::Frame) {
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
pub(super) fn build_jobs(loaded: &LoadedStack, frame_index: usize, enabled: &[bool], kinds: &[ChannelKind]) -> Vec<ChannelJob> {
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
pub(super) fn plan_volume(loaded: &LoadedStack, max_dim: u32, time: usize) -> crate::volume::VolumePlan {
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

/// Assemble the ray-march uniforms for the current camera + window. The volume's
/// depth axis is Z in the 4D case (else the frame axis); the box half-extents
/// fold in the per-axis scale so anisotropic voxels render with correct
/// proportions regardless of the (subsampled) texture size.
pub(super) fn build_volume_params(loaded: &LoadedStack, view: VolumeCam) -> render::VolumeParams {
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

/// Whether a prefetched result still matches the wanted jobs (same channels, in
/// order, with matching kind + dimensions). The generation/frame check happens
/// first; this guards against any residual layout mismatch before upload.
pub(super) fn prefetch_matches(result: &PrefetchResult, jobs: &[ChannelJob]) -> bool {
    result.channels.len() == jobs.len()
        && result.channels.iter().zip(jobs).all(|(ch, job)| {
            ch.channel == job.channel && ch.kind == job.kind && ch.width == job.width && ch.height == job.height
        })
}

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

use crate::camera::{build_volume_params, volume_camera};
use crate::prefetch::{decode_jobs, ChannelJob, Decoded, PrefetchResult};
use crate::stack::Stack;
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
        let Some(loaded) = &mut self.stack else { return outcome };

        let n_channels = loaded.channel_settings.len();
        if n_channels == 0 {
            return outcome;
        }

        // Per-channel GPU texture kind (R8Uint / R16Uint / R32F), picked from the
        // source format at load time — drives both texture allocation and the
        // decode path below.
        let kinds: Vec<ChannelKind> = loaded.channel_settings.iter().map(|s| s.kind).collect();

        if let Some(first) = loaded.tiff.frames.first() {
            renderer.ensure_size(first.width, first.height, &kinds);
        }

        if !loaded.luts_uploaded {
            for c in 0..n_channels {
                renderer.upload_lut(c, &loaded.tiff.meta.channel_display[c].lut);
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

        if let Err(e) = sync_movie(loaded, renderer, &kinds, self.playback.playing && !self.decode_parallel) {
            self.status = Some(format!("Failed to decode frame: {e:#}"));
        }

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
        renderer.set_params(&uniforms, n_channels as u32, self.uv_offset, self.uv_scale);
        outcome
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
) -> anyhow::Result<()> {
    // Skip disabled channels (the shader multiplies them out). Re-upload when
    // the frame moves *or* the enabled set changes; an enabled-set change also
    // bumps the prefetch generation so an in-flight prefetch under the old set
    // is recognized as stale.
    let enabled: Vec<bool> = loaded.channel_settings.iter().map(|s| s.enabled).collect();
    if loaded.last_enabled != enabled {
        loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
    }
    let mut result = Ok(());
    if loaded.last_uploaded != Some(loaded.frame_index) || loaded.last_enabled != enabled {
        let frame_index = loaded.frame_index;
        let want_gen = loaded.prefetch_gen;
        let jobs = build_jobs(loaded, frame_index, &enabled, kinds);

        // Use a prefetched frame if one is ready and matches exactly
        // (generation, frame index, and channel layout); otherwise decode
        // inline. A mismatch only costs a little redundant work — it can
        // never upload the wrong frame.
        let mut used_prefetch = false;
        if let Some(p) = &loaded.prefetch {
            if let Some(ready) = p.take_matching(want_gen, frame_index) {
                if prefetch_matches(&ready, &jobs) {
                    for ch in &ready.channels {
                        upload(renderer, ch.channel, ch.width, ch.height, &ch.data);
                    }
                    used_prefetch = true;
                }
            }
        }
        if !used_prefetch {
            // One call decodes every enabled channel; RGB planes share a single
            // decompression pass inside `decode_jobs`.
            match decode_jobs(&loaded.tiff.mmap, &loaded.tiff.frames, loaded.tiff.byte_order, &jobs) {
                Ok(decoded) => {
                    for (job, data) in jobs.iter().zip(decoded) {
                        upload(renderer, job.channel, job.width, job.height, &data);
                    }
                }
                Err(e) => result = Err(e),
            }
        }
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

/// Send one decoded channel to whichever texture its format lives in.
fn upload(renderer: &mut Renderer, channel: usize, width: u32, height: u32, data: &Decoded) {
    match data {
        Decoded::U8(v) => renderer.upload_channel_u8(channel, width, height, v),
        Decoded::U16(v) => renderer.upload_channel_u16(channel, width, height, v),
        Decoded::F32(v) => renderer.upload_channel_f32(channel, width, height, v),
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
    let is_4d = loaded.tiff.meta.slices > 1;
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
    let mut windows = [(0.0f32, 0.0f32, false, false); MAX_CHANNELS];
    let n = loaded.channel_settings.len().min(MAX_CHANNELS);
    for (slot, s) in windows.iter_mut().zip(&loaded.channel_settings) {
        let float = s.kind == ChannelKind::Float;
        let (lo, hi) = if float { (s.min, s.max) } else { (s.min / 65535.0, s.max / 65535.0) };
        *slot = (lo, hi, float, s.enabled);
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

/// The per-channel decode jobs for `frame_index`'s enabled channels, used both
/// to decode inline and to ask the prefetch worker for the next frame. Maps each
/// display channel to its IFD/plane: for RGB, all channels are sample planes of
/// one IFD per frame; otherwise each channel is its own IFD in ImageJ's default
/// `xyczt` plane order (channel fastest, then Z — frozen at slice 0 — then time).
pub fn build_jobs(loaded: &Stack, frame_index: usize, enabled: &[bool], kinds: &[ChannelKind]) -> Vec<ChannelJob> {
    let Some((width, height)) = loaded.dimensions() else { return Vec::new() };
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

/// Snapshot everything the volume builder needs (see [`VolumePlan`]): the
/// dimensions come from the stack's (possibly manually overridden) metadata so a
/// channels/frames swap is honored, `time` is the 4D timepoint to build.
pub fn plan_volume(loaded: &Stack, max_dim: u32, time: usize) -> VolumePlan {
    VolumePlan {
        kinds: loaded.channel_settings.iter().map(|s| s.kind).collect(),
        rgb: loaded.rgb,
        channels: loaded.tiff.meta.channels,
        slices: loaded.tiff.meta.slices,
        frames: loaded.tiff.meta.frames,
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

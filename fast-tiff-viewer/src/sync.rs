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

use crate::camera::{build_volume_params, VolumeChannel};
use crate::display::{ChannelMasks, Revision};
use crate::document::Document;
use crate::histogram::{self, Binning, HistogramSource};
use crate::prefetch::{
    build_jobs, decode_jobs_borrowed, ChannelJob, Decoded, PrefetchResult, Samples,
};
use crate::residency::{Residency, VolumeResidency};
use crate::roi::{self, Roi, Serving};
use crate::viewer::{DecodeMode, ViewMode, Viewer, VolumeView};
use crate::volume::VolumePlan;
use crate::window;
use crate::worker::CancelToken;
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
        // Read before the document is borrowed mutably: residency depends on which
        // part of the image is on screen, which is the frontend's business.
        let (uv_offset, uv_scale) = (self.uv_offset, self.uv_scale);
        let viewport = self.viewport_px;
        // The playback clock, while a movie plays: what the histograms'
        // throttle on this thread is measured against.
        let clock = if self.playback.playing {
            self.playback.last_time
        } else {
            None
        };
        let histograms_wanted = self.histograms_wanted;
        let Some(loaded) = &mut self.document else {
            return outcome;
        };
        // What the renderer holds that no longer describes the document as it
        // stands goes first, all of it in one place: the frame after a scrub,
        // a toggle or an override, the LUTs after a pick, the volume after an
        // override or a new file. Nothing that changes the display model
        // touches what is resident; it moves a revision, and this compares.
        // Everything below may take what is still recorded as current, and
        // what is to be sent first is what it returns.
        let residency = &mut self.residency;
        let pending = residency.begin_sync(&loaded.display, loaded.frame());
        // A volume of another document, or of the axes before an override, is no
        // use to this one: its GPU memory goes back now, this being the first
        // call that has the renderer.
        if pending.release_volume {
            renderer.release_volume();
        }
        // Whatever below changes the histograms, the frontend drew this
        // repaint's plot before this call and needs another to show them.
        let histograms_before = loaded.services.histograms.version();

        // The display model as of this call, read once: every cache consulted
        // and every request made below is keyed on this revision, so they all
        // agree on which version of the model they serve. Nothing in this call
        // changes the model; the frontend's toggles and overrides have already
        // moved it, at the moment they were made.
        //
        // With it, each channel's GPU texture kind (R8Uint / R16Uint / R32F),
        // picked from the source format at load time — drives both texture
        // allocation and the decode path below — and which channels are on.
        // Held by the display and copied out, so no repaint collects them.
        let current = Current {
            revision: loaded.display.revision(),
            masks: loaded.display.masks(),
            histograms: HistogramDemand {
                wanted: histograms_wanted,
                inline: histograms_wanted && loaded.services.histograms.may_bin_inline(clock),
            },
        };
        let kinds = current.masks.kinds();
        let n_channels = current.masks.len();
        if n_channels == 0 {
            return outcome;
        }

        // Work out what should be resident for this view, and size the textures
        // to it. For everything that fits a texture this is the whole frame at
        // full resolution, exactly as before; only a frame past the device limit
        // gets a window.
        let (frame_w, frame_h) = loaded
            .tiff
            .frames
            .first()
            .map_or((0, 0), |f| (f.width, f.height));
        let large_mode = self.large_image_mode;
        let target = plan_residency(
            loaded, residency, renderer, current, uv_offset, uv_scale, viewport, large_mode,
        );
        // Size the textures to what is being *shown*, not to what is being
        // prepared: resizing now would blank the picture for as long as the new
        // window takes to build, which is the stall this avoids.
        if let Some(r) = target {
            let (tw, th) = r.show.texture_size();
            renderer.ensure_size(tw, th, kinds);
        }

        if pending.upload_luts {
            for (c, lut) in loaded.display.luts().iter().enumerate().take(n_channels) {
                renderer.upload_lut(c, lut);
            }
            residency.luts_uploaded(&loaded.display);
        }

        if self.view_mode == ViewMode::Volume {
            // Nothing will consult the overview until the 2D view comes back,
            // and a volume build wants every byte it can get. Same argument as
            // cutting the band-cache budget once the window worker starts.
            residency.overview = None;
            outcome.needs_repaint = sync_volume(
                loaded,
                &mut self.volume,
                &mut residency.volume,
                renderer,
                &mut self.status,
            );
            // The volume holds no 2D frame to bin, so the frame's histograms
            // are decoded for, as the 2D view does for a channel switched off:
            // during 4D playback once a throttle interval, as nothing else
            // starts them.
            if current.histograms.inline {
                histogram::complete(loaded, residency, true);
            }
            outcome.needs_repaint |= note_histograms(loaded, histograms_before, current, clock);
            return outcome;
        }

        // The decode-parallelism choice for this thread's decodes: Auto follows
        // the playback-keeping-up latch, Serial/Threaded force it off/on. This
        // thread's own, so a save or a volume build running elsewhere keeps its
        // own; the workers serving this view are handed it with each request.
        // A frame too large for one texture is decoded a band at a time, with
        // the user waiting on each one — a second of wall clock that threads cut
        // to a fraction. `Auto` is otherwise driven by the playback latch, which
        // never fires here because nobody plays back a gigapixel stack; an
        // explicit Serial is still honoured, since that is a deliberate choice
        // about CPU use.
        let windowed =
            target.is_some_and(|r| !window::covers_whole_frame(&r.show, frame_w, frame_h));
        let parallel = self.decode_mode.parallel(self.decode_parallel)
            || (windowed && self.decode_mode == DecodeMode::Auto);
        fast_tiff_lib::set_thread_parallel_decode(Some(parallel));

        if let Some(r) = target {
            let read_ahead = self.playback.playing && !self.decode_parallel;
            let moving = self.view_moving;
            match sync_movie(
                loaded, residency, renderer, current, read_ahead, r, large_mode, moving,
            ) {
                Ok(pending) => outcome.needs_repaint |= pending,
                Err(e) => self.status = Some(format!("Failed to decode frame: {e:#}")),
            }
            // What showing the frame did not bin: the channels switched off,
            // a frame already on screen when the histograms were first asked
            // for, or for a frame past the texture limit, its overview. While
            // a movie plays, only the channels switched off, of a frame whose
            // histograms showing it started: a repaint between two frames
            // decodes nothing, and a frame shown while the throttle was shut
            // leaves its histograms to a later one.
            if current.histograms.inline {
                histogram::complete(loaded, residency, clock.is_none());
            }
        }
        outcome.needs_repaint |= note_histograms(loaded, histograms_before, current, clock);

        // Window/level goes to the shader in the units its texture actually
        // holds: the file's own for 8-bit (R8Uint holds the raw bytes) and
        // float (R32F holds the raw values), the decoder's 16-bit display scale
        // for everything on R16Uint. The window is held in file units, so that
        // conversion is [`crate::units::SampleUnits::to_texture`], shared with
        // the CPU compositor so the two cannot drift. `is_float` tells the
        // shader which texture to sample; the two integer formats share one
        // sampler. Bounded by MAX_CHANNELS, so built on the stack: this runs
        // every repaint.
        let blank = ChannelUniform {
            min: 0.0,
            max: 0.0,
            enabled: false,
            is_float: false,
        };
        let mut uniforms = [blank; MAX_CHANNELS];
        for (slot, s) in uniforms.iter_mut().zip(loaded.display.settings()) {
            let (min, max) = s.units.to_texture(s.window);
            *slot = ChannelUniform {
                min,
                max,
                enabled: s.enabled,
                is_float: s.kind() == ChannelKind::Float,
            };
        }
        let uniforms = &uniforms[..n_channels];
        // The frontend asked in units of the whole image; the shader has to be
        // told where that lands on whatever is actually resident.
        //
        // The resident window rather than `target.show`, because a finer
        // window may have landed from the worker during the call just made —
        // every path that uploads records the window in the same breath
        // (`Residency::uploaded`), so it is what the textures hold, and
        // mapping from the window that was merely *planned* would draw one
        // frame with the wrong corner of the image.
        let shown = residency.roi.or_else(|| target.map(|r| r.show));
        let (off, scale) = match (shown, loaded.tiff.frames.first()) {
            (Some(r), Some(f)) => r.map_uv(f.width, f.height, uv_offset, uv_scale),
            _ => (uv_offset, uv_scale),
        };
        renderer.set_params(uniforms, n_channels as u32, off, scale);
        outcome
    }
}

/// The display model as one [`Viewer::sync`] call sees it, read once at the
/// top: the revision every cache and request is keyed on, and the per-channel
/// formats and flags that go with it. And what the frontend wants of the
/// frame's histograms.
#[derive(Clone, Copy)]
struct Current {
    revision: Revision,
    masks: ChannelMasks,
    histograms: HistogramDemand,
}

/// What a sync does for the histograms of the frame it shows (see
/// [`crate::histogram`]).
#[derive(Clone, Copy)]
struct HistogramDemand {
    /// The frontend shows them
    /// ([`Viewer::histograms_wanted`](crate::Viewer::histograms_wanted)): keep
    /// what the read-ahead binned.
    wanted: bool,
    /// And this thread may bin, and decode, for them now: wanted, and not
    /// within the playback throttle of the last time it did.
    inline: bool,
}

/// Close a sync's work on the histograms: when it changed them, start the
/// playback throttle if this thread did the binning, and say whether the
/// frontend needs another frame to draw them. Not while a movie plays: the
/// next frame of it is already scheduled and draws them, a frame later, where
/// asking for one more would draw every frame of the movie twice.
fn note_histograms(
    loaded: &mut Document,
    before: u64,
    current: Current,
    clock: Option<f64>,
) -> bool {
    let histograms = &mut loaded.services.histograms;
    let changed = histograms.version() != before;
    if changed && current.histograms.inline {
        histograms.binned_inline(clock);
    }
    changed && clock.is_none()
}

/// Decide which window of the current frame should be on the GPU, recording
/// what that says about the frame (its stride, whether it is past the texture
/// limit) in the residency. Returns the window now being targeted, or `None`
/// when there is no frame to show.
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
#[allow(clippy::too_many_arguments)]
fn plan_residency(
    loaded: &Document,
    residency: &mut Residency,
    renderer: &Renderer,
    current: Current,
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
    let bytes_per_texel: usize = current
        .masks
        .kinds()
        .iter()
        .map(|k| k.bytes_per_sample())
        .sum();

    // The ordinary case: it fits, so keep doing exactly what was done before.
    // Nothing below this line runs for an image of ordinary size — no planning,
    // no cropping, no extra copy — which is what keeps the machinery for huge
    // frames from costing anything on the frames that do not need it.
    residency.over_texture_limit = roi::needs_window(w, h, max_axis);
    if !residency.over_texture_limit {
        residency.gpu_stride = 1;
        let full = Roi::full(w, h);
        return Some(Serving {
            show: full,
            want: full,
            required: full,
            ready: true,
        });
    }

    let budget = roi::Budget::new(max_axis, bytes_per_texel, mode);
    let plan = roi::plan(w, h, uv_offset, uv_scale, viewport, budget);

    // An overview held here is of this frame under this revision: the
    // reconcile at the top of the sync dropped any other. A stale one let
    // through would steer `show` at a window whose pixels nobody holds — and
    // the upload site, finding nothing to upload, would fall through to
    // assembling that frame-spanning window synchronously, which is slower
    // than the case this exists to avoid.
    let overview = residency.overview.as_ref().map(|o| o.built.roi);

    // Which of the four cases applies is decided in `roi`, away from the GPU,
    // so it can be tested without one.
    let serving = roi::serve(residency.roi, overview, &plan);
    residency.gpu_stride = serving.show.stride;
    Some(serving)
}

/// 2D path: decode and upload this frame's enabled channels (from the prefetch
/// worker when its result matches, else inline), then queue the read-ahead for
/// the next frame when `read_ahead` is set.
#[allow(clippy::too_many_arguments)]
fn sync_movie(
    loaded: &mut Document,
    residency: &mut Residency,
    renderer: &mut Renderer,
    current: Current,
    read_ahead: bool,
    plan: Serving,
    mode: roi::LargeImageMode,
    view_moving: bool,
) -> anyhow::Result<bool> {
    let roi = plan.show;
    let revision = current.revision;
    let (kinds, enabled) = (current.masks.kinds(), current.masks.enabled());
    // Whether showing the frame binned anything, by the time the read-ahead
    // is asked for the next one.
    let histograms_before = loaded.services.histograms.version();
    // A view of the whole of a frame past the texture limit is binned as it
    // is built (see `capture_overview`).
    let bin = current.histograms.inline;
    let mut result = Ok(());
    let frame_index = loaded.frame();
    let (fw, fh) = loaded
        .tiff
        .frames
        .first()
        .map_or((0, 0), |f| (f.width, f.height));
    // Skip disabled channels (the shader multiplies them out). Re-upload when
    // the textures do not hold this frame as the display now stands — the
    // reconcile dropped the record when the frame or the display revision
    // moved: a channel switched on has nothing in its texture yet, and after a
    // reinterpretation of the axes every channel may read another IFD. A
    // window smaller than the frame also has to be re-cut when the view
    // leaves it. Both are [`Residency::needs_upload`].
    let partial = !window::covers_whole_frame(&roi, fw, fh);
    if residency.needs_upload(&roi) {
        let jobs = build_jobs(loaded, frame_index, enabled);

        let channels: Vec<usize> = jobs.iter().map(|j| j.channel).collect();
        // The window asked for is the overview already held, cut to exactly
        // this texture size. Upload it and skip the decode — this is the arm
        // that makes zooming out cost an upload instead of a rebuild, and
        // without it `show` being the coarse whole-frame window would send the
        // interface thread off to assemble that window synchronously, which is
        // more work than not having an overview at all.
        let from_overview = residency
            .overview
            .as_ref()
            .is_some_and(|o| o.can_upload(&roi, frame_index, revision));
        if from_overview {
            let (tw, th) = roi.texture_size();
            let o = residency.overview.as_ref().expect("checked");
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
            let assembled = window::assemble(
                &loaded.tiff,
                &mut loaded.services.bands,
                revision.layout,
                &jobs,
                &roi,
                frame_index,
                &CancelToken::never(),
            );
            match assembled {
                Ok(planes) => {
                    let (tw, th) = roi.texture_size();
                    for (job, plane) in jobs.iter().zip(&planes) {
                        upload_texture(renderer, job.channel, tw, th, plane);
                    }
                    // If that was the whole frame, keep it: it is what the next
                    // view leaving its window will fall back on.
                    let built = window::Built {
                        frame_index,
                        roi,
                        channels,
                        planes,
                    };
                    capture_overview(loaded, residency, built, fw, fh, revision, mode, bin);
                }
                Err(e) => result = Err(e),
            }
        } else {
            // The whole frame is wanted, so take the ordinary path — including
            // the prefetched frame when one is ready and matches exactly
            // (display revision, frame index, and channel layout). A mismatch
            // only costs a little redundant work; it can never upload the
            // wrong frame.
            let mut used_prefetch = false;
            let ready = loaded
                .services
                .prefetch
                .as_ref()
                .and_then(|p| p.take_matching(revision, frame_index));
            if let Some(ready) = ready {
                if prefetch_matches(&ready, &jobs) {
                    for ch in &ready.channels {
                        upload(
                            renderer,
                            ch.channel,
                            ch.width,
                            ch.height,
                            &roi,
                            ch.data.view(),
                        );
                    }
                    used_prefetch = true;
                    if current.histograms.wanted {
                        // While it goes on reading ahead, a frame it did not
                        // bin is left to it: the next is asked for binned.
                        let bin_here = current.histograms.inline && !read_ahead;
                        keep_read_ahead_histograms(loaded, ready, bin_here);
                    }
                }
            }
            if !used_prefetch {
                // The channels whose histograms this decode is to give, asked
                // before it, because the planes it returns borrow the document.
                let missing = if current.histograms.inline {
                    histogram::prepare(loaded, HistogramSource::Frame)
                } else {
                    [false; MAX_CHANNELS]
                };
                // One call decodes every enabled channel; RGB planes share a
                // single decompression pass. Borrowed: an uncompressed plane
                // goes from the mapping straight into its texture.
                match decode_jobs_borrowed(
                    &loaded.tiff.data,
                    &loaded.tiff.frames,
                    loaded.tiff.byte_order,
                    &jobs,
                ) {
                    Ok(decoded) => {
                        for (job, data) in jobs.iter().zip(&decoded) {
                            upload(
                                renderer,
                                job.channel,
                                job.width,
                                job.height,
                                &roi,
                                data.view(),
                            );
                        }
                        // Binned from the planes just uploaded, which is all
                        // a histogram of them costs.
                        let binned = histogram::bin_shown(
                            loaded,
                            &missing,
                            jobs.iter()
                                .zip(&decoded)
                                .map(|(job, data)| (job.channel, data.view())),
                        );
                        loaded.services.histograms.put(binned);
                    }
                    Err(e) => result = Err(e),
                }
            }
        }
        residency.uploaded(roi, frame_index, revision);
    }

    // Read-ahead: while playing and keeping up (serial regime), ask the worker
    // to prepare the next frame — decode it (compressed) or touch its pages
    // (uncompressed) — so reaching it costs only the upload. Skipped when
    // behind (parallel decode handles that).
    let n = loaded.frame_count();
    if read_ahead && n > 1 {
        // Started on the first frame read ahead: a stack that is never played
        // needs no thread for it.
        loaded.services.start_prefetch(&loaded.tiff);
        if let Some(p) = &loaded.services.prefetch {
            let next = (loaded.frame() + 1) % n;
            let next_jobs = build_jobs(loaded, next, enabled);
            // The worker bins the next frame when this thread could have
            // binned this one and did not: once a throttle interval, and
            // with the binning off this thread. Not every frame, since a
            // frame it bins is published that much later (see
            // `histogram::PLAYBACK_INTERVAL`). A repeat of a request in
            // flight keeps what it asked, so a throttle that opens between
            // two frames bins the one after the next.
            let binned = loaded.services.histograms.version() != histograms_before;
            let binning = (current.histograms.inline && !binned).then(|| Binning::of(loaded));
            p.request(revision, next, next_jobs, binning);
        }
    }

    // A finer window was asked for and what is on screen is the older, coarser
    // one. Take the finished article if it has landed, otherwise ask for it and
    // keep drawing until it does. All of it under this call's revision: a
    // window built, or still being built, before an override or a toggle is
    // neither taken nor counted as in flight.
    let mut pending = false;
    if !plan.ready && plan.want != plan.show {
        let frame_index = loaded.frame();
        let jobs = build_jobs(loaded, frame_index, enabled);

        // Started on first need: a thread is only worth it for a frame too
        // large to upload whole, which is a small minority of what gets
        // opened.
        loaded.services.start_window_worker(&loaded.tiff);

        // Asked against what the view *needs*, not against the rectangle that
        // was requested: the request moves with every frame of a gesture, so
        // insisting the answer match it threw away most of what was built.
        let ready = loaded
            .services
            .window_worker
            .as_ref()
            .and_then(|w| w.take_matching(frame_index, &plan.required, revision));
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
                residency.uploaded(roi, frame_index, revision);
                residency.pending_window = None;
                capture_overview(loaded, residency, built, fw, fh, revision, mode, bin);
            }
            None => match &loaded.services.window_worker {
                Some(worker) => {
                    // Only ask when what is already being built would not do.
                    // A gesture re-plans every frame, so without this the queue
                    // fills with near-identical requests — 240 of them for 19
                    // distinct windows, measured — and the worker spends the
                    // whole gesture starting over. A window still recorded as
                    // asked for is of this frame under this revision: one asked
                    // for before a scrub, an override or a toggle went at the
                    // reconcile, and no longer counts as in flight.
                    let in_flight = residency
                        .pending_window
                        .is_some_and(|(_, r, _)| r.serves(&plan.required));
                    // While the view is still moving, whatever were started now
                    // would be planned against a zoom that has moved on before
                    // it finished. An animated step passes through every
                    // sampling level between the two rungs, and is at each of
                    // them for a few frames; only the one it lands on is worth
                    // decoding.
                    if !in_flight && !view_moving {
                        worker.request(frame_index, plan.want, revision, jobs);
                        residency.pending_window = Some((frame_index, plan.want, revision));
                    }
                    // Keep the frames coming until it lands; nothing else would
                    // wake the interface once the user stops moving.
                    pending = true;
                }
                None => {
                    // No worker — no threads, or the thread could not start.
                    // Take the stall rather than never sharpening.
                    let assembled = window::assemble(
                        &loaded.tiff,
                        &mut loaded.services.bands,
                        revision.layout,
                        &jobs,
                        &plan.want,
                        frame_index,
                        &CancelToken::never(),
                    );
                    if let Ok(planes) = assembled {
                        let (tw, th) = plan.want.texture_size();
                        renderer.ensure_size(tw, th, kinds);
                        for (job, plane) in jobs.iter().zip(&planes) {
                            upload_texture(renderer, job.channel, tw, th, plane);
                        }
                        // Its twin above records this; leaving it out here
                        // means the next sync re-decodes a window already
                        // uploaded.
                        residency.uploaded(plan.want, frame_index, revision);
                        let built = window::Built {
                            frame_index,
                            roi: plan.want,
                            channels: jobs.iter().map(|j| j.channel).collect(),
                            planes,
                        };
                        capture_overview(loaded, residency, built, fw, fh, revision, mode, bin);
                    }
                }
            },
        }
    }
    result.map(|()| pending)
}

/// Keep the histograms of a frame taken from the read-ahead: the ones its
/// worker binned, and when this thread is to bin (`bin_here`), the channels
/// it did not, from the planes the result holds.
fn keep_read_ahead_histograms(loaded: &mut Document, ready: PrefetchResult, bin_here: bool) {
    let PrefetchResult {
        channels,
        mut histograms,
        ..
    } = ready;
    if histograms.is_empty() && !bin_here {
        return;
    }
    let mut missing = histogram::prepare(loaded, HistogramSource::Frame);
    if bin_here {
        for h in &histograms {
            if let Some(m) = missing.get_mut(h.channel) {
                *m = false;
            }
        }
        histograms.extend(histogram::bin_shown(
            loaded,
            &missing,
            channels.iter().map(|ch| (ch.channel, ch.data.view())),
        ));
    }
    loaded.services.histograms.put(histograms);
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
/// it is checked against the current frame and display revision on every read
/// anyway (see [`Residency::reconcile`]).
///
/// `revision` is the one `built` was decoded under, which is this sync's: the
/// window worker's results are only taken under the revision they were asked
/// for under.
///
/// With `bin`, a build spanning the frame is also binned for the frame's
/// histograms, kept or not: it is the view of the whole frame they are made
/// from, and one too large to keep can be binned only now. Without this, a
/// frame whose overview is never kept (4K, several float channels) would have
/// no histogram however far it was zoomed out.
#[allow(clippy::too_many_arguments)]
fn capture_overview(
    loaded: &mut Document,
    residency: &mut Residency,
    built: window::Built,
    fw: u32,
    fh: u32,
    revision: Revision,
    mode: roi::LargeImageMode,
    bin: bool,
) {
    if bin && built.frame_index == loaded.frame() && window::spans_whole_frame(&built.roi, fw, fh) {
        histogram::bin_whole_view(loaded, &built);
    }
    let budget = window::overview_budget(mode);
    if let Some(o) = window::Overview::capture(built, fw, fh, revision, budget) {
        let (tw, th) = o.built.roi.texture_size();
        log::debug!(
            "overview retained: {}x{} at 1/{}, {} MB",
            tw,
            th,
            o.built.roi.stride,
            o.bytes() / (1 << 20)
        );
        residency.overview = Some(o);
    }
}

/// Send one channel's window to whichever texture its format lives in.
///
/// The common case — the whole frame, full resolution — hands the decoded plane
/// straight to the GPU with no copy, which is what it did before any of this
/// existed. Only a real window pays for the cut.
fn upload(
    renderer: &mut Renderer,
    channel: usize,
    width: u32,
    height: u32,
    roi: &Roi,
    data: Samples<'_>,
) {
    if window::covers_whole_frame(roi, width, height) {
        match data {
            Samples::U8(v) => renderer.upload_channel_u8(channel, width, height, v),
            Samples::U16(v) => renderer.upload_channel_u16(channel, width, height, v),
            Samples::F32(v) => renderer.upload_channel_f32(channel, width, height, v),
        }
        return;
    }
    let (tw, th) = roi.texture_size();
    upload_texture(
        renderer,
        channel,
        tw,
        th,
        &window::cut(data, width, height, roi),
    );
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
fn sync_volume(
    loaded: &mut Document,
    view: &mut VolumeView,
    res: &mut VolumeResidency,
    renderer: &mut Renderer,
    status: &mut Option<String>,
) -> bool {
    let mut needs_repaint = false;
    let is_4d = loaded.display.dims().slices > 1;
    let time = if is_4d { loaded.frame() } else { 0 };

    let generation = res.generation;
    // In a browser the GPU can refuse a volume after its upload returned (see
    // `take_late_upload_error`): handled as if the upload had failed — when it
    // is the volume still wanted. One of a document since closed is already being
    // replaced, and the new one owes it nothing.
    if let Some((error, bytes)) = renderer.take_late_upload_error() {
        if res.uploaded_generation == Some(res.generation) {
            out_of_memory(res, status, &error, bytes);
        }
    }
    // Until the GPU has said there was room for the volume, it is not drawn,
    // and nothing else may ask for the frame that will draw it.
    if renderer.volume_pending() {
        needs_repaint = true;
    }

    if res.built_frame != Some(time) {
        let budget = *res
            .planned_budget
            .get_or_insert_with(|| volume_budget(renderer, res.budget_cap));
        let plan = plan_volume(loaded, renderer, time, budget, view);
        let mut handled = false;
        // Started on first 3D use, over the same stack as the window — so a
        // stack held in memory gets one too.
        if let Some(builder) = loaded.services.start_volume_builder(&loaded.tiff) {
            if let Some(built) = builder.take_matching(res.generation, time) {
                let generation = res.generation;
                if let Some(volume) = built {
                    upload_volume(renderer, &volume, res, status);
                }
                // Mark built even on failure so we don't retry every frame (the
                // canvas just stays black) — unless the upload ran out of
                // memory and started a smaller rebuild, a new generation.
                if res.generation == generation {
                    res.built_frame = Some(time);
                    res.requested = None;
                }
                handled = true;
            } else if builder.request(res.generation, plan.clone()) {
                // Asked every frame until it is answered: a repeat of the
                // request in flight is one lock and changes nothing, and it is
                // what notices the worker has died.
                res.requested = Some((res.generation, time));
                // In flight: poll again next frame (the previous volume /
                // loading screen stays up meanwhile).
                needs_repaint = true;
                handled = true;
            }
            // Refused: the worker died (a build panicked, this one included),
            // so fall through to the synchronous build.
        }
        if !handled {
            let generation = res.generation;
            let inline = CancelToken::never();
            if let Ok(Some(volume)) = crate::volume::build_volume(&loaded.tiff, &plan, &inline) {
                upload_volume(renderer, &volume, res, status);
            }
            if res.generation == generation {
                res.built_frame = Some(time);
            }
        }
    }

    renderer.set_volume_interp(view.interp);
    renderer.set_volume_target(view.viewport_px[0], view.viewport_px[1], view.downscale);

    // Per-channel window/level, in the sampled texture's units: normalized the
    // way the volume builder normalized the voxels (integers as unorm, floats
    // over the channel's span), which is
    // [`crate::units::SampleUnits::to_unorm`].
    // Bounded by MAX_CHANNELS, so it lives on the stack — this runs every 3D
    // frame and has no business calling the allocator.
    let blank = VolumeChannel {
        min: 0.0,
        max: 0.0,
        is_float: false,
        enabled: false,
        albedo_t: 1.0,
    };
    let mut windows = [blank; MAX_CHANNELS];
    let n = loaded.display.channel_count().min(MAX_CHANNELS);
    for (c, (slot, s)) in windows
        .iter_mut()
        .zip(loaded.display.settings())
        .enumerate()
    {
        let is_float = s.kind() == ChannelKind::Float;
        let (min, max) = s.units.to_unorm(s.window);
        // The isosurface paints one fixed colour, taken from the LUT's
        // brightest point rather than its top entry — a contrast-stretched
        // palette ends black, and sampling the top would render nothing at all.
        let albedo_t = loaded
            .display
            .lut(c)
            .map_or(1.0, scivis_render::brightest_lut_t);
        *slot = VolumeChannel {
            min,
            max,
            is_float,
            enabled: s.enabled,
            albedo_t,
        };
    }
    let windows = &windows[..n];

    let (w, h) = loaded.dimensions().unwrap_or((1, 1));
    let cam = view.camera((w, h, loaded.volume_depth()));
    // Cache the box extents so the orbit re-pivot can ray-cast the box.
    view.cam.box_he = cam.box_he;
    // The gain, decided here rather than in the shader because which of the
    // three answers applies depends on the mode — see [`crate::brightness`].
    let brightness = match view.render {
        scivis_render::VolumeRender::Mip => view.mip_brightness,
        scivis_render::VolumeRender::Alpha if view.volume_autolevel => {
            crate::brightness::autolevel(view.density)
        }
        scivis_render::VolumeRender::Alpha => view.volume_brightness,
        // Surface returns a shaded hit rather than an accumulation, and has its
        // own shading constants.
        scivis_render::VolumeRender::Surface => 1.0,
    };
    let mut params = build_volume_params(
        &cam,
        windows,
        view.aspect,
        view.render,
        view.density,
        view.iso,
        brightness,
    );
    // However large the volume and the panel, a frame stays a bounded amount
    // of GPU work: see `scivis_render::FRAME_SAMPLE_BUDGET`.
    params.max_steps = view.max_steps(windows.len());
    renderer.set_volume_params(params);
    // A rebuild started this frame (the GPU ran out of memory): its loading
    // and its result need frames of their own, wherever the view is.
    needs_repaint || res.generation != generation
}

/// Snapshot everything the volume builder needs (see [`VolumePlan`]): the
/// dimensions come from the stack's (possibly manually overridden) metadata so a
/// channels/frames swap is honored, `time` is the 4D timepoint to build, and
/// `budget` the bytes it may take on the GPU.
pub fn plan_volume(
    loaded: &Document,
    renderer: &Renderer,
    time: usize,
    budget: usize,
    view: &VolumeView,
) -> VolumePlan {
    // The volume format each channel's kind maps to, on the stack: bounded by
    // MAX_CHANNELS, and asked for on every 3D frame a build is outstanding.
    let masks = loaded.display.masks();
    let mut volume_kinds = [scivis_render::VolumeKind::U8; MAX_CHANNELS];
    for (slot, kind) in volume_kinds.iter_mut().zip(masks.kinds()) {
        *slot = kind.volume_kind();
    }
    let volume_kinds = &volume_kinds[..masks.len()];
    let dims = loaded.display.dims();
    VolumePlan {
        format: renderer.volume_texel_format(scivis_render::pack::common_kind(volume_kinds)),
        units: loaded.display.settings().iter().map(|s| s.units).collect(),
        rgb: loaded.display.rgb(),
        channels: dims.channels,
        slices: dims.slices,
        frames: dims.frames,
        time,
        max_dim: renderer.max_3d_texture_size(),
        budget,
        scale: view.scale,
        detail_cap: crate::volume::detail_cap(view.screen_px),
    }
}

/// The share of the GPU memory the renderer reports free that a volume may
/// take; the rest is left for the frame textures, the window and everyone
/// else on the GPU.
pub const VOLUME_SHARE_OF_FREE: f64 = 0.5;

/// The bytes a volume may take on the GPU: half of what the renderer reports
/// free (the current volume counted as free, since it is replaced), or
/// [`crate::volume::DEFAULT_VOLUME_BUDGET`] when it cannot tell; within
/// [`crate::volume::MIN_VOLUME_BUDGET`]..=[`crate::volume::MAX_VOLUME_BUDGET`],
/// and no more than `cap` (set after an upload ran out of memory).
pub fn volume_budget(renderer: &Renderer, cap: Option<usize>) -> usize {
    let base = renderer
        .volume_memory_available()
        .map(|free| (free as f64 * VOLUME_SHARE_OF_FREE) as usize)
        .unwrap_or(crate::volume::DEFAULT_VOLUME_BUDGET);
    base.clamp(
        crate::volume::MIN_VOLUME_BUDGET,
        crate::volume::MAX_VOLUME_BUDGET,
    )
    .min(cap.unwrap_or(usize::MAX))
}

/// Upload a built volume. When the GPU ran out of memory, rebuild it at no
/// more than half the size that failed (and say so), unless that is below the
/// smallest budget there is.
fn upload_volume(
    renderer: &mut Renderer,
    volume: &crate::volume::BuiltVolume,
    res: &mut VolumeResidency,
    status: &mut Option<String>,
) {
    match renderer.upload_packed_volume(volume) {
        Ok(()) => res.uploaded_generation = Some(res.generation),
        Err(e @ scivis_render::VolumeUploadError::OutOfMemory(_)) => {
            out_of_memory(res, status, &e, volume.layout.total_bytes());
        }
        Err(e) => *status = Some(format!("3D: {e}")),
    }
}

/// A volume of `bytes` did not fit in GPU memory: rebuild it at no more than
/// half that, and say so, unless that is below the smallest budget there is.
///
/// Half of what failed, not half the budget: a volume smaller than its budget
/// (the stack is that small) or than the ceiling would otherwise come back
/// the same size and fail the same way.
fn out_of_memory(
    res: &mut VolumeResidency,
    status: &mut Option<String>,
    why: &scivis_render::VolumeUploadError,
    bytes: usize,
) {
    let cap = bytes / 2;
    if cap < crate::volume::MIN_VOLUME_BUDGET {
        *status = Some(format!(
            "3D: not enough GPU memory even for a small volume ({why})"
        ));
        return;
    }
    log::warn!("3D volume upload ran out of GPU memory ({why}); rebuilding smaller");
    *status =
        Some("3D: the volume did not fit in GPU memory; showing it at a lower resolution".into());
    res.budget_cap = Some(cap);
    res.rebuild();
}

/// Whether a prefetched result still matches the wanted jobs (same channels, in
/// order, with matching kind + dimensions). The revision/frame check happens
/// first; this guards against any residual layout mismatch before upload.
pub fn prefetch_matches(result: &PrefetchResult, jobs: &[ChannelJob]) -> bool {
    result.channels.len() == jobs.len()
        && result.channels.iter().zip(jobs).all(|(ch, job)| {
            ch.channel == job.channel
                && ch.kind == job.kind
                && ch.width == job.width
                && ch.height == job.height
        })
}

//! suite2p stabilization: motion-correct a timelapse.
//!
//! Living tissue moves under the objective — breathing, heartbeat, the animal
//! shifting. Over a few minutes a field drifts by tens of pixels, and anything
//! measured per-pixel afterwards is then measuring a different piece of tissue
//! at each timepoint. This puts the frames back on top of each other first.
//!
//! # The algorithm is suite2p's
//!
//! [`suite2p_registration`] is the port; this file is only the plugin around
//! it — the dialog, reading planes, and assembling the result. Every parameter
//! below carries the name and the default it has in suite2p's `default_ops`, so
//! a value copied from a lab's `ops.npy` means the same thing here.
//!
//! # Reference by one channel, applied to all
//!
//! A two-channel recording is one movie photographed twice at once, so both
//! channels move together. The shifts are measured on the channel the dialog
//! names — usually the structural one, which is brighter and steadier — and
//! applied to every channel, exactly as suite2p's `align_by_chan` does. Aligning
//! each channel independently would move them relative to each other, which is
//! the one thing a two-channel recording must not do.

mod params;

use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, Params, PixelType, Plane, PlaneData, Plugin,
    PluginError, PluginInfo,
};
use suite2p_registration::{
    compute_reference, fft::Fft2, masks::reference_filters_normed, nonrigid, shift_frame, Frames,
    Settings, Shift,
};

/// Correct a chunk of read planes and move them into the result.
///
/// Split out only because it is called from two places — once when the chunk
/// fills and once for whatever is left at the end of a batch — and getting one
/// of those wrong loses planes off the end of the recording.
fn flush(
    pending: &mut Vec<Vec<f32>>,
    at: &mut Vec<usize>,
    out: &mut Vec<PlaneData>,
    ly: usize,
    lx: usize,
    shifts: &[Shift],
    warp: Option<(&nonrigid::Blocks, &[Vec<nonrigid::BlockShift>])>,
    settings: &Settings,
) {
    if pending.is_empty() {
        return;
    }
    suite2p_registration::pipeline::apply_batch(pending, ly, lx, at, shifts, warp, settings);
    out.extend(pending.drain(..).map(PlaneData::F32));
    at.clear();
}

/// Motion correction for a timelapse, by suite2p's method.
#[derive(Default)]
pub struct Stabilize;

impl Plugin for Stabilize {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.stabilize", "suite2p stabilization")
            .menu_path("Stabilization")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Rigid motion correction by phase correlation, ported from suite2p. \
                 Measures a shift per frame against a reference built from the \
                 recording itself.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        params::declare(&host.image())
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        // The time axis is what moves. A single frame has no motion to correct,
        // and a Z stack's slices are different depths rather than the same
        // plane at different times — registering those would be aligning
        // anatomy to itself.
        if info.frames < 2 {
            return Err(PluginError::unsupported(
                "stabilization needs a time series: this stack has one timepoint",
            ));
        }
        let (ly, lx) = (info.height as usize, info.width as usize);
        if ly == 0 || lx == 0 {
            return Err(PluginError::unsupported("this stack has no pixels"));
        }

        let settings = params::settings_from(params);
        // Refused, not quietly downgraded. Someone who picked a device and got
        // the CPU instead would have no way to tell the difference from a slow
        // GPU run.
        if let Some(why) = settings
            .backend
            .unavailable_reason(info.height as usize, info.width as usize)
        {
            return Err(PluginError::unsupported(why));
        }
        // suite2p's `align_by_chan2` is 1-based `align_by_chan` under the hood;
        // here it is a switch, because a recording has one or two channels and
        // "the other one" is the only choice anyone makes.
        let align_by = usize::from(settings.align_by_chan2).min(info.channels.saturating_sub(1));

        // The reference is built from a sample rather than the whole recording:
        // 300 frames is plenty to find what the field looks like at rest, and a
        // twenty-minute recording is tens of thousands. Reading all of them to
        // build one average would be the slowest part of the run by far.
        let step = (info.frames / settings.nimg_init).max(1);
        let mut sample: Vec<Vec<f32>> = Vec::new();
        let mut plane = Vec::new();
        for t in (0..info.frames).step_by(step) {
            if !host.progress(0.1 * sample.len() as f32 / settings.nimg_init as f32) {
                return Ok(Outcome::Cancelled);
            }
            host.read_plane_f32(Plane::new(align_by, 0, t), &mut plane)?;
            sample.push(plane.clone());
        }
        let bidi = if settings.do_bidiphase && settings.bidiphase == 0 {
            suite2p_registration::bidiphase::compute(&sample, ly, lx)
        } else {
            settings.bidiphase
        };
        if bidi != 0 {
            for f in sample.iter_mut() {
                suite2p_registration::bidiphase::shift(f, ly, lx, bidi);
            }
        }

        let sample_movie = Frames {
            ly,
            lx,
            frames: &sample,
        };
        let Some(reference) = compute_reference(&sample_movie, &settings, &mut |f| {
            host.progress(0.1 + 0.2 * f)
        }) else {
            return Ok(Outcome::Cancelled);
        };
        drop(sample);

        // Now one pass over the recording: measure each timepoint's shift on
        // the chosen channel, then apply it to every channel of that timepoint.
        // Streamed rather than held, so the movie is never resident twice.
        let mut fft = Fft2::new(ly, lx);
        let filters = reference_filters_normed(
            &mut fft,
            &reference,
            ly,
            lx,
            settings.spatial_taper,
            settings.smooth_sigma,
            settings.norm_frames,
        );

        // The deformation grid, when one was asked for. Built once from the
        // reference: every frame's blocks are the same blocks, which is what
        // makes the per-block shifts comparable between frames.
        let mut nonrigid = settings.nonrigid.then(|| {
            let blocks = nonrigid::make_blocks(ly, lx, settings.block_size, settings.subpixel);
            let filters = nonrigid::block_filters(
                &reference,
                lx,
                &blocks,
                settings.spatial_taper,
                settings.smooth_sigma,
            );
            (blocks, filters)
        });
        let clip = settings
            .norm_frames
            .then(|| suite2p_registration::pipeline::percentile_range(&reference));

        let slices = info.slices.max(1);
        let channels = info.channels.max(1);
        let mut planes: Vec<PlaneData> = Vec::with_capacity(channels * slices * info.frames);
        let mut shifts: Vec<Shift> = Vec::with_capacity(info.frames);
        let mut plane = Vec::new();

        // One pass over the recording, a batch at a time.
        //
        // A batch rather than a frame, because `measure_batch` is where the
        // chosen backend lives — measuring frame by frame here, which is what
        // this did, ran the whole recording on one thread whatever the dialog
        // said, and made the three backends indistinguishable. It is also what
        // `smooth_sigma_time` smooths across, so a frame-at-a-time pass ignored
        // that setting entirely on the recording while honouring it on the
        // reference.
        // Bounded by memory as well as by the setting: a batch is held as
        // `f32`, and 100 frames of a 2048-pixel-square recording is a gigabyte
        // and a half. At the default frame size the budget is far larger than
        // the default batch, so the common case is exactly what was asked for.
        const BATCH_BUDGET: usize = 256 << 20;
        let per_frame = (ly * lx * std::mem::size_of::<f32>()).max(1);
        let batch = settings
            .batch_size
            .max(1)
            .min((BATCH_BUDGET / per_frame).max(1));
        if batch < settings.batch_size && settings.smooth_sigma_time > 0.0 {
            // Only worth saying when it changes the answer. `smooth_sigma_time`
            // smooths within a batch, so a smaller batch is a shorter window;
            // with it off, the batch is a scheduling detail and nothing else.
            host.log(&format!(
                "batch size reduced from {} to {batch} to bound memory at this frame size;                  smooth_sigma_time smooths within a batch, so its window is shorter",
                settings.batch_size
            ));
        }
        // How many planes are resampled in one parallel sweep. A bound rather
        // than the whole batch: a recording with several channels and slices has
        // many planes per timepoint, and a batch of all of them at once would be
        // an unpredictable amount of memory.
        const APPLY_CHUNK: usize = 64;

        let mut batch_frames: Vec<Vec<f32>> = Vec::new();
        let mut pending: Vec<Vec<f32>> = Vec::new();
        let mut pending_at: Vec<usize> = Vec::new();

        let mut t0 = 0;
        while t0 < info.frames {
            let t1 = (t0 + batch).min(info.frames);
            if !host.progress(0.3 + 0.7 * t0 as f32 / info.frames as f32) {
                return Ok(Outcome::Cancelled);
            }

            // The measurement channel for this batch.
            batch_frames.clear();
            for t in t0..t1 {
                host.read_plane_f32(Plane::new(align_by, 0, t), &mut plane)?;
                if bidi != 0 {
                    suite2p_registration::bidiphase::shift(&mut plane, ly, lx, bidi);
                }
                batch_frames.push(std::mem::take(&mut plane));
            }

            let batch_shifts = suite2p_registration::pipeline::measure_batch(
                ly,
                lx,
                &filters,
                &batch_frames,
                &settings,
            );

            // The deformation is measured on the *rigidly corrected* frame, so a
            // block is looking for the few pixels the tissue stretched rather
            // than the tens the animal moved — a 64-pixel block cannot see 50
            // pixels of travel.
            //
            // Serial, unlike everything around it: each block carries its own
            // FFT plan and scratch, so a second thread would need a second set
            // of them. Non-rigid is off by default and this is the price of
            // turning it on.
            let fields = nonrigid.as_mut().map(|(blocks, block_filters)| {
                batch_frames
                    .iter()
                    .zip(&batch_shifts)
                    .map(|(frame, s)| {
                        let corrected = shift_frame(frame, ly, lx, s.dy, s.dx);
                        nonrigid::measure_blocks(
                            block_filters,
                            &corrected,
                            lx,
                            blocks,
                            &nonrigid::BlockSearch {
                                maxregshift_nr: settings.maxregshift_nr,
                                snr_thresh: settings.snr_thresh,
                                subpixel: settings.subpixel,
                                clip,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            });
            let warp = nonrigid
                .as_ref()
                .zip(fields.as_ref())
                .map(|((blocks, _), f)| (blocks, f.as_slice()));

            // Now the planes themselves. xyczt: channel fastest, then z, then t
            // — the order the host expects a result's planes in.
            for t in t0..t1 {
                for z in 0..slices {
                    for c in 0..channels {
                        // The measurement channel of a plain single-channel,
                        // single-slice recording is the plane being corrected,
                        // and it is already here. Re-reading and re-converting
                        // it is a second pass over the whole recording for
                        // nothing, and that is the common case.
                        if channels == 1 && slices == 1 && c == align_by {
                            pending.push(std::mem::take(&mut batch_frames[t - t0]));
                        } else {
                            host.read_plane_f32(Plane::new(c, z, t), &mut plane)?;
                            if bidi != 0 {
                                suite2p_registration::bidiphase::shift(&mut plane, ly, lx, bidi);
                            }
                            pending.push(std::mem::take(&mut plane));
                        }
                        pending_at.push(t - t0);
                        if pending.len() >= APPLY_CHUNK {
                            flush(
                                &mut pending,
                                &mut pending_at,
                                &mut planes,
                                ly,
                                lx,
                                &batch_shifts,
                                warp,
                                &settings,
                            );
                        }
                    }
                }
            }
            flush(
                &mut pending,
                &mut pending_at,
                &mut planes,
                ly,
                lx,
                &batch_shifts,
                warp,
                &settings,
            );

            shifts.extend_from_slice(&batch_shifts);
            t0 = t1;
        }

        let moved = shifts.iter().filter(|s| s.dy != 0 || s.dx != 0).count();
        let worst = shifts
            .iter()
            .map(|s| s.dy.abs().max(s.dx.abs()))
            .max()
            .unwrap_or(0);
        host.log(&format!(
            "registered {} frames on channel {}{}; {moved} moved, largest shift {worst} px{}",
            info.frames,
            align_by + 1,
            match &nonrigid {
                Some((b, _)) => format!(", non-rigid over {} blocks", b.blocks.len()),
                None => String::new(),
            },
            if bidi != 0 {
                format!("; bidirectional offset {bidi} px")
            } else {
                String::new()
            }
        ));

        Ok(Outcome::NewDocument(Box::new(ImageResult {
            width: info.width,
            height: info.height,
            channels,
            slices,
            frames: info.frames,
            // Float, because a shifted frame is the file's own samples moved,
            // and rounding them back to integers here would quantise every
            // pixel a second time for no gain.
            pixel_type: PixelType::F32,
            planes,
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-stabilized", host.stack_info().name),
        })))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

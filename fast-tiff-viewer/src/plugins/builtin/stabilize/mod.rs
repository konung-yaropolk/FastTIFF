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
//! The run is a feeder of the crate's [`Session`]: it reads the frames the
//! session names, reports progress, honours a cancel and stores what comes
//! back. Which frames build the reference, the comb, the batches and the
//! window they give the temporal smoothing, a second pass, the block field
//! and the bad frames are the session's, so the viewer and the C library
//! give the same answer for the same frames and settings. This used to
//! sequence the crate's stages itself, and drifted: it once measured frame by
//! frame on one thread, bounded its batches where the crate did not, never
//! ran the second pass of a two-step registration and never counted a bad
//! frame.
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

use super::shared::{self, Store, Stored};
use fasttiff_plugin_api::{
    HostContext, ImageSpec, ImageStream, Outcome, ParamDecl, Params, Plane, Plugin, PluginError,
    PluginInfo, StreamTarget,
};
use suite2p_registration::{work, Border, Measured, Registered, Session};

/// How many planes are corrected in one parallel sweep. A bound rather than
/// a whole batch: a recording with several channels and slices has many planes
/// per timepoint, and a batch of all of them at once would be an unpredictable
/// amount of memory.
const APPLY_CHUNK: usize = 64;

/// How a chunk of planes is put back where it belongs.
#[derive(Clone, Copy)]
enum Correction<'a, 's> {
    /// By the batch the last pass has just measured, at full size.
    Batch(&'a Measured<'s>),
    /// By the finished registration, moved and cut to the common area: a
    /// crop's area depends on every frame, so its planes wait for the end.
    Crop(&'a Registered, usize, usize),
}

/// The result as it is handed to the host: planes, as recorded, gathered into
/// chunks, corrected a chunk at a time, stored at the file's width and pushed.
///
/// A chunk is the most of the result this plugin ever holds: each corrected
/// plane is encoded into `stored` and pushed, and the host has written it
/// before the next one is made.
struct Writer {
    store: Store,
    stored: Stored,
    pending: Vec<Vec<f32>>,
    at: Vec<usize>,
}

impl Writer {
    /// Queue `plane`, of frame `t`, and correct and push the chunk once it is
    /// full. How many planes went out, for the progress bar.
    fn put(
        &mut self,
        plane: Vec<f32>,
        t: usize,
        how: Correction<'_, '_>,
        host: &mut dyn HostContext,
        out: &mut ImageStream,
    ) -> Result<usize, PluginError> {
        self.pending.push(plane);
        self.at.push(t);
        if self.pending.len() < APPLY_CHUNK {
            return Ok(0);
        }
        self.flush(how, host, out)
    }

    /// Correct and push whatever is queued. Called when a chunk fills and
    /// once for what is left at the end of a batch, and getting the second
    /// wrong loses planes off the end of the recording.
    fn flush(
        &mut self,
        how: Correction<'_, '_>,
        host: &mut dyn HostContext,
        out: &mut ImageStream,
    ) -> Result<usize, PluginError> {
        if self.pending.is_empty() {
            return Ok(0);
        }
        match how {
            Correction::Batch(measured) => measured.apply(&mut self.pending, &self.at),
            Correction::Crop(registered, ly, lx) => registered
                .apply_batch(&mut self.pending, ly, lx, &self.at)
                .map_err(no_common_area)?,
        }
        let n = self.pending.len();
        for v in self.pending.drain(..) {
            out.push(host, self.store.encode(&v, &mut self.stored))?;
        }
        self.at.clear();
        Ok(n)
    }
}

fn no_common_area(none: suite2p_registration::NoCommonArea) -> PluginError {
    shared::no_common_area(
        none,
        "a smaller Max shift — and Max block shift for a non-rigid run — keeps the search in \
         bounds",
    )
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
        params::declare(&host.image(), host.pending_params())
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        // The GPU context suite2p keeps between batches goes when the run
        // does, however it ends.
        #[cfg(feature = "gpu-registration")]
        let _release = ReleaseGpuOnDrop;
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
        if let Some(why) = settings.backend.unavailable_reason(ly, lx) {
            return Err(PluginError::unsupported(why));
        }
        // suite2p's `align_by_chan2` is 1-based `align_by_chan` under the hood;
        // here it is a switch, because a recording has one or two channels and
        // "the other one" is the only choice anyone makes.
        let align_by = usize::from(settings.align_by_chan2).min(info.channels.saturating_sub(1));

        let mut session = Session::new(ly, lx, info.frames, &settings);
        // The batch is bounded by memory in the crate, for every caller alike;
        // when that shortens the temporal smoothing's window, it is said.
        if let Some(note) = session.shortened_window() {
            host.log(&note);
        }
        let passes = session.passes();

        // One bar for the whole run, divided by how long each stage takes
        // rather than by how many stages there are — see `work`. Split evenly,
        // it gave a tenth to reading the sample (a blink), a fifth to building
        // the reference (most of the run) and the rest to the pass over the
        // recording, so it raced, froze for seconds, and raced again.
        let slices = info.slices.max(1);
        let channels = info.channels.max(1);
        let planes_per_frame = channels * slices;
        // A single-plane recording's measurement plane is the one corrected,
        // so it is not read a second time. See the pass below. A crop reads
        // every plane again: it moves them only once every frame is measured.
        let cropping = settings.border == Border::Crop;
        let reread = planes_per_frame > 1 || cropping;
        let plane_work = if settings.nonrigid {
            work::WARP
        } else {
            work::SHIFT
        } + if reread { work::READ } else { 0.0 };
        let sample_len = session.reference_frames().len();
        let reading = sample_len as f64 * work::READ;
        let referencing = work::reference(sample_len, settings.reference_iterations);
        let frames = info.frames as f64;
        // Every pass reads the sample, builds a reference and measures every
        // frame; the last also measures the deformation and moves the planes.
        let per_pass = reading + referencing + frames * (work::READ + work::CORRELATION);
        let blocks = if settings.nonrigid { work::BLOCKS } else { 0.0 };
        let last_pass = frames * (blocks + planes_per_frame as f64 * plane_work);
        let total = (passes as f64 * per_pass + last_pass).max(f64::MIN_POSITIVE);
        let at = move |done: f64| (done / total) as f32;

        // How a corrected plane is stored in the result (`shared::Store`).
        //
        // **The file's own width, always.** A stabilised recording is the recording
        // with its frames moved: a 16-bit acquisition should come back 16-bit, the
        // same size, and open in whatever opened the original. Widening it to float
        // doubles every byte of it — on disk, through the handover to the new window,
        // and in that window's memory for as long as it is open.
        //
        // What that costs depends on the correction:
        //
        // * A **rigid** run is `np.roll`. Every sample is the file's own, moved; no
        //   arithmetic touches a value, and `f32` carries every `u8`, `u16` and `i16`
        //   exactly. Nothing is lost at all.
        // * A **non-rigid** run interpolates between pixels, so it makes values that
        //   were never in the file, and storing them at the source's width rounds
        //   them — by at most half a sample step. suite2p does the same: its own
        //   `reg_tif` output is `int16`, not float.
        //
        // A float source stays float, having nothing to round to.
        let store = Store::of(info.pixel_type);
        let mut writer = Writer {
            store,
            stored: Stored::default(),
            pending: Vec::new(),
            at: Vec::new(),
        };
        // The result, declared up front and handed to the host a plane at a
        // time as each is corrected: it is the recording's size, and holding
        // it to hand back at the end is what made a long recording need two or
        // three times its size in memory. The file's metadata goes with it — a
        // stabilised recording is the recording, so its calibration still
        // holds. A crop's frame is not known until every frame is measured, so
        // that result is begun then.
        let source = host.stack_info().clone();
        let name = format!("{}-stabilized", source.stem());
        let spec_of = |width: u32, height: u32| ImageSpec {
            width,
            height,
            channels,
            slices,
            frames: info.frames,
            // The file's own width back again for a rigid run, float for one
            // that interpolated. See [`Store`].
            pixel_type: store.pixel_type(),
            channel_colors: Vec::new(),
            metadata: Some(source.clone()),
            name: name.clone(),
        };
        let mut out = if cropping {
            None
        } else {
            Some(ImageStream::begin(host, spec_of(info.width, info.height))?)
        };

        // An earlier run's trouble is not this one's.
        #[cfg(feature = "gpu-registration")]
        let _ = suite2p_registration::gpu::take_trouble();

        // The measurement channel of the batch being measured, as read — and
        // as recorded again once measured. Its buffers are kept from batch to
        // batch, except a single-plane recording's on the last pass, which go
        // on into the result.
        let mut batch_frames: Vec<Vec<f32>> = Vec::new();
        // Work done so far, in `work` units, reported as each piece lands.
        let mut done = 0.0;
        // Once, or twice for a two-step registration: a reference from the
        // frames the session names, then the recording a batch at a time.
        while session.needs_reference() {
            // A second pass's reference is built with the first pass's batch
            // buffers idle; held through it, they were a batch more at the
            // run's peak, which is the sample and the copy the pick makes.
            batch_frames.clear();
            let mut sample: Vec<Vec<f32>> = Vec::with_capacity(sample_len);
            for &t in session.reference_frames() {
                if !host.progress(at(done)) {
                    return Ok(Outcome::Cancelled);
                }
                let mut plane = Vec::new();
                host.read_plane_f32(Plane::new(align_by, 0, t), &mut plane)?;
                sample.push(plane);
                done += work::READ;
            }
            let base = done;
            let built = session.build_reference(sample, &mut |f| {
                host.progress(at(base + f as f64 * referencing))
            });
            if built.is_none() {
                return Ok(Outcome::Cancelled);
            }
            done = base + referencing;
            gpu_trouble(&settings)?;

            while let Some(batch) = session.next_batch() {
                let frames_here = batch.len() as f64;
                if !host.progress(at(done)) {
                    return Ok(Outcome::Cancelled);
                }
                batch_frames.resize_with(batch.len(), Vec::new);
                for (buf, t) in batch_frames.iter_mut().zip(batch.clone()) {
                    host.read_plane_f32(Plane::new(align_by, 0, t), buf)?;
                }
                done += frames_here * work::READ;
                host.progress(at(done));

                // In place: these buffers are the plugin's own, so the session
                // undoes the comb (or a first pass's shift) in them and hands
                // them back as recorded, rather than copying the batch to
                // correct it. A copy was a second batch held, and with a
                // small `nimg_init` that was the run's peak.
                let measured = session.measure_in_place(&mut batch_frames);
                gpu_trouble(&settings)?;
                done += frames_here * work::CORRELATION;
                if !measured.is_final() {
                    continue;
                }
                done += frames_here * blocks;
                if !host.progress(at(done)) {
                    return Ok(Outcome::Cancelled);
                }
                let Some(stream) = out.as_mut() else {
                    // A crop moves its planes once every frame is measured.
                    continue;
                };

                // Now the planes themselves, as recorded: the batch undoes the
                // comb as it moves them. xyczt — channel fastest, then z, then
                // t — the order the host expects a result's planes in.
                let how = Correction::Batch(&measured);
                for t in batch.clone() {
                    for z in 0..slices {
                        for c in 0..channels {
                            // The measurement channel of a plain single-channel,
                            // single-slice recording is the plane being
                            // corrected, and it is already here. Re-reading and
                            // re-converting it is a second pass over the whole
                            // recording for nothing, and that is the common
                            // case.
                            let plane = if planes_per_frame == 1 {
                                std::mem::take(&mut batch_frames[t - batch.start])
                            } else {
                                let mut plane = Vec::new();
                                host.read_plane_f32(Plane::new(c, z, t), &mut plane)?;
                                plane
                            };
                            let pushed = writer.put(plane, t, how, host, stream)?;
                            if pushed > 0 {
                                done += pushed as f64 * plane_work;
                                if !host.progress(at(done)) {
                                    return Ok(Outcome::Cancelled);
                                }
                            }
                        }
                    }
                }
                done += writer.flush(how, host, stream)? as f64 * plane_work;
            }
        }
        drop(batch_frames);
        let registered = session.finish();

        // A crop: every frame measured, so the area is known — then every
        // plane read again, moved and cut in one resampling.
        if cropping {
            let area = registered.common_area(ly, lx).map_err(no_common_area)?;
            // The cut frame's size is known now, so the result begins here.
            let stream = out.insert(ImageStream::begin(
                host,
                spec_of(area.width as u32, area.height as u32),
            )?);
            let how = Correction::Crop(&registered, ly, lx);
            for t in 0..info.frames {
                for z in 0..slices {
                    for c in 0..channels {
                        let mut plane = Vec::new();
                        host.read_plane_f32(Plane::new(c, z, t), &mut plane)?;
                        let pushed = writer.put(plane, t, how, host, stream)?;
                        if pushed > 0 {
                            done += pushed as f64 * plane_work;
                            if !host.progress(at(done)) {
                                return Ok(Outcome::Cancelled);
                            }
                        }
                    }
                }
            }
            writer.flush(how, host, stream)?;
            host.log(&if area.is_upright() {
                format!(
                    "cropped to the common area: {} x {} pixels from ({}, {})",
                    area.width, area.height, area.x, area.y
                )
            } else {
                format!(
                    "cropped to the common area: {} x {} pixels, turned {:.1} degrees, from \
                     ({:.1}, {:.1})",
                    area.width,
                    area.height,
                    area.angle.to_degrees(),
                    area.x,
                    area.y
                )
            });
        }
        // The estimate only divides the bar; the end of the work is the end of
        // it, whatever the sum came to.
        host.progress(1.0);
        host.log(&summary(&registered, align_by, passes));

        out.expect("begun before the planes, or once a crop's area was known")
            .finish(StreamTarget::NewDocument)
    }
}

/// What the run did, in a line: how much moved, and how much of the
/// recording looks bad — counted, never dropped, which the dialog promises.
fn summary(registered: &Registered, align_by: usize, passes: usize) -> String {
    let shifts = &registered.shifts;
    let moved = shifts.iter().filter(|s| s.dy != 0 || s.dx != 0).count();
    let worst = shifts
        .iter()
        .map(|s| s.dy.abs().max(s.dx.abs()))
        .max()
        .unwrap_or(0);
    let bad = registered.bad_frames.iter().filter(|&&b| b).count();
    format!(
        "registered {} frames on channel {}{}{}; {moved} moved, largest shift {worst} px{}; \
         {bad} flagged as bad (kept)",
        shifts.len(),
        align_by + 1,
        if passes > 1 { ", in two passes" } else { "" },
        match &registered.nonrigid {
            Some(nr) => format!(", non-rigid over {} blocks", nr.blocks.blocks.len()),
            None => String::new(),
        },
        if registered.bidiphase != 0 {
            format!("; bidirectional offset {} px", registered.bidiphase)
        } else {
            String::new()
        }
    )
}

/// Stop the run, saying why, if the device failed part of it.
///
/// A batch the device could not take goes to the processor, and one it lost
/// part-way comes back as no shift at all — quietly, inside the registration.
/// Someone who chose the GPU would get either a CPU run or unregistered frames
/// with no way to tell, so the device's own account of it ends the run.
fn gpu_trouble(settings: &suite2p_registration::Settings) -> Result<(), PluginError> {
    #[cfg(feature = "gpu-registration")]
    if settings.backend == suite2p_registration::Backend::Gpu {
        if let Some(why) = suite2p_registration::gpu::take_trouble() {
            return Err(PluginError::failed(format!(
                "stabilizing on the GPU failed: {why}. {} runs on the processor and \
                 needs no GPU memory.",
                suite2p_registration::Backend::MultiThread.label()
            )));
        }
    }
    let _ = settings;
    Ok(())
}

/// Lets go of suite2p's resident GPU context when dropped: the batch buffers
/// a run kept between batches are of no use once it has ended, and on a small
/// card they are memory the renderer and the next run need.
#[cfg(feature = "gpu-registration")]
struct ReleaseGpuOnDrop;

#[cfg(feature = "gpu-registration")]
impl Drop for ReleaseGpuOnDrop {
    fn drop(&mut self) {
        suite2p_registration::gpu::release_resident();
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

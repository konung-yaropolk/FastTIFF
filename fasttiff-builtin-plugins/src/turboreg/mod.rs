//! TurboReg stabilization: take rotation and distortion out of a series.
//!
//! The sibling [`stabilize`] plugin answers a narrower question faster. It
//! measures a *shift* by phase correlation, which is the right tool for a
//! two-photon timelapse that drifted under the objective. This one solves for a
//! *transformation* — rotation, scale, shear, or a bilinear warp — by
//! optimisation, and costs more per frame to do it.
//!
//! Reach for this one when the frames are not merely displaced: a histology
//! series where each section was placed on the slide by hand, a widefield
//! recording where the stage rotated, a time-lapse across a objective change.
//!
//! [`stabilize`]: crate::stabilize
//!
//! # The algorithm is EPFL's
//!
//! [`turboreg_registration`] is the implementation; this file is only the
//! plugin around it — the dialog, reading the planes the crate's
//! [`Session`] asks for, and streaming the result. What is fitted against
//! what, in what order and in batches of what size, and whether the GPU lost
//! frames on the way, is the session's to decide, as it is for the C
//! interface and `register_stack`, so the three cannot drift apart; and the
//! recording is never held, only a batch of its measurement at a time and then
//! one plane being moved.
//! See that crate's manifest for the licensing, which is not the obvious one.
//!
//! # Measured on one channel, applied to all
//!
//! A multi-channel recording is one specimen photographed several ways at once,
//! so every channel moved together. The transformation is measured on the
//! channel the dialog names — or on the mean of them, which is the default
//! because averaging raises what the search is matching — and applied to every
//! channel. Registering each channel independently would move them relative to
//! each other, which is the one thing a multi-channel recording must not do.

mod params;

use super::shared::{self, Store, Stored};
use fasttiff_plugin_api::{
    HostContext, ImageSpec, ImageStream, Outcome, ParamDecl, Params, Plane, Plugin, PluginError,
    PluginInfo,
};
use turboreg_registration::{Backend, Batch, Error, Pass, Reference, Session, Wanted};

/// Where a run went, for the log line — naming the card for a GPU run, since
/// "GPU" on a machine with an integrated and a discrete one does not say which.
fn ran_on(backend: Backend) -> String {
    match backend {
        Backend::SingleThread => "on one CPU thread".to_string(),
        Backend::MultiThread => "on every CPU core".to_string(),
        Backend::Gpu => {
            #[cfg(feature = "gpu-registration")]
            if let Some(name) = turboreg_registration::gpu::adapter_name() {
                return format!("on the GPU ({name})");
            }
            "on the GPU".to_string()
        }
    }
}

/// Relative cost of the pieces of a run, for dividing one progress bar.
///
/// Reading a plane is the unit. Registering a frame is the expensive part by a
/// long way — a pyramid of spline fits against tens of iterations each — and
/// resampling one is in between. Split evenly instead, the bar raced through
/// the reading and then sat still for the registration, which reads as a hang.
mod work {
    pub(super) const READ: f64 = 1.0;
    pub(super) const REGISTER: f64 = 40.0;
    pub(super) const WARP: f64 = 6.0;
}

/// The measurement image for timepoint `t` into `out`: one channel's plane, or
/// the mean of all of them. `scratch` and `sum` are the mean's working space,
/// kept by the caller so that a batch after the first allocates nothing.
fn measurement(
    host: &mut dyn HostContext,
    channel: Option<usize>,
    channels: usize,
    t: usize,
    out: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    sum: &mut Vec<f64>,
) -> Result<(), PluginError> {
    match channel {
        Some(c) => host.read_plane_f32(Plane::new(c, 0, t), out),
        // The mean of one channel is the channel.
        None if channels <= 1 => host.read_plane_f32(Plane::new(0, 0, t), out),
        None => {
            // The mean across channels. Accumulated in f64 so a 16-bit stack
            // with many channels cannot lose its low bits to repeated rounding.
            for c in 0..channels {
                host.read_plane_f32(Plane::new(c, 0, t), scratch)?;
                if c == 0 {
                    sum.clear();
                    sum.extend(scratch.iter().map(|&v| v as f64));
                } else {
                    for (a, &v) in sum.iter_mut().zip(scratch.iter()) {
                        *a += v as f64;
                    }
                }
            }
            let n = channels as f64;
            out.clear();
            out.extend(sum.iter().map(|&v| (v / n) as f32));
            Ok(())
        }
    }
}

/// A registration that could not start, or could not finish, as the run's
/// error.
fn plugin_error(e: Error) -> PluginError {
    match e {
        // The dialog does not offer this combination, but a saved parameter
        // set or a scripted call can still carry it.
        Error::Unchainable => {
            PluginError::unsupported(format!("{e}; use the first frame or the mean"))
        }
        // Refused, not quietly downgraded: someone who picked a backend and
        // got a different one would have no way to tell.
        Error::Unavailable(why) => PluginError::unsupported(why),
        // Frames the device failed are not fitted for want of the device, not
        // because the data would not fit: a stack that came back with them
        // unmoved would be no result, so the run fails and says what will run.
        Error::Device { .. } => PluginError::failed(format!(
            "{e}. {} runs on the processor and needs no GPU memory.",
            Backend::MultiThread.label()
        )),
        other => PluginError::failed(other.to_string()),
    }
}

/// Registration by the TurboReg/StackReg method.
#[derive(Default)]
pub struct TurboReg;

impl Plugin for TurboReg {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.turboreg", "TurboReg stabilization")
            .menu_path("Stabilization")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Rigid-body, affine or bilinear registration by the TurboReg/StackReg \
                 pyramid method. Solves for a transformation per frame, so it takes out \
                 rotation and distortion as well as drift.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        params::declare(&host.image(), host.pending_params())
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        // The time axis is what moves. A Z stack's slices are different depths
        // rather than the same plane at different times, and registering those
        // would be aligning anatomy to itself.
        if info.frames < 2 {
            return Err(PluginError::unsupported(
                "registration needs a time series: this stack has one timepoint",
            ));
        }
        let (ly, lx) = (info.height as usize, info.width as usize);
        if ly == 0 || lx == 0 {
            return Err(PluginError::unsupported("this stack has no pixels"));
        }

        let settings = params::settings_from(params);
        let mut session = Session::new(ly, lx, info.frames, &settings).map_err(plugin_error)?;

        let channels = info.channels.max(1);
        let slices = info.slices.max(1);
        let measure_on = params::measure_channel(params, channels);

        // One bar for the whole run, divided by how long each stage takes:
        // every frame the session asks for is read, the ones it fits are
        // registered, and then every plane is read and moved.
        let reads_per_measurement = measure_on.map_or(channels, |_| 1) as f64;
        let read = reads_per_measurement * work::READ;
        let plan = session.plan();
        let planes = (channels * slices) as f64;
        let total = (plan.reference.len() + plan.fit.len()) as f64 * read
            + plan.fit.len() as f64 * work::REGISTER
            + info.frames as f64 * planes * (work::READ + work::WARP);
        let total = total.max(f64::MIN_POSITIVE);
        let at = move |done: f64| (done / total) as f32;

        // The measurement, a batch at a time, as the session asks for it: the
        // frames of its target first where it builds one, then the stack.
        // Only a batch is ever held; the buffers are the next batch's too.
        let mut batch: Vec<Vec<f32>> = Vec::new();
        let (mut scratch, mut sum) = (Vec::new(), Vec::new());
        let mut done = 0.0;
        while let Some(Wanted { frames, pass }) = session.wanted() {
            let count = frames.len();
            if batch.len() < count {
                batch.resize_with(count, Vec::new);
            }
            for (frame, t) in batch.iter_mut().zip(frames) {
                if !host.progress(at(done)) {
                    return Ok(Outcome::Cancelled);
                }
                measurement(host, measure_on, channels, t, frame, &mut scratch, &mut sum)?;
                done += read;
            }
            if !host.progress(at(done)) {
                return Ok(Outcome::Cancelled);
            }
            session
                .feed(Batch::Split(&batch[..count]))
                .map_err(plugin_error)?;
            if pass == Pass::Fit {
                done += count as f64 * work::REGISTER;
            }
        }
        drop((batch, sum));
        // Frames the GPU failed fail the run here (`Error::Device`): the
        // session takes the device's reasons for its own batches.
        let mut result = session.finish().map_err(plugin_error)?;
        result.border = params::border_from(params);

        // What comes out: the frame, or under a crop the common area — found
        // from every frame's transformation before anything is resampled, so
        // a stack with nothing in common stops here.
        const IF_NOT_REAL: &str = "a fit ran away: a transformation with fewer degrees of \
                                   freedom, or another reference, may hold";
        let (height, width) = result
            .output_size(ly, lx)
            .map_err(|none| shared::no_common_area(none, IF_NOT_REAL))?;
        if result.border == turboreg_registration::Border::Crop {
            if let Ok(area) = result.common_area(ly, lx) {
                host.log(&if area.is_upright() {
                    format!(
                        "cropping to the common area: {} x {} pixels from ({}, {})",
                        area.width, area.height, area.x, area.y
                    )
                } else {
                    format!(
                        "cropping to the common area: {} x {} pixels, turned {:.1} degrees, \
                         from ({:.1}, {:.1})",
                        area.width,
                        area.height,
                        area.angle.to_degrees(),
                        area.x,
                        area.y
                    )
                });
            }
        }

        // Now one pass over the recording, applying each timepoint's
        // transformation to every plane of that timepoint — and handing each
        // moved plane to the host as soon as it exists. The result is the
        // recording's size, and it is never held: the host writes each plane
        // into the result's file and the next one reuses the buffer.
        let store = Store::of(info.pixel_type);
        let spec = ImageSpec {
            width: width as u32,
            height: height as u32,
            channels,
            slices,
            frames: info.frames,
            pixel_type: store.pixel_type(),
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-registered", host.stack_info().stem()),
        };
        let mut out = ImageStream::begin(host, spec)?;
        let mut stored = Stored::default();
        // xyczt: channel fastest, then z, then t — the order the host expects.
        for t in 0..info.frames {
            for z in 0..slices {
                for c in 0..channels {
                    host.read_plane_f32(Plane::new(c, z, t), &mut scratch)?;
                    done += work::READ;
                    let moved = result
                        .apply(&scratch, ly, lx, t)
                        .map_err(|none| shared::no_common_area(none, IF_NOT_REAL))?;
                    done += work::WARP;
                    out.push(host, store.encode(&moved, &mut stored))?;
                }
            }
            if !host.progress(at(done)) {
                return Ok(Outcome::Cancelled);
            }
        }
        host.progress(1.0);

        let moved = result
            .matrices
            .iter()
            .filter(|m| **m != turboreg_registration::Matrix::identity())
            .count();
        host.log(&format!(
            "registered {} frames by {} on {}{}, {}; {moved} moved{}",
            info.frames,
            settings.transformation.label().to_lowercase(),
            match measure_on {
                Some(c) => format!("channel {}", c + 1),
                None => "the channel mean".to_string(),
            },
            match settings.reference {
                Reference::Previous => ", chained to the previous frame",
                Reference::First => ", against the first frame",
                Reference::Mean => ", against the stack mean",
            },
            ran_on(settings.backend),
            if result.failed.is_empty() {
                String::new()
            } else {
                format!(", {} frame(s) could not be fitted", result.failed.len())
            }
        ));

        out.finish(shared::target(params))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

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
//! [`stabilize`]: crate::plugins::builtin::stabilize
//!
//! # The algorithm is EPFL's
//!
//! [`turboreg_registration`] is the implementation; this file is only the
//! plugin around it — the dialog, reading planes, and assembling the result.
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

use super::shared::{self, Store};
use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, Params, Plane, PlaneData, Plugin, PluginError,
    PluginInfo,
};
use turboreg_registration::{register_stack, Frames, Reference, Transformation};

/// Relative cost of the pieces of a run, for dividing one progress bar.
///
/// Reading a plane is the unit. Registering a frame is the expensive part by a
/// long way — a pyramid of spline fits against tens of iterations each — and
/// resampling one is in between. Split evenly instead, the bar raced through
/// the reading and then sat still for the registration, which reads as a hang.
mod work {
    pub const READ: f64 = 1.0;
    pub const REGISTER: f64 = 40.0;
    pub const WARP: f64 = 6.0;
}

/// The measurement image for one timepoint: one channel's plane, or the mean of
/// all of them.
fn measurement(
    host: &mut dyn HostContext,
    channel: Option<usize>,
    channels: usize,
    t: usize,
    scratch: &mut Vec<f32>,
) -> Result<Vec<f32>, PluginError> {
    match channel {
        Some(c) => {
            host.read_plane_f32(Plane::new(c, 0, t), scratch)?;
            Ok(scratch.clone())
        }
        None => {
            // The mean across channels. Accumulated in f64 so a 16-bit stack
            // with many channels cannot lose its low bits to repeated rounding.
            let mut acc: Vec<f64> = Vec::new();
            for c in 0..channels {
                host.read_plane_f32(Plane::new(c, 0, t), scratch)?;
                if acc.is_empty() {
                    acc = scratch.iter().map(|&v| v as f64).collect();
                } else {
                    for (a, &v) in acc.iter_mut().zip(scratch.iter()) {
                        *a += v as f64;
                    }
                }
            }
            let n = channels.max(1) as f64;
            Ok(acc.iter().map(|&v| (v / n) as f32).collect())
        }
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
        // Refused, not quietly downgraded: someone who picked a backend and got
        // a different one would have no way to tell.
        if let Some(why) = settings.backend.unavailable_reason() {
            return Err(PluginError::unsupported(why));
        }
        // The dialog does not offer this combination, but a saved parameter set
        // or a scripted call can still carry it.
        if settings.reference == Reference::Previous
            && settings.transformation == Transformation::Bilinear
        {
            return Err(PluginError::unsupported(
                "a bilinear warp cannot be chained onto another, so it cannot be registered \
                 against the previous frame; use the first frame or the mean",
            ));
        }

        let channels = info.channels.max(1);
        let slices = info.slices.max(1);
        let measure_on = params::measure_channel(params, channels);

        // One bar for the whole run, divided by how long each stage takes.
        let reads_per_measurement = measure_on.map_or(channels, |_| 1) as f64;
        let planes = (channels * slices) as f64;
        let total = info.frames as f64
            * (reads_per_measurement * work::READ
                + work::REGISTER
                + planes * (work::READ + work::WARP));
        let total = total.max(f64::MIN_POSITIVE);
        let at = move |done: f64| (done / total) as f32;

        // The measurement movie, one plane per timepoint.
        let mut scratch = Vec::new();
        let mut movie: Vec<Vec<f32>> = Vec::with_capacity(info.frames);
        for t in 0..info.frames {
            if !host.progress(at(t as f64 * reads_per_measurement * work::READ)) {
                return Ok(Outcome::Cancelled);
            }
            movie.push(measurement(host, measure_on, channels, t, &mut scratch)?);
        }
        let reading = info.frames as f64 * reads_per_measurement * work::READ;

        let frames = Frames {
            ly,
            lx,
            frames: &movie,
        };
        let registering = info.frames as f64 * work::REGISTER;
        let Some(result) = register_stack(&frames, &settings, &mut |f| {
            host.progress(at(reading + f as f64 * registering))
        }) else {
            return Ok(Outcome::Cancelled);
        };
        drop(movie);

        // Now one pass over the recording, applying each timepoint's
        // transformation to every plane of that timepoint.
        let store = Store::of(info.pixel_type);
        let mut out: Vec<PlaneData> = Vec::with_capacity(channels * slices * info.frames);
        let mut done = reading + registering;
        // xyczt: channel fastest, then z, then t — the order the host expects.
        for t in 0..info.frames {
            for z in 0..slices {
                for c in 0..channels {
                    host.read_plane_f32(Plane::new(c, z, t), &mut scratch)?;
                    done += work::READ;
                    let moved = result.apply(&scratch, ly, lx, t);
                    done += work::WARP;
                    out.push(store.plane(moved));
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
            "registered {} frames by {} on {}{}; {moved} moved{}",
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
            if result.failed.is_empty() {
                String::new()
            } else {
                format!(", {} frame(s) could not be fitted", result.failed.len())
            }
        ));

        Ok(shared::deliver(
            ImageResult {
                width: info.width,
                height: info.height,
                channels,
                slices,
                frames: info.frames,
                pixel_type: store.pixel_type(),
                planes: out,
                channel_colors: Vec::new(),
                metadata: Some(host.stack_info().clone()),
                name: format!("{}-registered", host.stack_info().name),
            },
            params,
        ))
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

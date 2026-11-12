//! Handing a result back plane by plane, while the run is still going.
//!
//! An [`ImageResult`](crate::ImageResult) is every plane at once, and a
//! registration that rewrites a recording the size of the machine's memory
//! cannot hold that. So a plugin whose result is as big as its input declares
//! the result first — an [`ImageSpec`], everything but the pixels — and pushes
//! each plane as soon as it is made:
//!
//! ```no_run
//! use fasttiff_plugin_api::*;
//!
//! fn run(host: &mut dyn HostContext) -> Result<Outcome, PluginError> {
//!     let info = host.image();
//!     let spec = ImageSpec {
//!         width: info.width, height: info.height,
//!         channels: info.channels, slices: info.slices, frames: info.frames,
//!         pixel_type: PixelType::F32,
//!         channel_colors: Vec::new(),
//!         metadata: Some(host.stack_info().clone()),
//!         name: format!("{}-copy", host.stack_info().stem()),
//!     };
//!     let mut out = ImageStream::begin(host, spec)?;
//!     let mut buf = Vec::new();
//!     for t in 0..info.frames.max(1) {
//!         for z in 0..info.slices.max(1) {
//!             for c in 0..info.channels.max(1) {
//!                 if !host.progress(t as f32 / info.frames.max(1) as f32) {
//!                     return Ok(Outcome::Cancelled);
//!                 }
//!                 host.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
//!                 out.push(host, PlaneRef::F32(&buf))?;
//!             }
//!         }
//!     }
//!     out.finish(StreamTarget::NewDocument)
//! }
//! ```
//!
//! # Why the host is passed to every push
//!
//! The stream does not borrow the host, because the plugin goes on reading
//! through the same host between pushes — a writer holding `&mut` to it would
//! lock the plugin out of its own input until the result was done. So the
//! stream is a small value of its own, which keeps the count and the checks,
//! and `push` is told where to push.
//!
//! # On a host that cannot take it
//!
//! A host is not obliged to take a result this way — one written before it
//! existed, or a test harness, says so through
//! [`HostContext::streams_results`]. The stream then keeps the planes itself
//! and [`finish`](ImageStream::finish) hands them back whole, as the
//! [`ImageResult`](crate::ImageResult) outcome the target names. A plugin
//! written against this works on either; only the memory it needs differs.
//!
//! # Cancelling
//!
//! Return [`Outcome::Cancelled`] (or an error) instead of finishing, and the
//! host throws away what was pushed: a stream that was not finished leaves no
//! result and no file behind.

use crate::host::HostContext;
use crate::plugin::{ImageSpec, Outcome, PlaneData, PlaneRef, PluginError, StreamTarget};

/// A result being handed to the host as it is made. See the
/// [module documentation](self).
///
/// Planes go in `xyczt` order — channel fastest, then slice, then time — the
/// order an [`ImageResult`](crate::ImageResult) holds them in. Each is checked
/// against the declaration as it is pushed, so a plugin that miscounts or
/// mis-sizes learns which plane did it, in its own words, before the host
/// has written anything wrong.
#[derive(Debug)]
pub struct ImageStream {
    spec: ImageSpec,
    /// `spec.plane_count()`, which `begin` checked can be counted.
    planes: usize,
    pushed: usize,
    /// The planes, when the host takes results only whole. `None` while
    /// they go to the host instead.
    kept: Option<Vec<PlaneData>>,
}

impl ImageStream {
    /// Declare the result to `host`, and start one.
    ///
    /// Fails when the shape cannot hold a picture, or when the host refuses
    /// the declaration — it may have nowhere to put a result that big.
    pub fn begin<H: HostContext + ?Sized>(
        host: &mut H,
        spec: ImageSpec,
    ) -> Result<ImageStream, PluginError> {
        spec.validate()?;
        let planes = spec.plane_count().ok_or_else(|| {
            PluginError::failed("result declares more planes than can be counted")
        })?;
        let kept = if host.streams_results() {
            host.begin_image(&spec)?;
            None
        } else {
            // Not reserved: the declaration is the plugin's word, and a
            // host that keeps planes grows by what actually arrives.
            Some(Vec::new())
        };
        Ok(ImageStream {
            spec,
            planes,
            pushed: 0,
            kept,
        })
    }

    /// Change what the result says about itself — its colours, and its
    /// metadata, channel names among it — at any point before it is finished.
    ///
    /// `change` is given the declaration to edit; its shape and its name are
    /// fixed by [`begin`](Self::begin), the first planes being in the result
    /// already, and changing either is refused. A colour picked once a
    /// channel's contents have been seen, a calibration measured from the
    /// data: the host writes a result's description when it finishes it, so
    /// nothing already written has to change.
    ///
    /// A host across the C boundary hears this as the sink's `set_info`,
    /// `set_channel` and `set_channel_name` again, which add to what it was
    /// told rather than replace it: a colour or name, once given, stays.
    pub fn describe<H: HostContext + ?Sized>(
        &mut self,
        host: &mut H,
        change: impl FnOnce(&mut ImageSpec),
    ) -> Result<(), PluginError> {
        let mut spec = self.spec.clone();
        change(&mut spec);
        let fixed = |s: &ImageSpec| {
            (
                s.width,
                s.height,
                s.channels,
                s.slices,
                s.frames,
                s.pixel_type,
                s.name.clone(),
            )
        };
        if fixed(&spec) != fixed(&self.spec) {
            return Err(PluginError::failed(
                "a streamed result's shape and name are fixed when it begins; only its \
                 colours and metadata can change",
            ));
        }
        if self.kept.is_none() {
            host.describe_image(&spec)?;
        }
        self.spec = spec;
        Ok(())
    }

    /// The next plane, borrowed: the buffer can be refilled for the one after.
    pub fn push<H: HostContext + ?Sized>(
        &mut self,
        host: &mut H,
        plane: PlaneRef<'_>,
    ) -> Result<(), PluginError> {
        self.check(plane)?;
        match &mut self.kept {
            Some(planes) => planes.push(plane.to_plane_data()),
            None => host.push_plane(plane)?,
        }
        self.pushed += 1;
        Ok(())
    }

    /// The next plane, given up: for a plugin that makes a fresh buffer per
    /// plane anyway. On a host that takes results whole this saves the copy
    /// [`push`](Self::push) would make; otherwise it is the same push, and
    /// the buffer is freed straight after it.
    pub fn push_owned<H: HostContext + ?Sized>(
        &mut self,
        host: &mut H,
        plane: PlaneData,
    ) -> Result<(), PluginError> {
        self.check(plane.as_plane_ref())?;
        match &mut self.kept {
            Some(planes) => planes.push(plane),
            None => host.push_plane(plane.as_plane_ref())?,
        }
        self.pushed += 1;
        Ok(())
    }

    fn check(&self, plane: PlaneRef<'_>) -> Result<(), PluginError> {
        let expect = self.planes;
        if self.pushed >= expect {
            return Err(PluginError::failed(format!(
                "pushed more than the {expect} planes the result declares"
            )));
        }
        self.spec.check_plane(self.pushed, plane)
    }

    /// What the result was declared as.
    pub fn spec(&self) -> &ImageSpec {
        &self.spec
    }

    /// Planes pushed so far.
    pub fn pushed(&self) -> usize {
        self.pushed
    }

    /// Planes still to push.
    pub fn remaining(&self) -> usize {
        self.planes - self.pushed
    }

    /// The run's outcome: the result is complete, and goes to `target`.
    ///
    /// Fails, with the same words a returned image that miscounted gets, when
    /// fewer planes were pushed than declared.
    pub fn finish(self, target: StreamTarget) -> Result<Outcome, PluginError> {
        if self.pushed != self.planes {
            return Err(self.spec.count_mismatch(self.pushed));
        }
        let Some(planes) = self.kept else {
            return Ok(Outcome::Streamed { target });
        };
        let image = Box::new(self.spec.with_planes(planes));
        Ok(match target {
            StreamTarget::NewDocument => Outcome::NewDocument(image),
            StreamTarget::ReplaceDocument => Outcome::ReplaceDocument(image),
            StreamTarget::SaveToFile { path } => Outcome::SaveToFile { image, path },
        })
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;

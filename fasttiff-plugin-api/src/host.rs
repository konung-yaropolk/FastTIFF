//! The handle a plugin uses to reach the host.

use crate::image::{ImageInfo, Plane, ViewParams};
use crate::meta::StackInfo;
use crate::PluginError;

/// What the host lends a plugin for the duration of one run.
///
/// Pull-based and buffer-filling on purpose. A plugin asks for one plane at a
/// time into a buffer it owns, rather than being handed the stack: a 4 GB
/// stack cannot be copied across a plugin boundary, and lending a borrowed
/// slice of the host's memory map is exactly the thing that stops being sound
/// the moment the plugin is a separate library compiled by a different
/// toolchain. Filling a caller-owned buffer works identically for a built-in
/// plugin, a `.dll` behind a C ABI, and (later) a subprocess writing into
/// shared memory — which is why the shape is this and not `-> &[u16]`.
///
/// The host guarantees, for the whole run:
///   * [`image`](Self::image) and [`view`](Self::view) do not change;
///   * a plane that was in range stays in range;
///   * every `read_*` either fills the buffer to exactly `plane_len()` or
///     returns `Err` — never a short read.
pub trait HostContext {
    /// The stack's shape, in file coordinates.
    fn image(&self) -> ImageInfo;

    /// The viewer's state when the run started.
    fn view(&self) -> &ViewParams;

    /// Name, calibration, channel names, and the rest of the file's metadata.
    fn stack_info(&self) -> &StackInfo;

    /// Decode one plane as `u16`.
    ///
    /// 8-bit data is widened (`v << 8 | v`, matching how the viewer treats it),
    /// signed data is offset into unsigned, float is rescaled through the
    /// channel's contrast window. A plugin that needs the untouched values
    /// should use [`read_plane_f32`](Self::read_plane_f32).
    ///
    /// `out` is resized to `image().plane_len()`.
    fn read_plane_u16(&mut self, plane: Plane, out: &mut Vec<u16>) -> Result<(), PluginError>;

    /// Decode one plane as `f32`, in the file's own units — no windowing, no
    /// rescaling. This is the one to process with.
    fn read_plane_f32(&mut self, plane: Plane, out: &mut Vec<f32>) -> Result<(), PluginError>;

    /// Report progress, `0.0..=1.0`.
    ///
    /// Returns `false` when the user has asked to cancel; a well-behaved plugin
    /// stops and returns [`crate::Outcome::Cancelled`]. A plugin that ignores it
    /// is not a correctness problem — the host discards the result of a
    /// cancelled run — but it does leave a core busy until it finishes.
    fn progress(&mut self, fraction: f32) -> bool;

    /// A line for the host to show the user. Not an error; use the returned
    /// `Err` for that.
    fn log(&mut self, message: &str);
}

/// Convenience helpers, available on every `HostContext` and not part of what
/// an implementor has to provide.
pub trait HostContextExt: HostContext {
    /// Read the plane the viewer is currently showing for display channel 0 —
    /// the common case for a filter that just wants "the frame on screen".
    fn read_current_plane_f32(&mut self, out: &mut Vec<f32>) -> Result<(), PluginError> {
        let t = self.view().frame_index;
        self.read_plane_f32(Plane::new(0, 0, t), out)
    }

    /// Every plane of one timepoint, channel-major.
    fn read_timepoint_f32(&mut self, t: usize, out: &mut Vec<Vec<f32>>) -> Result<(), PluginError> {
        let info = self.image();
        out.clear();
        for z in 0..info.slices.max(1) {
            for c in 0..info.channels.max(1) {
                let mut buf = Vec::new();
                self.read_plane_f32(Plane::new(c, z, t), &mut buf)?;
                out.push(buf);
            }
        }
        Ok(())
    }
}

impl<T: HostContext + ?Sized> HostContextExt for T {}

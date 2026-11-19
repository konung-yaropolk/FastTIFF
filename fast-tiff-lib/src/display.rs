//! The `u16` "display space" the `*_u16` readers decode into, as a value.
//!
//! [`read_plane_u16`](crate::read_plane_u16) and its siblings put every bit
//! depth on one `0..=65535` scale, so a renderer can treat every channel
//! alike. That scale is not the file's: an 8-bit 200 comes back as 51400, a
//! signed 16-bit -1 as 32767, a float as a fraction of the range the caller
//! asked for. Code that needs the file's value back, or the display value of a
//! file value (a contrast window, a histogram bin edge), asks a [`DisplayMap`]
//! rather than restating 257 or 32768 itself.
//!
//! The file's own units come from [`read_plane_f32`](crate::read_plane_f32)
//! and its siblings, for every bit depth.

use crate::{FrameInfo, SampleFormat};

/// How one frame's samples are mapped onto the `u16` the `*_u16` readers
/// return.
///
/// [`DisplayMap::of`] says which map a frame gets; the readers that take a
/// caller's buffer also return the map they applied, which is the only way to
/// learn it for a wide frame read with an auto range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DisplayMap {
    /// Unsigned 16-bit: the value is the sample.
    Identity,
    /// Unsigned 8-bit, widened onto `0..=65535`: `display = file × 257`, so
    /// 0 stays 0 and 255 becomes 65535.
    Widen8,
    /// Signed 8-bit, offset into unsigned, then widened:
    /// `display = (file + 128) × 257`.
    SignedWiden8,
    /// Signed 16-bit, offset into unsigned (a sign-bit flip):
    /// `display = file + 32768`.
    Signed16,
    /// 32- or 64-bit samples, integer or float, rescaled linearly so that
    /// `lo` maps to 0 and `hi` to 65535, clamped outside that range.
    Linear { lo: f32, hi: f32 },
}

impl DisplayMap {
    /// The map the `*_u16` readers apply to `frame`.
    ///
    /// `range` is the one a wide frame is rescaled from, the readers'
    /// `float_range`; it is ignored for 8- and 16-bit frames. `None` for a
    /// wide frame without a range (the readers then range each plane on its
    /// own, and report the map they used), and for a depth the u16 readers do
    /// not decode.
    pub fn of(frame: &FrameInfo, range: Option<(f32, f32)>) -> Option<Self> {
        let signed = frame.sample_format == SampleFormat::SignedInt;
        match frame.bits_per_sample {
            8 if signed => Some(Self::SignedWiden8),
            8 => Some(Self::Widen8),
            16 if signed => Some(Self::Signed16),
            16 => Some(Self::Identity),
            32 | 64 => range.map(|(lo, hi)| Self::Linear { lo, hi }),
            _ => None,
        }
    }

    /// The display value of a file value, unclamped and unrounded: where `file`
    /// sits on the `0..=65535` scale, possibly outside it. For a contrast
    /// window or a bin edge, which may lie beyond the data.
    pub fn to_display(self, file: f64) -> f64 {
        match self {
            Self::Identity => file,
            Self::Widen8 => file * 257.0,
            Self::SignedWiden8 => (file + 128.0) * 257.0,
            Self::Signed16 => file + 32768.0,
            Self::Linear { lo, hi } => {
                let (lo, span) = linear_span(lo, hi);
                (file - lo as f64) / span as f64 * 65535.0
            }
        }
    }

    /// The file value of a display value: the inverse of
    /// [`to_display`](Self::to_display), exact for every map.
    pub fn to_file(self, display: f64) -> f64 {
        match self {
            Self::Identity => display,
            Self::Widen8 => display / 257.0,
            Self::SignedWiden8 => display / 257.0 - 128.0,
            Self::Signed16 => display - 32768.0,
            Self::Linear { lo, hi } => {
                let (lo, span) = linear_span(lo, hi);
                display / 65535.0 * span as f64 + lo as f64
            }
        }
    }

    /// The `u16` a reader writes for a sample whose file value is `file`:
    /// [`to_display`](Self::to_display), clamped to the scale and rounded,
    /// computed the way the readers compute it (in `f32` for wide samples).
    pub fn display_sample(self, file: f64) -> u16 {
        match self {
            Self::Linear { lo, hi } => {
                let (lo, span) = linear_span(lo, hi);
                let t = ((file as f32 - lo) / span).clamp(0.0, 1.0);
                (t * 65535.0).round() as u16
            }
            other => other.to_display(file).round().clamp(0.0, 65535.0) as u16,
        }
    }
}

/// `lo` and the span the readers divide by: never zero, so a constant range
/// maps everything to 0 rather than to a division by zero.
fn linear_span(lo: f32, hi: f32) -> (f32, f32) {
    (lo, (hi - lo).max(f32::EPSILON))
}

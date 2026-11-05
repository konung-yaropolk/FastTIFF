//! Contrast windows, and the units they are in, as types.
//!
//! A contrast window is two sample values: the one shown black and the one
//! shown at full intensity. Which *units* those two numbers were in used to be
//! a convention, worked out again wherever a window met pixels. An 8-bit window
//! was held widened onto the 16-bit scale so one slider served every depth, a
//! signed 16-bit one sat offset into unsigned, a 32-bit integer one was in
//! whatever range its plane had been rescaled through, and a float one was in
//! the data's own values. Every site that converted keyed the conversion on
//! something different — the texture kind, the bit depth, the first frame — and
//! getting one wrong raised no error. It produced a black or blown-out picture,
//! which looks like a contrast setting rather than a units bug.
//!
//! So a window now has one unit: the file's own, what
//! [`read_plane_f32`](fast_tiff_lib::read_plane_f32) returns. An 8-bit
//! channel's window is in `0..=255`, a signed 16-bit one's can be negative, a
//! float one's is in the data's own values. That is the unit the sliders move
//! in, the labels show, a plugin is handed and a measurement is made in.
//!
//! [`Window`] is such a pair. [`SampleUnits`], fixed per channel when its
//! settings are built, is the only place a window is converted: into what the
//! channel's GPU texture holds ([`SampleUnits::to_texture`], for the 2D shader
//! and the CPU compositor that has to match it), into the 3D view's normalized
//! texels ([`SampleUnits::to_unorm`]), from a decoded texture sample back
//! ([`SampleUnits::texture_to_file`]), and to and from the display range an
//! ImageJ file declares ([`SampleUnits::to_imagej`],
//! [`SampleUnits::from_imagej`]). The numbers those conversions are made of
//! live in [`fast_tiff_lib::DisplayMap`], the decoder's own statement of what
//! its `u16` readers do, so they cannot drift from the decode either.
//!
//! Every conversion here is per window, never per sample: a window is two
//! numbers, converted once, and the pixels go to the GPU exactly as decoded.

use fast_tiff_lib::{DisplayMap, FrameInfo};
use scivis_render::ChannelKind;

/// What an `R8Uint` texel is divided by to become a normalized one: the
/// largest byte.
///
/// The 3D view samples normalized textures, because only those filter. The
/// volume builder encodes an 8-bit channel's samples with this, and
/// [`SampleUnits::to_unorm`] divides that channel's window by it, which is
/// what keeps a window and the voxels it is applied to on one scale.
pub const UNORM8_MAX: f32 = u8::MAX as f32;

/// The same for an `R16Uint` texel: the largest `u16`.
pub const UNORM16_MAX: f32 = u16::MAX as f32;

/// A contrast window, in the file's own sample units: `lo` is shown black and
/// `hi` at full intensity, with the channel's LUT colouring what lies between.
///
/// Used for slider tracks ([`ChannelSettings::bounds`](crate::ChannelSettings::bounds))
/// as well as windows, since a track is a span of the same values.
///
/// Nothing stops `lo > hi` being written into the fields; the controls keep
/// them ordered, and every consumer clamps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub lo: f32,
    pub hi: f32,
}

impl Window {
    pub const fn new(lo: f32, hi: f32) -> Self {
        Window { lo, hi }
    }

    /// `hi - lo`.
    pub fn width(self) -> f32 {
        self.hi - self.lo
    }

    /// Whether `v` lies inside the window, ends included.
    pub fn contains(self, v: f32) -> bool {
        self.lo <= v && v <= self.hi
    }

    /// Both ends moved into `bounds`, each on its own: a window that lies
    /// wholly past one end of `bounds` comes back as that end, twice, and it is
    /// for the caller to decide what an empty window should become.
    ///
    /// Panics, as [`f32::clamp`] does, when `bounds` is inverted or NaN.
    pub fn clamp_to(self, bounds: Window) -> Window {
        Window {
            lo: self.lo.clamp(bounds.lo, bounds.hi),
            hi: self.hi.clamp(bounds.lo, bounds.hi),
        }
    }

    /// The smallest window holding both.
    pub fn union(self, other: Window) -> Window {
        Window {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
}

impl From<(f32, f32)> for Window {
    fn from((lo, hi): (f32, f32)) -> Self {
        Window { lo, hi }
    }
}

impl From<Window> for (f32, f32) {
    fn from(w: Window) -> Self {
        (w.lo, w.hi)
    }
}

/// How one channel's samples are written down: in the file, in the channel's
/// GPU texture, and in the display range an ImageJ file states.
///
/// Fixed when the channel's settings are built (see
/// [`ChannelSettings::units`](crate::ChannelSettings::units)), from the frame
/// and the texture kind it is uploaded as. It also says which kind that is
/// ([`kind`](Self::kind)), so the texture format and the conversions a window
/// goes through cannot disagree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleUnits {
    texture: Texture,
}

/// What a channel's texture holds, relative to the file's values.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Texture {
    /// `R8Uint` of the file's own unsigned bytes: 8-bit samples, and 4-bit
    /// ones one to a byte. The texture's units are the file's.
    Raw8,
    /// `R32F` of the file's own values (64-bit ones narrowed). The texture's
    /// units are the file's. `span` is what the 3D volume normalizes the
    /// values by (see [`SampleUnits::volume_affine`]).
    Float { span: Window },
    /// `R16Uint` of the `*_u16` readers' display scale: the map's
    /// [`to_display`](DisplayMap::to_display) of the file's values.
    Display(DisplayMap),
}

impl SampleUnits {
    /// An [`Int8`](ChannelKind::Int8) channel's: an `R8Uint` texture of the
    /// file's own unsigned bytes, so the window needs no conversion on its way
    /// to the shader.
    pub const RAW8: SampleUnits = SampleUnits {
        texture: Texture::Raw8,
    };

    /// A [`Float`](ChannelKind::Float) channel's: an `R32F` texture of the
    /// file's own values, so here too the window goes to the shader as it is.
    ///
    /// `span` is the range the channel's 3D volume is normalized over: its
    /// data's range, fixed when the channel's settings are built. A float
    /// volume is stored as half floats on wgpu, which hold nothing past 65504
    /// and only about three significant digits; stored raw, photon counts or a
    /// deconvolution's output saturated to white, and data far from zero kept
    /// a dozen levels across its range. Normalized to its span it sits near
    /// `0..=1`, where the half float is at its finest.
    pub const fn float(span: Window) -> SampleUnits {
        SampleUnits {
            texture: Texture::Float { span },
        }
    }

    /// An [`Int16`](ChannelKind::Int16) channel's: an `R16Uint` texture
    /// holding the decoder's `u16` display scale of the file's values, `map`.
    pub const fn display(map: DisplayMap) -> SampleUnits {
        SampleUnits {
            texture: Texture::Display(map),
        }
    }

    /// The units of a channel of `frame` uploaded as `kind`.
    ///
    /// For `Int16` that is the map the `*_u16` readers apply to `frame`
    /// ([`DisplayMap::of`]), with `range` the one a 32- or 64-bit frame is
    /// rescaled through. Without a frame, or for a wide frame with no range,
    /// the display scale is taken as the channel's own units: there is nothing
    /// to convert it by, and a window held in it is exactly the window the
    /// texture is shown with.
    pub fn of(kind: ChannelKind, frame: Option<&FrameInfo>, range: Option<(f32, f32)>) -> Self {
        match kind {
            ChannelKind::Int8 => Self::RAW8,
            // No data to take a span from: the unit one, the window a float
            // channel opens on when there is nothing to measure.
            ChannelKind::Float => Self::float(Window::new(0.0, 1.0)),
            ChannelKind::Int16 => Self::display(
                frame
                    .and_then(|f| DisplayMap::of(f, range))
                    .unwrap_or(DisplayMap::Identity),
            ),
        }
    }

    /// The GPU texture format the channel uploads to.
    pub fn kind(self) -> ChannelKind {
        match self.texture {
            Texture::Raw8 => ChannelKind::Int8,
            Texture::Float { .. } => ChannelKind::Float,
            Texture::Display(_) => ChannelKind::Int16,
        }
    }

    /// The decoder's map between the file's values and the texture's, for an
    /// `Int16` channel; `None` for the two kinds whose textures hold the file's
    /// values themselves.
    pub fn display_map(self) -> Option<DisplayMap> {
        match self.texture {
            Texture::Display(map) => Some(map),
            Texture::Raw8 | Texture::Float { .. } => None,
        }
    }

    /// `w` in the units the channel's texture holds: what the 2D shader's
    /// window uniforms are, and what the CPU compositor compares decoded
    /// samples against so that it agrees with the shader.
    pub fn to_texture(self, w: Window) -> (f32, f32) {
        (self.file_to_texture(w.lo), self.file_to_texture(w.hi))
    }

    /// `w` for the 3D view, whose volumes are normalized: the texture window
    /// put through [`volume_affine`](Self::volume_affine), the same map the
    /// volume builder encodes that channel's samples with.
    pub fn to_unorm(self, w: Window) -> (f32, f32) {
        let (lo, hi) = self.to_texture(w);
        let (scale, offset) = self.volume_affine();
        (lo * scale + offset, hi * scale + offset)
    }

    /// The map from a value in the channel's texture units to the value its 3D
    /// volume stores: `v * scale + offset`, returned as `(scale, offset)`.
    ///
    /// One map for the voxels and the window both, so the two cannot be put on
    /// different scales. An integer channel is divided by the largest value
    /// its texture holds ([`UNORM8_MAX`], [`UNORM16_MAX`]), as a unorm texel
    /// is; a float channel is normalized over its span (see
    /// [`float`](Self::float)), `lo` to 0 and `hi` to 1.
    pub fn volume_affine(self) -> (f32, f32) {
        match self.texture {
            Texture::Raw8 => (1.0 / UNORM8_MAX, 0.0),
            Texture::Display(_) => (1.0 / UNORM16_MAX, 0.0),
            Texture::Float { span } => {
                let width = span.width();
                let width = if width.is_finite() && width > 0.0 {
                    width
                } else {
                    1.0
                };
                let lo = if span.lo.is_finite() { span.lo } else { 0.0 };
                (1.0 / width, -lo / width)
            }
        }
    }

    /// The file value of a sample as the channel's texture holds it: for code
    /// that has decoded texture samples in hand and needs to say what they
    /// are worth. The inverse of [`to_texture`](Self::to_texture).
    pub fn texture_to_file(self, v: f32) -> f32 {
        match self.texture {
            Texture::Raw8 | Texture::Float { .. } => v,
            Texture::Display(map) => map.to_file(v as f64) as f32,
        }
    }

    /// Everything the channel's texture can hold, in file units: `0..=255` for
    /// an 8-bit channel, the sample type's whole range for an `Int16` one (or
    /// the range a wide one is rescaled through).
    ///
    /// What a channel shows when there is nothing to fit a window to: an RGB or
    /// CMYK component, which is meant to be seen whole, or a channel whose
    /// pixels could not be scanned. A float texture has no whole range; a float
    /// channel gets `0..1`, the window it has always opened with when there was
    /// nothing to measure.
    pub fn full_range(self) -> Window {
        match self.texture {
            Texture::Raw8 => Window::new(0.0, UNORM8_MAX),
            Texture::Float { .. } => Window::new(0.0, 1.0),
            Texture::Display(_) => {
                Window::new(self.texture_to_file(0.0), self.texture_to_file(UNORM16_MAX))
            }
        }
    }

    /// `w` as an ImageJ file states it: the `min=`/`max=` a save writes.
    ///
    /// ImageJ gives a display range in the file's own units, with one
    /// exception: it holds a signed 16-bit image offset into unsigned (showing
    /// the signed values through a calibration function), and states that
    /// image's range in the offset space. So a signed 16-bit window is written
    /// offset — by the decoder's own map for it — and every other channel's is
    /// written as it is. [`from_imagej`](Self::from_imagej) is the inverse, so
    /// a range read from a file and saved again comes back unchanged.
    pub fn to_imagej(self, w: Window) -> (f64, f64) {
        match self.texture {
            Texture::Display(map @ DisplayMap::Signed16) => {
                (map.to_display(w.lo as f64), map.to_display(w.hi as f64))
            }
            _ => (w.lo as f64, w.hi as f64),
        }
    }

    /// The window an ImageJ file's display range `(min, max)` describes: the
    /// inverse of [`to_imagej`](Self::to_imagej).
    pub fn from_imagej(self, (lo, hi): (f64, f64)) -> Window {
        match self.texture {
            Texture::Display(map @ DisplayMap::Signed16) => {
                Window::new(map.to_file(lo) as f32, map.to_file(hi) as f32)
            }
            _ => Window::new(lo as f32, hi as f32),
        }
    }

    /// One file value on the texture's scale; see [`to_texture`](Self::to_texture).
    fn file_to_texture(self, v: f32) -> f32 {
        match self.texture {
            Texture::Raw8 | Texture::Float { .. } => v,
            Texture::Display(map) => map.to_display(v as f64) as f32,
        }
    }
}

#[cfg(test)]
#[path = "units_tests.rs"]
mod tests;

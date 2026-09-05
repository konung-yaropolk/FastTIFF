//! What a plugin is looking at, and how it asks for pixels.

/// The sample format a plane's pixels arrive in.
///
/// This is the *file's* format, not the display's. A plugin that wants to work
/// in one type regardless can always ask for `f32` and let the host convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelType {
    U8,
    U16,
    I16,
    F32,
}

impl PixelType {
    pub fn bytes(self) -> usize {
        match self {
            PixelType::U8 => 1,
            PixelType::U16 | PixelType::I16 => 2,
            PixelType::F32 => 4,
        }
    }
}

/// The shape of the stack, in **file** terms.
///
/// `channels` here is the number of channels the *file* has, not the number the
/// viewer is compositing. Those differ: the renderer has six texture slots
/// (`scivis_render::MAX_CHANNELS`), so a 12-channel spectral stack displays six
/// and the seventh is simply not on screen. That is a rendering limit and has
/// nothing to do with processing — a plugin that could only reach the channels
/// that happened to fit in the shader would be useless on exactly the data that
/// most needs processing. Plane requests are therefore in file coordinates, and
/// [`crate::ViewParams`] describes the display separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// Channels in the file, under the viewer's resolved c/z/t interpretation.
    pub channels: usize,
    /// Z slices, likewise resolved.
    pub slices: usize,
    /// Timepoints, likewise resolved.
    pub frames: usize,
    /// Samples per pixel in the file's IFDs: 3 for chunky RGB, 1 otherwise.
    /// A plugin rarely needs this — plane requests are per channel either way —
    /// but a plugin writing a result has to know what it is reproducing.
    pub samples_per_pixel: u16,
    pub pixel_type: PixelType,
}

impl ImageInfo {
    /// Pixels in one plane.
    pub fn plane_len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Total planes addressable as (c, z, t).
    pub fn plane_count(&self) -> usize {
        self.channels.max(1) * self.slices.max(1) * self.frames.max(1)
    }

    /// Whether `(c, z, t)` is in range.
    pub fn contains(&self, c: usize, z: usize, t: usize) -> bool {
        c < self.channels.max(1) && z < self.slices.max(1) && t < self.frames.max(1)
    }
}

/// Which plane: channel `c`, slice `z`, timepoint `t`, in file coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Plane {
    pub c: usize,
    pub z: usize,
    pub t: usize,
}

impl Plane {
    pub fn new(c: usize, z: usize, t: usize) -> Self {
        Plane { c, z, t }
    }
}

/// A 256-entry RGB lookup table — the same shape the renderer and
/// `fast-tiff-lib`'s metadata use.
pub type Lut = [[u8; 3]; 256];

/// One channel's display state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelView {
    /// Contrast window low end, in the channel's own sample units.
    pub min: f32,
    /// Contrast window high end.
    pub max: f32,
    /// Whether the channel is currently composited.
    pub enabled: bool,
}

/// What the viewer is showing — the "viewing params" a plugin may want to
/// respect (or export).
///
/// Everything here is a *snapshot* taken when the run started. The user can
/// move a contrast slider while a plugin works; the plugin does not see it, on
/// purpose, so that a long run cannot produce a result computed from two
/// different states.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewParams {
    /// The timepoint on screen when the run started.
    pub frame_index: usize,
    /// True when the 3D ray-marched view is showing rather than the 2D movie.
    pub volume_view: bool,
    /// Per *displayed* channel, in display order. This is the capped list — at
    /// most `scivis_render::MAX_CHANNELS` entries — because it describes what is
    /// on screen. Use [`ImageInfo::channels`] to reach the rest of the file.
    pub channels: Vec<ChannelView>,
    /// The LUT rendered for each displayed channel, same order as `channels`.
    pub luts: Vec<Lut>,
    /// 3D parameters, present whether or not the 3D view is currently showing,
    /// so a plugin can export a render without the user having switched to it.
    pub volume: VolumeView,
}

/// The 3D ray-marching parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeView {
    pub mode: VolumeMode,
    /// Alpha-DVR opacity scale. Ignored by the other modes.
    pub density: f32,
    /// Isosurface threshold, 0..1 in windowed units. Surface mode only.
    pub iso: f32,
    /// Camera position and basis, in volume space.
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolumeMode {
    /// Maximum intensity projection.
    Mip,
    /// Alpha compositing (direct volume rendering).
    Dvr,
    /// Isosurface.
    Surface,
}

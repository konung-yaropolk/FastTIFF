//! What a plugin is, and what it may hand back.

use crate::host::HostContext;
use crate::image::PixelType;
use crate::meta::StackInfo;
use crate::params::{ParamDecl, Params};

/// Who a plugin is. Shown in the menu and in error messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PluginInfo {
    /// Reverse-DNS, stable across versions: `"org.example.invert"`. Two plugins
    /// claiming the same id is an installation error the host reports rather
    /// than resolving, because picking one silently would make which of them
    /// runs depend on directory order.
    pub id: String,
    /// The menu entry.
    pub name: String,
    /// Optional submenu path, `"Filters/Smoothing"`. Empty means top level.
    pub menu_path: String,
    pub version: String,
    pub author: String,
    /// One line, shown as a tooltip.
    pub description: String,
}

impl PluginInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        PluginInfo {
            id: id.into(),
            name: name.into(),
            menu_path: String::new(),
            version: String::new(),
            author: String::new(),
            description: String::new(),
        }
    }

    pub fn menu_path(mut self, p: impl Into<String>) -> Self {
        self.menu_path = p.into();
        self
    }
    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }
    pub fn author(mut self, a: impl Into<String>) -> Self {
        self.author = a.into();
        self
    }
    pub fn description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
}

/// Why a run stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PluginError {
    /// The plugin cannot work on this kind of stack. Distinct from `Failed`
    /// because the host says so differently: this is "not applicable here",
    /// not "something went wrong".
    Unsupported(String),
    /// A plane was requested that the stack does not have.
    OutOfRange(String),
    /// Anything else.
    Failed(String),
}

impl PluginError {
    pub fn failed(msg: impl Into<String>) -> Self {
        PluginError::Failed(msg.into())
    }
    pub fn unsupported(msg: impl Into<String>) -> Self {
        PluginError::Unsupported(msg.into())
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginError::Unsupported(m) => write!(f, "not applicable: {m}"),
            PluginError::OutOfRange(m) => write!(f, "out of range: {m}"),
            PluginError::Failed(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for PluginError {}

/// An image a plugin produced, whole.
///
/// Planes are channel-major within a slice, slice-major within a timepoint —
/// the same `xyczt` order the reader uses — and every plane must be
/// `width * height` samples. The host validates that before doing anything with
/// it, so a plugin that miscounts gets a clear error rather than a corrupt file.
///
/// Every plane is held at once, which is right for a result the size of a
/// plane or a projection and wrong for one the size of the recording: a
/// stabilised timelapse returned this way exists in the plugin's memory in
/// full before the host has seen any of it. Such a result is streamed instead,
/// plane by plane, through an [`ImageStream`](crate::ImageStream) — which
/// takes an [`ImageSpec`], this struct without its planes.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageResult {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub pixel_type: PixelType,
    /// The pixel data, one entry per plane, in `xyczt` order.
    pub planes: Vec<PlaneData>,
    /// What colour each channel should be shown in, as RGB.
    ///
    /// Empty means "the host decides", which is the right default: a filter
    /// that transformed one channel has no opinion. It matters when the
    /// channels *mean* different things — an overlay of two conditions is
    /// unreadable in the host's default red/green if the convention in the
    /// field is magenta/green, and only the plugin knows that.
    ///
    /// Shorter than `channels` is fine; the rest fall back to the default.
    pub channel_colors: Vec<[u8; 3]>,
    /// Metadata to carry into the result document. A filter that preserves an
    /// image's geometry, such as stabilization, can clone this from the host
    /// so calibration and the source record survive the rewrite.
    pub metadata: Option<StackInfo>,
    /// What to call it. The host uses it for the window title and the default
    /// filename.
    pub name: String,
}

/// One plane's samples, in whichever type the result declares.
///
/// There is no `I16` arm, and that is not an oversight: signed 16-bit samples
/// travel in [`PlaneData::U16`] as their raw bit pattern (`v as u16`), with the
/// result declaring [`PixelType::I16`]. Two representations of the same sixteen
/// bits would only invite one of them to be filled in wrongly, and the bits are
/// what gets written to the file either way.
#[derive(Clone, Debug, PartialEq)]
pub enum PlaneData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl PlaneData {
    pub fn len(&self) -> usize {
        match self {
            PlaneData::U8(v) => v.len(),
            PlaneData::U16(v) => v.len(),
            PlaneData::F32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn pixel_type(&self) -> PixelType {
        match self {
            PlaneData::U8(_) => PixelType::U8,
            PlaneData::U16(_) => PixelType::U16,
            PlaneData::F32(_) => PixelType::F32,
        }
    }

    /// The same samples, borrowed: what a plane is pushed to the host as.
    pub fn as_plane_ref(&self) -> PlaneRef<'_> {
        match self {
            PlaneData::U8(v) => PlaneRef::U8(v),
            PlaneData::U16(v) => PlaneRef::U16(v),
            PlaneData::F32(v) => PlaneRef::F32(v),
        }
    }
}

/// One plane's samples, borrowed: the [`PlaneData`] a plugin pushes without
/// giving it up, so the buffer it filled can be filled again for the next one.
///
/// The same three lanes, and the same rule for signed samples: they travel in
/// [`PlaneRef::U16`] as their raw bits, the result declaring [`PixelType::I16`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaneRef<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

impl PlaneRef<'_> {
    pub fn len(&self) -> usize {
        match self {
            PlaneRef::U8(v) => v.len(),
            PlaneRef::U16(v) => v.len(),
            PlaneRef::F32(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn pixel_type(&self) -> PixelType {
        match self {
            PlaneRef::U8(_) => PixelType::U8,
            PlaneRef::U16(_) => PixelType::U16,
            PlaneRef::F32(_) => PixelType::F32,
        }
    }
    /// A copy of the samples, owned.
    pub fn to_plane_data(&self) -> PlaneData {
        match *self {
            PlaneRef::U8(v) => PlaneData::U8(v.to_vec()),
            PlaneRef::U16(v) => PlaneData::U16(v.to_vec()),
            PlaneRef::F32(v) => PlaneData::F32(v.to_vec()),
        }
    }
}

impl<'a> From<&'a PlaneData> for PlaneRef<'a> {
    fn from(p: &'a PlaneData) -> Self {
        p.as_plane_ref()
    }
}

/// Everything about a result image but its pixels: an [`ImageResult`] without
/// its planes.
///
/// What a result is declared with before its first plane exists, so a plugin
/// can hand its planes over as it makes them — see
/// [`ImageStream`](crate::ImageStream). The fields mean what they mean on
/// `ImageResult`, and are checked the same way.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    pub pixel_type: PixelType,
    /// What colour each channel should be shown in. See
    /// [`ImageResult::channel_colors`].
    pub channel_colors: Vec<[u8; 3]>,
    /// Metadata to carry into the result. See [`ImageResult::metadata`].
    pub metadata: Option<StackInfo>,
    /// What to call it. See [`ImageResult::name`].
    pub name: String,
}

impl ImageSpec {
    /// How many planes the declared shape has: `channels * slices * frames`,
    /// each counted as at least one. `None` when that is more than a `usize`
    /// can count — a shape [`validate`](Self::validate) refuses, and one a
    /// plugin across the C boundary can declare in a single call.
    pub fn plane_count(&self) -> Option<usize> {
        self.channels
            .max(1)
            .checked_mul(self.slices.max(1))?
            .checked_mul(self.frames.max(1))
    }

    /// Samples in one plane. Saturates rather than wrapping on a target whose
    /// `usize` cannot hold `width * height`, so a plane that big matches no
    /// real plane; [`validate`](Self::validate) refuses the shape anyway.
    pub fn plane_len(&self) -> usize {
        (self.width as usize).saturating_mul(self.height as usize)
    }

    /// Whether the shape can hold a picture at all: no zero dimension, and
    /// as many planes, of as many samples, as can be counted.
    pub fn validate(&self) -> Result<(), PluginError> {
        if self.width == 0 || self.height == 0 {
            return Err(PluginError::failed("result has a zero dimension"));
        }
        if (self.width as usize)
            .checked_mul(self.height as usize)
            .is_none()
        {
            return Err(PluginError::failed(format!(
                "result planes of {}x{} are more samples than can be addressed",
                self.width, self.height
            )));
        }
        if self.plane_count().is_none() {
            return Err(PluginError::failed(format!(
                "result declares {}x{}x{} planes, more than can be counted",
                self.channels, self.slices, self.frames
            )));
        }
        Ok(())
    }

    /// Whether `plane` can be plane `index` of this result: the right number
    /// of samples, in the declared lane.
    pub fn check_plane(&self, index: usize, plane: PlaneRef<'_>) -> Result<(), PluginError> {
        let expect_len = self.plane_len();
        if plane.len() != expect_len {
            return Err(PluginError::failed(format!(
                "result plane {index} has {} samples, expected {}x{} = {expect_len}",
                plane.len(),
                self.width,
                self.height
            )));
        }
        // `I16` is carried in the `U16` lane as raw bits — see `PlaneData` —
        // so that pairing is the one mismatch that is not one.
        let declared_ok = plane.pixel_type() == self.pixel_type
            || (self.pixel_type == PixelType::I16 && plane.pixel_type() == PixelType::U16);
        if !declared_ok {
            return Err(PluginError::failed(format!(
                "result plane {index} is {:?} but the result declares {:?}",
                plane.pixel_type(),
                self.pixel_type
            )));
        }
        Ok(())
    }

    /// The error for a result that ended with `carried` planes where this
    /// shape has [`plane_count`](Self::plane_count). One wording, whether the
    /// planes were returned or streamed.
    pub fn count_mismatch(&self, carried: usize) -> PluginError {
        let declared = self
            .plane_count()
            .map_or_else(|| "uncountably many".to_string(), |n| n.to_string());
        PluginError::failed(format!(
            "result declares {}x{}x{} = {declared} planes but carries {carried}",
            self.channels, self.slices, self.frames,
        ))
    }

    /// The whole result, these planes being its pixels. Not checked: see
    /// [`ImageResult::validate`].
    pub fn with_planes(self, planes: Vec<PlaneData>) -> ImageResult {
        let ImageSpec {
            width,
            height,
            channels,
            slices,
            frames,
            pixel_type,
            channel_colors,
            metadata,
            name,
        } = self;
        ImageResult {
            width,
            height,
            channels,
            slices,
            frames,
            pixel_type,
            planes,
            channel_colors,
            metadata,
            name,
        }
    }
}

impl ImageResult {
    /// Everything but the planes, copied.
    pub fn spec(&self) -> ImageSpec {
        ImageSpec {
            width: self.width,
            height: self.height,
            channels: self.channels,
            slices: self.slices,
            frames: self.frames,
            pixel_type: self.pixel_type,
            channel_colors: self.channel_colors.clone(),
            metadata: self.metadata.clone(),
            name: self.name.clone(),
        }
    }

    /// The declaration and the planes, apart — what a host that writes a
    /// result plane by plane takes it as.
    pub fn into_parts(self) -> (ImageSpec, Vec<PlaneData>) {
        let ImageResult {
            width,
            height,
            channels,
            slices,
            frames,
            pixel_type,
            planes,
            channel_colors,
            metadata,
            name,
        } = self;
        (
            ImageSpec {
                width,
                height,
                channels,
                slices,
                frames,
                pixel_type,
                channel_colors,
                metadata,
                name,
            },
            planes,
        )
    }

    /// Whether the plane count and every plane's length match the declared
    /// shape. The host checks this; a plugin can too.
    pub fn validate(&self) -> Result<(), PluginError> {
        let spec = self.spec();
        spec.validate()?;
        if Some(self.planes.len()) != spec.plane_count() {
            return Err(spec.count_mismatch(self.planes.len()));
        }
        for (i, p) in self.planes.iter().enumerate() {
            spec.check_plane(i, p.as_plane_ref())?;
        }
        Ok(())
    }
}

/// Where a streamed result goes: the three things a returned image can be
/// asked to do, for an image the host already has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamTarget {
    /// Open it in a new document. See [`Outcome::NewDocument`].
    NewDocument,
    /// Put it in place of the document the plugin ran on. Destructive; see
    /// [`Outcome::ReplaceDocument`].
    ReplaceDocument,
    /// Write it to this path. See [`Outcome::SaveToFile`].
    SaveToFile { path: String },
}

/// What a run produced.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Nothing to apply. The plugin may still have logged.
    Nothing,
    /// Show this message and stop.
    Message(String),
    /// Open the image in a new FastTIFF document.
    NewDocument(Box<ImageResult>),
    /// Replace the document the plugin was run on with this image.
    ///
    /// The other half of [`Outcome::NewDocument`], and the reason a tool can
    /// offer an "open in a new window" checkbox at all: FastTIFF is one stack
    /// per window, so a result either arrives in a window of its own or takes
    /// the place of the one it came from, and only the user knows which they
    /// wanted.
    ///
    /// Unlike the other outcomes this one is **destructive** — what was open is
    /// gone from that window unless it was saved — so a plugin should default
    /// to [`NewDocument`](Outcome::NewDocument) and reach for this only when
    /// asked.
    ///
    /// A host that predates it refuses the run with a message saying so, rather
    /// than silently opening a window the user did not ask for. There is no
    /// field to probe for it: it is a value, not a callback, so the refusal is
    /// the whole negotiation.
    ReplaceDocument(Box<ImageResult>),
    /// Write the image to this path. The host picks the format from the
    /// extension and reports what it wrote.
    SaveToFile {
        image: Box<ImageResult>,
        path: String,
    },
    /// The image was handed to the host plane by plane during the run, through
    /// an [`ImageStream`](crate::ImageStream); this says what to do with it.
    ///
    /// What [`ImageStream::finish`](crate::ImageStream::finish) returns, and
    /// the way to hand back a result the size of the recording: the host
    /// writes each plane into the result file as it arrives, so neither side
    /// ever holds the whole of it. Returned without a stream having been
    /// begun and filled, it is an error naming the plugin.
    Streamed { target: StreamTarget },
    /// Show this chart. The host draws it; see [`crate::plot`].
    ///
    /// If the plot asks for a canvas tool, the host arms it and runs this
    /// plugin again whenever the selection changes — so a plot is a function of
    /// what is selected, and the plugin stays a plain call that returns.
    Plot(Box<crate::plot::Plot>),
    /// Move the 3D view: play this animation on it, or render it to a file.
    /// The host switches to the 3D view to do either. See
    /// [`crate::animation`].
    ///
    /// The host checks the animation with [`Animation::validate`] and refuses
    /// one that fails, naming the plugin, as it does an image.
    ///
    /// Replaces whatever animation was already running, this plugin's or
    /// another's: there is one 3D view, and it can only turn one way at a time.
    ///
    /// [`Animation::validate`]: crate::Animation::validate
    Animate(Box<crate::animation::Animation>),
    /// Stop this plugin's animation, if one is running, and put the view back
    /// where it was when it started. An animation another plugin started is
    /// left alone. Not an error when nothing is running: a Stop button pressed
    /// just after the user took the view back has nothing left to stop, and
    /// that is fine.
    StopAnimation,
    /// Ask the user to draw a rectangle on the picture first, then run this
    /// plugin again with it as the [selection](crate::HostContext::selection),
    /// using the same parameters.
    ///
    /// For a tool whose input is a region, such as a crop. The host shows the
    /// image in 2D, puts `prompt` in its status line, and arms its rectangle
    /// tool. Enter runs the plugin again with the rectangle; Escape gives up,
    /// and the plugin is not run again. So a plugin returns this when its
    /// selection is empty, and does its work when it is not.
    ///
    /// A host that predates it refuses the run with a message saying so, as it
    /// does any outcome it does not know.
    SelectRegion { prompt: String },
    /// The user cancelled. The host discards everything and says so.
    Cancelled,
}

/// A FastTIFF plugin.
///
/// Implement this, derive `Default`, and either let the `fasttiff-plugin`
/// crate's export macro wrap it for a `.dll`/`.so` or have the app register it
/// as a built-in. Both are the same registration through the same C boundary,
/// so the trait is not merely the same either way — the treatment is: a plugin
/// can start life compiled into the app and be moved out to a shared library
/// without its code, or what happens to it, changing. The rules it is held to
/// are listed in the [crate documentation](crate#one-set-of-rules).
pub trait Plugin: Send {
    fn info(&self) -> PluginInfo;

    /// The dialog. Return an empty list to run with no dialog at all.
    ///
    /// Called every time the user picks the menu entry, not once at load, so a
    /// plugin may vary its dialog with what is open — hiding a "Z step" control
    /// for a stack with one slice, say.
    fn params(&self, _host: &dyn HostContext) -> Vec<ParamDecl> {
        Vec::new()
    }

    /// Do the work.
    ///
    /// Called on a worker thread, never the UI thread, so a long run does not
    /// freeze the application. `params` has already been clamped to what
    /// [`Plugin::params`] declared.
    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError>;
}

//! What a plugin is, and what it may hand back.

use crate::host::HostContext;
use crate::image::PixelType;
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

/// An image a plugin produced.
///
/// Planes are channel-major within a slice, slice-major within a timepoint —
/// the same `xyczt` order the reader uses — and every plane must be
/// `width * height` samples. The host validates that before doing anything with
/// it, so a plugin that miscounts gets a clear error rather than a corrupt file.
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
    /// What to call it. The host uses it for the window title and the default
    /// filename.
    pub name: String,
}

/// One plane's samples, in whichever type the result declares.
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
}

impl ImageResult {
    /// Whether the plane count and every plane's length match the declared
    /// shape. The host checks this; a plugin can too.
    pub fn validate(&self) -> Result<(), PluginError> {
        let expect_planes = self.channels.max(1) * self.slices.max(1) * self.frames.max(1);
        if self.planes.len() != expect_planes {
            return Err(PluginError::failed(format!(
                "result declares {}x{}x{} = {expect_planes} planes but carries {}",
                self.channels,
                self.slices,
                self.frames,
                self.planes.len()
            )));
        }
        if self.width == 0 || self.height == 0 {
            return Err(PluginError::failed("result has a zero dimension"));
        }
        let expect_len = self.width as usize * self.height as usize;
        for (i, p) in self.planes.iter().enumerate() {
            if p.len() != expect_len {
                return Err(PluginError::failed(format!(
                    "result plane {i} has {} samples, expected {}x{} = {expect_len}",
                    p.len(),
                    self.width,
                    self.height
                )));
            }
            if p.pixel_type() != self.pixel_type {
                return Err(PluginError::failed(format!(
                    "result plane {i} is {:?} but the result declares {:?}",
                    p.pixel_type(),
                    self.pixel_type
                )));
            }
        }
        Ok(())
    }
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
    /// Write the image to this path. The host picks the format from the
    /// extension and reports what it wrote.
    SaveToFile {
        image: Box<ImageResult>,
        path: String,
    },
    /// The user cancelled. The host discards everything and says so.
    Cancelled,
}

/// A FastTIFF plugin.
///
/// Implement this for a built-in plugin, or let the `fasttiff-plugin` crate's
/// export macro wrap it for a `.dll`/`.so`. The trait is the same either way,
/// which is the point: a plugin can start life compiled into the app, be moved
/// out to a shared library, and later run in a subprocess, without its code
/// changing.
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

//! A [`HostContext`] backed by a loaded [`Stack`].
//!
//! This is what a plugin actually talks to for the in-process lane. The
//! `.dll`/`.so` lane will wrap the same object behind a C vtable rather than
//! replace it, which is why the trait is implemented here — against the viewer's
//! own state — and not in the app.

use fasttiff_plugin_api::{
    ChannelView, DisplayMode, HostContext, ImageInfo, PixelType, Plane, PluginError, Spacing,
    StackInfo, ViewParams, VolumeMode, VolumeView,
};

use crate::dimensions::plane_index;
use crate::display::Dims;
use crate::stack::Stack;
use fast_tiff_lib::{read_plane_u16_into, SampleFormat, TiffStack};
use std::sync::Arc;

/// Everything a plugin can reach for one run.
///
/// Owns what it needs rather than borrowing the [`Stack`], which is what lets a
/// run happen on a worker thread while the window keeps drawing. The pixels are
/// not copied to get there: `tiff` is a second handle on the *same* indexed
/// file — one memory map, shared — and the two fields beside it are the plane
/// layout, which is three `usize`s and a `bool`.
///
/// Everything here is read-only for the run's duration. That is the whole
/// reason no lock is needed: a `TiffStack` is never mutated after it is
/// indexed, so a worker reading it cannot race the window reading it.
pub struct StackHost {
    tiff: Arc<TiffStack>,
    /// The c/z/t interpretation in force when the run started, copied because
    /// the user may change it while a long run is going and a plugin must not
    /// see the axes move underneath it.
    dims: Dims,
    /// Whether display channels are an IFD's sample planes rather than separate
    /// IFDs. Part of the same frozen layout as `dims`.
    rgb: bool,
    image: ImageInfo,
    view: ViewParams,
    info: StackInfo,
    /// Reused between plane reads, so a run that reads a thousand planes does
    /// not allocate a thousand buffers.
    ///
    /// Only the paths that genuinely need an intermediate touch it. The common
    /// case — an uncompressed, native-order, unsigned 16-bit frame, which is
    /// what this library's own writer and ImageJ both produce — converts
    /// straight from the memory map and leaves this empty.
    scratch: crate::planes::Scratch,
    /// The regions the user has drawn, if any. See
    /// [`HostContext::selection`](fasttiff_plugin_api::HostContext::selection).
    selection: Vec<fasttiff_plugin_api::Roi>,
    /// Messages the plugin logged, drained by the caller when the run ends.
    pub messages: Vec<String>,
    /// Set by the UI thread to ask the plugin to stop.
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Last reported progress, `0.0..=1.0`.
    pub progress: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl StackHost {
    pub fn new(stack: &Stack, view: ViewParams) -> Self {
        StackHost {
            image: describe_image(stack),
            info: describe_stack(stack),
            view,
            tiff: Arc::clone(&stack.tiff),
            dims: stack.display.dims,
            rgb: stack.display.rgb,
            scratch: Default::default(),
            selection: Vec::new(),
            messages: Vec::new(),
            cancel: None,
            progress: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Replace the metadata this host reports.
    ///
    /// The stack's own is right almost always; this exists for the cases where
    /// the caller knows better — a document assembled from a plugin result,
    /// which has metadata the written bytes do not carry — and for tests, which
    /// need to state exactly what a plugin should see.
    pub fn with_info(mut self, info: StackInfo) -> Self {
        self.info = info;
        self
    }

    /// Hand the plugin the regions the user has drawn.
    ///
    /// Snapshotted for the run, like the view is: a selection that changed
    /// underneath a running measurement would give a trace of two different
    /// regions at once. The host runs the plugin again for the new one instead.
    pub fn with_selection(mut self, selection: Vec<fasttiff_plugin_api::Roi>) -> Self {
        self.selection = selection;
        self
    }

    /// Wire up a cancel flag the UI can set while the run is in flight.
    pub fn with_cancel(
        mut self,
        flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        progress: std::sync::Arc<std::sync::atomic::AtomicU32>,
    ) -> Self {
        self.cancel = Some(flag);
        self.progress = progress;
        self
    }

    /// Resolve `(c, z, t)` to the IFD and sample plane holding it.
    ///
    /// Goes through [`plane_index`], the one definition the 2D and 3D views also
    /// use — a second formula here is exactly the bug that made that function
    /// necessary.
    fn locate(&self, p: Plane) -> Result<(usize, usize), PluginError> {
        if !self.image.contains(p.c, p.z, p.t) {
            return Err(PluginError::OutOfRange(format!(
                "plane (c{}, z{}, t{}) is outside {}x{}x{}",
                p.c, p.z, p.t, self.image.channels, self.image.slices, self.image.frames
            )));
        }
        let (ifd, sample) = plane_index(self.dims, self.rgb, p.c, p.z, p.t);
        if ifd >= self.tiff.frames.len() {
            // Reachable: the metadata can describe more planes than the file
            // holds (a multi-file OME set gives every file the whole
            // dataset's dimensions). `dimensions::clamp_to_available` trims
            // that at load, but a plugin asking directly deserves the reason
            // rather than a panic.
            return Err(PluginError::OutOfRange(format!(
                "plane (c{}, z{}, t{}) maps to IFD {ifd}, but the file has {}",
                p.c,
                p.z,
                p.t,
                self.tiff.frames.len()
            )));
        }
        Ok((ifd, sample))
    }
}

impl HostContext for StackHost {
    fn image(&self) -> ImageInfo {
        self.image
    }

    fn view(&self) -> &ViewParams {
        &self.view
    }

    fn stack_info(&self) -> &StackInfo {
        &self.info
    }

    fn selection(&self) -> fasttiff_plugin_api::Selection<'_> {
        &self.selection
    }

    fn read_plane_u16(&mut self, plane: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        let (ifd, sample) = self.locate(plane)?;
        let frame = &self.tiff.frames[ifd];
        // The contrast window is what a float plane is rescaled through; a
        // plugin wanting the raw values asks for f32 instead.
        let range = self.view.channels.get(plane.c).map(|c| (c.min, c.max));
        read_plane_u16_into(
            &self.tiff.data,
            frame,
            self.tiff.byte_order,
            range,
            sample,
            out,
        )
        .map_err(|e| {
            PluginError::failed(format!(
                "decoding (c{}, z{}, t{}): {e:#}",
                plane.c, plane.z, plane.t
            ))
        })
    }

    fn read_plane_f32(&mut self, plane: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        let (ifd, sample) = self.locate(plane)?;
        // Shared with `crate::measure`, so a plugin's mean and the app's own
        // plot of the same pixels cannot disagree. See `crate::planes`.
        crate::planes::plane_f32_into(&self.tiff, ifd, sample, &mut self.scratch, out).map_err(
            |e| {
                PluginError::failed(format!(
                    "decoding (c{}, z{}, t{}): {e:#}",
                    plane.c, plane.z, plane.t
                ))
            },
        )
    }

    fn progress(&mut self, fraction: f32) -> bool {
        let clamped = fraction.clamp(0.0, 1.0);
        self.progress.store(
            (clamped * 10_000.0) as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        !self
            .cancel
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn log(&mut self, message: &str) {
        // Bounded: a plugin logging in a loop must not grow until the machine
        // runs out of memory. The cap is generous enough for real reporting.
        const MAX: usize = 1000;
        if self.messages.len() < MAX {
            self.messages.push(message.to_string());
        } else if self.messages.len() == MAX {
            self.messages.push("… further messages suppressed".into());
        }
    }
}

/// The stack's shape in file terms — see [`ImageInfo`] on why this is not the
/// display's channel count.
pub fn describe_image(stack: &Stack) -> ImageInfo {
    let (width, height) = stack.dimensions().unwrap_or((0, 0));
    let f0 = stack.tiff.frames.first();
    let pixel_type = match f0.map(|f| (f.bits_per_sample, f.sample_format)) {
        Some((8, _)) => PixelType::U8,
        Some((32, _)) | Some((64, _)) => PixelType::F32,
        Some((_, SampleFormat::SignedInt)) => PixelType::I16,
        _ => PixelType::U16,
    };
    ImageInfo {
        width,
        height,
        channels: stack.display.dims.channels.max(1),
        slices: stack.display.dims.slices.max(1),
        frames: stack.display.dims.frames.max(1),
        samples_per_pixel: f0.map(|f| f.samples_per_pixel).unwrap_or(1),
        pixel_type,
    }
}

/// The file's metadata, flattened into the dependency-free shape the contract
/// uses.
pub fn describe_stack(stack: &Stack) -> StackInfo {
    let meta = &stack.tiff.meta;
    // `Stack::path` is a real path for a file-opened stack and a bare label for
    // one loaded from bytes (dropped, or picked in the browser build). Only the
    // former is something a plugin can reopen, so say which it is rather than
    // handing over a path that does not exist.
    let path = stack
        .path
        .exists()
        .then(|| stack.path.display().to_string());
    StackInfo {
        name: stack
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| stack.path.display().to_string()),
        path,
        mode: match meta.mode {
            fast_tiff_lib::DisplayMode::Composite => DisplayMode::Composite,
            fast_tiff_lib::DisplayMode::Color => DisplayMode::Color,
            _ => DisplayMode::Grayscale,
        },
        unit: meta.unit.clone(),
        spacing: Spacing {
            x: meta.pixel_width,
            y: meta.pixel_height,
            z: meta.spacing,
        },
        frame_interval_s: meta.frame_interval_s,
        channel_names: Vec::new(),
        calibration: meta.calibration,
        description: stack.tiff.description.clone(),
    }
}

/// The viewer's display state, snapshotted for the run.
pub fn describe_view(
    stack: &Stack,
    frame_index: usize,
    volume_view: bool,
    volume: VolumeView,
) -> ViewParams {
    let channels = stack
        .display
        .settings
        .iter()
        .map(|s| ChannelView {
            min: s.min,
            max: s.max,
            enabled: s.enabled,
        })
        .collect();
    ViewParams {
        frame_index,
        volume_view,
        channels,
        luts: stack.display.luts.clone(),
        volume,
    }
}

/// The 3D parameters, in the contract's shape.
pub fn describe_volume(
    mode: u32,
    density: f32,
    iso: f32,
    eye: [f32; 3],
    forward: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
) -> VolumeView {
    VolumeView {
        mode: match mode {
            1 => VolumeMode::Dvr,
            2 => VolumeMode::Surface,
            _ => VolumeMode::Mip,
        },
        density,
        iso,
        eye,
        forward,
        up,
        right,
    }
}

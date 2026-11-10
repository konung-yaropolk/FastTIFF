//! A [`HostContext`] backed by a loaded [`Document`].
//!
//! This is what a plugin actually talks to for the in-process lane. The
//! `.dll`/`.so` lane will wrap the same object behind a C vtable rather than
//! replace it, which is why the trait is implemented here — against the viewer's
//! own state — and not in the app.

use fasttiff_plugin_api::{
    ChannelView, DisplayMode, HostContext, ImageInfo, ImageSpec, PixelType, Plane, PlaneRef,
    PluginError, Spacing, StackInfo, ViewParams, VolumeMode, VolumeView,
};

use super::result::{ResultDest, ResultTiff, ResultWriter};
use crate::dimensions::plane_index;
use crate::display::Dims;
use crate::document::Document;
use fast_tiff_lib::{read_plane_u16_into, SampleFormat, TiffStack};
use std::sync::Arc;

/// Everything a plugin can reach for one run.
///
/// Owns what it needs rather than borrowing the [`Document`], which is what lets a
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
    /// The regions the user has drawn, if any. See
    /// [`HostContext::selection`](fasttiff_plugin_api::HostContext::selection).
    selection: Vec<fasttiff_plugin_api::Roi>,
    /// What the dialog holds, while this host exists to answer a `params`
    /// call. Empty for a run.
    pending: fasttiff_plugin_api::Params,
    /// Whether an animation the plugin being asked started is running. See
    /// [`HostContext::animating`].
    animating: bool,
    /// Messages the plugin logged, drained by the caller when the run ends.
    pub messages: Vec<String>,
    /// Set by the UI thread to ask the plugin to stop.
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Last reported progress, `0.0..=1.0`.
    pub progress: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// Where a result the plugin streams is written, when this host takes
    /// results plane by plane. `None` takes them only whole.
    results: Option<ResultDest>,
    /// The result being streamed, from the plugin's `begin_image` until the
    /// caller takes it. Dropping it unfinished deletes its file.
    streamed: Option<ResultWriter>,
}

impl StackHost {
    pub fn new(stack: &Document, view: ViewParams) -> Self {
        StackHost {
            image: describe_image(stack),
            info: describe_stack(stack),
            view,
            tiff: Arc::clone(&stack.tiff),
            dims: stack.display.dims(),
            rgb: stack.display.rgb(),
            selection: Vec::new(),
            pending: Default::default(),
            animating: false,
            messages: Vec::new(),
            cancel: None,
            progress: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            results: None,
            streamed: None,
        }
    }

    /// Take a result plane by plane, writing it to `dest` as it arrives.
    ///
    /// What the app's runs are built with, so a registration's result is
    /// never whole in memory: each plane the plugin pushes is written to the
    /// result's file and let go, and [`take_result`](Self::take_result) has
    /// the file when the run is over. Without this the host takes results
    /// only whole, which is what a test that wants an [`ImageResult`] in hand
    /// wants — and what every host did before.
    ///
    /// The result is written with the colour tables of the view the run was
    /// given (`ViewParams::luts`), the source's tables as the run started.
    ///
    /// [`ImageResult`]: fasttiff_plugin_api::ImageResult
    pub fn streaming_to(mut self, dest: ResultDest) -> Self {
        self.results = Some(dest);
        self
    }

    /// Where a streamed result is written, if this host takes one.
    pub fn result_dest(&self) -> Option<&ResultDest> {
        self.results.as_ref()
    }

    /// The result the plugin streamed in this run, finished: its file closed
    /// and readable. `None` when nothing was streamed.
    ///
    /// Fails, deleting what was written, when fewer planes arrived than were
    /// declared.
    pub fn take_result(&mut self) -> Option<anyhow::Result<ResultTiff>> {
        self.streamed.take().map(ResultWriter::finish)
    }

    /// Throw away whatever the plugin streamed: the run did not end with it.
    /// Its file goes with it.
    pub fn discard_result(&mut self) {
        self.streamed = None;
    }

    /// Whether the user has asked this run to stop.
    pub fn cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// A run's outcome with a streamed result read back into the image outcome
    /// its target names: the answer [`Plugin::run`] gave before results could
    /// be streamed.
    ///
    /// For tests and tools that want the pixels in hand. The app never does
    /// this — it opens the file — and anything else is handed back as it was,
    /// a result streamed for it discarded.
    ///
    /// [`Plugin::run`]: fasttiff_plugin_api::Plugin::run
    pub fn collect(
        &mut self,
        outcome: Result<fasttiff_plugin_api::Outcome, PluginError>,
    ) -> Result<fasttiff_plugin_api::Outcome, PluginError> {
        use fasttiff_plugin_api::{Outcome, StreamTarget};
        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                self.discard_result();
                return Err(e);
            }
        };
        let Outcome::Streamed { target } = outcome else {
            self.discard_result();
            return Ok(outcome);
        };
        let finished = self.take_result().ok_or_else(|| {
            PluginError::failed("the plugin said it streamed a result, and none arrived")
        })?;
        let image = finished
            .and_then(|r| r.to_image())
            .map_err(|e| PluginError::failed(format!("{e:#}")))?;
        let image = Box::new(image);
        Ok(match target {
            StreamTarget::NewDocument => Outcome::NewDocument(image),
            StreamTarget::ReplaceDocument => Outcome::ReplaceDocument(image),
            StreamTarget::SaveToFile { path } => Outcome::SaveToFile { image, path },
        })
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

    /// Hand the plugin the dialog values chosen so far.
    ///
    /// Only meaningful for the host built to ask a plugin what its dialog
    /// should be: the plugin reads these and may declare a different set of
    /// controls, which is how a dialog follows the method chosen in it. For a
    /// run the values arrive as the run's own argument and this stays empty.
    pub fn with_pending_params(mut self, pending: fasttiff_plugin_api::Params) -> Self {
        self.pending = pending;
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

    /// Say whether an animation this plugin started is running — what its
    /// dialog reads to offer Play or Stop.
    ///
    /// The caller answers for the plugin it is about to ask, not for the view:
    /// another plugin's animation is not this one's to stop.
    pub fn with_animating(mut self, animating: bool) -> Self {
        self.animating = animating;
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

    fn pending_params(&self) -> &fasttiff_plugin_api::Params {
        &self.pending
    }

    fn animating(&self) -> bool {
        self.animating
    }

    fn read_plane_u16(&mut self, plane: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        let (ifd, sample) = self.locate(plane)?;
        let frame = &self.tiff.frames[ifd];
        // The contrast window is what a 32- or 64-bit plane is rescaled
        // through. It is in the channel's own units, which for such a plane are
        // the values being rescaled, so it can be handed to the decoder as it
        // is. A plugin wanting the raw values asks for f32 instead.
        let range = self.view.channels.get(plane.c).map(|c| (c.min, c.max));
        read_plane_u16_into(
            &self.tiff.data,
            frame,
            self.tiff.byte_order,
            range,
            sample,
            out,
        )
        .map(|_map| ())
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
        crate::planes::plane_f32_into(&self.tiff, ifd, sample, out).map_err(|e| {
            PluginError::failed(format!(
                "decoding (c{}, z{}, t{}): {e:#}",
                plane.c, plane.z, plane.t
            ))
        })
    }

    fn progress(&mut self, fraction: f32) -> bool {
        // Through the shared encoding, which the bar decodes. This stored basis
        // points while the bar read permille, so a tenth drew a full bar.
        super::progress::store(&self.progress, fraction);
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

    fn streams_results(&self) -> bool {
        self.results.is_some()
    }

    fn begin_image(&mut self, spec: &ImageSpec) -> Result<(), PluginError> {
        let Some(dest) = &self.results else {
            return Err(PluginError::unsupported(
                "this host takes a result only whole, not plane by plane",
            ));
        };
        // Declaring again before any plane only restates the result, as it
        // does across the C boundary; after one, it would be a second result
        // the first's planes belong to.
        if self.streamed.as_ref().is_some_and(|w| w.pushed() > 0) {
            return Err(PluginError::failed(super::result::SECOND_RESULT));
        }
        // The restated one replaces the first, whose empty file goes with it.
        self.streamed = None;
        let w = ResultWriter::begin(spec.clone(), &self.view.luts, dest)
            .map_err(|e| PluginError::failed(format!("{e:#}")))?;
        self.streamed = Some(w);
        Ok(())
    }

    fn describe_image(&mut self, spec: &ImageSpec) -> Result<(), PluginError> {
        let Some(w) = &mut self.streamed else {
            return Err(PluginError::failed(
                "the result was described before it was declared",
            ));
        };
        w.redescribe(spec.clone())
            .map_err(|e| PluginError::failed(format!("{e:#}")))
    }

    fn push_plane(&mut self, plane: PlaneRef<'_>) -> Result<(), PluginError> {
        let Some(w) = &mut self.streamed else {
            return Err(PluginError::failed(
                "a plane was pushed before the result was declared",
            ));
        };
        w.push(plane)
            .map_err(|e| PluginError::failed(format!("{e:#}")))
    }
}

/// The stack's shape in file terms — see [`ImageInfo`] on why this is not the
/// display's channel count.
pub fn describe_image(stack: &Document) -> ImageInfo {
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
        channels: file_channels(stack),
        slices: stack.display.dims().slices.max(1),
        frames: stack.display.dims().frames.max(1),
        samples_per_pixel: f0.map(|f| f.samples_per_pixel).unwrap_or(1),
        pixel_type,
    }
}

/// The channels a plugin can read. For chunky RGB that is the samples of each
/// pixel: the viewer's dimensions count one channel there, because the three
/// colours share an IFD, and the plane reader addresses them as samples (see
/// `plane_index`). Reporting that one channel left every plugin reading red
/// alone and writing it back as a grayscale stack. CMYK is left as it was:
/// its samples are inks, which are not what is shown.
fn file_channels(stack: &Document) -> usize {
    let display = &stack.display;
    if display.rgb() && !display.cmyk() {
        display.channel_count().max(1)
    } else {
        display.dims().channels.max(1)
    }
}

/// The file's metadata, flattened into the dependency-free shape the contract
/// uses.
pub fn describe_stack(stack: &Document) -> StackInfo {
    let meta = &stack.tiff.meta;
    // `Document::path` is a real path for a file-opened stack and a bare label for
    // one loaded from bytes (dropped, or picked in the browser build). Only the
    // former is something a plugin can reopen, so say which it is rather than
    // handing over a path that does not exist. A plugin's result shown here has
    // a real file, but a scratch one in a temporary folder that goes when the
    // document does — not a source to tell a plugin about, or to write beside.
    let path = stack.source_path().map(|p| p.display().to_string());
    StackInfo {
        name: stack
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| stack.path.display().to_string()),
        path,
        // RGB is shown composited, its channels red, green and blue, whatever
        // the file says; a result made from it should open looking the same.
        mode: match meta.mode {
            _ if stack.display.rgb() && !stack.display.cmyk() => DisplayMode::Composite,
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
///
/// The windows go out in the file's own sample units, which is what the
/// contract promises ([`ChannelView::min`]) and what the viewer holds them in,
/// so they are copied rather than converted.
pub fn describe_view(
    stack: &Document,
    frame_index: usize,
    volume_view: bool,
    volume: VolumeView,
) -> ViewParams {
    let channels = stack
        .display
        .settings()
        .iter()
        .map(|s| ChannelView {
            min: s.window.lo,
            max: s.window.hi,
            enabled: s.enabled,
        })
        .collect();
    ViewParams {
        frame_index,
        volume_view,
        channels,
        luts: stack.display.luts().to_vec(),
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

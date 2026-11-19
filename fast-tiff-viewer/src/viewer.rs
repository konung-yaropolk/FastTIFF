//! [`Viewer`] — the frontend-agnostic viewer state: which document is open, what
//! it's showing, and everything the per-frame GPU sync keeps between frames.
//!
//! This is the state a *second* frontend would otherwise have to reinvent. It
//! deliberately holds no window, panel, zoom or pointer state: those are the
//! frontend's, and the eframe app keeps them in its own `ViewerApp` alongside a
//! `Viewer`. The dividing line is "would a browser UI need this to show the
//! right pixels?" — if yes it lives here. What the frontend's view comes to
//! each frame — the part of the image on screen, the size of the 3D view — is
//! handed to [`Viewer::sync`] as [`FrameInputs`](crate::FrameInputs) rather
//! than kept here.

use crate::camera::{CameraState, VolumeNavPrefs};
use crate::channels::refresh_pseudocolor;
use crate::dimensions::{apply_dimension_override, shape_note};
use crate::document::{Document, DocumentMut};
use crate::frame::Pose;
use crate::notice::{Notice, NoticeKind, Notices};
use crate::residency::Residency;
use crate::roi::Roi;
use scivis_render::{VolumeInterp, VolumeRender};
use std::path::PathBuf;

/// Playback rate used when the file's metadata doesn't specify `fps=`.
pub const DEFAULT_FPS: f64 = 30.0;

/// Which view the frontend should show: the 2D movie (scrub/play through
/// frames) or the GPU-ray-marched 3D volume built from the whole stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ViewMode {
    #[default]
    Movie,
    Volume,
}

/// How per-frame decoding is split across CPU cores. The choice maps onto
/// `fast-tiff-lib`'s parallel-decode hint each frame, in [`Viewer::sync`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DecodeMode {
    /// Serial by default; switch to threaded automatically when real-time
    /// playback starts dropping frames (a core saturating on decode).
    #[default]
    Auto,
    /// Always single-threaded (lowest total CPU; one core).
    Serial,
    /// Always multi-threaded for large frames (spreads load across cores).
    Threaded,
}

impl DecodeMode {
    pub fn label(self) -> &'static str {
        match self {
            DecodeMode::Auto => "Auto",
            DecodeMode::Serial => "Serial",
            DecodeMode::Threaded => "Threaded",
        }
    }

    /// The parallel-decode flag this mode feeds to `fast-tiff-lib`. `latched` is
    /// whether `Auto`'s adaptive trigger has fired (playback fell behind); it
    /// only matters in `Auto`.
    pub fn parallel(self, latched: bool) -> bool {
        match self {
            DecodeMode::Serial => false,
            DecodeMode::Threaded => true,
            DecodeMode::Auto => latched,
        }
    }
}

/// Looped-playback clock. Advances by real elapsed time so the movie runs at the
/// file's fps regardless of render cadence.
///
/// A frontend starts and stops it, sets its rate and says when the scrubber is
/// held; [`Viewer::tick_playback`] runs it.
#[derive(Clone, Copy, Debug)]
pub struct Playback {
    /// See [`playing`](Self::playing).
    pub(crate) playing: bool,
    /// Playback rate (frames/second). Seeded from the file's `fps=` metadata (or
    /// [`DEFAULT_FPS`]) on every load.
    pub(crate) fps: f64,
    /// Frontend clock time (seconds) at the previous frame while playing.
    /// `None` when not playing.
    pub(crate) last_time: Option<f64>,
    /// Fractional-frame carry so a non-integer frames-per-tick advance doesn't
    /// lose or gain time over a long playback.
    pub(crate) accumulator: f64,
    /// Smoothed "frames demanded per render" while playing: how many frames the
    /// elapsed real time wanted us to advance each tick. ~1 when keeping up; >1
    /// means renders are slower than the target fps (frames dropping — one core
    /// is saturated). Drives the `Auto` decode latch.
    pub(crate) demand_ema: f32,
    /// See [`set_scrubbing`](Self::set_scrubbing).
    pub(crate) scrubbing: bool,
}

impl Default for Playback {
    fn default() -> Self {
        Playback {
            playing: false,
            fps: DEFAULT_FPS,
            last_time: None,
            accumulator: 0.0,
            demand_ema: 1.0,
            scrubbing: false,
        }
    }
}

impl Playback {
    /// Whether the movie is playing. Still `true` while the scrubber is held:
    /// a hold suspends playback rather than stopping it.
    pub fn playing(&self) -> bool {
        self.playing
    }

    /// Playback rate, in frames a second: the file's `fps=` (or
    /// [`DEFAULT_FPS`]) when it was opened, until [`set_fps`](Self::set_fps).
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Play at `fps` frames a second from now on.
    pub fn set_fps(&mut self, fps: f64) {
        self.fps = fps;
    }

    /// Start playing, or pause, from a clean clock: the first tick after
    /// resuming does not jump by however long it was paused. A latched
    /// parallel decode stays latched across a pause — if the stack needed it,
    /// it still does.
    pub fn toggle(&mut self) {
        self.playing = !self.playing;
        self.restart();
    }

    /// Start playing from a clean clock (see [`toggle`](Self::toggle)).
    pub fn play(&mut self) {
        self.playing = true;
        self.restart();
    }

    /// Stop playing, for a view in which playback has no meaning.
    pub fn stop(&mut self) {
        self.playing = false;
        self.last_time = None;
    }

    /// Say whether the scrubber is held, every frame before
    /// [`Viewer::tick_playback`].
    ///
    /// While it is, playback is *suspended*, not stopped:
    /// [`playing`](Self::playing) stays set so releasing resumes without the
    /// user pressing play again. Both the clock and the scrubber write the
    /// frame index, so with the clock still running a held scrubber and
    /// playback take turns — which is seen as the picture flickering between
    /// the frame being scrubbed to and the next one along.
    pub fn set_scrubbing(&mut self, held: bool) {
        self.scrubbing = held;
    }

    /// Restart the clock, so the first tick after resuming doesn't jump by
    /// however long we were paused.
    fn restart(&mut self) {
        self.last_time = None;
        self.accumulator = 0.0;
        self.demand_ema = 1.0;
    }
}

/// The 3D volume view's camera, appearance and navigation preferences: how
/// the user has set the view up.
///
/// What the renderer holds of the volume, and the builds asked for, are the
/// viewer's residency instead (see [`Viewer::volume_built`]): they follow the
/// document, not the user's choices. How large the view is this frame, how
/// finely to render it and an animation's turn of it are the frontend's
/// decisions for the frame, handed to [`Viewer::sync`] as
/// [`VolumeInputs`](crate::frame::VolumeInputs).
#[derive(Clone, Copy, Debug)]
pub struct VolumeSettings {
    pub cam: CameraState,
    /// How fast the camera moves for the user's input, how coarsely the view
    /// renders while it does, and the guide drawn over it: the user's, kept
    /// across files.
    pub navigation: VolumeNavPrefs,
    /// Per-axis voxel scale (x, y, z) for the volume box. Seeded from the
    /// stack's pixel calibration + Z spacing on load (else 1:1:1).
    pub scale: [f32; 3],
    pub interp: VolumeInterp,
    pub render: VolumeRender,
    /// Alpha-DVR opacity scale (only used by the `Alpha` render mode).
    pub density: f32,
    /// Isosurface threshold in windowed units 0..1 (only used by `Surface`).
    pub iso: f32,
    /// The gain on the max-intensity render, as a fraction.
    ///
    /// Per mode rather than shared, because the two modes want different
    /// values: a MIP comes out hot and wants less than 1, where a volume
    /// composited at its default density wants exactly 1. One slider remembering
    /// whichever mode was last on screen would reset the other every time the
    /// mode changed.
    pub mip_brightness: f32,
    /// The gain on the volume render when [`Self::volume_autolevel`] is off.
    pub volume_brightness: f32,
    /// Whether the volume render's gain follows the density automatically.
    ///
    /// On by default: without it the density slider changes the brightness as
    /// much as the solidity, since a denser volume composites brighter.
    pub volume_autolevel: bool,
}

impl Default for VolumeSettings {
    fn default() -> Self {
        VolumeSettings {
            cam: CameraState::default(),
            navigation: VolumeNavPrefs::default(),
            scale: [1.0, 1.0, 1.0],
            interp: VolumeInterp::Linear,
            render: VolumeRender::Mip,
            density: crate::brightness::DENSITY_DEFAULT,
            iso: 0.1,
            mip_brightness: crate::brightness::MIP_DEFAULT,
            volume_brightness: 1.0,
            volume_autolevel: true,
        }
    }
}

impl VolumeSettings {
    /// The camera the ray-march uses this frame, for a volume of `dims`
    /// voxels: the user's, turned by `pose` when an animation has set one
    /// ([`VolumeInputs::pose`](crate::frame::VolumeInputs::pose)).
    ///
    /// The one place it is worked out, so the shader and anything drawn over
    /// the render — the coordinate box — cannot disagree about where the
    /// volume is.
    pub fn camera(
        &self,
        dims: (u32, u32, u32),
        pose: Option<&Pose>,
    ) -> crate::camera::VolumeCamera {
        let cam = crate::camera::volume_camera(&self.cam, self.scale, dims);
        match pose {
            // About the middle of the volume under the centre of the picture,
            // so it turns in place in the middle of the frame however the user
            // has panned or zoomed (see `VolumeCamera::pivot_distance`).
            Some(r) => cam.posed(r, cam.focus(cam.pivot_distance())),
            None => cam,
        }
    }

    /// The volume box's half-extents, for a volume of `dims` voxels: the box
    /// the shader draws, which an orbit's re-pivot
    /// ([`CameraState::repivot`]) and the wheel
    /// ([`CameraState::wheel_fly`]) ray-cast against. A pose turns the
    /// camera, not the box, so it takes none.
    pub fn box_he(&self, dims: (u32, u32, u32)) -> [f32; 3] {
        crate::camera::box_half_extents(self.scale, dims)
    }

    /// How many pixels tall a render of this view, turned by `pose`, has to
    /// be to keep every pixel of the stack in view, for a volume of `dims`
    /// voxels — see
    /// [`VolumeCamera::detail_height`](crate::camera::VolumeCamera::detail_height).
    pub fn detail_height(&self, dims: (u32, u32, u32), pose: Option<&Pose>) -> f32 {
        let cam = self.camera(dims, pose);
        cam.detail_height(cam.pivot_distance(), (dims.0, dims.1))
    }
}

/// The viewer's frontend-agnostic state.
#[derive(Default)]
pub struct Viewer {
    /// The open document, if any: the file and how it is shown. Replaced only
    /// by an open, which re-seeds everything per-document with it; see
    /// [`document`](Self::document).
    pub(crate) document: Option<Document>,
    /// What there is to tell the user, one notice per kind: see
    /// [`notices`](Self::notices). Set here (an open, an axis override) and by
    /// the sync (a decode, a volume upload).
    pub(crate) notices: Notices,
    pub view_mode: ViewMode,
    /// User preference (persists across files): tint a multi-channel *grayscale*
    /// stack with the standard channel palette (ch1 red, ch2 green, ch3 blue, …).
    /// Has no effect on stacks that already carry colors. See
    /// [`pseudocolor`](Self::pseudocolor).
    pub(crate) apply_pseudocolor: bool,
    /// User's decode-parallelism preference (persists across files).
    pub decode_mode: DecodeMode,
    /// A file being opened on a worker thread, if one is. The previous document
    /// stays on screen and usable until it lands — an open replaces a picture
    /// rather than removing one and then providing another.
    #[cfg(feature = "threads")]
    pub(crate) loading: Option<crate::loader::Loading>,
    /// Set when a load lands, cleared by [`poll_open`](Self::poll_open).
    ///
    /// The signal is the same whether the load ran on a worker or here on this
    /// thread, which is the point: a frontend resets the chrome for the new
    /// file on one event, and does not have to know which way the file arrived.
    /// Getting that wrong is what once left the browser build — which has no
    /// threads, so it always loads inline — opening every image at 100%,
    /// because the "a file arrived" signal only ever came from the worker path.
    load_landed: bool,
    /// How frames past the GPU's texture limit are kept on screen (persists
    /// across files). Only consulted for such frames; see
    /// [`crate::roi::LargeImageMode`].
    pub large_image_mode: crate::roi::LargeImageMode,
    /// Latched once playback falls behind in `Auto` mode. Reset per document.
    /// See [`decodes_in_parallel`](Self::decodes_in_parallel).
    pub(crate) decode_parallel: bool,
    pub playback: Playback,
    pub volume: VolumeSettings,
    /// What the renderer holds of [`document`](Self::document), in 2D and 3D: one
    /// view's side of the document, reconciled with its display model at the
    /// top of every [`sync`](Self::sync) and replaced whenever the document
    /// is. See `crate::residency`.
    pub(crate) residency: Residency,
}

/// What to put after "Failed to open file:" — the error's own words, unless the
/// file simply was not a TIFF.
///
/// This viewer reads more than TIFF: importer plugins bring OIR, NIfTI, DICOM
/// and PNG, and more can be loaded from disk. So when the TIFF reader says "not
/// a TIFF" it has answered a narrower question than the user asked. By the time
/// the bytes reach the TIFF reader every importer has already declined them, and
/// the honest summary of that is **"file not supported"** — naming TIFF would
/// suggest the one format this program happens to be named after is the only one
/// it has, which stopped being true.
///
/// Every other failure keeps its own message. "Malformed BigTIFF header" is a
/// *broken TIFF*, not an unsupported file, and flattening the two would throw
/// away the part that tells the user whether to go and find a better copy.
///
/// Keyed on the library's [`ErrorKind::NotTiff`] rather than on the message
/// text, so rewording the library cannot silently turn this back into a
/// passthrough. Every link of a library error carries the kind of the
/// innermost, so the first one in the chain is the one to ask.
///
/// [`ErrorKind::NotTiff`]: fast_tiff_lib::ErrorKind::NotTiff
fn open_failure(e: &anyhow::Error) -> String {
    let kind = e
        .chain()
        .find_map(|c| c.downcast_ref::<fast_tiff_lib::Error>())
        .map(fast_tiff_lib::Error::kind);
    if kind == Some(fast_tiff_lib::ErrorKind::NotTiff) {
        return "file not supported".to_string();
    }
    format!("{e:#}")
}

impl Viewer {
    pub fn new() -> Self {
        Viewer::default()
    }

    /// The open document, if any. One arrives only through an open
    /// ([`open`](Self::open), [`begin_open`](Self::begin_open),
    /// [`load_bytes`](Self::load_bytes)), which re-seeds playback, the 3D
    /// view and what the renderer holds along with it.
    pub fn document(&self) -> Option<&Document> {
        self.document.as_ref()
    }

    /// The open document, to change what can be changed of it in place: the
    /// frame on screen, a channel's window, whether a channel is on, its
    /// overlay. A [`DocumentMut`] rather than `&mut Document`, so it cannot
    /// be replaced in place.
    pub fn document_mut(&mut self) -> Option<DocumentMut<'_>> {
        self.document.as_mut().map(DocumentMut::new)
    }

    /// Whether multi-channel grayscale stacks are tinted with the standard
    /// channel palette: the user's preference, kept across files and set with
    /// [`set_pseudocolor`](Self::set_pseudocolor).
    pub fn pseudocolor(&self) -> bool {
        self.apply_pseudocolor
    }

    /// Whether frames are being decoded across cores: always under
    /// [`DecodeMode::Threaded`], never under [`DecodeMode::Serial`], and under
    /// [`DecodeMode::Auto`] once playback of this document has fallen behind.
    pub fn decodes_in_parallel(&self) -> bool {
        self.decode_mode.parallel(self.decode_parallel)
    }

    /// What there is to tell the user about the view on screen, in
    /// [`NoticeKind`] order: an open that failed, a frame that could not be
    /// decoded, a volume that could not be shown as built, and the
    /// document's shape.
    ///
    /// One per kind, each replaced only by its own kind and gone once it
    /// stops applying (see [`Notice`]), so one never hides another. Only
    /// those that apply to [`view_mode`](Self::view_mode)'s view: the note
    /// that 2D shows the first Z-slice alone stands down over the 3D view,
    /// which shows them all.
    pub fn notices(&self) -> impl Iterator<Item = &Notice> + '_ {
        let mode = self.view_mode;
        self.notices.iter().filter(move |n| n.applies_to(mode))
    }

    /// Drop the notice of `kind`, for a frontend that has moved on.
    ///
    /// For the report of an attempt, [`NoticeKind::OpenFailed`]: once the
    /// user has started other work, what the last open said is no longer
    /// news, and left up beside that work it reads as a comment on it. The
    /// rest describe the document on show, and come back only when the
    /// viewer finds them again.
    pub fn dismiss(&mut self, kind: NoticeKind) {
        self.notices.clear(kind);
    }

    /// Open `path`, replacing whatever was loaded. On success every derived
    /// per-document default is re-seeded: playback rate from the file's `fps=`,
    /// voxel scale from its calibration, a fresh 3D camera, and the 2D movie
    /// view.
    ///
    /// On failure the **previously loaded document stays open** and only
    /// a [`Notice::OpenFailed`] is added, carrying the error — dropping a
    /// corrupt file onto a window must not close the good image the user was
    /// looking at.
    ///
    /// Frontend-side chrome (window title, zoom, open pop-ups) is *not* touched
    /// — that's the frontend's to reset.
    #[cfg(feature = "mmap")]
    pub fn open(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let opened = Document::open(path, self.apply_pseudocolor);
        self.adopt(opened)
    }

    /// Start opening `source`, off this thread where that is possible.
    ///
    /// Returns immediately. [`poll_open`](Self::poll_open) reports when it
    /// lands; until then [`load_stage`](Self::load_stage) says how far it has
    /// got, and the previously open document is still there to look at.
    ///
    /// Without threads to spawn — a browser, or a failed spawn — this falls
    /// back to loading here and now, which blocks exactly as it always did.
    /// Slower to respond is a great deal better than not opening the file.
    pub fn begin_open(&mut self, source: crate::loader::LoadSource) {
        // A new attempt: what the last one said is spent.
        self.notices.clear(NoticeKind::OpenFailed);
        #[cfg(feature = "threads")]
        let source = match crate::loader::Loading::spawn(source, self.apply_pseudocolor) {
            Ok(loading) => {
                self.loading = Some(loading);
                return;
            }
            // No thread to be had. Load it here — slower to respond is far
            // better than not opening the file.
            Err(Some(source)) => source,
            Err(None) => {
                self.notices
                    .set(Notice::OpenFailed("could not start the loader".to_owned()));
                self.load_landed = true;
                return;
            }
        };
        let opened = source.load(self.apply_pseudocolor, &mut |_| {});
        let _ = self.adopt(opened);
        self.load_landed = true;
    }

    /// Take delivery of a finished load, if one has finished this frame.
    ///
    /// Returns `true` when a document was installed or an error recorded — the
    /// signal for a frontend to reset whatever it had arranged around the
    /// previous file.
    pub fn poll_open(&mut self) -> bool {
        #[cfg(feature = "threads")]
        if let Some(loading) = self.loading.as_mut() {
            if let Some(result) = loading.take() {
                self.loading = None;
                let _ = self.adopt(result);
                self.load_landed = true;
            }
        }
        std::mem::take(&mut self.load_landed)
    }

    /// How far the in-flight open has got, or `None` when nothing is loading.
    pub fn load_stage(&self) -> Option<crate::loader::LoadStage> {
        #[cfg(feature = "threads")]
        {
            self.loading.as_ref().map(|l| l.stage())
        }
        #[cfg(not(feature = "threads"))]
        None
    }

    /// The name of the file being opened, while one is.
    pub fn loading_name(&self) -> Option<String> {
        #[cfg(feature = "threads")]
        {
            self.loading.as_ref().map(|l| {
                l.name()
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            })
        }
        #[cfg(not(feature = "threads"))]
        None
    }

    /// Load a document from bytes already in memory — the browser's entry
    /// point, where `name` is just the picked file's name for display.
    /// Otherwise identical to [`Viewer::open`], including every per-document
    /// default it re-seeds.
    pub fn load_bytes(
        &mut self,
        bytes: impl Into<fast_tiff_lib::Bytes>,
        name: PathBuf,
    ) -> anyhow::Result<()> {
        let opened = Document::from_bytes(bytes, name, self.apply_pseudocolor);
        self.adopt(opened)
    }

    /// Install a freshly opened document (or record why it failed), re-seeding
    /// every per-document default. Shared so the two entry points can't drift.
    fn adopt(&mut self, opened: anyhow::Result<Document>) -> anyhow::Result<()> {
        match opened {
            Ok(doc) => {
                self.playback = Playback {
                    fps: doc.tiff.meta.fps.unwrap_or(DEFAULT_FPS),
                    ..Default::default()
                };
                // 3D defaults for the new document: per-axis voxel scale from the
                // pixel calibration (X/YResolution) + Z spacing (else 1:1:1),
                // and a fresh orbit.
                self.volume.scale = doc.tiff.meta.voxel_scale();
                self.volume.cam.reset();
                // Nothing the renderer holds is of this document. A fresh
                // residency says so: its first sync uploads this document's
                // frame and LUTs, gives the last document's volume back, and
                // starts from a fresh volume budget, since what did not fit
                // before may have been the last document's problem.
                self.residency = Residency::default();
                // Always start a newly-opened file in the 2D movie view.
                self.view_mode = ViewMode::Movie;
                // Nothing said of the last document's frame or volume is about
                // this one, and the open that brought it succeeded.
                self.notices.new_document();
                self.notices.shape(shape_note(
                    doc.display.dims(),
                    doc.display.triple_axis_warning,
                    doc.display.plane_mismatch,
                ));
                // New document: re-evaluate decode parallelism from scratch (its
                // per-frame decode cost is different).
                self.decode_parallel = false;
                self.document = Some(doc);
                Ok(())
            }
            Err(e) => {
                // Beside whatever is said of the document still on show,
                // which is still true of it.
                self.notices.set(Notice::OpenFailed(open_failure(&e)));
                Err(e)
            }
        }
    }

    /// Reassign which axis counts mean channels / Z / time, then refresh what
    /// is derived from that here: the LUTs and the note about the shape. What
    /// the renderer holds — the frame, the volume — follows at the next sync,
    /// which finds the display's channel layout moved.
    ///
    /// The counts are a reassignment of the axes in effect, so their product
    /// is theirs: a triple that would address more or fewer planes (or none)
    /// is ignored, as no reading of the file it describes.
    ///
    /// Ignored too on a stack whose channels are the sample planes of each
    /// IFD ([`ChannelLayout::shown_as_rgb`](crate::ChannelLayout::shown_as_rgb):
    /// RGB and CMYK). The file's format fixes those channels, and the axis
    /// counts, which say one channel there, would rebuild the display around
    /// that one: the red plane, alone. A palette stack's channels are IFDs of
    /// their own, as an intensity stack's are, and are rebuilt as an open
    /// with those counts builds them (each window pinned to the identity, and
    /// each channel wearing the file's ColorMap, with no pseudocolor over it),
    /// so it is reassigned like one.
    pub fn set_dimension_order(&mut self, channels: usize, slices: usize, frames: usize) {
        if let Some(doc) = &mut self.document {
            if doc.display.layout().shown_as_rgb() {
                return;
            }
            let now = doc.display.dims();
            let planes = |c: usize, z: usize, t: usize| c.checked_mul(z)?.checked_mul(t);
            if planes(channels, slices, frames) != planes(now.channels, now.slices, now.frames) {
                return;
            }
            apply_dimension_override(doc, channels, slices, frames);
            // The swap rebuilds channel_display from `mode`, so re-apply the
            // pseudocolor preference on top of the fresh LUTs.
            refresh_pseudocolor(doc, self.apply_pseudocolor);
            self.notices.shape(shape_note(
                doc.display.dims(),
                doc.display.triple_axis_warning,
                doc.display.plane_mismatch,
            ));
        }
    }

    /// Apply the pseudocolor preference to the current document (and remember
    /// it for documents opened later).
    pub fn set_pseudocolor(&mut self, on: bool) {
        self.apply_pseudocolor = on;
        if let Some(doc) = &mut self.document {
            refresh_pseudocolor(doc, on);
        }
    }

    /// Show the single-channel LUT selector's option `sel`
    /// ([`GrayLuts`](crate::GrayLuts)): the file's own "Built-in" map first
    /// when it has one, then grayscale, the named channel colours, and the
    /// perceptual colormaps.
    ///
    /// Restoring "Built-in" works because opening a file never overwrites
    /// `tiff.meta`: the file's colours are still there to go back to. No-op
    /// when no document is open, the selector doesn't apply to it
    /// ([`Display::gray_luts`](crate::Display::gray_luts) is `None`), or `sel`
    /// is not one of its options.
    pub fn set_gray_lut(&mut self, sel: usize) {
        if let Some(doc) = &mut self.document {
            doc.display.select_gray_lut(sel);
        }
    }

    /// Subsampling applied to the 2D frame on its way to the GPU, as of the
    /// last [`sync`](Self::sync): `1` when what is on screen is the file's own
    /// resolution, higher when the view had to be coarsened to fit a texture.
    ///
    /// It changes what the user is looking at, and a viewer that quietly
    /// showed a reduced image as if it were the data would be lying: a
    /// frontend reads this to say so. On a frame bigger than one texture it
    /// falls to 1 as you zoom in, which is the signal that you are now seeing
    /// real pixels.
    pub fn gpu_stride(&self) -> u32 {
        self.residency.gpu_stride
    }

    /// Whether the frame on screen is past the device's per-axis texture
    /// limit, and so can only be shown through a window at all, as of the
    /// last [`sync`](Self::sync) (it depends on the device).
    ///
    /// A frontend reads it to decide whether the large-image controls are
    /// worth showing — for every ordinary image they are not, and an option
    /// that does nothing is worse than no option.
    pub fn over_texture_limit(&self) -> bool {
        self.residency.over_texture_limit
    }

    /// The window of the frame the 2D textures hold, in the frame's own
    /// pixels and at its sampling stride; `None` before anything is uploaded.
    /// The whole frame at stride 1 for anything that fits a texture.
    pub fn resident_window(&self) -> Option<Roi> {
        self.residency.roi
    }

    /// Whether a volume of the stack as it stands is on the GPU, for the 3D
    /// view to draw: of some timepoint, the previous one's while 4D playback
    /// builds the next. False until the first build lands — a frontend shows
    /// a loading state meanwhile — and from the moment the axes are
    /// reassigned, which makes it another volume.
    pub fn volume_built(&self) -> bool {
        self.document
            .as_ref()
            .is_some_and(|s| self.residency.volume.built_for(s.display.revision().layout))
    }

    /// Whether a volume build has been asked for and not yet landed.
    pub fn volume_building(&self) -> bool {
        self.residency.volume.requested.is_some()
    }

    /// Which volume the 3D view is of: moves when a new document is opened or its
    /// axes are reassigned — when what is shown is a different volume — and
    /// not when the same one is rebuilt smaller after the GPU ran out of
    /// memory. What a frontend compares to tell that what it was showing has
    /// gone. `0` with no document open.
    pub fn volume_identity(&self) -> u64 {
        self.document
            .as_ref()
            .map_or(0, |s| s.display.revision().layout)
    }

    /// Whether the stack has enough depth to build a volume from.
    pub fn can_show_volume(&self) -> bool {
        self.document
            .as_ref()
            .is_some_and(|s| s.display.dims().frames >= 2)
    }

    /// Whether the stack has a time axis *separate* from the volume's depth —
    /// i.e. playback still means something in the 3D view. False for an ordinary
    /// 3D stack, where the frame axis *is* the depth.
    pub fn is_4d(&self) -> bool {
        self.document
            .as_ref()
            .is_some_and(|s| s.display.dims().slices > 1)
    }

    /// Advance looped playback to match `now` (the frontend's monotonic clock, in
    /// seconds). Returns the number of frames stepped. Also maintains the
    /// keeping-up estimate that latches parallel decoding in `Auto` mode.
    ///
    /// Call once per rendered frame while playing; the frontend is responsible
    /// for scheduling the next repaint (at `1.0 / fps`).
    pub fn tick_playback(&mut self, now: f64) -> usize {
        // Scrubber held: suspend, do not stop. `playing` is left set so release
        // resumes on its own, and the clock baseline is dropped so that release
        // does not then pay back every frame that "should" have played during
        // the drag in one burst — the point of holding it is to look at one
        // frame, and being thrown forward on letting go is the same surprise
        // from the other side.
        if self.playback.scrubbing {
            self.playback.last_time = None;
            self.playback.accumulator = 0.0;
            return 0;
        }
        let Some(doc) = &mut self.document else {
            self.playback.last_time = None;
            self.playback.accumulator = 0.0;
            return 0;
        };
        let n = doc.frame_count();
        if !self.playback.playing || n <= 1 {
            self.playback.playing = self.playback.playing && n > 1;
            self.playback.last_time = None;
            self.playback.accumulator = 0.0;
            return 0;
        }
        let fps = self.playback.fps.max(0.1);
        let mut stepped = 0;
        if let Some(last) = self.playback.last_time {
            // `demand` = frames the elapsed real time wanted this render to
            // cover. ~1 when keeping up; >1 means renders are slower than the
            // fps target (frames dropping → a core is saturated). Once the
            // smoothed demand crosses the threshold, latch parallel decoding.
            let demand = (now - last) * fps;
            self.playback.demand_ema = self.playback.demand_ema * 0.9 + demand as f32 * 0.1;
            if self.decode_mode == DecodeMode::Auto && self.playback.demand_ema > 1.3 {
                self.decode_parallel = true;
            }
            self.playback.accumulator += demand;
            if self.playback.accumulator >= 1.0 {
                stepped = self.playback.accumulator.floor() as usize;
                self.playback.accumulator -= stepped as f64;
                doc.set_frame((doc.frame() + stepped) % n);
            }
        }
        self.playback.last_time = Some(now);
        stepped
    }
}

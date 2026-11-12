//! The loaded document and its per-channel display settings — the model every
//! other core module operates on, and the thing a frontend binds its widgets to.
//!
//! This was the viewer app's private `LoadedStack`, then the core's `Stack`;
//! "document" is the plugin API's word for it too (a result opens as a
//! `StreamTarget::NewDocument` or a `StreamTarget::ReplaceDocument`). What a
//! renderer has uploaded of one is not here: the [`Viewer`](crate::Viewer)
//! keeps that beside it. Nothing in it was ever GUI-specific; it moved here
//! so a second frontend gets the c/z/t interpretation, contrast defaults, LUT
//! selection and read-ahead wiring for free instead of reimplementing them.

use crate::channels::refresh_pseudocolor;
use crate::dimensions::{apply_resolved_dimensions_reporting, setup_cmyk, setup_rgb};
use crate::display::Display;
use crate::loader::LoadStage;
use crate::prefetch::Prefetcher;
pub use crate::units::{SampleUnits, Window};
use crate::volume::VolumeBuilder;
use fast_tiff_lib::TiffStack;
use scivis_render::ChannelKind;
use std::path::PathBuf;
use std::sync::Arc;

/// One display channel's contrast window and GPU upload format.
///
/// Every window here — [`window`](Self::window), [`bounds`](Self::bounds),
/// [`initial`](Self::initial) — is in the file's own sample units, what
/// [`read_plane_f32`](fast_tiff_lib::read_plane_f32) returns: `0..=255` for an
/// 8-bit channel, possibly negative for a signed one, the data's own values for
/// a float one. [`units`](Self::units) is the one place they are converted to
/// anything else; see [`crate::units`].
///
/// A plain value: the display hands these out by shared reference only
/// ([`Display::settings`]), and its setters
/// ([`set_enabled`](Display::set_enabled), [`set_window`](Display::set_window))
/// are how a shown channel changes, so a toggle is always recorded in the
/// display's [`Revision`](crate::display::Revision).
#[derive(Clone, Copy, Debug)]
pub struct ChannelSettings {
    /// The contrast window: `lo` shows black, `hi` full intensity.
    pub window: Window,
    /// Whether the channel is composited. A disabled channel is not decoded
    /// for the frame on screen.
    pub enabled: bool,
    /// The full track a contrast range-slider should span. Derived from the
    /// channel's data range (and widened to include any metadata window) at
    /// load time so both handles always sit somewhere on the track.
    pub bounds: Window,
    /// The window this channel started with — what the file asked for, or what
    /// was derived for it when no window was declared.
    ///
    /// Kept per channel rather than as a snapshot held elsewhere because every
    /// path that rebuilds the settings builds whole `ChannelSettings` values:
    /// opening a file, reinterpreting a mislabeled axis, switching an RGB or
    /// CMYK stack onto its component channels. A separate list would have to be
    /// re-seeded at each of those, and any one that was missed would leave a
    /// stale window paired with the wrong channel — or a list of the wrong
    /// length. As a field it cannot drift: whatever constructs a channel states
    /// where it began.
    pub initial: Window,
    /// How this channel's samples are written down in the file and in its GPU
    /// texture. It says which texture format the channel uploads to
    /// ([`kind`](Self::kind)): `Int8` (R8Uint, raw 8-bit — zero-copy), `Float`
    /// (R32F, raw float, window/level on the GPU), or `Int16` (R16Uint — the
    /// default, incl. RGB planes and any data the CPU widens, offsets or
    /// rescales onto the 16-bit scale). That drives both texture allocation and
    /// the decode path in `crate::sync`. And it holds the conversions the
    /// window goes through on its way to the shader, the 3D view and a saved
    /// file.
    pub units: SampleUnits,
}

impl ChannelSettings {
    /// A switched-on channel opening at `window`, with a slider track of just
    /// that window.
    pub fn new(window: Window, units: SampleUnits) -> Self {
        ChannelSettings {
            window,
            enabled: true,
            bounds: window,
            initial: window,
            units,
        }
    }

    /// Which GPU texture format this channel uploads to; see
    /// [`units`](Self::units).
    pub fn kind(&self) -> ChannelKind {
        self.units.kind()
    }
}

/// An open TIFF stack plus everything the viewer derives from it: the
/// document a view shows.
///
/// What is here holds for every view of it — the file, how it is shown, the
/// frame on screen, the overlay a save writes, and the document's services
/// (its background workers and the caches valid for any view). What one
/// renderer has uploaded of it is not: that is the viewer's own residency,
/// which is why a channel toggle or a dimension override only ever edits the
/// [`display`](Self::display).
pub struct Document {
    /// The file's bytes and frame index, shared rather than owned.
    ///
    /// `Arc` because a plugin, an export or a save runs on a worker thread and
    /// needs the same pixels the window is showing. Handing the worker a clone
    /// of this costs a refcount bump; the alternatives were re-opening the file
    /// (a second mmap and a second walk of the IFD chain) or copying the bytes,
    /// which on a multi-gigabyte stack is the whole file twice in RAM.
    ///
    /// Shared *immutably*, which is what makes it safe with no lock at all:
    /// nothing mutates a `TiffStack` after it is indexed. Everything the viewer
    /// does change — the frame index, the caches, the display model — lives in
    /// the fields below and stays with the window.
    pub tiff: Arc<TiffStack>,
    pub path: PathBuf,
    /// The frame on screen, on the scrub/time axis; see [`frame`](Self::frame).
    /// Below [`frame_count`](Self::frame_count): [`set_frame`](Self::set_frame)
    /// clamps, and reinterpreting the axes puts it back to 0.
    frame_index: usize,
    /// How the viewer is *showing* this document: the c/z/t interpretation, the
    /// per-channel LUTs and contrast windows, and the layout facts derived at
    /// load. Everything mutable about the presentation lives here, so
    /// [`tiff`](Self::tiff)`.meta` stays exactly as the file was parsed — see
    /// [`crate::display`].
    pub display: Display,
    /// The ImageJ overlay a save writes into the TIFF. Opening a file fills it
    /// with the point ROIs of the file's own overlay
    /// ([`StackMeta::overlay`](fast_tiff_lib::StackMeta::overlay)), so that a
    /// save keeps them; the point counter's "Add into TIFF's overlay" replaces
    /// it, and its window starts from it. Only point ROIs are read, so a
    /// file's other shapes are not kept. While it is empty a save is exactly
    /// what it was without one; see [`crate::save`] for what a non-empty one
    /// changes.
    pub overlay: Vec<fast_tiff_lib::PointRoi>,
    /// The workers and caches every view of this document shares.
    pub(crate) services: Services,
    /// Set when this document is a plugin's result shown in this window: the
    /// document it was made from, and the scratch file it is read from, which
    /// it deletes when it goes. `path` is then that scratch file — named as
    /// the result, so the title is right — and is not the user's to be offered
    /// as the document's source; see [`source_path`](Self::source_path) and
    /// [`save_dir`](Self::save_dir).
    ///
    /// Last on purpose: fields drop in order, so `tiff` — the mapping of that
    /// file — is let go and the workers holding it are told to stop before the
    /// file is deleted. Their threads are not joined: each lets go of its share
    /// of the mapping a step later (see [`crate::worker`]), which the scratch
    /// file's retried delete allows for.
    pub derived: Option<crate::plugins::result::Derived>,
}

/// What a document runs and keeps for whoever shows it: the background
/// workers over its pixels, and the caches that hold for any view of it.
///
/// Every worker starts on first need, so a document nobody shows — a test's,
/// a plugin's input, one opened and closed again — runs no thread at all.
/// Each is a `LatestWorker` ([`crate::worker`]): a newer request, or dropping
/// it with the document, stops the job under way. Each is `None` while it has
/// not been needed, when its thread could not start, and always in a build
/// without threads; the work is then done inline, the same path either way.
#[derive(Default)]
pub(crate) struct Services {
    /// Background read-ahead: decode-ahead for compressed stacks, page-touch
    /// for uncompressed ones. See [`crate::prefetch`].
    pub(crate) prefetch: Option<Prefetcher>,
    /// Builds windows of a frame too large for one texture, off the
    /// interface thread, so zooming does not stop the program while a finer
    /// one is prepared.
    pub(crate) window_worker: Option<crate::window::WindowWorker>,
    /// Builds 3D volumes. Dropping it with the document cancels a build in
    /// flight: a closed file's half-built volume, and the memory it holds,
    /// goes within a plane rather than being finished beside the next file's.
    pub(crate) volume_builder: Option<VolumeBuilder>,
    /// Decoded strip bands, so moving the window re-uses what has already been
    /// decompressed instead of decompressing it again.
    ///
    /// Only ever filled for frames too large to upload whole; an ordinary
    /// image never puts anything in it. Keyed by display channel, and so
    /// emptied by the next build under another channel layout
    /// ([`BandCache::hold_layout`](crate::bandcache::BandCache::hold_layout)).
    /// See [`crate::bandcache`].
    pub(crate) bands: crate::bandcache::BandCache,
    /// The histograms of the frame on screen; see [`Document::histograms`].
    pub(crate) histograms: crate::histogram::FrameHistograms,
}

// Started by the per-frame GPU path (`crate::sync`), which needs a backend.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
impl Services {
    /// The read-ahead worker over `tiff`, started if it is not running.
    pub(crate) fn start_prefetch(&mut self, tiff: &Arc<TiffStack>) -> Option<&Prefetcher> {
        if self.prefetch.is_none() {
            let compressed = tiff
                .frames
                .first()
                .is_some_and(|f| f.compression != fast_tiff_lib::Compression::None);
            self.prefetch = Prefetcher::new(Arc::clone(tiff), !compressed);
        }
        self.prefetch.as_ref()
    }

    /// The window worker over `tiff`, started if it is not running.
    pub(crate) fn start_window_worker(
        &mut self,
        tiff: &Arc<TiffStack>,
    ) -> Option<&crate::window::WindowWorker> {
        if self.window_worker.is_none() {
            self.window_worker = crate::window::WindowWorker::new(Arc::clone(tiff));
            if self.window_worker.is_some() {
                // The worker keeps its own bands, and from here on it is the
                // one decoding windows. Holding a second full cache here would
                // be memory for a thread that has stopped asking.
                self.bands.set_budget(crate::bandcache::IDLE_CACHE_BYTES);
            }
        }
        self.window_worker.as_ref()
    }

    /// The volume builder over `tiff`, started if it is not running.
    pub(crate) fn start_volume_builder(&mut self, tiff: &Arc<TiffStack>) -> Option<&VolumeBuilder> {
        if self.volume_builder.is_none() {
            self.volume_builder = VolumeBuilder::new(Arc::clone(tiff));
        }
        self.volume_builder.as_ref()
    }
}

impl Document {
    /// Open `path` and derive the full display model: resolved c/z/t roles, RGB
    /// or palette setup, per-channel contrast windows and the file's own LUT.
    /// No worker starts here: the document's background workers each start
    /// when a view first needs one.
    ///
    /// `apply_pseudocolor` carries the frontend's persistent preference onto the
    /// new stack (it only affects multi-channel grayscale stacks — see
    /// [`crate::channels::pseudocolor_applicable`]).
    #[cfg(feature = "mmap")]
    pub fn open(path: PathBuf, apply_pseudocolor: bool) -> anyhow::Result<Self> {
        Self::open_reporting(path, apply_pseudocolor, &mut |_| {})
    }

    /// [`open`](Self::open), reporting how far it has got.
    ///
    /// Opening is two costs: indexing the IFD chain, whose length is not known
    /// until it has been walked, and then one contrast scan per channel, which
    /// is countable. `on_stage` therefore sees an unquantified `Reading`
    /// followed by a counted `Contrast`, which is as honest as the work allows.
    #[cfg(feature = "mmap")]
    pub fn open_reporting(
        path: PathBuf,
        apply_pseudocolor: bool,
        on_stage: &mut dyn FnMut(LoadStage),
    ) -> anyhow::Result<Self> {
        on_stage(LoadStage::Reading);
        let tiff = TiffStack::open(&path)?;
        Self::from_tiff_reporting(tiff, path, apply_pseudocolor, on_stage)
    }

    /// Same, for a stack already in memory. `name` is only a label — it fills
    /// the `path` field so a frontend can show a filename — so a browser can
    /// pass the picked file's name and nothing will try to touch the
    /// filesystem. The background workers share the stack itself, so they need
    /// no file either.
    ///
    /// `bytes` is owned (a `Vec<u8>`) or shared (an `Arc<[u8]>`), and is kept
    /// rather than copied either way — see [`fast_tiff_lib::Bytes`].
    pub fn from_bytes(
        bytes: impl Into<fast_tiff_lib::Bytes>,
        name: PathBuf,
        apply_pseudocolor: bool,
    ) -> anyhow::Result<Self> {
        Self::from_bytes_reporting(bytes, name, apply_pseudocolor, &mut |_| {})
    }

    /// [`from_bytes`](Self::from_bytes), reporting progress — see
    /// [`open_reporting`](Self::open_reporting).
    pub fn from_bytes_reporting(
        bytes: impl Into<fast_tiff_lib::Bytes>,
        name: PathBuf,
        apply_pseudocolor: bool,
        on_stage: &mut dyn FnMut(LoadStage),
    ) -> anyhow::Result<Self> {
        on_stage(LoadStage::Reading);
        let tiff = TiffStack::from_bytes(bytes)?;
        Self::from_tiff_reporting(tiff, name, apply_pseudocolor, on_stage)
    }

    /// The shared tail of both constructors: derive the display model from an
    /// already-indexed stack.
    fn from_tiff_reporting(
        tiff: TiffStack,
        path: PathBuf,
        apply_pseudocolor: bool,
        on_stage: &mut dyn FnMut(LoadStage),
    ) -> anyhow::Result<Self> {
        // A copy: `tiff.meta` stays as the file was parsed, and this is the
        // document's, which the point counter may replace.
        let overlay = tiff.meta.overlay.clone();
        // No worker is started here: each starts when a view first needs it
        // (see `Services`).
        let mut stack = Document {
            tiff: Arc::new(tiff),
            path,
            frame_index: 0,
            display: Display::default(),
            overlay,
            services: Services::default(),
            derived: None,
        };

        let (c, z, f) = (
            stack.tiff.meta.channels,
            stack.tiff.meta.slices,
            stack.tiff.meta.frames,
        );
        apply_resolved_dimensions_reporting(
            &mut stack,
            fast_tiff_lib::resolve_dimensions(c, z, f),
            &mut |done, total| on_stage(LoadStage::Contrast { done, total }),
        );
        stack.display.has_z_axis = stack.display.dims().slices > 1;
        // Chunky RGB overrides the channel layout: the sample planes of each IFD
        // become red/green/blue display channels.
        if stack.tiff.frames.first().is_some_and(|f| f.is_rgb()) {
            setup_rgb(&mut stack);
        }
        // CMYK (Separated) frames become three converted RGB channels. Checked
        // separately from `is_rgb` rather than folded into it: `is_rgb` means
        // photometric=2, and these planes are derived, not stored.
        if stack.tiff.frames.first().is_some_and(|f| f.is_cmyk()) {
            setup_cmyk(&mut stack);
        }
        // Palette (indexed) images: the ColorMap is already installed as the
        // channel LUT by the lib; flag it so the fixed identity contrast window
        // is kept and its slider can be hidden.
        stack.display.palette = stack.tiff.frames.first().is_some_and(|f| f.is_palette());
        // Remember the file's own LUT (a single-channel image carrying a
        // ColorMap / ImageJ LUT) so the LUT selector can offer a "Built-in"
        // option that restores it. Captured now, before pseudocolor could touch
        // it (it won't, for an explicit-LUT channel, but capture first anyway).
        //
        // `has_explicit_luts` alone is not the right test: it is only set for a
        // *coloured* map, and a palette image's ColorMap is just as much its
        // display transfer when it happens to be grey. Contrast-stretched
        // indexed exports are exactly that — a grayscale ramp that maps index
        // 16 to grey 46 and blacks out the unused tail. Dropping it left the
        // user stuck: picking any colormap replaced the ramp with the generic
        // linear one, the image went dark, and with the contrast slider
        // suppressed for palette stacks there was no way back.
        let has_own_lut = stack.tiff.meta.has_explicit_luts || stack.display.palette;
        stack.display.builtin_lut = (has_own_lut && stack.display.channel_count() == 1)
            .then(|| stack.tiff.meta.channel_display[0].lut);
        refresh_pseudocolor(&mut stack, apply_pseudocolor);

        Ok(stack)
    }

    /// The histograms of the frame on screen, binned from the decodes that
    /// showed it while a frontend asks for them
    /// ([`Viewer::histograms_wanted`](crate::Viewer::histograms_wanted)). A
    /// frontend plots these rather than decoding the frame again; see
    /// [`crate::histogram`].
    pub fn histograms(&self) -> &crate::histogram::FrameHistograms {
        &self.services.histograms
    }

    /// Frame count on the scrub/time axis (at least 1).
    pub fn frame_count(&self) -> usize {
        self.display.dims().frames.max(1)
    }

    /// The frame on screen: its index on the scrub/time axis, from 0.
    pub fn frame(&self) -> usize {
        self.frame_index
    }

    /// Show frame `i`, or the last frame when `i` is past it, so that a step
    /// or a scrub off the end lands on the end rather than on a frame that is
    /// not there.
    pub fn set_frame(&mut self, i: usize) {
        self.frame_index = i.min(self.frame_count() - 1);
    }

    /// The stack's pixel dimensions, from the first frame.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.tiff.frames.first().map(|f| (f.width, f.height))
    }

    /// The file this document is, when it is a file the user has: `None` for
    /// one loaded from bytes, whose `path` is only a name, and for a plugin's
    /// result, whose file is scratch.
    pub fn source_path(&self) -> Option<&std::path::Path> {
        (self.derived.is_none() && self.path.exists()).then_some(self.path.as_path())
    }

    /// The file this document stands for: its own, or for a plugin's result
    /// the one it was made from — what a result made from *this* document is
    /// said to be made from in turn.
    pub fn original(&self) -> Option<PathBuf> {
        match &self.derived {
            Some(d) => d.made_from().map(std::path::Path::to_path_buf),
            None => self.source_path().map(std::path::Path::to_path_buf),
        }
    }

    /// Where "Save as…" should start: the folder of the file this document
    /// stands for (see [`original`](Self::original)), when there is one.
    pub fn save_dir(&self) -> Option<PathBuf> {
        self.original()
            .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
            .filter(|d| d.is_dir())
    }

    /// The volume's depth axis: Z for a 4D stack (`slices > 1`), else the whole
    /// frame axis. Mirrors what `volume::build_volume` assembles.
    pub fn volume_depth(&self) -> u32 {
        let dims = self.display.dims();
        let slices = dims.slices.max(1);
        if slices > 1 {
            slices as u32
        } else {
            dims.frames.max(1) as u32
        }
    }
}

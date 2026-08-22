//! The loaded stack and its per-channel display settings — the model every
//! other core module operates on, and the thing a frontend binds its widgets to.
//!
//! This was the viewer app's private `LoadedStack`. Nothing in it was ever
//! GUI-specific; it moved here so a second frontend gets the c/z/t
//! interpretation, contrast defaults, LUT selection and read-ahead wiring for
//! free instead of reimplementing them.

use crate::channels::{build_channel_settings, refresh_pseudocolor};
use crate::dimensions::{apply_resolved_dimensions, setup_cmyk, setup_rgb};
use crate::display::Display;
use crate::prefetch::Prefetcher;
use crate::volume::VolumeBuilder;
use fast_tiff_lib::TiffStack;
use scivis_render::ChannelKind;
use std::path::PathBuf;

/// One display channel's contrast window and GPU upload format.
#[derive(Clone, Copy, Debug)]
pub struct ChannelSettings {
    pub min: f32,
    pub max: f32,
    pub enabled: bool,
    /// The full track range `(lo, hi)` a contrast range-slider should span, in
    /// raw sample units. Derived from the channel's data range (and widened to
    /// include any metadata window) at load time so both handles always sit
    /// somewhere on the track.
    pub bounds: (f32, f32),
    /// The window this channel started with, in the same units as
    /// [`min`](Self::min)/[`max`](Self::max) — what the file asked for, or what
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
    pub initial: (f32, f32),
    /// Which GPU texture format this channel uploads to (picked from the source
    /// pixel format): `Int8` (R8Uint, raw 8-bit — zero-copy), `Float` (R32F, raw
    /// float, window/level on the GPU), or `Int16` (R16Uint — the default, incl.
    /// RGB planes and any data the CPU widens/rescales into 0..65535). Drives
    /// both texture allocation and the decode path in [`crate::sync`]. For float
    /// channels `min`/`max` are the contrast window in the data's own float
    /// units (matching how ImageJ shows float-image contrast).
    pub kind: ChannelKind,
}

/// An open TIFF stack plus everything the viewer derives from it.
pub struct Stack {
    pub tiff: TiffStack,
    pub path: PathBuf,
    pub frame_index: usize,
    /// Subsampling applied on the way to the GPU: `1` when what is on screen is
    /// the file's own resolution, higher when the view had to be coarsened to
    /// fit a texture.
    ///
    /// Recorded here rather than kept inside the upload because it changes what
    /// the user is looking at, and a viewer that quietly showed a reduced image
    /// as if it were the data would be lying. The frontend reads this to say
    /// so. On a frame bigger than one texture it falls to 1 as you zoom in,
    /// which is the signal that you are now seeing real pixels.
    pub gpu_stride: u32,
    /// The window of the frame currently on the GPU, when only a window of it
    /// is. `None` means the whole frame is resident — the ordinary case.
    pub roi: Option<crate::roi::Roi>,
    /// Builds windows off the interface thread, so zooming does not stop the
    /// program while a finer one is prepared. `None` when the thread could not
    /// start, or in a build without threads; windows are then built inline.
    pub window_worker: Option<crate::window::WindowWorker>,
    /// Decoded strip bands, so moving the window re-uses what has already been
    /// decompressed instead of decompressing it again.
    ///
    /// Only ever filled for frames too large to upload whole; an ordinary image
    /// never puts anything in it. See [`crate::bandcache`].
    pub bands: crate::bandcache::BandCache,
    /// The whole frame at a coarse sampling, decoded and held in RAM — what a
    /// view leaving the resident window falls back to, so it can show correct
    /// pixels at once instead of stopping while a new window is cut.
    ///
    /// `None` until a frame-spanning window has been built for this file, which
    /// the opening fit-to-window view does anyway. Never invalidated where its
    /// inputs change; validated at every use. See [`crate::window::Overview`].
    pub overview: Option<crate::window::Overview>,

    pub last_uploaded: Option<usize>,
    /// The per-channel `enabled` flags as of the last GPU upload. A disabled
    /// channel is skipped during upload (the shader multiplies it out anyway),
    /// so re-enabling one must re-upload it even when the frame index is
    /// unchanged — a difference here forces that.
    pub last_enabled: Vec<bool>,
    pub luts_uploaded: bool,
    /// How the viewer is *showing* this stack: the c/z/t interpretation, the
    /// per-channel LUTs and contrast windows, and the layout facts derived at
    /// load. Everything mutable about the presentation lives here, so
    /// [`tiff`](Self::tiff)`.meta` stays exactly as the file was parsed — see
    /// [`crate::display`].
    pub display: Display,
    /// Background read-ahead worker (own mmap): decode-ahead for compressed
    /// stacks, page-touch for uncompressed ones. `None` when the worker failed
    /// to start, or always under `--no-default-features` (no `threads`); the
    /// inline decode path covers both. See [`crate::prefetch`].
    pub prefetch: Option<Prefetcher>,
    /// Bumped whenever the decode plan changes (dimension-order swap, enabled-set
    /// change) so an in-flight prefetch decoded under the old plan is recognized
    /// as stale and ignored rather than uploaded.
    pub prefetch_gen: u64,
    /// Background 3D-volume builder (own mmap), spawned lazily on the first 3D
    /// use. `None` after a failed spawn (`volume_builder_tried` set) — volume
    /// builds then fall back to running synchronously.
    pub volume_builder: Option<VolumeBuilder>,
    pub volume_builder_tried: bool,
}

impl Stack {
    /// Open `path` and derive the full display model: resolved c/z/t roles, RGB
    /// or palette setup, per-channel contrast windows, the file's own LUT, and
    /// the read-ahead worker.
    ///
    /// `apply_pseudocolor` carries the frontend's persistent preference onto the
    /// new stack (it only affects multi-channel grayscale stacks — see
    /// [`crate::channels::pseudocolor_applicable`]).
    #[cfg(feature = "mmap")]
    pub fn open(path: PathBuf, apply_pseudocolor: bool) -> anyhow::Result<Self> {
        Self::from_tiff(TiffStack::open(&path)?, path, apply_pseudocolor)
    }

    /// Same, for a stack already in memory. `name` is only a label — it fills
    /// the `path` field so a frontend can show a filename and the background
    /// workers have something to reopen — so a browser can pass the picked
    /// file's name and nothing will try to touch the filesystem.
    pub fn from_bytes(bytes: Vec<u8>, name: PathBuf, apply_pseudocolor: bool) -> anyhow::Result<Self> {
        Self::from_tiff(TiffStack::from_bytes(bytes)?, name, apply_pseudocolor)
    }

    /// The shared tail of both constructors: derive the display model from an
    /// already-indexed stack.
    fn from_tiff(tiff: TiffStack, path: PathBuf, apply_pseudocolor: bool) -> anyhow::Result<Self> {
        // Spin up the read-ahead worker: decode-ahead for compressed stacks,
        // page-touch for uncompressed (see the `prefetch` field).
        // Only the read-ahead worker cares, and it only exists with `mmap`.
        #[cfg(feature = "mmap")]
        let compressed = tiff
            .frames
            .first()
            .is_some_and(|f| f.compression != fast_tiff_lib::Compression::None);
        // The worker reopens the file by path, so it only exists where there is
        // a filesystem. Without it every frame decodes inline — the same path
        // taken whenever a spawn fails.
        #[cfg(feature = "mmap")]
        let prefetch = Prefetcher::new(path.clone(), !compressed);
        #[cfg(not(feature = "mmap"))]
        let prefetch = None;

        let mut stack = Stack {
            tiff,
            path,
            display: Display::default(),
            frame_index: 0,
            // Corrected on the first sync, once a renderer is on hand to say
            // what this device can hold.
            gpu_stride: 1,
            roi: None,
            overview: None,
            window_worker: None,
            bands: Default::default(),
            last_uploaded: None,
            last_enabled: Vec::new(),
            luts_uploaded: false,
            prefetch,
            prefetch_gen: 0,
            volume_builder: None,
            volume_builder_tried: false,
        };

        let (c, z, f) = (
            stack.tiff.meta.channels,
            stack.tiff.meta.slices,
            stack.tiff.meta.frames,
        );
        apply_resolved_dimensions(&mut stack, fast_tiff_lib::resolve_dimensions(c, z, f));
        stack.display.has_z_axis = stack.display.dims.slices > 1;
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

    /// Rebuild the per-channel settings from the current metadata. Called after
    /// anything that changes the channel count or display info.
    pub fn rebuild_channel_settings(&mut self) {
        self.display.settings = build_channel_settings(&self.tiff, self.display.dims.channels);
    }

    /// Frame count on the scrub/time axis (at least 1).
    pub fn frame_count(&self) -> usize {
        self.display.dims.frames.max(1)
    }

    /// The stack's pixel dimensions, from the first frame.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.tiff.frames.first().map(|f| (f.width, f.height))
    }

    /// The volume's depth axis: Z for a 4D stack (`slices > 1`), else the whole
    /// frame axis. Mirrors what `volume::build_volume` assembles.
    pub fn volume_depth(&self) -> u32 {
        let slices = self.display.dims.slices.max(1);
        if slices > 1 {
            slices as u32
        } else {
            self.display.dims.frames.max(1) as u32
        }
    }
}

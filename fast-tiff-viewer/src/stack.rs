//! The loaded stack and its per-channel display settings — the model every
//! other core module operates on, and the thing a frontend binds its widgets to.
//!
//! This was the viewer app's private `LoadedStack`. Nothing in it was ever
//! GUI-specific; it moved here so a second frontend gets the c/z/t
//! interpretation, contrast defaults, LUT selection and read-ahead wiring for
//! free instead of reimplementing them.

use crate::channels::{build_channel_settings, refresh_pseudocolor};
use crate::dimensions::{apply_resolved_dimensions, setup_rgb};
use crate::prefetch::Prefetcher;
use crate::volume::VolumeBuilder;
use fast_tiff_lib::TiffStack;
use stack_renderer::ChannelKind;
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
    pub channel_settings: Vec<ChannelSettings>,
    pub frame_index: usize,
    pub last_uploaded: Option<usize>,
    /// The per-channel `enabled` flags as of the last GPU upload. A disabled
    /// channel is skipped during upload (the shader multiplies it out anyway),
    /// so re-enabling one must re-upload it even when the frame index is
    /// unchanged — a difference here forces that.
    pub last_enabled: Vec<bool>,
    pub luts_uploaded: bool,
    /// Set once at load time when the file genuinely has channels, Z, and
    /// time all present simultaneously — Z then stays permanently frozen
    /// at its first slice (see `fast_tiff_lib::resolve_dimensions`). Kept around
    /// so the warning note is still shown correctly after a manual
    /// channels/frames swap, which never touches Z.
    pub triple_axis_warning: bool,
    /// True when each IFD is a chunky RGB image: the "channels" are then the
    /// red/green/blue sample planes of a *single* IFD per frame (deinterleaved
    /// on upload), not separate IFDs. Flips how a display channel maps to file
    /// data.
    pub rgb: bool,
    /// True for a palette-color (indexed) stack: the single channel's pixels are
    /// indices into the file's ColorMap, installed as its display LUT by
    /// `fast_tiff_lib`. Its contrast window is fixed to an identity map (index →
    /// LUT entry), so a frontend should suppress the contrast slider for it —
    /// see `channels::build_channel_settings`.
    pub palette: bool,
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
    /// Whether the file had a real Z axis (`slices > 1`) as loaded. Gates the
    /// three-way (c/z/t) dimension-order choice, and deliberately snapshots
    /// the *load-time* shape rather than the current one: a permutation that
    /// assigns 1 to Z must not collapse the choice to the two-way c/t swap
    /// and strand Z there.
    pub has_z_axis: bool,
    /// Which LUT the single-channel color selector currently shows (selector
    /// index; `0` = the leading "Built-in LUT" when the file has one, else plain
    /// grayscale). Only meaningful while `channels::gray_lut_applicable` holds.
    /// Lives on the stack — not the viewer — so it resets to the default for
    /// each newly opened file.
    pub gray_lut_sel: usize,
    /// The file's own display LUT for the single channel, if it supplied one
    /// (a ColorMap or ImageJ LUT). Kept verbatim so the selector's "Built-in
    /// LUT" option can restore it after the user tries another. `None` when the
    /// file carries no LUT (or isn't single-channel).
    pub builtin_lut: Option<stack_renderer::Lut>,
}

impl Stack {
    /// Open `path` and derive the full display model: resolved c/z/t roles, RGB
    /// or palette setup, per-channel contrast windows, the file's own LUT, and
    /// the read-ahead worker.
    ///
    /// `apply_pseudocolor` carries the frontend's persistent preference onto the
    /// new stack (it only affects multi-channel grayscale stacks — see
    /// [`crate::channels::pseudocolor_applicable`]).
    pub fn open(path: PathBuf, apply_pseudocolor: bool) -> anyhow::Result<Self> {
        let tiff = TiffStack::open(&path)?;
        // Spin up the read-ahead worker: decode-ahead for compressed stacks,
        // page-touch for uncompressed (see the `prefetch` field).
        let compressed = tiff
            .frames
            .first()
            .is_some_and(|f| f.compression != fast_tiff_lib::Compression::None);
        let prefetch = Prefetcher::new(path.clone(), !compressed);

        let mut stack = Stack {
            tiff,
            path,
            channel_settings: Vec::new(),
            frame_index: 0,
            last_uploaded: None,
            last_enabled: Vec::new(),
            luts_uploaded: false,
            triple_axis_warning: false,
            rgb: false,
            palette: false,
            prefetch,
            prefetch_gen: 0,
            volume_builder: None,
            volume_builder_tried: false,
            gray_lut_sel: 0,
            builtin_lut: None,
            has_z_axis: false,
        };

        let (c, z, f) = (
            stack.tiff.meta.channels,
            stack.tiff.meta.slices,
            stack.tiff.meta.frames,
        );
        apply_resolved_dimensions(&mut stack, fast_tiff_lib::resolve_dimensions(c, z, f));
        stack.has_z_axis = stack.tiff.meta.slices > 1;
        // Chunky RGB overrides the channel layout: the sample planes of each IFD
        // become red/green/blue display channels.
        if stack.tiff.frames.first().is_some_and(|f| f.is_rgb()) {
            setup_rgb(&mut stack);
        }
        // Palette (indexed) images: the ColorMap is already installed as the
        // channel LUT by the lib; flag it so the fixed identity contrast window
        // is kept and its slider can be hidden.
        stack.palette = stack.tiff.frames.first().is_some_and(|f| f.is_palette());
        // Remember the file's own LUT (a single-channel image carrying a
        // ColorMap / ImageJ LUT) so the LUT selector can offer a "Built-in LUT"
        // option that restores it. Captured now, before pseudocolor could touch
        // it (it won't, for an explicit-LUT channel, but capture first anyway).
        stack.builtin_lut = (stack.tiff.meta.has_explicit_luts && stack.channel_settings.len() == 1)
            .then(|| stack.tiff.meta.channel_display[0].lut);
        refresh_pseudocolor(&mut stack, apply_pseudocolor);

        Ok(stack)
    }

    /// Rebuild the per-channel settings from the current metadata. Called after
    /// anything that changes the channel count or display info.
    pub fn rebuild_channel_settings(&mut self) {
        self.channel_settings = build_channel_settings(&self.tiff);
    }

    /// Frame count on the scrub/time axis (at least 1).
    pub fn frame_count(&self) -> usize {
        self.tiff.meta.frames.max(1)
    }

    /// The stack's pixel dimensions, from the first frame.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.tiff.frames.first().map(|f| (f.width, f.height))
    }

    /// The volume's depth axis: Z for a 4D stack (`slices > 1`), else the whole
    /// frame axis. Mirrors what `volume::build_volume` assembles.
    pub fn volume_depth(&self) -> u32 {
        let slices = self.tiff.meta.slices.max(1);
        if slices > 1 {
            slices as u32
        } else {
            self.tiff.meta.frames.max(1) as u32
        }
    }
}

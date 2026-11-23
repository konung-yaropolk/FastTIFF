//! [`Display`] — how the viewer is *showing* a stack, as opposed to what the
//! file said.
//!
//! `Document::tiff.meta` is the parsed file: whatever `fast-tiff-lib` read out of
//! the TIFF, and nothing else. It is never written to after `open`. Everything
//! the viewer or the user decides on top of it lives here instead:
//!
//! * the **interpretation** — how the flat plane chain is read as channels x Z
//!   x time, which routinely differs from the file's own labels (a mislabeled
//!   `channels=100` is really a frame count) and which the user can override;
//! * the **appearance** — the LUT actually rendered per channel and its
//!   contrast window, which the pseudocolor toggle and the LUT selector both
//!   rewrite;
//! * the **channel layout** derived once at load ([`ChannelLayout`]: what a
//!   display channel is — an IFD's intensities, an RGB or CMYK sample plane,
//!   or palette indices).
//!
//! Keeping the two apart is what makes "what did the file say?" answerable at
//! any time — the metadata panel shows the real thing, and the LUT selector's
//! "Built-in" option can restore the file's colours because nothing overwrote
//! them. When display state was written back into `meta`, that answer was lost
//! the moment a user touched a control, and each new display option needed its
//! own rescue copy of the value it was about to clobber.
//!
//! # Revisions
//!
//! Everything derived from the display model — a decoded window, a read-ahead
//! frame, a retained overview, a histogram — has to know whether it still
//! describes the model as it is now. [`Display`] answers that with a
//! [`Revision`]: one counter per concern, moved by the method that makes the
//! change, at the moment it makes it. A consumer records the revision it was
//! made under and compares; nothing has to diff copies of the settings, and
//! nothing has to remember to reset a cache when the model moves, because the
//! fields that would move it are private to the methods that move the counter.
//!
//! The LUTs have a counter of their own, apart from the [`Revision`]: a LUT is
//! applied as a frame is drawn, so nothing decoded depends on it, and only
//! what holds a copy of the LUTs themselves (a renderer's LUT texture) is
//! keyed on it.

use crate::document::{ChannelSettings, Window};
use fast_tiff_lib::{DisplayMode, StackMeta};
use scivis_render::{ChannelKind, Lut, MAX_CHANNELS};
use std::sync::atomic::{AtomicU64, Ordering};

/// How the plane chain is read as channels x Z-slices x time frames.
///
/// Seeded from the file's own labels, then re-resolved by
/// [`fast_tiff_lib::resolve_dimensions`] (which reclassifies implausible ones)
/// and optionally overridden by the user. `channels * slices * frames` always
/// equals the plane count — a change here only moves planes between axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
}

impl Default for Dims {
    fn default() -> Self {
        Dims {
            channels: 1,
            slices: 1,
            frames: 1,
        }
    }
}

/// Which version of the display model something was made from, one counter
/// per concern.
///
/// Everything derived from the display model — a decoded window, a read-ahead
/// frame, a retained overview, a histogram — has to know whether it still
/// describes the model as it is now. [`Display`] answers that with a
/// `Revision`, moved by the method that makes the change, at the moment it
/// makes it. A consumer records the revision it was made under and compares;
/// nothing has to diff copies of the settings, and nothing has to remember to
/// reset a cache when the model moves. The LUTs are not part of it: a LUT is
/// applied as a frame is drawn, so nothing decoded depends on it.
///
/// Every value is drawn from one process-wide sequence, so two revisions are
/// equal only when they are the same version of the same stack's model: a key
/// recorded against a stack that has since been closed can never match the
/// stack opened after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Revision {
    /// Moves when what a display channel decodes to changes: which IFD and
    /// sample plane it reads (the c/z/t interpretation, a dimension override,
    /// RGB or CMYK setup), how many channels there are, or a channel's texture
    /// format. Anything decoded under another `layout` is pixels of the wrong
    /// plane, or a buffer of the wrong format.
    pub layout: u64,
    /// Moves when a channel is switched on or off, and whenever the channels
    /// themselves are replaced (which a `layout` move always comes with).
    /// What depends on *which* channels were decoded — a frame read ahead for
    /// the enabled ones, a window built for them — keys on this as well; what
    /// covers every channel regardless, such as the histograms, does not.
    pub enabled: u64,
}

/// What a display channel is: which planes of the file it reads, and how its
/// values become colour. Decided once, from the file's first frame, when the
/// document is opened; the variants exclude each other because the TIFF
/// photometric interpretation they come from does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Each display channel is an image of intensities in IFDs of its own,
    /// the channels interleaved `xyczt`, each shown through its LUT and
    /// contrast window: a grayscale or a composite stack.
    #[default]
    Intensity,
    /// The display channels are the sample planes of each IFD (chunky or
    /// planar RGB), each shown through its colour ramp so that blending them
    /// gives back the pixel; samples past the third start switched off.
    Rgb,
    /// A CMYK (Separated) stack, its inks converted to three red, green and
    /// blue channels on decode. Addressed like [`Rgb`](Self::Rgb), the
    /// channels being planes of one IFD; told apart so that a frontend can say
    /// "CMYK", and a reader can tell the channels are derived, not stored.
    Cmyk,
    /// A palette-colour (indexed) stack: pixels are ColorMap indices, so the
    /// contrast window is pinned to an identity map and a frontend should
    /// show no contrast control for it.
    Palette,
}

impl ChannelLayout {
    /// Whether the display channels are the sample planes of each IFD, shown
    /// as red, green and blue ([`Rgb`](Self::Rgb), and
    /// [`Cmyk`](Self::Cmyk) once converted) rather than IFDs of their own.
    /// It flips how a display channel maps to the file's planes.
    pub fn shown_as_rgb(self) -> bool {
        matches!(self, ChannelLayout::Rgb | ChannelLayout::Cmyk)
    }

    /// Whether channels in this layout have a contrast window to adjust:
    /// every layout but [`Palette`](Self::Palette), whose window is a
    /// fixed index-to-LUT identity.
    pub fn has_contrast(self) -> bool {
        self != ChannelLayout::Palette
    }
}

/// The next value in the sequence every [`Revision`] is drawn from.
fn next_revision() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Each shown channel's texture format and on/off flag, as one small `Copy`
/// value.
///
/// Kept by [`Display`] beside the settings and rewritten by the methods that
/// change them, so a repaint reads it instead of collecting it from the
/// settings. Bounded by [`MAX_CHANNELS`], so it costs no allocation, and being
/// `Copy` it can be held across a mutable borrow of the stack it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChannelMasks {
    len: usize,
    kinds: [ChannelKind; MAX_CHANNELS],
    enabled: [bool; MAX_CHANNELS],
}

impl ChannelMasks {
    fn of(settings: &[ChannelSettings]) -> Self {
        let mut masks = ChannelMasks {
            len: settings.len().min(MAX_CHANNELS),
            ..ChannelMasks::default()
        };
        for (c, s) in settings.iter().take(MAX_CHANNELS).enumerate() {
            masks.kinds[c] = s.kind();
            masks.enabled[c] = s.enabled;
        }
        masks
    }

    /// How many channels are shown.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Each channel's texture format, in channel order.
    pub(crate) fn kinds(&self) -> &[ChannelKind] {
        &self.kinds[..self.len]
    }

    /// Whether each channel is switched on, in channel order.
    pub(crate) fn enabled(&self) -> &[bool] {
        &self.enabled[..self.len]
    }
}

impl Default for ChannelMasks {
    fn default() -> Self {
        ChannelMasks {
            len: 0,
            kinds: [ChannelKind::Int16; MAX_CHANNELS],
            enabled: [false; MAX_CHANNELS],
        }
    }
}

/// Everything the viewer decided about how to show the stack.
///
/// What a decode depends on — the interpretation, the channel layout, the
/// channels and which of them are on — is changed only through methods that
/// move the matching counter of [`revision`](Self::revision). So are the
/// LUTs, which a renderer's copy of them is keyed on, with a counter of their
/// own. A frontend reads all of it and changes what the user can: a
/// channel's window and whether it is on through a [`DisplayMut`]
/// ([`Document::display_mut`](crate::Document::display_mut)), and the
/// interpretation and the LUTs through the [`Viewer`](crate::Viewer)'s
/// operations.
///
/// Never handed out by value or by `&mut`, and neither cloned nor made empty
/// outside this crate: a display is its document's, and replacing one whole
/// would break what its methods keep, a `channels * slices * frames` that is
/// the document's plane count and a [`Revision`] no other display has.
#[derive(Debug)]
pub struct Display {
    /// The c/z/t interpretation in effect (see [`Dims`]).
    dims: Dims,
    /// Display mode in effect. Starts at the file's `mode=`, but RGB setup
    /// promotes it to `Color`.
    pub(crate) mode: DisplayMode,
    /// Per-channel contrast window and GPU upload format, one per shown
    /// channel, never more than [`MAX_CHANNELS`]. Kept separate from
    /// [`luts`](Self::luts) so it stays small and `Copy` — a `Lut` is 768
    /// bytes, and these are copied freely.
    settings: Vec<ChannelSettings>,
    /// The LUT actually rendered for each channel, same length and order as
    /// [`settings`](Self::settings). Starts as the file's own and then reflects
    /// whatever the user picked. Changed only through the methods that move
    /// [`lut_revision`](Self::lut_revision).
    luts: Vec<Lut>,
    /// The file's own LUT for a single-channel stack (a TIFF ColorMap or an
    /// ImageJ LUT), kept verbatim so the selector's "Built-in" option can put it
    /// back after the user has tried another. Also the LUT a reinterpretation
    /// down to one channel installed for it, when that is none of the
    /// selector's other options (see
    /// [`reselect_gray_lut`](Self::reselect_gray_lut)). `None` when the file
    /// supplied no LUT, or when the stack isn't single-channel.
    pub(crate) builtin_lut: Option<Lut>,
    /// What a display channel is (see [`ChannelLayout`]). Changed only
    /// through [`set_layout`](Self::set_layout), which moves
    /// [`Revision::layout`].
    layout: ChannelLayout,
    /// Set when the file's metadata described more image planes than the file
    /// contains, as `(declared, available)`. The dimensions above have been cut
    /// down to what fits; this is what was claimed before that.
    ///
    /// Not necessarily a broken file: a multi-file OME set gives every file the
    /// whole dataset's dimensions and points at its siblings. Either way the
    /// reader is looking at an arrangement the metadata does not back up, so it
    /// is said out loud rather than quietly corrected.
    pub(crate) plane_mismatch: Option<(usize, usize)>,
    /// Whether the file had a real Z axis (`slices > 1`) **as loaded**.
    /// Deliberately a load-time snapshot: an override that assigns 1 to Z must
    /// not collapse the three-way c/z/t choice to a two-way c/t swap and strand
    /// Z there.
    pub(crate) has_z_axis: bool,
    /// Which entry the single-channel LUT selector shows (`0` = the leading
    /// "Built-in" option when the file has one, else plain grayscale). Read
    /// only through [`gray_luts`](Self::gray_luts), which is `None` while the
    /// selector does not apply, and put on the option that is shown whenever
    /// the LUTs are reseeded, which discards the choice it describes.
    gray_lut_sel: usize,
    /// Each channel's texture format and on/off flag, kept in step with
    /// [`settings`](Self::settings) by every method that writes either.
    masks: ChannelMasks,
    /// See [`Revision`].
    revision: Revision,
    /// Which version of [`luts`](Self::luts) this is; see
    /// [`lut_revision`](Self::lut_revision).
    lut_revision: u64,
}

impl Display {
    /// A display of nothing, for a document to derive its own into: no
    /// channels, and the zero [`Revision`] that no derived one has.
    pub(crate) fn empty() -> Self {
        Display {
            dims: Dims::default(),
            plane_mismatch: None,
            // Grayscale until the file (or RGB setup) says otherwise — the same
            // conservative default `fast-tiff-lib` uses for a missing `mode=`.
            mode: DisplayMode::Grayscale,
            settings: Vec::new(),
            luts: Vec::new(),
            builtin_lut: None,
            layout: ChannelLayout::default(),
            has_z_axis: false,
            gray_lut_sel: 0,
            masks: ChannelMasks::default(),
            revision: Revision::default(),
            lut_revision: 0,
        }
    }

    /// A copy of this display as it is now, for a save to write from while
    /// the window carries on changing the original. A snapshot, not a second
    /// display: it is read, never installed in a document.
    pub(crate) fn snapshot(&self) -> Self {
        Display {
            settings: self.settings.clone(),
            luts: self.luts.clone(),
            ..*self
        }
    }

    /// How many channels are actually shown (capped at the shader's slot count).
    pub fn channel_count(&self) -> usize {
        self.settings.len()
    }

    /// The c/z/t interpretation in effect (see [`Dims`]).
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// What a display channel is: an IFD's intensities, an RGB or CMYK sample
    /// plane, or palette indices (see [`ChannelLayout`]).
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Display mode in effect: the file's `mode=`, or `Color` once RGB or
    /// CMYK setup has made the channels colour components.
    pub fn mode(&self) -> DisplayMode {
        self.mode
    }

    /// `(declared, available)` when the file's metadata described more image
    /// planes than the file holds, and the interpretation was cut down to
    /// what it does hold; `None` when the two agreed. Said to the user as
    /// [`ShapeNote::PlaneMismatch`](crate::ShapeNote::PlaneMismatch).
    pub fn plane_mismatch(&self) -> Option<(usize, usize)> {
        self.plane_mismatch
    }

    /// Whether the file had a real Z axis (`slices > 1`) as it was loaded,
    /// whatever the axes have been reassigned to since: a frontend offering
    /// the c/z/t order keeps offering three axes for such a file.
    pub fn has_z_axis(&self) -> bool {
        self.has_z_axis
    }

    /// The single-channel LUT selector, while it applies: exactly one
    /// channel, not RGB. `None` otherwise — for several channels the
    /// pseudocolor preference colours them instead
    /// ([`Viewer::set_pseudocolor`](crate::Viewer::set_pseudocolor)).
    /// Its choice is made with
    /// [`Viewer::set_gray_lut`](crate::Viewer::set_gray_lut).
    pub fn gray_luts(&self) -> Option<GrayLuts<'_>> {
        (!self.layout.shown_as_rgb() && self.settings.len() == 1).then_some(GrayLuts {
            builtin: self.builtin_lut.as_ref(),
            selected: self.gray_lut_sel,
        })
    }

    /// Every shown channel's settings, in channel order. Changed only through
    /// a [`DisplayMut`] ([`set_enabled`](DisplayMut::set_enabled),
    /// [`set_window`](DisplayMut::set_window) and the contrast buttons), or
    /// replaced whole by a reinterpretation
    /// ([`Viewer::set_dimension_order`](crate::Viewer::set_dimension_order)).
    pub fn settings(&self) -> &[ChannelSettings] {
        &self.settings
    }

    /// Channel `c`'s contrast window, in the file's own units; `None` past the
    /// last channel.
    pub fn window(&self, c: usize) -> Option<Window> {
        self.settings.get(c).map(|s| s.window)
    }

    /// Whether channel `c` is switched on; `false` past the last channel.
    pub fn enabled(&self, c: usize) -> bool {
        self.settings.get(c).is_some_and(|s| s.enabled)
    }

    /// Which version of the model this is; see [`Revision`].
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// Which version of the LUTs this is: moved by every method that changes
    /// one, at the moment it does, and drawn from the same process-wide
    /// sequence as the [`Revision`], so no two stacks' LUTs share a number.
    /// What holds a copy of them (a renderer's LUT texture) records the number
    /// it copied and compares.
    pub(crate) fn lut_revision(&self) -> u64 {
        self.lut_revision
    }

    /// Each channel's texture format and on/off flag, without collecting them.
    pub(crate) fn masks(&self) -> ChannelMasks {
        self.masks
    }

    /// What [`DisplayMut::set_enabled`] does.
    pub(crate) fn set_enabled(&mut self, c: usize, on: bool) -> bool {
        match self.settings.get_mut(c) {
            Some(s) if s.enabled != on => {
                s.enabled = on;
                self.masks.enabled[c] = on;
                self.revision.enabled = next_revision();
                true
            }
            _ => false,
        }
    }

    /// What [`DisplayMut::set_window`] does.
    pub(crate) fn set_window(&mut self, c: usize, window: Window) -> bool {
        if !self.layout.has_contrast() {
            return false;
        }
        match self.settings.get_mut(c) {
            Some(s) if s.window != window => {
                s.window = window;
                true
            }
            _ => false,
        }
    }

    /// Install a new c/z/t interpretation. Moves [`Revision::layout`]: every
    /// display channel may now read another IFD.
    pub(crate) fn set_dims(&mut self, dims: Dims) {
        self.dims = dims;
        self.revision.layout = next_revision();
    }

    /// Install what a display channel is. Moves [`Revision::layout`]: under
    /// [`ChannelLayout::Rgb`] or [`ChannelLayout::Cmyk`] a channel addresses a
    /// sample plane of the frame's IFD rather than an IFD of its own.
    pub(crate) fn set_layout(&mut self, layout: ChannelLayout) {
        self.layout = layout;
        self.revision.layout = next_revision();
    }

    /// Show the single-channel LUT selector's option `sel`: install its LUT
    /// and remember the choice. Returns whether it applied — `false`, with
    /// nothing changed, while the selector does not apply
    /// ([`gray_luts`](Self::gray_luts) is `None`) or `sel` is not one of its
    /// options.
    pub(crate) fn select_gray_lut(&mut self, sel: usize) -> bool {
        let Some(lut) = self
            .gray_luts()
            .filter(|luts| sel < luts.count())
            .map(|luts| luts.lut(sel))
        else {
            return false;
        };
        self.gray_lut_sel = sel;
        self.set_lut(0, lut);
        true
    }

    /// Replace every channel's settings: a new channel count, new formats, new
    /// windows. Moves both counters of the [`Revision`]. Anything past
    /// [`MAX_CHANNELS`] is dropped, as the shader has no slot for it.
    pub(crate) fn set_settings(&mut self, mut settings: Vec<ChannelSettings>) {
        settings.truncate(MAX_CHANNELS);
        self.masks = ChannelMasks::of(&settings);
        self.settings = settings;
        self.revision.layout = next_revision();
        self.revision.enabled = next_revision();
    }

    /// The LUT for channel `c`, by reference — a `Lut` is 768 bytes, so callers
    /// that only need to read one (tinting a widget, uploading a texture row)
    /// should never copy it.
    pub fn lut(&self, c: usize) -> Option<&Lut> {
        self.luts.get(c)
    }

    /// Every channel's rendered LUT, in channel order.
    pub fn luts(&self) -> &[Lut] {
        &self.luts
    }

    /// Replace channel `c`'s rendered LUT. Returns whether anything changed;
    /// when it did, the LUT revision has moved, and nothing else has.
    pub(crate) fn set_lut(&mut self, c: usize, lut: Lut) -> bool {
        match self.luts.get_mut(c) {
            Some(slot) if *slot != lut => {
                *slot = lut;
                self.lut_revision = next_revision();
                true
            }
            _ => false,
        }
    }

    /// Replace every channel's LUT, as a reinterpretation of the channels does
    /// (RGB or CMYK setup). Moves the LUT revision.
    pub(crate) fn set_luts(&mut self, luts: Vec<Lut>) {
        self.luts = luts;
        self.lut_revision = next_revision();
    }

    /// Reset the per-channel LUTs to what `meta` supplies for `channels`
    /// channels under the current mode, discarding any user choice. Used
    /// whenever the channel count changes and the old LUTs no longer correspond
    /// to the new channels.
    ///
    /// The file's own LUTs are kept when the count is unchanged (the usual case
    /// after re-resolving dimensions) so genuine per-channel colours from an
    /// ImageJ block survive; when it changes they're regenerated from `mode`,
    /// which also stops a collapsed single-channel stack from wearing a stale
    /// composite (e.g. red) LUT.
    ///
    /// Moves the LUT revision either way: it comes with a new channel layout,
    /// whose channels the old LUTs were not uploaded for. When the LUTs are
    /// regenerated, what the single-channel selector had chosen is no longer
    /// what is shown, so it goes to the option that is
    /// ([`reselect_gray_lut`](Self::reselect_gray_lut)).
    ///
    /// A [`Palette`](ChannelLayout::Palette) stack's channels all wear the
    /// file's ColorMap, which the metadata gives as the first channel's LUT:
    /// every channel's pixels are indices into that one map, and through any
    /// other LUT they are numbers, not the picture.
    pub(crate) fn reseed_luts(&mut self, meta: &StackMeta, channels: usize) {
        let keep = channels == self.luts.len();
        let palette = self.layout == ChannelLayout::Palette;
        self.lut_revision = next_revision();
        self.luts = (0..channels)
            .map(|c| {
                if keep {
                    self.luts[c]
                } else {
                    meta.channel_display
                        .get(if palette { 0 } else { c })
                        .map(|d| d.lut)
                        .unwrap_or_else(|| fast_tiff_lib::default_lut_for(self.mode, c))
                }
            })
            .collect();
        if !keep {
            self.reselect_gray_lut(channels);
        }
    }

    /// Put the single-channel LUT selector on the option that is shown, for a
    /// display of `channels` channels whose LUTs were just installed without
    /// the selector choosing one (an open, or a reinterpretation that
    /// regenerated them): the first option whose LUT channel 0 wears, or,
    /// when it is none of them, that LUT as the leading "Built-in" option.
    ///
    /// What is installed is the file's LUT for the channel, or its mode's
    /// default, and neither need be option 0: a composite file's first
    /// channel is red. Without this the selector would name one LUT while
    /// another is shown. Nothing when the selector does not apply.
    pub(crate) fn reselect_gray_lut(&mut self, channels: usize) {
        self.gray_lut_sel = 0;
        let Some(&shown) = self.luts.first() else {
            return;
        };
        if channels != 1 || self.layout.shown_as_rgb() {
            return;
        }
        let options = GrayLuts {
            builtin: self.builtin_lut.as_ref(),
            selected: 0,
        };
        match (0..options.count()).find(|&option| options.lut(option) == shown) {
            Some(option) => self.gray_lut_sel = option,
            None => self.builtin_lut = Some(shown),
        }
    }

    /// The number of channels a stack with `channels` in its metadata will
    /// actually show, clamped to what the compositing shader has slots for.
    pub(crate) fn shown_channels(channels: usize) -> usize {
        channels.min(MAX_CHANNELS)
    }
}

/// A document's [`Display`], lent out to change what the user changes of it
/// directly: a channel's contrast window, whether a channel is on, and the
/// contrast buttons. Had from
/// [`Document::display_mut`](crate::Document::display_mut); everything else
/// of the display reads through it.
///
/// A handle rather than `&mut Display`, so that a display is only ever
/// changed through these, never replaced, swapped or taken whole: that is
/// what keeps its interpretation the document's plane count and its
/// [`Revision`] its own.
///
/// ```compile_fail
/// # fn swap(a: &mut fast_tiff_viewer::Document, b: &mut fast_tiff_viewer::Document) {
/// // There is no `&mut Display` to swap through.
/// std::mem::swap(&mut *a.display_mut(), &mut *b.display_mut());
/// # }
/// ```
#[derive(Debug)]
pub struct DisplayMut<'a> {
    display: &'a mut Display,
}

impl std::ops::Deref for DisplayMut<'_> {
    type Target = Display;

    fn deref(&self) -> &Display {
        self.display
    }
}

impl<'a> DisplayMut<'a> {
    pub(crate) fn new(display: &'a mut Display) -> Self {
        DisplayMut { display }
    }

    /// Switch channel `c` on or off. Returns whether that changed anything;
    /// when it did, [`Revision::enabled`] has moved, and nothing else has.
    pub fn set_enabled(&mut self, c: usize, on: bool) -> bool {
        self.display.set_enabled(c, on)
    }

    /// Set channel `c`'s contrast window. Returns whether that changed
    /// anything.
    ///
    /// No revision moves: the window is applied as the frame is drawn, and
    /// nothing decoded or binned depends on it. Refused, changing nothing, for
    /// a [`Palette`](ChannelLayout::Palette) channel, as
    /// [`auto_contrast`](Self::auto_contrast) and
    /// [`reset_contrast`](Self::reset_contrast) are: its window is the pinned
    /// index-to-LUT identity, which nothing could then put back.
    pub fn set_window(&mut self, c: usize, window: Window) -> bool {
        self.display.set_window(c, window)
    }

    /// Fit every contrast window to where its data actually is, using
    /// histograms already computed for the frame on screen.
    ///
    /// Reads the histograms rather than rescanning the pixels: they describe
    /// the frame the user is looking at and are already in hand, so Auto
    /// costs nothing and — more importantly — agrees with the plot it is
    /// clicked underneath. Opening a file instead scans frame 0, which is the
    /// right choice there and the wrong one here: on a stack whose brightness
    /// drifts, seeding from frame 0 is a reasonable default, but auto-ing to
    /// it while looking at frame 900 would be visibly wrong.
    ///
    /// A channel with no histogram, or one whose histogram counted nothing,
    /// keeps the window it had. Palette stacks are skipped entirely: their
    /// window is a pinned index-to-LUT identity, not a contrast choice, and
    /// moving it would remap every colour.
    pub fn auto_contrast(&mut self, hists: &[crate::histogram::Histogram]) {
        self.display.auto_contrast(hists);
    }

    /// Put every contrast window back where it was when the file was opened —
    /// the undo for any amount of dragging, and for
    /// [`auto_contrast`](Self::auto_contrast).
    ///
    /// The window the file asked for, not the full slider track. Those differ
    /// whenever a file declares a display range narrower than its data, which
    /// is the normal case for anything saved out of ImageJ: opening the window
    /// to the full track would "reset" such a file to something it has never
    /// been shown as. Restoring what it opened with is the state the user is
    /// actually trying to get back to, and it costs nothing to remember — see
    /// [`ChannelSettings::initial`].
    ///
    /// Reopening a *different* interpretation of the same file — a
    /// dimension-order change, for instance — rebuilds the settings and with
    /// them the window this returns to, which is right: the starting point is
    /// whatever the current reading of the file starts from.
    pub fn reset_contrast(&mut self) {
        self.display.reset_contrast();
    }
}

/// The single-channel LUT selector: the options a frontend lists, and which
/// one is shown. Had from [`Display::gray_luts`], only while it applies.
///
/// The options are, in order: the file's own LUT as "Built-in" when it has
/// one, plain grayscale, the named channel colours, then the perceptual
/// colormaps. An option is its index in that list.
#[derive(Clone, Copy, Debug)]
pub struct GrayLuts<'a> {
    /// The file's own LUT, when it has one: the leading option.
    builtin: Option<&'a Lut>,
    /// The option on show.
    selected: usize,
}

impl GrayLuts<'_> {
    /// The option on show.
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// How many options there are.
    pub fn count(&self) -> usize {
        crate::channels::gray_lut_option_count() + self.offset()
    }

    /// Option `option`'s name, for the list.
    ///
    /// # Panics
    ///
    /// When `option` is not below [`count`](Self::count).
    pub fn name(&self, option: usize) -> &'static str {
        if self.builtin.is_some() && option == 0 {
            "Built-in"
        } else {
            crate::channels::gray_lut_name(option - self.offset())
        }
    }

    /// The colour to tint option `option`'s entry with — its LUT's middle,
    /// see [`crate::channels::ui_tint`] — without building its LUT; `None`
    /// for one whose middle is grey.
    ///
    /// A `Lut` is 768 bytes and a frontend asks this of *every* option on
    /// every frame its list is open, so the generated ramps' entry is worked
    /// out from their base colour instead.
    ///
    /// # Panics
    ///
    /// When `option` is not below [`count`](Self::count).
    pub fn tint(&self, option: usize) -> Option<[u8; 3]> {
        if let (Some(lut), 0) = (self.builtin, option) {
            return crate::channels::ui_tint(lut);
        }
        crate::channels::gray_lut_tint(option - self.offset())
    }

    /// The 256-entry LUT of option `option`, by value, for installing.
    fn lut(&self, option: usize) -> Lut {
        match self.builtin {
            Some(lut) if option == 0 => *lut,
            _ => crate::channels::gray_lut_for(option - self.offset()),
        }
    }

    /// 1 when the leading "Built-in" option is there, else 0.
    fn offset(&self) -> usize {
        self.builtin.is_some() as usize
    }
}

#[cfg(test)]
#[path = "display_tests.rs"]
mod tests;

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
//! * a few **layout facts** derived once at load (chunky/planar RGB, palette).
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
/// per concern. See the [module docs](self).
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
    // Read by the per-frame GPU path (`crate::sync`), which needs a backend.
    #[cfg_attr(
        not(any(feature = "backend-wgpu", feature = "backend-glow")),
        allow(dead_code)
    )]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Each channel's texture format, in channel order.
    #[cfg_attr(
        not(any(feature = "backend-wgpu", feature = "backend-glow")),
        allow(dead_code)
    )]
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
/// What a decode depends on — the interpretation, the RGB layout, the channels
/// and which of them are on — is private, and changed only through methods
/// that move the matching counter of [`revision`](Self::revision). So are the
/// LUTs, which a renderer's copy of them is keyed on, with a counter of their
/// own. The rest is appearance nothing is keyed on, and stays an open field.
#[derive(Clone, Debug)]
pub struct Display {
    /// The c/z/t interpretation in effect (see [`Dims`]).
    dims: Dims,
    /// Display mode in effect. Starts at the file's `mode=`, but RGB setup
    /// promotes it to `Color`.
    pub mode: DisplayMode,
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
    /// back after the user has tried another. `None` when the file supplied no
    /// LUT, or when the stack isn't single-channel.
    pub builtin_lut: Option<Lut>,
    /// True when each IFD is an RGB image and the "channels" are its sample
    /// planes rather than separate IFDs. Flips how a display channel maps to
    /// file data.
    rgb: bool,
    /// True when the stack is CMYK (Separated) and is being shown through the
    /// ink->RGB conversion. [`rgb`](Self::rgb) is *also* set — the display
    /// channels are sample planes of one IFD either way, which is what the
    /// decode addressing keys on — so this exists purely so the UI can say
    /// "CMYK" rather than "RGB", and so a reader of this struct can tell that
    /// the three channels are derived rather than stored.
    cmyk: bool,
    /// True for a palette-color (indexed) stack: pixels are ColorMap indices,
    /// so the contrast window is pinned to an identity map and a frontend
    /// should suppress the contrast slider.
    pub palette: bool,
    /// Set at load when the file genuinely has channels, Z *and* time all > 1.
    /// Z is then frozen at its first slice. Describes the file, so it survives a
    /// manual dimension-order change (which never touches Z).
    pub triple_axis_warning: bool,
    /// Set when the file's metadata described more image planes than the file
    /// contains, as `(declared, available)`. The dimensions above have been cut
    /// down to what fits; this is what was claimed before that.
    ///
    /// Not necessarily a broken file: a multi-file OME set gives every file the
    /// whole dataset's dimensions and points at its siblings. Either way the
    /// reader is looking at an arrangement the metadata does not back up, so it
    /// is said out loud rather than quietly corrected.
    pub plane_mismatch: Option<(usize, usize)>,
    /// Whether the file had a real Z axis (`slices > 1`) **as loaded**.
    /// Deliberately a load-time snapshot: an override that assigns 1 to Z must
    /// not collapse the three-way c/z/t choice to a two-way c/t swap and strand
    /// Z there.
    pub has_z_axis: bool,
    /// Which entry the single-channel LUT selector currently shows (`0` = the
    /// leading "Built-in" option when the file has one, else plain grayscale).
    /// Only meaningful while [`crate::channels::gray_lut_applicable`] holds.
    pub gray_lut_sel: usize,
    /// Each channel's texture format and on/off flag, kept in step with
    /// [`settings`](Self::settings) by every method that writes either.
    masks: ChannelMasks,
    /// See [`Revision`].
    revision: Revision,
    /// Which version of [`luts`](Self::luts) this is; see
    /// [`lut_revision`](Self::lut_revision).
    lut_revision: u64,
}

impl Default for Display {
    fn default() -> Self {
        Display {
            dims: Dims::default(),
            plane_mismatch: None,
            // Grayscale until the file (or RGB setup) says otherwise — the same
            // conservative default `fast-tiff-lib` uses for a missing `mode=`.
            mode: DisplayMode::Grayscale,
            settings: Vec::new(),
            luts: Vec::new(),
            builtin_lut: None,
            rgb: false,
            cmyk: false,
            palette: false,
            triple_axis_warning: false,
            has_z_axis: false,
            gray_lut_sel: 0,
            masks: ChannelMasks::default(),
            revision: Revision::default(),
            lut_revision: 0,
        }
    }
}

impl Display {
    /// How many channels are actually shown (capped at the shader's slot count).
    pub fn channel_count(&self) -> usize {
        self.settings.len()
    }

    /// The c/z/t interpretation in effect (see [`Dims`]).
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Whether the display channels are the sample planes of each IFD (chunky
    /// or planar RGB, and CMYK shown as RGB) rather than IFDs of their own.
    pub fn rgb(&self) -> bool {
        self.rgb
    }

    /// Whether the stack is CMYK shown through the ink-to-RGB conversion;
    /// [`rgb`](Self::rgb) holds too whenever this does.
    pub fn cmyk(&self) -> bool {
        self.cmyk
    }

    /// Every shown channel's settings, in channel order. Changed only through
    /// [`set_enabled`](Self::set_enabled) and [`set_window`](Self::set_window),
    /// or replaced whole by a reinterpretation (see [`crate::dimensions`]).
    pub fn settings(&self) -> &[ChannelSettings] {
        &self.settings
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
    // Read by the per-frame GPU path (`crate::sync`), which needs a backend.
    #[cfg_attr(
        not(any(feature = "backend-wgpu", feature = "backend-glow")),
        allow(dead_code)
    )]
    pub(crate) fn lut_revision(&self) -> u64 {
        self.lut_revision
    }

    /// Each channel's texture format and on/off flag, without collecting them.
    pub(crate) fn masks(&self) -> ChannelMasks {
        self.masks
    }

    /// Switch channel `c` on or off. Returns whether that changed anything;
    /// when it did, [`Revision::enabled`] has moved, and nothing else has.
    pub fn set_enabled(&mut self, c: usize, on: bool) -> bool {
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

    /// Set channel `c`'s contrast window. Returns whether that changed
    /// anything.
    ///
    /// No revision moves: the window is applied as the frame is drawn, and
    /// nothing decoded or binned depends on it.
    pub fn set_window(&mut self, c: usize, window: Window) -> bool {
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

    /// Show each IFD's sample planes as the display channels: chunky or planar
    /// RGB, or with `cmyk` a CMYK stack converted to RGB. Moves
    /// [`Revision::layout`], since a channel then addresses a plane of the
    /// frame's IFD rather than an IFD of its own.
    pub(crate) fn set_sample_planes(&mut self, cmyk: bool) {
        self.rgb = true;
        self.cmyk = cmyk;
        self.revision.layout = next_revision();
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
    pub fn set_lut(&mut self, c: usize, lut: Lut) -> bool {
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
    /// whose channels the old LUTs were not uploaded for.
    pub fn reseed_luts(&mut self, meta: &StackMeta, channels: usize) {
        let keep = channels == self.luts.len();
        self.lut_revision = next_revision();
        self.luts = (0..channels)
            .map(|c| {
                if keep {
                    self.luts[c]
                } else {
                    meta.channel_display
                        .get(c)
                        .map(|d| d.lut)
                        .unwrap_or_else(|| fast_tiff_lib::default_lut_for(self.mode, c))
                }
            })
            .collect();
    }

    /// The number of channels a stack with `channels` in its metadata will
    /// actually show, clamped to what the compositing shader has slots for.
    pub fn shown_channels(channels: usize) -> usize {
        channels.min(MAX_CHANNELS)
    }
}

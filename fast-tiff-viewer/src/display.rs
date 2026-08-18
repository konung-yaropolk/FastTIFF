//! [`Display`] — how the viewer is *showing* a stack, as opposed to what the
//! file said.
//!
//! `Stack::tiff.meta` is the parsed file: whatever `fast-tiff-lib` read out of
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

use crate::stack::ChannelSettings;
use fast_tiff_lib::{DisplayMode, StackMeta};
use scivis_render::{Lut, MAX_CHANNELS};

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
        Dims { channels: 1, slices: 1, frames: 1 }
    }
}

/// Everything the viewer decided about how to show the stack.
#[derive(Clone, Debug)]
pub struct Display {
    /// The c/z/t interpretation in effect (see [`Dims`]).
    pub dims: Dims,
    /// Display mode in effect. Starts at the file's `mode=`, but RGB setup
    /// promotes it to `Color`.
    pub mode: DisplayMode,
    /// Per-channel contrast window and GPU upload format, one per shown
    /// channel. Kept separate from [`luts`](Self::luts) so it stays small and
    /// `Copy` — a `Lut` is 768 bytes, and these are copied freely.
    pub settings: Vec<ChannelSettings>,
    /// The LUT actually rendered for each channel, same length and order as
    /// [`settings`](Self::settings). Starts as the file's own and then reflects
    /// whatever the user picked.
    pub luts: Vec<Lut>,
    /// The file's own LUT for a single-channel stack (a TIFF ColorMap or an
    /// ImageJ LUT), kept verbatim so the selector's "Built-in" option can put it
    /// back after the user has tried another. `None` when the file supplied no
    /// LUT, or when the stack isn't single-channel.
    pub builtin_lut: Option<Lut>,
    /// True when each IFD is an RGB image and the "channels" are its sample
    /// planes rather than separate IFDs. Flips how a display channel maps to
    /// file data.
    pub rgb: bool,
    /// True for a palette-color (indexed) stack: pixels are ColorMap indices,
    /// so the contrast window is pinned to an identity map and a frontend
    /// should suppress the contrast slider.
    pub palette: bool,
    /// Set at load when the file genuinely has channels, Z *and* time all > 1.
    /// Z is then frozen at its first slice. Describes the file, so it survives a
    /// manual dimension-order change (which never touches Z).
    pub triple_axis_warning: bool,
    /// Whether the file had a real Z axis (`slices > 1`) **as loaded**.
    /// Deliberately a load-time snapshot: an override that assigns 1 to Z must
    /// not collapse the three-way c/z/t choice to a two-way c/t swap and strand
    /// Z there.
    pub has_z_axis: bool,
    /// Which entry the single-channel LUT selector currently shows (`0` = the
    /// leading "Built-in" option when the file has one, else plain grayscale).
    /// Only meaningful while [`crate::channels::gray_lut_applicable`] holds.
    pub gray_lut_sel: usize,
}

impl Default for Display {
    fn default() -> Self {
        Display {
            dims: Dims::default(),
            // Grayscale until the file (or RGB setup) says otherwise — the same
            // conservative default `fast-tiff-lib` uses for a missing `mode=`.
            mode: DisplayMode::Grayscale,
            settings: Vec::new(),
            luts: Vec::new(),
            builtin_lut: None,
            rgb: false,
            palette: false,
            triple_axis_warning: false,
            has_z_axis: false,
            gray_lut_sel: 0,
        }
    }
}

impl Display {
    /// How many channels are actually shown (capped at the shader's slot count).
    pub fn channel_count(&self) -> usize {
        self.settings.len()
    }

    /// The LUT for channel `c`, by reference — a `Lut` is 768 bytes, so callers
    /// that only need to read one (tinting a widget, uploading a texture row)
    /// should never copy it.
    pub fn lut(&self, c: usize) -> Option<&Lut> {
        self.luts.get(c)
    }

    /// Replace channel `c`'s rendered LUT. Returns whether anything changed, so
    /// a caller can skip re-uploading when it didn't.
    pub fn set_lut(&mut self, c: usize, lut: Lut) -> bool {
        match self.luts.get_mut(c) {
            Some(slot) if *slot != lut => {
                *slot = lut;
                true
            }
            _ => false,
        }
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
    pub fn reseed_luts(&mut self, meta: &StackMeta, channels: usize) {
        let keep = channels == self.luts.len();
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

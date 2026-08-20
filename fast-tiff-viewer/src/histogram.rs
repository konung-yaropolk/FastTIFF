//! Intensity histograms of the frame currently on screen, one per display
//! channel.
//!
//! Every channel is binned over **one shared track** — the union of all the
//! channels' slider bounds (`ChannelSettings::bounds`) — rather than over the
//! raw sample type or each channel's own range.
//!
//! The shared axis is the whole point. These are drawn overlaid on a single
//! canvas, and curves plotted on per-channel axes would each be centred in their
//! own range: three channels with quite different distributions come out as one
//! bell sitting on top of itself, which says the opposite of the truth. On a
//! common axis a dim channel sits left of a bright one, which is the comparison
//! a composite histogram exists to make. For the overwhelmingly common case of
//! channels that share a range the two are identical anyway.
//!
//! Decoding is the expensive part, so a caller should hold the result and
//! recompute only when the frame — or the interpretation of it — changes.

use crate::prefetch::{build_jobs, decode_jobs, Decoded};
use crate::stack::Stack;
use scivis_render::{ChannelKind, Lut};

/// Bins per histogram. Matches the 256 entries of a [`Lut`], so one bin is
/// exactly one LUT entry for 8-bit and palette data.
pub const BINS: usize = 256;

/// How many samples one channel's histogram looks at, at most.
///
/// Same reasoning as the auto-contrast scan in [`crate::channels`]: this runs on
/// the UI thread whenever the frame changes while the histogram is on screen,
/// and a 4K frame is 16M samples per channel. A strided subsample of this many
/// points reproduces the *shape* of the distribution — which is all a histogram
/// is read for — without making scrubbing lurch.
const SAMPLE_BUDGET: usize = 1 << 20; // 1048576

/// One channel's binned intensity distribution.
#[derive(Clone, Debug)]
pub struct Histogram {
    /// Which display channel this describes.
    pub channel: usize,
    /// Sample counts, low track value first. Always [`BINS`] long.
    pub bins: Vec<u32>,
    /// The shared track this was binned over: `lo` is bin 0's left edge and `hi`
    /// the last bin's right edge. The same for every channel in one call, which
    /// is what makes the bins comparable across channels.
    pub lo: f32,
    pub hi: f32,
    /// The tallest bin, for scaling the plot. Never zero once any sample landed.
    pub peak: u32,
    /// How many samples were counted (after subsampling).
    pub counted: u64,
}

/// Histogram every display channel of the stack's current frame.
///
/// Returns an empty vec when nothing is loaded or the frame fails to decode —
/// a histogram is a read-only view, so a decode error here is not worth
/// surfacing separately; the image itself will already be reporting it.
pub fn frame_histograms(stack: &Stack) -> Vec<Histogram> {
    let n = stack.display.settings.len();
    if n == 0 {
        return Vec::new();
    }
    // Every channel, including any the user has switched off. A frontend that
    // only plots the enabled ones can then honour a checkbox by redrawing
    // instead of decoding the frame again, which is what makes toggling one
    // feel instant.
    let enabled = vec![true; n];
    let kinds: Vec<ChannelKind> = stack.display.settings.iter().map(|s| s.kind).collect();
    let (lo, hi) = shared_track(stack);
    let jobs = build_jobs(stack, stack.frame_index, &enabled, &kinds);
    let Ok(decoded) = decode_jobs(&stack.tiff.data, &stack.tiff.frames, stack.tiff.byte_order, &jobs)
    else {
        return Vec::new();
    };
    jobs.iter()
        .zip(decoded)
        .map(|(job, data)| bin(job.channel, &data, lo, hi))
        .collect()
}

/// The axis every channel is binned onto: the union of their slider tracks, so
/// no channel's data falls off either end.
///
/// Public because a frontend drawing the plot needs the same number to label an
/// axis or place a marker, and recomputing it there would be one more place for
/// the two to disagree.
pub fn shared_track(stack: &Stack) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for s in &stack.display.settings {
        lo = lo.min(s.bounds.0);
        hi = hi.max(s.bounds.1);
    }
    // No channels, or bounds that are somehow not finite: a degenerate track
    // would divide by zero in `bin`, so hand back something usable instead.
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return (0.0, u16::MAX as f32);
    }
    (lo, hi)
}

/// Bin one decoded channel over the track `lo..hi`.
fn bin(channel: usize, data: &Decoded, lo: f32, hi: f32) -> Histogram {
    let mut bins = vec![0u32; BINS];
    let span = (hi - lo).max(f32::EPSILON);
    let mut counted = 0u64;

    // Shared tail: value -> bin. `as usize` on a negative or NaN float saturates
    // to 0 in Rust, and the clamp catches the high end, so an out-of-track
    // sample piles into the nearest edge bin rather than being dropped — the
    // same thing the contrast handles do with out-of-track values.
    let mut add = |v: f32| {
        let t = (v - lo) / span * BINS as f32;
        let idx = (t as usize).min(BINS - 1);
        bins[idx] += 1;
        counted += 1;
    };

    match data {
        // 8-bit channels keep their raw 0..255 samples, while their slider
        // track lives in the widened 0..65535 space every other integer channel
        // uses (see `build_channel_settings`). 257 is that widening: it is what
        // `(v << 8) | v` computes, so this lands on the same axis the 16-bit
        // path produces for identical data.
        Decoded::U8(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                add(s as f32 * 257.0);
            }
        }
        Decoded::U16(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                add(s as f32);
            }
        }
        // Float channels are binned in their own units, matching how their
        // contrast window is expressed. NaNs and infinities belong to no bin.
        Decoded::F32(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                if s.is_finite() {
                    add(s);
                }
            }
        }
    }

    let peak = bins.iter().copied().max().unwrap_or(0);
    Histogram { channel, bins, lo, hi, peak, counted }
}

/// Step between sampled pixels so a frame contributes at most
/// [`SAMPLE_BUDGET`] points. A prime-ish stride would be better on a
/// power-of-two-wide image, but unlike the min/max scan a histogram is a
/// *distribution* — hitting the same column of every row still samples the same
/// population, so a plain quotient is fine here.
fn stride(len: usize) -> usize {
    (len / SAMPLE_BUDGET).max(1)
}

/// Fill colour for a channel's histogram: the brightest colour that channel is
/// drawn in, so a plot is matched to its channel by eye without a legend.
///
/// The *brightest* entry rather than the top one, because the top of a LUT is
/// not reliably its brightest point. A contrast-stretched palette export blacks
/// out the unused tail of its colour table, so entry 255 is literally black and
/// a histogram painted with it is invisible against the plot. Perceptual
/// colormaps have the same shape at the other end.
///
/// The floor covers a LUT whose brightest entry is still too dark to see at all
/// — an all-black colour table is legal — by falling back to a neutral. Judged
/// on the largest channel rather than luminance: pure blue is dim by luminance
/// and perfectly visible on screen.
///
/// Raw RGB rather than a toolkit colour, like the other tint helpers in
/// [`crate::channels`] — deciding *which* colour is display logic, converting it
/// is the frontend's.
pub fn fill_tint(lut: &Lut) -> [u8; 3] {
    /// Below this, the brightest channel of the colour is too dark to read
    /// against the plot background.
    const MIN_VISIBLE: u8 = 64;
    const NEUTRAL: [u8; 3] = [180, 180, 180];

    let brightest = lut
        .iter()
        .copied()
        .max_by_key(|c| c[0] as u32 + c[1] as u32 + c[2] as u32)
        .unwrap_or(NEUTRAL);
    if brightest.iter().copied().max().unwrap_or(0) < MIN_VISIBLE {
        NEUTRAL
    } else {
        brightest
    }
}

/// Alpha for `n` histograms sharing one canvas.
///
/// Every plot is drawn over the same axes, so the fills stack; without thinning
/// them the last channel drawn simply hides the rest. Falling off as `1/sqrt(n)`
/// lets each added channel show through the ones before it, and the floor stops
/// a six-channel stack from fading into the background entirely.
///
/// Deliberately well short of opaque even for a single channel. These are read
/// as overlapping shapes rather than as solid bars, the outline stroke carries
/// each curve's edge at full strength, and a translucent fill lets the plot
/// background — and anything drawn behind a curve — stay visible.
pub fn fill_alpha(n: usize) -> u8 {
    const SOLO: f32 = 120.0;
    const FLOOR: f32 = 40.0;
    let n = n.max(1) as f32;
    (SOLO / n.sqrt()).max(FLOOR) as u8
}

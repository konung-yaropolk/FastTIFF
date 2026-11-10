//! Intensity histograms of the frame currently on screen, one per display
//! channel.
//!
//! Every channel is binned over **one shared track** — the union of all the
//! channels' slider bounds (`ChannelSettings::bounds`) — rather than over the
//! raw sample type or each channel's own range. Like the bounds, the track is
//! in the file's own sample units, so a bin edge, a slider handle and a
//! contrast window all mean the same value.
//!
//! The shared axis is the whole point. These are drawn overlaid on a single
//! canvas, and curves plotted on per-channel axes would each be centred in their
//! own range: three channels with quite different distributions come out as one
//! bell sitting on top of itself, which says the opposite of the truth. On a
//! common axis a dim channel sits left of a bright one, which is the comparison
//! a composite histogram exists to make. For the overwhelmingly common case of
//! channels that share a range the two are identical anyway.
//!
//! # Made from the decodes that show the frame
//!
//! Decoding is the expensive part, and the frame on screen has already been
//! decoded once, to draw it. So the histograms are binned from those planes:
//! the ones `crate::sync` decodes inline, or takes from the read-ahead — whose
//! worker can bin the frame it decodes ahead while it holds it, off the
//! interface thread. They are held on the stack as its [`FrameHistograms`],
//! one of the document's services: whichever view decoded the frame, they are
//! the frame's.
//!
//! What no decode for the screen covers is decoded for them, and that is not
//! nothing:
//!
//! - the channels switched off, on every frame shown. On a chunky RGB or CMYK
//!   frame the channels are planes of one IFD, and the ones on screen were
//!   decoded in one pass without it, so a channel switched off is a second
//!   decompression of the whole frame;
//! - every channel, once, when they are first asked for on a frame already on
//!   screen, or when a movie stops on a frame shown between two binnings;
//! - in the 3D view, the 2D frame, which the volume does not hold.
//!
//! A frame too large for one texture is never decoded whole: its histograms
//! are binned from a view of the whole of it at a coarse stride
//! ([`HistogramSource::Overview`]), as that view is built and from the
//! overview kept for zooming out. Zoomed in after moving to another frame
//! there is no such view of it, and so no histogram, until a zoom out builds
//! one.
//!
//! All of that only while a frontend asks for them
//! ([`Viewer::histograms_wanted`](crate::Viewer::histograms_wanted)). While a
//! movie plays, a frame is binned at most once every [`PLAYBACK_INTERVAL`] of
//! its clock: by the read-ahead's worker when it is reading ahead, which is
//! asked to bin only then, since a frame it bins is published that much later;
//! otherwise by the interface thread, from the planes it decoded. Nothing else
//! is decoded for them during playback but the channels switched off of the
//! frame binned.
//!
//! [`frame_histograms`] decodes and bins every channel outright, the way the
//! held histograms used to be made. It is the reference they are tested
//! against, and what a caller without a renderer, and so without a sync, uses.

use crate::document::Document;
use crate::prefetch::{build_jobs, decode_jobs_borrowed, Decoded, Samples};
use crate::residency::Residency;
use crate::units::{SampleUnits, Window};
use crate::window::Overview;
use scivis_render::{Lut, MAX_CHANNELS};

/// Bins per histogram. Matches the 256 entries of a [`Lut`], so one bin is
/// exactly one LUT entry for 8-bit and palette data.
pub const BINS: usize = 256;

/// How many samples one channel's histogram looks at, at most.
///
/// Same reasoning as the auto-contrast scan in [`crate::channels`]: a 4K frame
/// is 16M samples per channel, and binning all of them on every frame would be
/// tens of milliseconds the picture is waiting for. A strided subsample of this
/// many points reproduces the *shape* of the distribution — which is all a
/// histogram is read for — without making scrubbing lurch.
const SAMPLE_BUDGET: usize = 1 << 20; // 1048576

/// The least time, on the playback clock, between two binnings on the
/// interface thread while a movie plays.
///
/// Binning is no decode, but a million samples is still about 4 ms a channel
/// (measured in release on a 2048 x 2048 frame), and at a hundred frames a
/// second that would be most of the thread drawing them. Nobody reads a
/// distribution redrawn that often; a few updates a second still looks live.
/// The read-ahead worker is held to the same pace: asked to bin a frame only
/// when this thread could have and did not, since a frame it bins is
/// published that much later, and one not ready when it is wanted is decoded
/// again here.
pub(crate) const PLAYBACK_INTERVAL: f64 = 0.25;

/// One channel's binned intensity distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct Histogram {
    /// Which display channel this describes.
    pub channel: usize,
    /// Sample counts, low track value first. Always [`BINS`] long.
    pub bins: Vec<u32>,
    /// The shared track this was binned over, in file units: `lo` is bin 0's
    /// left edge and `hi` the last bin's right edge. The same for every channel
    /// in one call, which is what makes the bins comparable across channels.
    pub lo: f32,
    pub hi: f32,
    /// The tallest bin, for scaling the plot. Never zero once any sample landed.
    pub peak: u32,
    /// How many samples were counted (after subsampling).
    pub counted: u64,
}

/// What a frame's histograms are a function of: the frame, the channel layout
/// it is decoded through
/// ([`Revision::layout`](crate::display::Revision::layout)), and each
/// channel's slider track, the axis the bins are laid out on. Held with the
/// histograms ([`FrameHistograms`]) to say whether they still describe the
/// stack.
///
/// Not which channels are on: every channel is histogrammed, on or off, so a
/// frontend that plots only the enabled ones honours a checkbox by redrawing
/// rather than decoding the frame again. And not the contrast windows, which
/// slide along the distribution rather than change it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HistogramKey {
    frame: usize,
    layout: u64,
    bounds: Vec<Window>,
}

impl HistogramKey {
    /// The key of `stack`'s histograms as it stands.
    pub(crate) fn of(stack: &Document) -> Self {
        HistogramKey {
            frame: stack.frame(),
            layout: stack.display.revision().layout,
            bounds: stack.display.settings().iter().map(|s| s.bounds).collect(),
        }
    }

    /// Whether histograms built under this key still describe `stack`: the
    /// same as `*self == HistogramKey::of(stack)`, without building one, as a
    /// sync asks on every repaint.
    pub(crate) fn describes(&self, stack: &Document) -> bool {
        self.frame == stack.frame()
            && self.layout == stack.display.revision().layout
            && self
                .bounds
                .iter()
                .eq(stack.display.settings().iter().map(|s| &s.bounds))
    }
}

/// Where a frame's histograms were binned from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistogramSource {
    /// The frame itself, at its own resolution: the planes decoded to show it,
    /// with at most a million samples of each binned.
    Frame,
    /// The overview of a frame too large for one texture
    /// ([`crate::window::Overview`]): every `stride`-th pixel along each axis
    /// of the whole frame, which is never decoded whole. The same
    /// distribution sampled differently, so the shape is the frame's but the
    /// counts are not the counts [`Frame`](Self::Frame) would give, and a
    /// feature narrower than the stride can be missed.
    Overview { stride: u32 },
}

/// What binning a frame's channels takes besides their samples: the shared
/// track ([`shared_track`]) and each channel's units. What the read-ahead is
/// handed, so the worker can bin the frame it decodes.
#[derive(Clone, Debug)]
pub(crate) struct Binning {
    track: Window,
    units: Vec<SampleUnits>,
}

impl Binning {
    /// The binning of `stack`'s frames as it stands.
    pub(crate) fn of(stack: &Document) -> Self {
        Binning {
            track: shared_track(stack),
            units: stack.display.settings().iter().map(|s| s.units).collect(),
        }
    }

    /// Bin `channel`'s `data`, or `None` for a channel this has no units for.
    pub(crate) fn bin(&self, channel: usize, data: Samples<'_>) -> Option<Histogram> {
        let units = *self.units.get(channel)?;
        Some(bin(channel, data, self.track, units))
    }
}

/// The histograms of the frame on screen, kept with the stack
/// ([`Document::histograms`]) and made from the decodes that show it (see the
/// [module docs](self)).
///
/// Held until something replaces them, which is what a plot wants: after a
/// scrub, or a frame of playback, these are for a moment the last frame's —
/// until the next sync bins the new one — rather than nothing.
/// [`shown`](Self::shown) is what may be plotted for a stack as it now stands,
/// and [`source`](Self::source) what it was binned from.
#[derive(Debug, Default)]
pub struct FrameHistograms {
    held: Option<Held>,
    /// Moved by every change, so a sync that changed them can ask for the
    /// frame that draws them.
    version: u64,
    /// The playback clock when the interface thread last binned, for
    /// [`PLAYBACK_INTERVAL`].
    inline_at: Option<f64>,
}

/// Histograms binned under one key from one source.
#[derive(Debug)]
struct Held {
    key: HistogramKey,
    source: HistogramSource,
    /// One per channel binned, in channel order.
    hists: Vec<Histogram>,
    /// The channels binned under this key, or tried and failed: what is not to
    /// be decoded for it again.
    done: [bool; MAX_CHANNELS],
}

impl FrameHistograms {
    /// The histograms to plot for `stack`, one per channel binned, in channel
    /// order: those held, as long as they were binned under its channel
    /// layout. They may still be of a frame the stack has just left; a
    /// channel layout they were not binned under gives none, since their
    /// channel numbers would name other channels.
    pub fn shown(&self, stack: &Document) -> &[Histogram] {
        match &self.held {
            Some(held) if held.key.layout == stack.display.revision().layout => &held.hists,
            _ => &[],
        }
    }

    /// Where the held histograms were binned from, when any are held.
    pub fn source(&self) -> Option<HistogramSource> {
        self.held.as_ref().map(|held| held.source)
    }

    /// Whether the held histograms are of `stack` as it stands: the frame on
    /// screen, under its channel layout and slider tracks.
    pub(crate) fn describes(&self, stack: &Document) -> bool {
        self.held
            .as_ref()
            .is_some_and(|held| held.key.describes(stack))
    }
}

// The rest is driven by `crate::sync`, which needs a GPU backend.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
impl FrameHistograms {
    /// Moves whenever the held histograms change.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// Whether the interface thread may bin now: always, unless a movie is
    /// playing (`clock` is its clock) and it binned less than
    /// [`PLAYBACK_INTERVAL`] ago.
    pub(crate) fn may_bin_inline(&self, clock: Option<f64>) -> bool {
        match (clock, self.inline_at) {
            // A clock that went backwards is a new playback, not a short gap.
            (Some(now), Some(at)) => now - at >= PLAYBACK_INTERVAL || now < at,
            _ => true,
        }
    }

    /// Record that the interface thread binned at `clock`.
    pub(crate) fn binned_inline(&mut self, clock: Option<f64>) {
        self.inline_at = clock;
    }

    /// Add `hists`, binned under the key [`prepare`] last made these hold.
    /// A channel already binned under it keeps what it has.
    pub(crate) fn put(&mut self, hists: Vec<Histogram>) {
        let Some(held) = &mut self.held else {
            return;
        };
        let mut changed = false;
        for h in hists {
            match held.done.get_mut(h.channel) {
                Some(done) if !*done => *done = true,
                _ => continue,
            }
            let at = held.hists.partition_point(|x| x.channel < h.channel);
            held.hists.insert(at, h);
            changed = true;
        }
        if changed {
            self.version += 1;
        }
    }

    /// Record that the channels `tried` selects were decoded for this key,
    /// whether or not that gave a histogram: one that failed is not retried
    /// until the key moves, the way a failed frame is not decoded again for
    /// every repaint.
    fn tried(&mut self, tried: &[bool; MAX_CHANNELS]) {
        if let Some(held) = &mut self.held {
            for (done, &t) in held.done.iter_mut().zip(tried) {
                *done |= t;
            }
        }
    }

    /// Drop whatever is held.
    fn clear(&mut self) {
        if self.held.take().is_some() {
            self.version += 1;
        }
    }

    /// Whether what is held was binned from `source` for `stack` as it
    /// stands, so binning more of it adds to it rather than starting over.
    fn holds(&self, stack: &Document, source: HistogramSource) -> bool {
        self.held
            .as_ref()
            .is_some_and(|held| held.source == source && held.key.describes(stack))
    }
}

/// Make `stack`'s held histograms those of the stack as it stands, binned from
/// `source` — kept when they already are, emptied when they describe anything
/// else — and say which channels are still to be binned for it.
///
/// One flag per channel, indexed by channel; the form [`build_jobs`] takes.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn prepare(stack: &mut Document, source: HistogramSource) -> [bool; MAX_CHANNELS] {
    if !stack.services.histograms.holds(stack, source) {
        let key = HistogramKey::of(stack);
        let histograms = &mut stack.services.histograms;
        histograms.held = Some(Held {
            key,
            source,
            hists: Vec::new(),
            done: [false; MAX_CHANNELS],
        });
        histograms.version += 1;
    }
    let mut missing = [false; MAX_CHANNELS];
    if let Some(held) = &stack.services.histograms.held {
        let n = stack.display.channel_count().min(MAX_CHANNELS);
        for (m, done) in missing.iter_mut().zip(&held.done).take(n) {
            *m = !done;
        }
    }
    missing
}

/// Bin the planes just decoded to show the frame, for the channels `missing`
/// selects (see [`prepare`]); the rest are skipped. `planes` pairs each plane
/// with its display channel.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn bin_shown<'a>(
    stack: &Document,
    missing: &[bool; MAX_CHANNELS],
    planes: impl IntoIterator<Item = (usize, Samples<'a>)>,
) -> Vec<Histogram> {
    // Made only once there is something to bin: a frame whose channels are all
    // binned already is asked about on every repaint.
    let mut binning = None;
    planes
        .into_iter()
        .filter(|(channel, _)| missing.get(*channel).copied().unwrap_or(false))
        .filter_map(|(channel, data)| {
            binning
                .get_or_insert_with(|| Binning::of(stack))
                .bin(channel, data)
        })
        .collect()
}

/// Finish the held histograms of the frame on screen once a sync has shown
/// it, with what nothing else binned: decode the channels still missing — the
/// ones switched off, or every one when histograms are first asked for on a
/// frame already on screen — and only those. A frame past the texture limit
/// (as `residency`, the view's, found it) is never decoded whole: its
/// histograms are binned from the overview that view holds, when it is of
/// this frame, and otherwise there are none.
///
/// `start` is whether this may start the histograms of a frame that has
/// none, decoding every channel for them. Not while a movie plays in the 2D
/// view: a played frame's histograms are started from the planes decoded to
/// show it, by this thread or the read-ahead's, at the throttle's pace, and a
/// repaint that shows no new frame would otherwise decode the one on screen a
/// second time. Then this only adds the channels switched off to histograms
/// already started for this frame.
///
/// Returns how many channels it decoded. None after a channel toggle, since
/// every channel is binned under a key a toggle does not move.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn complete(stack: &mut Document, residency: &Residency, start: bool) -> usize {
    if residency.over_texture_limit {
        from_overview(stack, residency.overview.as_ref());
        return 0;
    }
    if !start
        && !stack
            .services
            .histograms
            .holds(stack, HistogramSource::Frame)
    {
        return 0;
    }
    let missing = prepare(stack, HistogramSource::Frame);
    if !missing.contains(&true) {
        return 0;
    }
    let jobs = build_jobs(stack, stack.frame(), &missing);
    // Borrowed, as for the screen: an uncompressed plane is binned straight
    // out of the mapping, with no copy.
    let binned = match decode_jobs_borrowed(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    ) {
        Ok(planes) => bin_shown(
            stack,
            &missing,
            jobs.iter()
                .zip(&planes)
                .map(|(job, data)| (job.channel, data.view())),
        ),
        // A frame that does not decode has nothing to show; the image is
        // reporting it already.
        Err(_) => Vec::new(),
    };
    stack.services.histograms.put(binned);
    stack.services.histograms.tried(&missing);
    jobs.len()
}

/// [`complete`] for a frame past the texture limit: bin `overview`, which is
/// the whole frame at the overview's stride, when it is of this frame and
/// channel layout. Its channels are the ones that were on when it was built,
/// so a channel switched off then has no histogram until the next overview.
///
/// Held, or binned as it was built ([`bin_whole_view`]), what is binned from
/// a view of the whole frame stays while it describes the frame: zoomed in on
/// it, there is no such view, and the histograms are still the frame's.
fn from_overview(stack: &mut Document, overview: Option<&Overview>) {
    let layout = stack.display.revision().layout;
    let of_this_frame =
        overview.filter(|o| o.built.frame_index == stack.frame() && o.revision.layout == layout);
    let Some(overview) = of_this_frame else {
        // Nothing of this frame to bin. What is held is kept only while it
        // is still this frame's: another frame's would be the wrong picture,
        // with nothing to say so.
        if !stack.services.histograms.describes(stack) {
            stack.services.histograms.clear();
        }
        return;
    };
    bin_whole_view(stack, &overview.built);
}

/// Bin `view`, a window spanning the whole of the frame on screen at its
/// stride, into the held histograms ([`HistogramSource::Overview`]): what a
/// frame past the texture limit is histogrammed from, as it is built — a view
/// too large to keep as the overview is binned only then — and as the
/// overview later.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn bin_whole_view(stack: &mut Document, view: &crate::window::Built) {
    let source = HistogramSource::Overview {
        stride: view.roi.stride.max(1),
    };
    let missing = prepare(stack, source);
    let binned = bin_shown(
        stack,
        &missing,
        view.channels
            .iter()
            .copied()
            .zip(view.planes.iter().map(Decoded::view)),
    );
    stack.services.histograms.put(binned);
}

/// Histogram every display channel of the stack's current frame, decoding
/// every one of them: the whole-frame reference the held histograms
/// ([`FrameHistograms`]) are made to equal, and what a caller with no renderer
/// uses. A frontend showing the frame should read [`Document::histograms`]
/// instead, which are binned from the decodes that show it.
///
/// Returns an empty vec when nothing is loaded or the frame fails to decode —
/// a histogram is a read-only view, so a decode error here is not worth
/// surfacing separately; the image itself will already be reporting it.
/// What the result depends on, and so when to call this again, is the frame,
/// the channel layout and each channel's slider track: not which channels are
/// on, nor their contrast windows.
pub fn frame_histograms(stack: &Document) -> Vec<Histogram> {
    if stack.display.settings().is_empty() {
        return Vec::new();
    }
    // Every channel, including any the user has switched off. A frontend that
    // only plots the enabled ones can then honour a checkbox by redrawing
    // instead of decoding the frame again, which is what makes toggling one
    // feel instant — and is why `HistogramKey` leaves the enabled set out.
    let every = [true; MAX_CHANNELS];
    let jobs = build_jobs(stack, stack.frame(), &every);
    let Ok(decoded) = decode_jobs_borrowed(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    ) else {
        return Vec::new();
    };
    let binning = Binning::of(stack);
    jobs.iter()
        .zip(&decoded)
        .filter_map(|(job, data)| binning.bin(job.channel, data.view()))
        .collect()
}

/// The axis every channel is binned onto, in file units: the union of their
/// slider tracks, so no channel's data falls off either end.
///
/// Public because a frontend drawing the plot needs the same number to label an
/// axis or place a marker, and recomputing it there would be one more place for
/// the two to disagree.
pub fn shared_track(stack: &Document) -> Window {
    let settings = stack.display.settings();
    let track = settings
        .iter()
        .map(|s| s.bounds)
        .reduce(Window::union)
        .unwrap_or(Window::new(f32::INFINITY, f32::NEG_INFINITY));
    // No channels, or bounds that are somehow not finite: a degenerate track
    // would divide by zero in `bin`, so hand back something usable instead —
    // the whole of what the first channel's texture can show.
    if !(track.lo.is_finite() && track.hi.is_finite() && track.hi > track.lo) {
        return settings
            .first()
            .map_or(Window::new(0.0, 1.0), |s| s.units.full_range());
    }
    track
}

/// Bin one channel's samples, which are in `units`' texture units, over the
/// file-units `track`.
fn bin(channel: usize, data: Samples<'_>, track: Window, units: SampleUnits) -> Histogram {
    let mut bins = vec![0u32; BINS];
    // The samples are binned as decoded, against the track put into the same
    // units once, rather than each sample being converted: the conversion is
    // affine and increasing, so every sample lands in the bin it would have,
    // and a histogram of a 4K frame costs no extra pass.
    let (lo, hi) = units.to_texture(track);
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
        Samples::U8(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                add(s as f32);
            }
        }
        Samples::U16(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                add(s as f32);
            }
        }
        // NaNs and infinities belong to no bin.
        Samples::F32(v) => {
            for &s in v.iter().step_by(stride(v.len())) {
                if s.is_finite() {
                    add(s);
                }
            }
        }
    }

    let peak = bins.iter().copied().max().unwrap_or(0);
    Histogram {
        channel,
        bins,
        lo: track.lo,
        hi: track.hi,
        peak,
        counted,
    }
}

/// Step between sampled pixels so a frame contributes at most
/// [`SAMPLE_BUDGET`] points. A prime-ish stride would be better on a
/// power-of-two-wide image, but unlike the min/max scan a histogram is a
/// *distribution* — hitting the same column of every row still samples the same
/// population, so a plain quotient is fine here.
fn stride(len: usize) -> usize {
    (len / SAMPLE_BUDGET).max(1)
}

/// Fraction of one channel of samples that the auto window may clip off each
/// end, 0.35% — the same saturation ImageJ applies for its own Auto button, so
/// a stack looks here the way it looks there.
///
/// Some clipping is the entire point. The full data range is what the slider
/// already spans, so a window fitted to the exact min and max would be the
/// no-op the Reset button gives you. Real frames carry a few hot pixels and a
/// dark-current floor, and letting those set the ends leaves everything of
/// interest squeezed into the middle. Throwing away a third of a percent at
/// each end costs a handful of outliers and buys the rest of the histogram.
pub const AUTO_SATURATION: f64 = 0.0035;

/// A contrast window fitted to where this channel of data actually is: the
/// narrowest span that keeps all but [`AUTO_SATURATION`] of the samples at each
/// end, in the same units as [`Histogram::lo`]/[`Histogram::hi`] and therefore
/// as [`crate::ChannelSettings::window`] — the file's own.
///
/// `None` when nothing was counted, which is the one case with no defensible
/// answer — an empty frame, or a decode that failed. Callers should leave the
/// window alone rather than substituting something.
///
/// The result is never empty: a channel of one single value still yields a
/// one-bin-wide window rather than a zero-width one that would divide by zero
/// downstream and blank the image.
pub fn auto_window(hist: &Histogram) -> Option<Window> {
    if hist.counted == 0 {
        return None;
    }
    // Ceil, so even a tiny frame clips at least one sample and Auto does
    // something visible; `budget` counts samples that may be discarded.
    let budget = (hist.counted as f64 * AUTO_SATURATION).ceil() as u64;

    // Walk in from each end until more than `budget` samples have been passed.
    // The bin that breaks the budget is kept: it holds the first sample that is
    // not spare, so cutting it would clip more than asked for.
    let mut acc = 0u64;
    let mut lo_bin = 0usize;
    for (i, &c) in hist.bins.iter().enumerate() {
        acc += c as u64;
        if acc > budget {
            lo_bin = i;
            break;
        }
    }
    acc = 0;
    let mut hi_bin = BINS - 1;
    for (i, &c) in hist.bins.iter().enumerate().rev() {
        acc += c as u64;
        if acc > budget {
            hi_bin = i;
            break;
        }
    }
    // Both ends chewing through the same lone spike can cross the walks over
    // each other. Collapse to the single bin they agree on instead of returning
    // a backwards range.
    if hi_bin < lo_bin {
        hi_bin = lo_bin;
    }

    let width = (hist.hi - hist.lo) / BINS as f32;
    let lo = hist.lo + lo_bin as f32 * width;
    // The upper edge of `hi_bin`, not its lower one: the bin covers values up
    // to there, and a window ending at its start would clip the very samples
    // that earned it a place.
    let hi = hist.lo + (hi_bin + 1) as f32 * width;
    Some(Window::new(lo, hi.max(lo + width)))
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

#[cfg(test)]
#[path = "histogram_tests.rs"]
mod tests;

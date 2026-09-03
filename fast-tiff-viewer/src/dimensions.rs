//! Stack-shape interpretation: applying resolved (or manually overridden)
//! channel/Z/time roles to a loaded stack, RGB plane setup, and the derived
//! status line. Split from `app.rs`.

use crate::channels::build_channel_settings_reporting;
use crate::display::Dims;
use crate::stack::{ChannelSettings, Stack};
use scivis_render::{ChannelKind, MAX_CHANNELS};

/// The number of IFDs `dims` addresses: one past the highest index
/// [`crate::prefetch::build_jobs`] can produce for it.
///
/// Mirrors that function's arithmetic deliberately — the whole purpose of
/// knowing this is to stop it asking for a plane the file does not have, so a
/// second, independent formula here would be a second thing to get wrong.
/// Chunky RGB addresses one IFD per frame (its channels are sample planes of
/// that IFD); everything else gives each channel its own.
pub fn planes_addressed(dims: Dims, rgb: bool) -> usize {
    let (c, z, f) = (dims.channels.max(1), dims.slices.max(1), dims.frames.max(1));
    if rgb {
        (f - 1).saturating_mul(z).saturating_add(1)
    } else {
        (f - 1)
            .saturating_mul(z)
            .saturating_mul(c)
            .saturating_add(c)
    }
}

/// Cut `dims` down until every plane it addresses is one the file actually has.
///
/// Returns `(declared, available)` when it had to cut, so the caller can say so;
/// `None` when the metadata and the file already agreed.
///
/// A file can describe more planes than it contains, and the metadata is not
/// necessarily wrong to do so — a multi-file OME set gives every file the *whole*
/// dataset's dimensions and points at its siblings, so one file of a two-channel
/// pair declares twice the planes it holds. Trusting the declaration means
/// addressing IFDs past the end of the chain, which is a decode error on every
/// frame the scrubber reaches.
///
/// Time is cut first, because it is the slowest-varying axis in the order these
/// planes are addressed in and a file that is short is short at the end. Only if
/// a single frame still does not fit — fewer IFDs in the whole file than there
/// are channels — are the channels cut too, and Z never needs cutting because at
/// one frame it drops out of the arithmetic entirely.
pub fn clamp_to_available(dims: &mut Dims, rgb: bool, available: usize) -> Option<(usize, usize)> {
    let declared = planes_addressed(*dims, rgb);
    if available == 0 || declared <= available {
        return None;
    }
    let (c, z) = (dims.channels.max(1), dims.slices.max(1));
    if rgb {
        // (f - 1) * z + 1 <= available
        dims.frames = (available - 1) / z + 1;
    } else if available < c {
        // Not even one frame's worth of channels. Keep a frame, drop channels.
        dims.frames = 1;
        dims.channels = available;
    } else {
        // (f - 1) * z * c + c <= available
        dims.frames = (available - c) / (z * c) + 1;
    }
    dims.frames = dims.frames.max(1);
    dims.channels = dims.channels.max(1);
    // Belt and braces: the formulas above are exact, but they are arithmetic on
    // numbers a file chose, and being wrong here puts the decode error back.
    if planes_addressed(*dims, rgb) > available {
        dims.frames = 1;
        dims.slices = 1;
        dims.channels = dims.channels.min(available).max(1);
    }
    Some((declared, available))
}

/// The status note shown at the top of the window, derived from the
/// stack's current (resolved) dimensions. Shared between the initial load
/// and the manual dimension-order override so the two can't drift out of
/// sync with each other.
pub fn compute_status(
    dims: Dims,
    triple_axis_warning: bool,
    plane_mismatch: Option<(usize, usize)>,
) -> Option<String> {
    // Ahead of the others: those describe how the file has been *interpreted*,
    // this one says the file disagrees with itself. Whatever is on screen is
    // built from an arrangement the metadata does not actually back up, and the
    // reader should know that before anything else.
    if let Some((declared, available)) = plane_mismatch {
        Some(format!(
            "Warning: this file's metadata describes {declared} image plane(s) but the file              contains {available}. Showing {} channel(s) × {} Z-slice(s) × {} frame(s), which is              what fits — the rest may live in a companion file (a multi-file OME set gives every              file the whole dataset's dimensions), or the file may be truncated.",
            dims.channels, dims.slices, dims.frames
        ))
    } else if triple_axis_warning {
        Some(format!(
            "Warning: this file has channels, Z-slices, and time frames all present at once \
             ({} channel(s) × {} Z-slice(s) × {} frame(s)). Z isn't shown as a separate axis here \
             — only the first Z-slice is used; to see the whole picture, use 3D view.",
            dims.channels, dims.slices, dims.frames
        ))
    } else if dims.channels > MAX_CHANNELS {
        Some(format!(
            "Note: stack has {} channels; showing the first {MAX_CHANNELS}.",
            dims.channels
        ))
    } else {
        None
    }
}

/// Applies a (possibly newly resolved) channel/slice/frame interpretation
/// to a stack: updates the metadata, rebuilds channel_display +
/// channel_settings to match the new channel count, and resets the scrub
/// position. The one place that does this, so the manual channels/frames
/// swap can't drift out of sync with `open_file` the way `self.status`
/// previously did.
pub fn apply_resolved_dimensions(loaded: &mut Stack, resolved: fast_tiff_lib::ResolvedDimensions) {
    apply_resolved_dimensions_reporting(loaded, resolved, &mut |_, _| {});
}

/// [`apply_resolved_dimensions`], reporting the per-channel contrast scans as
/// they finish — the slow part of opening a file, and the only part whose
/// length is known ahead of time.
pub fn apply_resolved_dimensions_reporting(
    loaded: &mut Stack,
    resolved: fast_tiff_lib::ResolvedDimensions,
    on_channel: &mut dyn FnMut(usize, usize),
) {
    // Note what is *not* here: `loaded.tiff.meta` is left exactly as parsed.
    // The interpretation is ours, not the file's, so it lives in `display` —
    // which is what keeps "what did the file actually say?" answerable after
    // the user reassigns the axes (see `crate::display`).
    let mut dims = Dims {
        channels: resolved.channels,
        slices: resolved.slices,
        frames: resolved.frames,
    };
    // Chunky RGB and CMYK put their channels in one IFD's sample planes, which
    // changes how many IFDs a given shape addresses. Asked of the frame rather
    // than of `display.rgb`, because on a fresh load `setup_rgb` has not run
    // yet — it runs after this.
    let shares_an_ifd = loaded
        .tiff
        .frames
        .first()
        .is_some_and(|f| f.is_rgb() || f.is_cmyk());
    loaded.display.plane_mismatch =
        clamp_to_available(&mut dims, shares_an_ifd, loaded.tiff.frames.len());
    let resolved = fast_tiff_lib::ResolvedDimensions {
        channels: dims.channels,
        slices: dims.slices,
        frames: dims.frames,
        ..resolved
    };
    loaded.display.dims = dims;
    loaded.display.triple_axis_warning = resolved.triple_axis_warning;
    loaded.display.mode = loaded.tiff.meta.mode;
    let shown = crate::display::Display::shown_channels(resolved.channels);
    loaded.display.reseed_luts(&loaded.tiff.meta, shown);
    loaded.display.settings =
        build_channel_settings_reporting(&loaded.tiff, resolved.channels, on_channel);
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
    // Which IFD each display channel reads just changed, and the decoded-band
    // cache is keyed by *display channel*, not by IFD — so every band it holds
    // now describes the wrong plane. Keeping them would splice pixels from the
    // old interpretation into the next window, which is worse than the decode
    // they save: a wrong picture rather than a slow one. Only ever non-empty
    // for a frame too large to upload whole, so this costs nothing for an
    // ordinary image.
    loaded.bands.clear();
    // Same argument for the retained overview: its planes were decoded through
    // the old channel-to-IFD mapping. It is checked against `prefetch_gen` on
    // every read as well, so this is belt and braces — but it returns the
    // memory now rather than at the next sync.
    loaded.overview = None;
}

/// Which sample planes of an `spp`-sample RGB frame get a display channel, and
/// which of those start enabled — one entry per channel, `true` = on. The whole
/// policy `setup_rgb` applies, split out so it's testable without a
/// file-and-GPU-backed `Stack`. See `setup_rgb` for the reasoning.
///
/// Beyond `MAX_CHANNELS` the shader has no slot to composite into, so further
/// samples are dropped — no real file has 7+ samples/pixel.
pub fn rgb_channel_plan(spp: usize) -> Vec<bool> {
    (0..spp.min(MAX_CHANNELS)).map(|c| c < 3).collect()
}

/// Reconfigures a freshly-loaded RGB stack (chunky or planar): every sample
/// plane becomes a display channel with an identity full-range window (so true
/// colors show without any contrast tweaking). Additively blending the red,
/// green and blue ramps in the composite shader reconstructs the original RGB
/// pixel. Frame navigation still walks IFDs (one full-color image per IFD) —
/// see `Stack::rgb`.
///
/// Samples past the third (TIFF ExtraSamples — alpha, or anything else a writer
/// packed in) get channels too, but **start disabled**. They're real data the
/// user may want: `tifffile` writes any `(4, H, W)` array as RGB + one extra
/// sample, so for scientific stacks the fourth plane is a measurement, not
/// transparency. Compositing it on by default would wreck genuine RGBA images
/// though — an opaque alpha plane is a constant full-intensity channel, which
/// the additive shader would blend in as a solid color wash over the picture.
/// Off-by-default is the only setting that's harmless for both: the channel row
/// is visible, and one click shows it.
pub fn setup_rgb(loaded: &mut Stack) {
    let spp = loaded
        .tiff
        .frames
        .first()
        .map(|f| f.samples_per_pixel as usize)
        .unwrap_or(3);
    let plan = rgb_channel_plan(spp);
    let planes = plan.len();
    loaded.display.rgb = true;
    loaded.display.mode = fast_tiff_lib::DisplayMode::Color;
    // 0 = red, 1 = green, 2 = blue, then the composite palette for any extras.
    loaded.display.luts = (0..planes)
        .map(fast_tiff_lib::default_composite_lut)
        .collect();
    // Unsigned 8-bit RGB deinterleaves into raw u8 planes (`read_plane_u8`) and
    // rides the R8Uint path — half the texture memory + upload of widening each
    // plane to u16. Deeper or signed RGB still widens to u16 via `read_plane_u16`.
    // The window stays in 0..65535 either way; `sync_gpu` rescales it to 0..255
    // for an 8-bit (Int8) channel.
    let kind = if loaded.tiff.frames.first().is_some_and(|f| {
        f.bits_per_sample == 8 && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt
    }) {
        ChannelKind::Int8
    } else {
        ChannelKind::Int16
    };
    loaded.display.settings = plan
        .iter()
        .map(|&enabled| ChannelSettings {
            min: 0.0,
            max: 65535.0,
            enabled,
            bounds: (0.0, 65535.0),
            initial: (0.0, 65535.0),
            kind,
        })
        .collect();
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Reconfigures a freshly-loaded CMYK (Separated) stack: its four ink planes
/// become **three** red/green/blue display channels, converted on decode by
/// [`fast_tiff_lib::read_planes_rgb_u8`] and friends.
///
/// Why convert rather than show four ink channels: the compositing shader is
/// *additive* (`color += lut[value]` per channel), which is exactly right for
/// RGB and exactly wrong for CMYK, where inks *subtract* from white. Four ink
/// ramps blended additively would produce nonsense. Converting to RGB first
/// means the existing, proven RGB display path renders a Separated file
/// correctly with no shader involvement at all.
///
/// The trade is that the four ink measurements collapse to three display
/// values and the K plate is no longer separately visible. The raw ink planes
/// remain available from the library (`read_planes_u8`/`read_planes_u16` still
/// return one plane per sample), so a future "show separations" mode has
/// everything it needs.
pub fn setup_cmyk(loaded: &mut Stack) {
    // Both flags: `rgb` drives the plane-of-one-IFD decode addressing shared
    // with real RGB, `cmyk` records that these channels are derived.
    loaded.display.rgb = true;
    loaded.display.cmyk = true;
    loaded.display.mode = fast_tiff_lib::DisplayMode::Color;
    loaded.display.luts = (0..3).map(fast_tiff_lib::default_composite_lut).collect(); // R, G, B
                                                                                      // The conversion outputs in the source's own width: 8-bit inks give 8-bit
                                                                                      // components (the zero-widening R8Uint upload), 16-bit gives 16.
    let kind = if loaded
        .tiff
        .frames
        .first()
        .is_some_and(|f| f.bits_per_sample == 8)
    {
        ChannelKind::Int8
    } else {
        ChannelKind::Int16
    };
    loaded.display.settings = (0..3)
        .map(|_| ChannelSettings {
            min: 0.0,
            max: 65535.0,
            enabled: true,
            bounds: (0.0, 65535.0),
            initial: (0.0, 65535.0),
            kind,
        })
        .collect();
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Applies a manual dimension-order change from the dropdown: reassigns the
/// channels / Z-slices / time-frames roles to the given counts (the product
/// stays the stack's plane count — the selector only offers permutations).
/// For stacks without a real Z axis the selector passes `slices` through
/// unchanged, so it stays a plain channels/time swap. The triple-axis
/// warning flag is carried over — it describes the file, not the current
/// role assignment.
pub fn apply_dimension_override(loaded: &mut Stack, channels: usize, slices: usize, frames: usize) {
    let resolved = fast_tiff_lib::ResolvedDimensions {
        channels,
        slices,
        frames,
        triple_axis_warning: loaded.display.triple_axis_warning,
    };
    apply_resolved_dimensions(loaded, resolved);
    // The channel->IFD mapping just changed, so invalidate any in-flight prefetch
    // decoded under the old mapping.
    loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
}

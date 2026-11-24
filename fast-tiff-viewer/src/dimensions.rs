//! Stack-shape interpretation: applying resolved (or manually overridden)
//! channel/Z/time roles to a loaded stack, RGB plane setup, and the derived
//! note about its shape. Split from `app.rs`.

use crate::channels::build_channel_settings_reporting;
use crate::display::{ChannelLayout, Dims};
use crate::document::{ChannelSettings, Document};
use crate::notice::ShapeNote;
use crate::units::SampleUnits;
use scivis_render::{ChannelKind, MAX_CHANNELS};

/// Which IFD, and which sample plane inside it, holds channel `c` of slice `z`
/// at timepoint `t`.
///
/// The layout is ImageJ's default `xyczt` plane order: channel fastest, then Z,
/// then time. Chunky RGB is the exception — its channels are sample *planes* of
/// one IFD, so the channel index selects the plane and does not advance the IFD.
///
/// No bounds checking: `dims` describes what the metadata claims, and whether
/// the file actually has that plane is [`planes_addressed`]'s question (for the
/// 2D axis) or the decoder's.
pub(crate) fn plane_index(dims: Dims, rgb: bool, c: usize, z: usize, t: usize) -> (usize, usize) {
    let (channels, slices) = (dims.channels.max(1), dims.slices.max(1));
    if rgb {
        (t * slices + z, c)
    } else {
        (t * slices * channels + z * channels + c, 0)
    }
}

/// Minimum IFD count for every timepoint to have one complete slice.
///
/// Mirrors that function's arithmetic deliberately — the whole purpose of
/// knowing this is to stop it asking for a plane the file does not have, so a
/// second, independent formula here would be a second thing to get wrong.
/// Chunky RGB addresses one IFD per frame (its channels are sample planes of
/// that IFD); everything else gives each channel its own.
pub(crate) fn planes_addressed(dims: Dims, rgb: bool) -> usize {
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
pub(crate) fn clamp_to_available(
    dims: &mut Dims,
    rgb: bool,
    available: usize,
) -> Option<(usize, usize)> {
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

/// The note about the stack's shape worth showing the user, derived from its
/// current (resolved) dimensions; `None` when there is nothing to say. Shared
/// between the initial load and the manual dimension-order override so the
/// two can't drift out of sync with each other. Its words are its
/// [`Display`](std::fmt::Display).
pub(crate) fn shape_note(dims: Dims, plane_mismatch: Option<(usize, usize)>) -> Option<ShapeNote> {
    // Ahead of the others: those describe how the file has been *interpreted*,
    // this one says the file disagrees with itself. Whatever is on screen is
    // built from an arrangement the metadata does not actually back up, and the
    // reader should know that before anything else.
    if let Some((declared, available)) = plane_mismatch {
        Some(ShapeNote::PlaneMismatch {
            declared,
            available,
            dims,
        })
    } else if dims.channels > MAX_CHANNELS {
        Some(ShapeNote::TooManyChannels(dims.channels))
    } else {
        None
    }
}

/// Applies a (possibly newly resolved) channel/slice/frame interpretation
/// to a stack: updates the metadata, rebuilds channel_display +
/// channel_settings to match the new channel count, and resets the scrub
/// position. The one place that does this, so the manual channels/frames
/// swap can't drift out of sync with `open_file` the way the status note
/// previously did.
pub(crate) fn apply_resolved_dimensions(
    loaded: &mut Document,
    resolved: fast_tiff_lib::ResolvedDimensions,
) {
    apply_resolved_dimensions_reporting(loaded, resolved, &mut |_, _| {});
}

/// [`apply_resolved_dimensions`], reporting the per-channel contrast scans as
/// they finish — the slow part of opening a file, and the only part whose
/// length is known ahead of time.
pub(crate) fn apply_resolved_dimensions_reporting(
    loaded: &mut Document,
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
    // than of `display.layout()`, because on a fresh load `setup_rgb` has not run
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
    // Both move the display's layout revision, which is what everything
    // decoded under the old interpretation — the frame on the GPU, a frame
    // read ahead, a window being built, the retained overview, the decoded
    // bands, the volume — is checked against before it is used again: the
    // viewer's residency at the top of its next sync, each worker and cache
    // as it is asked. Nothing here has to know about any of them.
    loaded.display.set_dims(dims);
    loaded.display.mode = loaded.tiff.meta.mode;
    let shown = crate::display::Display::shown_channels(resolved.channels);
    loaded.display.reseed_luts(&loaded.tiff.meta, shown);
    loaded
        .display
        .set_settings(build_channel_settings_reporting(
            &loaded.tiff,
            resolved.channels,
            on_channel,
        ));
    loaded.set_frame(0);
    loaded.set_slice(0);
}

/// Which sample planes of an `spp`-sample RGB frame get a display channel, and
/// which of those start enabled — one entry per channel, `true` = on. The whole
/// policy `setup_rgb` applies, split out so it's testable without a
/// file-and-GPU-backed `Document`. See `setup_rgb` for the reasoning.
///
/// Beyond `MAX_CHANNELS` the shader has no slot to composite into, so further
/// samples are dropped — no real file has 7+ samples/pixel.
pub(crate) fn rgb_channel_plan(spp: usize) -> Vec<bool> {
    (0..spp.min(MAX_CHANNELS)).map(|c| c < 3).collect()
}

/// Reconfigures a freshly-loaded RGB stack (chunky or planar): every sample
/// plane becomes a display channel with an identity full-range window (so true
/// colors show without any contrast tweaking). Additively blending the red,
/// green and blue ramps in the composite shader reconstructs the original RGB
/// pixel. Frame navigation still walks IFDs (one full-color image per IFD) —
/// see `Display::rgb`.
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
pub(crate) fn setup_rgb(loaded: &mut Document) {
    let spp = loaded
        .tiff
        .frames
        .first()
        .map(|f| f.samples_per_pixel as usize)
        .unwrap_or(3);
    let plan = rgb_channel_plan(spp);
    let planes = plan.len();
    loaded.display.set_layout(ChannelLayout::Rgb);
    loaded.display.mode = fast_tiff_lib::DisplayMode::Color;
    // 0 = red, 1 = green, 2 = blue, then the composite palette for any extras.
    loaded.display.set_luts(
        (0..planes)
            .map(fast_tiff_lib::default_composite_lut)
            .collect(),
    );
    // Unsigned 8-bit RGB is read as its stored bytes, one plane per channel
    // (`units::Stored`, through `prefetch::decode_jobs`), and rides the R8Uint
    // path — half the texture memory + upload of widening each plane to u16.
    // Deeper or signed RGB is read on the 16-bit display scale
    // (`units::Display`). Either way each component opens on the whole of its
    // type's range.
    let first = loaded.tiff.frames.first();
    let kind = if first.is_some_and(|f| {
        f.bits_per_sample == 8 && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt
    }) {
        ChannelKind::Int8
    } else {
        ChannelKind::Int16
    };
    let units = SampleUnits::of(kind, first, None);
    loaded.display.set_settings(
        plan.iter()
            .map(|&enabled| ChannelSettings {
                enabled,
                ..ChannelSettings::new(units.full_range(), units)
            })
            .collect(),
    );
    loaded.set_frame(0);
    loaded.set_slice(0);
}

/// Reconfigures a freshly-loaded CMYK (Separated) stack: its four ink planes
/// become **three** red/green/blue display channels, converted on decode: the
/// display path reads such a frame in [`fast_tiff_lib::units::CmykRgb8`] or
/// [`CmykRgb16`](fast_tiff_lib::units::CmykRgb16) (see
/// [`decode_jobs`](crate::prefetch::decode_jobs)).
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
/// remain available from the library (a [`Frame`](fast_tiff_lib::Frame) read
/// in `units::Stored` or `units::Display` still gives one plane per ink), so a
/// future "show separations" mode has everything it needs.
///
/// Like [`setup_rgb`] and [`apply_resolved_dimensions`], this edits only the
/// display model (and the frame index), so it is as safe to call on a stack
/// already on screen as on one just opened: the display's revisions moving is
/// what has a viewer drop everything it holds of the old mapping.
pub(crate) fn setup_cmyk(loaded: &mut Document) {
    // Addressed as RGB is (the channels are planes of one IFD), and told
    // apart from it so the channels are known to be derived.
    loaded.display.set_layout(ChannelLayout::Cmyk);
    loaded.display.mode = fast_tiff_lib::DisplayMode::Color;
    // R, G, B.
    loaded
        .display
        .set_luts((0..3).map(fast_tiff_lib::default_composite_lut).collect());
    // The conversion outputs in the source's own width: 8-bit inks give 8-bit
    // components (the zero-widening R8Uint upload), 16-bit gives 16. The
    // components are the conversion's output rather than the file's inks, so
    // their units are that output's: raw bytes on the 8-bit path, the 16-bit
    // scale itself otherwise. Each opens on the whole of it.
    let units = if loaded
        .tiff
        .frames
        .first()
        .is_some_and(|f| f.bits_per_sample == 8)
    {
        SampleUnits::RAW8
    } else {
        SampleUnits::display(fast_tiff_lib::DisplayMap::Identity)
    };
    loaded.display.set_settings(
        (0..3)
            .map(|_| ChannelSettings::new(units.full_range(), units))
            .collect(),
    );
    loaded.set_frame(0);
    loaded.set_slice(0);
}

/// Applies a manual dimension-order change from the dropdown: reassigns the
/// channels / Z-slices / time-frames roles to the given counts (the product
/// stays the one in effect — [`Viewer::set_dimension_order`](crate::Viewer::set_dimension_order)
/// ignores any other).
/// For stacks without a real Z axis the selector passes `slices` through
/// unchanged, so it stays a plain channels/time swap.
pub(crate) fn apply_dimension_override(
    loaded: &mut Document,
    channels: usize,
    slices: usize,
    frames: usize,
) {
    let resolved = fast_tiff_lib::ResolvedDimensions {
        channels,
        slices,
        frames,
        triple_axis_warning: channels > 1 && slices > 1 && frames > 1,
    };
    // Moves the layout revision, so a frame read ahead or a window built under
    // the old channel-to-IFD mapping is refused when it arrives.
    apply_resolved_dimensions(loaded, resolved);
}

#[cfg(test)]
#[path = "plane_addressing_tests.rs"]
mod addressing_tests;
#[cfg(test)]
#[path = "plane_clamp_tests.rs"]
mod clamp_tests;
#[cfg(test)]
#[path = "cmyk_display_tests.rs"]
mod cmyk_tests;
#[cfg(test)]
#[path = "dimensions_tests.rs"]
mod tests;

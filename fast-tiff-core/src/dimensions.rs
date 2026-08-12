//! Stack-shape interpretation: applying resolved (or manually overridden)
//! channel/Z/time roles to a loaded stack, RGB plane setup, and the derived
//! status line. Split from `app.rs`.

use super::*;

use super::channels::{build_channel_settings, resize_channel_display};
use crate::render::{ChannelKind, MAX_CHANNELS};

/// The status note shown at the top of the window, derived from the
/// stack's current (resolved) dimensions. Shared between the initial load
/// and the manual dimension-order override so the two can't drift out of
/// sync with each other.
pub(super) fn compute_status(meta: &fast_tiff_lib::StackMeta, triple_axis_warning: bool) -> Option<String> {
    if triple_axis_warning {
        Some(format!(
            "Warning: this file has channels, Z-slices, and time frames all present at once \
             ({} channel(s) × {} Z-slice(s) × {} frame(s)). Z isn't shown as a separate axis here \
             — only the first Z-slice is used; scrubbing covers channels × time only.",
            meta.channels, meta.slices, meta.frames
        ))
    } else if meta.channels > MAX_CHANNELS {
        Some(format!(
            "Note: stack has {} channels; showing the first {MAX_CHANNELS}.",
            meta.channels
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
pub(super) fn apply_resolved_dimensions(loaded: &mut LoadedStack, resolved: fast_tiff_lib::ResolvedDimensions) {
    loaded.tiff.meta.channels = resolved.channels;
    loaded.tiff.meta.slices = resolved.slices;
    loaded.tiff.meta.frames = resolved.frames;
    loaded.triple_axis_warning = resolved.triple_axis_warning;
    resize_channel_display(&mut loaded.tiff.meta, resolved.channels);
    loaded.channel_settings = build_channel_settings(&loaded.tiff);
    loaded.frame_index = 0;
    loaded.last_uploaded = None;
    loaded.luts_uploaded = false;
}

/// Which sample planes of an `spp`-sample RGB frame get a display channel, and
/// which of those start enabled — one entry per channel, `true` = on. The whole
/// policy `setup_rgb` applies, split out so it's testable without a
/// file-and-GPU-backed `LoadedStack`. See `setup_rgb` for the reasoning.
///
/// Beyond `MAX_CHANNELS` the shader has no slot to composite into, so further
/// samples are dropped — no real file has 7+ samples/pixel.
pub(super) fn rgb_channel_plan(spp: usize) -> Vec<bool> {
    (0..spp.min(MAX_CHANNELS)).map(|c| c < 3).collect()
}

/// Reconfigures a freshly-loaded RGB stack (chunky or planar): every sample
/// plane becomes a display channel with an identity full-range window (so true
/// colors show without any contrast tweaking). Additively blending the red,
/// green and blue ramps in the composite shader reconstructs the original RGB
/// pixel. Frame navigation still walks IFDs (one full-color image per IFD) —
/// see `LoadedStack::rgb`.
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
pub(super) fn setup_rgb(loaded: &mut LoadedStack) {
    let spp = loaded.tiff.frames.first().map(|f| f.samples_per_pixel as usize).unwrap_or(3);
    let plan = rgb_channel_plan(spp);
    let planes = plan.len();
    loaded.rgb = true;
    loaded.tiff.meta.mode = fast_tiff_lib::DisplayMode::Color;
    loaded.tiff.meta.channel_display = (0..planes)
        .map(|c| fast_tiff_lib::ChannelDisplay {
            lut: fast_tiff_lib::default_composite_lut(c), // 0 = red, 1 = green, 2 = blue
            range: None,
        })
        .collect();
    // Unsigned 8-bit RGB deinterleaves into raw u8 planes (`read_plane_u8`) and
    // rides the R8Uint path — half the texture memory + upload of widening each
    // plane to u16. Deeper or signed RGB still widens to u16 via `read_plane_u16`.
    // The window stays in 0..65535 either way; `sync_gpu` rescales it to 0..255
    // for an 8-bit (Int8) channel.
    let kind = if loaded
        .tiff
        .frames
        .first()
        .is_some_and(|f| f.bits_per_sample == 8 && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt)
    {
        ChannelKind::Int8
    } else {
        ChannelKind::Int16
    };
    loaded.channel_settings = plan
        .iter()
        .map(|&enabled| ChannelSettings {
            min: 0.0,
            max: 65535.0,
            enabled,
            bounds: (0.0, 65535.0),
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
pub(super) fn apply_dimension_override(loaded: &mut LoadedStack, channels: usize, slices: usize, frames: usize) {
    let resolved = fast_tiff_lib::ResolvedDimensions {
        channels,
        slices,
        frames,
        triple_axis_warning: loaded.triple_axis_warning,
    };
    apply_resolved_dimensions(loaded, resolved);
    // The channel->IFD mapping just changed, so invalidate any in-flight prefetch
    // decoded under the old mapping.
    loaded.prefetch_gen = loaded.prefetch_gen.wrapping_add(1);
}

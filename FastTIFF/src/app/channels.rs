//! Per-channel display settings and LUT/color helpers: contrast bounds from
//! the data or metadata, slider tints, the grayscale color/colormap selector
//! tables, and the pseudocolor toggle. Split from `app.rs`.

use super::*;
use crate::render::{ChannelKind, MAX_CHANNELS};
use egui::Color32;
use fast_tiff_lib::TiffStack;

/// Actual pixel min/max of channel `c`'s first frame, for integer-format
/// data. Used as the auto-contrast fallback when no display range came
/// from the file's metadata.
pub(super) fn first_frame_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    let pixels = fast_tiff_lib::read_frame_u16(&tiff.mmap, frame, tiff.byte_order, None).ok()?;
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    for &p in pixels.iter() {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if hi <= lo {
        hi = lo.saturating_add(1);
    }
    Some((lo as f32, hi as f32))
}

/// Actual float min/max of channel `c`'s first frame, for 32-bit float
/// data — matches ImageJ auto-ranging a float image to its own values
/// rather than assuming a fixed integer-shaped scale.
pub(super) fn first_frame_float_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    fast_tiff_lib::frame_float_minmax(&tiff.mmap, frame, tiff.byte_order).ok()?
}

/// Resizes `meta.channel_display` to `new_channels` entries, preserving the
/// per-channel display range. When the channel count is *unchanged* (the usual
/// case after `resolve_dimensions`), the existing LUTs are kept — including any
/// custom per-channel colors supplied by the IJMetadata block. When the count
/// *changes* (a mislabeled `channels=N` collapsing to a single channel, or a
/// manual channels/frames swap), the old LUTs no longer correspond to the new
/// channels, so they're regenerated from `mode` — which also avoids leaving a
/// collapsed grayscale stack wearing a stale composite (e.g. red) LUT.
pub(super) fn resize_channel_display(meta: &mut fast_tiff_lib::StackMeta, new_channels: usize) {
    let old = std::mem::take(&mut meta.channel_display);
    let mode = meta.mode;
    let keep_luts = new_channels == old.len();
    meta.channel_display = (0..new_channels)
        .map(|c| fast_tiff_lib::ChannelDisplay {
            lut: if keep_luts {
                old[c].lut
            } else {
                fast_tiff_lib::default_lut_for(mode, c)
            },
            range: old.get(c).and_then(|d| d.range),
        })
        .collect();
}

/// The contrast range-slider's track bounds: the channel's data min/max
/// (when known) unioned with the current display `window`, so both handles
/// always land on the track and the user has a little headroom to widen the
/// window past the metadata defaults. Falls back to the window alone when the
/// data range is unknown, and pads a degenerate (zero-width) range so the
/// slider stays usable.
pub(super) fn slider_bounds(window: (f32, f32), data: Option<(f32, f32)>) -> (f32, f32) {
    let (mut lo, mut hi) = window;
    if let Some((dlo, dhi)) = data {
        lo = lo.min(dlo);
        hi = hi.max(dhi);
    }
    if !(hi > lo) {
        let pad = lo.abs().max(1.0);
        lo -= pad;
        hi += pad;
    }
    (lo, hi)
}

/// The channel's display color for tinting its contrast slider, taken from the
/// top (full-intensity) entry of its LUT. Returns `None` for a plain grayscale
/// LUT (`r == g == b`), so only genuinely colored channels — composite/RGB, or a
/// pseudocolored grayscale stack — get a tinted slider; grayscale ones keep the
/// default selection color.
pub(super) fn channel_tint(lut: &[[u8; 3]; 256]) -> Option<Color32> {
    let [r, g, b] = lut[255];
    if r == g && g == b {
        None
    } else {
        Some(Color32::from_rgb(r, g, b))
    }
}

/// Like `channel_tint`, but taken from the LUT's *low* (index 0) entry — the
/// color the darkest samples map to. Used by the single-channel grayscale color
/// selector, where the low end better identifies a perceptual colormap (whose
/// high end washes out to near-white) than the top does. `None` for a
/// grayscale/black low entry (`r == g == b`), so plain grayscale and the pure
/// single-hue channel ramps — whose low end is black — keep the default color.
pub(super) fn ui_tint(lut: &[[u8; 3]; 256]) -> Option<Color32> {
    let [r, g, b] = lut[127];
    (!(r == g && g == b)).then(|| Color32::from_rgb(r, g, b))
}

/// The named channel colors the single-channel grayscale color selector offers
/// (after plain grayscale, before the perceptual colormaps). Order matches the
/// palette `fast_tiff_lib::default_composite_lut` cycles through, so color option
/// *i* maps to `default_composite_lut(i)`. White is omitted — its ramp is
/// identical to grayscale.
pub(super) const GRAY_LUT_COLOR_NAMES: [&str; 6] = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"];

/// Number of options in the grayscale color selector: plain grayscale, then the
/// named channel colors, then the perceptual colormaps (`crate::colormap`).
pub(super) fn gray_lut_option_count() -> usize {
    1 + GRAY_LUT_COLOR_NAMES.len() + crate::colormap::NAMES.len()
}

/// Display name for grayscale-color option `sel` (see `gray_lut_for` for the
/// index layout).
pub(super) fn gray_lut_name(sel: usize) -> &'static str {
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        "Grayscale"
    } else if sel <= colors {
        GRAY_LUT_COLOR_NAMES[sel - 1]
    } else {
        crate::colormap::NAMES[sel - 1 - colors]
    }
}

/// The 256-entry LUT for grayscale-color option `sel`: `0` = plain grayscale,
/// `1..=6` = the named channel colors, the rest = the perceptual colormaps —
/// the same order the selector lists them in.
pub(super) fn gray_lut_for(sel: usize) -> [[u8; 3]; 256] {
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        fast_tiff_lib::grayscale_lut()
    } else if sel <= colors {
        fast_tiff_lib::default_composite_lut(sel - 1)
    } else {
        crate::colormap::LUTS[sel - 1 - colors]
    }
}

/// Builds the UI-level per-channel settings (window/level, enabled,
/// float-encoding range) from `tiff.meta`'s current channel count and
/// display info.
pub(super) fn build_channel_settings(tiff: &TiffStack) -> Vec<ChannelSettings> {
    (0..tiff.meta.channels.min(MAX_CHANNELS))
        .map(|c| {
            let disp = &tiff.meta.channel_display[c];
            let frame = tiff.frames.get(c);
            let is_float = frame
                .is_some_and(|f| f.sample_format == fast_tiff_lib::SampleFormat::Float && f.bits_per_sample == 32);
            // Unsigned single-sample 8-bit can upload raw (R8Uint) instead of
            // being widened to 16-bit on the CPU each frame.
            let is_u8 = frame.is_some_and(|f| {
                f.bits_per_sample == 8
                    && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt
                    && f.samples_per_pixel == 1
            });

            if is_float {
                let data = first_frame_float_minmax(tiff, c);
                let (lo, hi) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .or(data)
                    .unwrap_or((0.0, 1.0));
                let bounds = slider_bounds((lo, hi), data);
                ChannelSettings { min: lo, max: hi, enabled: true, bounds, kind: ChannelKind::Float }
            } else {
                let data = first_frame_minmax(tiff, c);
                let (min, max) = disp
                    .range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    // No display range in metadata at all (not even
                    // ImageDescription min=/max=) — fall back to the actual
                    // min/max of channel c's first frame.
                    .or(data)
                    .unwrap_or((0.0, 65535.0));
                let bounds = slider_bounds((min, max), data);
                // min/max stay in the widened 0..65535 space (slider unchanged);
                // for an 8-bit (R8Uint) channel `sync_gpu` rescales them to the
                // 0..255 the texture actually holds.
                let kind = if is_u8 { ChannelKind::Int8 } else { ChannelKind::Int16 };
                ChannelSettings { min, max, enabled: true, bounds, kind }
            }
        })
        .collect()
}

/// Whether the "apply pseudocolor" option is meaningful for this stack: only
/// multi-channel grayscale stacks (composite files already carry colors; RGB is
/// handled separately) can be optionally tinted with the channel palette.
pub(super) fn pseudocolor_applicable(loaded: &LoadedStack) -> bool {
    !loaded.rgb
        && loaded.channel_settings.len() > 1
        && loaded.tiff.meta.mode == fast_tiff_lib::DisplayMode::Grayscale
        // The file's own per-channel LUTs win: don't override them with the
        // grayscale/pseudocolor default.
        && !loaded.tiff.meta.has_explicit_luts
}

/// Whether the single-channel grayscale color/colormap selector applies: exactly
/// one channel, genuinely grayscale (not RGB or composite, no file-supplied
/// LUTs). The multi-channel case is covered by the pseudocolor toggle instead.
pub(super) fn gray_lut_applicable(loaded: &LoadedStack) -> bool {
    !loaded.rgb
        && loaded.channel_settings.len() == 1
        && loaded.tiff.meta.mode == fast_tiff_lib::DisplayMode::Grayscale
        && !loaded.tiff.meta.has_explicit_luts
}

/// Sets the per-channel LUTs of an applicable (multi-channel grayscale) stack:
/// the standard channel palette (ch1 red, ch2 green, …) when `apply` is true,
/// plain grayscale otherwise. No-op for stacks that carry their own colors.
pub(super) fn refresh_pseudocolor(loaded: &mut LoadedStack, apply: bool) {
    if !pseudocolor_applicable(loaded) {
        return;
    }
    for (c, disp) in loaded.tiff.meta.channel_display.iter_mut().enumerate() {
        disp.lut = if apply {
            fast_tiff_lib::default_composite_lut(c)
        } else {
            fast_tiff_lib::grayscale_lut()
        };
    }
    loaded.luts_uploaded = false; // force re-upload on the next sync
}

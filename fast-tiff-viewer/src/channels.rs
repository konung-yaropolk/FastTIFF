//! Per-channel display settings and LUT/color helpers: contrast bounds from
//! the data or metadata, slider tints, the grayscale color/colormap selector
//! tables, and the pseudocolor toggle. Split from `app.rs`.

use crate::display::Display;
use crate::stack::{ChannelSettings, Stack};
use fast_tiff_lib::TiffStack;
use scivis_render::{ChannelKind, Lut};

/// How many samples an auto-contrast scan looks at, at most. The scan runs once
/// per channel while a file is opening — six of them for a 6-channel stack,
/// before the first frame can be drawn — so on a large image it is the visible
/// cost of an open. A strided sample of this many points estimates the display
/// range just as well: the result only seeds slider positions the user can
/// drag, and a range off by one outlier pixel is imperceptible.
const MINMAX_SAMPLE_BUDGET: usize = 1 << 18; // 262144

/// Actual pixel min/max of channel `c`'s first frame, for integer-format
/// data. Used as the auto-contrast fallback when no display range came
/// from the file's metadata.
///
/// Scans at most [`MINMAX_SAMPLE_BUDGET`] samples, striding across the frame
/// when it is larger. Decoding still costs a frame, but the scan itself stops
/// being O(pixels) on a 4K image.
pub fn first_frame_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    let pixels = fast_tiff_lib::read_frame_u16(&tiff.data, frame, tiff.byte_order, None).ok()?;
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    // A prime-ish stride avoids landing on the same column of every row, which
    // a power-of-two stride would do on a power-of-two-wide image and could
    // miss a whole feature.
    let stride = (pixels.len() / MINMAX_SAMPLE_BUDGET).max(1);
    for &p in pixels.iter().step_by(stride) {
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
pub fn first_frame_float_minmax(tiff: &TiffStack, channel: usize) -> Option<(f32, f32)> {
    let idx = channel.min(tiff.frames.len().saturating_sub(1));
    let frame = tiff.frames.get(idx)?;
    fast_tiff_lib::frame_float_minmax(&tiff.data, frame, tiff.byte_order).ok()?
}

/// The contrast range-slider's track bounds: the channel's data min/max
/// (when known) unioned with the current display `window`, so both handles
/// always land on the track and the user has a little headroom to widen the
/// window past the metadata defaults. Falls back to the window alone when the
/// data range is unknown, and pads a degenerate (zero-width) range so the
/// slider stays usable.
pub fn slider_bounds(window: (f32, f32), data: Option<(f32, f32)>) -> (f32, f32) {
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
/// frontend's default selection color.
///
/// Returns raw RGB rather than a toolkit color type: picking *whether* there is
/// a tint is display logic that belongs here, while turning it into an
/// `egui::Color32` (or a CSS string) belongs to the frontend.
pub fn channel_tint(lut: &Lut) -> Option<[u8; 3]> {
    let [r, g, b] = lut[255];
    (!(r == g && g == b)).then_some([r, g, b])
}

/// Like `channel_tint`, but taken from the LUT's *low* (index 0) entry — the
/// color the darkest samples map to. Used by the single-channel grayscale color
/// selector, where the low end better identifies a perceptual colormap (whose
/// high end washes out to near-white) than the top does. `None` for a
/// grayscale/black low entry (`r == g == b`), so plain grayscale and the pure
/// single-hue channel ramps — whose low end is black — keep the default color.
pub fn ui_tint(lut: &Lut) -> Option<[u8; 3]> {
    let [r, g, b] = lut[127];
    (!(r == g && g == b)).then_some([r, g, b])
}

/// The named channel colors the single-channel grayscale color selector offers
/// (after plain grayscale, before the perceptual colormaps). Order matches the
/// palette `fast_tiff_lib::default_composite_lut` cycles through, so color option
/// *i* maps to `default_composite_lut(i)`. White is omitted — its ramp is
/// identical to grayscale.
pub const GRAY_LUT_COLOR_NAMES: [&str; 6] = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"];

/// Number of options in the grayscale color selector: plain grayscale, then the
/// named channel colors, then the perceptual colormaps (`crate::colormap`).
pub fn gray_lut_option_count() -> usize {
    1 + GRAY_LUT_COLOR_NAMES.len() + crate::colormap::NAMES.len()
}

/// Display name for grayscale-color option `sel` (see `gray_lut_for` for the
/// index layout).
pub fn gray_lut_name(sel: usize) -> &'static str {
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
pub fn gray_lut_for(sel: usize) -> Lut {
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        fast_tiff_lib::grayscale_lut()
    } else if sel <= colors {
        fast_tiff_lib::default_composite_lut(sel - 1)
    } else {
        crate::colormap::LUTS[sel - 1 - colors]
    }
}

// When the file carries its own LUT (`builtin_lut`), the selector prepends a
// leading "Built-in" entry (the default) ahead of the computed options — so the
// built-in colormap is one option among grayscale/colors/colormaps. The helpers
// below fold that offset in, so the selector index (`Display::gray_lut_sel`) is
// `0 = Built-in` when present, else `0 = Grayscale`.

/// 1 when a leading "Built-in" option is present, else 0.
fn gray_lut_offset(display: &Display) -> usize {
    display.builtin_lut.is_some() as usize
}

/// Total selector options, including the leading "Built-in" when present.
pub fn gray_lut_count(display: &Display) -> usize {
    gray_lut_option_count() + gray_lut_offset(display)
}

/// Display name for selector index `sel` ("Built-in" at 0 when present).
pub fn gray_lut_sel_name(display: &Display, sel: usize) -> &'static str {
    let off = gray_lut_offset(display);
    if off == 1 && sel == 0 {
        "Built-in"
    } else {
        gray_lut_name(sel - off)
    }
}

/// The 256-entry LUT for selector index `sel` (the file's own map at 0 when
/// present, otherwise the computed option).
///
/// Returns by value because the caller is *installing* it. To merely tint a
/// widget, use [`gray_lut_sel_tint`] instead — it avoids building the LUT at all.
pub fn gray_lut_sel_lut(display: &Display, sel: usize) -> Lut {
    let off = gray_lut_offset(display);
    match display.builtin_lut {
        Some(lut) if sel == 0 => lut,
        _ => gray_lut_for(sel - off),
    }
}

/// The tint colour for selector index `sel`, without materialising its LUT.
///
/// A `Lut` is 768 bytes and the selector asks about *every* option on every
/// frame the combo is open, so building one per option per frame just to read a
/// single entry is pure waste. Only entry 127 matters (see [`ui_tint`]), and for
/// the generated ramps that entry is computable directly from the base colour.
pub fn gray_lut_sel_tint(display: &Display, sel: usize) -> Option<[u8; 3]> {
    let off = gray_lut_offset(display);
    if off == 1 && sel == 0 {
        return display.builtin_lut.as_ref().and_then(ui_tint);
    }
    let sel = sel - off;
    let colors = GRAY_LUT_COLOR_NAMES.len();
    if sel == 0 {
        None // plain grayscale: entry 127 is grey, so no tint
    } else if sel <= colors {
        // The channel ramps are `base * t`; at entry 127 that is `base * 127/255`.
        let base = fast_tiff_lib::composite_color(sel - 1);
        let scaled = [
            ((base[0] as u16 * 127) / 255) as u8,
            ((base[1] as u16 * 127) / 255) as u8,
            ((base[2] as u16 * 127) / 255) as u8,
        ];
        (!(scaled[0] == scaled[1] && scaled[1] == scaled[2])).then_some(scaled)
    } else {
        // A perceptual colormap is a `const` table — index it, don't copy it.
        ui_tint(&crate::colormap::LUTS[sel - 1 - colors])
    }
}

/// Builds the per-channel settings (window/level, enabled, float-encoding
/// range) for `channels` display channels.
///
/// Reads `tiff.meta` purely as a source of *defaults* — the file's suggested
/// display window — and never writes to it; the result is owned by
/// [`Display::settings`](crate::display::Display::settings).
pub fn build_channel_settings(tiff: &TiffStack, channels: usize) -> Vec<ChannelSettings> {
    (0..Display::shown_channels(channels))
        .map(|c| {
            // The metadata may describe fewer channels than we're showing (a
            // dimension override can raise the count), so fall back to "no
            // suggested window" rather than indexing past it.
            let meta_range = tiff.meta.channel_display.get(c).and_then(|d| d.range);
            let frame = tiff.frames.get(c);
            // Palette (indexed) channel: pixels are direct ColorMap indices, and
            // the ColorMap is already this channel's LUT. Pin the window to a
            // texel-centered identity so index i maps to LUT entry i exactly —
            // t = (i + 0.5)/256 lands on texel i's center, so even a linearly
            // filtered LUT returns entry i with no blend into a neighbour. The
            // window is in the 0..65535 UI space that `sync_gpu` divides by 257
            // for the 8-bit texture, so -0.5 and 255.5 in index space are:
            if frame.is_some_and(|f| f.is_palette()) {
                const W: f32 = 257.0; // the 8-bit → 0..65535 widening sync_gpu undoes
                let (lo, hi) = (-0.5 * W, 255.5 * W);
                return ChannelSettings { min: lo, max: hi, enabled: true, bounds: (lo, hi), kind: ChannelKind::Int8 };
            }
            // 32- and 64-bit IEEE floats both go to the R32F GPU path (64-bit is
            // downcast to f32 on decode). Integer data of every width — including
            // 64-bit int — takes the u16-rescale path below instead.
            let is_float = frame.is_some_and(|f| {
                f.sample_format == fast_tiff_lib::SampleFormat::Float
                    && (f.bits_per_sample == 32 || f.bits_per_sample == 64)
            });
            // Unsigned single-sample 8-bit uploads raw (R8Uint) instead of being
            // widened to 16-bit on the CPU each frame. 4-bit indices decode to
            // one byte each, so they ride the same R8Uint path.
            let is_u8 = frame.is_some_and(|f| {
                matches!(f.bits_per_sample, 4 | 8)
                    && f.sample_format == fast_tiff_lib::SampleFormat::UnsignedInt
                    && f.samples_per_pixel == 1
            });

            if is_float {
                let data = first_frame_float_minmax(tiff, c);
                let (lo, hi) = meta_range
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .or(data)
                    .unwrap_or((0.0, 1.0));
                let bounds = slider_bounds((lo, hi), data);
                ChannelSettings { min: lo, max: hi, enabled: true, bounds, kind: ChannelKind::Float }
            } else {
                let data = first_frame_minmax(tiff, c);
                let (min, max) = meta_range
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
pub fn pseudocolor_applicable(loaded: &Stack) -> bool {
    !loaded.display.rgb
        && loaded.display.channel_count() > 1
        && loaded.display.mode == fast_tiff_lib::DisplayMode::Grayscale
        // The file's own per-channel LUTs win: don't override them with the
        // grayscale/pseudocolor default.
        && !loaded.tiff.meta.has_explicit_luts
}

/// Whether the single-channel LUT/colormap selector applies: exactly one
/// channel that isn't RGB. It applies whether the channel is plain grayscale
/// *or* already carries a file-supplied LUT — in the latter case the selector
/// leads with a "Built-in LUT" option (the default) so the file's colors show,
/// with grayscale / channel colors / colormaps available to override. The
/// multi-channel case is covered by the pseudocolor toggle instead.
pub fn gray_lut_applicable(loaded: &Stack) -> bool {
    !loaded.display.rgb && loaded.display.channel_count() == 1
}

/// Sets the per-channel LUTs of an applicable (multi-channel grayscale) stack:
/// the standard channel palette (ch1 red, ch2 green, …) when `apply` is true,
/// plain grayscale otherwise. No-op for stacks that carry their own colors.
pub fn refresh_pseudocolor(loaded: &mut Stack, apply: bool) {
    if !pseudocolor_applicable(loaded) {
        return;
    }
    for c in 0..loaded.display.channel_count() {
        let lut = if apply {
            fast_tiff_lib::default_composite_lut(c)
        } else {
            fast_tiff_lib::grayscale_lut()
        };
        loaded.display.set_lut(c, lut);
    }
    loaded.luts_uploaded = false; // force re-upload on the next sync
}

//! Per-channel display settings and LUT/color helpers: contrast bounds from
//! the data or metadata, slider tints, the grayscale color/colormap selector
//! tables, and the pseudocolor toggle. Split from `app.rs`.

use crate::display::Display;
use crate::histogram::{auto_window, Histogram};
use crate::stack::{ChannelSettings, Stack};
use crate::units::{SampleUnits, Window};
use fast_tiff_lib::{DisplayMap, TiffStack};
use scivis_render::{ChannelKind, Lut};
use std::borrow::Cow;

/// How many samples an auto-contrast scan looks at, at most. The scan runs once
/// per channel while a file is opening — six of them for a 6-channel stack,
/// before the first frame can be drawn — so on a large image it is the visible
/// cost of an open. A strided sample of this many points estimates the display
/// range just as well: the result only seeds slider positions the user can
/// drag, and a range off by one outlier pixel is imperceptible.
const MINMAX_SAMPLE_BUDGET: usize = 1 << 18; // 262144

/// Actual pixel min/max of channel `c`'s first frame with contrast in it, for
/// integer-format data, in the file's own units — with the map the plane was
/// decoded through. Used as the auto-contrast fallback when no display range
/// came from the file's metadata.
///
/// The map is the plane's [`DisplayMap`], which an `Int16` channel's texture
/// holds: fixed for 8- and 16-bit data, and for a 32- or 64-bit integer plane
/// the range the decoder rescaled it through — which only the decoder knows,
/// since it picks the range from the plane itself.
///
/// Scans at most [`MINMAX_SAMPLE_BUDGET`] samples, striding across the frame
/// when it is larger. Decoding still costs a frame, but the scan itself stops
/// being O(pixels) on a 4K image.
pub fn channel_minmax(
    tiff: &TiffStack,
    channel: usize,
    step: usize,
) -> Option<(Window, DisplayMap)> {
    let one = |idx: usize| -> Option<((f32, f32), DisplayMap)> {
        let frame = tiff.frames.get(idx)?;
        // An 8- or 16-bit frame's map is fixed, and its plane can be borrowed
        // straight from the file. A wide one is ranged on its own plane, and
        // the reader that ranges it is the one that says what it picked.
        let mut owned = Vec::new();
        let (pixels, map) = match DisplayMap::of(frame, None) {
            Some(map) => (
                fast_tiff_lib::read_frame_u16(&tiff.data, frame, tiff.byte_order, None).ok()?,
                map,
            ),
            None => {
                let map = fast_tiff_lib::read_frame_u16_into(
                    &tiff.data,
                    frame,
                    tiff.byte_order,
                    None,
                    &mut owned,
                )
                .ok()?;
                (Cow::Borrowed(&owned[..]), map)
            }
        };
        let (mut lo, mut hi) = (u16::MAX, 0u16);
        // A prime-ish stride avoids landing on the same column of every row,
        // which a power-of-two stride would do on a power-of-two-wide image and
        // could miss a whole feature.
        let stride = (pixels.len() / MINMAX_SAMPLE_BUDGET).max(1);
        for &p in pixels.iter().step_by(stride) {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        Some(((lo as f32, hi as f32), map))
    };
    // Scanned, and widened when flat, on the decoder's display scale; only the
    // answer is put into file units.
    let ((lo, hi), map) = scan_for_contrast(tiff, channel, step, one)?;
    let file = |v: f32| map.to_file(v as f64) as f32;
    Some((Window::new(file(lo), file(hi)), map))
}

/// Actual float min/max of channel `c`'s first frame with contrast in it, for
/// 32-bit float data — matches ImageJ auto-ranging a float image to its own
/// values rather than assuming a fixed integer-shaped scale.
pub fn channel_float_minmax(tiff: &TiffStack, channel: usize, step: usize) -> Option<Window> {
    let measure = |idx: usize| {
        let frame = tiff.frames.get(idx)?;
        let range =
            fast_tiff_lib::frame_float_minmax(&tiff.data, frame, tiff.byte_order).ok()??;
        Some((range, ()))
    };
    scan_for_contrast(tiff, channel, step, measure).map(|(range, ())| range.into())
}

/// Where to look for a plane with contrast in it, as fractions of the
/// channel's own plane count.
///
/// Ordered, and the order is the whole idea. The first probe is plane 0, so the
/// common case costs exactly one decode and nothing about opening an ordinary
/// stack changes. After that it bisects — the middle, then the quarters, then
/// the eighths — which finds content wherever it happens to be in two or three
/// probes rather than walking towards it.
///
/// Walking was the first attempt and it is the wrong shape. "How many empty
/// planes does a volume start with" has no safe constant: the file that
/// prompted this, an SPM statistical map, begins with **exactly eight** blank
/// slices, so a budget of eight found nothing and a budget of nine would have
/// looked like it worked. Bisection has no such number in it.
const PROBES: [(usize, usize); 8] = [
    (0, 1),
    (1, 2),
    (1, 4),
    (3, 4),
    (1, 8),
    (3, 8),
    (5, 8),
    (7, 8),
];

/// A plane of channel `channel` that has some spread in it, measured with
/// `measure`, stepping by `step` to stay on the same channel.
///
/// # Why not just plane 0
///
/// Because the first plane of a volume is routinely empty, and a window fitted
/// to an empty plane has no width. The old behaviour took plane 0's min and
/// max, found them equal, and widened by exactly one unit — leaving a window
/// one value wide, which maps everything below it to black and everything above
/// to white. The picture came out as a two-tone mask, and nothing about it
/// suggested the contrast was at fault rather than the data.
///
/// Anything with air at the top of it lands here: a head CT, a brain MRI, a
/// z-stack with empty optical sections before the specimen. A statistical map
/// is the worst case, being exact zeros everywhere outside its mask.
///
/// `step` is the channel count, because planes are `xyczt` with the channel
/// fastest — so a channel's own planes are every `step`-th one. Stepping by 1
/// would sample a *different* channel and fit this channel's window to another
/// channel's data.
///
/// If every probe is flat, the first answer stands, widened by the smallest
/// amount that is still a window — which is what a genuinely uniform image
/// should get and what this has always done.
///
/// `measure` returns the plane's range and whatever else the caller needs to
/// read it by (the map an integer plane was decoded through), which comes back
/// with the probe that was chosen.
fn scan_for_contrast<T: Copy>(
    tiff: &TiffStack,
    channel: usize,
    step: usize,
    measure: impl Fn(usize) -> Option<((f32, f32), T)>,
) -> Option<((f32, f32), T)> {
    let step = step.max(1);
    let total = tiff.frames.len();
    if total == 0 {
        return None;
    }
    let start = channel.min(total - 1);
    // How many planes belong to this channel.
    let planes = (total - start).div_ceil(step).max(1);

    let mut first = None;
    let mut tried = [usize::MAX; PROBES.len()];
    for (slot, &(num, den)) in PROBES.iter().enumerate() {
        let n = (planes.saturating_mul(num) / den).min(planes - 1);
        // A short stack maps several fractions onto one plane; decoding it
        // repeatedly would make opening a two-plane file eight times as slow
        // for no new information.
        if tried[..slot].contains(&n) {
            continue;
        }
        tried[slot] = n;
        let Some(((lo, hi), with)) = measure(start + n * step) else {
            continue;
        };
        if hi > lo {
            return Some(((lo, hi), with));
        }
        first = first.or(Some(((lo, hi), with)));
    }
    first.map(|((lo, hi), with)| ((lo, if hi > lo { hi } else { lo + 1.0 }), with))
}

/// Apply a Shift-drag's delta from one channel's contrast handles to all the
/// others.
///
/// `before` is every channel's window as of the start of the frame, same order
/// as `settings`. Whichever channel differs now is the one the user dragged,
/// and each *enabled* other channel moves by the same amount, in file units,
/// clamped to its own bounds.
///
/// Switched-off channels are deliberately left alone. A frontend shows their
/// handles as inert and takes no input on them, so dragging a neighbour would
/// otherwise be the one way to move a control that is meant to be untouchable —
/// and it would do it invisibly, since that channel is not on screen for the
/// result to show up in.
///
/// A no-op when nothing moved, so a caller can run it unconditionally on any
/// frame the modifier is held.
pub fn shift_sync(settings: &mut [ChannelSettings], before: &[Window]) {
    let moved = settings.iter().enumerate().find_map(|(c, s)| {
        let b = before.get(c)?;
        let (dlo, dhi) = (s.window.lo - b.lo, s.window.hi - b.hi);
        (dlo != 0.0 || dhi != 0.0).then_some((c, dlo, dhi))
    });
    let Some((src, dlo, dhi)) = moved else {
        return;
    };
    for (i, s) in settings.iter_mut().enumerate() {
        if i == src || !s.enabled {
            continue;
        }
        let w = Window::new(s.window.lo + dlo, s.window.hi + dhi).clamp_to(s.bounds);
        s.window = Window::new(w.lo.min(w.hi), w.hi);
    }
}

/// The contrast range-slider's track bounds: the channel's data min/max
/// (when known) unioned with the current display `window`, so both handles
/// always land on the track and the user has a little headroom to widen the
/// window past the metadata defaults. Falls back to the window alone when the
/// data range is unknown, and pads a degenerate (zero-width) range so the
/// slider stays usable. File units in, file units out.
pub fn slider_bounds(window: Window, data: Option<Window>) -> Window {
    let Window { mut lo, mut hi } = data.map_or(window, |d| window.union(d));
    // Not `hi <= lo`: a NaN bound compares false both ways, and has to be
    // padded like an empty range rather than let through.
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        let pad = lo.abs().max(1.0);
        lo -= pad;
        hi += pad;
    }
    Window::new(lo, hi)
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
    build_channel_settings_reporting(tiff, channels, &mut |_, _| {})
}

/// [`build_channel_settings`], reporting `(done, total)` as each channel's scan
/// finishes.
///
/// This is the slow half of opening a file — every channel decodes a frame to
/// find its display range — and the only part of an open whose length is known
/// in advance, so it is the part a progress bar can be honest about.
pub fn build_channel_settings_reporting(
    tiff: &TiffStack,
    channels: usize,
    on_channel: &mut dyn FnMut(usize, usize),
) -> Vec<ChannelSettings> {
    let total = Display::shown_channels(channels);
    on_channel(0, total);
    (0..total)
        .map(|c| {
            // The *resolved* channel count, not `total`: `total` is capped at
            // what can be shown, while the plane stride is whatever the file
            // actually interleaves. A seven-channel file shows six and still
            // has its channel's planes seven apart.
            let settings = build_one(tiff, c, channels);
            on_channel(c + 1, total);
            settings
        })
        .collect()
}

/// One channel's contrast window and upload format.
///
/// `channels` is how many planes apart this channel's own planes are; see
/// [`scan_for_contrast`], which needs it to stay on one channel while looking
/// for a plane with something in it.
fn build_one(tiff: &TiffStack, c: usize, channels: usize) -> ChannelSettings {
    {
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
        // indices are the file's samples, so the window is in index space.
        if frame.is_some_and(|f| f.is_palette()) {
            return ChannelSettings::new(Window::new(-0.5, 255.5), SampleUnits::RAW8);
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

        let (units, data) = if is_float {
            (SampleUnits::FLOAT, channel_float_minmax(tiff, c, channels))
        } else {
            let scan = channel_minmax(tiff, c, channels);
            let units = if is_u8 {
                SampleUnits::RAW8
            } else {
                // What the texture will hold. A 32- or 64-bit integer plane is
                // rescaled through a range taken from the plane itself, so the
                // one the scan's plane was read with is the channel's.
                //
                // The decode still ranges every plane on its own, as it always
                // has; this records the scale the window was measured on, so
                // the window reaches the shader exactly as it did before units
                // were types.
                let range = match scan {
                    Some((_, DisplayMap::Linear { lo, hi })) => Some((lo, hi)),
                    _ => None,
                };
                SampleUnits::of(ChannelKind::Int16, frame, range)
            };
            (units, scan.map(|(window, _)| window))
        };
        // The file's own window when it declares one (ImageJ's `min=`/`max=`,
        // in ImageJ's units for the type), else the data's range, else the
        // whole of what the texture can show. All in file units.
        let window = meta_range
            .map(|range| units.from_imagej(range))
            .or(data)
            .unwrap_or_else(|| units.full_range());
        ChannelSettings {
            bounds: slider_bounds(window, data),
            ..ChannelSettings::new(window, units)
        }
    }
}

/// Fit every contrast window to where its data actually is, using histograms
/// already computed for the frame on screen.
///
/// Reads the histograms rather than rescanning the pixels: they describe the
/// frame the user is looking at and are already in hand, so Auto costs nothing
/// and — more importantly — agrees with the plot it is clicked underneath.
/// [`build_channel_settings`] instead scans frame 0 at load, which is the right
/// choice there and the wrong one here: on a stack whose brightness drifts,
/// seeding from frame 0 is a reasonable default, but auto-ing to it while
/// looking at frame 900 would be visibly wrong.
///
/// A channel with no histogram, or one whose histogram counted nothing, keeps
/// the window it had. Palette stacks are skipped entirely: their window is a
/// pinned index-to-LUT identity, not a contrast choice, and moving it would
/// remap every colour.
pub fn auto_contrast(loaded: &mut Stack, hists: &[Histogram]) {
    if loaded.display.palette {
        return;
    }
    for h in hists {
        let Some(s) = loaded.display.settings.get_mut(h.channel) else {
            continue;
        };
        let Some(fitted) = auto_window(h) else {
            continue;
        };
        // The histogram spans the union of every channel track, so a window
        // fitted on it can reach past this one. Clamping keeps both handles
        // somewhere the slider can actually draw them.
        s.window = fitted.clamp_to(s.bounds);
        // Both ends landing on the same side of the track would leave an empty
        // window, which divides by zero on the way to the shader. Fall back to
        // showing everything, which is at least a picture.
        if s.window.hi <= s.window.lo {
            s.window = s.bounds;
        }
    }
}

/// Put every contrast window back where it was when the file was opened — the
/// undo for any amount of dragging, and for [`auto_contrast`].
///
/// The window the file asked for, not the full slider track. Those differ
/// whenever a file declares a display range narrower than its data, which is
/// the normal case for anything saved out of ImageJ: opening the window to the
/// full track would "reset" such a file to something it has never been shown
/// as. Restoring what it opened with is the state the user is actually trying
/// to get back to, and it costs nothing to remember — see
/// [`ChannelSettings::initial`].
///
/// Reopening a *different* interpretation of the same file — a dimension-order
/// change, for instance — rebuilds the settings and with them the window this
/// returns to, which is right: the starting point is whatever the current
/// reading of the file starts from.
pub fn reset_contrast(loaded: &mut Stack) {
    if loaded.display.palette {
        return;
    }
    for s in loaded.display.settings.iter_mut() {
        s.window = s.initial;
    }
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

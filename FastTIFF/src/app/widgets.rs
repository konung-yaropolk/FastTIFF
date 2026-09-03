//! Small bespoke UI pieces shared by the panels and the pop-up windows: the
//! two-handle contrast range slider, the per-channel contrast block built from
//! it, and calibrated value formatting. Split from `app.rs`.

use super::tint_color;
use egui::{Color32, RichText};
use fast_tiff_viewer::channels::{channel_tint, shift_sync, ui_tint};
use fast_tiff_viewer::Stack;

/// Formats a raw sample value for display, applying the stack's linear
/// calibration (`c0 + c1 * raw`) when present so the user sees real values;
/// otherwise shows the raw value. Picks a coarse/fine precision by magnitude.
pub(super) fn format_calibrated(calibration: Option<(f64, f64)>, raw: f32) -> String {
    let v = match calibration {
        Some((c0, c1)) => c0 + c1 * raw as f64,
        None => raw as f64,
    };
    if v.abs() >= 100.0 || v.fract().abs() < 1e-6 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

/// The contrast range sliders never draw narrower than this, no matter how
/// small the window gets — below it the two handles collide and the slider
/// stops being usable. The value text to the right clips first.
pub(super) const MIN_CONTRAST_SLIDER_W: f32 = 60.0;

/// Width kept clear at the right of an [`ContrastLayout::Inline`] row for the
/// window's value text, which is drawn after the slider.
const VALUE_RESERVE: f32 = 120.0;

/// How much of their colour a switched-off channel's bar and handles keep.
///
/// Split rather than one blanket fade — which is all `Ui::disable` offers, and
/// why the slider paints itself instead of relying on it. The bar recedes so
/// the enabled channels read first, while the handles stay at full strength:
/// they mark where this channel's window sits, which is exactly what you look
/// at before deciding whether to switch it back on. Nothing depends on the
/// slider *looking* dead — it takes no input either way, and the unticked
/// checkbox already says so.
const DISABLED_BAR_OPACITY: f32 = 0.25;
const DISABLED_HANDLE_OPACITY: f32 = 1.0;

/// Radius of a range slider's draggable handles.
///
/// A handle is centred *on* the value it marks, so at either end of the track
/// half of it hangs past the track's edge. Any layout that runs a slider up to
/// the edge of its container has to leave this much room on both sides, or the
/// end handles are sliced in half by the frame.
const HANDLE_RADIUS: f32 = 6.0;

/// A two-handle horizontal range slider editing `(min, max)` within the
/// inclusive track `[lo, hi]` (all in raw sample units). The handles can't
/// cross. `salt` disambiguates the interaction ids when several sliders share
/// a parent (e.g. one per channel). `tint`, when set, colors the selected span
/// with the channel's display color (composite/RGB or pseudocolor); otherwise
/// the default selection color is used.
///
/// `enabled` is the channel's own on/off state, not the surrounding `Ui`'s:
/// a switched-off channel's slider takes no input and dims its handles, but
/// keeps most of the colour in its bar. Blanket-fading the whole widget (what
/// `Ui::disable` does) washed the bar out along with everything else, and that
/// bar is the one part still worth reading — it says where that channel's
/// window sits, which is what you want to see before deciding to switch it back
/// on. The handles are what you cannot drag, so those are what look inert.
///
/// Returns the rect the track was drawn in, so a caller can align something
/// above it — the histogram window plots each channel across exactly the span
/// its own slider covers.
pub(super) fn range_slider(
    ui: &mut egui::Ui,
    salt: u64,
    min: &mut f32,
    max: &mut f32,
    lo: f32,
    hi: f32,
    width: f32,
    tint: Option<Color32>,
    enabled: bool,
) -> egui::Rect {
    // Defensive: keep the handles inside the track and ordered, even if the
    // values were pushed out of range elsewhere (e.g. by the shift-sync).
    *min = (*min).clamp(lo, hi);
    *max = (*max).clamp(lo, hi).max(*min);
    let height = 18.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let span = (hi - lo).max(f32::EPSILON);
    let x_of = |v: f32| rect.left() + ((v - lo) / span).clamp(0.0, 1.0) * rect.width();
    let v_of = |x: f32| lo + ((x - rect.left()) / rect.width()).clamp(0.0, 1.0) * span;
    let track_y = rect.center().y;
    let visuals = ui.visuals().clone();

    // Track + the selected span between the two handles.
    let track = egui::Rect::from_min_max(
        egui::pos2(rect.left(), track_y - 2.0),
        egui::pos2(rect.right(), track_y + 2.0),
    );
    ui.painter()
        .rect_filled(track, 2.0, visuals.widgets.inactive.bg_fill);
    let sel = egui::Rect::from_min_max(
        egui::pos2(x_of(*min), track_y - 2.0),
        egui::pos2(x_of(*max), track_y + 2.0),
    );
    let bar = tint.unwrap_or(visuals.selection.bg_fill);
    let bar = if enabled {
        bar
    } else {
        bar.gamma_multiply(DISABLED_BAR_OPACITY)
    };
    ui.painter().rect_filled(sel, 2.0, bar);

    let radius = HANDLE_RADIUS;

    // Which handle the in-flight drag is moving — see `dragged_handle`. Held
    // for the whole drag, so the handle you grabbed keeps being the one that
    // moves and pushes the other along; being pushed never makes a handle the
    // dragged one, so it cannot steal the drag from underneath.
    let drag_id = ui.id().with((salt, "range_drag"));

    let mut dragging = false;
    for is_max in [false, true] {
        let mut active = false;
        if enabled {
            let id = ui
                .id()
                .with((salt, if is_max { "range_max" } else { "range_min" }));
            let at = if is_max { *max } else { *min };
            let hit = egui::Rect::from_center_size(
                egui::pos2(x_of(at), track_y),
                egui::vec2(radius * 2.5, height),
            );
            let resp = ui.interact(hit, id, egui::Sense::drag());
            dragging |= resp.dragged();
            if resp.dragged() {
                if let Some(p) = resp.interact_pointer_pos() {
                    let v = v_of(p.x);
                    let held = ui.data(|d| d.get_temp::<bool>(drag_id));
                    if let Some(moving) = dragged_handle(*min, *max, v, held, is_max) {
                        ui.data_mut(|d| d.insert_temp(drag_id, moving));
                        // The handle being dragged goes exactly where the
                        // pointer is, and shoves the other along ahead of it
                        // instead of stopping against it. Reversing leaves the
                        // pushed one where it was pushed to, so a pair driven
                        // to one end can be opened back up from there.
                        if moving {
                            *max = v;
                            *min = (*min).min(*max);
                        } else {
                            *min = v;
                            *max = (*max).max(*min);
                        }
                    }
                }
            }
            active = resp.dragged() || resp.hovered();
        }
        let col = handle_color(&visuals, active);
        let col = if enabled {
            col
        } else {
            col.gamma_multiply(DISABLED_HANDLE_OPACITY)
        };
        // Read the value back rather than reusing `at`: this handle may have
        // just moved, and painting the pre-drag position lags it by a frame.
        let at = if is_max { *max } else { *min };
        ui.painter()
            .circle_filled(egui::pos2(x_of(at), track_y), radius, col);
    }

    // The committed handle only means anything while a drag is actually
    // happening, so drop it the moment none is. Clearing on `drag_stopped`
    // instead would hang the whole thing on catching one edge event: miss it
    // — the pointer leaves the window, the app loses focus, the release lands
    // on a frame the widget is not drawn — and the choice leaks into the next
    // drag, which then moves the handle you did not grab.
    if !dragging {
        ui.data_mut(|d| d.remove::<bool>(drag_id));
    }

    rect
}

/// Which handle a drag is moving: `Some(true)` for max, `Some(false)` for min,
/// or `None` while it cannot yet be told.
///
/// `grabbed` is the handle the pointer caught, `held` the choice already made
/// earlier in this same drag (`None` on its first frame).
///
/// Once a drag has committed to a handle it keeps it for the rest of the drag,
/// which is what makes the moving handle *push* the other rather than stop
/// against it or hand over to it. The only decision is therefore the first one:
///
/// * apart — the handle you grabbed;
/// * stacked — the side the pointer is on, since at a shared value a pointer to
///   the right can only mean max and one to the left only min, whichever widget
///   happened to win the grab. Without this a stack is a trap: the grabbed
///   handle may be the one with nowhere to go, and the pair never separates;
/// * stacked with the pointer still exactly on the pair — `None`. There is no
///   direction to read yet, so nothing moves and nothing is committed; the next
///   frame that the pointer is off the shared value decides.
fn dragged_handle(min: f32, max: f32, v: f32, held: Option<bool>, grabbed: bool) -> Option<bool> {
    if held.is_some() {
        return held;
    }
    if min == max {
        return if v > max {
            Some(true)
        } else if v < min {
            Some(false)
        } else {
            None
        };
    }
    Some(grabbed)
}

pub(super) fn handle_color(visuals: &egui::Visuals, active: bool) -> Color32 {
    if active {
        visuals.widgets.active.fg_stroke.color
    } else {
        visuals.widgets.inactive.fg_stroke.color
    }
}

/// How one channel's contrast controls are arranged.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ContrastLayout {
    /// Checkbox, slider and values all on one line. Compact, for the bottom
    /// panel, where every row costs height the image would otherwise have.
    Inline,
    /// Checkbox and values on one line, the slider full-width beneath them.
    /// Two lines per channel instead of one, spent to give the slider the whole
    /// width — and with it anything aligned to the slider, which is why the
    /// histogram window uses this: the plot ends up as wide as the window.
    Stacked,
}

/// What sits at the left of a contrast row.
enum RowHead<'a> {
    /// A per-channel enable toggle, with optional explanatory hover text.
    Toggle {
        label: String,
        enabled: &'a mut bool,
        hover: Option<&'a str>,
    },
    /// A plain caption, for a single-channel stack — switching off the only
    /// channel would just blank the image, so it gets no checkbox.
    Caption(&'a str),
}

/// Draw a row's head. Returns whether the channel it governs is switched on —
/// the head itself always stays live, since for a disabled channel it is the
/// only way back.
fn draw_head(ui: &mut egui::Ui, head: RowHead<'_>) -> bool {
    match head {
        RowHead::Toggle {
            label,
            enabled,
            hover,
        } => {
            // Fixed-width checkbox so every inline slider starts at the same x
            // regardless of label length.
            let check = ui.add_sized(egui::vec2(48.0, 18.0), egui::Checkbox::new(enabled, label));
            if let Some(hover) = hover {
                check.on_hover_text(hover);
            }
            *enabled
        }
        RowHead::Caption(text) => {
            ui.label(text);
            true
        }
    }
}

/// One channel's contrast controls. Returns the slider's horizontal span.
///
/// `window` is the channel's `(min, max, bounds)` — passed as separate borrows
/// rather than the whole `ChannelSettings` so `head` can hold `&mut enabled`
/// from the same struct at the same time.
fn contrast_row(
    ui: &mut egui::Ui,
    layout: ContrastLayout,
    salt: u64,
    head: RowHead<'_>,
    window: (&mut f32, &mut f32, (f32, f32)),
    calibration: Option<(f64, f64)>,
    tint: Option<Color32>,
) -> egui::Rangef {
    let (min, max, (lo, hi)) = window;
    let value = format!(
        "{} – {}",
        format_calibrated(calibration, *min),
        format_calibrated(calibration, *max),
    );
    match layout {
        ContrastLayout::Inline => {
            let mut track = egui::Rangef::NOTHING;
            ui.horizontal(|ui| {
                let live = draw_head(ui, head);
                // A switched-off channel is not being composited, so its window
                // is not in effect. The slider handles that itself (it dims its
                // handles but keeps its bar readable); the values it reports
                // grey out wholesale, since a number for a window that is not
                // applied is just noise.
                // Reserve room for the value text on the right; the slider
                // fills what is left of the row.
                let w = (ui.available_width() - VALUE_RESERVE).max(MIN_CONTRAST_SLIDER_W);
                track = range_slider(ui, salt, min, max, lo, hi, w, tint, live).x_range();
                ui.add_enabled_ui(live, |ui| {
                    ui.label(RichText::new(value).small());
                });
            });
            track
        }
        ContrastLayout::Stacked => {
            let mut live = true;
            ui.horizontal(|ui| {
                live = draw_head(ui, head);
                // Values pushed to the far right, so the head and the numbers
                // read as end labels on the slider spanning the line below.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_enabled_ui(live, |ui| {
                        ui.label(RichText::new(value).small());
                    });
                });
            });
            // Inset by a handle's radius at each end: the slider runs the full
            // width of its container here, and a handle parked at either
            // extreme is centred on the track's edge, so without this half of
            // it is cut off by the window frame.
            let mut track = egui::Rangef::NOTHING;
            ui.horizontal(|ui| {
                ui.add_space(HANDLE_RADIUS);
                let w = (ui.available_width() - HANDLE_RADIUS).max(MIN_CONTRAST_SLIDER_W);
                track = range_slider(ui, salt, min, max, lo, hi, w, tint, live).x_range();
            });
            // Separate this channel's slider from the next one's head, which
            // would otherwise sit tight against it and read as one block.
            ui.add_space(4.0);
            track
        }
    }
}

/// The per-channel contrast controls: one enable-checkbox and two-handle window
/// slider per channel, or a single unlabelled row when the stack has one
/// channel, plus the Shift-to-move-all-together behaviour.
///
/// Shared by the bottom panel and the histogram window. They are the same
/// control in two places — a value dragged in one is visibly live in the other,
/// because both are editing `loaded.display.settings` directly — and writing it
/// once is what keeps them that way.
///
/// `layout` picks how each channel's three pieces are arranged — see
/// [`ContrastLayout`].
///
/// `track` overrides the range each slider spans. By default a channel's
/// slider spans its own `ChannelSettings::bounds`, which gives the finest drag
/// resolution per channel. Pass `Some` to put every channel on one common
/// range instead: the histogram window does, so that a handle's position means
/// the same thing as a position on the plot above it. Widening a track can only
/// relax the clamp `range_slider` applies, so an override never alters a value.
///
/// Returns the horizontal span the sliders were drawn across, so a caller can
/// align a plot with the track underneath it. `None` when no slider was drawn:
/// a palette (indexed) stack has no contrast window at all.
pub(super) fn contrast_controls(
    ui: &mut egui::Ui,
    loaded: &mut Stack,
    layout: ContrastLayout,
    track: Option<(f32, f32)>,
) -> Option<egui::Rangef> {
    let mut span: Option<egui::Rangef> = None;
    let calibration = loaded.tiff.meta.calibration;
    let rgb = loaded.display.rgb;
    // A palette channel's window is a fixed index→LUT identity, so
    // there's nothing to adjust — its contrast slider is suppressed.
    let palette = loaded.display.palette;
    // Tint for the single-channel contrast slider: the low (dark) end
    // of its chosen color LUT (grayscale/black → None → the default
    // selection color). Snapshot here, before the mutable borrow of
    // `channel_settings` below.
    let single_tint = (loaded.display.settings.len() == 1)
        .then(|| tint_color(loaded.display.lut(0).and_then(ui_tint)))
        .flatten();
    if loaded.display.settings.len() > 1 {
        ui.separator();
        // Hold Shift while dragging one channel's slider to move every
        // channel's window by the same amount. Snapshot the values
        // first so we can detect which one moved and by how much.
        let shift = ui.input(|i| i.modifiers.shift);
        let before: Vec<(f32, f32)> = loaded
            .display
            .settings
            .iter()
            .map(|s| (s.min, s.max))
            .collect();
        // Per-channel slider tints from each channel's display LUT —
        // colored only for composite/RGB or pseudocolor stacks, `None`
        // (default color) for plain grayscale.
        let tints: Vec<Option<Color32>> = loaded
            .tiff
            .meta
            .channel_display
            .iter()
            .map(|cd| tint_color(channel_tint(&cd.lut)))
            .collect();
        // One entry per channel, stacked vertically.
        for (c, settings) in loaded.display.settings.iter_mut().enumerate() {
            let label = if rgb {
                // Sample planes past RGBA have no conventional letter — number
                // them instead.
                ["R", "G", "B", "A"]
                    .get(c)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("S{}", c + 1))
            } else {
                format!("Ch {}", c + 1)
            };
            // Samples past RGB start off (see `setup_rgb`) — say why, on the
            // row it applies to.
            let hover = (rgb && c >= 3).then_some(
                "Extra sample plane beyond RGB (TIFF ExtraSamples). Usually alpha, \
                 but writers also put real data here — a (4, H, W) array saved by \
                 tifffile lands as RGB + this. Off by default because compositing \
                 an opaque alpha plane washes the image out; enable it to see the \
                 data.",
            );
            span = Some(contrast_row(
                ui,
                layout,
                c as u64,
                RowHead::Toggle {
                    label,
                    enabled: &mut settings.enabled,
                    hover,
                },
                (
                    &mut settings.min,
                    &mut settings.max,
                    track.unwrap_or(settings.bounds),
                ),
                calibration,
                tints.get(c).copied().flatten(),
            ));
        }
        // Shift-sync: if a slider moved this frame, carry the same delta to
        // every other enabled channel.
        if shift {
            shift_sync(&mut loaded.display.settings, &before);
        }
        ui.label(
            RichText::new("Hold Shift while dragging to adjust all channels together.")
                .small()
                .weak(),
        );
    } else if !palette {
        if let Some(settings) = loaded.display.settings.first_mut() {
            ui.separator();
            // A caption rather than a checkbox: switching off the only channel
            // would just blank the image.
            span = Some(contrast_row(
                ui,
                layout,
                0,
                RowHead::Caption("Contrast:"),
                (
                    &mut settings.min,
                    &mut settings.max,
                    track.unwrap_or(settings.bounds),
                ),
                calibration,
                single_tint,
            ));
        }
    }
    // No `else`: a palette (indexed) image shows no contrast row at
    // all. Its pixels are colour-table indices rather than
    // intensities — index 37 isn't brighter than 12, the map
    // decides — so there is no window to adjust. Changing the table
    // is the LUT selector's job.

    span
}

/// A small bar-chart icon button, for opening the histogram window.
///
/// Painted rather than set as a glyph, for the same reason the play/pause icons
/// are: the bundled font's coverage of symbol characters can't be relied on
/// (the toolbar already has one ASCII arrow standing in for `→`), and a tofu box
/// in the panel is worse than a dozen lines of painting.
pub(super) fn histogram_button(ui: &mut egui::Ui) -> egui::Response {
    let resp = ui
        .add_sized(egui::vec2(24.0, 18.0), egui::Button::new(""))
        .on_hover_text("Channel histogram and contrast");
    let color = ui.style().interact(&resp).fg_stroke.color;
    let r = resp.rect.shrink2(egui::vec2(6.0, 5.0));
    // Three bars, uneven heights — a histogram at a glance rather than a
    // bar chart's even steps.
    let pitch = r.width() / 3.0;
    let bar_w = (pitch * 0.62).max(1.0);
    for (i, frac) in [0.5_f32, 1.0, 0.72].iter().enumerate() {
        let x = r.left() + i as f32 * pitch;
        let bar = egui::Rect::from_min_max(
            egui::pos2(x, r.bottom() - r.height() * frac),
            egui::pos2(x + bar_w, r.bottom()),
        );
        ui.painter().rect_filled(bar, 0.0, color);
    }
    resp
}

#[cfg(test)]
#[path = "widgets_tests.rs"]
mod tests;

//! Small bespoke UI pieces shared by the panels: the two-handle contrast
//! range slider and calibrated value formatting. Split from `app.rs`.

use egui::Color32;

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
pub(super) const MIN_CONTRAST_SLIDER_W: f32 = 80.0;

/// A two-handle horizontal range slider editing `(min, max)` within the
/// inclusive track `[lo, hi]` (all in raw sample units). The handles can't
/// cross. `salt` disambiguates the interaction ids when several sliders share
/// a parent (e.g. one per channel). `tint`, when set, colors the selected span
/// with the channel's display color (composite/RGB or pseudocolor); otherwise
/// the default selection color is used.
pub(super) fn range_slider(
    ui: &mut egui::Ui,
    salt: u64,
    min: &mut f32,
    max: &mut f32,
    lo: f32,
    hi: f32,
    width: f32,
    tint: Option<Color32>,
) {
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
    ui.painter().rect_filled(track, 2.0, visuals.widgets.inactive.bg_fill);
    let sel = egui::Rect::from_min_max(
        egui::pos2(x_of(*min), track_y - 2.0),
        egui::pos2(x_of(*max), track_y + 2.0),
    );
    ui.painter().rect_filled(sel, 2.0, tint.unwrap_or(visuals.selection.bg_fill));

    let radius = 6.0;
    // min handle.
    {
        let id = ui.id().with((salt, "range_min"));
        let hit = egui::Rect::from_center_size(egui::pos2(x_of(*min), track_y), egui::vec2(radius * 2.5, height));
        let resp = ui.interact(hit, id, egui::Sense::drag());
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                *min = v_of(p.x).min(*max);
            }
        }
        let col = handle_color(&visuals, resp.dragged() || resp.hovered());
        ui.painter().circle_filled(egui::pos2(x_of(*min), track_y), radius, col);
    }
    // max handle.
    {
        let id = ui.id().with((salt, "range_max"));
        let hit = egui::Rect::from_center_size(egui::pos2(x_of(*max), track_y), egui::vec2(radius * 2.5, height));
        let resp = ui.interact(hit, id, egui::Sense::drag());
        if resp.dragged() {
            if let Some(p) = resp.interact_pointer_pos() {
                *max = v_of(p.x).max(*min);
            }
        }
        let col = handle_color(&visuals, resp.dragged() || resp.hovered());
        ui.painter().circle_filled(egui::pos2(x_of(*max), track_y), radius, col);
    }
}

pub(super) fn handle_color(visuals: &egui::Visuals, active: bool) -> Color32 {
    if active {
        visuals.widgets.active.fg_stroke.color
    } else {
        visuals.widgets.inactive.fg_stroke.color
    }
}

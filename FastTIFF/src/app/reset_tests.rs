//! What a new document resets, part by part.
//!
//! `finish_open` used to null a list of fields one by one, and a field added
//! without a line there carried the last file's state into the next. Each part
//! of the chrome now resets itself; these pin what each one drops and what it
//! deliberately keeps, so a change to either is a change someone made on
//! purpose. The canvas tools' reset is pinned with the tools, in
//! `canvas_tool_tests`, and the pop-ups' in `windows_tests`.

use super::*;

/// A view with every field moved off its fresh value.
fn used_view() -> View2d {
    let mut glide = kinetic::Glide::default();
    glide.push(egui::vec2(40.0, -25.0), 1.0 / 60.0);
    assert!(glide.is_moving(), "the fixture's glide should be moving");
    View2d {
        zoom: 3.0,
        pan: egui::vec2(12.0, 34.0),
        pending_initial_fit: false,
        fit_zoom: Some(0.4),
        resize_to_zoom: true,
        zoom_reposition: Some((2.0, egui::pos2(5.0, 6.0))),
        zoom_glide: Some(ZoomGlide {
            target: 4.0,
            anchor: egui::pos2(7.0, 8.0),
        }),
        scroll_accum: 0.75,
        image_origin: egui::pos2(-3.0, 9.0),
        panel_rect: egui::Rect::from_min_size(egui::pos2(0.0, 30.0), egui::vec2(640.0, 480.0)),
        pan_glide: glide,
        was_pannable: true,
        minimap_drag: true,
        autoscroll: Some(egui::pos2(100.0, 200.0)),
    }
}

/// The transform and every gesture over it go back to where a freshly opened
/// picture starts, with the opening fit still to come.
#[test]
fn a_new_document_resets_the_view_to_a_fresh_one_awaiting_its_fit() {
    let mut view = used_view();
    view.reset();
    let fresh = View2d::default();
    assert_eq!(view.zoom, fresh.zoom);
    assert_eq!(view.pan, fresh.pan);
    assert!(view.pending_initial_fit, "the opening fit is still to come");
    assert_eq!(view.fit_zoom, fresh.fit_zoom);
    assert_eq!(view.resize_to_zoom, fresh.resize_to_zoom);
    assert_eq!(view.zoom_glide, fresh.zoom_glide);
    assert_eq!(view.was_pannable, fresh.was_pannable);
    assert_eq!(view.minimap_drag, fresh.minimap_drag);
    assert_eq!(view.autoscroll, fresh.autoscroll);
    assert!(
        !view.pan_glide.is_moving(),
        "no momentum from the last picture"
    );
}

/// What the reset leaves alone: the last layout's geometry, which the next
/// frame recomputes; a zoom reposition, which a zoom step's own frame takes;
/// and wheel travel short of a frame. None of it was ever reset, and a reset
/// that started dropping it would be a change of behaviour.
#[test]
fn a_new_document_keeps_the_views_layout_bookkeeping() {
    let used = used_view();
    let mut view = used_view();
    view.reset();
    assert_eq!(view.image_origin, used.image_origin);
    assert_eq!(view.panel_rect, used.panel_rect);
    assert_eq!(view.zoom_reposition, used.zoom_reposition);
    assert_eq!(view.scroll_accum, used.scroll_accum);
}

/// The channels panel is rebuilt wholesale, a grow caught in flight included.
#[test]
fn a_new_document_rebuilds_the_channels_panel() {
    let mut panel = PanelLayout {
        expanded: true,
        ..Default::default()
    };
    panel.note_info_row(Some("Saved".into()), 40.0);
    panel.arm_grow(60.0);
    panel.reset();
    assert!(!panel.expanded);
    assert_eq!(panel.grow_armed, 0);
    assert_eq!(panel.last_info, None);
    assert_eq!(panel.last_h, 0.0);
}

/// The 3D preferences and the panel's bookkeeping start where the app always
/// started them; a new document has no reset for them at all.
#[test]
fn the_volume_ui_starts_with_the_default_preferences() {
    let ui = VolumeUi::default();
    assert!(!ui.prefs.show_coord_box);
    assert_eq!(ui.prefs.move_speed, 1.0);
    assert_eq!(ui.prefs.scroll_speed, 1.0);
    assert_eq!(ui.prefs.moving_downscale, render::MOVING_DOWNSCALE);
    assert!(ui.rect.is_none() && ui.pending_copy.is_none());
    assert!(!ui.pressed);
    assert!(ui.last_camera.is_none() && ui.moved_at.is_none());
}

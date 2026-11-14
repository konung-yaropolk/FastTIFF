//! What a new document resets, part by part.
//!
//! `finish_open` used to null a list of fields one by one, and a field added
//! without a line there carried the last file's state into the next. Each part
//! of the chrome now resets itself; these pin what each one drops and what it
//! deliberately keeps, so a change to either is a change someone made on
//! purpose. The canvas tools' reset is pinned with the tools, in
//! `canvas_tool_tests`, and the pop-ups' in `windows_tests`.

use super::*;

/// A view moved off its fresh value in every way the window can: zoomed,
/// panned, stepped, autoscrolling, with a resize, a reposition and a
/// navigator drag in flight. The core view's own reset is pinned with it,
/// in `fast-tiff-viewer`'s `view2d/tests.rs`.
fn used_view() -> CanvasView {
    use fast_tiff_viewer::view2d::{ImagePx, LogicalPt, LogicalRect, LogicalVec};
    let mut model = View2d::default();
    model.set_panel(LogicalRect::from_min_size(
        LogicalPt::new(0.0, 30.0),
        LogicalVec::new(640.0, 480.0),
    ));
    let layout = model.measure(ImagePx::new(4000, 3000), LogicalVec::new(640.0, 480.0));
    model.open_fitted(0.4);
    model.drag(LogicalVec::new(-12.0, -34.0));
    model.toggle_autoscroll(Some(LogicalPt::new(100.0, 200.0)), &layout);
    let step = model.step_zoom(1, None).expect("a rung above");
    assert!(model.is_zoom_gliding());
    CanvasView {
        model,
        resize_to_zoom: true,
        zoom_reposition: Some(step),
        minimap_drag: true,
    }
}

/// The view and every gesture over it go back to where a freshly opened
/// picture starts, with the opening fit still to come; the window's resize
/// and the navigator's drag go with the picture they were for.
#[test]
fn a_new_document_resets_the_view_to_a_fresh_one_awaiting_its_fit() {
    let mut view = used_view();
    view.reset();
    let fresh = View2d::default();
    assert_eq!(view.model.zoom(), fresh.zoom());
    assert_eq!(view.model.pan(), fresh.pan());
    assert!(view.model.awaits_fit(), "the opening fit is still to come");
    assert_eq!(view.model.fit_rung(), None);
    assert!(!view.model.is_zoom_gliding());
    assert_eq!(view.model.autoscroll(), None);
    assert!(!view.resize_to_zoom);
    assert!(!view.minimap_drag);
}

/// What the reset leaves alone: the last layout's geometry, which the next
/// frame recomputes, and a zoom reposition, which a zoom step's own frame
/// takes. None of it was ever reset, and a reset that started dropping it
/// would be a change of behaviour.
#[test]
fn a_new_document_keeps_the_views_layout_bookkeeping() {
    let used = used_view();
    let mut view = used_view();
    view.reset();
    assert_eq!(view.model.image_origin(), used.model.image_origin());
    assert_eq!(view.model.panel(), used.model.panel());
    assert_eq!(view.zoom_reposition, used.zoom_reposition);
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

/// The 3D preferences start where the app always started them — the core's
/// defaults, which are the renderer's — and the panel's bookkeeping empty; a
/// new document has no reset for either.
#[test]
fn the_volume_ui_starts_with_the_default_preferences() {
    let prefs = Viewer::new().volume.navigation;
    assert!(!prefs.show_coord_box);
    assert_eq!(prefs.move_speed, 1.0);
    assert_eq!(prefs.scroll_speed, 1.0);
    assert_eq!(prefs.moving_downscale, render::MOVING_DOWNSCALE);
    let ui = VolumeUi::default();
    assert!(ui.rect.is_none() && ui.pending_copy.is_none());
    assert!(!ui.pressed);
    assert!(ui.last_camera.is_none() && ui.moved_at.is_none());
}

fn run_on(document: u64, outcome: fast_tiff_viewer::plugins::Product) -> PluginRun {
    PluginRun {
        index: 0,
        run_on: document,
        params: fasttiff_plugin_api::Params::new(),
        name: "trace".into(),
        messages: Vec::new(),
        outcome: Ok(outcome),
        made_from: None,
    }
}

/// A plot measured on the last document, or a rectangle asked for on it,
/// does not come back over the next one when the run ends after that one
/// landed: the reset put the canvas tools away, and a late result would
/// reopen one on a picture it never read. What is not drawn on the picture
/// still lands, and so does a plot of the document still on show.
#[test]
fn a_canvas_result_from_the_last_document_is_not_shown_on_the_next() {
    use fast_tiff_viewer::plugins::Product;
    let plot = || Product::Plot(Box::new(fasttiff_plugin_api::Plot::new("trace")));
    let pick = || Product::SelectRegion("Draw a rectangle".into());
    let news = || Product::Message("done".into());

    // `finish_open` counts the document that landed after the run began.
    let (before, after) = (0, 1);
    assert!(run_on(before, plot()).outlived(after), "a plot");
    assert!(run_on(before, pick()).outlived(after), "a region pick");
    assert!(!run_on(before, news()).outlived(after), "a message");
    assert!(!run_on(after, plot()).outlived(after), "its own plot");
    assert!(!run_on(after, pick()).outlived(after), "its own pick");

    // A failed run says so whatever is on show.
    let mut failed = run_on(before, plot());
    failed.outcome = Err("it broke".into());
    assert!(!failed.outlived(after));
}

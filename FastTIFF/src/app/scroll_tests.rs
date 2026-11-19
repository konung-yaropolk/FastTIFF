//! Which device gets which gesture.
//!
//! The regression these exist for: routing trackpad swipes to panning must not
//! take frame-scrubbing away from the mouse wheel. That is easy to break and,
//! on a machine with no trackpad, impossible to notice.

use super::*;

fn wheel(unit: egui::MouseWheelUnit, delta: egui::Vec2) -> egui::Event {
    // `phase` is a trackpad nicety egui passes through from winit; routing does
    // not depend on it, so the tests use the "no information" value.
    egui::Event::MouseWheel {
        unit,
        delta,
        phase: egui::TouchPhase::Move,
        modifiers: egui::Modifiers::default(),
    }
}

fn wheel_with(
    unit: egui::MouseWheelUnit,
    delta: egui::Vec2,
    modifiers: egui::Modifiers,
) -> egui::Event {
    egui::Event::MouseWheel {
        unit,
        delta,
        phase: egui::TouchPhase::Move,
        modifiers,
    }
}

fn ctrl_wheel(unit: egui::MouseWheelUnit, delta: egui::Vec2) -> egui::Event {
    wheel_with(unit, delta, egui::Modifiers::CTRL)
}

/// Ctrl and Cmd as each platform reports them: Ctrl on Windows and Linux,
/// and in a browser on any platform (`ctrl` and `command`); a Mac's Ctrl
/// (`ctrl` alone); a Mac's Cmd, natively or in a browser (`mac_cmd` and
/// `command`).
const ZOOM_MODIFIERS: [(&str, egui::Modifiers); 3] = [
    (
        "Ctrl",
        egui::Modifiers {
            ctrl: true,
            command: true,
            ..egui::Modifiers::NONE
        },
    ),
    ("a Mac's Ctrl", egui::Modifiers::CTRL),
    (
        "a Mac's Cmd",
        egui::Modifiers {
            mac_cmd: true,
            command: true,
            ..egui::Modifiers::NONE
        },
    ),
];

#[test]
fn a_mouse_wheel_always_scrubs_frames() {
    // Zoomed in or not, a wheel steps frames — one notch, one frame.
    for pannable in [false, true] {
        let out = classify(
            &[wheel(egui::MouseWheelUnit::Line, egui::vec2(0.0, -1.0))],
            pannable,
        );
        assert_eq!(out.notches, -1.0, "pannable={pannable}");
        assert_eq!(
            out.swipe,
            egui::Vec2::ZERO,
            "a wheel must never pan (pannable={pannable})"
        );
    }
}

#[test]
fn a_paging_wheel_scrubs_too() {
    let out = classify(
        &[wheel(egui::MouseWheelUnit::Page, egui::vec2(0.0, 2.0))],
        true,
    );
    assert_eq!(out.notches, 2.0);
    assert_eq!(out.swipe, egui::Vec2::ZERO);
}

#[test]
fn a_trackpad_pans_when_there_is_something_to_pan() {
    let d = egui::vec2(-14.0, 33.0);
    let out = classify(&[wheel(egui::MouseWheelUnit::Point, d)], true);
    assert_eq!(
        out.swipe, d,
        "the swipe should pass through at full precision"
    );
    assert_eq!(out.notches, 0.0, "panning must not also scrub");
}

#[test]
fn a_trackpad_scrubs_when_the_image_already_fits() {
    // Panning an image with no overflow would do nothing, so the gesture is
    // better spent on frames than wasted.
    let out = classify(
        &[wheel(egui::MouseWheelUnit::Point, egui::vec2(0.0, -100.0))],
        false,
    );
    assert_eq!(out.swipe, egui::Vec2::ZERO);
    assert!(
        out.notches < 0.0,
        "should have scrubbed, got {}",
        out.notches
    );
}

#[test]
fn horizontal_trackpad_movement_pans_and_never_scrubs() {
    // Sideways is meaningless as a frame step; it is the axis a wheel does not
    // have and the reason panning by trackpad is worth doing.
    let out = classify(
        &[wheel(egui::MouseWheelUnit::Point, egui::vec2(40.0, 0.0))],
        true,
    );
    assert_eq!(out.swipe, egui::vec2(40.0, 0.0));
    assert_eq!(out.notches, 0.0);
    // And with nothing to pan it contributes no frame steps either.
    let out = classify(
        &[wheel(egui::MouseWheelUnit::Point, egui::vec2(40.0, 0.0))],
        false,
    );
    assert_eq!(out.notches, 0.0, "sideways scroll should not step frames");
}

#[test]
fn ctrl_or_cmd_scroll_is_left_for_the_zoom_handler() {
    for (held, modifiers) in ZOOM_MODIFIERS {
        for unit in [egui::MouseWheelUnit::Line, egui::MouseWheelUnit::Point] {
            let out = classify(&[wheel_with(unit, egui::vec2(0.0, -30.0), modifiers)], true);
            assert_eq!(
                out,
                Wheel::default(),
                "a scroll with {held} should be ignored here ({unit:?})"
            );
        }
    }
}

/// Every combination of modifiers: a scroll that egui turns into a zoom
/// (`zoom_delta`) is one `classify` leaves alone, and one egui leaves as a
/// scroll is one it counts. On a Mac, where Cmd is not `ctrl`, testing
/// `ctrl` alone counted Cmd+scroll, which egui zooms on, as frames too.
#[test]
fn what_egui_zooms_is_what_is_left_alone() {
    for bits in 0..32u8 {
        let modifiers = egui::Modifiers {
            alt: bits & 1 != 0,
            ctrl: bits & 2 != 0,
            shift: bits & 4 != 0,
            mac_cmd: bits & 8 != 0,
            command: bits & 16 != 0,
        };
        // A small trackpad delta, which egui applies on the frame it arrives
        // rather than smoothing it over the next few.
        let event = wheel_with(egui::MouseWheelUnit::Point, egui::vec2(0.0, 5.0), modifiers);
        let ctx = egui::Context::default();
        let mut zoom = 1.0;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![event.clone()],
                modifiers,
                ..Default::default()
            },
            |ui| zoom = ui.input(|i| i.zoom_delta()),
        );
        let zoomed = zoom != 1.0;
        assert_eq!(zoom_held(modifiers), zoomed, "{modifiers:?}: {zoom}");
        assert_eq!(
            classify(&[event], true) == Wheel::default(),
            zoomed,
            "{modifiers:?}"
        );
    }
}

#[test]
fn a_frames_events_accumulate() {
    // Several events can land in one frame; all of them count.
    let out = classify(
        &[
            wheel(egui::MouseWheelUnit::Point, egui::vec2(1.0, 2.0)),
            wheel(egui::MouseWheelUnit::Point, egui::vec2(3.0, 4.0)),
            ctrl_wheel(egui::MouseWheelUnit::Point, egui::vec2(99.0, 99.0)),
        ],
        true,
    );
    assert_eq!(out.swipe, egui::vec2(4.0, 6.0));
}

#[test]
fn unrelated_events_are_ignored() {
    let out = classify(
        &[
            egui::Event::PointerMoved(egui::pos2(1.0, 2.0)),
            egui::Event::Zoom(1.2),
        ],
        true,
    );
    assert_eq!(out, Wheel::default());
}

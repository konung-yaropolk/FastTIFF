use super::*;
use fast_tiff_lib::RoiPoint;

/// A 100 x 50 image, and a press reaching 3 image pixels.
const SIZE: (u32, u32) = (100, 50);
const R: f32 = 3.0;

fn tool() -> Session {
    Session::new(PointCounter::new())
}

fn click(x: f32, y: f32) -> Gesture {
    Gesture {
        at: Some((x, y)),
        pressed_at: Some((x, y)),
        clicked: true,
        ..Default::default()
    }
}

/// The frame a drag is first seen on: pressed at `from`, already at `to`.
fn drag_start(from: (f32, f32), to: (f32, f32)) -> Gesture {
    Gesture {
        at: Some(to),
        pressed_at: Some(from),
        drag_started: true,
        dragging: true,
        ..Default::default()
    }
}

#[test]
fn the_cursor_says_what_the_left_button_would_do() {
    let mut t = tool();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    assert_eq!(
        cursor(&t, Some((50.0, 10.0)), 0, R),
        egui::CursorIcon::Crosshair
    );
    assert_eq!(cursor(&t, None, 0, R), egui::CursorIcon::Crosshair);
    assert_eq!(cursor(&t, Some((11.0, 10.0)), 0, R), egui::CursorIcon::Grab);
    t.track(drag_start((10.0, 10.0), (20.0, 10.0)), 0, R, SIZE);
    assert_eq!(
        cursor(&t, Some((50.0, 40.0)), 0, R),
        egui::CursorIcon::Grabbing
    );
}

/// The picture at `zoom`, its top-left corner at the panel's.
fn at_zoom(zoom: f32) -> ImageXform {
    ImageXform {
        origin: egui::pos2(0.0, 0.0),
        zoom,
        size: SIZE,
    }
}

fn roi(points: usize) -> PointRoi {
    PointRoi {
        hyperstack_position: [0; 3],
        name: "G".into(),
        color: [0, 0, 0],
        points: vec![
            RoiPoint {
                x: 1.0,
                y: 1.0,
                position: 0
            };
            points
        ],
    }
}

#[test]
fn the_summary_counts_points_and_groups() {
    assert_eq!(summary(&[roi(1)]), "1 point in 1 group");
    assert_eq!(summary(&[roi(2), roi(3)]), "5 points in 2 groups");
    assert_eq!(summary(&[]), "0 points in 0 groups");
}

#[test]
fn the_roi_set_is_named_after_the_document() {
    use std::path::Path;
    assert_eq!(
        file_name(Path::new("C:/data/cells.tif")),
        "cells-points.zip"
    );
    assert_eq!(file_name(Path::new("/data/run 3.tiff")), "run 3-points.zip");
    assert_eq!(file_name(Path::new("")), "stack-points.zip");
}

/// What has the keyboard while Ctrl/Cmd+Z is pressed.
#[derive(Clone, Copy)]
enum Holder {
    Nothing,
    /// A group's name being edited.
    TextField,
    /// A button reached with Tab.
    Button,
}

/// Run one frame with Ctrl/Cmd+Z pressed, and report what `undo_hotkey`
/// decided.
fn undo_chord_taken(holder: Holder) -> bool {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        }],
        ..Default::default()
    };
    let mut taken = false;
    let _ = ctx.run_ui(input, |ui| {
        match holder {
            Holder::Nothing => {}
            Holder::TextField => {
                let mut name = String::from("Group 1");
                ui.add(egui::TextEdit::singleline(&mut name))
                    .request_focus();
            }
            Holder::Button => {
                ui.button("Add group").request_focus();
            }
        }
        taken = undo_hotkey(ui.ctx());
    });
    taken
}

#[test]
fn the_undo_chord_is_taken_when_nothing_has_focus() {
    assert!(undo_chord_taken(Holder::Nothing));
}

/// In a group's name, Ctrl+Z is the text field's own undo.
#[test]
fn a_focused_text_field_keeps_the_undo_chord() {
    assert!(!undo_chord_taken(Holder::TextField));
}

/// A button with the focus has no undo of its own: the chord is the points'.
#[test]
fn a_focused_button_does_not_keep_the_undo_chord() {
    assert!(undo_chord_taken(Holder::Button));
}

#[test]
fn no_chord_means_no_undo() {
    let ctx = egui::Context::default();
    let mut taken = true;
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        taken = undo_hotkey(ui.ctx())
    });
    assert!(!taken);
}

/// The hint's separators and the copy button's icon are glyphs the bundled
/// fonts have, not tofu boxes.
#[test]
fn the_window_text_has_glyphs() {
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let id = egui::FontId::proportional(12.0);
    for text in [HINT, super::super::ICON_COPY, "Save ImageJ overlay…"] {
        assert!(
            ctx.fonts_mut(|f| f.has_glyphs(&id, text)),
            "{text:?} has a character the font lacks"
        );
    }
}

/// The window draws, with its table and buttons, and asks for nothing when
/// nothing is clicked — on every frame of a few, so the fitted height and the
/// table's state settle without a panic.
#[test]
fn the_window_draws_and_asks_for_nothing_by_itself() {
    let ctx = egui::Context::default();
    let mut t = tool();
    t.counter.add_group();
    t.track(click(10.0, 10.0), 0, R, SIZE);
    for _ in 0..3 {
        let mut action = CounterAction::Close;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            action = counter_window(ui.ctx(), &mut t, 10, true);
        });
        assert_eq!(action, CounterAction::None);
    }
    assert_eq!(t.counter.groups.len(), 2);
    assert_eq!(t.counter.points.len(), 1);
}

/// Every point shown on the frame gets a mark, and its number is counted over
/// every frame: the second point of a group is "2" even when the first is on
/// another frame and not drawn.
#[test]
fn marks_are_drawn_for_the_frame_in_view_numbered_over_all_frames() {
    let ctx = egui::Context::default();
    let mut t = tool();
    t.counter.per_frame = true;
    t.track(click(10.0, 10.0), 0, R, SIZE);
    t.track(click(20.0, 10.0), 1, R, SIZE);
    let mut texts = Vec::new();
    let output = ctx.run_ui(egui::RawInput::default(), |ui| {
        let painter = ui.painter().clone();
        draw(&painter, &t.counter, 1, &at_zoom(2.0));
    });
    for clipped in &output.shapes {
        if let egui::Shape::Text(text) = &clipped.shape {
            texts.push(text.galley.text().to_string());
        }
    }
    // One number, outlined: drawn nine times, all of them "2".
    assert_eq!(texts.len(), 9, "{texts:?}");
    assert!(texts.iter().all(|s| s == "2"), "{texts:?}");
    let lines = output
        .shapes
        .iter()
        .filter(|c| matches!(c.shape, egui::Shape::LineSegment { .. }))
        .count();
    assert_eq!(lines, 4, "one cross: two arms, each over its outline");
}

// ------------------------------------------------ starting from the overlay

/// A 100 x 50 time series of `frames` frames, written with `rois` as its
/// ImageJ overlay and opened as the app opens a file.
fn stack_with_overlay(frames: usize, rois: Vec<PointRoi>) -> Document {
    use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(100, 50, SampleType::U8)
        .metadata(StackMetaWrite::new(1, 1).overlay(rois));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames {
        w.write_frame_u8(&vec![f as u8; 100 * 50]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    Document::from_bytes(bytes, "counted.tif".into(), false).unwrap()
}

fn named(name: &str, color: [u8; 3], points: &[(f32, f32, u32)]) -> PointRoi {
    PointRoi {
        hyperstack_position: [0; 3],
        name: name.into(),
        color,
        points: points
            .iter()
            .map(|&(x, y, position)| RoiPoint { x, y, position })
            .collect(),
    }
}

/// A file with points opens the counter on them: a group per ROI, each point
/// on its frame, every one shown until asked otherwise — and the status line
/// says how many came from where.
#[test]
fn a_session_starts_from_the_files_overlay() {
    let rois = vec![
        named("Neurons", [1, 2, 3], &[(10.5, 20.5, 1), (30.5, 5.5, 4)]),
        named("Glia", [4, 5, 6], &[(50.5, 40.5, 2)]),
    ];
    let mut stack = stack_with_overlay(5, rois.clone());
    assert_eq!(stack.overlay, rois, "the stack opened with its overlay");

    let t = Session::for_stack(&stack);
    let names: Vec<&str> = t.counter.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["Neurons", "Glia"]);
    let points: Vec<(usize, f32, f32, Option<usize>)> = t
        .counter
        .points
        .iter()
        .map(|p| (p.group, p.x, p.y, p.frame))
        .collect();
    assert_eq!(
        points,
        [
            (0, 10.5, 20.5, Some(0)),
            (0, 30.5, 5.5, Some(3)),
            (1, 50.5, 40.5, Some(1))
        ]
    );
    assert!(!t.counter.per_frame);
    assert!(t.drag.is_none());
    assert_eq!(
        loaded_message(&stack).as_deref(),
        Some("Loaded 3 points in 2 groups from the file's overlay.")
    );

    // What "Add into TIFF's overlay" put there is not the file's.
    stack.overlay = vec![named("Mine", [0; 3], &[(1.5, 1.5, 0)])];
    assert_eq!(
        loaded_message(&stack).as_deref(),
        Some("Loaded 1 point in 1 group from the overlay added to this image.")
    );
}

/// No overlay: the session is a new one, and nothing is said about it.
#[test]
fn without_an_overlay_a_session_starts_empty() {
    let stack = stack_with_overlay(5, Vec::new());
    assert!(stack.overlay.is_empty());
    let t = Session::for_stack(&stack);
    assert_eq!(t.counter.groups.len(), 1);
    assert_eq!(t.counter.groups[0].name, "Group 1");
    assert!(t.counter.points.is_empty());
    assert!(
        !t.counter.per_frame,
        "every point on every frame to begin with"
    );
    assert_eq!(loaded_message(&stack), None);
}

/// "Show only current frame's tags" starts off for loaded points as for new
/// ones, and the window leaves it as it is. With it on, a point the overlay
/// put on every slice is still on every frame, and one on a frame only there.
#[test]
fn loaded_points_start_shown_on_every_frame() {
    let rois = vec![
        named("All", [9; 3], &[(10.5, 10.5, 0)]),
        named("Some", [8; 3], &[(30.5, 10.5, 3)]),
    ];
    let stack = stack_with_overlay(5, rois);
    let mut t = Session::for_stack(&stack);
    assert!(!t.counter.per_frame);
    let ctx = egui::Context::default();
    for _ in 0..3 {
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            counter_window(ui.ctx(), &mut t, stack.display.dims().frames, true);
        });
    }
    assert!(!t.counter.per_frame, "the window leaves it as it is");
    assert_eq!(t.counter.points[0].frame, None);
    assert_eq!(t.counter.points[1].frame, Some(2));
    assert!(
        t.counter.hit(30.5, 10.5, 4, R).is_some(),
        "every point shown"
    );
    t.counter.per_frame = true;
    assert!(t.counter.hit(10.5, 10.5, 4, R).is_some(), "on every frame");
    assert!(t.counter.hit(30.5, 10.5, 4, R).is_none(), "only on its own");
    assert!(t.counter.hit(30.5, 10.5, 2, R).is_some());
}

/// Past `MAX_NUMBERED` marks on screen the numbers are left out and the
/// crosses still drawn; below it, every mark is numbered.
#[test]
fn a_crowded_frame_is_drawn_without_numbers() {
    let ctx = egui::Context::default();
    let mut t = tool();
    for i in 0..(MAX_NUMBERED + 1) {
        t.counter
            .add(1.0 + (i % 100) as f32, 1.0 + (i / 100) as f32 * 0.5, 0);
    }
    let shapes = |t: &Session| {
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let painter = ui.painter().clone();
            draw(&painter, &t.counter, 0, &at_zoom(1.0));
        });
        let texts = output
            .shapes
            .iter()
            .filter(|c| matches!(c.shape, egui::Shape::Text(_)))
            .count();
        let lines = output
            .shapes
            .iter()
            .filter(|c| matches!(c.shape, egui::Shape::LineSegment { .. }))
            .count();
        (texts, lines)
    };
    let (texts, lines) = shapes(&t);
    assert_eq!(texts, 0, "too many to number");
    assert_eq!(lines, 4 * (MAX_NUMBERED + 1), "every cross is still drawn");
    t.counter.points.pop();
    let (texts, _) = shapes(&t);
    assert_eq!(texts, 9 * MAX_NUMBERED, "each number and its outline");
}

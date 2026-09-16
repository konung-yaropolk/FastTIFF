use super::*;

/// The case that was broken: a record of `"key"\t"value"` lines, joined by `\n`.
#[test]
fn every_line_break_becomes_crlf() {
    assert_eq!(
        crlf("\"Name\"\t\"a.oir\"\n\"Date\"\t\"today\"\n"),
        "\"Name\"\t\"a.oir\"\r\n\"Date\"\t\"today\"\r\n"
    );
}

/// Text that is already in Windows form is not given a second `\r`.
#[test]
fn crlf_is_not_doubled() {
    assert_eq!(crlf("a\r\nb\r\n"), "a\r\nb\r\n");
    assert_eq!(crlf("a\r\nb\nc"), "a\r\nb\r\nc");
}

#[test]
fn text_without_line_breaks_is_unchanged() {
    assert_eq!(crlf(""), "");
    assert_eq!(crlf("12.5, 40"), "12.5, 40");
    // A lone carriage return is not a line break to rewrite.
    assert_eq!(crlf("a\rb"), "a\rb");
}

#[test]
fn a_break_at_either_end_is_kept() {
    assert_eq!(crlf("\nmiddle\n"), "\r\nmiddle\r\n");
}

/// Only copies are touched; everything else a frame asks of the platform goes
/// through as it was.
#[test]
fn only_copy_commands_are_rewritten() {
    let mut commands = vec![
        egui::OutputCommand::CopyText("one\ntwo".into()),
        egui::OutputCommand::OpenUrl(egui::OpenUrl::same_tab("https://example.com/a\nb")),
    ];
    to_windows_line_endings(&mut commands);
    match &commands[0] {
        egui::OutputCommand::CopyText(t) => assert_eq!(t, "one\r\ntwo"),
        other => panic!("the copy became {other:?}"),
    }
    match &commands[1] {
        egui::OutputCommand::OpenUrl(u) => assert_eq!(u.url, "https://example.com/a\nb"),
        other => panic!("the link became {other:?}"),
    }
}

// ------------------------------------------------- copies from a dialog window

/// Run a frame in which a pop-out dialog copies text, and return what each
/// window put on the clipboard: (the dialog's commands, the main window's).
///
/// A dialog is its own egui viewport, and the backend gives each viewport its
/// own pass and hands *that pass's* output to the clipboard. This stands in for
/// that: the renderer callback below is what eframe installs, and it runs the
/// dialog's UI in a pass of its own exactly as eframe's does.
fn frame_with_a_dialog(copies: bool) -> (Vec<egui::OutputCommand>, Vec<egui::OutputCommand>) {
    use std::sync::{Arc, Mutex};

    let from_dialog: Arc<Mutex<Vec<egui::OutputCommand>>> = Arc::default();
    let sink = Arc::clone(&from_dialog);
    egui::Context::set_immediate_viewport_renderer(move |ctx, mut viewport| {
        let mut input = egui::RawInput {
            viewport_id: viewport.ids.this,
            ..Default::default()
        };
        // The backend describes the window it is about to draw into; egui
        // refuses to run a pass for a viewport it has not been told about.
        input.viewports.clear();
        input
            .viewports
            .insert(viewport.ids.this, egui::ViewportInfo::default());
        let out = ctx.run_ui(input, |ui| (viewport.viewport_ui_cb)(ui));
        sink.lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(out.platform_output.commands);
    });

    let ctx = egui::Context::default();
    // Real windows rather than windows drawn inside the main one, which is what
    // the desktop backend selects and what makes a dialog its own viewport.
    ctx.set_embed_viewports(false);
    let out = ctx.run_ui(egui::RawInput::default(), |ui| {
        let ctx = ui.ctx().clone();
        // Through the real dialog rather than a stand-in viewport, so this
        // covers the call site too: take the conversion out of `dialog::show`
        // and this test fails.
        let mut open = true;
        let spec = crate::app::dialog::Dialog {
            id: "metadata",
            title: "File metadata",
            size: egui::vec2(360.0, 600.0),
            scroll: false,
            resizable: true,
        };
        crate::app::dialog::show(&ctx, spec, &mut open, |ui| {
            if copies {
                ui.ctx().copy_text("one\ntwo".to_string());
            }
        });
        // What the main window does at the end of its own frame.
        to_system_line_endings(&ctx);
    });

    let dialog = std::mem::take(&mut *from_dialog.lock().unwrap_or_else(|e| e.into_inner()));
    (dialog, out.platform_output.commands)
}

fn copied(commands: &[egui::OutputCommand]) -> Option<String> {
    commands.iter().find_map(|c| match c {
        egui::OutputCommand::CopyText(t) => Some(t.clone()),
        _ => None,
    })
}

/// A copy made in a pop-out dialog — the metadata window is one — is converted
/// like any other.
///
/// The bug this pins: the conversion used to run only on the main window's
/// output, so text copied from the metadata window reached the clipboard with
/// bare `\n`. Windows 11's Notepad accepts that and Windows 7's does not, which
/// is a difference no test on one machine would have shown.
#[test]
fn a_copy_from_a_dialog_window_is_converted() {
    let (dialog, root) = frame_with_a_dialog(true);
    assert_eq!(
        copied(&dialog).as_deref(),
        Some(if cfg!(windows) {
            "one\r\ntwo"
        } else {
            "one\ntwo"
        }),
        "the dialog's copy was not converted"
    );
    assert!(
        copied(&root).is_none(),
        "the dialog's copy also reached the main window's output, which would \
         mean this test is not exercising two viewports at all"
    );
}

/// The fixture copies nothing unless asked, so the test above really did
/// observe the copy made inside the dialog.
#[test]
fn a_dialog_that_copies_nothing_puts_nothing_on_the_clipboard() {
    let (dialog, root) = frame_with_a_dialog(false);
    assert_eq!(copied(&dialog), None);
    assert_eq!(copied(&root), None);
}

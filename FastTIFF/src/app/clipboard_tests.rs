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

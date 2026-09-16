//! Text leaving the app through the clipboard, in the line endings the system
//! expects.

/// Put this pass's copies into the system's line endings.
///
/// Called once per *viewport* per frame, not once per frame. Each egui viewport
/// — and every pop-out dialog is one, see `dialog::show` — carries its own
/// platform output, and the backend hands each one to the clipboard separately.
/// Converting only the root window's output left every dialog copying what it
/// always did, which is how the metadata window went on pasting as a single
/// line into Windows 7's Notepad while the main window was fine.
pub(super) fn to_system_line_endings(ctx: &egui::Context) {
    #[cfg(windows)]
    ctx.output_mut(|o| to_windows_line_endings(&mut o.commands));
    // Everywhere else `\n` is already the convention.
    #[cfg(not(windows))]
    let _ = ctx;
}

/// Rewrite every copy in `commands` to use `\r\n` between lines.
///
/// egui copies text exactly as it holds it — `\n` between lines — and the
/// clipboard crate under it hands that to the system unchanged. On Windows the
/// convention for clipboard text is `\r\n`, and the standard text controls a
/// great many programs paste into show a bare `\n` as nothing at all. So a
/// many-line acquisition record copied out of the metadata window landed in
/// Notepad as a single line.
///
/// Done to the frame's output rather than to any one widget, so every copy in
/// the app is covered — the metadata window, the coordinate readout, whatever
/// is added next — without each having to remember it. Only called on Windows:
/// everywhere else `\n` is the convention already.
#[cfg(any(windows, test))]
pub(super) fn to_windows_line_endings(commands: &mut [egui::OutputCommand]) {
    for command in commands {
        if let egui::OutputCommand::CopyText(text) = command {
            *text = crlf(text);
        }
    }
}

/// `text` with every line break as `\r\n`. A break that already is one is left
/// alone, so nothing is doubled.
#[cfg(any(windows, test))]
fn crlf(text: &str) -> String {
    if !text.contains('\n') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + text.matches('\n').count());
    let mut previous = '\0';
    for c in text.chars() {
        if c == '\n' && previous != '\r' {
            out.push('\r');
        }
        out.push(c);
        previous = c;
    }
    out
}

#[cfg(test)]
#[path = "clipboard_tests.rs"]
mod tests;

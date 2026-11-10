//! The pop-up windows: 3D render settings, the file-metadata viewer, and the
//! channel histogram, one child module each.
//!
//! Each is written once, against a plain [`egui::Ui`], and handed to
//! [`super::dialog`] to be put wherever the host can put it: its own top-level
//! window on the desktop, a window inside the canvas in the browser. Nothing
//! here needs to know which it got.
//!
//! A window is an [`AppWindow`]: what it is called, how to draw it, and
//! whether it stays up when another file is opened. [`Windows`] holds them,
//! with whether each is open, so a new one is a file implementing the trait
//! and a line in [`Windows::new`]; the toolbar or the bottom bar then asks for
//! it by its [`WindowId`] with [`UiCommand::ToggleWindow`].
//!
//! [`UiCommand::ToggleWindow`]: super::command::UiCommand::ToggleWindow

use super::*;

mod histogram;
mod metadata;
mod render_settings;

/// Which pop-up a toggle means. The name is also the id its dialog is shown
/// under, which is what egui remembers its place and size by.
///
/// Each window's module declares its own, as an associated constant, so adding
/// one touches nothing here but the registration in [`Windows::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WindowId(&'static str);

/// What a pop-up may read and edit while it is drawn.
pub(super) struct Host<'a> {
    /// Everything that isn't GUI: the document, its display, the 3D view.
    pub(super) core: &'a mut Viewer,
    /// The 3D preferences the render-settings window edits.
    pub(super) prefs: &'a mut VolumePrefs,
}

/// One pop-up window.
pub(super) trait AppWindow {
    /// Which one this is.
    fn id(&self) -> WindowId;

    /// Draw it. Called only while it is open, every frame; `open` is cleared
    /// when it closes — by its own close button, or because there is nothing
    /// left for it to show.
    fn show(&mut self, ctx: &egui::Context, open: &mut bool, host: &mut Host<'_>);

    /// Whether it stays up when another document lands in place of the one it
    /// was opened over. `core` already holds the new one, and `web` is whether
    /// this is the browser build, where a pop-up is a window inside the canvas
    /// rather than one of its own.
    fn outlives_document(&self, core: &Viewer, web: bool) -> bool;
}

/// Every pop-up, and whether it is open.
pub(super) struct Windows {
    /// In the order they are drawn, which is the order they were registered.
    windows: Vec<(bool, Box<dyn AppWindow>)>,
}

impl Windows {
    /// The app's pop-ups, all closed.
    pub(super) fn new() -> Self {
        Self::of(vec![
            Box::new(metadata::MetadataWindow),
            Box::new(histogram::HistogramWindow::default()),
            Box::new(render_settings::RenderSettingsWindow),
        ])
    }

    /// These windows, all closed, drawn in this order.
    fn of(windows: Vec<Box<dyn AppWindow>>) -> Self {
        Windows {
            windows: windows.into_iter().map(|w| (false, w)).collect(),
        }
    }

    pub(super) fn is_open(&self, id: WindowId) -> bool {
        self.windows.iter().any(|(open, w)| *open && w.id() == id)
    }

    /// Open it if it is closed, close it if it is open.
    pub(super) fn toggle(&mut self, id: WindowId) {
        for (open, _) in self.windows.iter_mut().filter(|(_, w)| w.id() == id) {
            *open = !*open;
        }
    }

    /// Draw every open window.
    pub(super) fn show(&mut self, ctx: &egui::Context, host: &mut Host<'_>) {
        for (open, window) in self.windows.iter_mut().filter(|(open, _)| *open) {
            window.show(ctx, open, host);
        }
    }

    /// Close whatever does not outlive the document it was opened over, now
    /// that `core` holds another, on the desktop or (`web`) in a browser.
    pub(super) fn reset(&mut self, core: &Viewer, web: bool) {
        for (open, window) in &mut self.windows {
            *open = *open && window.outlives_document(core, web);
        }
    }
}

#[cfg(test)]
#[path = "windows_tests.rs"]
mod tests;

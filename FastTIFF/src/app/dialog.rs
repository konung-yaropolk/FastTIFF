//! Where a pop-up is drawn: its own top-level window on the desktop, a window
//! inside the canvas in the browser.
//!
//! The three pop-ups — 3D render settings, file metadata, the histogram — are
//! all *companions* to the image rather than steps in front of it. You set a
//! contrast window while watching the frame change, or read the metadata while
//! scrubbing. An in-canvas window is the wrong shape for that: it sits on top
//! of the very image it describes, it cannot be dragged to a second monitor,
//! and it competes for the same pixels the picture needs.
//!
//! On the desktop that is fixable, because the platform has real windows, and
//! egui can open one per [viewport](egui::ViewportId). In a browser it is not:
//! the app is a single canvas element, and a popup would be a separate
//! document the page cannot lay out or style. So the two hosts genuinely differ
//! and this module is the seam — every pop-up is written once, against a
//! [`egui::Ui`], and drawn wherever the host can put it.

#[cfg(target_arch = "wasm32")]
use super::scale::unscaled;

/// How one pop-up wants to be presented.
pub(super) struct Dialog<'a> {
    /// Stable identity for the window. On the desktop it seeds the
    /// [`egui::ViewportId`], so it must be unique among dialogs and must not
    /// change between frames — a new id is a new window.
    pub id: &'a str,
    /// Title bar text. Native windows get this from the OS, so it is the only
    /// label a user has for a window that has been dragged away from the app.
    pub title: &'a str,
    /// Size the window opens at.
    pub size: egui::Vec2,
    /// Wrap the body in a vertical scroll area.
    ///
    /// Not simply always on: a body that sizes itself from
    /// [`egui::Ui::available_height`] — the histogram, whose plot claims
    /// whatever the controls leave — measures infinity inside a scroll area and
    /// collapses. Those bodies manage their own overflow.
    pub scroll: bool,
    /// Whether the user can resize it. Honoured only in the browser: a native
    /// window is always resizable, and refusing to be would leave a user whose
    /// screen is shorter than the content with no way to reach the rest of it.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub resizable: bool,
}

/// Draw `body` as `dialog`, closing it when the user dismisses the window.
///
/// `open` is cleared when the window is closed from its own chrome (the OS
/// close button, or the `x` on the egui window). The caller still owns the flag
/// otherwise — a toolbar button toggling it off simply stops calling this, and
/// the window goes away with it.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn show(
    ctx: &egui::Context,
    dialog: Dialog<'_>,
    open: &mut bool,
    mut body: impl FnMut(&mut egui::Ui),
) {
    // egui diffs this builder against the previous frame's and issues commands
    // only for what changed, so a `size` that stays the same leaves a window the
    // user has since moved or resized alone. That makes `size` a live property
    // here rather than the opening size it is on the web: a caller whose value
    // does change — the histogram, which sizes itself to the channel count — is
    // choosing to resize the window when the open file changes shape. For that
    // one it is the right call, since a window kept at the old file's height
    // would clip the controls of a file with more channels.
    let builder = egui::ViewportBuilder::default()
        .with_title(dialog.title)
        .with_inner_size(dialog.size);

    let mut dismissed = false;
    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of(dialog.id),
        builder,
        |ui, _class| {
            // `_class` says whether we really got a window or egui fell back to
            // embedding one — nothing here needs to care, since either way the
            // body just fills what it is given.
            //
            // The callback hands us a bare root `Ui` with no background or
            // margin, so the panel is what makes the window look like a window.
            egui::CentralPanel::default().show_inside(ui, |ui| {
                if dialog.scroll {
                    egui::ScrollArea::vertical().show(ui, &mut body);
                } else {
                    body(ui);
                }
            });
            if ui.ctx().input(|i| i.viewport().close_requested()) {
                dismissed = true;
            }
        },
    );
    if dismissed {
        *open = false;
    }
}

/// The browser has no second window to open, so the pop-up stays an egui window
/// floating over the canvas — drawn at native size even though the chrome
/// around it is enlarged (see [`unscaled`]).
#[cfg(target_arch = "wasm32")]
pub(super) fn show(
    ctx: &egui::Context,
    dialog: Dialog<'_>,
    open: &mut bool,
    mut body: impl FnMut(&mut egui::Ui),
) {
    unscaled(ctx, |ctx| {
        let win = egui::Window::new(dialog.title)
            .id(egui::Id::new(dialog.id))
            .open(open)
            .resizable(dialog.resizable)
            .vscroll(dialog.scroll);
        // Only the height is conditional. A scrolling body has no height of its
        // own to ask for, so the window hugs its content and stops growing when
        // it runs out of canvas — which matters far more here than on a desktop,
        // since a browser viewport cannot be made taller to see the rest. A body
        // that instead divides up the height it is given — the histogram, whose
        // plot claims whatever the controls leave — needs an explicit one for
        // there to be anything to divide.
        let win = if dialog.scroll {
            win.default_width(dialog.size.x)
        } else {
            win.default_size(dialog.size)
        };
        win.show(ctx, |ui| body(ui));
    });
}

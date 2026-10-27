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
    /// Make the window exactly as tall as its body, following it as it grows
    /// and shrinks — the 3D settings, whose rows change with the render mode.
    /// Up to most of the screen's height, past which the body scrolls; needs
    /// `scroll`. `size` is then only the width, and the height the window
    /// opens at the first time.
    ///
    /// Neither host does this by itself. A native window keeps the size it was
    /// opened at, whatever is in it, and an egui window grows to fit its
    /// content but never shrinks back: open the settings in Volume mode, switch
    /// to Max intensity, and the rows that went leave a blank space behind.
    pub fit_height: bool,
    /// Whether the user can resize it. Honoured only in the browser: a native
    /// window is always resizable, and refusing to be would leave a user whose
    /// screen is shorter than the content with no way to reach the rest of it.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub resizable: bool,
    /// Whether it floats above the main window.
    ///
    /// Not on by default, and not right for every pop-up. It earns its place
    /// where the controls act on the picture *continuously* — the 3-D settings,
    /// whose every slider changes the render while you drag it. There the main
    /// window has to be visible and the pop-up has to be reachable at the same
    /// time, and a window that sinks behind the thing it is adjusting turns
    /// each change into click-raise-drag-click.
    ///
    /// The metadata viewer and the histogram are read *about* the picture
    /// rather than used to change it, so they stay ordinary windows a user can
    /// put behind the app and forget.
    pub always_on_top: bool,
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
    //
    // A window that fits its height opens at the height it last fitted, so it
    // does not open at `size` and jump to its content a frame later. That is
    // the only height the builder is given: it has to stay the same while the
    // window is open, or the diff above would resize it to that height, and to
    // `size`'s width with it.
    let fit_key = egui::Id::new(dialog.id).with("fit_height");
    let mut fit = dialog
        .fit_height
        .then(|| FitHeight::load(ctx, fit_key, dialog.size.y));
    let height = fit.map_or(dialog.size.y, |f| f.opened_at);
    let mut builder = egui::ViewportBuilder::default()
        .with_title(dialog.title)
        .with_inner_size(egui::vec2(dialog.size.x, height));
    if dialog.always_on_top {
        // The window manager's own always-on-top, so it holds against every
        // other application too — which is the point when the other
        // application is a reference image or a paper.
        builder = builder.with_always_on_top();
    }

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
            let window = ui
                .ctx()
                .input(|i| i.viewport().inner_rect)
                .unwrap_or_else(|| ui.max_rect());
            let mut wanted = None;
            egui::CentralPanel::default().show_inside(ui, |ui| {
                // What the window adds around the body: the panel's margins.
                let chrome = window.height() - ui.max_rect().height();
                if dialog.scroll {
                    let shown = egui::ScrollArea::vertical().show(ui, &mut body);
                    wanted = Some(shown.content_size.y + chrome);
                } else {
                    body(ui);
                }
            });
            if let (Some(fit), Some(wanted)) = (fit.as_mut(), wanted) {
                fit.apply(ui.ctx(), window, wanted);
            }
            if ui.ctx().input(|i| i.viewport().close_requested()) {
                dismissed = true;
            }
            // This window is its own viewport, with its own platform output
            // that the backend hands to the clipboard by itself, so its copies
            // are converted here rather than by the main window.
            super::clipboard::to_system_line_endings(ui.ctx());
        },
    );
    if let Some(fit) = fit {
        ctx.data_mut(|d| d.insert_temp(fit_key, fit));
    }
    if dismissed {
        *open = false;
    }
}

/// What a native [`Dialog::fit_height`] window keeps between frames.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct FitHeight {
    /// The height its content last asked for, in points.
    fitted: f32,
    /// The height this opening of the window started at.
    opened_at: f32,
    /// The main window's frame it was last drawn in. A window not drawn in
    /// the frame before has just been opened.
    frame: u64,
    /// The height last asked of the window manager: at opening, the height
    /// the window was opened at.
    asked: f32,
}

#[cfg(not(target_arch = "wasm32"))]
impl FitHeight {
    fn load(ctx: &egui::Context, key: egui::Id, first: f32) -> Self {
        let frame = ctx.cumulative_frame_nr();
        let mut fit = ctx.data(|d| d.get_temp::<Self>(key)).unwrap_or(Self {
            fitted: first,
            opened_at: first,
            frame,
            asked: first,
        });
        if frame > fit.frame + 1 {
            fit.opened_at = fit.fitted;
            fit.asked = fit.opened_at;
        }
        fit.frame = frame;
        fit
    }

    /// Size `window` to `wanted`, its body's height plus what goes around it.
    ///
    /// Asked once per change of height, not every frame the window is some
    /// other height. A window manager may decline — a tiling one, or any for a
    /// maximized window, and a minimized window has no height to compare — and
    /// asking again each frame would then never stop: every request schedules
    /// another frame. So a window the user has made another height stays so
    /// until its content changes.
    fn apply(&mut self, ctx: &egui::Context, window: egui::Rect, wanted: f32) {
        // Most of the screen at the most, leaving room for the title bar and
        // the taskbar; past that the body scrolls.
        let most = ctx
            .input(|i| i.viewport().monitor_size)
            .map_or(f32::INFINITY, |m| m.y * 0.85);
        let height = wanted.min(most).max(64.0).ceil();
        self.fitted = height;
        let held = ctx.input(|i| {
            let v = i.viewport();
            [v.minimized, v.maximized, v.fullscreen].contains(&Some(true))
        });
        if !held && (height - self.asked).abs() > 0.5 {
            self.asked = height;
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                window.width(),
                height,
            )));
        }
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
        // A window that fits its height scrolls its own body, to measure it,
        // and is held to that height, which is how it shrinks: egui only ever
        // grows a window to its content. The measurement is a frame old, so a
        // body that grows shows a scroll bar for that one frame.
        let fit = dialog.fit_height && dialog.scroll;
        let fit_key = egui::Id::new(dialog.id).with("fit_height");
        let fitted = ctx.data(|d| d.get_temp::<f32>(fit_key)).filter(|_| fit);
        let mut win = egui::Window::new(dialog.title)
            .id(egui::Id::new(dialog.id))
            .open(open)
            .resizable(dialog.resizable)
            .vscroll(dialog.scroll && !fit);
        if let Some(height) = fitted {
            // At most the canvas: the window clamps its own maximum to that.
            win = win.min_height(height).max_height(height);
        }
        if dialog.always_on_top {
            // In the browser there is no window manager to ask, only egui's own
            // layer order — so "on top" means the foreground layer, above the
            // canvas and above the other pop-ups.
            win = win.order(egui::Order::Foreground);
        }
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
        win.show(ctx, |ui| {
            if !fit {
                body(ui);
                return;
            }
            let shown = egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, &mut body);
            let height = shown.content_size.y.ceil();
            if fitted != Some(height) {
                ctx.data_mut(|d| d.insert_temp(fit_key, height));
                ctx.request_repaint();
            }
        });
    });
}

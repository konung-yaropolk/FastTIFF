//! The point counter: click on the picture to count things, in named,
//! coloured groups.
//!
//! An app-native tool listed in the Plugins menu, not a plugin. A plugin is
//! constructed afresh for every run and keeps nothing between them, and what
//! crosses the plugin boundary has no point shape — while this tool is a
//! session: a list of points that grows click by click, an undo history, and
//! marks drawn over the picture for as long as the window is open.
//!
//! Its code is with the other built-ins, in
//! [`fast_tiff_viewer::plugins::builtin::point_counter`]: the model — groups,
//! points, undo, the export as ImageJ point ROIs — and the gestures on the
//! picture ([`Session::track`]), tested there without a window. This module is
//! the egui side of it, which that crate has no toolkit for:
//!
//! * the window: the table of groups and counts, and the buttons;
//! * the pointer turned into a [`Gesture`], by the canvas, and the cursor;
//! * the marks drawn over the picture ([`draw`]);
//! * where the two exports go: a RoiSet `.zip` the ROI Manager opens, or the
//!   open document's overlay ([`Document::overlay`]), which the next save writes
//!   into the TIFF.
//!
//! # Gestures
//!
//! While the window is open the primary button on the picture is the
//! counter's, as it is a region tool's while one is armed: a click adds a
//! point to the selected group on the frame in view, a press on a point drags
//! it, and Alt+click on a point deletes it. Every other button still pans, the
//! wheel still scrubs frames, and Ctrl+wheel still zooms. Ctrl/Cmd+Z takes
//! back the last add, move or delete, from the main window or the counter's
//! own.
//!
//! # Opening
//!
//! A session starts from the document's overlay when it has one — the point
//! ROIs the file was opened with, or those added to it earlier — so points
//! counted in ImageJ, or counted here and saved, come back as groups to go on
//! counting in. Without one it starts with a single empty group.
//!
//! # Closing
//!
//! Closing the window ends the session: the points go, and the marks with
//! them. What was added to the TIFF's overlay is the document's by then, and
//! stays until the file is saved or another one is opened.
//!
//! [`Document::overlay`]: fast_tiff_viewer::Document::overlay

use egui::{Color32, Pos2, RichText, Stroke, Vec2};
use fast_tiff_lib::PointRoi;
use fast_tiff_viewer::plugins::builtin::point_counter::{CountedPoint, PointCounter};
pub(super) use fast_tiff_viewer::plugins::builtin::point_counter::{Gesture, Session};
use fast_tiff_viewer::Document;

use super::dialog;

/// The tool's name: its menu entry and its window's title.
pub(super) const NAME: &str = "Point counter";

/// The window's id. The desktop's native window is keyed on it, which is also
/// how the menu raises a window that is already open.
const WINDOW_ID: &str = "point_counter";

/// How near a press has to land to a point to take hold of it, in screen
/// points. On screen rather than in the image so that a point is as easy to
/// grab at 3% as at 3200%: divided by the zoom before it reaches
/// [`PointCounter::hit`].
pub(super) const HIT_RADIUS: f32 = 6.0;

/// The mark: a cross with arms this long either side of the point, in screen
/// points, drawn this wide.
const ARM: f32 = 5.0;
const STROKE: f32 = 1.5;

/// The outline under every mark and number.
///
/// Dark, not [`contrasting_ink`](super::minimap::contrasting_ink)'s choice:
/// that one is chosen against the interface's background, and these sit on
/// the picture, in colours of their own. A dark edge is what keeps a
/// saturated colour readable on a bright cell and on a black background alike
/// — and Set1's yellow, outlined in white, would vanish.
const HALO: Color32 = Color32::from_black_alpha(200);
const HALO_STROKE: f32 = STROKE + 2.0;

/// The size of the number beside each mark.
const LABEL_SIZE: f32 = 11.0;

/// Past this many marks on screen their numbers are left out, and only the
/// crosses drawn. A number costs nine passes over its text (eight for the
/// outline), and a file's overlay can put tens of thousands of points on one
/// frame; that many numbers would be unreadable on top of each other anyway.
const MAX_NUMBERED: usize = 2_000;

/// Where the window keeps the row to bring into view on its next frame: a
/// group just added, which with more than a dozen is below the table's fold.
const SCROLL_TO: &str = "point_counter_scroll_to";

/// What the window says the gestures are, under the buttons.
pub(super) const HINT: &str = "Click to add · drag to move · Alt+click to delete · Ctrl+Z to undo";

/// The cursor over the picture: a closed hand while dragging, an open one
/// over a point that can be taken hold of, cross-hairs for placing one.
pub(super) fn cursor(
    session: &Session,
    hover: Option<(f32, f32)>,
    frame: usize,
    radius: f32,
) -> egui::CursorIcon {
    if session.drag.is_some() {
        egui::CursorIcon::Grabbing
    } else if hover.is_some_and(|(x, y)| session.counter.hit(x, y, frame, radius).is_some()) {
        egui::CursorIcon::Grab
    } else {
        egui::CursorIcon::Crosshair
    }
}

/// The image position under a point on screen, continuous: pixel `(i, j)`
/// covers `[i, i+1) x [j, j+1)`, so the middle of it is `(i + 0.5, j + 0.5)`.
/// Not floored the way [`plot::pixel_at`](super::plot::pixel_at) is — a point
/// is where it was clicked, and the ImageJ export puts it on the same pixel.
pub(super) fn image_at(pos: Pos2, origin: Pos2, zoom: f32) -> (f32, f32) {
    ((pos.x - origin.x) / zoom, (pos.y - origin.y) / zoom)
}

/// The screen position of an image position: [`image_at`] the other way.
pub(super) fn screen_at((x, y): (f32, f32), origin: Pos2, zoom: f32) -> Pos2 {
    origin + Vec2::new(x * zoom, y * zoom)
}

/// Whether Ctrl/Cmd+Z was pressed this frame, in the viewport `ctx` is
/// drawing.
///
/// Skipped while text has the keyboard, for the reason
/// [`copy_view::hotkey`](super::copy_view::hotkey) is: in a group's name the
/// chord is the text field's own undo, and taking back a point instead would
/// read as the field being broken. Any other widget with the focus — a button
/// just reached with Tab — has no undo of its own, and leaves it to the points.
pub(super) fn undo_hotkey(ctx: &egui::Context) -> bool {
    if super::copy_view::text_has_the_keyboard(ctx) {
        return false;
    }
    ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z))
}

/// Draw the points shown on `frame` over the picture: a cross in the group's
/// colour, and the point's number in its group beside it, both outlined so
/// they read on any part of the image.
pub(super) fn draw(
    painter: &egui::Painter,
    counter: &PointCounter,
    frame: usize,
    origin: Pos2,
    zoom: f32,
) {
    // Room for a mark and its number whose point is just off the panel.
    let room = painter.clip_rect().expand(ARM + 4.0 * LABEL_SIZE);
    let font = egui::FontId::proportional(LABEL_SIZE);
    let halo = Stroke::new(HALO_STROKE, HALO);
    let on_screen = |i: usize, p: &CountedPoint| {
        counter.shown_on(i, frame) && room.contains(screen_at((p.x, p.y), origin, zoom))
    };
    let numbered = counter
        .points
        .iter()
        .enumerate()
        .filter(|&(i, p)| on_screen(i, p))
        .take(MAX_NUMBERED + 1)
        .count()
        <= MAX_NUMBERED;
    // A point's number counts every earlier point of its group, on every
    // frame, so the count runs over all of them, drawn or not — in one pass
    // rather than asking `number_in_group` per point, which is quadratic.
    let mut numbers = vec![0usize; counter.groups.len()];
    for (i, p) in counter.points.iter().enumerate() {
        let Some(n) = numbers.get_mut(p.group) else {
            continue;
        };
        *n += 1;
        let number = *n;
        if !on_screen(i, p) {
            continue;
        }
        let c = screen_at((p.x, p.y), origin, zoom);
        let [r, g, b] = counter.groups[p.group].color;
        let color = Color32::from_rgb(r, g, b);
        // Both halos first, a point longer at each end, so neither arm's
        // outline is drawn over the other arm.
        let (x, y) = (Vec2::new(ARM, 0.0), Vec2::new(0.0, ARM));
        let (hx, hy) = (Vec2::new(ARM + 1.0, 0.0), Vec2::new(0.0, ARM + 1.0));
        painter.line_segment([c - hx, c + hx], halo);
        painter.line_segment([c - hy, c + hy], halo);
        painter.line_segment([c - x, c + x], Stroke::new(STROKE, color));
        painter.line_segment([c - y, c + y], Stroke::new(STROKE, color));
        if !numbered {
            continue;
        }

        // The number up and to the right, clear of the cross, outlined by
        // drawing it dark one point off in each direction first.
        let galley = painter.layout_no_wrap(number.to_string(), font.clone(), color);
        let at = c + Vec2::new(ARM + 1.0, -(ARM + 1.0) - galley.size().y * 0.6);
        for (dx, dy) in [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ] {
            painter.galley_with_override_text_color(at + Vec2::new(dx, dy), galley.clone(), HALO);
        }
        painter.galley(at, galley, color);
    }
}

/// What the status line says when a session on `stack` starts from its
/// overlay — how much, and from where — or `None` when it starts empty.
///
/// "The file's" while the overlay is the one the file was opened with; once
/// "Add into TIFF's overlay" has replaced it, it is what was added, which a
/// user who closed the window and opened it again should not take for the
/// file's.
pub(super) fn loaded_message(stack: &Document) -> Option<String> {
    if stack.overlay.is_empty() {
        return None;
    }
    let from = if stack.overlay == stack.tiff.meta.overlay {
        "the file's overlay"
    } else {
        "the overlay added to this image"
    };
    Some(format!("Loaded {} from {from}.", summary(&stack.overlay)))
}

/// `"12 points in 3 groups"`: what a set of ROIs holds, for the status line.
pub(super) fn summary(rois: &[PointRoi]) -> String {
    let points: usize = rois.iter().map(|r| r.points.len()).sum();
    let groups = rois.len();
    let s = |n: usize| if n == 1 { "" } else { "s" };
    format!("{points} point{} in {groups} group{}", s(points), s(groups))
}

/// The name a ROI set is offered under: the document's own, with
/// `-points.zip` — beside the image, and clearly about it.
pub(super) fn file_name(document: &std::path::Path) -> String {
    let stem = document
        .file_stem()
        .map(|s| s.to_string_lossy().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "stack".to_string());
    format!("{stem}-points.zip")
}

/// What the window asks the app to do next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CounterAction {
    None,
    /// The window was closed.
    Close,
    /// "Save ImageJ overlay…": write the points as a RoiSet `.zip`.
    SaveRois,
    /// "Add into TIFF's overlay": attach them to the open document.
    AddToTiff,
}

/// The counter's window: the table, the buttons, and the hint.
///
/// `frames` is the open stack's frame count; with one, "Show only current
/// frame's tags" has no other frame to hide points on and is greyed.
/// `imagej_reads_overlay` is whether ImageJ will read an overlay back from a
/// save of the stack ([`overlay_readable_by_imagej`](fast_tiff_viewer::save::overlay_readable_by_imagej)),
/// which the button's description has to tell the truth about. The copy is done here,
/// in the viewport the button was clicked in, whose clipboard output the
/// dialog converts to the system's line endings.
pub(super) fn counter_window(
    ctx: &egui::Context,
    tool: &mut Session,
    frames: usize,
    imagej_reads_overlay: bool,
) -> CounterAction {
    let mut action = CounterAction::None;
    let mut open = true;
    let spec = dialog::Dialog {
        id: WINDOW_ID,
        title: NAME,
        size: egui::vec2(320.0, 260.0),
        // As tall as the table, which grows a row per group, and scrolling
        // only past most of the screen.
        scroll: true,
        fit_height: true,
        resizable: true,
        // A companion to the picture, like the plot: the points are clicked
        // onto the image, so it is the image that has to stay reachable.
        always_on_top: false,
    };
    dialog::show(ctx, spec, &mut open, |ui| {
        // Nothing here is text to select: a click on a count belongs to its
        // row, which it selects, and a label would take the click instead.
        ui.style_mut().interaction.selectable_labels = false;
        // Ctrl/Cmd+Z while this window has the keyboard. On the desktop it is
        // a window of its own, and the main window never hears the key.
        if undo_hotkey(ui.ctx()) {
            tool.undo();
        }
        let counter = &mut tool.counter;
        group_table(ui, counter);
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Add group")
                .on_hover_text(
                    "A new group, in the next colour. New points go to the selected row.",
                )
                .clicked()
            {
                let index = counter.add_group();
                // Into view on the next frame: the table is drawn above.
                ui.data_mut(|d| d.insert_temp(egui::Id::new(SCROLL_TO), index));
            }
            ui.add_enabled(
                frames > 1,
                egui::Checkbox::new(&mut counter.per_frame, "Show only current frame's tags"),
            )
            .on_hover_text(
                "On: each point is shown only on the frame it was placed on. Off: every \
                 point is shown on every frame. Either way each point keeps its frame — \
                 the counts are over all frames, and saving ties each point to its own.",
            )
            .on_disabled_hover_text("This image has a single frame");
        });
        ui.horizontal_wrapped(|ui| {
            // Named for what each platform does, as the toolbar's save is: a
            // browser cannot ask where, only hand the file to its downloads.
            let save = if cfg!(target_arch = "wasm32") {
                "Download ImageJ overlay"
            } else {
                "Save ImageJ overlay…"
            };
            if ui
                .button(save)
                .on_hover_text(
                    "An ImageJ ROI set (.zip) of the points, one multi-point selection per \
                     group, for ImageJ's ROI Manager",
                )
                .clicked()
            {
                action = CounterAction::SaveRois;
            }
            if ui
                .button(format!("{} Copy", super::ICON_COPY))
                .on_hover_text("Copy the table, tab-separated, to paste into a spreadsheet")
                .clicked()
            {
                ui.ctx().copy_text(counter.table_text());
            }
            let about = if imagej_reads_overlay {
                "Attach the points to this image as an ImageJ overlay. The next save writes \
                 them into the TIFF, uncompressed, and ImageJ shows them when it opens it."
            } else {
                "Attach the points to this image as an ImageJ overlay. The next save writes \
                 them into the TIFF, but this stack is too big to save uncompressed (over \
                 4 GiB), which is the only way ImageJ reads an overlay: use \
                 \u{201c}Save ImageJ overlay\u{2026}\u{201d} for ImageJ."
            };
            if ui
                .button("Add into TIFF's overlay")
                .on_hover_text(about)
                .clicked()
            {
                action = CounterAction::AddToTiff;
            }
        });
        // Weak, not small as well: the pop-up is drawn at native size even in
        // the browser, where small text in it is too small to read.
        ui.label(RichText::new(HINT).weak());
    });
    if !open {
        return CounterAction::Close;
    }
    action
}

/// The table: a row per group — its colour, its name to edit, its count — and
/// the total under it. Clicking a row selects it: new points go to the
/// selected group.
///
/// Rows are laid out only where they are in view, and the table scrolls past
/// a dozen of them: a file's overlay brings a group per ROI, which can be
/// thousands, and laying every one of them out on every frame would slow the
/// whole window down with it. The total is under the table rather than in it,
/// so that it stays in view however the rows are scrolled.
fn group_table(ui: &mut egui::Ui, counter: &mut PointCounter) {
    use egui_extras::{Column, TableBuilder};

    let counts = counter.counts();
    let total = counter.points.len();
    let mut selected = counter.selected;
    let row_height = ui.spacing().interact_size.y + 4.0;
    let right = egui::Layout::right_to_left(egui::Align::Center);
    let scroll_to: Option<usize> = ui.data_mut(|d| d.remove_temp(egui::Id::new(SCROLL_TO)));
    let groups = &mut counter.groups;
    let mut table = TableBuilder::new(ui);
    if let Some(row) = scroll_to {
        table = table.scroll_to_row(row, None);
    }
    table
        .id_salt("point_counter_table")
        .striped(true)
        .sense(egui::Sense::click())
        .vscroll(true)
        .max_scroll_height(12.0 * row_height)
        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
        .column(Column::remainder().at_least(140.0).clip(true))
        .column(Column::exact(64.0))
        .header(row_height, |mut header| {
            header.col(|ui| {
                ui.strong("Group");
            });
            header.col(|ui| {
                ui.with_layout(right, |ui| ui.strong("Points"));
            });
        })
        .body(|body| {
            body.rows(row_height, groups.len(), |mut row| {
                let g = row.index();
                let group = &mut groups[g];
                row.set_selected(g == selected);
                row.col(|ui| {
                    swatch(ui, group.color);
                    let name = ui.add(
                        egui::TextEdit::singleline(&mut group.name)
                            .id_salt(("point_counter_group", g))
                            .desired_width(f32::INFINITY),
                    );
                    // Editing a name is choosing that row too; the field
                    // takes the click before the row can see it.
                    if name.gained_focus() || name.clicked() {
                        selected = g;
                    }
                });
                row.col(|ui| {
                    ui.with_layout(right, |ui| ui.label(counts[g].to_string()));
                });
                if row.response().clicked() {
                    selected = g;
                }
            });
        });
    ui.separator();
    ui.horizontal(|ui| {
        ui.strong("Total");
        ui.with_layout(right, |ui| ui.strong(total.to_string()));
    });
    counter.selected = selected;
}

/// Bring the counter's window to the front: a menu entry for a window that is
/// already open, and has gone behind the main one or another, should show it.
fn raise_window(ctx: &egui::Context) {
    // A window of its own on the desktop, so the window manager's focus; in
    // the browser an egui window, so the top of egui's own order.
    #[cfg(not(target_arch = "wasm32"))]
    ctx.send_viewport_cmd_to(
        egui::ViewportId::from_hash_of(WINDOW_ID),
        egui::ViewportCommand::Focus,
    );
    #[cfg(target_arch = "wasm32")]
    ctx.move_to_top(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new(WINDOW_ID),
    ));
}

/// A group's colour, as a small square before its name.
fn swatch(ui: &mut egui::Ui, [r, g, b]: [u8; 3]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, Color32::from_rgb(r, g, b));
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0_f32, ui.visuals().weak_text_color()),
        egui::StrokeKind::Inside,
    );
}

impl super::ViewerApp {
    /// The Plugins menu's "Point counter": open the window, or bring it to the
    /// front if it is already open — with its points, which reopening must not
    /// throw away.
    ///
    /// A new session starts from the document's overlay when it has one, and
    /// says so on the status line.
    pub(super) fn open_point_counter(&mut self, ctx: &egui::Context) {
        let Some(stack) = self.core.document.as_ref() else {
            self.core.status = Some("Open an image first".into());
            return;
        };
        // Points are placed on the picture. An animation running in 3D stops
        // on its own when the view leaves it.
        self.core.view_mode = fast_tiff_viewer::ViewMode::Movie;
        if self.point_counter.is_none() {
            let loaded = loaded_message(stack);
            self.point_counter = Some(Session::for_stack(stack));
            if let Some(message) = loaded {
                self.report_done(message);
            }
        } else {
            raise_window(ctx);
        }
    }

    /// Draw the counter's window while it is open, and act on its buttons.
    /// Called once a frame.
    pub(super) fn poll_point_counter(&mut self, ctx: &egui::Context) {
        let Some(tool) = self.point_counter.as_mut() else {
            return;
        };
        // The points were placed on a picture that is no longer there.
        let Some(stack) = self.core.document.as_ref() else {
            self.point_counter = None;
            return;
        };
        let imagej_reads_overlay = fast_tiff_viewer::save::overlay_readable_by_imagej(stack);
        match counter_window(ctx, tool, stack.display.dims().frames, imagej_reads_overlay) {
            CounterAction::None => {}
            CounterAction::Close => self.point_counter = None,
            CounterAction::SaveRois => self.save_point_rois(),
            CounterAction::AddToTiff => self.add_points_to_overlay(),
        }
    }

    /// The points as ImageJ ROIs: one per group that has any, each point on
    /// the stack position of its frame, and placed so that ImageJ draws it on
    /// a hyperstack too — see [`PointCounter::rois_for`].
    fn point_rois(&self) -> Option<Vec<PointRoi>> {
        let stack = self.core.document.as_ref()?;
        let tool = self.point_counter.as_ref()?;
        Some(tool.counter.rois_for(stack))
    }

    /// "Save ImageJ overlay…": the points as a RoiSet `.zip`, which ImageJ's
    /// ROI Manager opens (and File > Open hands to it). Asked where on the
    /// desktop; handed to the downloads in a browser.
    fn save_point_rois(&mut self) {
        let Some(rois) = self.point_rois() else {
            return;
        };
        if rois.is_empty() {
            self.core.status = Some(format!("{NAME}: no points to save yet"));
            return;
        }
        let Some(stack) = self.core.document.as_ref() else {
            return;
        };
        let suggested = file_name(&stack.path);
        let bytes = match fast_tiff_lib::roi_set_zip(&rois) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.core.status = Some(format!("Could not save the ROI set: {e}"));
                return;
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let (name, wrote) = {
            let mut dialog = rfd::FileDialog::new()
                .add_filter("ImageJ ROI set (*.zip)", &["zip"])
                .set_file_name(&suggested);
            if let Some(dir) = stack.save_dir() {
                dialog = dialog.set_directory(dir);
            }
            let Some(path) = dialog.save_file() else {
                // Cancelled: not a failure, and not worth a status line.
                return;
            };
            // ImageJ tells a ROI set by its extension, so a name typed
            // without one gets it rather than becoming a file nothing opens.
            let path = if path.extension().is_none() {
                path.with_extension("zip")
            } else {
                path
            };
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());
            (
                name,
                std::fs::write(&path, &bytes).map_err(|e| e.to_string()),
            )
        };
        #[cfg(target_arch = "wasm32")]
        let (name, wrote) = {
            let wrote = crate::web_save::download_as(&bytes, &suggested, "application/zip");
            (suggested, wrote)
        };

        match wrote {
            Ok(()) => self.report_done(format!("Saved {} to {name}", summary(&rois))),
            Err(e) => self.core.status = Some(format!("Could not save {name}: {e}")),
        }
    }

    /// "Add into TIFF's overlay": attach the points to the open document, so
    /// the next save writes them into the TIFF as an ImageJ overlay.
    ///
    /// Replaces what was attached before rather than adding to it: the
    /// overlay is the counter as it stands, and pressing the button twice
    /// must not write every point twice.
    fn add_points_to_overlay(&mut self) {
        let Some(rois) = self.point_rois() else {
            return;
        };
        let Some(stack) = self.core.document.as_mut() else {
            return;
        };
        if rois.is_empty() {
            // Deleting every point and adding again is how an overlay added
            // by mistake comes off.
            if stack.overlay.is_empty() {
                self.core.status = Some(format!("{NAME}: no points to add yet"));
            } else {
                stack.overlay.clear();
                self.report_done(
                    "Removed the overlay: there are no points. Save the file to write it \
                     without one."
                        .to_string(),
                );
            }
            return;
        }
        let message = if fast_tiff_viewer::save::overlay_readable_by_imagej(stack) {
            format!(
                "Added {} to the overlay. Save the file to write it (saved uncompressed, so \
                 ImageJ can read it).",
                summary(&rois)
            )
        } else {
            format!(
                "Added {} to the overlay. Saving writes it into the file, but this stack is \
                 too big to save uncompressed, so ImageJ will not show it: use \
                 \u{201c}Save ImageJ overlay\u{2026}\u{201d} for ImageJ.",
                summary(&rois)
            )
        };
        stack.overlay = rois;
        self.report_done(message);
    }
}

#[cfg(test)]
#[path = "point_counter_tests.rs"]
mod tests;

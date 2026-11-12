//! Asking the user for a rectangle on behalf of a plugin, then running it
//! again with that rectangle.
//!
//! A plugin whose input is a region (Crop is the first) returns
//! `Outcome::SelectRegion` when it has none. The host then:
//!
//! 1. shows the image in 2D, since a rectangle is drawn on the picture;
//! 2. puts the plugin's prompt in the status line;
//! 3. arms a canvas tool ([`CanvasTool`]) ahead of the others — the point
//!    counter, a plot's regions — drawing rectangles only, one at a time: a
//!    new drag replaces the last.
//!
//! Enter runs the plugin again, with the same parameters and the rectangle as
//! its selection. Escape gives up and the plugin is not run again. So does
//! going to 3D, or opening another file: the rectangle was drawn on a picture
//! that is no longer there.

use fast_tiff_viewer::{ViewMode, Viewer};
use fasttiff_plugin_api::{Params, Roi, Shape};

use super::canvas_tool::{CanvasGesture, CanvasTool, ImageXform, Layer, Priority, ToolEvent};
use super::plot;

/// A rectangle being drawn for a plugin.
pub(super) struct RegionPick {
    /// Which plugin asked, and what its dialog was answered with: the second
    /// run is the same question with a rectangle added.
    pub plugin: usize,
    pub params: Params,
    /// The plugin's name, for the status line.
    pub name: String,
    /// The rectangle drawn so far.
    pub roi: Option<Roi>,
    /// A drag in flight. See [`plot::track_drag`].
    pub drag: Option<plot::DragCorners>,
}

impl RegionPick {
    /// What the status line says once a rectangle is drawn.
    pub fn drawn_message(&self, roi: &Roi) -> String {
        format!(
            "{}: {} x {} pixels from ({}, {}). Press Enter to apply, Esc to cancel, \
             or drag again",
            self.name, roi.w, roi.h, roi.x, roi.y
        )
    }
}

/// Draw the rectangle and the drag in flight, over the picture.
pub(super) fn draw(painter: &egui::Painter, pick: &RegionPick, xf: &ImageXform) {
    let stroke = egui::Stroke::new(1.5_f32, plot::palette(0));
    if let Some(roi) = &pick.roi {
        plot::stroke_roi(painter, xf.roi_rect(roi), roi.shape, stroke);
    }
    if let Some((from, to)) = pick.drag {
        if let Some(preview) = plot::drag_region(from, to, Shape::Rect, xf.size.0, xf.size.1) {
            plot::stroke_roi(
                painter,
                xf.roi_rect(&preview),
                preview.shape,
                egui::Stroke::new(1.0_f32, plot::palette(0)),
            );
        }
    }
}

/// The canvas tool: ahead of every other, being what the user was just asked
/// to do. It draws regions, and hides a plot's while it is armed.
impl CanvasTool for RegionPick {
    fn priority(&self) -> Priority {
        Priority::Asked
    }

    fn layer(&self) -> Layer {
        Layer::Regions
    }

    /// In whole pixels, by [`plot::track_drag`] as a plot's regions are: from
    /// the pixel the button went down on to the one it came up on, committed
    /// on release. A new drag replaces the last rectangle; one wholly off the
    /// picture keeps it. The status line says what was drawn.
    fn track(&mut self, g: &CanvasGesture, xf: &ImageXform, _frame: usize) -> Option<ToolEvent> {
        let (from, to) = plot::track_drag(&mut self.drag, g, xf)?;
        let roi = plot::drag_region(from, to, Shape::Rect, xf.size.0, xf.size.1)?;
        self.roi = Some(roi);
        Some(ToolEvent::Status(self.drawn_message(&roi)))
    }

    fn draw(&self, painter: &egui::Painter, xf: &ImageXform, _frame: usize) {
        draw(painter, self, xf);
    }

    /// Enter and Escape while the rectangle is being drawn, and giving up
    /// when the picture it is drawn on goes away.
    fn tick(&mut self, ctx: &egui::Context, core: &Viewer, busy: bool) -> Option<ToolEvent> {
        let cancelled = || ToolEvent::Close(format!("{}: cancelled", self.name));
        if core.document.is_none() || core.view_mode != ViewMode::Movie {
            return Some(cancelled());
        }
        // Keys typed into a text field are that field's.
        if ctx.egui_wants_keyboard_input() {
            return None;
        }
        let (enter, escape) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if escape {
            Some(cancelled())
        } else if enter {
            let Some(roi) = self.roi else {
                return Some(ToolEvent::Status(format!(
                    "{}: drag a rectangle on the image first, then press Enter (Esc cancels)",
                    self.name
                )));
            };
            // Kept, not dropped, while something else is running: the user
            // can press Enter again once it is done.
            if busy {
                return Some(ToolEvent::Busy);
            }
            Some(ToolEvent::Run {
                plugin: self.plugin,
                params: self.params.clone(),
                rois: vec![roi],
            })
        } else {
            None
        }
    }
}

impl super::ViewerApp {
    /// The plugin at `plugin` (called `name`) asked for a rectangle.
    pub(super) fn start_region_pick(
        &mut self,
        plugin: usize,
        params: Params,
        name: &str,
        prompt: String,
    ) {
        if self.core.document.is_none() {
            self.core.status = Some("Open an image first".into());
            return;
        }
        // A rectangle is drawn on the picture. An animation running in 3D
        // stops on its own when the view leaves it.
        self.core.view_mode = fast_tiff_viewer::ViewMode::Movie;
        self.tools.region_pick = Some(RegionPick {
            plugin,
            params,
            name: name.to_string(),
            roi: None,
            drag: None,
        });
        self.core.status = Some(prompt);
    }
}

#[cfg(test)]
#[path = "region_pick_tests.rs"]
mod tests;

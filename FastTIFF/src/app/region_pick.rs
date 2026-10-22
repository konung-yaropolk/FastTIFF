//! Asking the user for a rectangle on behalf of a plugin, then running it
//! again with that rectangle.
//!
//! A plugin whose input is a region (Crop is the first) returns
//! `Outcome::SelectRegion` when it has none. The host then:
//!
//! 1. shows the image in 2D, since a rectangle is drawn on the picture;
//! 2. puts the plugin's prompt in the status line;
//! 3. arms the canvas tool the plots use, drawing rectangles only, one at a
//!    time: a new drag replaces the last.
//!
//! Enter runs the plugin again, with the same parameters and the rectangle as
//! its selection. Escape gives up and the plugin is not run again. So does
//! going to 3D, or opening another file: the rectangle was drawn on a picture
//! that is no longer there.

use fasttiff_plugin_api::{Params, Roi, Shape};

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
    /// A drag in flight: the pixel it started on and the pixel it is over.
    pub drag: Option<((i64, i64), (i64, i64))>,
}

impl RegionPick {
    /// Take one frame of a drag on the picture: `px` is the pixel under the
    /// pointer, `pressed` the pixel the button went down on, and
    /// `started`/`dragging`/`stopped` what the primary button did. Returns the
    /// rectangle when a drag ends on one.
    ///
    /// The rectangle starts where the button went down, not where the drag was
    /// noticed: a drag is only reported once the pointer has moved a few
    /// pixels, and a corner that far from the press is not the one aimed at.
    pub fn track(
        &mut self,
        px: Option<(i64, i64)>,
        pressed: Option<(i64, i64)>,
        (started, dragging, stopped): (bool, bool, bool),
        size: (u32, u32),
    ) -> Option<Roi> {
        if let Some(px) = px {
            match self.drag {
                None if started => self.drag = Some((pressed.unwrap_or(px), px)),
                Some((from, _)) if dragging || stopped => self.drag = Some((from, px)),
                _ => {}
            }
        }
        if !stopped {
            return None;
        }
        let (from, to) = self.drag.take()?;
        let roi = plot::drag_region(from, to, Shape::Rect, size.0, size.1)?;
        self.roi = Some(roi);
        Some(roi)
    }

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
pub(super) fn draw(
    painter: &egui::Painter,
    pick: &RegionPick,
    origin: egui::Pos2,
    zoom: f32,
    size: (u32, u32),
) {
    let stroke = egui::Stroke::new(1.5_f32, plot::palette(0));
    if let Some(roi) = &pick.roi {
        plot::stroke_roi(
            painter,
            plot::roi_rect(roi, origin, zoom),
            roi.shape,
            stroke,
        );
    }
    if let Some((from, to)) = pick.drag {
        if let Some(preview) = plot::drag_region(from, to, Shape::Rect, size.0, size.1) {
            plot::stroke_roi(
                painter,
                plot::roi_rect(&preview, origin, zoom),
                preview.shape,
                egui::Stroke::new(1.0_f32, plot::palette(0)),
            );
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
        if self.core.stack.is_none() {
            self.core.status = Some("Open an image first".into());
            return;
        }
        // A rectangle is drawn on the picture. An animation running in 3D
        // stops on its own when the view leaves it.
        self.core.view_mode = fast_tiff_viewer::ViewMode::Movie;
        self.region_pick = Some(RegionPick {
            plugin,
            params,
            name: name.to_string(),
            roi: None,
            drag: None,
        });
        self.core.status = Some(prompt);
    }

    /// Enter and Escape while a rectangle is being drawn, and giving up when
    /// the picture it is drawn on goes away. Called once a frame.
    pub(super) fn tick_region_pick(&mut self, ctx: &egui::Context) {
        let Some(pick) = self.region_pick.as_ref() else {
            return;
        };
        let name = pick.name.clone();
        if self.core.stack.is_none() || self.core.view_mode != fast_tiff_viewer::ViewMode::Movie {
            self.region_pick = None;
            self.core.status = Some(format!("{name}: cancelled"));
            return;
        }
        // Keys typed into a text field are that field's.
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let (enter, escape) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if escape {
            self.region_pick = None;
            self.core.status = Some(format!("{name}: cancelled"));
        } else if enter {
            let Some(roi) = pick.roi else {
                self.core.status = Some(format!(
                    "{name}: drag a rectangle on the image first, then press Enter (Esc cancels)"
                ));
                return;
            };
            // Kept, not dropped, while something else is running: the user
            // can press Enter again once it is done.
            if self.job.is_some() {
                self.core.status = Some(self.busy_message());
                return;
            }
            let pick = self.region_pick.take().expect("checked above");
            self.picked = Some((pick.plugin, vec![roi]));
            self.run_plugin(pick.plugin, pick.params);
        }
    }
}

#[cfg(test)]
#[path = "region_pick_tests.rs"]
mod tests;

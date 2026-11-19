//! egui's geometry and the viewer's: the one place this app converts between
//! them.
//!
//! The 2D view is the core's ([`fast_tiff_viewer::view2d::View2d`]), kept in logical
//! points — which is what egui calls a point, so a conversion is a copy of
//! the two components and changes no number. Everything that reads input or
//! paints stays in egui's types and crosses here: `.logical()` on the way in,
//! `.egui()` on the way out.

use fast_tiff_viewer::view2d::{LogicalPt, LogicalRect, LogicalVec};

/// An egui position, offset or rect, as the viewer's.
pub(super) trait Logical {
    type Logical;
    fn logical(self) -> Self::Logical;
}

/// The viewer's position, offset or rect, as egui's.
pub(super) trait Egui {
    type Egui;
    fn egui(self) -> Self::Egui;
}

impl Logical for egui::Pos2 {
    type Logical = LogicalPt;
    fn logical(self) -> LogicalPt {
        LogicalPt::new(self.x, self.y)
    }
}

impl Logical for egui::Vec2 {
    type Logical = LogicalVec;
    fn logical(self) -> LogicalVec {
        LogicalVec::new(self.x, self.y)
    }
}

impl Logical for egui::Rect {
    type Logical = LogicalRect;
    fn logical(self) -> LogicalRect {
        LogicalRect {
            min: self.min.logical(),
            max: self.max.logical(),
        }
    }
}

impl Egui for LogicalPt {
    type Egui = egui::Pos2;
    fn egui(self) -> egui::Pos2 {
        egui::pos2(self.x, self.y)
    }
}

impl Egui for LogicalVec {
    type Egui = egui::Vec2;
    fn egui(self) -> egui::Vec2 {
        egui::vec2(self.x, self.y)
    }
}

impl Egui for LogicalRect {
    type Egui = egui::Rect;
    fn egui(self) -> egui::Rect {
        egui::Rect::from_min_max(self.min.egui(), self.max.egui())
    }
}

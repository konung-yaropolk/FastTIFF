//! Regions a user drew on the picture.
//!
//! The host owns the drawing — a plugin cannot draw, and that is not going to
//! change — but the *shape* has to be a type both sides name, or the host and
//! the plugin would each have their own idea of which pixels an ellipse covers
//! and quietly disagree. So the geometry lives here, in the contract, and is
//! the same arithmetic wherever it runs.
//!
//! # Why pixels, and why integers
//!
//! A selection is over samples. Half a pixel is not something the data has, and
//! a region whose edge fell between samples would measure differently depending
//! on how it was rounded later. The interface snaps a drag to this grid, so what
//! is drawn is exactly what is measured.

/// What a region is shaped like.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Rect,
    Ellipse,
}

impl Shape {
    /// What to call it in the tool's controls.
    pub fn label(self) -> &'static str {
        match self {
            Shape::Rect => "Rectangle",
            Shape::Ellipse => "Ellipse",
        }
    }
}

/// A region of interest, in whole image pixels.
///
/// `x`/`y` are the top-left corner from the image's top-left; `w`/`h` are the
/// extent. A region with a zero extent is not constructible through
/// [`Roi::from_corners`], because a region covering nothing has no mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Roi {
    pub shape: Shape,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Roi {
    /// A region from two corners in any order, clamped to a `width`x`height`
    /// image.
    ///
    /// Takes corners rather than an origin and a size because that is what a
    /// drag produces, and normalising here means no caller has to think about a
    /// drag that went up and to the left. Clamping happens before the extent is
    /// worked out, so a drag that wandered off the image keeps the part that did
    /// not rather than being refused whole.
    ///
    /// `None` when nothing is left — a click rather than a drag, or a drag
    /// entirely outside the image.
    pub fn from_corners(a: (i64, i64), b: (i64, i64), width: u32, height: u32) -> Option<Roi> {
        let x0 = a.0.min(b.0).clamp(0, width as i64);
        let x1 = a.0.max(b.0).clamp(0, width as i64);
        let y0 = a.1.min(b.1).clamp(0, height as i64);
        let y1 = a.1.max(b.1).clamp(0, height as i64);
        let (w, h) = ((x1 - x0) as u32, (y1 - y0) as u32);
        if w == 0 || h == 0 {
            return None;
        }
        Some(Roi {
            shape: Shape::Rect,
            x: x0 as u32,
            y: y0 as u32,
            w,
            h,
        })
    }

    /// This region with a different shape.
    pub fn with_shape(self, shape: Shape) -> Roi {
        Roi { shape, ..self }
    }

    /// Whether the pixel at `(x, y)` is inside.
    ///
    /// For an ellipse this asks whether the pixel's *centre* is inside, which is
    /// the rule ImageJ uses. A pixel is a sample, not an area, so a rule that
    /// weighted partial coverage would be inventing a value the detector never
    /// produced.
    pub fn contains(&self, x: u32, y: u32) -> bool {
        if x < self.x || y < self.y || x >= self.x + self.w || y >= self.y + self.h {
            return false;
        }
        match self.shape {
            Shape::Rect => true,
            Shape::Ellipse => {
                let (rx, ry) = (self.w as f64 / 2.0, self.h as f64 / 2.0);
                let (cx, cy) = (self.x as f64 + rx, self.y as f64 + ry);
                let dx = (x as f64 + 0.5 - cx) / rx;
                let dy = (y as f64 + 0.5 - cy) / ry;
                dx * dx + dy * dy <= 1.0
            }
        }
    }

    /// How many pixels this region actually covers.
    ///
    /// Counted rather than computed: `pi*rx*ry` is the area of the continuous
    /// ellipse, and the number of pixel centres inside it is a different — and,
    /// for a small region, quite different — number. Anything averaging over a
    /// region has to divide by what it summed.
    pub fn pixel_count(&self) -> usize {
        match self.shape {
            Shape::Rect => self.w as usize * self.h as usize,
            Shape::Ellipse => self.pixels().count(),
        }
    }

    /// Every pixel inside, in row-major order.
    ///
    /// The one place the containment rule is applied to a whole region, so a
    /// caller cannot walk the bounding box and forget the shape.
    pub fn pixels(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (self.y..self.y.saturating_add(self.h)).flat_map(move |y| {
            (self.x..self.x.saturating_add(self.w))
                .filter(move |&x| self.contains(x, y))
                .map(move |x| (x, y))
        })
    }

    /// Indices into a row-major `width`-wide plane, for the pixels inside.
    ///
    /// Resolved once by anything that will read many planes through the same
    /// region: re-running the containment test per plane is the same answer
    /// computed a thousand times.
    pub fn indices(&self, width: u32, height: u32) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.pixel_count());
        for (x, y) in self.pixels() {
            if x < width && y < height {
                out.push(y as usize * width as usize + x as usize);
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "selection_tests.rs"]
mod tests;

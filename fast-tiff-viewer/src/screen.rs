//! The units on screen: positions and sizes in logical points, the pixels a
//! GPU renders into, and the image's size in its own pixels. One module, so
//! one place holds the conversions between them.
//!
//! A frontend lays its panels out in *logical points* — what a toolkit such
//! as egui calls a point, scaled to the display by its pixels-per-point —
//! and that is the unit the 2D view's pan, zoom and panel are kept in
//! ([`LogicalPt`] for a place, [`LogicalVec`] for an offset or a size,
//! [`LogicalRect`] for an area). The image's size is in the image's own
//! pixels ([`ImagePx`]); the zoom is how many points one of those is across.
//! What the GPU renders into is physical pixels, [`PhysicalPx`], the unit of
//! every size on screen the sync is handed, and the one conversion to it is
//! [`LogicalVec::to_physical`].
//!
//! The arithmetic is written to the bit as a toolkit's own two-component
//! types do it — each component on its own, in the order written — so a
//! frontend converting at its edge gets the numbers its own types would have
//! given.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A size in physical pixels: the display device's own, which is what a GPU
/// renders into and what decides how much detail is worth decoding.
///
/// Fractional, because a panel laid out in logical points on a display
/// scaled by 1.25 or 1.5 is rarely a whole number of them; [`whole`](Self::whole)
/// is the size a render target is made at. Zero means "not known". A
/// frontend thinking in logical points gets one with
/// [`LogicalVec::to_physical`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhysicalPx {
    pub width: f32,
    pub height: f32,
}

impl PhysicalPx {
    /// A size the frontend has not told: residency falls back to the texture
    /// budget, and a volume keeps whatever detail the GPU can hold.
    pub const UNKNOWN: PhysicalPx = PhysicalPx::new(0.0, 0.0);

    pub const fn new(width: f32, height: f32) -> Self {
        PhysicalPx { width, height }
    }

    /// Whether this is a size at all: finite, and at least one pixel each way.
    pub fn is_known(self) -> bool {
        self.width.is_finite() && self.height.is_finite() && self.width >= 1.0 && self.height >= 1.0
    }

    /// Whole pixels, at least one each way: the size a render target is made
    /// at.
    pub fn whole(self) -> [u32; 2] {
        [
            self.width.round().max(1.0) as u32,
            self.height.round().max(1.0) as u32,
        ]
    }

    /// The longer side in whole pixels; `0` when not known.
    pub fn longest_side(self) -> u32 {
        self.width.max(self.height).round().max(0.0) as u32
    }
}

/// A place on screen, in logical points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogicalPt {
    pub x: f32,
    pub y: f32,
}

/// An offset or a size on screen, in logical points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogicalVec {
    pub x: f32,
    pub y: f32,
}

/// An area on screen, in logical points: `min` its top-left corner, `max`
/// its bottom-right.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogicalRect {
    pub min: LogicalPt,
    pub max: LogicalPt,
}

/// The size of an image in its own pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImagePx {
    pub width: u32,
    pub height: u32,
}

impl LogicalPt {
    pub const ZERO: LogicalPt = LogicalPt::new(0.0, 0.0);

    pub const fn new(x: f32, y: f32) -> Self {
        LogicalPt { x, y }
    }

    /// The nearer-the-origin corner of the two, each axis on its own.
    pub fn min(self, other: LogicalPt) -> LogicalPt {
        LogicalPt::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// The further-from-the-origin corner of the two, each axis on its own.
    pub fn max(self, other: LogicalPt) -> LogicalPt {
        LogicalPt::new(self.x.max(other.x), self.y.max(other.y))
    }
}

impl LogicalVec {
    pub const ZERO: LogicalVec = LogicalVec::new(0.0, 0.0);

    pub const fn new(x: f32, y: f32) -> Self {
        LogicalVec { x, y }
    }

    /// How long it is, as `hypot` measures it.
    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    /// The smaller of the two on each axis.
    pub fn min(self, other: LogicalVec) -> LogicalVec {
        LogicalVec::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// This size on a display of `pixels_per_point`: the one way from
    /// points to the pixels a GPU renders.
    pub fn to_physical(self, pixels_per_point: f32) -> PhysicalPx {
        PhysicalPx::new(self.x * pixels_per_point, self.y * pixels_per_point)
    }
}

impl LogicalRect {
    /// The rect at `min` that is `size` across.
    pub fn from_min_size(min: LogicalPt, size: LogicalVec) -> Self {
        LogicalRect {
            min,
            max: min + size,
        }
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    pub fn size(&self) -> LogicalVec {
        self.max - self.min
    }

    pub fn center(&self) -> LogicalPt {
        LogicalPt::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
        )
    }

    /// Whether `p` is inside, its edges included.
    pub fn contains(&self, p: LogicalPt) -> bool {
        self.min.x <= p.x && p.x <= self.max.x && self.min.y <= p.y && p.y <= self.max.y
    }

    /// The part of this rect inside `other`, which is not
    /// [`positive`](Self::is_positive) when they do not overlap.
    pub fn intersect(self, other: LogicalRect) -> LogicalRect {
        LogicalRect {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Whether it covers any area at all.
    pub fn is_positive(&self) -> bool {
        self.min.x < self.max.x && self.min.y < self.max.y
    }
}

impl ImagePx {
    pub const fn new(width: u32, height: u32) -> Self {
        ImagePx { width, height }
    }

    /// How large the image is on screen at `zoom` points per pixel.
    pub fn at_zoom(self, zoom: f32) -> LogicalVec {
        LogicalVec::new(self.width as f32 * zoom, self.height as f32 * zoom)
    }
}

impl Sub for LogicalPt {
    type Output = LogicalVec;
    fn sub(self, rhs: LogicalPt) -> LogicalVec {
        LogicalVec::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Add<LogicalVec> for LogicalPt {
    type Output = LogicalPt;
    fn add(self, rhs: LogicalVec) -> LogicalPt {
        LogicalPt::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub<LogicalVec> for LogicalPt {
    type Output = LogicalPt;
    fn sub(self, rhs: LogicalVec) -> LogicalPt {
        LogicalPt::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Add for LogicalVec {
    type Output = LogicalVec;
    fn add(self, rhs: LogicalVec) -> LogicalVec {
        LogicalVec::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for LogicalVec {
    type Output = LogicalVec;
    fn sub(self, rhs: LogicalVec) -> LogicalVec {
        LogicalVec::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl AddAssign for LogicalVec {
    fn add_assign(&mut self, rhs: LogicalVec) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for LogicalVec {
    fn sub_assign(&mut self, rhs: LogicalVec) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for LogicalVec {
    type Output = LogicalVec;
    fn mul(self, factor: f32) -> LogicalVec {
        LogicalVec::new(self.x * factor, self.y * factor)
    }
}

impl MulAssign<f32> for LogicalVec {
    fn mul_assign(&mut self, factor: f32) {
        self.x *= factor;
        self.y *= factor;
    }
}

impl Div<f32> for LogicalVec {
    type Output = LogicalVec;
    fn div(self, factor: f32) -> LogicalVec {
        LogicalVec::new(self.x / factor, self.y / factor)
    }
}

impl Neg for LogicalVec {
    type Output = LogicalVec;
    fn neg(self) -> LogicalVec {
        LogicalVec::new(-self.x, -self.y)
    }
}

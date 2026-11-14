//! [`FrameInputs`]: what a frontend decides each frame and hands to
//! [`Viewer::sync`](crate::Viewer::sync).
//!
//! Everything the core needs to know about the frontend's view, and nothing it
//! keeps: which part of the 2D image is on screen and in how many pixels, how
//! large the 3D view is and how finely to render it, whether the view is still
//! moving, and whether the histograms are being shown. The frontend works these
//! out from its own pan, zoom, panel and animation state, and passes them as one
//! value, so the frame contract is the signature of `sync` rather than a set of
//! fields to remember to write, in the right order, before calling it.
//!
//! Sizes on screen are [`PhysicalPx`] everywhere: the device's own pixels,
//! which is what a GPU renders into and what decides how much detail is worth
//! decoding. A frontend thinking in logical points converts once, with
//! [`PhysicalPx::from_points`].

/// A size in physical pixels: the display device's own, which is what a GPU
/// renders into.
///
/// Fractional, because a panel laid out in logical points on a display
/// scaled by 1.25 or 1.5 is rarely a whole number of them; [`whole`](Self::whole)
/// is the size a render target is made at. Zero means "not known".
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

    /// `width` x `height` logical points on a display of `pixels_per_point`.
    pub fn from_points(width: f32, height: f32, pixels_per_point: f32) -> Self {
        PhysicalPx::new(width * pixels_per_point, height * pixels_per_point)
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

/// What a frontend decides for one frame. See the [module docs](self).
///
/// Kept by the frontend from one frame to the next where that matters: a
/// frame that does not lay the 2D view out (the 3D view is on screen, or the
/// panel has no room) leaves [`image`](Self::image) as the last frame that did,
/// which is what the residency of the 2D frame is still planned against.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameInputs {
    /// The 2D view: which part of the image is on screen, in how many pixels.
    pub image: ImageInputs,
    /// The 3D view: how large it is, how finely to render it, and an
    /// animation's turn of the volume.
    pub volume: VolumeInputs,
    /// Whether the 2D view is still moving under an animation or gesture, so
    /// any window started now would be planned against a zoom that has
    /// already changed by the time it is decoded.
    ///
    /// Set while a zoom step is gliding out or a pinch is live. Windows already
    /// finished are still taken — a better picture is always worth having —
    /// but no new one is started until the view settles. Without this an
    /// animated zoom decodes at every sampling level it passes through on the
    /// way, which is work the user never sees: the view is only at each of
    /// them for a few frames, and the level it is heading for is the only one
    /// that will still be wanted.
    ///
    /// The 3D view says how it is moving through
    /// [`VolumeInputs::downscale`] instead.
    pub moving: bool,
    /// Whether the frontend is showing the histograms of the frame on screen
    /// ([`Document::histograms`](crate::Document::histograms)): set while its
    /// histogram panel is open.
    ///
    /// While it is, the sync bins each frame it decodes for the screen (during
    /// playback, a few times a second, and by the read-ahead when it is
    /// reading ahead), and decodes what that did not cover: the channels
    /// switched off, a frame already on screen when the panel opened, and in
    /// the 3D view the 2D frame. A frame past the texture limit is binned from
    /// a view of the whole of it instead. Off, none of that is done, and the
    /// histograms held are left as they are. See [`crate::histogram`].
    pub histograms: bool,
}

/// The 2D view's side of [`FrameInputs`]: the part of the image on screen, and
/// the room it is drawn in.
///
/// The image is always drawn into the on-screen visible rect, with pan and
/// zoom done through the UV window — never through an oversized viewport,
/// which a GPU backend would clamp to the framebuffer (squashing the image
/// instead of zooming).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageInputs {
    /// The visible part of the image, `uv_offset .. uv_offset + uv_scale`, in
    /// fractions of the whole image (0..1 each way).
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
    /// The size of the area the image is drawn into.
    ///
    /// The other half of the same question: the UV window says *which* part
    /// of the image is on screen, this how much room there is to draw it, and
    /// together they decide how finely the image needs to be sampled. Without
    /// it a huge image is sampled as finely as the texture budget allows,
    /// which can be a hundred times more detail than the display can show,
    /// paid for on every zoom step. [`PhysicalPx::UNKNOWN`] falls back to the
    /// texture budget.
    pub viewport: PhysicalPx,
}

impl Default for ImageInputs {
    /// The whole image, in a viewport not yet known.
    fn default() -> Self {
        ImageInputs {
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
            viewport: PhysicalPx::UNKNOWN,
        }
    }
}

impl ImageInputs {
    /// The part of an image laid out at `image_min` with size `image_size`
    /// that shows through the rect `visible_min` + `visible_size`, all four in
    /// logical points, on a display of `pixels_per_point`.
    ///
    /// `visible` is the image's rect cut to the panel, which is all a frontend
    /// draws: the image overflowing the panel is what pan and zoom are.
    pub fn of_visible(
        image_min: [f32; 2],
        image_size: [f32; 2],
        visible_min: [f32; 2],
        visible_size: [f32; 2],
        pixels_per_point: f32,
    ) -> Self {
        let inv = [1.0 / image_size[0].max(1.0), 1.0 / image_size[1].max(1.0)];
        ImageInputs {
            uv_offset: [
                (visible_min[0] - image_min[0]) * inv[0],
                (visible_min[1] - image_min[1]) * inv[1],
            ],
            uv_scale: [visible_size[0] * inv[0], visible_size[1] * inv[1]],
            viewport: PhysicalPx::from_points(visible_size[0], visible_size[1], pixels_per_point),
        }
    }
}

/// A turn of the volume, row-major, in volume coordinates: what an animation
/// sets on top of the user's camera (see [`VolumeInputs::pose`]).
pub type Pose = [[f32; 3]; 3];

/// The 3D view's side of [`FrameInputs`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeInputs {
    /// The size of the 3D view: what bounds the work of a frame (see
    /// `scivis_render::max_steps`), what the renderer renders at, and the
    /// ray-march projection's aspect ratio ([`aspect`](Self::aspect)).
    pub viewport: PhysicalPx,
    /// How many times smaller per axis to render the view than it is shown:
    /// 1 renders every pixel. A frontend sets
    /// [`scivis_render::MOVING_DOWNSCALE`] while the user is moving the view,
    /// where a sharp frame is wasted, and 1 once it is still.
    pub downscale: u32,
    /// The size of the screen (unknown when the frontend cannot tell). A
    /// volume keeps at most about twice its longest side in voxels along any
    /// axis — more than that cannot be seen even zoomed in twofold — however
    /// much GPU memory there is.
    pub screen: PhysicalPx,
    /// A turn of the volume about the middle of the volume under the centre
    /// of the view (see `VolumeCamera::pivot_distance`), on top of the user's
    /// camera. Set by a frontend playing an animation, frame by frame; `None`
    /// the rest of the time. See [`VolumeCamera::posed`].
    ///
    /// Separate from the camera on purpose: the camera
    /// ([`VolumeSettings::cam`](crate::VolumeSettings::cam)) is what the user
    /// set up and the animation is seen from, and leaving it untouched is what
    /// lets a frontend tell that the user has taken the view back (it changed)
    /// and put the view back exactly as it was when the animation ends.
    ///
    /// [`VolumeCamera::posed`]: crate::camera::VolumeCamera::posed
    pub pose: Option<Pose>,
}

impl Default for VolumeInputs {
    /// A sharp, unturned view the size of a common screen, on a screen of
    /// unknown size: what a frame before the 3D view is first laid out plans
    /// against.
    fn default() -> Self {
        VolumeInputs {
            viewport: PhysicalPx::new(1920.0, 1080.0),
            downscale: 1,
            screen: PhysicalPx::UNKNOWN,
            pose: None,
        }
    }
}

impl VolumeInputs {
    /// The view's aspect ratio (width / height), for the ray-march projection,
    /// within 0.1..=10.
    pub fn aspect(&self) -> f32 {
        (self.viewport.width / self.viewport.height.max(1.0)).clamp(0.1, 10.0)
    }

    /// The most samples a ray of this view may take
    /// (`VolumeParams::max_steps`) with `channels` channels sampled `interp`:
    /// what keeps a frame of the whole viewport within the frame budget (see
    /// `scivis_render::max_steps`).
    ///
    /// The whole viewport's even while the view renders reduced
    /// ([`downscale`](Self::downscale)), so a reduced frame takes as many
    /// steps along each ray as a sharp one and does a fraction of the work,
    /// which is what reducing it is for. Worked out from the reduced size, its
    /// rays could take up to sixteen times the steps, and a large volume would
    /// move barely faster than it stands.
    pub fn max_steps(&self, channels: usize, interp: scivis_render::VolumeInterp) -> i32 {
        let [w, h] = self.viewport.whole();
        scivis_render::max_steps(w as f64 * h as f64, channels, interp)
    }
}

#[cfg(test)]
#[path = "frame_tests.rs"]
mod tests;

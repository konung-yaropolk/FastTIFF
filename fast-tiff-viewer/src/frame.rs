//! [`FrameInputs`]: what a frontend decides each frame and hands to
//! [`Viewer::sync`](crate::Viewer::sync).

use crate::screen::{LogicalPt, LogicalRect, LogicalVec, PhysicalPx};

/// What a frontend decides for one frame, and hands to
/// [`Viewer::sync`](crate::Viewer::sync).
///
/// Everything the core needs to know about the frontend's view, and nothing it
/// keeps: which part of the 2D image is on screen and in how many pixels, how
/// large the 3D view is and how finely to render it, whether the view is still
/// moving, and whether the histograms are being shown. The frontend works these
/// out from its own pan, zoom, panel and animation state, and passes them as one
/// value, so the frame contract is the signature of `sync` rather than a set of
/// fields to remember to write, in the right order, before calling it.
///
/// Sizes on screen are [`PhysicalPx`] everywhere: the device's own pixels,
/// which is what a GPU renders into and what decides how much detail is worth
/// decoding. A frontend thinking in logical points converts once, with
/// [`LogicalVec::to_physical`](crate::view2d::LogicalVec::to_physical); the
/// 2D view's part is worked out for it, by
/// [`ImagePlacement::inputs`](crate::view2d::ImagePlacement::inputs).
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
    /// Set while a zoom step is gliding out or a pinch is live: a 2D view's
    /// [`View2d::is_moving`](crate::view2d::View2d::is_moving). Windows already
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
    /// The part of an image laid out at `image_min`, `image_size` across,
    /// that shows through `visible`, on a display of `pixels_per_point`.
    ///
    /// `visible` is the image's rect cut to the panel, which is all a frontend
    /// draws: the image overflowing the panel is what pan and zoom are. The
    /// size is the image's own rather than its rect's, which is not always the
    /// same number once added to the corner and taken off it again.
    pub(crate) fn of_visible(
        image_min: LogicalPt,
        image_size: LogicalVec,
        visible: LogicalRect,
        pixels_per_point: f32,
    ) -> Self {
        let inv = [1.0 / image_size.x.max(1.0), 1.0 / image_size.y.max(1.0)];
        let offset = visible.min - image_min;
        let size = visible.size();
        ImageInputs {
            uv_offset: [offset.x * inv[0], offset.y * inv[1]],
            uv_scale: [size.x * inv[0], size.y * inv[1]],
            viewport: size.to_physical(pixels_per_point),
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
    /// 1 renders every pixel. A frontend sets the user's reduction while the
    /// user is moving the view, where a sharp frame is wasted, and 1 once it
    /// is still: what
    /// [`VolumeNavPrefs::downscale_after`](crate::camera::VolumeNavPrefs::downscale_after)
    /// says.
    pub downscale: u32,
    /// The size of the screen (unknown when the frontend cannot tell). A
    /// volume keeps at most about twice its longest side in voxels along any
    /// axis — more than that cannot be seen even zoomed in twofold — however
    /// much GPU memory there is.
    pub screen: PhysicalPx,
    /// A turn of the volume about the middle of the volume under the centre
    /// of the view (see `VolumeCamera::pivot_distance`), on top of the user's
    /// camera. Set by a frontend playing an animation, frame by frame; `None`
    /// the rest of the time.
    ///
    /// Separate from the camera on purpose: the camera
    /// ([`VolumeSettings::cam`](crate::VolumeSettings::cam)) is what the user
    /// set up and the animation is seen from, and leaving it untouched is what
    /// lets a frontend tell that the user has taken the view back (it changed)
    /// and put the view back exactly as it was when the animation ends.
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
    pub(crate) fn max_steps(&self, channels: usize, interp: scivis_render::VolumeInterp) -> i32 {
        let [w, h] = self.viewport.whole();
        scivis_render::max_steps(w as f64 * h as f64, channels, interp)
    }
}

#[cfg(test)]
#[path = "frame_tests.rs"]
mod tests;

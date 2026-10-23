//! Moving the 3D view: a motion a plugin computes and the host carries out.
//!
//! The same split as [`crate::plot`], for the same reasons. A plugin cannot run
//! a loop on the host's frame clock — [`Plugin::run`](crate::Plugin::run) is
//! one call that returns — and it cannot draw. So it does not animate the view;
//! it *declares* an animation and the host plays it. An [`Animation`] is a list
//! of [`Pose`]s, one per frame, and the rate to show them at. The host shows
//! them on the 3D view in turn and starts again from the first, or renders each
//! of them once and writes the frames to a file.
//!
//! What the plugin keeps is everything that makes the motion mean anything:
//! about which axis, how far, how fast, and where a cycle ends.
//!
//! # A pose
//!
//! A pose is the volume's orientation, relative to how it lay when the
//! animation started, as a rotation. The camera does not move: the volume
//! turns in front of it, about a point at the **centre of the view**: the
//! middle of the stretch of volume that lies along the line of sight through
//! the middle of the frame as the user left it. That is the volume's own
//! centre until they pan, and after a pan or a zoom it is still in the middle
//! of the picture and inside the volume. A volume moved to one side turns
//! where it is seen, rather than swinging about a point off to the side or
//! far in front of it. Whatever the user had set up — the angle they were
//! looking from, how close, how far to one side — is where the animation is
//! seen from, and identity is "exactly the view the user was looking at".
//!
//! The rotation is in **volume coordinates**: `x` along the image's width, `y`
//! along its height, `z` through the slices — the axis a z-stack is a stack
//! along. A rotation about `x` is therefore a tumble about the image's
//! horizontal axis whatever the view, which is the axis a person means when
//! they ask for one.
//!
//! # A loop has no seam
//!
//! The host loops the list: after the last pose comes the first. A turntable
//! whose last pose is the first one again shows that orientation twice running,
//! which is a stutter on screen and a duplicated frame in a saved movie. So a
//! full turn in `n` frames is the angles `0, 360/n, … 360(n-1)/n`, never `360`
//! itself. [`Animation::cycle`] samples a period exactly that way.
//!
//! # Stopping
//!
//! The view belongs to the user. The host stops an animation by itself the
//! moment they take the view back — a drag, the wheel, a key that moves the
//! camera, the reset button — or leave the 3D view, and it then shows the view
//! the animation started from. A plugin stops its own with
//! [`Outcome::StopAnimation`](crate::Outcome::StopAnimation), and asks whether
//! one of its own is running with
//! [`HostContext::animating`](crate::HostContext::animating), which is how a
//! dialog knows whether to offer Play or Stop.
//!
//! # It crosses a shared library too
//!
//! As of plugin ABI 1.5 an animation travels the C contract like a plot does,
//! and the pose list arrives whole: a plugin in a `.dll` animates the view
//! exactly as one compiled into the app does. An older host refuses an
//! animation with a message saying so.

use crate::PluginError;

/// The volume's orientation in one frame: a rotation about the centre of the
/// view, relative to how it lay when the animation started.
///
/// See the [module documentation](self) for the coordinates and the pivot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Row-major: `rotation[r][c]` is row `r`, column `c`. A point of the
    /// volume at `p` when the animation started, measured from the pivot, is
    /// at `rotation · p` from it in this frame.
    pub rotation: [[f32; 3]; 3],
}

impl Default for Pose {
    fn default() -> Self {
        Pose::IDENTITY
    }
}

impl Pose {
    /// The volume as it was: the view the user was looking at.
    pub const IDENTITY: Pose = Pose {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    /// A pose from a rotation matrix, row-major.
    ///
    /// Taken as given; [`Animation::validate`] is what refuses a matrix that
    /// is not a rotation.
    pub fn from_matrix(rotation: [[f32; 3]; 3]) -> Pose {
        Pose { rotation }
    }

    /// A turn of `radians` about `axis`, right-handed: positive is
    /// anticlockwise looking from the tip of the axis towards the origin.
    ///
    /// The axis need not be unit length. One of zero length has no direction
    /// to turn about, and gives the identity.
    pub fn rotation(axis: [f32; 3], radians: f32) -> Pose {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        // NaN as well as zero: an axis with a NaN in it has no direction
        // either, and must not turn the volume into NaNs.
        if len.is_nan() || len <= 1e-12 || !radians.is_finite() {
            return Pose::IDENTITY;
        }
        let [x, y, z] = [axis[0] / len, axis[1] / len, axis[2] / len];
        // Rodrigues, written out: one rotation matrix, no accumulated error.
        let (s, c) = radians.sin_cos();
        let t = 1.0 - c;
        Pose {
            rotation: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
        }
    }

    /// This pose followed by `next`: the volume turned by `self`, then by
    /// `next`.
    pub fn then(&self, next: &Pose) -> Pose {
        let (a, b) = (&next.rotation, &self.rotation);
        let mut m = [[0.0f32; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            }
        }
        Pose { rotation: m }
    }

    /// Where the point `p`, in volume coordinates at the start, is in this
    /// frame.
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.rotation;
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
        ]
    }

    /// Whether this is a rotation, to within `tolerance` per entry of
    /// `RᵀR − I`, and not a mirror.
    ///
    /// The two are different failures and both are refused. A matrix that
    /// stretches or skews draws the volume distorted; one that mirrors it draws
    /// it inside out, which looks plausible and is wrong.
    pub fn is_rotation(&self, tolerance: f32) -> bool {
        let m = &self.rotation;
        if m.iter().flatten().any(|v| !v.is_finite()) {
            return false;
        }
        for i in 0..3 {
            for j in 0..3 {
                let dot = m[0][i] * m[0][j] + m[1][i] * m[1][j] + m[2][i] * m[2][j];
                let want = if i == j { 1.0 } else { 0.0 };
                if (dot - want).abs() > tolerance {
                    return false;
                }
            }
        }
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        det > 0.0
    }
}

/// What the host does with an [`Animation`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum AnimationOutput {
    /// Play it on the 3D view, looping, until it is stopped.
    #[default]
    Play,
    /// Render every pose once, in order, and write the frames to a TIFF: an
    /// RGB movie of exactly one cycle, at the animation's frame rate.
    ///
    /// An empty `path` asks the host to ask the user where — the usual case
    /// for a "Save…" button, which has nowhere better to get a name from.
    SaveTiff {
        /// Where to write. Empty for the host to ask.
        path: String,
    },
}

/// How large the frames of a saved movie are.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MovieResolution {
    /// As large as it takes for every pixel of the stack in view to be at
    /// least a pixel of the movie, and never smaller than the 3D view is on
    /// screen. The host caps the long side at what it can render.
    #[default]
    Max,
    /// The 3D view's own size on screen, in pixels.
    Window,
}

/// A motion of the 3D view, declared for the host to carry out.
///
/// See the [module documentation](self).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Animation {
    /// One pose per frame, in order. The host goes back to the first after
    /// the last.
    pub poses: Vec<Pose>,
    /// Frames per second, both on screen and in a saved movie.
    pub fps: f32,
    pub output: AnimationOutput,
    /// The size of a saved movie's frames. Not used when playing.
    pub resolution: MovieResolution,
}

impl Animation {
    /// More frames than a cycle needs, and short of what a runaway loop
    /// allocates: an hour at thirty frames a second.
    pub const MAX_POSES: usize = 108_000;

    /// The frame rates a host will take. Fractional and slow is fine — a
    /// slideshow of orientations — but nothing that cannot be a frame rate.
    pub const FPS_RANGE: std::ops::RangeInclusive<f32> = 0.1..=240.0;

    /// How far a pose may be from a rotation and still be accepted, per entry
    /// of `RᵀR − I`. Far looser than `f32` rounding, far tighter than any
    /// matrix someone meant as a stretch.
    pub const TOLERANCE: f32 = 1e-3;

    /// An animation to play.
    pub fn new(poses: Vec<Pose>, fps: f32) -> Animation {
        Animation {
            poses,
            fps,
            output: AnimationOutput::Play,
            resolution: MovieResolution::Max,
        }
    }

    /// One period of a cyclic motion in `frames` frames, without its seam.
    ///
    /// `pose_at` is given the phase in radians — `2π · i / frames` for frame
    /// `i` — and returns the pose there. The phase reaches `2π · (frames − 1) /
    /// frames` and never `2π`, so a motion that comes back to where it started
    /// after a full period loops with every frame shown once. See the
    /// [module documentation](self#a-loop-has-no-seam).
    pub fn cycle(frames: usize, fps: f32, mut pose_at: impl FnMut(f32) -> Pose) -> Animation {
        let n = frames.max(1);
        let poses = (0..n)
            .map(|i| pose_at(std::f32::consts::TAU * i as f32 / n as f32))
            .collect();
        Animation::new(poses, fps)
    }

    /// Write the frames to a TIFF instead of playing them. An empty path asks
    /// the user where.
    pub fn save_tiff(mut self, path: impl Into<String>) -> Animation {
        self.output = AnimationOutput::SaveTiff { path: path.into() };
        self
    }

    /// Save the movie at this size. See [`MovieResolution`].
    pub fn resolution(mut self, resolution: MovieResolution) -> Animation {
        self.resolution = resolution;
        self
    }

    /// How long one pass through the poses takes, in seconds.
    pub fn period_s(&self) -> f64 {
        self.poses.len() as f64 / self.fps as f64
    }

    /// Whether the host can carry this out. The host checks; a plugin can too.
    pub fn validate(&self) -> Result<(), PluginError> {
        if self.poses.is_empty() {
            return Err(PluginError::failed("the animation has no poses"));
        }
        if self.poses.len() > Self::MAX_POSES {
            return Err(PluginError::failed(format!(
                "the animation has {} poses, more than the {} a host will take",
                self.poses.len(),
                Self::MAX_POSES
            )));
        }
        if !Self::FPS_RANGE.contains(&self.fps) {
            return Err(PluginError::failed(format!(
                "the animation's frame rate, {} per second, is outside {}..={}",
                self.fps,
                Self::FPS_RANGE.start(),
                Self::FPS_RANGE.end()
            )));
        }
        if let Some(i) = self
            .poses
            .iter()
            .position(|p| !p.is_rotation(Self::TOLERANCE))
        {
            return Err(PluginError::failed(format!(
                "pose {i} of the animation is not a rotation: it would stretch, skew \
                 or mirror the volume"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "animation_tests.rs"]
mod tests;

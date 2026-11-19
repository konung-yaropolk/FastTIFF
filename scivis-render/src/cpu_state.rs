//! What both backends keep on the CPU side of their GPU resources.
//!
//! Each backend once kept this for itself, field for field alike: which
//! channel textures are allocated for what, and of the volume what is
//! uploaded, the params it is drawn with, the occupancy worked out for them
//! and the switches the host set. The two copies of the logic over it —
//! when to reallocate, when the occupancy is stale, what the acceleration
//! uniform says — were the same lines in both, and kept the same by hand.
//! Here they are once; each backend holds its GPU objects beside them.

use crate::pack::{self, BrickMaxima, Visibility, VolumeLayout};
use crate::{ChannelKind, VolumeInterp, VolumeParams, MAX_CHANNELS};

/// What a channel's two 2D textures are allocated as: an integer one (8-bit
/// or 16-bit) and a float one, of which the one carrying the channel's data
/// is the image's size and the other a 1x1 dummy. Both are 1x1 for a channel
/// that is not there; the dummy integer texture is a 16-bit one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChannelTextures {
    /// The integer texture holds 8-bit samples, rather than 16-bit ones.
    pub(crate) int_8bit: bool,
    pub(crate) int_size: (u32, u32),
    pub(crate) float_size: (u32, u32),
}

impl ChannelTextures {
    /// The textures of a channel of `kind` (`None`: not there) of a `size`
    /// image.
    pub(crate) fn of(kind: Option<ChannelKind>, size: (u32, u32)) -> ChannelTextures {
        let (int_8bit, int_size, float_size) = match kind {
            Some(ChannelKind::Int8) => (true, size, (1, 1)),
            Some(ChannelKind::Int16) => (false, size, (1, 1)),
            Some(ChannelKind::Float) => (false, (1, 1), size),
            None => (false, (1, 1), (1, 1)),
        };
        ChannelTextures {
            int_8bit,
            int_size,
            float_size,
        }
    }
}

/// The image size and channel kinds the 2D textures are allocated for.
pub(crate) struct ChannelAllocation {
    size: (u32, u32),
    /// `None` for a channel that is not there, whose textures are dummies.
    kinds: [Option<ChannelKind>; MAX_CHANNELS],
}

impl Default for ChannelAllocation {
    /// What a backend allocates when it is made: dummies everywhere.
    fn default() -> Self {
        ChannelAllocation {
            size: (1, 1),
            kinds: [None; MAX_CHANNELS],
        }
    }
}

impl ChannelAllocation {
    /// Record a `width` x `height` image of `kinds` (the first
    /// [`MAX_CHANNELS`] of them), and say whether that is not what was
    /// allocated: then the backend remakes the textures, as
    /// [`textures`](Self::textures) says. `false` when nothing changed, which
    /// is the every-frame case and costs a comparison.
    ///
    /// A `bool`, not the textures in an `Option`, and inlined into each
    /// backend's `ensure_size`, because that is the form the comparison had
    /// when each backend wrote it out itself. Under the release profile's fat
    /// LTO, returning the 120-byte array in an `Option` made the no-op 0.3 ns
    /// slower than the copies it replaced (4.7 ns on wgpu); as a `bool` it is
    /// a nanosecond faster than they were, on wgpu and on glow.
    #[inline]
    pub(crate) fn update(&mut self, width: u32, height: u32, kinds: &[ChannelKind]) -> bool {
        let want: [Option<ChannelKind>; MAX_CHANNELS] =
            std::array::from_fn(|c| kinds.get(c).copied());
        if self.size == (width, height) && self.kinds == want {
            return false;
        }
        self.size = (width, height);
        self.kinds = want;
        true
    }

    /// What each channel's textures are to be, for what is recorded.
    pub(crate) fn textures(&self) -> [ChannelTextures; MAX_CHANNELS] {
        self.kinds.map(|kind| ChannelTextures::of(kind, self.size))
    }
}

/// Which bricks of the volume can be seen, as the occupancy texture holds
/// it: one byte per brick (see [`pack::occupancy`]), `dims` bricks a side,
/// worked out for `vis`.
pub(crate) struct Occupancy {
    pub(crate) bytes: Vec<u8>,
    pub(crate) dims: (u32, u32, u32),
    pub(crate) vis: Visibility,
}

/// What a backend knows of its volume on the CPU side: what is uploaded, the
/// params it is drawn with, and the switches the host set.
pub(crate) struct VolumeState {
    /// The uploaded volume's layout and brick maxima; `None` when
    /// placeholders are bound.
    pub(crate) layout: Option<VolumeLayout>,
    pub(crate) bricks: Option<BrickMaxima>,
    /// What the occupancy texture was last worked out for.
    pub(crate) occupancy_for: Option<Visibility>,
    /// The ray-march params, stashed by `set_volume_params` for the paint.
    pub(crate) params: Option<VolumeParams>,
    pub(crate) interp: VolumeInterp,
    /// Empty-space skipping and mip levels, both on unless switched off.
    pub(crate) skip_empty: bool,
    pub(crate) mip_levels: bool,
    /// How many times smaller per axis to render the view (at least 1).
    pub(crate) downscale: u32,
}

impl Default for VolumeState {
    fn default() -> Self {
        VolumeState {
            layout: None,
            bricks: None,
            occupancy_for: None,
            params: None,
            interp: VolumeInterp::Linear,
            skip_empty: true,
            mip_levels: true,
            downscale: 1,
        }
    }
}

impl VolumeState {
    /// Bytes the uploaded volume holds on the GPU, every level of every
    /// texture: what a new volume would free, since it replaces this one.
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.layout.as_ref().map_or(0, |l| l.total_bytes() as u64)
    }

    /// Forget the volume: placeholders are bound now.
    pub(crate) fn forget(&mut self) {
        self.layout = None;
        self.bricks = None;
        self.occupancy_for = None;
    }

    /// Which bricks can be seen under the current params, when that is not
    /// what the occupancy texture holds; `None` when it is, or there is no
    /// volume or no params yet. Record its `vis` in
    /// [`occupancy_for`](Self::occupancy_for) once it is uploaded.
    pub(crate) fn stale_occupancy(&self) -> Option<Occupancy> {
        let (bricks, params) = (self.bricks.as_ref()?, self.params.as_ref()?);
        let vis = Visibility::of(params);
        if self.occupancy_for == Some(vis) {
            return None;
        }
        Some(Occupancy {
            bytes: pack::occupancy(bricks, &vis),
            dims: bricks.dims,
            vis,
        })
    }

    /// The params a capture of `width` x `height` pixels renders, when one
    /// can start: there are params to render, the picture is within the
    /// device's 2D limit of `max_2d` a side (each tile is a 2D render
    /// target), and no capture is `busy` still to be taken. What each backend
    /// checks beyond this (wgpu: that its target format reads back) is its
    /// own.
    pub(crate) fn capture_params(
        &self,
        width: u32,
        height: u32,
        max_2d: u32,
        busy: bool,
    ) -> Result<VolumeParams, String> {
        let Some(params) = self.params else {
            return Err("there is no volume to render".into());
        };
        if width == 0 || height == 0 || width > max_2d || height > max_2d {
            return Err(format!(
                "a {width} x {height} picture is beyond what this GPU can render (up to {max_2d} a side)"
            ));
        }
        if busy {
            return Err("the last capture has not been taken yet".into());
        }
        Ok(params)
    }

    /// The volume shaders' `accel` uniform for drawing `p` `height` pixels
    /// tall (see [`crate::uniforms::volume_accel`]). Empty space is skipped
    /// only where the occupancy was worked out for `p`'s own windows: a
    /// capture renders the params it was asked for over several frames, while
    /// the view's change underneath it.
    pub(crate) fn accel(&self, p: &VolumeParams, height: u32) -> [f32; 4] {
        let levels = self.layout.as_ref().map_or(1, |l| l.levels);
        crate::uniforms::volume_accel(
            p,
            height,
            levels,
            self.mip_levels,
            self.skip_empty && self.occupancy_for == Some(Visibility::of(p)),
        )
    }
}

#[cfg(test)]
#[path = "cpu_state_tests.rs"]
mod tests;

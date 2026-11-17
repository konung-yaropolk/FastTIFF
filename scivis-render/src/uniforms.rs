//! What the two backends hand their shaders, worked out once on the CPU.
//!
//! Each backend used to pack these for itself, line for line alike: the
//! composite's per-channel block, the LUT's RGB-to-RGBA expansion and the
//! volume's acceleration vector. A change made to one copy and not the other
//! drew differently on the two backends with nothing to say so. Here they
//! are plain data both consume. The composite's block is `#[repr(C)]` and
//! laid out as both shading languages declare it — `Params` in
//! `composite.wgsl`, written byte for byte into a uniform buffer by the wgpu
//! backend, and `ch_params` in `composite.frag.glsl`, set as one `vec4` array
//! by the glow backend — so neither takes it apart again.
//!
//! The composite's block is public, so that a host or a test double can see
//! exactly what reaches the GPU for what it passed
//! [`set_params`](crate::ImageRenderer::set_params): FastTIFF's viewer
//! records it in the double its sync is tested with. Its padding is not, so
//! [`CompositeParams::new`] (or `Default`) is the only way to make one. The
//! LUT row and the acceleration vector are crate-private: they are the
//! backends' business, and a host hands over a [`Lut`] and [`VolumeParams`].

use crate::{ChannelUniform, Lut, VolumeParams, MAX_CHANNELS};

/// One channel of the 2D composite, as the shaders read it: `ChannelParams`
/// in `composite.wgsl`, one `vec4` of `ch_params` in `composite.frag.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    any(feature = "backend-wgpu", feature = "backend-glow"),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct ChannelParams {
    /// The window, in the units of the channel's texture.
    pub min_max: [f32; 2],
    /// 1.0 on, 0.0 off.
    pub enabled: f32,
    /// 1.0 if the channel's data is in its float texture, 0.0 if in its
    /// integer one; the other is a 1x1 dummy.
    pub is_float: f32,
}

/// What a channel the host did not give is sent as: off, and integer. Its
/// window is never applied; any with a span the shader need not clamp will
/// do.
const NO_CHANNEL: ChannelParams = ChannelParams {
    min_max: [0.0, 65535.0],
    enabled: 0.0,
    is_float: 0.0,
};

/// The whole 2D composite's uniform: `Params` in `composite.wgsl`, field for
/// field (`src/wgpu_backend_tests.rs` holds the two to it). The glow backend
/// sets `channels` as `ch_params` and the rest as uniforms of their own; the
/// padding is the WGSL struct's, a `vec4` boundary.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    any(feature = "backend-wgpu", feature = "backend-glow"),
    derive(bytemuck::Pod, bytemuck::Zeroable)
)]
pub struct CompositeParams {
    pub channels: [ChannelParams; MAX_CHANNELS],
    /// The visible part of the image: `sampled_uv = uv_offset + uv * uv_scale`.
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
    pub num_channels: u32,
    // Crate-visible, for the layout test against naga (`wgpu_backend_tests.rs`).
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

impl CompositeParams {
    /// The block for what
    /// [`ImageRenderer::set_params`](crate::ImageRenderer::set_params) is
    /// given: the first [`MAX_CHANNELS`] of `channels`, the rest off.
    /// Inlined, as it is built on every frame's `set_params`, where each
    /// backend once built it in place.
    #[inline]
    pub fn new(
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) -> CompositeParams {
        let mut block = CompositeParams {
            channels: [NO_CHANNEL; MAX_CHANNELS],
            uv_offset,
            uv_scale,
            num_channels,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        for (slot, c) in block.channels.iter_mut().zip(channels) {
            *slot = ChannelParams {
                min_max: [c.min, c.max],
                enabled: if c.enabled { 1.0 } else { 0.0 },
                is_float: if c.is_float { 1.0 } else { 0.0 },
            };
        }
        block
    }
}

impl Default for CompositeParams {
    /// Nothing to composite, the whole image in view.
    fn default() -> Self {
        CompositeParams::new(&[], 0, [0.0, 0.0], [1.0, 1.0])
    }
}

/// Entries in a LUT, which is the width of a row of the LUT texture.
///
/// This, [`lut_rgba`] and [`volume_accel`] are used by the backends alone, so
/// a build with neither has no caller; their tests run in every build.
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) const LUT_WIDTH: usize = 256;

/// `lut` as the LUT texture holds it: a row of RGBA8 texels, opaque. On the
/// stack, as it is sent on every LUT upload; inlined, so the row is built
/// where the upload reads it rather than copied there.
///
/// Written as indexed stores into a zeroed row, which is the form the
/// backends had: measured on glow, a 255-filled row written a texel at a time
/// made each LUT upload some 60 ns slower, the same work in isolation.
#[inline]
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn lut_rgba(lut: &Lut) -> [u8; LUT_WIDTH * 4] {
    let mut rgba = [0u8; LUT_WIDTH * 4];
    for (i, px) in lut.iter().enumerate() {
        rgba[i * 4] = px[0];
        rgba[i * 4 + 1] = px[1];
        rgba[i * 4 + 2] = px[2];
        rgba[i * 4 + 3] = 255;
    }
    rgba
}

/// The volume shaders' `accel` uniform for a render `height` pixels tall:
/// the size of one pixel at unit distance, the highest mip level a maximum
/// projection may read, the brick size, and whether to skip empty bricks.
///
/// A height of 0 means the size of a pixel is not known yet (the wgpu
/// backend before the host has set a target), and then no coarser level than
/// the finest is read. A capture tile's narrowed params give the same pixel
/// size as the whole picture's (see [`crate::tile_params`]).
#[cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]
pub(crate) fn volume_accel(
    p: &VolumeParams,
    height: u32,
    levels: u32,
    mip_levels: bool,
    skip_empty: bool,
) -> [f32; 4] {
    [
        2.0 * p.tan_half_fov / height.max(1) as f32,
        if mip_levels && height > 0 {
            levels.saturating_sub(1) as f32
        } else {
            0.0
        },
        crate::pack::BRICK as f32,
        if skip_empty { 1.0 } else { 0.0 },
    ]
}

#[cfg(test)]
#[path = "uniforms_tests.rs"]
mod tests;

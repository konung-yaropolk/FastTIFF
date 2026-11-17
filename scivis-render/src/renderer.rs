//! What every backend does, as two traits a host is generic over.
//!
//! The backends were once kept in step by hand: each had an
//! `ImageRenderResources` of its own with some thirty inherent methods of the
//! same names and signatures, and nothing tied them together. A host picked
//! one with a `#[cfg]` re-export, and only a build of that host with the other
//! backend noticed when the two drifted. Now both implement [`ImageRenderer`]
//! and [`VolumeRenderer`], so a host's per-frame code is written once against
//! the traits — FastTIFF's whole decode-to-GPU sync is — and compiles with no
//! backend at all, which is also what lets it be tested against a double with
//! no GPU.
//!
//! Static dispatch throughout: a host's code is generic over the traits and is
//! compiled for the one backend it hands in, so a call costs what the inherent
//! method it replaced did. Nothing here is meant as `dyn`.
//!
//! What is not in a trait is what cannot share a signature: construction (a
//! wgpu device and queue, or a GL context), painting (into a wgpu render pass,
//! or with a live GL context), and the wgpu backend's `prepare_volume`, which
//! records into the host's command encoder. Those stay inherent, and are named
//! only where a host hands the renderer to its drawing layer.

use crate::pack::{self, PackedVolume, TexelFormat, VolumeLayout};
use crate::{
    ChannelKind, ChannelUniform, Lut, VolumeImage, VolumeInterp, VolumeKind, VolumeParams,
    VolumeUploadError,
};

/// The 2D composite: up to [`MAX_CHANNELS`](crate::MAX_CHANNELS) channels of
/// an image, each windowed and coloured through its own LUT, and summed.
///
/// Three tiers, from least to most frequent: [`ensure_size`] when the image's
/// shape or channel layout changes, [`upload_lut`] when a channel's colours
/// do, and the uploads and [`set_params`] every frame. `ensure_size`
/// reallocates the channel textures, so every channel's pixels are to be
/// uploaded again after it changed anything; the LUTs it leaves as they are.
///
/// Every call that changes what the renderer holds takes `&mut self`, the
/// uploads included. wgpu and glow only queue those writes, and so could take
/// `&self`, but that is a property of those two and not of a renderer: one
/// that keeps the pixels itself (a CPU compositor, the recording double
/// FastTIFF's viewer tests its sync with) would need interior mutability for
/// it, and would no longer be `Sync`. Every caller holds the renderer
/// mutably anyway, as it also sizes and parameterizes it.
///
/// [`ensure_size`]: Self::ensure_size
/// [`upload_lut`]: Self::upload_lut
/// [`set_params`]: Self::set_params
pub trait ImageRenderer {
    /// Largest per-axis 2D-texture dimension the device supports.
    ///
    /// A frame wider or taller than this cannot be uploaded at all: wgpu
    /// rejects the texture and, its errors being fatal by default, takes the
    /// process with it. A host subsamples or windows the frame to fit
    /// instead, as FastTIFF's viewer does (`fast-tiff-viewer/src/roi.rs`).
    fn max_2d_texture_size(&self) -> u32;

    /// Allocate textures for a `width` x `height` image with one
    /// [`ChannelKind`] per display channel. An `Int8` or `Int16` channel gets
    /// a full-size R8Uint or R16Uint texture and a `Float` one a full-size
    /// R32F texture, the other of its pair a 1x1 dummy; channels past
    /// `kinds.len()` are unused (both 1x1). No-op when nothing changed, so a
    /// host may call it every frame; otherwise every uploaded pixel is gone.
    fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind]);

    /// Upload one `Int8` channel's raw 8-bit samples, `width * height` of
    /// them, row-major. No widening on the CPU: the window is in 0..255
    /// units instead. Each upload must match the kind and size
    /// [`ensure_size`](Self::ensure_size) was given for its channel; a
    /// channel past `MAX_CHANNELS` is ignored.
    fn upload_channel_u8(&mut self, channel: usize, width: u32, height: u32, samples: &[u8]);

    /// Upload one `Int16` channel's raw 16-bit samples; as
    /// [`upload_channel_u8`](Self::upload_channel_u8).
    fn upload_channel_u16(&mut self, channel: usize, width: u32, height: u32, samples: &[u16]);

    /// Upload one `Float` channel's raw 32-bit float samples; as
    /// [`upload_channel_u8`](Self::upload_channel_u8).
    fn upload_channel_f32(&mut self, channel: usize, width: u32, height: u32, samples: &[f32]);

    /// Set channel `channel`'s LUT. Sticky: it stays until replaced. The 3D
    /// volume is coloured through the same LUTs.
    fn upload_lut(&mut self, channel: usize, lut: &Lut);

    /// This frame's per-channel window/level, on/off and float flags, how
    /// many channels to composite, and the visible part of the image:
    /// `sampled_uv = uv_offset + uv * uv_scale`, so `([0, 0], [1, 1])` shows
    /// all of it. The windows are in the units of each channel's texture: raw
    /// 0..255 for `Int8`, raw 0..65535 for `Int16`, the data's own for
    /// `Float`.
    fn set_params(
        &mut self,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    );
}

/// The 3D view: a volume packed by [`crate::pack`], ray-marched through the
/// channels' LUTs.
///
/// A supertrait of [`ImageRenderer`] because the volume has no colours of its
/// own: it is drawn through the LUTs [`ImageRenderer::upload_lut`] uploads.
pub trait VolumeRenderer: ImageRenderer {
    /// What this device can hold of a volume: its size limit, and the texel
    /// format each kind of sample is stored as. Fixed for the renderer's
    /// life, and cheap.
    fn volume_caps(&self) -> VolumeCaps;

    /// GPU memory a new volume could take, in bytes: what the driver says is
    /// left of this process's budget, plus what the current volume holds (a
    /// new one replaces it). `None` when the driver will not say — see each
    /// backend for where — or the GPU is integrated, whose memory is the
    /// host's RAM. A builder should take a share of it, not all of it.
    ///
    /// Asks the driver on every call, which can cost a system call or two:
    /// ask once per volume, not once per frame.
    fn volume_memory_available(&self) -> Option<u64>;

    /// Upload a volume packed by [`crate::pack`], every level of every
    /// texture, and the brick maxima empty space is skipped with. A volume of
    /// the shape and format already there (a 4D timepoint step) refills it
    /// in place.
    ///
    /// Refused as [`VolumeUploadError::Unsupported`] when it is not one
    /// [`VolumeCaps::check`] passes, or its buffers are not the sizes its
    /// layout says ([`PackedVolume::check`]). Fails with
    /// [`VolumeUploadError::OutOfMemory`] when the GPU had no room, leaving
    /// 1x1x1 placeholders and no volume: build a smaller one and try again.
    fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError>;

    /// (Re)upload a volume given one buffer per channel — `w * h * d` samples
    /// of its `kind`, native-endian, x fastest — packed here, on the calling
    /// thread, as [`volume_caps`](Self::volume_caps) says to store it. A
    /// builder with a thread of its own should pack there instead and call
    /// [`upload_packed_volume`](Self::upload_packed_volume).
    fn upload_volumes(
        &mut self,
        w: u32,
        h: u32,
        d: u32,
        channels: &[(VolumeKind, Vec<u8>)],
    ) -> Result<(), VolumeUploadError> {
        let kinds: Vec<VolumeKind> = channels.iter().map(|(k, _)| *k).collect();
        let format = self.volume_caps().texel_format(pack::common_kind(&kinds));
        let borrowed: Vec<(VolumeKind, &[u8])> =
            channels.iter().map(|(k, b)| (*k, b.as_slice())).collect();
        self.upload_packed_volume(&PackedVolume::from_channels((w, h, d), &borrowed, format))
    }

    /// Give the volume's GPU memory back: 1x1x1 placeholders in its place,
    /// until the next upload makes a new one.
    ///
    /// For a host whose volume has no further use — the stack it was built
    /// from closed or replaced. Without it the textures stayed resident until
    /// the next volume arrived, which for a session that never returned to 3D
    /// was never: up to half the free VRAM of a 2 GB card, held for nothing.
    fn release_volume(&mut self);

    /// An upload that returned `Ok` and was then refused for want of memory,
    /// and the bytes its volume took. Only WebGPU does this: it says whether
    /// there was room through a promise an upload cannot wait for. The volume
    /// is gone, as after an upload that failed, and `None` is returned from
    /// then on: build a smaller one. Ask once a frame; until the answer is
    /// in, the volume is not drawn.
    fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)>;

    /// Whether the volume is waiting for the device to say there was room for
    /// it (WebGPU only; see [`take_late_upload_error`]). It is not drawn until
    /// then, and nothing but the host can ask for the frame that draws it:
    /// keep frames coming while this is true.
    ///
    /// [`take_late_upload_error`]: Self::take_late_upload_error
    fn volume_pending(&mut self) -> bool;

    /// How the volume is sampled. Cheap and idempotent: set it every 3D
    /// frame.
    fn set_volume_interp(&mut self, interp: VolumeInterp);

    /// Switch the two accelerations on or off: skipping parts of the volume
    /// that cannot be seen (exact: the picture is the same either way) and,
    /// in the maximum projection, reading coarser mip levels where a voxel is
    /// smaller than a pixel. Both are on unless switched off; off is for
    /// comparing, measuring, or ruling them out.
    fn set_volume_acceleration(&mut self, skip_empty: bool, mip_levels: bool);

    /// The size the 3D view is painted at, in pixels, and how many times
    /// smaller per axis to render it: 1 renders every pixel, and more renders
    /// a reduced picture the paint scales up — for a view that is moving,
    /// where a sharp frame is wasted (see [`crate::MOVING_DOWNSCALE`]). Set
    /// every frame. The glow backend takes the size from the viewport the
    /// host has set when it paints, and uses only `downscale`.
    fn set_volume_target(&mut self, width: u32, height: u32, downscale: u32);

    /// This frame's camera, per-channel windows and mode. Cheap: set it every
    /// 3D frame.
    fn set_volume_params(&mut self, params: VolumeParams);

    /// Render the volume off screen at `width` x `height` pixels, with the
    /// params last given to [`set_volume_params`](Self::set_volume_params),
    /// and read it back; collect the picture with
    /// [`take_volume_capture`](Self::take_volume_capture). Every pixel is a
    /// ray, so a larger capture has more detail rather than larger pixels;
    /// the aspect ratio is the size's. Rendered in tiles of
    /// [`CAPTURE_TILE`](crate::CAPTURE_TILE) pixels, so that no one GPU job
    /// runs long enough for the system to reset the GPU. One at a time.
    fn capture_volume(&mut self, width: u32, height: u32) -> Result<(), String>;

    /// The picture [`capture_volume`](Self::capture_volume) asked for, once
    /// it is all in: `None` while it is still being rendered (wgpu reads back
    /// a frame or more later, glow at once), or when none was asked for. Call
    /// once a frame until it answers.
    fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>>;
}

/// What a backend's device can hold of a volume: what a builder must know
/// before it packs one, and what an upload is checked against.
///
/// The backends store volumes differently on purpose, and this is where it is
/// said. Both read a volume through a filtering sampler (trilinear and
/// tricubic sampling need it). OpenGL 3 filters 16-bit unorm and 32-bit float
/// textures as core, so the glow backend keeps every kind at its own width.
/// wgpu filters a 16-bit unorm only under `TEXTURE_FORMAT_16BIT_NORM`, which
/// the backend asks for where the adapter has it
/// (`wgpu_backend::optional_features`), and a full float only under
/// `FLOAT32_FILTERABLE`, which it does not ask for: many mobile GPUs and most
/// browsers lack it, and a half float is enough for a volume normalized to
/// `0..=1`, as the FastTIFF viewer normalizes them. So on wgpu a float volume
/// is a half float, and a 16-bit one too without the feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeCaps {
    /// Largest per-axis 3D-texture dimension the device supports. A volume
    /// past it is refused: reduce it to fit before packing, there is no
    /// automatic downscale.
    pub max_3d_size: u32,
    /// 16-bit normalized textures can be sampled filtered.
    pub unorm16: bool,
    /// Full 32-bit float textures can be sampled filtered.
    pub float32: bool,
}

impl VolumeCaps {
    /// How a volume whose channels are `kind` is stored: what a builder
    /// should pack it as (use [`pack::common_kind`] for mixed channels).
    /// 8-bit stays 8-bit; 16-bit is a 16-bit unorm where the device filters
    /// one and a half float where it does not; float is a full float where
    /// the device filters one, else a half float.
    pub fn texel_format(&self, kind: VolumeKind) -> TexelFormat {
        match kind {
            VolumeKind::U8 => TexelFormat::Unorm8,
            VolumeKind::U16 if self.unorm16 => TexelFormat::Unorm16,
            VolumeKind::F32 if self.float32 => TexelFormat::Float32,
            VolumeKind::U16 | VolumeKind::F32 => TexelFormat::Float16,
        }
    }

    /// Whether a volume of `layout` can be uploaded: stored in a format
    /// [`texel_format`](Self::texel_format) gives for some kind — which is
    /// all a backend allocates textures for — and within the size limit.
    pub fn check(&self, layout: &VolumeLayout) -> Result<(), VolumeUploadError> {
        let stored =
            [VolumeKind::U8, VolumeKind::U16, VolumeKind::F32].map(|k| self.texel_format(k));
        if !stored.contains(&layout.format) {
            return Err(VolumeUploadError::Unsupported(format!(
                "this GPU stores 8-bit, 16-bit and float volumes as {:?}, {:?} and {:?}, \
                 not {:?}: pack it as VolumeCaps::texel_format says",
                stored[0], stored[1], stored[2], layout.format
            )));
        }
        let max = self.max_3d_size;
        let (w, h, d) = layout.dims;
        if w > max || h > max || d > max {
            return Err(VolumeUploadError::Unsupported(format!(
                "a {w} x {h} x {d} volume is beyond this GPU's 3D texture limit of {max}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "renderer_tests.rs"]
mod tests;

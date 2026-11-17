//! [`VolumeCaps`]: how each backend stores a volume, and what an upload is
//! held to.
//!
//! The two backends store volumes differently on purpose: each used to answer
//! `volume_texel_format` and refuse uploads with a policy of its own. The
//! policy is one function of the kind and the caps now, and these pin it, for
//! the caps each backend reports written out by hand below, as each backend's
//! own policy was, so moving them here changed nothing. What a real backend
//! reports is set in its `new`, from its device, and `tests/it/caps.rs` holds
//! each, on a real device, to the flags written here for it.

use super::*;
use crate::pack::{BrickMaxima, PackedVolume, TexelFormat, VolumeLayout};
use crate::{ChannelKind, VolumeImage, VolumeInterp, VolumeParams};

/// What the glow backend reports: OpenGL 3 filters every kind at its own
/// width.
const GLOW: VolumeCaps = VolumeCaps {
    max_3d_size: 2048,
    unorm16: true,
    float32: true,
};

/// What the wgpu backend reports on a device with
/// `TEXTURE_FORMAT_16BIT_NORM` (most desktops), and without (WebGL, some
/// mobile GPUs). It never asks for `FLOAT32_FILTERABLE`.
const WGPU: VolumeCaps = VolumeCaps {
    max_3d_size: 2048,
    unorm16: true,
    float32: false,
};
const WGPU_NO_UNORM16: VolumeCaps = VolumeCaps {
    unorm16: false,
    ..WGPU
};

const KINDS: [VolumeKind; 3] = [VolumeKind::U8, VolumeKind::U16, VolumeKind::F32];

/// Each backend's `volume_texel_format` as it was, kind by kind.
#[test]
fn each_backend_stores_a_volume_as_it_did() {
    use TexelFormat::*;
    let table = |caps: VolumeCaps| KINDS.map(|k| caps.texel_format(k));
    assert_eq!(table(GLOW), [Unorm8, Unorm16, Float32], "glow");
    assert_eq!(table(WGPU), [Unorm8, Unorm16, Float16], "wgpu");
    assert_eq!(
        table(WGPU_NO_UNORM16),
        [Unorm8, Float16, Float16],
        "wgpu without 16-bit norm"
    );
}

fn layout(dims: (u32, u32, u32), format: TexelFormat) -> VolumeLayout {
    VolumeLayout::new(dims, 2, format)
}

/// An upload is refused in a format the device does not store volumes in —
/// which is all a backend allocates textures for — and past its 3D limit on
/// any axis; it passes in every format it stores, up to the limit.
#[test]
fn an_upload_is_held_to_what_the_device_stores() {
    use TexelFormat::*;
    let unsupported =
        |r: Result<(), VolumeUploadError>| matches!(r, Err(VolumeUploadError::Unsupported(_)));
    for (caps, refused) in [
        (GLOW, vec![Float16]),
        (WGPU, vec![Float32]),
        (WGPU_NO_UNORM16, vec![Unorm16, Float32]),
    ] {
        for format in [Unorm8, Unorm16, Float16, Float32] {
            let r = caps.check(&layout((8, 8, 8), format));
            assert_eq!(
                unsupported(r.clone()),
                refused.contains(&format),
                "{caps:?} with {format:?}: {r:?}"
            );
        }
        let max = caps.max_3d_size;
        assert_eq!(caps.check(&layout((max, 1, max), Unorm8)), Ok(()));
        for dims in [(max + 1, 1, 1), (1, max + 1, 1), (1, 1, max + 1)] {
            assert!(unsupported(caps.check(&layout(dims, Unorm8))), "{dims:?}");
        }
    }
}

/// A renderer that keeps what it is handed: enough of one to see what the
/// trait's own `upload_volumes` packs.
struct Packs {
    caps: VolumeCaps,
    uploaded: Option<(VolumeLayout, BrickMaxima)>,
}

impl ImageRenderer for Packs {
    fn max_2d_texture_size(&self) -> u32 {
        4096
    }
    fn ensure_size(&mut self, _: u32, _: u32, _: &[ChannelKind]) {}
    fn upload_channel_u8(&mut self, _: usize, _: u32, _: u32, _: &[u8]) {}
    fn upload_channel_u16(&mut self, _: usize, _: u32, _: u32, _: &[u16]) {}
    fn upload_channel_f32(&mut self, _: usize, _: u32, _: u32, _: &[f32]) {}
    fn upload_lut(&mut self, _: usize, _: &Lut) {}
    fn set_params(&mut self, _: &[ChannelUniform], _: u32, _: [f32; 2], _: [f32; 2]) {}
}

impl VolumeRenderer for Packs {
    fn volume_caps(&self) -> VolumeCaps {
        self.caps
    }
    fn volume_memory_available(&self) -> Option<u64> {
        None
    }
    fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError> {
        volume.check().map_err(VolumeUploadError::Unsupported)?;
        self.caps.check(&volume.layout)?;
        self.uploaded = Some((volume.layout.clone(), volume.bricks.clone()));
        Ok(())
    }
    fn release_volume(&mut self) {}
    fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)> {
        None
    }
    fn volume_pending(&mut self) -> bool {
        false
    }
    fn set_volume_interp(&mut self, _: VolumeInterp) {}
    fn set_volume_acceleration(&mut self, _: bool, _: bool) {}
    fn set_volume_target(&mut self, _: u32, _: u32, _: u32) {}
    fn set_volume_params(&mut self, _: VolumeParams) {}
    fn capture_volume(&mut self, _: u32, _: u32) -> Result<(), String> {
        Ok(())
    }
    fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>> {
        None
    }
}

/// `upload_volumes` packs whole channels as the device stores their common
/// kind, which is what each backend's own copy of it did, and uploads a
/// volume that passes the checks an upload makes.
#[test]
fn upload_volumes_packs_as_the_device_stores() {
    let (w, h, d) = (5, 4, 3);
    let voxels = (w * h * d) as usize;
    let u16s = vec![0u8; voxels * 2];
    let f32s = vec![0u8; voxels * 4];
    for (caps, want) in [
        (GLOW, TexelFormat::Float32),
        (WGPU, TexelFormat::Float16),
        (WGPU_NO_UNORM16, TexelFormat::Float16),
    ] {
        let mut r = Packs {
            caps,
            uploaded: None,
        };
        // A 16-bit and a float channel: float is what they have in common.
        let channels = [
            (VolumeKind::U16, u16s.clone()),
            (VolumeKind::F32, f32s.clone()),
        ];
        r.upload_volumes(w, h, d, &channels).expect("uploads");
        let (layout, bricks) = r.uploaded.expect("uploaded");
        assert_eq!(
            (layout.dims, layout.channels, layout.format),
            ((w, h, d), 2, want),
            "{caps:?}"
        );
        assert_eq!(bricks.max.len(), 2);
    }
}

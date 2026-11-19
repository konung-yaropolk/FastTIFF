//! [`Recorder`]: a renderer that keeps a list of what it is sent, for testing
//! [`Viewer::sync`](crate::Viewer::sync) with no GPU.
//!
//! The sync is generic over scivis-render's [`ImageRenderer`] and
//! [`VolumeRenderer`], so this stands where a backend would, and says
//! afterwards what reached it and in what order: which textures were sized,
//! which channel planes and LUTs were uploaded, what the shader was given, and
//! what the volume path asked. What it answers — the texture limits, how
//! volumes are stored, the free memory, whether an upload fits — is set by the
//! test, which is what lets the budget and the out-of-memory paths run here.
//!
//! It holds what it is sent to what a backend needs of it, and panics where a
//! backend would fail (a wgpu validation error is fatal) or the trait's
//! contract is broken: textures within the 2D limit it reports, a plane of
//! the kind and size [`ensure_size`](ImageRenderer::ensure_size) last gave
//! its channel, and a volume that passes the checks an upload makes. So every
//! test of the sync also checks the contract between the sync and a backend.

use scivis_render::pack::{PackedVolume, TexelFormat};
use scivis_render::uniforms::CompositeParams;
use scivis_render::{
    ChannelKind, ChannelUniform, ImageRenderer, Lut, VolumeCaps, VolumeImage, VolumeInterp,
    VolumeParams, VolumeRenderer, VolumeUploadError,
};
use std::cell::Cell;
use std::collections::VecDeque;

/// One call a [`Recorder`] was sent.
#[derive(Clone, Debug)]
pub(crate) enum Call {
    EnsureSize {
        width: u32,
        height: u32,
        kinds: Vec<ChannelKind>,
    },
    /// A channel plane: its texture's kind, size, and its first sample, which
    /// the tests' stacks make say which plane of the file it is.
    Upload {
        channel: usize,
        kind: ChannelKind,
        width: u32,
        height: u32,
        first: f64,
    },
    /// A channel's LUT, by its brightest entry, which tells a ramp from a
    /// tinted one.
    Lut {
        channel: usize,
        top: [u8; 3],
    },
    /// What the composite's shader is given, packed as both backends pack it.
    Params(CompositeParams),
    ReleaseVolume,
    /// An upload of a packed volume, refused or not.
    UploadVolume {
        dims: (u32, u32, u32),
        channels: usize,
        format: TexelFormat,
        bytes: usize,
        /// Each channel's voxel at the volume's origin, as stored (a unorm
        /// read back as `0..=1`), which the tests' stacks make say which
        /// plane of the file the first slice was built from. Empty for a
        /// volume whose buffers are not the sizes its layout says.
        corner: Vec<f32>,
    },
    VolumeInterp(VolumeInterp),
    VolumeTarget {
        width: u32,
        height: u32,
        downscale: u32,
    },
    VolumeParams(VolumeParams),
}

/// A renderer that records every call it is sent; see the [module docs](self).
pub(crate) struct Recorder {
    /// Read with [`take`](Self::take).
    pub(crate) calls: Vec<Call>,
    /// What the last `ensure_size` allocated: the size and each channel's
    /// kind. `None` before the first.
    pub(crate) allocated: Option<(u32, u32, Vec<ChannelKind>)>,
    /// How many times the free memory was asked for. A `Cell`, as a backend
    /// answers it through `&self`.
    pub(crate) memory_asks: Cell<usize>,
    /// The 2D-texture limit it reports.
    pub(crate) max_2d: u32,
    /// How it says it stores volumes.
    pub(crate) caps: VolumeCaps,
    /// The free GPU memory it reports.
    pub(crate) free: Option<u64>,
    /// What its next volume uploads return, in turn; `Ok` once these run out.
    pub(crate) upload_results: VecDeque<Result<(), VolumeUploadError>>,
    /// An error it reports once, late, through `take_late_upload_error`.
    pub(crate) late: Option<(VolumeUploadError, usize)>,
    /// Whether it says the volume is still waiting to hear there was room.
    pub(crate) pending: bool,
}

impl Default for Recorder {
    /// A device with room for any 2D frame the tests make, that stores
    /// volumes as the glow backend does, and will not say how much memory it
    /// has.
    fn default() -> Self {
        Recorder {
            calls: Vec::new(),
            allocated: None,
            memory_asks: Cell::new(0),
            max_2d: 16384,
            caps: VolumeCaps {
                max_3d_size: 2048,
                unorm16: true,
                float32: true,
            },
            free: None,
            upload_results: VecDeque::new(),
            late: None,
            pending: false,
        }
    }
}

impl Recorder {
    /// The calls since the last `take`, in order, and forget them.
    pub(crate) fn take(&mut self) -> Vec<Call> {
        std::mem::take(&mut self.calls)
    }

    /// Record a plane, after holding it to what a backend needs of one: as
    /// many samples as its size says, which a GPU would otherwise read past,
    /// and the kind and size of the texture `ensure_size` made its channel,
    /// which a backend writes it into. A plane of another kind or size breaks
    /// the trait's contract, and what a backend then does varies, from a
    /// validation error to garbled pixels, so the recorder refuses it outright.
    fn upload(
        &mut self,
        channel: usize,
        kind: ChannelKind,
        (width, height): (u32, u32),
        len: usize,
        first: impl FnOnce() -> f64,
    ) {
        assert_eq!(
            len,
            width as usize * height as usize,
            "channel {channel}: {len} samples for a {width} x {height} texture"
        );
        let Some((aw, ah, kinds)) = &self.allocated else {
            panic!("channel {channel} was uploaded before any ensure_size");
        };
        assert_eq!(
            (kinds.get(channel), width, height),
            (Some(&kind), *aw, *ah),
            "channel {channel}: a {kind:?} {width} x {height} plane, into the textures \
             ensure_size made for {kinds:?} at {aw} x {ah}"
        );
        self.calls.push(Call::Upload {
            channel,
            kind,
            width,
            height,
            first: first(),
        });
    }
}

impl ImageRenderer for Recorder {
    fn max_2d_texture_size(&self) -> u32 {
        self.max_2d
    }

    /// Held to the limit it reports: both backends refuse a texture past it
    /// (wgpu with a validation error, which is fatal), so a sync that sized
    /// one past the device's limit would fail on any GPU, uploaded to or not.
    fn ensure_size(&mut self, width: u32, height: u32, kinds: &[ChannelKind]) {
        assert!(
            (1..=self.max_2d).contains(&width) && (1..=self.max_2d).contains(&height),
            "ensure_size({width} x {height}) on a device whose 2D textures are at most \
             {0} x {0}",
            self.max_2d
        );
        self.allocated = Some((width, height, kinds.to_vec()));
        self.calls.push(Call::EnsureSize {
            width,
            height,
            kinds: kinds.to_vec(),
        });
    }

    fn upload_channel_u8(&mut self, channel: usize, width: u32, height: u32, samples: &[u8]) {
        self.upload(
            channel,
            ChannelKind::Int8,
            (width, height),
            samples.len(),
            || samples[0] as f64,
        );
    }

    fn upload_channel_u16(&mut self, channel: usize, width: u32, height: u32, samples: &[u16]) {
        self.upload(
            channel,
            ChannelKind::Int16,
            (width, height),
            samples.len(),
            || samples[0] as f64,
        );
    }

    fn upload_channel_f32(&mut self, channel: usize, width: u32, height: u32, samples: &[f32]) {
        self.upload(
            channel,
            ChannelKind::Float,
            (width, height),
            samples.len(),
            || samples[0] as f64,
        );
    }

    fn upload_lut(&mut self, channel: usize, lut: &Lut) {
        let top = *lut
            .iter()
            .max_by_key(|c| c.iter().map(|&v| v as u32).sum::<u32>())
            .expect("256 entries");
        self.calls.push(Call::Lut { channel, top });
    }

    fn set_params(
        &mut self,
        channels: &[ChannelUniform],
        num_channels: u32,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) {
        self.calls.push(Call::Params(CompositeParams::new(
            channels,
            num_channels,
            uv_offset,
            uv_scale,
        )));
    }
}

/// Each channel's voxel at `volume`'s origin, level 0, as stored; empty when
/// its buffers are not the sizes its layout says.
fn corner(volume: &PackedVolume) -> Vec<f32> {
    if volume.check().is_err() {
        return Vec::new();
    }
    let layout = &volume.layout;
    let bytes = layout.format.bytes();
    (0..layout.channels)
        .map(|c| {
            let (t, component) = layout.place(c);
            layout
                .format
                .decode(&volume.levels[t][0][component * bytes..])
        })
        .collect()
}

impl VolumeRenderer for Recorder {
    fn volume_caps(&self) -> VolumeCaps {
        self.caps
    }

    fn volume_memory_available(&self) -> Option<u64> {
        self.memory_asks.set(self.memory_asks.get() + 1);
        self.free
    }

    /// Checked as a backend checks it, then answered as the test said.
    fn upload_packed_volume(&mut self, volume: &PackedVolume) -> Result<(), VolumeUploadError> {
        let layout = &volume.layout;
        self.calls.push(Call::UploadVolume {
            dims: layout.dims,
            channels: layout.channels,
            format: layout.format,
            bytes: layout.total_bytes(),
            corner: corner(volume),
        });
        volume.check().map_err(VolumeUploadError::Unsupported)?;
        self.caps.check(layout)?;
        self.upload_results.pop_front().unwrap_or(Ok(()))
    }

    fn release_volume(&mut self) {
        self.calls.push(Call::ReleaseVolume);
    }

    fn take_late_upload_error(&mut self) -> Option<(VolumeUploadError, usize)> {
        self.late.take()
    }

    fn volume_pending(&mut self) -> bool {
        self.pending
    }

    fn set_volume_interp(&mut self, interp: VolumeInterp) {
        self.calls.push(Call::VolumeInterp(interp));
    }

    fn set_volume_acceleration(&mut self, _skip_empty: bool, _mip_levels: bool) {}

    fn set_volume_target(&mut self, width: u32, height: u32, downscale: u32) {
        self.calls.push(Call::VolumeTarget {
            width,
            height,
            downscale,
        });
    }

    fn set_volume_params(&mut self, params: VolumeParams) {
        self.calls.push(Call::VolumeParams(params));
    }

    fn capture_volume(&mut self, _width: u32, _height: u32) -> Result<(), String> {
        Err("the recorder renders nothing".into())
    }

    fn take_volume_capture(&mut self) -> Option<Result<VolumeImage, String>> {
        None
    }
}

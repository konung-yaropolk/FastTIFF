//! The plugin's side of the host callbacks: an [`FtHost`] wearing the ordinary
//! [`HostContext`] trait, so plugin code never sees a raw pointer.

use crate::abi::*;
use crate::api::{
    ChannelView, DisplayMode, HostContext, ImageInfo, PixelType, Plane, PluginError, Spacing,
    StackInfo, ViewParams, VolumeMode, VolumeView,
};

/// A [`HostContext`] backed by the host's C callbacks.
pub struct CHost {
    host: FtHost,
    image: ImageInfo,
    view: ViewParams,
    info: StackInfo,
}

impl CHost {
    /// Build from the host's table, reading everything that is snapshot-shaped
    /// once so a plugin's repeated `image()` calls do not cross the boundary.
    ///
    /// # Safety
    /// `host` must point at a valid `FtHost` whose callbacks remain valid for
    /// the lifetime of the returned value.
    pub unsafe fn new(host: *const FtHost) -> Result<CHost, FtStatus> {
        // `fits` reads only the four-byte prologue. The obvious spelling —
        // copy the table, then look at its `struct_size` — is undefined
        // behaviour when the host is older than this plugin: the copy reads
        // `size_of::<FtHost>()` bytes out of a shorter allocation, so the check
        // that was supposed to prevent that has already been overtaken by it.
        if !crate::abi::fits(host) {
            crate::last_error::set(
                "the host's callback table is older than this plugin's ABI expects",
            );
            return Err(FtStatus::BadArgument);
        }
        let h = *host;

        let mut ii = core::mem::zeroed::<FtImageInfo>();
        ii.struct_size = core::mem::size_of::<FtImageInfo>() as u32;
        if (h.image_info)(h.ctx, &mut ii) != FtStatus::Ok {
            return Err(FtStatus::Error);
        }
        let mut vp = core::mem::zeroed::<FtViewParams>();
        vp.struct_size = core::mem::size_of::<FtViewParams>() as u32;
        if (h.view_params)(h.ctx, &mut vp) != FtStatus::Ok {
            return Err(FtStatus::Error);
        }

        let mut channels = Vec::new();
        for i in 0..vp.shown_channels.min(64) {
            let mut cv = FtChannelView {
                // Ours to declare: the host writes only this many bytes.
                struct_size: core::mem::size_of::<FtChannelView>() as u32,
                _pad0: 0,
                min: 0.0,
                max: 0.0,
                enabled: 0,
                _pad: 0,
            };
            if (h.channel_view)(h.ctx, i, &mut cv) == FtStatus::Ok {
                channels.push(ChannelView {
                    min: cv.min,
                    max: cv.max,
                    enabled: cv.enabled != 0,
                });
            }
        }

        let name = (h.stack_name)(h.ctx).as_str().unwrap_or("").to_string();
        let path = (h.stack_path)(h.ctx).as_str().unwrap_or("").to_string();

        // The file's own units. A plugin that measures anything is quietly
        // wrong without these, so they are read up front with everything else
        // rather than left to a callback a plugin might not know to call.
        let mut si = core::mem::zeroed::<FtStackInfo>();
        si.struct_size = core::mem::size_of::<FtStackInfo>() as u32;
        let has_meta = (h.stack_info)(h.ctx, &mut si) == FtStatus::Ok;
        let opt = |flag: u32, v: f64| (has_meta && si.present & flag != 0).then_some(v);
        let string = |which: u32| {
            let s = (h.stack_string)(h.ctx, which, 0).as_str().unwrap_or("");
            (!s.is_empty()).then(|| s.to_string())
        };
        let mut channel_names = Vec::new();
        for i in 0..ii.channels.min(1024) {
            match (h.stack_string)(h.ctx, FT_STRING_CHANNEL_NAME, i).as_str() {
                Some(n) if !n.is_empty() => channel_names.push(n.to_string()),
                // Names run out; the api documents the list may be short.
                _ => break,
            }
        }

        Ok(CHost {
            image: ImageInfo {
                width: ii.width,
                height: ii.height,
                channels: ii.channels as usize,
                slices: ii.slices as usize,
                frames: ii.frames as usize,
                samples_per_pixel: ii.samples_per_pixel as u16,
                pixel_type: match ii.pixel_type {
                    FtPixelType::U8 => PixelType::U8,
                    FtPixelType::I16 => PixelType::I16,
                    FtPixelType::F32 => PixelType::F32,
                    _ => PixelType::U16,
                },
            },
            view: ViewParams {
                frame_index: vp.frame_index as usize,
                volume_view: vp.volume_view != 0,
                channels,
                // LUTs are not carried across the boundary in ABI 1. A plugin
                // that needs one can be added to a later minor without breaking
                // anything, because this struct is built here rather than
                // shared; leaving it empty is honest about what ABI 1 provides.
                luts: Vec::new(),
                volume: VolumeView {
                    mode: match vp.volume_mode {
                        1 => VolumeMode::Dvr,
                        2 => VolumeMode::Surface,
                        _ => VolumeMode::Mip,
                    },
                    density: vp.density,
                    iso: vp.iso,
                    eye: vp.eye,
                    forward: vp.forward,
                    up: vp.up,
                    right: vp.right,
                },
            },
            info: StackInfo {
                name,
                path: if path.is_empty() { None } else { Some(path) },
                mode: match si.mode {
                    1 => DisplayMode::Composite,
                    2 => DisplayMode::Color,
                    _ => DisplayMode::Grayscale,
                },
                unit: string(FT_STRING_UNIT),
                spacing: Spacing {
                    x: opt(FT_HAS_SPACING_X, si.spacing_x),
                    y: opt(FT_HAS_SPACING_Y, si.spacing_y),
                    z: opt(FT_HAS_SPACING_Z, si.spacing_z),
                },
                frame_interval_s: opt(FT_HAS_FRAME_INTERVAL, si.frame_interval_s),
                channel_names,
                calibration: (has_meta && si.present & FT_HAS_CALIBRATION != 0)
                    .then_some((si.calibration_offset, si.calibration_scale)),
                description: string(FT_STRING_DESCRIPTION),
            },
            host: h,
        })
    }
}

impl HostContext for CHost {
    fn image(&self) -> ImageInfo {
        self.image
    }

    fn view(&self) -> &ViewParams {
        &self.view
    }

    fn stack_info(&self) -> &StackInfo {
        &self.info
    }

    fn read_plane_u16(&mut self, plane: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        let n = self.image.plane_len();
        out.clear();
        out.resize(n, 0);
        // SAFETY: `out` holds exactly `n` writable `u16`s, and `n` is the
        // plane length the host itself reported.
        let st = unsafe {
            (self.host.read_plane_u16)(
                self.host.ctx,
                plane.c as u64,
                plane.z as u64,
                plane.t as u64,
                out.as_mut_ptr(),
                n as u64,
            )
        };
        status_to_result(st, plane)
    }

    fn read_plane_f32(&mut self, plane: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        let n = self.image.plane_len();
        out.clear();
        out.resize(n, 0.0);
        // SAFETY: as above.
        let st = unsafe {
            (self.host.read_plane_f32)(
                self.host.ctx,
                plane.c as u64,
                plane.z as u64,
                plane.t as u64,
                out.as_mut_ptr(),
                n as u64,
            )
        };
        status_to_result(st, plane)
    }

    fn progress(&mut self, fraction: f32) -> bool {
        // SAFETY: the host's callback, called with its own context.
        unsafe { (self.host.progress)(self.host.ctx, fraction) != 0 }
    }

    fn log(&mut self, message: &str) {
        // SAFETY: the borrow outlives the call, which is all the contract asks.
        unsafe { (self.host.log)(self.host.ctx, FtStr::from_str(message)) }
    }
}

fn status_to_result(st: FtStatus, plane: Plane) -> Result<(), PluginError> {
    match st {
        FtStatus::Ok => Ok(()),
        FtStatus::OutOfRange => Err(PluginError::OutOfRange(format!(
            "plane (c{}, z{}, t{})",
            plane.c, plane.z, plane.t
        ))),
        other => Err(PluginError::failed(format!(
            "the host could not decode (c{}, z{}, t{}): {other:?}",
            plane.c, plane.z, plane.t
        ))),
    }
}

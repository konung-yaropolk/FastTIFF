//! The plugin's side of the host callbacks: an [`FtHost`] wearing the ordinary
//! [`HostContext`] trait, so plugin code never sees a raw pointer.

use crate::abi::*;
use crate::api::{
    ChannelView, DisplayMode, HostContext, ImageInfo, PixelType, Plane, PluginError, Roi,
    Selection, Shape, Spacing, StackInfo, ViewParams, VolumeMode, VolumeView,
};

/// Stands in for `stack_info` on a host too old to have it.
///
/// Answering `Unsupported` rather than filling in zeros: a plugin cannot tell
/// "this file states no calibration" from "this host cannot tell me", and the
/// difference decides whether a measurement is in microns or in pixels.
unsafe extern "C" fn no_stack_info(
    _ctx: *mut core::ffi::c_void,
    _out: *mut FtStackInfo,
) -> FtStatus {
    FtStatus::Unsupported
}

/// Stands in for `stack_string` on a host too old to have it.
unsafe extern "C" fn no_stack_string(
    _ctx: *mut core::ffi::c_void,
    _which: u32,
    _index: u64,
) -> FtStr {
    FtStr::EMPTY
}

/// Stands in for `selection_count` on a host too old to have it.
///
/// Zero, which is not a lie: a host that offers no selection tool has no
/// regions, and the api already reads an empty selection as the whole frame.
/// The difference from `stack_info` — where a missing callback answers
/// `Unsupported` rather than zeros — is that there a zero would be mistaken
/// for a measurement, and here it is the truth.
unsafe extern "C" fn no_selection_count(_ctx: *mut core::ffi::c_void) -> u64 {
    0
}

/// Stands in for `selection_roi` on a host too old to have it. Never reached:
/// `no_selection_count` says there are none.
unsafe extern "C" fn no_selection_roi(
    _ctx: *mut core::ffi::c_void,
    _index: u64,
    _out: *mut FtRoi,
) -> FtStatus {
    FtStatus::Unsupported
}

/// More regions than anyone draws, so a host answering nonsense cannot make a
/// plugin allocate without bound before it has looked at a single one.
const MAX_REGIONS: u64 = 4096;

/// The host's callback table, copied into a value this plugin may safely hold.
///
/// [`FtHost`] grows, so a host built against an earlier minor allocates fewer
/// bytes than this plugin's `FtHost` has. Reading one as a whole — `*host`, or
/// `&*host` — is then a load past the end of the host's allocation, undefined
/// behaviour before a single field has been looked at. So it is a byte copy
/// bounded by what the host says it allocated, with the fields the copy did not
/// reach filled in with stubs.
///
/// Stubs, not `zeroed()`: every field past the prologue is a function pointer,
/// and a null function pointer is not a valid one whether or not it is ever
/// called. Writing the uncovered suffix explicitly is what makes the value
/// initialised.
///
/// The host's own `struct_size` is copied rather than replaced, so
/// `ft_covers!` on the result still asks about the host.
///
/// # Safety
/// `host` must point at a table the host allocated, valid for its own declared
/// size, for the duration of the call.
pub unsafe fn host_of(host: *const FtHost) -> Result<FtHost, FtStatus> {
    // `covers` reads only the four-byte prologue. The obvious spelling — copy
    // the table, then look at its `struct_size` — is the very bug this
    // prevents: the copy reads `size_of::<FtHost>()` bytes out of a shorter
    // allocation, so the check meant to prevent that has already been
    // overtaken by it.
    if !crate::abi::covers(host, FtHost::CORE) {
        crate::last_error::set(
            "the host's callback table is too small to be a FastTIFF plugin host",
        );
        return Err(FtStatus::BadArgument);
    }
    let declared = crate::abi::declared_size(host) as usize;
    let mut buf = core::mem::MaybeUninit::<FtHost>::uninit();
    core::ptr::copy_nonoverlapping(
        host.cast::<u8>(),
        buf.as_mut_ptr().cast::<u8>(),
        declared.min(core::mem::size_of::<FtHost>()),
    );
    // Field by field rather than in the pairs they were appended in: a field
    // the copy reached only halfway is covered by neither, and checking each
    // one separately is what guarantees every byte is written. The assertion
    // `offset_of!(FtHost, selection_roi) + p == size_of` in the ABI crate is
    // what fails if a later append forgets to add its line here.
    let p = buf.as_mut_ptr();
    if !crate::abi::ft_covers!(host, FtHost, stack_info) {
        core::ptr::addr_of_mut!((*p).stack_info).write(no_stack_info);
    }
    if !crate::abi::ft_covers!(host, FtHost, stack_string) {
        core::ptr::addr_of_mut!((*p).stack_string).write(no_stack_string);
    }
    if !crate::abi::ft_covers!(host, FtHost, selection_count) {
        core::ptr::addr_of_mut!((*p).selection_count).write(no_selection_count);
    }
    if !crate::abi::ft_covers!(host, FtHost, selection_roi) {
        core::ptr::addr_of_mut!((*p).selection_roi).write(no_selection_roi);
    }
    Ok(buf.assume_init())
}

/// A [`HostContext`] backed by the host's C callbacks.
pub struct CHost {
    host: FtHost,
    image: ImageInfo,
    view: ViewParams,
    info: StackInfo,
    /// Snapshotted with everything else, because that is what it is: the host
    /// re-runs the plugin when the regions change rather than changing them
    /// under a call that is already going.
    selection: Vec<Roi>,
}

impl CHost {
    /// Build from the host's table, reading everything that is snapshot-shaped
    /// once so a plugin's repeated `image()` calls do not cross the boundary.
    ///
    /// # Safety
    /// `host` must point at a valid `FtHost` whose callbacks remain valid for
    /// the lifetime of the returned value.
    pub unsafe fn new(host: *const FtHost) -> Result<CHost, FtStatus> {
        // Only the *core* is required, not the whole table. Requiring all of
        // it would refuse a host that merely predates a callback this plugin
        // might not even use, which is the opposite of what `struct_size` is
        // for. Each optional callback is checked before it is called.
        //
        // `covers` reads only the four-byte prologue. The obvious spelling —
        // copy the table, then look at its `struct_size` — is undefined
        // behaviour when the host is older: the copy reads
        // `size_of::<FtHost>()` bytes out of a shorter allocation, so the check
        // meant to prevent that has already been overtaken by it.
        let h = host_of(host)?;
        // Asked of the copy, which carries the host's own declared size — so
        // these still say what the *host* has, not what this plugin has.
        //
        // `stack_string` decides both metadata callbacks: they were appended
        // together, and a host with only the first of them is not one this
        // plugin has any reason to trust the calibration of.
        let has_metadata = crate::abi::ft_covers!(&h as *const FtHost, FtHost, stack_string);
        // Likewise the later of the selection pair.
        let has_selection = crate::abi::ft_covers!(&h as *const FtHost, FtHost, selection_roi);

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
        //
        // A host too old to have them leaves every field `None`, which is the
        // honest answer — the same one a file that states no calibration gives.
        let mut si = core::mem::zeroed::<FtStackInfo>();
        si.struct_size = core::mem::size_of::<FtStackInfo>() as u32;
        let has_meta = has_metadata && (h.stack_info)(h.ctx, &mut si) == FtStatus::Ok;
        let opt = |flag: u32, v: f64| (has_meta && si.present & flag != 0).then_some(v);
        let string = |which: u32| {
            if !has_metadata {
                return None;
            }
            let s = (h.stack_string)(h.ctx, which, 0).as_str().unwrap_or("");
            (!s.is_empty()).then(|| s.to_string())
        };
        let mut channel_names = Vec::new();
        for i in 0..if has_metadata {
            ii.channels.min(1024)
        } else {
            0
        } {
            match (h.stack_string)(h.ctx, FT_STRING_CHANNEL_NAME, i).as_str() {
                Some(n) if !n.is_empty() => channel_names.push(n.to_string()),
                // Names run out; the api documents the list may be short.
                _ => break,
            }
        }

        // The regions the plot is a function of. Read here with the rest of
        // the run's inputs so a plugin sees one consistent set for the whole
        // call, however many times it asks.
        let mut selection = Vec::new();
        if has_selection {
            let n = (h.selection_count)(h.ctx).min(MAX_REGIONS);
            for i in 0..n {
                let mut r = core::mem::zeroed::<FtRoi>();
                // Ours to declare: the host writes only this many bytes.
                r.struct_size = core::mem::size_of::<FtRoi>() as u32;
                if (h.selection_roi)(h.ctx, i, &mut r) != FtStatus::Ok {
                    break;
                }
                // A region covering nothing has no mean, and the api promises
                // it cannot be constructed; dropping it here keeps that true
                // on this side of the boundary too.
                if r.w == 0 || r.h == 0 {
                    continue;
                }
                selection.push(Roi {
                    // A shape from a newer host: its bounding box is the only
                    // thing this plugin can honestly use, and that is a rect.
                    shape: match r.shape {
                        1 => Shape::Ellipse,
                        _ => Shape::Rect,
                    },
                    x: r.x,
                    y: r.y,
                    w: r.w,
                    h: r.h,
                });
            }
        }

        Ok(CHost {
            selection,
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

    fn selection(&self) -> Selection<'_> {
        &self.selection
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

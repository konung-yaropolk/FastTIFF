// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! A C interface, for programs in C, C++, or anything else that can call C.
//!
//! Behind the `capi` feature, and built as a library of its own by
//! `cargo rustc`, which picks the crate type at build time:
//!
//! ```text
//! cargo rustc -p suite2p-registration --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! The declarations are in `include/suite2p_registration.h`, written by hand.
//! A test reads it and checks it against this module — every constant's
//! value, the settings structure's fields in order, and the list of functions
//! — so a field or function added here and not there fails the build of the
//! tests rather than a C program.
//!
//! The shape follows the Rust API — measure once with [`s2p_register`], then
//! move each frame (and each channel) with [`s2p_registered_apply`] — with the
//! habits a C caller expects: every call returns a status, `S2P_OK` or why
//! not; [`s2p_last_error`] says why in words; the result is an opaque handle
//! freed with [`s2p_registered_free`]. Nothing unwinds into C: a panic inside
//! the registration comes back as [`S2P_FAILED`], as long as the library was
//! built with `panic = "unwind"` (the default, and this workspace's release
//! profile).

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{bidiphase, register, Backend, Border, Frames, Registered, Settings};

/// The call did what it was asked.
pub const S2P_OK: c_int = 0;
/// The progress callback asked to stop; nothing was produced.
pub const S2P_CANCELLED: c_int = 1;
/// A pointer, size, index or setting the call cannot take.
pub const S2P_INVALID_ARGUMENT: c_int = 2;
/// The backend asked for cannot run here: no GPU in this build or on this
/// machine, no threads in this build, or a frame size the GPU path does not
/// take.
pub const S2P_UNAVAILABLE: c_int = 3;
/// The registration itself failed: a panic inside it, or a GPU run the device
/// could not finish.
pub const S2P_FAILED: c_int = 4;

/// [`Backend::SingleThread`].
pub const S2P_BACKEND_SINGLE_THREAD: c_int = 0;
/// [`Backend::MultiThread`].
pub const S2P_BACKEND_MULTI_THREAD: c_int = 1;
/// [`Backend::Gpu`].
pub const S2P_BACKEND_GPU: c_int = 2;

/// [`Border::Native`].
pub const S2P_BORDER_NATIVE: c_int = 0;
/// [`Border::Edge`].
pub const S2P_BORDER_EDGE: c_int = 1;
/// [`Border::Constant`].
pub const S2P_BORDER_CONSTANT: c_int = 2;
/// [`Border::Crop`]: frames come back cut to the common area, smaller than
/// they went in; [`s2p_registered_output_size`] says how big.
pub const S2P_BORDER_CROP: c_int = 3;

/// The frames moved so far apart that no pixel is covered by all of them
/// ([`NoCommonArea`](crate::NoCommonArea)): there is no common area, so
/// [`Border::Crop`] has nothing to crop to. From anything that needs the area
/// — a crop's frames or size, or the area itself under any border. The
/// registration itself stands; a border that fills does not have this
/// problem.
pub const S2P_NO_COMMON_AREA: c_int = 5;

/// [`Settings`], in C's terms. Field for field the same meaning, under the
/// same suite2p names; enumerations are `S2P_BACKEND_*` and `S2P_BORDER_*`.
///
/// Fill it with [`s2p_settings_default`] and change what differs — a
/// structure built field by field would break the first time one is added.
/// `align_by_chan2` is not here: which channel to measure on is the caller's
/// choice of what to pass.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct S2pSettings {
    /// [`Settings::nimg_init`].
    pub nimg_init: usize,
    /// [`Settings::reference_iterations`].
    pub reference_iterations: usize,
    /// [`Settings::batch_size`].
    pub batch_size: usize,
    /// [`Settings::block_size`], rows then columns.
    pub block_size: [usize; 2],
    /// [`Settings::subpixel`].
    pub subpixel: usize,
    /// [`Settings::maxregshift`].
    pub maxregshift: f64,
    /// [`Settings::maxregshift_nr`].
    pub maxregshift_nr: f64,
    /// [`Settings::smooth_sigma`].
    pub smooth_sigma: f64,
    /// [`Settings::smooth_sigma_time`].
    pub smooth_sigma_time: f64,
    /// [`Settings::spatial_taper`].
    pub spatial_taper: f64,
    /// [`Settings::snr_thresh`].
    pub snr_thresh: f64,
    /// [`Settings::th_badframes`].
    pub th_badframes: f64,
    /// [`Settings::bidiphase`].
    pub bidiphase: i32,
    /// [`Settings::backend`], an `S2P_BACKEND_*`.
    pub backend: c_int,
    /// [`Settings::border`], an `S2P_BORDER_*`.
    pub border: c_int,
    /// [`Settings::norm_frames`].
    pub norm_frames: bool,
    /// [`Settings::two_step_registration`].
    pub two_step_registration: bool,
    /// [`Settings::do_bidiphase`].
    pub do_bidiphase: bool,
    /// [`Settings::nonrigid`].
    pub nonrigid: bool,
}

impl From<&Settings> for S2pSettings {
    fn from(s: &Settings) -> Self {
        S2pSettings {
            nimg_init: s.nimg_init,
            reference_iterations: s.reference_iterations,
            batch_size: s.batch_size,
            block_size: s.block_size,
            subpixel: s.subpixel,
            maxregshift: s.maxregshift,
            maxregshift_nr: s.maxregshift_nr,
            smooth_sigma: s.smooth_sigma,
            smooth_sigma_time: s.smooth_sigma_time,
            spatial_taper: s.spatial_taper,
            snr_thresh: s.snr_thresh,
            th_badframes: s.th_badframes,
            bidiphase: s.bidiphase,
            backend: match s.backend {
                Backend::SingleThread => S2P_BACKEND_SINGLE_THREAD,
                Backend::MultiThread => S2P_BACKEND_MULTI_THREAD,
                Backend::Gpu => S2P_BACKEND_GPU,
            },
            border: match s.border {
                Border::Native => S2P_BORDER_NATIVE,
                Border::Edge => S2P_BORDER_EDGE,
                Border::Constant => S2P_BORDER_CONSTANT,
                Border::Crop => S2P_BORDER_CROP,
            },
            norm_frames: s.norm_frames,
            two_step_registration: s.two_step_registration,
            do_bidiphase: s.do_bidiphase,
            nonrigid: s.nonrigid,
        }
    }
}

impl S2pSettings {
    /// Back to [`Settings`], refusing what C can say and Rust cannot: an
    /// enumeration value out of range, a count of zero, a number that is not
    /// one.
    fn to_settings(self) -> Result<Settings, String> {
        let backend = backend_of(self.backend)?;
        let border = border_of(self.border)?;
        for (name, v) in [
            ("nimg_init", self.nimg_init),
            ("batch_size", self.batch_size),
            ("block_size[0]", self.block_size[0]),
            ("block_size[1]", self.block_size[1]),
            ("subpixel", self.subpixel),
        ] {
            if v == 0 {
                return Err(format!("{name} is 0; it has to be at least 1"));
            }
        }
        for (name, v) in [
            ("maxregshift", self.maxregshift),
            ("maxregshift_nr", self.maxregshift_nr),
            ("smooth_sigma", self.smooth_sigma),
            ("smooth_sigma_time", self.smooth_sigma_time),
            ("spatial_taper", self.spatial_taper),
            ("snr_thresh", self.snr_thresh),
            ("th_badframes", self.th_badframes),
        ] {
            if !v.is_finite() {
                return Err(format!("{name} is {v}, not a number it can use"));
            }
        }
        Ok(Settings {
            nimg_init: self.nimg_init,
            reference_iterations: self.reference_iterations,
            batch_size: self.batch_size,
            block_size: self.block_size,
            subpixel: self.subpixel,
            maxregshift: self.maxregshift,
            maxregshift_nr: self.maxregshift_nr,
            smooth_sigma: self.smooth_sigma,
            smooth_sigma_time: self.smooth_sigma_time,
            spatial_taper: self.spatial_taper,
            snr_thresh: self.snr_thresh,
            th_badframes: self.th_badframes,
            bidiphase: self.bidiphase,
            backend,
            border,
            norm_frames: self.norm_frames,
            two_step_registration: self.two_step_registration,
            do_bidiphase: self.do_bidiphase,
            nonrigid: self.nonrigid,
            ..Settings::default()
        })
    }
}

fn backend_of(v: c_int) -> Result<Backend, String> {
    match v {
        S2P_BACKEND_SINGLE_THREAD => Ok(Backend::SingleThread),
        S2P_BACKEND_MULTI_THREAD => Ok(Backend::MultiThread),
        S2P_BACKEND_GPU => Ok(Backend::Gpu),
        _ => Err(format!("backend {v} is not one of the S2P_BACKEND_* values")),
    }
}

fn border_of(v: c_int) -> Result<Border, String> {
    match v {
        S2P_BORDER_NATIVE => Ok(Border::Native),
        S2P_BORDER_EDGE => Ok(Border::Edge),
        S2P_BORDER_CONSTANT => Ok(Border::Constant),
        S2P_BORDER_CROP => Ok(Border::Crop),
        _ => Err(format!("border {v} is not one of the S2P_BORDER_* values")),
    }
}

/// What a registration produced, behind a handle: [`Registered`] and the
/// frame size it was measured at, so every later call can check a buffer
/// against it.
pub struct S2pRegistered {
    inner: Registered,
    ly: usize,
    lx: usize,
}

/// Called with the fraction done, from `0.0` to `1.0`, on the thread that
/// called [`s2p_register`]; return non-zero to go on, zero to stop. `user` is
/// whatever was passed alongside it.
pub type S2pProgress = Option<unsafe extern "C" fn(fraction: f32, user: *mut c_void) -> c_int>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Held for the length of a GPU run; see [`s2p_register`].
#[cfg(feature = "gpu")]
static GPU_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Record why a call failed, for [`s2p_last_error`], and return its status.
///
/// `try_with`, not `with`: a call made while the thread is being torn down —
/// from a C++ `thread_local` destructor, say — finds the slot already gone,
/// and `with` would panic there, which past `extern "C"` is an abort. The
/// status still comes back; only the message is lost.
fn fail(status: c_int, why: impl Into<String>) -> c_int {
    let why = CString::new(why.into().replace('\0', " ")).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|e| *e.borrow_mut() = why);
    status
}

/// Run `work`, turning a panic into [`S2P_FAILED`] rather than letting it
/// unwind into a caller that has no idea what one is.
fn guarded(work: impl FnOnce() -> c_int) -> c_int {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(status) => status,
        Err(panic) => {
            let why = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "no message".to_string());
            fail(S2P_FAILED, format!("the registration panicked: {why}"))
        }
    }
}

/// The handle behind `r`, or the status to return for a null one.
///
/// # Safety
///
/// `r` is null or came from [`s2p_register`] and has not been freed.
unsafe fn handle<'a>(r: *const S2pRegistered) -> Result<&'a S2pRegistered, c_int> {
    r.as_ref()
        .ok_or_else(|| fail(S2P_INVALID_ARGUMENT, "the result handle is NULL"))
}

/// Why the last call on this thread that did not return `S2P_OK` failed, as
/// a NUL-terminated UTF-8 string. Empty before any has.
///
/// The pointer stays valid until the next failure on the same thread; copy
/// the text to keep it. Empty, too, on a thread being torn down.
#[no_mangle]
pub extern "C" fn s2p_last_error() -> *const c_char {
    LAST_ERROR
        .try_with(|e| e.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}

/// The library's version, `"major.minor.patch"`, NUL-terminated and static.
/// The header is only good for the version it shipped with.
#[no_mangle]
pub extern "C" fn s2p_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Fill `out` with suite2p's defaults.
///
/// # Safety
///
/// `out` is null or points to writable memory for one [`S2pSettings`].
#[no_mangle]
pub unsafe extern "C" fn s2p_settings_default(out: *mut S2pSettings) -> c_int {
    if out.is_null() {
        return fail(S2P_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(S2pSettings::from(&Settings::default()));
    S2P_OK
}

/// Whether `backend` can run frames of `ly` rows by `lx` columns here:
/// `S2P_OK`, or `S2P_UNAVAILABLE` with the reason in [`s2p_last_error`].
///
/// For `S2P_BACKEND_GPU` this opens the device the first time it is asked.
#[no_mangle]
pub extern "C" fn s2p_backend_available(backend: c_int, ly: usize, lx: usize) -> c_int {
    guarded(|| match backend_of(backend) {
        Err(why) => fail(S2P_INVALID_ARGUMENT, why),
        Ok(backend) => match backend.unavailable_reason(ly, lx) {
            Some(why) => fail(S2P_UNAVAILABLE, why),
            None => S2P_OK,
        },
    })
}

/// Register a movie: `n_frames` frames of `ly` rows by `lx` columns, one
/// after another in `frames`, each row-major. On `S2P_OK`, `*out` is the
/// result, to be freed with [`s2p_registered_free`]; on anything else it is
/// NULL.
///
/// `settings` may be NULL for the defaults. `progress` may be NULL; `user` is
/// handed to it untouched. The frames are copied, so the caller's buffer is
/// free again as soon as this returns.
///
/// GPU runs started from several threads take turns; CPU runs do not.
///
/// # Safety
///
/// `frames` points to `n_frames * ly * lx` readable floats; `settings` is null
/// or points to one [`S2pSettings`]; `out` points to writable memory for one
/// pointer; `progress`, if given, is safe to call with `user`.
#[no_mangle]
pub unsafe extern "C" fn s2p_register(
    frames: *const f32,
    n_frames: usize,
    ly: usize,
    lx: usize,
    settings: *const S2pSettings,
    progress: S2pProgress,
    user: *mut c_void,
    out: *mut *mut S2pRegistered,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(S2P_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        if n_frames == 0 || ly == 0 || lx == 0 {
            return fail(
                S2P_INVALID_ARGUMENT,
                format!("{n_frames} frame(s) of {lx} x {ly} is nothing to register"),
            );
        }
        let Some(len) = n_frames
            .checked_mul(ly)
            .and_then(|v| v.checked_mul(lx))
            .filter(|&v| v <= isize::MAX as usize / std::mem::size_of::<f32>())
        else {
            return fail(S2P_INVALID_ARGUMENT, "the movie is larger than memory can address");
        };
        if frames.is_null() {
            return fail(S2P_INVALID_ARGUMENT, "frames is NULL");
        }
        let settings = match settings.as_ref() {
            None => Settings::default(),
            Some(s) => match s.to_settings() {
                Ok(s) => s,
                Err(why) => return fail(S2P_INVALID_ARGUMENT, why),
            },
        };
        if let Some(why) = settings.backend.unavailable_reason(ly, lx) {
            return fail(S2P_UNAVAILABLE, why);
        }

        let movie: Vec<Vec<f32>> = std::slice::from_raw_parts(frames, len)
            .chunks_exact(ly * lx)
            .map(<[f32]>::to_vec)
            .collect();
        // GPU runs take turns. Whether the device failed part of a run is
        // reported through one slot per process (`gpu::take_trouble`), so two
        // at once could clear or take each other's report — a failed run
        // coming back as OK. They share one device anyway.
        #[cfg(feature = "gpu")]
        let _turn = (settings.backend == Backend::Gpu)
            .then(|| GPU_TURN.lock().unwrap_or_else(|e| e.into_inner()));
        // An earlier run's trouble is not this one's.
        #[cfg(feature = "gpu")]
        let _ = crate::gpu::take_trouble();
        let mut on_progress = |fraction: f32| match progress {
            Some(report) => report(fraction, user) != 0,
            None => true,
        };
        let Some(inner) = register(
            &Frames {
                ly,
                lx,
                frames: &movie,
            },
            &settings,
            &mut on_progress,
        ) else {
            return fail(S2P_CANCELLED, "the progress callback asked to stop");
        };
        // A batch the device could not take ran on the processor instead, or
        // came back unmeasured; either way it is not the run that was asked
        // for, and a caller that asked for the GPU is told.
        #[cfg(feature = "gpu")]
        if settings.backend == Backend::Gpu {
            if let Some(why) = crate::gpu::take_trouble() {
                return fail(S2P_FAILED, format!("the GPU could not finish the run: {why}"));
            }
        }
        out.write(Box::into_raw(Box::new(S2pRegistered { inner, ly, lx })));
        S2P_OK
    })
}

/// Free a result. NULL is ignored.
///
/// # Safety
///
/// `r` is null or came from [`s2p_register`] and has not been freed.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_free(r: *mut S2pRegistered) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

/// How many frames the result has a shift for; 0 for NULL.
///
/// # Safety
///
/// `r` is null or came from [`s2p_register`] and has not been freed.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_frames(r: *const S2pRegistered) -> usize {
    r.as_ref().map_or(0, |r| r.inner.shifts.len())
}

/// Frame `t`'s rigid shift and the correlation peak that found it. Any of
/// the three outputs may be NULL.
///
/// `(dy, dx)` is where the frame's content sits relative to the reference:
/// what the reference has at `(y, x)`, frame `t` has at `(y + dy, x + dx)`.
/// Bringing the frame onto the reference moves it by `(-dy, -dx)`, which is
/// what [`s2p_registered_apply`] does. suite2p's own sign, as in its `ops`.
///
/// # Safety
///
/// `r` is null or a live result; each output is null or writable.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_shift(
    r: *const S2pRegistered,
    t: usize,
    dy: *mut i32,
    dx: *mut i32,
    corr: *mut f32,
) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    let Some(s) = r.inner.shifts.get(t) else {
        return fail(S2P_INVALID_ARGUMENT, format!("frame {t} is past the end"));
    };
    if !dy.is_null() {
        dy.write(s.dy);
    }
    if !dx.is_null() {
        dx.write(s.dx);
    }
    if !corr.is_null() {
        corr.write(s.corr);
    }
    S2P_OK
}

/// Whether frame `t` looks like an outlier ([`Registered::bad_frames`]).
/// Reported, never dropped: which frames to discard is the caller's decision.
///
/// # Safety
///
/// `r` is null or a live result; `bad` points to one writable `bool`.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_is_bad(
    r: *const S2pRegistered,
    t: usize,
    bad: *mut bool,
) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    if bad.is_null() {
        return fail(S2P_INVALID_ARGUMENT, "bad is NULL");
    }
    let Some(&b) = r.inner.bad_frames.get(t) else {
        return fail(S2P_INVALID_ARGUMENT, format!("frame {t} is past the end"));
    };
    bad.write(b);
    S2P_OK
}

/// The bidirectional-phase offset the run corrected for, in columns; 0 for
/// none, and for NULL.
///
/// # Safety
///
/// `r` is null or a live result.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_bidiphase(r: *const S2pRegistered) -> i32 {
    r.as_ref().map_or(0, |r| r.inner.bidiphase)
}

/// Whether the run measured a non-rigid block field, which
/// [`s2p_registered_apply`] then folds in.
///
/// # Safety
///
/// `r` is null or a live result.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_is_nonrigid(r: *const S2pRegistered) -> bool {
    r.as_ref().is_some_and(|r| r.inner.nonrigid.is_some())
}

/// Copy the reference every frame was aligned to into `out`, which holds
/// `len` floats; `len` has to be the frame's `ly * lx`.
///
/// # Safety
///
/// `r` is null or a live result; `out` points to `len` writable floats.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_reference(
    r: *const S2pRegistered,
    out: *mut f32,
    len: usize,
) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    if out.is_null() || len != r.ly * r.lx {
        return fail(
            S2P_INVALID_ARGUMENT,
            format!("the reference needs {} floats, not {len}", r.ly * r.lx),
        );
    }
    std::ptr::copy_nonoverlapping(r.inner.reference.as_ptr(), out, len);
    S2P_OK
}

/// Change what [`s2p_registered_apply`] writes where a moved frame no longer
/// covers the frame, an `S2P_BORDER_*`. The run's own setting until then.
///
/// # Safety
///
/// `r` is null or a live result, not in use on another thread.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_set_border(r: *mut S2pRegistered, border: c_int) -> c_int {
    let Some(r) = r.as_mut() else {
        return fail(S2P_INVALID_ARGUMENT, "the result handle is NULL");
    };
    match border_of(border) {
        Ok(b) => {
            r.inner.border = b;
            S2P_OK
        }
        Err(why) => fail(S2P_INVALID_ARGUMENT, why),
    }
}

/// The size, rows then columns, of a frame [`s2p_registered_apply`] writes:
/// the frame size the run measured, or under `S2P_BORDER_CROP` the common
/// area's — and `S2P_NO_COMMON_AREA` when there is none. Either output may be
/// NULL.
///
/// # Safety
///
/// `r` is null or a live result; `ly` and `lx` are each null or writable.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_output_size(
    r: *const S2pRegistered,
    ly: *mut usize,
    lx: *mut usize,
) -> c_int {
    guarded(|| {
        let r = match handle(r) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match r.inner.output_size(r.ly, r.lx) {
            Ok((h, w)) => {
                if !ly.is_null() {
                    ly.write(h);
                }
                if !lx.is_null() {
                    lx.write(w);
                }
                S2P_OK
            }
            Err(none) => fail(S2P_NO_COMMON_AREA, none.to_string()),
        }
    })
}

/// The largest rectangle every frame covers once moved, at any angle — what
/// `S2P_BORDER_CROP` keeps ([`Area`](crate::Area)). Where its first pixel sits
/// in the frame (`y`, `x`, fractional for a turned area), its size, and its
/// angle in radians (0 upright; positive turns its rows clockwise as an image
/// is shown). Asked under any border; `S2P_NO_COMMON_AREA` when the frames
/// share no pixel. Any output may be NULL.
///
/// # Safety
///
/// `r` is null or a live result; each output is null or writable.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_common_area(
    r: *const S2pRegistered,
    y: *mut f64,
    x: *mut f64,
    height: *mut usize,
    width: *mut usize,
    angle: *mut f64,
) -> c_int {
    guarded(|| {
        let r = match handle(r) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match r.inner.common_area(r.ly, r.lx) {
            Ok(a) => {
                for (out, v) in [(y, a.y), (x, a.x), (angle, a.angle)] {
                    if !out.is_null() {
                        out.write(v);
                    }
                }
                for (out, v) in [(height, a.height), (width, a.width)] {
                    if !out.is_null() {
                        out.write(v);
                    }
                }
                S2P_OK
            }
            Err(none) => fail(S2P_NO_COMMON_AREA, none.to_string()),
        }
    })
}

/// Put frame `t` back where it belongs: the run's bidirectional-phase
/// correction, then its shift — or its warp, when there is a block field.
/// `frame` holds one frame of the size the run measured; `out` holds what
/// [`s2p_registered_output_size`] says, which is the same size unless the
/// border is `S2P_BORDER_CROP`. They may be the same buffer.
///
/// Any frame of timepoint `t` can be passed, not only the one measured: that
/// is how a second channel is moved with the first.
///
/// Under `S2P_BORDER_CROP` with no common area this is `S2P_NO_COMMON_AREA`,
/// and `out` is left alone.
///
/// # Safety
///
/// `r` is null or a live result; `frame` points to `ly * lx` readable floats
/// and `out` to as many writable ones as the output size.
#[no_mangle]
pub unsafe extern "C" fn s2p_registered_apply(
    r: *const S2pRegistered,
    frame: *const f32,
    t: usize,
    out: *mut f32,
) -> c_int {
    guarded(|| {
        let r = match handle(r) {
            Ok(r) => r,
            Err(status) => return status,
        };
        if frame.is_null() || out.is_null() {
            return fail(S2P_INVALID_ARGUMENT, "frame or out is NULL");
        }
        if t >= r.inner.shifts.len() {
            return fail(S2P_INVALID_ARGUMENT, format!("frame {t} is past the end"));
        }
        let (ly, lx) = (r.ly, r.lx);
        // Copied first, so `out` may be `frame`.
        let mut moved = std::slice::from_raw_parts(frame, ly * lx).to_vec();
        bidiphase::shift(&mut moved, ly, lx, r.inner.bidiphase);
        let moved = match r.inner.apply(&moved, ly, lx, t) {
            Ok(moved) => moved,
            Err(none) => return fail(S2P_NO_COMMON_AREA, none.to_string()),
        };
        std::ptr::copy(moved.as_ptr(), out, moved.len());
        S2P_OK
    })
}

#[cfg(test)]
#[path = "capi_tests.rs"]
mod tests;

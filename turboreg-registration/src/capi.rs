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
//! cargo rustc -p turboreg-registration --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! The declarations are in `include/turboreg_registration.h`, written by hand.
//! A test reads it and checks it against this module — every constant's
//! value, the settings structure's fields in order, and the list of functions
//! — so a field or function added here and not there fails the build of the
//! tests rather than a C program.
//!
//! The shape follows the Rust API — measure once with [`tr_register`], then
//! move each frame (and each channel) with [`tr_registered_apply`] — with the
//! habits a C caller expects: every call returns a status, `TR_OK` or why
//! not; [`tr_last_error`] says why in words; the result is an opaque handle
//! freed with [`tr_registered_free`]. Nothing unwinds into C: a panic inside
//! the registration comes back as [`TR_FAILED`], as long as the library was
//! built with `panic = "unwind"` (the default, and this workspace's release
//! profile).

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{register_stack, Backend, Border, Frames, Reference, Registered, Settings, Transformation};

/// The call did what it was asked.
pub const TR_OK: c_int = 0;
/// The progress callback asked to stop; nothing was produced.
pub const TR_CANCELLED: c_int = 1;
/// A pointer, size, index or setting the call cannot take.
pub const TR_INVALID_ARGUMENT: c_int = 2;
/// The backend asked for cannot run here: no GPU in this build or on this
/// machine, a frame too large for the card's limits, or no threads in this
/// build.
pub const TR_UNAVAILABLE: c_int = 3;
/// The registration itself failed: a panic inside it, or frames the GPU could
/// not register for want of the device.
pub const TR_FAILED: c_int = 4;

/// [`Transformation::Translation`].
pub const TR_TRANSFORMATION_TRANSLATION: c_int = 0;
/// [`Transformation::RigidBody`].
pub const TR_TRANSFORMATION_RIGID_BODY: c_int = 1;
/// [`Transformation::ScaledRotation`].
pub const TR_TRANSFORMATION_SCALED_ROTATION: c_int = 2;
/// [`Transformation::Affine`].
pub const TR_TRANSFORMATION_AFFINE: c_int = 3;
/// [`Transformation::Bilinear`].
pub const TR_TRANSFORMATION_BILINEAR: c_int = 4;

/// [`Reference::Previous`].
pub const TR_REFERENCE_PREVIOUS: c_int = 0;
/// [`Reference::First`].
pub const TR_REFERENCE_FIRST: c_int = 1;
/// [`Reference::Mean`].
pub const TR_REFERENCE_MEAN: c_int = 2;

/// [`Backend::SingleThread`].
pub const TR_BACKEND_SINGLE_THREAD: c_int = 0;
/// [`Backend::MultiThread`].
pub const TR_BACKEND_MULTI_THREAD: c_int = 1;
/// [`Backend::Gpu`].
pub const TR_BACKEND_GPU: c_int = 2;

/// [`Border::Native`].
pub const TR_BORDER_NATIVE: c_int = 0;
/// [`Border::Edge`].
pub const TR_BORDER_EDGE: c_int = 1;
/// [`Border::Constant`].
pub const TR_BORDER_CONSTANT: c_int = 2;
/// [`Border::Crop`]: frames come back cut to the common area, smaller than
/// they went in; [`tr_registered_output_size`] says how big.
pub const TR_BORDER_CROP: c_int = 3;

/// The frames moved so far apart that no pixel is covered by all of them
/// ([`NoCommonArea`](crate::NoCommonArea)): there is no common area, so
/// [`Border::Crop`] has nothing to crop to. From anything that needs the area
/// — a crop's frames or size, or the area itself under any border. The
/// registration itself stands; a border that fills does not have this
/// problem.
pub const TR_NO_COMMON_AREA: c_int = 5;

/// [`Settings`], in C's terms, plus the [`Border`] the result applies with.
/// Enumerations are the `TR_TRANSFORMATION_*`, `TR_REFERENCE_*`,
/// `TR_BACKEND_*` and `TR_BORDER_*` values.
///
/// Fill it with [`tr_settings_default`] and change what differs — a
/// structure built field by field would break the first time one is added.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TrSettings {
    /// [`Settings::n_frames`].
    pub n_frames: usize,
    /// [`Settings::moving_average`].
    pub moving_average: usize,
    /// [`Settings::max_pyramid_depth`].
    pub max_pyramid_depth: usize,
    /// [`Settings::transformation`], a `TR_TRANSFORMATION_*`.
    pub transformation: c_int,
    /// [`Settings::reference`], a `TR_REFERENCE_*`.
    pub reference: c_int,
    /// [`Settings::backend`], a `TR_BACKEND_*`.
    pub backend: c_int,
    /// [`Registered::border`] for the result, a `TR_BORDER_*`.
    pub border: c_int,
}

impl TrSettings {
    fn from_settings(s: &Settings, border: Border) -> Self {
        TrSettings {
            n_frames: s.n_frames,
            moving_average: s.moving_average,
            max_pyramid_depth: s.max_pyramid_depth,
            transformation: Transformation::all()
                .iter()
                .position(|t| *t == s.transformation)
                .unwrap_or(0) as c_int,
            reference: Reference::all()
                .iter()
                .position(|r| *r == s.reference)
                .unwrap_or(0) as c_int,
            backend: Backend::all()
                .iter()
                .position(|b| *b == s.backend)
                .unwrap_or(0) as c_int,
            border: Border::all().iter().position(|b| *b == border).unwrap_or(0) as c_int,
        }
    }

    /// Back to [`Settings`] and a [`Border`], refusing what C can say and Rust
    /// cannot: an enumeration value out of range, a count of zero, and the
    /// one combination the registration itself refuses.
    fn to_settings(self) -> Result<(Settings, Border), String> {
        let transformation = pick(&Transformation::all(), self.transformation, "TRANSFORMATION")?;
        let reference = pick(&Reference::all(), self.reference, "REFERENCE")?;
        let backend = backend_of(self.backend)?;
        let border = border_of(self.border)?;
        if self.n_frames == 0 || self.moving_average == 0 {
            return Err("n_frames and moving_average have to be at least 1".to_string());
        }
        if reference == Reference::Previous && transformation == Transformation::Bilinear {
            return Err(
                "a bilinear warp cannot be chained onto another, so it cannot be registered \
                 against the previous frame; use TR_REFERENCE_FIRST or TR_REFERENCE_MEAN"
                    .to_string(),
            );
        }
        Ok((
            Settings {
                transformation,
                reference,
                n_frames: self.n_frames,
                moving_average: self.moving_average,
                backend,
                max_pyramid_depth: self.max_pyramid_depth,
            },
            border,
        ))
    }
}

/// The `index`th of `all`, which is how each `TR_<WHAT>_*` value is numbered.
fn pick<T: Copy>(all: &[T], index: c_int, what: &str) -> Result<T, String> {
    usize::try_from(index)
        .ok()
        .and_then(|i| all.get(i).copied())
        .ok_or_else(|| format!("{} {index} is not one of the TR_{what}_* values", what.to_lowercase()))
}

fn backend_of(v: c_int) -> Result<Backend, String> {
    pick(&Backend::all(), v, "BACKEND")
}

fn border_of(v: c_int) -> Result<Border, String> {
    pick(&Border::all(), v, "BORDER")
}

/// What a registration produced, behind a handle: [`Registered`] and the
/// frame size it was measured at, so every later call can check a buffer
/// against it.
pub struct TrRegistered {
    inner: Registered,
    ly: usize,
    lx: usize,
}

/// Called with the fraction done, from `0.0` to `1.0`, on the thread that
/// called [`tr_register`]; return non-zero to go on, zero to stop. `user` is
/// whatever was passed alongside it.
pub type TrProgress = Option<unsafe extern "C" fn(fraction: f32, user: *mut c_void) -> c_int>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Held for the length of a GPU run; see [`tr_register`].
#[cfg(feature = "gpu")]
static GPU_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Record why a call failed, for [`tr_last_error`], and return its status.
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

/// Run `work`, turning a panic into [`TR_FAILED`] rather than letting it
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
            fail(TR_FAILED, format!("the registration panicked: {why}"))
        }
    }
}

/// The handle behind `r`, or the status to return for a null one.
///
/// # Safety
///
/// `r` is null or came from [`tr_register`] and has not been freed.
unsafe fn handle<'a>(r: *const TrRegistered) -> Result<&'a TrRegistered, c_int> {
    r.as_ref()
        .ok_or_else(|| fail(TR_INVALID_ARGUMENT, "the result handle is NULL"))
}

/// Why the last call on this thread that did not return `TR_OK` failed, as
/// a NUL-terminated UTF-8 string. Empty before any has.
///
/// The pointer stays valid until the next failure on the same thread; copy
/// the text to keep it. Empty, too, on a thread being torn down.
#[no_mangle]
pub extern "C" fn tr_last_error() -> *const c_char {
    LAST_ERROR
        .try_with(|e| e.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}

/// The library's version, `"major.minor.patch"`, NUL-terminated and static.
/// The header is only good for the version it shipped with.
#[no_mangle]
pub extern "C" fn tr_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Fill `out` with the defaults: rigid body against the first frame, and
/// TurboReg's own border.
///
/// # Safety
///
/// `out` is null or points to writable memory for one [`TrSettings`].
#[no_mangle]
pub unsafe extern "C" fn tr_settings_default(out: *mut TrSettings) -> c_int {
    if out.is_null() {
        return fail(TR_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(TrSettings::from_settings(&Settings::default(), Border::default()));
    TR_OK
}

/// Whether `backend` can run frames of `ly` rows by `lx` columns here:
/// `TR_OK`, or `TR_UNAVAILABLE` with the reason in [`tr_last_error`].
///
/// For `TR_BACKEND_GPU` this opens the device the first time it is asked.
#[no_mangle]
pub extern "C" fn tr_backend_available(backend: c_int, ly: usize, lx: usize) -> c_int {
    guarded(|| match backend_of(backend) {
        Err(why) => fail(TR_INVALID_ARGUMENT, why),
        Ok(backend) => match backend.unavailable_reason(ly, lx) {
            Some(why) => fail(TR_UNAVAILABLE, why),
            None => TR_OK,
        },
    })
}

/// Register a stack: `n_frames` frames of `ly` rows by `lx` columns, one
/// after another in `frames`, each row-major. On `TR_OK`, `*out` is the
/// result, to be freed with [`tr_registered_free`]; on anything else it is
/// NULL.
///
/// `settings` may be NULL for the defaults. `progress` may be NULL; `user` is
/// handed to it untouched. The frames are copied, so the caller's buffer is
/// free again as soon as this returns. GPU runs started from several threads
/// take turns; CPU runs do not.
///
/// Frames the fit gave up on keep the identity and are reported by
/// [`tr_registered_failed`], not refused — except under `TR_BACKEND_GPU`, where
/// frames lost for want of GPU memory make the whole call `TR_FAILED`.
///
/// # Safety
///
/// `frames` points to `n_frames * ly * lx` readable floats; `settings` is null
/// or points to one [`TrSettings`]; `out` points to writable memory for one
/// pointer; `progress`, if given, is safe to call with `user`.
#[no_mangle]
pub unsafe extern "C" fn tr_register(
    frames: *const f32,
    n_frames: usize,
    ly: usize,
    lx: usize,
    settings: *const TrSettings,
    progress: TrProgress,
    user: *mut c_void,
    out: *mut *mut TrRegistered,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(TR_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        if n_frames == 0 || ly == 0 || lx == 0 {
            return fail(
                TR_INVALID_ARGUMENT,
                format!("{n_frames} frame(s) of {lx} x {ly} is nothing to register"),
            );
        }
        let Some(len) = n_frames
            .checked_mul(ly)
            .and_then(|v| v.checked_mul(lx))
            .filter(|&v| v <= isize::MAX as usize / std::mem::size_of::<f32>())
        else {
            return fail(TR_INVALID_ARGUMENT, "the stack is larger than memory can address");
        };
        if frames.is_null() {
            return fail(TR_INVALID_ARGUMENT, "frames is NULL");
        }
        let (settings, border) = match settings.as_ref() {
            None => (Settings::default(), Border::default()),
            Some(s) => match s.to_settings() {
                Ok(both) => both,
                Err(why) => return fail(TR_INVALID_ARGUMENT, why),
            },
        };
        if let Some(why) = settings.backend.unavailable_reason(ly, lx) {
            return fail(TR_UNAVAILABLE, why);
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
        let Some(mut inner) = register_stack(
            &Frames {
                ly,
                lx,
                frames: &movie,
            },
            &settings,
            &mut on_progress,
        ) else {
            return fail(TR_CANCELLED, "the progress callback asked to stop");
        };
        // Frames the device failed are not fitted for want of the device, not
        // because the data would not fit: a caller that asked for the GPU is
        // told, rather than handed frames that will not move.
        #[cfg(feature = "gpu")]
        if settings.backend == Backend::Gpu && !inner.failed.is_empty() {
            if let Some(why) = crate::gpu::take_trouble() {
                return fail(
                    TR_FAILED,
                    format!(
                        "{} of {n_frames} frames could not be registered on the GPU: {why}",
                        inner.failed.len()
                    ),
                );
            }
        }
        inner.border = border;
        out.write(Box::into_raw(Box::new(TrRegistered { inner, ly, lx })));
        TR_OK
    })
}

/// Free a result. NULL is ignored.
///
/// # Safety
///
/// `r` is null or came from [`tr_register`] and has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_free(r: *mut TrRegistered) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

/// How many frames the result has a transformation for; 0 for NULL.
///
/// # Safety
///
/// `r` is null or came from [`tr_register`] and has not been freed.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_frames(r: *const TrRegistered) -> usize {
    r.as_ref().map_or(0, |r| r.inner.matrices.len())
}

/// Frame `t`'s transformation, as eight coefficients into `c`:
///
/// `x' = c[0] + c[1] x + c[2] y + c[3] x y`,
/// `y' = c[4] + c[5] x + c[6] y + c[7] x y`.
///
/// It maps **reference** coordinates `(x, y)` to **moving** coordinates — the
/// direction a resampler pulls in, so it is used as it stands, not inverted.
/// `c[3]` and `c[7]` are zero for every model but bilinear.
///
/// # Safety
///
/// `r` is null or a live result; `c` points to 8 writable doubles.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_matrix(r: *const TrRegistered, t: usize, c: *mut f64) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    if c.is_null() {
        return fail(TR_INVALID_ARGUMENT, "c is NULL");
    }
    let Some(m) = r.inner.matrices.get(t) else {
        return fail(TR_INVALID_ARGUMENT, format!("frame {t} is past the end"));
    };
    std::ptr::copy_nonoverlapping(m.c.as_ptr(), c, 8);
    TR_OK
}

/// Whether the fit gave up on frame `t`, which then keeps the identity.
/// Reported, never dropped: which frames to discard is the caller's decision.
///
/// # Safety
///
/// `r` is null or a live result; `failed` points to one writable `bool`.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_failed(r: *const TrRegistered, t: usize, failed: *mut bool) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    if failed.is_null() {
        return fail(TR_INVALID_ARGUMENT, "failed is NULL");
    }
    if t >= r.inner.matrices.len() {
        return fail(TR_INVALID_ARGUMENT, format!("frame {t} is past the end"));
    }
    failed.write(r.inner.failed.contains(&t));
    TR_OK
}

/// Whether there is a single reference to copy out: there is for
/// `TR_REFERENCE_FIRST` and `TR_REFERENCE_MEAN`, not for
/// `TR_REFERENCE_PREVIOUS`, whose target changes every frame.
///
/// # Safety
///
/// `r` is null or a live result.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_has_reference(r: *const TrRegistered) -> bool {
    r.as_ref().is_some_and(|r| r.inner.reference.is_some())
}

/// Copy the image every frame was matched against into `out`, which holds
/// `len` floats; `len` has to be the frame's `ly * lx`.
///
/// # Safety
///
/// `r` is null or a live result; `out` points to `len` writable floats.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_reference(r: *const TrRegistered, out: *mut f32, len: usize) -> c_int {
    let r = match handle(r) {
        Ok(r) => r,
        Err(status) => return status,
    };
    let Some(reference) = &r.inner.reference else {
        return fail(
            TR_INVALID_ARGUMENT,
            "this run has no single reference: TR_REFERENCE_PREVIOUS matches each frame to the one before it",
        );
    };
    if out.is_null() || len != r.ly * r.lx {
        return fail(
            TR_INVALID_ARGUMENT,
            format!("the reference needs {} floats, not {len}", r.ly * r.lx),
        );
    }
    std::ptr::copy_nonoverlapping(reference.as_ptr(), out, len);
    TR_OK
}

/// Change what [`tr_registered_apply`] writes where a moved frame no longer
/// covers the frame, a `TR_BORDER_*`. The settings' border until then.
///
/// # Safety
///
/// `r` is null or a live result, not in use on another thread.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_set_border(r: *mut TrRegistered, border: c_int) -> c_int {
    let Some(r) = r.as_mut() else {
        return fail(TR_INVALID_ARGUMENT, "the result handle is NULL");
    };
    match border_of(border) {
        Ok(b) => {
            r.inner.border = b;
            TR_OK
        }
        Err(why) => fail(TR_INVALID_ARGUMENT, why),
    }
}

/// The size, rows then columns, of a frame [`tr_registered_apply`] writes:
/// the frame size the run measured, or under `TR_BORDER_CROP` the common
/// area's — and `TR_NO_COMMON_AREA` when there is none. Either output may be
/// NULL.
///
/// # Safety
///
/// `r` is null or a live result; `ly` and `lx` are each null or writable.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_output_size(
    r: *const TrRegistered,
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
                TR_OK
            }
            Err(none) => fail(TR_NO_COMMON_AREA, none.to_string()),
        }
    })
}

/// The largest rectangle every frame covers once moved, at any angle — what
/// `TR_BORDER_CROP` keeps ([`Area`](crate::Area)). Where its first pixel sits
/// in the frame (`y`, `x`, fractional for a turned area), its size, and its
/// angle in radians (0 upright; positive turns its rows clockwise as an image
/// is shown). Asked under any border; `TR_NO_COMMON_AREA` when the frames
/// share no pixel. Any output may be NULL.
///
/// # Safety
///
/// `r` is null or a live result; each output is null or writable.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_common_area(
    r: *const TrRegistered,
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
                TR_OK
            }
            Err(none) => fail(TR_NO_COMMON_AREA, none.to_string()),
        }
    })
}

/// Put frame `t` back where it belongs: resample it through its
/// transformation. `frame` holds one frame of the size the run measured;
/// `out` holds what [`tr_registered_output_size`] says, which is the same
/// size unless the border is `TR_BORDER_CROP`. They may be the same buffer.
///
/// Any frame of timepoint `t` can be passed, not only the one measured: that
/// is how a second channel is moved with the first.
///
/// Under `TR_BORDER_CROP` with no common area this is `TR_NO_COMMON_AREA`,
/// and `out` is left alone.
///
/// # Safety
///
/// `r` is null or a live result; `frame` points to `ly * lx` readable floats
/// and `out` to as many writable ones as the output size.
#[no_mangle]
pub unsafe extern "C" fn tr_registered_apply(
    r: *const TrRegistered,
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
            return fail(TR_INVALID_ARGUMENT, "frame or out is NULL");
        }
        if t >= r.inner.matrices.len() {
            return fail(TR_INVALID_ARGUMENT, format!("frame {t} is past the end"));
        }
        let (ly, lx) = (r.ly, r.lx);
        // Read in full before anything is written, so `out` may be `frame`.
        let moved = match r.inner.apply(std::slice::from_raw_parts(frame, ly * lx), ly, lx, t) {
            Ok(moved) => moved,
            Err(none) => return fail(TR_NO_COMMON_AREA, none.to_string()),
        };
        std::ptr::copy(moved.as_ptr(), out, moved.len());
        TR_OK
    })
}

#[cfg(test)]
#[path = "capi_tests.rs"]
mod tests;

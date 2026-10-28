//! A C interface, for programs in C, C++, or anything else that can call C.
//!
//! Behind the `capi` feature, and built as a library of its own by
//! `cargo rustc`, which picks the crate type at build time:
//!
//! ```text
//! cargo rustc -p oir-reader --release --features capi --crate-type cdylib,staticlib
//! ```
//!
//! The declarations are in `include/oir_reader.h`, written by hand. A test
//! reads it and checks it against this module — every constant's value, each
//! structure's fields in order, and the list of functions — so a field or
//! function added here and not there fails the tests rather than a C program.
//!
//! The shape follows the Rust API — [`oir_read`] (or [`oir_read_bytes`]) reads
//! a whole acquisition into an opaque handle, everything else asks that handle
//! about it, and [`oir_acquisition_free`] lets it go — with the habits a C
//! caller expects: every call that can fail returns a status, `OIR_OK` or why
//! not, and [`oir_last_error`] says why in words. A function that hands back
//! text follows `snprintf`: it writes what fits in the caller's buffer and
//! returns the length it needed, or a negative status when it could not answer
//! at all. Nothing unwinds into C: a panic inside the reader comes back as
//! [`OIR_FAILED`], as long as the library was built with `panic = "unwind"`
//! (the default, and this workspace's release profile).
//!
//! # Unsafe code
//!
//! The crate is `#![deny(unsafe_code)]`, and stays so. Two places are allowed
//! it: mapping a file (`map.rs`, one call), and this module, because taking
//! pointers from C is what it is for. This one reads the caller's arguments,
//! writes the caller's outputs, and calls only the safe API — the reader
//! itself, and what it promises about memory, is the same code whether or not
//! this module is built.

#![allow(unsafe_code)]

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    Access, Acquisition, BlockSource, Channel, Error, ErrorKind, Options, Planes, ReadPath, Record,
    SampleType,
};

/// The call did what it was asked.
pub const OIR_OK: c_int = 0;
/// The progress callback asked to stop; nothing was produced.
pub const OIR_CANCELLED: c_int = 1;
/// A pointer, size, index or enumeration value the call cannot take.
pub const OIR_INVALID_ARGUMENT: c_int = 2;
/// The call cannot work in this build: reading by path where there is no
/// filesystem (`wasm32-unknown-unknown`). Read the bytes with
/// [`oir_read_bytes`] there.
pub const OIR_UNAVAILABLE: c_int = 3;
/// The reader failed in a way that is not the file's fault: a panic inside it.
pub const OIR_FAILED: c_int = 4;
/// A part could not be opened or read ([`ErrorKind::Io`]).
pub const OIR_IO_ERROR: c_int = 5;
/// Not an OIR file: the signature is missing ([`ErrorKind::NotOir`]).
pub const OIR_NOT_OIR: c_int = 6;
/// Damaged, truncated, contradictory or unrecognisable structure or metadata
/// ([`ErrorKind::Malformed`]).
pub const OIR_MALFORMED: c_int = 7;
/// No image plane this reader recognises — usually the parts that hold them are
/// not there ([`ErrorKind::NoPlanes`]).
pub const OIR_NO_PLANES: c_int = 8;

/// [`Access::Auto`].
pub const OIR_ACCESS_AUTO: c_int = 0;
/// [`Access::Buffered`].
pub const OIR_ACCESS_BUFFERED: c_int = 1;

/// [`SampleType::U8`].
pub const OIR_SAMPLE_U8: c_int = 0;
/// [`SampleType::U16`].
pub const OIR_SAMPLE_U16: c_int = 1;

/// [`ReadPath::Mapped`].
pub const OIR_READ_MAPPED: c_int = 0;
/// [`ReadPath::Buffered`].
pub const OIR_READ_BUFFERED: c_int = 1;
/// [`ReadPath::Memory`].
pub const OIR_READ_MEMORY: c_int = 2;

/// [`BlockSource::Index`].
pub const OIR_BLOCKS_INDEX: c_int = 0;
/// [`BlockSource::Walk`].
pub const OIR_BLOCKS_WALK: c_int = 1;

/// [`Record::file_version`].
pub const OIR_FIELD_FILE_VERSION: c_int = 0;
/// [`Record::created`].
pub const OIR_FIELD_CREATED: c_int = 1;
/// [`Record::system_name`].
pub const OIR_FIELD_SYSTEM_NAME: c_int = 2;
/// [`Record::system_version`].
pub const OIR_FIELD_SYSTEM_VERSION: c_int = 3;
/// [`Record::objective`].
pub const OIR_FIELD_OBJECTIVE: c_int = 4;
/// [`Record::magnification`].
pub const OIR_FIELD_MAGNIFICATION: c_int = 5;
/// [`Record::numerical_aperture`].
pub const OIR_FIELD_NUMERICAL_APERTURE: c_int = 6;
/// [`Record::scanner`].
pub const OIR_FIELD_SCANNER: c_int = 7;
/// [`Record::sampling_speed_us`].
pub const OIR_FIELD_SAMPLING_SPEED_US: c_int = 8;
/// [`Record::sequential_mode`].
pub const OIR_FIELD_SEQUENTIAL_MODE: c_int = 9;
/// [`Record::integration`].
pub const OIR_FIELD_INTEGRATION: c_int = 10;
/// [`Record::integration_count`].
pub const OIR_FIELD_INTEGRATION_COUNT: c_int = 11;
/// [`Record::rotation_deg`].
pub const OIR_FIELD_ROTATION_DEG: c_int = 12;
/// [`Record::pan_x_um`].
pub const OIR_FIELD_PAN_X_UM: c_int = 13;
/// [`Record::pan_y_um`].
pub const OIR_FIELD_PAN_Y_UM: c_int = 14;
/// [`Record::zoom`].
pub const OIR_FIELD_ZOOM: c_int = 15;
/// [`Record::mirror_turret`].
pub const OIR_FIELD_MIRROR_TURRET: c_int = 16;
/// [`Record::laser_wavelength_nm`].
pub const OIR_FIELD_LASER_WAVELENGTH_NM: c_int = 17;
/// [`Record::laser_transmissivity_pct`].
pub const OIR_FIELD_LASER_TRANSMISSIVITY_PCT: c_int = 18;

/// [`Channel::name`].
pub const OIR_CHANNEL_NAME: c_int = 0;
/// [`Channel::detector`].
pub const OIR_CHANNEL_DETECTOR: c_int = 1;
/// [`Channel::barrier_filter`].
pub const OIR_CHANNEL_BARRIER_FILTER: c_int = 2;
/// [`Channel::dichroic`].
pub const OIR_CHANNEL_DICHROIC: c_int = 3;
/// [`Channel::dye`].
pub const OIR_CHANNEL_DYE: c_int = 4;
/// [`Channel::emission_nm`].
pub const OIR_CHANNEL_EMISSION_NM: c_int = 5;
/// [`Channel::bits`].
pub const OIR_CHANNEL_BITS: c_int = 6;
/// [`Channel::pmt_voltage`].
pub const OIR_CHANNEL_PMT_VOLTAGE: c_int = 7;
/// [`Channel::label`]: the detector, or the channel's own name without one.
pub const OIR_CHANNEL_LABEL: c_int = 8;

/// Whether a path names a file this build can open: not in a browser.
const NO_FILESYSTEM: bool = cfg!(all(target_arch = "wasm32", target_os = "unknown"));

/// [`Options`], in C's terms. Fill it with [`oir_options_default`] and change
/// what differs — a structure built field by field would break the first time
/// one is added.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OirOptions {
    /// [`Options::access`], an `OIR_ACCESS_*`.
    pub access: c_int,
}

impl OirOptions {
    /// Back to [`Options`], refusing an enumeration value out of range.
    fn to_options(self) -> Result<Options, String> {
        let access = match self.access {
            OIR_ACCESS_AUTO => Access::Auto,
            OIR_ACCESS_BUFFERED => Access::Buffered,
            v => return Err(format!("access {v} is not one of the OIR_ACCESS_* values")),
        };
        Ok(Options::new().access(access))
    }
}

/// [`Shape`](crate::Shape), with what its methods work out written down.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OirShape {
    /// Pixels per row.
    pub width: u32,
    /// Rows per plane.
    pub height: u32,
    /// Channels recorded.
    pub channels: usize,
    /// Z slices.
    pub slices: usize,
    /// Time points.
    pub frames: usize,
    /// An `OIR_SAMPLE_*`.
    pub sample_type: c_int,
    /// Bytes per sample: 1 or 2.
    pub bytes_per_sample: usize,
    /// Samples per plane, `width * height`.
    pub plane_len: usize,
    /// Planes in the stack, `channels * slices * frames`.
    pub plane_count: usize,
}

/// The numbers of the acquisition's [`Record`], and the counts of its lists.
///
/// A number the file does not state is NaN, or 0 for a count or a size;
/// `roundtrip` is -1. The timing — `frame_interval_s` and `duration_s` — is for
/// the stack as read, which may be a frame short of what the record states.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OirRecordInfo {
    /// [`Record::pixel_x_um`].
    pub pixel_x_um: f64,
    /// [`Record::pixel_y_um`].
    pub pixel_y_um: f64,
    /// [`Record::z_step_um`].
    pub z_step_um: f64,
    /// [`Acquisition::frame_interval_s`].
    pub frame_interval_s: f64,
    /// [`Record::duration_s`] for the frames the stack holds.
    pub duration_s: f64,
    /// [`Record::width`]: the frame width as the record states it.
    pub width: u32,
    /// [`Record::height`].
    pub height: u32,
    /// [`Record::slices`]: the slices a z-stack was configured for.
    pub slices: usize,
    /// How many [`Record::frame_times_ms`]; [`oir_frame_times`] copies them.
    pub frame_time_count: usize,
    /// How many [`Record::channels`]; [`oir_channel_string`] reads them.
    pub channel_count: usize,
    /// How many [`Record::events`]; [`oir_event_name`] and
    /// [`oir_event_time_ms`] read them.
    pub event_count: usize,
    /// [`Record::roundtrip`]: 1 both ways, 0 one way, -1 not stated.
    pub roundtrip: c_int,
    /// [`Record::free_run`].
    pub free_run: bool,
}

/// The cheap facts about one [`Part`](crate::Part); its path and why it could
/// not be mapped are text, from [`oir_part_path`] and [`oir_part_map_error`].
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OirPartInfo {
    /// Bytes read: the file's length when it was opened.
    pub size: u64,
    /// An `OIR_READ_*`.
    pub read_path: c_int,
    /// An `OIR_BLOCKS_*`.
    pub block_source: c_int,
    /// Blocks it holds.
    pub block_count: usize,
    /// Image planes first seen in it.
    pub planes_added: usize,
}

/// An acquisition, read, behind a handle: [`Acquisition`], every plane of it
/// in memory.
pub struct OirAcquisition {
    inner: Acquisition,
}

/// Called with the fraction of the acquisition's bytes read, from `0.0` to
/// `1.0`, on the thread that called [`oir_read`] or [`oir_read_bytes`]; return
/// non-zero to go on, zero to stop. `user` is whatever was passed alongside it.
pub type OirProgress = Option<unsafe extern "C" fn(fraction: f32, user: *mut c_void) -> c_int>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Record why a call failed, for [`oir_last_error`], and return its status.
fn fail(status: c_int, why: impl Into<String>) -> c_int {
    let why = CString::new(why.into().replace('\0', " ")).unwrap_or_default();
    // `try_with`: a call from a thread being torn down (a C++ `thread_local`
    // destructor) finds the slot gone, and `with` would panic there, which
    // past `extern "C"` is an abort. The status still comes back.
    let _ = LAST_ERROR.try_with(|e| *e.borrow_mut() = why);
    status
}

/// [`fail`], for a call that returns a length: the status, negated.
fn fail_len(status: c_int, why: impl Into<String>) -> isize {
    -(fail(status, why) as isize)
}

/// The status for a reader error, and the reason with the part it was reading
/// and the operating system's own words.
fn fail_read(e: &Error) -> c_int {
    let status = match e.kind() {
        ErrorKind::Io => OIR_IO_ERROR,
        ErrorKind::NotOir => OIR_NOT_OIR,
        ErrorKind::Malformed => OIR_MALFORMED,
        ErrorKind::NoPlanes => OIR_NO_PLANES,
        ErrorKind::Cancelled => OIR_CANCELLED,
    };
    let mut why = e.to_string();
    if let Some(path) = e.path() {
        why = format!("{}: {why}", path.display());
    }
    if let Some(cause) = std::error::Error::source(e) {
        why = format!("{why} ({cause})");
    }
    fail(status, why)
}

fn panicked(panic: Box<dyn Any + Send>) -> String {
    let why = panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string());
    format!("the reader panicked: {why}")
}

/// Run `work`, turning a panic into [`OIR_FAILED`] rather than letting it
/// unwind into a caller that has no idea what one is.
fn guarded(work: impl FnOnce() -> c_int) -> c_int {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|p| fail(OIR_FAILED, panicked(p)))
}

/// [`guarded`], for a call that returns a length.
fn guarded_len(work: impl FnOnce() -> isize) -> isize {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|p| fail_len(OIR_FAILED, panicked(p)))
}

/// The acquisition behind `a`, or the status to return for a null one.
///
/// # Safety
///
/// `a` is null or came from [`oir_read`] or [`oir_read_bytes`] and has not
/// been freed.
unsafe fn handle<'a>(a: *const OirAcquisition) -> Result<&'a Acquisition, c_int> {
    a.as_ref()
        .map(|a| &a.inner)
        .ok_or_else(|| fail(OIR_INVALID_ARGUMENT, "the acquisition handle is NULL"))
}

/// A NUL-terminated UTF-8 argument, or the status to return for it.
///
/// # Safety
///
/// `s` is null or points to a NUL-terminated string.
unsafe fn text_arg<'a>(s: *const c_char, name: &str) -> Result<&'a str, c_int> {
    if s.is_null() {
        return Err(fail(OIR_INVALID_ARGUMENT, format!("{name} is NULL")));
    }
    CStr::from_ptr(s)
        .to_str()
        .map_err(|_| fail(OIR_INVALID_ARGUMENT, format!("{name} is not UTF-8")))
}

/// Hand `s` to C the way `snprintf` would: as much as fits in `cap` bytes with
/// a NUL after it — cut at a character boundary, never inside one — and the
/// length of the whole, without the NUL, as the answer.
///
/// # Safety
///
/// `buf` is null or points to `cap` writable bytes.
unsafe fn put_text(s: &str, buf: *mut c_char, cap: usize) -> isize {
    if buf.is_null() {
        if cap != 0 {
            return fail_len(
                OIR_INVALID_ARGUMENT,
                format!("buf is NULL but cap is {cap}; pass 0 to ask for the length"),
            );
        }
    } else if cap != 0 {
        let mut n = s.len().min(cap - 1);
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf.cast::<u8>(), n);
        buf.add(n).write(0);
    }
    // No string is longer than isize::MAX bytes.
    s.len() as isize
}

/// Element `i` of `list`, or the status to return for one past the end.
fn index_in<'a, T>(list: &'a [T], i: usize, what: &str) -> Result<&'a T, c_int> {
    list.get(i).ok_or_else(|| {
        fail(
            OIR_INVALID_ARGUMENT,
            format!("{what} {i} is past the end; there are {}", list.len()),
        )
    })
}

/// The status of a `Result<_, c_int>` as a length.
fn len_of(r: Result<isize, c_int>) -> isize {
    r.unwrap_or_else(|status| -(status as isize))
}

/// Why the last call on this thread that did not return `OIR_OK` failed, as a
/// NUL-terminated UTF-8 string. Empty before any has.
///
/// The pointer stays valid until the next failure on the same thread; copy the
/// text to keep it.
#[no_mangle]
pub extern "C" fn oir_last_error() -> *const c_char {
    LAST_ERROR
        .try_with(|e| e.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}

/// The library's version, `"major.minor.patch"`, NUL-terminated and static.
/// The header is only good for the version it shipped with.
#[no_mangle]
pub extern "C" fn oir_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Whether this build memory-maps the parts it reads ([`MMAP_SUPPORTED`]).
/// When it is false, `OIR_ACCESS_AUTO` reads every part through a buffer.
///
/// [`MMAP_SUPPORTED`]: crate::MMAP_SUPPORTED
#[no_mangle]
pub extern "C" fn oir_mmap_supported() -> bool {
    crate::MMAP_SUPPORTED
}

/// Whether the `len` bytes at `head`, the opening bytes of a file, carry the
/// OIR signature ([`is_oir`](crate::is_oir)); the first sixteen are enough.
/// False for NULL.
///
/// # Safety
///
/// `head` is null or points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_is_oir(head: *const u8, len: usize) -> bool {
    // The signature is all that is looked at, so no more is read than it.
    !head.is_null() && crate::is_oir(std::slice::from_raw_parts(head, len.min(16)))
}

/// Fill `out` with the defaults: `OIR_ACCESS_AUTO`.
///
/// # Safety
///
/// `out` is null or points to writable memory for one [`OirOptions`].
#[no_mangle]
pub unsafe extern "C" fn oir_options_default(out: *mut OirOptions) -> c_int {
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    out.write(OirOptions {
        access: OIR_ACCESS_AUTO,
    });
    OIR_OK
}

/// Read the acquisition `path` begins — every part of a split one, see
/// [`oir_acquisition_part`] — into memory ([`Options::gather`], then
/// [`assemble`](crate::Gathered::assemble)). On `OIR_OK`, `*out` is the
/// acquisition, to be freed with [`oir_acquisition_free`]; on anything else it
/// is NULL.
///
/// `path` is NUL-terminated UTF-8. `options` may be NULL for the defaults,
/// `progress` NULL for none; `user` is handed to it untouched.
///
/// # Safety
///
/// `path` points to a NUL-terminated string; `options` is null or points to
/// one [`OirOptions`]; `out` points to writable memory for one pointer;
/// `progress`, if given, is safe to call with `user`.
#[no_mangle]
pub unsafe extern "C" fn oir_read(
    path: *const c_char,
    options: *const OirOptions,
    progress: OirProgress,
    user: *mut c_void,
    out: *mut *mut OirAcquisition,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(OIR_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        let path = match text_arg(path, "path") {
            Ok(p) => p,
            Err(status) => return status,
        };
        let options = match options.as_ref() {
            None => Options::new(),
            Some(o) => match o.to_options() {
                Ok(o) => o,
                Err(why) => return fail(OIR_INVALID_ARGUMENT, why),
            },
        };
        if NO_FILESYSTEM {
            return fail(
                OIR_UNAVAILABLE,
                "this build has no filesystem to read a path from; use oir_read_bytes",
            );
        }
        let mut on_progress = |fraction: f32| match progress {
            Some(report) => report(fraction, user) != 0,
            None => true,
        };
        match options
            .gather(path, &mut on_progress)
            .and_then(crate::Gathered::assemble)
        {
            Ok(inner) => {
                out.write(Box::into_raw(Box::new(OirAcquisition { inner })));
                OIR_OK
            }
            Err(e) => fail_read(&e),
        }
    })
}

/// Read an acquisition from its parts, already in memory and in order
/// ([`Options::gather_bytes`]): `n_parts` parts, part `i` the `lens[i]` bytes
/// at `parts[i]`. Nothing is copied but the pixels, so the caller's buffers are
/// free again as soon as this returns. On `OIR_OK`, `*out` is the acquisition;
/// on anything else it is NULL.
///
/// `progress` may be NULL; `user` is handed to it untouched.
///
/// # Safety
///
/// `parts` and `lens` each point to `n_parts` readable elements (either may be
/// null when `n_parts` is 0); each `parts[i]` is null with `lens[i]` 0, or
/// points to `lens[i]` readable bytes; `out` points to writable memory for one
/// pointer; `progress`, if given, is safe to call with `user`.
#[no_mangle]
pub unsafe extern "C" fn oir_read_bytes(
    parts: *const *const u8,
    lens: *const usize,
    n_parts: usize,
    progress: OirProgress,
    user: *mut c_void,
    out: *mut *mut OirAcquisition,
) -> c_int {
    guarded(|| {
        if out.is_null() {
            return fail(OIR_INVALID_ARGUMENT, "out is NULL");
        }
        out.write(std::ptr::null_mut());
        let mut slices: Vec<&[u8]> = Vec::new();
        if n_parts > 0 {
            if parts.is_null() || lens.is_null() {
                return fail(OIR_INVALID_ARGUMENT, "parts or lens is NULL");
            }
            if n_parts > isize::MAX as usize / std::mem::size_of::<usize>() {
                return fail(
                    OIR_INVALID_ARGUMENT,
                    "n_parts is larger than memory can hold",
                );
            }
            let parts = std::slice::from_raw_parts(parts, n_parts);
            let lens = std::slice::from_raw_parts(lens, n_parts);
            for (i, (&p, &len)) in parts.iter().zip(lens).enumerate() {
                if len > isize::MAX as usize {
                    return fail(
                        OIR_INVALID_ARGUMENT,
                        format!("part {i} is larger than memory can hold"),
                    );
                }
                if p.is_null() {
                    if len != 0 {
                        return fail(
                            OIR_INVALID_ARGUMENT,
                            format!("part {i} is NULL but {len} bytes long"),
                        );
                    }
                    slices.push(&[]);
                } else {
                    slices.push(std::slice::from_raw_parts(p, len));
                }
            }
        }
        let mut on_progress = |fraction: f32| match progress {
            Some(report) => report(fraction, user) != 0,
            None => true,
        };
        match Options::new()
            .gather_bytes(&slices, &mut on_progress)
            .and_then(crate::Gathered::assemble)
        {
            Ok(inner) => {
                out.write(Box::into_raw(Box::new(OirAcquisition { inner })));
                OIR_OK
            }
            Err(e) => fail_read(&e),
        }
    })
}

/// Free an acquisition, and every plane in it. NULL is ignored.
///
/// # Safety
///
/// `a` is null or came from [`oir_read`] or [`oir_read_bytes`] and has not
/// been freed.
#[no_mangle]
pub unsafe extern "C" fn oir_acquisition_free(a: *mut OirAcquisition) {
    if !a.is_null() {
        drop(Box::from_raw(a));
    }
}

/// The stack the planes make up.
///
/// # Safety
///
/// `a` is null or a live acquisition; `out` is null or points to writable
/// memory for one [`OirShape`].
#[no_mangle]
pub unsafe extern "C" fn oir_shape(a: *const OirAcquisition, out: *mut OirShape) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    let s = a.shape();
    out.write(OirShape {
        width: s.width,
        height: s.height,
        channels: s.channels,
        slices: s.slices,
        frames: s.frames,
        sample_type: match s.sample_type {
            SampleType::U8 => OIR_SAMPLE_U8,
            SampleType::U16 => OIR_SAMPLE_U16,
        },
        bytes_per_sample: s.sample_type.bytes(),
        plane_len: s.plane_len(),
        plane_count: s.plane_count(),
    });
    OIR_OK
}

/// Where channel `c` of slice `z` at time `t` is among the planes — channel
/// fastest, then slice, then time ([`Shape::plane_index`]).
///
/// [`Shape::plane_index`]: crate::Shape::plane_index
///
/// # Safety
///
/// `a` is null or a live acquisition; `index` is null or points to one
/// writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn oir_plane_index(
    a: *const OirAcquisition,
    c: usize,
    z: usize,
    t: usize,
    index: *mut usize,
) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    if index.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "index is NULL");
    }
    let s = a.shape();
    match s.plane_index(c, z, t) {
        Some(i) => {
            index.write(i);
            OIR_OK
        }
        None => fail(
            OIR_INVALID_ARGUMENT,
            format!(
                "channel {c}, slice {z}, frame {t} is outside a stack of {} channel(s), {} slice(s), {} frame(s)",
                s.channels, s.slices, s.frames
            ),
        ),
    }
}

/// Planes left out because the acquisition did not finish them
/// ([`Acquisition::dropped_planes`]); 0 for NULL.
///
/// # Safety
///
/// `a` is null or a live acquisition.
#[no_mangle]
pub unsafe extern "C" fn oir_dropped_planes(a: *const OirAcquisition) -> usize {
    a.as_ref().map_or(0, |a| a.inner.dropped_planes())
}

/// Plane `i`'s samples as native-endian bytes, whichever their type.
fn plane_bytes(planes: &Planes, i: usize) -> Option<&[u8]> {
    match planes {
        Planes::U8(p) => p.get(i).map(Vec::as_slice),
        Planes::U16(p) => p.get(i).map(|p| bytemuck::cast_slice(p.as_slice())),
    }
}

/// Plane `i`, refusing a null handle and an index past the end.
///
/// # Safety
///
/// `a` is null or a live acquisition.
unsafe fn plane_of<'a>(a: *const OirAcquisition, i: usize) -> Result<&'a [u8], c_int> {
    let a = handle(a)?;
    plane_bytes(a.planes(), i).ok_or_else(|| {
        fail(
            OIR_INVALID_ARGUMENT,
            format!("plane {i} is past the end; there are {}", a.planes().len()),
        )
    })
}

/// Copy plane `i` into `out`: `plane_len` samples of the acquisition's sample
/// type, native-endian, `plane_len * bytes_per_sample` bytes. `out_bytes` is
/// the room at `out`, and has to be at least that; nothing past it is written.
///
/// # Safety
///
/// `a` is null or a live acquisition; `out` is null or points to `out_bytes`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_plane_copy(
    a: *const OirAcquisition,
    i: usize,
    out: *mut c_void,
    out_bytes: usize,
) -> c_int {
    let plane = match plane_of(a, i) {
        Ok(p) => p,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    if out_bytes < plane.len() {
        return fail(
            OIR_INVALID_ARGUMENT,
            format!("a plane needs {} bytes, not {out_bytes}", plane.len()),
        );
    }
    std::ptr::copy_nonoverlapping(plane.as_ptr(), out.cast::<u8>(), plane.len());
    OIR_OK
}

/// Plane `i` where it already is, without a copy: `*data` points to its
/// `plane_len` samples of the acquisition's sample type (`uint8_t` or
/// `uint16_t`, native-endian, aligned for it), valid until the acquisition is
/// freed. Read-only.
///
/// # Safety
///
/// `a` is null or a live acquisition; `data` is null or points to one writable
/// pointer.
#[no_mangle]
pub unsafe extern "C" fn oir_plane_data(
    a: *const OirAcquisition,
    i: usize,
    data: *mut *const c_void,
) -> c_int {
    if data.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "data is NULL");
    }
    data.write(std::ptr::null());
    match plane_of(a, i) {
        Ok(p) => {
            data.write(p.as_ptr().cast());
            OIR_OK
        }
        Err(status) => status,
    }
}

/// The numbers the acquisition record states, and the counts of its lists.
///
/// # Safety
///
/// `a` is null or a live acquisition; `out` is null or points to writable
/// memory for one [`OirRecordInfo`].
#[no_mangle]
pub unsafe extern "C" fn oir_record_info(
    a: *const OirAcquisition,
    out: *mut OirRecordInfo,
) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    let r = a.record();
    out.write(OirRecordInfo {
        pixel_x_um: r.pixel_x_um().unwrap_or(f64::NAN),
        pixel_y_um: r.pixel_y_um().unwrap_or(f64::NAN),
        z_step_um: r.z_step_um().unwrap_or(f64::NAN),
        frame_interval_s: a.frame_interval_s().unwrap_or(f64::NAN),
        duration_s: r.duration_s(a.shape().frames).unwrap_or(f64::NAN),
        width: r.width().unwrap_or(0),
        height: r.height().unwrap_or(0),
        slices: r.slices().unwrap_or(0),
        frame_time_count: r.frame_times_ms().len(),
        channel_count: r.channels().len(),
        event_count: r.events().len(),
        roundtrip: r.roundtrip().map_or(-1, c_int::from),
        free_run: r.free_run(),
    });
    OIR_OK
}

/// Copy every frame timestamp the acquisition recorded, in milliseconds, sorted
/// ([`Record::frame_times_ms`]): `frame_time_count` doubles, from
/// [`oir_record_info`]. `len` is the room at `out` and has to be at least that.
///
/// # Safety
///
/// `a` is null or a live acquisition; `out` is null or points to `len`
/// writable doubles.
#[no_mangle]
pub unsafe extern "C" fn oir_frame_times(
    a: *const OirAcquisition,
    out: *mut f64,
    len: usize,
) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    let times = a.record().frame_times_ms();
    if times.is_empty() {
        return OIR_OK;
    }
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    if len < times.len() {
        return fail(
            OIR_INVALID_ARGUMENT,
            format!("there are {} frame times, and room for {len}", times.len()),
        );
    }
    std::ptr::copy_nonoverlapping(times.as_ptr(), out, times.len());
    OIR_OK
}

fn record_field(r: &Record, field: c_int) -> Result<Option<&str>, String> {
    Ok(match field {
        OIR_FIELD_FILE_VERSION => r.file_version(),
        OIR_FIELD_CREATED => r.created(),
        OIR_FIELD_SYSTEM_NAME => r.system_name(),
        OIR_FIELD_SYSTEM_VERSION => r.system_version(),
        OIR_FIELD_OBJECTIVE => r.objective(),
        OIR_FIELD_MAGNIFICATION => r.magnification(),
        OIR_FIELD_NUMERICAL_APERTURE => r.numerical_aperture(),
        OIR_FIELD_SCANNER => r.scanner(),
        OIR_FIELD_SAMPLING_SPEED_US => r.sampling_speed_us(),
        OIR_FIELD_SEQUENTIAL_MODE => r.sequential_mode(),
        OIR_FIELD_INTEGRATION => r.integration(),
        OIR_FIELD_INTEGRATION_COUNT => r.integration_count(),
        OIR_FIELD_ROTATION_DEG => r.rotation_deg(),
        OIR_FIELD_PAN_X_UM => r.pan_x_um(),
        OIR_FIELD_PAN_Y_UM => r.pan_y_um(),
        OIR_FIELD_ZOOM => r.zoom(),
        OIR_FIELD_MIRROR_TURRET => r.mirror_turret(),
        OIR_FIELD_LASER_WAVELENGTH_NM => r.laser_wavelength_nm(),
        OIR_FIELD_LASER_TRANSMISSIVITY_PCT => r.laser_transmissivity_pct(),
        v => return Err(format!("field {v} is not one of the OIR_FIELD_* values")),
    })
}

fn channel_field(c: &Channel, field: c_int) -> Result<Option<&str>, String> {
    Ok(match field {
        OIR_CHANNEL_NAME => c.name(),
        OIR_CHANNEL_DETECTOR => c.detector(),
        OIR_CHANNEL_BARRIER_FILTER => c.barrier_filter(),
        OIR_CHANNEL_DICHROIC => c.dichroic(),
        OIR_CHANNEL_DYE => c.dye(),
        OIR_CHANNEL_EMISSION_NM => c.emission_nm(),
        OIR_CHANNEL_BITS => c.bits(),
        OIR_CHANNEL_PMT_VOLTAGE => c.pmt_voltage(),
        OIR_CHANNEL_LABEL => c.label(),
        v => return Err(format!("field {v} is not one of the OIR_CHANNEL_* values")),
    })
}

/// One of the record's text values, an `OIR_FIELD_*`, exactly as the file
/// states it; empty when it does not. The `snprintf` convention: see the
/// header.
///
/// # Safety
///
/// `a` is null or a live acquisition; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_record_string(
    a: *const OirAcquisition,
    field: c_int,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    len_of((|| {
        let a = handle(a)?;
        let value =
            record_field(a.record(), field).map_err(|why| fail(OIR_INVALID_ARGUMENT, why))?;
        Ok(put_text(value.unwrap_or(""), buf, cap))
    })())
}

/// One of channel `index`'s text values, an `OIR_CHANNEL_*`, exactly as the
/// file states it; empty when it does not. The channels are the record's — in
/// recording order, `channel_count` of them — which usually but not always
/// number the stack's. The `snprintf` convention: see the header.
///
/// # Safety
///
/// `a` is null or a live acquisition; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_channel_string(
    a: *const OirAcquisition,
    index: usize,
    field: c_int,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    len_of((|| {
        let a = handle(a)?;
        let channel = index_in(a.record().channels(), index, "channel")?;
        let value = channel_field(channel, field).map_err(|why| fail(OIR_INVALID_ARGUMENT, why))?;
        Ok(put_text(value.unwrap_or(""), buf, cap))
    })())
}

/// Event `index`'s name ([`Event::name`](crate::Event::name)). The `snprintf`
/// convention: see the header.
///
/// # Safety
///
/// `a` is null or a live acquisition; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_event_name(
    a: *const OirAcquisition,
    index: usize,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    len_of((|| {
        let a = handle(a)?;
        let event = index_in(a.record().events(), index, "event")?;
        Ok(put_text(event.name(), buf, cap))
    })())
}

/// When event `index` happened, in milliseconds from the start of the
/// acquisition ([`Event::time_ms`](crate::Event::time_ms)).
///
/// # Safety
///
/// `a` is null or a live acquisition; `time_ms` is null or points to one
/// writable double.
#[no_mangle]
pub unsafe extern "C" fn oir_event_time_ms(
    a: *const OirAcquisition,
    index: usize,
    time_ms: *mut f64,
) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    if time_ms.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "time_ms is NULL");
    }
    match index_in(a.record().events(), index, "event") {
        Ok(e) => {
            time_ms.write(e.time_ms());
            OIR_OK
        }
        Err(status) => status,
    }
}

/// The record as FluoView exports it beside an OIR, `"key"\t"value"` lines,
/// for a stack of exactly the frames read ([`Acquisition::fluoview_text`]);
/// `file_name`, NUL-terminated UTF-8, is the name it states. Empty when the
/// file carried nothing recognisable. The `snprintf` convention: see the
/// header. The text is written afresh on every call, so asking for the length
/// and then the text does the work twice — cheap beside the read.
///
/// # Safety
///
/// `a` is null or a live acquisition; `file_name` points to a NUL-terminated
/// string; `buf` is null or points to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_fluoview_text(
    a: *const OirAcquisition,
    file_name: *const c_char,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    guarded_len(|| {
        len_of((|| {
            let a = handle(a)?;
            let name = text_arg(file_name, "file_name")?;
            let text = a.fluoview_text(name).unwrap_or_default();
            Ok(put_text(&text, buf, cap))
        })())
    })
}

/// How many files the acquisition was read from; 0 for NULL.
///
/// # Safety
///
/// `a` is null or a live acquisition.
#[no_mangle]
pub unsafe extern "C" fn oir_part_count(a: *const OirAcquisition) -> usize {
    a.as_ref().map_or(0, |a| a.inner.parts().len())
}

/// What happened to part `index`.
///
/// # Safety
///
/// `a` is null or a live acquisition; `out` is null or points to writable
/// memory for one [`OirPartInfo`].
#[no_mangle]
pub unsafe extern "C" fn oir_part_info(
    a: *const OirAcquisition,
    index: usize,
    out: *mut OirPartInfo,
) -> c_int {
    let a = match handle(a) {
        Ok(a) => a,
        Err(status) => return status,
    };
    if out.is_null() {
        return fail(OIR_INVALID_ARGUMENT, "out is NULL");
    }
    let p = match index_in(a.parts(), index, "part") {
        Ok(p) => p,
        Err(status) => return status,
    };
    out.write(OirPartInfo {
        size: p.size(),
        read_path: match p.read_path() {
            ReadPath::Mapped => OIR_READ_MAPPED,
            ReadPath::Buffered => OIR_READ_BUFFERED,
            ReadPath::Memory => OIR_READ_MEMORY,
        },
        block_source: match p.block_source() {
            BlockSource::Index => OIR_BLOCKS_INDEX,
            BlockSource::Walk => OIR_BLOCKS_WALK,
        },
        block_count: p.block_count(),
        planes_added: p.planes_added(),
    });
    OIR_OK
}

/// Part `index`'s file, as UTF-8; empty for a part read from memory. The
/// `snprintf` convention: see the header.
///
/// # Safety
///
/// `a` is null or a live acquisition; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_part_path(
    a: *const OirAcquisition,
    index: usize,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    len_of((|| {
        let a = handle(a)?;
        let part = index_in(a.parts(), index, "part")?;
        let path = part.path().map(|p| p.to_string_lossy()).unwrap_or_default();
        Ok(put_text(&path, buf, cap))
    })())
}

/// Why part `index` could not be memory-mapped, when mapping was tried and
/// failed and it was read through a buffer instead; empty otherwise
/// ([`Part::map_error`](crate::Part::map_error)). The `snprintf` convention:
/// see the header.
///
/// # Safety
///
/// `a` is null or a live acquisition; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_part_map_error(
    a: *const OirAcquisition,
    index: usize,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    len_of((|| {
        let a = handle(a)?;
        let part = index_in(a.parts(), index, "part")?;
        Ok(put_text(part.map_error().unwrap_or(""), buf, cap))
    })())
}

/// The files of the acquisition `path` begins, without reading them
/// ([`acquisition_parts`](crate::acquisition_parts)), or the status to return.
unsafe fn parts_of(path: *const c_char) -> Result<Vec<std::path::PathBuf>, c_int> {
    let path = text_arg(path, "path")?;
    if NO_FILESYSTEM {
        return Err(fail(
            OIR_UNAVAILABLE,
            "this build has no filesystem to look for parts in",
        ));
    }
    Ok(crate::acquisition_parts(path))
}

/// How many files make up the acquisition `path` begins: `path` and the
/// continuations beside it, up to the first gap. Only directory entries are
/// read; `path` counts whether or not it exists.
///
/// # Safety
///
/// `path` points to a NUL-terminated string; `count` is null or points to one
/// writable `size_t`.
#[no_mangle]
pub unsafe extern "C" fn oir_acquisition_part_count(
    path: *const c_char,
    count: *mut usize,
) -> c_int {
    guarded(|| {
        if count.is_null() {
            return fail(OIR_INVALID_ARGUMENT, "count is NULL");
        }
        match parts_of(path) {
            Ok(parts) => {
                count.write(parts.len());
                OIR_OK
            }
            Err(status) => status,
        }
    })
}

/// File `index` of the acquisition `path` begins, as UTF-8 — index 0 is
/// `path` itself. The directory is looked at again on every call. The
/// `snprintf` convention: see the header.
///
/// # Safety
///
/// `path` points to a NUL-terminated string; `buf` is null or points to `cap`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn oir_acquisition_part(
    path: *const c_char,
    index: usize,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    guarded_len(|| {
        len_of((|| {
            let parts = parts_of(path)?;
            let part = index_in(&parts, index, "part")?;
            Ok(put_text(&part.to_string_lossy(), buf, cap))
        })())
    })
}

#[cfg(test)]
#[path = "capi_tests.rs"]
mod tests;

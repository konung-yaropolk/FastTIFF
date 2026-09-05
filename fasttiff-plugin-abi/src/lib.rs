//! The binary contract between FastTIFF and a plugin shared library.
//!
//! # The rule
//!
//! Nothing crosses this boundary but fixed-width integers, `f32`/`f64`, raw
//! pointers, `extern "C"` function pointers, and `#[repr(C)]` structs and
//! `#[repr(u32)]` enums built only from those.
//!
//! No `Vec`, no `String`, no `&mut`, no `dyn Trait`, no `Result`, no
//! `Option<T>` of a non-pointer, no `egui::Ui`, no monomorphised generic.
//! Those have a layout Rust does not promise to keep stable between compiler
//! versions, optimisation levels or dependency resolutions — which is exactly
//! the set of things a third party building a plugin in five years will have
//! different from the host that loads it. The types here have a layout fixed by
//! the *target triple*, which both sides already agree on.
//!
//! # Five things make it sound
//!
//! **The major version is in the symbol name.** A v1 host looks up
//! [`QUERY_SYMBOL`] — `ft_plugin_v1_query`. A v2 host will look for
//! `ft_plugin_v2_query` and simply not find it in a v1 library, so the
//! mismatch is a clean missing-symbol error at load rather than two sides
//! disagreeing about a struct layout after the call has already begun.
//!
//! **Every struct starts with `struct_size`.** Fields are only ever appended.
//! Both sides read `min(theirs, mine)` bytes and treat the rest as absent, so a
//! plugin built against an older minor version runs on a newer host and vice
//! versa. [`ABI_MINOR`] says which fields to expect.
//!
//! **Layout is pinned by tests, not by hope.** `layout_tests.rs` asserts
//! hard-coded sizes and offsets. Reordering a field fails CI rather than a
//! user's microscope.
//!
//! **No allocator is shared.** The plugin never frees host memory and the host
//! never frees plugin memory. Pixels flow host → plugin into a buffer the
//! *plugin* allocated and sized; results flow plugin → host through a sink the
//! host copies out of before the call returns. A plugin linked against a
//! different `malloc` is therefore fine.
//!
//! **Panics never unwind across the boundary.** Unwinding out of an
//! `extern "C"` function is undefined behaviour, and this workspace builds with
//! `panic = "unwind"`. Every entry point on both sides is wrapped in
//! `catch_unwind` and converted to [`FtStatus::Panic`].
//!
//! # What a plugin author sees
//!
//! None of this, ideally. The `fasttiff-plugin` crate wraps it: an author
//! implements the ordinary Rust `Plugin` or `Importer` trait from
//! `fasttiff-plugin-api` and writes `export_plugin!`.

#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

/// The contract's major version. Part of [`QUERY_SYMBOL`].
pub const ABI_MAJOR: u32 = 1;

/// The contract's minor version: how many optional trailing fields exist.
/// Bumped when a field is appended; never when one changes meaning.
pub const ABI_MINOR: u32 = 0;

/// The one symbol a plugin library must export, NUL-terminated for `dlsym`.
///
/// The major version is part of the name on purpose — see the module docs.
pub const QUERY_SYMBOL: &[u8] = b"ft_plugin_v1_query\0";

/// The signature of [`QUERY_SYMBOL`].
///
/// Called once when the host first loads the library. The plugin registers
/// everything it provides through `registrar` and returns [`FtStatus::Ok`].
pub type FtQueryFn = unsafe extern "C" fn(registrar: *mut FtRegistrar) -> FtStatus;

/// Define an *open* enumeration: a `#[repr(transparent)]` `u32` newtype with
/// named constants, rather than a Rust `enum`.
///
/// This is not stylistic. A Rust enum with variants `0..=6` holding the value
/// `7` is **undefined behaviour**, and the whole forward-compatibility story
/// here is that a plugin built against a later minor version may hand back a
/// value this host has never heard of. An `enum` in that position would make
/// the documented evolution path UB on receipt, before any code could check
/// it. A newtype has no invalid values, so an unknown one is data — matched
/// with a wildcard and refused, or ignored, as each site decides.
///
/// The constants keep the `Ok`/`U16` spelling of variants deliberately: they
/// read the same at every call site, and the wire numbers stay visible in one
/// place.
macro_rules! open_enum {
    (
        $(#[$outer:meta])*
        $name:ident {
            $( $(#[$inner:meta])* $variant:ident = $value:literal ),* $(,)?
        }
    ) => {
        $(#[$outer])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub u32);

        #[allow(non_upper_case_globals)]
        impl $name {
            $( $(#[$inner])* pub const $variant: $name = $name($value); )*
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.0 {
                    $( $value => f.write_str(stringify!($variant)), )*
                    other => write!(f, "{}({})", stringify!($name), other),
                }
            }
        }
    };
}

open_enum! {
    /// How a call ended.
    ///
    /// These numbers are part of the contract. Never renumber, never remove;
    /// append only.
    FtStatus {
        Ok = 0,
        /// The plugin failed. Its reason is available from `last_error`.
        Error = 1,
        /// The user cancelled.
        Cancelled = 2,
        /// A panic was caught at the boundary. The plugin is not trusted again
        /// during this run.
        Panic = 3,
        /// Not applicable to this stack — distinct from `Error`, and the host
        /// says so differently.
        Unsupported = 4,
        /// A pointer was null, a length absurd, or a string not UTF-8.
        BadArgument = 5,
        /// A plane outside the stack was requested.
        OutOfRange = 6,
    }
}

/// Borrowed UTF-8. **Not** NUL-terminated, and valid only for the duration of
/// the call that produced it.
///
/// The receiver validates with `from_utf8` — never the unchecked form — and
/// copies before returning. A plugin that hands back rubbish gets
/// [`FtStatus::BadArgument`], not a crash.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtStr {
    /// May be null only when `len` is 0.
    pub ptr: *const u8,
    pub len: u64,
}

impl FtStr {
    pub const EMPTY: FtStr = FtStr {
        ptr: core::ptr::null(),
        len: 0,
    };

    /// Borrow a `&str` as an `FtStr`. The borrow must outlive the call.
    // Not `std::str::FromStr`: that trait is fallible and owns its output,
    // while this borrows and cannot fail.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> FtStr {
        FtStr {
            ptr: s.as_ptr(),
            len: s.len() as u64,
        }
    }

    /// # Safety
    /// `ptr` must be valid for `len` bytes for the lifetime `'a`.
    pub unsafe fn as_str<'a>(&self) -> Option<&'a str> {
        if self.len == 0 {
            return Some("");
        }
        if self.ptr.is_null() {
            return None;
        }
        // A length past what any real string can be is a corrupt struct, not a
        // long name; refuse rather than construct an enormous slice.
        if self.len > (1 << 30) {
            return None;
        }
        let bytes = core::slice::from_raw_parts(self.ptr, self.len as usize);
        core::str::from_utf8(bytes).ok()
    }
}

// ------------------------------------------------- versioned-struct handling

/// The `struct_size` a caller declared, read without forming a reference.
///
/// Every struct that crosses this boundary begins with `struct_size: u32`, and
/// this reads exactly that — four bytes — and nothing else.
///
/// The distinction matters more than it looks. `&*ptr` on a struct the other
/// side built to an *older*, smaller layout is undefined behaviour the instant
/// the reference exists, whether or not the trailing fields are ever touched:
/// a reference must be dereferenceable for the whole of its type. So the size
/// check cannot be written as "make the reference, then look at its
/// `struct_size`" — by then the damage is done. It has to be this, first.
///
/// # Safety
/// `p` must point at four readable, `u32`-aligned bytes. That is the floor of
/// the contract: a caller passing a pointer that cannot even carry its own
/// prologue is not participating in this ABI, and no check can rescue it. (The
/// same floor every versioned-struct ABI assumes — Vulkan's `sType`, COM's
/// `cbSize`.)
#[inline]
pub unsafe fn declared_size<T>(p: *const T) -> u32 {
    p.cast::<u32>().read()
}

/// Whether a `&T` may be formed from `p`: non-null, and at least `size_of::<T>()`
/// bytes according to the caller's own `struct_size`.
///
/// # Safety
/// As [`declared_size`].
#[inline]
pub unsafe fn fits<T>(p: *const T) -> bool {
    !p.is_null() && declared_size(p) as usize >= core::mem::size_of::<T>()
}

/// Fill a caller-supplied out-parameter, writing only as much as the caller
/// said it allocated.
///
/// The mirror of [`fits`] for the other direction. When the receiver is older
/// than the writer, its struct is a *prefix* of the writer's — fields are only
/// ever appended — so writing `min(theirs, ours)` bytes of a locally built
/// value gives it exactly the fields it knows about and touches nothing beyond.
/// Writing `size_of::<T>()` unconditionally would run past the end of a struct
/// the caller allocated to a smaller layout.
///
/// The caller's own `struct_size` is preserved, not overwritten with ours: it
/// describes their allocation, not our idea of it.
///
/// Returns `false` without writing if `out` is null or declares less than the
/// four bytes of its own prologue.
///
/// # Safety
/// `out` must point at `declared_size(out)` writable bytes, correctly aligned
/// for `T`, and `T` must begin with `struct_size: u32`.
pub unsafe fn write_prefix<T: Copy>(out: *mut T, mut value: T) -> bool {
    if out.is_null() {
        return false;
    }
    let declared = declared_size(out.cast_const()) as usize;
    if declared < core::mem::size_of::<u32>() {
        return false;
    }
    // Keep the size the caller wrote; the rest of `value` is ours.
    *(&mut value as *mut T).cast::<u32>() = declared as u32;
    let n = declared.min(core::mem::size_of::<T>());
    core::ptr::copy_nonoverlapping((&value as *const T).cast::<u8>(), out.cast::<u8>(), n);
    true
}

open_enum! {
    /// Sample format of a plane, matching `fasttiff_plugin_api::PixelType`.
    FtPixelType {
        U8 = 0,
        U16 = 1,
        I16 = 2,
        F32 = 3,
    }
}

open_enum! {
    /// A dialog control's kind, matching `fasttiff_plugin_api::ParamKind`.
    FtParamKind {
        Int = 0,
        Float = 1,
        Bool = 2,
        Choice = 3,
        Text = 4,
        Path = 5,
        Label = 6,
    }
}

/// One declared control.
///
/// A single struct rather than a tagged union, because a union's layout is the
/// hardest thing to keep stable across compilers and the waste here is a few
/// dozen bytes per control in a dialog that has perhaps ten.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtParamDecl {
    pub struct_size: u32,
    pub kind: FtParamKind,
    pub key: FtStr,
    pub label: FtStr,
    /// Empty when there is none.
    pub help: FtStr,
    /// `Int`: the default, min and max. `Choice`: the default index in
    /// `i_default`. Unused by other kinds.
    pub i_default: i64,
    pub i_min: i64,
    pub i_max: i64,
    /// `Float`: the default, min and max.
    pub f_default: f64,
    pub f_min: f64,
    pub f_max: f64,
    /// `Bool`: the default, as 0 or 1.
    pub b_default: u32,
    /// `Path`: 1 for a save dialog, 0 for open.
    pub save: u32,
    /// `Text` and `Path`: the default.
    pub s_default: FtStr,
    /// `Choice`: the options, as an array the plugin owns for the call.
    pub options: *const FtStr,
    pub option_count: u64,
}

/// One value the user chose.///
/// **Arrayed: this struct's size is frozen.** It is passed as `*const` plus a
/// count, so both sides walk it with `ptr.add(i)` — using *their own*
/// `size_of` as the stride. Appending a field would change that stride on one
/// side only, and every element after the first would be read from the wrong
/// address: the append-a-field rule that the rest of this contract relies on
/// does not reach through an array. To carry more per-element data in a later
/// version, add a *parallel* array with its own count, or a new struct; never
/// grow this one. `layout_tests.rs` pins the size so a well-meaning addition
/// fails the build rather than a user's microscope.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtValue {
    pub struct_size: u32,
    pub kind: FtParamKind,
    pub key: FtStr,
    pub i: i64,
    pub f: f64,
    pub b: u32,
    pub _pad: u32,
    pub s: FtStr,
}

/// The stack's shape, in file coordinates.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtImageInfo {
    pub struct_size: u32,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u32,
    pub channels: u64,
    pub slices: u64,
    pub frames: u64,
    pub pixel_type: FtPixelType,
    pub _pad: u32,
}

/// One displayed channel's contrast window.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtChannelView {
    /// Present for the same reason every other struct here has one: the host
    /// *writes* into this, and without a declared size it would write
    /// `size_of::<FtChannelView>()` bytes into whatever an older plugin
    /// allocated. See [`write_prefix`].
    pub struct_size: u32,
    pub _pad0: u32,
    pub min: f32,
    pub max: f32,
    /// 0 or 1.
    pub enabled: u32,
    pub _pad: u32,
}

/// What the viewer is showing, snapshotted for the run.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtViewParams {
    pub struct_size: u32,
    /// 0 or 1.
    pub volume_view: u32,
    pub frame_index: u64,
    /// How many entries `channel` may be asked for. Capped by the renderer's
    /// slot count, so it can be fewer than `FtImageInfo::channels`.
    pub shown_channels: u64,
    /// 0 = MIP, 1 = alpha DVR, 2 = isosurface.
    pub volume_mode: u32,
    pub _pad: u32,
    pub density: f32,
    pub iso: f32,
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
}

/// Callbacks the host provides. Every one takes the opaque `ctx` first.
///
/// A plugin must check `struct_size` before using any field: a host older than
/// the plugin will have a shorter struct, and reading past it is exactly the
/// kind of thing this prologue exists to prevent.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtHost {
    pub struct_size: u32,
    pub _pad: u32,
    pub ctx: *mut c_void,

    /// Fill `out` with the stack's shape.
    pub image_info: unsafe extern "C" fn(ctx: *mut c_void, out: *mut FtImageInfo) -> FtStatus,
    /// Fill `out` with the viewer's state.
    pub view_params: unsafe extern "C" fn(ctx: *mut c_void, out: *mut FtViewParams) -> FtStatus,
    /// One displayed channel's window; `index < FtViewParams::shown_channels`.
    pub channel_view:
        unsafe extern "C" fn(ctx: *mut c_void, index: u64, out: *mut FtChannelView) -> FtStatus,
    /// The document's name. Borrowed; copy before returning.
    pub stack_name: unsafe extern "C" fn(ctx: *mut c_void) -> FtStr,
    /// The file's path, or an empty string for a stack loaded from bytes.
    pub stack_path: unsafe extern "C" fn(ctx: *mut c_void) -> FtStr,

    /// Decode one plane as `f32`, in the file's own units, into a buffer the
    /// **plugin** owns. `cap` must be at least `width * height`; the host
    /// writes exactly that many and never more.
    pub read_plane_f32: unsafe extern "C" fn(
        ctx: *mut c_void,
        c: u64,
        z: u64,
        t: u64,
        out: *mut f32,
        cap: u64,
    ) -> FtStatus,
    /// As [`read_plane_f32`](Self::read_plane_f32), in `u16` display units.
    pub read_plane_u16: unsafe extern "C" fn(
        ctx: *mut c_void,
        c: u64,
        z: u64,
        t: u64,
        out: *mut u16,
        cap: u64,
    ) -> FtStatus,

    /// Report progress `0.0..=1.0`. Returns 0 when the user has cancelled.
    pub progress: unsafe extern "C" fn(ctx: *mut c_void, fraction: f32) -> u32,
    /// A line for the host to show.
    pub log: unsafe extern "C" fn(ctx: *mut c_void, message: FtStr),

    /// The file's spacing, calibration and display mode.
    pub stack_info: unsafe extern "C" fn(ctx: *mut c_void, out: *mut FtStackInfo) -> FtStatus,
    /// One of the file's strings, chosen by `which` (an `FT_STRING_*`
    /// constant). `index` is used only by `FT_STRING_CHANNEL_NAME`. Returns an
    /// empty string when there is none — a single call rather than one
    /// callback per string, because the set of strings is the part of this
    /// contract most likely to grow, and a selector grows without touching the
    /// table's layout.
    pub stack_string: unsafe extern "C" fn(ctx: *mut c_void, which: u32, index: u64) -> FtStr,
}

open_enum! {
    /// What a plugin decided to produce. Matches `fasttiff_plugin_api::Outcome`.
    FtOutcomeKind {
        Nothing = 0,
        Message = 1,
        NewDocument = 2,
        SaveToFile = 3,
    }
}

/// The file's own metadata: what a plugin needs to turn pixel indices into
/// microns and seconds.
///
/// Not optional decoration. A plugin that measures anything — a distance, a
/// volume, a rate — is wrong without the calibration, and wrong *quietly*: the
/// numbers still come out, in the wrong units. Before this crossed, a plugin in
/// a shared library saw zeros here while the identical plugin compiled into the
/// app saw the real values, which is the worst possible way for the two to
/// differ.
///
/// The strings (`unit`, `description`) are not in here because they are
/// borrowed, not owned; they come from their own callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtStackInfo {
    pub struct_size: u32,
    /// 0 grayscale, 1 composite, 2 color — matching
    /// `fasttiff_plugin_api::DisplayMode`.
    pub mode: u32,
    /// Which of the numbers below the file actually stated: see the
    /// `FT_HAS_*` constants. `Option<f64>` has no C spelling, and a sentinel
    /// value would be indistinguishable from a real measurement, so presence
    /// is carried separately.
    pub present: u32,
    pub _pad: u32,
    /// Physical pixel size, in `unit`s.
    pub spacing_x: f64,
    pub spacing_y: f64,
    /// Z step between slices.
    pub spacing_z: f64,
    /// Seconds between timepoints.
    pub frame_interval_s: f64,
    /// Linear calibration: a raw sample `r` means `offset + scale * r`.
    pub calibration_offset: f64,
    pub calibration_scale: f64,
}

/// Flags for [`FtStackInfo::present`].
pub const FT_HAS_SPACING_X: u32 = 1 << 0;
pub const FT_HAS_SPACING_Y: u32 = 1 << 1;
pub const FT_HAS_SPACING_Z: u32 = 1 << 2;
pub const FT_HAS_FRAME_INTERVAL: u32 = 1 << 3;
pub const FT_HAS_CALIBRATION: u32 = 1 << 4;

/// Which string [`FtHost::stack_string`] should return.
pub const FT_STRING_UNIT: u32 = 0;
/// The file's `ImageDescription`, verbatim.
pub const FT_STRING_DESCRIPTION: u32 = 1;
/// The name of channel `index`, when the file names them.
pub const FT_STRING_CHANNEL_NAME: u32 = 2;

/// Where a plugin puts its result. The host owns everything pushed here and
/// copies it before the call returns, so no allocation crosses the boundary.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtSink {
    pub struct_size: u32,
    pub _pad: u32,
    pub ctx: *mut c_void,

    /// Declare the result's shape. Must precede any `push_plane`.
    pub begin_image: unsafe extern "C" fn(
        ctx: *mut c_void,
        width: u32,
        height: u32,
        channels: u64,
        slices: u64,
        frames: u64,
        pixel_type: FtPixelType,
        name: FtStr,
    ) -> FtStatus,
    /// One plane, in `xyczt` order. `data` points at `len` *samples* of the
    /// declared type; the host copies immediately.
    pub push_plane:
        unsafe extern "C" fn(ctx: *mut c_void, data: *const c_void, len: u64) -> FtStatus,
    /// What to do with what was pushed.
    pub set_outcome:
        unsafe extern "C" fn(ctx: *mut c_void, kind: FtOutcomeKind, text: FtStr) -> FtStatus,
    /// Metadata for the result: what an importer read out of the file it just
    /// parsed. Optional — a filter transforming an open stack has nothing to
    /// add here, and the host keeps what the source already had.
    pub set_info: unsafe extern "C" fn(
        ctx: *mut c_void,
        info: *const FtStackInfo,
        unit: FtStr,
        description: FtStr,
    ) -> FtStatus,
}

/// Where a plugin declares its dialog, one control at a time.
///
/// A callback rather than a returned array, so the host owns the collection and
/// no ownership question arises at all.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtParamSink {
    pub struct_size: u32,
    pub _pad: u32,
    pub ctx: *mut c_void,
    pub push: unsafe extern "C" fn(ctx: *mut c_void, decl: *const FtParamDecl) -> FtStatus,
}

/// A processing plugin's entry points.
///
/// Stateless on purpose: there is no opaque instance handle whose lifetime both
/// sides must agree on, and nothing to leak if a call fails. A plugin that
/// needs state keeps it in its own statics, where its own allocator owns it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtPluginVtable {
    pub struct_size: u32,
    pub _pad: u32,
    /// Declare the dialog.
    pub params: unsafe extern "C" fn(host: *const FtHost, sink: *const FtParamSink) -> FtStatus,
    /// Do the work.
    pub run: unsafe extern "C" fn(
        host: *const FtHost,
        values: *const FtValue,
        value_count: u64,
        sink: *const FtSink,
    ) -> FtStatus,
    /// Why the last call returned a non-`Ok` status. Borrowed from the plugin;
    /// the host copies it at once.
    pub last_error: unsafe extern "C" fn() -> FtStr,
}

/// An importer's entry points.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtImporterVtable {
    pub struct_size: u32,
    pub _pad: u32,
    /// How sure this importer is about a file, given its first bytes.
    /// Returns an [`FtConfidence`].
    pub probe: unsafe extern "C" fn(path: FtStr, head: *const u8, head_len: u64) -> u32,
    /// Declare a dialog for this file, if any.
    pub params: unsafe extern "C" fn(path: FtStr, sink: *const FtParamSink) -> FtStatus,
    /// Read the file into `sink`.
    pub import: unsafe extern "C" fn(
        path: FtStr,
        values: *const FtValue,
        value_count: u64,
        host: *const FtHost,
        sink: *const FtSink,
    ) -> FtStatus,
    pub last_error: unsafe extern "C" fn() -> FtStr,
}

open_enum! {
    /// How sure an importer is. Matches `fasttiff_plugin_api::Confidence`.
    ///
    /// Carried as a bare `u32` in [`FtImporterVtable::probe`]'s return, because
    /// that is the only place it crosses.
    FtConfidence {
        No = 0,
        Maybe = 1,
        Certain = 2,
    }
}

/// One file type an importer offers.///
/// **Arrayed: this struct's size is frozen.** It is passed as `*const` plus a
/// count, so both sides walk it with `ptr.add(i)` — using *their own*
/// `size_of` as the stride. Appending a field would change that stride on one
/// side only, and every element after the first would be read from the wrong
/// address: the append-a-field rule that the rest of this contract relies on
/// does not reach through an array. To carry more per-element data in a later
/// version, add a *parallel* array with its own count, or a new struct; never
/// grow this one. `layout_tests.rs` pins the size so a well-meaning addition
/// fails the build rather than a user's microscope.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtFileType {
    pub struct_size: u32,
    pub _pad: u32,
    pub description: FtStr,
    /// Lowercase, without the dot.
    pub extensions: *const FtStr,
    pub extension_count: u64,
}

/// A plugin being registered.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtPluginDesc {
    pub struct_size: u32,
    pub _pad: u32,
    pub id: FtStr,
    pub name: FtStr,
    pub menu_path: FtStr,
    pub version: FtStr,
    pub author: FtStr,
    pub description: FtStr,
    pub vtable: *const FtPluginVtable,
}

/// An importer being registered.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtImporterDesc {
    pub struct_size: u32,
    pub _pad: u32,
    pub id: FtStr,
    pub name: FtStr,
    pub version: FtStr,
    pub author: FtStr,
    pub description: FtStr,
    pub file_types: *const FtFileType,
    pub file_type_count: u64,
    pub vtable: *const FtImporterVtable,
}

/// What the plugin registers itself through, handed to [`FtQueryFn`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FtRegistrar {
    pub struct_size: u32,
    /// The host's ABI minor version. A plugin built against a newer minor must
    /// not use fields the host does not have.
    pub host_abi_minor: u32,
    pub ctx: *mut c_void,
    /// The plugin's ABI minor version, written by the plugin.
    pub plugin_abi_minor: u32,
    pub _pad: u32,
    pub add_plugin: unsafe extern "C" fn(ctx: *mut c_void, desc: *const FtPluginDesc) -> FtStatus,
    pub add_importer:
        unsafe extern "C" fn(ctx: *mut c_void, desc: *const FtImporterDesc) -> FtStatus,
}

// ------------------------------------------------------------ layout, pinned
//
// Asserted in a `const` block rather than a `#[test]`, deliberately.
//
// A layout is a compile-time property, so a compile-time assertion checks it
// for **every target this crate is built for** — including the 32-bit ones
// nobody runs a test suite on. The runtime version could only check the host
// that happened to run `cargo test`, and the earlier form of these tests
// quietly returned early on any target that was not 64-bit, which meant they
// passed green on i686 while asserting nothing at all.
//
// Reordering a field, changing a type, or inserting one anywhere but the end
// now fails the build, on the target where it is wrong, before anything ships.
// The alternative is a plugin built last year reading a host's struct with last
// year's offsets and getting silent nonsense.
//
// When the contract genuinely grows, a field is *appended* and `ABI_MINOR` is
// bumped; the numbers below then change only by adding new ones. The two
// arrayed structs are the exception — see their doc comments.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};

    // Every struct that crosses begins with `struct_size`, or the versioning
    // scheme has nothing to stand on.
    assert!(offset_of!(FtStr, ptr) == 0);
    assert!(offset_of!(FtParamDecl, struct_size) == 0);
    assert!(offset_of!(FtValue, struct_size) == 0);
    assert!(offset_of!(FtImageInfo, struct_size) == 0);
    assert!(offset_of!(FtChannelView, struct_size) == 0);
    assert!(offset_of!(FtViewParams, struct_size) == 0);
    assert!(offset_of!(FtHost, struct_size) == 0);
    assert!(offset_of!(FtStackInfo, struct_size) == 0);
    assert!(offset_of!(FtSink, struct_size) == 0);
    assert!(offset_of!(FtParamSink, struct_size) == 0);
    assert!(offset_of!(FtPluginVtable, struct_size) == 0);
    assert!(offset_of!(FtImporterVtable, struct_size) == 0);
    assert!(offset_of!(FtFileType, struct_size) == 0);
    assert!(offset_of!(FtPluginDesc, struct_size) == 0);
    assert!(offset_of!(FtImporterDesc, struct_size) == 0);
    assert!(offset_of!(FtRegistrar, struct_size) == 0);

    // The open enumerations are `u32` on the wire.
    assert!(size_of::<FtStatus>() == 4);
    assert!(size_of::<FtPixelType>() == 4);
    assert!(size_of::<FtParamKind>() == 4);
    assert!(size_of::<FtOutcomeKind>() == 4);
    assert!(size_of::<FtConfidence>() == 4);

    // A pointer followed by a `u64` is 8-aligned on every target, so these
    // sizes are the same on 32- and 64-bit and can be written as one number.
    assert!(size_of::<FtStr>() == 16);
    assert!(align_of::<FtStr>() == 8);
    assert!(offset_of!(FtStr, len) == 8);

    assert!(size_of::<FtImageInfo>() == 48);
    assert!(offset_of!(FtImageInfo, width) == 4);
    assert!(offset_of!(FtImageInfo, height) == 8);
    assert!(offset_of!(FtImageInfo, samples_per_pixel) == 12);
    assert!(offset_of!(FtImageInfo, channels) == 16);
    assert!(offset_of!(FtImageInfo, slices) == 24);
    assert!(offset_of!(FtImageInfo, frames) == 32);
    assert!(offset_of!(FtImageInfo, pixel_type) == 40);

    assert!(size_of::<FtChannelView>() == 24);
    assert!(offset_of!(FtChannelView, min) == 8);
    assert!(offset_of!(FtChannelView, max) == 12);
    assert!(offset_of!(FtChannelView, enabled) == 16);

    assert!(offset_of!(FtViewParams, volume_view) == 4);
    assert!(offset_of!(FtViewParams, frame_index) == 8);
    assert!(offset_of!(FtViewParams, shown_channels) == 16);
    assert!(offset_of!(FtViewParams, volume_mode) == 24);
    assert!(offset_of!(FtViewParams, density) == 32);
    assert!(offset_of!(FtViewParams, iso) == 36);
    assert!(offset_of!(FtViewParams, eye) == 40);

    assert!(offset_of!(FtParamDecl, kind) == 4);
    assert!(offset_of!(FtParamDecl, key) == 8);
    assert!(offset_of!(FtParamDecl, label) == 24);
    assert!(offset_of!(FtParamDecl, help) == 40);

    assert!(offset_of!(FtValue, kind) == 4);
    assert!(offset_of!(FtValue, key) == 8);

    assert!(offset_of!(FtFileType, description) == 8);

    // Arrayed, so their size is part of the contract in a way the others' is
    // not: both sides walk them with their own `size_of` as the stride.
    assert!(size_of::<FtValue>() == 64);
    assert!(size_of::<FtFileType>() == 40);
    assert!(align_of::<FtValue>() == 8);
    assert!(align_of::<FtFileType>() == 8);

    // The tables are a size, a pad, and then nothing but pointers, so their
    // size scales with the target's pointer width rather than being fixed.
    let p = size_of::<*const u8>();
    assert!(size_of::<FtPluginVtable>() == 8 + 3 * p);
    assert!(size_of::<FtImporterVtable>() == 8 + 4 * p);
    assert!(size_of::<FtParamSink>() == 8 + 2 * p);
    assert!(size_of::<FtSink>() == 8 + 5 * p);
    assert!(size_of::<FtHost>() == 8 + 12 * p);

    assert!(size_of::<FtStackInfo>() == 64);
    assert!(offset_of!(FtStackInfo, mode) == 4);
    assert!(offset_of!(FtStackInfo, present) == 8);
    assert!(offset_of!(FtStackInfo, spacing_x) == 16);
    assert!(offset_of!(FtStackInfo, calibration_scale) == 56);

    // The wire numbers of every open enumeration. Renumbering silently changes
    // what a plugin built against the old values is telling the host.
    assert!(FtStatus::Ok.0 == 0);
    assert!(FtStatus::Error.0 == 1);
    assert!(FtStatus::Cancelled.0 == 2);
    assert!(FtStatus::Panic.0 == 3);
    assert!(FtStatus::Unsupported.0 == 4);
    assert!(FtStatus::BadArgument.0 == 5);
    assert!(FtStatus::OutOfRange.0 == 6);

    assert!(FtPixelType::U8.0 == 0);
    assert!(FtPixelType::U16.0 == 1);
    assert!(FtPixelType::I16.0 == 2);
    assert!(FtPixelType::F32.0 == 3);

    assert!(FtParamKind::Int.0 == 0);
    assert!(FtParamKind::Float.0 == 1);
    assert!(FtParamKind::Bool.0 == 2);
    assert!(FtParamKind::Choice.0 == 3);
    assert!(FtParamKind::Text.0 == 4);
    assert!(FtParamKind::Path.0 == 5);
    assert!(FtParamKind::Label.0 == 6);

    assert!(FtConfidence::No.0 == 0);
    assert!(FtConfidence::Maybe.0 == 1);
    assert!(FtConfidence::Certain.0 == 2);

    assert!(FtOutcomeKind::Nothing.0 == 0);
    assert!(FtOutcomeKind::Message.0 == 1);
    assert!(FtOutcomeKind::NewDocument.0 == 2);
    assert!(FtOutcomeKind::SaveToFile.0 == 3);
};

/// Whether `name` is the entry symbol this ABI's host looks up.
///
/// A symbol name cannot be built from a constant, so `export_plugin!` spells
/// `ft_plugin_v1_query` out as a literal. This lets the macro assert, at
/// compile time in every plugin, that its literal still matches
/// [`QUERY_SYMBOL`] — otherwise bumping [`ABI_MAJOR`] would update the constant
/// and the host's lookup while leaving every plugin quietly exporting the old
/// name.
pub const fn is_query_symbol(name: &[u8]) -> bool {
    if name.len() != QUERY_SYMBOL.len() {
        return false;
    }
    let mut i = 0;
    while i < name.len() {
        if name[i] != QUERY_SYMBOL[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod layout_tests;

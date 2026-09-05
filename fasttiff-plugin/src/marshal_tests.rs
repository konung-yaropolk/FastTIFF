//! Version skew, from the plugin's side.
//!
//! The contract promises that "an older plugin runs on a newer host and vice
//! versa". Only half of that was true: a plugin refused any host whose tables
//! were smaller than its own, so installing a plugin built against a newer ABI
//! than the running application failed the run with `BadArgument` and no
//! message at all — which is exactly how it was reported.
//!
//! The fix is that a size check answers "is this field there", not "is this
//! host at least as new as me". These pin both directions of that.

use super::*;

unsafe extern "C" fn begin(
    _c: *mut core::ffi::c_void,
    _w: u32,
    _h: u32,
    _ch: u64,
    _sl: u64,
    _f: u64,
    _t: FtPixelType,
    _n: FtStr,
) -> FtStatus {
    FtStatus::Ok
}
unsafe extern "C" fn push(
    _c: *mut core::ffi::c_void,
    _d: *const core::ffi::c_void,
    _l: u64,
) -> FtStatus {
    FtStatus::Ok
}
unsafe extern "C" fn outcome(_c: *mut core::ffi::c_void, _k: FtOutcomeKind, _t: FtStr) -> FtStatus {
    FtStatus::Ok
}
unsafe extern "C" fn info(
    _c: *mut core::ffi::c_void,
    _i: *const FtStackInfo,
    _u: FtStr,
    _d: FtStr,
) -> FtStatus {
    FtStatus::Ok
}

/// Counts the calls a newer host would receive, so "skipped" can be told from
/// "called and ignored".
static CHANNEL_CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

unsafe extern "C" fn channel(_c: *mut core::ffi::c_void, _i: u64, _n: FtStr, _r: u32) -> FtStatus {
    CHANNEL_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    FtStatus::Ok
}

/// A sink declaring `size` bytes, whatever this build's `FtSink` really is.
fn sink(size: usize) -> FtSink {
    FtSink {
        struct_size: size as u32,
        _pad: 0,
        ctx: core::ptr::null_mut(),
        begin_image: begin,
        push_plane: push,
        set_outcome: outcome,
        set_info: info,
        set_channel: channel,
    }
}

/// The reported failure: a host from before `set_channel` existed.
#[test]
fn a_host_older_than_this_plugin_still_gets_its_result() {
    let older = core::mem::offset_of!(FtSink, set_info);
    assert!(
        older >= FtSink::CORE && older < core::mem::size_of::<FtSink>(),
        "the fixture must be smaller than today's sink but still a valid one"
    );
    let s = sink(older);
    crate::last_error::set("");
    // SAFETY: every callback in the fixture is real; only the declared size is
    // smaller than this build's.
    let st = unsafe { write_outcome(&s, Outcome::Nothing) };
    assert_eq!(
        st,
        FtStatus::Ok,
        "an older host was refused: {:?}",
        unsafe { crate::last_error::get().as_str() }
    );
}

/// The optional callback must actually be skipped, not called into a table
/// that does not contain it.
#[test]
fn an_optional_callback_is_skipped_rather_than_called_past_the_end() {
    let img = crate::api::ImageResult {
        width: 2,
        height: 1,
        channels: 2,
        slices: 1,
        frames: 1,
        pixel_type: crate::api::PixelType::U8,
        planes: vec![
            crate::api::PlaneData::U8(vec![1, 2]),
            crate::api::PlaneData::U8(vec![3, 4]),
        ],
        channel_colors: vec![[255, 0, 255], [0, 255, 0]],
        name: "x".into(),
    };

    // A host that has the callback receives it.
    CHANNEL_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
    let full = sink(core::mem::size_of::<FtSink>());
    let st = unsafe { write_image(&full, &img, "x", FtOutcomeKind::NewDocument, "") };
    assert_eq!(st, FtStatus::Ok);
    assert_eq!(
        CHANNEL_CALLS.load(core::sync::atomic::Ordering::Relaxed),
        2,
        "a host that has set_channel should have been given both colours"
    );

    // A host that does not still gets the image, just without the colours.
    CHANNEL_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
    let old = sink(core::mem::offset_of!(FtSink, set_channel));
    let st = unsafe { write_image(&old, &img, "x", FtOutcomeKind::NewDocument, "") };
    assert_eq!(st, FtStatus::Ok, "an older host lost the whole image");
    assert_eq!(
        CHANNEL_CALLS.load(core::sync::atomic::Ordering::Relaxed),
        0,
        "set_channel was called on a host whose table does not contain it"
    );
}

/// There is still a floor. A table too small to receive a result at all is not
/// an old host, it is not a host.
#[test]
fn a_sink_below_the_core_is_refused_with_a_reason() {
    let s = sink(FtSink::CORE - 1);
    crate::last_error::set("");
    let st = unsafe { write_outcome(&s, Outcome::Nothing) };
    assert_eq!(st, FtStatus::BadArgument);
    let msg = unsafe { crate::last_error::get().as_str().unwrap_or("") }.to_string();
    assert!(
        msg.contains("too small"),
        "a refusal must say why, not just fail: {msg:?}"
    );
}

/// Every refusal in the run path has to leave something behind, or the host can
/// only report the status — which is what "failed (BadArgument)" was.
#[test]
fn no_refusal_in_the_run_path_is_silent() {
    let s = sink(FtSink::CORE - 1);
    crate::last_error::set("");
    unsafe { write_outcome(&s, Outcome::Nothing) };
    assert!(
        !unsafe { crate::last_error::get().as_str().unwrap_or("") }.is_empty(),
        "write_outcome refused without saying why"
    );
}

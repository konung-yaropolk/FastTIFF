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

/// A chart as `begin_plot` received it: title, x label, y label, x start,
/// x step, and the raw `FtSelectionKind`.
type BegunPlot = (String, String, String, f64, f64, u32);

/// A curve as `push_series` received it: label, values, and the raw colour.
type PushedSeries = (String, Vec<f32>, u32);

/// What the plot callbacks were handed, so a marshalled chart can be read back
/// rather than merely counted.
static PLOT: std::sync::Mutex<Option<BegunPlot>> = std::sync::Mutex::new(None);
static SERIES: std::sync::Mutex<Vec<PushedSeries>> = std::sync::Mutex::new(Vec::new());
static LAST_KIND: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);

/// Held for the length of any test that reads those three back.
///
/// The recording callbacks are statics because a C function pointer has no
/// closure to carry state in, so tests that use them share one set of globals
/// and `cargo test` runs them on several threads at once. Without this, two
/// plot tests interleave and each reads the other's chart — a flake that
/// looks like a marshalling bug.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take it without caring whether an earlier failure poisoned it: a poisoned
/// lock here would replace a real assertion failure with a confusing one.
fn alone() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

unsafe extern "C" fn begin_plot(
    _c: *mut core::ffi::c_void,
    title: FtStr,
    x_label: FtStr,
    y_label: FtStr,
    x_start: f64,
    x_step: f64,
    wants: FtSelectionKind,
) -> FtStatus {
    *PLOT.lock().unwrap() = Some((
        title.as_str().unwrap_or("").into(),
        x_label.as_str().unwrap_or("").into(),
        y_label.as_str().unwrap_or("").into(),
        x_start,
        x_step,
        wants.0,
    ));
    SERIES.lock().unwrap().clear();
    FtStatus::Ok
}

unsafe extern "C" fn push_series(
    _c: *mut core::ffi::c_void,
    label: FtStr,
    values: *const f32,
    len: u64,
    color: u32,
) -> FtStatus {
    let v = if len == 0 {
        Vec::new()
    } else {
        core::slice::from_raw_parts(values, len as usize).to_vec()
    };
    SERIES
        .lock()
        .unwrap()
        .push((label.as_str().unwrap_or("").into(), v, color));
    FtStatus::Ok
}

unsafe extern "C" fn recording_outcome(
    _c: *mut core::ffi::c_void,
    k: FtOutcomeKind,
    _t: FtStr,
) -> FtStatus {
    LAST_KIND.store(k.0, core::sync::atomic::Ordering::Relaxed);
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
        begin_plot,
        push_series,
    }
}

/// The same, with a `set_outcome` that remembers what it was told.
fn recording_sink(size: usize) -> FtSink {
    FtSink {
        set_outcome: recording_outcome,
        ..sink(size)
    }
}

/// Counts what an importer's progress callback received.
static PROGRESS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

unsafe extern "C" fn host_progress(_c: *mut core::ffi::c_void, _f: f32) -> u32 {
    PROGRESS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    1
}

unsafe extern "C" fn host_log(_c: *mut core::ffi::c_void, _m: FtStr) {}

unsafe extern "C" fn host_image_info(_c: *mut core::ffi::c_void, _o: *mut FtImageInfo) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn host_view(_c: *mut core::ffi::c_void, _o: *mut FtViewParams) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn host_channel(
    _c: *mut core::ffi::c_void,
    _i: u64,
    _o: *mut FtChannelView,
) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn host_str(_c: *mut core::ffi::c_void) -> FtStr {
    FtStr::EMPTY
}
unsafe extern "C" fn host_read_f32(
    _c: *mut core::ffi::c_void,
    _a: u64,
    _b: u64,
    _d: u64,
    _o: *mut f32,
    _n: u64,
) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn host_read_u16(
    _c: *mut core::ffi::c_void,
    _a: u64,
    _b: u64,
    _d: u64,
    _o: *mut u16,
    _n: u64,
) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn host_count(_c: *mut core::ffi::c_void) -> u64 {
    0
}
unsafe extern "C" fn host_roi(_c: *mut core::ffi::c_void, _i: u64, _o: *mut FtRoi) -> FtStatus {
    FtStatus::OutOfRange
}

/// A host table declaring `size` bytes, whatever this build's `FtHost` is.
fn host_table(size: usize) -> FtHost {
    FtHost {
        struct_size: size as u32,
        _pad: 0,
        ctx: core::ptr::null_mut(),
        image_info: host_image_info,
        view_params: host_view,
        channel_view: host_channel,
        stack_name: host_str,
        stack_path: host_str,
        read_plane_f32: host_read_f32,
        read_plane_u16: host_read_u16,
        progress: host_progress,
        log: host_log,
        stack_info: no_stack_info_fixture,
        stack_string: no_stack_string_fixture,
        selection_count: host_count,
        selection_roi: host_roi,
    }
}

unsafe extern "C" fn no_stack_info_fixture(
    _c: *mut core::ffi::c_void,
    _o: *mut FtStackInfo,
) -> FtStatus {
    FtStatus::Unsupported
}
unsafe extern "C" fn no_stack_string_fixture(
    _c: *mut core::ffi::c_void,
    _w: u32,
    _i: u64,
) -> FtStr {
    FtStr::EMPTY
}

/// The smallest importer that reports progress and produces a result.
#[derive(Default)]
struct Tiny;

impl crate::api::Importer for Tiny {
    fn info(&self) -> crate::api::PluginInfo {
        crate::api::PluginInfo::new("dev.test.tiny", "Tiny")
    }

    fn file_types(&self) -> Vec<crate::api::FileType> {
        vec![crate::api::FileType::new("Tiny", &["tiny"])]
    }

    fn import(
        &mut self,
        _request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<crate::api::ImportResult, crate::api::PluginError> {
        host.progress(0.5);
        host.log("reading");
        Ok(crate::api::ImportResult {
            image: crate::api::ImageResult {
                width: 1,
                height: 1,
                channels: 1,
                slices: 1,
                frames: 1,
                pixel_type: PixelType::U8,
                planes: vec![PlaneData::U8(vec![7])],
                channel_colors: Vec::new(),
                metadata: None,
                name: "tiny".into(),
            },
            info: None,
        })
    }
}

/// A chart with every field filled in, so nothing that fails to cross can hide
/// behind a value that happens to be the default.
fn a_plot() -> crate::api::Plot {
    crate::api::Plot::new("Mean over time")
        .labels("Time (s)", "Mean value")
        .scale(2.5, 0.25)
        .wants(crate::api::SelectionKind::Regions)
        .push(crate::api::Series::new("Region 1", vec![1.0, 2.0, f32::NAN]).color([10, 20, 30]))
        .push(crate::api::Series::new("Region 2", vec![4.0, 5.0, 6.0]))
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
        metadata: None,
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

// ------------------------------------------------------- registrar skew

/// Counts what a newer host would receive, so "skipped" can be told from
/// "called and ignored" — the same distinction the sink tests above draw.
static EXPORTER_CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static EXPORTER_ID: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
// The two registrar tests reset and inspect the counters above. Cargo runs
// tests concurrently, so make that shared fixture exclusive rather than
// letting one test's legitimate callback look like another's ABI violation.
static EXPORTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

unsafe extern "C" fn add_plugin_stub(
    _c: *mut core::ffi::c_void,
    _d: *const FtPluginDesc,
) -> FtStatus {
    FtStatus::Ok
}
unsafe extern "C" fn add_importer_stub(
    _c: *mut core::ffi::c_void,
    _d: *const FtImporterDesc,
) -> FtStatus {
    FtStatus::Ok
}
unsafe extern "C" fn add_exporter_counting(
    _c: *mut core::ffi::c_void,
    d: *const FtExporterDesc,
) -> FtStatus {
    EXPORTER_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    if let Some(id) = (*d).id.as_str() {
        *EXPORTER_ID.lock().unwrap() = id.to_string();
    }
    FtStatus::Ok
}

/// A registrar declaring `size` bytes, whatever this build's really is.
fn registrar(size: usize) -> FtRegistrar {
    FtRegistrar {
        struct_size: size as u32,
        host_abi_minor: 0,
        ctx: core::ptr::null_mut(),
        plugin_abi_minor: 0,
        _pad: 0,
        add_plugin: add_plugin_stub,
        add_importer: add_importer_stub,
        add_exporter: add_exporter_counting,
    }
}

#[derive(Default)]
struct Csv;

impl Exporter for Csv {
    fn info(&self) -> crate::api::PluginInfo {
        crate::api::PluginInfo::new("dev.test.csv", "CSV")
    }
    fn file_types(&self) -> Vec<crate::api::FileType> {
        vec![crate::api::FileType::new("CSV", &["csv"])]
    }
    fn export(
        &mut self,
        _r: &ExportRequest,
        _h: &mut dyn crate::api::HostContext,
    ) -> Result<(), crate::api::PluginError> {
        Ok(())
    }
}

/// The direction the whole `registrar_of` dance exists for: a host from before
/// exporters existed must still load a library that carries one.
#[test]
fn a_host_without_add_exporter_installs_everything_else() {
    let _exclusive = EXPORTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let older = FtRegistrar::CORE;
    assert!(
        older < core::mem::size_of::<FtRegistrar>(),
        "the fixture must be smaller than today's registrar but still a valid one"
    );
    let mut host = registrar(older);
    EXPORTER_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);

    // SAFETY: every callback in the fixture is real; only the declared size is
    // smaller than this build's.
    let mut copy = unsafe { registrar_of(&mut host) }.expect("an older host must be accepted");
    assert_eq!(
        register_exporter::<Csv>(&mut copy),
        FtStatus::Ok,
        "an exporter with nowhere to go must not fail the whole library"
    );
    assert_eq!(
        EXPORTER_CALLS.load(core::sync::atomic::Ordering::Relaxed),
        0,
        "add_exporter was called on a host whose table does not contain it"
    );
    // And the host still learns what this plugin knows about, which is the one
    // thing written back through the pointer rather than into the copy.
    assert_eq!(host.plugin_abi_minor, crate::abi::ABI_MINOR);
}

/// The other direction: a host that does have the field is given the exporter,
/// descriptor and all.
#[test]
fn a_host_with_add_exporter_receives_it() {
    let _exclusive = EXPORTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut host = registrar(core::mem::size_of::<FtRegistrar>());
    EXPORTER_CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
    EXPORTER_ID.lock().unwrap().clear();

    // SAFETY: as above; this fixture declares its true size.
    let mut copy = unsafe { registrar_of(&mut host) }.expect("a current host must be accepted");
    assert_eq!(register_exporter::<Csv>(&mut copy), FtStatus::Ok);
    assert_eq!(
        EXPORTER_CALLS.load(core::sync::atomic::Ordering::Relaxed),
        1
    );
    assert_eq!(*EXPORTER_ID.lock().unwrap(), "dev.test.csv");
}

/// There is still a floor, as there is for the sink.
#[test]
fn a_registrar_below_the_core_is_refused_with_a_reason() {
    let mut host = registrar(FtRegistrar::CORE - 1);
    crate::last_error::set("");
    // SAFETY: the fixture is a real registrar; only its declared size lies.
    // `expect_err` would need the registrar to be `Debug`, and a table of raw
    // function pointers has nothing worth printing.
    let Err(st) = (unsafe { registrar_of(&mut host) }) else {
        panic!("a table this small is not a host");
    };
    assert_eq!(st, FtStatus::BadArgument);
    let msg = unsafe { crate::last_error::get().as_str().unwrap_or("") }.to_string();
    assert!(
        msg.contains("too small"),
        "a refusal must say why, not just fail: {msg:?}"
    );
}

// --------------------------------------------------------------------- plots

/// Every field of a chart reaches the host, including the ones a default would
/// hide: the x scale, the tool it asks for, and a series with no colour.
#[test]
fn a_plot_crosses_field_for_field() {
    let _alone = alone();
    let s = recording_sink(core::mem::size_of::<FtSink>());
    // SAFETY: every callback in the fixture is real, and the declared size is
    // this build's own.
    let st = unsafe { write_outcome(&s, Outcome::Plot(Box::new(a_plot()))) };
    assert_eq!(st, FtStatus::Ok, "{:?}", unsafe {
        crate::last_error::get().as_str()
    });

    let got = PLOT
        .lock()
        .unwrap()
        .clone()
        .expect("begin_plot was not called");
    assert_eq!(got.0, "Mean over time");
    assert_eq!(got.1, "Time (s)");
    assert_eq!(got.2, "Mean value");
    // The x axis is the chart's whole calibration; a dropped one silently
    // relabels every point on it.
    assert_eq!(got.3, 2.5);
    assert_eq!(got.4, 0.25);
    assert_eq!(
        got.5,
        FtSelectionKind::Regions.0,
        "the tool request did not cross, so the host would never call again"
    );

    let series = SERIES.lock().unwrap().clone();
    assert_eq!(series.len(), 2);
    assert_eq!(series[0].0, "Region 1");
    assert_eq!(series[0].1[..2], [1.0, 2.0]);
    // A gap is a value, and has to stay one.
    assert!(series[0].1[2].is_nan(), "the gap did not cross");
    assert_eq!(series[0].2, 0x000A_141E, "the colour did not cross");
    assert_eq!(series[1].0, "Region 2");
    assert_eq!(series[1].1, vec![4.0, 5.0, 6.0]);
    // No colour is not black. Black is a colour a plugin may choose.
    assert_eq!(series[1].2, FT_COLOR_NONE);

    assert_eq!(
        LAST_KIND.load(core::sync::atomic::Ordering::Relaxed),
        FtOutcomeKind::Plot.0,
        "the chart was pushed but never finished"
    );
}

/// A series with no points still crosses, and its pointer is never read.
#[test]
fn an_empty_series_crosses_without_its_pointer_being_read() {
    let _alone = alone();
    let s = sink(core::mem::size_of::<FtSink>());
    let plot =
        crate::api::Plot::new("Nothing").push(crate::api::Series::new("Empty region", Vec::new()));
    // SAFETY: as above.
    let st = unsafe { write_outcome(&s, Outcome::Plot(Box::new(plot))) };
    assert_eq!(st, FtStatus::Ok);
    let series = SERIES.lock().unwrap().clone();
    assert_eq!(series.len(), 1);
    assert!(series[0].1.is_empty());
}

/// A host from before plots says so, rather than reporting a success that
/// nothing came of.
///
/// The alternative — pushing into a table that stops short of `begin_plot` —
/// is a call through whatever happens to sit past the end of the host's
/// allocation.
#[test]
fn a_host_too_old_for_plots_refuses_with_a_reason() {
    let _alone = alone();
    let older = core::mem::offset_of!(FtSink, begin_plot);
    assert!(
        older >= FtSink::CORE && older < core::mem::size_of::<FtSink>(),
        "the fixture must be a real minor-1 sink"
    );
    let s = sink(older);
    crate::last_error::set("");
    // SAFETY: the callbacks are real; only the declared size is older.
    let st = unsafe { write_outcome(&s, Outcome::Plot(Box::new(a_plot()))) };
    assert_eq!(st, FtStatus::Unsupported);
    let msg = unsafe { crate::last_error::get().as_str().unwrap_or("").to_string() };
    assert!(msg.contains("too old"), "unhelpful: {msg}");
    assert!(
        msg.contains("1.2"),
        "the message must name the version: {msg}"
    );
}

/// And that same older host still gets everything it can take.
#[test]
fn a_host_too_old_for_plots_still_gets_its_other_results() {
    let _alone = alone();
    let s = recording_sink(core::mem::offset_of!(FtSink, begin_plot));
    // SAFETY: as above.
    let st = unsafe { write_outcome(&s, Outcome::Message("still fine".into())) };
    assert_eq!(st, FtStatus::Ok);
    assert_eq!(
        LAST_KIND.load(core::sync::atomic::Ordering::Relaxed),
        FtOutcomeKind::Message.0
    );
}

// ------------------------------------------------------ holding a short sink

/// A heap block of exactly `size` bytes, aligned as a `T`.
///
/// The point of the fixture is that the allocation *ends* where the host says
/// its table ends, so a copy that took `size_of::<T>()` bytes would run off a
/// real allocation rather than into slack. That rules out a `Vec<u8>`, which
/// has alignment 1 — every one of these is read through `declared_size`, which
/// loads a `u32` — and it rules out over-allocating to get the alignment,
/// which would put the slack back.
struct Exactly<T> {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    _t: core::marker::PhantomData<T>,
}

impl<T: Copy> Exactly<T> {
    /// The first `size` bytes of `value`, in an allocation that size.
    fn holding(value: &T, size: usize) -> Self {
        assert!(size <= core::mem::size_of::<T>());
        let layout =
            std::alloc::Layout::from_size_align(size, core::mem::align_of::<T>()).expect("layout");
        // SAFETY: a non-zero size, and the alignment comes from `T` itself.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null(), "the fixture could not be allocated");
        // SAFETY: `value` is a whole `T` and `size` is at most its size, so
        // this copies a prefix of it into a block that large.
        unsafe {
            core::ptr::copy_nonoverlapping((value as *const T).cast::<u8>(), ptr, size);
        }
        Exactly {
            ptr,
            layout,
            _t: core::marker::PhantomData,
        }
    }

    fn as_ptr(&self) -> *const T {
        self.ptr.cast()
    }
}

impl<T> Drop for Exactly<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer and layout are the ones `holding` allocated with.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// A sink the host allocated to its own, smaller layout is copied, not
/// referenced.
///
/// `&*sink` over a shorter allocation is undefined behaviour the moment the
/// reference exists — before any field is read, and so before any `covers`
/// check could rescue it. The fixture is allocated to exactly the declared
/// size, so a copy that took `size_of::<FtSink>()` bytes would run off the end
/// of it. Without a sanitiser the over-read is silent, so this case earns its
/// keep under `cargo miri test`; what it checks unconditionally is the
/// behaviour that over-read was in aid of.
#[test]
fn a_shorter_sink_is_copied_within_what_the_host_allocated() {
    let _alone = alone();
    let declared = core::mem::offset_of!(FtSink, begin_plot);
    let short = Exactly::holding(&sink(declared), declared);

    // SAFETY: the block holds a well-formed prefix of an `FtSink` whose own
    // `struct_size` says how much of one — which is the whole contract.
    let copied = unsafe { sink_of(short.as_ptr()) }.expect("a minor-1 sink");

    // The host's declared size survives, so every `ft_covers!` on the copy
    // still asks about the host rather than about this plugin.
    assert_eq!(copied.struct_size as usize, declared);
    // SAFETY: `copied` is a fully initialised local.
    assert!(!unsafe { crate::abi::ft_covers!(&copied as *const FtSink, FtSink, begin_plot) });
    // SAFETY: as above.
    assert!(unsafe { crate::abi::ft_covers!(&copied as *const FtSink, FtSink, set_channel) });

    // And it is usable: stubbing the tail did not disturb the real callbacks.
    crate::last_error::set("");
    // SAFETY: `copied` is a valid sink for the duration of the call.
    assert_eq!(
        unsafe { write_outcome(&copied, Outcome::Nothing) },
        FtStatus::Ok
    );
}

/// A sink that cannot even carry the core is refused rather than copied.
#[test]
fn a_sink_smaller_than_the_core_is_refused() {
    let small = sink(FtSink::CORE - 1);
    // SAFETY: the pointer is to a real, full-size allocation; only the
    // declared size is too small, which is the case under test.
    let r = unsafe { sink_of(&small as *const FtSink) };
    assert_eq!(r.err(), Some(FtStatus::BadArgument));

    // SAFETY: `sink_of` checks for null before reading anything.
    let r = unsafe { sink_of(core::ptr::null()) };
    assert_eq!(r.err(), Some(FtStatus::BadArgument));
}

// ------------------------------------------------------- holding a short host

/// The host's table gets the same treatment as its sink.
///
/// It is the table that grew this time, so an already-released host allocates
/// two pointers fewer than this plugin's `FtHost` has.
#[test]
fn a_shorter_host_is_copied_within_what_the_host_allocated() {
    let declared = core::mem::offset_of!(FtHost, selection_count);
    assert!(
        declared >= FtHost::CORE && declared < core::mem::size_of::<FtHost>(),
        "the fixture must be a real minor-1 host"
    );
    let short = Exactly::holding(&host_table(declared), declared);

    // SAFETY: the block holds a well-formed prefix of an `FtHost`, sized by
    // its own `struct_size`.
    let copied = unsafe { crate::host::host_of(short.as_ptr()) }.expect("a minor-1 host");

    assert_eq!(copied.struct_size as usize, declared);
    // SAFETY: `copied` is a fully initialised local.
    assert!(!unsafe { crate::abi::ft_covers!(&copied as *const FtHost, FtHost, selection_count) });
    // SAFETY: as above. The real callbacks below `CORE` came across intact.
    assert_eq!(unsafe { (copied.progress)(copied.ctx, 0.5) }, 1);
}

/// And an importer runs against one, which is the path that read it whole.
///
/// `import_shim` used to copy the host's table by value with no size check at
/// all — a full `size_of::<FtHost>()` load out of whatever the host allocated.
/// That was byte-exact while the table had not grown, and a read past the end
/// of every shipped host's table the moment it did.
#[test]
fn an_importer_runs_against_a_host_from_an_older_minor() {
    let _alone = alone();
    let declared = core::mem::offset_of!(FtHost, selection_count);
    let short = Exactly::holding(&host_table(declared), declared);
    let s = recording_sink(core::mem::size_of::<FtSink>());

    PROGRESS.store(0, core::sync::atomic::Ordering::Relaxed);
    // SAFETY: a well-formed minor-1 host table and this build's own sink.
    let st = unsafe {
        import_shim::<Tiny>(
            FtStr::from_str("x.tiny"),
            core::ptr::null(),
            0,
            short.as_ptr(),
            &s as *const FtSink,
        )
    };
    assert_eq!(st, FtStatus::Ok, "{:?}", unsafe {
        crate::last_error::get().as_str()
    });
    // The callbacks it did have were reached, so the copy carried them.
    assert_eq!(PROGRESS.load(core::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(
        LAST_KIND.load(core::sync::atomic::Ordering::Relaxed),
        FtOutcomeKind::NewDocument.0
    );
}

/// A host too small to carry even the core leaves an importer without progress
/// or logging, rather than calling through pointers read past the end of it.
#[test]
fn an_importer_survives_a_host_that_is_not_one() {
    let _alone = alone();
    let s = recording_sink(core::mem::size_of::<FtSink>());
    let tiny = host_table(FtHost::CORE - 1);

    PROGRESS.store(0, core::sync::atomic::Ordering::Relaxed);
    // SAFETY: the pointer is to a real, full-size allocation; only the
    // declared size is too small, which is the case under test.
    let st = unsafe {
        import_shim::<Tiny>(
            FtStr::from_str("x.tiny"),
            core::ptr::null(),
            0,
            &tiny as *const FtHost,
            &s as *const FtSink,
        )
    };
    // The import still happens — an importer needs no host to read a file.
    assert_eq!(st, FtStatus::Ok);
    assert_eq!(
        PROGRESS.load(core::sync::atomic::Ordering::Relaxed),
        0,
        "progress must go nowhere rather than through a pointer that was never there"
    );
}

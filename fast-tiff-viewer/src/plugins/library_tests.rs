//! Plugins that lie.
//!
//! `plugin_library.rs` proves the boundary is correct for a plugin that
//! behaves. This proves it is *safe* for one that does not — which is the case
//! that actually matters, because a plugin is third-party code the host has no
//! control over, and "it crashed the microscope software" is not an acceptable
//! outcome for any of it.
//!
//! Every fixture here is a deliberately hostile `ft_plugin_v1_query`, fed to
//! the real [`register_from`] with the real sinks and host callbacks. The
//! assertions are always the same two things: **the host does not crash**, and
//! **it says what was wrong**. Nothing here checks a pixel.
//!
//! The list is drawn from what a C boundary can actually be handed: a struct
//! from a different ABI version, a null where a pointer was promised, bytes
//! that are not UTF-8, a length that cannot be real, more data than was
//! declared, less data than was declared, and a panic.

use super::*;
use fasttiff_plugin_api::{ImageInfo, Plane, StackInfo, ViewParams, VolumeMode, VolumeView};

// ------------------------------------------------------------ a host to run in

/// The smallest thing that satisfies `HostContext`: one 2x2 plane of known
/// values, so a plugin reading pixels gets something and the host's range
/// checks have something to reject.
struct FakeHost {
    view: ViewParams,
    info: StackInfo,
    /// Set when the host is asked to stop, so a test can check the flag
    /// crossed rather than trusting that it did.
    cancel: bool,
    logged: Vec<String>,
}

impl Default for FakeHost {
    fn default() -> Self {
        FakeHost {
            view: ViewParams {
                frame_index: 0,
                volume_view: false,
                channels: Vec::new(),
                luts: Vec::new(),
                volume: VolumeView {
                    mode: VolumeMode::Mip,
                    density: 1.0,
                    iso: 0.5,
                    eye: [0.0; 3],
                    forward: [0.0, 0.0, 1.0],
                    up: [0.0, 1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                },
            },
            info: StackInfo {
                name: "fake".into(),
                ..Default::default()
            },
            cancel: false,
            logged: Vec::new(),
        }
    }
}

impl HostContext for FakeHost {
    fn image(&self) -> ImageInfo {
        ImageInfo {
            width: 2,
            height: 2,
            channels: 1,
            slices: 1,
            frames: 1,
            samples_per_pixel: 1,
            pixel_type: PixelType::F32,
        }
    }
    fn view(&self) -> &ViewParams {
        &self.view
    }
    fn stack_info(&self) -> &StackInfo {
        &self.info
    }
    fn read_plane_u16(&mut self, p: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        if p != Plane::new(0, 0, 0) {
            return Err(PluginError::OutOfRange(format!("{p:?}")));
        }
        *out = vec![1, 2, 3, 4];
        Ok(())
    }
    fn read_plane_f32(&mut self, p: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        if p != Plane::new(0, 0, 0) {
            return Err(PluginError::OutOfRange(format!("{p:?}")));
        }
        *out = vec![1.0, 2.0, 3.0, 4.0];
        Ok(())
    }
    fn progress(&mut self, _f: f32) -> bool {
        !self.cancel
    }
    fn log(&mut self, m: &str) {
        self.logged.push(m.to_string());
    }
}

// ------------------------------------------------------------------- fixtures

fn src() -> &'static Path {
    Path::new("hostile.dll")
}

/// Run a hostile query function through the real loader.
fn load(query: abi::FtQueryFn) -> Result<Loaded, String> {
    // SAFETY: each fixture below has the signature `FtQueryFn` names, and is
    // linked into this test binary for its whole life.
    unsafe { register_from(query, src()) }
}

const NO_DECLS: unsafe extern "C" fn(*const abi::FtHost, *const abi::FtParamSink) -> abi::FtStatus = {
    unsafe extern "C" fn f(_h: *const abi::FtHost, _s: *const abi::FtParamSink) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    f
};

/// The simplest valid control, for fixtures that only care how many arrive.
fn label_decl() -> abi::FtParamDecl {
    abi::FtParamDecl {
        struct_size: std::mem::size_of::<abi::FtParamDecl>() as u32,
        kind: abi::FtParamKind::Label,
        key: abi::FtStr::from_str("k"),
        label: abi::FtStr::from_str("l"),
        help: abi::FtStr::EMPTY,
        i_default: 0,
        i_min: 0,
        i_max: 0,
        f_default: 0.0,
        f_min: 0.0,
        f_max: 0.0,
        b_default: 0,
        save: 0,
        s_default: abi::FtStr::EMPTY,
        options: std::ptr::null(),
        option_count: 0,
    }
}

unsafe extern "C" fn no_error() -> abi::FtStr {
    abi::FtStr::EMPTY
}

/// A `last_error` that hands back bytes that are not UTF-8.
unsafe extern "C" fn garbage_error() -> abi::FtStr {
    static BAD: [u8; 3] = [0xFF, 0xFE, 0xFD];
    abi::FtStr {
        ptr: BAD.as_ptr(),
        len: 3,
    }
}

/// Build a descriptor around a `run` function, with everything else valid.
fn desc_for(
    vtable: &'static abi::FtPluginVtable,
    id: &'static str,
) -> &'static mut abi::FtPluginDesc {
    Box::leak(Box::new(abi::FtPluginDesc {
        struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32,
        _pad: 0,
        id: abi::FtStr::from_str(id),
        name: abi::FtStr::from_str("Hostile"),
        menu_path: abi::FtStr::EMPTY,
        version: abi::FtStr::from_str("1"),
        author: abi::FtStr::EMPTY,
        description: abi::FtStr::EMPTY,
        vtable,
    }))
}

/// Register one plugin whose `run` is `$run`, then load it.
macro_rules! plugin_with_run {
    ($run:expr) => {{
        static VT: std::sync::OnceLock<abi::FtPluginVtable> = std::sync::OnceLock::new();
        let vt = VT.get_or_init(|| abi::FtPluginVtable {
            struct_size: std::mem::size_of::<abi::FtPluginVtable>() as u32,
            _pad: 0,
            params: NO_DECLS,
            run: $run,
            last_error: no_error,
        });
        let desc = desc_for(vt, "dev.test.hostile");
        let mut c = Collector {
            source: src().to_path_buf(),
            ..Default::default()
        };
        // SAFETY: `desc` is a valid, fully-initialised descriptor that outlives
        // the call, which is all `add_plugin_cb` requires.
        let st = unsafe {
            add_plugin_cb(
                &mut c as *mut Collector as *mut std::ffi::c_void,
                desc as *const abi::FtPluginDesc,
            )
        };
        assert_eq!(st, abi::FtStatus::Ok, "{:?}", c.problems);
        c.plugins.remove(0)
    }};
}

// ----------------------------------------------------------------- the tests

/// The most basic lie: claiming to be from a version of the ABI whose structs
/// are larger than the ones actually provided. Reading the fields past the end
/// would be reading whatever happened to follow in the plugin's memory.
#[test]
fn a_descriptor_shorter_than_this_abi_is_refused() {
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        static VT: abi::FtPluginVtable = abi::FtPluginVtable {
            // Correct size, so the descriptor is the only thing wrong.
            struct_size: 0,
            _pad: 0,
            params: NO_DECLS,
            run: ok_run,
            last_error: no_error,
        };
        let mut vt = VT;
        vt.struct_size = std::mem::size_of::<abi::FtPluginVtable>() as u32;
        let desc = abi::FtPluginDesc {
            // One byte short of this ABI's descriptor.
            struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32 - 1,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.short"),
            name: abi::FtStr::from_str("Short"),
            menu_path: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_plugin)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("a short descriptor must be refused");
    assert!(err.contains("older than this ABI"), "{err}");
}

#[test]
fn a_vtable_shorter_than_this_abi_is_refused() {
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        let vt = abi::FtPluginVtable {
            struct_size: 8, // just the header
            _pad: 0,
            params: NO_DECLS,
            run: ok_run,
            last_error: no_error,
        };
        let desc = abi::FtPluginDesc {
            struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.shortvt"),
            name: abi::FtStr::EMPTY,
            menu_path: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_plugin)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("a short vtable must be refused");
    assert!(err.contains("vtable is older"), "{err}");
}

#[test]
fn a_null_vtable_is_refused_rather_than_called() {
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        let desc = abi::FtPluginDesc {
            struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.null"),
            name: abi::FtStr::EMPTY,
            menu_path: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            vtable: std::ptr::null(),
        };
        let r = &mut *reg;
        (r.add_plugin)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("a null vtable must be refused");
    assert!(err.contains("null vtable"), "{err}");
}

#[test]
fn a_name_that_is_not_utf8_is_reported_rather_than_shown() {
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        static BAD: [u8; 4] = [b'a', 0xC3, 0x28, b'z'];
        static VT: abi::FtPluginVtable = abi::FtPluginVtable {
            struct_size: 0,
            _pad: 0,
            params: NO_DECLS,
            run: ok_run,
            last_error: no_error,
        };
        let mut vt = VT;
        vt.struct_size = std::mem::size_of::<abi::FtPluginVtable>() as u32;
        let desc = abi::FtPluginDesc {
            struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.badname"),
            name: abi::FtStr {
                ptr: BAD.as_ptr(),
                len: 4,
            },
            menu_path: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_plugin)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("invalid UTF-8 must be refused");
    assert!(err.contains("not valid UTF-8"), "{err}");
}

/// A length no string can have, pointing at a valid byte. Constructing the
/// slice would read gigabytes of whatever follows.
#[test]
fn an_absurd_string_length_is_refused_rather_than_read() {
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        static ONE: [u8; 1] = [b'x'];
        static VT: abi::FtPluginVtable = abi::FtPluginVtable {
            struct_size: 0,
            _pad: 0,
            params: NO_DECLS,
            run: ok_run,
            last_error: no_error,
        };
        let mut vt = VT;
        vt.struct_size = std::mem::size_of::<abi::FtPluginVtable>() as u32;
        let desc = abi::FtPluginDesc {
            struct_size: std::mem::size_of::<abi::FtPluginDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.longstr"),
            name: abi::FtStr {
                ptr: ONE.as_ptr(),
                len: u64::MAX,
            },
            menu_path: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_plugin)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("an absurd length must be refused");
    assert!(err.contains("not valid UTF-8"), "{err}");
}

#[test]
fn a_library_that_registers_nothing_says_so() {
    unsafe extern "C" fn query(_reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("registering nothing is an error");
    assert!(err.contains("registered nothing"), "{err}");
}

#[test]
fn a_query_that_fails_is_reported_with_its_status() {
    unsafe extern "C" fn query(_reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        abi::FtStatus::Unsupported
    }
    let err = load(query).expect_err("a failed query must not load");
    assert!(err.contains("Unsupported"), "{err}");
}

/// A plugin that panicked while registering reports it, and is not loaded.
///
/// The fixture *returns* `Panic` rather than panicking, because a genuine
/// unwind cannot reach the host at all: since Rust 1.81 every `extern "C"`
/// function carries an abort shim, so a panic escaping a plugin's entry point
/// terminates the process inside the **plugin**, before any host frame runs.
/// No `catch_unwind` on this side can change that, which is precisely why
/// `fasttiff_plugin::export_plugin!` wraps every entry point in its own guard
/// and converts the panic to this status. `plugin_library.rs` tests that guard
/// against the real example plugin; this tests what the host does with the
/// answer.
#[test]
fn a_query_that_reports_a_panic_is_not_loaded() {
    unsafe extern "C" fn query(_reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        abi::FtStatus::Panic
    }
    let err = load(query).expect_err("a panicking query must not load");
    assert!(err.contains("Panic"), "{err}");
}

// --- what a plugin can do once it is running -------------------------------

/// A `run` that behaves, for the fixtures that only care about registration.
unsafe extern "C" fn ok_run(
    _h: *const abi::FtHost,
    _v: *const abi::FtValue,
    _n: u64,
    sink: *const abi::FtSink,
) -> abi::FtStatus {
    let s = &*sink;
    (s.set_outcome)(s.ctx, abi::FtOutcomeKind::Nothing, abi::FtStr::EMPTY)
}

/// A panic during a run must name the plugin *and* the file it came from: with
/// several plugins installed, "something panicked" is not actionable.
#[test]
fn a_panic_inside_run_becomes_an_error_naming_the_plugin() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        // What a guarded plugin returns after catching its own panic; see
        // `a_query_that_reports_a_panic_is_not_loaded` for why it cannot
        // actually unwind to here.
        abi::FtStatus::Panic
    }
    let mut p = plugin_with_run!(run);
    let mut host = FakeHost::default();
    let err = p
        .run(&mut host, &Params::new())
        .expect_err("a panicking run must be an error");
    assert!(err.to_string().contains("panicked"), "{err}");
    assert!(err.to_string().contains("hostile.dll"), "{err}");
}

#[test]
fn a_run_that_pushes_more_planes_than_it_declared_is_stopped() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.begin_image)(
            s.ctx,
            2,
            2,
            1,
            1,
            1,
            abi::FtPixelType::F32,
            abi::FtStr::from_str("x"),
        );
        let plane = [0.0f32; 4];
        // One declared, three pushed.
        for _ in 0..3 {
            (s.push_plane)(s.ctx, plane.as_ptr() as *const std::ffi::c_void, 4);
        }
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an over-long push must be refused");
    assert!(err.to_string().contains("more than the 1 planes"), "{err}");
}

#[test]
fn a_plane_of_the_wrong_length_is_refused_before_it_is_copied() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.begin_image)(
            s.ctx,
            2,
            2,
            1,
            1,
            1,
            abi::FtPixelType::F32,
            abi::FtStr::from_str("x"),
        );
        let plane = [0.0f32; 4];
        // Four samples exist; a million are claimed.
        (s.push_plane)(s.ctx, plane.as_ptr() as *const std::ffi::c_void, 1_000_000);
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("a mis-sized plane must be refused");
    assert!(err.to_string().contains("1000000 samples where 4"), "{err}");
}

#[test]
fn a_result_shape_that_cannot_exist_is_refused_before_allocating() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        // 4 G x 4 G pixels, times a billion planes.
        (s.begin_image)(
            s.ctx,
            u32::MAX,
            u32::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            abi::FtPixelType::F32,
            abi::FtStr::EMPTY,
        );
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an impossible shape must be refused");
    assert!(err.to_string().contains("impossible"), "{err}");
}

#[test]
fn pushing_a_plane_before_declaring_the_result_is_refused() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        let plane = [0.0f32; 4];
        (s.push_plane)(s.ctx, plane.as_ptr() as *const std::ffi::c_void, 4);
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an undeclared push must be refused");
    assert!(err.to_string().contains("before declaring"), "{err}");
}

#[test]
fn declaring_more_planes_than_are_pushed_is_caught_by_validation() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.begin_image)(
            s.ctx,
            2,
            2,
            1,
            1,
            4,
            abi::FtPixelType::F32,
            abi::FtStr::EMPTY,
        );
        let plane = [0.0f32; 4];
        (s.push_plane)(s.ctx, plane.as_ptr() as *const std::ffi::c_void, 4);
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("a short result must be refused");
    assert!(err.to_string().contains("4 planes but carries 1"), "{err}");
}

#[test]
fn returning_ok_without_an_outcome_is_an_error_not_an_empty_document() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("no outcome is an error");
    assert!(
        err.to_string().contains("without saying what to do"),
        "{err}"
    );
}

/// A plugin that fails but whose `last_error` is unreadable must still produce
/// a message a user can act on.
#[test]
fn an_unreadable_last_error_falls_back_to_the_status() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        abi::FtStatus::Error
    }
    static VT: std::sync::OnceLock<abi::FtPluginVtable> = std::sync::OnceLock::new();
    let vt = VT.get_or_init(|| abi::FtPluginVtable {
        struct_size: std::mem::size_of::<abi::FtPluginVtable>() as u32,
        _pad: 0,
        params: NO_DECLS,
        run,
        last_error: garbage_error,
    });
    let desc = desc_for(vt, "dev.test.garbage");
    let mut c = Collector {
        source: src().to_path_buf(),
        ..Default::default()
    };
    // SAFETY: a valid descriptor that outlives the call.
    let st = unsafe {
        add_plugin_cb(
            &mut c as *mut Collector as *mut std::ffi::c_void,
            desc as *const abi::FtPluginDesc,
        )
    };
    assert_eq!(st, abi::FtStatus::Ok);
    let err = c.plugins[0]
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("Error must stay an error");
    assert!(err.to_string().contains("failed (Error)"), "{err}");
}

/// A plugin whose `params` fails must lose its dialog, not the application —
/// and must not be offered a half-built one either.
#[test]
fn a_failed_dialog_declaration_yields_no_controls() {
    unsafe extern "C" fn params(
        _h: *const abi::FtHost,
        sink: *const abi::FtParamSink,
    ) -> abi::FtStatus {
        // Push one control, then fail: a partial dialog is worse than none,
        // because the user would fill in values the plugin never declared.
        let s = &*sink;
        let decl = label_decl();
        (s.push)(s.ctx, &decl);
        abi::FtStatus::Panic
    }
    static VT: std::sync::OnceLock<abi::FtPluginVtable> = std::sync::OnceLock::new();
    let vt = VT.get_or_init(|| abi::FtPluginVtable {
        struct_size: std::mem::size_of::<abi::FtPluginVtable>() as u32,
        _pad: 0,
        params,
        run: ok_run,
        last_error: no_error,
    });
    let desc = desc_for(vt, "dev.test.paramspanic");
    let mut c = Collector {
        source: src().to_path_buf(),
        ..Default::default()
    };
    // SAFETY: a valid descriptor that outlives the call.
    unsafe {
        add_plugin_cb(
            &mut c as *mut Collector as *mut std::ffi::c_void,
            desc as *const abi::FtPluginDesc,
        )
    };
    let host = FakeHost::default();
    assert!(c.plugins[0].params(&host).is_empty());
}

/// A runaway `params` loop must be stopped by the host rather than filling
/// memory with controls nobody could use.
#[test]
fn a_dialog_with_no_end_is_capped() {
    unsafe extern "C" fn params(
        _h: *const abi::FtHost,
        sink: *const abi::FtParamSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        let decl = label_decl();
        // A plugin with a bug in its own loop. The host must stop taking them.
        for _ in 0..100_000 {
            if (s.push)(s.ctx, &decl) != abi::FtStatus::Ok {
                return abi::FtStatus::Ok;
            }
        }
        abi::FtStatus::Ok
    }
    static VT: std::sync::OnceLock<abi::FtPluginVtable> = std::sync::OnceLock::new();
    let vt = VT.get_or_init(|| abi::FtPluginVtable {
        struct_size: std::mem::size_of::<abi::FtPluginVtable>() as u32,
        _pad: 0,
        params,
        run: ok_run,
        last_error: no_error,
    });
    let desc = desc_for(vt, "dev.test.runaway");
    let mut c = Collector {
        source: src().to_path_buf(),
        ..Default::default()
    };
    // SAFETY: a valid descriptor that outlives the call.
    unsafe {
        add_plugin_cb(
            &mut c as *mut Collector as *mut std::ffi::c_void,
            desc as *const abi::FtPluginDesc,
        )
    };
    let host = FakeHost::default();
    assert_eq!(c.plugins[0].params(&host).len(), 256);
}

/// The host's own callbacks are called from plugin frames, so they must refuse
/// nonsense too — here, a buffer smaller than the plane the plugin asked for.
///
/// The capacity the plugin *declares* is the only thing the host can check
/// against; a plugin that declares 4 while owning 1 has a bug the host cannot
/// see. So the fixture declares the truth, and the host must refuse it rather
/// than fill a plane's worth of samples into it.
#[test]
fn the_host_refuses_to_write_past_a_buffer_the_plugin_undersized() {
    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let h = &*host;
        // The plane is 4 samples; this buffer holds 1, honestly declared.
        let mut buf = [0.0f32; 1];
        let st = (h.read_plane_f32)(h.ctx, 0, 0, 0, buf.as_mut_ptr(), 1);
        let s = &*sink;
        let text = if st == abi::FtStatus::BadArgument {
            "refused"
        } else {
            "wrote"
        };
        (s.set_outcome)(
            s.ctx,
            abi::FtOutcomeKind::Message,
            abi::FtStr::from_str(text),
        );
        abi::FtStatus::Ok
    }

    let mut p = plugin_with_run!(run);
    let out = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect("run");
    assert_eq!(
        out,
        Outcome::Message("refused".into()),
        "the host must not write into a buffer smaller than the plane"
    );
}

/// A plane the stack does not have must come back as `OutOfRange`, not as
/// zeros the plugin would silently process.
#[test]
fn a_plane_outside_the_stack_is_refused() {
    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let h = &*host;
        let mut buf = [0.0f32; 4];
        let st = (h.read_plane_f32)(h.ctx, 0, 0, 99, buf.as_mut_ptr(), 4);
        let s = &*sink;
        (s.set_outcome)(
            s.ctx,
            abi::FtOutcomeKind::Message,
            abi::FtStr::from_str(if st == abi::FtStatus::OutOfRange {
                "out of range"
            } else {
                "allowed"
            }),
        );
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let out = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect("run");
    assert_eq!(out, Outcome::Message("out of range".into()));
}

/// Cancellation has to reach the plugin, or a long run cannot be stopped.
#[test]
fn a_cancelled_host_reports_false_through_the_boundary() {
    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let h = &*host;
        let keep = (h.progress)(h.ctx, 0.5);
        let s = &*sink;
        if keep == 0 {
            (s.set_outcome)(s.ctx, abi::FtOutcomeKind::Nothing, abi::FtStr::EMPTY);
            return abi::FtStatus::Cancelled;
        }
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::Nothing, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);

    let mut host = FakeHost::default();
    assert_eq!(
        p.run(&mut host, &Params::new()).expect("run"),
        Outcome::Nothing
    );

    let mut host = FakeHost {
        cancel: true,
        ..Default::default()
    };
    assert_eq!(
        p.run(&mut host, &Params::new()).expect("run"),
        Outcome::Cancelled
    );
}

/// What a plugin logs must arrive as text, and rubbish must simply not arrive.
#[test]
fn logging_crosses_and_unreadable_log_lines_are_dropped() {
    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        static BAD: [u8; 2] = [0xFF, 0xFE];
        let h = &*host;
        (h.log)(h.ctx, abi::FtStr::from_str("hello"));
        (h.log)(
            h.ctx,
            abi::FtStr {
                ptr: BAD.as_ptr(),
                len: 2,
            },
        );
        let s = &*sink;
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::Nothing, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let mut host = FakeHost::default();
    p.run(&mut host, &Params::new()).expect("run");
    assert_eq!(host.logged, vec!["hello".to_string()]);
}

/// The host's callbacks run in plugin frames, so a panic in *host* code has the
/// same abort shim to worry about — and here `catch_unwind` genuinely does the
/// work, because the guard is inside the callback rather than outside it.
#[test]
fn a_panic_in_the_hosts_own_callback_is_contained() {
    /// A host whose `log` is broken. Contrived, but the class is not: every
    /// callback runs viewer code, and viewer code can have bugs.
    struct BrokenHost(FakeHost);
    impl HostContext for BrokenHost {
        fn image(&self) -> ImageInfo {
            self.0.image()
        }
        fn view(&self) -> &ViewParams {
            self.0.view()
        }
        fn stack_info(&self) -> &StackInfo {
            self.0.stack_info()
        }
        fn read_plane_u16(&mut self, p: Plane, o: &mut Vec<u16>) -> Result<(), PluginError> {
            self.0.read_plane_u16(p, o)
        }
        fn read_plane_f32(&mut self, p: Plane, o: &mut Vec<f32>) -> Result<(), PluginError> {
            self.0.read_plane_f32(p, o)
        }
        fn progress(&mut self, _f: f32) -> bool {
            panic!("the host's progress reporting is broken");
        }
        fn log(&mut self, _m: &str) {
            panic!("the host's logging is broken");
        }
    }

    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let h = &*host;
        // Both of these panic inside the host. The plugin must survive them
        // and be able to finish; a broken host callback is not a reason to
        // lose the work a plugin has already done.
        (h.log)(h.ctx, abi::FtStr::from_str("anything"));
        let keep = (h.progress)(h.ctx, 0.5);
        let s = &*sink;
        (s.set_outcome)(
            s.ctx,
            abi::FtOutcomeKind::Message,
            // A panicking `progress` reads as "stop", which is the safe way
            // round: it cannot make a run continue forever.
            abi::FtStr::from_str(if keep == 0 {
                "told to stop"
            } else {
                "continued"
            }),
        );
        abi::FtStatus::Ok
    }

    let mut p = plugin_with_run!(run);
    let mut host = BrokenHost(FakeHost::default());
    let out = p
        .run(&mut host, &Params::new())
        .expect("the run must survive");
    assert_eq!(out, Outcome::Message("told to stop".into()));
}

// --- values from a version this host has never heard of ---------------------

/// The reason every enumeration crossing this boundary is a `u32` newtype and
/// not a Rust `enum`.
///
/// A plugin built against a later minor version may hand back a value this host
/// does not know — that is the *documented* evolution path, not an abuse. With
/// a real `enum` in the signature, receiving it would be undefined behaviour
/// before a single line of checking code could run. With a newtype it is
/// ordinary data, and these tests are about what the host then does with it.
#[test]
fn an_unknown_status_is_data_rather_than_undefined_behaviour() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        // Not a status this ABI defines. Under `#[repr(u32)] enum` this line
        // alone would be UB on the host's side of the return.
        abi::FtStatus(4242)
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an unknown status cannot be a success");
    // Reported by number, so a user can quote it.
    assert!(err.to_string().contains("4242"), "{err}");
}

#[test]
fn an_unknown_outcome_kind_is_refused_rather_than_ignored() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind(99), abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an unknown outcome must not be silently dropped");
    assert!(
        err.to_string().contains("needs a newer FastTIFF"),
        "the message should say what to do about it: {err}"
    );
}

#[test]
fn an_unknown_sample_format_is_refused() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.begin_image)(
            s.ctx,
            2,
            2,
            1,
            1,
            1,
            abi::FtPixelType(77),
            abi::FtStr::EMPTY,
        );
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let err = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect_err("an unknown sample format must be refused");
    assert!(err.to_string().contains("sample format"), "{err}");
}

/// A control the host cannot draw becomes a *visible* label rather than a gap.
/// Dropping it silently would show a dialog quietly missing a setting the
/// plugin expects the user to make.
#[test]
fn an_unknown_control_kind_becomes_a_visible_label() {
    unsafe extern "C" fn params(
        _h: *const abi::FtHost,
        sink: *const abi::FtParamSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        let mut decl = label_decl();
        decl.kind = abi::FtParamKind(64);
        decl.label = abi::FtStr::from_str("Wavelet order");
        (s.push)(s.ctx, &decl);
        abi::FtStatus::Ok
    }
    static VT: std::sync::OnceLock<abi::FtPluginVtable> = std::sync::OnceLock::new();
    let vt = VT.get_or_init(|| abi::FtPluginVtable {
        struct_size: std::mem::size_of::<abi::FtPluginVtable>() as u32,
        _pad: 0,
        params,
        run: ok_run,
        last_error: no_error,
    });
    let desc = desc_for(vt, "dev.test.futurekind");
    let mut c = Collector {
        source: src().to_path_buf(),
        ..Default::default()
    };
    // SAFETY: a valid descriptor that outlives the call.
    unsafe {
        add_plugin_cb(
            &mut c as *mut Collector as *mut std::ffi::c_void,
            desc as *const abi::FtPluginDesc,
        )
    };
    let decls = c.plugins[0].params(&FakeHost::default());
    assert_eq!(decls.len(), 1, "the control must not vanish");
    assert_eq!(decls[0].kind, ParamKind::Label);
    assert!(
        decls[0].label.contains("Wavelet order") && decls[0].label.contains("newer FastTIFF"),
        "the user should see which control is missing and why: {:?}",
        decls[0].label
    );
}

/// Signed samples: declared `I16`, carried in the `U16` lane as raw bits.
/// This pairing used to be rejected by `validate`, which made `PixelType::I16`
/// a type a plugin could name and never successfully use.
#[test]
fn a_signed_result_crosses_and_validates() {
    unsafe extern "C" fn run(
        _h: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let s = &*sink;
        (s.begin_image)(
            s.ctx,
            2,
            2,
            1,
            1,
            1,
            abi::FtPixelType::I16,
            abi::FtStr::from_str("signed"),
        );
        // The bit patterns of -2, -1, 0, 1.
        let plane: [u16; 4] = [(-2i16) as u16, (-1i16) as u16, 0u16, 1u16];
        (s.push_plane)(s.ctx, plane.as_ptr() as *const std::ffi::c_void, 4);
        (s.set_outcome)(s.ctx, abi::FtOutcomeKind::NewDocument, abi::FtStr::EMPTY);
        abi::FtStatus::Ok
    }
    let mut p = plugin_with_run!(run);
    let out = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect("run");
    let Outcome::NewDocument(img) = out else {
        panic!("expected a document, got {out:?}");
    };
    assert_eq!(img.pixel_type, PixelType::I16, "the declaration is kept");
    assert_eq!(
        img.planes,
        vec![PlaneData::U16(vec![65534, 65535, 0, 1])],
        "the bits must arrive untouched"
    );
    img.validate()
        .expect("I16 in the U16 lane is a valid pairing");

    // And it becomes a real signed TIFF rather than being reinterpreted.
    let stack = crate::plugins::to_stack(&img, None, false).expect("open");
    assert_eq!(stack.dimensions(), Some((2, 2)));
}

// --- structs from a version this host has never heard of --------------------

/// The other half of the versioning scheme: a plugin whose out-parameter is
/// *smaller* than this host's must be filled only as far as it goes.
#[test]
fn the_host_fills_only_as_much_of_an_out_param_as_the_plugin_allocated() {
    // A plugin from an older minor: it allocated room for the first three
    // fields of `FtImageInfo` and said so.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct OldImageInfo {
        struct_size: u32,
        width: u32,
        height: u32,
        /// A canary the host must not touch.
        guard: u32,
    }

    unsafe extern "C" fn run(
        host: *const abi::FtHost,
        _v: *const abi::FtValue,
        _n: u64,
        sink: *const abi::FtSink,
    ) -> abi::FtStatus {
        let h = &*host;
        let mut old = OldImageInfo {
            // Only the first twelve bytes are ours.
            struct_size: 12,
            width: 0,
            height: 0,
            guard: 0xDEAD_BEEF,
        };
        let st = (h.image_info)(h.ctx, (&mut old as *mut OldImageInfo).cast());
        let s = &*sink;
        let ok = st == abi::FtStatus::Ok
            && old.width == 2
            && old.height == 2
            && old.guard == 0xDEAD_BEEF;
        (s.set_outcome)(
            s.ctx,
            abi::FtOutcomeKind::Message,
            abi::FtStr::from_str(if ok { "intact" } else { "clobbered" }),
        );
        abi::FtStatus::Ok
    }

    let mut p = plugin_with_run!(run);
    let out = p
        .run(&mut FakeHost::default(), &Params::new())
        .expect("run");
    assert_eq!(
        out,
        Outcome::Message("intact".into()),
        "the host wrote past the end of the plugin's struct"
    );
}

/// An importer that claims a preposterous number of file types is a corrupt
/// descriptor, and reading them would walk off the end of its array.
#[test]
fn an_importer_with_an_absurd_file_type_count_is_refused() {
    unsafe extern "C" fn probe(_p: abi::FtStr, _h: *const u8, _n: u64) -> u32 {
        abi::FtConfidence::No.0
    }
    unsafe extern "C" fn params(_p: abi::FtStr, _s: *const abi::FtParamSink) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    unsafe extern "C" fn import(
        _p: abi::FtStr,
        _v: *const abi::FtValue,
        _n: u64,
        _h: *const abi::FtHost,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        let vt = abi::FtImporterVtable {
            struct_size: std::mem::size_of::<abi::FtImporterVtable>() as u32,
            _pad: 0,
            probe,
            params,
            import,
            last_error: no_error,
        };
        let ext = abi::FtStr::from_str("zzz");
        let ft = abi::FtFileType {
            struct_size: std::mem::size_of::<abi::FtFileType>() as u32,
            _pad: 0,
            description: abi::FtStr::from_str("Nonsense"),
            extensions: &ext,
            extension_count: 1,
        };
        let desc = abi::FtImporterDesc {
            struct_size: std::mem::size_of::<abi::FtImporterDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.manytypes"),
            name: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            file_types: &ft,
            // One exists; four billion are claimed.
            file_type_count: u64::MAX,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_importer)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("an absurd count must be refused");
    assert!(err.contains("absurd number of file types"), "{err}");
}

/// The same, one level down: a file type whose extension array is a lie.
#[test]
fn an_importer_with_a_malformed_file_type_is_refused() {
    unsafe extern "C" fn probe(_p: abi::FtStr, _h: *const u8, _n: u64) -> u32 {
        0
    }
    unsafe extern "C" fn params(_p: abi::FtStr, _s: *const abi::FtParamSink) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    unsafe extern "C" fn import(
        _p: abi::FtStr,
        _v: *const abi::FtValue,
        _n: u64,
        _h: *const abi::FtHost,
        _s: *const abi::FtSink,
    ) -> abi::FtStatus {
        abi::FtStatus::Ok
    }
    unsafe extern "C" fn query(reg: *mut abi::FtRegistrar) -> abi::FtStatus {
        let vt = abi::FtImporterVtable {
            struct_size: std::mem::size_of::<abi::FtImporterVtable>() as u32,
            _pad: 0,
            probe,
            params,
            import,
            last_error: no_error,
        };
        let ext = abi::FtStr::from_str("zzz");
        let ft = abi::FtFileType {
            struct_size: std::mem::size_of::<abi::FtFileType>() as u32,
            _pad: 0,
            description: abi::FtStr::from_str("Nonsense"),
            extensions: &ext,
            extension_count: 10_000,
        };
        let desc = abi::FtImporterDesc {
            struct_size: std::mem::size_of::<abi::FtImporterDesc>() as u32,
            _pad: 0,
            id: abi::FtStr::from_str("dev.test.manyexts"),
            name: abi::FtStr::EMPTY,
            version: abi::FtStr::EMPTY,
            author: abi::FtStr::EMPTY,
            description: abi::FtStr::EMPTY,
            file_types: &ft,
            file_type_count: 1,
            vtable: &vt,
        };
        let r = &mut *reg;
        (r.add_importer)(r.ctx, &desc);
        abi::FtStatus::Ok
    }
    let err = load(query).expect_err("a malformed file type must be refused");
    assert!(err.contains("malformed file type"), "{err}");
}

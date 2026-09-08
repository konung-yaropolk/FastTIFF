//! Loading plugins from shared libraries.
//!
//! The host half of the C boundary: it opens a library, looks up
//! [`QUERY_SYMBOL`](fasttiff_plugin_abi::QUERY_SYMBOL), lets the plugin
//! register what it provides, and wraps each registration in a type that
//! implements the ordinary [`Plugin`]/[`Importer`] traits — so nothing above
//! this module can tell a loaded plugin from a built-in one.
//!
//! Everything here assumes the plugin is **wrong, not merely foreign**:
//!
//! * Its strings might not be UTF-8, or might claim a gigabyte of length.
//! * Its descriptors might be shorter than this ABI's, or claim to be longer.
//! * It might push more planes than it declared, or planes of the wrong size.
//! * It might panic, which must never unwind back into a Rust host across an
//!   `extern "C"` frame.
//!
//! Each of those is checked at the point it crosses, and the failure is a
//! message naming the plugin rather than a crash.
//!
//! # Libraries are never unloaded
//!
//! Once opened, a [`Library`] is leaked for the process's lifetime. Calling
//! `dlclose` while a plugin's thread, a registered TLS destructor, or a
//! function pointer the host still holds is alive is a family of crashes that
//! only ever appear on someone else's machine. Reinstalling a plugin therefore
//! needs a restart, which is the trade every long-lived plugin host makes.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fasttiff_plugin_abi as abi;
use fasttiff_plugin_api::{
    Confidence, ExportRequest, Exporter, FileType, HostContext, ImageResult, ImportHost,
    ImportRequest, ImportResult, Importer, Outcome, ParamDecl, ParamKind, ParamValue, Params,
    PixelType, PlaneData, Plugin, PluginError, PluginInfo, StackInfo,
};

use super::{Origin, Registry};

/// What one library gave us.
pub struct Loaded {
    pub plugins: Vec<LoadedPlugin>,
    pub importers: Vec<LoadedImporter>,
    pub exporters: Vec<LoadedExporter>,
}

impl std::fmt::Debug for Loaded {
    /// Counts rather than contents: the fields are function pointers into
    /// another binary, and printing them helps nobody.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Loaded {{ plugins: {}, importers: {}, exporters: {} }}",
            self.plugins.len(),
            self.importers.len(),
            self.exporters.len()
        )
    }
}

/// A plugin living in a shared library.
pub struct LoadedPlugin {
    info: PluginInfo,
    vtable: abi::FtPluginVtable,
    /// Kept so the error can name the file rather than just the plugin.
    source: PathBuf,
}

/// An importer living in a shared library.
pub struct LoadedImporter {
    info: PluginInfo,
    file_types: Vec<FileType>,
    vtable: abi::FtImporterVtable,
    source: PathBuf,
}

/// An exporter living in a shared library.
pub struct LoadedExporter {
    info: PluginInfo,
    file_types: Vec<FileType>,
    vtable: abi::FtExporterVtable,
    source: PathBuf,
}

// ---------------------------------------------------------------- discovery

/// Load every plugin library on the search path into `registry`.
///
/// Problems are recorded on the registry rather than returned: one unloadable
/// library must not stop the others, and the user needs to be told which one
/// and why.
pub fn load_all(registry: &mut Registry) {
    for dir in super::search_paths() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| super::is_library(p))
            .collect();
        // Deterministic order, so which of two clashing plugins wins does not
        // depend on the filesystem.
        files.sort();
        for file in files {
            match load_library(&file) {
                Ok(loaded) => {
                    for p in loaded.plugins {
                        registry.add(Box::new(p), Origin::Library);
                    }
                    for i in loaded.importers {
                        registry.add_importer(Box::new(i), Origin::Library);
                    }
                    for e in loaded.exporters {
                        registry.add_exporter(Box::new(e), Origin::Library);
                    }
                }
                Err(e) => registry.problems.push(format!(
                    "{}: {e}",
                    file.file_name().unwrap_or_default().to_string_lossy()
                )),
            }
        }
    }
}

/// What a library registered, collected while its query function runs.
#[derive(Default)]
struct Collector {
    plugins: Vec<LoadedPlugin>,
    importers: Vec<LoadedImporter>,
    exporters: Vec<LoadedExporter>,
    source: PathBuf,
    problems: Vec<String>,
}

/// Open one library and let it register itself.
pub fn load_library(path: &Path) -> Result<Loaded, String> {
    // SAFETY: opening a library runs its initialisers, which is inherent to
    // loading a plugin at all; there is no safe version of this.
    let lib =
        unsafe { libloading::Library::new(path) }.map_err(|e| format!("could not load: {e}"))?;

    let query: libloading::Symbol<abi::FtQueryFn> = unsafe {
        lib.get(abi::QUERY_SYMBOL).map_err(|_| {
            format!(
                "not a FastTIFF plugin, or built for a different ABI: no `{}` symbol",
                String::from_utf8_lossy(&abi::QUERY_SYMBOL[..abi::QUERY_SYMBOL.len() - 1])
            )
        })?
    };
    let query = *query;

    // SAFETY: `query` came from this library's export table, so it has the
    // signature the symbol name promises.
    let loaded = unsafe { register_from(query, path) };
    if loaded.is_ok() {
        // Deliberately leaked: see the module docs on never unloading.
        std::mem::forget(lib);
    }
    loaded
}

/// Everything `load_library` does except finding the function: run a plugin's
/// query entry point and adapt whatever it registers.
///
/// Split out because it is where all the risk lives — every string, descriptor
/// and vtable a plugin hands over is validated here, while `dlopen` and `dlsym`
/// above are two library calls with nothing to get wrong. Tests reach this
/// directly with a query function linked into the test binary, which is the
/// only way to exercise the marshalling against a *freshly compiled* plugin:
/// `cargo test` does not rebuild a `cdylib`, so a test that could only go
/// through `load_library` would keep passing against a stale artifact.
///
/// # Safety
/// `query` must be a real [`FtQueryFn`](abi::FtQueryFn) — in practice, one
/// produced by `fasttiff_plugin::export_plugin!` — and the code behind it must
/// stay loaded for as long as anything it registers is used.
pub unsafe fn register_from(query: abi::FtQueryFn, source: &Path) -> Result<Loaded, String> {
    let mut collector = Collector {
        source: source.to_path_buf(),
        ..Default::default()
    };
    let mut reg = abi::FtRegistrar {
        struct_size: std::mem::size_of::<abi::FtRegistrar>() as u32,
        host_abi_minor: abi::ABI_MINOR,
        ctx: &mut collector as *mut Collector as *mut std::ffi::c_void,
        plugin_abi_minor: 0,
        _pad: 0,
        add_plugin: add_plugin_cb,
        add_importer: add_importer_cb,
        add_exporter: add_exporter_cb,
    };

    // A panic must not unwind out of the plugin and through this frame.
    let status = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| query(&mut reg)))
        .map_err(|_| "panicked while registering".to_string())?;

    if status != abi::FtStatus::Ok {
        return Err(format!("registration failed: {status:?}"));
    }
    // Specific complaints first: "registered nothing" is true whenever a
    // descriptor was refused, and is the least useful thing to say about it.
    if !collector.problems.is_empty() {
        return Err(collector.problems.join("; "));
    }
    if collector.plugins.is_empty()
        && collector.importers.is_empty()
        && collector.exporters.is_empty()
    {
        return Err("loaded, but registered nothing".into());
    }
    // A plugin built against a newer minor version than this host is fine —
    // that is what the additive rule buys — but it is worth saying, because it
    // is the first thing to suspect if the plugin then behaves oddly.
    if reg.plugin_abi_minor > abi::ABI_MINOR {
        log::info!(
            "{}: built against plugin ABI 1.{}, this host provides 1.{};              anything newer than 1.{} is ignored",
            source.display(),
            reg.plugin_abi_minor,
            abi::ABI_MINOR,
            abi::ABI_MINOR
        );
    }

    Ok(Loaded {
        plugins: collector.plugins,
        importers: collector.importers,
        exporters: collector.exporters,
    })
}

/// Copy a borrowed plugin string, refusing anything that is not one.
///
/// # Safety
/// `s` must be a valid `FtStr` for the duration of this call.
unsafe fn owned(s: abi::FtStr, what: &str, problems: &mut Vec<String>) -> String {
    match s.as_str() {
        Some(v) => v.to_string(),
        None => {
            problems.push(format!("{what} is not valid UTF-8"));
            String::new()
        }
    }
}

unsafe extern "C" fn add_plugin_cb(
    ctx: *mut std::ffi::c_void,
    desc: *const abi::FtPluginDesc,
) -> abi::FtStatus {
    let Some(c) = (ctx as *mut Collector).as_mut() else {
        return abi::FtStatus::BadArgument;
    };
    // The size check comes *before* the reference, not after it: `&*desc` on a
    // descriptor built to an older, smaller layout is undefined behaviour the
    // moment it exists, whether or not the missing fields are ever read. See
    // `abi::fits`.
    if desc.is_null() {
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(desc) {
        c.problems
            .push("a plugin descriptor is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let d = &*desc;
    if d.vtable.is_null() {
        c.problems.push("a plugin registered a null vtable".into());
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(d.vtable) {
        c.problems
            .push("a plugin's vtable is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let vt = &*d.vtable;

    let info = PluginInfo {
        id: owned(d.id, "a plugin id", &mut c.problems),
        name: owned(d.name, "a plugin name", &mut c.problems),
        menu_path: owned(d.menu_path, "a plugin menu path", &mut c.problems),
        version: owned(d.version, "a plugin version", &mut c.problems),
        author: owned(d.author, "a plugin author", &mut c.problems),
        description: owned(d.description, "a plugin description", &mut c.problems),
    };
    c.plugins.push(LoadedPlugin {
        info,
        vtable: *vt,
        source: c.source.clone(),
    });
    abi::FtStatus::Ok
}

unsafe extern "C" fn add_importer_cb(
    ctx: *mut std::ffi::c_void,
    desc: *const abi::FtImporterDesc,
) -> abi::FtStatus {
    let Some(c) = (ctx as *mut Collector).as_mut() else {
        return abi::FtStatus::BadArgument;
    };
    // Checked before the reference is formed; see `add_plugin_cb`.
    if desc.is_null() {
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(desc) {
        c.problems
            .push("an importer descriptor is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let d = &*desc;
    if d.vtable.is_null() {
        c.problems
            .push("an importer registered a null vtable".into());
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(d.vtable) {
        c.problems
            .push("an importer's vtable is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let vt = &*d.vtable;

    let Some(file_types) = file_types_of(
        d.file_types,
        d.file_type_count,
        "an importer",
        &mut c.problems,
    ) else {
        return abi::FtStatus::BadArgument;
    };

    let info = PluginInfo {
        id: owned(d.id, "an importer id", &mut c.problems),
        name: owned(d.name, "an importer name", &mut c.problems),
        menu_path: String::new(),
        version: owned(d.version, "an importer version", &mut c.problems),
        author: owned(d.author, "an importer author", &mut c.problems),
        description: owned(d.description, "an importer description", &mut c.problems),
    };
    c.importers.push(LoadedImporter {
        info,
        file_types,
        vtable: *vt,
        source: c.source.clone(),
    });
    abi::FtStatus::Ok
}

unsafe extern "C" fn add_exporter_cb(
    ctx: *mut std::ffi::c_void,
    desc: *const abi::FtExporterDesc,
) -> abi::FtStatus {
    let Some(c) = (ctx as *mut Collector).as_mut() else {
        return abi::FtStatus::BadArgument;
    };
    // Checked before the reference is formed; see `add_plugin_cb`.
    if desc.is_null() {
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(desc) {
        c.problems
            .push("an exporter descriptor is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let d = &*desc;
    if d.vtable.is_null() {
        c.problems
            .push("an exporter registered a null vtable".into());
        return abi::FtStatus::BadArgument;
    }
    if !abi::fits(d.vtable) {
        c.problems
            .push("an exporter's vtable is older than this ABI".into());
        return abi::FtStatus::BadArgument;
    }
    let vt = &*d.vtable;

    let Some(file_types) = file_types_of(
        d.file_types,
        d.file_type_count,
        "an exporter",
        &mut c.problems,
    ) else {
        return abi::FtStatus::BadArgument;
    };

    let info = PluginInfo {
        id: owned(d.id, "an exporter id", &mut c.problems),
        name: owned(d.name, "an exporter name", &mut c.problems),
        menu_path: String::new(),
        version: owned(d.version, "an exporter version", &mut c.problems),
        author: owned(d.author, "an exporter author", &mut c.problems),
        description: owned(d.description, "an exporter description", &mut c.problems),
    };
    c.exporters.push(LoadedExporter {
        info,
        file_types,
        vtable: *vt,
        source: c.source.clone(),
    });
    abi::FtStatus::Ok
}

/// Copy the file types a descriptor declares, or `None` if the list is not one.
///
/// Shared by importers and exporters: both descriptors end in the same
/// `(*const FtFileType, u64)` pair, and a foreign one needs the same checks
/// whichever it came from. `what` names the kind in the message, which is the
/// only thing that differs between the two.
///
/// # Safety
/// `types` must be null, or point at `count` `FtFileType`s that stay valid for
/// the duration of this call.
unsafe fn file_types_of(
    types: *const abi::FtFileType,
    count: u64,
    what: &str,
    problems: &mut Vec<String>,
) -> Option<Vec<FileType>> {
    // A file-type count past anything real is a corrupt descriptor.
    const MAX_TYPES: u64 = 256;
    const MAX_EXTS: u64 = 256;
    if count > MAX_TYPES {
        problems.push(format!("{what} declares an absurd number of file types"));
        return None;
    }
    let mut out = Vec::new();
    if types.is_null() {
        return Some(out);
    }
    for i in 0..count {
        // Indexed with this side's stride, which is sound only because
        // `FtFileType` is an *arrayed* struct and therefore frozen in size
        // — see its doc comment. A struct reached through an array cannot
        // use the append-a-field rule: the stride is baked into the writer's
        // pointer arithmetic and the reader's, and the two would silently
        // disagree.
        let element = types.add(i as usize);
        if !abi::fits(element) || (*element).extension_count > MAX_EXTS {
            problems.push(format!("{what} declares a malformed file type"));
            return None;
        }
        let t = &*element;
        let mut exts = Vec::new();
        if !t.extensions.is_null() {
            for j in 0..t.extension_count {
                let e = *t.extensions.add(j as usize);
                if let Some(s) = e.as_str() {
                    exts.push(s.to_lowercase());
                }
            }
        }
        out.push(FileType {
            description: owned(t.description, "a file type description", problems),
            extensions: exts,
        });
    }
    Some(out)
}

// ------------------------------------------------------------- host callbacks

/// The host state a running plugin can reach, behind the C table.
struct HostCell<'a> {
    inner: &'a mut dyn HostContext,
    name: String,
    path: String,
}

/// Run `f`, converting a panic in the *host's own* callback into a status.
///
/// The host's callbacks are called from plugin frames, so a panic here would
/// unwind through `extern "C"` exactly as a plugin's would.
fn host_guard<F: FnOnce() -> abi::FtStatus>(f: F) -> abi::FtStatus {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(abi::FtStatus::Panic)
}

unsafe extern "C" fn cb_image_info(
    ctx: *mut std::ffi::c_void,
    out: *mut abi::FtImageInfo,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(c) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let i = c.inner.image();
        // Filled through `write_prefix`, which writes only as many bytes as the
        // plugin said it allocated. Assigning through `&mut *out` would write
        // this ABI's whole struct into whatever an older plugin provided.
        let filled = abi::FtImageInfo {
            struct_size: 0, // replaced with the caller's own inside write_prefix
            width: i.width,
            height: i.height,
            samples_per_pixel: i.samples_per_pixel as u32,
            channels: i.channels as u64,
            slices: i.slices as u64,
            frames: i.frames as u64,
            pixel_type: match i.pixel_type {
                PixelType::U8 => abi::FtPixelType::U8,
                PixelType::U16 => abi::FtPixelType::U16,
                PixelType::I16 => abi::FtPixelType::I16,
                PixelType::F32 => abi::FtPixelType::F32,
            },
            _pad: 0,
        };
        if abi::write_prefix(out, filled) {
            abi::FtStatus::Ok
        } else {
            abi::FtStatus::BadArgument
        }
    })
}

unsafe extern "C" fn cb_view_params(
    ctx: *mut std::ffi::c_void,
    out: *mut abi::FtViewParams,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(c) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let v = c.inner.view();
        // Through `write_prefix`; see `cb_image_info`.
        let filled = abi::FtViewParams {
            struct_size: 0,
            volume_view: u32::from(v.volume_view),
            frame_index: v.frame_index as u64,
            shown_channels: v.channels.len() as u64,
            volume_mode: match v.volume.mode {
                fasttiff_plugin_api::VolumeMode::Dvr => 1,
                fasttiff_plugin_api::VolumeMode::Surface => 2,
                _ => 0,
            },
            _pad: 0,
            density: v.volume.density,
            iso: v.volume.iso,
            eye: v.volume.eye,
            forward: v.volume.forward,
            up: v.volume.up,
            right: v.volume.right,
        };
        if abi::write_prefix(out, filled) {
            abi::FtStatus::Ok
        } else {
            abi::FtStatus::BadArgument
        }
    })
}

unsafe extern "C" fn cb_channel_view(
    ctx: *mut std::ffi::c_void,
    index: u64,
    out: *mut abi::FtChannelView,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(c) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        match c.inner.view().channels.get(index as usize) {
            Some(cv) => {
                // Through `write_prefix`; see `cb_image_info`.
                let filled = abi::FtChannelView {
                    struct_size: 0,
                    _pad0: 0,
                    min: cv.min,
                    max: cv.max,
                    enabled: u32::from(cv.enabled),
                    _pad: 0,
                };
                if abi::write_prefix(out, filled) {
                    abi::FtStatus::Ok
                } else {
                    abi::FtStatus::BadArgument
                }
            }
            None => abi::FtStatus::OutOfRange,
        }
    })
}

unsafe extern "C" fn cb_stack_name(ctx: *mut std::ffi::c_void) -> abi::FtStr {
    match (ctx as *mut HostCell).as_ref() {
        Some(c) => abi::FtStr::from_str(&c.name),
        None => abi::FtStr::EMPTY,
    }
}

unsafe extern "C" fn cb_stack_path(ctx: *mut std::ffi::c_void) -> abi::FtStr {
    match (ctx as *mut HostCell).as_ref() {
        Some(c) => abi::FtStr::from_str(&c.path),
        None => abi::FtStr::EMPTY,
    }
}

unsafe extern "C" fn cb_read_plane_f32(
    ctx: *mut std::ffi::c_void,
    c: u64,
    z: u64,
    t: u64,
    out: *mut f32,
    cap: u64,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(cell) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let need = cell.inner.image().plane_len() as u64;
        // The plugin states the buffer's capacity; writing past it would be a
        // buffer overflow in the *plugin's* memory, caused by the host.
        if out.is_null() || cap < need {
            return abi::FtStatus::BadArgument;
        }
        let mut buf = Vec::new();
        match cell.inner.read_plane_f32(
            fasttiff_plugin_api::Plane::new(c as usize, z as usize, t as usize),
            &mut buf,
        ) {
            Ok(()) => {
                let n = buf.len().min(need as usize);
                std::ptr::copy_nonoverlapping(buf.as_ptr(), out, n);
                abi::FtStatus::Ok
            }
            Err(PluginError::OutOfRange(_)) => abi::FtStatus::OutOfRange,
            Err(_) => abi::FtStatus::Error,
        }
    })
}

unsafe extern "C" fn cb_read_plane_u16(
    ctx: *mut std::ffi::c_void,
    c: u64,
    z: u64,
    t: u64,
    out: *mut u16,
    cap: u64,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(cell) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let need = cell.inner.image().plane_len() as u64;
        if out.is_null() || cap < need {
            return abi::FtStatus::BadArgument;
        }
        let mut buf = Vec::new();
        match cell.inner.read_plane_u16(
            fasttiff_plugin_api::Plane::new(c as usize, z as usize, t as usize),
            &mut buf,
        ) {
            Ok(()) => {
                let n = buf.len().min(need as usize);
                std::ptr::copy_nonoverlapping(buf.as_ptr(), out, n);
                abi::FtStatus::Ok
            }
            Err(PluginError::OutOfRange(_)) => abi::FtStatus::OutOfRange,
            Err(_) => abi::FtStatus::Error,
        }
    })
}

unsafe extern "C" fn cb_stack_info(
    ctx: *mut std::ffi::c_void,
    out: *mut abi::FtStackInfo,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(c) = (ctx as *mut HostCell).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let i = c.inner.stack_info();
        let mut present = 0u32;
        let mut flag = |set: bool, bit: u32| {
            if set {
                present |= bit;
            }
        };
        flag(i.spacing.x.is_some(), abi::FT_HAS_SPACING_X);
        flag(i.spacing.y.is_some(), abi::FT_HAS_SPACING_Y);
        flag(i.spacing.z.is_some(), abi::FT_HAS_SPACING_Z);
        flag(i.frame_interval_s.is_some(), abi::FT_HAS_FRAME_INTERVAL);
        flag(i.calibration.is_some(), abi::FT_HAS_CALIBRATION);
        let (c0, c1) = i.calibration.unwrap_or((0.0, 0.0));
        let filled = abi::FtStackInfo {
            struct_size: 0,
            mode: match i.mode {
                fasttiff_plugin_api::DisplayMode::Composite => 1,
                fasttiff_plugin_api::DisplayMode::Color => 2,
                _ => 0,
            },
            present,
            _pad: 0,
            spacing_x: i.spacing.x.unwrap_or(0.0),
            spacing_y: i.spacing.y.unwrap_or(0.0),
            spacing_z: i.spacing.z.unwrap_or(0.0),
            frame_interval_s: i.frame_interval_s.unwrap_or(0.0),
            calibration_offset: c0,
            calibration_scale: c1,
        };
        if abi::write_prefix(out, filled) {
            abi::FtStatus::Ok
        } else {
            abi::FtStatus::BadArgument
        }
    })
}

unsafe extern "C" fn cb_stack_string(
    ctx: *mut std::ffi::c_void,
    which: u32,
    index: u64,
) -> abi::FtStr {
    let Some(c) = (ctx as *mut HostCell).as_mut() else {
        return abi::FtStr::EMPTY;
    };
    // Borrowed from the `StackInfo` the cell holds, which outlives the call —
    // the contract requires the plugin to copy before returning, and it does.
    let s = match which {
        abi::FT_STRING_UNIT => c.inner.stack_info().unit.as_deref(),
        abi::FT_STRING_DESCRIPTION => c.inner.stack_info().description.as_deref(),
        abi::FT_STRING_CHANNEL_NAME => c
            .inner
            .stack_info()
            .channel_names
            .get(index as usize)
            .map(|n| n.as_str()),
        // A selector this host does not know: an empty string is the right
        // answer, because every string here is optional anyway.
        _ => None,
    };
    s.map(abi::FtStr::from_str).unwrap_or(abi::FtStr::EMPTY)
}

unsafe extern "C" fn cb_progress(ctx: *mut std::ffi::c_void, fraction: f32) -> u32 {
    match (ctx as *mut HostCell).as_mut() {
        Some(c) => {
            let keep = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                c.inner.progress(fraction)
            }))
            .unwrap_or(false);
            u32::from(keep)
        }
        None => 0,
    }
}

unsafe extern "C" fn cb_log(ctx: *mut std::ffi::c_void, message: abi::FtStr) {
    if let Some(c) = (ctx as *mut HostCell).as_mut() {
        if let Some(m) = message.as_str() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.inner.log(m)));
        }
    }
}

fn host_table(cell: &mut HostCell) -> abi::FtHost {
    abi::FtHost {
        struct_size: std::mem::size_of::<abi::FtHost>() as u32,
        _pad: 0,
        ctx: cell as *mut HostCell as *mut std::ffi::c_void,
        image_info: cb_image_info,
        view_params: cb_view_params,
        channel_view: cb_channel_view,
        stack_name: cb_stack_name,
        stack_path: cb_stack_path,
        read_plane_f32: cb_read_plane_f32,
        read_plane_u16: cb_read_plane_u16,
        progress: cb_progress,
        log: cb_log,
        stack_info: cb_stack_info,
        stack_string: cb_stack_string,
    }
}

// ------------------------------------------------------------------- sinks

/// Collects what a plugin pushes. Everything is copied here, inside the call.
#[derive(Default)]
struct ResultSink {
    width: u32,
    height: u32,
    channels: usize,
    slices: usize,
    frames: usize,
    pixel_type: Option<PixelType>,
    name: String,
    planes: Vec<PlaneData>,
    kind: Option<abi::FtOutcomeKind>,
    text: String,
    problem: Option<String>,
    /// What an importer said about the file it parsed. `None` when the plugin
    /// did not say, which is the normal case for a filter.
    info: Option<StackInfo>,
    /// Per-channel colours, by index. Sparse: a plugin may colour some
    /// channels and leave the rest to the host.
    colors: std::collections::BTreeMap<u64, [u8; 3]>,
}

unsafe extern "C" fn sink_begin(
    ctx: *mut std::ffi::c_void,
    width: u32,
    height: u32,
    channels: u64,
    slices: u64,
    frames: u64,
    pixel_type: abi::FtPixelType,
    name: abi::FtStr,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut ResultSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        // A declared size that cannot exist must be refused before anything is
        // allocated from it.
        let px = (width as u64).checked_mul(height as u64);
        let planes = channels
            .checked_mul(slices)
            .and_then(|v| v.checked_mul(frames));
        let (Some(px), Some(planes)) = (px, planes) else {
            s.problem = Some("the plugin declared an impossible result size".into());
            return abi::FtStatus::BadArgument;
        };
        if width == 0 || height == 0 || planes == 0 {
            s.problem = Some("the plugin declared an empty result".into());
            return abi::FtStatus::BadArgument;
        }
        // 16 Gsamples is far past any real image and short of overflowing.
        if px.saturating_mul(planes) > (1u64 << 34) {
            s.problem = Some("the plugin declared a result too large to be real".into());
            return abi::FtStatus::BadArgument;
        }
        s.width = width;
        s.height = height;
        s.channels = channels as usize;
        s.slices = slices as usize;
        s.frames = frames as usize;
        s.pixel_type = Some(match pixel_type {
            abi::FtPixelType::U8 => PixelType::U8,
            abi::FtPixelType::U16 => PixelType::U16,
            abi::FtPixelType::F32 => PixelType::F32,
            // Signed samples travel in the unsigned lane as raw bits — see
            // `PlaneData` — so the declaration is kept and `push_plane` below
            // reads the same sixteen bits as `u16`. The writer then emits them
            // as `SampleType::I16`, which is the same bit pattern again.
            abi::FtPixelType::I16 => PixelType::I16,
            other => {
                s.problem = Some(format!(
                    "the plugin declared a sample format this version of                      FastTIFF does not know ({other:?})"
                ));
                return abi::FtStatus::BadArgument;
            }
        });
        s.name = name.as_str().unwrap_or("plugin result").to_string();
        s.planes.clear();
        abi::FtStatus::Ok
    })
}

unsafe extern "C" fn sink_push_plane(
    ctx: *mut std::ffi::c_void,
    data: *const std::ffi::c_void,
    len: u64,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut ResultSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        let Some(ty) = s.pixel_type else {
            s.problem = Some("the plugin pushed a plane before declaring the result".into());
            return abi::FtStatus::BadArgument;
        };
        let want = s.width as u64 * s.height as u64;
        if data.is_null() || len != want {
            s.problem = Some(format!(
                "the plugin pushed a plane of {len} samples where {want} were declared"
            ));
            return abi::FtStatus::BadArgument;
        }
        let expect = s.channels.max(1) * s.slices.max(1) * s.frames.max(1);
        if s.planes.len() >= expect {
            s.problem = Some(format!(
                "the plugin pushed more than the {expect} planes it declared"
            ));
            return abi::FtStatus::BadArgument;
        }
        let n = len as usize;
        // Copied here, inside the call: the plugin's memory is not the host's
        // to hold on to.
        let plane = match ty {
            PixelType::U8 => {
                PlaneData::U8(std::slice::from_raw_parts(data as *const u8, n).to_vec())
            }
            PixelType::F32 => {
                PlaneData::F32(std::slice::from_raw_parts(data as *const f32, n).to_vec())
            }
            // U16 and I16 are both sixteen bits in the `U16` lane; which of
            // them they mean is what the declared `pixel_type` says.
            _ => PlaneData::U16(std::slice::from_raw_parts(data as *const u16, n).to_vec()),
        };
        s.planes.push(plane);
        abi::FtStatus::Ok
    })
}

unsafe extern "C" fn sink_set_outcome(
    ctx: *mut std::ffi::c_void,
    kind: abi::FtOutcomeKind,
    text: abi::FtStr,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut ResultSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        s.kind = Some(kind);
        s.text = text.as_str().unwrap_or("").to_string();
        abi::FtStatus::Ok
    })
}

unsafe extern "C" fn sink_set_info(
    ctx: *mut std::ffi::c_void,
    info: *const abi::FtStackInfo,
    unit: abi::FtStr,
    description: abi::FtStr,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut ResultSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        // Checked before the reference is formed; see `add_plugin_cb`.
        if !abi::fits(info) {
            return abi::FtStatus::BadArgument;
        }
        let i = &*info;
        let opt = |flag: u32, v: f64| (i.present & flag != 0).then_some(v);
        let text = |s: abi::FtStr| s.as_str().filter(|v| !v.is_empty()).map(|v| v.to_string());
        s.info = Some(StackInfo {
            // The host fills in the name and path: it knows which file it asked
            // for, and a plugin naming a *different* one is not something to
            // take at face value.
            name: String::new(),
            path: None,
            mode: match i.mode {
                1 => fasttiff_plugin_api::DisplayMode::Composite,
                2 => fasttiff_plugin_api::DisplayMode::Color,
                _ => fasttiff_plugin_api::DisplayMode::Grayscale,
            },
            unit: text(unit),
            spacing: fasttiff_plugin_api::Spacing {
                x: opt(abi::FT_HAS_SPACING_X, i.spacing_x),
                y: opt(abi::FT_HAS_SPACING_Y, i.spacing_y),
                z: opt(abi::FT_HAS_SPACING_Z, i.spacing_z),
            },
            frame_interval_s: opt(abi::FT_HAS_FRAME_INTERVAL, i.frame_interval_s),
            channel_names: Vec::new(),
            calibration: (i.present & abi::FT_HAS_CALIBRATION != 0)
                .then_some((i.calibration_offset, i.calibration_scale)),
            description: text(description),
        });
        abi::FtStatus::Ok
    })
}

unsafe extern "C" fn sink_set_channel(
    ctx: *mut std::ffi::c_void,
    index: u64,
    _name: abi::FtStr,
    rgb: u32,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut ResultSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        // More channels than any real image, so a runaway loop cannot fill
        // memory one entry at a time.
        if index > 4096 {
            return abi::FtStatus::OutOfRange;
        }
        s.colors
            .insert(index, [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]);
        abi::FtStatus::Ok
    })
}

fn sink_table(sink: &mut ResultSink) -> abi::FtSink {
    abi::FtSink {
        struct_size: std::mem::size_of::<abi::FtSink>() as u32,
        _pad: 0,
        ctx: sink as *mut ResultSink as *mut std::ffi::c_void,
        begin_image: sink_begin,
        push_plane: sink_push_plane,
        set_outcome: sink_set_outcome,
        set_info: sink_set_info,
        set_channel: sink_set_channel,
    }
}

impl ResultSink {
    /// Turn what was pushed into an outcome, or say what was wrong with it.
    fn finish(self, name: &str) -> Result<Outcome, PluginError> {
        if let Some(p) = self.problem {
            return Err(PluginError::failed(p));
        }
        let Some(kind) = self.kind else {
            return Err(PluginError::failed(format!(
                "{name} returned without saying what to do with its result"
            )));
        };
        match kind {
            abi::FtOutcomeKind::Nothing => Ok(Outcome::Nothing),
            abi::FtOutcomeKind::Message => Ok(Outcome::Message(self.text)),
            abi::FtOutcomeKind::NewDocument | abi::FtOutcomeKind::SaveToFile => {
                // Dense, up to the highest channel the plugin coloured: the
                // result carries a colour per channel or none at all, and a
                // half-filled list would silently mean "black" for the rest.
                let channel_colors = match self.colors.keys().next_back() {
                    Some(&highest) => (0..=highest)
                        .map(|i| {
                            self.colors.get(&i).copied().unwrap_or_else(|| {
                                fast_tiff_lib::metadata::composite_color(i as usize)
                            })
                        })
                        .collect(),
                    None => Vec::new(),
                };
                let image = ImageResult {
                    width: self.width,
                    height: self.height,
                    channels: self.channels,
                    slices: self.slices,
                    frames: self.frames,
                    pixel_type: self.pixel_type.unwrap_or(PixelType::U16),
                    planes: self.planes,
                    channel_colors,
                    name: self.name,
                };
                // The plugin may simply have pushed too few planes; the host
                // checks rather than trusting the declaration.
                image.validate()?;
                if kind == abi::FtOutcomeKind::NewDocument {
                    Ok(Outcome::NewDocument(Box::new(image)))
                } else {
                    Ok(Outcome::SaveToFile { image: Box::new(image), path: self.text })
                }
            }
            // A kind from a newer ABI. The plugin did the work and expects
            // something to happen to the result; quietly doing nothing would
            // look like a plugin that silently fails.
            other => Err(PluginError::failed(format!(
                "{name} asked for something this version of FastTIFF does not                  know how to do ({other:?}); the plugin needs a newer FastTIFF"
            ))),
        }
    }
}

/// Collects a plugin's declared dialog.
#[derive(Default)]
struct DeclSink {
    decls: Vec<ParamDecl>,
}

unsafe extern "C" fn decl_push(
    ctx: *mut std::ffi::c_void,
    decl: *const abi::FtParamDecl,
) -> abi::FtStatus {
    host_guard(|| {
        let Some(s) = (ctx as *mut DeclSink).as_mut() else {
            return abi::FtStatus::BadArgument;
        };
        // Checked before the reference is formed; see `add_plugin_cb`.
        if !abi::fits(decl) {
            return abi::FtStatus::BadArgument;
        }
        let d = &*decl;
        if (d.struct_size as usize) < std::mem::size_of::<abi::FtParamDecl>() {
            return abi::FtStatus::BadArgument;
        }
        // A dialog with thousands of controls is a runaway loop, not a dialog.
        if s.decls.len() >= 256 {
            return abi::FtStatus::BadArgument;
        }
        let Some(key) = d.key.as_str() else {
            return abi::FtStatus::BadArgument;
        };
        let label = d.label.as_str().unwrap_or("").to_string();
        let help = d
            .help
            .as_str()
            .filter(|h| !h.is_empty())
            .map(|h| h.to_string());

        let kind = match d.kind {
            abi::FtParamKind::Int => ParamKind::Int {
                default: d.i_default,
                min: d.i_min,
                max: d.i_max,
            },
            abi::FtParamKind::Float => ParamKind::Float {
                default: d.f_default,
                min: d.f_min,
                max: d.f_max,
            },
            abi::FtParamKind::Bool => ParamKind::Bool {
                default: d.b_default != 0,
            },
            abi::FtParamKind::Choice => {
                if d.option_count > 1024 {
                    return abi::FtStatus::BadArgument;
                }
                let mut options = Vec::new();
                if !d.options.is_null() {
                    for i in 0..d.option_count {
                        match (*d.options.add(i as usize)).as_str() {
                            Some(o) => options.push(o.to_string()),
                            None => return abi::FtStatus::BadArgument,
                        }
                    }
                }
                ParamKind::Choice {
                    default: d.i_default.max(0) as usize,
                    options,
                }
            }
            abi::FtParamKind::Text => ParamKind::Text {
                default: d.s_default.as_str().unwrap_or("").to_string(),
            },
            abi::FtParamKind::Path => ParamKind::Path {
                default: d.s_default.as_str().unwrap_or("").to_string(),
                save: d.save != 0,
            },
            abi::FtParamKind::Label => ParamKind::Label,
            // A control kind from a newer ABI. Dropping it would show the user
            // a dialog quietly missing a setting the plugin expects them to
            // make; refusing the whole dialog would hide the reason. A label in
            // the control's own place says exactly what is wrong, where the
            // user is looking.
            _ => ParamKind::Label,
        };
        let label = if d.kind.0 > abi::FtParamKind::Label.0 {
            format!("{label} — needs a newer FastTIFF")
        } else {
            label
        };
        s.decls.push(ParamDecl {
            key: key.to_string(),
            label,
            help,
            kind,
        });
        abi::FtStatus::Ok
    })
}

fn decl_sink_table(sink: &mut DeclSink) -> abi::FtParamSink {
    abi::FtParamSink {
        struct_size: std::mem::size_of::<abi::FtParamSink>() as u32,
        _pad: 0,
        ctx: sink as *mut DeclSink as *mut std::ffi::c_void,
        push: decl_push,
    }
}

/// Turn `Params` into the flat array the ABI passes.
///
/// The returned strings must outlive the borrowed `FtValue`s, so both are
/// returned together and the caller keeps them alive across the call.
fn values_to_c(params: &Params) -> (Vec<abi::FtValue>, Vec<(String, String)>) {
    let mut owned: Vec<(String, String)> = Vec::new();
    for (k, v) in params.iter() {
        let s = match v {
            ParamValue::Text(t) | ParamValue::Path(t) => t.clone(),
            _ => String::new(),
        };
        owned.push((k.to_string(), s));
    }
    let values = params
        .iter()
        .zip(owned.iter())
        .map(|((_, v), (k, s))| {
            let mut out = abi::FtValue {
                struct_size: std::mem::size_of::<abi::FtValue>() as u32,
                kind: abi::FtParamKind::Label,
                key: abi::FtStr::from_str(k),
                i: 0,
                f: 0.0,
                b: 0,
                _pad: 0,
                s: abi::FtStr::from_str(s),
            };
            match v {
                ParamValue::Int(i) => {
                    out.kind = abi::FtParamKind::Int;
                    out.i = *i;
                }
                ParamValue::Float(f) => {
                    out.kind = abi::FtParamKind::Float;
                    out.f = *f;
                }
                ParamValue::Bool(b) => {
                    out.kind = abi::FtParamKind::Bool;
                    out.b = u32::from(*b);
                }
                ParamValue::Choice(c) => {
                    out.kind = abi::FtParamKind::Choice;
                    out.i = *c as i64;
                }
                ParamValue::Text(_) => out.kind = abi::FtParamKind::Text,
                ParamValue::Path(_) => out.kind = abi::FtParamKind::Path,
            }
            out
        })
        .collect();
    (values, owned)
}

/// Read a plugin's `last_error`, defensively.
unsafe fn last_error_of(f: unsafe extern "C" fn() -> abi::FtStr, fallback: &str) -> String {
    let s = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f()))
        .ok()
        .and_then(|s| s.as_str().map(|v| v.to_string()))
        .unwrap_or_default();
    if s.trim().is_empty() {
        fallback.to_string()
    } else {
        s
    }
}

/// What to say when a plugin panicked.
///
/// The plugin's guard stores only the panic payload, because it does not know
/// its own name or which file it was loaded from; this side knows both and
/// nothing about the panic. Neither half is much use alone: "panicked" does not
/// say which of six installed plugins to remove, and the payload does not say
/// where to look.
fn panic_message(name: &str, source: &Path, detail: String) -> String {
    let detail = detail.trim();
    if detail.is_empty() {
        format!("{name} panicked (from {})", source.display())
    } else {
        format!("{name} panicked (from {}): {detail}", source.display())
    }
}

/// One plugin call may not be re-entered while another is running: the sinks
/// and host cells are addressed by raw pointer, and a plugin calling back into
/// the host to run another plugin would alias them.
static CALL_LOCK: Mutex<()> = Mutex::new(());

impl Plugin for LoadedPlugin {
    fn info(&self) -> PluginInfo {
        self.info.clone()
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // `params` takes `&dyn`, but the C table is uniform over `&mut`; the
        // callbacks it can reach from here are read-only.
        let mut shim = ReadOnly(host);
        let mut cell = HostCell {
            name: host.stack_info().name.clone(),
            path: host.stack_info().path.clone().unwrap_or_default(),
            inner: &mut shim,
        };
        let table = host_table(&mut cell);
        let mut sink = DeclSink::default();
        let sink_table = decl_sink_table(&mut sink);
        // SAFETY: both tables point at locals that outlive the call, and the
        // plugin's own shim catches its panics; `catch_unwind` here is the
        // second line for a plugin that was not built with this crate.
        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.params)(&table, &sink_table)
        }))
        .unwrap_or(abi::FtStatus::Panic);
        if st != abi::FtStatus::Ok {
            return Vec::new();
        }
        sink.decls
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let name = self.info.name.clone();
        // `_owned` holds the strings the `FtValue`s point at. The leading
        // underscore keeps it alive to the end of this scope — a bare `_` would
        // drop it here and leave every key and text value dangling for the
        // whole call.
        let (values, _owned) = values_to_c(params);
        let mut cell = HostCell {
            name: host.stack_info().name.clone(),
            path: host.stack_info().path.clone().unwrap_or_default(),
            inner: host,
        };
        let table = host_table(&mut cell);
        let mut sink = ResultSink::default();
        let sink_table = sink_table(&mut sink);

        // SAFETY: every pointer handed over refers to a local that outlives
        // this call, and the plugin copies nothing it is not given.
        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.run)(
                &table,
                if values.is_empty() {
                    std::ptr::null()
                } else {
                    values.as_ptr()
                },
                values.len() as u64,
                &sink_table,
            )
        }))
        .unwrap_or(abi::FtStatus::Panic);

        match st {
            abi::FtStatus::Ok => sink.finish(&name),
            abi::FtStatus::Cancelled => Ok(Outcome::Cancelled),
            abi::FtStatus::Unsupported => Err(PluginError::unsupported(unsafe {
                last_error_of(self.vtable.last_error, "not applicable to this stack")
            })),
            abi::FtStatus::OutOfRange => Err(PluginError::OutOfRange(unsafe {
                last_error_of(
                    self.vtable.last_error,
                    "asked for a plane that does not exist",
                )
            })),
            abi::FtStatus::Panic => Err(PluginError::failed(panic_message(
                &name,
                &self.source,
                // SAFETY: the plugin's `last_error`, copied before returning.
                unsafe { last_error_of(self.vtable.last_error, "") },
            ))),
            other => Err(PluginError::failed(unsafe {
                last_error_of(self.vtable.last_error, &format!("failed ({other:?})"))
            })),
        }
    }
}

/// A `HostContext` with no stack behind it, wrapping an [`ImportHost`].
///
/// An importer runs before anything is open, so there is nothing to answer
/// `image()` or `read_plane_*` with. But it still needs the other half of the
/// table: a long import must be able to report progress, be cancelled, and log
/// — and the ABI has exactly one host table, so the choice is this or passing
/// null and throwing all three away.
struct ImportOnly<'a>(&'a mut dyn ImportHost);

impl HostContext for ImportOnly<'_> {
    fn image(&self) -> fasttiff_plugin_api::ImageInfo {
        // Zero, honestly: there is no image yet. `plane_len()` is then 0, so
        // the pixel readers below refuse every request rather than appearing to
        // offer a 1x1 one.
        fasttiff_plugin_api::ImageInfo {
            width: 0,
            height: 0,
            channels: 0,
            slices: 0,
            frames: 0,
            samples_per_pixel: 0,
            pixel_type: PixelType::U16,
        }
    }
    fn view(&self) -> &fasttiff_plugin_api::ViewParams {
        // A default view, built once: there is no viewer state to describe
        // before the file has been read.
        static EMPTY: std::sync::OnceLock<fasttiff_plugin_api::ViewParams> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(|| fasttiff_plugin_api::ViewParams {
            frame_index: 0,
            volume_view: false,
            channels: Vec::new(),
            luts: Vec::new(),
            volume: fasttiff_plugin_api::VolumeView {
                mode: fasttiff_plugin_api::VolumeMode::Mip,
                density: 1.0,
                iso: 0.5,
                eye: [0.0; 3],
                forward: [0.0, 0.0, 1.0],
                up: [0.0, 1.0, 0.0],
                right: [1.0, 0.0, 0.0],
            },
        })
    }
    fn stack_info(&self) -> &StackInfo {
        static EMPTY: std::sync::OnceLock<StackInfo> = std::sync::OnceLock::new();
        EMPTY.get_or_init(StackInfo::default)
    }
    fn read_plane_u16(
        &mut self,
        _p: fasttiff_plugin_api::Plane,
        _o: &mut Vec<u16>,
    ) -> Result<(), PluginError> {
        Err(PluginError::unsupported(
            "no stack is open during an import",
        ))
    }
    fn read_plane_f32(
        &mut self,
        _p: fasttiff_plugin_api::Plane,
        _o: &mut Vec<f32>,
    ) -> Result<(), PluginError> {
        Err(PluginError::unsupported(
            "no stack is open during an import",
        ))
    }
    fn progress(&mut self, fraction: f32) -> bool {
        self.0.progress(fraction)
    }
    fn log(&mut self, message: &str) {
        self.0.log(message)
    }
}

/// Lets a `&dyn HostContext` be used where the C table needs `&mut dyn`.
///
/// Only the read-only callbacks are reachable during `params`; the pixel
/// readers take `&mut self` on the trait and are not called there.
struct ReadOnly<'a>(&'a dyn HostContext);

impl HostContext for ReadOnly<'_> {
    fn image(&self) -> fasttiff_plugin_api::ImageInfo {
        self.0.image()
    }
    fn view(&self) -> &fasttiff_plugin_api::ViewParams {
        self.0.view()
    }
    fn stack_info(&self) -> &StackInfo {
        self.0.stack_info()
    }
    fn read_plane_u16(
        &mut self,
        _p: fasttiff_plugin_api::Plane,
        _o: &mut Vec<u16>,
    ) -> Result<(), PluginError> {
        Err(PluginError::failed(
            "pixels cannot be read while declaring a dialog",
        ))
    }
    fn read_plane_f32(
        &mut self,
        _p: fasttiff_plugin_api::Plane,
        _o: &mut Vec<f32>,
    ) -> Result<(), PluginError> {
        Err(PluginError::failed(
            "pixels cannot be read while declaring a dialog",
        ))
    }
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
}

impl Importer for LoadedImporter {
    fn info(&self) -> PluginInfo {
        self.info.clone()
    }

    fn file_types(&self) -> Vec<FileType> {
        self.file_types.clone()
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        let p = path.to_string_lossy().to_string();
        let c = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.probe)(abi::FtStr::from_str(&p), head.as_ptr(), head.len() as u64)
        }))
        .unwrap_or(abi::FtConfidence::No.0);
        match c {
            2 => Confidence::Certain,
            1 => Confidence::Maybe,
            _ => Confidence::No,
        }
    }

    fn params(&self, path: &Path) -> Vec<ParamDecl> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let p = path.to_string_lossy().to_string();
        let mut sink = DeclSink::default();
        let table = decl_sink_table(&mut sink);
        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.params)(abi::FtStr::from_str(&p), &table)
        }))
        .unwrap_or(abi::FtStatus::Panic);
        if st != abi::FtStatus::Ok {
            return Vec::new();
        }
        sink.decls
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let name = self.info.name.clone();
        let p = request.path.to_string_lossy().to_string();
        // Kept alive for the call; see the note in `Plugin::run`.
        let (values, _owned) = values_to_c(&request.params);
        let mut sink = ResultSink::default();
        let sink_table = sink_table(&mut sink);

        // An import has no stack, but it still needs progress, cancellation and
        // logging, so it gets a host table whose pixel readers refuse rather
        // than no table at all.
        let mut shim = ImportOnly(host);
        let mut cell = HostCell {
            name: String::new(),
            path: p.clone(),
            inner: &mut shim,
        };
        let table = host_table(&mut cell);

        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.import)(
                abi::FtStr::from_str(&p),
                if values.is_empty() {
                    std::ptr::null()
                } else {
                    values.as_ptr()
                },
                values.len() as u64,
                &table,
                &sink_table,
            )
        }))
        .unwrap_or(abi::FtStatus::Panic);

        match st {
            abi::FtStatus::Ok => {
                // Taken before `finish` consumes the sink.
                let declared = sink.info.take();
                match sink.finish(&name)? {
                    Outcome::NewDocument(image) => Ok(ImportResult {
                        // What the importer read out of the file, with the name and
                        // path filled in by the host — which is the side that knows
                        // them. Before this crossed, everything the importer had
                        // learned about spacing and calibration was dropped here.
                        info: Some(StackInfo {
                            name: image.name.clone(),
                            path: Some(p),
                            ..declared.unwrap_or_default()
                        }),
                        image: *image,
                    }),
                    _ => Err(PluginError::failed(format!(
                        "{name} did not return an image"
                    ))),
                }
            }
            abi::FtStatus::Unsupported => Err(PluginError::unsupported(unsafe {
                last_error_of(self.vtable.last_error, "cannot read this file")
            })),
            abi::FtStatus::Panic => Err(PluginError::failed(panic_message(
                &name,
                &self.source,
                // SAFETY: the plugin's `last_error`, copied before returning.
                unsafe { last_error_of(self.vtable.last_error, "") },
            ))),
            other => Err(PluginError::failed(unsafe {
                last_error_of(self.vtable.last_error, &format!("failed ({other:?})"))
            })),
        }
    }
}

impl Exporter for LoadedExporter {
    fn info(&self) -> PluginInfo {
        self.info.clone()
    }

    fn file_types(&self) -> Vec<FileType> {
        self.file_types.clone()
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // As in `Plugin::params`: the trait takes `&dyn`, the C table is
        // uniform over `&mut`, and everything reachable from here is read-only.
        let mut shim = ReadOnly(host);
        let mut cell = HostCell {
            name: host.stack_info().name.clone(),
            path: host.stack_info().path.clone().unwrap_or_default(),
            inner: &mut shim,
        };
        let table = host_table(&mut cell);
        let mut sink = DeclSink::default();
        let sink_table = decl_sink_table(&mut sink);
        // SAFETY: both tables point at locals that outlive the call, and the
        // plugin's own shim catches its panics; `catch_unwind` here is the
        // second line for a plugin that was not built with this crate.
        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.params)(&table, &sink_table)
        }))
        .unwrap_or(abi::FtStatus::Panic);
        if st != abi::FtStatus::Ok {
            return Vec::new();
        }
        sink.decls
    }

    fn export(
        &mut self,
        request: &ExportRequest,
        host: &mut dyn HostContext,
    ) -> Result<(), PluginError> {
        let _lock = CALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let name = self.info.name.clone();
        let p = request.path.to_string_lossy().to_string();
        // Kept alive for the call; see the note in `Plugin::run`.
        let (values, _owned) = values_to_c(&request.params);
        let mut cell = HostCell {
            name: host.stack_info().name.clone(),
            path: host.stack_info().path.clone().unwrap_or_default(),
            inner: host,
        };
        let table = host_table(&mut cell);

        // Nothing comes back but a status: the plugin wrote the file itself.
        let st = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            (self.vtable.export)(
                abi::FtStr::from_str(&p),
                if values.is_empty() {
                    std::ptr::null()
                } else {
                    values.as_ptr()
                },
                values.len() as u64,
                &table,
            )
        }))
        .unwrap_or(abi::FtStatus::Panic);

        match st {
            abi::FtStatus::Ok => Ok(()),
            // The trait has no cancelled outcome, and an export that stopped
            // because the user said so is still an export that did not happen —
            // so it is reported, in the same words a built-in exporter uses.
            abi::FtStatus::Cancelled => Err(PluginError::unsupported("cancelled")),
            abi::FtStatus::Unsupported => Err(PluginError::unsupported(unsafe {
                last_error_of(self.vtable.last_error, "cannot write this stack")
            })),
            abi::FtStatus::OutOfRange => Err(PluginError::OutOfRange(unsafe {
                last_error_of(
                    self.vtable.last_error,
                    "asked for a plane that does not exist",
                )
            })),
            abi::FtStatus::Panic => Err(PluginError::failed(panic_message(
                &name,
                &self.source,
                // SAFETY: the plugin's `last_error`, copied before returning.
                unsafe { last_error_of(self.vtable.last_error, "") },
            ))),
            other => Err(PluginError::failed(unsafe {
                last_error_of(self.vtable.last_error, &format!("failed ({other:?})"))
            })),
        }
    }
}

#[cfg(test)]
#[path = "library_tests.rs"]
mod tests;

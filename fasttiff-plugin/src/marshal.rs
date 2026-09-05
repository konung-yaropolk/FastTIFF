//! Turning Rust plugin types into C vtables, and values back again.
//!
//! The shims here are generic `extern "C"` functions: `params_shim::<Invert>`
//! is a perfectly good `unsafe extern "C" fn` pointer, which is what lets one
//! generic implementation serve every plugin without a macro generating a
//! module per type.
//!
//! **The host copies everything during the registering call.** Descriptors,
//! vtables, strings and planes are all borrowed for the duration of the call
//! that supplies them and copied by the host before it returns. That is what
//! keeps allocation from crossing the boundary, and it means a plugin may build
//! its descriptor on the stack.

use crate::abi::*;
use crate::api::{
    Confidence, ImportHost, ImportRequest, Importer, Outcome, ParamDecl, ParamKind, ParamValue,
    Params, PixelType, PlaneData, Plugin,
};
use crate::{guard, status_of, CHost};

// ---------------------------------------------------------------- registering

/// Register one plugin type with the host.
pub fn register_plugin<T: Plugin + Default + 'static>(reg: &mut FtRegistrar) -> FtStatus {
    let info = T::default().info();
    let vt = FtPluginVtable {
        struct_size: core::mem::size_of::<FtPluginVtable>() as u32,
        _pad: 0,
        params: params_shim::<T>,
        run: run_shim::<T>,
        last_error: crate::last_error_shim,
    };
    let desc = FtPluginDesc {
        struct_size: core::mem::size_of::<FtPluginDesc>() as u32,
        _pad: 0,
        id: FtStr::from_str(&info.id),
        name: FtStr::from_str(&info.name),
        menu_path: FtStr::from_str(&info.menu_path),
        version: FtStr::from_str(&info.version),
        author: FtStr::from_str(&info.author),
        description: FtStr::from_str(&info.description),
        vtable: &vt,
    };
    // SAFETY: `desc` and everything it borrows outlive this call, and the host
    // copies during it.
    unsafe { (reg.add_plugin)(reg.ctx, &desc) }
}

/// Register one importer type with the host.
pub fn register_importer<T: Importer + Default + 'static>(reg: &mut FtRegistrar) -> FtStatus {
    let probe = T::default();
    let info = probe.info();
    let types = probe.file_types();

    // Extensions must stay alive while the descriptor is read, so they are
    // flattened here rather than built inside the loop below.
    let ext_storage: Vec<Vec<FtStr>> = types
        .iter()
        .map(|t| t.extensions.iter().map(|e| FtStr::from_str(e)).collect())
        .collect();
    let c_types: Vec<FtFileType> = types
        .iter()
        .zip(ext_storage.iter())
        .map(|(t, exts)| FtFileType {
            struct_size: core::mem::size_of::<FtFileType>() as u32,
            _pad: 0,
            description: FtStr::from_str(&t.description),
            extensions: exts.as_ptr(),
            extension_count: exts.len() as u64,
        })
        .collect();

    let vt = FtImporterVtable {
        struct_size: core::mem::size_of::<FtImporterVtable>() as u32,
        _pad: 0,
        probe: probe_shim::<T>,
        params: import_params_shim::<T>,
        import: import_shim::<T>,
        last_error: crate::last_error_shim,
    };
    let desc = FtImporterDesc {
        struct_size: core::mem::size_of::<FtImporterDesc>() as u32,
        _pad: 0,
        id: FtStr::from_str(&info.id),
        name: FtStr::from_str(&info.name),
        version: FtStr::from_str(&info.version),
        author: FtStr::from_str(&info.author),
        description: FtStr::from_str(&info.description),
        file_types: c_types.as_ptr(),
        file_type_count: c_types.len() as u64,
        vtable: &vt,
    };
    // SAFETY: as above — everything borrowed here outlives the call.
    unsafe { (reg.add_importer)(reg.ctx, &desc) }
}

// --------------------------------------------------------------------- shims

unsafe extern "C" fn params_shim<T: Plugin + Default>(
    host: *const FtHost,
    sink: *const FtParamSink,
) -> FtStatus {
    guard(|| {
        if sink.is_null() {
            crate::last_error::set("the host passed no dialog sink");
            return FtStatus::BadArgument;
        }
        let h = match CHost::new(host) {
            Ok(h) => h,
            Err(s) => return s,
        };
        let decls = T::default().params(&h);
        push_decls(&*sink, &decls)
    })
}

unsafe extern "C" fn run_shim<T: Plugin + Default>(
    host: *const FtHost,
    values: *const FtValue,
    value_count: u64,
    sink: *const FtSink,
) -> FtStatus {
    guard(|| {
        if sink.is_null() {
            return FtStatus::BadArgument;
        }
        let mut h = match CHost::new(host) {
            Ok(h) => h,
            Err(s) => return s,
        };
        let params = match values_from_c(values, value_count) {
            Some(p) => p,
            None => {
                crate::last_error::set(
                    "the host's dialog values could not be read; it may be built against \
                     a different plugin ABI",
                );
                return FtStatus::BadArgument;
            }
        };
        match T::default().run(&mut h, &params) {
            Ok(o) => write_outcome(&*sink, o),
            Err(e) => status_of(&e),
        }
    })
}

unsafe extern "C" fn probe_shim<T: Importer + Default>(
    path: FtStr,
    head: *const u8,
    head_len: u64,
) -> u32 {
    // Not `guard`: this returns a confidence, not a status, so a panic answers
    // "definitely not mine" — which is the safe direction, because the file
    // then goes to another importer or the built-in reader.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(p) = path.as_str() else {
            return FtConfidence::No.0;
        };
        let bytes: &[u8] = if head.is_null() || head_len == 0 {
            &[]
        } else if head_len > (1 << 24) {
            // An absurd head length is a corrupt call, not a big file.
            return FtConfidence::No.0;
        } else {
            core::slice::from_raw_parts(head, head_len as usize)
        };
        match T::default().probe(std::path::Path::new(p), bytes) {
            Confidence::No => FtConfidence::No.0,
            Confidence::Maybe => FtConfidence::Maybe.0,
            Confidence::Certain => FtConfidence::Certain.0,
        }
    }));
    r.unwrap_or(FtConfidence::No.0)
}

unsafe extern "C" fn import_params_shim<T: Importer + Default>(
    path: FtStr,
    sink: *const FtParamSink,
) -> FtStatus {
    guard(|| {
        if sink.is_null() {
            return FtStatus::BadArgument;
        }
        let Some(p) = path.as_str() else {
            return FtStatus::BadArgument;
        };
        let decls = T::default().params(std::path::Path::new(p));
        push_decls(&*sink, &decls)
    })
}

unsafe extern "C" fn import_shim<T: Importer + Default>(
    path: FtStr,
    values: *const FtValue,
    value_count: u64,
    host: *const FtHost,
    sink: *const FtSink,
) -> FtStatus {
    guard(|| {
        if sink.is_null() {
            return FtStatus::BadArgument;
        }
        let Some(p) = path.as_str() else {
            return FtStatus::BadArgument;
        };
        let params = match values_from_c(values, value_count) {
            Some(v) => v,
            None => return FtStatus::BadArgument,
        };

        // An importer runs before anything is open, so the host table may be
        // absent; progress and logging then go nowhere, which is correct.
        struct Progress(Option<FtHost>);
        impl ImportHost for Progress {
            fn progress(&mut self, f: f32) -> bool {
                match &self.0 {
                    Some(h) => unsafe { (h.progress)(h.ctx, f) != 0 },
                    None => true,
                }
            }
            fn log(&mut self, m: &str) {
                if let Some(h) = &self.0 {
                    unsafe { (h.log)(h.ctx, FtStr::from_str(m)) }
                }
            }
        }
        let mut ph = Progress(if host.is_null() { None } else { Some(*host) });

        let request = ImportRequest {
            path: std::path::PathBuf::from(p),
            params,
        };
        match T::default().import(&request, &mut ph) {
            Ok(r) => {
                let name = r
                    .info
                    .as_ref()
                    .map(|i| i.name.clone())
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| r.image.name.clone());
                // Skipped rather than failed on a host that predates it: an
                // import without its metadata is worth having, and refusing
                // one because the host is a version behind is not.
                if let Some(info) = r.info.as_ref() {
                    if crate::abi::ft_covers!(&*sink as *const FtSink, FtSink, set_info) {
                        let st = write_stack_info(&*sink, info);
                        if st != FtStatus::Ok {
                            return st;
                        }
                    }
                }
                write_image(&*sink, &r.image, &name, FtOutcomeKind::NewDocument, "")
            }
            Err(e) => status_of(&e),
        }
    })
}

// ---------------------------------------------------------------- marshalling

/// Hand the file's spacing, calibration and display mode to the host.
///
/// # Safety
/// `sink` must be a valid `FtSink` for the duration of the call.
unsafe fn write_stack_info(sink: &FtSink, info: &crate::api::StackInfo) -> FtStatus {
    let mut present = 0u32;
    let mut set = |cond: bool, bit: u32| {
        if cond {
            present |= bit;
        }
    };
    set(info.spacing.x.is_some(), FT_HAS_SPACING_X);
    set(info.spacing.y.is_some(), FT_HAS_SPACING_Y);
    set(info.spacing.z.is_some(), FT_HAS_SPACING_Z);
    set(info.frame_interval_s.is_some(), FT_HAS_FRAME_INTERVAL);
    set(info.calibration.is_some(), FT_HAS_CALIBRATION);
    let (c0, c1) = info.calibration.unwrap_or((0.0, 0.0));
    let c = FtStackInfo {
        struct_size: core::mem::size_of::<FtStackInfo>() as u32,
        mode: match info.mode {
            crate::api::DisplayMode::Composite => 1,
            crate::api::DisplayMode::Color => 2,
            _ => 0,
        },
        present,
        _pad: 0,
        spacing_x: info.spacing.x.unwrap_or(0.0),
        spacing_y: info.spacing.y.unwrap_or(0.0),
        spacing_z: info.spacing.z.unwrap_or(0.0),
        frame_interval_s: info.frame_interval_s.unwrap_or(0.0),
        calibration_offset: c0,
        calibration_scale: c1,
    };
    // Both strings are borrowed for the length of this call, which is all the
    // host needs: it copies before returning.
    (sink.set_info)(
        sink.ctx,
        &c,
        info.unit
            .as_deref()
            .map(FtStr::from_str)
            .unwrap_or(FtStr::EMPTY),
        info.description
            .as_deref()
            .map(FtStr::from_str)
            .unwrap_or(FtStr::EMPTY),
    )
}

/// Push a declaration list through the host's sink, one control at a time.
unsafe fn push_decls(sink: &FtParamSink, decls: &[ParamDecl]) -> FtStatus {
    if (sink.struct_size as usize) < core::mem::size_of::<FtParamSink>() {
        crate::last_error::set("the host's dialog sink is older than this plugin's ABI");
        return FtStatus::BadArgument;
    }
    for d in decls {
        // The option strings must outlive the push, so they live here.
        let opts: Vec<FtStr> = match &d.kind {
            ParamKind::Choice { options, .. } => {
                options.iter().map(|o| FtStr::from_str(o)).collect()
            }
            _ => Vec::new(),
        };
        let mut c = FtParamDecl {
            struct_size: core::mem::size_of::<FtParamDecl>() as u32,
            kind: FtParamKind::Label,
            key: FtStr::from_str(&d.key),
            label: FtStr::from_str(&d.label),
            help: d
                .help
                .as_deref()
                .map(FtStr::from_str)
                .unwrap_or(FtStr::EMPTY),
            i_default: 0,
            i_min: 0,
            i_max: 0,
            f_default: 0.0,
            f_min: 0.0,
            f_max: 0.0,
            b_default: 0,
            save: 0,
            s_default: FtStr::EMPTY,
            options: opts.as_ptr(),
            option_count: opts.len() as u64,
        };
        match &d.kind {
            ParamKind::Int { default, min, max } => {
                c.kind = FtParamKind::Int;
                c.i_default = *default;
                c.i_min = *min;
                c.i_max = *max;
            }
            ParamKind::Float { default, min, max } => {
                c.kind = FtParamKind::Float;
                c.f_default = *default;
                c.f_min = *min;
                c.f_max = *max;
            }
            ParamKind::Bool { default } => {
                c.kind = FtParamKind::Bool;
                c.b_default = u32::from(*default);
            }
            ParamKind::Choice { default, .. } => {
                c.kind = FtParamKind::Choice;
                c.i_default = *default as i64;
            }
            ParamKind::Text { default } => {
                c.kind = FtParamKind::Text;
                c.s_default = FtStr::from_str(default);
            }
            ParamKind::Path { default, save } => {
                c.kind = FtParamKind::Path;
                c.s_default = FtStr::from_str(default);
                c.save = u32::from(*save);
            }
            ParamKind::Label => c.kind = FtParamKind::Label,
        }
        let st = (sink.push)(sink.ctx, &c);
        if st != FtStatus::Ok {
            return st;
        }
    }
    FtStatus::Ok
}

/// Read the host's value array back into [`Params`].
///
/// # Safety
/// `values` must point at `count` valid `FtValue`s, or be null with `count` 0.
pub unsafe fn values_from_c(values: *const FtValue, count: u64) -> Option<Params> {
    let mut p = Params::new();
    if count == 0 {
        return Some(p);
    }
    if values.is_null() {
        return None;
    }
    // A count past anything a dialog could hold is a corrupt call.
    if count > 4096 {
        return None;
    }
    for i in 0..count as usize {
        // `FtValue` is arrayed, so its size is frozen and this side's stride is
        // the right one; the per-element check is still what makes forming the
        // reference sound. See `FtValue`'s doc comment.
        let element = values.add(i);
        if !crate::abi::fits(element) {
            return None;
        }
        let v = &*element;
        let key = v.key.as_str()?.to_string();
        let value = match v.kind {
            FtParamKind::Int => ParamValue::Int(v.i),
            FtParamKind::Float => ParamValue::Float(v.f),
            FtParamKind::Bool => ParamValue::Bool(v.b != 0),
            FtParamKind::Choice => ParamValue::Choice(v.i.max(0) as usize),
            FtParamKind::Text => ParamValue::Text(v.s.as_str()?.to_string()),
            FtParamKind::Path => ParamValue::Path(v.s.as_str()?.to_string()),
            FtParamKind::Label => continue,
            // A kind this plugin's ABI does not know: the host is newer than
            // the crate this plugin was built with. Skipping the control is
            // right — the plugin never declared it, so it cannot want it, and
            // failing the whole call over a value it does not use would make
            // every plugin break on a host upgrade.
            _ => continue,
        };
        p.set(key, value);
    }
    Some(p)
}

/// Write an [`Outcome`] into the host's sink.
///
/// # Safety
/// `sink` must be a valid `FtSink` for the duration of the call.
pub unsafe fn write_outcome(sink: &FtSink, outcome: Outcome) -> FtStatus {
    // Only the core is required. Refusing a host that merely predates an
    // optional callback would fail the whole run with nothing to say — which
    // is exactly what it did before this check was narrowed.
    if !crate::abi::covers(sink as *const FtSink, FtSink::CORE) {
        crate::last_error::set("the host's result sink is too small to be a FastTIFF plugin host");
        return FtStatus::BadArgument;
    }
    match outcome {
        Outcome::Nothing => (sink.set_outcome)(sink.ctx, FtOutcomeKind::Nothing, FtStr::EMPTY),
        Outcome::Cancelled => FtStatus::Cancelled,
        Outcome::Message(m) => {
            (sink.set_outcome)(sink.ctx, FtOutcomeKind::Message, FtStr::from_str(&m))
        }
        Outcome::NewDocument(img) => {
            let name = img.name.clone();
            write_image(sink, &img, &name, FtOutcomeKind::NewDocument, "")
        }
        Outcome::SaveToFile { image, path } => {
            let name = image.name.clone();
            write_image(sink, &image, &name, FtOutcomeKind::SaveToFile, &path)
        }
    }
}

/// Declare a result's shape, push every plane, then say what to do with it.
unsafe fn write_image(
    sink: &FtSink,
    img: &crate::api::ImageResult,
    name: &str,
    kind: FtOutcomeKind,
    text: &str,
) -> FtStatus {
    // Validated on this side too, not only the host's: a plugin that miscounts
    // should learn about it from its own error rather than from the host
    // refusing something it cannot explain.
    if let Err(e) = img.validate() {
        return status_of(&e);
    }
    let ty = match img.pixel_type {
        PixelType::U8 => FtPixelType::U8,
        PixelType::U16 => FtPixelType::U16,
        PixelType::I16 => FtPixelType::I16,
        PixelType::F32 => FtPixelType::F32,
    };
    let st = (sink.begin_image)(
        sink.ctx,
        img.width,
        img.height,
        img.channels as u64,
        img.slices as u64,
        img.frames as u64,
        ty,
        FtStr::from_str(name),
    );
    if st != FtStatus::Ok {
        return st;
    }
    // Colours before planes, so a host that refuses one has not yet copied a
    // gigabyte of pixels. Skipped entirely on a host that predates the
    // callback — the image is still right, it just comes out in the host's
    // default colours.
    if crate::abi::ft_covers!(sink as *const FtSink, FtSink, set_channel) {
        for (i, color) in img.channel_colors.iter().enumerate() {
            let rgb = (color[0] as u32) << 16 | (color[1] as u32) << 8 | color[2] as u32;
            let st = (sink.set_channel)(sink.ctx, i as u64, FtStr::EMPTY, rgb);
            if st != FtStatus::Ok {
                return st;
            }
        }
    }
    for p in &img.planes {
        let (ptr, len) = match p {
            PlaneData::U8(v) => (v.as_ptr() as *const core::ffi::c_void, v.len() as u64),
            PlaneData::U16(v) => (v.as_ptr() as *const core::ffi::c_void, v.len() as u64),
            PlaneData::F32(v) => (v.as_ptr() as *const core::ffi::c_void, v.len() as u64),
        };
        let st = (sink.push_plane)(sink.ctx, ptr, len);
        if st != FtStatus::Ok {
            return st;
        }
    }
    (sink.set_outcome)(sink.ctx, kind, FtStr::from_str(text))
}

#[cfg(test)]
#[path = "marshal_tests.rs"]
mod tests;

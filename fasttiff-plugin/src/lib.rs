//! Write a FastTIFF plugin as an ordinary Rust type.
//!
//! ```
//! use fasttiff_plugin::api::*;
//!
//! #[derive(Default)]
//! struct Invert;
//!
//! impl Plugin for Invert {
//!     fn info(&self) -> PluginInfo {
//!         PluginInfo::new("com.example.invert", "Invert").menu_path("Filters")
//!     }
//!
//!     fn run(&mut self, host: &mut dyn HostContext, _p: &Params)
//!         -> Result<Outcome, PluginError>
//!     {
//!         let info = host.image();
//!         let mut buf = Vec::new();
//!         host.read_plane_f32(Plane::new(0, 0, host.view().frame_index), &mut buf)?;
//!
//!         let hi = buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
//!         let out: Vec<f32> = buf.iter().map(|&v| hi - v).collect();
//!
//!         Ok(Outcome::NewDocument(Box::new(ImageResult {
//!             width: info.width,
//!             height: info.height,
//!             channels: 1,
//!             slices: 1,
//!             frames: 1,
//!             pixel_type: PixelType::F32,
//!             planes: vec![PlaneData::F32(out)],
//!             channel_colors: Vec::new(),
//!             metadata: None,
//!             name: format!("{}-inverted", host.stack_info().name),
//!         })))
//!     }
//! }
//!
//! fasttiff_plugin::export_plugin! { plugins: [Invert] }
//! ```
//!
//! This example is compiled and linked by `cargo test`, so the snippet a plugin
//! author copies is one that is known to build.
//!
//! `Cargo.toml` needs `crate-type = ["cdylib"]`, and the resulting
//! `.dll`/`.so`/`.dylib` goes in the plugin folder — the app's
//! **Plugins ▸ Open plugin folder…** shows where that is.
//!
//! # What this crate is for
//!
//! Everything in `fasttiff-plugin-abi` is `extern "C"`, `#[repr(C)]`, raw
//! pointers and manual length checks, because Rust has no stable ABI and a
//! plugin built by a different compiler cannot exchange Rust types with the
//! host. That contract is frozen forever; nobody should have to *write*
//! against it. This crate is the translation, and unlike the ABI crate it may
//! change freely, because it compiles into the plugin rather than sitting
//! between two binaries.
//!
//! It also carries the two obligations an author would otherwise have to
//! remember. Every generated entry point catches unwinding, because a panic
//! crossing an `extern "C"` boundary is undefined behaviour and this workspace
//! builds with `panic = "unwind"`. And no allocation crosses the boundary: the
//! host copies every descriptor, string and plane during the call that supplies
//! it, so the plugin's allocator and the host's never meet.

pub use fasttiff_plugin_abi as abi;
pub use fasttiff_plugin_api as api;

mod host;
pub mod marshal;

pub use host::CHost;
pub use marshal::{register_exporter, register_importer, register_plugin, registrar_of};

/// Remember why the last call failed, for the host to ask about.
///
/// A thread-local, because the ABI's `last_error` takes no instance: the
/// vtable is stateless by design, so there is nothing to hang an error on but
/// the thread that produced it.
pub mod last_error {
    use std::cell::RefCell;

    thread_local! {
        static LAST: RefCell<String> = const { RefCell::new(String::new()) };
    }

    pub fn set(msg: impl Into<String>) {
        LAST.with(|l| *l.borrow_mut() = msg.into());
    }

    /// The stored message, borrowed.
    ///
    /// Valid until the next `set` on this thread. The contract requires the
    /// host to copy it before returning, which it does.
    pub fn get() -> crate::abi::FtStr {
        LAST.with(|l| {
            let b = l.borrow();
            crate::abi::FtStr {
                ptr: b.as_ptr(),
                len: b.len() as u64,
            }
        })
    }
}

/// Run `f`, turning a panic into a status rather than letting it unwind across
/// the boundary.
pub fn guard<F: FnOnce() -> abi::FtStatus>(f: F) -> abi::FtStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(s) => s,
        Err(e) => {
            let what = e
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| e.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked".into());
            // The payload alone: the host adds the plugin's name and the file
            // it came from, which this side does not know and the user needs.
            last_error::set(what);
            abi::FtStatus::Panic
        }
    }
}

/// Map a plugin error onto the ABI's status, recording its text.
///
/// The *message* is stored, not the error's `Display` form. The status already
/// carries the kind, and the host rebuilds the same `PluginError` on its side —
/// so storing `e.to_string()` here would put "not applicable: " in the text and
/// the host would put it there again, giving the user "not applicable: not
/// applicable: this stack has one Z slice". Each side says the part it knows.
pub fn status_of(e: &api::PluginError) -> abi::FtStatus {
    let (status, message) = match e {
        api::PluginError::Unsupported(m) => (abi::FtStatus::Unsupported, m),
        api::PluginError::OutOfRange(m) => (abi::FtStatus::OutOfRange, m),
        api::PluginError::Failed(m) => (abi::FtStatus::Error, m),
    };
    last_error::set(message.clone());
    status
}

/// The `last_error` entry point every generated vtable shares.
///
/// # Safety
/// Called by the host, which copies the result before returning.
pub unsafe extern "C" fn last_error_shim() -> abi::FtStr {
    last_error::get()
}

/// Generate the library's `ft_plugin_v1_query` entry point.
///
/// Every named type must implement the trait for the list it appears in —
/// [`api::Plugin`], [`api::Importer`] or [`api::Exporter`] — and `Default`: the
/// vtable is stateless, so an instance is built per call and dropped inside it,
/// and no opaque handle's lifetime has to be agreed between the two binaries.
///
/// The lists are optional and may be given in any combination, as long as they
/// stay in the order `plugins`, `importers`, `exporters`:
///
/// ```ignore
/// fasttiff_plugin::export_plugin! { plugins: [Invert], exporters: [Csv] }
/// fasttiff_plugin::export_plugin! { exporters: [Csv] }
/// ```
///
/// An exporter registers only on a host new enough to have asked for one. That
/// is not an error: `add_exporter` was appended to the registrar in ABI minor
/// 1, and a library that also carries filters or importers should install those
/// rather than refuse to load because one of the three has nowhere to go.
#[macro_export]
macro_rules! export_plugin {
    (plugins: [$($p:ty),* $(,)?]
     $(, importers: [$($i:ty),* $(,)?])?
     $(, exporters: [$($e:ty),* $(,)?])? $(,)?) => {
        $crate::__export!([$($p),*], [$($($i),*)?], [$($($e),*)?]);
    };
    (importers: [$($i:ty),* $(,)?] $(, exporters: [$($e:ty),* $(,)?])? $(,)?) => {
        $crate::__export!([], [$($i),*], [$($($e),*)?]);
    };
    (exporters: [$($e:ty),* $(,)?] $(,)?) => {
        $crate::__export!([], [], [$($e),*]);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __export {
    ([$($p:ty),*], [$($i:ty),*], [$($e:ty),*]) => {
        /// The one symbol the host looks up. Its name carries the ABI major
        /// version, so a host of a different major simply does not find it.
        ///
        /// The name has to be spelled out — a symbol cannot be built from a
        /// constant — so it is the one place the major version is duplicated.
        /// The assertion below is what stops that duplicate from drifting:
        /// bumping `ABI_MAJOR` without changing this literal would leave every
        /// plugin exporting `v1` while hosts looked for `v2`, and the symptom
        /// would be every plugin in the world silently failing to load.
        const _: () = assert!(
            $crate::abi::is_query_symbol(b"ft_plugin_v1_query\0"),
            "the exported symbol name no longer matches the ABI's QUERY_SYMBOL"
        );

        #[no_mangle]
        pub unsafe extern "C" fn ft_plugin_v1_query(
            reg: *mut $crate::abi::FtRegistrar,
        ) -> $crate::abi::FtStatus {
            $crate::guard(|| {
                use $crate::abi::FtStatus;
                if reg.is_null() {
                    return FtStatus::BadArgument;
                }
                // Never `&mut *reg`: on a host that predates a field this
                // plugin knows about, a reference to the whole struct over a
                // shorter allocation is undefined behaviour on creation. A
                // size check that *refused* such a host would be safe and
                // wrong — it would decline to install filters the host could
                // have run. `registrar_of` copies what is there, stubs what is
                // not, and writes this plugin's minor version back.
                let mut r = match unsafe { $crate::registrar_of(reg) } {
                    Ok(r) => r,
                    Err(st) => return st,
                };
                let r = &mut r;
                $(
                    let st = $crate::register_plugin::<$p>(r);
                    if st != FtStatus::Ok {
                        return st;
                    }
                )*
                $(
                    let st = $crate::register_importer::<$i>(r);
                    if st != FtStatus::Ok {
                        return st;
                    }
                )*
                $(
                    let st = $crate::register_exporter::<$e>(r);
                    if st != FtStatus::Ok {
                        return st;
                    }
                )*
                FtStatus::Ok
            })
        }
    };
}

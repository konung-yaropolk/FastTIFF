//! Making the `windows` crates run on Windows 7.
//!
//! # The problem
//!
//! `combase.dll` does not exist before Windows 8. The COM functions that live
//! in it there live in `ole32.dll` on Windows 7, and on Windows 8+ `ole32`
//! simply forwards to `combase` — so `ole32` is the name that works everywhere.
//!
//! The modern `windows`/`windows-sys` crates import from `combase.dll`, and a
//! static import is resolved by the loader at process start. So a build that
//! imports one single function from `combase.dll` does not fail when that
//! function is called: it fails to start at all, with a dialog naming a DLL the
//! user has never heard of.
//!
//! In this application that is *one* function — `CoTaskMemFree`, used to free
//! the strings the file dialog returns. Every other COM entry point already
//! comes from `ole32.dll`. That was established by reading the import table of
//! the linked binary, and `no_combase_import_survives` re-establishes it — a
//! dependency bump can introduce a second `combase` import at any time, and
//! nothing else would notice until a Windows 7 user could not start the
//! program.
//!
//! # The fix
//!
//! Not a patched dependency, and not a vendored fork — the crates stay exactly
//! as they are. Instead this defines the *import symbol itself*.
//!
//! A `raw-dylib` import (which is how `windows-link` declares these) makes the
//! compiler synthesise a small import library defining `__imp_CoTaskMemFree`,
//! the indirect-call slot the calling code goes through. A linker pulls a
//! member out of a library only when a symbol is still undefined, but it always
//! links every object file it is given outright. Defining `__imp_CoTaskMemFree`
//! here, in the binary crate, therefore satisfies the reference before the
//! import library is ever consulted — and with nothing left to resolve from
//! `combase.dll`, it does not appear in the import table at all.
//!
//! The slot holds a pointer to [`co_task_mem_free`] below, which finds the real
//! function in `ole32.dll` at run time. Resolution is deliberately dynamic:
//! declaring `extern "system" { fn CoTaskMemFree }` against `ole32` would ask
//! the linker for `__imp_CoTaskMemFree` from `ole32`'s import library, which is
//! the very symbol being defined here, and the two would collide.
//!
//! # Only on Windows 7 targets
//!
//! The *symbol* is gated on `cfg(win7)`, which `build.rs` sets when the target
//! triple contains `-win7-`. Every other build keeps the ordinary
//! `combase.dll` import, which is what should happen: this is a workaround for
//! an old platform rather than an improvement, and applying it on Windows 10
//! would add an indirection and a `LoadLibrary` for nothing.
//!
//! The forwarding underneath is compiled on every Windows build and covered by
//! the tests below, so it cannot rot unnoticed while nobody is building for
//! Windows 7.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The address of `ole32!CoTaskMemFree`, resolved once.
///
/// `0` means "not looked up yet"; `1` means "looked up and not there", which is
/// distinct because a failed lookup must not be retried on every free.
static REAL: AtomicUsize = AtomicUsize::new(0);

const NOT_FOUND: usize = 1;

#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

/// Stands in for `combase!CoTaskMemFree`.
///
/// # Safety
/// Called by the `windows` crates in place of the real thing, with the same
/// contract: `pv` is either null or a pointer from the COM task allocator.
#[cfg_attr(not(win7), allow(dead_code))]
unsafe extern "system" fn co_task_mem_free(pv: *mut c_void) {
    let mut addr = REAL.load(Ordering::Relaxed);
    if addr == 0 {
        // `ole32.dll` is already in the process — every other COM entry point
        // is statically imported from it — so this is a refcount bump rather
        // than a load, and a race here is harmless: both threads find the same
        // address and store it.
        let module = LoadLibraryA(c"ole32.dll".as_ptr().cast());
        addr = if module.is_null() {
            NOT_FOUND
        } else {
            match GetProcAddress(module, c"CoTaskMemFree".as_ptr().cast()) {
                p if p.is_null() => NOT_FOUND,
                p => p as usize,
            }
        };
        REAL.store(addr, Ordering::Relaxed);
    }
    if addr != NOT_FOUND {
        let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(addr);
        f(pv);
    }
    // Nothing to do if it could not be found: leaking a dialog's path string is
    // survivable, and there is no way to report an error through this signature
    // anyway. It cannot happen in practice — `CoTaskMemFree` has been exported
    // from `ole32.dll` since it existed.
}

/// The indirect-call slot the `windows` crates go through.
///
/// Spelled with `#[export_name]` rather than `#[no_mangle]` because the symbol
/// differs between architectures: 32-bit x86 decorates `__stdcall` exports with
/// a leading underscore and the argument-byte count, and the name has to match
/// the one the compiler generated for the `raw-dylib` import exactly.
#[cfg(all(win7, target_arch = "x86_64"))]
#[export_name = "__imp_CoTaskMemFree"]
pub static IMP_CO_TASK_MEM_FREE: unsafe extern "system" fn(*mut c_void) = co_task_mem_free;

#[cfg(all(win7, target_arch = "x86"))]
#[export_name = "__imp__CoTaskMemFree@4"]
pub static IMP_CO_TASK_MEM_FREE: unsafe extern "system" fn(*mut c_void) = co_task_mem_free;

/// The DLLs a linked PE imports from, and the functions taken from each.
///
/// A deliberately small reader: enough of the PE format to walk the import
/// directory and no more. Written out rather than pulled in as a dependency
/// because it exists to check one property of one file, and a build dependency
/// that parses executables is a larger thing to own than sixty lines.
#[cfg_attr(not(test), allow(dead_code))]
pub fn pe_imports(exe: &[u8]) -> Option<Vec<(String, Vec<String>)>> {
    let u16_at =
        |o: usize| -> Option<u16> { Some(u16::from_le_bytes(exe.get(o..o + 2)?.try_into().ok()?)) };
    let u32_at =
        |o: usize| -> Option<u32> { Some(u32::from_le_bytes(exe.get(o..o + 4)?.try_into().ok()?)) };
    let u64_at =
        |o: usize| -> Option<u64> { Some(u64::from_le_bytes(exe.get(o..o + 8)?.try_into().ok()?)) };

    let pe = u32_at(0x3c)? as usize;
    if exe.get(pe..pe + 4)? != b"PE\0\0" {
        return None;
    }
    let sections = u16_at(pe + 6)? as usize;
    let opt_size = u16_at(pe + 20)? as usize;
    let opt = pe + 24;
    // 0x20b is PE32+, i.e. 64-bit: the data directories sit further in, and
    // the import lookup table is 64 bits per entry rather than 32.
    let pe32plus = u16_at(opt)? == 0x20b;
    let dirs = opt + if pe32plus { 112 } else { 96 };
    let import_rva = u32_at(dirs + 8)? as usize;

    let table = opt + opt_size;
    let mut secs = Vec::with_capacity(sections);
    for i in 0..sections {
        let h = table + 40 * i;
        secs.push((
            u32_at(h + 12)? as usize, // virtual address
            u32_at(h + 8)? as usize,  // virtual size
            u32_at(h + 20)? as usize, // raw pointer
            u32_at(h + 16)? as usize, // raw size
        ));
    }
    let to_file = |rva: usize| -> Option<usize> {
        secs.iter()
            .find(|(va, vsz, _, rsz)| rva >= *va && rva < va + (*vsz).max(*rsz))
            .map(|(va, _, raw, _)| raw + (rva - va))
    };
    let cstr = |at: usize| -> Option<String> {
        let end = exe[at..].iter().position(|b| *b == 0)? + at;
        Some(String::from_utf8_lossy(&exe[at..end]).into_owned())
    };

    let mut out = Vec::new();
    let mut entry = to_file(import_rva)?;
    loop {
        let lookup = u32_at(entry)? as usize;
        let name_rva = u32_at(entry + 12)? as usize;
        let address = u32_at(entry + 16)? as usize;
        if name_rva == 0 {
            break;
        }
        let dll = cstr(to_file(name_rva)?)?;
        let mut funcs = Vec::new();
        let mut at = to_file(if lookup != 0 { lookup } else { address })?;
        loop {
            let (value, ordinal_bit) = if pe32plus {
                (u64_at(at)?, 1u64 << 63)
            } else {
                (u32_at(at)? as u64, 1u64 << 31)
            };
            if value == 0 {
                break;
            }
            // An ordinal import carries no name to record.
            if value & ordinal_bit == 0 {
                funcs.push(cstr(to_file((value & 0x7fff_ffff) as usize)? + 2)?);
            }
            at += if pe32plus { 8 } else { 4 };
        }
        out.push((dll, funcs));
        entry += 20;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The redirection is only worth anything if `ole32.dll` really exports the
    /// function. It has since Windows 95 — but this is the assumption the whole
    /// workaround rests on, and it costs nothing to check rather than believe.
    #[test]
    fn ole32_exports_the_function_combase_would_have_provided() {
        // SAFETY: both are ordinary Win32 calls with NUL-terminated names.
        let addr = unsafe {
            let m = LoadLibraryA(c"ole32.dll".as_ptr().cast());
            assert!(!m.is_null(), "ole32.dll should always be loadable");
            GetProcAddress(m, c"CoTaskMemFree".as_ptr().cast())
        };
        assert!(
            !addr.is_null(),
            "ole32.dll does not export CoTaskMemFree, so redirecting to it \
             would leave the Windows 7 build calling nothing"
        );
    }

    /// Freeing a null pointer is a documented no-op, which makes it the one call
    /// that can be made safely here — and it exercises the whole path: the
    /// one-time lookup, the cached address, and the call through it.
    #[test]
    fn the_shim_resolves_and_forwards() {
        // SAFETY: `CoTaskMemFree(NULL)` is defined to do nothing.
        unsafe {
            co_task_mem_free(std::ptr::null_mut());
            // Twice, so the cached branch is taken as well as the lookup.
            co_task_mem_free(std::ptr::null_mut());
        }
        let cached = REAL.load(Ordering::Relaxed);
        assert!(
            cached != 0 && cached != NOT_FOUND,
            "the shim did not find ole32!CoTaskMemFree (cached {cached:#x})"
        );
    }

    /// Modules Windows 7 does not have, so a *static* import of any of them
    /// stops the program starting rather than failing when first called.
    const TOO_NEW: &[(&str, &str)] = &[
        (
            "combase.dll",
            "introduced in Windows 8; redirect to ole32.dll the way this module does",
        ),
        (
            "api-ms-win-core-synch-l1-2-0.dll",
            "WaitOnAddress/WakeByAddress are Windows 8+. `std` avoids them on the \
             *-win7-windows-msvc targets, so this appearing means a dependency \
             imports them directly and needs a version that still supports \
             Windows 7",
        ),
    ];

    /// Nothing a Windows 7 machine cannot load may survive into its build.
    ///
    /// Ignored by default because it needs a *linked binary*, and which one
    /// depends on how it was built. Point it at the Windows 7 executable — the
    /// path is read here, in the test process, whose working directory is this
    /// crate rather than the workspace root, hence the `..`:
    ///
    /// ```text
    /// FASTTIFF_EXE=../target/x86_64-win7-windows-msvc/release/FastTIFF.exe \
    ///     cargo test -p FastTIFF --bin FastTIFF -- --ignored no_combase
    /// ```
    ///
    /// This is the check that matters over time. The redirection covers the one
    /// function that is imported today; a dependency bump can add another at
    /// any point, and the symptom is a Windows 7 machine refusing to start the
    /// program with a message about a DLL nobody has heard of.
    #[test]
    #[ignore = "needs a linked binary; set FASTTIFF_EXE"]
    fn no_combase_import_survives() {
        let Ok(path) = std::env::var("FASTTIFF_EXE") else {
            eprintln!("set FASTTIFF_EXE to a linked .exe to run this");
            return;
        };
        let bytes = std::fs::read(&path).expect("read the executable");
        let imports = pe_imports(&bytes).expect("parse the import table");
        assert!(
            !imports.is_empty(),
            "no imports found; is {path} a PE file?"
        );

        let mut problems = Vec::new();
        for (dll, funcs) in &imports {
            if let Some((_, why)) = TOO_NEW.iter().find(|(n, _)| dll.eq_ignore_ascii_case(n)) {
                problems.push(format!("  {dll} — {why}\n    imports: {funcs:?}"));
            }
        }
        assert!(
            problems.is_empty(),
            "{path} imports from modules Windows 7 does not have:\n{}",
            problems.join("\n")
        );

        // Not vacuous: something must have been read, and the COM functions
        // have to be coming from somewhere.
        let ole32 = imports
            .iter()
            .find(|(dll, _)| dll.eq_ignore_ascii_case("ole32.dll"));
        assert!(
            ole32.is_some(),
            "no ole32.dll import either — the import table was probably misread"
        );
        eprintln!(
            "{path}: {} DLLs, all of them present on Windows 7",
            imports.len()
        );
    }
}

use std::path::{Path, PathBuf, MAIN_SEPARATOR};
use std::process::Command;

fn main() {
    // --- Compile the vendored TinyTIFF reader (C) ---
    let vendor = PathBuf::from("vendor/tinytiff");
    let mut build = cc::Build::new();
    build
        .file(vendor.join("tinytiffreader.c"))
        .file(vendor.join("tinytiff_ctools_internal.c"))
        .include(&vendor)
        .opt_level(3)
        .warnings(false); // silence the LARGE_FILE_SUPPORT #warning (gcc/clang)
    // MSVC: `#warning` is a hard error (C1021), so take the large-file branch
    // instead — MSVC has _ftelli64/_fseeki64 and strcpy_s (what upstream
    // TinyTIFF's CMake detects and defines on Windows).
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build.define("HAVE_FTELLI64", None);
        build.define("HAVE_STRCPY_S", None);
    }
    build.compile("tinytiff");

    println!("cargo:rerun-if-changed=vendor/tinytiff");

    // --- libtiff (C), used whenever one can be found ---
    //
    // Not vendored, unlike TinyTIFF next door, and the asymmetry is deliberate.
    // TinyTIFF is two .c files with no codecs and no configuration surface, so a
    // vendored copy *is* TinyTIFF. libtiff is 43 files and 1.5 MB even stripped
    // to its dependency-free core, needs two hand-written config headers that
    // differ per platform, and — the part that decides it — cannot read Deflate
    // or Zstd without zlib and libzstd. Those codecs are `#ifdef`-guarded and
    // wire to `_notConfigured()` when absent, which is 29% of this matrix. A
    // vendored build would be measuring a stripped libtiff nobody runs.
    //
    // So it is found, not shipped. Present by default when the machine has one;
    // quietly absent when it does not, which the run header states either way.
    // `--features libtiff` turns absence into a hard error, for CI that requires
    // the comparison to be complete.
    println!("cargo:rustc-check-cfg=cfg(libtiff)");
    match find_libtiff() {
        Some(found) => {
            found.emit();
            println!("cargo:rustc-cfg=libtiff");
        }
        None if std::env::var_os("CARGO_FEATURE_LIBTIFF").is_some() => panic!("{}", NO_LIBTIFF),
        None => {}
    }

    // --- Report the exact toolchain + library version in the bench header ---
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "rustc (unknown)".into());
    println!("cargo:rustc-env=BENCH_RUSTC_VERSION={rustc_version}");

    // fast-tiff-lib version from the parent crate manifest (path dependency).
    let lib_version = std::fs::read_to_string("../Cargo.toml")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.trim_start().starts_with("version"))
                .and_then(|l| l.split('"').nth(1).map(str::to_owned))
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=FAST_TIFF_LIB_VERSION={lib_version}");
    println!("cargo:rerun-if-changed=../Cargo.toml");
}

/// A libtiff we can link, and how.
struct Found {
    /// Directory to add to the link search path.
    dir: PathBuf,
    /// What to pass to `-l`. Either the plain stem (`tiff`, which the linker
    /// resolves) or a verbatim filename for a GNU-style import library, which
    /// MSVC's linker will not find by stem.
    link: String,
    /// Where the runtime DLL lives, when linking an import library on Windows.
    /// The exe cannot start without it, and copying the one file is not enough:
    /// Windows resolves libtiff's *own* imports from the exe's directory, and an
    /// MSYS2 libtiff pulls in a closure of about a dozen DLLs.
    runtime_bin: Option<PathBuf>,
}

impl Found {
    fn emit(&self) {
        println!("cargo:rustc-link-search=native={}", self.dir.display());
        println!("cargo:rustc-link-lib=dylib{}", self.link);
        if let Some(bin) = &self.runtime_bin {
            // One separator style: probed prefixes are written with forward
            // slashes and joined with the platform separator, which otherwise
            // shows up in a user-facing message as C:/msys64\clang64\lib.
            let tidy = |p: &Path| p.display().to_string().replace(MAIN_SEPARATOR, "/");
            println!(
                "cargo:warning=libtiff linked from {}. Put {} on PATH before running, or the \
                 benchmark will not start (STATUS_DLL_NOT_FOUND).",
                tidy(&self.dir),
                tidy(bin)
            );
        }
    }
}

const NO_LIBTIFF: &str = "
--features libtiff was requested, but no libtiff could be found to link against.

The feature only forces the issue. Without it the benchmark uses a libtiff when
one is present and omits it when not; the run header says which happened.
Any of these makes one present:

   * Linux/macOS   apt install libtiff-dev   /   brew install libtiff
   * MSYS2         pacman -S mingw-w64-clang-x86_64-libtiff
   * vcpkg         vcpkg install tiff:x64-windows-static-md
   * explicit      LIBTIFF_DIR=<prefix>  or  LIBTIFF_LIB_DIR=<dir holding the lib>
";

/// Find a libtiff, most explicit source first.
///
/// Nothing here emits `rerun-if-changed` for a candidate path. Cargo re-runs a
/// build script on every build when told to watch a path that does not exist,
/// and with this crate's `lto = true` that is a full relink each time — even
/// when the script's output is byte-identical. Only the environment variables
/// are watched; installing a libtiff into an already-probed prefix is rare, and
/// `cargo clean -p tiff_read_bench` covers it.
fn find_libtiff() -> Option<Found> {
    println!("cargo:rerun-if-env-changed=LIBTIFF_DIR");
    println!("cargo:rerun-if-env-changed=LIBTIFF_LIB_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=MSYSTEM_PREFIX");

    explicit_libtiff()
        .or_else(pkg_config_libtiff)
        .or_else(vcpkg_libtiff)
        .or_else(prefix_libtiff)
        .or_else(bare_libtiff)
}

fn is_msvc() -> bool {
    std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
}

fn is_windows() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
}

/// Look in one directory for anything that names libtiff.
///
/// Always `dylib`, never `static`: pointed at an MSYS2 prefix a static link
/// would take the MinGW `libtiff.a` and drag its CRT into an MSVC build.
fn probe_dir(dir: &Path, runtime_bin: Option<PathBuf>) -> Option<Found> {
    if !dir.is_dir() {
        return None;
    }
    // A name the linker resolves by itself.
    for name in ["tiff.lib", "libtiff.so", "libtiff.dylib", "libtiff.a"] {
        if dir.join(name).is_file() {
            // The MinGW static archive is not usable from MSVC; keep looking.
            if name == "libtiff.a" && is_msvc() {
                continue;
            }
            return Some(Found { dir: dir.to_path_buf(), link: "=tiff".into(), runtime_bin: None });
        }
    }
    // A GNU-style import library. MSVC's linker will not find this by stem, and
    // whether rustc resolves it on your behalf depends on the rustc version, so
    // name the file verbatim — that works on every version.
    for name in ["libtiff.dll.a", "libtiff-6.dll.a", "libtiff-5.dll.a"] {
        if dir.join(name).is_file() {
            let link =
                if is_msvc() { format!(":+verbatim={name}") } else { "=tiff".to_string() };
            return Some(Found { dir: dir.to_path_buf(), link, runtime_bin });
        }
    }
    None
}

/// `LIBTIFF_LIB_DIR`, or `LIBTIFF_DIR` with the usual `lib/` under it.
fn explicit_libtiff() -> Option<Found> {
    if let Some(dir) = std::env::var_os("LIBTIFF_LIB_DIR") {
        let dir = PathBuf::from(dir);
        // An explicit answer is trusted even when the probe recognises nothing
        // in it: the caller knows something we do not.
        return probe_dir(&dir, None)
            .or(Some(Found { dir, link: "=tiff".into(), runtime_bin: None }));
    }
    let prefix = PathBuf::from(std::env::var_os("LIBTIFF_DIR")?);
    from_prefix(&prefix)
}

/// A `lib/` + `bin/` prefix — the shape every package manager installs into.
fn from_prefix(prefix: &Path) -> Option<Found> {
    let bin = prefix.join("bin");
    let runtime = (is_windows() && bin.is_dir()).then_some(bin);
    probe_dir(&prefix.join("lib"), runtime.clone()).or_else(|| probe_dir(prefix, runtime))
}

/// The prefixes package managers actually use.
///
/// MSYS2 matters most here: it is the common way to have a libtiff on Windows,
/// and its `libtiff-4.pc` reports MSYS-style paths that `link.exe` cannot use —
/// so pkg-config throws the case away rather than handling it. Scanning the
/// prefix directly is the fix.
fn prefix_libtiff() -> Option<Found> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    // Inside an MSYS2 shell these name the active environment exactly.
    for var in ["MSYSTEM_PREFIX", "MINGW_PREFIX"] {
        if let Some(p) = std::env::var_os(var) {
            candidates.push(PathBuf::from(p));
        }
    }
    if is_windows() {
        for root in ["C:/msys64", "C:/msys32", "C:/tools/msys64"] {
            // clang64 first: its import library is the one verified to link
            // under MSVC, and the others differ in CRT.
            for env in ["clang64", "ucrt64", "mingw64", "clangarm64", "mingw32", "clang32"] {
                candidates.push(PathBuf::from(root).join(env));
            }
        }
    } else {
        if let Some(conda) = std::env::var_os("CONDA_PREFIX") {
            candidates.push(PathBuf::from(conda));
        }
        candidates.extend(
            [
                "/opt/homebrew", // Homebrew on Apple Silicon; NOT on ld64's default path
                "/usr/local",    // Homebrew on Intel, and the usual local prefix
                "/opt/local",    // MacPorts
                "/usr",
            ]
            .iter()
            .map(PathBuf::from),
        );
    }

    candidates.iter().find_map(|p| from_prefix(p))
}

/// vcpkg: `VCPKG_ROOT` if set, else the conventional drive-root checkout.
fn vcpkg_libtiff() -> Option<Found> {
    let roots: Vec<PathBuf> = std::env::var_os("VCPKG_ROOT")
        .map(|r| vec![PathBuf::from(r)])
        .unwrap_or_else(|| ["C:/vcpkg", "C:/dev/vcpkg"].iter().map(PathBuf::from).collect());
    // `-static-md` first: static libtiff against the dynamic CRT is how rustc
    // links on MSVC, and it leaves no DLL to find at run time.
    let triplets = ["x64-windows-static-md", "x64-windows", "x64-windows-static"];

    for root in &roots {
        for triplet in triplets {
            if let Some(found) = from_prefix(&root.join("installed").join(triplet)) {
                return Some(found);
            }
        }
    }
    None
}

/// Let the linker resolve `-l tiff` from its own default paths.
///
/// Unix only. On MSVC there are no default library paths worth trying, and the
/// failure would be `LNK1181: cannot open input file 'tiff.lib'` — a message
/// naming neither libtiff nor the thing that asked for it.
fn bare_libtiff() -> Option<Found> {
    (!is_msvc()).then(|| Found {
        dir: PathBuf::from("/usr/lib"),
        link: "=tiff".into(),
        runtime_bin: None,
    })
}

/// pkg-config, which is how it is found on Linux and in a real Unix prefix.
///
/// Skipped on MSVC: a `pkg-config` on PATH there is almost always MSYS2's, and
/// it answers for the MinGW world — MSYS-style paths `link.exe` cannot parse,
/// naming a GNU archive it could not use if it found it. `prefix_libtiff`
/// handles that case properly by scanning the prefix instead.
fn pkg_config_libtiff() -> Option<Found> {
    if is_msvc() {
        return None;
    }
    let out = Command::new("pkg-config").args(["--libs", "libtiff-4"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let flags = String::from_utf8_lossy(&out.stdout);
    let dir = flags
        .split_whitespace()
        .find_map(|t| t.strip_prefix("-L"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/lib"));
    Some(Found { dir, link: "=tiff".into(), runtime_bin: None })
}

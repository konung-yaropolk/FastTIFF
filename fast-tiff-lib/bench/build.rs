use std::path::PathBuf;
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

    // --- Link the system libtiff (C) only when the feature asks for it ---
    // (Linux: libtiff-dev. Prefer pkg-config; fall back to a bare `-l tiff`.)
    if std::env::var_os("CARGO_FEATURE_LIBTIFF").is_some() && pkg_config_libtiff().is_err() {
        println!("cargo:rustc-link-lib=dylib=tiff");
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

fn pkg_config_libtiff() -> Result<(), ()> {
    let out = Command::new("pkg-config")
        .args(["--libs", "libtiff-4"])
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
        return Err(());
    }
    let flags = String::from_utf8_lossy(&out.stdout);
    for tok in flags.split_whitespace() {
        if let Some(lib) = tok.strip_prefix("-l") {
            println!("cargo:rustc-link-lib=dylib={lib}");
        } else if let Some(path) = tok.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        }
    }
    Ok(())
}

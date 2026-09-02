use std::path::{Path, PathBuf};
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
    if is_msvc() {
        build.define("HAVE_FTELLI64", None);
        build.define("HAVE_STRCPY_S", None);
    }
    build.compile("tinytiff");

    println!("cargo:rerun-if-changed=vendor/tinytiff");

    // --- Compile the vendored libtiff (C) ---
    //
    // Vendored rather than found, so every machine measures the same libtiff and
    // nothing has to be installed first. The codecs are what used to make that a
    // bad trade — a libtiff without zlib and libzstd cannot read Deflate or
    // Zstd, and its `#ifdef`-guarded codecs wire to `_notConfigured()` rather
    // than failing loudly — so both are built from source here too, through the
    // `libz-sys` and `zstd-sys` crates. What comes out is a complete libtiff for
    // every codec this benchmark exercises.
    println!("cargo:rustc-check-cfg=cfg(libtiff)");
    let libtiff_version = build_libtiff();
    println!("cargo:rustc-cfg=libtiff");
    println!("cargo:rustc-env=BENCH_LIBTIFF_VERSION={libtiff_version}");

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

fn is_msvc() -> bool {
    std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
}

fn is_windows() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
}

/// Build the vendored libtiff, returning the version it will report.
///
/// libtiff is normally configured by CMake or autotools, which probe the host
/// and write `tiffconf.h`, `tif_config.h` and `tiffvers.h`. Neither build system
/// is available here, so those three are written directly: the probing they do
/// is only interesting for platforms this benchmark does not target, and for the
/// ones it does the answers are known.
fn build_libtiff() -> String {
    let src = PathBuf::from("vendor/libtiff");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is always set"));
    let windows = is_windows();
    let version = libtiff_version(&src);

    write_config_headers(&out, &version, windows);

    let mut build = cc::Build::new();
    // `out` before `src`: the generated headers must win over the `.h.in`
    // templates sitting next to the sources.
    build.include(&out).include(&src).opt_level(3).warnings(false);

    // The codec libraries, from the `links` crates that built them from source.
    // Their headers arrive as `DEP_*_INCLUDE`; cargo has already arranged the
    // linking itself, so nothing needs to be said about that here.
    for var in ["DEP_Z_INCLUDE", "DEP_ZSTD_INCLUDE"] {
        println!("cargo:rerun-if-env-changed={var}");
        if let Some(dir) = std::env::var_os(var) {
            build.include(PathBuf::from(dir));
        }
    }

    for file in LIBTIFF_SOURCES {
        build.file(src.join(file));
    }
    // File I/O: one of these two and never both — they define the same symbols.
    build.file(src.join(if windows { "tif_win32.c" } else { "tif_unix.c" }));

    build.compile("tiff");

    println!("cargo:rerun-if-changed=vendor/libtiff");
    version
}

/// The libtiff sources this build compiles.
///
/// Everything whose dependencies can be satisfied from source — which is
/// everything except the codecs needing a library the benchmark has no use for:
/// JPEG (`tif_jpeg`, `tif_jpeg_12`, `tif_ojpeg`), JBIG, LERC, LZMA and WebP.
/// Their `#ifdef`s are simply left undefined, which is exactly how libtiff
/// builds itself when those libraries are absent.
///
/// `tif_stream.cxx` is the C++ interface and `mkspans.c` a generator for the
/// CCITT tables that are already committed as `tif_fax3sm.c`; neither belongs in
/// the library.
const LIBTIFF_SOURCES: &[&str] = &[
    "tif_aux.c",
    "tif_close.c",
    "tif_codec.c",
    "tif_color.c",
    "tif_compress.c",
    "tif_dir.c",
    "tif_dirinfo.c",
    "tif_dirread.c",
    "tif_dirwrite.c",
    "tif_dumpmode.c",
    "tif_error.c",
    "tif_extension.c",
    "tif_fax3.c",
    "tif_fax3sm.c",
    "tif_flush.c",
    "tif_getimage.c",
    "tif_hash_set.c",
    "tif_luv.c",
    "tif_lzw.c",
    "tif_next.c",
    "tif_open.c",
    "tif_packbits.c",
    "tif_pixarlog.c",
    "tif_predict.c",
    "tif_print.c",
    "tif_read.c",
    "tif_strip.c",
    "tif_swab.c",
    "tif_thunder.c",
    "tif_tile.c",
    "tif_version.c",
    "tif_warning.c",
    "tif_write.c",
    "tif_zip.c",
    "tif_zstd.c",
];

/// The libtiff version, read from the symbol-version map.
///
/// Derived rather than hard-coded, so dropping a newer libtiff into `vendor/`
/// reports the newer version instead of quietly lying about it. Each
/// `LIBTIFF_x.y[.z]` node in the map is a release that added symbols, and the
/// highest is the release these sources belong to.
///
/// Nodes come in both two- and three-component forms — libtiff switched to
/// naming the patch level at 4.6.1 — and both have to parse. Handling only the
/// two-component form does not fail loudly; it silently reports the newest
/// version that happens to have two components, which is how this first claimed
/// to be 4.5 while shipping 4.7.2.
fn libtiff_version(src: &Path) -> String {
    let map = std::fs::read_to_string(src.join("libtiff.map")).unwrap_or_default();
    let highest = map
        .lines()
        .filter_map(|line| line.trim().strip_prefix("LIBTIFF_"))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(parse_version)
        .max();
    match highest {
        // A two-component node names no patch level, and inventing a `.0` would
        // be a claim the sources do not make.
        Some((major, minor, None)) => format!("{major}.{minor}.x (vendored)"),
        Some((major, minor, Some(patch))) => format!("{major}.{minor}.{patch} (vendored)"),
        None => "unknown (vendored)".to_string(),
    }
}

/// `"4.7"` or `"4.7.2"` into a comparable tuple. Anything else is not a version.
fn parse_version(text: &str) -> Option<(u32, u32, Option<u32>)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = match parts.next() {
        Some(p) => Some(p.parse().ok()?),
        None => None,
    };
    // A fourth component means this is not the shape we think it is.
    parts.next().is_none().then_some((major, minor, patch))
}

/// Write the three headers libtiff's build system would otherwise generate.
fn write_config_headers(out: &Path, version: &str, windows: bool) {
    let size_of_size_t =
        if std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("32") { 4 } else { 8 };
    let big_endian = i32::from(std::env::var("CARGO_CFG_TARGET_ENDIAN").as_deref() == Ok("big"));

    // --- tiffconf.h: the public compile-time configuration ---
    //
    // The codec list is the part that matters. Everything enabled is either
    // self-contained in these sources or backed by a library built from source
    // alongside them; nothing is enabled that would degrade to
    // `_notConfigured()` and fail at run time instead of at build time.
    let tiffconf = format!(
        r#"#ifndef _TIFFCONF_
#define _TIFFCONF_

#include <stddef.h>
#include <stdint.h>

#define TIFF_INT8_T signed char
#define TIFF_UINT8_T unsigned char
#define TIFF_INT16_T signed short
#define TIFF_UINT16_T unsigned short
#define TIFF_INT32_T signed int
#define TIFF_UINT32_T unsigned int
#define TIFF_INT64_T int64_t
#define TIFF_UINT64_T uint64_t
/* ptrdiff_t rather than a concrete type: it is signed and pointer-sized on
 * every target here, which is what tmsize_t has to be, and it stays right on
 * LLP64 (Windows) and LP64 (Unix) alike. */
#define TIFF_SSIZE_T ptrdiff_t

#define HAVE_IEEEFP 1
#define HOST_FILLORDER FILLORDER_LSB2MSB
#define HOST_BIGENDIAN {big_endian}

/* Codecs needing nothing beyond these sources. */
#define CCITT_SUPPORT 1
#define LOGLUV_SUPPORT 1
#define LZW_SUPPORT 1
#define NEXT_SUPPORT 1
#define PACKBITS_SUPPORT 1
#define THUNDER_SUPPORT 1
/* Codecs backed by a library built from source next to this one. */
#define ZIP_SUPPORT 1
#define PIXARLOG_SUPPORT 1
#define ZSTD_SUPPORT 1

#define SUBIFD_SUPPORT 1
#define DEFAULT_EXTRASAMPLE_AS_ALPHA 1
#define STRIPCHOP_DEFAULT TIFF_STRIPCHOP
#define CHECK_JPEG_YCBCR_SUBSAMPLING 1

#define COLORIMETRY_SUPPORT
#define YCBCR_SUPPORT
#define CMYK_SUPPORT
#define ICC_SUPPORT
#define PHOTOSHOP_SUPPORT
#define IPTC_SUPPORT

#endif /* _TIFFCONF_ */
"#
    );

    // --- tif_config.h: what the host probe would have found ---
    let platform = if windows {
        "#define HAVE_IO_H 1\n#define USE_WIN32_FILEIO 1\n#define HAVE_SETMODE 1\n"
    } else {
        "#define HAVE_UNISTD_H 1\n#define HAVE_STRINGS_H 1\n#define HAVE_FSEEKO 1\n#define HAVE_MMAP 1\n"
    };
    let tif_config = format!(
        r#"#ifndef TIF_CONFIG_H
#define TIF_CONFIG_H

#include "tiffconf.h"
#include <inttypes.h>

#define HAVE_ASSERT_H 1
#define HAVE_FCNTL_H 1
#define HAVE_SYS_TYPES_H 1
{platform}
#define PACKAGE "tiff"
#define PACKAGE_NAME "LibTIFF Software"
#define PACKAGE_TARNAME "tiff"
#define PACKAGE_VERSION "{version}"
#define PACKAGE_BUGREPORT ""
#define PACKAGE_URL "http://www.simplesystems.org/libtiff/"

#define SIZEOF_SIZE_T {size_of_size_t}
#define WORDS_BIGENDIAN {big_endian}

/* Read a strip in pieces rather than whole, and load the strile arrays lazily.
 * Both are on by default in libtiff's own build, and both matter to a reader
 * benchmark: without the second, opening a many-strip file pays for the whole
 * offset table up front, which is work no reader here is being asked to do. */
#define CHUNKY_STRIP_READ_SUPPORT 1
#define DEFER_STRILE_LOAD 1
#define TIFF_MAX_DIR_COUNT 1048576
#define STRIP_SIZE_DEFAULT 8192

/* Printf formats for tmsize_t, mirroring tif_config.h.cmake.in. */
#if !defined(__MINGW32__)
#  define TIFF_SIZE_FORMAT "zu"
#endif
#if SIZEOF_SIZE_T == 8
#  define TIFF_SSIZE_FORMAT PRId64
#  if defined(__MINGW32__)
#    define TIFF_SIZE_FORMAT PRIu64
#  endif
#elif SIZEOF_SIZE_T == 4
#  define TIFF_SSIZE_FORMAT PRId32
#  if defined(__MINGW32__)
#    define TIFF_SIZE_FORMAT PRIu32
#  endif
#else
#  error "Unsupported size_t size"
#endif

#endif /* TIF_CONFIG_H */
"#
    );

    // --- tiffvers.h: what TIFFGetVersion() reports ---
    let tiffvers = format!(
        "#define TIFFLIB_VERSION_STR \"LIBTIFF, Version {version}\\nCopyright (c) 1988-1996 Sam \
         Leffler\\nCopyright (c) 1991-1996 Silicon Graphics, Inc.\"\n\
         #define TIFFLIB_VERSION 0\n\
         #define TIFFLIB_VERSION_STR_MAJ_MIN_MIC \"{version}\"\n"
    );

    for (name, body) in
        [("tiffconf.h", tiffconf), ("tif_config.h", tif_config), ("tiffvers.h", tiffvers)]
    {
        std::fs::write(out.join(name), body)
            .unwrap_or_else(|e| panic!("writing {name} into OUT_DIR: {e}"));
    }
}

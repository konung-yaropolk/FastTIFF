//! Windows-only build step: embeds the application icon and version-info
//! metadata (read from `[package.metadata.winres]` in Cargo.toml) into the
//! compiled `.exe` as a Win32 resource. On other platforms this is a no-op.

fn main() {
    // `cfg(win7)` for the Tier-3 Windows 7 targets, whose triples are
    // `x86_64-win7-windows-msvc` and `i686-win7-windows-msvc`. There is no
    // built-in cfg for them, and the distinction matters: `src/win7_compat.rs`
    // redirects a COM import that only needs redirecting on a platform without
    // `combase.dll`, and doing it anywhere else would be pure overhead.
    println!("cargo:rustc-check-cfg=cfg(win7)");
    println!("cargo:rerun-if-env-changed=TARGET");
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("-win7-")
    {
        println!("cargo:rustc-cfg=win7");
    }

    // `#[cfg(windows)]` in a build script describes the **host**, because that
    // is what the script itself is compiled for — and so does the
    // `[target.'cfg(windows)'.build-dependencies]` gate that supplies `winres`.
    // Keep it, since it is what decides whether the crate is even linkable
    // here, and ask Cargo about the *target* separately below.
    #[cfg(windows)]
    {
        // Cross-compiling from Windows to wasm still runs this script, and
        // without the target check it would hand a Win32 resource to a
        // non-Windows target and fail — printing
        //     winres: could not embed icon/metadata: Can only compile resource
        //     file when target_env is "gnu" or "msvc"
        // on every `cargo build --target wasm32-unknown-unknown`. Nothing was
        // broken, but a line saying "could not embed" reads like a build error,
        // which is exactly how it was reported. An executable resource is
        // meaningless off Windows, so skip it rather than try and warn.
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if target_os != "windows" || !matches!(target_env.as_str(), "gnu" | "msvc") {
            return;
        }

        // Rebuild the resource if the icon changes.
        println!("cargo:rerun-if-changed=icon/icon.ico");

        // `WindowsResource::new()` automatically reads the string properties
        // from `[package.metadata.winres]`; we only need to point it at the
        // icon. CWD here is the crate root, so the path is relative to it.
        let mut res = winres::WindowsResource::new();
        res.set_icon("icon/icon.ico");
        if let Err(e) = res.compile() {
            // Don't hard-fail the build if the resource compiler isn't
            // available (e.g. a minimal toolchain without the Windows SDK) —
            // the program still builds, just without the embedded icon.
            println!("cargo:warning=winres: could not embed icon/metadata: {e}");
        }
    }
}

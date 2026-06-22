//! Windows-only build step: embeds the application icon and version-info
//! metadata (read from `[package.metadata.winres]` in Cargo.toml) into the
//! compiled `.exe` as a Win32 resource. On other platforms this is a no-op.

fn main() {
    #[cfg(windows)]
    {
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

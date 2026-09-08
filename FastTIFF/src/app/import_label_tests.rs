//! The window title given to a file an importer converted.

use super::*;
use std::path::Path;

/// What the title is for: an OIR opened through the importer says so, keeping
/// the name and the extension it was opened under.
///
/// The path is assembled from components rather than written out. A backslash
/// separates directories on Windows and is an ordinary character in a file
/// name everywhere else, so `E:\data\stack.oir` — which this used to say —
/// names one file on one platform and a single oddly-named file on the other
/// two. It passed where it was written and nowhere else.
#[test]
fn an_imported_file_keeps_its_own_name_and_extension() {
    let path = Path::new("data").join("Alzheimer mRuby Z stack.oir");
    assert_eq!(
        imported_label(&path, "unused"),
        "Alzheimer mRuby Z stack.oir - imported"
    );
}

/// The extension is the point of keeping the full file name rather than the
/// stem: what reaches the viewer is TIFF bytes, and a title reading
/// `stack.tif` would name a file that exists nowhere while hiding which vendor
/// format it actually came from.
#[test]
fn the_extension_is_kept_rather_than_replaced_by_the_tiff_it_became() {
    let label = imported_label(Path::new("/scans/stack.oir"), "unused");
    assert!(label.starts_with("stack.oir"), "{label}");
    assert!(!label.contains(".tif"), "{label}");
}

/// A directory path ends in `..`, which names no file. An importer that
/// synthesises rather than reads has its own name for the result, and that is
/// what should show rather than an empty title.
#[test]
fn a_path_that_names_no_file_falls_back_to_the_plugins_own_name() {
    assert_eq!(
        imported_label(Path::new("/scans/.."), "Synthesised"),
        "Synthesised"
    );
}

/// A name the platform allows but Rust cannot hold as a `str` must still
/// produce a title rather than being dropped or panicking: the user picked
/// this file, so it has to appear. Built through the platform's own `OsString`
/// conversion, because a `Path` made from a `&str` is valid UTF-8 by
/// construction and would test nothing.
#[test]
fn a_name_the_platform_allows_but_rust_cannot_hold_still_produces_a_title() {
    // An unpaired surrogate on Windows, a stray continuation byte elsewhere —
    // both are legal file names and neither is valid Unicode.
    #[cfg(windows)]
    let name: std::ffi::OsString = {
        use std::os::windows::ffi::OsStringExt;
        let mut units: Vec<u16> = "odd".encode_utf16().collect();
        units.push(0xD800);
        units.extend(".oir".encode_utf16());
        std::ffi::OsString::from_wide(&units)
    };
    #[cfg(unix)]
    let name: std::ffi::OsString = {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(b"odd\xFFname.oir".to_vec())
    };

    let label = imported_label(Path::new(&name), "unused");
    assert!(label.ends_with(" - imported"), "{label}");
    assert!(label.starts_with("odd"), "the name was dropped: {label}");
    assert!(label.contains(".oir"), "the extension was dropped: {label}");
    assert!(
        label.contains('\u{FFFD}'),
        "no replacement character, so the lossy path was never taken and this \
         test proves nothing: {label}"
    );
}

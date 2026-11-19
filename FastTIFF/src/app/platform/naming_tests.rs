//! What a download gets called.
//!
//! The browser decides *where* the file goes and we cannot test that; the name
//! is the whole of what this module decides, and it is the thing the user sees
//! in their downloads folder a week later.

use super::suggested_name;

#[test]
fn an_ordinary_name_keeps_its_stem_and_gains_a_tif() {
    assert_eq!(suggested_name("scan.tif"), "scan.tif");
    assert_eq!(suggested_name("scan.tiff"), "scan.tif");
    assert_eq!(suggested_name("scan.lsm"), "scan.tif");
    assert_eq!(suggested_name("scan"), "scan.tif");
}

/// The reason the stem is taken at the *first* dot. `scan.ome.tif` is one file
/// type, not a file called `scan.ome`; saving it should offer `scan.tif`, not
/// put a second extension on a name that already looks like it has one.
#[test]
fn a_multi_part_extension_is_not_mistaken_for_part_of_the_name() {
    assert_eq!(suggested_name("scan.ome.tif"), "scan.tif");
    assert_eq!(suggested_name("scan.ome.tiff"), "scan.tif");
}

/// A browser document arrives under a bare name, but a stack that came from a
/// plugin or a drop may carry a path. The name is what is wanted, never the
/// directories — a `/` in a `download` attribute is ignored by some browsers
/// and turned into `_` by others, and neither is a filename anyone asked for.
#[test]
fn a_path_contributes_only_its_last_component() {
    assert_eq!(suggested_name("/data/2026/scan.tif"), "scan.tif");
    assert_eq!(suggested_name(r"C:\data\scan.tif"), "scan.tif");
    assert_eq!(suggested_name("/data/2026/scan"), "scan.tif");
}

/// Nothing usable still has to produce a filename: a download with an empty
/// `download` attribute is named by the browser, from the blob URL, and comes
/// out as a GUID with no extension.
#[test]
fn a_name_with_nothing_in_it_falls_back() {
    for empty in ["", "   ", ".tif", "/", r"C:\", "..."] {
        assert_eq!(
            suggested_name(empty),
            "stack.tif",
            "{empty:?} should fall back"
        );
    }
}

/// Surrounding space comes from a title, not from the file, and would be kept
/// verbatim by the browser.
#[test]
fn surrounding_space_is_not_part_of_the_name() {
    assert_eq!(suggested_name("  scan.tif  "), "scan.tif");
    assert_eq!(suggested_name(" scan "), "scan.tif");
}

// ------------------------------------- the name a plugin's own output keeps

use super::name_from_path;

/// The plugin's name survives intact, extension and all — the same file the
/// desktop would have written, so one plugin does not produce two results.
#[test]
fn a_plugins_chosen_name_is_kept_whole() {
    assert_eq!(name_from_path("derivative.tif"), "derivative.tif");
    assert_eq!(name_from_path("trace.csv"), "trace.csv");
    assert_eq!(name_from_path("scan.ome.tif"), "scan.ome.tif");
}

/// The directory cannot be honoured by a download and is dropped rather than
/// passed through to be mangled by the `download` attribute.
#[test]
fn the_directory_a_plugin_asked_for_is_dropped() {
    assert_eq!(name_from_path("/tmp/out/derivative.tif"), "derivative.tif");
    assert_eq!(name_from_path(r"C:\work\derivative.tif"), "derivative.tif");
}

#[test]
fn a_path_naming_no_file_still_produces_one() {
    for empty in ["", "   ", "/", r"C:\", "/tmp/", "."] {
        assert_eq!(
            name_from_path(empty),
            "result.tif",
            "{empty:?} should fall back"
        );
    }
}

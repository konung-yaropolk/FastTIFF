//! What [`register`](fasttiff_builtin_plugins::register) installs, as the
//! application's registry holds it.
//!
//! The registry's own bookkeeping — duplicate ids, ranking importers, the
//! dialogs' file types — is unit-tested beside it in the viewer, with plugins
//! of its own. This is the other half: that the set compiled here arrives
//! whole and cleanly, as built-ins, through the C boundary, and that the
//! features decide that set and nothing else does. Gated per feature, unlike
//! the suites beside it, since which plugins are installed is exactly the
//! question.

use fast_tiff_viewer::plugins::{Origin, Registry};
use fasttiff_builtin_plugins::register;
use fasttiff_plugin_api::Plugin;

/// What holds for every configuration: whatever was compiled in registers, and
/// registers cleanly. Which plugins those are is a build-time choice (the
/// `plugin-*` features), so the identity assertions below are tied to the
/// features that put them there rather than to a list this test keeps its own
/// copy of — a copy would only ever go stale.
#[test]
fn the_builtins_are_installed_and_uniquely_identified() {
    let reg = Registry::new(register);
    assert!(
        reg.problems.is_empty(),
        "a clean build registers cleanly: {:?}",
        reg.problems
    );

    let mut ids: Vec<&str> = reg.entries().iter().map(|e| e.info.id.as_str()).collect();
    let before = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), before, "built-in ids must be unique");
    assert!(reg.entries().iter().all(|e| e.origin == Origin::BuiltIn));
    assert!(reg.importers().iter().all(|e| e.origin == Origin::BuiltIn));
    assert!(reg.exporters().iter().all(|e| e.origin == Origin::BuiltIn));

    #[cfg(feature = "builtin-plugins")]
    assert!(
        !reg.is_empty(),
        "a build that asked for the built-in plugins must have them"
    );
    #[cfg(feature = "plugin-image-tools")]
    assert!(reg.find("dev.fasttiff.invert").is_some());
    #[cfg(feature = "plugin-zproject")]
    assert!(reg.find("dev.fasttiff.zproject").is_some());
}

/// The other half of the switch: a plugin left out is *gone*, not merely
/// hidden. Without this, a feature could silently do nothing and every other
/// test here would still pass.
#[test]
fn a_plugin_left_out_of_the_build_is_absent_from_the_registry() {
    let reg = Registry::new(register);
    for (feature_on, id) in [
        (cfg!(feature = "plugin-image-tools"), "dev.fasttiff.invert"),
        (cfg!(feature = "plugin-zproject"), "dev.fasttiff.zproject"),
        (cfg!(feature = "plugin-plot-axis"), "dev.fasttiff.plot-axis"),
        (
            cfg!(feature = "plugin-image-tools"),
            "dev.fasttiff.stacktools.keeper",
        ),
        (
            cfg!(feature = "plugin-suite2p-stabilize"),
            "dev.fasttiff.stabilize",
        ),
        (
            cfg!(feature = "plugin-deconvolve"),
            "dev.fasttiff.deconvolve.run",
        ),
    ] {
        assert_eq!(
            reg.find(id).is_some(),
            feature_on,
            "{id} must be installed exactly when its feature is on"
        );
    }
}

/// The app's own plugins are in the registry through the same boundary as a
/// library's — they simply have no file behind them.
#[test]
fn the_built_ins_come_through_the_boundary_with_no_file_behind_them() {
    let reg = Registry::new(register);
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
    for e in reg.entries() {
        assert_eq!(e.origin, Origin::BuiltIn);
        assert!(e.plugin.source().is_none(), "{} has a source", e.info.id);
        assert_eq!(e.plugin.info(), e.info);
    }
    for e in reg.importers() {
        assert!(e.importer.source().is_none(), "{} has a source", e.info.id);
    }
    for e in reg.exporters() {
        assert!(e.exporter.source().is_none(), "{} has a source", e.info.id);
    }
}

#[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
#[test]
fn the_builtin_importer_is_installed_and_offers_its_file_types() {
    let reg = Registry::new(register);
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
    // OIR is the importer that ships, and the one this checks for.
    assert!(
        reg.importers()
            .iter()
            .any(|e| e.info.id == "dev.fasttiff.oir"),
        "the OIR importer should be installed"
    );
    let types = reg.open_file_types();
    assert!(
        types
            .iter()
            .any(|t| t.extensions.contains(&"oir".to_string())),
        "the Open dialog should learn about .oir: {types:?}"
    );
}

/// The MRI readers, each present exactly when its feature is on and offering
/// the extensions the Open dialog needs.
///
/// The extension list is the part worth asserting rather than the registration:
/// a `.nii.gz` has `gz` for its extension and nothing finer can be declared, so
/// if that one slipped out of the list the importer would still be installed
/// and would simply never be offered the files it exists for.
#[test]
fn the_mri_importers_are_installed_exactly_when_their_features_are() {
    let reg = Registry::new(register);
    for (feature_on, id, exts) in [
        (
            cfg!(all(feature = "plugin-nifti", not(target_arch = "wasm32"))),
            "dev.fasttiff.nifti",
            &["nii", "gz"][..],
        ),
        (
            cfg!(all(feature = "plugin-dicom", not(target_arch = "wasm32"))),
            "dev.fasttiff.dicom",
            &["dcm"][..],
        ),
    ] {
        let entry = reg.importers().iter().find(|e| e.info.id == id);
        assert_eq!(
            entry.is_some(),
            feature_on,
            "{id} must be installed exactly when its feature is on"
        );
        if let Some(entry) = entry {
            let declared: Vec<&str> = entry
                .file_types
                .iter()
                .flat_map(|t| t.extensions.iter().map(String::as_str))
                .collect();
            for want in exts {
                assert!(
                    declared.contains(want),
                    "{id} should offer .{want}: {declared:?}"
                );
            }
            // And the Open dialog learns about them.
            let dialog = reg.open_file_types();
            assert!(
                dialog
                    .iter()
                    .any(|t| exts.iter().any(|e| t.extensions.contains(&e.to_string()))),
                "{id}'s types should reach the Open dialog"
            );
        }
    }
}

/// No built-in claims a TIFF: the host reads those itself, and an importer
/// that asked for `.tif` would be offered every file the app opens.
#[test]
fn no_builtin_importer_claims_a_tiff() {
    let reg = Registry::new(register);
    let claimed = reg.claimed_extensions();
    assert!(!claimed.iter().any(|e| e == "tif" || e == "tiff"));
    assert!(!reg.claims_extension(std::path::Path::new("a.tif")));
}

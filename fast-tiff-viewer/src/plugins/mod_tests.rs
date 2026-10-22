use super::*;
use fasttiff_plugin_api::{Confidence, FileType, HostContext, Outcome, Params, PluginError};

/// A plugin type that only says who it is.
///
/// One *type* per stub rather than one value with fields: a registry entry is
/// a type registered through the C ABI, constructed afresh for every call, so
/// there is no instance to carry a field in. That is the rule every plugin
/// lives by, these included.
macro_rules! stub {
    ($ty:ident, $id:expr, $name:expr, $path:expr) => {
        #[derive(Default)]
        struct $ty;
        impl Plugin for $ty {
            fn info(&self) -> PluginInfo {
                PluginInfo::new($id, $name).menu_path($path)
            }
            fn run(
                &mut self,
                _h: &mut dyn HostContext,
                _p: &Params,
            ) -> Result<Outcome, PluginError> {
                Ok(Outcome::Nothing)
            }
        }
    };
}

stub!(First, "new.id", "First", "");
stub!(Second, "new.id", "Second", "");
stub!(Nameless, "  ", "Nameless", "");
stub!(Alpha, "a.1", "Alpha", "Tools");
stub!(Beta, "a.2", "Beta", "Tools");
stub!(Top, "a.3", "Top", "");

/// What holds for every configuration: whatever was compiled in registers, and
/// registers cleanly. Which plugins those are is a build-time choice (the
/// `plugin-*` features), so the identity assertions below are tied to the
/// features that put them there rather than to a list this test keeps its own
/// copy of — a copy would only ever go stale.
#[test]
fn the_builtins_are_installed_and_uniquely_identified() {
    let reg = Registry::new();
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

    #[cfg(feature = "builtin-plugins")]
    assert!(
        !reg.is_empty(),
        "a build that asked for the built-in plugins must have them"
    );
    #[cfg(feature = "plugin-invert")]
    assert!(reg.find("dev.fasttiff.invert").is_some());
    #[cfg(feature = "plugin-zproject")]
    assert!(reg.find("dev.fasttiff.zproject").is_some());
}

/// The other half of the switch: a plugin left out is *gone*, not merely
/// hidden. Without this, a feature could silently do nothing and every other
/// test here would still pass.
#[test]
fn a_plugin_left_out_of_the_build_is_absent_from_the_registry() {
    let reg = Registry::new();
    for (feature_on, id) in [
        (cfg!(feature = "plugin-invert"), "dev.fasttiff.invert"),
        (cfg!(feature = "plugin-zproject"), "dev.fasttiff.zproject"),
        (cfg!(feature = "plugin-plot-axis"), "dev.fasttiff.plot-axis"),
        (
            cfg!(feature = "plugin-stack-tools"),
            "dev.fasttiff.stacktools.keeper",
        ),
        (cfg!(feature = "plugin-suite2p-stabilize"), "dev.fasttiff.stabilize"),
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

/// Which plugin runs must not depend on directory order, so a duplicate id is
/// reported rather than resolved.
#[test]
fn a_duplicate_id_is_refused_and_reported() {
    let mut reg = Registry::new();
    let n = reg.len();
    assert!(reg.add::<First>(Origin::Library));
    assert_eq!(reg.len(), n + 1);

    assert!(!reg.add::<Second>(Origin::Library));
    assert_eq!(reg.len(), n + 1, "the duplicate must not be installed");
    assert!(
        reg.problems
            .iter()
            .any(|p| p.contains("new.id") && p.contains("Second")),
        "the clash must be reported: {:?}",
        reg.problems
    );
}

#[test]
fn a_plugin_without_an_id_is_refused() {
    let mut reg = Registry::new();
    assert!(!reg.add::<Nameless>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("Nameless")));
}

#[test]
fn grouping_collects_each_menu_path_once_and_covers_everything() {
    let mut reg = Registry::new();
    reg.add::<Alpha>(Origin::Library);
    reg.add::<Beta>(Origin::Library);
    reg.add::<Top>(Origin::Library);

    let groups = reg.grouped();
    let mut paths: Vec<&str> = groups.iter().map(|(p, _)| *p).collect();
    let before = paths.len();
    paths.sort_unstable();
    paths.dedup();
    assert_eq!(paths.len(), before, "each menu path appears exactly once");

    let total: usize = groups.iter().map(|(_, v)| v.len()).sum();
    assert_eq!(
        total,
        reg.len(),
        "grouping must not drop or duplicate an entry"
    );

    let tools = groups
        .iter()
        .find(|(p, _)| *p == "Tools")
        .expect("Tools group");
    assert_eq!(tools.1.len(), 2);
    assert!(groups.iter().any(|(p, _)| p.is_empty()), "top level exists");

    // Every index a group hands back must address the entry it describes.
    for (_, items) in &groups {
        for (i, info) in items {
            assert_eq!(&reg.entries()[*i].info, *info);
        }
    }
}

// ---- importers ----

/// An importer type with a fixed id, extensions and confidence; see [`stub!`]
/// for why each is a type of its own.
macro_rules! stub_importer {
    ($ty:ident, $id:expr, $exts:expr, $conf:expr) => {
        #[derive(Default)]
        struct $ty;
        impl fasttiff_plugin_api::Importer for $ty {
            fn info(&self) -> PluginInfo {
                PluginInfo::new($id, "Stub importer")
            }
            fn file_types(&self) -> Vec<FileType> {
                vec![FileType::new("Stub", $exts)]
            }
            fn probe(&self, _p: &Path, _h: &[u8]) -> Confidence {
                $conf
            }
            fn import(
                &mut self,
                _r: &fasttiff_plugin_api::ImportRequest,
                _h: &mut dyn fasttiff_plugin_api::ImportHost,
            ) -> Result<fasttiff_plugin_api::ImportResult, PluginError> {
                Err(PluginError::failed("stub"))
            }
        }
    };
}

stub_importer!(MaybeZz, "s.maybe", &["zz"], Confidence::Maybe);
stub_importer!(CertainZz, "s.certain", &["zz"], Confidence::Certain);
stub_importer!(NoZz, "s.no", &["zz"], Confidence::No);
stub_importer!(DupQ1, "dup.id", &["q1"], Confidence::Maybe);
stub_importer!(DupQ2, "dup.id", &["q2"], Confidence::Maybe);
stub_importer!(SameD1, "d.1", &["same"], Confidence::Maybe);
stub_importer!(SameD2, "d.2", &["same"], Confidence::Maybe);

#[cfg(all(feature = "plugin-oir", not(target_arch = "wasm32")))]
#[test]
fn the_builtin_importer_is_installed_and_offers_its_file_types() {
    let reg = Registry::new();
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
    // OIR is the importer that ships. Netpbm is a worked example behind an
    // off-by-default feature, so asserting on it here would fail the ordinary
    // build; when it is enabled it must register too, and it is checked below.
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

/// The example importer still registers when it is asked for, so the feature is
/// a switch rather than a way for the file to rot unnoticed.
#[cfg(feature = "plugin-netpbm")]
#[test]
fn the_example_importer_registers_when_enabled() {
    let reg = Registry::new();
    assert!(
        reg.importers()
            .iter()
            .any(|e| e.info.id == "dev.fasttiff.netpbm"),
        "the Netpbm example should be installed under its feature"
    );
    assert!(
        reg.open_file_types()
            .iter()
            .any(|t| t.extensions.contains(&"pgm".to_string())),
        "the Open dialog should learn about .pgm"
    );
}

#[test]
fn importers_are_ranked_by_confidence_not_by_name() {
    let mut reg = Registry::new();
    reg.add_importer::<MaybeZz>(Origin::Library);
    reg.add_importer::<CertainZz>(Origin::Library);

    let ranked = reg.importers_for(Path::new("f.zz"), b"whatever");
    assert!(ranked.len() >= 2);
    assert_eq!(
        reg.importers()[ranked[0].0].info.id,
        "s.certain",
        "the confident importer must be offered first"
    );
    // And one that declines is not offered at all.
    reg.add_importer::<NoZz>(Origin::Library);
    let ranked = reg.importers_for(Path::new("f.zz"), b"whatever");
    assert!(
        !ranked
            .iter()
            .any(|(i, _)| reg.importers()[*i].info.id == "s.no"),
        "an importer answering No must not be offered"
    );
}

/// An importer with no file types could never be reached, so installing it
/// silently would be a trap.
#[test]
fn an_importer_declaring_no_file_types_is_refused() {
    #[derive(Default)]
    struct Empty;
    impl fasttiff_plugin_api::Importer for Empty {
        fn info(&self) -> PluginInfo {
            PluginInfo::new("empty.one", "Empty")
        }
        fn file_types(&self) -> Vec<FileType> {
            Vec::new()
        }
        fn import(
            &mut self,
            _r: &fasttiff_plugin_api::ImportRequest,
            _h: &mut dyn fasttiff_plugin_api::ImportHost,
        ) -> Result<fasttiff_plugin_api::ImportResult, PluginError> {
            Err(PluginError::failed("empty"))
        }
    }
    let mut reg = Registry::new();
    assert!(!reg.add_importer::<Empty>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("no file types")));
}

#[test]
fn duplicate_importer_ids_are_refused() {
    let mut reg = Registry::new();
    assert!(reg.add_importer::<DupQ1>(Origin::Library));
    assert!(!reg.add_importer::<DupQ2>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("dup.id")));
}

/// A file dialog must not show the same format twice.
#[test]
fn open_file_types_are_deduplicated() {
    let mut reg = Registry::new();
    reg.add_importer::<SameD1>(Origin::Library);
    reg.add_importer::<SameD2>(Origin::Library);
    let types = reg.open_file_types();
    let n = types
        .iter()
        .filter(|t| t.extensions == vec!["same".to_string()])
        .count();
    assert_eq!(n, 1, "one row per format: {types:?}");
}

/// Reading the head of a file that does not exist must not fail the caller —
/// the probes treat an empty head as "no evidence".
#[test]
fn reading_the_head_of_a_missing_file_yields_nothing() {
    assert!(Registry::read_head(Path::new("/definitely/not/here.zz")).is_empty());
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
    let reg = Registry::new();
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

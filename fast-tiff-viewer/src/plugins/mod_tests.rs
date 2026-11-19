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

/// The built-ins arrive by being handed to the registry: what `new`'s argument
/// registers is installed, through the C boundary, as `BuiltIn` — and sorted
/// into menu order, whatever order it was registered in.
#[test]
fn what_the_application_hands_in_is_installed_as_built_in_and_sorted() {
    use fasttiff_plugin::Registrar;
    let reg = Registry::new(|r| {
        r.plugin::<Top>();
        r.plugin::<Beta>();
        r.plugin::<Alpha>();
        r.importer::<SameD1>();
        r.exporter::<StubCsv>();
    });
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
    let names: Vec<&str> = reg.entries().iter().map(|e| e.info.name.as_str()).collect();
    assert_eq!(
        names,
        ["Top", "Alpha", "Beta"],
        "menu order: path, then name"
    );
    assert!(reg.entries().iter().all(|e| e.origin == Origin::BuiltIn));
    assert!(reg.entries().iter().all(|e| e.plugin.source().is_none()));
    assert_eq!(reg.importers().len(), 1);
    assert_eq!(reg.importers()[0].origin, Origin::BuiltIn);
    assert!(reg.importers()[0].importer.source().is_none());
    assert_eq!(reg.exporters().len(), 1);
    assert_eq!(reg.exporters()[0].origin, Origin::BuiltIn);
}

/// The registry knows no plugin by name: with nothing handed in, nothing is
/// installed, and nothing is wrong.
#[test]
fn with_no_built_ins_handed_in_the_registry_is_empty() {
    let reg = Registry::new(|_| {});
    assert!(reg.is_empty());
    assert!(reg.importers().is_empty() && reg.exporters().is_empty());
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
}

/// A built-in the registry refuses is reported, and the others are still
/// installed: each is registered on its own, so one bad one costs only itself.
#[test]
fn a_refused_built_in_is_reported_and_the_rest_installed() {
    use fasttiff_plugin::Registrar;
    let reg = Registry::new(|r| {
        r.plugin::<First>();
        r.plugin::<Second>();
        r.plugin::<Alpha>();
    });
    assert_eq!(reg.len(), 2, "the duplicate is refused, the others kept");
    assert!(reg.find("a.1").is_some());
    assert!(
        reg.problems
            .iter()
            .any(|p| p.contains("new.id") && p.contains("Second")),
        "{:?}",
        reg.problems
    );
}

/// Which plugin runs must not depend on directory order, so a duplicate id is
/// reported rather than resolved.
#[test]
fn a_duplicate_id_is_refused_and_reported() {
    let mut reg = Registry::empty();
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
    let mut reg = Registry::empty();
    assert!(!reg.add::<Nameless>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("Nameless")));
}

#[test]
fn grouping_collects_each_menu_path_once_and_covers_everything() {
    let mut reg = Registry::empty();
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

/// An exporter that only says what it writes.
#[derive(Default)]
struct StubCsv;
impl fasttiff_plugin_api::Exporter for StubCsv {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("s.csv", "Stub exporter")
    }
    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Stub", &["csv"])]
    }
    fn export(
        &mut self,
        _r: &fasttiff_plugin_api::ExportRequest,
        _h: &mut dyn HostContext,
    ) -> Result<(), PluginError> {
        Err(PluginError::failed("stub"))
    }
}

#[test]
fn importers_are_ranked_by_confidence_not_by_name() {
    let mut reg = Registry::empty();
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
    let mut reg = Registry::empty();
    assert!(!reg.add_importer::<Empty>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("no file types")));
}

#[test]
fn duplicate_importer_ids_are_refused() {
    let mut reg = Registry::empty();
    assert!(reg.add_importer::<DupQ1>(Origin::Library));
    assert!(!reg.add_importer::<DupQ2>(Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("dup.id")));
}

/// A file dialog must not show the same format twice.
#[test]
fn open_file_types_are_deduplicated() {
    let mut reg = Registry::empty();
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

/// The claimed extensions are what `claims_extension` decides by — each once,
/// sorted — so a host holding the list while the registry is away decides the
/// same way. A TIFF is never among them unless an importer asks for it.
#[test]
fn the_claimed_extensions_agree_with_claims_extension() {
    let mut reg = Registry::empty();
    reg.add_importer::<MaybeZz>(Origin::Library);
    reg.add_importer::<SameD1>(Origin::Library);
    reg.add_importer::<SameD2>(Origin::Library);
    reg.add_importer::<DupQ1>(Origin::Library);
    let claimed = reg.claimed_extensions();
    assert_eq!(claimed, ["q1", "same", "zz"]);
    let mut sorted = claimed.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(claimed, sorted, "sorted, each once");
    for ext in &claimed {
        assert!(
            reg.claims_extension(std::path::Path::new(&format!("a.{ext}"))),
            "{ext}"
        );
    }
    assert!(!claimed.iter().any(|e| e == "tif" || e == "tiff"));
    assert!(!reg.claims_extension(std::path::Path::new("a.tif")));
}

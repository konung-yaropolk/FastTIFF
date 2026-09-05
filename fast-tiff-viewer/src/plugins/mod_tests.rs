use super::*;
use fasttiff_plugin_api::{Confidence, FileType, HostContext, Outcome, Params, PluginError};

struct Stub(&'static str, &'static str, &'static str);

impl Plugin for Stub {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(self.0, self.1).menu_path(self.2)
    }
    fn run(&mut self, _h: &mut dyn HostContext, _p: &Params) -> Result<Outcome, PluginError> {
        Ok(Outcome::Nothing)
    }
}

#[test]
fn the_builtins_are_installed_and_uniquely_identified() {
    let reg = Registry::new();
    assert!(!reg.is_empty(), "the built-in plugins must be present");
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
    assert!(reg.find("dev.fasttiff.invert").is_some());
    assert!(reg.find("dev.fasttiff.zproject").is_some());
    assert!(reg.entries().iter().all(|e| e.origin == Origin::BuiltIn));
}

/// Which plugin runs must not depend on directory order, so a duplicate id is
/// reported rather than resolved.
#[test]
fn a_duplicate_id_is_refused_and_reported() {
    let mut reg = Registry::new();
    let n = reg.len();
    assert!(reg.add(Box::new(Stub("new.id", "First", "")), Origin::Library));
    assert_eq!(reg.len(), n + 1);

    assert!(!reg.add(Box::new(Stub("new.id", "Second", "")), Origin::Library));
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
    assert!(!reg.add(Box::new(Stub("  ", "Nameless", "")), Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("Nameless")));
}

#[test]
fn grouping_collects_each_menu_path_once_and_covers_everything() {
    let mut reg = Registry::new();
    reg.add(Box::new(Stub("a.1", "Alpha", "Tools")), Origin::Library);
    reg.add(Box::new(Stub("a.2", "Beta", "Tools")), Origin::Library);
    reg.add(Box::new(Stub("a.3", "Top", "")), Origin::Library);

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

struct StubImporter(&'static str, &'static [&'static str], Confidence);

impl fasttiff_plugin_api::Importer for StubImporter {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(self.0, "Stub importer")
    }
    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Stub", self.1)]
    }
    fn probe(&self, _p: &Path, _h: &[u8]) -> Confidence {
        self.2
    }
    fn import(
        &mut self,
        _r: &fasttiff_plugin_api::ImportRequest,
        _h: &mut dyn fasttiff_plugin_api::ImportHost,
    ) -> Result<fasttiff_plugin_api::ImportResult, PluginError> {
        Err(PluginError::failed("stub"))
    }
}

#[test]
fn the_builtin_importer_is_installed_and_offers_its_file_types() {
    let reg = Registry::new();
    assert!(reg.problems.is_empty(), "{:?}", reg.problems);
    assert!(
        reg.importers()
            .iter()
            .any(|e| e.info.id == "dev.fasttiff.netpbm"),
        "the Netpbm importer should be installed"
    );
    let types = reg.open_file_types();
    assert!(
        types
            .iter()
            .any(|t| t.extensions.contains(&"pgm".to_string())),
        "the Open dialog should learn about .pgm: {types:?}"
    );
    assert!(reg.claims_extension(Path::new("x.ppm")));
    assert!(
        !reg.claims_extension(Path::new("x.tif")),
        "TIFF is the app's own job"
    );
}

/// The signature decides which importer runs, not the extension — otherwise two
/// plugins claiming `.tif` could never be told apart.
#[test]
fn importers_are_ranked_by_confidence_not_by_name() {
    let mut reg = Registry::new();
    reg.add_importer(
        Box::new(StubImporter("s.maybe", &["zz"], Confidence::Maybe)),
        Origin::Library,
    );
    reg.add_importer(
        Box::new(StubImporter("s.certain", &["zz"], Confidence::Certain)),
        Origin::Library,
    );

    let ranked = reg.importers_for(Path::new("f.zz"), b"whatever");
    assert!(ranked.len() >= 2);
    assert_eq!(
        reg.importers()[ranked[0].0].info.id,
        "s.certain",
        "the confident importer must be offered first"
    );
    // And one that declines is not offered at all.
    reg.add_importer(
        Box::new(StubImporter("s.no", &["zz"], Confidence::No)),
        Origin::Library,
    );
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
    assert!(!reg.add_importer(Box::new(Empty), Origin::Library));
    assert!(reg.problems.iter().any(|p| p.contains("no file types")));
}

#[test]
fn duplicate_importer_ids_are_refused() {
    let mut reg = Registry::new();
    assert!(reg.add_importer(
        Box::new(StubImporter("dup.id", &["q1"], Confidence::Maybe)),
        Origin::Library
    ));
    assert!(!reg.add_importer(
        Box::new(StubImporter("dup.id", &["q2"], Confidence::Maybe)),
        Origin::Library
    ));
    assert!(reg.problems.iter().any(|p| p.contains("dup.id")));
}

/// A file dialog must not show the same format twice.
#[test]
fn open_file_types_are_deduplicated() {
    let mut reg = Registry::new();
    reg.add_importer(
        Box::new(StubImporter("d.1", &["same"], Confidence::Maybe)),
        Origin::Library,
    );
    reg.add_importer(
        Box::new(StubImporter("d.2", &["same"], Confidence::Maybe)),
        Origin::Library,
    );
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

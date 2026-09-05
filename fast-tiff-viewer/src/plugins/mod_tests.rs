use super::*;
use fasttiff_plugin_api::{HostContext, Outcome, Params, PluginError};

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

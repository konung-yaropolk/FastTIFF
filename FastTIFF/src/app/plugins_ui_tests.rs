use super::*;

fn info(name: &str, path: &str) -> PluginInfo {
    PluginInfo::new(format!("test.{name}"), name).menu_path(path)
}

/// What the menu shows, as `(submenu, [entry names])`.
fn layout<'a>(groups: &[(&'a str, Vec<Item<'a>>)]) -> Vec<(&'a str, Vec<String>)> {
    groups
        .iter()
        .map(|(path, items)| (*path, items.iter().map(|i| i.name().to_string()).collect()))
        .collect()
}

/// With no plugin under "Analysis" the counter gets a submenu of its own, in
/// the place the registry's sort would have put it.
#[test]
fn the_point_counter_makes_its_submenu_in_sorted_place() {
    let (top, filter, stack) = (
        info("Invert", ""),
        info("Gauss", "Filters"),
        info("Z", "Stack"),
    );
    let groups = vec![
        ("", vec![(0, &top)]),
        ("Filters", vec![(1, &filter)]),
        ("Stack", vec![(2, &stack)]),
    ];
    assert_eq!(
        layout(&with_app_tools(groups)),
        vec![
            ("", vec!["Invert".to_string()]),
            ("Analysis", vec!["Point counter".to_string()]),
            ("Filters", vec!["Gauss".to_string()]),
            ("Stack", vec!["Z".to_string()]),
        ]
    );
}

/// A plugin that chose "Analysis" shares the submenu, sorted by name with the
/// counter, and keeps its index for the run.
#[test]
fn the_point_counter_joins_the_analysis_plugins_in_name_order() {
    let (a, z) = (info("Derivatives", "Analysis"), info("Profile", "Analysis"));
    let groups = vec![("Analysis", vec![(4, &a), (7, &z)])];
    let merged = with_app_tools(groups);
    assert_eq!(
        layout(&merged),
        vec![(
            "Analysis",
            vec![
                "Derivatives".to_string(),
                "Point counter".to_string(),
                "Profile".to_string(),
            ]
        )]
    );
    let indices: Vec<Option<usize>> = merged[0]
        .1
        .iter()
        .map(|item| match item {
            Item::Plugin(i, _) => Some(*i),
            Item::PointCounter => None,
        })
        .collect();
    assert_eq!(indices, vec![Some(4), None, Some(7)]);
}

/// No plugins at all — `plugins-none`, nothing installed — still lists it.
#[test]
fn the_point_counter_is_listed_with_no_plugins() {
    assert_eq!(
        layout(&with_app_tools(Vec::new())),
        vec![("Analysis", vec!["Point counter".to_string()])]
    );
}

/// The real registry, whatever it holds in this build: the counter appears
/// exactly once, and every plugin is still there.
#[test]
fn every_plugin_is_still_listed_beside_the_counter() {
    let registry = Registry::new();
    let merged = with_app_tools(registry.grouped());
    let all: Vec<&Item> = merged.iter().flat_map(|(_, items)| items).collect();
    let counters = all
        .iter()
        .filter(|i| matches!(i, Item::PointCounter))
        .count();
    assert_eq!(counters, 1);
    assert_eq!(all.len(), registry.len() + 1);
    let paths: Vec<&str> = merged.iter().map(|(p, _)| *p).collect();
    let mut sorted = paths.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(paths, sorted, "submenus in order, each once");
}

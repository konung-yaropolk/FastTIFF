use super::Cached;
use std::cell::Cell;

/// Worked out once per key, however often it is asked for.
#[test]
fn a_value_is_worked_out_once_per_key() {
    let made = Cell::new(0);
    let mut title: Cached<u64, String> = Cached::default();
    let mut ask = |key: u64| {
        title
            .get(key, || {
                made.set(made.get() + 1);
                format!("document {key}")
            })
            .clone()
    };
    assert_eq!(ask(1), "document 1");
    assert_eq!(ask(1), "document 1");
    assert_eq!(made.get(), 1, "asked again under the same key");
    assert_eq!(ask(2), "document 2");
    assert_eq!(made.get(), 2, "a new key is worked out afresh");
    // And going back is a new key too: only the last one is kept.
    assert_eq!(ask(1), "document 1");
    assert_eq!(made.get(), 3);
}

/// What is sent somewhere is news only when the key moved *and* the value
/// with it.
#[test]
fn a_value_is_news_only_when_it_changed() {
    let mut title: Cached<u64, String> = Cached::default();
    assert_eq!(
        title.changed(0, || "FastTIFF".into()).map(String::as_str),
        Some("FastTIFF"),
        "the first value is news"
    );
    assert_eq!(title.changed(0, || unreachable!("same key")), None);
    // A failed open moves the key and leaves the same document up.
    assert_eq!(title.changed(1, || "FastTIFF".into()), None);
    assert_eq!(
        title.changed(2, || "a.tif".into()).map(String::as_str),
        Some("a.tif")
    );
    assert_eq!(title.get(2, || unreachable!("held")), "a.tif");
}

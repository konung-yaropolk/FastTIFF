//! The edge choice, read back by what the dialog shows.

use super::*;
use fasttiff_plugin_api::ParamValue;

/// Every option reads back as the edge handling its label names. Looked up by
/// label, not by position: the list's order is the dialog's to choose, and a
/// test that hard-codes positions cannot see a label and its meaning come
/// apart — which is how picking "Crop" once ran the raw output.
#[test]
fn each_option_is_what_its_label_says() {
    let ParamKind::Choice { default, options } = edges_param("raw").kind else {
        panic!("the edge choice is a choice");
    };
    let named = |label: &str| options.iter().position(|o| o == label).expect(label);
    for (label, want) in [
        ("Fill with 0", Edges::Zero),
        ("Edge replicate", Edges::Replicate),
        ("Algorithm's raw output", Edges::Raw),
        ("Crop to the common area", Edges::Crop),
    ] {
        let mut p = Params::new();
        p.set(EDGES, ParamValue::Choice(named(label)));
        assert_eq!(edges_from(&p), want, "{label}");
    }
    assert_eq!(options.len(), 4, "an option no test names: {options:?}");
    // An untouched dialog fills with 0.
    assert_eq!(options[default], "Fill with 0");
    assert_eq!(edges_from(&Params::new()), Edges::Zero);
}

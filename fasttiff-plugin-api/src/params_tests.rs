use super::*;

fn decls() -> Vec<ParamDecl> {
    vec![
        ParamDecl::new(
            "radius",
            "Radius",
            ParamKind::Int {
                default: 3,
                min: 1,
                max: 64,
            },
        ),
        ParamDecl::new(
            "sigma",
            "Sigma",
            ParamKind::Float {
                default: 1.5,
                min: 0.1,
                max: 10.0,
            },
        ),
        ParamDecl::new("norm", "Normalise", ParamKind::Bool { default: true }),
        ParamDecl::new(
            "mode",
            "Mode",
            ParamKind::Choice {
                default: 1,
                options: vec!["Mean".into(), "Max".into()],
            },
        ),
        ParamDecl::new(
            "tag",
            "Tag",
            ParamKind::Text {
                default: "out".into(),
            },
        ),
        ParamDecl::new("note", "", ParamKind::Label),
    ]
}

#[test]
fn defaults_come_from_the_declarations() {
    let p = Params::defaults(&decls());
    assert_eq!(p.int("radius", 0), 3);
    assert_eq!(p.float("sigma", 0.0), 1.5);
    assert!(p.bool("norm", false));
    assert_eq!(p.choice("mode", 0), 1);
    assert_eq!(p.text("tag", ""), "out");
    // A Label is not a value and must not appear.
    assert!(p.get("note").is_none());
}

/// The host clamps before the plugin runs, so a plugin never has to defend
/// against a radius of -1 it already said was impossible.
#[test]
fn clamp_forces_values_into_their_declared_range() {
    let mut p = Params::new();
    p.set("radius", ParamValue::Int(9999));
    p.set("sigma", ParamValue::Float(-5.0));
    let c = p.clamp_to(&decls());
    assert_eq!(c.int("radius", 0), 64);
    assert_eq!(c.float("sigma", 0.0), 0.1);
}

/// NaN survives `f64::clamp` as NaN, which would reach the plugin as a value it
/// was promised could not occur. The declared default is the only sane answer.
#[test]
fn clamp_replaces_nan_with_the_default() {
    let mut p = Params::new();
    p.set("sigma", ParamValue::Float(f64::NAN));
    assert_eq!(p.clamp_to(&decls()).float("sigma", 0.0), 1.5);
}

/// A choice index past the end of the option list must not become an
/// out-of-bounds index in the host's renderer.
#[test]
fn clamp_bounds_a_choice_to_its_options() {
    let mut p = Params::new();
    p.set("mode", ParamValue::Choice(50));
    assert_eq!(p.clamp_to(&decls()).choice("mode", 0), 1);

    // An empty option list is a plugin bug; it must still not panic.
    let empty = vec![ParamDecl::new(
        "x",
        "X",
        ParamKind::Choice {
            default: 3,
            options: Vec::new(),
        },
    )];
    let mut q = Params::new();
    q.set("x", ParamValue::Choice(7));
    assert_eq!(q.clamp_to(&empty).choice("x", 99), 0);
}

/// A value of the wrong variant means the two sides disagree about the dialog.
/// Falling back to the declared default keeps the run going; silently coercing
/// a `Text` into an `Int` would hide the disagreement.
#[test]
fn clamp_replaces_a_wrongly_typed_value_with_the_default() {
    let mut p = Params::new();
    p.set("radius", ParamValue::Text("not a number".into()));
    assert_eq!(p.clamp_to(&decls()).int("radius", 0), 3);
}

/// Keys the plugin never declared are dropped rather than passed through, so a
/// stale saved dialog cannot smuggle a value into a later version.
#[test]
fn clamp_drops_undeclared_keys() {
    let mut p = Params::defaults(&decls());
    p.set("removed_in_v2", ParamValue::Int(1));
    let c = p.clamp_to(&decls());
    assert!(c.get("removed_in_v2").is_none());
    assert_eq!(c.int("radius", 0), 3, "declared keys survive");
}

/// Reading a key that was never set, or reading it as the wrong type, yields
/// the caller's default instead of panicking.
#[test]
fn accessors_fall_back_rather_than_panic() {
    let p = Params::defaults(&decls());
    assert_eq!(p.int("nonexistent", 42), 42);
    assert_eq!(p.float("norm", 7.0), 7.0, "a Bool is not a Float");
    assert_eq!(p.text("radius", "fallback"), "fallback");
}

/// Clamping is idempotent — the host may do it more than once.
#[test]
fn clamp_is_idempotent() {
    let mut p = Params::new();
    p.set("radius", ParamValue::Int(500));
    let once = p.clamp_to(&decls());
    let twice = once.clamp_to(&decls());
    assert_eq!(once, twice);
}

/// A declaration whose min and max are the wrong way round is a plugin bug that
/// must not become a `clamp` panic in the host.
#[test]
fn clamp_survives_an_inverted_range() {
    let bad = vec![
        ParamDecl::new(
            "i",
            "I",
            ParamKind::Int {
                default: 5,
                min: 10,
                max: 0,
            },
        ),
        ParamDecl::new(
            "f",
            "F",
            ParamKind::Float {
                default: 5.0,
                min: 10.0,
                max: 0.0,
            },
        ),
    ];
    let mut p = Params::new();
    p.set("i", ParamValue::Int(7));
    p.set("f", ParamValue::Float(7.0));
    let c = p.clamp_to(&bad);
    assert!((0..=10).contains(&c.int("i", -1)));
    assert!((0.0..=10.0).contains(&c.float("f", -1.0)));
}

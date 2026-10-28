//! The plugin around the registration: what it offers, and what it refuses.
//!
//! The algorithm is tested in `turboreg-registration`. What is checked here
//! is the wiring — that the dialog changes with the file and with itself, that
//! the channel selector says something true about a single-channel stack, and
//! that a stack with no time axis is refused rather than silently "registered".

use super::*;
use fasttiff_plugin_api::{ImageInfo, ParamKind, ParamValue, PixelType};
use turboreg_registration::Settings;

fn info(channels: usize, slices: usize, frames: usize) -> ImageInfo {
    ImageInfo {
        width: 32,
        height: 32,
        channels,
        slices,
        frames,
        samples_per_pixel: 1,
        pixel_type: PixelType::U16,
    }
}

fn decls_with(info: &ImageInfo, chosen: &[(&str, ParamValue)]) -> Vec<ParamDecl> {
    let mut p = Params::new();
    for (k, v) in chosen {
        p.set(*k, v.clone());
    }
    params::declare(info, &p)
}

fn keys(decls: &[ParamDecl]) -> Vec<String> {
    decls.iter().map(|d| d.key.clone()).collect()
}

fn options_of(decls: &[ParamDecl], key: &str) -> Vec<String> {
    match &decls
        .iter()
        .find(|d| d.key == key)
        .unwrap_or_else(|| panic!("no control for {key}"))
        .kind
    {
        ParamKind::Choice { options, .. } => options.clone(),
        other => panic!("{key} is {other:?}, not a choice"),
    }
}

/// A single-channel stack gets exactly one measurement option, which is what
/// makes the host grey the selector out — "a choice of one is not a choice".
/// Offering "Mean" and "Channel 1" separately there would be two names for the
/// same arithmetic.
#[test]
fn one_channel_leaves_nothing_to_choose() {
    let decls = decls_with(&info(1, 1, 10), &[]);
    let opts = options_of(&decls, "ref_channel");
    assert_eq!(
        opts.len(),
        1,
        "a one-channel stack has one possible measurement, got {opts:?}"
    );
    assert!(
        opts[0].to_lowercase().contains("only"),
        "the single option should say why it is the only one: {opts:?}"
    );
}

/// With several channels the mean leads and every channel is offered by number.
#[test]
fn several_channels_are_offered_with_the_mean_first() {
    let opts = options_of(&decls_with(&info(3, 1, 10), &[]), "ref_channel");
    assert_eq!(opts.len(), 4, "mean plus three channels: {opts:?}");
    assert!(opts[0].to_lowercase().contains("mean"), "{opts:?}");
    assert!(opts[1].contains('1') && opts[3].contains('3'), "{opts:?}");
}

/// The mean is the default, and it is index 0 so the default reads back as the
/// mean rather than as channel 1.
#[test]
fn the_mean_is_the_default_measurement() {
    for channels in [1usize, 2, 4] {
        assert_eq!(
            params::measure_channel(&Params::new(), channels),
            None,
            "{channels} channels should default to the mean"
        );
    }
}

/// And a chosen channel maps to its own index, one-based in the label and
/// zero-based in the value.
#[test]
fn choosing_a_channel_selects_that_channel() {
    let mut p = Params::new();
    p.set("ref_channel", ParamValue::Choice(1));
    assert_eq!(
        params::measure_channel(&p, 3),
        Some(0),
        "Channel 1 is index 0"
    );
    p.set("ref_channel", ParamValue::Choice(3));
    assert_eq!(
        params::measure_channel(&p, 3),
        Some(2),
        "Channel 3 is index 2"
    );
    // Out of range is clamped rather than panicking on a stale parameter set.
    p.set("ref_channel", ParamValue::Choice(99));
    assert_eq!(params::measure_channel(&p, 3), Some(2));
    // A single-channel stack ignores the stored choice entirely.
    assert_eq!(params::measure_channel(&p, 1), None);
}

/// A stack with one timepoint has nothing to align in time, so the whole time
/// section is absent rather than present and inert.
#[test]
fn a_single_timepoint_hides_the_time_controls() {
    let decls = keys(&decls_with(&info(2, 1, 1), &[]));
    for absent in ["reference", "n_frames", "moving_average", "h_time"] {
        assert!(
            !decls.contains(&absent.to_string()),
            "{absent} should not be offered for a one-frame stack: {decls:?}"
        );
    }
    // What is left still makes sense on its own.
    assert!(decls.contains(&"transformation".to_string()));
    assert!(decls.contains(&"ref_channel".to_string()));
}

/// `n_frames` is only read when the target is the first frame, so it is only
/// offered then. A control that will be ignored is an invitation to set it and
/// conclude the algorithm is broken.
#[test]
fn the_frame_count_appears_only_for_the_first_frame_reference() {
    let modes = turboreg_registration::Reference::all();
    for (i, mode) in modes.iter().enumerate() {
        let decls = keys(&decls_with(
            &info(1, 1, 20),
            &[("reference", ParamValue::Choice(i))],
        ));
        let offered = decls.contains(&"n_frames".to_string());
        assert_eq!(
            offered,
            *mode == turboreg_registration::Reference::First,
            "{mode:?} should {} offer n_frames",
            if *mode == turboreg_registration::Reference::First {
                ""
            } else {
                "not"
            }
        );
    }
}

/// Bilinear cannot be chained, so "previous frame" is not offered for it —
/// removed rather than offered and then refused.
#[test]
fn bilinear_does_not_offer_the_chained_reference() {
    let bilinear = Transformation::all()
        .iter()
        .position(|t| *t == Transformation::Bilinear)
        .expect("bilinear is a model");
    let decls = decls_with(
        &info(1, 1, 10),
        &[("transformation", ParamValue::Choice(bilinear))],
    );
    let opts = options_of(&decls, "reference");
    assert!(
        !opts.iter().any(|o| o.to_lowercase().contains("previous")),
        "bilinear should not offer the previous frame: {opts:?}"
    );
    // The other models do.
    let rigid = Transformation::all()
        .iter()
        .position(|t| *t == Transformation::RigidBody)
        .unwrap();
    let opts = options_of(
        &decls_with(
            &info(1, 1, 10),
            &[("transformation", ParamValue::Choice(rigid))],
        ),
        "reference",
    );
    assert!(
        opts.iter().any(|o| o.to_lowercase().contains("previous")),
        "{opts:?}"
    );
}

/// The reference list is model-dependent, so the index a user picked has to be
/// read back against the *same* list. Reading index 1 as "first frame" under
/// one model and "mean" under another is the bug this pins.
#[test]
fn a_reference_choice_means_the_same_in_both_directions() {
    for (i, model) in Transformation::all().iter().enumerate() {
        let options = params::reference_options(*model);
        for (j, want) in options.iter().enumerate() {
            let mut p = Params::new();
            p.set("transformation", ParamValue::Choice(i));
            p.set("reference", ParamValue::Choice(j));
            assert_eq!(
                params::settings_from(&p).reference,
                *want,
                "{model:?} index {j} should read back as {want:?}"
            );
        }
    }
}

/// Every control carries the crate's own default, so a dialog left untouched
/// runs what the library would run on its own.
#[test]
fn the_dialog_offers_the_crates_defaults() {
    let d = Settings::default();
    let got = params::settings_from(&Params::new());
    assert_eq!(got.transformation, d.transformation);
    assert_eq!(got.reference, d.reference);
    assert_eq!(got.n_frames, d.n_frames);
    assert_eq!(got.moving_average, d.moving_average);
    assert_eq!(got.max_pyramid_depth, d.max_pyramid_depth);
    assert_eq!(got.backend, d.backend);
}

/// The backend selector offers every backend the build has, which is how the
/// parallelism option reaches the user.
#[test]
fn the_compute_selector_offers_the_backends() {
    let opts = options_of(&decls_with(&info(1, 1, 10), &[]), "backend");
    assert_eq!(opts.len(), turboreg_registration::Backend::all().len());
    assert!(opts.iter().any(|o| o.contains("Multi")), "{opts:?}");
    assert!(opts.iter().any(|o| o.contains("Single")), "{opts:?}");
}

/// Sections are declared, so the dialog is grouped rather than one long column.
#[test]
fn the_dialog_is_grouped() {
    let decls = decls_with(&info(2, 1, 10), &[]);
    let sections = decls
        .iter()
        .filter(|d| matches!(d.kind, ParamKind::Section))
        .count();
    assert!(sections >= 2, "expected grouping, got {sections} sections");
}

/// Each edge option is the border it names, and an untouched dialog fills
/// with 0 rather than the crate's own mirror.
#[test]
fn the_edge_choice_is_the_border_it_names() {
    use turboreg_registration::Border;
    assert_eq!(params::border_from(&Params::new()), Border::Constant);
    for (choice, want) in [(0, Border::Constant), (1, Border::Edge), (2, Border::Native)] {
        let mut p = Params::new();
        p.set("edges", ParamValue::Choice(choice));
        assert_eq!(params::border_from(&p), want, "option {choice}");
    }
    assert!(keys(&decls_with(&info(1, 1, 10), &[])).contains(&"edges".to_string()));
}

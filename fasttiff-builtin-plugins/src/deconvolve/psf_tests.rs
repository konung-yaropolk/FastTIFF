//! The Generate PSF dialog and what it produces.

use super::super::tests::{keys, settle, TestHost};
use super::*;
use fasttiff_plugin_api::{ParamValue, Spacing};

fn params(host: &mut TestHost, overrides: &[(&str, ParamValue)]) -> Params {
    settle(&GeneratePsf, host, overrides).1
}

fn declared(host: &mut TestHost, overrides: &[(&str, ParamValue)]) -> Vec<String> {
    keys(&settle(&GeneratePsf, host, overrides).0)
}

/// The Gaussian model, which is the fast one, with a small grid.
fn quick(host: &mut TestHost, extra: &[(&str, ParamValue)]) -> Params {
    let mut o: Vec<(&str, ParamValue)> = vec![
        ("model", ParamValue::Choice(2)), // Gaussian
        ("width", ParamValue::Int(9)),
        ("height", ParamValue::Int(9)),
        ("slices", ParamValue::Int(5)),
    ];
    o.extend(extra.iter().cloned());
    params(host, &o)
}

fn image_of(outcome: Outcome) -> Box<ImageResult> {
    match outcome {
        Outcome::NewDocument(i) | Outcome::ReplaceDocument(i) => i,
        other => panic!("expected an image, got {other:?}"),
    }
}

// ----------------------------------------------------------------- the dialog

/// Every key `run` reads must be one `params` declares for *some* model, or
/// the control is unreachable and the plugin silently uses its default. The
/// host drops undeclared keys in `clamp_to`, so a key that is never declared
/// is invisible at run time and total in effect.
#[test]
fn every_key_the_run_reads_is_reachable_from_some_model() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let mut all: Vec<String> = Vec::new();
    for m in 0..Model::ALL.len() {
        all.extend(declared(&mut host, &[("model", ParamValue::Choice(m))]));
    }
    for key in [
        "model",
        "mode",
        "na",
        "wavelength",
        "ni",
        "sa",
        "ns",
        "depth",
        "ng",
        "tg",
        "tg0",
        "ti0",
        "pixel",
        "step",
        "width",
        "height",
        "slices",
        "normalise",
        "new_window",
    ] {
        assert!(
            all.iter().any(|d| d == key),
            "{key} is read but no model declares it"
        );
    }
}

/// The point of the dynamic dialog: a model is asked only for what it uses.
#[test]
fn each_model_is_asked_only_for_what_it_uses() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let gl_only = ["ns", "depth", "ng", "tg", "tg0", "ti0"];

    let gl = declared(&mut host, &[("model", ParamValue::Choice(1))]);
    for key in gl_only {
        assert!(gl.iter().any(|d| d == key), "Gibson & Lanni wants {key}");
    }
    assert!(gl.iter().any(|d| d == "sa"), "and it has a wavefront");

    // Every other model: none of the index-mismatch terms, because none of
    // them has anywhere to put one.
    for m in [0usize, 2, 3] {
        let d = declared(&mut host, &[("model", ParamValue::Choice(m))]);
        for key in gl_only {
            assert!(
                !d.iter().any(|x| x == key),
                "model {m} should not ask for {key}"
            );
        }
    }

    // The geometric model has no diffraction, so no wavelength and no
    // wavefront to aberrate.
    let defocus = declared(&mut host, &[("model", ParamValue::Choice(3))]);
    assert!(!defocus.iter().any(|d| d == "wavelength"));
    assert!(!defocus.iter().any(|d| d == "sa"));
    // And the Gaussian has a wavelength (it sizes itself from one) but no
    // wavefront either.
    let gauss = declared(&mut host, &[("model", ParamValue::Choice(2))]);
    assert!(gauss.iter().any(|d| d == "wavelength"));
    assert!(!gauss.iter().any(|d| d == "sa"));
}

/// The dialog is grouped, and the groups are sections rather than labels —
/// which is what makes the host draw a rule rather than a line of text.
#[test]
fn the_dialog_is_grouped_into_sections() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let (decls, _) = settle(&GeneratePsf, &mut host, &[]);
    let sections: Vec<&str> = decls
        .iter()
        .filter(|d| d.kind == ParamKind::Section)
        .map(|d| d.label.as_str())
        .collect();
    assert!(sections.len() >= 4, "{sections:?}");
    assert_eq!(sections[0], "Model");
    // A heading with nothing under it is worse than no heading: the first
    // declaration after each one has to be a control.
    for (i, d) in decls.iter().enumerate() {
        if d.kind == ParamKind::Section {
            match decls.get(i + 1) {
                Some(next) => assert_ne!(
                    next.kind,
                    ParamKind::Section,
                    "the {:?} heading has nothing under it",
                    d.label
                ),
                None => panic!("the dialog ends on the {:?} heading", d.label),
            }
        }
    }
}

/// A value set for a control, lost when the model changes and found again
/// when it changes back. Going to look at another model must not silently
/// reset what was typed.
#[test]
fn a_setting_survives_a_trip_through_another_model() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    // Settle on Gibson & Lanni with a specimen index nobody would default to.
    let (_, values) = settle(
        &GeneratePsf,
        &mut host,
        &[
            ("model", ParamValue::Choice(1)),
            ("ns", ParamValue::Float(1.41)),
        ],
    );
    assert_eq!(values.float("ns", 0.0), 1.41);

    // Now the host loop, by hand: switch to Gaussian, which drops `ns`, then
    // back. What the user set has to come back with it.
    let mut live = values.clone();
    for model in [2usize, 1] {
        live.set("model", ParamValue::Choice(model));
        host.pending = live.clone();
        let decls = GeneratePsf.params(&host);
        for (k, v) in Params::defaults(&decls).iter() {
            if live.get(k).is_none() {
                live.set(k, v.clone());
            }
        }
    }
    host.pending = Params::new();
    assert_eq!(
        live.float("ns", 0.0),
        1.41,
        "the specimen index was reset by a visit to another model"
    );
}

#[test]
fn the_model_and_mode_lists_are_the_ones_the_code_knows() {
    let host = TestHost::new(8, 8, 1, 1, 1);
    let decls = GeneratePsf.params(&host);
    let find = |k: &str| {
        decls
            .iter()
            .find(|d| d.key == k)
            .map(|d| d.kind.clone())
            .expect("declared")
    };
    match find("model") {
        ParamKind::Choice { options, .. } => {
            assert_eq!(options.len(), Model::ALL.len());
            assert_eq!(options[0], Model::ALL[0].label());
        }
        other => panic!("{other:?}"),
    }
    match find("mode") {
        ParamKind::Choice { options, .. } => assert_eq!(options.len(), Mode::ALL.len()),
        other => panic!("{other:?}"),
    }
}

/// The sampling defaults are the whole reason this plugin looks at the host.
#[test]
fn the_sampling_defaults_come_from_the_open_stack() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    host.info.unit = Some("micron".into());
    host.info.spacing = Spacing {
        x: Some(0.065),
        y: Some(0.065),
        z: Some(0.3),
    };
    let decls = GeneratePsf.params(&host);
    let default_of = |k: &str| match decls.iter().find(|d| d.key == k).map(|d| d.kind.clone()) {
        Some(ParamKind::Float { default, .. }) => default,
        other => panic!("{k}: {other:?}"),
    };
    assert!((default_of("pixel") - 65.0).abs() < 1e-9);
    assert!((default_of("step") - 300.0).abs() < 1e-9);
}

/// A file calibrated in something else is left alone rather than silently
/// treated as microns — a stack in inches would otherwise produce a PSF
/// 25,400 times the right size, and the dialog would look reasonable.
#[test]
fn a_stack_calibrated_in_something_else_does_not_seed_the_defaults() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    host.info.unit = Some("inch".into());
    host.info.spacing = Spacing {
        x: Some(0.065),
        y: Some(0.065),
        z: Some(0.3),
    };
    let decls = GeneratePsf.params(&host);
    match decls
        .iter()
        .find(|d| d.key == "pixel")
        .map(|d| d.kind.clone())
    {
        Some(ParamKind::Float { default, .. }) => assert_eq!(default, 100.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_flat_stack_defaults_to_a_flat_psf() {
    let flat = TestHost::new(8, 8, 1, 1, 1);
    let volume = TestHost::new(8, 8, 1, 12, 1);
    let slices = |h: &TestHost| match GeneratePsf
        .params(h)
        .iter()
        .find(|d| d.key == "slices")
        .map(|d| d.kind.clone())
    {
        Some(ParamKind::Int { default, .. }) => default,
        other => panic!("{other:?}"),
    };
    assert_eq!(slices(&flat), 1);
    assert!(slices(&volume) > 1);
}

// -------------------------------------------------------------------- the run

#[test]
fn it_produces_a_float_volume_of_the_asked_for_shape() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(&mut host, &[]);
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    assert_eq!((image.width, image.height), (9, 9));
    assert_eq!((image.channels, image.slices, image.frames), (1, 5, 1));
    assert_eq!(image.pixel_type, PixelType::F32);
    assert_eq!(image.planes.len(), 5);
    image
        .validate()
        .expect("the result must be self-consistent");
}

/// The default normalisation is the one a deconvolution needs: convolving by
/// a kernel that sums to 1 leaves total intensity alone.
#[test]
fn the_default_normalisation_sums_to_one() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(&mut host, &[]);
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    let total: f64 = image
        .planes
        .iter()
        .map(|p| match p {
            PlaneData::F32(v) => v.iter().map(|&x| x as f64).sum::<f64>(),
            other => panic!("{:?}", other.pixel_type()),
        })
        .sum();
    assert!((total - 1.0).abs() < 1e-5, "sums to {total}");
}

#[test]
fn the_other_normalisations_do_what_they_say() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);

    let p = quick(&mut host, &[("normalise", ParamValue::Choice(1))]);
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    let peak = image
        .planes
        .iter()
        .map(|p| match p {
            PlaneData::F32(v) => v.iter().copied().fold(0.0f32, f32::max),
            other => panic!("{:?}", other.pixel_type()),
        })
        .fold(0.0f32, f32::max);
    assert!((peak - 1.0).abs() < 1e-6, "maximum is {peak}");

    // Raw: the Gaussian model's own peak is exactly 1 at the centre, so the
    // distinguishing property is the *sum*, which is not 1.
    let p = quick(&mut host, &[("normalise", ParamValue::Choice(2))]);
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    let total: f64 = image
        .planes
        .iter()
        .map(|p| match p {
            PlaneData::F32(v) => v.iter().map(|&x| x as f64).sum::<f64>(),
            other => panic!("{:?}", other.pixel_type()),
        })
        .sum();
    assert!(total > 1.5, "raw values should not be normalised: {total}");
}

/// The result carries its own calibration, in microns, so that deconvolving
/// with it can check the sampling matches.
#[test]
fn the_result_is_calibrated_in_microns() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(
        &mut host,
        &[
            ("pixel", ParamValue::Float(80.0)),
            ("step", ParamValue::Float(400.0)),
        ],
    );
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    let meta = image.metadata.as_ref().expect("a PSF describes itself");
    assert_eq!(meta.unit.as_deref(), Some("micron"));
    assert_eq!(meta.spacing.x, Some(0.08));
    assert_eq!(meta.spacing.z, Some(0.4));
    assert_eq!(meta.description, None, "it must not carry the source's");
}

#[test]
fn an_aperture_larger_than_its_medium_is_refused_with_the_reason() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(
        &mut host,
        &[
            ("na", ParamValue::Float(1.6)),
            ("ni", ParamValue::Float(1.33)),
        ],
    );
    let err = GeneratePsf
        .run(&mut host, &p)
        .expect_err("impossible optics");
    assert!(matches!(err, PluginError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("sin"), "{err}");
}

#[test]
fn undersampling_is_mentioned_before_the_work_starts() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(
        &mut host,
        &[
            ("pixel", ParamValue::Float(400.0)),
            ("na", ParamValue::Float(1.4)),
            ("wavelength", ParamValue::Float(520.0)),
        ],
    );
    GeneratePsf.run(&mut host, &p).expect("run");
    assert!(host.said("Nyquist"), "{:?}", host.logged);
}

#[test]
fn the_checkbox_decides_where_the_result_goes() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(&mut host, &[("new_window", ParamValue::Bool(true))]);
    assert!(matches!(
        GeneratePsf.run(&mut host, &p).expect("run"),
        Outcome::NewDocument(_)
    ));
    let p = quick(&mut host, &[("new_window", ParamValue::Bool(false))]);
    assert!(matches!(
        GeneratePsf.run(&mut host, &p).expect("run"),
        Outcome::ReplaceDocument(_)
    ));
}

#[test]
fn cancelling_is_not_an_error() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    host.cancel_after = Some(0);
    let p = quick(&mut host, &[("model", ParamValue::Choice(0))]);
    assert_eq!(
        GeneratePsf
            .run(&mut host, &p)
            .expect("cancel is not failure"),
        Outcome::Cancelled
    );
}

#[test]
fn the_window_is_named_after_the_model_and_the_wavelength() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = quick(&mut host, &[("wavelength", ParamValue::Float(488.0))]);
    let image = image_of(GeneratePsf.run(&mut host, &p).expect("run"));
    assert!(image.name.contains("gaussian"), "{}", image.name);
    assert!(image.name.contains("488"), "{}", image.name);
}

#[test]
fn it_lives_in_the_deconvolution_menu() {
    let info = GeneratePsf.info();
    assert_eq!(info.menu_path, "Deconvolution");
    assert_eq!(info.id, "dev.fasttiff.deconvolve.psf");
}

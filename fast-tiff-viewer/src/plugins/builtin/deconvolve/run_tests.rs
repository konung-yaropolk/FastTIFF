//! The Deconvolve dialog, and the whole plugin end to end.

use super::super::tests::{keys, settle, write_f32_tiff, TestHost};
use super::*;
use fasttiff_plugin_api::{ParamValue, PixelType};

fn params(host: &mut TestHost, overrides: &[(&str, ParamValue)]) -> Params {
    settle(&Deconvolve, host, overrides).1
}

fn declared(host: &mut TestHost, overrides: &[(&str, ParamValue)]) -> Vec<String> {
    keys(&settle(&Deconvolve, host, overrides).0)
}

fn image_of(outcome: Outcome) -> Box<ImageResult> {
    match outcome {
        Outcome::NewDocument(i) | Outcome::ReplaceDocument(i) => i,
        other => panic!("expected an image, got {other:?}"),
    }
}

fn floats(p: &PlaneData) -> &[f32] {
    match p {
        PlaneData::F32(v) => v,
        other => panic!("expected float samples, got {:?}", other.pixel_type()),
    }
}

/// A 1-pixel PSF written to a TIFF: the identity, so anything that comes back
/// changed came back changed for a reason other than the blur.
fn delta_psf_file(name: &str) -> String {
    write_f32_tiff(name, 1, 1, &[vec![1.0]])
        .display()
        .to_string()
}

// ----------------------------------------------------------------- the dialog

/// Every key `run` reads has to be reachable from some combination of the
/// choices, or the control is unreachable and the plugin silently uses its
/// default. `clamp_to` drops undeclared keys, so a key no selection declares
/// is invisible at run time and total in effect.
#[test]
fn every_key_the_run_reads_is_reachable_from_some_choice() {
    // Several frames, so the timepoint selector is on offer.
    let mut host = TestHost::new(8, 8, 1, 4, 3);
    let mut all: Vec<String> = Vec::new();
    for source in 0..2 {
        for m in 0..Method::ALL.len() {
            all.extend(declared(
                &mut host,
                &[
                    ("psf_source", ParamValue::Choice(source)),
                    ("method", ParamValue::Choice(m)),
                ],
            ));
        }
    }
    for key in [
        "psf_source",
        "psf_path",
        "sigma_xy",
        "sigma_z",
        "psf_origin",
        "normalise_psf",
        "method",
        "iterations",
        "lambda",
        "gamma",
        "step",
        "threshold",
        "low_pass",
        "stop_delta",
        "nonneg",
        "dimensionality",
        "boundary",
        "scope",
        "output",
        "new_window",
    ] {
        assert!(
            all.iter().any(|d| d == key),
            "{key} is read but no choice declares it"
        );
    }
}

/// Each method is asked for its own parameters and no others. This is the
/// whole point of the dynamic dialog: eight tuning controls exist between
/// the nine methods and no method reads more than three of them.
#[test]
fn each_method_is_asked_only_for_what_it_reads() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    // method -> the tuning controls it should offer. Anything in the union
    // and not in the row must be absent.
    let expected: [(Method, &[&str]); 9] = [
        (Method::RichardsonLucy, &["iterations", "stop_delta"]),
        (
            Method::RichardsonLucyTv,
            &["iterations", "stop_delta", "lambda"],
        ),
        (Method::Wiener, &["gamma"]),
        (Method::RegularisedInverse, &["lambda"]),
        (Method::NaiveInverse, &["threshold"]),
        (Method::Landweber, &["iterations", "stop_delta", "step"]),
        (Method::VanCittert, &["iterations", "stop_delta", "step"]),
        (
            Method::TikhonovMiller,
            &["iterations", "stop_delta", "step", "lambda"],
        ),
        (Method::Mrnsd, &["iterations", "stop_delta"]),
    ];
    let tuning = [
        "iterations",
        "stop_delta",
        "lambda",
        "gamma",
        "step",
        "threshold",
    ];
    for (i, (method, wanted)) in expected.iter().enumerate() {
        assert_eq!(Method::ALL[i], *method, "the list drifted from Method::ALL");
        let d = declared(&mut host, &[("method", ParamValue::Choice(i))]);
        for key in tuning {
            let should = wanted.contains(&key);
            let does = d.iter().any(|x| x == key);
            assert_eq!(
                does,
                should,
                "{}: {key} is {} and should not be",
                method.label(),
                if does { "offered" } else { "missing" }
            );
        }
    }
}

/// The PSF source decides which half of that group is shown. Asking for a
/// file path *and* a Gaussian width at once is asking the user which of two
/// answers will be ignored.
#[test]
fn the_psf_source_decides_which_controls_appear() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);

    let file = declared(&mut host, &[("psf_source", ParamValue::Choice(0))]);
    for key in ["psf_path", "psf_origin", "normalise_psf"] {
        assert!(file.iter().any(|d| d == key), "a file needs {key}");
    }
    assert!(!file.iter().any(|d| d == "sigma_xy"));

    let gauss = declared(&mut host, &[("psf_source", ParamValue::Choice(1))]);
    assert!(gauss.iter().any(|d| d == "sigma_xy"));
    assert!(
        gauss.iter().any(|d| d == "sigma_z"),
        "a volume has an axial width"
    );
    // A Gaussian built here is centred and sums to one by construction, so
    // there is nothing to ask about either.
    for key in ["psf_path", "psf_origin", "normalise_psf"] {
        assert!(
            !gauss.iter().any(|d| d == key),
            "a built-in Gaussian should not ask about {key}"
        );
    }
}

/// A flat stack has no axial width to give the Gaussian, and one timepoint
/// is not a choice of timepoints.
#[test]
fn controls_with_nothing_to_choose_are_not_shown() {
    let mut flat = TestHost::new(8, 8, 1, 1, 1);
    let d = declared(&mut flat, &[("psf_source", ParamValue::Choice(1))]);
    assert!(!d.iter().any(|x| x == "sigma_z"));
    assert!(!d.iter().any(|x| x == "scope"));

    let mut movie = TestHost::new(8, 8, 1, 1, 4);
    let d = declared(&mut movie, &[]);
    assert!(d.iter().any(|x| x == "scope"));
}

#[test]
fn the_dialog_is_grouped_into_sections() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let (decls, _) = settle(&Deconvolve, &mut host, &[]);
    let sections: Vec<&str> = decls
        .iter()
        .filter(|d| d.kind == ParamKind::Section)
        .map(|d| d.label.as_str())
        .collect();
    assert_eq!(
        sections,
        vec!["Point spread function", "Algorithm", "Image"]
    );
    for (i, d) in decls.iter().enumerate() {
        if d.kind == ParamKind::Section {
            assert!(
                decls
                    .get(i + 1)
                    .is_some_and(|n| n.kind != ParamKind::Section),
                "the {:?} heading has nothing under it",
                d.label
            );
        }
    }
}

/// Switching method and back must not lose what was typed for the first one.
#[test]
fn a_setting_survives_a_trip_through_another_method() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let (_, values) = settle(
        &Deconvolve,
        &mut host,
        &[
            ("method", ParamValue::Choice(0)),
            ("iterations", ParamValue::Int(77)),
        ],
    );
    assert_eq!(values.int("iterations", 0), 77);

    // The host loop by hand: to Wiener, which has no iteration count, and
    // back to Richardson-Lucy, which does.
    let mut live = values.clone();
    for m in [2usize, 0] {
        live.set("method", ParamValue::Choice(m));
        host.pending = live.clone();
        let decls = Deconvolve.params(&host);
        for (k, v) in Params::defaults(&decls).iter() {
            if live.get(k).is_none() {
                live.set(k, v.clone());
            }
        }
    }
    host.pending = Params::new();
    assert_eq!(
        live.int("iterations", 0),
        77,
        "the iteration count was reset by a visit to Wiener"
    );
}

#[test]
fn the_method_list_is_the_one_the_code_knows() {
    let host = TestHost::new(8, 8, 1, 1, 1);
    let decls = Deconvolve.params(&host);
    match decls
        .iter()
        .find(|d| d.key == "method")
        .map(|d| d.kind.clone())
    {
        Some(ParamKind::Choice { options, .. }) => {
            assert_eq!(options.len(), Method::ALL.len());
            assert_eq!(options[0], Method::RichardsonLucy.label());
        }
        other => panic!("{other:?}"),
    }
}

/// The PSF control is a path, and it opens rather than saves — a save dialog
/// here would offer to overwrite the user's PSF.
#[test]
fn the_psf_control_is_an_open_dialog() {
    let host = TestHost::new(8, 8, 1, 1, 1);
    let decls = Deconvolve.params(&host);
    match decls
        .iter()
        .find(|d| d.key == "psf_path")
        .map(|d| d.kind.clone())
    {
        Some(ParamKind::Path { save, default }) => {
            assert!(!save, "picking a PSF must not be a save dialog");
            assert!(default.is_empty());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_dialog_offers_all_three_shapes() {
    let host = TestHost::new(8, 8, 1, 4, 1);
    let decls = Deconvolve.params(&host);
    match decls
        .iter()
        .find(|d| d.key == "dimensionality")
        .map(|d| d.kind.clone())
    {
        Some(ParamKind::Choice { options, .. }) => {
            assert_eq!(options.len(), 3);
            assert!(options[2].contains("frames as Z"), "{:?}", options);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn it_lives_in_the_deconvolution_menu() {
    let info = Deconvolve.info();
    assert_eq!(info.menu_path, "Deconvolution");
    assert_eq!(info.id, "dev.fasttiff.deconvolve.run");
}

// ------------------------------------------------------------------ the work

/// The end-to-end claim: a blurred stack comes back closer to what it was
/// before the blur.
#[test]
fn it_sharpens_a_blurred_image() {
    const W: usize = 32;
    let truth: Vec<f32> = (0..W * W)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            if (10..14).contains(&x) && (8..20).contains(&y) {
                1.0
            } else {
                0.05
            }
        })
        .collect();
    // Blur it with a separable 3-tap kernel, by hand, so the fixture does not
    // come from the same code the test is checking. Two passes of [1 2 1]/4
    // along each axis is a Gaussian of sigma 1 — which is the PSF the dialog
    // is then asked for, so the forward model is right and what is left to
    // test is whether the inversion works.
    let mut blurred = truth.clone();
    for pass in 0..4 {
        let src = blurred.clone();
        let along_x = pass % 2 == 0;
        for y in 0..W {
            for x in 0..W {
                let at = |d: isize| {
                    let (px, py) = if along_x {
                        ((x as isize + d).clamp(0, W as isize - 1) as usize, y)
                    } else {
                        (x, (y as isize + d).clamp(0, W as isize - 1) as usize)
                    };
                    src[py * W + px]
                };
                blurred[y * W + x] = (at(-1) + 2.0 * at(0) + at(1)) / 4.0;
            }
        }
    }

    let mut host = TestHost::new(W as u32, W as u32, 1, 1, 1);
    host.set_plane(Plane::new(0, 0, 0), &blurred);
    let p = params(
        &mut host,
        &[
            ("psf_source", ParamValue::Choice(1)),
            ("sigma_xy", ParamValue::Float(1.0)),
            ("sigma_z", ParamValue::Float(0.0)),
            ("iterations", ParamValue::Int(50)),
        ],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    let out = floats(&image.planes[0]);

    let err = |v: &[f32]| -> f64 {
        v.iter()
            .zip(&truth)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        err(out) < err(&blurred) * 0.8,
        "deconvolution left {:.4} of a starting {:.4}",
        err(out),
        err(&blurred)
    );
}

/// Every plane must come back as itself: same channel, same slice, same
/// timepoint. With a delta PSF the arithmetic is the identity, so anything
/// that moves has been indexed wrongly — which is the failure that looks like
/// a working plugin until someone opens a two-channel stack.
#[test]
fn every_plane_comes_back_where_it_started() {
    let (c, z, t) = (2usize, 3usize, 2usize);
    let mut host = TestHost::new(4, 4, c, z, t);
    for ti in 0..t {
        for zi in 0..z {
            for ci in 0..c {
                // A value that encodes its own address.
                let tag = (ti * 100 + zi * 10 + ci) as f32;
                host.set_plane(Plane::new(ci, zi, ti), &[tag; 16]);
            }
        }
    }
    // All three shapes: each walks the planes in a different order, and each
    // has to put them back in the same one.
    for shape in 0..3 {
        let p = params(
            &mut host,
            &[
                ("psf_path", ParamValue::Path(delta_psf_file("delta.tif"))),
                ("dimensionality", ParamValue::Choice(shape)),
            ],
        );
        let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
        assert_eq!((image.channels, image.slices, image.frames), (c, z, t));

        for ti in 0..t {
            for zi in 0..z {
                for ci in 0..c {
                    let tag = (ti * 100 + zi * 10 + ci) as f32;
                    let at = ti * (z * c) + zi * c + ci;
                    let got = floats(&image.planes[at]);
                    assert!(
                        got.iter().all(|v| (v - tag).abs() < 0.01),
                        "shape {shape}: plane (c{ci}, z{zi}, t{ti}) came back as {} not {tag}",
                        got[0]
                    );
                }
            }
        }
    }
}

/// A z-stack that opened as a time series can still be deconvolved as a
/// volume, and the option is the only way to say so.
///
/// `resolve_dimensions` folds a single-timepoint z-stack into frames, because
/// far more files shaped that way are movies. The consequence for this plugin
/// is that "3D (whole volume)" on such a file is 2-D per frame — correct, and
/// not what the user wanted. The two halves of this test are that the default
/// really does leave the frames unmixed, and that the third option really
/// does mix them.
#[test]
fn frames_can_be_treated_as_z() {
    let frames = 5;
    let mut host = TestHost::new(4, 4, 1, 1, frames);
    for t in 0..frames {
        host.set_plane(Plane::new(0, 0, t), &[if t == 2 { 9.0 } else { 0.0 }; 16]);
    }
    // A PSF that is flat laterally and three deep: it can only do something
    // visible by mixing along the volume axis.
    let path = write_f32_tiff("along-t.tif", 1, 1, &[vec![1.0], vec![1.0], vec![1.0]])
        .display()
        .to_string();

    let along_t = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(path.clone())),
            ("dimensionality", ParamValue::Choice(2)),
            ("method", ParamValue::Choice(2)), // Wiener: one shot, no iteration count
            ("gamma", ParamValue::Float(1.0)), // heavily damped, so it stays a blur
        ],
    );
    let mixed = image_of(Deconvolve.run(&mut host, &along_t).expect("run"));
    assert_eq!(mixed.frames, frames);
    assert!(
        floats(&mixed.planes[1])[0] > 0.01,
        "frame 1 should have picked up signal from frame 2: {:?}",
        floats(&mixed.planes[1])[0]
    );

    // The default, on the same file, leaves every frame to itself.
    let whole = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(path)),
            ("dimensionality", ParamValue::Choice(0)),
            ("method", ParamValue::Choice(2)),
            ("gamma", ParamValue::Float(1.0)),
        ],
    );
    let apart = image_of(Deconvolve.run(&mut host, &whole).expect("run"));
    assert!(
        floats(&apart.planes[1])[0].abs() < 1e-6,
        "a per-frame run must not pull signal in from another frame: {:?}",
        floats(&apart.planes[1])[0]
    );
    // And it says so, because the result is correct and is not what was asked
    // for.
    assert!(host.said("really Z slices"), "{:?}", host.logged);
}

/// The grid it is about to allocate is reported before it is allocated.
#[test]
fn the_working_memory_is_stated_up_front() {
    let mut host = TestHost::new(16, 16, 1, 1, 1);
    let p = params(
        &mut host,
        &[("psf_path", ParamValue::Path(delta_psf_file("mem.tif")))],
    );
    Deconvolve.run(&mut host, &p).expect("run");
    assert!(host.said("working grid"), "{:?}", host.logged);
    assert!(host.said("GB"), "{:?}", host.logged);
}

#[test]
fn a_missing_psf_file_stops_the_run_before_any_work() {
    let mut host = TestHost::new(8, 8, 1, 1, 1);
    let p = params(&mut host, &[]);
    let err = Deconvolve.run(&mut host, &p).expect_err("no PSF chosen");
    assert!(matches!(err, PluginError::Unsupported(_)), "{err:?}");
}

#[test]
fn a_psf_larger_than_the_image_is_refused_with_both_sizes() {
    let mut host = TestHost::new(4, 4, 1, 1, 1);
    let path = write_f32_tiff("too-big.tif", 9, 9, &[vec![1.0; 81]]);
    let p = params(
        &mut host,
        &[("psf_path", ParamValue::Path(path.display().to_string()))],
    );
    let err = Deconvolve.run(&mut host, &p).expect_err("too big");
    let text = err.to_string();
    assert!(text.contains("9x9") && text.contains("4x4"), "{text}");
}

#[test]
fn a_psf_that_cannot_be_normalised_is_refused() {
    let mut host = TestHost::new(8, 8, 1, 1, 1);
    let path = write_f32_tiff("empty.tif", 3, 3, &[vec![0.0; 9]]);
    let p = params(
        &mut host,
        &[("psf_path", ParamValue::Path(path.display().to_string()))],
    );
    let err = Deconvolve.run(&mut host, &p).expect_err("all zero");
    assert!(err.to_string().contains("background"), "{err}");
}

/// Slice by slice with a 3-D PSF takes the in-focus slice and says so, rather
/// than refusing or silently using the whole thing.
#[test]
fn two_dimensional_mode_uses_the_psfs_focal_slice() {
    let mut host = TestHost::new(8, 8, 1, 3, 1);
    let planes = vec![vec![0.0f32; 9], vec![1.0f32; 9], vec![0.0f32; 9]];
    let path = write_f32_tiff("three-slice.tif", 3, 3, &planes);
    let p = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(path.display().to_string())),
            ("dimensionality", ParamValue::Choice(1)),
            ("iterations", ParamValue::Int(2)),
        ],
    );
    Deconvolve.run(&mut host, &p).expect("run");
    assert!(host.said("in-focus slice (2)"), "{:?}", host.logged);
}

/// A one-slice PSF on a volume cannot remove out-of-focus light, and the user
/// has to be told: the result looks sharper and is not what they think.
#[test]
fn a_flat_psf_on_a_volume_is_called_out() {
    let mut host = TestHost::new(8, 8, 1, 4, 1);
    let p = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(delta_psf_file("flat.tif"))),
            ("iterations", ParamValue::Int(2)),
        ],
    );
    Deconvolve.run(&mut host, &p).expect("run");
    assert!(host.said("out-of-focus"), "{:?}", host.logged);
}

#[test]
fn one_timepoint_can_be_done_on_its_own() {
    let mut host = TestHost::new(4, 4, 1, 1, 5);
    host.view.frame_index = 3;
    for t in 0..5 {
        host.set_plane(Plane::new(0, 0, t), &[t as f32 + 1.0; 16]);
    }
    let p = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(delta_psf_file("one-t.tif"))),
            ("scope", ParamValue::Choice(1)),
        ],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    assert_eq!(image.frames, 1);
    assert_eq!(image.planes.len(), 1);
    // And it is the timepoint that was on screen, not the first one.
    assert!((floats(&image.planes[0])[0] - 4.0).abs() < 0.01);
}

#[test]
fn the_output_type_is_float_by_default_and_the_sources_on_request() {
    let mut host = TestHost::new(4, 4, 1, 1, 1);
    host.image.pixel_type = PixelType::U16;
    host.set_plane(Plane::new(0, 0, 0), &[1000.0; 16]);
    let path = ParamValue::Path(delta_psf_file("type.tif"));

    let p = params(&mut host, &[("psf_path", path.clone())]);
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    assert_eq!(image.pixel_type, PixelType::F32);

    let p = params(
        &mut host,
        &[("psf_path", path), ("output", ParamValue::Choice(1))],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    assert_eq!(image.pixel_type, PixelType::U16);
    match &image.planes[0] {
        PlaneData::U16(v) => assert!(
            v.iter().all(|&x| (x as i32 - 1000).abs() <= 1),
            "{:?}",
            &v[..4]
        ),
        other => panic!("{:?}", other.pixel_type()),
    }
}

#[test]
fn the_checkbox_decides_where_the_result_goes() {
    let mut host = TestHost::new(4, 4, 1, 1, 1);
    let path = ParamValue::Path(delta_psf_file("where.tif"));
    let p = params(
        &mut host,
        &[
            ("psf_path", path.clone()),
            ("new_window", ParamValue::Bool(false)),
        ],
    );
    assert!(matches!(
        Deconvolve.run(&mut host, &p).expect("run"),
        Outcome::ReplaceDocument(_)
    ));
    let p = params(
        &mut host,
        &[("psf_path", path), ("new_window", ParamValue::Bool(true))],
    );
    assert!(matches!(
        Deconvolve.run(&mut host, &p).expect("run"),
        Outcome::NewDocument(_)
    ));
}

#[test]
fn cancelling_is_not_an_error() {
    let mut host = TestHost::new(16, 16, 1, 1, 1);
    host.cancel_after = Some(0);
    let p = params(
        &mut host,
        &[("psf_path", ParamValue::Path(delta_psf_file("cancel.tif")))],
    );
    assert_eq!(
        Deconvolve
            .run(&mut host, &p)
            .expect("cancel is not failure"),
        Outcome::Cancelled
    );
}

/// The geometry is unchanged, so the calibration has to travel with it —
/// a deconvolved stack that lost its pixel size is no longer measurable.
#[test]
fn the_calibration_survives() {
    let mut host = TestHost::new(4, 4, 1, 1, 1);
    host.info.unit = Some("micron".into());
    host.info.spacing = fasttiff_plugin_api::Spacing {
        x: Some(0.1),
        y: Some(0.1),
        z: Some(0.5),
    };
    let p = params(
        &mut host,
        &[("psf_path", ParamValue::Path(delta_psf_file("cal.tif")))],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    let meta = image.metadata.as_ref().expect("metadata");
    assert_eq!(meta.spacing.x, Some(0.1));
    assert_eq!(meta.spacing.z, Some(0.5));
}

#[test]
fn a_divergent_run_is_an_error_rather_than_a_result() {
    let mut host = TestHost::new(16, 16, 1, 1, 1);
    for i in 0..256 {
        host.pixels[i] = (i % 7) as f32;
    }
    let p = params(
        &mut host,
        &[
            ("psf_source", ParamValue::Choice(1)),
            ("sigma_xy", ParamValue::Float(2.0)),
            ("sigma_z", ParamValue::Float(0.0)),
            ("method", ParamValue::Choice(6)), // Van Cittert
            ("step", ParamValue::Float(2.0)),
            ("iterations", ParamValue::Int(1000)),
            ("stop_delta", ParamValue::Float(0.0)),
            ("nonneg", ParamValue::Bool(false)),
        ],
    );
    let err = Deconvolve
        .run(&mut host, &p)
        .expect_err("this must diverge");
    assert!(err.to_string().contains("diverged"), "{err}");
    assert!(
        err.to_string().contains("step"),
        "the message must say what to change: {err}"
    );
}

/// The Gaussian source exists so the tool can be used with no file at all.
#[test]
fn the_built_in_gaussian_needs_no_file() {
    let mut host = TestHost::new(16, 16, 1, 1, 1);
    host.set_plane(Plane::new(0, 0, 0), &[5.0; 256]);
    let p = params(
        &mut host,
        &[
            ("psf_source", ParamValue::Choice(1)),
            ("sigma_xy", ParamValue::Float(1.5)),
            ("iterations", ParamValue::Int(3)),
        ],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    assert_eq!(image.planes.len(), 1);
    assert!(host.said("Gaussian PSF"), "{:?}", host.logged);
}

#[test]
fn the_gaussian_kernel_is_odd_sized_and_sums_to_something() {
    let psf = gaussian_psf(1.0, 0.0);
    assert_eq!(psf.dims.z, 1, "sigma 0 in z is a flat kernel");
    assert_eq!(psf.dims.x % 2, 1, "an odd extent puts the peak on a voxel");
    assert_eq!(psf.dims, Dims::new(7, 7, 1), "three sigma each way");
    assert_eq!(
        grid::brightest(&psf.data, psf.dims),
        (3, 3, 0),
        "the peak belongs at the centre"
    );
    let total: f32 = psf.data.iter().sum();
    assert!(total > 0.0);
}

/// The name says which method produced it, because two deconvolutions of one
/// stack are told apart by nothing else.
#[test]
fn the_window_is_named_after_the_method() {
    let mut host = TestHost::new(4, 4, 1, 1, 1);
    let p = params(
        &mut host,
        &[
            ("psf_path", ParamValue::Path(delta_psf_file("name.tif"))),
            ("method", ParamValue::Choice(2)), // Wiener
        ],
    );
    let image = image_of(Deconvolve.run(&mut host, &p).expect("run"));
    assert!(image.name.ends_with("-wiener"), "{}", image.name);
}

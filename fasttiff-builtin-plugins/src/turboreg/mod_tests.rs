//! The plugin around the registration: what it offers, and what it refuses.
//!
//! The algorithm is tested in `turboreg-registration`. What is checked here
//! is the wiring — that the dialog changes with the file and with itself, that
//! the channel selector says something true about a single-channel stack, and
//! that a stack with no time axis is refused rather than silently "registered".

use super::*;
use fasttiff_plugin_api::{ImageInfo, ParamKind, ParamValue, PixelType};
use turboreg_registration::{Settings, Transformation};

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
    for (choice, want) in [
        (0, Border::Constant),
        (1, Border::Edge),
        (2, Border::Crop),
        (3, Border::Native),
    ] {
        let mut p = Params::new();
        p.set("edges", ParamValue::Choice(choice));
        assert_eq!(params::border_from(&p), want, "option {choice}");
    }
    assert!(keys(&decls_with(&info(1, 1, 10), &[])).contains(&"edges".to_string()));
}

// ------------------------------------------------------------ the run itself

use fasttiff_plugin_api::{Outcome, PlaneData, StackInfo, ViewParams, VolumeMode, VolumeView};
use turboreg_registration::{register_stack, Backend, Frames, Reference};

/// A stack in memory, `xyczt`, that keeps the planes it was asked for, and
/// where the bar stood when each was asked for, and can be told to stop.
struct Memory {
    image: ImageInfo,
    view: ViewParams,
    info: StackInfo,
    planes: Vec<Vec<f32>>,
    reads: Vec<Plane>,
    /// The progress last reported before each of `reads`.
    read_at: Vec<f32>,
    progress: f32,
    progress_calls: usize,
    stop_after: Option<usize>,
}

impl Memory {
    /// `channels` textures drifting a little further each of `frames` frames.
    fn drifting(w: u32, h: u32, channels: usize, frames: usize) -> Self {
        let (w, h) = (w as usize, h as usize);
        let mut planes = Vec::new();
        for t in 0..frames {
            let (dx, dy) = (0.7 * t as f32, -0.45 * t as f32 + 0.05 * (t * t) as f32);
            for c in 0..channels {
                let k = 1.0 + 0.35 * c as f32;
                planes.push(
                    (0..w * h)
                        .map(|i| {
                            let (x, y) = ((i % w) as f32 - dx, (i / w) as f32 - dy);
                            300.0
                                + ((x * 0.21 * k).sin() * (y * 0.17).cos()) * 90.0
                                + (x * 0.09 + y * 0.13 * k).sin() * 30.0
                                + 4.0 * ((t * 7 + c * 3) % 5) as f32
                        })
                        .collect(),
                );
            }
        }
        Memory {
            image: ImageInfo {
                width: w as u32,
                height: h as u32,
                channels,
                slices: 1,
                frames,
                samples_per_pixel: 1,
                pixel_type: PixelType::F32,
            },
            view: ViewParams {
                frame_index: 0,
                volume_view: false,
                channels: Vec::new(),
                luts: Vec::new(),
                volume: VolumeView {
                    mode: VolumeMode::Mip,
                    density: 1.0,
                    iso: 0.5,
                    eye: [0.0; 3],
                    forward: [0.0, 0.0, 1.0],
                    up: [0.0, 1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                },
            },
            info: StackInfo {
                name: "drift".into(),
                ..Default::default()
            },
            planes,
            reads: Vec::new(),
            read_at: Vec::new(),
            progress: 0.0,
            progress_calls: 0,
            stop_after: None,
        }
    }

    fn plane(&self, p: Plane) -> &[f32] {
        &self.planes[p.t * self.image.channels + p.c]
    }
}

impl HostContext for Memory {
    fn image(&self) -> ImageInfo {
        self.image
    }
    fn view(&self) -> &ViewParams {
        &self.view
    }
    fn stack_info(&self) -> &StackInfo {
        &self.info
    }
    fn read_plane_u16(&mut self, _: Plane, _: &mut Vec<u16>) -> Result<(), PluginError> {
        unreachable!("TurboReg reads float planes")
    }
    fn read_plane_f32(&mut self, p: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        self.reads.push(p);
        self.read_at.push(self.progress);
        out.clear();
        out.extend_from_slice(self.plane(p));
        Ok(())
    }
    fn progress(&mut self, fraction: f32) -> bool {
        self.progress = fraction;
        self.progress_calls += 1;
        self.stop_after.is_none_or(|n| self.progress_calls <= n)
    }
    fn log(&mut self, _: &str) {}
}

/// The dialog's values for a run.
fn chosen(
    reference: Reference,
    window: usize,
    n_frames: usize,
    measure: usize,
    backend: Backend,
) -> Params {
    let mut p = Params::new();
    let model = Transformation::all()
        .iter()
        .position(|t| *t == Transformation::Translation)
        .unwrap();
    p.set("transformation", ParamValue::Choice(model));
    let options = params::reference_options(Transformation::Translation);
    let r = options.iter().position(|r| *r == reference).unwrap();
    p.set("reference", ParamValue::Choice(r));
    p.set("moving_average", ParamValue::Int(window as i64));
    p.set("n_frames", ParamValue::Int(n_frames as i64));
    p.set("ref_channel", ParamValue::Choice(measure));
    let b = Backend::all().iter().position(|b| *b == backend).unwrap();
    p.set("backend", ParamValue::Choice(b));
    p
}

/// The run reads the measurement a batch at a time, as the crate's session
/// asks for it, and the stack it hands back is the one the crate registers
/// from memory and moves, plane for plane, to the bit: the measurement (a
/// channel, or the channel mean in `f64`), the target's frames read first,
/// the smoothing across batch edges, the border. Then every plane is read
/// once, in `xyczt` order, to be moved.
#[test]
fn the_run_registers_what_the_crate_registers_from_memory() {
    let (w, h, channels, frames) = (40u32, 32u32, 2usize, 9usize);
    let mut cases = Vec::new();
    for reference in Reference::all() {
        for window in [1, 3] {
            cases.push(chosen(reference, window, 2, 0, Backend::SingleThread));
        }
    }
    cases.push(chosen(Reference::First, 1, 1, 2, Backend::SingleThread));
    if Backend::MultiThread.available() {
        cases.push(chosen(Reference::Mean, 3, 1, 0, Backend::MultiThread));
        cases.push(chosen(Reference::Previous, 1, 1, 1, Backend::MultiThread));
    }
    for p in cases {
        let mut host = Memory::drifting(w, h, channels, frames);
        let settings = params::settings_from(&p);
        let measure_on = params::measure_channel(&p, channels);
        let what = format!("{settings:?}, measured on {measure_on:?}");
        let Outcome::NewDocument(out) = TurboReg.run(&mut host, &p).expect("run") else {
            panic!("{what}: registration should produce a new document");
        };

        // The crate, from memory.
        let (ly, lx) = (h as usize, w as usize);
        let movie: Vec<Vec<f32>> = (0..frames)
            .map(|t| match measure_on {
                Some(c) => host.plane(Plane::new(c, 0, t)).to_vec(),
                None => {
                    let mut sum: Vec<f64> = host
                        .plane(Plane::new(0, 0, t))
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                    for c in 1..channels {
                        for (a, &v) in sum.iter_mut().zip(host.plane(Plane::new(c, 0, t))) {
                            *a += v as f64;
                        }
                    }
                    sum.iter().map(|&v| (v / channels as f64) as f32).collect()
                }
            })
            .collect();
        let mut want = register_stack(
            &Frames {
                ly,
                lx,
                frames: &movie,
            },
            &settings,
            &mut |_| true,
        )
        .expect("the crate registers it");
        want.border = params::border_from(&p);
        assert_eq!(out.planes.len(), frames * channels, "{what}");
        for t in 0..frames {
            for c in 0..channels {
                let moved = want
                    .apply(host.plane(Plane::new(c, 0, t)), ly, lx, t)
                    .unwrap();
                let PlaneData::F32(got) = &out.planes[t * channels + c] else {
                    panic!("{what}: a float stack comes back float");
                };
                assert!(
                    got.iter()
                        .map(|v| v.to_bits())
                        .eq(moved.iter().map(|v| v.to_bits())),
                    "{what}: frame {t} channel {c}"
                );
            }
        }

        // What was read: the session's plan, a measurement per frame it asked
        // for, and then every plane once.
        let plan = turboreg_registration::Session::new(ly, lx, frames, &settings)
            .unwrap()
            .plan();
        let measure = |t: usize| -> Vec<Plane> {
            match measure_on {
                Some(c) => vec![Plane::new(c, 0, t)],
                None => (0..channels).map(|c| Plane::new(c, 0, t)).collect(),
            }
        };
        let expect: Vec<Plane> = plan
            .reference
            .chain(plan.fit)
            .flat_map(measure)
            .chain((0..frames).flat_map(|t| (0..channels).map(move |c| Plane::new(c, 0, t))))
            .collect();
        assert_eq!(host.reads, expect, "{what}");
    }
}

/// Stopping is honoured at every point the bar moves: while the target's
/// frames are read, while the stack is fitted, and while it is moved — and
/// there and then, the call that said stop being the last. A run that only
/// noticed when the moved planes were streamed would also end `Cancelled`.
#[test]
fn a_run_stops_wherever_it_is_asked_to() {
    let p = chosen(Reference::Mean, 3, 1, 0, Backend::SingleThread);
    let mut whole = Memory::drifting(24, 20, 2, 5);
    assert!(matches!(
        TurboReg.run(&mut whole, &p),
        Ok(Outcome::NewDocument(_))
    ));
    let calls = whole.progress_calls;
    for stop_after in [0, 3, calls / 2, calls - 2] {
        let mut host = Memory::drifting(24, 20, 2, 5);
        host.stop_after = Some(stop_after);
        assert!(
            matches!(TurboReg.run(&mut host, &p), Ok(Outcome::Cancelled)),
            "stopped after {stop_after} of {calls} progress calls"
        );
        assert_eq!(
            host.progress_calls,
            stop_after + 1,
            "the run went on after call {} said stop",
            stop_after + 1
        );
    }
}

/// What the crate cannot run, or could not finish, comes back as the run's
/// error in the host's terms: a refusal for what cannot run here, a failure
/// for frames the GPU lost, which says what will run instead.
#[test]
fn a_registration_that_cannot_run_or_finish_says_why() {
    let unchainable = plugin_error(Error::Unchainable);
    assert!(
        matches!(&unchainable, PluginError::Unsupported(why) if why.contains("first frame")),
        "{unchainable}"
    );
    assert!(matches!(
        plugin_error(Error::Unavailable("no card".into())),
        PluginError::Unsupported(why) if why == "no card"
    ));
    let device = plugin_error(Error::Device {
        failed: 3,
        frames: 9,
        why: "no room".into(),
    });
    let PluginError::Failed(why) = &device else {
        panic!("{device}");
    };
    assert!(
        why.starts_with("3 of 9 frames could not be registered on the GPU: no room. ")
            && why.contains(Backend::MultiThread.label()),
        "{why}"
    );
}

/// The measurement is read in step with the fits, never more than a batch
/// ahead of them: before a frame's measurement is read, every frame of the
/// batches before its own has been registered, which the bar shows. The run
/// used to read the whole measurement movie first, with the bar at the
/// reading's share alone, and only then register any of it. On one thread a
/// batch is a frame; on every core, a stack of a few batches.
#[test]
fn the_measurement_is_read_a_batch_at_a_time() {
    let (w, h) = (24u32, 20u32);
    let mut runs = vec![(Backend::SingleThread, 6)];
    if Backend::MultiThread.available() {
        let s = Settings {
            backend: Backend::MultiThread,
            ..Settings::default()
        };
        let per = turboreg_registration::Session::new(h as usize, w as usize, 1000, &s)
            .unwrap()
            .batch_len();
        runs.push((Backend::MultiThread, 2 * per + 3));
    }
    for (backend, frames) in runs {
        let p = chosen(Reference::Previous, 1, 1, 0, backend);
        let settings = params::settings_from(&p);
        let session =
            turboreg_registration::Session::new(h as usize, w as usize, frames, &settings).unwrap();
        let per = session.batch_len();
        assert!(per < frames, "{backend:?}: more than one batch");
        assert_eq!(session.plan().fit, 0..frames);

        let mut host = Memory::drifting(w, h, 1, frames);
        assert!(matches!(
            TurboReg.run(&mut host, &p),
            Ok(Outcome::NewDocument(_))
        ));
        // One channel, one slice: a read per frame to measure, each fitted
        // (frame 0 is read for frame 1 to be fitted against), then a read per
        // frame to move.
        let n = frames as f64;
        let total = n * work::READ + n * work::REGISTER + n * (work::READ + work::WARP);
        assert_eq!(host.reads.len(), 2 * frames);
        for t in 0..frames {
            assert_eq!(host.reads[t], Plane::new(0, 0, t));
            let registered = (t - t % per) as f64;
            assert!(
                host.read_at[t] as f64 * total + 1e-3 >= registered * work::REGISTER,
                "{backend:?}: frame {t} was read with the bar at {}, before the {registered} \
                 frames ahead of it were registered",
                host.read_at[t]
            );
        }
    }
}

//! The dialog, and reading it back into [`Settings`].
//!
//! Split from the run for the same reason the stabilization plugin's is: the
//! controls are the whole of the algorithm's options, and a list that long in
//! the middle of the run makes both harder to read.
//!
//! # The two "mean"s
//!
//! There are two independent choices here that both have an option called
//! "mean", and conflating them is the mistake this note exists to prevent:
//!
//! * **Reference channel** — *which channel is measured*. A multi-channel
//!   recording is one specimen photographed several ways at once, so every
//!   channel moves together; the transformation is measured on one of them (or
//!   on their mean) and applied to all. Measuring each channel separately would
//!   move them relative to each other, which is the one thing a multi-channel
//!   recording must not do.
//! * **Reference frame** — *what each frame is aligned to*: the previous frame,
//!   the first, or the average of the stack. This is pystackreg's `reference=`.
//!
//! One picks a channel, the other picks a target in time. The labels say so.

use fasttiff_plugin_api::{ImageInfo, ParamDecl, ParamKind, Params};
use turboreg_registration::{Backend, Reference, Settings, Transformation};

/// A group heading, drawn by the host as bold text with a rule under it.
fn section(key: &str, text: &str) -> ParamDecl {
    ParamDecl::new(key, text, ParamKind::Section)
}

/// Which reference modes a transformation can use.
///
/// Bilinear cannot chain: the composition of two bilinear maps is not bilinear,
/// so "previous frame" has nothing to compose and pystackreg refuses it. The
/// option is removed rather than offered-and-rejected, which means the list is
/// model-dependent — so this one function builds it for both the dialog and the
/// read-back, and the index a user picked keeps its meaning.
pub(super) fn reference_options(model: Transformation) -> Vec<Reference> {
    match model {
        Transformation::Bilinear => vec![Reference::First, Reference::Mean],
        _ => Reference::all().to_vec(),
    }
}

/// The transformation currently chosen, which several other controls depend on.
fn chosen_model(chosen: &Params) -> Transformation {
    let d = Settings::default();
    let default = Transformation::all()
        .iter()
        .position(|t| *t == d.transformation)
        .unwrap_or(1);
    *Transformation::all()
        .get(chosen.choice("transformation", default))
        .unwrap_or(&d.transformation)
}

/// The reference mode currently chosen, resolved against the model's own list.
fn chosen_reference(chosen: &Params, model: Transformation) -> Reference {
    let options = reference_options(model);
    let d = Settings::default();
    let default = options.iter().position(|r| *r == d.reference).unwrap_or(0);
    *options
        .get(chosen.choice("reference", default))
        .unwrap_or(&d.reference)
}

/// How many channels the measurement selector should offer, and their labels.
///
/// With one channel there is exactly one thing the measurement can be, and
/// "mean of one channel" and "channel 1" are the same arithmetic. So a
/// single-channel stack gets a single option — which the host draws greyed out,
/// because a choice of one is not a choice. Offering "Mean" and "Channel 1"
/// separately there would be two names for one behaviour and an invitation to
/// wonder which is which.
fn channel_options(channels: usize) -> Vec<String> {
    if channels <= 1 {
        return vec!["Mean (only channel)".to_string()];
    }
    let mut v = vec!["Mean of all channels".to_string()];
    v.extend((1..=channels).map(|c| format!("Channel {c}")));
    v
}

/// Build the dialog for a stack of this shape, and for what has been chosen in
/// it so far.
///
/// `chosen` is what the dialog holds right now, which is empty the first time
/// it is built and reads back as the declared defaults — so this one function
/// produces both the first dialog and every one after a change.
///
/// # What is hidden, and why hiding is honest here
///
/// Two controls do nothing unless something else is set, and a control that
/// will be ignored is not neutral: it is an invitation to set it and then
/// conclude the algorithm is broken when nothing changes. `n_frames` is only
/// read when the reference is the first frame; the reference *frame* selector
/// is meaningless for a stack with one timepoint. Both are therefore absent
/// rather than greyed.
///
/// The measurement-channel selector is the exception that is greyed rather than
/// hidden, and the difference is worth stating: it does not depend on another
/// setting, it depends on the *file*. Hiding it would leave no way to learn
/// that the choice exists, and the answer to "which channel was this measured
/// on" would be unavailable exactly when there is only one and it is obvious.
pub(super) fn declare(info: &ImageInfo, chosen: &Params) -> Vec<ParamDecl> {
    let d = Settings::default();
    let model = chosen_model(chosen);
    let mut decls = vec![section("h_what", "What to correct")];

    decls.push(
        ParamDecl::new(
            "transformation",
            "Transformation",
            ParamKind::Choice {
                default: Transformation::all()
                    .iter()
                    .position(|t| *t == d.transformation)
                    .unwrap_or(1),
                options: Transformation::all()
                    .iter()
                    .map(|t| t.label().to_string())
                    .collect(),
            },
        )
        .help(
            "How much freedom the correction has. More is not better: every extra degree of \
             freedom is another way for a weak match to go wrong, and a specimen that only \
             drifted does not need rotation taken out of it.",
        ),
    );

    // Which channel is measured. Always offered, so the answer to "what was
    // this measured on" is always on screen; a single-channel stack gets one
    // option and the host greys it.
    decls.push(
        ParamDecl::new(
            "ref_channel",
            "Measure on",
            ParamKind::Choice {
                default: 0,
                options: channel_options(info.channels.max(1)),
            },
        )
        .help(
            "Which channel the transformation is measured on. It is applied to every channel, \
             because the channels are one specimen photographed at once and must stay registered \
             to each other. The mean is the default: averaging the channels raises what the \
             search is matching.",
        ),
    );

    // The time axis. A stack with one timepoint has nothing to align in time.
    if info.frames > 1 {
        decls.push(section("h_time", "What to align to"));
        let options = reference_options(model);
        decls.push(
            ParamDecl::new(
                "reference",
                "Reference frame",
                ParamKind::Choice {
                    default: options.iter().position(|r| *r == d.reference).unwrap_or(0),
                    options: options.iter().map(|r| r.label().to_string()).collect(),
                },
            )
            .help(if model == Transformation::Bilinear {
                "A bilinear warp cannot be chained onto another, so aligning to the previous \
                 frame is not offered for it."
            } else {
                "The first frame is the default: nothing accumulates, and the target is a \
                 frame that was actually photographed. The stack mean is sharper on a long, \
                 noisy recording. Aligning to the previous frame suits a steady drift but \
                 compounds its own error down the stack."
            }),
        );
        if chosen_reference(chosen, model) == Reference::First {
            decls.push(
                ParamDecl::new(
                    "n_frames",
                    "Reference frames average",
                    ParamKind::Int {
                        default: d.n_frames as i64,
                        min: 1,
                        max: info.frames.max(1) as i64,
                    },
                )
                .help("How many frames from the start are averaged to make the target. One uses the first frame alone."),
            );
        }
        decls.push(
            ParamDecl::new(
                "moving_average",
                "Moving average",
                ParamKind::Int {
                    default: d.moving_average as i64,
                    min: 1,
                    max: 31,
                },
            )
            .help(
                "Average this many neighbouring frames before measuring, for a recording too \
                 dim to register frame by frame. The window is centred, so it raises the signal \
                 without moving it. One disables it.",
            ),
        );
    }

    decls.push(section("h_run", "How it runs"));
    decls.push(
        ParamDecl::new(
            "backend",
            "Compute",
            ParamKind::Choice {
                default: Backend::all()
                    .iter()
                    .position(|b| *b == d.backend)
                    .unwrap_or(0),
                options: Backend::all()
                    .iter()
                    .map(|b| b.label().to_string())
                    .collect(),
            },
        )
        .help(
            "The answer is identical on each, bit for bit; multi-thread spreads every \
             frame's work over all cores, and also runs separate frames at once. Aligning \
             to the previous frame takes its frames in turn, since each needs the last, \
             but still uses every core within each one.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "max_depth",
            "Pyramid levels",
            ParamKind::Int {
                default: d.max_pyramid_depth as i64,
                min: 0,
                max: 8,
            },
        )
        .help(
            "How many times the image is halved for the coarse-to-fine search. 0 means as deep \
             as the image allows, which is what finds a large displacement. Fewer is faster and \
             can miss one.",
        ),
    );

    decls.push(crate::plugins::builtin::shared::in_new_window());
    decls
}

/// Which channel to measure on: `None` for the mean of all of them.
pub(super) fn measure_channel(params: &Params, channels: usize) -> Option<usize> {
    if channels <= 1 {
        return None;
    }
    match params.choice("ref_channel", 0) {
        0 => None,
        n => Some((n - 1).min(channels - 1)),
    }
}

/// Read the dialog back.
pub(super) fn settings_from(params: &Params) -> Settings {
    let d = Settings::default();
    let model = chosen_model(params);
    let options = reference_options(model);
    let reference = *options
        .get(params.choice(
            "reference",
            options.iter().position(|r| *r == d.reference).unwrap_or(0),
        ))
        .unwrap_or(&d.reference);
    Settings {
        transformation: model,
        reference,
        n_frames: params.int("n_frames", d.n_frames as i64).max(1) as usize,
        moving_average: params.int("moving_average", d.moving_average as i64).max(1) as usize,
        backend: *Backend::all()
            .get(
                params.choice(
                    "backend",
                    Backend::all()
                        .iter()
                        .position(|b| *b == d.backend)
                        .unwrap_or(0),
                ),
            )
            .unwrap_or(&d.backend),
        max_pyramid_depth: params.int("max_depth", d.max_pyramid_depth as i64).max(0) as usize,
    }
}

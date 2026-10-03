//! The dialog, and reading it back into [`Settings`].
//!
//! Split from the run for one reason: the controls are the *whole* of suite2p's
//! registration options, and a list that long buried in the middle of the
//! algorithm makes both harder to read. Every key here is the name suite2p uses
//! in `default_ops`, so a value copied from a lab's `ops.npy` means the same
//! thing.

use fasttiff_plugin_api::{ImageInfo, ParamDecl, ParamKind, Params};
use suite2p_registration::{Backend, Settings};

/// A group heading, drawn by the host as bold text with a rule under it.
fn section(key: &str, text: &str) -> ParamDecl {
    ParamDecl::new(key, text, ParamKind::Section)
}

/// Build the dialog for a stack of this shape, and for what has been chosen
/// in it so far.
///
/// # Grouping
///
/// Seven sections, running from the decisions that change a run most to the
/// ones most people never touch:
///
///   1. where it runs, and whether the correction may deform;
///   2. what is registered against what, and how far it may move;
///   3. how the reference is built;
///   4. the non-rigid grid;
///   5. the settings for a recording too dim to register frame by frame;
///   6. the scanner's own artefact;
///   7. what is reported afterwards.
///
/// This was a *comment* until the contract grew [`ParamKind::Section`]: the
/// order was the only grouping expressible, and a reader had to be told where
/// the boundaries were. They are now drawn.
///
/// Two of the help texts point at their neighbours ("given below", "above"),
/// so those pairs stay adjacent and in order.
///
/// # What is hidden, and why hiding is honest here
///
/// Three groups of controls do nothing unless something else is set, and
/// suite2p's own documentation is where that is written down rather than the
/// dialog. Four of them are the non-rigid grid, which the rigid pass does not
/// read at all; one is the bidirectional-phase switch, which suite2p ignores
/// whenever a fixed offset is given. Showing a control that will be ignored
/// is not neutral — it is an invitation to set it and conclude the algorithm
/// is broken when nothing changes.
///
/// `chosen` is what the dialog holds right now, which is empty the first time
/// it is built and reads back as the declared defaults — so this one function
/// produces both the first dialog and every one after a change. See
/// `HostContext::pending_params`.
pub(super) fn declare(info: &ImageInfo, chosen: &Params) -> Vec<ParamDecl> {
    let d = Settings::default();

    // ---- what runs, and what kind of correction ----------------------------

    // Built by pushing rather than as one literal: four of the controls below
    // are conditional, and a list that is half literal and half appended is
    // harder to read than one that is all appended.
    let mut decls = vec![section("h_correction", "Correction")];
    // Where it runs. Not a suite2p option — suite2p takes CUDA if it finds it —
    // but "it was slower than I expected" and "it gave a different answer" are
    // worth being able to ask separately.
    decls.push(
        ParamDecl::new(
            "backend",
            "Compute",
            ParamKind::Choice {
                default: Backend::all()
                    .iter()
                    .position(|b| *b == d.backend)
                    .unwrap_or(1),
                options: Backend::all()
                    .iter()
                    .map(|b| b.label().to_string())
                    .collect(),
            },
        )
        .help("The answer is the same on each; only how many frames are in flight differs."),
    );
    decls.push(
        ParamDecl::new(
            "nonrigid",
            "Non-rigid",
            ParamKind::Bool {
                default: d.nonrigid,
            },
        )
        .help(
            "Correct tissue that deforms rather than merely sliding: a shift per \
             block, interpolated to a shift per pixel. Measured on top of the rigid \
             correction rather than instead of it, and slower. Its own settings \
             appear below when it is on.",
        ),
    );

    // ---- what is registered, and how far it may move -----------------------

    decls.push(section("h_align", "Alignment"));
    if info.channels > 1 {
        decls.push(
            ParamDecl::new(
                "align_by_chan2",
                "Align by channel 2",
                ParamKind::Bool {
                    default: d.align_by_chan2,
                },
            )
            .help(
                "Measure the shifts on the second channel — usually the structural \
                 one — rather than the first. Applied to every channel either way.",
            ),
        );
    }
    decls.push(
        ParamDecl::new(
            "maxregshift",
            "Max shift",
            ParamKind::Float {
                default: d.maxregshift,
                min: 0.01,
                max: 0.5,
            },
        )
        .help(
            "The largest shift allowed, as a fraction of the smaller frame \
             dimension. It has to cover the motion plus wherever the reference \
             landed — a shift beyond it is clipped, which reads as nearly working.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "smooth_sigma",
            "Spatial smoothing",
            ParamKind::Float {
                default: d.smooth_sigma,
                min: 0.0,
                max: 10.0,
            },
        )
        .help("~1 suits two-photon; 3-5 is recommended for one-photon."),
    );
    // Next to the smoothing it has to stay above: the help says so.
    decls.push(
        ParamDecl::new(
            "spatial_taper",
            "Edge taper",
            ParamKind::Float {
                default: d.spatial_taper,
                min: 0.0,
                max: 200.0,
            },
        )
        .help(
            "How much of the border to fade before correlating. Keep it well above \
             3x the spatial smoothing, or the smoothing reaches into the FFT's wrap.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "norm_frames",
            "Normalize frames",
            ParamKind::Bool {
                default: d.norm_frames,
            },
        )
        .help(
            "Clip to the reference's 1st and 99th percentile before correlating, so \
             one bright speck cannot pull the peak.",
        ),
    );

    // ---- how the reference is built ----------------------------------------

    decls.push(section("h_reference", "Reference"));
    decls.push(
        ParamDecl::new(
            "nimg_init",
            "Reference frames",
            ParamKind::Int {
                default: d.nimg_init as i64,
                min: 2,
                max: 5000,
            },
        )
        .help("How many frames are sampled to build the reference image."),
    );
    decls.push(
        ParamDecl::new(
            "two_step_registration",
            "Two-step registration",
            ParamKind::Bool {
                default: d.two_step_registration,
            },
        )
        .help("Register, then register the result again. For low-SNR recordings."),
    );

    // ---- the non-rigid grid, only when the switch above is on --------------

    // The rigid pass never reads any of these four. Offered while they are
    // dead, the usual outcome is someone raising the block shift, seeing no
    // difference, and concluding the non-rigid correction does not work.
    if chosen.bool("nonrigid", d.nonrigid) {
        decls.push(section("h_nonrigid", "Non-rigid grid"));
        decls.push(
            ParamDecl::new(
                "block_size",
                "Block size",
                ParamKind::Int {
                    default: d.block_size[0] as i64,
                    min: 16,
                    max: 512,
                },
            )
            .help("The square block the field is divided into, in pixels."),
        );
        decls.push(
            ParamDecl::new(
                "maxregshiftNR",
                "Max block shift",
                ParamKind::Float {
                    default: d.maxregshift_nr,
                    min: 0.0,
                    max: 50.0,
                },
            )
            .help("How far a block may move on top of the rigid shift."),
        );
        decls.push(
            ParamDecl::new(
                "snr_thresh",
                "Block SNR threshold",
                ParamKind::Float {
                    default: d.snr_thresh,
                    min: 1.0,
                    max: 5.0,
                },
            )
            .help(
                "A block below this is smoothed against its neighbours until it clears \
                 the bar. 1.0 means no smoothing.",
            ),
        );
        decls.push(
            ParamDecl::new(
                "subpixel",
                "Subpixel precision",
                ParamKind::Int {
                    default: d.subpixel as i64,
                    min: 1,
                    max: 50,
                },
            )
            .help(
                "Block shifts are resolved to 1/this of a pixel. suite2p's rigid pass \
                 is whole pixels, and so is this one.",
            ),
        );
    }

    // ---- a recording too dim to register frame by frame --------------------

    decls.push(section("h_dim", "Low signal"));
    decls.push(
        ParamDecl::new(
            "smooth_sigma_time",
            "Temporal smoothing",
            ParamKind::Float {
                default: d.smooth_sigma_time,
                min: 0.0,
                max: 10.0,
            },
        )
        .help(
            "Smooth the correlation maps along time before taking the peak — for a \
             recording too dim to locate frame by frame. Smoothed within a batch, \
             so it is the one setting the batch size can change the answer through.",
        ),
    );
    // Below the temporal smoothing, which its help refers to as "above".
    decls.push(
        ParamDecl::new(
            "batch_size",
            "Batch size",
            ParamKind::Int {
                default: d.batch_size as i64,
                min: 1,
                max: 2000,
            },
        )
        .help(
            "How many frames are processed at once. A memory and scheduling knob \
             only, except through the temporal smoothing above.",
        ),
    );

    // ---- the scanner's own artefact ----------------------------------------

    decls.push(section("h_scanner", "Scanner"));
    // suite2p's rule: a non-zero fixed offset wins and the measurement never
    // runs. So the switch is offered only while there is no fixed offset to
    // beat it — rather than shown next to the thing that silently overrides
    // it, which is how the rule gets discovered the hard way.
    if chosen.int("bidiphase", d.bidiphase as i64) == 0 {
        decls.push(
            ParamDecl::new(
                "do_bidiphase",
                "Correct bidirectional phase",
                ParamKind::Bool {
                    default: d.do_bidiphase,
                },
            )
            .help(
                "Measure and undo the comb a resonant scanner leaves. Offered only \
                 while the offset below is 0, because suite2p's rule is that a fixed \
                 offset wins and the measurement is then never run.",
            ),
        );
    }
    // Below the switch, which its help refers to as "below".
    decls.push(
        ParamDecl::new(
            "bidiphase",
            "Fixed bidirectional phase offset",
            ParamKind::Int {
                default: d.bidiphase as i64,
                min: -20,
                max: 20,
            },
        )
        .help("A known scanner offset, in pixels. 0 means measure it instead."),
    );

    // ---- what is reported afterwards ---------------------------------------

    decls.push(section("h_report", "Reporting"));
    decls.push(
        ParamDecl::new(
            "th_badframes",
            "Bad-frame threshold",
            ParamKind::Float {
                default: d.th_badframes,
                min: 0.0,
                max: 10.0,
            },
        )
        .help(
            "How far a frame may deviate before it is reported as an outlier. Smaller \
             flags more. Flagged frames are counted, never discarded — which frames to \
             drop is yours to decide.",
        ),
    );
    decls
}

/// Read the dialog back. Anything absent keeps suite2p's default.
pub(super) fn settings_from(params: &Params) -> Settings {
    let d = Settings::default();
    let block = params.int("block_size", d.block_size[0] as i64).max(1) as usize;
    Settings {
        align_by_chan2: params.bool("align_by_chan2", d.align_by_chan2),
        nimg_init: params.int("nimg_init", d.nimg_init as i64).max(2) as usize,
        maxregshift: params.float("maxregshift", d.maxregshift),
        smooth_sigma: params.float("smooth_sigma", d.smooth_sigma),
        smooth_sigma_time: params.float("smooth_sigma_time", d.smooth_sigma_time),
        spatial_taper: params.float("spatial_taper", d.spatial_taper),
        norm_frames: params.bool("norm_frames", d.norm_frames),
        two_step_registration: params.bool("two_step_registration", d.two_step_registration),
        batch_size: params.int("batch_size", d.batch_size as i64).max(1) as usize,
        do_bidiphase: params.bool("do_bidiphase", d.do_bidiphase),
        bidiphase: params.int("bidiphase", d.bidiphase as i64) as i32,
        nonrigid: params.bool("nonrigid", d.nonrigid),
        // One control for a square block, which is what everyone uses. suite2p
        // takes a pair; a non-square block is expressible there and has no
        // sensible dialog here, so it is offered as one number rather than two
        // that are almost always equal.
        block_size: [block, block],
        maxregshift_nr: params.float("maxregshiftNR", d.maxregshift_nr),
        snr_thresh: params.float("snr_thresh", d.snr_thresh),
        subpixel: params.int("subpixel", d.subpixel as i64).max(1) as usize,
        th_badframes: params.float("th_badframes", d.th_badframes),
        backend: *Backend::all()
            .get(params.choice("backend", 1))
            .unwrap_or(&Backend::MultiThread),
        ..d
    }
}

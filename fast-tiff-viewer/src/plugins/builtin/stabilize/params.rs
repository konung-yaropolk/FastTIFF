//! The dialog, and reading it back into [`Settings`].
//!
//! Split from the run for one reason: the controls are the *whole* of suite2p's
//! registration options, and a list that long buried in the middle of the
//! algorithm makes both harder to read. Every key here is the name suite2p uses
//! in `default_ops`, so a value copied from a lab's `ops.npy` means the same
//! thing.

use fasttiff_plugin_api::{ImageInfo, ParamDecl, ParamKind, Params};
use suite2p_registration::{Backend, Settings};

/// Build the dialog for a stack of this shape.
pub(super) fn declare(info: &ImageInfo) -> Vec<ParamDecl> {
    let d = Settings::default();
    let mut decls = Vec::new();

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

    if info.channels > 1 {
        decls.push(
            ParamDecl::new(
                "align_by_chan2",
                "Align by channel 2 (align_by_chan2)",
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
            "nimg_init",
            "Reference frames (nimg_init)",
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
            "maxregshift",
            "Max shift (maxregshift)",
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
            "Spatial smoothing (smooth_sigma)",
            ParamKind::Float {
                default: d.smooth_sigma,
                min: 0.0,
                max: 10.0,
            },
        )
        .help("~1 suits two-photon; 3-5 is recommended for one-photon."),
    );
    decls.push(
        ParamDecl::new(
            "smooth_sigma_time",
            "Temporal smoothing (smooth_sigma_time)",
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
    decls.push(
        ParamDecl::new(
            "spatial_taper",
            "Edge taper (spatial_taper)",
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
            "Normalize frames (norm_frames)",
            ParamKind::Bool {
                default: d.norm_frames,
            },
        )
        .help(
            "Clip to the reference's 1st and 99th percentile before correlating, so \
             one bright speck cannot pull the peak.",
        ),
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
    decls.push(
        ParamDecl::new(
            "batch_size",
            "Batch size (batch_size)",
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

    decls.push(
        ParamDecl::new(
            "do_bidiphase",
            "Correct bidirectional phase (do_bidiphase)",
            ParamKind::Bool {
                default: d.do_bidiphase,
            },
        )
        .help(
            "Measure and undo the comb a resonant scanner leaves. Ignored when a \
             fixed offset is given below, which is suite2p's rule.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "bidiphase",
            "Fixed bidi offset (bidiphase)",
            ParamKind::Int {
                default: d.bidiphase as i64,
                min: -20,
                max: 20,
            },
        )
        .help("A known scanner offset, in pixels. 0 means measure it instead."),
    );

    decls.push(
        ParamDecl::new(
            "nonrigid",
            "Non-rigid (nonrigid)",
            ParamKind::Bool {
                default: d.nonrigid,
            },
        )
        .help(
            "Correct tissue that deforms rather than merely sliding: a shift per \
             block, interpolated to a shift per pixel. Measured on top of the rigid \
             correction rather than instead of it, and slower.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "block_size",
            "Block size (block_size)",
            ParamKind::Int {
                default: d.block_size[0] as i64,
                min: 16,
                max: 512,
            },
        )
        .help("The square block the field is divided into, in pixels. Non-rigid only."),
    );
    decls.push(
        ParamDecl::new(
            "maxregshiftNR",
            "Max block shift (maxregshiftNR)",
            ParamKind::Float {
                default: d.maxregshift_nr,
                min: 0.0,
                max: 50.0,
            },
        )
        .help("How far a block may move on top of the rigid shift. Non-rigid only."),
    );
    decls.push(
        ParamDecl::new(
            "snr_thresh",
            "Block SNR threshold (snr_thresh)",
            ParamKind::Float {
                default: d.snr_thresh,
                min: 1.0,
                max: 5.0,
            },
        )
        .help(
            "A block below this is smoothed against its neighbours until it clears \
             the bar. 1.0 means no smoothing. Non-rigid only.",
        ),
    );
    decls.push(
        ParamDecl::new(
            "subpixel",
            "Subpixel precision (subpixel)",
            ParamKind::Int {
                default: d.subpixel as i64,
                min: 1,
                max: 50,
            },
        )
        .help(
            "Shifts are resolved to 1/this of a pixel. Non-rigid only — suite2p's \
             rigid pass is whole pixels, and so is this one.",
        ),
    );

    decls.push(
        ParamDecl::new(
            "th_badframes",
            "Bad-frame threshold (th_badframes)",
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

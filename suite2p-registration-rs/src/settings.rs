// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.
//
// Ported from suite2p (https://github.com/MouseLand/suite2p), Copyright © 2023
// Howard Hughes Medical Institute, authored by Carsen Stringer and Marius
// Pachitariu, and licensed GPL-3.

//! Registration parameters, named as suite2p names them.

/// Where the arithmetic runs.
///
/// The phase correlation is the same in every case — what differs is how many
/// frames are in flight at once. A frame's shift does not depend on any other
/// frame's, which is what makes this a free choice rather than a trade.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Backend {
    /// One frame at a time. Slowest, and the one to reach for when a result
    /// looks wrong — it removes scheduling from the list of suspects.
    SingleThread,
    /// A frame per core, through rayon.
    #[default]
    MultiThread,
    /// On the graphics card.
    Gpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::SingleThread => "Single-thread CPU",
            Backend::MultiThread => "Multi-thread CPU",
            Backend::Gpu => "GPU",
        }
    }

    /// Every backend, in the order a selector should offer them.
    pub fn all() -> [Backend; 3] {
        [Backend::SingleThread, Backend::MultiThread, Backend::Gpu]
    }

    /// Whether this backend can run a frame of this size.
    ///
    /// The GPU path is a radix-2 FFT, so it takes power-of-two frames only —
    /// 512x512 yes, 1024x768 no. And it is only compiled in with the `gpu`
    /// feature.
    ///
    /// Checked rather than silently downgraded: a run that said GPU and used
    /// the processor is indistinguishable from a slow one, and there would be
    /// no way to tell which had happened.
    pub fn available_for(self, ly: usize, lx: usize) -> bool {
        self.unavailable_reason(ly, lx).is_none()
    }

    /// Why it cannot run, for a caller to show.
    pub fn unavailable_reason(self, _ly: usize, _lx: usize) -> Option<String> {
        match self {
            Backend::Gpu => {
                #[cfg(not(feature = "gpu"))]
                {
                    Some(
                        "this build has no GPU backend: it is behind the crate's `gpu`                          feature, which was not enabled"
                            .to_string(),
                    )
                }
                #[cfg(feature = "gpu")]
                {
                    if !crate::gpu::size_supported(_ly, _lx) {
                        Some(format!(
                            "the GPU backend needs power-of-two frame sizes (its FFT is                              radix-2); this stack is {_lx}x{_ly}. Use a CPU backend."
                        ))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        }
    }
}

/// Registration parameters.
///
/// Every field carries the name it has in suite2p's `default_ops`, so a value
/// copied from a lab's `ops.npy` means here what it meant there. The defaults
/// are suite2p's own except where noted.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Settings {
    // --- what is registered -------------------------------------------
    /// `align_by_chan2`: measure the shifts on the second channel rather than
    /// the first. Applied to every channel either way — a two-channel recording
    /// is one field photographed twice, and the channels must not drift apart.
    pub align_by_chan2: bool,

    // --- the reference -------------------------------------------------
    /// `nimg_init`: how many frames are sampled to build the reference.
    pub nimg_init: usize,
    /// How many refinement passes build the reference. suite2p's `niter`, which
    /// its options do not expose — it is 8 in the source.
    pub reference_iterations: usize,

    // --- rigid ---------------------------------------------------------
    /// `maxregshift`: the largest shift allowed, as a fraction of the smaller
    /// frame dimension.
    ///
    /// It has to cover the motion *plus* wherever the reference landed. A shift
    /// asked for beyond this limit is clipped to it, which reads as the
    /// registration nearly working.
    pub maxregshift: f64,
    /// `smooth_sigma`: Gaussian smoothing of the reference's spectrum, in
    /// pixels. ~1 suits two-photon; 3–5 is recommended for one-photon.
    pub smooth_sigma: f64,
    /// `smooth_sigma_time`: Gaussian smoothing of the correlation maps along
    /// time. Nought is none. Worth raising on a recording so dim that a single
    /// frame cannot be located on its own.
    pub smooth_sigma_time: f64,
    /// `spatial_taper`: how much of the border to fade before correlating.
    /// Keep it well above `3 * smooth_sigma`, or the smoothing reaches into the
    /// FFT's wrap-around.
    pub spatial_taper: f64,
    /// `norm_frames`: clip the reference and the frames to the reference's 1st
    /// and 99th percentile before correlating, so one bright speck cannot pull
    /// the peak.
    pub norm_frames: bool,
    /// `two_step_registration`: register, then register the result again.
    /// For recordings too dim to locate in one pass.
    pub two_step_registration: bool,
    /// `batch_size`: how many frames are held and processed at once.
    ///
    /// Only a memory and scheduling knob — the answer does not depend on it,
    /// except through `smooth_sigma_time`, which smooths *within* a batch.
    pub batch_size: usize,

    // --- bidirectional phase --------------------------------------------
    /// `do_bidiphase`: measure and undo the resonant scanner's comb.
    pub do_bidiphase: bool,
    /// `bidiphase`: a known offset to apply instead of measuring one. A stated
    /// value wins over measuring, which is suite2p's rule.
    pub bidiphase: i32,

    // --- non-rigid -------------------------------------------------------
    /// `nonrigid`: correct tissue that deforms rather than merely sliding.
    pub nonrigid: bool,
    /// `block_size`: the block the field is divided into, in pixels.
    pub block_size: [usize; 2],
    /// `maxregshiftNR`: the largest *additional* shift a block may take,
    /// in pixels, on top of the rigid one.
    pub maxregshift_nr: f64,
    /// `snr_thresh`: a block whose correlation peak is less than this many
    /// times its next-best peak is smoothed against its neighbours until it
    /// clears the bar. 1.0 means no smoothing.
    pub snr_thresh: f64,
    /// `subpixel`: shifts are resolved to `1/subpixel` of a pixel.
    pub subpixel: usize,

    // --- quality ---------------------------------------------------------
    /// `th_badframes`: how far a frame may deviate before it is called bad and
    /// left out of the crop calculation. Smaller excludes more.
    pub th_badframes: f64,

    // --- where it runs ----------------------------------------------------
    /// Not a suite2p option — suite2p picks CUDA if it is there. Here it is a
    /// choice, because "it was slower than I expected" and "it gave a different
    /// answer" are questions worth being able to answer separately.
    pub backend: Backend,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            align_by_chan2: false,
            nimg_init: 300,
            reference_iterations: 8,
            maxregshift: 0.1,
            smooth_sigma: 1.15,
            smooth_sigma_time: 0.0,
            spatial_taper: 50.0,
            norm_frames: true,
            two_step_registration: false,
            batch_size: 100,
            do_bidiphase: false,
            bidiphase: 0,
            nonrigid: false,
            block_size: [64, 64],
            maxregshift_nr: 10.0,
            snr_thresh: 1.25,
            subpixel: 10,
            th_badframes: 1.0,
            backend: Backend::default(),
        }
    }
}

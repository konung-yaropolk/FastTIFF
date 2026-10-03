//! `Deconvolution > Deconvolve`: undo a measured or computed blur.
//!
//! The dialog is the union of what ImageJ's deconvolution plugins ask for, and
//! the work is done one `(channel, timepoint)` volume at a time: read it,
//! embed it in a padded grid, run the chosen algorithm, crop it back.
//!
//! # Why a volume at a time
//!
//! A 3-D PSF describes how light from one point spreads *through* the stack,
//! most of it into the slices above and below. Deconvolving slice by slice
//! with a 2-D PSF therefore cannot remove out-of-focus light — the thing most
//! people are deconvolving to remove — because the information about where it
//! came from is in the axis being ignored. Slice-by-slice is offered because
//! it is fast, because a single-plane image has no other option, and because
//! ImageJ offers it; it is not the default when there is a Z axis to use.
//!
//! # The axis may not be the one it is called
//!
//! `resolve_dimensions` folds a single-timepoint z-stack into *frames*,
//! because far more files shaped that way are movies and the movie reading is
//! right far more often. The consequence here is specific and would otherwise
//! be silent: an ordinary 101-slice confocal acquisition arrives as 101 frames
//! of one slice, so "3D (whole volume)" on it is a hundred and one 2-D runs —
//! correct for what the host says the file is, and not what the user meant.
//! Hence [`Shape::VolumeAlongT`], and hence the note the run logs whenever it
//! sees that shape.
//!
//! Channels are deconvolved independently with the same PSF. Strictly a PSF is
//! wavelength-dependent and each channel deserves its own, which is what the
//! two-run workflow is for: deconvolve, then deconvolve the other channel with
//! the other PSF. Doing it in one pass would need a PSF per channel in the
//! dialog, and the honest version of that is more fields than it is worth.

use super::algorithms::{self, Method, Settings, Stopped};
use super::fft::{Dims, Transform};
use super::grid::{self, Edge, Grid, Operator};
use super::{heading, load_psf, Psf, MAX_WORKING_BYTES, MENU};
use crate::plugins::builtin::stack_tools::{deliver, in_new_window, Store};
use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, ParamKind, Params, Plane, PlaneData, Plugin,
    PluginError, PluginInfo,
};

/// Deconvolve the open stack.
pub struct Deconvolve;

/// Where the PSF comes from.
const SOURCE_FILE: usize = 0;

/// What goes into one transform together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// One plane at a time.
    TwoD,
    /// All of a timepoint's Z slices.
    VolumeAlongZ,
    /// All of a slice's timepoints — for a z-stack that opened as a movie.
    VolumeAlongT,
}

impl Plugin for Deconvolve {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.deconvolve.run", "Deconvolve…")
            .menu_path(MENU)
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Deconvolve with a PSF image: Richardson-Lucy, Wiener, Tikhonov, \
                 Landweber, Van Cittert, Tikhonov-Miller or MRNSD.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let info = host.image();
        let volume = info.slices.max(1) > 1;
        // What has been chosen already. Empty the first time, which reads
        // back as the declared defaults, so one code path produces both the
        // first dialog and every one after a change.
        let chosen = host.pending_params();
        let from_file = chosen.choice("psf_source", SOURCE_FILE) == SOURCE_FILE;
        let method = *Method::ALL
            .get(chosen.choice("method", 0))
            .unwrap_or(&Method::RichardsonLucy);

        let mut v = vec![
            heading("h_psf", "Point spread function"),
            ParamDecl::new(
                "psf_source",
                "PSF from",
                ParamKind::Choice {
                    default: 0,
                    options: vec!["A TIFF file".into(), "A Gaussian, from its width".into()],
                },
            )
            .help(
                "A measured or computed PSF is what makes this quantitative. The Gaussian \
                 is for when you have neither and want to see what the data does.",
            ),
        ];

        if from_file {
            v.extend([
                ParamDecl::new(
                    "psf_path",
                    "PSF file",
                    ParamKind::Path {
                        default: String::new(),
                        save: false,
                    },
                )
                .help(
                    "A TIFF whose slices are the PSF Z planes. Deconvolution > Generate \
                     PSF makes one; a bead image works too.",
                ),
                // Both of these are questions about a file. A Gaussian built
                // here is centred and normalised by construction, so asking
                // would be asking about something that cannot be otherwise.
                ParamDecl::new(
                    "psf_origin",
                    "PSF origin",
                    ParamKind::Choice {
                        default: 0,
                        options: vec!["Brightest voxel".into(), "Geometric centre".into()],
                    },
                )
                .help(
                    "Which voxel of the PSF is the emitter. Every voxel this is out by \
                     shifts the whole result by the same amount. The brightest voxel is \
                     right for a measured PSF; the centre is right for a computed one \
                     with strong aberration, where the peak is genuinely off-centre.",
                ),
                ParamDecl::new(
                    "normalise_psf",
                    "Normalize PSF to sum 1",
                    ParamKind::Bool { default: true },
                )
                .help(
                    "Leave this on unless you know why not: a PSF that does not sum to 1 \
                     scales the whole result by the amount it is out by.",
                ),
            ]);
        } else {
            v.push(ParamDecl::new(
                "sigma_xy",
                "Gaussian sigma, XY (px)",
                ParamKind::Float {
                    default: 2.0,
                    min: 0.2,
                    max: 50.0,
                },
            ));
            // A flat stack has no axial width to give it.
            if volume {
                v.push(
                    ParamDecl::new(
                        "sigma_z",
                        "Gaussian sigma, Z (px)",
                        ParamKind::Float {
                            default: 1.5,
                            min: 0.0,
                            max: 50.0,
                        },
                    )
                    .help("0 makes the Gaussian flat in Z, which is a 2-D PSF."),
                );
            }
        }

        v.extend([
            heading("h_algo", "Algorithm"),
            ParamDecl::new(
                "method",
                "Method",
                ParamKind::Choice {
                    default: 0,
                    options: Method::ALL.iter().map(|m| m.label().to_string()).collect(),
                },
            )
            .help(
                "Richardson-Lucy is the default for fluorescence: it assumes Poisson \
                 photon noise, which is what a camera gives. Wiener is one shot and \
                 instant. The rest are here because ImageJ plugins offer them.",
            ),
        ]);

        // From here the controls belong to particular methods. Showing all of
        // them at once was the alternative, and it asks the user to know
        // which three of eight the method they picked will read — which is
        // exactly the knowledge the dialog is supposed to supply.
        if method.is_iterative() {
            v.push(
                ParamDecl::new(
                    "iterations",
                    "Iterations",
                    ParamKind::Int {
                        default: 10,
                        min: 1,
                        max: 1000,
                    },
                )
                .help("10 to 50 is the usual range."),
            );
        }
        if matches!(
            method,
            Method::RegularisedInverse | Method::TikhonovMiller | Method::RichardsonLucyTv
        ) {
            let label = if method == Method::RichardsonLucyTv {
                "Total-variation weight"
            } else {
                "Regularization (Tikhonov)"
            };
            v.push(
                ParamDecl::new(
                    "lambda",
                    label,
                    ParamKind::Float {
                        default: 0.01,
                        min: 0.000_001,
                        max: 1.0,
                    },
                )
                .help(
                    "How hard to insist the answer is smooth. For total variation, 0.001 \
                     to 0.01 is the useful range and more makes staircase artefacts.",
                ),
            );
        }
        if method == Method::Wiener {
            v.push(
                ParamDecl::new(
                    "gamma",
                    "Wiener gamma (noise / signal)",
                    ParamKind::Float {
                        default: 0.001,
                        min: 0.000_001,
                        max: 1.0,
                    },
                )
                .help("Larger is smoother. The Wiener filter only parameter."),
            );
        }
        if matches!(
            method,
            Method::Landweber | Method::VanCittert | Method::TikhonovMiller
        ) {
            v.push(
                ParamDecl::new(
                    "step",
                    "Step size",
                    ParamKind::Float {
                        default: 1.0,
                        min: 0.01,
                        max: 2.0,
                    },
                )
                .help("Above 1 is faster and may diverge, which the run will say if it does."),
            );
        }
        if method == Method::NaiveInverse {
            v.push(
                ParamDecl::new(
                    "threshold",
                    "Inverse-filter cutoff",
                    ParamKind::Float {
                        default: 0.001,
                        min: 0.000_001,
                        max: 1.0,
                    },
                )
                .help(
                    "As a fraction of the PSF spectrum peak. The naive inverse filter \
                     gives up below this instead of dividing by nearly nothing.",
                ),
            );
        }
        v.push(
            ParamDecl::new(
                "low_pass",
                "Low-pass smoothing (px)",
                ParamKind::Float {
                    default: 0.0,
                    min: 0.0,
                    max: 10.0,
                },
            )
            .help(
                "Gaussian smoothing as sigma in pixels, applied to every iteration and \
                 to a one-shot result. 0 is off. Bob Dougherty remedy for an iterative \
                 method turning noise into texture.",
            ),
        );
        if method.is_iterative() {
            v.push(
                ParamDecl::new(
                    "stop_delta",
                    "Stop when change is under (%)",
                    ParamKind::Float {
                        default: 0.01,
                        min: 0.0,
                        max: 10.0,
                    },
                )
                .help("0 runs every iteration asked for."),
            );
        }
        v.push(
            ParamDecl::new(
                "nonneg",
                "Force non-negative",
                ParamKind::Bool { default: true },
            )
            .help(
                "An intensity below zero is not a measurement. Richardson-Lucy and MRNSD \
                 cannot produce one anyway.",
            ),
        );

        v.extend([
            heading("h_image", "Image"),
            ParamDecl::new(
                "dimensionality",
                "Work in",
                ParamKind::Choice {
                    default: 0,
                    options: vec![
                        "3D (whole volume)".into(),
                        "2D (slice by slice)".into(),
                        "3D, treating frames as Z".into(),
                    ],
                },
            )
            .help(
                "3-D is what removes out-of-focus light, and needs a PSF with Z slices. \
                 2-D treats every plane as its own image. The third is for a z-stack \
                 that opened as a time series — FastTIFF reads a single-timepoint \
                 z-stack as frames, because far more files shaped that way are movies, \
                 and this is how to say that yours is not.",
            ),
            ParamDecl::new(
                "boundary",
                "Edges",
                ParamKind::Choice {
                    default: 0,
                    options: Edge::ALL.iter().map(|e| e.label().to_string()).collect(),
                },
            )
            .help(
                "An FFT convolution wraps, so the image is padded and the padding is \
                 filled this way. Mirroring is the one that invents least.",
            ),
        ]);
        // Nothing to choose when there is one timepoint.
        if info.frames.max(1) > 1 {
            v.push(ParamDecl::new(
                "scope",
                "Apply to",
                ParamKind::Choice {
                    default: 0,
                    options: vec!["The whole stack".into(), "The current timepoint".into()],
                },
            ));
        }
        v.extend([
            ParamDecl::new(
                "output",
                "Output",
                ParamKind::Choice {
                    default: 0,
                    options: vec!["32-bit float".into(), "Same as the source".into()],
                },
            )
            .help(
                "Deconvolution produces values between and beyond the ones it was given; \
                 float keeps them. Matching the source rounds and clips, which is what \
                 you want only when something downstream needs the original type.",
            ),
            in_new_window(),
        ]);
        v
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let info = host.image();
        if info.plane_len() == 0 {
            return Err(PluginError::unsupported("the stack has no pixels"));
        }
        let (channels, slices, frames) =
            (info.channels.max(1), info.slices.max(1), info.frames.max(1));

        // ---- the PSF -----------------------------------------------------
        let mut psf = if params.choice("psf_source", SOURCE_FILE) == SOURCE_FILE {
            let p = load_psf(params.text("psf_path", ""))?;
            host.log(&format!(
                "PSF {}x{}x{} from {}",
                p.dims.x,
                p.dims.y,
                p.dims.z,
                params.text("psf_path", "")
            ));
            p
        } else {
            let (sxy, sz) = (params.float("sigma_xy", 2.0), params.float("sigma_z", 1.5));
            let p = gaussian_psf(sxy, if slices > 1 { sz } else { 0.0 });
            host.log(&format!(
                "Gaussian PSF {}x{}x{}, sigma {sxy} px laterally and {sz} px axially",
                p.dims.x, p.dims.y, p.dims.z
            ));
            p
        };

        // Which axis the volume runs along, which is a real question here
        // and not a formality: `resolve_dimensions` folds a single-timepoint
        // z-stack into frames, so the Z axis of a 101-slice acquisition is
        // very often the T axis by the time a plugin sees it.
        let shape = match params.choice("dimensionality", 0) {
            1 => Shape::TwoD,
            2 => Shape::VolumeAlongT,
            _ if slices == 1 => Shape::TwoD,
            _ => Shape::VolumeAlongZ,
        };
        let two_d = shape == Shape::TwoD;
        if two_d && psf.dims.z > 1 {
            // Not an error: a 3-D PSF is the one most people have, and its
            // central slice is the in-focus one, which is exactly the 2-D PSF.
            let mid = psf.dims.z / 2;
            let plane = psf.dims.x * psf.dims.y;
            psf.data = psf.data[mid * plane..(mid + 1) * plane].to_vec();
            psf.dims = Dims::new(psf.dims.x, psf.dims.y, 1);
            host.log(&format!(
                "working slice by slice, so only the PSF's in-focus slice ({}) is used",
                mid + 1
            ));
        }
        if !two_d && psf.dims.z == 1 && slices > 1 {
            host.log(
                "the PSF has one slice, so this cannot remove out-of-focus light — it is a \
                 lateral sharpening of every slice",
            );
        }
        if psf.dims.x > info.width as usize || psf.dims.y > info.height as usize {
            return Err(PluginError::unsupported(format!(
                "the PSF is {}x{} and the image is only {}x{} — crop the PSF, or generate \
                 a smaller one",
                psf.dims.x, psf.dims.y, info.width, info.height
            )));
        }
        let origin = if params.choice("psf_origin", 0) == 0 {
            grid::brightest(&psf.data, psf.dims)
        } else {
            grid::centre(psf.dims)
        };
        if params.bool("normalise_psf", true) && !grid::normalise(&mut psf.data) {
            return Err(PluginError::unsupported(
                "that PSF sums to zero or less, so it cannot be normalized — it is \
                 probably not a PSF, or it needs its background subtracting first",
            ));
        }

        // ---- the settings ------------------------------------------------
        let method = *Method::ALL
            .get(params.choice("method", 0))
            .unwrap_or(&Method::RichardsonLucy);
        let settings = Settings {
            method,
            iterations: params.int("iterations", 10).max(1) as usize,
            lambda: params.float("lambda", 0.01) as f32,
            gamma: params.float("gamma", 0.001) as f32,
            step: params.float("step", 1.0) as f32,
            threshold: params.float("threshold", 0.001) as f32,
            low_pass: params.float("low_pass", 0.0) as f32,
            stop_delta: params.float("stop_delta", 0.01) as f32,
            nonneg: params.bool("nonneg", true),
        };
        let edge = *Edge::ALL
            .get(params.choice("boundary", 0))
            .unwrap_or(&Edge::Mirror);

        // ---- what to work on ---------------------------------------------
        let whole = params.choice("scope", 0) == 0;
        let times: Vec<usize> = if whole {
            (0..frames).collect()
        } else {
            vec![host.view().frame_index.min(frames - 1)]
        };
        let out_frames = times.len();
        let store = if params.choice("output", 0) == 0 {
            Store::F32
        } else {
            Store::of(info.pixel_type)
        };

        // One job is one volume: the planes that go into a transform
        // together, in depth order. Building the list up front rather than
        // nesting loops is what lets all three shapes share one body — and it
        // is the only place the plane order has to be got right.
        let jobs: Vec<Vec<(usize, usize)>> = match shape {
            Shape::TwoD => (0..out_frames)
                .flat_map(|ti| (0..slices).map(move |z| vec![(z, ti)]))
                .collect(),
            Shape::VolumeAlongZ => (0..out_frames)
                .map(|ti| (0..slices).map(|z| (z, ti)).collect())
                .collect(),
            Shape::VolumeAlongT => (0..slices)
                .map(|z| (0..out_frames).map(|ti| (z, ti)).collect())
                .collect(),
        };
        let depth = jobs.first().map(|j| j.len()).unwrap_or(1);
        // On the user's choice, not on `shape`: a stack with one Z slice has
        // already been forced to `TwoD` by the time `shape` is decided, so a
        // test of `shape` here could never fire — which is exactly the case
        // the message exists for.
        if params.choice("dimensionality", 0) == 0 && slices == 1 && out_frames > 1 {
            host.log(
                "this stack has one Z slice and many frames, so a whole-volume run is a \
                 2-D one per frame. If those frames are really Z slices, choose \
                 \"3D, treating frames as Z\".",
            );
        }
        let total_chunks = (channels * jobs.len()).max(1);
        let src = Dims::new(info.width as usize, info.height as usize, depth);

        let plan = Grid::plan(src, psf.dims, edge);
        // Two complex working buffers at eight bytes a voxel and four real
        // ones at four, which is what the operator and the iterative methods
        // hold between them. Reported rather than merely enforced: a run that
        // is about to take four gigabytes should say so before it takes them,
        // not after the machine starts swapping.
        let bytes = plan.padded().len() as u64 * 32;
        host.log(&format!(
            "working grid {}x{}x{}, about {:.1} GB",
            plan.padded().x,
            plan.padded().y,
            plan.padded().z,
            bytes as f64 / (1u64 << 30) as f64
        ));
        if bytes > MAX_WORKING_BYTES {
            return Err(PluginError::unsupported(format!(
                "deconvolving this needs a {}x{}x{} grid — about {:.1} GB of working \
                 memory, more than this will allocate. Deconvolve a crop, a single \
                 timepoint, or slice by slice.",
                plan.padded().x,
                plan.padded().y,
                plan.padded().z,
                bytes as f64 / (1u64 << 30) as f64
            )));
        }
        let placed = grid::place_psf(plan.padded(), &psf.data, psf.dims, origin);
        let mut op = Operator::new(Transform::new(plan.padded()), &placed);

        host.log(&format!(
            "{} on {}x{}x{} padded to {}x{}x{} ({})",
            method.label(),
            src.x,
            src.y,
            src.z,
            plan.padded().x,
            plan.padded().y,
            plan.padded().z,
            edge.label()
        ));

        // ---- the work ----------------------------------------------------
        let mut planes: Vec<Option<PlaneData>> = vec![None; channels * slices * out_frames];
        let mut volume = Vec::new();
        let mut padded = Vec::new();
        let mut cropped = Vec::new();
        let mut buf = Vec::new();
        let mut done = 0usize;
        let mut reported: Option<Stopped> = None;

        for c in 0..channels {
            for job in &jobs {
                volume.clear();
                for &(z, ti) in job {
                    if !host.progress(done as f32 / total_chunks as f32) {
                        return Ok(Outcome::Cancelled);
                    }
                    host.read_plane_f32(Plane::new(c, z, times[ti]), &mut buf)?;
                    volume.extend_from_slice(&buf);
                }
                debug_assert_eq!(job.len(), src.z);

                plan.embed(&volume, &mut padded);
                let base = done;
                let result = algorithms::run(&mut op, &padded, &settings, &mut |f| {
                    host.progress((base as f32 + f) / total_chunks as f32)
                });
                let Some(result) = result else {
                    return Ok(Outcome::Cancelled);
                };
                reported.get_or_insert(result.stopped.clone());
                plan.crop(&result.image, &mut cropped);

                // Back to `xyczt`, from whatever order the volume ran in.
                let plane = src.x * src.y;
                for (&(z, ti), slice) in job.iter().zip(cropped.chunks_exact(plane)) {
                    let at = ti * (slices * channels) + z * channels + c;
                    planes[at] = Some(store.plane(slice.to_vec()));
                }
                done += 1;
            }
        }

        match reported {
            Some(Stopped::Converged { iterations, delta }) => host.log(&format!(
                "stopped after {iterations} iteration(s): the estimate moved {delta:.4}%, \
                 under the threshold"
            )),
            Some(Stopped::Diverged(it)) => {
                return Err(PluginError::failed(format!(
                    "the estimate diverged on iteration {it}. Lower the step size, or \
                     raise the regularization."
                )))
            }
            Some(Stopped::Completed(n)) if n > 0 => host.log(&format!("ran {n} iteration(s)")),
            _ => {}
        }

        let planes: Vec<PlaneData> = planes
            .into_iter()
            .map(|p| p.ok_or_else(|| PluginError::failed("a plane was not computed")))
            .collect::<Result<_, _>>()?;

        let image = ImageResult {
            width: info.width,
            height: info.height,
            channels,
            slices,
            frames: out_frames,
            pixel_type: store.pixel_type(),
            planes,
            channel_colors: Vec::new(),
            metadata: Some(host.stack_info().clone()),
            name: format!("{}-{}", host.stack_info().name, method.tag()),
        };
        image.validate()?;
        Ok(deliver(image, params))
    }
}

/// A separable Gaussian PSF, for when there is no file.
///
/// Three sigma each way, which holds 99.7% of it; the rest would cost grid
/// size for nothing. An odd extent so the peak lands on a voxel.
fn gaussian_psf(sigma_xy: f64, sigma_z: f64) -> Psf {
    let extent = |s: f64| -> usize {
        if s <= 0.0 {
            return 1;
        }
        2 * (3.0 * s).ceil().max(1.0) as usize + 1
    };
    let dims = Dims::new(extent(sigma_xy), extent(sigma_xy), extent(sigma_z));
    let (cx, cy, cz) = (dims.x / 2, dims.y / 2, dims.z / 2);
    let g = |d: f64, s: f64| {
        if s <= 0.0 {
            1.0
        } else {
            (-0.5 * (d / s).powi(2)).exp()
        }
    };
    let mut data = vec![0.0f32; dims.len()];
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let v = g(x as f64 - cx as f64, sigma_xy)
                    * g(y as f64 - cy as f64, sigma_xy)
                    * g(z as f64 - cz as f64, sigma_z);
                data[dims.at(x, y, z)] = v as f32;
            }
        }
    }
    Psf { data, dims }
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;

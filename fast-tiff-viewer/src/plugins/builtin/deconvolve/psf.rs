//! `Deconvolution > Generate PSF`: a point spread function from the optics.
//!
//! The dialog is ImageJ's *Diffraction PSF 3D* and the BIG group's *PSF
//! Generator* put together: a model, the optics, the sampling, and what to
//! normalise to. The numerics are all in [`super::optics`]; this is the dialog
//! and the bookkeeping.
//!
//! # Defaults come from the open stack
//!
//! The pixel size and Z step default to the stack's own calibration, because a
//! PSF is only usable against an image sampled the way it was. Getting that
//! wrong is the quiet failure of theoretical deconvolution — a PSF computed at
//! 100 nm pixels and applied to a 65 nm image is simply the wrong kernel, and
//! nothing downstream can tell. Seeding the fields from the file is what makes
//! the common case right without the user having to look the numbers up.

use super::fft::Dims;
use super::optics::{self, Mode, Model, Optics};
use super::{fresh_info, heading, MAX_VOXELS, MENU};
use crate::plugins::builtin::stack_tools::{deliver, in_new_window};
use fasttiff_plugin_api::{
    HostContext, ImageResult, Outcome, ParamDecl, ParamKind, Params, PixelType, PlaneData, Plugin,
    PluginError, PluginInfo,
};

/// Compute a theoretical PSF.
pub struct GeneratePsf;

/// The defaults for sampling, taken from the open stack where it says.
fn sampling_defaults(host: &dyn HostContext) -> (f64, f64, i64) {
    let info = host.stack_info();
    let image = host.image();
    // The file's spacing is in whatever `unit` says; micrometres is what
    // microscopy writes and what the models want. A file calibrated in
    // anything else is left to the defaults rather than silently scaled.
    let micron = info
        .unit
        .as_deref()
        .map(|u| {
            let u = u.trim().to_ascii_lowercase();
            u == "micron" || u == "microns" || u == "um" || u == "micrometer" || u == "\u{b5}m"
        })
        .unwrap_or(false);
    let pixel = info
        .spacing
        .x
        .filter(|v| micron && *v > 0.0 && v.is_finite())
        .map(|v| v * 1000.0)
        .unwrap_or(100.0);
    let step = info
        .spacing
        .z
        .filter(|v| micron && *v > 0.0 && v.is_finite())
        .map(|v| v * 1000.0)
        .unwrap_or(250.0);
    // A PSF for a single-plane image is a single plane. Anything else would
    // make the user notice afterwards, when the deconvolution refuses.
    let slices = if image.slices.max(1) > 1 { 33 } else { 1 };
    (pixel, step, slices)
}

impl Plugin for GeneratePsf {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.deconvolve.psf", "Generate PSF…")
            .menu_path(MENU)
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Compute a theoretical point spread function — Gaussian, Born & Wolf, \
                 Gibson & Lanni or geometric defocus — from the objective and the sampling.",
            )
    }

    fn params(&self, host: &dyn HostContext) -> Vec<ParamDecl> {
        let (pixel, step, slices) = sampling_defaults(host);
        // What the dialog holds already, so it can follow what has been
        // chosen in it. Empty the first time it opens, which reads back as
        // the declared defaults — so the first dialog and the dialog after a
        // change are produced by the same code.
        let chosen = host.pending_params();
        let model = *Model::ALL
            .get(chosen.choice("model", 0))
            .unwrap_or(&Model::BornWolf);
        // The two that solve the diffraction integral. The other two have no
        // wavefront, so an aberration of it means nothing to them.
        let diffracts = matches!(model, Model::BornWolf | Model::GibsonLanni);

        let mut v = vec![
            heading("h_model", "Model"),
            ParamDecl::new(
                "model",
                "Model",
                ParamKind::Choice {
                    default: 0,
                    options: Model::ALL.iter().map(|m| m.label().to_string()).collect(),
                },
            )
            .help(
                "Born & Wolf is the diffraction PSF of a clean widefield system and the \
                 one to start with. Gibson & Lanni adds the refractive-index mismatches, \
                 which is what matters when imaging deep into an aqueous specimen with an \
                 oil lens. Gaussian is a crude but instant approximation. Defocus is \
                 geometric optics with no diffraction at all.",
            ),
            ParamDecl::new(
                "mode",
                "Imaging mode",
                ParamKind::Choice {
                    default: 0,
                    options: Mode::ALL.iter().map(|m| m.label().to_string()).collect(),
                },
            )
            .help(
                "Confocal and two-photon square the PSF, which is the small-pinhole \
                 approximation. For two-photon, enter the excitation wavelength.",
            ),
            heading("h_optics", "Objective"),
            ParamDecl::new(
                "na",
                "Numerical aperture",
                ParamKind::Float {
                    default: 1.4,
                    min: 0.05,
                    max: 1.7,
                },
            ),
        ];

        // The geometric model has no wavelength to diffract.
        if model != Model::Defocus {
            v.push(
                ParamDecl::new(
                    "wavelength",
                    "Wavelength (nm)",
                    ParamKind::Float {
                        default: 520.0,
                        min: 200.0,
                        max: 1500.0,
                    },
                )
                .help("Emission, except for two-photon, where it is the excitation."),
            );
        }
        v.push(
            ParamDecl::new(
                "ni",
                "Immersion refractive index",
                ParamKind::Float {
                    default: 1.515,
                    min: 1.0,
                    max: 2.0,
                },
            )
            .help("1.0 air, 1.33 water, 1.406 glycerol, 1.515 oil."),
        );
        if diffracts {
            v.push(
                ParamDecl::new(
                    "sa",
                    "Spherical aberration (um)",
                    ParamKind::Float {
                        default: 0.0,
                        min: -20.0,
                        max: 20.0,
                    },
                )
                .help(
                    "Longitudinal spherical aberration at full aperture: how far the \
                     marginal rays focus from the paraxial ones. Zero for a \
                     well-corrected system.",
                ),
            );
        }

        // The index-mismatch terms, which only one model has anywhere to put.
        if model == Model::GibsonLanni {
            v.extend([
                heading("h_gl", "Specimen and coverslip"),
                ParamDecl::new(
                    "ns",
                    "Specimen refractive index",
                    ParamKind::Float {
                        default: 1.33,
                        min: 1.0,
                        max: 2.0,
                    },
                ),
                ParamDecl::new(
                    "depth",
                    "Depth below coverslip (um)",
                    ParamKind::Float {
                        default: 0.0,
                        min: 0.0,
                        max: 500.0,
                    },
                )
                .help(
                    "The PSF of an oil lens looking into water gets worse with depth, and \
                     this is the parameter that says so.",
                ),
                ParamDecl::new(
                    "ng",
                    "Coverslip refractive index",
                    ParamKind::Float {
                        default: 1.515,
                        min: 1.0,
                        max: 2.0,
                    },
                ),
                ParamDecl::new(
                    "tg",
                    "Coverslip thickness, actual (um)",
                    ParamKind::Float {
                        default: 170.0,
                        min: 0.0,
                        max: 2000.0,
                    },
                ),
                ParamDecl::new(
                    "tg0",
                    "Coverslip thickness, design (um)",
                    ParamKind::Float {
                        default: 170.0,
                        min: 0.0,
                        max: 2000.0,
                    },
                )
                .help(
                    "Only the difference from the actual thickness matters; equal means \
                     no error.",
                ),
                ParamDecl::new(
                    "ti0",
                    "Working distance (um)",
                    ParamKind::Float {
                        default: 150.0,
                        min: 1.0,
                        max: 20000.0,
                    },
                ),
            ]);
        }

        v.extend([
            heading("h_sampling", "Sampling"),
            ParamDecl::new(
                "pixel",
                "Pixel size (nm)",
                ParamKind::Float {
                    default: pixel,
                    min: 1.0,
                    max: 10000.0,
                },
            )
            .help("Taken from the open stack when it is calibrated in microns."),
            ParamDecl::new(
                "step",
                "Z step (nm)",
                ParamKind::Float {
                    default: step,
                    min: 1.0,
                    max: 100000.0,
                },
            ),
            ParamDecl::new(
                "width",
                "Width (px)",
                ParamKind::Int {
                    default: 65,
                    min: 3,
                    max: 4096,
                },
            )
            .help("An odd size puts the peak on a voxel rather than between four."),
            ParamDecl::new(
                "height",
                "Height (px)",
                ParamKind::Int {
                    default: 65,
                    min: 3,
                    max: 4096,
                },
            ),
            ParamDecl::new(
                "slices",
                "Slices",
                ParamKind::Int {
                    default: slices,
                    min: 1,
                    max: 2048,
                },
            )
            .help("1 for a 2-D PSF. Defaults to a volume when the open stack is one."),
            heading("h_out", "Output"),
            ParamDecl::new(
                "normalise",
                "Normalization",
                ParamKind::Choice {
                    default: 0,
                    options: vec![
                        "Sum = 1 (preserves intensity)".into(),
                        "Maximum = 1".into(),
                        "None (raw model values)".into(),
                    ],
                },
            )
            .help(
                "Sum = 1 is what a deconvolution wants: convolving by it leaves total \
                 intensity unchanged. The others are for looking at.",
            ),
            in_new_window(),
        ]);
        v
    }

    fn run(&mut self, host: &mut dyn HostContext, params: &Params) -> Result<Outcome, PluginError> {
        let model = *Model::ALL
            .get(params.choice("model", 0))
            .unwrap_or(&Model::BornWolf);
        let mode = *Mode::ALL
            .get(params.choice("mode", 0))
            .unwrap_or(&Mode::Widefield);

        let pixel_nm = params.float("pixel", 100.0);
        let step_nm = params.float("step", 250.0);
        let o = Optics {
            na: params.float("na", 1.4),
            lambda: params.float("wavelength", 520.0) / 1000.0,
            ni: params.float("ni", 1.515),
            ni0: params.float("ni", 1.515),
            ng: params.float("ng", 1.515),
            ng0: params.float("ng", 1.515),
            ns: params.float("ns", 1.33),
            ti0: params.float("ti0", 150.0),
            tg: params.float("tg", 170.0),
            tg0: params.float("tg0", 170.0),
            depth: params.float("depth", 0.0),
            sa: params.float("sa", 0.0),
            pixel: pixel_nm / 1000.0,
            step: step_nm / 1000.0,
        };
        if o.na >= o.ni {
            return Err(PluginError::unsupported(format!(
                "a numerical aperture of {:.2} is impossible in a medium of index {:.3} — \
                 NA is n sin(theta), so it cannot reach n",
                o.na, o.ni
            )));
        }

        let dims = Dims::new(
            params.int("width", 65).max(1) as usize,
            params.int("height", 65).max(1) as usize,
            params.int("slices", 1).max(1) as usize,
        );
        if dims.len() > MAX_VOXELS {
            return Err(PluginError::unsupported(format!(
                "{}x{}x{} is {} voxels, more than this will generate at once",
                dims.x,
                dims.y,
                dims.z,
                dims.len()
            )));
        }

        // Said before the work rather than after, because it is the thing most
        // likely to be wrong and the run can take a minute.
        let nyquist = o.lambda / (4.0 * o.na) * 1000.0;
        if pixel_nm > nyquist {
            host.log(&format!(
                "note: {pixel_nm:.0} nm pixels undersample this objective — Nyquist is \
                 about {nyquist:.0} nm at NA {:.2} and {:.0} nm. Deconvolution cannot \
                 recover detail the camera did not sample.",
                o.na,
                o.lambda * 1000.0
            ));
        }
        if model == Model::Gaussian {
            let (sxy, sz) = o.gaussian_sigma();
            host.log(&format!(
                "Gaussian sigma {:.3} um laterally ({:.2} px), {:.3} um axially ({:.2} px)",
                sxy,
                sxy / o.pixel,
                sz,
                sz / o.step
            ));
        }

        let mut data = match optics::generate(model, mode, &o, dims, &mut |f| host.progress(f)) {
            Some(d) => d,
            None => return Ok(Outcome::Cancelled),
        };

        match params.choice("normalise", 0) {
            0 => {
                if !super::grid::normalise(&mut data) {
                    return Err(PluginError::failed(
                        "the model produced nothing to normalise — check the optics",
                    ));
                }
            }
            1 => {
                let peak = data.iter().copied().fold(0.0f32, f32::max);
                if peak > 0.0 {
                    for v in data.iter_mut() {
                        *v /= peak;
                    }
                }
            }
            _ => {}
        }

        let name = format!(
            "psf-{}-{}nm-na{:.2}",
            model.tag(),
            params.float("wavelength", 520.0).round() as i64,
            o.na
        );
        host.log(&format!(
            "{} {}: {}x{}x{} at {pixel_nm:.0} nm / {step_nm:.0} nm",
            model.label(),
            mode.label(),
            dims.x,
            dims.y,
            dims.z
        ));

        let plane = dims.x * dims.y;
        let image = ImageResult {
            width: dims.x as u32,
            height: dims.y as u32,
            channels: 1,
            slices: dims.z,
            frames: 1,
            pixel_type: PixelType::F32,
            planes: data
                .chunks_exact(plane)
                .map(|c| PlaneData::F32(c.to_vec()))
                .collect(),
            channel_colors: Vec::new(),
            metadata: Some(fresh_info(name, pixel_nm / 1000.0, step_nm / 1000.0)),
            name: format!(
                "psf-{}-{}nm",
                model.tag(),
                params.float("wavelength", 520.0).round() as i64
            ),
        };
        image.validate()?;
        Ok(deliver(image, params))
    }
}

#[cfg(test)]
#[path = "psf_tests.rs"]
mod tests;

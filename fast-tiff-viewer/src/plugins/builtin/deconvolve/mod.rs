//! Deconvolution: the `Deconvolution` menu's two entries.
//!
//! [`GeneratePsf`] computes a point spread function from the optics.
//! [`Deconvolve`] uses one — generated here, measured from beads, or supplied
//! by anything else — to undo the blur it describes.
//!
//! They are two plugins rather than one because they are two decisions. A PSF
//! is a property of the microscope and is made once for a given objective,
//! wavelength and sampling; which algorithm to run and for how many iterations
//! is a property of the experiment and gets changed a dozen times. ImageJ
//! splits them the same way, and for the same reason.
//!
//! # The parts
//!
//! * [`fft`] — the separable 3-D transform everything convolves through.
//! * [`grid`] — padding, boundary handling, PSF placement, and the blur
//!   operator and its adjoint. The edge handling lives here because it is a
//!   property of the grid, and getting it wrong is the most common way a
//!   deconvolution produces a bright rim and a convincing-looking lie.
//! * [`optics`] — the theoretical PSF models.
//! * [`algorithms`] — the nine ways of inverting the blur.
//! * [`psf`] and [`run`] — the two plugins, which are dialogs and bookkeeping
//!   over the above and contain no numerics of their own.
//!
//! # Reading a PSF from a file
//!
//! [`Deconvolve`] opens a TIFF the user picked, which no other plugin here
//! does: the rest of them work on what the host handed them. The contract has
//! no "read me this other image" call and should not grow one lightly — the
//! host would have to decide what a plugin may open, and a filter that reads
//! arbitrary paths is a different security proposition from one that does not.
//!
//! So this reads it itself, with `fast-tiff-lib`, which is a published crate
//! any plugin may depend on and not a reach into the viewer. That keeps the
//! property this directory is built on: every plugin here could be moved out
//! to a `.dll` without changing a line. A third-party deconvolution plugin
//! would do exactly this.

use fasttiff_plugin_api::{ParamDecl, ParamKind, PluginError, Spacing, StackInfo};

mod algorithms;
mod fft;
mod grid;
mod optics;
mod par;
mod psf;
mod run;

pub use psf::GeneratePsf;
pub use run::Deconvolve;

use fft::Dims;

/// The submenu both plugins sit in.
const MENU: &str = "Deconvolution";

/// The most voxels a PSF may have.
///
/// Separate from the working-memory budget below and far smaller, because a
/// PSF is a kernel: one larger than this is a file picked by mistake, not a
/// large job.
const MAX_VOXELS: usize = 1 << 26;

/// The most working memory a deconvolution may allocate.
///
/// Counted as 32 bytes a voxel of the *padded* grid — two complex buffers at
/// eight and four real ones at four, which is what the operator and the
/// iterative methods hold between them.
///
/// Six gigabytes is deliberately generous. A 1024x1024 stack of 101 slices,
/// which is an ordinary confocal acquisition and not a stress test, pads to
/// about 140 million voxels and so needs four and a half; a cap that refused
/// it would make whole-volume deconvolution a feature for small images only.
/// The run says what it is about to take before it takes it, so the number is
/// the user's to act on rather than a surprise.
const MAX_WORKING_BYTES: u64 = 6 << 30;

/// A heading in a dialog that would otherwise be twenty controls in a column.
///
/// Drawn by the host as bold text with a rule under it, and it also ends the
/// previous group — see `plugins_ui::groups`. Both dialogs here need it for
/// the same reason: they are long because the thing they do has many knobs,
/// not because they are badly designed, and the remedy for a long form is
/// grouping rather than hiding.
fn heading(key: &str, text: &str) -> ParamDecl {
    ParamDecl::new(key, text, ParamKind::Section)
}

/// A PSF, as pixels and a shape.
#[derive(Debug)]
pub(crate) struct Psf {
    pub data: Vec<f32>,
    pub dims: Dims,
}

/// Read a PSF from a TIFF.
///
/// Every IFD is a Z slice. That is the right reading for a PSF whatever the
/// file's own metadata says it is — a PSF has no channels and no timepoints,
/// and a stack of 32 planes is a 32-slice PSF whether it was saved as slices
/// or as frames. A file with several samples per pixel contributes its first,
/// with a note, because an RGB PSF is a mistake rather than a format.
pub(crate) fn load_psf(path: &str) -> Result<Psf, PluginError> {
    use fast_tiff_lib::{SampleFormat, TiffStack};

    if path.trim().is_empty() {
        return Err(PluginError::unsupported(
            "no PSF file chosen — pick one with the … button, or make one with \
             Deconvolution > Generate PSF",
        ));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| PluginError::failed(format!("could not read the PSF file {path}: {e}")))?;
    let tiff = TiffStack::from_bytes(bytes)
        .map_err(|e| PluginError::failed(format!("{path} is not a TIFF this can read: {e:#}")))?;

    let first = tiff
        .frames
        .first()
        .ok_or_else(|| PluginError::failed(format!("{path} has no images in it")))?;
    let dims = Dims::new(
        first.width as usize,
        first.height as usize,
        tiff.frames.len(),
    );
    if dims.x == 0 || dims.y == 0 {
        return Err(PluginError::failed(format!("{path} has a zero dimension")));
    }
    if tiff
        .frames
        .iter()
        .any(|f| f.width != first.width || f.height != first.height)
    {
        return Err(PluginError::failed(format!(
            "the slices of {path} are not all the same size, so it cannot be one PSF"
        )));
    }
    if dims.len() > MAX_VOXELS {
        return Err(PluginError::unsupported(format!(
            "that PSF is {}x{}x{} voxels, which is more than this can hold",
            dims.x, dims.y, dims.z
        )));
    }

    let mut data = Vec::with_capacity(dims.len());
    let (mut u8buf, mut u16buf, mut f32buf) = (Vec::new(), Vec::new(), Vec::new());
    for (i, frame) in tiff.frames.iter().enumerate() {
        let fail = |e: anyhow::Error| {
            PluginError::failed(format!("decoding slice {} of the PSF: {e:#}", i + 1))
        };
        match frame.bits_per_sample {
            32 | 64 => {
                fast_tiff_lib::read_plane_f32_into(
                    &tiff.data,
                    frame,
                    tiff.byte_order,
                    0,
                    &mut f32buf,
                )
                .map_err(fail)?;
                data.extend_from_slice(&f32buf);
            }
            8 | 4 => {
                fast_tiff_lib::read_plane_u8_into(
                    &tiff.data,
                    frame,
                    tiff.byte_order,
                    0,
                    &mut u8buf,
                )
                .map_err(fail)?;
                data.extend(u8buf.iter().map(|&v| v as f32));
            }
            _ => {
                fast_tiff_lib::read_plane_u16_into(
                    &tiff.data,
                    frame,
                    tiff.byte_order,
                    None,
                    0,
                    &mut u16buf,
                )
                .map_err(fail)?;
                // That reader offsets signed samples into unsigned display
                // space so that signed and unsigned files look alike. A PSF is
                // arithmetic, not a display, so the offset comes back off.
                if frame.sample_format == SampleFormat::SignedInt {
                    data.extend(u16buf.iter().map(|&v| v as f32 - 32768.0));
                } else {
                    data.extend(u16buf.iter().map(|&v| v as f32));
                }
            }
        }
    }
    if data.len() != dims.len() {
        return Err(PluginError::failed(format!(
            "the PSF decoded to {} samples, expected {}x{}x{}",
            data.len(),
            dims.x,
            dims.y,
            dims.z
        )));
    }
    Ok(Psf { data, dims })
}

/// The metadata a generated image carries.
///
/// Built fresh rather than cloned from the open stack. A PSF is not that
/// stack: its spacing is whatever was asked for, its name is its own, and
/// carrying the source's `description` would hand the written file a second
/// opinion about its own shape — the ImageJ dialect keeps the first occurrence
/// of a key, and a carried `slices=` has been exactly that bug before.
fn fresh_info(name: String, pixel_um: f64, step_um: f64) -> StackInfo {
    StackInfo {
        path: None,
        name,
        mode: fasttiff_plugin_api::DisplayMode::Grayscale,
        unit: Some("micron".into()),
        spacing: Spacing {
            x: Some(pixel_um),
            y: Some(pixel_um),
            z: Some(step_um),
        },
        frame_interval_s: None,
        channel_names: Vec::new(),
        calibration: None,
        description: None,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

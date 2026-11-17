//! A built-in importer for NIfTI-1 volumes — `.nii` and `.nii.gz`.
//!
//! NIfTI is what MRI leaves the scanner as, once anything has touched it: FSL,
//! SPM, AFNI, FreeSurfer and `dcm2niix` all read and write it, and a research
//! dataset distributed for reuse is almost always `.nii.gz`. It is also an
//! unusually pleasant format to read — a fixed 348-byte header and then the
//! voxels, with no container, no index and no compression of its own.
//!
//! # What arrives in the viewer
//!
//! A NIfTI volume is `x * y * z * t`, which maps onto this viewer's axes
//! exactly: `z` becomes the slice axis and `t` the frame axis, so a structural
//! scan opens as a z-stack you scroll through and an fMRI series opens as a 4-D
//! stack with both. The voxel sizes from `pixdim` become the stack's spacing,
//! so the 3-D view and the readouts are in millimetres rather than in voxels.
//!
//! # What it does not do, and why
//!
//! **No reorientation.** NIfTI carries the transform from voxel indices to
//! scanner coordinates in its qform and sform matrices, and this reader ignores
//! them — see [`header`] for the longer version. The short version is that this
//! viewer shows an array of planes, not a volume in a patient's frame, so there
//! is nothing here that could honour a direction matrix. The planes are
//! therefore in the file's own storage order, which is what every array-level
//! tool shows and is *not* necessarily radiological or neurological convention.
//! If left and right matter for what you are doing, they matter more than this
//! viewer can promise, and the orientation has to come from a tool that models
//! it.
//!
//! **Desktop only**, as every importer is: an `ImportRequest` carries a
//! filesystem path and a browser tab has none.
//!
//! **NIfTI-1 only.** NIfTI-2 is detected and refused by name rather than
//! mis-read, which it would be: its 540-byte header puts different fields where
//! NIfTI-1's are, so the first 348 bytes of one parse into a plausible,
//! entirely wrong image.

mod header;

use fasttiff_plugin_api::{
    Confidence, DisplayMode, FileType, ImageResult, ImportHost, ImportRequest, ImportResult,
    Importer, PixelType, PlaneData, PluginError, PluginInfo, Spacing, StackInfo,
};
use header::{Datatype, Hdr};
use std::path::Path;

#[derive(Default)]
pub struct Nifti;

/// The gzip member header, which is how a `.nii.gz` announces itself.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Whether this path names the gzipped form.
///
/// On the whole name rather than on [`Path::extension`], which for
/// `brain.nii.gz` answers `gz` and knows nothing about the `.nii` in front of
/// it. That is also why [`Nifti::file_types`] has to claim `gz` outright, and
/// why [`Nifti::probe`] has to be the thing that declines someone else's
/// tarball.
fn named_nii_gz(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.to_ascii_lowercase())
        .is_some_and(|n| n.ends_with(".nii.gz"))
}

/// Whether this path names the plain form.
fn named_nii(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("nii"))
}

/// Read the file, inflating it if it is gzipped.
///
/// Decided on the content rather than the name: a `.nii` that is really
/// gzipped is common enough (a careless `gzip` leaves the name alone if told
/// to) and costs two bytes to detect, where trusting the name would produce
/// "not a NIfTI file" about a file that is one.
fn read_maybe_gzipped(path: &Path) -> Result<Vec<u8>, PluginError> {
    use std::io::Read;

    let raw = std::fs::read(path)
        .map_err(|e| PluginError::failed(format!("reading {}: {e}", path.display())))?;
    if raw.first_chunk::<2>() != Some(&GZIP_MAGIC) {
        return Ok(raw);
    }
    // `MultiGzDecoder`, not `GzDecoder`: a `.gz` may hold several concatenated
    // members, and `GzDecoder` stops after the first — which on such a file
    // yields a short read that looks exactly like a truncated volume. The
    // multi-member form is what `gzip` itself and Python's `gzip` module both
    // produce when appending, so it does occur.
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(&raw[..])
        .read_to_end(&mut out)
        .map_err(|e| {
            PluginError::failed(format!(
                "{} is gzipped but could not be decompressed: {e}",
                path.display()
            ))
        })?;
    Ok(out)
}

/// Pull one plane's voxels out of the volume and convert them.
///
/// `at` is the plane's byte offset and `n` its voxel count. NIfTI stores x
/// fastest, then y, then z, then t, so a plane *is* a contiguous run and this
/// never has to stride — which is the one thing that makes reading a 4-D
/// volume as a stack of planes free rather than a transpose.
fn plane_of(
    data: &[u8],
    at: usize,
    n: usize,
    datatype: Datatype,
    swap: bool,
) -> Result<PlaneData, PluginError> {
    let width = datatype.bytes();
    let end = at
        .checked_add(n * width)
        .ok_or_else(|| PluginError::failed("the NIfTI volume's size overflows"))?;
    let raw = data
        .get(at..end)
        .ok_or_else(|| PluginError::failed("the NIfTI file ends in the middle of a plane"))?;

    // One closure per width, so the byte-order branch is taken once per plane
    // rather than once per voxel.
    macro_rules! read {
        ($t:ty, $w:expr, $f:expr) => {{
            let f = $f;
            raw.chunks_exact($w)
                .map(|c| {
                    let b = <[u8; $w]>::try_from(c).expect("chunks_exact gives exact chunks");
                    f(if swap {
                        <$t>::from_be_bytes(b)
                    } else {
                        <$t>::from_le_bytes(b)
                    })
                })
                .collect()
        }};
    }

    Ok(match datatype {
        Datatype::U8 => PlaneData::U8(raw.to_vec()),
        // Colour goes through `colour_planes_of`, which has to emit one plane
        // per component. Reaching here with it would mean a packed plane
        // nothing could window, so it is a bug rather than a bad file.
        Datatype::Rgb24 | Datatype::Rgba32 => {
            return Err(PluginError::failed(
                "internal: a colour NIfTI plane was read through the                  single-channel path",
            ))
        }
        // `i8` to the `U16` lane with the sign preserved as `i16` bits, which
        // is how the contract carries signed samples: there is no `PlaneData`
        // arm for them, and `PixelType::I16` is what tells the host to read
        // these sixteen bits as signed.
        Datatype::I8 => PlaneData::U16(raw.iter().map(|&b| b as i8 as i16 as u16).collect()),
        Datatype::U16 => PlaneData::U16(read!(u16, 2, |v| v)),
        Datatype::I16 => PlaneData::U16(read!(i16, 2, |v: i16| v as u16)),
        Datatype::F32 => PlaneData::F32(read!(f32, 4, |v| v)),
        Datatype::I32 => PlaneData::F32(read!(i32, 4, |v| v as f32)),
        Datatype::U32 => PlaneData::F32(read!(u32, 4, |v| v as f32)),
        Datatype::F64 => PlaneData::F32(read!(f64, 8, |v| v as f32)),
    })
}

/// One plane's worth of interleaved colour, split into a plane per component.
///
/// NIfTI stores a colour voxel as its components in a row — `r g b r g b` for
/// RGB24, `r g b a` for RGBA32 — so a plane of `n` voxels is `n * channels`
/// bytes and each component is every `channels`-th one.
///
/// Never byte-swapped, however the rest of the file is: the components are
/// single bytes, so there is nothing to reverse. A reader that swapped by
/// bytes-per-voxel would reverse each voxel's components and turn every red
/// pixel blue on a big-endian file — which is why [`Datatype::swaps`] exists
/// separately from [`Datatype::bytes`], mirroring the `swapsize` column
/// `nifti1_io.c` keeps beside its `nbyper`.
fn colour_planes_of(
    data: &[u8],
    at: usize,
    n_px: usize,
    channels: usize,
) -> Result<Vec<PlaneData>, PluginError> {
    let need = n_px
        .checked_mul(channels)
        .and_then(|v| at.checked_add(v))
        .ok_or_else(|| PluginError::failed("the NIfTI volume's size overflows"))?;
    let raw = data
        .get(at..need)
        .ok_or_else(|| PluginError::failed("the NIfTI file ends in the middle of a plane"))?;
    Ok((0..channels)
        .map(|c| {
            PlaneData::U8(
                raw.iter()
                    .skip(c)
                    .step_by(channels)
                    .copied()
                    .take(n_px)
                    .collect(),
            )
        })
        .collect())
}

/// What the host should call the samples of a plane built by [`plane_of`].
fn pixel_type(datatype: Datatype) -> PixelType {
    match datatype {
        // A colour component is a byte, and each becomes its own channel.
        Datatype::U8 | Datatype::Rgb24 | Datatype::Rgba32 => PixelType::U8,
        Datatype::U16 => PixelType::U16,
        // Both signed types travel as `i16` bits; see `plane_of`.
        Datatype::I8 | Datatype::I16 => PixelType::I16,
        Datatype::F32 | Datatype::I32 | Datatype::U32 | Datatype::F64 => PixelType::F32,
    }
}

impl Importer for Nifti {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.nifti", "NIfTI (MRI)")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Read NIfTI-1 MRI volumes, plain or gzipped: z becomes the slice \
                 axis and t the frame axis, with voxel sizes from pixdim.",
            )
    }

    fn file_types(&self) -> Vec<FileType> {
        // `gz` as well as `nii`, because `brain.nii.gz` has `gz` for its
        // extension and the host matches on that one component. The cost is
        // that every `.gz` is offered to this importer; `probe` is what gives
        // the ones that are not NIfTI straight back.
        vec![FileType::new("NIfTI volume", &["nii", "gz"])]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        let gzipped = head.first_chunk::<2>() == Some(&GZIP_MAGIC);
        // A gzip member says nothing about what is inside it without
        // inflating, and inflating in a probe would mean decompressing every
        // `.gz` in a directory listing to answer a question about one file.
        // The name is the evidence available, and for `.nii.gz` it is good
        // evidence.
        if gzipped {
            return if named_nii_gz(path) {
                Confidence::Certain
            } else {
                // Somebody else's archive. Declined rather than attempted, so
                // the host's "nothing here can read this" is what the user
                // sees instead of a decompression failure.
                Confidence::No
            };
        }
        // Uncompressed: the header speaks for itself, both ways round, which
        // also means a NIfTI under the wrong name is still read.
        match Hdr::parse(head) {
            Ok(h) if &h.magic() == b"n+1\0" => Confidence::Certain,
            // A `.hdr`/`.img` pair is a NIfTI this reader cannot complete.
            // Claiming it would be the difference between a clear refusal and
            // falling through to the TIFF reader — see `import`.
            Ok(h) if &h.magic() == b"ni1\0" && named_nii(path) => Confidence::Maybe,
            _ if named_nii(path) => Confidence::Maybe,
            _ => Confidence::No,
        }
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let bytes = read_maybe_gzipped(&request.path)?;
        let hdr = Hdr::parse(&bytes)?;
        let shape = header::shape_of(&hdr)?;

        let n_px = (shape.width as usize)
            .checked_mul(shape.height as usize)
            .ok_or_else(|| PluginError::failed("the NIfTI plane size overflows"))?;
        let n_planes = shape
            .slices
            .checked_mul(shape.frames)
            .ok_or_else(|| PluginError::failed("the NIfTI volume's plane count overflows"))?;

        host.log(&format!(
            "NIfTI-1: {}x{}, {} slice(s), {} frame(s), {} channel(s), {:?}{}",
            shape.width,
            shape.height,
            shape.slices,
            shape.frames,
            shape.datatype.channels(),
            shape.datatype,
            if hdr.byte_swapped() {
                ", byte-swapped"
            } else {
                ""
            }
        ));
        if let Some(note) = shape.datatype.widened_from() {
            host.log(note);
        }
        if let Some((inter, slope)) = shape.calibration {
            host.log(&format!(
                "intensities are calibrated: value = {inter} + {slope} * stored"
            ));
        }

        // `xyczt` order: channel fastest, then z, then t. A greyscale volume
        // has one channel, so that is `z` within `t` — which is also the order
        // the file stores them in, and why this walks the volume forwards and
        // never has to stride. A colour volume's components come out of each
        // voxel in the same pass, which keeps them innermost as required.
        let channels = shape.datatype.channels();
        let mut planes = Vec::with_capacity(n_planes * channels);
        for i in 0..n_planes {
            if !host.progress(i as f32 / n_planes.max(1) as f32) {
                return Err(PluginError::failed("cancelled"));
            }
            let at = shape
                .data_at
                .checked_add(i * n_px * shape.datatype.bytes())
                .ok_or_else(|| PluginError::failed("the NIfTI volume's size overflows"))?;
            if channels > 1 {
                planes.extend(colour_planes_of(&bytes, at, n_px, channels)?);
            } else {
                planes.push(plane_of(
                    &bytes,
                    at,
                    n_px,
                    shape.datatype,
                    hdr.byte_swapped() && shape.datatype.swaps(),
                )?);
            }
        }
        host.progress(1.0);

        let name = request
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "volume.nii".into());

        Ok(ImportResult {
            image: ImageResult {
                width: shape.width,
                height: shape.height,
                channels,
                slices: shape.slices,
                frames: shape.frames,
                pixel_type: pixel_type(shape.datatype),
                planes,
                channel_colors: Vec::new(),
                metadata: None,
                name: name.clone(),
            },
            info: Some(StackInfo {
                path: Some(request.path.display().to_string()),
                name,
                // One channel of signal intensity, which is the truth about
                // what MRI is rather than a default — and a composite for the
                // colour types, whose channels are components of one picture
                // and are meaningless shown one at a time.
                mode: if channels > 1 {
                    DisplayMode::Composite
                } else {
                    DisplayMode::Grayscale
                },
                unit: shape.unit,
                spacing: Spacing {
                    x: shape.spacing.0,
                    y: shape.spacing.1,
                    z: shape.spacing.2,
                },
                frame_interval_s: shape.frame_interval_s,
                channel_names: Vec::new(),
                calibration: shape.calibration,
                // `descrip` is 80 bytes of whatever wrote the file — the
                // pipeline and its version, usually. Carried through as the
                // description so it is visible in the metadata panel rather
                // than discarded.
                description: shape.descrip,
            }),
        })
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

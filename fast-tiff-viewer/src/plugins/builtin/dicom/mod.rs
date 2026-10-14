//! A built-in importer for DICOM images — `.dcm`, one file or a whole series.
//!
//! DICOM is what comes off the scanner, and it is one file per *slice*: a
//! 200-slice MRI is 200 files in a directory, each a complete object with the
//! whole study's metadata repeated in it. So opening one file and showing one
//! slice would be technically correct and almost useless, and this reader
//! gathers the series instead — see [`series`].
//!
//! # Scope: uncompressed transfer syntaxes
//!
//! The pixel data is read for Implicit VR Little Endian, Explicit VR Little
//! Endian and Explicit VR Big Endian. Everything compressed — JPEG in its
//! several forms, JPEG-LS, JPEG 2000, RLE, deflated — is **refused by name**,
//! with the conversion command that would fix it.
//!
//! That is a real limit and worth being plain about. It is also the right place
//! to stop: each of those is a whole codec, two of them are patent-adjacent,
//! and `dcmdjpeg`, `gdcmconv --raw` and `dcm2niix` all exist and are what
//! people already use. Research MRI exported for analysis is usually
//! uncompressed; clinical PACS exports often are not.
//!
//! # What is deliberately not read
//!
//! **Anything identifying.** `PatientName`, `PatientID`, `PatientBirthDate`
//! and the rest are in every one of these files and none of them are touched.
//! The document is named after the file on disk, and the description comes from
//! `SeriesDescription` and `Modality`. A viewer that put a patient's name in a
//! window title would put it in screenshots, in bug reports and on projectors.
//!
//! **The patient coordinate system**, beyond using it to order the slices.
//! `ImageOrientationPatient` says how the grid sits in the scanner; this viewer
//! shows planes and a voxel size, so there is nothing here that could honour a
//! direction matrix. Left and right are therefore not guaranteed, exactly as
//! for [`nifti`](super::nifti).
//!
//! **Desktop only**, as every importer is: an `ImportRequest` carries a
//! filesystem path and a browser tab has none.

mod jpeg;
mod lossless;
mod palette;
mod parse;
mod pixels;
mod rle;
mod series;

use fasttiff_plugin_api::{
    Confidence, DisplayMode, FileType, ImageResult, ImportHost, ImportRequest, ImportResult,
    Importer, ParamDecl, ParamKind, PluginError, PluginInfo, Spacing, StackInfo,
};
use parse::{Codec, Found, Syntax, Tag};
use pixels::Format;
use std::path::Path;

pub struct Dicom;

/// The key of the "read the whole series" switch.
const WHOLE_SERIES: &str = "whole_series";

const MODALITY: Tag = (0x0008, 0x0060);
const SERIES_DESCRIPTION: Tag = (0x0008, 0x103E);
const SLICE_THICKNESS: Tag = (0x0018, 0x0050);
const SPACING_BETWEEN_SLICES: Tag = (0x0018, 0x0088);
const PIXEL_SPACING: Tag = (0x0028, 0x0030);

/// Every tag this reader copies out of a dataset.
///
/// One list, used for the file being opened and for each sibling considered for
/// the series, so the two cannot drift — a sibling judged on a tag the main
/// file was not read for would be a bug that only appeared on multi-series
/// directories.
fn wanted() -> Vec<Tag> {
    vec![
        MODALITY,
        SERIES_DESCRIPTION,
        SLICE_THICKNESS,
        SPACING_BETWEEN_SLICES,
        PIXEL_SPACING,
        pixels::ROWS,
        pixels::COLUMNS,
        pixels::SAMPLES_PER_PIXEL,
        pixels::PHOTOMETRIC,
        pixels::PLANAR_CONFIG,
        pixels::NUMBER_OF_FRAMES,
        pixels::BITS_ALLOCATED,
        pixels::BITS_STORED,
        pixels::HIGH_BIT,
        pixels::PIXEL_REPRESENTATION,
        pixels::RESCALE_INTERCEPT,
        pixels::RESCALE_SLOPE,
        series::SERIES_UID,
        series::INSTANCE_NUMBER,
        series::IMAGE_POSITION,
        series::IMAGE_ORIENTATION,
    ]
    .into_iter()
    // The colour table, for a palette image. Always asked for rather than
    // asked for conditionally: the dataset is walked once, before anything
    // knows whether this file is palette-colour, so the only chance to collect
    // them is on that pass.
    .chain(palette::TAGS)
    .collect()
}

/// One file, read far enough to know what it is.
///
/// No visibility modifier: a module's private items are visible to its own
/// descendants, and `series` is one. Marking it `pub(super)` made it reachable
/// from outside this plugin while the types in its fields were not, which is
/// exactly the mismatch the compiler complains about.
struct Slice {
    found: Found,
    syntax: Syntax,
    codec: Codec,
    /// The pixel bytes, already decoded if they needed to be.
    ///
    /// Owned rather than a range into the file, because a decoded frame does
    /// not exist in the file to point at. One entry per frame for a compressed
    /// file, where each had to be decoded separately; one entry holding every
    /// frame for a native one, which is the single contiguous value it is on
    /// disk.
    pixels: Pixels,
}

/// Where a file's samples came from.
enum Pixels {
    /// The value as stored: every frame, end to end.
    Native(Vec<u8>),
    /// One decoded frame each.
    Frames(Vec<Vec<u8>>),
}

impl Pixels {
    /// One frame's samples, and whether they are at an offset within a larger
    /// buffer.
    ///
    /// Native pixel data is addressed by frame *offset* because the frames
    /// share one buffer; a decoded frame is its own buffer and so is always at
    /// offset zero. Returning both rather than normalising keeps the native
    /// path from copying every frame out of the value it is already in.
    fn frame(&self, i: usize) -> Option<(&[u8], usize)> {
        match self {
            Pixels::Native(b) => Some((b.as_slice(), i)),
            Pixels::Frames(v) => v.get(i).map(|f| (f.as_slice(), 0)),
        }
    }
}

/// Read a file's metadata and get its pixels, decoding them if they are
/// compressed.
fn read_one(path: &Path) -> Result<Slice, PluginError> {
    let bytes = std::fs::read(path)
        .map_err(|e| PluginError::failed(format!("reading {}: {e}", path.display())))?;
    let (syntax, codec, data_at) = parse::meta(&bytes)?;
    let (found, native) = parse::scan(&bytes, data_at, syntax, &wanted())?;
    let big_endian = syntax == Syntax::ExplicitBe;

    // Before anything about the samples: an object with no pixel data at all is
    // not an image, and saying so beats complaining that it does not describe
    // the samples it does not have. A structured report has neither, and only
    // one of those two messages tells the reader what kind of file they opened.
    if found.encapsulated_at.is_none() && native.is_none() {
        return Err(PluginError::unsupported(format!(
            "{} has no pixel data. Some DICOM objects carry only measurements \
             or a structured report",
            path.display()
        )));
    }
    let fmt = Format::of(&found, big_endian)?;

    let pixels = match (codec, found.encapsulated_at, native) {
        // Native and contiguous, which is the ordinary case.
        (Codec::Native, None, Some(range)) => {
            let mut data = bytes[range].to_vec();
            if big_endian {
                pixels::to_little_endian(&mut data, fmt);
            }
            Pixels::Native(data)
        }
        // A compressed syntax, with the item stream a compressed syntax must
        // have.
        (codec, Some(at), _) => {
            let frames = parse::fragments(&bytes, at)?;
            let mut out = Vec::with_capacity(frames.len());
            for (i, frame) in frames.iter().enumerate() {
                out.push(match codec {
                    Codec::Rle => rle::decode(frame, fmt)?,
                    Codec::JpegBaseline | Codec::JpegLossless => jpeg::decode(frame, fmt)?,
                    Codec::Native => {
                        return Err(PluginError::failed(format!(
                            "{} declares an uncompressed transfer syntax but its \
                             pixel data is encapsulated, so the two disagree",
                            path.display()
                        )))
                    }
                });
                let _ = i;
            }
            Pixels::Frames(out)
        }
        // A compressed syntax whose pixel data is *not* encapsulated. The
        // standard requires it to be, so this is a file whose two halves
        // disagree — and decoding the value as though it were one fragment
        // would be a guess.
        (_, None, Some(_)) => {
            return Err(PluginError::failed(format!(
                "{} declares a compressed transfer syntax but its pixel data is \
                 stored as one plain value, which the encapsulation rules do \
                 not allow",
                path.display()
            )))
        }
        // Ruled out above, and kept as an arm rather than folded into a
        // catch-all so the match stays exhaustive — a catch-all here would
        // silently swallow a codec added later.
        (_, None, None) => unreachable!("checked before the format is read"),
    };
    Ok(Slice {
        found,
        syntax,
        codec,
        pixels,
    })
}

impl Importer for Dicom {
    fn info(&self) -> PluginInfo {
        PluginInfo::new("dev.fasttiff.dicom", "DICOM (MRI/CT)")
            .version(env!("CARGO_PKG_VERSION"))
            .author("FastTIFF")
            .description(
                "Read uncompressed DICOM images, gathering the whole slice \
                 series from the folder and ordering it by position.",
            )
    }

    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("DICOM image", &["dcm", "dicom"])]
    }

    fn probe(&self, path: &Path, head: &[u8]) -> Confidence {
        // The `DICM` marker at offset 128 is the only evidence a DICOM file
        // offers, and it is conclusive. A `.dcm` without it is something else
        // under a borrowed name.
        if parse::has_magic(head) {
            Confidence::Certain
        } else if !head.is_empty() {
            Confidence::No
        } else if self.file_types().iter().any(|t| t.matches(path)) {
            Confidence::Maybe
        } else {
            Confidence::No
        }
    }

    fn params(&self, _path: &Path) -> Vec<ParamDecl> {
        // Declared, and currently answered by its own default: the host does
        // not yet put an importer's dialog on screen (see `start_import` in the
        // app). Declaring it anyway is what makes this a switch the moment it
        // does, and documents that the gathering is a choice rather than a
        // quirk.
        vec![ParamDecl::new(
            WHOLE_SERIES,
            "Read the whole series from this folder",
            ParamKind::Bool { default: true },
        )
        .help(
            "On, every DICOM in the folder belonging to the same series is \
             gathered into one stack, ordered by slice position — which is what \
             a scanner's one-file-per-slice export needs. Off, only the chosen \
             file is read.",
        )]
    }

    fn import(
        &mut self,
        request: &ImportRequest,
        host: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        let first = read_one(&request.path)?;
        let big_endian = first.syntax == Syntax::ExplicitBe;
        let fmt = Format::of(&first.found, big_endian)?;

        // The series, or just this file. A multi-frame file is already a volume
        // and is never extended with its siblings: a cine loop plus its
        // neighbours would be two different axes stacked into one.
        // The colour table, when the samples are indices into one. Read from
        // the opened file: a series shares its palette, and a series whose
        // slices disagreed about their colour table would not be one stack.
        let table = fmt
            .palette
            .then(|| palette::Palette::of(&first.found, big_endian))
            .transpose()?;
        // What comes out: three channels for a palette image, whatever the file
        // says it stores per pixel.
        let out_channels = if table.is_some() { 3 } else { fmt.samples };

        let gather = request.params.bool(WHOLE_SERIES, true) && fmt.frames == 1;
        let paths = if gather {
            series::gather(&request.path, &first.found, host)
        } else {
            vec![request.path.clone()]
        };

        if let Some(what) = description(&first.found) {
            host.log(&what);
        }
        host.log(&format!(
            "DICOM {:?}: {}x{}, {} bit(s){}{}",
            first.syntax,
            fmt.columns,
            fmt.rows,
            fmt.bits_stored,
            if fmt.signed { " signed" } else { "" },
            match (fmt.samples, fmt.palette) {
                (_, true) => ", palette colour",
                (3, _) => ", colour",
                _ => "",
            }
        ));
        if fmt.inverted {
            // Not inverted here. The samples are what the file holds and the
            // rescale calibration is stated against them; flipping them would
            // make every reported value wrong in a way nothing downstream could
            // detect. `Filters > Invert` is one click and leaves the choice
            // visible.
            host.log(
                "this file is MONOCHROME1, where the lowest value is white — \
                 use Filters > Invert to show it the way it was intended",
            );
        }

        // Each slice's planes, in order. A greyscale series gives one plane per
        // slice, a colour one gives three, and `xyczt` wants the channels
        // innermost — which is the order they come out in.
        let mut planes = Vec::with_capacity(paths.len() * out_channels * fmt.frames);
        let total = paths.len().max(1);
        for (i, path) in paths.iter().enumerate() {
            if !host.progress(i as f32 / total as f32) {
                return Err(PluginError::failed("cancelled"));
            }
            // The file that was opened is already in hand; re-reading it would
            // double the largest read in the common single-file case.
            //
            // Matched by **path**, not by being first in the list. The list is
            // in slice order and the file the user picked is wherever its own
            // position puts it — usually not the front. Keying this on `i == 0`
            // gave the first slice of the stack the *opened* file's pixels,
            // which is a stack with one slice duplicated and one missing: a
            // plausible volume with a seam in it, and nothing reported.
            let reuse = *path == request.path;
            let slice = if reuse { None } else { Some(read_one(path)?) };
            let s = slice.as_ref().unwrap_or(&first);

            // Every slice has to agree about its shape, or they are not one
            // stack. A series with a scout image dropped in the middle of it is
            // the usual cause, and stacking it anyway would show one slice
            // stretched across the others' geometry.
            let this = Format::of(&s.found, s.syntax == Syntax::ExplicitBe)?;
            if (this.rows, this.columns, this.bits_allocated, this.samples)
                != (fmt.rows, fmt.columns, fmt.bits_allocated, fmt.samples)
            {
                return Err(PluginError::failed(format!(
                    "{} is {}x{} where the series is {}x{}, so they are not one \
                     stack. Re-open with the series switch off to see this file \
                     alone",
                    path.display(),
                    this.columns,
                    this.rows,
                    fmt.columns,
                    fmt.rows
                )));
            }

            // A decoder hands back the native interleaved layout, so the
            // file's `PlanarConfiguration` describes the arrangement of the
            // *compressed* data and no longer applies — RLE's segments are
            // planar by construction and are de-planarised on the way out, and
            // a JPEG decoder gives interleaved components. Leaving the field in
            // place would read a correctly decoded colour frame as three
            // vertical bands of wrong colour.
            let this = Format {
                planar: this.planar && s.codec == Codec::Native,
                ..this
            };
            for frame in 0..this.frames {
                let (data, at) = s.pixels.frame(frame).ok_or_else(|| {
                    PluginError::failed(format!(
                        "{} declares {} frame(s) and holds fewer",
                        path.display(),
                        this.frames
                    ))
                })?;
                match &table {
                    Some(p) => planes.extend(p.expand(data, this, at)?),
                    None => planes.extend(pixels::planes_of_frame(data, this, at)?),
                }
            }
        }
        host.progress(1.0);

        // Slices and frames. One file per slice is the scanner's layout, so the
        // gathered files are the z axis; a multi-frame file's frames are the z
        // axis for the same reason — both are positions through the body, and
        // nothing here is a timepoint. A 4-D DICOM (a cine, a perfusion series)
        // would need the temporal tags to split, which is more than this reads.
        let slices = paths.len() * fmt.frames;

        let name = request
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "series.dcm".into());

        Ok(ImportResult {
            image: ImageResult {
                width: fmt.columns,
                height: fmt.rows,
                channels: out_channels,
                slices,
                frames: 1,
                pixel_type: match &table {
                    Some(p) => p.pixel_type(),
                    None => fmt.pixel_type(),
                },
                planes,
                channel_colors: Vec::new(),
                metadata: None,
                name: name.clone(),
            },
            info: Some(StackInfo {
                path: Some(request.path.display().to_string()),
                name,
                mode: if out_channels == 3 {
                    DisplayMode::Composite
                } else {
                    DisplayMode::Grayscale
                },
                // DICOM states its distances in millimetres, always — there is
                // no unit field to read because the standard fixes it.
                unit: Some("mm".to_string()),
                spacing: spacing_of(&first.found),
                frame_interval_s: None,
                channel_names: Vec::new(),
                calibration: pixels::rescale(&first.found),
                description: description(&first.found),
            }),
        })
    }
}

/// The voxel size, from the three tags that between them give it.
///
/// `PixelSpacing` is **row spacing then column spacing** — y before x, which is
/// the reverse of how everything else here is written and is the single easiest
/// thing to get backwards in a DICOM reader. On the usual isotropic in-plane
/// scan it makes no difference, which is what makes it dangerous: the bug
/// appears only on anisotropic data, where it silently transposes the aspect.
fn spacing_of(f: &Found) -> Spacing {
    let plane = f.nums(PIXEL_SPACING);
    let positive = |v: Option<&f64>| v.copied().filter(|v| v.is_finite() && *v > 0.0);
    Spacing {
        x: positive(plane.get(1)),
        y: positive(plane.first()),
        // `SpacingBetweenSlices` is centre-to-centre and is what a stack's z
        // step means. `SliceThickness` is how thick a slice is, which with a
        // gap between slices is a smaller number — so it is the fallback, not
        // the first choice, and using it where the real spacing exists would
        // compress the volume along z.
        z: positive(f.num(SPACING_BETWEEN_SLICES).as_ref())
            .or_else(|| positive(f.num(SLICE_THICKNESS).as_ref())),
    }
}

/// What to call the series, from the non-identifying tags.
///
/// `Modality` and `SeriesDescription` only. Both are about the acquisition —
/// "MR", "T1 MPRAGE SAG" — and neither is about a person.
fn description(f: &Found) -> Option<String> {
    let modality = f.text(MODALITY);
    let series = f.text(SERIES_DESCRIPTION);
    match (modality, series) {
        (Some(m), Some(s)) => Some(format!("{m}: {s}")),
        (Some(m), None) => Some(m),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

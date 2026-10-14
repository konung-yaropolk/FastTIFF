//! Turning DICOM's Pixel Data into planes, which is five tags deciding one
//! thing between them.
//!
//! `BitsAllocated`, `BitsStored`, `HighBit`, `PixelRepresentation` and
//! `SamplesPerPixel` are not independent descriptions of the samples — they are
//! one description spread over five elements, and every one of them has a way
//! of being ignored that leaves a picture looking fine.
//!
//! * `PixelRepresentation` ignored turns every negative sample into a very
//!   bright one. On CT, where air is −1000 HU, that is a scan with a brilliant
//!   white background instead of a black one — obvious. On MRI with a few
//!   negative voxels from a phase or a subtraction, it is a sprinkling of hot
//!   pixels that look like noise.
//! * `BitsStored` ignored, on the usual 12-bits-in-16 CT or CR, leaves the top
//!   four bits as whatever the scanner put there. Often zero, so it works —
//!   until the file where they are not, and then the image is scaled by
//!   sixteen in bands.
//! * `RescaleSlope` ignored gives an image that is right in every way except
//!   its numbers, so every measurement taken off it is wrong and nothing looks
//!   wrong.
//!
//! All five are therefore read, and a combination this does not handle is
//! refused by name rather than approximated.

use super::parse::{Found, Tag};
use fasttiff_plugin_api::{PixelType, PlaneData, PluginError};

pub(super) const ROWS: Tag = (0x0028, 0x0010);
pub(super) const COLUMNS: Tag = (0x0028, 0x0011);
pub(super) const SAMPLES_PER_PIXEL: Tag = (0x0028, 0x0002);
pub(super) const PHOTOMETRIC: Tag = (0x0028, 0x0004);
pub(super) const PLANAR_CONFIG: Tag = (0x0028, 0x0006);
pub(super) const NUMBER_OF_FRAMES: Tag = (0x0028, 0x0008);
pub(super) const BITS_ALLOCATED: Tag = (0x0028, 0x0100);
pub(super) const BITS_STORED: Tag = (0x0028, 0x0101);
pub(super) const HIGH_BIT: Tag = (0x0028, 0x0102);
pub(super) const PIXEL_REPRESENTATION: Tag = (0x0028, 0x0103);
pub(super) const RESCALE_INTERCEPT: Tag = (0x0028, 0x1052);
pub(super) const RESCALE_SLOPE: Tag = (0x0028, 0x1053);

/// How one file's samples are laid out and what they mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Format {
    pub(super) rows: u32,
    pub(super) columns: u32,
    /// How many images are in this one file's Pixel Data.
    pub(super) frames: usize,
    /// 1 for greyscale, 3 for colour.
    pub(super) samples: usize,
    pub(super) bits_allocated: u16,
    pub(super) bits_stored: u16,
    /// The position of the most significant stored bit within the allocated
    /// word. `bits_stored - 1` in every ordinary file.
    pub(super) high_bit: u16,
    /// Whether the samples are two's-complement signed.
    pub(super) signed: bool,
    /// Whether colour samples are stored plane-by-plane rather than
    /// interleaved per pixel.
    pub(super) planar: bool,
    /// Whether the photometric interpretation is `MONOCHROME1`, where the
    /// *lowest* value is white.
    pub(super) inverted: bool,
    /// Whether the samples are indices into a colour table rather than
    /// intensities. See [`super::palette`].
    pub(super) palette: bool,
}

impl Format {
    /// Bytes one sample occupies.
    pub(super) fn sample_bytes(self) -> usize {
        (self.bits_allocated as usize).div_ceil(8)
    }

    /// What the host should call these samples.
    pub(super) fn pixel_type(self) -> PixelType {
        match (self.bits_allocated, self.signed) {
            (8, false) => PixelType::U8,
            // An 8-bit signed image is widened to the signed 16-bit lane: the
            // contract has no signed byte, and `U8` would make every negative
            // sample bright.
            (8, true) => PixelType::I16,
            (_, false) => PixelType::U16,
            (_, true) => PixelType::I16,
        }
    }

    /// Read the five tags, and refuse what cannot be read.
    pub(super) fn of(f: &Found, big_endian: bool) -> Result<Self, PluginError> {
        let need = |tag: Tag, what: &str| -> Result<u16, PluginError> {
            f.us(tag, big_endian).ok_or_else(|| {
                PluginError::failed(format!(
                    "the DICOM file does not say its {what}, which is required \
                     to read its pixels"
                ))
            })
        };
        let rows = need(ROWS, "row count")? as u32;
        let columns = need(COLUMNS, "column count")? as u32;
        if rows == 0 || columns == 0 {
            return Err(PluginError::failed(format!(
                "the DICOM file declares a {columns}x{rows} image"
            )));
        }
        let bits_allocated = need(BITS_ALLOCATED, "bits per sample")?;
        // 8 and 16 are what scanners write. 1-bit overlays exist and are a
        // different thing entirely; 32-bit float pixel data exists in a few
        // parametric-map objects and is rare enough that an untested path for
        // it would be worse than a refusal.
        if !matches!(bits_allocated, 8 | 16) {
            return Err(PluginError::unsupported(format!(
                "this DICOM file allocates {bits_allocated} bits per sample; \
                 8 and 16 are read here"
            )));
        }
        let bits_stored = f.us(BITS_STORED, big_endian).unwrap_or(bits_allocated);
        if bits_stored == 0 || bits_stored > bits_allocated {
            return Err(PluginError::failed(format!(
                "the DICOM file stores {bits_stored} bits in {bits_allocated} \
                 allocated, which is not possible"
            )));
        }
        // `HighBit` defaults to the top of the stored range, which is what
        // every ordinary file has. A different value means the samples sit
        // shifted inside the word, and is honoured rather than assumed away.
        let high_bit = f
            .us(HIGH_BIT, big_endian)
            .unwrap_or(bits_stored.saturating_sub(1));
        if high_bit >= bits_allocated || high_bit + 1 < bits_stored {
            return Err(PluginError::failed(format!(
                "the DICOM file puts its high bit at {high_bit} with \
                 {bits_stored} bits stored in {bits_allocated}, which do not fit \
                 together"
            )));
        }
        let samples = f.us(SAMPLES_PER_PIXEL, big_endian).unwrap_or(1) as usize;
        if !matches!(samples, 1 | 3) {
            return Err(PluginError::unsupported(format!(
                "this DICOM file has {samples} samples per pixel; 1 (greyscale) \
                 and 3 (colour) are read here"
            )));
        }
        let photometric = f.text(PHOTOMETRIC).unwrap_or_default();
        // Palette colour: the samples are indices into the colour table in
        // (0028,1101..1203), which [`super::palette`] applies. Recorded here
        // rather than refused, because showing the indices as intensities is a
        // picture made of arbitrary numbers — a Doppler palette runs through
        // blue, red and green in whatever order the vendor chose.
        let palette = photometric.starts_with("PALETTE");
        // YBR_* is JPEG's colour space and arrives with JPEG transfer
        // syntaxes, which are already refused — but an uncompressed YBR_FULL
        // does exist, and showing its channels as RGB would be a plausible
        // picture in the wrong colours.
        if photometric.starts_with("YBR") {
            return Err(PluginError::unsupported(format!(
                "this DICOM file's pixels are in the {photometric} colour \
                 space, which is not converted here"
            )));
        }
        let frames = f
            .num(NUMBER_OF_FRAMES)
            .map(|v| v.max(1.0) as usize)
            .unwrap_or(1);
        Ok(Format {
            rows,
            columns,
            frames,
            samples,
            bits_allocated,
            bits_stored,
            high_bit,
            signed: f.us(PIXEL_REPRESENTATION, big_endian).unwrap_or(0) == 1,
            planar: f.us(PLANAR_CONFIG, big_endian).unwrap_or(0) == 1,
            inverted: photometric == "MONOCHROME1",
            palette,
        })
    }

    /// Bytes one frame of this format occupies.
    pub(super) fn frame_bytes(self) -> Option<usize> {
        (self.rows as usize)
            .checked_mul(self.columns as usize)?
            .checked_mul(self.samples)?
            .checked_mul(self.sample_bytes())
    }
}

/// `(intercept, slope)` when the file asks for rescaling, else `None`.
///
/// This is what turns stored values into Hounsfield units on CT, and into
/// whatever the vendor's units are on MRI. The host carries it as a
/// calibration rather than applying it, which keeps the stored values intact
/// and is the same arrangement the TIFF reader uses.
///
/// Identity is reported as nothing, so an ordinary MRI does not arrive carrying
/// a redundant `0 + 1 * x`.
pub(super) fn rescale(f: &Found) -> Option<(f64, f64)> {
    let slope = f.num(RESCALE_SLOPE).unwrap_or(1.0);
    let inter = f.num(RESCALE_INTERCEPT).unwrap_or(0.0);
    (slope.is_finite() && inter.is_finite() && slope != 0.0 && !(slope == 1.0 && inter == 0.0))
        .then_some((inter, slope))
}

/// Pull one frame's samples out of Pixel Data and make planes of them.
///
/// Returns one plane per channel, so a greyscale frame gives one and an RGB
/// frame gives three — which is what the host's `xyczt` layout wants, and what
/// makes a colour DICOM open as three channels rather than as a packed image
/// nothing can window.
pub(super) fn planes_of_frame(
    data: &[u8],
    fmt: Format,
    frame: usize,
) -> Result<Vec<PlaneData>, PluginError> {
    let n_px = (fmt.rows as usize) * (fmt.columns as usize);
    let stride = fmt
        .frame_bytes()
        .ok_or_else(|| PluginError::failed("the DICOM frame size overflows"))?;
    let at = stride
        .checked_mul(frame)
        .ok_or_else(|| PluginError::failed("the DICOM frame offset overflows"))?;
    let raw = data
        .get(at..at + stride)
        .ok_or_else(|| PluginError::failed("the DICOM pixel data ends in the middle of a frame"))?;

    let word = fmt.sample_bytes();
    // How far the stored bits sit above the bottom of the word, and the mask
    // that keeps only them. For the ordinary file where `high_bit` is
    // `bits_stored - 1` the shift is zero and the mask is everything, so this
    // costs nothing where it is not needed.
    let shift = (fmt.high_bit + 1 - fmt.bits_stored) as u32;
    let mask: u32 = if fmt.bits_stored >= 32 {
        u32::MAX
    } else {
        (1u32 << fmt.bits_stored) - 1
    };
    // Where the sign bit of a stored sample is, once shifted down.
    let sign_bit: u32 = 1 << (fmt.bits_stored - 1);

    // One sample, as the signed-or-unsigned number it represents, widened to
    // `i32` so both fit before the lane is chosen.
    let sample = |i: usize| -> i32 {
        let v = match word {
            1 => raw[i] as u32,
            _ => u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as u32,
        };
        let v = (v >> shift) & mask;
        if fmt.signed && (v & sign_bit) != 0 {
            // Two's complement, sign-extended from `bits_stored` to 32. The
            // subtraction is the extension: a 12-bit 0xFFF becomes −1 rather
            // than 4095.
            (v as i32) - ((sign_bit as i32) << 1)
        } else {
            v as i32
        }
    };

    // Which raw sample index a given channel and pixel is at. Interleaved is
    // the default (`PlanarConfiguration` 0) and is what almost everything
    // writes; planar puts each channel's whole plane one after another, and
    // reading one as the other gives a picture in three vertical bands of
    // wrong colour.
    let index = |c: usize, p: usize| -> usize {
        if fmt.planar {
            c * n_px + p
        } else {
            p * fmt.samples + c
        }
    };

    let mut out = Vec::with_capacity(fmt.samples);
    for c in 0..fmt.samples {
        out.push(match fmt.pixel_type() {
            PixelType::U8 => PlaneData::U8((0..n_px).map(|p| sample(index(c, p)) as u8).collect()),
            // Both the unsigned and the signed 16-bit lanes are `U16` bits;
            // `pixel_type` is what tells the host how to read them.
            _ => PlaneData::U16(
                (0..n_px)
                    .map(|p| sample(index(c, p)) as i16 as u16)
                    .collect(),
            ),
        });
    }
    Ok(out)
}

/// Big-endian Pixel Data, swapped in place to the little-endian the sample
/// reader assumes.
///
/// Done once per file rather than branching per sample. Explicit VR Big Endian
/// is retired but files from the 1990s and some older research archives are in
/// it, and a 16-bit image read with the bytes the wrong way round is not subtly
/// wrong — it is noise, which at least fails loudly.
pub(super) fn to_little_endian(data: &mut [u8], fmt: Format) {
    if fmt.sample_bytes() == 2 {
        for pair in data.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
    }
}

#[cfg(test)]
#[path = "pixels_tests.rs"]
pub(crate) mod tests;

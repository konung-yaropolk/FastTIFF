//! The NIfTI-1 header: 348 fixed bytes, and the decisions that come out of it.
//!
//! Laid out field by field at absolute offsets rather than parsed with a struct
//! cast, which is the usual way this is done in C and cannot be done safely
//! here — the file's byte order is not known until the header has been read,
//! and a `#[repr(C)]` cast would also be asserting a padding layout the format
//! does not promise. Reading each field by offset costs nothing and makes the
//! byte-swap a property of the reader rather than of the struct.
//!
//! # Byte order comes out of the header itself
//!
//! There is no magic number saying which end is which. `sizeof_hdr` is the
//! trick the format uses instead: it is always 348, so a reader tries it both
//! ways and the one that gives 348 is the file's order. This is the whole
//! mechanism, it is what `nifti1.h` itself documents, and it is why every field
//! below goes through [`Hdr::i16`] and friends rather than being read directly.
//!
//! # What is deliberately not read
//!
//! The qform/sform orientation matrices. They say how the voxel grid sits in
//! the scanner, which is what a neuroimaging tool needs to overlay two scans —
//! and this viewer has nowhere to put that: it shows a stack of planes with a
//! voxel size, not a volume in a patient's frame of reference. Reading them to
//! throw them away would suggest otherwise. The spacings from `pixdim` are
//! taken, because those *are* what the viewer can honour.

use fasttiff_plugin_api::PluginError;

/// The size every NIfTI-1 header declares, and the byte-order probe.
pub(super) const SIZEOF_HDR: i32 = 348;

/// NIfTI-2's header size, recognised only so it can be refused by name.
const SIZEOF_HDR_NIFTI2: i32 = 540;

/// Where the voxels start in a single-file `.nii` with no header extension:
/// the 348-byte header plus the 4-byte extension flag.
///
/// `vox_offset` is authoritative and larger values are honoured as written — a
/// header extension pushes the voxels back, legitimately. This is the *floor*,
/// and the value a file that writes 0 must have meant.
const UNEXTENDED_VOX_OFFSET: f32 = 352.0;

/// A NIfTI-1 header, with the byte order it turned out to be in.
///
/// `Debug` because a test asserting on a parse failure needs it, and because a
/// header that is 348 bytes of borrowed slice has nothing to hide.
#[derive(Debug)]
pub(super) struct Hdr<'a> {
    bytes: &'a [u8],
    /// Whether the file's multi-byte fields are big-endian.
    swap: bool,
}

/// How a NIfTI datatype maps onto what the host can hold.
///
/// The host has four sample types and NIfTI has eighteen, so most of this is
/// widening. The two interesting rows are the ones that are *not* lossless, and
/// they are called out in [`Datatype::widened_from`] so a reader of this file
/// can see at a glance which conversions change a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Datatype {
    /// `NIFTI_TYPE_UINT8`.
    U8,
    /// `NIFTI_TYPE_INT16`. Signed, and the host's `PixelType::I16` carries it
    /// exactly — in the `U16` lane, which is the contract's own arrangement.
    I16,
    /// `NIFTI_TYPE_UINT16`.
    U16,
    /// `NIFTI_TYPE_INT8`, widened to `i16`. Lossless: every `i8` is an `i16`.
    I8,
    /// `NIFTI_TYPE_FLOAT32`.
    F32,
    /// `NIFTI_TYPE_INT32`, converted to `f32`. **Lossy past 2^24.**
    I32,
    /// `NIFTI_TYPE_UINT32`, converted to `f32`. **Lossy past 2^24.**
    U32,
    /// `NIFTI_TYPE_FLOAT64`, narrowed to `f32`. **Lossy.**
    F64,
    /// `NIFTI_TYPE_RGB24`: three bytes per voxel, interleaved.
    Rgb24,
    /// `NIFTI_TYPE_RGBA32`: four bytes per voxel, interleaved.
    ///
    /// The alpha arrives as a fourth channel rather than being composited
    /// away. A viewer with per-channel windows can show or hide it, which is
    /// more use than a reader deciding it is not data — and in a segmentation
    /// or a surface rendering saved to NIfTI it very much is.
    Rgba32,
}

impl Datatype {
    /// The NIfTI code, or `None` for one this reader will not take.
    ///
    /// The codes are `nifti1.h`'s `NIFTI_TYPE_*`. The ones left out are left
    /// out on purpose rather than for want of a branch:
    ///
    /// * `DT_BINARY` (1) is one bit per voxel, and nothing produces it.
    /// * the complex types (32, 1792, 2048) are two numbers per voxel; showing
    ///   the real part and calling it the image would be a quiet lie, and the
    ///   host has no complex plane to put them in.
    /// * `RGB24` (128) and `RGBA32` (2304) *are* read, as three and four
    ///   channels. They are not what an MRI volume is, but they are what a
    ///   cryosection atlas is — the Visible Human is distributed this way —
    ///   and a colour volume refused is a file that does not open at all.
    /// * the 64-bit integers (1024, 1280) and `FLOAT128` (1536) exceed `f32`'s
    ///   range as well as its precision, so there is no conversion that is
    ///   merely lossy rather than wrong.
    pub(super) fn from_code(code: i16) -> Option<Self> {
        Some(match code {
            2 => Datatype::U8,
            4 => Datatype::I16,
            8 => Datatype::I32,
            16 => Datatype::F32,
            64 => Datatype::F64,
            128 => Datatype::Rgb24,
            256 => Datatype::I8,
            512 => Datatype::U16,
            768 => Datatype::U32,
            2304 => Datatype::Rgba32,
            _ => return None,
        })
    }

    /// How many channels a voxel of this type carries.
    ///
    /// One for everything but the colour types, whose components become
    /// separate channels — which is what the host's `xyczt` layout wants and
    /// what lets each be windowed. A packed RGB plane would be a picture
    /// nothing could adjust.
    pub(super) fn channels(self) -> usize {
        match self {
            Datatype::Rgb24 => 3,
            Datatype::Rgba32 => 4,
            _ => 1,
        }
    }

    /// Whether this type's bytes are swapped on a byte-swapped file.
    ///
    /// The colour types are **not**, and this is the trap: their components are
    /// single bytes, so there is nothing to reverse — but a reader that swaps
    /// by bytes-per-voxel would reverse each voxel's components and turn every
    /// red pixel blue. `nifti1_io.c` carries a separate `swapsize` column
    /// beside `nbyper` for exactly this, and RGB24's is 1 where its nbyper
    /// is 3.
    pub(super) fn swaps(self) -> bool {
        !matches!(
            self,
            Datatype::Rgb24 | Datatype::Rgba32 | Datatype::U8 | Datatype::I8
        )
    }

    /// Bytes per voxel on disk.
    pub(super) fn bytes(self) -> usize {
        match self {
            Datatype::I8 | Datatype::U8 => 1,
            Datatype::I16 | Datatype::U16 => 2,
            Datatype::Rgb24 => 3,
            Datatype::F32 | Datatype::I32 | Datatype::U32 | Datatype::Rgba32 => 4,
            Datatype::F64 => 8,
        }
    }

    /// Whether the conversion to the host's type loses information, and what
    /// to say about it. `None` when it is exact.
    ///
    /// Reported rather than silent. Someone measuring intensities in a volume
    /// that arrived as `int32` has a right to know it is being carried as
    /// `f32`, because past 16,777,216 consecutive integers stop being
    /// distinguishable — and a voxel count or a label map is exactly the kind
    /// of `int32` volume where that matters.
    pub(super) fn widened_from(self) -> Option<&'static str> {
        match self {
            Datatype::I32 => Some(
                "int32 voxels are carried as 32-bit float: values beyond \
                 16777216 lose their last bits",
            ),
            Datatype::U32 => Some(
                "uint32 voxels are carried as 32-bit float: values beyond \
                 16777216 lose their last bits",
            ),
            Datatype::F64 => Some(
                "float64 voxels are carried as 32-bit float, which keeps about \
                 7 significant digits of each",
            ),
            _ => None,
        }
    }
}

/// The spatial unit `xyzt_units` names, as a word for [`StackInfo::unit`].
///
/// [`StackInfo::unit`]: fasttiff_plugin_api::StackInfo::unit
///
/// The byte packs two units into one: the spatial one in its low three bits and
/// the temporal one in the next three. Masking is therefore not optional — the
/// raw byte of a file with seconds on its time axis is 2 | 8 = 10, which read
/// whole is not a unit at all.
pub(super) fn space_unit(xyzt_units: u8) -> Option<&'static str> {
    match xyzt_units & 0x07 {
        1 => Some("meter"),
        2 => Some("mm"),
        3 => Some("micron"),
        _ => None,
    }
}

/// Seconds per timepoint, from `pixdim[4]` and the time unit in `xyzt_units`.
///
/// The repetition time of an fMRI series, which the viewer shows on its
/// scrubber. Converted to seconds here because that is the unit the host's
/// `frame_interval_s` is in, and because a file is as likely to say
/// milliseconds: a TR of 2000 ms read as 2000 s would put a ten-minute scan at
/// forty days.
pub(super) fn time_interval_s(pixdim4: f64, xyzt_units: u8) -> Option<f64> {
    if !(pixdim4.is_finite() && pixdim4 > 0.0) {
        return None;
    }
    match xyzt_units & 0x38 {
        8 => Some(pixdim4),                // seconds
        16 => Some(pixdim4 / 1_000.0),     // milliseconds
        24 => Some(pixdim4 / 1_000_000.0), // microseconds
        // Hz, ppm and rad/s are spectroscopy axes, not time. A file that says
        // one of those has no interval to report, and guessing seconds would
        // put a wrong number on the scrubber.
        _ => None,
    }
}

impl<'a> Hdr<'a> {
    /// Read the 348-byte header, settling the byte order on the way.
    pub(super) fn parse(bytes: &'a [u8]) -> Result<Self, PluginError> {
        if bytes.len() < 348 {
            return Err(PluginError::unsupported(format!(
                "a NIfTI header is 348 bytes and this file has {}",
                bytes.len()
            )));
        }
        // `sizeof_hdr` both ways. Exactly one of them can be 348 — the value
        // byte-swapped is 0x5C010000, which is 1543569408 — so this is a
        // decision and not a guess.
        let native = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let swapped = i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let swap = match (native, swapped) {
            (SIZEOF_HDR, _) => false,
            (_, SIZEOF_HDR) => true,
            // NIfTI-2 is a different 540-byte header with 64-bit dimensions. It
            // is recognised here only so that it can be refused by name:
            // reading its first 348 bytes as a NIfTI-1 header would find a
            // plausible `datatype` at an offset that means something else, and
            // produce an image rather than an error.
            (SIZEOF_HDR_NIFTI2, _) | (_, SIZEOF_HDR_NIFTI2) => {
                return Err(PluginError::unsupported(
                    "this is a NIfTI-2 file (540-byte header). Only NIfTI-1 is \
                     read here; convert it with `nifti_tool -copy_im` or \
                     nibabel, or ask for NIfTI-2 support",
                ));
            }
            _ => {
                return Err(PluginError::unsupported(format!(
                    "not a NIfTI file: its first field is {native}, and a NIfTI-1 \
                     header declares {SIZEOF_HDR} there (in either byte order)"
                )));
            }
        };
        Ok(Hdr { bytes, swap })
    }

    /// Whether the file's numbers are the other way round from this machine's.
    pub(super) fn byte_swapped(&self) -> bool {
        self.swap
    }

    fn i16(&self, at: usize) -> i16 {
        let b = [self.bytes[at], self.bytes[at + 1]];
        if self.swap {
            i16::from_be_bytes(b)
        } else {
            i16::from_le_bytes(b)
        }
    }

    fn f32(&self, at: usize) -> f32 {
        let b = [
            self.bytes[at],
            self.bytes[at + 1],
            self.bytes[at + 2],
            self.bytes[at + 3],
        ];
        if self.swap {
            f32::from_be_bytes(b)
        } else {
            f32::from_le_bytes(b)
        }
    }

    // ---- the fields this reader uses, at their offsets in `nifti1.h` ----

    /// `dim[i]`, `i` in `0..8`. `dim[0]` is how many of the rest are
    /// meaningful; `dim[1..=3]` are x, y, z; `dim[4]` is time.
    pub(super) fn dim(&self, i: usize) -> i16 {
        self.i16(40 + 2 * i)
    }

    /// `pixdim[i]`: the extent of one voxel along `dim[i]`.
    ///
    /// `pixdim[0]` is not a size at all — it carries the `qfac` sign for the
    /// qform matrix — which is why the caller asks for 1..=4 and never 0.
    pub(super) fn pixdim(&self, i: usize) -> f32 {
        self.f32(76 + 4 * i)
    }

    /// `datatype`: which `NIFTI_TYPE_*` the voxels are.
    pub(super) fn datatype(&self) -> i16 {
        self.i16(70)
    }

    /// `bitpix`: bits per voxel, which must agree with `datatype`.
    pub(super) fn bitpix(&self) -> i16 {
        self.i16(72)
    }

    /// `vox_offset`: where the voxels start, as a float, because the format
    /// says so.
    pub(super) fn vox_offset(&self) -> f32 {
        self.f32(108)
    }

    /// `scl_slope`, `scl_inter`: the linear calibration `inter + slope * raw`.
    pub(super) fn scl(&self) -> (f32, f32) {
        (self.f32(112), self.f32(116))
    }

    /// `xyzt_units`: the spatial and temporal units, packed into one byte.
    pub(super) fn xyzt_units(&self) -> u8 {
        self.bytes[123]
    }

    /// `descrip`: 80 bytes of free text, NUL-padded.
    pub(super) fn descrip(&self) -> Option<String> {
        let raw = &self.bytes[148..228];
        let text: String = raw
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect();
        let text = text.trim().to_string();
        (!text.is_empty()).then_some(text)
    }

    /// `magic`: `"n+1\0"` for a single file, `"ni1\0"` for a `.hdr`/`.img`
    /// pair.
    pub(super) fn magic(&self) -> [u8; 4] {
        [
            self.bytes[344],
            self.bytes[345],
            self.bytes[346],
            self.bytes[347],
        ]
    }
}

/// The shape and scaling a reader needs, checked and in the host's terms.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Shape {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) slices: usize,
    pub(super) frames: usize,
    pub(super) datatype: Datatype,
    /// Where the voxels begin.
    pub(super) data_at: usize,
    /// `(inter, slope)` when the file asks for scaling, else `None`.
    pub(super) calibration: Option<(f64, f64)>,
    pub(super) spacing: (Option<f64>, Option<f64>, Option<f64>),
    pub(super) unit: Option<String>,
    pub(super) frame_interval_s: Option<f64>,
    pub(super) descrip: Option<String>,
}

/// Turn a header into the shape to read, refusing what cannot be read.
pub(super) fn shape_of(h: &Hdr) -> Result<Shape, PluginError> {
    // The single-file magic. A `.hdr`/`.img` pair declares `ni1` and keeps its
    // voxels in a second file; this reader is handed one path and so cannot
    // follow that, and saying which half is missing is more use than a size
    // mismatch further down.
    let magic = h.magic();
    match &magic {
        b"n+1\0" => {}
        b"ni1\0" => {
            return Err(PluginError::unsupported(
                "this is the header half of a NIfTI `.hdr`/`.img` pair, which \
                 keeps its voxels in a separate file. Open the single-file \
                 `.nii` form instead",
            ));
        }
        other => {
            return Err(PluginError::unsupported(format!(
                "the NIfTI magic is {:?}, which is neither `n+1` (single file) \
                 nor `ni1` (header/image pair)",
                String::from_utf8_lossy(other)
            )));
        }
    }

    let ndim = h.dim(0);
    if !(1..=7).contains(&ndim) {
        return Err(PluginError::failed(format!(
            "the NIfTI header says it has {ndim} dimensions, and the format \
             allows 1 to 7"
        )));
    }

    // `dim[i]` for an axis the file does not have is documented as 1, but
    // files in the wild write 0 — and a zero would make the volume empty
    // rather than flat. Treated as 1, which is what every other reader does.
    let axis = |i: usize| -> Result<usize, PluginError> {
        let v = h.dim(i);
        if i as i16 > ndim {
            return Ok(1);
        }
        if v < 0 {
            return Err(PluginError::failed(format!(
                "the NIfTI header declares dim[{i}] = {v}"
            )));
        }
        Ok((v as usize).max(1))
    };

    let (nx, ny, nz, nt) = (axis(1)?, axis(2)?, axis(3)?, axis(4)?);
    if nx == 0 || ny == 0 {
        return Err(PluginError::failed("the NIfTI volume has no extent"));
    }
    // Axes past time. `dim[5]` is a vector per voxel (a DTI tensor, a
    // deformation field), `dim[6]` and `dim[7]` are unnamed. Folding them into
    // the time axis would show them as a movie, which for a 3-vector field is
    // three frames that are not timepoints — so they are refused rather than
    // reinterpreted.
    for i in 5..=7 {
        if i as i16 <= ndim && h.dim(i) > 1 {
            return Err(PluginError::unsupported(format!(
                "this volume has {} components along dim[{i}], which is a \
                 vector or tensor per voxel rather than an image axis. This \
                 reader shows x, y, z and time",
                h.dim(i)
            )));
        }
    }

    let code = h.datatype();
    let datatype = Datatype::from_code(code).ok_or_else(|| {
        PluginError::unsupported(format!(
            "NIfTI datatype {code} ({}) is not read here",
            datatype_name(code)
        ))
    })?;
    // `bitpix` is redundant with `datatype` and the format requires them to
    // agree. A disagreement means one of the two is wrong and there is no way
    // to tell which, so neither is trusted.
    let want_bits = (datatype.bytes() * 8) as i16;
    if h.bitpix() != want_bits {
        return Err(PluginError::failed(format!(
            "the NIfTI header is inconsistent: datatype {code} is {want_bits} \
             bits per voxel but `bitpix` says {}",
            h.bitpix()
        )));
    }

    // `vox_offset` is a float in the format, and larger than the floor is the
    // authoritative answer — a header extension legitimately pushes the voxels
    // back. Smaller than the floor is not: in a single-file NIfTI the first 352
    // bytes *are* the header, so reading voxels from inside them would decode
    // the header as picture and produce a band of noise across the first rows.
    //
    // Zero is excepted, and only zero. Some writers leave the field at its
    // default and mean "straight after the header", which is the one reading
    // that is not ambiguous — there is nowhere else the voxels of a single-file
    // volume could be. Anything else in between is a file whose header
    // disagrees with itself, and guessing which half is wrong is not this
    // reader's business.
    // `vox_offset` is a float in the format, and larger than the floor is the
    // authoritative answer: a header extension legitimately pushes the voxels
    // back.
    //
    // Smaller than the floor is **clamped up to it**, not refused, and that is
    // the format's own rule rather than a kindness. `nifti1.h`, under DETAILS
    // ABOUT vox_offset, states that in a `.nii` file a `vox_offset` below 352
    // is *equivalent to* 352. Refusing those would reject files the format
    // declares valid, and writers that leave the field at its default do
    // produce them — so this was a conformance bug here before it was a
    // leniency. The rule is unambiguous as well as binding: there is nowhere
    // else the voxels of a single-file volume could be.
    //
    // Negative and non-finite are still refused, because those are not a small
    // `vox_offset` — they are a field that does not hold one at all.
    let vox = h.vox_offset();
    if !vox.is_finite() || vox < 0.0 {
        return Err(PluginError::failed(format!(
            "the NIfTI header puts its voxels at offset {vox}"
        )));
    }
    let data_at = vox.max(UNEXTENDED_VOX_OFFSET) as usize;

    // Scaling. Slope 0 is the format's "no scaling", not a scale factor of
    // nothing — applied literally it would flatten every voxel to `inter`,
    // which is a uniform image that looks like a failed acquisition rather
    // than like a bug. Identity is also skipped, so a file that says
    // `1 * raw + 0` produces no calibration rather than a redundant one.
    // Scaling is not defined for the colour types and must not be applied to
    // them: `nifti1.h` says so, and an RGB triple multiplied by a slope is not
    // a colour. A file carrying both is carrying one of them by mistake.
    let (slope, inter) = if datatype.channels() > 1 {
        (1.0, 0.0)
    } else {
        h.scl()
    };
    let calibration =
        (slope.is_finite() && inter.is_finite() && slope != 0.0 && !(slope == 1.0 && inter == 0.0))
            .then_some((inter as f64, slope as f64));

    let positive = |v: f32| (v.is_finite() && v > 0.0).then_some(v as f64);
    Ok(Shape {
        width: nx as u32,
        height: ny as u32,
        slices: nz,
        frames: nt,
        datatype,
        data_at,
        calibration,
        spacing: (
            positive(h.pixdim(1)),
            positive(h.pixdim(2)),
            positive(h.pixdim(3)),
        ),
        unit: space_unit(h.xyzt_units()).map(str::to_string),
        frame_interval_s: time_interval_s(h.pixdim(4) as f64, h.xyzt_units()),
        descrip: h.descrip(),
    })
}

/// What a datatype code is called, so a refusal can name it.
///
/// Every code in `nifti1.h`, including the ones this reader takes, because the
/// table is also what makes the refusal messages specific — "complex64 is not
/// read here" is actionable where "datatype 32" is not.
fn datatype_name(code: i16) -> &'static str {
    match code {
        0 => "unknown",
        1 => "1-bit binary",
        2 => "uint8",
        4 => "int16",
        8 => "int32",
        16 => "float32",
        32 => "complex64",
        64 => "float64",
        128 => "RGB24",
        256 => "int8",
        512 => "uint16",
        768 => "uint32",
        1024 => "int64",
        1280 => "uint64",
        1536 => "float128",
        1792 => "complex128",
        2048 => "complex256",
        2304 => "RGBA32",
        _ => "an unrecognised type",
    }
}

#[cfg(test)]
#[path = "header_tests.rs"]
pub(super) mod tests;

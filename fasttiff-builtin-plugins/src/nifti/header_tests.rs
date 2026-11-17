//! The 348 bytes, field by field.
//!
//! Every test here builds a header by writing numbers at absolute offsets and
//! then asks the reader what it made of them. That is deliberate and it is the
//! only way these are worth anything: a test that used the reader's own
//! constants to place the fields would agree with it about a wrong offset. The
//! offsets below are written out as literals, from `nifti1.h`, so the test and
//! the code are two independent statements of the same layout.

use super::*;

/// A header builder: 348 bytes of zero, with fields written at their offsets.
pub(crate) struct Build {
    b: Vec<u8>,
    be: bool,
}

impl Build {
    /// A valid minimal NIfTI-1 single-file header for a `w x h x d x t`
    /// volume of `datatype`.
    pub(crate) fn new(w: i16, h: i16, d: i16, t: i16, datatype: i16, bitpix: i16) -> Self {
        let mut it = Build {
            b: vec![0u8; 348],
            be: false,
        };
        it.i32_at(0, 348) // sizeof_hdr
            .i16_at(40, if t > 1 { 4 } else { 3 }) // dim[0]
            .i16_at(42, w) // dim[1]
            .i16_at(44, h) // dim[2]
            .i16_at(46, d) // dim[3]
            .i16_at(48, t) // dim[4]
            .i16_at(70, datatype)
            .i16_at(72, bitpix)
            .f32_at(108, 352.0) // vox_offset
            .magic(b"n+1\0");
        it
    }

    /// Write every multi-byte field the other way round, as a scanner on a
    /// big-endian machine once did and as `nifti1.h` requires readers to cope
    /// with.
    pub(crate) fn big_endian(mut self) -> Self {
        // Rewritten rather than byte-reversed in place: the fields have to be
        // re-emitted in the new order, and a blanket reverse would also
        // reverse `magic` and `descrip`, which are bytes and not numbers.
        let (w, h, d, t) = (self.i16(42), self.i16(44), self.i16(46), self.i16(48));
        let (dt, bp) = (self.i16(70), self.i16(72));
        let vox = self.f32(108);
        let (slope, inter) = (self.f32(112), self.f32(116));
        let units = self.b[123];
        let pixdims: Vec<f32> = (0..8).map(|i| self.f32(76 + 4 * i)).collect();
        let ndim = self.i16(40);
        let magic = [self.b[344], self.b[345], self.b[346], self.b[347]];

        self.b = vec![0u8; 348];
        self.be = true;
        self.i32_at(0, 348)
            .i16_at(40, ndim)
            .i16_at(42, w)
            .i16_at(44, h)
            .i16_at(46, d)
            .i16_at(48, t)
            .i16_at(70, dt)
            .i16_at(72, bp)
            .f32_at(108, vox)
            .f32_at(112, slope)
            .f32_at(116, inter)
            .magic(&magic);
        for (i, p) in pixdims.iter().enumerate() {
            self.f32_at(76 + 4 * i, *p);
        }
        self.b[123] = units;
        self
    }

    pub(crate) fn i16_at(&mut self, at: usize, v: i16) -> &mut Self {
        let b = if self.be {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        };
        self.b[at..at + 2].copy_from_slice(&b);
        self
    }

    pub(crate) fn i32_at(&mut self, at: usize, v: i32) -> &mut Self {
        let b = if self.be {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        };
        self.b[at..at + 4].copy_from_slice(&b);
        self
    }

    pub(crate) fn f32_at(&mut self, at: usize, v: f32) -> &mut Self {
        let b = if self.be {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        };
        self.b[at..at + 4].copy_from_slice(&b);
        self
    }

    pub(crate) fn magic(&mut self, m: &[u8; 4]) -> &mut Self {
        self.b[344..348].copy_from_slice(m);
        self
    }

    /// `pixdim[i]`, the voxel extent along `dim[i]`.
    pub(crate) fn pixdim(&mut self, i: usize, v: f32) -> &mut Self {
        self.f32_at(76 + 4 * i, v)
    }

    /// `scl_slope` and `scl_inter`.
    pub(crate) fn scl(&mut self, slope: f32, inter: f32) -> &mut Self {
        self.f32_at(112, slope).f32_at(116, inter)
    }

    /// `xyzt_units`, one byte with two units packed into it.
    pub(crate) fn units(&mut self, v: u8) -> &mut Self {
        self.b[123] = v;
        self
    }

    /// `descrip`, 80 bytes of NUL-padded text.
    pub(crate) fn descrip(&mut self, s: &str) -> &mut Self {
        let n = s.len().min(80);
        self.b[148..148 + n].copy_from_slice(&s.as_bytes()[..n]);
        self
    }

    fn i16(&self, at: usize) -> i16 {
        let b = [self.b[at], self.b[at + 1]];
        if self.be {
            i16::from_be_bytes(b)
        } else {
            i16::from_le_bytes(b)
        }
    }

    fn f32(&self, at: usize) -> f32 {
        let b = [self.b[at], self.b[at + 1], self.b[at + 2], self.b[at + 3]];
        if self.be {
            f32::from_be_bytes(b)
        } else {
            f32::from_le_bytes(b)
        }
    }

    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.b.clone()
    }

    /// The header plus the 4-byte extension flag, which is where the voxels of
    /// an ordinary single-file volume begin.
    pub(crate) fn with_data(&self, data: &[u8]) -> Vec<u8> {
        let mut out = self.b.clone();
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(data);
        out
    }
}

/// A `u16` volume's bytes, each voxel equal to its own index so a
/// mis-addressed read is visible rather than merely wrong-looking.
pub(crate) fn counted_u16(n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (i as u16).to_le_bytes()).collect()
}

fn shape(b: &[u8]) -> Result<Shape, PluginError> {
    shape_of(&Hdr::parse(b)?)
}

#[test]
fn a_minimal_header_gives_the_shape_it_declares() {
    let s = shape(&Build::new(4, 3, 2, 1, 4, 16).bytes()).expect("a valid header");
    assert_eq!((s.width, s.height, s.slices, s.frames), (4, 3, 2, 1));
    assert_eq!(s.datatype, Datatype::I16);
    assert_eq!(s.data_at, 352);
}

/// The whole byte-order mechanism: there is no flag, only `sizeof_hdr` read
/// both ways. A reader that assumed little-endian would get 1543569408 here and
/// a dimension of 1024 from a `dim[1]` of 4.
#[test]
fn a_big_endian_header_is_detected_and_read() {
    let mut b = Build::new(4, 3, 2, 5, 4, 16);
    b.pixdim(1, 2.5).pixdim(4, 3.0).units(2 | 8);
    let be = b.big_endian();
    let raw = be.bytes();
    let h = Hdr::parse(&raw).expect("a byte-swapped header is still a header");
    assert!(h.byte_swapped(), "should have been detected as swapped");
    let s = shape_of(&h).expect("a valid header");
    assert_eq!((s.width, s.height, s.slices, s.frames), (4, 3, 2, 5));
    assert_eq!(s.spacing.0, Some(2.5), "pixdim must swap too");
    assert_eq!(s.frame_interval_s, Some(3.0));
}

#[test]
fn something_that_is_not_a_nifti_header_is_declined() {
    let junk = vec![0x42u8; 400];
    let err = Hdr::parse(&junk).expect_err("junk is not a header");
    assert!(
        format!("{err}").contains("not a NIfTI"),
        "should say what it is not: {err}"
    );
    // And a file too short to hold a header at all.
    assert!(Hdr::parse(&[0u8; 40]).is_err());
}

/// NIfTI-2 puts different fields where NIfTI-1's are, so reading its first 348
/// bytes as a NIfTI-1 header does not fail — it succeeds and produces a
/// different, wrong image. Refusing it by name is the only safe answer.
#[test]
fn a_nifti2_header_is_refused_by_name_rather_than_misread() {
    let mut b = vec![0u8; 560];
    b[0..4].copy_from_slice(&540i32.to_le_bytes());
    b[4..8].copy_from_slice(b"n+2\0");
    let err = Hdr::parse(&b).expect_err("NIfTI-2 must not be read as NIfTI-1");
    assert!(
        format!("{err}").contains("NIfTI-2"),
        "should name the format it found: {err}"
    );
    // Both byte orders, since NIfTI-2 has the same no-flag problem.
    let mut be = vec![0u8; 560];
    be[0..4].copy_from_slice(&540i32.to_be_bytes());
    assert!(Hdr::parse(&be).is_err());
}

/// The datatype table, against the widths `nifti1.h` gives for each code. A
/// width that disagreed with the code would read every voxel from the wrong
/// offset, which looks like noise for a float and like a plausible picture for
/// an integer.
#[test]
fn every_datatype_this_reader_takes_has_the_width_the_spec_gives() {
    for (code, bytes, channels) in [
        (2, 1, 1),    // uint8
        (4, 2, 1),    // int16
        (8, 4, 1),    // int32
        (16, 4, 1),   // float32
        (64, 8, 1),   // float64
        (128, 3, 3),  // RGB24
        (256, 1, 1),  // int8
        (512, 2, 1),  // uint16
        (768, 4, 1),  // uint32
        (2304, 4, 4), // RGBA32
    ] {
        let dt = Datatype::from_code(code).unwrap_or_else(|| panic!("code {code} should be read"));
        assert_eq!(dt.bytes(), bytes, "code {code} is {bytes} bytes per voxel");
        assert_eq!(
            dt.channels(),
            channels,
            "code {code} has {channels} channel(s)"
        );
    }
    // And the ones deliberately refused, each for a reason given in
    // `Datatype::from_code`.
    for code in [0, 1, 32, 1024, 1280, 1536, 1792, 2048] {
        assert!(
            Datatype::from_code(code).is_none(),
            "code {code} should not be read"
        );
    }
}

/// The colour types' components are single bytes, so there is nothing to
/// reverse on a byte-swapped file — and swapping by bytes-per-voxel would
/// reverse each voxel's components and turn every red pixel blue. `nifti1_io.c`
/// keeps a `swapsize` column beside its `nbyper` for exactly this, and RGB24's
/// is 1 where its nbyper is 3.
#[test]
fn the_colour_types_are_never_byte_swapped() {
    for code in [128, 2304] {
        let dt = Datatype::from_code(code).expect("read");
        assert!(!dt.swaps(), "code {code} must not be swapped");
    }
    // Nor are the single-byte scalar types, for the simpler reason that one
    // byte has no order.
    for code in [2, 256] {
        assert!(!Datatype::from_code(code).expect("read").swaps());
    }
    // Everything wider does swap.
    for code in [4, 8, 16, 64, 512, 768] {
        assert!(
            Datatype::from_code(code).expect("read").swaps(),
            "code {code} should swap"
        );
    }
}

/// Scaling is not defined for the colour types, and an RGB triple multiplied by
/// a slope is not a colour. A file carrying both is carrying one by mistake.
#[test]
fn a_colour_volume_reports_no_calibration_even_if_the_file_states_one() {
    let mut b = Build::new(4, 3, 2, 1, 128, 24); // RGB24
    b.scl(2.0, -5.0);
    assert_eq!(shape(&b.bytes()).expect("valid").calibration, None);
}

/// `bitpix` is redundant with `datatype`, and a file where they disagree has
/// one of them wrong with no way to tell which.
#[test]
fn a_datatype_that_disagrees_with_bitpix_is_refused() {
    // int16 (4) declared as 8 bits.
    let err = shape(&Build::new(4, 3, 1, 1, 4, 8).bytes()).expect_err("should refuse");
    assert!(
        format!("{err}").contains("bitpix"),
        "should name the field: {err}"
    );
}

/// Slope 0 is the format's "no scaling". Applied literally it multiplies every
/// voxel by nothing, which yields a uniform image — plausible as a failed
/// acquisition and so not obviously a bug.
#[test]
fn a_slope_of_zero_means_no_scaling_rather_than_a_scale_of_zero() {
    let mut b = Build::new(4, 3, 1, 1, 4, 16);
    b.scl(0.0, 7.0);
    assert_eq!(shape(&b.bytes()).expect("valid").calibration, None);
}

/// An identity calibration is no calibration. Reporting it would put a
/// redundant `0 + 1 * x` on every ordinary file.
#[test]
fn an_identity_calibration_is_not_reported() {
    let mut b = Build::new(4, 3, 1, 1, 4, 16);
    b.scl(1.0, 0.0);
    assert_eq!(shape(&b.bytes()).expect("valid").calibration, None);
}

#[test]
fn a_real_calibration_is_reported_as_intercept_then_slope() {
    let mut b = Build::new(4, 3, 1, 1, 4, 16);
    b.scl(2.0, -1024.0);
    assert_eq!(
        shape(&b.bytes()).expect("valid").calibration,
        Some((-1024.0, 2.0)),
        "the host's pair is (c0, c1) for c0 + c1 * raw"
    );
}

/// The units byte packs two units into one, and the file this is taken from is
/// the common case: millimetres and seconds together are `2 | 8 == 10`, which
/// read whole is not a unit at all.
#[test]
fn the_packed_units_byte_is_masked_per_axis() {
    assert_eq!(space_unit(2 | 8), Some("mm"));
    assert_eq!(space_unit(3 | 16), Some("micron"));
    assert_eq!(space_unit(1), Some("meter"));
    assert_eq!(space_unit(0), None);
    // Time without space, and space without time.
    assert_eq!(time_interval_s(2.0, 2 | 8), Some(2.0));
    assert_eq!(time_interval_s(2.0, 2), None);
}

/// A TR is as likely to be in milliseconds, and 2000 s where 2 s was meant
/// would put a ten-minute scan at forty days.
#[test]
fn a_repetition_time_is_converted_to_seconds() {
    assert_eq!(time_interval_s(2.0, 8), Some(2.0));
    assert_eq!(time_interval_s(2000.0, 16), Some(2.0));
    assert_eq!(time_interval_s(2_000_000.0, 24), Some(2.0));
    // Hz, ppm and rad/s are spectroscopy axes, not time: there is no interval
    // to report and guessing seconds would put a wrong number on the scrubber.
    for spectral in [32u8, 40, 48] {
        assert_eq!(time_interval_s(2.0, spectral), None, "unit {spectral}");
    }
    // And nothing nonsensical gets through.
    assert_eq!(time_interval_s(0.0, 8), None);
    assert_eq!(time_interval_s(-1.0, 8), None);
    assert_eq!(time_interval_s(f32::NAN as f64, 8), None);
}

/// The `.hdr`/`.img` pair keeps its voxels in a second file. Saying so beats a
/// size mismatch, which is what reading the header half alone would produce.
#[test]
fn the_header_half_of_a_pair_is_refused_with_its_own_message() {
    let mut b = Build::new(4, 3, 1, 1, 4, 16);
    b.magic(b"ni1\0");
    let err = shape(&b.bytes()).expect_err("should refuse");
    let msg = format!("{err}");
    assert!(msg.contains(".hdr"), "should name the pair: {msg}");
}

/// A `vox_offset` below the header's own length is **clamped**, not refused.
///
/// This test exists because the obvious reading is the wrong one. Voxels inside
/// the header cannot be meant literally — reading them there would decode the
/// header as picture — so refusing looks like the careful answer, and it is
/// what this reader did at first. The format disagrees: `nifti1.h`, under
/// DETAILS ABOUT vox_offset, says that in a `.nii` file a `vox_offset` below
/// 352 is *equivalent to* 352. That makes such files valid and refusing them a
/// conformance bug, and writers that leave the field at its default do produce
/// them.
#[test]
fn a_vox_offset_inside_the_header_is_clamped_to_just_after_it() {
    for small in [0.0f32, 4.0, 100.0, 351.0, 351.9] {
        let mut b = Build::new(4, 3, 1, 1, 4, 16);
        b.f32_at(108, small);
        assert_eq!(
            shape(&b.bytes()).expect("valid").data_at,
            352,
            "vox_offset {small} should clamp to 352"
        );
    }
}

/// Negative and non-finite are a different thing: not a small offset but a
/// field that does not hold one, so there is nothing to clamp towards.
#[test]
fn a_vox_offset_that_is_not_an_offset_is_refused() {
    for bad in [-1.0f32, -352.0, f32::NAN, f32::INFINITY] {
        let mut b = Build::new(4, 3, 1, 1, 4, 16);
        b.f32_at(108, bad);
        assert!(
            shape(&b.bytes()).is_err(),
            "vox_offset {bad} should be refused"
        );
    }
}

/// A header extension pushes the voxels back legitimately, and the field is
/// then authoritative.
#[test]
fn a_vox_offset_past_the_header_is_honoured() {
    let mut b = Build::new(4, 3, 1, 1, 4, 16);
    b.f32_at(108, 1024.0);
    assert_eq!(shape(&b.bytes()).expect("valid").data_at, 1024);
}

/// `dim[5]` is a vector per voxel — a DTI tensor, a deformation field. Folding
/// it into the time axis would show a 3-vector field as three timepoints,
/// which is a plausible-looking movie of something that is not one.
#[test]
fn a_vector_valued_volume_is_refused_rather_than_shown_as_a_movie() {
    let mut b = Build::new(4, 3, 2, 1, 16, 32);
    b.i16_at(40, 5).i16_at(50, 3); // dim[0] = 5, dim[5] = 3
    let err = shape(&b.bytes()).expect_err("should refuse");
    assert!(
        format!("{err}").contains("dim[5]"),
        "should name the axis: {err}"
    );
}

/// The format says an unused axis is 1, and files in the wild write 0. A zero
/// taken literally makes the volume empty rather than flat.
#[test]
fn a_dimension_written_as_zero_is_read_as_one() {
    let b = Build::new(4, 3, 0, 0, 4, 16);
    let s = shape(&b.bytes()).expect("valid");
    assert_eq!((s.slices, s.frames), (1, 1));
}

#[test]
fn the_voxel_sizes_and_description_come_through() {
    let mut b = Build::new(4, 3, 2, 1, 4, 16);
    b.pixdim(1, 0.9375)
        .pixdim(2, 0.9375)
        .pixdim(3, 1.2)
        .units(2)
        .descrip("FSL5.0");
    let s = shape(&b.bytes()).expect("valid");
    // Compared with a tolerance, because `pixdim` is `f32` on the wire and the
    // host's spacing is `f64`: 1.2 widened from a float is 1.2000000476837158,
    // and a reader that produced exactly 1.2 would have got it from somewhere
    // other than the file. 0.9375 is exact in both, being a dyadic fraction —
    // which is why a real scanner's isotropic spacing often compares equal and
    // the odd one out does not.
    let near = |got: Option<f64>, want: f64| got.is_some_and(|v| (v - want).abs() < 1e-6);
    assert!(near(s.spacing.0, 0.9375), "x: {:?}", s.spacing.0);
    assert!(near(s.spacing.1, 0.9375), "y: {:?}", s.spacing.1);
    assert!(near(s.spacing.2, 1.2), "z: {:?}", s.spacing.2);
    assert_eq!(s.unit.as_deref(), Some("mm"));
    assert_eq!(s.descrip.as_deref(), Some("FSL5.0"));
    // A zero or negative pixdim is no information, not a zero-sized voxel.
    let mut z = Build::new(4, 3, 2, 1, 4, 16);
    z.pixdim(1, 0.0).pixdim(2, -1.0);
    let s = shape(&z.bytes()).expect("valid");
    assert_eq!((s.spacing.0, s.spacing.1), (None, None));
}

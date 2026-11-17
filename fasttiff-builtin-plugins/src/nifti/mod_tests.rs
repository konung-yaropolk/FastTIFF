//! The NIfTI importer, end to end through a real file.
//!
//! The header's own arithmetic is tested next door; this is about the part that
//! only shows up once a whole file is read: that every voxel ends up in the
//! plane it belongs to, that the gzipped and plain forms agree byte for byte,
//! and that a signed volume keeps its signs.
//!
//! Every file here is built in the test rather than committed as a fixture.
//! NIfTI is small enough to synthesise exactly — 348 bytes and then the voxels
//! — and a synthesised file is the only kind whose every byte is known, which
//! is what lets "voxel 37 is in plane 2 at index 5" be asserted rather than
//! eyeballed. No real MRI data is needed, or wanted: a patient's scan is not
//! something to commit to a repository.

use super::header::tests::{counted_u16, Build};
use super::*;
use fasttiff_plugin_api::Params;

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fasttiff-nifti-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let p = dir.join(name);
    let _ = std::fs::remove_file(&p);
    p
}

#[derive(Default)]
struct Notes {
    lines: Vec<String>,
    progress: Vec<f32>,
}

impl ImportHost for Notes {
    fn progress(&mut self, fraction: f32) -> bool {
        self.progress.push(fraction);
        true
    }
    fn log(&mut self, message: &str) {
        self.lines.push(message.to_string());
    }
}

impl Notes {
    fn said(&self, needle: &str) -> bool {
        self.lines.iter().any(|l| l.contains(needle))
    }
}

/// A host that refuses at once, for the cancellation path.
struct Stops;
impl ImportHost for Stops {
    fn progress(&mut self, _fraction: f32) -> bool {
        false
    }
    fn log(&mut self, _message: &str) {}
}

fn import(path: &std::path::Path, host: &mut dyn ImportHost) -> Result<ImportResult, PluginError> {
    Nifti.import(
        &ImportRequest {
            path: path.to_path_buf(),
            params: Params::new(),
        },
        host,
    )
}

/// Gzip some bytes, so the compressed path is exercised against a file this
/// test made rather than one checked in.
fn gzipped(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(bytes).expect("gzip");
    e.finish().expect("gzip")
}

/// The headline property: a 4-D volume arrives as slices within frames, with
/// every voxel in the plane it was written to.
///
/// The voxels count 0, 1, 2, … across the whole volume, so a plane that came
/// from the wrong offset holds the wrong run of numbers — which is checked
/// exactly rather than by a checksum, because an off-by-one plane and an
/// off-by-one *row* are different bugs and the numbers say which.
#[test]
fn every_voxel_of_a_four_dimensional_volume_lands_in_its_own_plane() {
    let (w, h, d, t) = (4i16, 3i16, 2i16, 5i16);
    let n_px = (w as usize) * (h as usize);
    let n_planes = (d as usize) * (t as usize);
    let b = Build::new(w, h, d, t, 512, 16); // uint16
    let path = temp("4d.nii");
    std::fs::write(&path, b.with_data(&counted_u16(n_px * n_planes))).expect("write");

    let mut notes = Notes::default();
    let got = import(&path, &mut notes).expect("import");
    assert_eq!(got.image.width, w as u32);
    assert_eq!(got.image.height, h as u32);
    assert_eq!(got.image.slices, d as usize);
    assert_eq!(got.image.frames, t as usize);
    assert_eq!(got.image.channels, 1);
    assert_eq!(got.image.planes.len(), n_planes);
    assert_eq!(got.image.pixel_type, PixelType::U16);

    // `xyczt` with one channel is z within t, which is also the file's own
    // order — so plane `i` holds voxels `i * n_px ..`.
    for (i, plane) in got.image.planes.iter().enumerate() {
        let PlaneData::U16(v) = plane else {
            panic!("plane {i} is not u16: {plane:?}")
        };
        let want: Vec<u16> = (0..n_px).map(|j| (i * n_px + j) as u16).collect();
        assert_eq!(v, &want, "plane {i} holds the wrong run of voxels");
    }
    assert!(
        notes.progress.last().is_some_and(|&p| p >= 1.0),
        "should finish at 100%: {:?}",
        notes.progress
    );
    let _ = std::fs::remove_file(&path);
}

/// A structural scan: one timepoint, several slices, which is the common case
/// and must not come out as frames.
#[test]
fn a_three_dimensional_volume_opens_as_a_z_stack_not_a_movie() {
    let b = Build::new(4, 3, 6, 1, 512, 16);
    let path = temp("3d.nii");
    std::fs::write(&path, b.with_data(&counted_u16(4 * 3 * 6))).expect("write");
    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!((got.image.slices, got.image.frames), (6, 1));
    let _ = std::fs::remove_file(&path);
}

/// The two distribution forms have to agree. `.nii.gz` is how a dataset is
/// actually shipped, and a decompression that dropped a member or mangled the
/// tail would give a short volume rather than an error.
#[test]
fn the_gzipped_form_reads_identically_to_the_plain_one() {
    let b = Build::new(5, 4, 3, 2, 512, 16);
    let raw = b.with_data(&counted_u16(5 * 4 * 3 * 2));

    let plain = temp("same.nii");
    std::fs::write(&plain, &raw).expect("write");
    let packed = temp("same.nii.gz");
    std::fs::write(&packed, gzipped(&raw)).expect("write");

    let a = import(&plain, &mut Notes::default()).expect("plain");
    let z = import(&packed, &mut Notes::default()).expect("gzipped");
    assert_eq!(
        a.image.planes, z.image.planes,
        "the voxels must be the same"
    );
    assert_eq!(a.image.width, z.image.width);
    assert_eq!(a.image.slices, z.image.slices);
    assert_eq!(a.image.frames, z.image.frames);
    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&packed);
}

/// Gzip detected by content, not by name — a `.nii` that was compressed in
/// place without being renamed is common enough, and costs two bytes to spot.
#[test]
fn a_gzipped_file_under_a_plain_name_is_still_read() {
    let b = Build::new(4, 3, 1, 1, 512, 16);
    let path = temp("misnamed.nii");
    std::fs::write(&path, gzipped(&b.with_data(&counted_u16(12)))).expect("write");
    assert!(import(&path, &mut Notes::default()).is_ok());
    let _ = std::fs::remove_file(&path);
}

/// Signed voxels are what MRI and CT actually contain, and the contract has no
/// `PlaneData::I16` — they travel as the same sixteen bits in the `U16` lane
/// with `PixelType::I16` declared. Getting that wrong makes every negative
/// voxel read as a very bright one, which still looks like a picture.
#[test]
fn signed_voxels_keep_their_sign_through_the_unsigned_lane() {
    let values: [i16; 6] = [-32768, -1024, -1, 0, 1, 32767];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b = Build::new(6, 1, 1, 1, 4, 16); // int16
    let path = temp("signed.nii");
    std::fs::write(&path, b.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(
        got.image.pixel_type,
        PixelType::I16,
        "the host has to be told these are signed"
    );
    let PlaneData::U16(v) = &got.image.planes[0] else {
        panic!("expected the u16 lane")
    };
    let back: Vec<i16> = v.iter().map(|&u| u as i16).collect();
    assert_eq!(back, values, "the bits must round-trip through the lane");
    let _ = std::fs::remove_file(&path);
}

/// `int8` is widened to the same lane, losslessly, and must keep its sign too.
#[test]
fn signed_bytes_are_widened_without_losing_their_sign() {
    let values: [i8; 4] = [-128, -1, 0, 127];
    let data: Vec<u8> = values.iter().map(|&v| v as u8).collect();
    let b = Build::new(4, 1, 1, 1, 256, 8); // int8
    let path = temp("int8.nii");
    std::fs::write(&path, b.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(got.image.pixel_type, PixelType::I16);
    let PlaneData::U16(v) = &got.image.planes[0] else {
        panic!("expected the u16 lane")
    };
    let back: Vec<i16> = v.iter().map(|&u| u as i16).collect();
    assert_eq!(back, vec![-128i16, -1, 0, 127]);
    let _ = std::fs::remove_file(&path);
}

/// Float volumes are the output of every processing pipeline, so they are the
/// common case for anything that is not straight off a scanner.
#[test]
fn float_voxels_arrive_as_floats() {
    let values: [f32; 4] = [-1.5, 0.0, 0.25, 1e6];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b = Build::new(4, 1, 1, 1, 16, 32); // float32
    let path = temp("f32.nii");
    std::fs::write(&path, b.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(got.image.pixel_type, PixelType::F32);
    assert_eq!(got.image.planes[0], PlaneData::F32(values.to_vec()));
    let _ = std::fs::remove_file(&path);
}

/// A widened type says so, because someone measuring a label map carried as
/// `f32` has a right to know its larger values are no longer distinguishable.
#[test]
fn a_lossy_conversion_is_reported_rather_than_performed_silently() {
    let b = Build::new(2, 1, 1, 1, 8, 32); // int32
    let path = temp("i32.nii");
    std::fs::write(&path, b.with_data(&[0u8; 8])).expect("write");
    let mut notes = Notes::default();
    import(&path, &mut notes).expect("import");
    assert!(
        notes.said("int32") && notes.said("16777216"),
        "should say what it did and where it stops being exact: {:?}",
        notes.lines
    );
    let _ = std::fs::remove_file(&path);
}

/// The metadata the viewer can actually honour: voxel sizes, their unit, the
/// repetition time, the calibration and whatever wrote the file.
#[test]
fn the_scan_geometry_reaches_the_stack_info() {
    let mut b = Build::new(4, 3, 2, 4, 512, 16);
    b.pixdim(1, 2.0)
        .pixdim(2, 2.0)
        .pixdim(3, 3.0)
        .pixdim(4, 2500.0)
        .units(2 | 16) // mm and milliseconds
        .scl(2.0, -1000.0)
        .descrip("dcm2niix");
    let path = temp("meta.nii");
    std::fs::write(&path, b.with_data(&counted_u16(4 * 3 * 2 * 4))).expect("write");

    let info = import(&path, &mut Notes::default())
        .expect("import")
        .info
        .expect("an importer that read a header should say what it said");
    assert_eq!(info.unit.as_deref(), Some("mm"));
    assert_eq!(info.spacing.x, Some(2.0));
    assert_eq!(info.spacing.z, Some(3.0));
    assert_eq!(
        info.frame_interval_s,
        Some(2.5),
        "2500 ms is 2.5 s, not 2500 s"
    );
    assert_eq!(info.calibration, Some((-1000.0, 2.0)));
    assert_eq!(info.description.as_deref(), Some("dcm2niix"));
    assert_eq!(info.mode, DisplayMode::Grayscale);
    assert!(info.path.is_some(), "the file it came from");
    let _ = std::fs::remove_file(&path);
}

/// A file that promises more voxels than it holds is an error with a reason,
/// not a panic on a slice index and not a volume with a plane of zeros.
#[test]
fn a_truncated_volume_is_reported_rather_than_padded() {
    let b = Build::new(8, 8, 4, 1, 512, 16);
    let path = temp("short.nii");
    // Two planes' worth of voxels where the header declares four.
    std::fs::write(&path, b.with_data(&counted_u16(8 * 8 * 2))).expect("write");
    let err = import(&path, &mut Notes::default()).expect_err("should refuse");
    assert!(
        format!("{err}").contains("ends in the middle"),
        "should say what is wrong: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cancelling_stops_the_read() {
    let b = Build::new(4, 3, 8, 1, 512, 16);
    let path = temp("cancel.nii");
    std::fs::write(&path, b.with_data(&counted_u16(4 * 3 * 8))).expect("write");
    let err = import(&path, &mut Stops).expect_err("a refusing host stops it");
    assert!(format!("{err}").contains("cancelled"), "{err}");
    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------------------------- probing

/// The reader has to claim `.gz` outright, because `brain.nii.gz` has `gz` for
/// its extension and nothing finer can be declared. `probe` is therefore the
/// only thing standing between this importer and every tarball in the
/// directory, and this is the test that says it does.
#[test]
fn someone_elses_gzip_is_declined() {
    let gz = gzipped(b"this is not a volume");
    let p = std::path::Path::new("sources.tar.gz");
    assert_eq!(Nifti.probe(p, &gz), Confidence::No);
    // And one that *is* named like a NIfTI is taken, on the name, because
    // looking inside would mean inflating every `.gz` to answer a question
    // about one file.
    let named = std::path::Path::new("brain.nii.gz");
    assert_eq!(Nifti.probe(named, &gz), Confidence::Certain);
}

/// An uncompressed header speaks for itself, in either byte order — so a NIfTI
/// under the wrong name is still read, which is the point of probing at all.
#[test]
fn an_uncompressed_header_is_recognised_whatever_the_file_is_called() {
    let b = Build::new(4, 3, 2, 1, 512, 16);
    let head = b.bytes();
    assert_eq!(
        Nifti.probe(std::path::Path::new("scan.dat"), &head),
        Confidence::Certain
    );
    let be = Build::new(4, 3, 2, 1, 512, 16).big_endian().bytes();
    assert_eq!(
        Nifti.probe(std::path::Path::new("scan.dat"), &be),
        Confidence::Certain
    );
    // Junk under a `.nii` name is still offered, on the extension, so the
    // refusal comes from `import` with a reason rather than from the host with
    // "nothing here can read this".
    assert_eq!(
        Nifti.probe(std::path::Path::new("scan.nii"), &[0x42; 400]),
        Confidence::Maybe
    );
    // Junk under a name nobody claims is not this importer's business.
    assert_eq!(
        Nifti.probe(std::path::Path::new("scan.dat"), &[0x42; 400]),
        Confidence::No
    );
}

/// The file types have to include `gz`, or `.nii.gz` — which is how nearly
/// every NIfTI is distributed — would never reach the importer at all.
#[test]
fn the_declared_extensions_cover_both_distribution_forms() {
    let types = Nifti.file_types();
    let exts: Vec<&str> = types
        .iter()
        .flat_map(|t| t.extensions.iter().map(String::as_str))
        .collect();
    assert!(exts.contains(&"nii"), "{exts:?}");
    assert!(exts.contains(&"gz"), "{exts:?}");
    assert!(types
        .iter()
        .any(|t| t.matches(std::path::Path::new("a.nii"))));
    assert!(types
        .iter()
        .any(|t| t.matches(std::path::Path::new("a.nii.gz"))));
}

// ------------------------------------------------------------------- colour

/// A colour volume opens as a channel per component.
///
/// The file that prompted this is the Visible Human cryosection atlas, which is
/// distributed as `RGB24` — so "not what an MRI is" was true and beside the
/// point: refusing it meant a file that would not open at all.
///
/// The components are interleaved per voxel, so this is also the test that the
/// deinterleaving is right: a reader that took the first third of the bytes as
/// red would produce three plausible greyscale planes of the wrong thing.
#[test]
fn an_rgb_volume_opens_as_three_channels_in_component_order() {
    let (w, h, d) = (2i16, 2i16, 2i16);
    let n_px = (w as usize) * (h as usize);
    // Voxel i gets (i, i + 100, i + 200), so every component of every voxel is
    // a distinct number and a mix-up is visible rather than merely wrong.
    let mut data = Vec::new();
    for i in 0..(n_px * d as usize) {
        data.extend_from_slice(&[i as u8, (i + 100) as u8, (i + 200) as u8]);
    }
    let b = Build::new(w, h, d, 1, 128, 24); // RGB24
    let path = temp("rgb.nii");
    std::fs::write(&path, b.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(got.image.channels, 3);
    assert_eq!(got.image.pixel_type, PixelType::U8);
    assert_eq!(
        got.image.planes.len(),
        3 * d as usize,
        "a plane per component per slice"
    );
    assert_eq!(
        got.info.as_ref().expect("info").mode,
        DisplayMode::Composite,
        "three components of one picture, not three greys"
    );

    // `xyczt` wants the channel innermost, so slice z's planes are at
    // 3*z .. 3*z+3 in red, green, blue order.
    for z in 0..d as usize {
        let base = z * n_px;
        for (c, offset) in [(0usize, 0u8), (1, 100), (2, 200)] {
            let PlaneData::U8(v) = &got.image.planes[z * 3 + c] else {
                panic!("plane {} of slice {z} is not u8", c)
            };
            let want: Vec<u8> = (0..n_px).map(|i| (base + i) as u8 + offset).collect();
            assert_eq!(v, &want, "slice {z}, component {c}");
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// RGBA keeps its alpha as a fourth channel rather than compositing it away.
/// In a segmentation or a surface rendering saved to NIfTI the alpha is data.
#[test]
fn an_rgba_volume_keeps_its_alpha_as_a_fourth_channel() {
    let data: Vec<u8> = (0..4u8).flat_map(|i| [i, i + 10, i + 20, i + 30]).collect();
    let b = Build::new(2, 2, 1, 1, 2304, 32); // RGBA32
    let path = temp("rgba.nii");
    std::fs::write(&path, b.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(got.image.channels, 4);
    assert_eq!(got.image.planes.len(), 4);
    assert_eq!(got.image.planes[3], PlaneData::U8(vec![30, 31, 32, 33]));
    let _ = std::fs::remove_file(&path);
}

/// A byte-swapped colour volume must *not* have its components reversed — they
/// are single bytes, so there is nothing to swap, and swapping by
/// bytes-per-voxel would turn every red pixel blue.
#[test]
fn a_byte_swapped_colour_volume_keeps_its_components_in_order() {
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let be = Build::new(2, 2, 1, 1, 128, 24).big_endian();
    let path = temp("rgb_be.nii");
    std::fs::write(&path, be.with_data(&data)).expect("write");

    let got = import(&path, &mut Notes::default()).expect("import");
    assert_eq!(got.image.channels, 3);
    assert_eq!(
        got.image.planes[0],
        PlaneData::U8(vec![1, 4, 7, 10]),
        "red is every third byte, whatever the header's byte order"
    );
    let _ = std::fs::remove_file(&path);
}

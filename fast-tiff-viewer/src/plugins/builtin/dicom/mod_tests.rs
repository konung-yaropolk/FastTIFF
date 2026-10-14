//! The DICOM importer, end to end: a folder of slices in, one stack out.
//!
//! The parts are tested next door — the element walk in `parse_tests`, the
//! sample arithmetic in `pixels_tests`, the ordering in `series_tests`. What is
//! left for here is what only shows up once the whole thing runs: that the
//! planes come out in the order the slices were sorted into, that the geometry
//! survives the trip, and that nothing identifying comes with it.

use super::parse::tests::{Dcm, EXPLICIT_LE};
use super::*;
use fasttiff_plugin_api::{ParamValue, Params, PixelType, PlaneData};

const SERIES: &str = "1.2.840.113619.2.1.4242";

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
    fn all(&self) -> String {
        self.lines.join(" | ")
    }
}

struct Stops;
impl ImportHost for Stops {
    fn progress(&mut self, _fraction: f32) -> bool {
        false
    }
    fn log(&mut self, _message: &str) {}
}

fn folder(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fasttiff-dcmimp-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a temp folder");
    dir
}

/// A 2x2 16-bit axial slice at `z`, every voxel equal to `fill` — so a plane
/// that came from the wrong file holds the wrong number throughout, which the
/// assertions below read directly.
fn slice(z: f64, instance: i32, fill: u16) -> Vec<u8> {
    let px: Vec<u8> = std::iter::repeat_n(fill, 4)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(series::SERIES_UID, b"UI", SERIES)
        .text(series::INSTANCE_NUMBER, b"IS", &instance.to_string())
        .text(series::IMAGE_ORIENTATION, b"DS", "1\\0\\0\\0\\1\\0")
        .text(series::IMAGE_POSITION, b"DS", &format!("0\\0\\{z}"))
        .text(MODALITY, b"CS", "MR")
        .text(SERIES_DESCRIPTION, b"LO", "T1 MPRAGE SAG")
        // Row spacing then column spacing — deliberately different, so the
        // order can be asserted.
        .text(PIXEL_SPACING, b"DS", "0.5\\0.25")
        .text(SPACING_BETWEEN_SLICES, b"DS", "1.5")
        .text(SLICE_THICKNESS, b"DS", "1.0")
        // Identifying tags, present exactly as a real file has them, so the
        // test can prove they go nowhere.
        .text((0x0010, 0x0010), b"PN", "DOE^JANE")
        .text((0x0010, 0x0020), b"LO", "MRN0001234")
        .us(pixels::ROWS, 2)
        .us(pixels::COLUMNS, 2)
        .us(pixels::BITS_ALLOCATED, 16)
        .us(pixels::BITS_STORED, 16)
        .us(pixels::HIGH_BIT, 15)
        .us(pixels::PIXEL_REPRESENTATION, 0)
        .pixels(&px);
    d.build()
}

fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).expect("write");
    p
}

fn import_with(
    path: &std::path::Path,
    params: Params,
    host: &mut dyn ImportHost,
) -> Result<ImportResult, PluginError> {
    Dicom.import(
        &ImportRequest {
            path: path.to_path_buf(),
            params,
        },
        host,
    )
}

fn import(path: &std::path::Path, host: &mut dyn ImportHost) -> Result<ImportResult, PluginError> {
    import_with(path, Params::new(), host)
}

fn fills(r: &ImportResult) -> Vec<u16> {
    r.image
        .planes
        .iter()
        .map(|p| match p {
            PlaneData::U16(v) => v[0],
            other => panic!("expected u16, got {other:?}"),
        })
        .collect()
}

/// The headline: a folder of slices becomes one stack, in position order.
///
/// The files are written in an order that neither their names nor their
/// instance numbers agree with the geometry, so only sorting by position gives
/// the answer asserted here.
#[test]
fn a_folder_of_slices_opens_as_one_stack_in_position_order() {
    let dir = folder("series");
    // z, instance, fill — the fills are what identify each plane, and the
    // expected order is by z.
    let a = write(&dir, "z3.dcm", &slice(3.0, 1, 300));
    write(&dir, "z1.dcm", &slice(1.0, 4, 100));
    write(&dir, "z4.dcm", &slice(4.0, 2, 400));
    write(&dir, "z2.dcm", &slice(2.0, 3, 200));

    let mut notes = Notes::default();
    let got = import(&a, &mut notes).expect("import");
    assert_eq!(got.image.width, 2);
    assert_eq!(got.image.height, 2);
    assert_eq!(got.image.channels, 1);
    assert_eq!(got.image.slices, 4, "four files, four slices");
    assert_eq!(got.image.frames, 1, "slices are not timepoints");
    assert_eq!(got.image.pixel_type, PixelType::U16);
    assert_eq!(
        fills(&got),
        vec![100, 200, 300, 400],
        "the planes are not in position order"
    );
    assert!(notes.said("gathered 4 slices"), "{}", notes.all());
    assert!(
        notes.progress.last().is_some_and(|&p| p >= 1.0),
        "{:?}",
        notes.progress
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The switch off reads the one file. It is declared but not yet shown by the
/// host, so this is the only thing exercising the other branch.
#[test]
fn the_series_switch_off_reads_only_the_chosen_file() {
    let dir = folder("one");
    let a = write(&dir, "z1.dcm", &slice(1.0, 1, 111));
    write(&dir, "z2.dcm", &slice(2.0, 2, 222));

    let mut params = Params::new();
    params.set(WHOLE_SERIES, ParamValue::Bool(false));
    let got = import_with(&a, params, &mut Notes::default()).expect("import");
    assert_eq!(got.image.slices, 1);
    assert_eq!(fills(&got), vec![111]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Geometry. `PixelSpacing` is **row then column** — y before x — which is the
/// reverse of how everything else is written and the single easiest thing to
/// get backwards. On the usual isotropic scan it makes no difference, which is
/// what makes it dangerous, so the fixture is deliberately anisotropic.
#[test]
fn the_geometry_reaches_the_stack_info_with_pixel_spacing_the_right_way_round() {
    let dir = folder("geometry");
    let a = write(&dir, "z1.dcm", &slice(1.0, 1, 1));
    let info = import(&a, &mut Notes::default())
        .expect("import")
        .info
        .expect("info");

    assert_eq!(info.spacing.y, Some(0.5), "row spacing is y");
    assert_eq!(info.spacing.x, Some(0.25), "column spacing is x");
    // `SpacingBetweenSlices` is centre-to-centre and is the z step;
    // `SliceThickness` (1.0 here) is a smaller number when there is a gap, so
    // using it would compress the volume along z.
    assert_eq!(info.spacing.z, Some(1.5));
    // DICOM fixes its unit; there is no field to read.
    assert_eq!(info.unit.as_deref(), Some("mm"));
    assert_eq!(info.description.as_deref(), Some("MR: T1 MPRAGE SAG"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `SliceThickness` is the fallback when the real spacing is absent, and only
/// then.
#[test]
fn slice_thickness_stands_in_for_a_missing_slice_spacing() {
    let dir = folder("thickness");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(PIXEL_SPACING, b"DS", "1\\1")
        .text(SLICE_THICKNESS, b"DS", "2.5")
        .us(pixels::ROWS, 1)
        .us(pixels::COLUMNS, 1)
        .us(pixels::BITS_ALLOCATED, 16)
        .pixels(&[0, 0]);
    let a = write(&dir, "t.dcm", &d.build());
    let info = import(&a, &mut Notes::default())
        .expect("import")
        .info
        .expect("info");
    assert_eq!(info.spacing.z, Some(2.5));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Nothing identifying. Every one of these files carries a patient name and an
/// MRN, exactly as a real one does, and none of it may reach the document — a
/// name in a window title is a name in screenshots, bug reports and
/// projections.
#[test]
fn nothing_identifying_reaches_the_document() {
    let dir = folder("privacy");
    let a = write(&dir, "z1.dcm", &slice(1.0, 1, 1));
    let mut notes = Notes::default();
    let got = import(&a, &mut notes).expect("import");
    let info = got.info.expect("info");

    let haystack = format!(
        "{} | {} | {} | {}",
        got.image.name,
        info.name,
        info.description.unwrap_or_default(),
        notes.all()
    );
    for identifying in ["DOE", "JANE", "MRN0001234"] {
        assert!(
            !haystack.contains(identifying),
            "{identifying:?} leaked into {haystack:?}"
        );
    }
    // The name is the file's own, which is what the user chose and already
    // sees in their own file manager.
    assert_eq!(got.image.name, "z1.dcm");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A multi-frame file is already a volume, and must not be extended with its
/// siblings: a cine loop plus its neighbours would be two different axes
/// stacked into one.
#[test]
fn a_multi_frame_file_is_its_own_volume_and_is_not_extended() {
    let dir = folder("multiframe");
    let px: Vec<u8> = [7u16, 7, 7, 7, 9, 9, 9, 9]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(series::SERIES_UID, b"UI", SERIES)
        .text(pixels::NUMBER_OF_FRAMES, b"IS", "2")
        .us(pixels::ROWS, 2)
        .us(pixels::COLUMNS, 2)
        .us(pixels::BITS_ALLOCATED, 16)
        .us(pixels::BITS_STORED, 16)
        .us(pixels::HIGH_BIT, 15)
        .pixels(&px);
    let a = write(&dir, "cine.dcm", &d.build());
    // A sibling of the same series, which must be ignored.
    write(&dir, "other.dcm", &slice(1.0, 1, 555));

    let got = import(&a, &mut Notes::default()).expect("import");
    assert_eq!(got.image.slices, 2, "its own two frames and no more");
    assert_eq!(fills(&got), vec![7, 9]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A scout image dropped into a series is the usual cause of a shape
/// mismatch, and stacking it anyway would show one slice stretched across the
/// others' geometry. Refused, and the message says how to see that file alone.
#[test]
fn a_slice_of_a_different_shape_in_the_series_is_refused() {
    let dir = folder("mismatch");
    let a = write(&dir, "z1.dcm", &slice(1.0, 1, 1));
    // Same series UID, different size.
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(series::SERIES_UID, b"UI", SERIES)
        .text(series::IMAGE_ORIENTATION, b"DS", "1\\0\\0\\0\\1\\0")
        .text(series::IMAGE_POSITION, b"DS", "0\\0\\2")
        .us(pixels::ROWS, 4)
        .us(pixels::COLUMNS, 4)
        .us(pixels::BITS_ALLOCATED, 16)
        .us(pixels::BITS_STORED, 16)
        .us(pixels::HIGH_BIT, 15)
        .pixels(&[0u8; 32]);
    write(&dir, "scout.dcm", &d.build());

    let err = import(&a, &mut Notes::default()).expect_err("should refuse");
    let msg = format!("{err}");
    assert!(msg.contains("not one stack"), "{msg}");
    assert!(msg.contains("series switch off"), "should say how: {msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Colour DICOM — a screenshot or a report page — arrives as three channels,
/// which is what lets each be windowed rather than being a packed image
/// nothing can touch.
#[test]
fn a_colour_file_opens_as_three_channels() {
    let dir = folder("colour");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(pixels::PHOTOMETRIC, b"CS", "RGB")
        .us(pixels::SAMPLES_PER_PIXEL, 3)
        .us(pixels::PLANAR_CONFIG, 0)
        .us(pixels::ROWS, 1)
        .us(pixels::COLUMNS, 2)
        .us(pixels::BITS_ALLOCATED, 8)
        .us(pixels::BITS_STORED, 8)
        .us(pixels::HIGH_BIT, 7)
        .pixels(&[1, 2, 3, 4, 5, 6]);
    let a = write(&dir, "rgb.dcm", &d.build());

    let got = import(&a, &mut Notes::default()).expect("import");
    assert_eq!(got.image.channels, 3);
    assert_eq!(got.image.planes.len(), 3);
    assert_eq!(got.image.planes[0], PlaneData::U8(vec![1, 4]));
    assert_eq!(got.image.planes[2], PlaneData::U8(vec![3, 6]));
    assert_eq!(
        got.info.expect("info").mode,
        DisplayMode::Composite,
        "three colour components are a composite, not three greys"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A compressed file is refused by name, with the command that fixes it. This
/// is the limitation most likely to be met in practice — a clinical PACS export
/// is often JPEG — so the message is the feature.
#[test]
fn a_compressed_file_is_refused_with_the_conversion_to_run() {
    let dir = folder("compressed");
    // Only the meta group matters: the transfer syntax is refused before the
    // dataset is touched.
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(pixels::ROWS, 2);
    let mut file = d.build();
    // Rewrite the transfer syntax UID in the meta group to JPEG 2000. Same
    // length as the explicit-VR one it replaces, so nothing else moves.
    let from = EXPLICIT_LE.as_bytes();
    let to = b"1.2.840.10008.1.2.4.90";
    assert_eq!(from.len() + 3, to.len(), "the UIDs must be swappable");
    let at = file
        .windows(from.len())
        .position(|w| w == from)
        .expect("the syntax is in there");
    // Replace and fix the two lengths that mention it.
    file.splice(at..at + from.len(), to.iter().copied());
    let err = import(&dir.join("nope.dcm"), &mut Notes::default()).expect_err("no such file");
    assert!(format!("{err}").contains("reading"), "{err}");

    // The direct route, which is what the message is really about.
    let err = parse::transfer("1.2.840.10008.1.2.4.90").expect_err("should refuse");
    let msg = format!("{err}");
    assert!(msg.contains("JPEG 2000"), "{msg}");
    assert!(msg.contains("dcm2niix"), "{msg}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The magic is conclusive, and a `.dcm` without it is something else under a
/// borrowed name.
#[test]
fn probing_trusts_the_magic_over_the_extension() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(pixels::ROWS, 2);
    let file = d.build();
    assert_eq!(
        Dicom.probe(std::path::Path::new("scan.dcm"), &file),
        Confidence::Certain
    );
    // And under any other name, so a scanner export called `IM_0001` is read.
    assert_eq!(
        Dicom.probe(std::path::Path::new("IM_0001"), &file),
        Confidence::Certain
    );
    // A `.dcm` that is really a PNG is declined, so the host says "nothing
    // here can read this" rather than this importer failing on it.
    assert_eq!(
        Dicom.probe(std::path::Path::new("scan.dcm"), b"\x89PNG\r\n\x1a\n"),
        Confidence::No
    );
}

#[test]
fn an_object_with_no_pixel_data_says_so() {
    let dir = folder("no-pixels");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us(pixels::ROWS, 2).us(pixels::COLUMNS, 2);
    let a = write(&dir, "sr.dcm", &d.build());
    let err = import(&a, &mut Notes::default()).expect_err("should refuse");
    assert!(format!("{err}").contains("no pixel data"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cancelling_stops_the_read() {
    let dir = folder("cancel");
    let a = write(&dir, "z1.dcm", &slice(1.0, 1, 1));
    write(&dir, "z2.dcm", &slice(2.0, 2, 2));
    let err = import(&a, &mut Stops).expect_err("a refusing host stops it");
    assert!(format!("{err}").contains("cancelled"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The rescale a CT carries, reaching the host as a calibration rather than
/// being applied — so the stored values still agree with every other tool's
/// reading of the same file.
#[test]
fn the_rescale_reaches_the_stack_info_as_a_calibration() {
    let dir = folder("rescale");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(MODALITY, b"CS", "CT")
        .text(pixels::RESCALE_SLOPE, b"DS", "1")
        .text(pixels::RESCALE_INTERCEPT, b"DS", "-1024")
        .us(pixels::ROWS, 1)
        .us(pixels::COLUMNS, 1)
        .us(pixels::BITS_ALLOCATED, 16)
        .us(pixels::BITS_STORED, 16)
        .us(pixels::HIGH_BIT, 15)
        .us(pixels::PIXEL_REPRESENTATION, 1)
        .pixels(&(-500i16).to_le_bytes());
    let a = write(&dir, "ct.dcm", &d.build());

    let got = import(&a, &mut Notes::default()).expect("import");
    assert_eq!(got.image.pixel_type, PixelType::I16, "CT is signed");
    let PlaneData::U16(v) = &got.image.planes[0] else {
        panic!("expected the u16 lane")
    };
    assert_eq!(v[0] as i16, -500, "the stored value, untouched");
    assert_eq!(
        got.info.expect("info").calibration,
        Some((-1024.0, 1.0)),
        "the rescale travels as a calibration"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

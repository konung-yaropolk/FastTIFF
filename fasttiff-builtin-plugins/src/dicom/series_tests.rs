//! Putting a folder of slices in the right order.
//!
//! The ordering is the part of this reader most able to be wrong without
//! looking wrong, so most of these tests are about producing an order that a
//! naive reader would get differently — an interleaved acquisition, a series
//! whose filenames sort the other way, a folder with two series in it.

use super::super::parse::tests::{Dcm, EXPLICIT_LE};
use super::*;
use fasttiff_plugin_api::PluginError;

const SERIES_A: &str = "1.2.840.113619.2.1.11111";
const SERIES_B: &str = "1.2.840.113619.2.1.22222";

/// A host that keeps what it was told.
#[derive(Default)]
struct Notes {
    lines: Vec<String>,
}

impl ImportHost for Notes {
    fn progress(&mut self, _fraction: f32) -> bool {
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

/// One slice file: a 1x1 image at a stated position, in a stated series.
///
/// Axial by default — row along +x, column along +y — so the slice normal is
/// +z and the position along it is simply the z coordinate. That keeps the
/// expected order readable in the tests that use it.
fn slice_file(uid: &str, instance: i32, z: f64) -> Vec<u8> {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(SERIES_UID, b"UI", uid)
        .text(INSTANCE_NUMBER, b"IS", &instance.to_string())
        .text(IMAGE_ORIENTATION, b"DS", "1\\0\\0\\0\\1\\0")
        .text(IMAGE_POSITION, b"DS", &format!("0\\0\\{z}"))
        .us((0x0028, 0x0010), 1) // Rows
        .us((0x0028, 0x0011), 1) // Columns
        .us((0x0028, 0x0100), 8) // BitsAllocated
        .us((0x0028, 0x0101), 8) // BitsStored
        .us((0x0028, 0x0102), 7) // HighBit
        .pixels(&[(instance as u8), 0]);
    d.build()
}

/// A slice file with no geometry at all, only an instance number.
fn slice_without_geometry(uid: &str, instance: i32) -> Vec<u8> {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(SERIES_UID, b"UI", uid)
        .text(INSTANCE_NUMBER, b"IS", &instance.to_string())
        .us((0x0028, 0x0010), 1)
        .us((0x0028, 0x0011), 1)
        .us((0x0028, 0x0100), 8)
        .us((0x0028, 0x0101), 8)
        .us((0x0028, 0x0102), 7)
        .pixels(&[(instance as u8), 0]);
    d.build()
}

/// A fresh folder, named after the test so two tests never share one.
fn folder(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fasttiff-dicom-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a temp folder");
    dir
}

fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).expect("write");
    p
}

/// Read a file's tags the way `gather` does, for building the `Found` its
/// caller passes in.
fn found_of(path: &std::path::Path) -> Result<Found, PluginError> {
    Ok(super::read_one(path)?.found)
}

// ------------------------------------------------------------- the geometry

/// The normal is the cross product of the two direction cosines, and for an
/// axial slice that is +z.
#[test]
fn the_slice_normal_is_the_cross_product_of_the_direction_cosines() {
    // Axial: row along +x, column along +y, so the normal is +z.
    let n = normal(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("a normal");
    assert!((n[2] - 1.0).abs() < 1e-9, "{n:?}");
    // Sagittal: row along +y, column along −z, so the normal is... +x.
    let n = normal(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0]).expect("a normal");
    assert!((n[0].abs() - 1.0).abs() < 1e-9, "{n:?}");
    // And it is a unit vector whatever the input's scale.
    let n = normal(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0]).expect("a normal");
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-9, "should be normalised: {n:?}");
}

/// A degenerate orientation gives no normal, rather than one of zero length —
/// projecting onto which would make every slice's position zero, which looks
/// like geometry and is not.
#[test]
fn a_degenerate_orientation_has_no_normal() {
    assert!(normal(&[0.0; 6]).is_none(), "all zeros");
    assert!(
        normal(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).is_none(),
        "two parallel cosines"
    );
    assert!(normal(&[1.0, 0.0, 0.0]).is_none(), "too few values");
    assert!(normal(&[]).is_none(), "none at all");
}

/// The headline: an interleaved acquisition. Instance numbers count the order
/// the scanner collected the slices in, which here is every other one down and
/// then the rest back up — so sorting by instance number interleaves the
/// volume, and sorting by position does not.
#[test]
fn slices_are_ordered_by_position_and_not_by_instance_number() {
    let dir = folder("interleaved");
    // z = 0, 2, 4 collected first (instances 1..3), then z = 1, 3, 5
    // (instances 4..6) — which is what an interleaved MR acquisition does.
    let order = [(1, 0.0), (2, 2.0), (3, 4.0), (4, 1.0), (5, 3.0), (6, 5.0)];
    let mut first = None;
    for (instance, z) in order {
        let p = write(
            &dir,
            &format!("IM_{instance}.dcm"),
            &slice_file(SERIES_A, instance, z),
        );
        if instance == 1 {
            first = Some(p);
        }
    }
    let first = first.expect("a first file");
    let f = found_of(&first).expect("readable");

    let got = gather(&first, &f, &mut Notes::default());
    assert_eq!(got.len(), 6);
    // The positions, in the order gathered. Ascending z is the right answer;
    // ascending instance number would be 0, 2, 4, 1, 3, 5.
    let zs: Vec<f64> = got
        .iter()
        .map(|p| {
            let f = found_of(p).expect("readable");
            f.nums(IMAGE_POSITION)[2]
        })
        .collect();
    assert_eq!(
        zs,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "the volume came out interleaved"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two series in one folder is the normal state of a study export. Mixing them
/// would stack a localiser into a T1 and show it as a slice of it.
#[test]
fn a_folder_with_two_series_gathers_only_the_one_asked_for() {
    let dir = folder("two-series");
    let a1 = write(&dir, "a1.dcm", &slice_file(SERIES_A, 1, 0.0));
    write(&dir, "a2.dcm", &slice_file(SERIES_A, 2, 1.0));
    write(&dir, "b1.dcm", &slice_file(SERIES_B, 1, 0.0));
    write(&dir, "b2.dcm", &slice_file(SERIES_B, 2, 1.0));
    write(&dir, "b3.dcm", &slice_file(SERIES_B, 3, 2.0));

    let f = found_of(&a1).expect("readable");
    let mut notes = Notes::default();
    let got = gather(&a1, &f, &mut notes);
    assert_eq!(got.len(), 2, "only series A: {got:?}");
    assert!(
        notes.said("leaving 3 file(s) from other series"),
        "should say what it left: {:?}",
        notes.lines
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Files that are not DICOM at all — a readme, a thumbnail — cost one open and
/// are then ignored.
#[test]
fn files_that_are_not_dicom_are_passed_over() {
    let dir = folder("mixed");
    let a1 = write(&dir, "a1.dcm", &slice_file(SERIES_A, 1, 0.0));
    write(&dir, "a2.dcm", &slice_file(SERIES_A, 2, 1.0));
    write(&dir, "README.txt", b"this is not a scan");
    write(&dir, "thumb.png", b"\x89PNG\r\n\x1a\n");
    let f = found_of(&a1).expect("readable");
    assert_eq!(gather(&a1, &f, &mut Notes::default()).len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A scanner export is as likely to be `IM_0001` or a bare UID with no
/// extension, so gathering cannot require one — requiring it would collect a
/// fraction of the series and present it as the whole thing, which is the
/// failure mode with no symptom.
#[test]
fn slices_without_a_dcm_extension_are_still_gathered() {
    let dir = folder("no-extension");
    let a1 = write(&dir, "IM_0001", &slice_file(SERIES_A, 1, 0.0));
    write(&dir, "IM_0002", &slice_file(SERIES_A, 2, 1.0));
    write(&dir, "1.2.840.113619.9999", &slice_file(SERIES_A, 3, 2.0));
    let f = found_of(&a1).expect("readable");
    assert_eq!(gather(&a1, &f, &mut Notes::default()).len(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

/// No geometry anywhere: instance number is the fallback, and the user is told
/// so — because that is exactly when an interleaved acquisition comes out
/// interleaved and nothing else would reveal it.
#[test]
fn a_series_without_positions_falls_back_to_instance_number_and_says_so() {
    let dir = folder("no-geometry");
    // Written in an order the filenames do not agree with, so the instance
    // number has to be what decides.
    let third = write(&dir, "c.dcm", &slice_without_geometry(SERIES_A, 3));
    write(&dir, "a.dcm", &slice_without_geometry(SERIES_A, 1));
    write(&dir, "b.dcm", &slice_without_geometry(SERIES_A, 2));

    let f = found_of(&third).expect("readable");
    let mut notes = Notes::default();
    let got = gather(&third, &f, &mut notes);
    let instances: Vec<f64> = got
        .iter()
        .map(|p| found_of(p).expect("readable").num(INSTANCE_NUMBER).unwrap())
        .collect();
    assert_eq!(instances, vec![1.0, 2.0, 3.0]);
    assert!(
        notes.said("interleaved"),
        "should warn what this ordering cannot promise: {:?}",
        notes.lines
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// One file missing its position must not decide the order of the rest. The
/// positioned slices keep their order and the odd one goes to the end.
#[test]
fn one_slice_without_geometry_does_not_reorder_the_others() {
    let dir = folder("one-odd");
    let a1 = write(&dir, "a1.dcm", &slice_file(SERIES_A, 1, 0.0));
    write(&dir, "a2.dcm", &slice_file(SERIES_A, 2, 1.0));
    write(&dir, "a3.dcm", &slice_file(SERIES_A, 3, 2.0));
    let odd = write(&dir, "odd.dcm", &slice_without_geometry(SERIES_A, 99));

    let f = found_of(&a1).expect("readable");
    let mut notes = Notes::default();
    let got = gather(&a1, &f, &mut notes);
    assert_eq!(got.len(), 4);
    assert_eq!(
        got.last().map(|p| p.as_path()),
        Some(odd.as_path()),
        "the file with no position belongs at the end: {got:?}"
    );
    assert!(notes.said("no position"), "{:?}", notes.lines);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without a series UID there is nothing to match siblings against, and
/// gathering on "every DICOM here" would merge unrelated acquisitions.
#[test]
fn a_file_with_no_series_uid_is_read_alone() {
    let dir = folder("no-uid");
    let mut d = Dcm::new(EXPLICIT_LE);
    d.us((0x0028, 0x0010), 1)
        .us((0x0028, 0x0011), 1)
        .us((0x0028, 0x0100), 8)
        .pixels(&[1, 0]);
    let lone = write(&dir, "lone.dcm", &d.build());
    write(&dir, "other.dcm", &slice_file(SERIES_A, 1, 0.0));

    let f = found_of(&lone).expect("readable");
    let mut notes = Notes::default();
    let got = gather(&lone, &f, &mut notes);
    assert_eq!(got, vec![lone.clone()]);
    assert!(notes.said("no Series Instance UID"), "{:?}", notes.lines);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The file asked for is in the list exactly once, however the folder
/// enumerates — a duplicate would show one slice twice and shift the rest.
#[test]
fn the_chosen_file_appears_exactly_once() {
    let dir = folder("once");
    let a1 = write(&dir, "a1.dcm", &slice_file(SERIES_A, 1, 0.0));
    write(&dir, "a2.dcm", &slice_file(SERIES_A, 2, 1.0));
    let f = found_of(&a1).expect("readable");
    let got = gather(&a1, &f, &mut Notes::default());
    assert_eq!(got.iter().filter(|p| **p == a1).count(), 1, "{got:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A lone file in its own folder is a series of one, not an error.
#[test]
fn a_single_file_is_a_series_of_one() {
    let dir = folder("alone");
    let only = write(&dir, "only.dcm", &slice_file(SERIES_A, 1, 0.0));
    let f = found_of(&only).expect("readable");
    assert_eq!(gather(&only, &f, &mut Notes::default()), vec![only.clone()]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------- the comparison itself

/// `sort_by` is entitled to panic on an inconsistent comparator, and a NaN read
/// from a file would make one. `Along` filters non-finite values out and this
/// uses `total_cmp` anyway — the cost is nothing and what it prevents is a
/// crash on a malformed file.
#[test]
fn the_comparison_is_total_and_orders_the_fallbacks_consistently() {
    use std::cmp::Ordering;
    assert_eq!(
        order(Along::Normal(1.0), Along::Normal(2.0)),
        Ordering::Less
    );
    assert_eq!(
        order(Along::Normal(2.0), Along::Normal(1.0)),
        Ordering::Greater
    );
    assert_eq!(
        order(Along::Normal(1.0), Along::Normal(1.0)),
        Ordering::Equal
    );
    // A real position beats an instance number, which beats nothing.
    assert_eq!(
        order(Along::Normal(99.0), Along::Instance(0.0)),
        Ordering::Less
    );
    assert_eq!(order(Along::Instance(0.0), Along::Unknown), Ordering::Less);
    assert_eq!(order(Along::Unknown, Along::Unknown), Ordering::Equal);
    // And a NaN, which `Along` should never produce, still compares rather
    // than poisoning the sort.
    let _ = order(Along::Normal(f64::NAN), Along::Normal(1.0));
    let _ = order(Along::Normal(1.0), Along::Normal(f64::NAN));
}

/// A position read from a file that is not a number is no position, so the
/// slice falls back rather than carrying a NaN into the sort.
#[test]
fn a_file_whose_position_is_not_a_number_falls_back() {
    let mut d = Dcm::new(EXPLICIT_LE);
    d.text(SERIES_UID, b"UI", SERIES_A)
        .text(INSTANCE_NUMBER, b"IS", "7")
        .text(IMAGE_ORIENTATION, b"DS", "1\\0\\0\\0\\1\\0")
        .text(IMAGE_POSITION, b"DS", "not\\a\\number")
        .us((0x0028, 0x0010), 1)
        .us((0x0028, 0x0011), 1)
        .us((0x0028, 0x0100), 8)
        .us((0x0028, 0x0101), 8)
        .us((0x0028, 0x0102), 7)
        .pixels(&[1, 0]);
    let dir = folder("bad-position");
    let p = write(&dir, "bad.dcm", &d.build());
    let f = found_of(&p).expect("readable");
    assert_eq!(along(&f), Along::Instance(7.0));
    let _ = std::fs::remove_dir_all(&dir);
}

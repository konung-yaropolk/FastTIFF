//! The toolbar's physical-size readout.

use super::*;

#[test]
fn a_calibrated_frame_reports_its_real_size() {
    assert_eq!(physical_size(Some(0.325), Some(0.325), Some("um"), 600, 400).as_deref(), Some("195 × 130 um"));
}

#[test]
fn anisotropic_pixels_are_not_collapsed() {
    // Common in microscopy, and a reader cannot tell a square field from a
    // rounded one if both axes are not shown.
    assert_eq!(physical_size(Some(0.2), Some(0.5), Some("um"), 1000, 1000).as_deref(), Some("200 × 500 um"));
}

#[test]
fn an_uncalibrated_or_unlabelled_frame_reports_nothing() {
    // A number without its unit is not information — it is a number that will
    // be read as whichever unit the reader expects.
    assert_eq!(physical_size(Some(0.3), Some(0.3), None, 100, 100), None);
    assert_eq!(physical_size(Some(0.3), Some(0.3), Some(""), 100, 100), None);
    assert_eq!(physical_size(None, Some(0.3), Some("um"), 100, 100), None);
    assert_eq!(physical_size(Some(0.3), None, Some("um"), 100, 100), None);
}

#[test]
fn nonsense_calibration_is_declined_rather_than_printed() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let got = physical_size(Some(bad), Some(0.3), Some("um"), 100, 100);
        assert_eq!(got, None, "{bad} was printed");
    }
}

#[test]
fn magnitudes_are_rounded_for_reading() {
    // Enough places to tell neighbouring values apart, and no trailing zeros —
    // a round number should look round.
    assert_eq!(trim_num(1234.6), "1235");
    // Rust's formatter rounds halves to even, so this is 1234 rather than
    // 1235. Recorded rather than worked around: it is a display readout, and a
    // unit in the last place either way is not worth its own rounding code.
    assert_eq!(trim_num(1234.5), "1234");
    assert_eq!(trim_num(195.0), "195");
    assert_eq!(trim_num(12.34), "12.3");
    assert_eq!(trim_num(1.5), "1.5");
    assert_eq!(trim_num(0.125), "0.125");
    assert_eq!(trim_num(2.0), "2");
}

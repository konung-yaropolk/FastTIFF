use super::*;

/// What goes in comes out, across the whole range.
#[test]
fn a_fraction_survives_the_round_trip() {
    let p = AtomicU32::new(UNKNOWN);
    for f in [0.0f32, 0.001, 0.1, 0.25, 0.5, 0.999, 1.0] {
        store(&p, f);
        let back = load(&p).expect("a stored fraction reads back");
        assert!((back - f).abs() <= 0.0005, "stored {f}, read {back}");
    }
}

/// A tenth is a tenth.
///
/// The specific failure this module was written for: one end scaled by ten
/// thousand and the other by a thousand, so 0.1 read back as a full bar.
#[test]
fn a_tenth_does_not_read_as_a_full_bar() {
    let p = AtomicU32::new(UNKNOWN);
    store(&p, 0.1);
    assert_eq!(load(&p), Some(0.1));
}

#[test]
fn nothing_reported_is_a_spinner_not_zero() {
    let p = AtomicU32::new(UNKNOWN);
    assert_eq!(load(&p), None);
    store(&p, 0.4);
    clear(&p);
    assert_eq!(load(&p), None);
}

#[test]
fn out_of_range_is_clamped_and_never_becomes_unknown() {
    let p = AtomicU32::new(UNKNOWN);
    store(&p, 7.0);
    assert_eq!(load(&p), Some(1.0));
    store(&p, -3.0);
    assert_eq!(load(&p), Some(0.0));
    store(&p, f32::NAN);
    assert_eq!(load(&p), Some(0.0));
    store(&p, f32::INFINITY);
    assert_eq!(load(&p), Some(1.0));
}

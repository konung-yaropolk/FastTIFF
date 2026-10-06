//! The pieces the four tools share: the plane index, the sample width, and the
//! checkbox that decides where a result goes.

use super::*;
use fasttiff_plugin_api::ParamValue;

// The plane order these tools rely on — channel fastest, then Z, then T — is
// pinned where it can actually be got wrong: in `tests/plugins.rs`, against a
// real stack whose every plane carries its own `(c, z, t)`. There is no
// formula here to unit-test, because none of the tools computes an index.
// Each chooses which `(z, t)` pairs to walk and hands them to `map_planes`
// already in order, so the order is the loop nesting and the integration
// tests are what hold it.

// ------------------------------------------------------------- sample width

/// A tool here changes which planes there are, not what a sample means, so the
/// width it was handed is the width it gives back.
#[test]
fn the_sample_width_survives() {
    for (src, want) in [
        (PixelType::U8, Store::U8),
        (PixelType::U16, Store::U16),
        (PixelType::I16, Store::I16),
        (PixelType::F32, Store::F32),
    ] {
        assert_eq!(Store::of(src), want);
        assert_eq!(want.pixel_type(), src);
    }
}

/// Every value of a width round-trips through `f32` exactly.
///
/// The whole justification for reading as float and storing back narrow:
/// `f32` has a 24-bit mantissa, so every `u16` and every `i16` is exactly
/// representable and nothing is lost on the way through.
#[test]
fn every_integer_sample_round_trips_exactly() {
    let bytes: Vec<u8> = (0..=u8::MAX).collect();
    let as_f32: Vec<f32> = bytes.iter().map(|&v| v as f32).collect();
    match Store::U8.plane(as_f32) {
        PlaneData::U8(back) => assert_eq!(back, bytes),
        other => panic!("{:?}", other.pixel_type()),
    }

    let words: Vec<u16> = (0..=u16::MAX).collect();
    let as_f32: Vec<f32> = words.iter().map(|&v| v as f32).collect();
    match Store::U16.plane(as_f32) {
        PlaneData::U16(back) => assert_eq!(back, words),
        other => panic!("{:?}", other.pixel_type()),
    }

    // Signed samples travel as the same sixteen bits in the `U16` lane.
    let signed: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    let as_f32: Vec<f32> = signed.iter().map(|&v| v as f32).collect();
    match Store::I16.plane(as_f32) {
        PlaneData::U16(back) => {
            let round: Vec<i16> = back.iter().map(|&v| v as i16).collect();
            assert_eq!(round, signed);
        }
        other => panic!("{:?}", other.pixel_type()),
    }
}

/// A sample that could not have come from the source is clamped, not wrapped.
#[test]
fn an_out_of_range_sample_is_clamped() {
    match Store::U8.plane(vec![-5.0, 300.0, 7.0]) {
        PlaneData::U8(v) => assert_eq!(v, vec![0, 255, 7]),
        other => panic!("{:?}", other.pixel_type()),
    }
    match Store::I16.plane(vec![-40000.0, 40000.0]) {
        PlaneData::U16(v) => {
            let s: Vec<i16> = v.iter().map(|&x| x as i16).collect();
            assert_eq!(s, vec![i16::MIN, i16::MAX]);
        }
        other => panic!("{:?}", other.pixel_type()),
    }
}

/// `NaN` lands at the bottom of the range rather than wherever a cast puts it.
///
/// `as u16` on a `NaN` is 0, which for a signed stack is the *middle* of the
/// range — mid-grey, which reads as data rather than as nothing.
#[test]
fn a_nan_sample_lands_at_the_bottom_of_the_range() {
    match Store::I16.plane(vec![f32::NAN]) {
        PlaneData::U16(v) => assert_eq!(v[0] as i16, i16::MIN),
        other => panic!("{:?}", other.pixel_type()),
    }
    match Store::U8.plane(vec![f32::NAN]) {
        PlaneData::U8(v) => assert_eq!(v[0], 0),
        other => panic!("{:?}", other.pixel_type()),
    }
}

/// The range each width inverts about, and the one that has none.
#[test]
fn only_float_has_no_range_of_its_own() {
    assert_eq!(Store::U8.range(), Some((0.0, 255.0)));
    assert_eq!(Store::U16.range(), Some((0.0, 65535.0)));
    assert_eq!(Store::I16.range(), Some((-32768.0, 32767.0)));
    assert_eq!(
        Store::F32.range(),
        None,
        "a float stack's range has to be measured, not assumed"
    );
}

// ----------------------------------------------------------- where it goes

/// The checkbox decides, and its default is the answer that cannot lose
/// anything.
#[test]
fn the_default_opens_a_window_rather_than_replacing_one() {
    let image = ImageResult {
        width: 1,
        height: 1,
        channels: 1,
        slices: 1,
        frames: 1,
        pixel_type: PixelType::U8,
        planes: vec![PlaneData::U8(vec![1])],
        channel_colors: Vec::new(),
        metadata: None,
        name: "x".into(),
    };

    // An untouched dialog.
    let decls = vec![in_new_window()];
    let params = Params::defaults(&decls);
    assert!(
        matches!(deliver(image.clone(), &params), Outcome::NewDocument(_)),
        "the default must not replace the open image"
    );

    // And an empty one: a tool whose dialog never reached the host must still
    // not destroy anything.
    assert!(matches!(
        deliver(image.clone(), &Params::new()),
        Outcome::NewDocument(_)
    ));

    let mut off = Params::new();
    off.set(NEW_WINDOW, ParamValue::Bool(false));
    assert!(matches!(deliver(image, &off), Outcome::ReplaceDocument(_)));
}

/// The checkbox's key is the one `deliver` reads.
///
/// Spelled in two places — the declaration and the read — and a drift between
/// them would silently always take the default, which is a checkbox that does
/// nothing.
#[test]
fn the_checkbox_key_matches_what_is_read() {
    assert_eq!(in_new_window().key, NEW_WINDOW);
}

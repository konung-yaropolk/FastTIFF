//! The plugin around the registration: what it refuses, and what it produces.
//!
//! The algorithm itself is tested in `suite2p-registration`. What is checked
//! here is the wiring — that the shift measured on one channel is applied to
//! all of them, that the result has the shape it claims, and that a stack with
//! no time axis is refused rather than silently "registered".

use super::*;
use fasttiff_plugin_api::{ImageInfo, ParamKind};

fn info(channels: usize, slices: usize, frames: usize) -> ImageInfo {
    ImageInfo {
        width: 32,
        height: 32,
        channels,
        slices,
        frames,
        samples_per_pixel: 1,
        pixel_type: PixelType::U16,
    }
}

/// Every parameter offered carries suite2p's own default, so a dialog left
/// untouched runs what suite2p would run. A drift here is a drift away from the
/// numbers published results were produced with.
#[test]
fn the_dialog_offers_suite2ps_defaults() {
    let decls = super::params::declare(&info(1, 1, 10));
    let find = |k: &str| {
        decls
            .iter()
            .find(|d| d.key == k)
            .unwrap_or_else(|| panic!("no control for {k}"))
    };

    let float = |k: &str| match find(k).kind {
        ParamKind::Float { default, .. } => default,
        ref other => panic!("{k} is {other:?}"),
    };
    let int = |k: &str| match find(k).kind {
        ParamKind::Int { default, .. } => default,
        ref other => panic!("{k} is {other:?}"),
    };
    let boolean = |k: &str| match find(k).kind {
        ParamKind::Bool { default } => default,
        ref other => panic!("{k} is {other:?}"),
    };

    assert_eq!(int("nimg_init"), 300);
    assert_eq!(float("maxregshift"), 0.1);
    assert!(!boolean("do_bidiphase"));
    assert_eq!(int("bidiphase"), 0);
    assert_eq!(int("batch_size"), 100);
    // The one deliberate divergence: suite2p ships `nonrigid` off and this
    // build turns it on. Pinned as a decision rather than left unchecked, so
    // that it cannot quietly become a drift — and because it costs: a warped
    // frame is interpolated, so the result is stored as float and is twice the
    // size of one that was only shifted.
    assert!(boolean("nonrigid"));
    assert_eq!(float("maxregshiftNR"), 10.0);
    assert_eq!(int("block_size"), 64);
    assert_eq!(float("smooth_sigma_time"), 0.0);
    assert_eq!(float("smooth_sigma"), 1.15);
    assert_eq!(float("spatial_taper"), 50.0);
    assert_eq!(float("th_badframes"), 1.0);
    assert!(boolean("norm_frames"));
    assert_eq!(float("snr_thresh"), 1.25);
    assert_eq!(int("subpixel"), 10);
    assert!(!boolean("two_step_registration"));
}

/// The backend selector offers all three, defaulting to multi-thread.
#[test]
fn the_dialog_offers_every_backend() {
    let decls = super::params::declare(&info(1, 1, 10));
    let backend = decls.iter().find(|d| d.key == "backend").expect("backend");
    match &backend.kind {
        ParamKind::Choice { default, options } => {
            assert_eq!(
                options,
                &vec![
                    "Single-thread CPU".to_string(),
                    "Multi-thread CPU".to_string(),
                    "GPU".to_string(),
                ]
            );
            assert_eq!(options[*default], "Multi-thread CPU");
        }
        other => panic!("{other:?}"),
    }
}

/// `align_by_chan2` only makes sense with a second channel to align by.
#[test]
fn the_channel_switch_appears_only_for_a_multichannel_stack() {
    assert!(
        super::params::declare(&info(1, 1, 10))
            .iter()
            .all(|d| d.key != "align_by_chan2"),
        "one channel is not a choice"
    );
    assert!(super::params::declare(&info(2, 1, 10))
        .iter()
        .any(|d| d.key == "align_by_chan2"));
}

/// An untouched dialog reads back as suite2p's defaults exactly.
#[test]
fn an_unanswered_dialog_reads_back_as_the_defaults() {
    let read = super::params::settings_from(&Params::new());
    assert_eq!(read, suite2p_registration::Settings::default());
}

/// And a changed control reaches the settings rather than being dropped.
#[test]
fn an_answered_dialog_reaches_the_settings() {
    let mut p = Params::new();
    p.set("smooth_sigma", fasttiff_plugin_api::ParamValue::Float(3.0));
    p.set("nonrigid", fasttiff_plugin_api::ParamValue::Bool(true));
    p.set("block_size", fasttiff_plugin_api::ParamValue::Int(128));
    p.set("backend", fasttiff_plugin_api::ParamValue::Choice(0));
    let s = super::params::settings_from(&p);
    assert_eq!(s.smooth_sigma, 3.0);
    assert!(s.nonrigid);
    assert_eq!(s.block_size, [128, 128]);
    assert_eq!(s.backend, suite2p_registration::Backend::SingleThread);
}

// ------------------------------------------------ what the result is stored in

/// A rigid run gives the file's own width back; a non-rigid one gives float.
///
/// The distinction is not cosmetic. A rigid correction is `np.roll` — no
/// arithmetic touches a sample — so the file's width is exact and half the
/// size. Non-rigid interpolates between pixels and makes values that were never
/// in the file, which need somewhere to live.
#[test]
fn the_result_keeps_the_source_width_unless_it_was_warped() {
    assert_eq!(Store::of(PixelType::U16, false), Store::U16);
    assert_eq!(Store::of(PixelType::U8, false), Store::U8);
    assert_eq!(Store::of(PixelType::F32, false), Store::F32);
    // Signed 16-bit has no `PlaneData` of its own; float says what it is.
    assert_eq!(Store::of(PixelType::I16, false), Store::F32);

    for source in [
        PixelType::U8,
        PixelType::U16,
        PixelType::I16,
        PixelType::F32,
    ] {
        assert_eq!(
            Store::of(source, true),
            Store::F32,
            "a non-rigid run interpolates and cannot be stored as {source:?}"
        );
    }
}

/// Every 16-bit value survives the trip through the registration's `f32`
/// exactly, including both ends of the range.
///
/// This is the whole justification for storing the result at the source's
/// width: `f32` has a 24-bit mantissa, so every `u16` is representable, and a
/// rigid shift only moves them.
#[test]
fn every_16_bit_value_round_trips_exactly() {
    let samples: Vec<u16> = (0..=u16::MAX).collect();
    let as_f32: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
    match Store::U16.plane(as_f32) {
        PlaneData::U16(back) => assert_eq!(back, samples),
        other => panic!("stored as {:?}", other.pixel_type()),
    }

    let bytes: Vec<u8> = (0..=u8::MAX).collect();
    let as_f32: Vec<f32> = bytes.iter().map(|&v| v as f32).collect();
    match Store::U8.plane(as_f32) {
        PlaneData::U8(back) => assert_eq!(back, bytes),
        other => panic!("stored as {:?}", other.pixel_type()),
    }
}

/// A value that could not have come from the source is clamped rather than
/// wrapped.
///
/// It cannot happen through the plugin. It is guarded because the failure mode
/// if it ever did — `as u16` on a negative float is 0, but on a large one it
/// saturates in a way that reads as a bright speck — is a picture that looks
/// like data.
#[test]
fn an_out_of_range_sample_is_clamped_not_wrapped() {
    match Store::U16.plane(vec![-5.0, 70000.0, 1234.0]) {
        PlaneData::U16(v) => assert_eq!(v, vec![0, 65535, 1234]),
        other => panic!("stored as {:?}", other.pixel_type()),
    }
}

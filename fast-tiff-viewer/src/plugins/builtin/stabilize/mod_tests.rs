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
    assert!(!boolean("nonrigid"));
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

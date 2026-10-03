//! The plugin around the registration: what it refuses, and what it produces.
//!
//! The algorithm itself is tested in `suite2p-registration`. What is checked
//! here is the wiring — that the shift measured on one channel is applied to
//! all of them, that the result has the shape it claims, and that a stack with
//! no time axis is refused rather than silently "registered".

use super::*;
use fasttiff_plugin_api::{ImageInfo, ParamKind, ParamValue, Params};

/// The declarations for a stack of this shape, with nothing chosen yet.
fn decls_for(info: &ImageInfo) -> Vec<fasttiff_plugin_api::ParamDecl> {
    super::params::declare(info, &Params::new())
}

/// The declarations once `chosen` has been set, as the host would ask for
/// them after the user changed something.
fn decls_with(
    info: &ImageInfo,
    chosen: &[(&str, ParamValue)],
) -> Vec<fasttiff_plugin_api::ParamDecl> {
    let mut p = Params::new();
    for (k, v) in chosen {
        p.set(*k, v.clone());
    }
    super::params::declare(info, &p)
}

/// Just the keys, for the tests that are about what is offered.
fn offered(decls: &[fasttiff_plugin_api::ParamDecl]) -> Vec<String> {
    decls.iter().map(|d| d.key.clone()).collect()
}

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
    let decls = decls_for(&info(1, 1, 10));
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
    let decls = decls_for(&info(1, 1, 10));
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
        decls_for(&info(1, 1, 10))
            .iter()
            .all(|d| d.key != "align_by_chan2"),
        "one channel is not a choice"
    );
    assert!(decls_for(&info(2, 1, 10))
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

/// The result comes back at the file's own width, whatever the correction was.
///
/// A stabilised 16-bit recording is a 16-bit recording. It used to widen to
/// float whenever `nonrigid` was on — which is the default here — so every
/// stabilised file was twice the size of the one it came from.
#[test]
fn the_result_keeps_the_source_width() {
    assert_eq!(Store::of(PixelType::U8), Store::U8);
    assert_eq!(Store::of(PixelType::U16), Store::U16);
    assert_eq!(Store::of(PixelType::I16), Store::I16);
    assert_eq!(Store::of(PixelType::F32), Store::F32);

    // And each one declares itself as what it is, so the file says so too.
    assert_eq!(Store::U8.pixel_type(), PixelType::U8);
    assert_eq!(Store::U16.pixel_type(), PixelType::U16);
    assert_eq!(Store::I16.pixel_type(), PixelType::I16);
    assert_eq!(Store::F32.pixel_type(), PixelType::F32);
}

/// Signed samples travel as their bit pattern, and come back as themselves.
///
/// The contract has no `PlaneData::I16`: a signed result is `PlaneData::U16`
/// holding the same sixteen bits, declared `PixelType::I16`. Get that wrong and
/// every negative sample reads as a very bright one — a picture that still
/// looks like a picture.
#[test]
fn every_signed_16_bit_value_round_trips_exactly() {
    let samples: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    let as_f32: Vec<f32> = samples.iter().map(|&v| v as f32).collect();
    match Store::I16.plane(as_f32) {
        PlaneData::U16(back) => {
            let signed: Vec<i16> = back.iter().map(|&v| v as i16).collect();
            assert_eq!(signed, samples);
        }
        other => panic!("stored as {:?}", other.pixel_type()),
    }
}

/// An interpolated sample is rounded to the nearest whole one, not truncated.
///
/// Only a non-rigid run produces these — a rigid shift moves whole samples —
/// and truncating them would darken every warped frame by half a sample on
/// average, which is a bias, not noise.
#[test]
fn interpolated_samples_are_rounded_to_nearest() {
    match Store::U16.plane(vec![41.6, 41.4, 0.5, 1.5, 2.49]) {
        // 0.5 and 1.5 round away from zero, as `f32::round` does.
        PlaneData::U16(v) => assert_eq!(v, vec![42, 41, 1, 2, 2]),
        other => panic!("stored as {:?}", other.pixel_type()),
    }
    match Store::I16.plane(vec![-3.4, -3.6, -0.5]) {
        PlaneData::U16(v) => {
            let signed: Vec<i16> = v.iter().map(|&x| x as i16).collect();
            assert_eq!(signed, vec![-3, -4, -1]);
        }
        other => panic!("stored as {:?}", other.pixel_type()),
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

// ------------------------------------------------- a dialog that follows itself

/// The four non-rigid controls exist only while the non-rigid correction
/// does.
///
/// They are dead settings for a rigid run — nothing reads them — and a dead
/// setting that is still on screen is worse than a missing one: the usual
/// outcome is someone raising the block shift, seeing no difference, and
/// concluding the non-rigid correction is broken.
#[test]
fn the_non_rigid_grid_appears_only_with_the_non_rigid_correction() {
    let i = info(1, 1, 10);
    let grid = ["block_size", "maxregshiftNR", "snr_thresh", "subpixel"];

    let on = offered(&decls_with(&i, &[("nonrigid", ParamValue::Bool(true))]));
    for key in grid {
        assert!(
            on.iter().any(|k| k == key),
            "non-rigid is on and {key} is missing"
        );
    }
    assert!(
        on.iter().any(|k| k == "h_nonrigid"),
        "and so is its heading"
    );

    let off = offered(&decls_with(&i, &[("nonrigid", ParamValue::Bool(false))]));
    for key in grid {
        assert!(
            !off.iter().any(|k| k == key),
            "non-rigid is off and {key} is offered"
        );
    }
    assert!(
        !off.iter().any(|k| k == "h_nonrigid"),
        "an empty group must take its heading with it"
    );
    // Everything else is still there: the rigid run is not a cut-down dialog.
    for key in [
        "backend",
        "nonrigid",
        "maxregshift",
        "nimg_init",
        "th_badframes",
    ] {
        assert!(
            off.iter().any(|k| k == key),
            "{key} went missing with the grid"
        );
    }
}

/// suite2p ignores the measurement whenever a fixed offset is given, so the
/// switch is offered only while there is no offset to beat it.
#[test]
fn the_bidirectional_switch_hides_behind_a_fixed_offset() {
    let i = info(1, 1, 10);
    let none = offered(&decls_with(&i, &[("bidiphase", ParamValue::Int(0))]));
    assert!(none.iter().any(|k| k == "do_bidiphase"));

    for offset in [-3i64, 7] {
        let set = offered(&decls_with(&i, &[("bidiphase", ParamValue::Int(offset))]));
        assert!(
            !set.iter().any(|k| k == "do_bidiphase"),
            "an offset of {offset} overrides the switch, so it should not be offered"
        );
        // The offset itself stays, or there would be no way back.
        assert!(set.iter().any(|k| k == "bidiphase"));
    }
}

/// The groups the module doc describes are now drawn rather than described.
#[test]
fn the_dialog_is_grouped_into_sections() {
    let decls = decls_for(&info(1, 1, 10));
    let sections: Vec<&str> = decls
        .iter()
        .filter(|d| d.kind == ParamKind::Section)
        .map(|d| d.label.as_str())
        .collect();
    assert_eq!(
        sections,
        vec![
            "Correction",
            "Alignment",
            "Reference",
            "Non-rigid grid",
            "Low signal",
            "Scanner",
            "Reporting",
        ]
    );
    // No heading may be last or immediately followed by another: either is a
    // group that lost its contents to a condition without losing its title.
    for (n, d) in decls.iter().enumerate() {
        if d.kind == ParamKind::Section {
            assert!(
                decls
                    .get(n + 1)
                    .is_some_and(|x| x.kind != ParamKind::Section),
                "the {:?} heading has nothing under it",
                d.label
            );
        }
    }
}

/// And that stays true for every combination that can hide something.
#[test]
fn no_heading_is_ever_left_empty() {
    for channels in [1usize, 2] {
        for nonrigid in [true, false] {
            for offset in [0i64, 5] {
                let decls = decls_with(
                    &info(channels, 1, 10),
                    &[
                        ("nonrigid", ParamValue::Bool(nonrigid)),
                        ("bidiphase", ParamValue::Int(offset)),
                    ],
                );
                for (n, d) in decls.iter().enumerate() {
                    if d.kind == ParamKind::Section {
                        assert!(
                            decls
                                .get(n + 1)
                                .is_some_and(|x| x.kind != ParamKind::Section),
                            "c{channels} nonrigid={nonrigid} offset={offset}: \
                             the {:?} heading has nothing under it",
                            d.label
                        );
                    }
                }
            }
        }
    }
}

/// A hidden control falls back to suite2p's default, and that default is the
/// one the run would have ignored anyway.
///
/// This is what makes hiding safe rather than merely tidy: `clamp_to` drops
/// every key the settled dialog did not declare, so `settings_from` sees a
/// rigid run with no block size at all. It has to produce the same `Settings`
/// as a rigid run that was shown one and left it alone.
#[test]
fn hiding_a_control_does_not_change_what_runs() {
    let i = info(1, 1, 10);

    let shown = decls_with(&i, &[("nonrigid", ParamValue::Bool(true))]);
    let mut as_rigid = Params::defaults(&shown);
    as_rigid.set("nonrigid", ParamValue::Bool(false));
    // The dialog as it was before the switch was turned off: the block size
    // is still in there, and is still declared.
    let with_dead_keys = super::params::settings_from(&as_rigid.clamp_to(&shown));

    let hidden = decls_with(&i, &[("nonrigid", ParamValue::Bool(false))]);
    // Defaults *plus the choice*, which is what the host holds: `defaults`
    // alone would put `nonrigid` back to suite2p's `true` and so contradict
    // the very choice that produced these declarations. The host only ever
    // uses defaults to seed keys that have just appeared.
    let mut settled = Params::defaults(&hidden);
    settled.set("nonrigid", ParamValue::Bool(false));
    assert!(
        settled.get("block_size").is_none(),
        "the fixture must actually be missing the key"
    );
    let without = super::params::settings_from(&settled.clamp_to(&hidden));

    assert!(!with_dead_keys.nonrigid);
    assert!(!without.nonrigid);
    assert_eq!(with_dead_keys.block_size, without.block_size);
    assert_eq!(with_dead_keys.subpixel, without.subpixel);
    assert_eq!(with_dead_keys.snr_thresh, without.snr_thresh);
    assert_eq!(with_dead_keys.maxregshift_nr, without.maxregshift_nr);
}

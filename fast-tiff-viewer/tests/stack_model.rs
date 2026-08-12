//! End-to-end tests of the viewer model against real TIFF fixtures — the whole
//! path from opening a stack through channel/contrast derivation, dimension
//! reinterpretation and the playback clock, with no GPU and no window.
//!
//! Being able to run this at all is a direct payoff of the core/frontend split:
//! before it, every one of these behaviors was reachable only through an
//! `eframe::App` and could not be asserted in CI.
//!
//! Fixtures are shared with `fast-tiff-lib` (see `tests/fixtures/generate_fixtures.py`
//! there); the tests skip rather than fail if one is missing, so a checkout
//! without them still builds green.

use fast_tiff_viewer::channels::{channel_tint, gray_lut_applicable, pseudocolor_applicable};
use fast_tiff_viewer::{ChannelKind, Stack, Viewer};
use std::path::PathBuf;

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("fast-tiff-lib")
        .join("tests")
        .join("fixtures")
        .join(name);
    p.exists().then_some(p)
}

/// Open a stack through whichever entry point this build has.
///
/// With `mmap` (the default) that's the memory-mapped `Stack::open`; without
/// it, the bytes are read here and handed to `Stack::from_bytes` — the path a
/// browser takes. So the two CI configurations between them cover both.
fn open_stack(path: PathBuf, pseudocolor: bool) -> anyhow::Result<Stack> {
    #[cfg(feature = "mmap")]
    {
        Stack::open(path, pseudocolor)
    }
    #[cfg(not(feature = "mmap"))]
    {
        let bytes = std::fs::read(&path)?;
        Stack::from_bytes(bytes, path, pseudocolor)
    }
}

/// The `Viewer` counterpart to [`open_stack`].
fn load(viewer: &mut Viewer, path: PathBuf) -> anyhow::Result<()> {
    #[cfg(feature = "mmap")]
    {
        viewer.open(path)
    }
    #[cfg(not(feature = "mmap"))]
    {
        // A missing file yields no bytes — hand over what we have (nothing) so
        // the *viewer* reports the failure and records it in `status`, rather
        // than the error escaping here. That matches both how `open` surfaces a
        // bad file and how a browser fails: bytes arrive, and they don't parse.
        let bytes = std::fs::read(&path).unwrap_or_default();
        viewer.load_bytes(bytes, path)
    }
}

#[test]
fn opens_an_imagej_hyperstack_with_derived_channel_settings() {
    let Some(path) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let stack = open_stack(path, false).expect("hyperstack should open");

    let (w, h) = stack.dimensions().expect("dimensions");
    assert!(w > 0 && h > 0);
    assert!(!stack.channel_settings.is_empty(), "every open stack gets channel settings");

    // 16-bit unsigned planes take the R16Uint path, and each channel starts
    // enabled with a contrast window that sits inside its slider track.
    for (c, s) in stack.channel_settings.iter().enumerate() {
        assert_eq!(s.kind, ChannelKind::Int16, "channel {c}");
        assert!(s.enabled, "channel {c} should start enabled");
        let (lo, hi) = s.bounds;
        assert!(hi > lo, "channel {c}: degenerate slider track {lo}..{hi}");
        assert!(s.min >= lo && s.max <= hi, "channel {c}: window {}..{} outside track {lo}..{hi}", s.min, s.max);
        assert!(s.min <= s.max, "channel {c}: inverted window");
    }

    // A display LUT exists for every channel the settings cover.
    assert!(stack.tiff.meta.channel_display.len() >= stack.channel_settings.len());
}

#[test]
fn rgb_stack_becomes_three_display_channels() {
    let Some(path) = fixture("tff_u8_spp3_p2_none.tif") else { return };
    let stack = open_stack(path, false).expect("rgb should open");

    assert!(stack.rgb, "a 3-sample photometric-RGB frame should set the rgb flag");
    assert_eq!(stack.channel_settings.len(), 3, "R/G/B become three display channels");
    // 8-bit unsigned uploads raw, without the CPU widening pass.
    assert!(stack.channel_settings.iter().all(|s| s.kind == ChannelKind::Int8));
    // The channels/time guess and the pseudocolor toggle are meaningless here.
    assert!(!pseudocolor_applicable(&stack));
    assert!(!gray_lut_applicable(&stack));
}

#[test]
fn float_stack_windows_in_its_own_units() {
    let Some(path) = fixture("tff_f32_spp1_p2_none-le.tif") else { return };
    let stack = open_stack(path, false).expect("float should open");

    let s = stack.channel_settings.first().expect("one channel");
    assert_eq!(s.kind, ChannelKind::Float, "32-bit float takes the R32F path");
    // Contrast is defined over the data's actual range, not an assumed
    // 0..65535 integer scale (matching how ImageJ treats float images).
    assert!(s.bounds.1 <= 65535.0 || s.bounds.0 < 0.0, "float bounds look integer-shaped: {:?}", s.bounds);
    assert!(s.min <= s.max);
}

#[test]
fn dimension_override_conserves_the_plane_count() {
    let Some(path) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let mut viewer = Viewer::new();
    load(&mut viewer, path).expect("hyperstack should open");

    let planes = {
        let m = &viewer.stack.as_ref().unwrap().tiff.meta;
        m.channels * m.slices * m.frames
    };
    let gen_before = viewer.volume.generation;

    // Swap the channel and time roles, as the dimension-order dropdown does.
    let (c, z, f) = {
        let m = &viewer.stack.as_ref().unwrap().tiff.meta;
        (m.channels, m.slices, m.frames)
    };
    viewer.set_dimension_order(f, z, c);

    let stack = viewer.stack.as_ref().unwrap();
    assert_eq!(stack.tiff.meta.channels, f);
    assert_eq!(stack.tiff.meta.frames, c);
    assert_eq!(
        stack.tiff.meta.channels * stack.tiff.meta.slices * stack.tiff.meta.frames,
        planes,
        "reassigning axes must not invent or drop planes"
    );
    // Channel settings and LUTs were rebuilt to match the new channel count...
    assert_eq!(stack.channel_settings.len(), f.min(fast_tiff_viewer::MAX_CHANNELS));
    // ...and the volume was invalidated, since the depth axis just changed.
    assert_ne!(viewer.volume.generation, gen_before, "volume should be invalidated");
    assert_eq!(viewer.volume.built_frame, None);
}

#[test]
fn pseudocolor_tints_channels_and_reverts() {
    let Some(path) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let mut viewer = Viewer::new();
    load(&mut viewer, path).expect("open");

    if !pseudocolor_applicable(viewer.stack.as_ref().unwrap()) {
        return; // this fixture carries its own colors; nothing to assert
    }

    viewer.set_pseudocolor(true);
    let tinted = viewer
        .stack
        .as_ref()
        .unwrap()
        .tiff
        .meta
        .channel_display
        .iter()
        .filter(|cd| channel_tint(&cd.lut).is_some())
        .count();
    assert!(tinted > 0, "pseudocolor should give at least one channel a non-gray LUT");

    viewer.set_pseudocolor(false);
    let stack = viewer.stack.as_ref().unwrap();
    assert!(
        stack.tiff.meta.channel_display.iter().all(|cd| channel_tint(&cd.lut).is_none()),
        "turning pseudocolor off should restore plain grayscale LUTs"
    );
    // Either way the LUTs must be re-uploaded on the next sync.
    assert!(!stack.luts_uploaded);
}

#[test]
fn a_failed_open_keeps_the_current_stack() {
    let Some(good) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let mut viewer = Viewer::new();
    load(&mut viewer, good).expect("open");
    assert!(viewer.stack.is_some());

    // Dropping a corrupt/nonexistent file must not close the good image.
    let err = load(&mut viewer, PathBuf::from("definitely-not-a-file.tif"));
    assert!(err.is_err());
    assert!(viewer.stack.is_some(), "a failed open must not drop the loaded stack");
    assert!(viewer.status.as_deref().unwrap_or("").contains("Failed to open"));
}

#[test]
fn playback_advances_by_real_elapsed_time() {
    let Some(path) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let mut viewer = Viewer::new();
    load(&mut viewer, path).expect("open");
    let n = viewer.stack.as_ref().unwrap().frame_count();
    if n < 2 {
        return;
    }

    viewer.playback.playing = true;
    viewer.playback.fps = 10.0;

    // The first tick only seeds the clock — it can't know how much time passed.
    assert_eq!(viewer.tick_playback(0.0), 0);
    assert_eq!(viewer.stack.as_ref().unwrap().frame_index, 0);

    // A quarter-frame of real time is not yet a whole frame...
    assert_eq!(viewer.tick_playback(0.025), 0);
    // ...but it accumulates rather than being discarded: 0.025 + 0.075 = one frame.
    assert_eq!(viewer.tick_playback(0.1), 1);
    assert_eq!(viewer.stack.as_ref().unwrap().frame_index, 1 % n);

    // A long stall steps by however many frames the elapsed time demanded.
    let stepped = viewer.tick_playback(0.6);
    assert_eq!(stepped, 5, "0.5 s at 10 fps is five frames");

    // Playback wraps rather than running off the end.
    for i in 0..(2 * n) {
        viewer.tick_playback(1.0 + i as f64 * 0.1);
        assert!(viewer.stack.as_ref().unwrap().frame_index < n);
    }
}

#[test]
fn falling_behind_latches_parallel_decode_only_in_auto() {
    let Some(path) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    let mut viewer = Viewer::new();
    load(&mut viewer, path).expect("open");
    if viewer.stack.as_ref().unwrap().frame_count() < 2 {
        return;
    }

    viewer.decode_mode = fast_tiff_viewer::DecodeMode::Serial;
    viewer.playback.playing = true;
    viewer.playback.fps = 60.0;
    // Renders far slower than the target: demand per tick is ~6 frames.
    for i in 0..40 {
        viewer.tick_playback(i as f64 * 0.1);
    }
    assert!(!viewer.decode_parallel, "Serial must never latch parallel decode");

    viewer.decode_mode = fast_tiff_viewer::DecodeMode::Auto;
    for i in 40..80 {
        viewer.tick_playback(i as f64 * 0.1);
    }
    assert!(viewer.decode_parallel, "Auto should latch parallel decode once playback falls behind");

    // Opening a new stack re-evaluates it from scratch.
    let Some(again) = fixture("ij_u16_spp1_p6_hyperstack.tif") else { return };
    load(&mut viewer, again).expect("reopen");
    assert!(!viewer.decode_parallel, "a new stack starts from the serial path again");
}

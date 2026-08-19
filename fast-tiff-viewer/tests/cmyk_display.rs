//! CMYK stacks, end to end through the viewer.
//!
//! The library tests cover the conversion itself. What is left to go wrong is
//! the wiring: whether a Separated file is recognised at load, whether it is
//! configured as three display channels rather than four ink ones, and whether
//! both decode paths — the batched one used when several channels are visible
//! and the per-plane one used when only a single channel is ticked — route to
//! the converting readers. A miss in either path shows up as a channel of raw
//! ink coverage rendered as if it were light, which is a plausible-looking
//! image rather than an obvious failure, so it is worth pinning.

use fast_tiff_viewer::prefetch::Decoded;
use fast_tiff_viewer::{ChannelKind, Stack};
use std::path::PathBuf;

const W: usize = 23;
const H: usize = 11;

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("fast-tiff-lib")
        .join("tests")
        .join("fixtures")
        .join(name);
    p.exists().then_some(p)
}

/// Open through whichever entry point this build has — memory-mapped by
/// default, byte-slice when `mmap` is off (the browser path). Both
/// configurations run in CI, so between them each entry point is covered.
fn open_stack(path: PathBuf) -> anyhow::Result<Stack> {
    #[cfg(feature = "mmap")]
    {
        Stack::open(path, false)
    }
    #[cfg(not(feature = "mmap"))]
    {
        let bytes = std::fs::read(&path)?;
        Stack::from_bytes(bytes, path, false)
    }
}

/// Decode `frame` with exactly the channels in `enabled` switched on.
fn decode(stack: &Stack, frame: usize, enabled: &[bool]) -> Vec<Decoded> {
    let kinds: Vec<_> = stack.display.settings.iter().map(|s| s.kind).collect();
    let jobs = fast_tiff_viewer::sync::build_jobs(stack, frame, enabled, &kinds);
    fast_tiff_viewer::prefetch::decode_jobs(&stack.tiff.data, &stack.tiff.frames, stack.tiff.byte_order, &jobs)
        .unwrap_or_else(|e| panic!("frame {frame} failed to decode: {e:#}"))
}

fn as_u8(d: &Decoded) -> &Vec<u8> {
    match d {
        Decoded::U8(v) => v,
        _ => panic!("expected an 8-bit channel"),
    }
}

/// What the four ink plates say each component should be, read straight from
/// the library's raw per-sample readers — the ones this feature was required
/// not to disturb.
fn reference_rgb(stack: &Stack, frame: usize) -> Vec<[u8; 3]> {
    let f = &stack.tiff.frames[frame];
    let raw = fast_tiff_lib::read_planes_u8(&stack.tiff.data, f, stack.tiff.byte_order).unwrap();
    assert_eq!(raw.len(), 4, "the raw reader must still yield one plane per ink");
    (0..W * H)
        .map(|i| {
            let k = raw[3][i] as f32 / 255.0;
            let c = |ink: u8| (((1.0 - ink as f32 / 255.0) * (1.0 - k)) * 255.0).round() as u8;
            [c(raw[0][i]), c(raw[1][i]), c(raw[2][i])]
        })
        .collect()
}

#[test]
fn cmyk_stack_becomes_three_converted_display_channels() {
    let Some(path) = fixture("tff_u8_spp4_p1_cmyk.tif") else { return };
    let stack = open_stack(path).expect("cmyk should open");

    assert!(stack.display.cmyk, "a Separated frame should set the cmyk flag");
    assert!(
        stack.display.rgb,
        "cmyk also sets rgb: the display channels are sample planes of one IFD either way, \
         which is what the decode addressing keys on"
    );
    assert_eq!(
        stack.display.settings.len(),
        3,
        "four inks become three components, not four channels"
    );
    assert!(stack.display.settings.iter().all(|s| s.kind == ChannelKind::Int8));
    assert!(stack.display.settings.iter().all(|s| s.enabled), "all three start visible");
}

/// 16-bit inks must stay 16-bit through the conversion rather than being
/// narrowed to fit the 8-bit upload path.
#[test]
fn sixteen_bit_cmyk_keeps_its_width() {
    let Some(path) = fixture("tff_u16_spp4_p1_cmyk.tif") else { return };
    let stack = open_stack(path).expect("16-bit cmyk should open");
    assert!(stack.display.cmyk);
    assert_eq!(stack.display.settings.len(), 3);
    assert!(stack.display.settings.iter().all(|s| s.kind == ChannelKind::Int16));

    let decoded = decode(&stack, 0, &[true, true, true]);
    assert_eq!(decoded.len(), 3);
    assert!(matches!(decoded[0], Decoded::U16(_)), "16-bit source must decode to 16-bit");
}

/// The batched path: several channels visible, so one decompression pass feeds
/// the whole conversion. This is the common case.
#[test]
fn batched_decode_yields_converted_components() {
    let Some(path) = fixture("tff_u8_spp4_p1_cmyk.tif") else { return };
    let stack = open_stack(path).expect("cmyk should open");
    let want = reference_rgb(&stack, 0);

    let decoded = decode(&stack, 0, &[true, true, true]);
    assert_eq!(decoded.len(), 3);
    for c in 0..3 {
        let got = as_u8(&decoded[c]);
        assert_eq!(got.len(), W * H, "component {c} pixel count");
        for i in 0..W * H {
            assert!(
                (got[i] as i16 - want[i][c] as i16).abs() <= 1,
                "component {c} px {i}: got {}, want {}",
                got[i],
                want[i][c]
            );
        }
    }
}

/// The per-plane path. Unticking channels until one is left drops the job count
/// to one, which falls out of the batched branch entirely — so this is a second
///, independently-reachable route to the same pixels, and it has to agree.
#[test]
fn single_channel_decode_matches_the_batched_result() {
    let Some(path) = fixture("tff_u8_spp4_p1_cmyk.tif") else { return };
    let stack = open_stack(path).expect("cmyk should open");
    let batched = decode(&stack, 0, &[true, true, true]);

    for c in 0..3 {
        let mut enabled = [false; 3];
        enabled[c] = true;
        let alone = decode(&stack, 0, &enabled);
        assert_eq!(alone.len(), 1, "exactly one job when one channel is visible");
        assert_eq!(
            as_u8(&alone[0]),
            as_u8(&batched[c]),
            "component {c} differs between the single-channel and batched decode paths"
        );
    }
}

/// Planar CMYK, multi-page: the layout is normalised by the plane gather before
/// the conversion sees it, so a planar file must decode to the same components
/// a chunky one would — and every page must decode, not just the first.
#[test]
fn planar_cmyk_decodes_every_page() {
    let Some(path) = fixture("tff_u8_spp4_p2_cmyk-planar.tif") else { return };
    let stack = open_stack(path).expect("planar cmyk should open");
    assert!(stack.display.cmyk);
    assert!(stack.tiff.frames[0].is_planar());

    let mut first_pixels = Vec::new();
    for frame in 0..stack.tiff.frames.len() {
        let want = reference_rgb(&stack, frame);
        let decoded = decode(&stack, frame, &[true, true, true]);
        assert_eq!(decoded.len(), 3, "page {frame}");
        for c in 0..3 {
            let got = as_u8(&decoded[c]);
            for i in 0..W * H {
                assert!(
                    (got[i] as i16 - want[i][c] as i16).abs() <= 1,
                    "page {frame} component {c} px {i}: got {}, want {}",
                    got[i],
                    want[i][c]
                );
            }
        }
        first_pixels.push(as_u8(&decoded[0])[0]);
    }
    assert_eq!(first_pixels.len(), 2, "both pages should be addressable");
}

/// A Separated file whose inks are not CMYK must not be converted. There is no
/// such fixture (neither writer emits InkSet), so this checks the gate the
/// viewer actually consults rather than asserting on a file.
#[test]
fn the_viewer_only_converts_what_the_library_calls_cmyk() {
    let Some(path) = fixture("tff_u8_spp3_p2_none.tif") else { return };
    let stack = open_stack(path).expect("rgb should open");
    assert!(!stack.tiff.frames[0].is_cmyk(), "a photometric-2 RGB frame is not CMYK");
    assert!(!stack.display.cmyk, "and must not be flagged as such");
    assert!(stack.display.rgb, "it is still ordinary RGB");
}

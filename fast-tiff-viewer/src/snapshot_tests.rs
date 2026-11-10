//! The composed image, checked against what the shader would produce.
//!
//! These tests are the reason the compositor is a separate module: the whole
//! risk in it is that the arithmetic drifts from `composite.wgsl` and a copied
//! picture stops matching the one on screen — brighter, banded, or thresholded
//! differently. That is a difference nobody notices by looking, so each step of
//! the pipeline is pinned here by value rather than by eye.

use super::*;
use crate::prefetch::build_jobs;
use crate::stack::{SampleUnits, Stack, Window};
use fast_tiff_lib::{DisplayMap, SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use scivis_render::ChannelKind;
use std::io::Cursor;

const W: u32 = 4;
const H: u32 = 3;

/// A grey ramp, which is the LUT that makes a composed value readable: with a
/// full window it maps a sample straight back to itself, so a wrong window or a
/// wrong LUT read shows up as a wrong *number* rather than as a wrong shade.
fn ramp() -> Lut {
    std::array::from_fn(|i| [i as u8, i as u8, i as u8])
}

/// A LUT that is zero everywhere except its top entry — for telling an
/// interpolated read from a nearest one.
fn only_top() -> Lut {
    let mut lut = [[0u8; 3]; 256];
    lut[255] = [255, 255, 255];
    lut
}

/// An 8-bit stack of `channels` channels whose sample values are given by `f`.
fn stack_u8(channels: usize, f: impl Fn(usize, u32) -> u8) -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::U8).metadata(StackMetaWrite::new(channels, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for c in 0..channels {
        let px: Vec<u8> = (0..W * H).map(|i| f(c, i)).collect();
        w.write_frame_u8(&px).expect("frame");
    }
    let bytes = w.finish().expect("finish").into_inner();
    Stack::from_bytes(bytes, "snapshot.tif".into(), false).expect("the stack should open")
}

/// Give every channel the same window and LUT, so a test that is not about the
/// window or the LUT is unaffected by whatever the file declared.
///
/// `min`/`max` are in **sample** units — the units the test reasons in, where a
/// full 8-bit window is 0..255 — which are the units the field holds. They
/// once were not: the field kept an 8-bit window on the 16-bit scale, and a
/// first version of this helper that wrote sample units straight into it made
/// every test agree with a compositor that was off by a factor of the
/// widening. So the helper still checks that the window reaches the texture's
/// units as the test meant it.
fn with(stack: &mut Stack, lut: Lut, min: f32, max: f32) {
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(min, max));
        stack.display.set_enabled(c, true);
        // These stacks are 8-bit, whose texture holds the samples themselves:
        // if the window does not arrive there as given, the helper is quietly
        // testing something else.
        let s = stack.display.settings()[c];
        let (lo, hi) = s.units.to_texture(s.window);
        assert!(
            (lo - min).abs() < 0.01 && (hi - max).abs() < 0.01,
            "the window helper should reach the texture as {min}..{max}, got {lo}..{hi}"
        );
    }
    let n = stack.display.luts().len();
    stack.display.set_luts(vec![lut; n]);
}

/// The copy is the image's own size, not the window's. This is the property
/// that separates "copy the view" from "screenshot the viewport", and it is
/// stated first because everything else assumes it.
#[test]
fn the_snapshot_is_the_frames_own_size() {
    let mut stack = stack_u8(1, |_, i| i as u8);
    with(&mut stack, ramp(), 0.0, 255.0);
    let snap = of_frame(&stack).expect("compose");
    assert_eq!((snap.width, snap.height), (W, H));
    assert_eq!(snap.rgba.len(), (W * H * 4) as usize);
}

/// A full window through a grey ramp is the identity, so every sample comes
/// back as itself. If the window arithmetic or the LUT read is off by even one
/// step, this is where it shows as a number.
#[test]
fn a_full_window_through_a_grey_ramp_returns_the_sample_itself() {
    let mut stack = stack_u8(1, |_, i| (i * 20) as u8);
    with(&mut stack, ramp(), 0.0, 255.0);
    let snap = of_frame(&stack).expect("compose");
    for i in 0..(W * H) {
        let want = (i * 20) as u8;
        let got = snap.pixel(i % W, i / W).expect("in range");
        assert_eq!(
            got,
            [want, want, want, 255],
            "pixel {i} should round-trip {want}"
        );
    }
}

/// Fully opaque, always. A clipboard image with a zero alpha channel pastes as
/// nothing at all in most programs, which looks like the copy having silently
/// failed.
#[test]
fn every_pixel_is_opaque() {
    let mut stack = stack_u8(1, |_, i| i as u8);
    with(&mut stack, ramp(), 0.0, 255.0);
    let snap = of_frame(&stack).expect("compose");
    assert!(
        snap.rgba.chunks(4).all(|p| p[3] == 255),
        "every pixel should be opaque"
    );
}

/// Channels add and the sum is clamped, which is ImageJ's composite mode and
/// what the shader's final `clamp` does. Accumulating in 8 bits instead would
/// wrap, turning an overlap of two bright channels dark.
#[test]
fn channels_blend_additively_and_clamp_rather_than_wrap() {
    // Two channels, both at full scale everywhere: 1.0 + 1.0 clamped to 1.0.
    let mut stack = stack_u8(2, |_, _| 255);
    with(&mut stack, ramp(), 0.0, 255.0);
    if stack.display.settings().len() < 2 {
        // The file was interpreted as one channel; the blend is not under test
        // here and a wrong-axis file would make this assert nothing.
        panic!(
            "expected two display channels, got {}",
            stack.display.settings().len()
        );
    }
    let snap = of_frame(&stack).expect("compose");
    assert_eq!(
        snap.pixel(0, 0).expect("in range"),
        [255, 255, 255, 255],
        "two saturated channels should clamp to white, not wrap to black"
    );
}

/// A switched-off channel contributes nothing — the shader multiplies it by
/// `enabled`, and this skips decoding it, which must come to the same picture.
#[test]
fn a_disabled_channel_contributes_nothing() {
    let mut stack = stack_u8(2, |c, _| if c == 0 { 100 } else { 255 });
    with(&mut stack, ramp(), 0.0, 255.0);
    assert!(stack.display.settings().len() >= 2, "need two channels");
    stack.display.set_enabled(1, false);
    let snap = of_frame(&stack).expect("compose");
    assert_eq!(
        snap.pixel(0, 0).expect("in range"),
        [100, 100, 100, 255],
        "only the enabled channel should show"
    );

    // And with both on, the same pixel is brighter — so the test above is
    // measuring the switch rather than a channel that was never there.
    stack.display.set_enabled(1, true);
    let both = of_frame(&stack).expect("compose");
    assert!(
        both.pixel(0, 0).expect("in range")[0] > 100,
        "enabling the second channel should brighten the pixel"
    );
}

/// The window's span has a one-unit floor, copied from the shader. A window
/// dragged shut would otherwise divide by something near zero; with the floor it
/// behaves as a one-unit window, which is a hard threshold at `min`.
#[test]
fn a_shut_window_becomes_a_one_unit_threshold_rather_than_a_division_by_zero() {
    assert_eq!(windowed(100.0, 100.0, 100.0), 0.0, "at the threshold");
    assert_eq!(windowed(101.0, 100.0, 100.0), 1.0, "one unit above it");
    assert_eq!(windowed(99.0, 100.0, 100.0), 0.0, "below it");
    assert!(
        windowed(1e9, 5.0, 5.0).is_finite(),
        "a shut window must not produce an infinity"
    );
    // Inverted handles are the same floor, not a negative span.
    assert_eq!(windowed(50.0, 100.0, 0.0), 0.0, "inverted, below min");
    assert!(windowed(150.0, 100.0, 0.0).is_finite());
}

/// Clamped at both ends, so a sample outside the window saturates instead of
/// running past the LUT.
#[test]
fn samples_outside_the_window_saturate() {
    assert_eq!(windowed(-1000.0, 0.0, 100.0), 0.0);
    assert_eq!(windowed(1000.0, 0.0, 100.0), 1.0);
    assert_eq!(windowed(50.0, 0.0, 100.0), 0.5);
}

/// The LUT ends are exact. The sampler clamps to edge, so `t = 0` is the first
/// entry and `t = 1` the last — not a half-texel blend into whatever is beyond,
/// which would make a full-scale pixel slightly dim.
#[test]
fn the_lut_ends_are_exact() {
    let lut = ramp();
    assert_eq!(lut_sample(&lut, 0.0), [0.0, 0.0, 0.0]);
    let top = lut_sample(&lut, 1.0);
    for v in top {
        assert!((v - 1.0).abs() < 1e-6, "t=1 should be the top entry");
    }
    // Out of range cannot escape the table.
    assert_eq!(lut_sample(&lut, -5.0), [0.0, 0.0, 0.0]);
    assert_eq!(lut_sample(&lut, 5.0), top);
}

/// Between entries the read interpolates, because the sampler filters linearly.
/// A nearest read would band a smooth ramp against the same ramp on screen.
#[test]
fn the_lut_is_read_with_interpolation_not_nearest() {
    let lut = only_top();
    // Halfway between entry 254 (black) and 255 (white) is the centre of the
    // last texel interval: t such that t*256 - 0.5 == 254.5.
    let t = 255.0 / 256.0;
    let mid = lut_sample(&lut, t);
    assert!(
        mid[0] > 0.1 && mid[0] < 0.9,
        "a linear sampler should blend the two entries, got {}",
        mid[0]
    );
    // A nearest read of the same position would give one or the other exactly.
    assert!(
        (mid[0] - 0.0).abs() > 1e-3 && (mid[0] - 1.0).abs() > 1e-3,
        "neither endpoint: {}",
        mid[0]
    );
}

/// A narrower window stretches the contrast, which is the control doing its
/// job: the same samples come out brighter.
#[test]
fn narrowing_the_window_brightens_the_picture() {
    let mut stack = stack_u8(1, |_, _| 40);
    with(&mut stack, ramp(), 0.0, 255.0);
    let wide = of_frame(&stack).expect("compose").pixel(0, 0).unwrap()[0];
    with(&mut stack, ramp(), 0.0, 60.0);
    let narrow = of_frame(&stack).expect("compose").pixel(0, 0).unwrap()[0];
    assert!(
        narrow > wide,
        "a tighter window should brighten a mid sample: {wide} -> {narrow}"
    );
}

/// The LUT actually colours the result — a red LUT gives red pixels. Cheap, but
/// it is the assertion that would fail if the channel's LUT were looked up by
/// the wrong index, which on a multi-channel stack is an easy mistake and looks
/// merely "wrong-coloured" rather than broken.
#[test]
fn each_channel_is_coloured_by_its_own_lut() {
    let mut stack = stack_u8(2, |_, _| 255);
    with(&mut stack, ramp(), 0.0, 255.0);
    assert!(stack.display.settings().len() >= 2, "need two channels");
    // Channel 0 red, channel 1 green: the sum must be yellow with no blue.
    stack
        .display
        .set_lut(0, std::array::from_fn(|i| [i as u8, 0, 0]));
    stack
        .display
        .set_lut(1, std::array::from_fn(|i| [0, i as u8, 0]));
    let snap = of_frame(&stack).expect("compose");
    assert_eq!(
        snap.pixel(0, 0).expect("in range"),
        [255, 255, 0, 255],
        "red plus green is yellow, and nothing should land in blue"
    );
}

/// Scrubbing to another frame copies *that* frame. The snapshot reads
/// `frame_index`, so this is really a check that it is read at all rather than
/// baked in at open time.
#[test]
fn the_snapshot_follows_the_current_frame() {
    // Three single-channel frames, each a flat value identifying itself.
    let opts = WriterOptions::new(W, H, SampleType::U8).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for v in [10u8, 120, 240] {
        w.write_frame_u8(&vec![v; (W * H) as usize]).expect("frame");
    }
    let bytes = w.finish().expect("finish").into_inner();
    let mut stack = Stack::from_bytes(bytes, "frames.tif".into(), false).expect("open");
    with(&mut stack, ramp(), 0.0, 255.0);

    for (i, want) in [10u8, 120, 240].iter().enumerate() {
        stack.frame_index = i;
        let snap = of_frame(&stack).expect("compose");
        assert_eq!(
            snap.pixel(0, 0).expect("in range")[0],
            *want,
            "frame {i} should copy its own pixels"
        );
    }
}

/// The window is in the file's units and the samples are in texture units, and
/// for a channel on the 16-bit texture those differ.
///
/// This is the conversion [`SampleUnits::to_texture`] exists for, and it is
/// asserted here because its consumer is this module. Every test above sets
/// the window by hand on 8-bit data, whose texture holds the samples
/// themselves, and so cannot see it — which is exactly how the bug it once
/// fixed got in: the compositor paired a window on one scale with samples on
/// another and every 8-bit image came out black, while the synthetic tests all
/// passed.
#[test]
fn a_window_is_converted_into_the_textures_units() {
    let full8 = Window::new(0.0, 255.0);
    assert_eq!(
        SampleUnits::RAW8.to_texture(full8),
        (0.0, 255.0),
        "an R8 texture holds the 8-bit samples themselves"
    );
    let widened = SampleUnits::display(DisplayMap::Widen8);
    assert_eq!(
        widened.to_texture(full8),
        (0.0, 65_535.0),
        "8-bit samples widened onto R16 take the window with them"
    );
    assert_eq!(
        SampleUnits::display(DisplayMap::Identity).to_texture(Window::new(17.0, 33029.0)),
        (17.0, 33029.0)
    );
    assert_eq!(
        SampleUnits::float(Window::new(-4.0, 4.0)).to_texture(Window::new(-1.5, 2.5)),
        (-1.5, 2.5)
    );
}

/// An 8-bit image composed with the window the *file* produced is not black.
///
/// The test that was missing. Everything else here overrides the window, so
/// nothing exercised the units the interface actually stores — and a whole
/// class of real file composed to zero while the suite stayed green. No `with`
/// call in this one on purpose: whatever opening the file derived is what is
/// under test.
#[test]
fn an_eight_bit_file_composes_with_its_own_window() {
    // A ramp rather than a flat plane: auto-contrast on a constant image is a
    // legitimately degenerate window, which composes to black for an honest
    // reason and would make this assert nothing.
    let last = W * H - 1;
    let stack = stack_u8(1, |_, i| (i * 255 / last) as u8);
    assert_eq!(
        stack.display.settings()[0].kind(),
        ChannelKind::Int8,
        "an 8-bit file should take the 8-bit texture path, or this tests nothing"
    );
    let snap = of_frame(&stack).expect("compose");
    let dark = snap.pixel(0, 0).expect("in range")[0];
    let bright = snap.pixel(W - 1, H - 1).expect("in range")[0];
    assert!(
        bright > 200,
        "the top of the ramp composed to {bright}; with the window on the 16-bit \
         scale the whole image would be black"
    );
    assert!(dark < 56, "and the bottom should stay dark: {dark}");
}

/// A signed 16-bit channel rides the 16-bit texture offset into unsigned,
/// while its window is in the signed values: the compositor has to put the one
/// on the other's scale. Samples -6..=5 here, window -4..0, so -4 and below are
/// black, -2 is mid-grey and 0 and above are white. Composed without the
/// conversion, every sample would sit far above the window and come out white.
#[test]
fn a_signed_window_composes_on_the_offset_texture() {
    let opts = WriterOptions::new(W, H, SampleType::I16).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    let px: Vec<u8> = (0..(W * H) as i16)
        .flat_map(|i| (i - 6).to_le_bytes())
        .collect();
    w.write_frame_bytes(&px).expect("frame");
    let bytes = w.finish().expect("finish").into_inner();
    let mut stack = Stack::from_bytes(bytes, "signed.tif".into(), false).expect("open");
    assert_eq!(stack.display.settings()[0].kind(), ChannelKind::Int16);
    stack.display.set_window(0, Window::new(-4.0, 0.0));
    let n = stack.display.luts().len();
    stack.display.set_luts(vec![ramp(); n]);

    let snap = of_frame(&stack).expect("compose");
    let grey = |i: u32| snap.pixel(i % W, i / W).expect("in range")[0];
    // Sample i - 6: pixel 2 is -4, pixel 4 is -2, pixel 6 is 0.
    assert_eq!(grey(0), 0, "below the window is black");
    assert_eq!(grey(2), 0, "the window's low end is black");
    assert!(
        (120..=136).contains(&grey(4)),
        "mid-window is mid-grey: {}",
        grey(4)
    );
    assert_eq!(grey(6), 255, "the window's high end is white");
    assert_eq!(grey(11), 255, "above the window is white");
}

/// The compositor as it was before it worked a band at a time: every enabled
/// channel decoded whole, blended into one frame-sized accumulator, then
/// converted. Kept verbatim as the oracle the banded one must equal.
fn whole_frame_oracle(stack: &Stack) -> Snapshot {
    use crate::prefetch::{decode_jobs, Decoded};
    fn sample(data: &Decoded, i: usize) -> Option<f32> {
        match data {
            Decoded::U8(v) => v.get(i).map(|&s| s as f32),
            Decoded::U16(v) => v.get(i).map(|&s| s as f32),
            Decoded::F32(v) => v.get(i).copied(),
        }
    }
    let (width, height) = stack.dimensions().expect("open");
    let settings = stack.display.settings();
    let masks = stack.display.masks();
    let jobs = build_jobs(stack, stack.frame_index, masks.enabled());
    let decoded = decode_jobs(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    )
    .expect("decode");
    let pixels = width as usize * height as usize;
    let mut acc = vec![[0.0f32; 3]; pixels];
    for (job, data) in jobs.iter().zip(&decoded) {
        let Some(s) = settings.get(job.channel) else {
            continue;
        };
        let Some(lut) = stack.display.lut(job.channel) else {
            continue;
        };
        let (min, max) = s.units.to_texture(s.window);
        for (i, out) in acc.iter_mut().enumerate() {
            let Some(value) = sample(data, i) else {
                break;
            };
            let rgb = lut_sample(lut, windowed(value, min, max));
            for (o, v) in out.iter_mut().zip(rgb) {
                *o += v;
            }
        }
    }
    let mut rgba = Vec::with_capacity(pixels * 4);
    for channel in &acc {
        for v in channel {
            rgba.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        rgba.push(255);
    }
    Snapshot {
        width,
        height,
        rgba,
    }
}

/// Seven rows in strips of two, so bands start and end mid-strip and a strip
/// is shared by two bands; five columns, so no row is a power of two wide.
const OW: u32 = 5;
const OH: u32 = 7;

/// A stack of `channels` channels of `sample` data in strips of two rows,
/// every sample distinct, each channel with a LUT of its own and a window that
/// clips both ends: a pixel read from the wrong row, channel or band shows as
/// a different byte.
fn oracle_stack(sample: SampleType, channels: usize, spp: u16) -> Stack {
    let opts = WriterOptions::new(OW, OH, sample)
        .rows_per_strip(2)
        .samples_per_pixel(spp)
        .metadata(StackMetaWrite::new(channels, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    let n = (OW * OH) as usize * spp as usize;
    for c in 0..channels {
        let value = |i: usize| (i * 37 + c * 101) % 251;
        match sample {
            SampleType::U8 => {
                let px: Vec<u8> = (0..n).map(|i| value(i) as u8).collect();
                w.write_frame_u8(&px).expect("frame");
            }
            SampleType::U16 => {
                let px: Vec<u16> = (0..n).map(|i| (value(i) * 211) as u16).collect();
                w.write_frame_u16(&px).expect("frame");
            }
            SampleType::F32 => {
                let px: Vec<f32> = (0..n).map(|i| value(i) as f32 / 7.0 - 3.0).collect();
                w.write_frame_f32(&px).expect("frame");
            }
            other => panic!("no fixture for {other:?}"),
        }
    }
    let bytes = w.finish().expect("finish").into_inner();
    let mut stack = Stack::from_bytes(bytes, "oracle.tif".into(), false).expect("open");
    for c in 0..stack.display.channel_count() {
        let s = stack.display.settings()[c];
        let (lo, hi) = (s.bounds.lo, s.bounds.hi);
        let span = hi - lo;
        stack
            .display
            .set_window(c, Window::new(lo + span * 0.1, hi - span * 0.2));
        let tint = [(c % 3 == 0) as u8, (c % 3 == 1) as u8, (c % 3 != 1) as u8];
        stack.display.set_lut(
            c,
            std::array::from_fn(|i| tint.map(|t| t * i as u8 + (255 - i as u8) / 8)),
        );
    }
    stack
}

/// Re-describe each `(ifd, rows_per_strip)`'s strips, which are uncompressed
/// and contiguous as the writer lays them out, as strips of that many rows:
/// the same bytes in another valid layout, which no writer here produces.
fn restrip(stack: &mut Stack, layouts: &[(usize, u32)]) {
    let mut tiff = fast_tiff_lib::TiffStack::from_bytes(stack.tiff.data.to_vec()).expect("index");
    for &(ifd, rows_per_strip) in layouts {
        let frame = &mut tiff.frames[ifd];
        let offsets = frame.strip_offsets.to_vec();
        let counts = frame.strip_byte_counts.to_vec();
        assert!(
            offsets
                .windows(2)
                .zip(&counts)
                .all(|(o, c)| o[0] + c == o[1]),
            "the writer's strips should be contiguous"
        );
        let total: u64 = counts.iter().sum();
        let strip = total / frame.height as u64 * rows_per_strip as u64;
        let starts: Vec<u64> = (0..total).step_by(strip as usize).collect();
        frame.strip_byte_counts = starts.iter().map(|&at| strip.min(total - at)).collect();
        frame.strip_offsets = starts.iter().map(|&at| offsets[0] + at).collect();
        frame.rows_per_strip = rows_per_strip;
    }
    stack.tiff = std::sync::Arc::new(tiff);
}

/// Composed a band at a time, the image is byte for byte the one the whole
/// frame composed at once gave: for several channels, through the
/// accumulator, with one switched off; for one, with no accumulator; for RGB
/// planes decoded together; for floats; for channels whose IFDs are laid out
/// in strips of different heights, so their bands end on different rows. And
/// with decoded and composed bands of every height from one row to more than
/// the frame.
#[test]
fn the_banded_image_is_byte_identical_to_the_whole_frame_one() {
    let cases: Vec<(&str, Stack)> = vec![
        ("three u16 channels", oracle_stack(SampleType::U16, 3, 1)),
        ("three u16 channels, one off", {
            let mut s = oracle_stack(SampleType::U16, 3, 1);
            s.display.set_enabled(1, false);
            s
        }),
        ("one u8 channel", oracle_stack(SampleType::U8, 1, 1)),
        ("two f32 channels", oracle_stack(SampleType::F32, 2, 1)),
        ("chunky RGB", oracle_stack(SampleType::U8, 1, 3)),
        ("three u16 channels in strips of 2, 3 and 7 rows", {
            let mut s = oracle_stack(SampleType::U16, 3, 1);
            restrip(&mut s, &[(1, 3), (2, OH)]);
            s
        }),
    ];
    for (name, stack) in &cases {
        let want = whole_frame_oracle(stack);
        assert!(
            want.rgba.chunks(4).any(|p| p[..3] != [0, 0, 0]),
            "{name}: the oracle composed a black image, which tests nothing"
        );
        for decode_rows in [1, 2, 3, 4, OH, 100] {
            for compose_rows in [1, 2, 3, OH, 100] {
                let got = compose(stack, OW, OH, decode_rows, compose_rows).expect("compose");
                assert_eq!(
                    got, want,
                    "{name}, decoded {decode_rows} and composed {compose_rows} rows at a time"
                );
            }
        }
        assert_eq!(of_frame(stack).expect("compose"), want, "{name}");
    }
    let layouts: Vec<u32> = cases[5].1.tiff.frames[..3]
        .iter()
        .map(|f| f.rows_per_strip)
        .collect();
    assert_eq!(layouts, [2, 3, OH], "the IFDs should disagree on strips");
    assert_eq!(
        cases[4].1.display.channel_count(),
        3,
        "RGB has three planes"
    );
}

/// A frame whose image is past the limit is refused with that reason, before
/// any decode or buffer. This one claims 100000 x 100000 pixels, a 40 GB
/// image no test machine could allocate, over strips that hold nothing that
/// size: reaching either the allocation or the decode would fail some other
/// way.
#[test]
fn an_oversized_frame_is_refused_before_anything_is_decoded_or_allocated() {
    let mut stack = stack_u8(2, |_, i| i as u8);
    let mut tiff = fast_tiff_lib::TiffStack::from_bytes(stack.tiff.data.to_vec()).expect("index");
    for frame in &mut tiff.frames {
        frame.width = 100_000;
        frame.height = 100_000;
    }
    stack.tiff = std::sync::Arc::new(tiff);
    let err = of_frame(&stack).expect_err("a 40 GB copy should be refused");
    assert_eq!(
        err.downcast_ref::<TooLarge>(),
        Some(&TooLarge {
            width: 100_000,
            height: 100_000
        }),
        "refused for another reason: {err:#}"
    );
    let message = format!("{err:#}");
    assert!(
        message.contains("100000 x 100000") && message.contains("crop"),
        "the refusal should say what is too large and what to do: {message}"
    );

    // The limit is the 8192 x 8192 image, inclusive.
    assert!(rgba_bytes(8192, 8192) <= MAX_SNAPSHOT_BYTES);
    assert!(rgba_bytes(8192, 8193) > MAX_SNAPSHOT_BYTES);
    // And the arithmetic does not wrap, here or where a `usize` is 32 bits.
    assert_eq!(rgba_bytes(65_536, 65_536), 1 << 34);
    assert_eq!(rgba_bytes(u32::MAX, u32::MAX), u64::MAX);
}

//! The histogram dialog's Auto and Reset buttons.
//!
//! Auto exists because the useful part of a microscopy frame is usually a
//! narrow band with a scattering of hot pixels and a dark floor around it. A
//! window fitted to the exact min and max is therefore the same window the
//! slider already spans, and shows nothing; the whole value is in throwing away
//! a fraction of a percent at each end. So what these tests pin is not "the
//! window changed" but *that the outliers are the part that got dropped* — and
//! that Reset puts back exactly what was there before.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::channels::{auto_contrast, reset_contrast};
use fast_tiff_viewer::histogram::{auto_window, frame_histograms, Histogram, BINS};
use fast_tiff_viewer::Stack;
use std::io::Cursor;

/// A one-frame u16 stack of `w * h` pixels: `outliers` of them pinned to 0,
/// `outliers` to 65535, and every other pixel at `bulk`.
///
/// The extremes are there for two reasons. They set the slider bounds to the
/// full 16-bit range, so the window Auto picks is measured against a track that
/// does not move between cases — and they are precisely the samples Auto is
/// supposed to discard, so their presence is what makes the test meaningful.
fn spiked_u16_stack(w: u32, h: u32, bulk: u16, outliers: usize) -> Vec<u8> {
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(1, 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    writer.write_frame_bytes(&spiked_frame(w, h, bulk, outliers)).unwrap();
    writer.finish().unwrap().into_inner()
}

/// The same pixel pattern as [`spiked_u16_stack`], as raw little-endian bytes.
fn spiked_frame(w: u32, h: u32, bulk: u16, outliers: usize) -> Vec<u8> {
    let n = (w * h) as usize;
    assert!(n > 2 * outliers, "the bulk has to outnumber the outliers");
    let mut px = vec![bulk; n];
    for i in 0..outliers {
        px[i] = 0;
        px[n - 1 - i] = u16::MAX;
    }
    px.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A multi-frame version: frame `f` has its bulk at `bulks[f]`, with the same
/// extremes in every frame so the track is identical across them.
fn drifting_u16_stack(bulks: &[u16], w: u32, h: u32, outliers: usize) -> Vec<u8> {
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(1, bulks.len()));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for &b in bulks {
        writer.write_frame_bytes(&spiked_frame(w, h, b, outliers)).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// A multi-channel, multi-timepoint stack: `per_frame[t][c]` is the flat value
/// of channel `c` at timepoint `t`. Planes are written in ImageJ order, channel
/// varying fastest, which is what the plane addressing expects.
fn composite_u16_stack(per_frame: &[&[u16]], w: u32, h: u32) -> Vec<u8> {
    let channels = per_frame[0].len();
    let opts = WriterOptions::new(w, h, SampleType::U16)
        .metadata(StackMetaWrite::new(channels, per_frame.len()));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for frame in per_frame {
        assert_eq!(frame.len(), channels, "every timepoint needs every channel");
        for &v in *frame {
            let bytes: Vec<u8> = (0..w * h).flat_map(|_| v.to_le_bytes()).collect();
            writer.write_frame_bytes(&bytes).unwrap();
        }
    }
    writer.finish().unwrap().into_inner()
}

/// Like [`composite_u16_stack`], but each plane is given as a pair of values
/// filling half its pixels each. A channel whose distribution has two modes can
/// straddle the end of its own track instead of sitting wholly inside or wholly
/// outside it, which is the only way one clamp fires without the other.
fn split_u16_stack(per_frame: &[&[(u16, u16)]], w: u32, h: u32) -> Vec<u8> {
    let channels = per_frame[0].len();
    let opts = WriterOptions::new(w, h, SampleType::U16)
        .metadata(StackMetaWrite::new(channels, per_frame.len()));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let n = (w * h) as usize;
    for frame in per_frame {
        for &(a, b) in *frame {
            let px: Vec<u8> =
                (0..n).flat_map(|i| if i < n / 2 { a.to_le_bytes() } else { b.to_le_bytes() }).collect();
            writer.write_frame_bytes(&px).unwrap();
        }
    }
    writer.finish().unwrap().into_inner()
}

/// [`spiked_u16_stack`] plus a declared display window, so the window the file
/// opens with is narrower than the track its slider spans. That gap is the
/// whole point of the distinction Reset draws, and most files do not have one:
/// with no declared window the settings start at the full data range, and the
/// two candidate meanings of Reset coincide.
fn windowed_u16_stack(w: u32, h: u32, bulk: u16, outliers: usize, window: (f64, f64)) -> Vec<u8> {
    let meta = StackMetaWrite::new(1, 1).range(window.0, window.1);
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(meta);
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    writer.write_frame_bytes(&spiked_frame(w, h, bulk, outliers)).unwrap();
    writer.finish().unwrap().into_inner()
}

/// The float counterpart of [`windowed_u16_stack`]. Float channels take their
/// own branch when the settings are built — they keep their window in the
/// source units instead of being rescaled onto 0..=65535 — so an integer stack
/// says nothing about whether that branch remembers where it started.
fn windowed_f32_stack(w: u32, h: u32, bulk: f32, extremes: (f32, f32), window: (f64, f64)) -> Vec<u8> {
    let meta = StackMetaWrite::new(1, 1).range(window.0, window.1);
    let opts = WriterOptions::new(w, h, SampleType::F32).metadata(meta);
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let n = (w * h) as usize;
    let mut px = vec![bulk; n];
    px[0] = extremes.0;
    px[n - 1] = extremes.1;
    let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
    writer.write_frame_bytes(&bytes).unwrap();
    writer.finish().unwrap().into_inner()
}

fn load(bytes: Vec<u8>) -> Stack {
    Stack::from_bytes(bytes, "auto.tif".into(), false).expect("should open")
}

/// 10_000 pixels, 20 at each extreme. The saturation budget is
/// `ceil(10_000 * 0.0035)` = 35 per end, comfortably more than 20, so both
/// spikes are inside the budget and get clipped.
const W: u32 = 100;
const H: u32 = 100;
const OUTLIERS: usize = 20;

#[test]
fn auto_window_discards_the_outliers_and_keeps_the_bulk() {
    let stack = load(spiked_u16_stack(W, H, 30_000, OUTLIERS));
    let hists = frame_histograms(&stack);
    assert_eq!(hists.len(), 1);

    let (lo, hi) = auto_window(&hists[0]).expect("a counted histogram has a window");
    assert!(lo > 0.0, "the dark spike at 0 should have been clipped, got lo={lo}");
    assert!(hi < u16::MAX as f32, "the bright spike at 65535 should have been clipped, got hi={hi}");
    assert!(lo <= 30_000.0 && 30_000.0 <= hi, "the bulk at 30000 must stay inside {lo}..{hi}");

    // One bin either side at most: the bulk is a single spike, so anything
    // wider means the walk overshot rather than stopping at the first bin that
    // broke the budget.
    let bin = (u16::MAX as f32) / BINS as f32;
    assert!(hi - lo <= 2.0 * bin, "window {lo}..{hi} is wider than the spike that earned it");
}

/// The budget has to be large enough to actually reach the outliers. With more
/// of them than the saturation allows, they are data rather than noise and the
/// window must keep them — otherwise Auto would clip whatever happens to be at
/// the ends regardless of how much of it there is.
#[test]
fn outliers_too_numerous_to_be_noise_are_kept() {
    // 500 at each end of 10_000 is 5%, far past the 0.35% budget.
    let stack = load(spiked_u16_stack(W, H, 30_000, 500));
    let hists = frame_histograms(&stack);
    let (lo, hi) = auto_window(&hists[0]).unwrap();
    let bin = (u16::MAX as f32) / BINS as f32;
    assert!(lo < bin, "the low spike is too big to discard, got lo={lo}");
    assert!(hi > u16::MAX as f32 - bin, "the high spike is too big to discard, got hi={hi}");
}

/// A frame of one single value has no spread to fit. The window must still come
/// back non-empty: a zero-width contrast window divides by zero on the way to
/// the shader and blanks the image.
#[test]
fn a_single_valued_frame_yields_a_non_empty_window() {
    let stack = load(spiked_u16_stack(W, H, 1234, 0));
    let hists = frame_histograms(&stack);
    let (lo, hi) = auto_window(&hists[0]).unwrap();
    assert!(hi > lo, "window {lo}..{hi} must not be empty");
    assert!(lo <= 1234.0 && 1234.0 <= hi, "the one value present must be inside {lo}..{hi}");
}

/// A histogram that counted nothing has no defensible answer, so Auto declines
/// rather than inventing one.
#[test]
fn an_empty_histogram_has_no_auto_window() {
    let empty = Histogram { channel: 0, bins: vec![0; BINS], lo: 0.0, hi: 65535.0, peak: 0, counted: 0 };
    assert!(auto_window(&empty).is_none());
}

#[test]
fn auto_narrows_the_window_and_reset_restores_it() {
    let mut stack = load(spiked_u16_stack(W, H, 30_000, OUTLIERS));
    let bounds = stack.display.settings[0].bounds;
    let initial = stack.display.settings[0].initial;
    let hists = frame_histograms(&stack);

    auto_contrast(&mut stack, &hists);
    let s = &stack.display.settings[0];
    assert!(
        s.max - s.min < (bounds.1 - bounds.0) * 0.5,
        "auto should have narrowed {:?} well inside {bounds:?}",
        (s.min, s.max)
    );
    assert!(s.min >= bounds.0 && s.max <= bounds.1, "both handles must stay on the track");

    reset_contrast(&mut stack);
    let s = &stack.display.settings[0];
    assert_eq!((s.min, s.max), initial, "reset should undo auto");
}

/// Reset is the undo for any amount of dragging, not just for Auto.
#[test]
fn reset_recovers_from_an_arbitrary_window() {
    let mut stack = load(spiked_u16_stack(W, H, 30_000, OUTLIERS));
    let initial = stack.display.settings[0].initial;
    stack.display.settings[0].min = 12_345.0;
    stack.display.settings[0].max = 12_346.0;

    reset_contrast(&mut stack);
    assert_eq!((stack.display.settings[0].min, stack.display.settings[0].max), initial);
}

/// The same, for a float stack. Float channels keep their contrast window in
/// the data's own units rather than being rescaled onto 0..=65535, which is a
/// separate branch of the settings builder — and one where "where this channel
/// started" is easy to conflate with the data range it was measured from.
#[test]
fn reset_returns_to_the_file_window_for_float_channels_too() {
    let declared = (10.0f32, 40.0f32);
    let mut stack = load(windowed_f32_stack(
        W,
        H,
        25.0,
        (-100.0, 200.0),
        (declared.0 as f64, declared.1 as f64),
    ));
    let s = &stack.display.settings[0];
    assert_eq!(s.kind, fast_tiff_viewer::ChannelKind::Float, "this must exercise the float branch");
    assert_eq!((s.min, s.max), declared, "float windows stay in the data units");
    let bounds = s.bounds;
    assert!(
        bounds.0 < declared.0 && bounds.1 > declared.1,
        "the track {bounds:?} should be widened to the data, past the declared {declared:?}"
    );

    stack.display.settings[0].min = 100.0;
    stack.display.settings[0].max = 150.0;
    reset_contrast(&mut stack);

    let s = &stack.display.settings[0];
    assert_eq!((s.min, s.max), declared, "reset should restore the file window");
    assert_ne!((s.min, s.max), bounds, "reset should NOT open the window to the full track");
}

/// What Reset means, on the only kind of file where the two candidate meanings
/// differ: one that declares a display window narrower than its own data.
///
/// Anything saved out of ImageJ normally does. Opening such a file to the full
/// track would show it in a state it has never been shown in, and call that a
/// reset — so Reset goes back to the window the file asked for, which is also
/// what the sliders were sitting at a moment ago.
#[test]
fn reset_returns_to_the_file_window_not_the_full_track() {
    let declared = (10_000.0f32, 40_000.0f32);
    let mut stack =
        load(windowed_u16_stack(W, H, 30_000, OUTLIERS, (declared.0 as f64, declared.1 as f64)));
    let s = &stack.display.settings[0];
    assert_eq!((s.min, s.max), declared, "the file window should be what the stack opens with");
    assert_eq!(s.initial, declared);
    let bounds = s.bounds;
    assert!(
        bounds.0 < declared.0 && bounds.1 > declared.1,
        "the track {bounds:?} should be wider than the declared window {declared:?},          or this test cannot tell the two resets apart"
    );

    // Drag both handles somewhere else entirely, then reset.
    stack.display.settings[0].min = 55_000.0;
    stack.display.settings[0].max = 60_000.0;
    reset_contrast(&mut stack);

    let s = &stack.display.settings[0];
    assert_eq!((s.min, s.max), declared, "reset should restore the file window");
    assert_ne!((s.min, s.max), bounds, "reset should NOT open the window to the full track");
}

/// The documented reason Auto reads histograms instead of rescanning frame 0:
/// on a stack whose brightness drifts, fitting to frame 0 while looking at a
/// later frame would be visibly wrong.
#[test]
fn auto_fits_the_frame_on_screen_not_frame_zero() {
    let mut stack = load(drifting_u16_stack(&[10_000, 50_000], W, H, OUTLIERS));
    assert!(stack.display.dims.frames >= 2, "need both frames addressable");

    stack.frame_index = 1;
    let hists = frame_histograms(&stack);
    auto_contrast(&mut stack, &hists);

    let s = &stack.display.settings[0];
    assert!(s.min <= 50_000.0 && 50_000.0 <= s.max, "frame 1 sits at 50000, window is {:?}", (s.min, s.max));
    assert!(
        !(s.min <= 10_000.0 && 10_000.0 <= s.max),
        "frame 0 at 10000 should be outside the window fitted to frame 1, got {:?}",
        (s.min, s.max)
    );
}

/// A palette stack has no contrast window to fit. Its `min`/`max` are a pinned
/// index-to-LUT identity, and nudging them would remap every colour in the
/// image — so both buttons must leave it exactly as it was.
#[test]
fn palette_windows_are_left_alone_by_both() {
    let mut stack = load(spiked_u16_stack(W, H, 30_000, OUTLIERS));
    let hists = frame_histograms(&stack);
    // The library builds this flag from the file's ColorMap; setting it here
    // tests the guard itself without needing a palette TIFF in this crate.
    stack.display.palette = true;
    // Deliberately *not* the bounds. On a real palette stack the window already
    // equals its bounds, so a Reset that ignored the guard would coincidentally
    // do nothing and the test would pass whether the guard were there or not.
    // A distinguishable window tests the contract instead of the coincidence.
    stack.display.settings[0].min = -128.5;
    stack.display.settings[0].max = 65_407.5;
    let before: Vec<(f32, f32)> = stack.display.settings.iter().map(|s| (s.min, s.max)).collect();

    auto_contrast(&mut stack, &hists);
    reset_contrast(&mut stack);

    let after: Vec<(f32, f32)> = stack.display.settings.iter().map(|s| (s.min, s.max)).collect();
    assert_eq!(before, after, "a palette window is an identity mapping, not a contrast choice");
}

/// The histogram spans the union of *every* channel's track, so a window fitted
/// on it can land outside the track of the one channel it belongs to. Both
/// handles still have to end up somewhere the slider can draw them.
///
/// Reaching that case needs two channels. On a single-channel stack the shared
/// axis and the channel's own track are the same range, so a fitted window is
/// inside the track by construction and the clamp can never fire. Here channel
/// 0 is dim at t0, which seeds it a narrow track, while channel 1 is bright and
/// widens the shared axis; at t1 channel 0 jumps to channel 1's level, so the
/// window fitted for it lands far past its own end. Clamped, both handles
/// collapse onto the track end, and the empty-window fallback then opens the
/// channel back up rather than leaving it showing nothing.
#[test]
fn a_window_fitted_past_the_track_is_pulled_back_onto_it() {
    let mut stack = load(composite_u16_stack(&[&[1_000, 60_000], &[60_000, 60_000]], W, H));
    assert_eq!(stack.display.settings.len(), 2, "should resolve as two channels");
    let bounds = stack.display.settings[0].bounds;
    assert!(bounds.1 < 60_000.0, "channel 0 track should be too narrow for 60000, got {bounds:?}");

    stack.frame_index = 1;
    let hists = frame_histograms(&stack);
    auto_contrast(&mut stack, &hists);

    let s = &stack.display.settings[0];
    assert!(
        s.min >= bounds.0 && s.max <= bounds.1,
        "handles left the track: {:?} is not inside {bounds:?}",
        (s.min, s.max)
    );
    assert!(s.max > s.min, "the window must not collapse to nothing");
}

/// The low end of the track, clamped on its own.
///
/// The previous test pins the high clamp, but not this one: when a window falls
/// entirely past one end, both handles land on that end, and the empty-window
/// fallback rewrites them both — so dropping either clamp leaves the result
/// unchanged. Only a window that *straddles* the track start separates them.
/// Channel 0 is bright at t0 (a track starting well above zero) and half-dark
/// at t1, so the window fitted for it reaches below its track while its upper
/// handle stays legitimately inside.
#[test]
fn the_low_handle_is_clamped_without_the_window_collapsing() {
    let mut stack = load(split_u16_stack(
        &[&[(50_000, 60_000), (0, 65_535)], &[(0, 55_000), (0, 65_535)]],
        W,
        H,
    ));
    let bounds = stack.display.settings[0].bounds;
    assert!(bounds.0 > 1_000.0, "channel 0 track should start well above zero, got {bounds:?}");

    stack.frame_index = 1;
    let hists = frame_histograms(&stack);
    auto_contrast(&mut stack, &hists);

    let s = &stack.display.settings[0];
    assert!(s.min >= bounds.0, "the low handle fell off the track: {} < {}", s.min, bounds.0);
    assert!(s.max <= bounds.1, "the high handle fell off the track: {} > {}", s.max, bounds.1);
    assert!(
        s.max > s.min && (s.min, s.max) != bounds,
        "this case should clamp one handle, not collapse to the fallback: {:?}",
        (s.min, s.max)
    );
}

/// Auto is handed whatever histograms the caller has cached. If they describe
/// fewer channels than the stack has — a stale cache, or a decode that failed
/// partway — the channels it says nothing about keep the window they had.
#[test]
fn channels_without_a_histogram_keep_their_window() {
    let mut stack = load(spiked_u16_stack(W, H, 30_000, OUTLIERS));
    let before: Vec<(f32, f32)> = stack.display.settings.iter().map(|s| (s.min, s.max)).collect();
    auto_contrast(&mut stack, &[]);
    let after: Vec<(f32, f32)> = stack.display.settings.iter().map(|s| (s.min, s.max)).collect();
    assert_eq!(before, after);
}

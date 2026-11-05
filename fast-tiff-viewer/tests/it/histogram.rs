//! Tests for the per-channel intensity histograms.
//!
//! The property that matters is *alignment*: bin `i` of channel `c` must sit at
//! the same fraction of the plot as the contrast handles do on that channel's
//! slider, because the two are drawn one above the other. That holds only if the
//! bins are laid out over `ChannelSettings::bounds` and 8-bit channels are
//! widened onto the same axis their slider uses — the two things these tests
//! pin down.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::histogram::{fill_alpha, frame_histograms, BINS};
use fast_tiff_viewer::Stack;
use std::io::Cursor;

/// A `frames`-plane u16 stack where plane `f` is filled with the single value
/// `base + f * step`, so every frame's histogram is one known spike.
fn flat_u16_stack(frames: usize, w: u32, h: u32, base: u16, step: u16) -> Vec<u8> {
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(1, frames));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames {
        let v = base + f as u16 * step;
        let bytes: Vec<u8> = (0..w * h).flat_map(|_| v.to_le_bytes()).collect();
        writer.write_frame_bytes(&bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// A composite stack of `values.len()` channels, channel `c` filled flat with
/// `values[c]` — three constants far apart in intensity, so where each channel's
/// spike lands is unambiguous.
fn multichannel_u16_stack(values: &[u16], w: u32, h: u32) -> Vec<u8> {
    let opts =
        WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(values.len(), 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for &v in values {
        let bytes: Vec<u8> = (0..w * h).flat_map(|_| v.to_le_bytes()).collect();
        writer.write_frame_bytes(&bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// One u8 plane holding a horizontal 0..=255 ramp, one column per value.
fn ramp_u8_stack() -> Vec<u8> {
    let (w, h) = (256u32, 4u32);
    let opts = WriterOptions::new(w, h, SampleType::U8).metadata(StackMetaWrite::new(1, 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let px: Vec<u8> = (0..h).flat_map(|_| (0..w).map(|x| x as u8)).collect();
    writer.write_frame_bytes(&px).unwrap();
    writer.finish().unwrap().into_inner()
}

fn load(bytes: Vec<u8>) -> Stack {
    Stack::from_bytes(bytes, "hist.tif".into(), false).expect("should open")
}

#[test]
fn every_sample_lands_in_exactly_one_bin() {
    let stack = load(flat_u16_stack(1, 32, 16, 1000, 0));
    let hists = frame_histograms(&stack);
    assert_eq!(hists.len(), 1);
    let h = &hists[0];
    assert_eq!(h.bins.len(), BINS);
    // Nothing dropped, nothing double-counted.
    assert_eq!(h.bins.iter().map(|&b| b as u64).sum::<u64>(), h.counted);
    assert_eq!(h.counted, 32 * 16);
    // A single-valued frame is one spike and 255 empty bins.
    assert_eq!(h.bins.iter().filter(|&&b| b > 0).count(), 1);
    assert_eq!(h.peak, 32 * 16);
}

#[test]
fn bins_span_the_slider_track_for_a_single_channel_stack() {
    // With one channel the shared track *is* that channel's slider track, so a
    // handle parked at a fraction of the track sits above the same fraction of
    // the plot.
    let stack = load(flat_u16_stack(1, 16, 16, 4000, 0));
    let settings = stack.display.settings[0];
    let h = &frame_histograms(&stack)[0];
    assert_eq!((h.lo, h.hi), settings.bounds);

    // And the spike lands where that mapping says it should.
    let (lo, hi) = settings.bounds;
    let expected = (((4000.0 - lo) / (hi - lo) * BINS as f32) as usize).min(BINS - 1);
    let occupied: Vec<usize> = h
        .bins
        .iter()
        .enumerate()
        .filter(|(_, &b)| b > 0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(occupied, vec![expected]);
}

#[test]
fn eight_bit_channels_are_widened_onto_the_slider_axis() {
    // 8-bit channels keep raw 0..255 samples while their track lives in the
    // widened 0..65535 space. Binning the raw values against that track would
    // pile every pixel of a full-range ramp into bin 0.
    let stack = load(ramp_u8_stack());
    assert_eq!(
        stack.display.settings[0].kind,
        fast_tiff_viewer::ChannelKind::Int8
    );
    let h = &frame_histograms(&stack)[0];
    assert_eq!(h.counted, 256 * 4);
    // A 0..=255 ramp over a 0..=255-derived track fills the plot, not one bin.
    let occupied = h.bins.iter().filter(|&&b| b > 0).count();
    assert!(
        occupied > BINS / 2,
        "ramp collapsed into {occupied} bin(s) — not widened?"
    );
}

#[test]
fn the_histogram_follows_the_current_frame() {
    // Frame 2 is a different constant from frame 0, so the spike must move.
    let mut stack = load(flat_u16_stack(3, 16, 16, 100, 5000));
    let first = frame_histograms(&stack)[0].clone();
    stack.frame_index = 2;
    let third = frame_histograms(&stack)[0].clone();
    let spike =
        |h: &fast_tiff_viewer::Histogram| h.bins.iter().position(|&b| b > 0).expect("a spike");
    assert_ne!(
        spike(&first),
        spike(&third),
        "histogram ignored the frame index"
    );
}

#[test]
fn peak_is_the_tallest_bin() {
    let stack = load(flat_u16_stack(1, 8, 8, 2000, 0));
    let h = &frame_histograms(&stack)[0];
    assert_eq!(h.peak, *h.bins.iter().max().unwrap());
    assert_eq!(h.peak, 8 * 8);
}

#[test]
fn overlay_alpha_thins_as_channels_are_added_but_stays_visible() {
    let a: Vec<u8> = (1..=6).map(fill_alpha).collect();
    assert!(
        a.windows(2).all(|w| w[0] >= w[1]),
        "alpha must not rise with channel count: {a:?}"
    );
    assert!(
        a[0] > a[5],
        "six channels should be thinner than one: {a:?}"
    );
    assert!(a[5] >= 40, "six channels faded to {} — invisible", a[5]);
    assert!(
        a[0] < 160,
        "a single channel at {} is not translucent",
        a[0]
    );
    // Degenerate input must not divide by zero or panic.
    assert!(fill_alpha(0) > 0);
}

#[test]
fn an_empty_stack_histograms_to_nothing_rather_than_panicking() {
    let mut stack = load(flat_u16_stack(1, 8, 8, 1, 0));
    stack.display.settings.clear();
    assert!(frame_histograms(&stack).is_empty());
    // A frame index past the end is a decode error, not a panic.
    let mut stack = load(flat_u16_stack(1, 8, 8, 1, 0));
    stack.frame_index = 999;
    assert!(frame_histograms(&stack).is_empty());
}

#[test]
fn channels_share_one_axis_and_land_at_different_places_on_it() {
    // The regression this guards: binning each channel over *its own* bounds
    // centres every distribution in its own range, so three channels that are
    // nothing alike all plot as the same curve stacked on itself — the exact
    // opposite of what an overlaid composite histogram is read for.
    let stack = load(multichannel_u16_stack(&[4_000, 30_000, 60_000], 32, 32));
    let hists = frame_histograms(&stack);
    assert_eq!(hists.len(), 3, "expected one histogram per channel");

    // One axis for all of them.
    let axis = (hists[0].lo, hists[0].hi);
    assert!(
        hists.iter().all(|h| (h.lo, h.hi) == axis),
        "channels binned on different axes"
    );
    assert_eq!(axis, fast_tiff_viewer::histogram::shared_track(&stack));

    // And on that axis the three spikes are ordered dim -> bright, distinctly.
    let spikes: Vec<usize> = hists
        .iter()
        .map(|h| h.bins.iter().position(|&b| b > 0).expect("a spike"))
        .collect();
    assert!(
        spikes[0] < spikes[1] && spikes[1] < spikes[2],
        "spikes not ordered: {spikes:?}"
    );
    assert!(
        spikes[2] - spikes[0] > BINS / 4,
        "spikes {spikes:?} are bunched together — are they on a shared axis?"
    );
}

#[test]
fn the_shared_track_covers_every_channels_bounds() {
    let stack = load(multichannel_u16_stack(&[4_000, 30_000, 60_000], 16, 16));
    let (lo, hi) = fast_tiff_viewer::histogram::shared_track(&stack);
    for s in &stack.display.settings {
        assert!(
            s.bounds.0 >= lo && s.bounds.1 <= hi,
            "{:?} escapes the track {lo}..{hi}",
            s.bounds
        );
    }
    // Nothing falls off either end of that track.
    for h in frame_histograms(&stack) {
        assert_eq!(h.bins.iter().map(|&b| b as u64).sum::<u64>(), h.counted);
    }
}

#[test]
fn fill_tint_stays_visible_for_luts_with_a_dark_top_entry() {
    use fast_tiff_viewer::histogram::fill_tint;
    let bright = |c: [u8; 3]| c.iter().copied().max().unwrap();

    // A contrast-stretched palette: a ramp that blacks out its unused tail. The
    // top entry is black, so taking it would paint an invisible histogram.
    let mut stretched = [[0u8; 3]; 256];
    for (i, e) in stretched.iter_mut().enumerate().take(140) {
        let v = (i * 255 / 139) as u8;
        *e = [v, v, v];
    }
    assert_eq!(
        stretched[255],
        [0, 0, 0],
        "fixture should have a black top entry"
    );
    assert!(
        bright(fill_tint(&stretched)) >= 64,
        "picked an invisible colour"
    );

    // An all-black table is legal and must still yield something drawable.
    assert!(bright(fill_tint(&[[0, 0, 0]; 256])) >= 64);

    // Ordinary ramps are unaffected: a channel keeps its own colour.
    let mut red = [[0u8; 3]; 256];
    for (i, e) in red.iter_mut().enumerate() {
        *e = [i as u8, 0, 0];
    }
    assert_eq!(fill_tint(&red), [255, 0, 0]);
    let mut grey = [[0u8; 3]; 256];
    for (i, e) in grey.iter_mut().enumerate() {
        *e = [i as u8; 3];
    }
    assert_eq!(fill_tint(&grey), [255, 255, 255]);
}

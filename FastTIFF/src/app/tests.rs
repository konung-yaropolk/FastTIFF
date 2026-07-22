//! Unit tests for `app` (see fast-tiff-lib's `*_tests.rs` convention).

use super::dimensions::rgb_channel_plan;
use super::DecodeMode;
use crate::render::MAX_CHANNELS;

#[test]
fn decode_mode_drives_parallel_flag() {
    // Serial is always off and Threaded always on, regardless of whether
    // Auto's "falling behind" latch happens to be set.
    assert!(!DecodeMode::Serial.parallel(false));
    assert!(!DecodeMode::Serial.parallel(true));
    assert!(DecodeMode::Threaded.parallel(false));
    assert!(DecodeMode::Threaded.parallel(true));
    // Auto follows the latch: serial until playback falls behind, then parallel.
    assert!(!DecodeMode::Auto.parallel(false));
    assert!(DecodeMode::Auto.parallel(true));
}

#[test]
fn rgb_extra_samples_get_channels_but_start_disabled() {
    // Plain RGB: three channels, all on.
    assert_eq!(rgb_channel_plan(3), vec![true, true, true]);

    // RGBA (what tifffile writes for a (4, H, W) array): the fourth sample is
    // reachable as a channel — the regression this guards — but starts off, so
    // an opaque alpha plane can't wash out a genuine RGBA image on open.
    assert_eq!(rgb_channel_plan(4), vec![true, true, true, false]);

    // More extras stay individually addressable...
    assert_eq!(rgb_channel_plan(5), vec![true, true, true, false, false]);
    // ...up to the shader's channel limit, past which samples are dropped.
    let many = rgb_channel_plan(MAX_CHANNELS + 3);
    assert_eq!(many.len(), MAX_CHANNELS);
    assert_eq!(many.iter().filter(|&&on| on).count(), 3, "only R/G/B start on");
}

//! The RGB channel plan: which sample planes of an RGB frame get a display
//! channel, and which of them start on.

use super::*;

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
    assert_eq!(
        many.iter().filter(|&&on| on).count(),
        3,
        "only R/G/B start on"
    );
}

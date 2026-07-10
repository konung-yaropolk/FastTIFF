//! Unit tests for `app` (see fast-tiff-lib's `*_tests.rs` convention).

use super::DecodeMode;

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

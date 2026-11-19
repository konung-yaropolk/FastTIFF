//! The desktop's own arithmetic.

use super::due;
use std::sync::atomic::AtomicU64;

/// Progress wakes the interface at most every 100 ms: the first report after
/// that long is due, the ones inside the next 100 ms are not, and only one of
/// several reports at the same instant claims the wake.
#[test]
fn progress_wakes_are_spaced() {
    let last = AtomicU64::new(0);
    assert!(!due(&last, 50, 100), "not yet 100 ms in");
    assert!(due(&last, 100, 100));
    assert!(!due(&last, 100, 100), "the same instant claims it once");
    assert!(!due(&last, 199, 100));
    assert!(due(&last, 200, 100));
    assert!(
        !due(&last, 150, 100),
        "a clock read late does not wake again"
    );
}

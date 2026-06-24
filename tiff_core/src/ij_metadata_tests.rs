use super::*;

fn check(c: usize, z: usize, f: usize, expected: ResolvedDimensions, label: &str) {
    let got = resolve_dimensions(c, z, f);
    assert_eq!(got, expected, "{label}: resolve_dimensions({c}, {z}, {f})");
    // Invariant: reclassifying axes must never invent or drop planes.
    assert_eq!(
        got.channels * got.slices * got.frames,
        c * z * f,
        "{label}: product changed"
    );
}

#[test]
fn fixes_mislabeled_large_channel_count() {
    // The actual reported bug: metadata says channels=100 (z=1, f=1).
    check(
        100,
        1,
        1,
        ResolvedDimensions { channels: 1, slices: 1, frames: 100, triple_axis_warning: false },
        "mislabeled channels=100",
    );
    check(
        100,
        1,
        7,
        ResolvedDimensions { channels: 1, slices: 1, frames: 700, triple_axis_warning: false },
        "mislabeled channels=100, real frames also present",
    );
}

#[test]
fn leaves_normal_stacks_untouched() {
    check(
        2,
        1,
        350,
        ResolvedDimensions { channels: 2, slices: 1, frames: 350, triple_axis_warning: false },
        "normal 2-channel timelapse",
    );
    check(
        1,
        1,
        500,
        ResolvedDimensions { channels: 1, slices: 1, frames: 500, triple_axis_warning: false },
        "normal single-channel timelapse",
    );
    check(
        1,
        1,
        1,
        ResolvedDimensions { channels: 1, slices: 1, frames: 1, triple_axis_warning: false },
        "single image",
    );
}

#[test]
fn folds_z_into_time() {
    check(
        1,
        50,
        1,
        ResolvedDimensions { channels: 1, slices: 1, frames: 50, triple_axis_warning: false },
        "pure z-stack becomes a 50-frame series",
    );
    check(
        2,
        3,
        1,
        ResolvedDimensions { channels: 2, slices: 1, frames: 3, triple_axis_warning: false },
        "2-channel z-stack: z folds into frames",
    );
}

#[test]
fn detects_swapped_channel_and_time_roles() {
    // Small value mislabeled as frames, large value mislabeled as channels.
    check(
        500,
        1,
        2,
        ResolvedDimensions { channels: 2, slices: 1, frames: 500, triple_axis_warning: false },
        "swapped roles recovered",
    );
}

#[test]
fn warns_on_genuine_triple_axis_stack() {
    check(
        3,
        10,
        20,
        ResolvedDimensions { channels: 3, slices: 10, frames: 20, triple_axis_warning: true },
        "channels + Z + time all present",
    );
}

#[test]
fn channel_size_boundary_is_inclusive_at_cutoff() {
    check(
        6,
        1,
        100,
        ResolvedDimensions { channels: 6, slices: 1, frames: 100, triple_axis_warning: false },
        "exactly at the cutoff (6) counts as channel-sized",
    );
    check(
        7,
        1,
        100,
        ResolvedDimensions { channels: 1, slices: 1, frames: 700, triple_axis_warning: false },
        "one past the cutoff does not count as channel-sized",
    );
}

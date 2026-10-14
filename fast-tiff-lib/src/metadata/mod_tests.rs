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
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 100,
            triple_axis_warning: false,
        },
        "mislabeled channels=100",
    );
    check(
        100,
        1,
        7,
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 700,
            triple_axis_warning: false,
        },
        "mislabeled channels=100, real frames also present",
    );
}

#[test]
fn leaves_normal_stacks_untouched() {
    check(
        2,
        1,
        350,
        ResolvedDimensions {
            channels: 2,
            slices: 1,
            frames: 350,
            triple_axis_warning: false,
        },
        "normal 2-channel timelapse",
    );
    check(
        1,
        1,
        1,
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 1,
            triple_axis_warning: false,
        },
        "single image",
    );
}

#[test]
fn folds_z_into_time() {
    check(
        1,
        50,
        1,
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 50,
            triple_axis_warning: false,
        },
        "pure z-stack becomes a 50-frame series",
    );
    check(
        2,
        3,
        1,
        ResolvedDimensions {
            channels: 2,
            slices: 1,
            frames: 3,
            triple_axis_warning: false,
        },
        "2-channel z-stack: z folds into frames",
    );
}

#[test]
fn detects_swapped_channel_and_time_roles() {
    check(
        500,
        1,
        2,
        ResolvedDimensions {
            channels: 2,
            slices: 1,
            frames: 500,
            triple_axis_warning: false,
        },
        "swapped roles recovered",
    );
}

#[test]
fn warns_on_genuine_triple_axis_stack() {
    check(
        3,
        10,
        20,
        ResolvedDimensions {
            channels: 3,
            slices: 10,
            frames: 20,
            triple_axis_warning: true,
        },
        "channels + Z + time all present",
    );
}

#[test]
fn channel_size_boundary_is_inclusive_at_cutoff() {
    check(
        6,
        1,
        100,
        ResolvedDimensions {
            channels: 6,
            slices: 1,
            frames: 100,
            triple_axis_warning: false,
        },
        "exactly at the cutoff (6) counts as channel-sized",
    );
    check(
        7,
        1,
        100,
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 700,
            triple_axis_warning: false,
        },
        "one past the cutoff does not count as channel-sized",
    );
}

#[test]
fn voxel_scale_reports_raw_calibration() {
    // A neutral (no-description) parse gives an uncalibrated 1:1:1 stack.
    let bare = parse(None, None, None, 1, None, None);
    assert_eq!(bare.source_format, MetadataFormat::None);
    assert_eq!(bare.voxel_scale(), [1.0, 1.0, 1.0]);

    // Filled in: 0.1 µm pixels, 0.5 µm z-step → the raw values (not normalized).
    let mut meta = bare;
    meta.pixel_width = Some(0.1);
    meta.pixel_height = Some(0.1);
    meta.spacing = Some(0.5);
    assert_eq!(meta.voxel_scale(), [0.1, 0.1, 0.5]);
}

#[test]
fn detect_classifies_each_dialect() {
    assert_eq!(detect(None), MetadataFormat::None);
    assert_eq!(
        detect(Some("ImageJ=1.54f\nchannels=2\n")),
        MetadataFormat::ImageJ
    );
    assert_eq!(
        detect(Some("<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\"/>")),
        MetadataFormat::Ome
    );
    // Free-form text in someone else's TIFF isn't mistaken for a known dialect.
    assert_eq!(
        detect(Some("Made with SomeMicroscope v3")),
        MetadataFormat::None
    );
}

#[test]
fn colormap_to_lut_scales_and_validates() {
    // 16-bit-scaled colormap (the TIFF6 spec form): planar R[256], G[256], B[256].
    let mut cm = vec![0u32; 768];
    cm[0] = 0xFFFF; // R[0] = full
    cm[256 + 1] = 0x8000; // G[1] = half
    cm[512 + 2] = 0x0100; // B[2] = 1 at 8-bit
    let lut = colormap_to_lut(&cm).expect("256-entry map");
    assert_eq!(lut[0], [255, 0, 0]); // 0xFFFF >> 8 = 255
    assert_eq!(lut[1], [0, 128, 0]); // 0x8000 >> 8 = 128
    assert_eq!(lut[2], [0, 0, 1]); // 0x0100 >> 8 = 1

    // 8-bit-scaled colormap (values never exceed 255): taken as-is, not shifted.
    let mut cm8 = vec![0u32; 768];
    cm8[0] = 200;
    cm8[256] = 100;
    cm8[512] = 50;
    let lut8 = colormap_to_lut(&cm8).unwrap();
    assert_eq!(lut8[0], [200, 100, 50]);

    // A 4-bit (16-entry) map: entries fill LUT slots 0..16, the tail repeats
    // the last entry (index 15 here is [10,20,30]).
    let mut cm4 = vec![0u32; 48];
    cm4[0] = 200; // R[0]
    cm4[16] = 100; // G[0]
    cm4[32] = 50; // B[0]
    cm4[15] = 10;
    cm4[16 + 15] = 20;
    cm4[32 + 15] = 30;
    let lut4 = colormap_to_lut(&cm4).unwrap();
    assert_eq!(lut4[0], [200, 100, 50]);
    assert_eq!(lut4[15], [10, 20, 30]);
    assert_eq!(lut4[255], [10, 20, 30], "tail repeats the last real entry");

    // Genuinely invalid: not a multiple of 3, or too large.
    assert!(colormap_to_lut(&[0u32; 49]).is_none());
    assert!(colormap_to_lut(&[0u32; 3 * 300]).is_none());
    assert!(colormap_to_lut(&[]).is_none());
}

#[test]
fn neutral_parse_still_honors_resolution_tags() {
    // A plain TIFF (no recognized description) with resolution tags reports
    // pixel size, so calibration survives even without a metadata dialect.
    let meta = parse(None, None, None, 4, Some(10.0), Some(10.0));
    assert_eq!(meta.source_format, MetadataFormat::None);
    assert_eq!(meta.pixel_width, Some(0.1)); // 1 / 10 px-per-unit
    assert_eq!(meta.pixel_height, Some(0.1));
    assert_eq!(meta.frames, 4); // inferred from the IFD count
}

/// A single-channel 4-D stack keeps all three axes.
///
/// This is the commonest 4-D shape there is — one channel, tens of slices, tens
/// of timepoints, which is what an fMRI or an ASL series looks like — and it
/// used to be folded into one long movie because the "keep all three" branch
/// also demanded more than one channel. Two things went wrong at once: the 3-D
/// view built its depth from slices × frames and showed every timepoint's
/// volume stacked end to end in a row, and the scrubber stepped through slices
/// and timepoints interleaved with no way to hold one still.
///
/// There is nothing to trade here. One channel leaves the channel slot free, so
/// all three numbers fit and no warning is needed.
#[test]
fn a_single_channel_four_dimensional_stack_keeps_its_slice_axis() {
    check(
        1,
        20,
        10,
        ResolvedDimensions {
            channels: 1,
            slices: 20,
            frames: 10,
            triple_axis_warning: false,
        },
        "an ASL series: 1 channel x 20 slices x 10 timepoints",
    );
    // Two slices and two frames is the smallest case that is genuinely 4-D,
    // and it must not be folded either.
    check(
        1,
        2,
        2,
        ResolvedDimensions {
            channels: 1,
            slices: 2,
            frames: 2,
            triple_axis_warning: false,
        },
        "the smallest real 4-D stack",
    );
}

/// With several channels there is nothing left to put Z in, so the warning
/// stands — the viewer shows one Z-slice and says so.
#[test]
fn a_multi_channel_four_dimensional_stack_still_warns() {
    check(
        3,
        20,
        10,
        ResolvedDimensions {
            channels: 3,
            slices: 20,
            frames: 10,
            triple_axis_warning: true,
        },
        "3 channels x 20 slices x 10 timepoints",
    );
}

/// A single timepoint is still folded, which is the long-standing behaviour and
/// deliberate: a one-timepoint z-stack and a movie are indistinguishable from
/// the plane count alone, and the movie reading is right far more often. The
/// 3-D view does not care — with one slice it builds its depth from the frame
/// axis instead — so nothing is lost by it.
#[test]
fn a_z_stack_with_one_timepoint_is_still_read_as_a_movie() {
    check(
        1,
        190,
        1,
        ResolvedDimensions {
            channels: 1,
            slices: 1,
            frames: 190,
            triple_axis_warning: false,
        },
        "a structural scan",
    );
}

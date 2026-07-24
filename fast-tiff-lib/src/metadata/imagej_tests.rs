use super::*;

#[test]
fn decodes_imagej_unit_escapes() {
    // ImageJ writes the micron unit as a literal Java \uXXXX escape.
    assert_eq!(decode_ij_escapes("\\u00B5m"), "µm");
    assert_eq!(decode_ij_escapes("um"), "um"); // plain ASCII untouched
    assert_eq!(decode_ij_escapes("pixel"), "pixel");
    // A malformed escape is left verbatim rather than dropped.
    assert_eq!(decode_ij_escapes("\\uZZ"), "\\uZZ");
    assert_eq!(decode_ij_escapes("\\u12"), "\\u12");
}

#[test]
fn parses_hyperstack_dimensions_and_calibration() {
    let desc = "ImageJ=1.54f\nimages=6\nchannels=2\nframes=3\nmode=composite\n\
                unit=micron\nfinterval=1.5\ncf=0\nc0=100\nc1=2\n";
    let meta = parse(Some(desc), None, None, 6, None, None);
    assert_eq!(meta.source_format, MetadataFormat::ImageJ);
    assert_eq!((meta.channels, meta.slices, meta.frames), (2, 1, 3));
    assert_eq!(meta.mode, DisplayMode::Composite);
    assert_eq!(meta.unit.as_deref(), Some("micron"));
    assert_eq!(meta.frame_interval_s, Some(1.5));
    assert_eq!(meta.calibration, Some((100.0, 2.0)));
}

#[test]
fn serialize_round_trips_through_parse() {
    // The neutral write builder → ImageJ text → parse back to the same values.
    let write = StackMetaWrite::new(2, 1)
        .mode(DisplayMode::Composite)
        .unit("micron")
        .fps(12.5)
        .range(10.0, 200.0)
        .calibration(5.0, 0.5);
    let desc = serialize(6, &write).unwrap(); // 6 planes = 2 channels x 3 frames
    let meta = parse(Some(&desc), None, None, 6, None, None);

    assert_eq!((meta.channels, meta.slices, meta.frames), (2, 1, 3));
    assert_eq!(meta.mode, DisplayMode::Composite);
    assert_eq!(meta.unit.as_deref(), Some("micron"));
    assert_eq!(meta.fps, Some(12.5));
    assert_eq!(meta.channel_display[0].range, Some((10.0, 200.0)));
    assert_eq!(meta.calibration, Some((5.0, 0.5)));
}

#[test]
fn serialize_rejects_indivisible_plane_count() {
    // 5 planes can't split into 2 channels evenly.
    let write = StackMetaWrite::new(2, 1);
    assert!(serialize(5, &write).is_err());
}

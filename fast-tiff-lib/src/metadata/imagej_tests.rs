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

#[test]
fn ij_metadata_luts_round_trip_and_carry_the_magic() {
    // Two distinct per-channel LUTs: a red ramp and a green ramp.
    let mut red = [[0u8; 3]; 256];
    let mut green = [[0u8; 3]; 256];
    for i in 0..256 {
        red[i] = [i as u8, 0, 0];
        green[i] = [0, i as u8, 0];
    }
    let (blob, counts) = serialize_ij_metadata(&[red, green]).expect("two LUTs → a block");

    // The blob must start with the (little-endian) ImageJ magic, and the
    // byte-count layout is [header(12), 768, 768].
    assert_eq!(&blob[..4], b"JIJI");
    assert_eq!(counts, vec![12, 768, 768]);

    // …and parse straight back to the same LUTs.
    let blocks = try_parse_ij_blocks(&blob, &counts).expect("our own block must parse");
    assert_eq!(blocks.luts.len(), 2);
    assert_eq!(blocks.luts[0][255], [255, 0, 0]);
    assert_eq!(blocks.luts[1][255], [0, 255, 0]);

    // No LUTs → no block.
    assert!(serialize_ij_metadata(&[]).is_none());
}

#[test]
fn parser_requires_the_magic() {
    // A magic-less header (the shape an earlier version wrongly assumed) is
    // rejected, so we can't silently misread non-ImageJ bytes.
    let bogus = vec![b'r', b'a', b'n', b'g', 0, 0, 0, 1, 1, 2, 3, 4];
    assert!(try_parse_ij_blocks(&bogus, &[8, 4]).is_none());
}

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

/// A carried-over description cannot redescribe the file it is carried into.
///
/// The real case, with the real numbers. Inverting a 101-slice ImageJ z-stack
/// gives a 101-*frame* result: the writer states `frames=101` and says nothing
/// about slices, because there is one. If the source's own `slices=101` rides
/// along in the trailing text it becomes the only `slices=` in the description
/// — "first occurrence wins" cannot help, there being no competing occurrence
/// — and the file reads back as 101 slices x 101 frames. 10,201 planes
/// declared against 101 present, reported on every window as a damaged file.
#[test]
fn a_carried_description_cannot_redescribe_the_shape() {
    let source = "ImageJ=1.54p\nimages=101\nslices=101\nunit=micron\nspacing=0.2\n\
                  loop=false\nmin=0.0\nmax=4095.0\n";
    let write = StackMetaWrite::new(1, 1)
        .unit("micron")
        .trailing(source.to_string());
    let desc = serialize(101, &write).unwrap();

    // The writer's own account of the shape, and only that.
    assert!(desc.contains("frames=101"), "{desc}");
    assert_eq!(desc.matches("slices=").count(), 0, "{desc}");
    assert_eq!(
        desc.matches("ImageJ=").count(),
        1,
        "two version markers: {desc}"
    );

    let meta = parse(Some(&desc), None, None, 101, None, None);
    assert_eq!(
        (meta.channels, meta.slices, meta.frames),
        (1, 1, 101),
        "the source's shape leaked into the result"
    );
    // The structured block still carries what it was told.
    assert_eq!(meta.unit.as_deref(), Some("micron"));
}

/// What `trailing` exists for still survives: a vendor record carries no
/// `key=value` lines of this dialect's, so none of it is dropped.
///
/// Olympus OIR's record is `"key"\t"value"`, which is what the importer writes
/// and what an analysis downstream reads to find out when the stimulus fired.
#[test]
fn a_vendor_record_passes_through_untouched() {
    let record = "\"[General]\"\t\"\"\n\"Name\"\t\"a.oir\"\n\"Scan Mode\"\t\"XY\"\n\
                  \"Image Size\"\t\"512 * 512 [pixel]\"\n";
    let write = StackMetaWrite::new(1, 1).trailing(record.to_string());
    let desc = serialize(4, &write).unwrap();
    for line in record.lines() {
        assert!(desc.contains(line), "dropped {line:?} from {desc}");
    }
}

/// And a line that merely *contains* an `=` inside a value is not mistaken for
/// one of this dialect's keys.
#[test]
fn an_equals_inside_a_vendor_value_is_not_a_key() {
    let record = "\"Comment\"\t\"gain=2, mode=fast\"\nfree text with = in it\n";
    let write = StackMetaWrite::new(1, 1).trailing(record.to_string());
    let desc = serialize(4, &write).unwrap();
    assert!(
        desc.contains("\"Comment\"\t\"gain=2, mode=fast\""),
        "{desc}"
    );
    assert!(desc.contains("free text with = in it"), "{desc}");
}

/// The keys the writer *does* emit were already protected, and still are.
#[test]
fn a_carried_description_cannot_override_what_the_writer_stated() {
    let source = "ImageJ=1.50a\nmode=grayscale\nunit=inch\nmin=7.0\nmax=9.0\n";
    let write = StackMetaWrite::new(2, 1)
        .mode(DisplayMode::Composite)
        .unit("micron")
        .range(10.0, 200.0)
        .trailing(source.to_string());
    let desc = serialize(6, &write).unwrap();
    let meta = parse(Some(&desc), None, None, 6, None, None);

    assert_eq!(meta.mode, DisplayMode::Composite);
    assert_eq!(meta.unit.as_deref(), Some("micron"));
    assert_eq!(meta.channel_display[0].range, Some((10.0, 200.0)));
}

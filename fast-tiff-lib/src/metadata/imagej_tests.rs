use super::*;
use crate::metadata::roi::{PointRoi, RoiPoint};

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
    let (blob, counts) = serialize_ij_metadata(&[red, green], &[]).expect("two LUTs → a block");

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
    assert!(serialize_ij_metadata(&[], &[]).is_none());
}

/// Two overlay ROIs, as the point counter hands them over: one on two slices,
/// one on a single slice. Mid-pixel, so they read back exactly.
fn overlay_rois() -> Vec<PointRoi> {
    let point = |x: f32, position: u32| RoiPoint {
        x,
        y: 2.5,
        position,
    };
    vec![
        PointRoi {
            hyperstack_position: [0; 3],
            name: "Group 1".into(),
            color: [0xe4, 0x1a, 0x1c],
            points: vec![point(1.5, 1), point(3.5, 2)],
        },
        PointRoi {
            hyperstack_position: [0; 3],
            name: "Group 2".into(),
            color: [0x37, 0x7e, 0xb8],
            points: vec![point(5.5, 1)],
        },
    ]
}

/// Two LUT ramps and the two overlay ROIs encoded, the shape the viewer's save
/// writes.
fn luts_and_rois() -> (Vec<[[u8; 3]; 256]>, Vec<Vec<u8>>) {
    let mut red = [[0u8; 3]; 256];
    let mut green = [[0u8; 3]; 256];
    for i in 0..256 {
        red[i] = [i as u8, 0, 0];
        green[i] = [0, i as u8, 0];
    }
    (
        vec![red, green],
        overlay_rois().iter().map(|r| r.to_imagej_bytes()).collect(),
    )
}

/// LUTs and an overlay in one block: both types in the directory, LUTs first
/// (ImageJ's order), byte counts `[header, 768, 768, roi, roi]`, and every ROI
/// body verbatim after the LUTs.
#[test]
fn ij_metadata_carries_luts_then_the_overlay() {
    let (luts, rois) = luts_and_rois();
    let (blob, counts) = serialize_ij_metadata(&luts, &rois).expect("a block");

    // "JIJI", then ("luts", 2) and ("over", 2), codes byte-reversed and
    // counts little-endian as the file is.
    let mut header = b"JIJI".to_vec();
    header.extend_from_slice(b"stul");
    header.extend_from_slice(&2u32.to_le_bytes());
    header.extend_from_slice(b"revo");
    header.extend_from_slice(&2u32.to_le_bytes());
    assert_eq!(&blob[..20], &header[..]);
    assert_eq!(
        counts,
        vec![20, 768, 768, rois[0].len() as u32, rois[1].len() as u32]
    );
    assert_eq!(blob.len() as u32, counts.iter().sum::<u32>());

    // The bodies, in order: the red ramp's R plane, …, then the ROIs as-is.
    assert_eq!(blob[20 + 255], 255, "first LUT's red at 255");
    let rois_at = 20 + 2 * 768;
    assert_eq!(&blob[rois_at..rois_at + rois[0].len()], &rois[0][..]);
    assert_eq!(&blob[rois_at + rois[0].len()..], &rois[1][..]);
    assert_eq!(&blob[rois_at..rois_at + 4], b"Iout");
}

/// A block with LUTs and an overlay, as this crate's writer makes one, gives
/// back both: the LUTs, and the overlay's points in order.
#[test]
fn a_block_with_an_overlay_yields_its_luts_and_its_points() {
    let (luts, rois) = luts_and_rois();
    let (blob, counts) = serialize_ij_metadata(&luts, &rois).unwrap();
    let blocks = try_parse_ij_blocks(&blob, &counts).expect("luts + over must parse");
    assert_eq!(blocks.luts.len(), 2);
    assert_eq!(blocks.luts[0][255], [255, 0, 0]);
    assert_eq!(blocks.luts[1][255], [0, 255, 0]);
    assert_eq!(blocks.overlay, overlay_rois());
    assert!(blocks.ranges.is_none());

    // And through the public parse, as a file's first IFD would be read.
    let desc = "ImageJ=1.54f\nimages=2\nchannels=2\nmode=composite\n";
    let meta = parse(Some(desc), Some(&blob), Some(&counts), 2, None, None);
    assert!(meta.has_explicit_luts);
    assert_eq!(meta.channel_display[1].lut[255], [0, 255, 0]);
    assert_eq!(meta.overlay, overlay_rois());

    // LUTs that do not match the channel count are stale and not applied —
    // but the overlay is about the file's IFDs, not its channels, and stays.
    let desc = "ImageJ=1.54f\nimages=3\nchannels=3\nmode=composite\n";
    let meta = parse(Some(desc), Some(&blob), Some(&counts), 3, None, None);
    assert!(!meta.has_explicit_luts);
    assert_eq!(meta.overlay, overlay_rois());
}

/// An overlay alone is a block of its own: a 12-byte header naming only
/// `over`. It carries no display information, and is read for its points.
#[test]
fn an_overlay_without_luts_is_a_block_of_its_own() {
    let (_, rois) = luts_and_rois();
    let (blob, counts) = serialize_ij_metadata(&[], &rois).expect("an overlay → a block");
    assert_eq!(&blob[..12], b"JIJIrevo\x02\0\0\0");
    assert_eq!(counts, vec![12, rois[0].len() as u32, rois[1].len() as u32]);
    let blocks = try_parse_ij_blocks(&blob, &counts).expect("an overlay alone must parse");
    assert!(blocks.luts.is_empty() && blocks.ranges.is_none());
    assert_eq!(blocks.overlay, overlay_rois());
    let desc = "ImageJ=1.54f\nimages=1\n";
    let meta = parse(Some(desc), Some(&blob), Some(&counts), 1, None, None);
    assert!(!meta.has_explicit_luts);
    assert_eq!(meta.channels, 1);
    assert_eq!(meta.overlay, overlay_rois());
}

/// A block in ImageJ's own byte order, as its `TiffEncoder` writes a grey
/// stack with an overlay: `IJIJ`, the type codes as they read, counts
/// big-endian, an `info` entry and the `over` ones. The `.roi` bodies are
/// big-endian in either order, so the points are the same.
#[test]
fn a_big_endian_block_with_info_and_an_overlay_is_read() {
    let rois: Vec<Vec<u8>> = overlay_rois().iter().map(|r| r.to_imagej_bytes()).collect();
    let info: Vec<u8> = "Acquired by hand"
        .encode_utf16()
        .flat_map(|u| u.to_be_bytes())
        .collect();
    let mut blob = b"IJIJ".to_vec();
    blob.extend_from_slice(b"info");
    blob.extend_from_slice(&1u32.to_be_bytes());
    blob.extend_from_slice(b"over");
    blob.extend_from_slice(&2u32.to_be_bytes());
    let counts = vec![
        20,
        info.len() as u32,
        rois[0].len() as u32,
        rois[1].len() as u32,
    ];
    blob.extend_from_slice(&info);
    blob.extend_from_slice(&rois[0]);
    blob.extend_from_slice(&rois[1]);

    let blocks = try_parse_ij_blocks(&blob, &counts).expect("info + over must parse");
    assert_eq!(blocks.overlay, overlay_rois());
    let desc = "ImageJ=1.54f\nimages=4\nslices=4\n";
    let meta = parse(Some(desc), Some(&blob), Some(&counts), 4, None, None);
    assert_eq!(meta.overlay, overlay_rois());
    assert_eq!(meta.slices, 4);

    // The same block spelt little-endian, as tifffile writes one.
    let mut le = b"JIJI".to_vec();
    le.extend_from_slice(b"ofni");
    le.extend_from_slice(&1u32.to_le_bytes());
    le.extend_from_slice(b"revo");
    le.extend_from_slice(&2u32.to_le_bytes());
    le.extend_from_slice(&blob[20..]);
    let blocks = try_parse_ij_blocks(&le, &counts).expect("the little-endian spelling");
    assert_eq!(blocks.overlay, overlay_rois());
}

/// An `over` entry that is not a point ROI it can read — garbage, a cut-off
/// one, a rectangle, an empty one — adds nothing, and the entries around it
/// are still read: the byte counts say where each starts, not the entry.
#[test]
fn an_unreadable_overlay_entry_is_skipped_and_the_rest_read() {
    let (luts, rois) = luts_and_rois();
    let mut rectangle = rois[0].clone();
    rectangle[6] = 1; // type 1: a rectangle
    let cut_off = rois[1][..rois[1].len() - 3].to_vec();
    let entries = vec![
        rois[0].clone(),
        b"not a ROI at all".to_vec(),
        rectangle,
        cut_off,
        Vec::new(),
        rois[1].clone(),
    ];
    let (blob, counts) = serialize_ij_metadata(&luts, &entries).unwrap();
    let blocks = try_parse_ij_blocks(&blob, &counts).expect("the block is sound");
    assert_eq!(blocks.luts.len(), 2, "the LUTs before them are read");
    assert_eq!(
        blocks.overlay,
        overlay_rois(),
        "the two good ones, in order"
    );

    // Nothing readable at all, and nothing else: no block worth having.
    let (blob, counts) = serialize_ij_metadata(&[], &[b"junk".to_vec()]).unwrap();
    assert!(try_parse_ij_blocks(&blob, &counts).is_none());
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

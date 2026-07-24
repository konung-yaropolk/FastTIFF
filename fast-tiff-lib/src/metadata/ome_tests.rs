use super::*;
use crate::metadata::{DisplayMode, MetadataFormat, StackMetaWrite, WriteGeometry};

/// A trimmed but realistic OME-XML block of the shape tifffile / Bio-Formats
/// emit — namespaced root, one Image/Pixels, two colored channels.
const SAMPLE_OME: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     UUID="urn:uuid:2f1e9b7a-0000-4000-8000-000000000001">
  <Image ID="Image:0" Name="demo">
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16"
            SizeX="512" SizeY="256" SizeC="2" SizeZ="1" SizeT="10"
            PhysicalSizeX="0.1" PhysicalSizeXUnit="µm"
            PhysicalSizeY="0.1" PhysicalSizeYUnit="µm"
            TimeIncrement="1.5" TimeIncrementUnit="s">
      <Channel ID="Channel:0:0" Name="DAPI" Color="-16776961" SamplesPerPixel="1"/>
      <Channel ID="Channel:0:1" Name="GFP" Color="16711935" SamplesPerPixel="1"/>
      <TiffData/>
    </Pixels>
  </Image>
</OME>"#;

#[test]
fn parses_pixels_core_and_channels() {
    let meta = parse(SAMPLE_OME, 20, None, None).expect("should parse OME-XML");
    assert_eq!(meta.source_format, MetadataFormat::Ome);
    assert_eq!((meta.channels, meta.slices, meta.frames), (2, 1, 10));
    assert_eq!(meta.pixel_width, Some(0.1));
    assert_eq!(meta.pixel_height, Some(0.1));
    assert_eq!(meta.unit.as_deref(), Some("µm"));
    assert_eq!(meta.frame_interval_s, Some(1.5));
    // Colored channels → composite, and the per-channel LUTs are kept.
    assert_eq!(meta.mode, DisplayMode::Composite);
    assert!(meta.has_explicit_luts);
    assert_eq!(meta.channel_display.len(), 2);
    // Color="16711935" = 0x00FF00FF → green channel: LUT tops out green-dominant.
    let top = meta.channel_display[1].lut[255];
    assert!(top[1] > top[0] && top[1] > top[2], "GFP channel should ramp toward green, got {top:?}");
}

#[test]
fn missing_sizes_default_and_infer_from_ifds() {
    let xml = r#"<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image><Pixels Type="uint8" SizeX="4" SizeY="4" SizeC="3"><TiffData/></Pixels></Image></OME>"#;
    // SizeZ/SizeT absent: Z defaults to 1, T inferred from the IFD count / (C*Z).
    let meta = parse(xml, 12, None, None).expect("parse");
    assert_eq!((meta.channels, meta.slices, meta.frames), (3, 1, 4));
}

#[test]
fn falls_back_to_resolution_tags_when_no_physical_size() {
    let xml = r#"<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image><Pixels Type="uint16" SizeX="4" SizeY="4" SizeC="1"><TiffData/></Pixels></Image></OME>"#;
    let meta = parse(xml, 1, Some(4.0), Some(4.0)).expect("parse");
    assert_eq!(meta.pixel_width, Some(0.25)); // 1 / 4 px-per-unit
}

#[test]
fn malformed_xml_returns_none() {
    assert!(parse("<OME><Image><Pixels", 1, None, None).is_none()); // truncated
    assert!(parse("<OME></OME>", 1, None, None).is_none()); // no Pixels
}

#[test]
fn serialize_round_trips_through_parse() {
    let write = StackMetaWrite::new(2, 3)
        .mode(DisplayMode::Composite)
        .unit("µm")
        .pixel_size(0.2, 0.2)
        .spacing(0.5)
        .frame_interval_s(2.0)
        .channel("DAPI", [0, 0, 255])
        .channel("GFP", [0, 255, 0]);
    let geom = WriteGeometry { width: 512, height: 512, samples_per_pixel: 1, ome_pixel_type: "uint16" };

    // 12 planes = 2 channels x 3 slices x 2 frames.
    let xml = serialize(12, &write, &geom).unwrap();
    assert_eq!(crate::metadata::detect(Some(&xml)), MetadataFormat::Ome, "output must be detected as OME");

    let meta = parse(&xml, 12, None, None).expect("round-trip parse");
    assert_eq!((meta.channels, meta.slices, meta.frames), (2, 3, 2));
    assert_eq!(meta.pixel_width, Some(0.2));
    assert_eq!(meta.spacing, Some(0.5));
    assert_eq!(meta.unit.as_deref(), Some("µm"));
    assert_eq!(meta.frame_interval_s, Some(2.0));
    assert_eq!(meta.mode, DisplayMode::Composite);
    // The DAPI (blue) channel's color survives the round-trip.
    let dapi_top = meta.channel_display[0].lut[255];
    assert!(dapi_top[2] > dapi_top[0] && dapi_top[2] > dapi_top[1], "DAPI should ramp toward blue, got {dapi_top:?}");
}

#[test]
fn serialize_escapes_special_characters_in_names() {
    let write = StackMetaWrite::new(1, 1).mode(DisplayMode::Composite).channel("A & B <\"x\">", [255, 0, 0]);
    let geom = WriteGeometry { width: 2, height: 2, samples_per_pixel: 1, ome_pixel_type: "uint8" };
    let xml = serialize(1, &write, &geom).unwrap();
    assert!(xml.contains("A &amp; B &lt;&quot;x&quot;&gt;"), "special chars must be escaped: {xml}");
    // And it must still parse back cleanly (quick-xml unescapes on read).
    assert!(parse(&xml, 1, None, None).is_some());
}

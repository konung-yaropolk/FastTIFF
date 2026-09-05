use super::*;
use fasttiff_plugin_api::PlaneData;

fn result(channels: usize, slices: usize, frames: usize, w: u32, h: u32) -> ImageResult {
    let n = (w * h) as usize;
    let mut planes = Vec::new();
    // xyczt order, each plane tagged so a re-ordering is visible.
    for t in 0..frames.max(1) {
        for z in 0..slices.max(1) {
            for c in 0..channels.max(1) {
                let tag = (c * 100 + z * 10 + t) as f32;
                planes.push(PlaneData::F32((0..n).map(|_| tag).collect()));
            }
        }
    }
    ImageResult {
        width: w,
        height: h,
        channels,
        slices,
        frames,
        pixel_type: PixelType::F32,
        planes,
        name: "result".into(),
    }
}

/// The round trip must preserve the shape and every pixel, in order — this is
/// what makes a plugin result a real document rather than an approximation.
#[test]
fn a_result_round_trips_through_the_ordinary_reader() {
    // All three axes above 1, so the resolver keeps the Z axis.
    let img = result(2, 3, 2, 5, 4);
    let stack = to_stack(&img, None, false).expect("the result should open");

    assert_eq!(stack.dimensions(), Some((5, 4)));
    assert_eq!(
        (
            stack.display.dims.channels,
            stack.display.dims.slices,
            stack.display.dims.frames
        ),
        (2, 3, 2),
        "the declared axes must survive the round trip"
    );

    // And the pixels landed where they were declared to be.
    let view = crate::plugins::describe_view(
        &stack,
        0,
        false,
        fasttiff_plugin_api::VolumeView {
            mode: fasttiff_plugin_api::VolumeMode::Mip,
            density: 1.0,
            iso: 0.5,
            eye: [0.0; 3],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        },
    );
    let mut host = crate::plugins::StackHost::new(&stack, view);
    use fasttiff_plugin_api::{HostContext, Plane};
    let mut buf = Vec::new();
    for t in 0..2 {
        for z in 0..3 {
            for c in 0..2 {
                host.read_plane_f32(Plane::new(c, z, t), &mut buf).unwrap();
                assert_eq!(
                    buf[0],
                    (c * 100 + z * 10 + t) as f32,
                    "plane (c{c}, z{z}, t{t}) came back as something else"
                );
            }
        }
    }
}

#[test]
fn every_pixel_type_survives_the_round_trip() {
    let cases: Vec<(PixelType, PlaneData)> = vec![
        (PixelType::U8, PlaneData::U8(vec![7u8; 4])),
        (PixelType::U16, PlaneData::U16(vec![4242u16; 4])),
        (PixelType::F32, PlaneData::F32(vec![1.5f32; 4])),
    ];
    for (ty, plane) in cases {
        let img = ImageResult {
            width: 2,
            height: 2,
            channels: 1,
            slices: 1,
            frames: 1,
            pixel_type: ty,
            planes: vec![plane],
            name: "t".into(),
        };
        let stack = to_stack(&img, None, false).unwrap_or_else(|e| panic!("{ty:?}: {e:#}"));
        assert_eq!(stack.dimensions(), Some((2, 2)), "{ty:?}");
    }
}

/// A malformed result must be refused here, before it becomes a corrupt file.
#[test]
fn a_malformed_result_is_refused_before_encoding() {
    let mut img = result(1, 1, 1, 2, 2);
    img.planes = vec![PlaneData::F32(vec![0.0; 3])]; // one sample short
    let err = to_tiff_bytes(&img, None).unwrap_err().to_string();
    assert!(err.contains("unusable"), "unexpected error: {err}");
}

/// A multi-channel result with no metadata should composite, because that is
/// what a plugin producing several channels almost always means.
#[test]
fn a_multichannel_result_defaults_to_composite() {
    let img = result(2, 1, 2, 3, 3);
    let stack = to_stack(&img, None, false).unwrap();
    assert_eq!(stack.tiff.meta.mode, fast_tiff_lib::DisplayMode::Composite);
}

/// Metadata an importer supplies must reach the opened document.
#[test]
fn supplied_metadata_reaches_the_document() {
    let img = result(1, 1, 1, 2, 2);
    let info = StackInfo {
        name: "from-instrument".into(),
        unit: Some("micron".into()),
        frame_interval_s: Some(0.25),
        mode: DisplayMode::Grayscale,
        ..Default::default()
    };
    let stack = to_stack(&img, Some(&info), false).unwrap();
    assert_eq!(stack.tiff.meta.unit.as_deref(), Some("micron"));
    assert_eq!(stack.tiff.meta.frame_interval_s, Some(0.25));
    assert!(
        stack.path.to_string_lossy().contains("from-instrument"),
        "the supplied name should title the document"
    );
}

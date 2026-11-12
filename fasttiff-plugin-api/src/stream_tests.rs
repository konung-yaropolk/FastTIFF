use super::*;
use crate::{ImageInfo, PixelType, Plane, StackInfo, ViewParams, VolumeMode, VolumeView};

/// A host with nothing to read, which either takes planes as they come (and
/// counts them, holding none) or does not stream at all.
struct Counting {
    streams: bool,
    begun: Option<ImageSpec>,
    pushed: Vec<usize>,
    view: ViewParams,
    info: StackInfo,
}

impl Counting {
    fn new(streams: bool) -> Self {
        Counting {
            streams,
            begun: None,
            pushed: Vec::new(),
            view: ViewParams {
                frame_index: 0,
                volume_view: false,
                channels: Vec::new(),
                luts: Vec::new(),
                volume: VolumeView {
                    mode: VolumeMode::Mip,
                    density: 1.0,
                    iso: 0.5,
                    eye: [0.0; 3],
                    forward: [0.0, 0.0, 1.0],
                    up: [0.0, 1.0, 0.0],
                    right: [1.0, 0.0, 0.0],
                },
            },
            info: StackInfo::default(),
        }
    }
}

impl HostContext for Counting {
    fn image(&self) -> ImageInfo {
        ImageInfo {
            width: 2,
            height: 2,
            channels: 1,
            slices: 1,
            frames: 1,
            samples_per_pixel: 1,
            pixel_type: PixelType::U16,
        }
    }
    fn view(&self) -> &ViewParams {
        &self.view
    }
    fn stack_info(&self) -> &StackInfo {
        &self.info
    }
    fn read_plane_u16(&mut self, _p: Plane, _o: &mut Vec<u16>) -> Result<(), PluginError> {
        unreachable!()
    }
    fn read_plane_f32(&mut self, _p: Plane, _o: &mut Vec<f32>) -> Result<(), PluginError> {
        unreachable!()
    }
    fn progress(&mut self, _f: f32) -> bool {
        true
    }
    fn log(&mut self, _m: &str) {}
    fn streams_results(&self) -> bool {
        self.streams
    }
    fn begin_image(&mut self, spec: &ImageSpec) -> Result<(), PluginError> {
        self.begun = Some(spec.clone());
        Ok(())
    }
    fn push_plane(&mut self, plane: PlaneRef<'_>) -> Result<(), PluginError> {
        self.pushed.push(plane.len());
        Ok(())
    }
}

fn spec(frames: usize, pixel_type: PixelType) -> ImageSpec {
    ImageSpec {
        width: 2,
        height: 2,
        channels: 1,
        slices: 1,
        frames,
        pixel_type,
        channel_colors: Vec::new(),
        metadata: None,
        name: "r".into(),
    }
}

#[test]
fn a_streaming_host_is_given_every_plane_and_the_run_says_so() {
    let mut host = Counting::new(true);
    let mut out = ImageStream::begin(&mut host, spec(3, PixelType::U16)).unwrap();
    assert_eq!(host.begun, Some(spec(3, PixelType::U16)));
    let buf = [1u16, 2, 3, 4];
    for _ in 0..3 {
        out.push(&mut host, PlaneRef::U16(&buf)).unwrap();
    }
    assert_eq!(out.remaining(), 0);
    assert_eq!(host.pushed, vec![4, 4, 4]);
    assert_eq!(
        out.finish(StreamTarget::ReplaceDocument).unwrap(),
        Outcome::Streamed {
            target: StreamTarget::ReplaceDocument
        }
    );
}

/// A host that predates streaming gets the result whole, as the outcome the
/// target names: the plugin that streamed is the same plugin there.
#[test]
fn a_host_that_takes_results_whole_gets_the_image_back() {
    let mut host = Counting::new(false);
    let mut out = ImageStream::begin(&mut host, spec(2, PixelType::I16)).unwrap();
    out.push(&mut host, PlaneRef::U16(&[1, 2, 3, 4])).unwrap();
    out.push_owned(&mut host, PlaneData::U16(vec![5, 6, 7, 8]))
        .unwrap();
    assert!(host.begun.is_none() && host.pushed.is_empty());
    let Outcome::SaveToFile { image, path } = out
        .finish(StreamTarget::SaveToFile {
            path: "x.tif".into(),
        })
        .unwrap()
    else {
        panic!("expected the image back");
    };
    assert_eq!(path, "x.tif");
    assert_eq!(
        image.planes,
        vec![
            PlaneData::U16(vec![1, 2, 3, 4]),
            PlaneData::U16(vec![5, 6, 7, 8])
        ]
    );
    image.validate().unwrap();
}

/// Checked as each plane arrives, in the words a returned image gets.
#[test]
fn a_miscounted_or_missized_stream_is_refused_where_it_goes_wrong() {
    let mut host = Counting::new(true);
    let mut out = ImageStream::begin(&mut host, spec(2, PixelType::F32)).unwrap();
    let e = out.push(&mut host, PlaneRef::F32(&[0.0; 3])).unwrap_err();
    assert!(e.to_string().contains("plane 0 has 3 samples"), "{e}");
    let e = out.push(&mut host, PlaneRef::U8(&[0; 4])).unwrap_err();
    assert!(e.to_string().contains("plane 0 is U8"), "{e}");
    assert!(host.pushed.is_empty(), "a refused plane reached the host");

    out.push(&mut host, PlaneRef::F32(&[0.0; 4])).unwrap();
    let e = out.finish(StreamTarget::NewDocument).unwrap_err();
    assert!(e.to_string().contains("2 planes but carries 1"), "{e}");

    let mut out = ImageStream::begin(&mut host, spec(1, PixelType::F32)).unwrap();
    out.push(&mut host, PlaneRef::F32(&[0.0; 4])).unwrap();
    let e = out.push(&mut host, PlaneRef::F32(&[0.0; 4])).unwrap_err();
    assert!(e.to_string().contains("more than the 1 planes"), "{e}");
}

#[test]
fn a_shape_with_no_pixels_is_refused_before_the_host_hears_of_it() {
    let mut host = Counting::new(true);
    let mut s = spec(1, PixelType::U8);
    s.width = 0;
    assert!(ImageStream::begin(&mut host, s).is_err());
    assert!(host.begun.is_none());
}

/// The spec and the image are two views of one declaration.
#[test]
fn an_image_comes_apart_into_its_spec_and_planes_and_back() {
    let image = spec(2, PixelType::U8)
        .with_planes(vec![PlaneData::U8(vec![1; 4]), PlaneData::U8(vec![2; 4])]);
    assert_eq!(image.spec(), spec(2, PixelType::U8));
    let (s, planes) = image.clone().into_parts();
    assert_eq!(s.with_planes(planes), image);
}

/// A declaration of more planes than can be counted is refused, rather than
/// wrapping to a small count or reserving room for it.
#[test]
fn an_uncountable_result_is_refused() {
    let spec = ImageSpec {
        width: 2,
        height: 2,
        channels: usize::MAX,
        slices: 2,
        frames: 2,
        pixel_type: PixelType::U16,
        channel_colors: Vec::new(),
        metadata: None,
        name: "too many".into(),
    };
    assert_eq!(spec.plane_count(), None);
    let e = spec.validate().unwrap_err().to_string();
    assert!(e.contains("more than can be counted"), "{e}");
    for streams in [true, false] {
        let mut host = Counting::new(streams);
        assert!(ImageStream::begin(&mut host, spec.clone()).is_err());
        assert!(host.begun.is_none(), "the host was told of it");
    }
}

/// A description changed after the planes is what the result says, on a host
/// that keeps the planes as on one that streams them; the shape and the name
/// are fixed.
#[test]
fn a_description_can_change_until_the_result_is_finished() {
    // Kept: the whole image comes back with the late colour.
    let mut host = Counting::new(false);
    let mut out = ImageStream::begin(&mut host, spec(1, PixelType::U16)).unwrap();
    out.push(&mut host, PlaneRef::U16(&[1, 2, 3, 4])).unwrap();
    out.describe(&mut host, |s| s.channel_colors = vec![[9, 8, 7]])
        .unwrap();
    assert!(out
        .describe(&mut host, |s| s.frames = 2)
        .unwrap_err()
        .to_string()
        .contains("fixed"));
    assert!(out.describe(&mut host, |s| s.name = "x".into()).is_err());
    let Outcome::NewDocument(image) = out.finish(StreamTarget::NewDocument).unwrap() else {
        panic!("a host that keeps planes gets the image whole");
    };
    assert_eq!(image.channel_colors, vec![[9, 8, 7]]);

    // Streamed: the host is told, and this one has no way to take it.
    let mut host = Counting::new(true);
    let mut out = ImageStream::begin(&mut host, spec(1, PixelType::U16)).unwrap();
    out.push(&mut host, PlaneRef::U16(&[1, 2, 3, 4])).unwrap();
    assert!(
        out.describe(&mut host, |s| s.channel_colors = vec![[9, 8, 7]])
            .is_err(),
        "a host without `describe_image` refuses, and the plugin hears it"
    );
}

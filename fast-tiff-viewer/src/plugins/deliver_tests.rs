use super::*;
use crate::plugins::describe_view;
use fasttiff_plugin_api::{
    ImageSpec, PixelType, PlaneData, PlaneRef, StackInfo, VolumeMode, VolumeView,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Arc;

/// A two-frame 4x3 stack to run against.
fn stack() -> crate::Document {
    let image = ImageResult {
        width: 4,
        height: 3,
        channels: 1,
        slices: 1,
        frames: 2,
        pixel_type: PixelType::U16,
        planes: vec![PlaneData::U16(vec![1; 12]), PlaneData::U16(vec![2; 12])],
        channel_colors: Vec::new(),
        metadata: None,
        name: "source".into(),
    };
    crate::plugins::to_stack(&image, None, false).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ft-deliver-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn files_in(dir: &std::path::Path) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries.map(|e| e.unwrap().path()).collect(),
        Err(_) => Vec::new(),
    }
}

/// A streaming host over `s`, writing to `dir`, with a stop button.
fn host(s: &crate::Document, dir: &std::path::Path) -> (StackHost, Arc<AtomicBool>) {
    let view = describe_view(
        s,
        0,
        false,
        VolumeView {
            mode: VolumeMode::Mip,
            density: 1.0,
            iso: 0.5,
            eye: [0.0; 3],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        },
    );
    let stop = Arc::new(AtomicBool::new(false));
    let h = StackHost::new(s, view)
        .with_cancel(Arc::clone(&stop), Arc::new(AtomicU32::new(0)))
        .streaming_to(ResultDest::Dir(dir.to_path_buf()));
    (h, stop)
}

fn spec() -> ImageSpec {
    ImageSpec {
        width: 4,
        height: 3,
        channels: 1,
        slices: 1,
        frames: 2,
        pixel_type: PixelType::U16,
        channel_colors: Vec::new(),
        metadata: Some(StackInfo::default()),
        name: "made".into(),
    }
}

fn stream_into(h: &mut StackHost) {
    h.begin_image(&spec()).unwrap();
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    h.push_plane(PlaneRef::U16(&[8; 12])).unwrap();
}

/// A platform that records what it was asked, and opens a "window" by keeping
/// the result (or refusing it).
#[derive(Default)]
struct Platform {
    phases: Vec<String>,
    same: bool,
    refuse: bool,
    opened: Vec<(Option<PathBuf>, ImageResult)>,
}

impl Deliver for Platform {
    fn phase(&mut self, label: &str) {
        self.phases.push(label.to_string());
    }
    fn report(&mut self, _fraction: f32) -> bool {
        true
    }
    fn same_document(&self) -> bool {
        self.same
    }
    fn open_in_new_window(&mut self, result: ResultTiff) -> Handover {
        if self.refuse {
            return Handover::Refused(Box::new(result), "no windows today".into());
        }
        let image = result.to_image().unwrap();
        let path = result.path().map(|p| p.to_path_buf());
        self.opened.push((path, image));
        // The result drops here, as if the "window" had read it and gone.
        Handover::Opened("made.tif".into())
    }
    fn save(&mut self, result: ResultTiff, path: &str) -> Result<String, String> {
        result
            .save_as(std::path::Path::new(path))
            .map(|()| path.to_string())
            .map_err(|e| e.to_string())
    }
}

#[test]
fn a_streamed_result_goes_to_a_new_window_as_the_file_it_was_written_to() {
    let dir = scratch("streamed");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let mut to = Platform::default();
    let product = finish(
        Outcome::Streamed {
            target: StreamTarget::NewDocument,
        },
        &mut h,
        &mut to,
    )
    .unwrap();
    assert!(matches!(product, Product::NewWindow(ref n) if n == "made.tif"));
    assert_eq!(
        to.phases,
        ["Opening a new window"],
        "nothing left to encode"
    );
    let (path, image) = &to.opened[0];
    assert_eq!(
        path.as_deref().and_then(|p| p.parent()?.parent()),
        Some(dir.as_path()),
        "handed over as the file it was streamed into"
    );
    assert_eq!(
        path.as_deref().and_then(|p| p.file_name()),
        Some(std::ffi::OsStr::new("made.tif")),
        "named as the result, in a directory of its own"
    );
    assert_eq!(image.planes[1], PlaneData::U16(vec![8; 12]));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A result that came back whole goes through the same writer to the same
/// kind of file, under a phase of its own.
#[test]
fn a_whole_result_is_written_where_a_streamed_one_would_be() {
    let dir = scratch("whole");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    let image = spec().with_planes(vec![
        PlaneData::U16(vec![3; 12]),
        PlaneData::U16(vec![4; 12]),
    ]);
    let mut to = Platform::default();
    let product = finish(
        Outcome::NewDocument(Box::new(image.clone())),
        &mut h,
        &mut to,
    )
    .unwrap();
    assert!(matches!(product, Product::NewWindow(_)));
    assert_eq!(to.phases, ["Encoding result", "Opening a new window"]);
    assert!(to.opened[0].0.is_some(), "written to a file");
    assert_eq!(to.opened[0].1, image);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Replace replaces only the document it was run on.
#[test]
fn a_replacement_replaces_only_the_document_it_was_made_from() {
    let dir = scratch("replace");
    let s = stack();
    let replace = || Outcome::Streamed {
        target: StreamTarget::ReplaceDocument,
    };

    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let mut here = Platform {
        same: true,
        ..Default::default()
    };
    let Product::Replace(result) = finish(replace(), &mut h, &mut here).unwrap() else {
        panic!("expected a replacement");
    };
    assert!(here.phases.is_empty() && here.opened.is_empty());
    assert_eq!(
        result.to_image().unwrap().planes[0],
        PlaneData::U16(vec![7; 12])
    );
    drop(result);

    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let mut moved_on = Platform::default();
    let product = finish(replace(), &mut h, &mut moved_on).unwrap();
    assert!(
        matches!(product, Product::NewWindow(_)),
        "the document it would replace has gone, so it gets a window"
    );
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A window that will not open hands the result back to be shown here.
#[test]
fn a_refused_window_brings_the_result_back() {
    let dir = scratch("refused");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let mut to = Platform {
        refuse: true,
        ..Default::default()
    };
    let Product::Inline { result, why } = finish(
        Outcome::Streamed {
            target: StreamTarget::NewDocument,
        },
        &mut h,
        &mut to,
    )
    .unwrap() else {
        panic!("expected the result back");
    };
    assert_eq!(why, "no windows today");
    assert_eq!(result.spec(), &spec());
    drop(result);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Saving moves the streamed file to where the plugin asked.
#[test]
fn a_streamed_result_is_saved_where_the_plugin_asked() {
    let dir = scratch("save");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let target = scratch("save-target");
    std::fs::create_dir_all(&target).unwrap();
    let path = target.join("out.tif");
    let product = finish(
        Outcome::Streamed {
            target: StreamTarget::SaveToFile {
                path: path.to_string_lossy().into(),
            },
        },
        &mut h,
        &mut Platform::default(),
    )
    .unwrap();
    assert!(matches!(product, Product::Saved(_)));
    let back = fast_tiff_lib::TiffStack::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(back.frames.len(), 2);
    assert!(files_in(&dir).is_empty(), "the temporary file was moved");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&target);
}

/// A run the user stopped leaves nothing — whether the plugin noticed the stop
/// or finished regardless — and so does one that ended with something other
/// than the image it began.
#[test]
fn a_stopped_or_abandoned_stream_leaves_no_file() {
    let dir = scratch("stopped");
    let s = stack();

    // The plugin noticed, and said so.
    let (mut h, _) = host(&s, &dir);
    h.begin_image(&spec()).unwrap();
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    assert_eq!(files_in(&dir).len(), 1, "being written as it goes");
    let p = finish(Outcome::Cancelled, &mut h, &mut Platform::default()).unwrap();
    assert!(matches!(p, Product::Cancelled));
    assert!(
        files_in(&dir).is_empty(),
        "a cancelled stream left its file"
    );

    // The plugin did not notice, and finished anyway.
    let (mut h, stop) = host(&s, &dir);
    stream_into(&mut h);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut to = Platform::default();
    let p = finish(
        Outcome::Streamed {
            target: StreamTarget::NewDocument,
        },
        &mut h,
        &mut to,
    )
    .unwrap();
    assert!(matches!(p, Product::Cancelled));
    assert!(to.opened.is_empty());
    assert!(files_in(&dir).is_empty(), "a stopped run's result was kept");

    // It began an image and ended with a message.
    let (mut h, _) = host(&s, &dir);
    stream_into(&mut h);
    let p = finish(
        Outcome::Message("no".into()),
        &mut h,
        &mut Platform::default(),
    )
    .unwrap();
    assert!(matches!(p, Product::Message(_)));
    assert!(
        files_in(&dir).is_empty(),
        "an abandoned stream left its file"
    );

    // And a host dropped mid-stream — a run that failed or panicked.
    let (mut h, _) = host(&s, &dir);
    h.begin_image(&spec()).unwrap();
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    drop(h);
    assert!(files_in(&dir).is_empty(), "a dropped host left its file");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Saying it streamed without streaming is an error, not an empty window.
#[test]
fn a_stream_that_never_came_is_an_error() {
    let dir = scratch("never");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    let e = finish(
        Outcome::Streamed {
            target: StreamTarget::NewDocument,
        },
        &mut h,
        &mut Platform::default(),
    )
    .unwrap_err();
    assert!(e.contains("none arrived"), "{e}");

    let (mut h, _) = host(&s, &dir);
    h.begin_image(&spec()).unwrap();
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    let e = finish(
        Outcome::Streamed {
            target: StreamTarget::NewDocument,
        },
        &mut h,
        &mut Platform::default(),
    )
    .unwrap_err();
    assert!(e.contains("2 planes but carries 1"), "{e}");
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stopped run's streamed result is thrown away before it is finished, so
/// one cut short by the stop is "cancelled", not "the plugin miscounted" —
/// and no directory is written for it.
#[test]
fn a_stopped_stream_is_discarded_before_it_is_finished() {
    let dir = scratch("stopped-short");
    let s = stack();
    let (mut h, stop) = host(&s, &dir);
    h.begin_image(&spec()).unwrap();
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let p = finish(
        Outcome::Streamed {
            target: StreamTarget::ReplaceDocument,
        },
        &mut h,
        &mut Platform::default(),
    )
    .expect("a stop is not a miscount");
    assert!(matches!(p, Product::Cancelled), "{p:?}");
    assert!(h.take_result().is_none(), "the result outlived the stop");
    assert!(files_in(&dir).is_empty(), "a stopped run's result was kept");
    let _ = std::fs::remove_dir_all(&dir);
}

/// In process as across the boundary: declaring again before the first plane
/// restates the result — its first file gone with it — and after one is a
/// second result, refused in the same words.
#[test]
fn the_host_takes_a_restatement_until_the_first_plane() {
    let dir = scratch("restated");
    let s = stack();
    let (mut h, _) = host(&s, &dir);
    h.begin_image(&spec()).unwrap();
    let mut wider = spec();
    wider.name = "restated".into();
    h.begin_image(&wider).unwrap();
    assert_eq!(
        files_in(&dir).len(),
        1,
        "the first declaration's file stayed"
    );
    h.push_plane(PlaneRef::U16(&[7; 12])).unwrap();
    assert_eq!(
        h.begin_image(&spec()).unwrap_err(),
        fasttiff_plugin_api::PluginError::failed(crate::plugins::result::SECOND_RESULT)
    );
    h.push_plane(PlaneRef::U16(&[8; 12])).unwrap();
    let done = h.take_result().unwrap().unwrap();
    assert_eq!(done.name(), "restated");
    drop(done);
    assert!(files_in(&dir).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

//! The registrations that stream their result, run the way the app runs them.
//!
//! Through the registry — so across the C boundary, the plugin's
//! `ImageStream` pushing through `fasttiff-plugin`'s host and the host's sink
//! into a [`StackHost`] that writes each plane to a file as it arrives — and
//! held to one thing: the file is exactly the result the same run hands back
//! whole to a host that does not stream. Then the other half of streaming: a
//! run stopped after its result has begun leaves nothing on disk.
#![cfg(feature = "builtin-plugins")]

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::plugins::{describe_view, Registry, ResultDest, StackHost};
use fast_tiff_viewer::Document;
use fasttiff_plugin_api::{
    HostContext, ImageResult, ImageSpec, Outcome, ParamKind, ParamValue, Params, Plane, PlaneRef,
    Plugin, PluginError, StackInfo, StreamTarget, ViewParams, VolumeMode, VolumeView,
};
use std::io::Cursor;
use std::path::{Path, PathBuf};

fn view() -> VolumeView {
    VolumeView {
        mode: VolumeMode::Mip,
        density: 1.0,
        iso: 0.5,
        eye: [0.0; 3],
        forward: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
    }
}

fn host(s: &Document) -> StackHost {
    StackHost::new(s, describe_view(s, 0, false, view()))
}

/// A directory of its own per test and process, empty.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ft-streamed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(e) => e.map(|e| e.unwrap().path()).collect(),
        Err(_) => Vec::new(),
    }
}

/// A field of blobs wandering along `path`, one channel, as a real stack.
fn wandering(side: u32, path: &[(i32, i32)]) -> Document {
    let base: Vec<f32> = {
        let mut f = vec![10.0f32; (side * side) as usize];
        for (cy, cx, amp) in [
            (20.0f32, 24.0f32, 200.0f32),
            (40.0, 44.0, 150.0),
            (28.0, 12.0, 120.0),
        ] {
            for y in 0..side {
                for x in 0..side {
                    let d2 = ((y as f32 - cy).powi(2) + (x as f32 - cx).powi(2)) / 8.0;
                    f[(y * side + x) as usize] += amp * (-d2).exp();
                }
            }
        }
        f
    };
    let opts = WriterOptions::new(side, side, SampleType::F32).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for &(dy, dx) in path {
        let px: Vec<f32> = (0..side * side)
            .map(|i| {
                let sy = ((i / side) as i32 - dy).rem_euclid(side as i32) as u32;
                let sx = ((i % side) as i32 - dx).rem_euclid(side as i32) as u32;
                base[(sy * side + sx) as usize]
            })
            .collect();
        w.write_frame_f32(&px).unwrap();
    }
    Document::from_bytes(w.finish().unwrap().into_inner(), "moving.tif".into(), false).unwrap()
}

const PATH: [(i32, i32); 6] = [(0, 0), (3, -2), (-4, 1), (2, 3), (-1, -3), (4, 2)];

/// The registry's own copy of a built-in, as the app runs it.
fn installed(reg: &mut Registry, id: &str) -> usize {
    reg.find(id)
        .unwrap_or_else(|| panic!("{id} is not installed"))
}

/// The edge choice labelled `label`, as the dialog would set it.
fn edges(reg: &Registry, index: usize, s: &Document, label: &str) -> ParamValue {
    let decls = reg.entries()[index].plugin.params(&host(s));
    let decl = decls
        .iter()
        .find(|d| d.key == "edges")
        .expect("an edge choice");
    let ParamKind::Choice { options, .. } = &decl.kind else {
        panic!("the edge choice is a choice");
    };
    ParamValue::Choice(options.iter().position(|o| o == label).expect(label))
}

/// Run plugin `index` on `s` twice — on a host that takes the result whole,
/// and on one that streams it into `dir` — and hand back the whole result and
/// the streamed one as read back from its file, with the file's path.
fn whole_and_streamed(
    reg: &mut Registry,
    index: usize,
    s: &Document,
    params: &Params,
    dir: &Path,
) -> (ImageResult, ImageResult, PathBuf) {
    let plugin = &mut reg.get_mut(index).unwrap().plugin;
    let Outcome::NewDocument(whole) = plugin.run(&mut host(s), params).expect("whole run") else {
        panic!("a host that does not stream gets the result whole");
    };

    let mut h = host(s).streaming_to(ResultDest::Dir(dir.to_path_buf()));
    let out = plugin.run(&mut h, params).expect("streamed run");
    assert_eq!(
        out,
        Outcome::Streamed {
            target: StreamTarget::NewDocument
        },
        "a streaming host gets the result as it is made"
    );
    let file = h.take_result().expect("a result was streamed").unwrap();
    let path = file.path().expect("written to a file").to_path_buf();
    assert!(
        path.starts_with(dir),
        "{} is not in {}",
        path.display(),
        dir.display()
    );
    let streamed = file.to_image().unwrap();
    // `file` drops here, and the file with it: nothing kept it.
    (*whole, streamed, path)
}

/// suite2p stabilization, rigid and warped, cropped and not: the file a
/// streaming host writes is the result a whole run returns, to the sample.
#[test]
fn a_streamed_stabilization_is_the_whole_one() {
    let s = wandering(64, &PATH);
    let mut reg = Registry::new();
    let index = installed(&mut reg, "dev.fasttiff.stabilize");
    let crop = edges(&reg, index, &s, "Crop to the common area");
    let dir = scratch("stabilize");
    for (nonrigid, cropped) in [(true, false), (false, true), (true, true)] {
        let mut params = Params::new();
        params.set("spatial_taper", ParamValue::Float(5.0));
        params.set("maxregshift", ParamValue::Float(0.3));
        params.set("nonrigid", ParamValue::Bool(nonrigid));
        if cropped {
            params.set("edges", crop.clone());
        }
        let (whole, streamed, path) = whole_and_streamed(&mut reg, index, &s, &params, &dir);
        let what = format!("non-rigid {nonrigid}, cropped {cropped}");
        if cropped {
            assert!(whole.width < 64 && whole.height < 64, "{what}: not cropped");
        }
        assert_eq!(
            streamed.spec(),
            whole.spec(),
            "{what}: declared differently"
        );
        assert!(streamed.planes == whole.planes, "{what}: the pixels differ");
        assert_eq!(
            path.file_name().unwrap(),
            "moving-stabilized.tif",
            "{what}: the file is not named as the result"
        );
    }
    assert!(
        entries(&dir).is_empty(),
        "results nobody kept: {:?}",
        entries(&dir)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// TurboReg, the same.
#[test]
fn a_streamed_registration_is_the_whole_one() {
    let path = [(0, 0), (2, -1), (-1, 2), (1, 1)];
    let s = wandering(64, &path);
    let mut reg = Registry::new();
    let index = installed(&mut reg, "dev.fasttiff.turboreg");
    let crop = edges(&reg, index, &s, "Crop to the common area");
    let dir = scratch("turboreg");
    for cropped in [false, true] {
        let mut params = Params::new();
        if cropped {
            params.set("edges", crop.clone());
        }
        let (whole, streamed, _) = whole_and_streamed(&mut reg, index, &s, &params, &dir);
        assert_eq!(streamed.spec(), whole.spec(), "cropped {cropped}");
        assert!(
            streamed.planes == whole.planes,
            "cropped {cropped}: the pixels differ"
        );
    }
    assert!(
        entries(&dir).is_empty(),
        "results nobody kept: {:?}",
        entries(&dir)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A streaming host whose user presses Stop as soon as the first plane of the
/// result has been written: the flag the window's Cancel button sets.
struct StopsOnceBegun {
    inner: StackHost,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pushed: usize,
}

impl HostContext for StopsOnceBegun {
    fn image(&self) -> fasttiff_plugin_api::ImageInfo {
        self.inner.image()
    }
    fn view(&self) -> &ViewParams {
        self.inner.view()
    }
    fn stack_info(&self) -> &StackInfo {
        self.inner.stack_info()
    }
    fn read_plane_u16(&mut self, p: Plane, out: &mut Vec<u16>) -> Result<(), PluginError> {
        self.inner.read_plane_u16(p, out)
    }
    fn read_plane_f32(&mut self, p: Plane, out: &mut Vec<f32>) -> Result<(), PluginError> {
        self.inner.read_plane_f32(p, out)
    }
    fn progress(&mut self, f: f32) -> bool {
        self.inner.progress(f)
    }
    fn log(&mut self, m: &str) {
        self.inner.log(m)
    }
    fn streams_results(&self) -> bool {
        self.inner.streams_results()
    }
    fn begin_image(&mut self, spec: &ImageSpec) -> Result<(), PluginError> {
        self.inner.begin_image(spec)
    }
    fn push_plane(&mut self, plane: PlaneRef<'_>) -> Result<(), PluginError> {
        self.pushed += 1;
        let pushed = self.inner.push_plane(plane);
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        pushed
    }
    fn describe_image(&mut self, spec: &ImageSpec) -> Result<(), PluginError> {
        self.inner.describe_image(spec)
    }
}

/// A platform that is never asked for anything: a stopped run has nothing
/// to open or save.
struct NoWindows;

impl fast_tiff_viewer::plugins::Deliver for NoWindows {
    fn phase(&mut self, _label: &str) {}
    fn report(&mut self, _fraction: f32) -> bool {
        true
    }
    fn same_document(&self) -> bool {
        true
    }
    fn open_in_new_window(
        &mut self,
        _result: fast_tiff_viewer::plugins::ResultTiff,
    ) -> fast_tiff_viewer::plugins::Handover {
        panic!("a stopped run's result was opened")
    }
    fn save(
        &mut self,
        _result: fast_tiff_viewer::plugins::ResultTiff,
        _path: &str,
    ) -> Result<String, String> {
        panic!("a stopped run's result was saved")
    }
}

/// Stopped after its result has begun — a file on disk with planes in it —
/// a run comes to "cancelled" and the host throws the result away, file,
/// directory and all: whether the plugin noticed the stop and returned
/// `Cancelled` (TurboReg checks after every timepoint) or finished the
/// stream anyway (stabilization only checks between chunks, and this
/// recording is one).
#[test]
fn a_run_stopped_after_its_result_began_leaves_no_file() {
    let s = wandering(64, &[(0, 0), (2, -1), (-1, 2), (1, 1)]);
    let mut reg = Registry::new();
    let dir = scratch("stopped");
    for (id, noticed) in [
        ("dev.fasttiff.turboreg", true),
        ("dev.fasttiff.stabilize", false),
    ] {
        let index = installed(&mut reg, id);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut h = StopsOnceBegun {
            inner: host(&s)
                .with_cancel(
                    std::sync::Arc::clone(&stop),
                    std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
                )
                .streaming_to(ResultDest::Dir(dir.clone())),
            stop,
            pushed: 0,
        };
        let plugin = &mut reg.get_mut(index).unwrap().plugin;
        let mut params = Params::new();
        params.set("spatial_taper", ParamValue::Float(5.0));
        let out = plugin
            .run(&mut h, &params)
            .expect("a stop is not a failure");
        assert_eq!(out == Outcome::Cancelled, noticed, "{id}: {out:?}");
        assert!(h.pushed > 0, "{id}: stopped before its result began");
        assert_eq!(entries(&dir).len(), 1, "{id}: the result was on disk");
        // What the app does next, with whatever the run came to.
        let product = fast_tiff_viewer::plugins::deliver::finish(out, &mut h.inner, &mut NoWindows)
            .expect("a stop is not a failure");
        assert!(
            matches!(product, fast_tiff_viewer::plugins::Product::Cancelled),
            "{id}: {product:?}"
        );
        assert!(
            entries(&dir).is_empty(),
            "{id}: a stopped run left {:?}",
            entries(&dir)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

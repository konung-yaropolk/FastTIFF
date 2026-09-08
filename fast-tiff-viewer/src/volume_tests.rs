//! The background volume builder's failure, which is not a rare case.
//!
//! The worker opens its own map of the file — which a stack that came from a
//! plugin does not have. An imported OIR is held in memory and its `path` is
//! the name it is shown under, so the open fails every time, and what the
//! caller does about that decides whether 3D works at all for imported data.
//!
//! It used to sit on "Loading 3D…" forever. The caller's check was "did
//! `request` succeed", and it succeeded: the request was sent while the worker
//! was still starting, before the open had been tried. Every frame after that
//! polled for a reply from a thread that had already exited.

use super::*;

/// Wait for `f`, up to a second. The worker has to be scheduled and try to open
/// a file first, so there is nothing to assert synchronously.
fn within_a_second(what: &str, mut f: impl FnMut() -> bool) {
    let start = std::time::Instant::now();
    while start.elapsed() < std::time::Duration::from_secs(1) {
        if f() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    panic!("{what} did not happen within a second");
}

/// A file that is not there is the ordinary case for an imported stack, and the
/// builder has to *say* it gave up rather than leaving the caller to infer it.
#[test]
fn a_builder_that_cannot_open_its_file_reports_it() {
    let path = std::env::temp_dir().join("fasttiff-no-such-stack-9f3a.tif");
    let _ = std::fs::remove_file(&path);
    let builder = VolumeBuilder::new(path).expect("the thread should spawn");

    within_a_second("the builder should report that it gave up", || {
        builder.failed()
    });
}

/// Why the flag exists rather than a failed `request`.
///
/// A request sent before the worker has finished failing is *accepted* — the
/// channel is open until the thread exits and drops the receiver. A caller
/// that reads that as "the worker is fine" waits for a reply forever, which is
/// exactly what the loading screen was doing.
#[test]
fn a_request_can_succeed_against_a_worker_that_is_already_doomed() {
    let path = std::env::temp_dir().join("fasttiff-no-such-stack-4b21.tif");
    let _ = std::fs::remove_file(&path);
    let builder = VolumeBuilder::new(path).expect("the thread should spawn");

    // Whether this particular send wins the race is a scheduling detail, so it
    // is not asserted either way. What is asserted is that the flag settles on
    // "gave up" regardless — the caller has something reliable to read.
    let _accepted = builder.request(0, plan());
    within_a_second("the builder should report that it gave up", || {
        builder.failed()
    });
    // And nothing is ever going to arrive for it.
    assert!(builder.take_matching(0, 0).is_none());
}

/// A builder over a real file does not claim to have failed.
#[test]
fn a_builder_over_a_readable_file_does_not_report_failure() {
    let path = std::env::temp_dir().join("fasttiff-volume-builder-ok.tif");
    std::fs::write(&path, tiny_tiff()).expect("write a stack to open");
    let builder = VolumeBuilder::new(path.clone()).expect("the thread should spawn");

    // Give the worker the same chance to fail that the tests above give it.
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(
        !builder.failed(),
        "a file that opens should not be reported as unopenable"
    );
    drop(builder);
    let _ = std::fs::remove_file(&path);
}

fn plan() -> VolumePlan {
    VolumePlan {
        kinds: vec![ChannelKind::Int8],
        rgb: false,
        channels: 1,
        slices: 1,
        frames: 1,
        time: 0,
        max_dim: 256,
    }
}

/// The smallest stack `TiffStack::open` will accept.
fn tiny_tiff() -> Vec<u8> {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let mut w = TiffWriter::new(
        std::io::Cursor::new(Vec::new()),
        WriterOptions::new(2, 2, SampleType::U8),
    )
    .expect("writer");
    w.write_frame_bytes(&[0u8; 4]).expect("frame");
    w.finish().expect("finish").into_inner()
}

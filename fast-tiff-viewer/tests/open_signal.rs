//! Opening a file has to report that it landed, whichever way it ran.
//!
//! A frontend resets everything that described the previous file — the zoom,
//! the fit-to-window request, the panel — on that one signal. When the signal
//! only came from the worker-thread path, the browser build, which has no
//! threads and always loads inline, never got it: every image opened at 100%
//! instead of fitted, and nothing looked broken enough to notice.
//!
//! Run under `--no-default-features` as well as the default, since that is the
//! configuration the browser builds and the one the bug lived in.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::{LoadSource, Viewer};
use std::io::Cursor;

fn tiff_bytes(frames: usize, w: u32, h: u32) -> Vec<u8> {
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(1, frames));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames {
        let px: Vec<u16> = (0..w * h).map(|i| (i as u16).wrapping_add(f as u16 * 7)).collect();
        let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_frame_bytes(&bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

/// Drive an open to completion the way a frontend does, and report how many
/// times the landing was signalled.
fn open_and_count_signals(viewer: &mut Viewer, source: LoadSource) -> usize {
    viewer.begin_open(source);
    let mut signals = 0;
    // Generous: a worker has to be scheduled, and this is a tiny file.
    for _ in 0..2000 {
        if viewer.poll_open() {
            signals += 1;
        }
        if signals > 0 && viewer.load_stage().is_none() {
            // Poll a few more times to prove the signal is not repeated.
            for _ in 0..5 {
                if viewer.poll_open() {
                    signals += 1;
                }
            }
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    signals
}

#[test]
fn an_open_signals_exactly_once() {
    let mut viewer = Viewer::new();
    let signals = open_and_count_signals(&mut viewer, LoadSource::Bytes(tiff_bytes(3, 16, 16), "t.tif".into()));
    assert_eq!(signals, 1, "the landing should be reported once and only once");
    assert!(viewer.stack.is_some(), "the stack should be installed by then");
}

#[test]
fn a_failed_open_still_signals() {
    // Otherwise the frontend waits for ever for a file that is never coming,
    // and the error it recorded is never surfaced.
    let mut viewer = Viewer::new();
    let signals = open_and_count_signals(&mut viewer, LoadSource::Bytes(b"not a tiff".to_vec(), "bad.tif".into()));
    assert_eq!(signals, 1);
    assert!(viewer.stack.is_none());
    assert!(viewer.status.is_some(), "a failure should leave something to show");
}

#[test]
fn nothing_is_signalled_when_nothing_was_opened() {
    let mut viewer = Viewer::new();
    for _ in 0..10 {
        assert!(!viewer.poll_open(), "an idle viewer should report no landing");
    }
}

#[test]
fn opening_a_second_file_signals_again() {
    let mut viewer = Viewer::new();
    open_and_count_signals(&mut viewer, LoadSource::Bytes(tiff_bytes(2, 8, 8), "a.tif".into()));
    let signals = open_and_count_signals(&mut viewer, LoadSource::Bytes(tiff_bytes(5, 8, 8), "b.tif".into()));
    assert_eq!(signals, 1, "the second open must report its landing too");
    assert_eq!(viewer.stack.as_ref().map(|s| s.frame_count()), Some(5));
}

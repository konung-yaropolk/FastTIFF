//! The pop-up registry: which window a toggle means, the order they are drawn
//! in, and which of them a new document closes.

use super::*;
use std::cell::RefCell;
use std::rc::Rc;

/// A window that only says it was drawn, and closes itself when told to.
struct Probe {
    id: WindowId,
    drawn: Rc<RefCell<Vec<&'static str>>>,
    closes_itself: bool,
    outlives: bool,
}

impl AppWindow for Probe {
    fn id(&self) -> WindowId {
        self.id
    }
    fn show(&mut self, _: &egui::Context, open: &mut bool, _: &mut Host<'_>) {
        self.drawn.borrow_mut().push(self.id.0);
        if self.closes_itself {
            *open = false;
        }
    }
    fn outlives_document(&self, _: &Viewer, _: bool) -> bool {
        self.outlives
    }
}

const A: WindowId = WindowId("a");
const B: WindowId = WindowId("b");
const C: WindowId = WindowId("c");

fn probes(drawn: &Rc<RefCell<Vec<&'static str>>>) -> Windows {
    let probe = |id, closes_itself, outlives| -> Box<dyn AppWindow> {
        Box::new(Probe {
            id,
            drawn: Rc::clone(drawn),
            closes_itself,
            outlives,
        })
    };
    Windows::of(vec![
        probe(A, false, true),
        probe(B, true, false),
        probe(C, false, false),
    ])
}

/// Draw `windows` once, as the frame does.
fn show(windows: &mut Windows) {
    let mut core = Viewer::new();
    let mut prefs = VolumePrefs::default();
    let mut host = Host {
        core: &mut core,
        prefs: &mut prefs,
    };
    windows.show(&egui::Context::default(), &mut host);
}

#[test]
fn every_window_starts_closed() {
    let windows = Windows::new();
    for id in [
        WindowId::METADATA,
        WindowId::HISTOGRAM,
        WindowId::RENDER_SETTINGS,
    ] {
        assert!(!windows.is_open(id), "{id:?}");
    }
}

/// A toggle opens its own window and no other, and a second one closes it.
#[test]
fn a_toggle_opens_and_closes_its_own_window() {
    let mut windows = Windows::new();
    windows.toggle(WindowId::HISTOGRAM);
    assert!(windows.is_open(WindowId::HISTOGRAM));
    assert!(!windows.is_open(WindowId::METADATA));
    assert!(!windows.is_open(WindowId::RENDER_SETTINGS));
    windows.toggle(WindowId::HISTOGRAM);
    assert!(!windows.is_open(WindowId::HISTOGRAM));
}

/// Open windows are drawn in the order they were registered, however they
/// were opened; closed ones are not drawn at all.
#[test]
fn open_windows_are_drawn_in_registration_order() {
    let drawn = Rc::new(RefCell::new(Vec::new()));
    let mut windows = probes(&drawn);
    windows.toggle(C);
    windows.toggle(A);
    show(&mut windows);
    assert_eq!(*drawn.borrow(), ["a", "c"]);
}

/// A window that closes itself while drawn — its close button, or nothing
/// left to show — is closed afterwards, and not drawn again.
#[test]
fn a_window_that_closes_itself_stays_closed() {
    let drawn = Rc::new(RefCell::new(Vec::new()));
    let mut windows = probes(&drawn);
    windows.toggle(B);
    show(&mut windows);
    assert!(!windows.is_open(B));
    show(&mut windows);
    assert_eq!(*drawn.borrow(), ["b"]);
}

/// A new document closes what does not outlive it, and opens nothing.
#[test]
fn a_new_document_closes_only_what_does_not_outlive_it() {
    let drawn = Rc::new(RefCell::new(Vec::new()));
    let mut windows = probes(&drawn);
    windows.toggle(A);
    windows.toggle(C);
    windows.reset(&Viewer::new(), false);
    assert!(windows.is_open(A), "outlives the document");
    assert!(!windows.is_open(B), "was closed, stays closed");
    assert!(!windows.is_open(C), "does not outlive it");
}

/// Every pop-up, open.
fn all_open() -> Windows {
    let mut windows = Windows::new();
    windows.toggle(WindowId::METADATA);
    windows.toggle(WindowId::HISTOGRAM);
    windows.toggle(WindowId::RENDER_SETTINGS);
    windows
}

/// A viewer holding a two-frame stack, which has a volume to show.
fn core_with_a_volume() -> Viewer {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(4, 4, SampleType::U8);
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..2u8 {
        w.write_frame_u8(&[f; 16]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    let mut core = Viewer::new();
    core.document = Some(Document::from_bytes(bytes, "two.tif".into(), false).unwrap());
    assert!(core.can_show_volume());
    core
}

/// On the desktop the metadata and the histogram stay up across a new
/// document, repointed at it; the 3D settings close when the new one has no
/// volume, since only a hidden button could reopen them.
#[test]
fn the_desktop_keeps_the_file_windows_and_closes_settings_without_a_volume() {
    let mut windows = all_open();
    let core = Viewer::new();
    assert!(!core.can_show_volume());
    windows.reset(&core, false);
    assert!(windows.is_open(WindowId::METADATA));
    assert!(windows.is_open(WindowId::HISTOGRAM));
    assert!(!windows.is_open(WindowId::RENDER_SETTINGS));
}

/// And over a new document with a volume, the 3D settings stay up as well.
#[test]
fn the_desktop_keeps_the_settings_while_there_is_a_volume() {
    let mut windows = all_open();
    windows.reset(&core_with_a_volume(), false);
    assert!(windows.is_open(WindowId::METADATA));
    assert!(windows.is_open(WindowId::HISTOGRAM));
    assert!(windows.is_open(WindowId::RENDER_SETTINGS));
}

/// In the browser a new document closes every pop-up — windows inside the
/// canvas, sized for the document that went — volume or not.
#[test]
fn the_browser_closes_every_pop_up() {
    for core in [Viewer::new(), core_with_a_volume()] {
        let mut windows = all_open();
        windows.reset(&core, true);
        assert!(!windows.is_open(WindowId::METADATA));
        assert!(!windows.is_open(WindowId::HISTOGRAM));
        assert!(!windows.is_open(WindowId::RENDER_SETTINGS));
    }
}

/// With nothing open, the metadata and histogram windows have nothing to show
/// and close themselves rather than drawing an empty frame.
#[test]
fn the_file_windows_close_without_a_document() {
    let mut windows = Windows::new();
    windows.toggle(WindowId::METADATA);
    windows.toggle(WindowId::HISTOGRAM);
    show(&mut windows);
    assert!(!windows.is_open(WindowId::METADATA));
    assert!(!windows.is_open(WindowId::HISTOGRAM));
}

/// The histogram's log scale is on until the user turns it off.
#[test]
fn the_histogram_starts_on_a_log_scale() {
    assert!(histogram::HistogramWindow::default().log);
}

/// Each window's id is the id its dialog has always been shown under, which
/// is what egui remembers its place and size by.
#[test]
fn the_window_ids_are_the_dialog_ids_they_always_were() {
    assert_eq!(WindowId::METADATA.0, "file_metadata");
    assert_eq!(WindowId::HISTOGRAM.0, "histogram");
    assert_eq!(WindowId::RENDER_SETTINGS.0, "render_settings");
}

//! Whole frames of the app, run headless (see `headless`): what the frame
//! does with what arrives, in the order `command::FRAME` takes its steps —
//! which no test of a helper the frame calls can see.

use super::headless::{stack, Headless, TempFile, CTRL, DT};
use super::platform::Target;
use super::screen::Egui as _;
use super::windows::WindowId;
use egui::{pos2, vec2, Key, Modifiers, PointerButton, Pos2};
use fast_tiff_viewer::{PhysicalPx, ViewMode};
use std::time::Duration;

/// An image too large for its window at 100%, in a maximized window, which
/// the desktop leaves the size of: the picture overflows the canvas both
/// ways, so a drag has somewhere to pan it.
fn pannable(frames: u8) -> Headless {
    let mut h = Headless::new(Target::Desktop)
        .sized(vec2(800.0, 600.0), 1.0)
        .maximized(true);
    h.open_bytes(stack(1000, 800, frames), "large.tif");
    h.settle();
    assert_eq!(h.app.view.model.zoom(), 1.0, "opened at 100%");
    h
}

/// Somewhere on the canvas, clear of the bars and of a tool window at the
/// top left.
const ON_CANVAS: Pos2 = pos2(600.0, 420.0);

/// How far the picture is panned, in points.
fn pan(h: &Headless) -> egui::Vec2 {
    h.app.view.model.pan().egui()
}

/// The middle of the canvas.
fn on_picture(h: &Headless) -> Pos2 {
    h.app.view.model.panel().egui().center()
}

/// The middle of the bottom bar's painted triangle whose first two corners
/// are level (`level`: the panel toggle) or plumb (the play button).
fn icon(h: &Headless, level: bool) -> Pos2 {
    let [a, b, c] = h
        .triangles()
        .into_iter()
        .find(|[a, b, _]| if level { a.y == b.y } else { a.x == b.x })
        .unwrap_or_else(|| panic!("no such icon: {:?}", h.triangles()));
    pos2((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0)
}

/// Open the channels panel, and wait for it to be laid out.
fn open_panel(h: &mut Headless) {
    h.settle();
    h.click(icon(h, true));
    h.hover(on_picture(h));
    h.settle();
    assert!(h.app.panel.expanded);
}

/// Choose `path` from the menus, a click at a time, each menu opening on
/// the frame after the click that opened it.
fn choose(h: &mut Headless, path: &[&str]) {
    for item in path {
        h.click_text(item);
        h.frame();
    }
}

/// A TIFF whose ImageDescription is OME-XML with `creator` on its root.
fn ome_stack(creator: &str) -> Vec<u8> {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(64, 48, SampleType::U8).description(format!(
        r#"<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" Creator="{creator}"/>"#
    ));
    let mut writer = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    writer.write_frame_u8(&[7; 64 * 48]).unwrap();
    writer.finish().unwrap().into_inner()
}

// ------------------------------------------------------------- opening

/// A file dropped on the desktop's window opens in it: the window is titled
/// after it once, sized around the picture at 100%, and the first frame is
/// planned for exactly the picture, in the device's pixels.
#[test]
fn a_dropped_file_is_titled_sized_and_planned() {
    let file = TempFile::new("dropped.tif", &stack(320, 200, 3));
    let mut h = Headless::new(Target::Desktop).sized(vec2(400.0, 300.0), 2.0);
    h.frame();
    assert_eq!(h.titles(), ["FastTIFF"], "an empty window is the app's");

    h.drop_file(file.path());
    h.run_until("the dropped file to land", |app| {
        app.core.document().is_some()
    });
    // What the frame it landed on planned: the picture, whole, at 100%.
    let image = h.app.sync_inputs.image;
    assert_eq!((image.uv_offset, image.uv_scale), ([0.0; 2], [1.0; 2]));
    assert_eq!(
        image.viewport,
        PhysicalPx::new(640.0, 400.0),
        "two pixels a point"
    );
    // The window takes the size it was asked for, and the next frame is laid
    // out in it.
    h.settle();
    let name = file.path().file_name().unwrap().to_string_lossy();
    assert_eq!(
        h.titles(),
        ["FastTIFF".to_string(), format!("{name} • FastTIFF")]
    );
    assert_eq!(h.size().x, 320.0, "as wide as the picture");
    assert!(h.shows("100%"), "{:?}", h.texts());
    assert_eq!(
        h.app.sync_inputs.image, image,
        "and every frame after plans the same"
    );

    // A file that is not a TIFF leaves the picture up, says why on the status
    // line, and does not retitle the window: the title is worked out again
    // for the landing, and is the same.
    let broken = TempFile::new("broken.tif", b"not a TIFF at all");
    h.drop_file(broken.path());
    h.run_until("the broken file to land", |app| {
        app.core.notices().next().is_some()
    });
    h.frame();
    assert!(h.shows("Failed to open file"), "{:?}", h.texts());
    assert_eq!(h.titles().len(), 2, "{:?}", h.titles());
    assert!(h.app.core.document().is_some(), "the last picture stays up");
}

/// In a browser a picture is fitted to the canvas the page gave it, exactly,
/// before it is first drawn — and the window is not touched.
///
/// Not maximized, unlike the harness's browser window: the desktop's sizing,
/// which is what runs here, would then size the window around the picture if
/// the fit had left the opening to it, so a window left alone says the fit
/// took the opening on the frame it landed.
#[test]
fn the_browser_fits_an_opened_picture_to_its_canvas() {
    let mut h = Headless::new(Target::Browser)
        .sized(vec2(800.0, 600.0), 1.0)
        .maximized(false);
    h.open_bytes(stack(1000, 500, 2), "wide.tif");
    let image = h.app.sync_inputs.image;
    assert_eq!((image.uv_offset, image.uv_scale), ([0.0; 2], [1.0; 2]));
    assert_eq!(
        image.viewport,
        PhysicalPx::new(800.0, 400.0),
        "fitted across the canvas, on the frame it landed"
    );
    assert_eq!(h.app.view.model.zoom(), 0.8);
    h.settle();
    assert!(h.shows("80%"), "{:?}", h.texts());
    assert!(h.resizes().is_empty(), "{:?}", h.resizes());
}

/// A pop-up open over one document stays up over the next on the desktop,
/// where it is a window of its own, and closes in a browser, where it lies
/// over the canvas sized for the document that went.
#[test]
fn a_new_document_closes_the_browsers_pop_ups_and_not_the_desktops() {
    for (target, stays) in [(Target::Desktop, true), (Target::Browser, false)] {
        let mut h = Headless::new(target)
            .sized(vec2(800.0, 600.0), 1.0)
            .maximized(true);
        h.open_bytes(stack(64, 48, 1), "first.tif");
        open_panel(&mut h);
        h.click_text("( i )");
        assert!(h.app.windows.is_open(WindowId::METADATA), "{target:?}");
        h.open_bytes(stack(64, 48, 1), "second.tif");
        assert_eq!(
            h.app.windows.is_open(WindowId::METADATA),
            stays,
            "{target:?}"
        );
    }
}

/// The metadata window shows the OME tree of the document on show: parsed
/// for the first, and again for the next, rather than kept from the last.
#[test]
fn the_metadata_window_reads_each_documents_own_ome() {
    let mut h = Headless::new(Target::Desktop)
        .sized(vec2(800.0, 900.0), 1.0)
        .maximized(true);
    h.open_bytes(ome_stack("first-file"), "first.ome.tif");
    open_panel(&mut h);
    h.click_text("( i )");
    h.frame();
    // In the ImageDescription's text, and not yet as a value of its own.
    assert!(h.shows("first-file") && !h.drew("first-file"));
    h.click_text("OME-TIFF metadata");
    h.frames(2);
    assert!(h.drew("first-file"), "{:?}", h.texts());

    h.open_bytes(ome_stack("second-file"), "second.ome.tif");
    h.frames(2);
    assert!(h.drew("second-file"), "{:?}", h.texts());
    assert!(!h.drew("first-file"));
}

// ------------------------------------------------------- the canvas tools

/// The primary button pans the picture until a tool is armed, and then it
/// is the tool's — the canvas does not pan under it — while the middle
/// button still pans. The point counter's click places a point where it
/// lands, and Ctrl+Z takes it back.
#[test]
fn an_armed_tool_has_the_primary_button_and_the_middle_one_still_pans() {
    let mut h = pannable(2);
    h.drag(ON_CANVAS, ON_CANVAS - vec2(100.0, 80.0));
    assert_eq!(pan(&h), vec2(100.0, 80.0), "nothing armed: it pans");

    choose(&mut h, &["☰", "Analysis", "Point counter"]);
    assert!(h.app.tools.point_counter.is_some(), "the menu armed it");
    let before = pan(&h);
    h.drag(ON_CANVAS, ON_CANVAS - vec2(100.0, 80.0));
    assert_eq!(pan(&h), before, "the counter has the primary button");
    h.drag_with(
        PointerButton::Middle,
        ON_CANVAS,
        ON_CANVAS - vec2(50.0, 40.0),
        5,
    );
    assert_eq!(pan(&h), before + vec2(50.0, 40.0), "the middle one pans");

    let origin = h.app.view.model.image_origin().egui();
    h.click(ON_CANVAS);
    let points = |h: &Headless| {
        let counter = &h.app.tools.point_counter.as_ref().unwrap().counter;
        counter
            .points
            .iter()
            .map(|p| (p.x, p.y))
            .collect::<Vec<_>>()
    };
    // Continuous coordinates, at 100%: the point under the pointer.
    let under = ON_CANVAS - origin;
    assert_eq!(points(&h), [(under.x, under.y)]);
    h.key(Key::Z, CTRL);
    assert_eq!(points(&h), []);
}

/// A plugin that asks for a rectangle gets the drag ahead of everything:
/// the drag draws it and the status line says what was drawn, and Escape
/// gives it back — the plugin is not run again, and the drag pans again.
#[test]
fn a_rectangle_a_plugin_asks_for_takes_the_drag_until_escape() {
    let mut h = pannable(1);
    choose(&mut h, &["☰", "Image", "Crop"]);
    h.run_until("Crop to ask for a rectangle", |app| {
        app.tools.region_pick.is_some()
    });
    h.frame();
    assert!(
        h.shows("Crop: drag a rectangle"),
        "its prompt is up: {:?}",
        h.texts()
    );

    let before = pan(&h);
    h.drag(ON_CANVAS, ON_CANVAS + vec2(60.0, 40.0));
    assert_eq!(pan(&h), before, "the drag drew instead");
    let roi = h
        .app
        .tools
        .region_pick
        .as_ref()
        .unwrap()
        .roi
        .expect("drawn");
    assert_eq!(
        (roi.w, roi.h),
        (61, 41),
        "from the pixel pressed to the one released, both in"
    );
    h.frame();
    assert!(h.shows("61 x 41 pixels"), "{:?}", h.texts());

    h.key(Key::Escape, Modifiers::NONE);
    assert!(h.app.tools.region_pick.is_none());
    assert!(h.app.job.is_none(), "nothing was run");
    assert!(h.shows("Crop: cancelled"), "{:?}", h.texts());
    h.drag(ON_CANVAS, ON_CANVAS - vec2(30.0, 20.0));
    assert_eq!(pan(&h), before + vec2(30.0, 20.0), "it pans again");
}

// --------------------------------------------------------------- repaints

/// A movie playing asks for the next frame at its own rate, and nothing
/// else; paused, and with the pointer at rest, the window asks for none.
#[test]
fn playing_asks_for_frames_at_the_movie_rate_and_an_idle_window_for_none() {
    let mut h = Headless::new(Target::Desktop).sized(vec2(800.0, 600.0), 1.0);
    h.open_bytes(stack(320, 200, 4), "movie.tif");
    h.settle();

    let play = icon(&h, false);
    h.click(play);
    h.hover(ON_CANVAS);
    assert!(h.app.core.playback.playing());
    // egui asks for frames of its own while the pointer is still moving.
    h.frames(30);
    let rate = Duration::from_secs_f64(1.0 / h.app.core.playback.fps());
    // What egui reports for a request at the movie's rate, which it brings
    // forward by a frame time; anything asked for sooner would be less.
    let at_rate = rate.saturating_sub(Duration::from_secs_f32(DT as f32));
    assert!(
        at_rate > Duration::from_millis(5),
        "a frame time short of it"
    );
    let mut frame = h.app.core.document().unwrap().frame();
    let mut moved = 0;
    for _ in 0..60 {
        h.frame();
        let after = h.repaint_after();
        assert!(
            after.abs_diff(at_rate) < Duration::from_millis(1),
            "asked after {after:?}, a request at the movie's rate ({rate:?}) being \
             reported as {at_rate:?}"
        );
        let now = h.app.core.document().unwrap().frame();
        moved += usize::from(now != frame);
        frame = now;
    }
    assert!(moved > 0, "a second of playing shows more than one frame");

    h.click(play);
    h.hover(ON_CANVAS);
    assert!(!h.app.core.playback.playing());
    h.settle();
    let paused_on = h.app.core.document().unwrap().frame();
    h.frames(30);
    assert_eq!(h.app.core.document().unwrap().frame(), paused_on);
}

// ------------------------------------------------------ the 2D/3D switch

/// Over to 3D and back leaves the 2D view where it was: the frame, the
/// zoom and the pan. And the switch reaches the canvas on the frame it was
/// clicked on (it is applied at `Phase::CopyPoll`, before the canvas).
#[test]
fn the_2d_3d_switch_keeps_the_frame_and_the_view() {
    let mut h = pannable(3);
    h.wheel(ON_CANVAS, -1.0);
    h.drag_with(
        PointerButton::Middle,
        ON_CANVAS,
        ON_CANVAS - vec2(40.0, 30.0),
        4,
    );
    h.settle();
    let (zoom, panned) = (h.app.view.model.zoom(), pan(&h));
    let image = h.app.sync_inputs.image;
    assert_eq!(h.app.core.document().unwrap().frame(), 1);
    assert_eq!(panned, vec2(40.0, 30.0));

    h.click_text("3D");
    assert_eq!(h.app.core.view_mode, ViewMode::Volume);
    assert!(
        h.shows("Loading 3D"),
        "drawn on the click frame: {:?}",
        h.texts()
    );
    assert!(!h.shows("100%"), "and no 2D zoom in the toolbar");
    h.frames(5);

    h.click_text("2D");
    h.frame();
    assert_eq!(h.app.core.view_mode, ViewMode::Movie);
    assert_eq!(h.app.core.document().unwrap().frame(), 1);
    assert_eq!((h.app.view.model.zoom(), pan(&h)), (zoom, panned));
    assert_eq!(
        h.app.sync_inputs.image, image,
        "the same part of the same frame"
    );
    assert!(h.shows("Frame 2 / 3") && h.shows("100%"), "{:?}", h.texts());
}

// ----------------------------------------------------- the commands' order

/// A wheel notch over the picture is queued by the canvas and applied at
/// `Phase::Canvas`: after the toolbar has said which frame it is on, and
/// before the frame ends — so the sync at its end shows the new frame, and
/// the toolbar says so on the next.
#[test]
fn a_wheel_notch_is_applied_after_the_canvas_and_before_the_frame_ends() {
    let mut h = Headless::new(Target::Desktop)
        .sized(vec2(800.0, 600.0), 1.0)
        .maximized(true);
    h.open_bytes(stack(320, 200, 3), "notched.tif");
    h.settle();
    h.wheel(on_picture(&h), -1.0);
    assert!(
        h.shows("Frame 1 / 3"),
        "the toolbar was drawn first: {:?}",
        h.texts()
    );
    assert_eq!(
        h.app.core.document().unwrap().frame(),
        1,
        "applied within the frame"
    );
    h.frame();
    assert!(h.shows("Frame 2 / 3"), "{:?}", h.texts());
}

// ------------------------------------------------------------- the panel

/// Opening the channels panel on the desktop grows the window by the
/// panel's height, so the picture keeps its size and the panel drops down
/// below it; closing it gives the height back.
#[test]
fn the_desktop_window_grows_by_the_panel_it_opens() {
    let mut h = Headless::new(Target::Desktop).sized(vec2(400.0, 300.0), 1.0);
    h.open_bytes(stack(320, 200, 2), "panel.tif");
    h.settle();
    let closed = h.size();
    let image = h.app.sync_inputs.image;
    assert_eq!(image.viewport, PhysicalPx::new(320.0, 200.0));

    let toggle = icon(&h, true);
    h.click(toggle);
    h.hover(on_picture(&h));
    h.settle();
    assert!(h.app.panel.expanded);
    assert_eq!(h.size().x, closed.x);
    assert!(h.size().y > closed.y, "{:?} -> {:?}", closed, h.size());
    assert_eq!(h.app.sync_inputs.image, image, "the picture keeps its size");

    h.click(toggle);
    h.hover(on_picture(&h));
    h.settle();
    assert!(!h.app.panel.expanded);
    assert_eq!(h.size(), closed);
    assert_eq!(h.app.sync_inputs.image, image);
}

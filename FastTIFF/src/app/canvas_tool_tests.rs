//! Who has the canvas, the picture's transform, and the tools' gestures.
//!
//! All three replaced code written inline in the canvas, before audit item
//! 18c: four hand-ordered chains deciding which tool had the pointer, two
//! conversions of one transform, and a gesture read from egui's response
//! separately by each tool's branch. That code is kept here, as it was but
//! for where its inputs come from, as the oracle the new code is held to.
//!
//! The oracles are scaffolding for that move, which changed nothing a tool
//! does, including what may be wrong: where a plot's region starts, and when
//! a press counts as taken, are the old branches' answers. Fixing one breaks
//! a comparison here; replace that comparison with an assertion of the
//! behaviour wanted rather than editing the "old" code to match.

use super::*;
use egui::PointerButton::{Middle, Primary};
use fast_tiff_viewer::plugins::builtin::point_counter::PointCounter;
use fasttiff_plugin_api::{Plot, SelectionKind, Shape};

// ------------------------------------------------------------ the arbitration

const ASKED: Claim = Claim {
    priority: Priority::Asked,
    layer: Layer::Regions,
};
const OPENED: Claim = Claim {
    priority: Priority::Opened,
    layer: Layer::Marks,
};
const OFFERED: Claim = Claim {
    priority: Priority::Offered,
    layer: Layer::Regions,
};

#[test]
fn with_no_claims_nobody_has_the_canvas() {
    let none: [(&str, Claim); 0] = [];
    for pinching in [false, true] {
        let a = arbitrate(&none, pinching);
        assert_eq!(a.owner, None);
        assert!(a.drawn.is_empty());
    }
}

/// The order the claims arrive in decides nothing: in every order, the
/// highest priority has the pointer, and of the other two the higher.
#[test]
fn the_highest_priority_has_the_pointer_in_any_order() {
    let claims = [("pick", ASKED), ("counter", OPENED), ("plot", OFFERED)];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for order in orders {
        let in_order: Vec<_> = order.iter().map(|&i| claims[i]).collect();
        assert_eq!(arbitrate(&in_order, false).owner, Some("pick"), "{order:?}");
        let rest = in_order.iter().copied().filter(|(k, _)| *k != "pick");
        assert_eq!(owner(rest), Some("counter"), "{order:?}");
    }
}

#[test]
fn of_equal_priorities_the_first_claim_wins() {
    assert_eq!(owner([("a", OFFERED), ("b", OFFERED)]), Some("a"));
    assert_eq!(owner([("b", OFFERED), ("a", OFFERED)]), Some("b"));
}

/// One tool's regions — the highest priority's — and the marks over them,
/// whatever order the claims come in.
#[test]
fn each_layer_draws_its_own_winner_regions_first() {
    let a = arbitrate(
        &[("counter", OPENED), ("plot", OFFERED), ("pick", ASKED)],
        false,
    );
    assert_eq!(a.drawn, vec!["pick", "counter"]);
    let a = arbitrate(&[("counter", OPENED), ("plot", OFFERED)], false);
    assert_eq!(a.owner, Some("counter"));
    assert_eq!(a.drawn, vec!["plot", "counter"], "outranked, still drawn");
}

/// A pinch is two fingers on the picture, not a press: no tool takes it,
/// and nothing is hidden for it.
#[test]
fn a_pinch_takes_the_pointer_from_every_tool_and_hides_none() {
    let claims = [("pick", ASKED), ("counter", OPENED), ("plot", OFFERED)];
    let a = arbitrate(&claims, true);
    assert_eq!(a.owner, None);
    assert_eq!(a.drawn, arbitrate(&claims, false).drawn);
}

fn region_pick() -> region_pick::RegionPick {
    region_pick::RegionPick {
        plugin: 1,
        params: Params::new(),
        name: "Crop".into(),
        roi: None,
        drag: None,
    }
}

fn plot_window(wants: SelectionKind) -> plot::PlotWindow {
    let mut chart = Plot::new("trace");
    chart.wants = wants;
    plot::PlotWindow::new(0, Params::new(), chart)
}

fn counter() -> point_counter::Session {
    point_counter::Session::new(PointCounter::new())
}

/// Every set of tools there can be: each of the three in use or not, and a
/// plot asking for regions or not.
fn every_combination() -> Vec<Tools> {
    let mut all = Vec::new();
    for pick in [false, true] {
        for count in [false, true] {
            for plot in [
                None,
                Some(SelectionKind::None),
                Some(SelectionKind::Regions),
            ] {
                all.push(Tools {
                    plot: plot.map(plot_window),
                    region_pick: pick.then(region_pick),
                    point_counter: count.then(counter),
                });
            }
        }
    }
    all
}

/// What the canvas's four chains decided before 18c.
#[derive(Debug, PartialEq)]
struct OldChains {
    selecting: bool,
    /// The tool whose branch took the gesture.
    routed: Option<ToolSlot>,
    drawn: Vec<ToolSlot>,
    /// Whether the cursor was the counter's.
    counter_cursor: bool,
}

/// The four chains, as `canvas.rs` had them; `on_map` is whether the drag in
/// flight began on the navigator.
fn old_chains(t: &Tools, pinching: bool, on_map: bool) -> OldChains {
    // (1) `selecting`.
    let selecting = (t.region_pick.is_some()
        || t.point_counter.is_some()
        || t.plot.as_ref().is_some_and(|p| p.wants_regions()))
        && !pinching;
    // (2) The gesture's routing.
    let routed = if let (true, Some(_)) = (selecting && !on_map, t.region_pick.as_ref()) {
        Some(ToolSlot::RegionPick)
    } else if let (true, Some(_)) = (selecting && !on_map, t.point_counter.as_ref()) {
        Some(ToolSlot::PointCounter)
    } else if selecting && !on_map {
        t.plot.as_ref().map(|_| ToolSlot::Plot)
    } else {
        None
    };
    // (3) The overlay.
    let mut drawn = Vec::new();
    if t.region_pick.is_some() {
        drawn.push(ToolSlot::RegionPick);
    } else if t.plot.as_ref().filter(|p| p.wants_regions()).is_some() {
        drawn.push(ToolSlot::Plot);
    }
    if t.point_counter.is_some() {
        drawn.push(ToolSlot::PointCounter);
    }
    // (4) The cursor.
    let counter_cursor = t
        .point_counter
        .as_ref()
        .filter(|_| selecting && t.region_pick.is_none())
        .is_some();
    OldChains {
        selecting,
        routed,
        drawn,
        counter_cursor,
    }
}

/// The drag rule, as `canvas.rs` had it: whether a drag pans the picture.
fn old_pans(
    selecting: bool,
    pannable: bool,
    (dragged, by_primary): (bool, bool),
    pinching: bool,
    on_map: bool,
) -> bool {
    let dragging_picture = dragged && !(selecting && by_primary);
    pannable && dragging_picture && !pinching && !on_map
}

/// The cursor chain, as `canvas.rs` had it. The counter's own cursor stands
/// for what `point_counter::cursor` gave, which the gestures below hold the
/// two to.
fn old_cursor(
    t: &Tools,
    selecting: bool,
    (autoscroll, over_map, on_map, pannable): (bool, bool, bool, bool),
    xf: &ImageXform,
) -> CursorIcon {
    if autoscroll {
        CursorIcon::AllScroll
    } else if over_map || on_map {
        CursorIcon::Move
    } else if let Some(tool) = t
        .point_counter
        .as_ref()
        .filter(|_| selecting && t.region_pick.is_none())
    {
        CanvasTool::cursor(tool, None, xf, 0).expect("the counter has a cursor")
    } else if pannable {
        CursorIcon::Grab
    } else {
        CursorIcon::Crosshair
    }
}

/// A drag this frame, as `(dragged, by_primary)`: none, by another button,
/// and by the primary one (which egui also reports as a drag).
const DRAGS: [(bool, bool); 3] = [(false, false), (true, false), (true, true)];

/// For every combination of tools, pinching or not, on the navigator or not,
/// the one arbitration — and what the canvas does with it — decides what each
/// of the four chains and the drag rule did.
#[test]
fn every_combination_of_tools_is_arbitrated_as_the_four_chains_did() {
    let xf = ImageXform {
        origin: Pos2::ZERO,
        zoom: 1.0,
        size: (20, 10),
    };
    let bools = [false, true];
    let mut seen = 0;
    for tools in every_combination() {
        for pinching in bools {
            for on_map in bools {
                let old = old_chains(&tools, pinching, on_map);
                let new = tools.arbitrate(pinching);
                let tool_cursor = |slot| tools.get(slot)?.cursor(None, &xf, 0);
                let case = format!(
                    "pick {} counter {} plot {:?} pinching {pinching} on map {on_map}",
                    tools.region_pick.is_some(),
                    tools.point_counter.is_some(),
                    tools.plot.as_ref().map(|p| p.plot.wants),
                );
                assert_eq!(new.owner.is_some(), old.selecting, "{case}");
                assert_eq!(new.routed(on_map), old.routed, "{case}");
                assert_eq!(new.drawn, old.drawn, "{case}");
                let cursor = new.owner.and_then(tool_cursor);
                assert_eq!(cursor.is_some(), old.counter_cursor, "{case}");
                for pannable in bools {
                    for drag in DRAGS {
                        assert_eq!(
                            pannable && new.pans(drag.0, drag.1, pinching, on_map),
                            old_pans(old.selecting, pannable, drag, pinching, on_map),
                            "{case} pannable {pannable} drag {drag:?}"
                        );
                    }
                    for autoscroll in bools {
                        for over_map in bools {
                            let navigator = over_map || on_map;
                            let pointer = (autoscroll, over_map, on_map, pannable);
                            assert_eq!(
                                new.cursor(autoscroll, navigator, tool_cursor, pannable),
                                old_cursor(&tools, old.selecting, pointer, &xf),
                                "{case} pointer {pointer:?}"
                            );
                        }
                    }
                }
                seen += 1;
            }
        }
    }
    assert_eq!(seen, 48);
}

/// A press is taken from the tools on the navigator, and anywhere while
/// autoscroll was on; off the map, or with no map drawn, it is theirs.
#[test]
fn a_press_on_the_map_or_out_of_autoscroll_is_taken() {
    let map = Rect::from_min_size(Pos2::new(160.0, 80.0), Vec2::new(40.0, 20.0));
    let on = Some(Pos2::new(170.0, 90.0));
    let off = Some(Pos2::new(110.0, 60.0));
    assert!(CanvasGesture::is_taken(false, Some(map), on));
    assert!(!CanvasGesture::is_taken(false, Some(map), off));
    assert!(!CanvasGesture::is_taken(false, None, on));
    assert!(!CanvasGesture::is_taken(false, Some(map), None));
    for at in [on, off, None] {
        assert!(CanvasGesture::is_taken(true, Some(map), at));
        assert!(CanvasGesture::is_taken(true, None, at));
    }
}

/// The table every dispatcher goes by lists each tool once, under its own
/// slot: with every tool in use, each entry is the very tool [`Tools::get`]
/// reaches by that slot. (The literal below names every field, so a tool added
/// to `Tools` is added here too.)
#[test]
fn the_table_lists_every_tool_once_under_its_own_slot() {
    let tools = Tools {
        plot: Some(plot_window(SelectionKind::Regions)),
        region_pick: Some(region_pick()),
        point_counter: Some(counter()),
    };
    let slots = tools.slots();
    for (i, slot) in slots.iter().enumerate() {
        assert!(!slots[i + 1..].contains(slot), "{slot:?} is listed twice");
    }
    for (slot, tool) in tools.table() {
        let listed = tool.expect("every tool is in use");
        let got = tools.get(slot).expect("every tool is in use");
        assert!(
            std::ptr::addr_eq(listed, got),
            "{slot:?} lists another tool"
        );
    }
}

/// The undo chord is the tools' exactly while the counter is open, as it was;
/// and a reset puts away what `finish_open` did: every tool on the desktop,
/// and in a browser every tool but a plot, which has always stayed up there.
#[test]
fn the_undo_chord_and_the_reset_follow_the_tools_in_use() {
    for web in [false, true] {
        for mut tools in every_combination() {
            let counting = tools.point_counter.is_some();
            assert_eq!(tools.takes_undo(), counting);
            let plot_kept = tools.plot.is_some() && web;
            tools.reset(web);
            assert!(tools.region_pick.is_none() && tools.point_counter.is_none());
            assert_eq!(tools.plot.is_some(), plot_kept, "web {web}");
        }
    }
}

// --------------------------------------------------------------- the transform

/// `plot::pixel_at`, as it was.
fn old_pixel_at(pos: Pos2, origin: Pos2, zoom: f32) -> (i64, i64) {
    (
        ((pos.x - origin.x) / zoom).floor() as i64,
        ((pos.y - origin.y) / zoom).floor() as i64,
    )
}

/// `point_counter::image_at`, as it was.
fn old_image_at(pos: Pos2, origin: Pos2, zoom: f32) -> (f32, f32) {
    ((pos.x - origin.x) / zoom, (pos.y - origin.y) / zoom)
}

/// `point_counter::screen_at`, as it was.
fn old_screen_at((x, y): (f32, f32), origin: Pos2, zoom: f32) -> Pos2 {
    origin + Vec2::new(x * zoom, y * zoom)
}

/// `plot::roi_rect`, as it was.
fn old_roi_rect(roi: &Roi, origin: Pos2, zoom: f32) -> Rect {
    Rect::from_min_size(
        origin + Vec2::new(roi.x as f32 * zoom, roi.y as f32 * zoom),
        Vec2::new(roi.w as f32 * zoom, roi.h as f32 * zoom),
    )
}

/// From far out to far in, through whole, fractional and awkward factors.
const ZOOMS: [f32; 13] = [
    0.03,
    0.1,
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    7.3,
    32.0,
];

/// Centred, scrolled, and scrolled a long way, on and off whole points.
const ORIGINS: [Pos2; 5] = [
    Pos2::new(0.0, 0.0),
    Pos2::new(100.0, 50.0),
    Pos2::new(-37.5, 12.25),
    Pos2::new(0.3, -1000.7),
    Pos2::new(-12345.6, 789.01),
];

/// The way back, for points and for regions. (The way there, `pixel_at` and
/// `image_at`, is held to its edges below.)
#[test]
fn the_transform_places_points_and_regions_as_the_old_functions_did() {
    let rois = [
        Roi::from_corners((0, 0), (1, 1), 20, 20).unwrap(),
        Roi::from_corners((3, 4), (17, 9), 20, 20).unwrap(),
        Roi::from_corners((0, 0), (20, 20), 20, 20)
            .unwrap()
            .with_shape(Shape::Ellipse),
    ];
    for origin in ORIGINS {
        for zoom in ZOOMS {
            let xf = ImageXform {
                origin,
                zoom,
                size: (20, 20),
            };
            for x in [-2.5, 0.0, 0.5, 3.25, 19.999, 20.0] {
                for y in [-0.75, 0.0, 0.5, 10.5, 20.0] {
                    assert_eq!(xf.screen_at((x, y)), old_screen_at((x, y), origin, zoom));
                }
            }
            for roi in &rois {
                assert_eq!(xf.roi_rect(roi), old_roi_rect(roi, origin, zoom));
            }
        }
    }
}

const ORIGIN: Pos2 = Pos2::new(100.0, 50.0);

fn at_4x() -> ImageXform {
    ImageXform {
        origin: ORIGIN,
        zoom: 4.0,
        size: (64, 64),
    }
}

#[test]
fn a_point_maps_to_the_pixel_it_is_within() {
    let xf = at_4x();
    // At 4x zoom, image pixel (0,0) covers screen x 100..104.
    assert_eq!(xf.pixel_at(Pos2::new(100.0, 50.0)), (0, 0));
    assert_eq!(xf.pixel_at(Pos2::new(103.9, 53.9)), (0, 0));
    assert_eq!(xf.pixel_at(Pos2::new(104.0, 54.0)), (1, 1));
    // Not rounded: three-quarters into a pixel is still that pixel.
    assert_eq!(xf.pixel_at(Pos2::new(107.0, 50.0)), (1, 0));
}

#[test]
fn a_point_left_of_the_image_maps_outside_it() {
    // Negative rather than clamped to 0 — clamping here would make a drag that
    // began off the image start at its corner instead of where it began.
    assert_eq!(at_4x().pixel_at(Pos2::new(96.0, 46.0)), (-1, -1));
}

/// The middle of a pixel is `.5`, at any zoom, and the conversion goes both
/// ways.
#[test]
fn screen_and_image_positions_are_continuous_and_invert() {
    let xf = at_4x();
    // At 4x, pixel (10, 20) spans screen x 140..144, y 130..134.
    assert_eq!(xf.image_at(egui::pos2(142.0, 132.0)), (10.5, 20.5));
    assert_eq!(xf.image_at(ORIGIN), (0.0, 0.0));
    let p = (37.25, 3.75);
    for zoom in [0.25, 1.0, 1.5, 32.0] {
        let xf = ImageXform { zoom, ..xf };
        let back = xf.image_at(xf.screen_at(p));
        assert!(
            (back.0 - p.0).abs() < 1e-4 && (back.1 - p.1).abs() < 1e-4,
            "{zoom}"
        );
    }
}

// ----------------------------------------------------------------- gestures

/// One frame of input: the pointer's events, and the modifiers held.
#[derive(Clone, Default)]
struct Frame {
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
}

fn moved(at: Pos2) -> Frame {
    Frame {
        events: vec![egui::Event::PointerMoved(at)],
        ..Default::default()
    }
}

fn button(
    at: Pos2,
    button: egui::PointerButton,
    pressed: bool,
    modifiers: egui::Modifiers,
) -> Frame {
    Frame {
        events: vec![egui::Event::PointerButton {
            pos: at,
            button,
            pressed,
            modifiers,
        }],
        modifiers,
    }
}

/// A press at `from`, a drag through `via` to `to`, the release there, and
/// a frame after it.
fn drag(
    with: egui::PointerButton,
    from: Pos2,
    via: Pos2,
    to: Pos2,
    modifiers: egui::Modifiers,
) -> Vec<Frame> {
    let held = |mut f: Frame| {
        f.modifiers = modifiers;
        f
    };
    vec![
        held(moved(from)),
        button(from, with, true, modifiers),
        held(moved(via)),
        held(moved(to)),
        button(to, with, false, modifiers),
        held(moved(to)),
    ]
}

/// A press and release at `at` without moving, and a frame after it.
fn click(at: Pos2, modifiers: egui::Modifiers) -> Vec<Frame> {
    let mut hover = moved(at);
    hover.modifiers = modifiers;
    vec![
        hover.clone(),
        button(at, Primary, true, modifiers),
        button(at, Primary, false, modifiers),
        hover,
    ]
}

/// Run `frames` through egui, over a canvas filling the window as the
/// picture's does, handing each frame's response to `each`.
fn run(frames: &[Frame], mut each: impl FnMut(&egui::Response, &egui::Ui)) {
    let ctx = egui::Context::default();
    // One pass a frame, so `each` sees every frame once.
    ctx.options_mut(|o| o.max_passes = std::num::NonZeroUsize::MIN);
    // A frame with nothing in it first, so the canvas is there to be hit.
    let frames = std::iter::once(Frame::default()).chain(frames.iter().cloned());
    for (i, frame) in frames.enumerate() {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            time: Some(i as f64 / 60.0),
            modifiers: frame.modifiers,
            events: frame.events,
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| {
            let (_, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            each(&response, ui);
        });
    }
}

/// The picture for the gestures: 40 x 20 pixels at 2.5x, so on screen from
/// (100, 50) to (200, 100).
const XF: ImageXform = ImageXform {
    origin: Pos2::new(100.0, 50.0),
    zoom: 2.5,
    size: (40, 20),
};

/// What the region pick's branch did with a frame, before 18c.
fn old_region_pick(
    pick: &mut region_pick::RegionPick,
    response: &egui::Response,
    ui: &egui::Ui,
) -> Option<String> {
    let (origin, zoom, (w, h)) = (XF.origin, XF.zoom, XF.size);
    let primary = egui::PointerButton::Primary;
    let px = response
        .interact_pointer_pos()
        .map(|pos| old_pixel_at(pos, origin, zoom));
    let pressed = ui
        .input(|i| i.pointer.press_origin())
        .map(|pos| old_pixel_at(pos, origin, zoom));
    let buttons = (
        response.drag_started_by(primary),
        response.dragged_by(primary),
        response.drag_stopped(),
    );
    let roi = pick.track(px, pressed, buttons, (w, h))?;
    Some(pick.drawn_message(&roi))
}

/// What the counter's branch did with a frame, before 18c.
fn old_point_counter(
    tool: &mut point_counter::Session,
    response: &egui::Response,
    ui: &egui::Ui,
    map: Option<Rect>,
    autoscroll_was_on: bool,
) {
    let (origin, zoom, (w, h)) = (XF.origin, XF.zoom, XF.size);
    let frame = 0;
    let to_image = |pos| old_image_at(pos, origin, zoom);
    let primary = egui::PointerButton::Primary;
    let at = response.interact_pointer_pos();
    let gesture = point_counter::Gesture {
        at: at.map(to_image),
        pressed_at: ui.input(|i| i.pointer.press_origin()).map(to_image),
        clicked: response.clicked_by(primary),
        alt: ui.input(|i| i.modifiers.alt),
        drag_started: response.drag_started_by(primary),
        dragging: response.dragged_by(primary),
        drag_stopped: response.drag_stopped(),
        taken: autoscroll_was_on || map.zip(at).is_some_and(|(outer, at)| outer.contains(at)),
    };
    tool.track(gesture, frame, point_counter::HIT_RADIUS / zoom, (w, h));
}

/// The counter's cursor, before 18c.
fn old_counter_cursor(
    tool: &point_counter::Session,
    response: &egui::Response,
) -> egui::CursorIcon {
    let (origin, zoom) = (XF.origin, XF.zoom);
    point_counter::cursor(
        tool,
        response
            .hover_pos()
            .map(|pos| old_image_at(pos, origin, zoom)),
        0,
        point_counter::HIT_RADIUS / zoom,
    )
}

/// What the plot's branch did with a frame, before 18c.
fn old_plot(win: &mut plot::PlotWindow, response: &egui::Response, ui: &egui::Ui) {
    let (origin, zoom, (w, h)) = (XF.origin, XF.zoom, XF.size);
    let add = ui.input(|i| i.modifiers.shift);
    if let Some(pos) = response.interact_pointer_pos() {
        let px = old_pixel_at(pos, origin, zoom);
        let primary = egui::PointerButton::Primary;
        match win.drag {
            None if response.drag_started_by(primary) => win.drag = Some((px, px)),
            Some((from, _)) if response.dragged_by(primary) => win.drag = Some((from, px)),
            _ => {}
        }
    }
    if response.drag_stopped() {
        if let Some((from, to)) = win.drag.take() {
            if let Some(roi) = plot::drag_region(from, to, win.shape, w, h) {
                if !add {
                    win.rois.clear();
                }
                win.rois.push(roi);
                win.stale = true;
            }
        }
    }
}

/// Each tool twice, one fed the old way and one the new, after `frames`.
struct Pair<T> {
    old: T,
    new: T,
}

struct Outcome {
    pick: Pair<region_pick::RegionPick>,
    counter: Pair<point_counter::Session>,
    plot: Pair<plot::PlotWindow>,
}

/// Feed `frames` to every tool the old way and the new, asserting after each
/// frame that the two agree: on what the tool holds, on what it asked the app
/// to say, and on the cursor.
fn compare(
    frames: &[Frame],
    counter_at_start: point_counter::Session,
    map: Option<Rect>,
    autoscroll_was_on: bool,
) -> Outcome {
    let seeded = || point_counter::Session::new(counter_at_start.counter.clone());
    let mut out = Outcome {
        pick: Pair {
            old: region_pick(),
            new: region_pick(),
        },
        counter: Pair {
            old: seeded(),
            new: seeded(),
        },
        plot: Pair {
            old: plot_window(SelectionKind::Regions),
            new: plot_window(SelectionKind::Regions),
        },
    };
    let mut n = 0;
    run(frames, |response, ui| {
        // The gesture, as the canvas reads it.
        let gesture = CanvasGesture {
            taken: CanvasGesture::is_taken(autoscroll_was_on, map, response.interact_pointer_pos()),
            ..CanvasGesture::read(response, ui)
        };
        assert_eq!(gesture.hover, response.hover_pos());

        let said = old_region_pick(&mut out.pick.old, response, ui);
        let asked = CanvasTool::track(&mut out.pick.new, &gesture, &XF, 0);
        assert_eq!(asked, said.map(ToolEvent::Status), "frame {n}");
        assert_eq!(out.pick.new.drag, out.pick.old.drag, "frame {n}");
        assert_eq!(out.pick.new.roi, out.pick.old.roi, "frame {n}");

        old_point_counter(&mut out.counter.old, response, ui, map, autoscroll_was_on);
        let asked = CanvasTool::track(&mut out.counter.new, &gesture, &XF, 0);
        assert_eq!(asked, None, "frame {n}");
        assert_eq!(
            out.counter.new.counter.points, out.counter.old.counter.points,
            "frame {n}"
        );
        assert_eq!(out.counter.new.drag, out.counter.old.drag, "frame {n}");
        assert_eq!(
            CanvasTool::cursor(&out.counter.new, gesture.hover, &XF, 0),
            Some(old_counter_cursor(&out.counter.old, response)),
            "frame {n}"
        );

        old_plot(&mut out.plot.old, response, ui);
        let asked = CanvasTool::track(&mut out.plot.new, &gesture, &XF, 0);
        assert_eq!(asked, None, "frame {n}");
        assert_eq!(out.plot.new.drag, out.plot.old.drag, "frame {n}");
        assert_eq!(out.plot.new.rois, out.plot.old.rois, "frame {n}");
        assert_eq!(out.plot.new.stale, out.plot.old.stale, "frame {n}");
        n += 1;
    });
    out
}

/// A counter with one point at image (4.2, 4.2): under screen (110.5, 60.5).
fn counter_with_a_point() -> point_counter::Session {
    let mut tool = counter();
    tool.counter.add(4.2, 4.2, 0);
    tool
}

const NONE: egui::Modifiers = egui::Modifiers::NONE;

/// A left drag across the picture draws the rectangle and the plot's region,
/// and moves the point it started on, exactly as before.
#[test]
fn a_left_drag_does_what_it_did() {
    let frames = drag(
        Primary,
        Pos2::new(110.0, 60.0),
        Pos2::new(140.0, 70.0),
        Pos2::new(170.0, 85.0),
        NONE,
    );
    let out = compare(&frames, counter_with_a_point(), None, false);
    // It was a drag, so each tool did something with it.
    assert!(out.pick.new.roi.is_some(), "a rectangle was drawn");
    assert_eq!(out.plot.new.rois.len(), 1, "a region was drawn");
    assert!(out.plot.new.stale);
    let p = out.counter.new.counter.points[0];
    assert!((p.x, p.y) != (4.2, 4.2), "the point was moved");
}

/// A click adds a point, and draws nothing; Alt+click on it takes it away.
#[test]
fn a_click_and_an_alt_click_do_what_they_did() {
    let mut frames = click(Pos2::new(150.0, 75.0), NONE);
    frames.extend(click(Pos2::new(110.5, 60.5), egui::Modifiers::ALT));
    let out = compare(&frames, counter_with_a_point(), None, false);
    assert!(out.pick.new.roi.is_none() && out.plot.new.rois.is_empty());
    let points = &out.counter.new.counter.points;
    assert_eq!(points.len(), 1, "one added, one deleted: {points:?}");
    assert_eq!((points[0].x, points[0].y), (20.0, 10.0));
}

/// Shift adds a plot region to the last; a middle drag pans and draws none.
#[test]
fn shift_adds_a_region_and_the_middle_button_draws_nothing() {
    let mut frames = drag(
        Primary,
        Pos2::new(105.0, 55.0),
        Pos2::new(120.0, 60.0),
        Pos2::new(130.0, 70.0),
        NONE,
    );
    frames.extend(drag(
        Primary,
        Pos2::new(150.0, 60.0),
        Pos2::new(170.0, 80.0),
        Pos2::new(190.0, 95.0),
        egui::Modifiers::SHIFT,
    ));
    frames.extend(drag(
        Middle,
        Pos2::new(110.5, 60.5),
        Pos2::new(160.0, 80.0),
        Pos2::new(180.0, 90.0),
        NONE,
    ));
    let out = compare(&frames, counter_with_a_point(), None, false);
    assert_eq!(out.plot.new.rois.len(), 2, "the second added to the first");
    let p = out.counter.new.counter.points[0];
    assert_eq!((p.x, p.y), (4.2, 4.2), "a middle drag moves no point");
}

/// A drag that leaves the picture keeps what is on it, and one that starts
/// off it starts where it began.
#[test]
fn a_drag_off_the_picture_does_what_it_did() {
    let mut frames = drag(
        Primary,
        Pos2::new(150.0, 75.0),
        Pos2::new(300.0, 200.0),
        Pos2::new(400.0, 300.0),
        NONE,
    );
    frames.extend(drag(
        Primary,
        Pos2::new(20.0, 20.0),
        Pos2::new(60.0, 40.0),
        Pos2::new(120.0, 70.0),
        NONE,
    ));
    let out = compare(&frames, counter_with_a_point(), None, false);
    assert!(out.pick.new.roi.is_some());
}

/// A click the navigator took, or one that ended autoscroll, adds no point —
/// and a rectangle and a plot, which only drag, go on as they did.
#[test]
fn a_taken_click_does_what_it_did() {
    let map = Rect::from_min_size(Pos2::new(160.0, 80.0), Vec2::new(40.0, 20.0));
    let mut frames = click(Pos2::new(170.0, 90.0), NONE);
    frames.extend(drag(
        Primary,
        Pos2::new(110.0, 60.0),
        Pos2::new(140.0, 70.0),
        Pos2::new(170.0, 85.0),
        NONE,
    ));
    let out = compare(&frames, counter_with_a_point(), Some(map), false);
    assert_eq!(
        out.counter.new.counter.points.len(),
        1,
        "the click was the map's"
    );
    let out = compare(&frames, counter_with_a_point(), None, true);
    assert_eq!(
        out.counter.new.counter.points.len(),
        1,
        "it ended autoscroll"
    );
    let p = out.counter.new.counter.points[0];
    assert_eq!((p.x, p.y), (4.2, 4.2), "and so did the drag");
    assert!(
        out.pick.new.roi.is_some(),
        "a rectangle is drawn regardless"
    );
}

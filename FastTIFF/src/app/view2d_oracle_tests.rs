//! The 2D view's arithmetic, moved from this app into the core crate
//! (`fast_tiff_viewer::view2d`), checked against the code it replaced.
//!
//! [`old`] is that code as it was here, in egui's types: the zoom ladder, the
//! fit and the opening zoom, the zoom glide, the pinch, the pan glide, the
//! autoscroll curve, wheel travel, and the canvas's inline layout — overflow,
//! pan clamping, the image's origin, the visible rect and the UV window and
//! viewport handed to the sync. Each test drives the old and the new over a
//! grid of inputs — fractional zooms, huge images, fractional panels and
//! display scales, degenerate frames — and asks for the same numbers, to the
//! bit. What each step *promises* is pinned with the code, in the core's
//! `view2d` tests; this is what says the move changed nothing.

use super::screen::{Egui as _, Logical as _};
use super::CanvasView;
use fast_tiff_viewer::view2d::{
    fit_zoom, opening_zoom, ImagePx, LogicalPt, LogicalVec, View2d, ZoomLadder,
};
use fast_tiff_viewer::ImageInputs;

/// The code as it was in this app before the move — in `app.rs`, the
/// `kinetic` and `autoscroll` modules (since removed), `Opening::zoom` and
/// the canvas — unchanged but for being gathered here.
mod old {
    pub const ZOOM_LEVELS: [f32; 21] = [
        0.031, 0.042, 0.063, 0.083, 0.125, 0.167, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
        6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
    ];
    const FIT_SNAP: f32 = 0.02;
    const ZOOM_GLIDE_TAU: f32 = 0.035;
    const ZOOM_GLIDE_SNAP: f32 = 0.01;
    const ZOOM_GLIDE_MAX_DT: f32 = 1.0 / 20.0;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZoomGlide {
        pub target: f32,
        pub anchor: egui::Pos2,
    }

    pub fn zoom_ladder(fit: Option<f32>) -> Vec<f32> {
        let mut levels = ZOOM_LEVELS.to_vec();
        if let Some(f) = fit.filter(|f| f.is_finite() && *f > 0.0) {
            if !levels.iter().any(|z| (z - f).abs() <= f * FIT_SNAP) {
                levels.insert(levels.partition_point(|z| *z < f), f);
            }
        }
        levels
    }

    pub fn stepped_zoom(current: f32, dir: i32, fit: Option<f32>) -> f32 {
        let levels = zoom_ladder(fit);
        let nearest = levels
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (**a - current)
                    .abs()
                    .partial_cmp(&(**b - current).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        let next = (nearest as i32 + dir).clamp(0, levels.len() as i32 - 1) as usize;
        levels[next]
    }

    pub fn zoom_bounds(fit: Option<f32>) -> (f32, f32) {
        let ladder = zoom_ladder(fit);
        (ladder[0], ladder[ladder.len() - 1])
    }

    pub fn fit_to_panel(img_w: f32, img_h: f32, avail: egui::Vec2) -> Option<f32> {
        if !(img_w >= 1.0 && img_h >= 1.0 && avail.x >= 1.0 && avail.y >= 1.0) {
            return None;
        }
        let z = (avail.x / img_w).min(avail.y / img_h).min(1.0);
        z.is_finite().then_some(z).filter(|z| *z > 0.0)
    }

    /// `Opening::SizeWindow`'s arm of `Opening::zoom`, with the room known.
    pub fn size_window(image: egui::Vec2, room: egui::Vec2, chrome_h: f32) -> f32 {
        for &z in ZOOM_LEVELS.iter().rev().filter(|&&z| z <= 1.0) {
            if image.x * z <= room.x && image.y * z + chrome_h <= room.y {
                return z;
            }
        }
        ZOOM_LEVELS[0]
    }

    // The app's `kinetic` module.
    const TAU: f32 = 0.12;
    const MIN_SPEED: f32 = 24.0;
    const SMOOTHING: f32 = 0.4;

    #[derive(Default, Clone, Copy, Debug)]
    pub struct Glide {
        velocity: egui::Vec2,
    }

    impl Glide {
        pub fn push(&mut self, delta: egui::Vec2, dt: f32) -> egui::Vec2 {
            if dt > 0.0 && dt.is_finite() {
                let sample = delta / dt;
                self.velocity += (sample - self.velocity) * SMOOTHING;
            }
            delta
        }
        pub fn coast(&mut self, dt: f32) -> egui::Vec2 {
            if !(dt > 0.0 && dt.is_finite() && self.is_moving()) {
                return egui::Vec2::ZERO;
            }
            let step = self.velocity * dt;
            self.velocity *= (-dt / TAU).exp();
            if self.velocity.length() < MIN_SPEED {
                self.velocity = egui::Vec2::ZERO;
            }
            step
        }
        pub fn is_moving(&self) -> bool {
            self.velocity.length() >= MIN_SPEED
        }
        pub fn stop(&mut self) {
            self.velocity = egui::Vec2::ZERO;
        }
        pub fn stop_axes(&mut self, x: bool, y: bool) {
            if x {
                self.velocity.x = 0.0;
            }
            if y {
                self.velocity.y = 0.0;
            }
        }
    }

    // The app's `autoscroll` module.
    const DEAD_ZONE: f32 = 18.0;
    const GAIN: f32 = 24.0;
    const MAX_SPEED: f32 = 40000.0;

    pub fn velocity(anchor: egui::Pos2, pointer: egui::Pos2) -> egui::Vec2 {
        let offset = pointer - anchor;
        let len = offset.length();
        if !len.is_finite() || len <= DEAD_ZONE {
            return egui::Vec2::ZERO;
        }
        let speed = ((len - DEAD_ZONE) * GAIN).min(MAX_SPEED);
        (offset / len) * speed
    }

    /// The fields of the app's `View2d` the arithmetic touched.
    #[derive(Clone, Copy, Debug)]
    pub struct View2d {
        pub zoom: f32,
        pub pan: egui::Vec2,
        pub fit_zoom: Option<f32>,
        pub zoom_reposition: Option<(f32, egui::Pos2)>,
        pub zoom_glide: Option<ZoomGlide>,
        pub scroll_accum: f32,
        pub image_origin: egui::Pos2,
        pub panel_rect: egui::Rect,
        pub pan_glide: Glide,
        pub autoscroll: Option<egui::Pos2>,
    }

    impl Default for View2d {
        fn default() -> Self {
            View2d {
                zoom: 1.0,
                pan: egui::Vec2::ZERO,
                fit_zoom: None,
                zoom_reposition: None,
                zoom_glide: None,
                scroll_accum: 0.0,
                image_origin: egui::Pos2::ZERO,
                panel_rect: egui::Rect::ZERO,
                pan_glide: Glide::default(),
                autoscroll: None,
            }
        }
    }

    impl View2d {
        pub fn zoom_settled(&self) -> f32 {
            self.zoom_glide.map_or(self.zoom, |g| g.target)
        }

        fn zoom_about(&mut self, anchor: egui::Pos2, new_zoom: f32) {
            let p = (anchor - self.image_origin) / self.zoom;
            self.pan = self.panel_rect.min - (anchor - p * new_zoom);
            self.zoom = new_zoom;
        }

        pub fn advance_zoom_glide(&mut self, dt: f32) -> bool {
            let Some(glide) = self.zoom_glide else {
                return false;
            };
            let dt = dt.clamp(0.0, ZOOM_GLIDE_MAX_DT);
            let from = self.zoom;
            if !(from > 0.0 && glide.target > 0.0) {
                self.zoom_glide = None;
                return false;
            }
            let alpha = 1.0 - (-dt / ZOOM_GLIDE_TAU).exp();
            let next = (from.ln() + (glide.target.ln() - from.ln()) * alpha).exp();
            let arrived = !next.is_finite() || (next / glide.target - 1.0).abs() <= ZOOM_GLIDE_SNAP;
            let new_zoom = if arrived { glide.target } else { next };
            self.zoom_about(glide.anchor, new_zoom);
            if arrived {
                self.zoom_glide = None;
            }
            !arrived
        }

        /// `zoom_input`'s pinch.
        pub fn pinch(&mut self, center_pos: egui::Pos2, zoom_delta: f32, translation: egui::Vec2) {
            let old_zoom = self.zoom;
            let (lo, hi) = zoom_bounds(self.fit_zoom);
            let new_zoom = (old_zoom * zoom_delta).clamp(lo, hi);
            let p = (center_pos - self.image_origin) / old_zoom;
            self.pan = self.panel_rect.min - (center_pos - p * new_zoom) - translation;
            self.zoom = new_zoom;
            self.zoom_glide = None;
        }

        /// `zoom_input`'s step, with the pointer at `latest`.
        pub fn step(&mut self, zoom_step: i32, latest: Option<egui::Pos2>) {
            let old_zoom = self.zoom;
            let from = self.zoom_settled();
            let new_zoom = stepped_zoom(from, zoom_step, self.fit_zoom);
            if (new_zoom - from).abs() > f32::EPSILON {
                let cursor = latest
                    .filter(|p| self.panel_rect.contains(*p))
                    .unwrap_or_else(|| self.panel_rect.center());
                self.zoom_glide = Some(ZoomGlide {
                    target: new_zoom,
                    anchor: cursor,
                });
                self.zoom_reposition = Some((old_zoom, cursor));
            }
        }
    }

    /// What the canvas worked out inline for one frame.
    #[derive(Debug)]
    pub struct Frame {
        pub img_px: egui::Vec2,
        pub overflow: egui::Vec2,
        pub pannable: bool,
        /// What the navigator centred a click on.
        pub shown: egui::Vec2,
        pub origin: egui::Pos2,
        pub full_rect: egui::Rect,
        pub visible: egui::Rect,
        pub inputs: Option<fast_tiff_viewer::ImageInputs>,
    }

    /// `canvas` and `image_canvas`, from the panel to the sync's inputs:
    /// the measurement, a drag, the clamp and the placement.
    pub fn frame(
        view: &mut View2d,
        (w, h): (u32, u32),
        available: egui::Vec2,
        panel_rect: egui::Rect,
        drag: Option<egui::Vec2>,
        ppp: f32,
    ) -> Frame {
        view.panel_rect = panel_rect;
        let img_px = egui::vec2(w as f32 * view.zoom, h as f32 * view.zoom);
        let overflow = egui::vec2(
            (img_px.x - available.x - 1.0).max(0.0),
            (img_px.y - available.y - 1.0).max(0.0),
        );
        let pannable = overflow.x > 0.0 || overflow.y > 0.0;
        let shown = egui::vec2(img_px.x.min(available.x), img_px.y.min(available.y));
        if let Some(delta) = drag {
            if pannable {
                view.pan_glide.stop();
                view.pan -= delta;
            }
        }
        view.pan.x = view.pan.x.clamp(0.0, overflow.x);
        view.pan.y = view.pan.y.clamp(0.0, overflow.y);
        let origin = egui::pos2(
            if overflow.x > 0.0 {
                panel_rect.min.x - view.pan.x
            } else {
                panel_rect.min.x + (available.x - img_px.x) * 0.5
            },
            if overflow.y > 0.0 {
                panel_rect.min.y - view.pan.y
            } else {
                panel_rect.min.y + (available.y - img_px.y) * 0.5
            },
        );
        view.image_origin = origin;
        let full_rect = egui::Rect::from_min_size(origin, img_px);
        let visible = full_rect.intersect(panel_rect);
        // What the canvas wrote on the core before each sync, the
        // expression as it was, and the 2D viewport as the core then read it.
        let inputs = visible.is_positive().then(|| {
            let inv = egui::vec2(1.0 / img_px.x.max(1.0), 1.0 / img_px.y.max(1.0));
            let uv_offset: [f32; 2] = ((visible.min - origin) * inv).into();
            let uv_scale: [f32; 2] = (visible.size() * inv).into();
            let viewport_px: [f32; 2] = (visible.size() * ppp).into();
            fast_tiff_viewer::ImageInputs {
                uv_offset,
                uv_scale,
                viewport: fast_tiff_viewer::PhysicalPx::new(viewport_px[0], viewport_px[1]),
            }
        });
        Frame {
            img_px,
            overflow,
            pannable,
            shown,
            origin,
            full_rect,
            visible,
            inputs,
        }
    }

    /// The canvas's swipe and coast, against this frame's overflow.
    pub fn glide(
        view: &mut View2d,
        pan_swipe: egui::Vec2,
        dt: f32,
        overflow: egui::Vec2,
        pannable: bool,
    ) -> bool {
        if !pannable {
            view.pan_glide.stop();
        }
        let motion = if pan_swipe == egui::Vec2::ZERO {
            view.pan_glide.coast(dt)
        } else {
            view.pan_glide.push(pan_swipe, dt)
        };
        if motion != egui::Vec2::ZERO {
            let want = view.pan - motion;
            view.pan.x = want.x.clamp(0.0, overflow.x);
            view.pan.y = want.y.clamp(0.0, overflow.y);
            view.pan_glide.stop_axes(
                (view.pan.x - want.x).abs() > 0.01,
                (view.pan.y - want.y).abs() > 0.01,
            );
        }
        view.pan_glide.is_moving()
    }

    /// The canvas's autoscroll step, with the pointer at `at`.
    pub fn autoscroll(view: &mut View2d, at: egui::Pos2, dt: f32) -> bool {
        let Some(anchor) = view.autoscroll else {
            return false;
        };
        let v = velocity(anchor, at);
        view.pan += v * dt;
        v != egui::Vec2::ZERO
    }

    /// The canvas's Shift+wheel accumulation.
    pub fn scrub_fast(view: &mut View2d, by: f32) -> i32 {
        view.scroll_accum += by;
        let steps = view.scroll_accum.trunc();
        view.scroll_accum -= steps;
        steps as i32
    }

    /// The canvas's plain wheel accumulation.
    pub fn scrub_notches(view: &mut View2d, notches: f32) -> i32 {
        view.scroll_accum -= notches;
        let steps = view.scroll_accum.trunc();
        view.scroll_accum -= steps;
        steps as i32
    }
}

// ------------------------------------------------------------- comparisons

/// What is being compared, worked out only if the comparison fails: the grids
/// are large, and a message formatted for every case would be most of the
/// test's time.
type What<'a> = &'a dyn Fn() -> String;

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

#[track_caller]
fn assert_same(a: f32, b: f32, what: What) {
    assert!(
        same(a, b),
        "{}: core {a:?} ({:#x}) vs old {b:?} ({:#x})",
        what(),
        a.to_bits(),
        b.to_bits()
    );
}

#[track_caller]
fn assert_same_vec(a: egui::Vec2, b: egui::Vec2, what: What) {
    assert_same(a.x, b.x, &|| format!("{}, x", what()));
    assert_same(a.y, b.y, &|| format!("{}, y", what()));
}

#[track_caller]
fn assert_same_pos(a: egui::Pos2, b: egui::Pos2, what: What) {
    assert_same_vec(a.to_vec2(), b.to_vec2(), what);
}

#[track_caller]
fn assert_same_rect(a: egui::Rect, b: egui::Rect, what: What) {
    assert_same_pos(a.min, b.min, &|| format!("{}, min", what()));
    assert_same_pos(a.max, b.max, &|| format!("{}, max", what()));
}

#[track_caller]
fn assert_same_inputs(a: &ImageInputs, b: &ImageInputs, what: What) {
    for i in 0..2 {
        let uv = |name: &str| format!("{}, {name}[{i}]", what());
        assert_same(a.uv_offset[i], b.uv_offset[i], &|| uv("uv_offset"));
        assert_same(a.uv_scale[i], b.uv_scale[i], &|| uv("uv_scale"));
    }
    let viewport = || format!("{}, viewport", what());
    assert_same(a.viewport.width, b.viewport.width, &viewport);
    assert_same(a.viewport.height, b.viewport.height, &viewport);
}

/// Fit factors: none, nonsense, tiny, on and near every level, between them,
/// past either end, and awkward fractions.
fn fits() -> Vec<Option<f32>> {
    let mut out = vec![
        None,
        Some(f32::NAN),
        Some(f32::INFINITY),
        Some(f32::NEG_INFINITY),
        Some(0.0),
        Some(-0.5),
        Some(f32::MIN_POSITIVE),
        Some(1.0e-6),
        Some(0.004),
        Some(0.287),
        Some(1.0 / 3.0),
        Some(0.5391),
        Some(0.999),
        Some(50.0),
        Some(1.0e9),
    ];
    for z in old::ZOOM_LEVELS {
        for k in [0.97, 0.98, 0.9801, 0.99, 1.0, 1.01, 1.0199, 1.02, 1.03] {
            out.push(Some(z * k));
        }
    }
    out
}

/// Zooms a view can be at: the levels and between them, fractional, tiny and
/// huge.
fn zooms() -> Vec<f32> {
    let mut out = vec![
        1.0e-6,
        0.001,
        0.004,
        0.0287,
        0.1,
        0.287,
        1.0 / 3.0,
        0.5391,
        0.7,
        1.0001,
        1.37,
        2.5,
        7.9,
        31.9,
        32.0,
        33.0,
        100.0,
        1.0e6,
    ];
    for z in old::ZOOM_LEVELS {
        out.extend([z, z * 0.999, z * 1.001, z * 1.25]);
    }
    out
}

/// Image sizes, from a pixel to a gigapixel mosaic, square and elongated, and
/// past the point where `f32` holds a width exactly.
fn images() -> Vec<(u32, u32)> {
    vec![
        (1, 1),
        (2, 3),
        (7, 5),
        (512, 512),
        (1000, 800),
        (1920, 1080),
        (4096, 4096),
        (20_000, 4),
        (4, 20_000),
        (40_000, 30_000),
        (100_000, 100_000),
        (16_777_217, 3),
        (3_000_000_000, 2),
        (u32::MAX, u32::MAX),
    ]
}

/// Rooms a panel or a monitor can offer, laid out and not yet.
fn rooms() -> Vec<egui::Vec2> {
    vec![
        egui::vec2(0.0, 0.0),
        egui::vec2(0.5, 0.5),
        egui::vec2(1.0, 1.0),
        egui::vec2(800.0, 0.0),
        egui::vec2(333.3, 222.7),
        egui::vec2(640.0, 480.0),
        egui::vec2(1536.0, 812.8),
        egui::vec2(1920.0, 1080.0),
        egui::vec2(2560.0, 1349.6),
        egui::vec2(16384.0, 9000.0),
        egui::vec2(f32::NAN, 600.0),
        egui::vec2(f32::INFINITY, f32::INFINITY),
    ]
}

/// Panels: at the origin, under a toolbar of fractional height, with an odd
/// size, and one whose `max` the layout rounded away from `min + size` — the
/// room is then the size asked for, as `available_size` gives it.
fn panels() -> Vec<(egui::Rect, egui::Vec2)> {
    let mut out = Vec::new();
    for (min, size) in [
        (egui::pos2(0.0, 0.0), egui::vec2(640.0, 480.0)),
        (egui::pos2(0.0, 33.333), egui::vec2(1536.0, 777.1)),
        (egui::pos2(0.1, 0.2), egui::vec2(0.3, 0.7)),
        (egui::pos2(12.25, 41.0), egui::vec2(1919.75, 1003.5)),
        (egui::pos2(0.0, 28.0), egui::vec2(1.0, 1.0)),
    ] {
        out.push((egui::Rect::from_min_size(min, size), size));
        let rounded = egui::Rect::from_min_max(min, (min + size).round());
        out.push((rounded, size));
    }
    out
}

const SCALES: [f32; 5] = [1.0, 1.25, 1.5, 1.75, 2.0];

// ------------------------------------------------------------------- tests

#[test]
fn the_ladder_steps_and_ends_are_the_old_ones() {
    let mut currents = zooms();
    currents.extend([0.0, -1.0, f32::INFINITY, f32::MIN_POSITIVE]);
    for fit in fits() {
        let ladder = ZoomLadder::of(fit);
        let want = old::zoom_ladder(fit);
        assert_eq!(ladder.rungs().len(), want.len(), "fit {fit:?}");
        for (a, b) in ladder.rungs().iter().zip(&want) {
            assert_same(*a, *b, &|| format!("a rung of the ladder for {fit:?}"));
        }
        let (lo, hi) = ladder.bounds();
        let (old_lo, old_hi) = old::zoom_bounds(fit);
        assert_same(lo, old_lo, &|| format!("the bottom for {fit:?}"));
        assert_same(hi, old_hi, &|| format!("the top for {fit:?}"));
        for &current in &currents {
            for dir in [-2, -1, 0, 1, 2] {
                assert_same(
                    ladder.step(current, dir),
                    old::stepped_zoom(current, dir, fit),
                    &|| format!("a step {dir} from {current} with fit {fit:?}"),
                );
            }
        }
    }
}

#[test]
fn the_fit_and_the_opening_zoom_are_the_old_ones() {
    for (w, h) in images().into_iter().chain([(0, 0), (0, 100), (100, 0)]) {
        let image = ImagePx::new(w, h);
        let as_vec = egui::vec2(w as f32, h as f32);
        for room in rooms() {
            let what = || format!("{w}x{h} in {room:?}");
            let new = fit_zoom(image, room.logical());
            let was = old::fit_to_panel(as_vec.x, as_vec.y, room);
            assert_eq!(
                new.map(f32::to_bits),
                was.map(f32::to_bits),
                "fit of {}",
                what()
            );
            for chrome in [0.0, 33.5, 60.0, 250.0, 2000.0] {
                assert_same(
                    opening_zoom(image, room.logical(), chrome),
                    old::size_window(as_vec, room, chrome),
                    &|| format!("opening zoom of {} under {chrome} of chrome", what()),
                );
            }
        }
    }
}

/// The canvas's layout of a frame — the measurement, a drag, the clamp, the
/// placement and the sync's inputs — over images, zooms, pans, panels and
/// display scales.
#[test]
fn a_frames_layout_is_the_old_one() {
    let pans = [
        egui::vec2(0.0, 0.0),
        egui::vec2(-50.0, -0.25),
        egui::vec2(0.5, 12.75),
        egui::vec2(1234.567, 89.01),
        egui::vec2(1.0e7, 1.0e7),
    ];
    let drags = [
        None,
        Some(egui::vec2(3.5, -7.25)),
        Some(egui::vec2(-1.0e4, 1.0e4)),
    ];
    let mut checked = 0;
    for (w, h) in images() {
        for zoom in zooms() {
            for &(panel, available) in &panels() {
                for pan in pans {
                    for drag in drags {
                        let ppp = SCALES[checked % SCALES.len()];
                        checked += 1;
                        let what = || {
                            format!(
                                "{w}x{h} at {zoom}, pan {pan:?}, drag {drag:?}, \
                                 panel {panel:?} ({available:?}), {ppp} px/pt"
                            )
                        };

                        let mut was = old::View2d {
                            zoom,
                            pan,
                            ..Default::default()
                        };
                        let want = old::frame(&mut was, (w, h), available, panel, drag, ppp);

                        let mut view = View2d::default();
                        view.open_at_level(zoom);
                        view.pan_to(pan.logical());
                        view.set_panel(panel.logical());
                        let layout = view.measure(ImagePx::new(w, h), available.logical());
                        if let Some(delta) = drag.filter(|_| layout.pannable()) {
                            view.drag(delta.logical());
                        }
                        view.clamp_pan(&layout);
                        let placed = view.place(&layout);

                        assert_same_vec(layout.size.egui(), want.img_px, &what);
                        assert_same_vec(layout.overflow.egui(), want.overflow, &what);
                        assert_eq!(layout.pannable(), want.pannable, "{}", what());
                        assert_same_vec(layout.shown().egui(), want.shown, &what);
                        assert_same_vec(view.pan().egui(), was.pan, &what);
                        assert_same_pos(placed.origin.egui(), want.origin, &what);
                        assert_same_pos(view.image_origin().egui(), was.image_origin, &what);
                        assert_same_rect(placed.full.egui(), want.full_rect, &what);
                        assert_same_rect(placed.visible.egui(), want.visible, &what);
                        assert_eq!(placed.is_visible(), want.inputs.is_some(), "{}", what());
                        if let Some(old_inputs) = &want.inputs {
                            assert_same_inputs(&placed.inputs(ppp), old_inputs, &what);
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "only {checked} frames compared");
}

/// Wheel steps and the glides they start, frame after frame, with the canvas
/// placing the picture between them as it does — so each glide frame turns
/// about the origin the last one left.
#[test]
fn zoom_steps_and_their_glides_are_the_old_ones() {
    // A frame's wheel notch (0: none) and its duration.
    let scripts: [&[(i32, f32)]; 6] = [
        &[
            (1, 1.0 / 60.0),
            (0, 1.0 / 60.0),
            (0, 1.0 / 60.0),
            (0, 1.0 / 60.0),
        ],
        &[
            (-1, 1.0 / 144.0),
            (-1, 1.0 / 144.0),
            (0, 1.0 / 144.0),
            (-1, 1.0 / 144.0),
        ],
        &[(1, 0.2), (1, 0.0), (0, 1.0 / 30.0), (-1, 1.0 / 30.0)],
        &[
            (1, f32::NAN),
            (0, 1.0 / 60.0),
            (1, 1.0 / 60.0),
            (1, 1.0 / 60.0),
        ],
        &[
            (-1, 1.0 / 60.0),
            (-1, 1.0 / 60.0),
            (-1, 1.0 / 60.0),
            (-1, 1.0 / 60.0),
        ],
        &[
            (1, 1.0 / 60.0),
            (1, 1.0 / 60.0),
            (1, 1.0 / 60.0),
            (1, 1.0 / 60.0),
        ],
    ];
    let pointers = [
        None,
        Some(egui::pos2(17.5, 300.25)),
        Some(egui::pos2(-40.0, 10.0)),
        Some(egui::pos2(1000.0, 500.0)),
    ];
    let fit_choices = [None, Some(0.287), Some(0.004), Some(0.5391)];
    let mut frames = 0;
    for (w, h) in [(1000, 800), (40_000, 30_000), (7, 5), (100_000, 3)] {
        for start in [1.0e-6, 0.004, 0.287, 1.0 / 3.0, 1.0, 1.37, 31.9] {
            for fit in fit_choices {
                for &(panel, available) in &panels() {
                    for pointer in pointers {
                        for (n, script) in scripts.iter().enumerate() {
                            let ppp = SCALES[frames % SCALES.len()];
                            let mut was = old::View2d {
                                zoom: start,
                                fit_zoom: fit,
                                ..Default::default()
                            };
                            let mut view = CanvasView::default();
                            // The fit's own rung, then the zoom moved off it.
                            if let Some(f) = fit {
                                view.model.open_fitted(f);
                            }
                            view.model.open_at_level(start);
                            assert_eq!(view.model.fit_rung(), fit, "the fixture's fit");
                            for (i, &(notch, dt)) in script.iter().enumerate() {
                                frames += 1;
                                let what = || {
                                    format!(
                                        "{w}x{h} from {start} (fit {fit:?}), pointer \
                                         {pointer:?}, script {n} frame {i}, panel {panel:?}"
                                    )
                                };
                                // The zoom input, before the panels.
                                if notch != 0 {
                                    was.step(notch, pointer);
                                    let at = pointer.map(|p| p.logical());
                                    if let Some(step) = view.model.step_zoom(notch, at) {
                                        view.zoom_reposition = Some(step);
                                    }
                                }
                                assert_same(view.model.zoom_settled(), was.zoom_settled(), &what);
                                match (view.zoom_reposition, was.zoom_reposition) {
                                    (Some(step), Some((from, cursor))) => {
                                        assert_same(step.from, from, &what);
                                        assert_same_pos(step.anchor.egui(), cursor, &what);
                                    }
                                    (None, None) => {}
                                    other => panic!("{}: repositions {other:?}", what()),
                                }
                                let running = view.model.advance_zoom_glide(dt);
                                assert_eq!(running, was.advance_zoom_glide(dt), "{}", what());
                                assert_eq!(
                                    view.model.is_zoom_gliding(),
                                    was.zoom_glide.is_some(),
                                    "{}",
                                    what()
                                );
                                assert_same(view.model.zoom(), was.zoom, &what);
                                assert_same_vec(view.model.pan().egui(), was.pan, &what);
                                // The canvas.
                                let want =
                                    old::frame(&mut was, (w, h), available, panel, None, ppp);
                                view.model.set_panel(panel.logical());
                                let room = available.logical();
                                let layout = view.model.measure(ImagePx::new(w, h), room);
                                view.model.clamp_pan(&layout);
                                let placed = view.model.place(&layout);
                                assert_same_pos(placed.origin.egui(), want.origin, &what);
                                if let Some(old_inputs) = &want.inputs {
                                    assert_same_inputs(&placed.inputs(ppp), old_inputs, &what);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(frames > 10_000, "only {frames} frames compared");
}

/// A pinch from a laid-out view, over zooms, fits, ratios and slides.
#[test]
fn a_pinch_is_the_old_one() {
    // Not NaN, the one ratio the core refuses: the old code stored the NaN
    // zoom, and the next step panicked on it (see the core's
    // `a_pinch_that_is_not_a_ratio_zooms_nothing`).
    let ratios = [0.0, 1.0e-6, 0.5, 0.93, 1.0, 1.0001, 1.07, 2.0, 1.0e6];
    let slides = [
        egui::Vec2::ZERO,
        egui::vec2(3.25, -9.5),
        egui::vec2(-400.0, 1.0e5),
    ];
    let centers = [
        egui::pos2(0.0, 0.0),
        egui::pos2(321.5, 270.25),
        egui::pos2(-5.0, 9000.0),
    ];
    for (w, h) in [(1000, 800), (40_000, 30_000), (3, 100_000)] {
        for zoom in zooms() {
            for fit in [None, Some(0.287), Some(0.004), Some(50.0)] {
                for &(panel, available) in &panels()[..4] {
                    for ratio in ratios {
                        for slide in slides {
                            for center in centers {
                                let what = || {
                                    format!(
                                        "{w}x{h} at {zoom} (fit {fit:?}) by {ratio} about \
                                         {center:?} sliding {slide:?}, panel {panel:?}"
                                    )
                                };
                                let mut was = old::View2d {
                                    zoom,
                                    fit_zoom: fit,
                                    ..Default::default()
                                };
                                old::frame(&mut was, (w, h), available, panel, None, 1.0);
                                let mut view = View2d::default();
                                if let Some(f) = fit {
                                    view.open_fitted(f);
                                }
                                view.open_at_level(zoom);
                                view.set_panel(panel.logical());
                                let room = available.logical();
                                let layout = view.measure(ImagePx::new(w, h), room);
                                view.clamp_pan(&layout);
                                view.place(&layout);

                                was.pinch(center, ratio, slide);
                                view.pinch(center.logical(), ratio, slide.logical());
                                assert_same(view.zoom(), was.zoom, &what);
                                assert_same_vec(view.pan().egui(), was.pan, &what);
                                assert!(!view.is_zoom_gliding());
                            }
                        }
                    }
                }
            }
        }
    }
}

/// One frame of a swipe script: the trackpad's swipe, a drag, and the frame's
/// duration.
type SwipeFrame = (Option<egui::Vec2>, Option<egui::Vec2>, f32);

/// Swipes, coasts and drags, frame after frame, against an overflow that
/// lets them run and one that stops them at an edge.
#[test]
fn the_pan_glide_is_the_old_one() {
    let swipe = |x: f32, y: f32| Some(egui::vec2(x, y));
    let scripts: [&[SwipeFrame]; 5] = [
        &[
            (swipe(-20.0, 0.0), None, 1.0 / 60.0),
            (swipe(-25.0, 3.0), None, 1.0 / 60.0),
            (swipe(-30.0, 4.5), None, 1.0 / 60.0),
        ],
        &[
            (swipe(40.0, -40.0), None, 1.0 / 144.0),
            (swipe(40.0, -40.0), None, 0.0),
            (swipe(40.0, -40.0), None, f32::NAN),
            (swipe(0.2, 0.2), None, 1.0 / 144.0),
        ],
        &[
            (swipe(-300.0, -300.0), None, 1.0 / 30.0),
            (None, swipe(5.0, 5.0), 1.0 / 30.0),
        ],
        &[
            (swipe(1.0e4, -1.0e4), None, 1.0 / 60.0),
            (swipe(1.0e4, -1.0e4), None, 1.0 / 60.0),
        ],
        &[(swipe(-7.5, 0.0), None, 1.0 / 60.0)],
    ];
    let starts = [
        egui::vec2(0.0, 0.0),
        egui::vec2(500.25, 300.5),
        egui::vec2(1.0e9, 1.0e9),
    ];
    let mut frames = 0;
    for (w, h) in [(4000, 3000), (1000, 800), (100_000, 5), (10, 10)] {
        for zoom in [0.287, 1.0, 1.37, 8.0] {
            for start in starts {
                for &(panel, available) in &panels()[..4] {
                    for (n, script) in scripts.iter().enumerate() {
                        let mut was = old::View2d {
                            zoom,
                            pan: start,
                            ..Default::default()
                        };
                        let mut view = View2d::default();
                        view.open_at_level(zoom);
                        view.pan_to(start.logical());
                        // The script, then the coast it leaves.
                        let coast = std::iter::repeat_n((None, None, 1.0 / 60.0), 400);
                        for (i, (swiped, drag, dt)) in
                            script.iter().copied().chain(coast).enumerate()
                        {
                            frames += 1;
                            let what = || {
                                format!(
                                    "{w}x{h} at {zoom} from {start:?}, script {n} frame {i}, \
                                     panel {panel:?}"
                                )
                            };
                            let want = old::frame(&mut was, (w, h), available, panel, drag, 1.0);
                            view.set_panel(panel.logical());
                            let layout = view.measure(ImagePx::new(w, h), available.logical());
                            if let Some(delta) = drag.filter(|_| layout.pannable()) {
                                view.drag(delta.logical());
                            }
                            view.clamp_pan(&layout);
                            view.place(&layout);
                            let s = swiped.unwrap_or(egui::Vec2::ZERO);
                            let moving = view.glide_pan(s.logical(), dt, &layout);
                            let was_moving =
                                old::glide(&mut was, s, dt, want.overflow, want.pannable);
                            assert_eq!(moving, was_moving, "{}", what());
                            assert_same_vec(view.pan().egui(), was.pan, &what);
                            if !moving && i >= script.len() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(frames > 1_000, "only {frames} frames compared");
}

/// The autoscroll curve and the pan it moves, over anchors, pointers and
/// frame times, non-finite ones included.
#[test]
fn autoscroll_is_the_old_one() {
    let points = [
        egui::pos2(0.0, 0.0),
        egui::pos2(400.0, 300.0),
        egui::pos2(417.9, 300.0),
        egui::pos2(418.0, 300.0),
        egui::pos2(418.1, 300.1),
        egui::pos2(-2000.0, 4000.0),
        egui::pos2(1.0e5, -3.0),
        egui::pos2(f32::NAN, 300.0),
        egui::pos2(400.0, f32::INFINITY),
    ];
    let room = egui::vec2(640.0, 480.0);
    for anchor in points {
        for at in points {
            for dt in [0.0, 1.0 / 144.0, 1.0 / 60.0, 0.1] {
                let what = || format!("anchored {anchor:?}, pointer {at:?}, dt {dt}");
                let mut was = old::View2d {
                    pan: egui::vec2(100.0, 100.0),
                    autoscroll: Some(anchor),
                    ..Default::default()
                };
                let mut view = View2d::default();
                view.pan_to(LogicalVec::new(100.0, 100.0));
                let layout = view.measure(ImagePx::new(4000, 4000), room.logical());
                view.toggle_autoscroll(Some(anchor.logical()), &layout);
                let moved = view.autoscroll_by(at.logical(), dt);
                assert_eq!(moved, old::autoscroll(&mut was, at, dt), "{}", what());
                assert_same_vec(view.pan().egui(), was.pan, &what);
            }
        }
    }
    // Off, nothing moves.
    let mut view = View2d::default();
    assert!(!view.autoscroll_by(LogicalPt::new(900.0, 900.0), 0.1));
    assert_eq!(view.pan(), LogicalVec::ZERO);
}

/// Wheel travel in both of the canvas's forms — the fast scroll's added
/// travel and the wheel's subtracted notches — over runs of deltas.
#[test]
fn wheel_travel_is_the_old_one() {
    let deltas = [
        0.0,
        0.4,
        0.4,
        0.4,
        -3.0,
        0.999,
        1.0e-7,
        -1.0e-7,
        2.5,
        -0.5,
        7.25,
        0.1,
        0.2,
        0.3,
        -0.6000001,
        1.0e6,
        -1.0e6,
        0.333_333_34,
    ];
    let mut was = old::View2d::default();
    let mut view = View2d::default();
    for (i, &d) in deltas.iter().enumerate() {
        assert_eq!(
            view.scrub(d),
            old::scrub_fast(&mut was, d),
            "fast scroll {i}"
        );
        assert_eq!(
            view.scrub(-d),
            old::scrub_notches(&mut was, d),
            "notches {i}"
        );
        if i % 5 == 4 {
            view.forget_scrub();
            was.scroll_accum = 0.0;
        }
    }
    // What is left over carries on, to the bit, in both.
    let mut was = old::View2d::default();
    let mut view = View2d::default();
    for &d in &deltas {
        view.scrub(-d);
        old::scrub_notches(&mut was, d);
        view.scrub(d * 0.37);
        old::scrub_fast(&mut was, d * 0.37);
    }
    assert_eq!(view.scrub(0.0), old::scrub_fast(&mut was, 0.0));
    assert_eq!(view.scrub(0.999_999), old::scrub_fast(&mut was, 0.999_999));
}

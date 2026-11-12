//! The central panel: the picture, in 2D or 3D, and what the pointer does to
//! it there.

use super::*;

impl ViewerApp {
    /// Draw the central panel, queueing what the wheel over the picture asks
    /// for (see [`Phase::Canvas`]).
    pub(super) fn canvas(&mut self, ui: &mut egui::Ui, pinching: bool, cmds: &Commands) {
        // Central panel: the image is drawn at exactly `image_size × zoom`. When
        // that fits the panel it's centered (letterboxed); when it's larger
        // (zoomed past what the monitor-capped window can show) it overflows and
        // is pannable by dragging. Aspect ratio is always preserved.
        // Zero inner margin: the window is sized to exactly the image, so any
        // panel padding would make the available area smaller than the image
        // and produce a small spurious pan/overflow even when it should fit.
        egui::CentralPanel::default()
            .frame(egui::Frame::default().inner_margin(egui::Margin::ZERO))
            .show_inside(ui, |ui| {
                if self.core.document.is_none() {
                    ui.centered_and_justified(|ui| {
                        ui.label(TARGET.welcome_text());
                    });
                    return;
                }

                let available = ui.available_size();
                let (panel_rect, response) =
                    ui.allocate_exact_size(available, egui::Sense::click_and_drag());
                self.view.panel_rect = panel_rect;

                // Fit the image to the canvas on open, where the platform fits
                // a fresh picture to its canvas rather than sizing a window
                // around it (the browser: see `Opening::FitCanvas`).
                //
                // Fitted to the canvas *exactly*, not to the nearest fixed level: a
                // 4:3 image in a 16:9 canvas has no fixed level that fills it, so
                // snapping would either crop a sliver or leave a visible margin.
                // The exact factor then joins this image's zoom ladder, so scrolling
                // back out returns to the fitted view instead of stepping past it —
                // which is the whole reason it has to be a rung and not a one-off.
                //
                // Held back until the 2D view is the one on screen: in 3D the zoom
                // is unused, and consuming the flag there would leave the image
                // unfitted when the user came back to 2D.
                let opening = TARGET.opening();
                if opening == platform::Opening::FitCanvas
                    && self.view.pending_initial_fit
                    && self.core.view_mode == ViewMode::Movie
                {
                    let dims = self
                        .core
                        .document
                        .as_ref()
                        .and_then(|l| l.tiff.frames.first())
                        .map(|f| egui::vec2(f.width as f32, f.height as f32));
                    match dims.and_then(|image| opening.zoom(image, Some(available), 0.0)) {
                        Some(z) => {
                            self.view.zoom = z;
                            self.view.fit_zoom = Some(z);
                            self.view.pan = egui::Vec2::ZERO;
                            self.view.pending_initial_fit = false;
                        }
                        // No canvas size yet — it has not been laid out. Come back
                        // next frame rather than fitting to nothing.
                        None => ui.ctx().request_repaint(),
                    }
                }

                // 3D volume view: drive the camera per the active nav mode and paint
                // the GPU ray-march. The 2D pan/UV/scrub path is bypassed.
                if self.core.view_mode == ViewMode::Volume {
                    self.volume_canvas(ui, panel_rect, response);
                } else {
                    self.image_canvas(ui, available, panel_rect, response, pinching, cmds);
                }
            });
    }

    /// The 3D view: the camera driven by this frame's input, the ray-march, and
    /// the coordinate box over it.
    fn volume_canvas(
        &mut self,
        ui: &mut egui::Ui,
        panel_rect: egui::Rect,
        response: egui::Response,
    ) {
        self.core.volume.aspect =
            (panel_rect.width() / panel_rect.height().max(1.0)).clamp(0.1, 10.0);
        let ppp = ui.ctx().pixels_per_point();
        self.core.volume.viewport_px = [
            (panel_rect.width() * ppp).round().max(1.0) as u32,
            (panel_rect.height() * ppp).round().max(1.0) as u32,
        ];
        // The screen, for how much detail a volume is worth keeping.
        if let Some(monitor) = ui.ctx().input(|i| i.viewport().monitor_size) {
            self.core.volume.screen_px = (monitor.x.max(monitor.y) * ppp).round().max(0.0) as u32;
        }

        // Until the first volume is built, show a black loading screen so
        // the heavy decode doesn't freeze on the previous view (`sync_gpu`
        // defers the initial build until after this frame paints).
        if !self.core.volume_built() {
            ui.painter()
                .rect_filled(panel_rect, 0.0, egui::Color32::BLACK);
            ui.painter().text(
                panel_rect.center(),
                egui::Align2::CENTER_CENTER,
                "Loading 3D…",
                egui::FontId::proportional(16.0),
                egui::Color32::from_gray(150),
            );
            ui.ctx().request_repaint();
            return;
        }

        let animating = self.drive_volume_camera(ui, &response, panel_rect);
        // A press, moving or not, is the user reaching for the
        // view: an animation stops for it on the next frame.
        if response.is_pointer_button_down_on() {
            self.volume_ui.pressed = true;
        }
        self.volume_motion(ui.ctx());
        // Where the view is, for a copy of it. Recorded at the one
        // place that knows — a copy happens a frame later, by which
        // time this panel may not be being laid out at all.
        self.volume_ui.rect = Some(panel_rect);
        response.on_hover_cursor(egui::CursorIcon::Crosshair);
        ui.painter()
            .with_clip_rect(panel_rect)
            .add(render::paint_volume_callback(&self.render, panel_rect));
        // Coordinate-box overlay, drawn on top of the ray-march with the
        // 2D painter (aligned via the same camera; see `overlay`).
        if self.volume_ui.prefs.show_coord_box {
            let painter = ui.painter().with_clip_rect(panel_rect);
            self.draw_coord_box(&painter, panel_rect);
        }
        // Keep repainting while a drag or held key keeps the camera moving.
        if animating {
            ui.ctx().request_repaint();
        }
    }

    /// The 2D picture: panning, the navigator, autoscroll, the canvas tools,
    /// the image itself and what is drawn over it, the cursor, and the wheel.
    fn image_canvas(
        &mut self,
        ui: &mut egui::Ui,
        available: egui::Vec2,
        panel_rect: egui::Rect,
        response: egui::Response,
        pinching: bool,
        cmds: &Commands,
    ) {
        let Some(loaded) = &self.core.document else {
            return;
        };
        let (Some(w), Some(h)) = (
            loaded.tiff.frames.first().map(|f| f.width),
            loaded.tiff.frames.first().map(|f| f.height),
        ) else {
            return;
        };

        let img_px = egui::vec2(w as f32 * self.view.zoom, h as f32 * self.view.zoom);
        // A 1px tolerance so sub-pixel rounding between the window size and
        // the panel's available area doesn't register as a pannable overflow.
        let overflow = egui::vec2(
            (img_px.x - available.x - 1.0).max(0.0),
            (img_px.y - available.y - 1.0).max(0.0),
        );
        let pannable = overflow.x > 0.0 || overflow.y > 0.0;

        // A canvas tool takes the *primary* drag: while one has it,
        // dragging on the picture with the left button draws rather than
        // pans. Every other button still pans — see the rule below — and a
        // tool claims it only while it is in use (its window open, or a
        // plugin waiting for a rectangle, which Escape gives up), so this
        // cannot strand someone with an unpannable view.
        //
        // Which tool has it, and which are drawn, is decided here, once,
        // from what each tool claims (see `canvas_tool`), and the drag rule,
        // the tool's gesture, the overlay and the cursor below all go by it.
        let tools = self.tools.arbitrate(pinching);
        let frame = loaded.frame();

        // The navigator's frame box, worked out before anything acts on a
        // drag: a press that lands on the map has to claim the gesture
        // early enough that the picture underneath does not pan as well.
        //
        // Gated on the same `steady_pannable` that decides whether the map
        // is *drawn*, so it is never clickable while invisible — an
        // invisible hit target in the corner of the image would be a
        // mystery rather than a feature. The `was_pannable` bookkeeping
        // moves up here with it, which is where it belongs anyway: it is
        // part of this frame's decision, not of the painting.
        let settled = !self.view.pending_initial_fit;
        let steady_pannable = pannable && self.view.was_pannable && settled;
        self.view.was_pannable = pannable && settled;
        let map = steady_pannable
            .then(|| minimap::frame_box(panel_rect, img_px.x, img_px.y))
            .flatten();

        // Who owns the drag in flight. Decided once, when it starts, and
        // the primary button only: the middle button belongs to panning
        // and to autoscroll everywhere on the picture, including over the
        // map. Without that split a middle-click on the map would both
        // jump the view and arm autoscroll, from one press.
        if response.drag_started() {
            self.view.minimap_drag = response.drag_started_by(egui::PointerButton::Primary)
                && map
                    .zip(response.interact_pointer_pos())
                    .is_some_and(|(outer, at)| outer.contains(at));
        }
        if response.drag_stopped() {
            self.view.minimap_drag = false;
        }
        let on_map = self.view.minimap_drag;

        // Where on the map the pointer is acting, if it is.
        //
        // Two ways in, and the second is easy to miss: egui only calls a
        // gesture a drag once the pointer has moved past its click
        // threshold, so a careful single click never sets `dragged()` at
        // all. Without the `clicked()` arm, clicking the map — the first
        // thing anyone tries — would do nothing.
        //
        // Mid-drag the position is used wherever it is, including outside
        // the map: dragging the view box to an edge and carrying on is how
        // you reach the corner of a large image, and clamping that to the
        // map's bounds would make the gesture stop short of the edge.
        let map_action = map.and_then(|outer| {
            let at = response.interact_pointer_pos()?;
            let taken =
                on_map || (response.clicked_by(egui::PointerButton::Primary) && outer.contains(at));
            taken.then_some((outer, at))
        });

        // Dragging or clicking the navigator moves the view to the pointer.
        //
        // Anywhere in the frame box counts, not only on the view box. At
        // `MIN_VIEW_BOX` that box is a four-point target, which is not
        // something to ask anyone to hit; and "click the map to go there"
        // is what a navigator is for. Grabbing the box and dragging it is
        // then the same code path rather than a second handler, which is
        // what keeps the two gestures from disagreeing about where the
        // view ends up.
        if let Some((outer, at)) = map_action {
            // As with the picture: a hand on it ends any glide.
            self.view.pan_glide.stop();
            // How much of the image is on screen, which is what `pan_for`
            // centres on the clicked point: the panel on an axis that
            // overflows, the whole image on one that fits.
            let view = egui::vec2(img_px.x.min(available.x), img_px.y.min(available.y));
            self.view.pan = minimap::pan_for(outer, img_px, view, at);
        }

        // Autoscroll: the sticky mode a middle-click turns on, where the
        // view drifts towards wherever the pointer sits. See
        // [`autoscroll`] for the curve and why the mode is worth having
        // beside the drag.
        //
        // A *click* toggles it, never a drag, and that is what lets the
        // two gestures share the middle button without a mode switch: a
        // middle press that moves is a pan (the rule below), a middle
        // press that does not is this. egui will not report `clicked` for
        // a pointer that travelled past its click threshold, so the two
        // are mutually exclusive by construction rather than by a guard
        // that could be got wrong.
        //
        // Whether it was on as this frame began is kept for the tools: a
        // left-click or a drag that ends the mode below is spent on that,
        // and must not also add or move a point or start a region
        // (`CanvasGesture::taken`).
        let autoscroll_was_on = self.view.autoscroll.is_some();
        if response.clicked_by(egui::PointerButton::Middle) {
            // Toggling off from anywhere, so a second middle-click always
            // means stop — including the one that would otherwise re-anchor
            // the mode somewhere new and leave no way out but the keyboard.
            self.view.autoscroll = match self.view.autoscroll {
                Some(_) => None,
                // Only where there is something to scroll. A mode that
                // swallowed the click and then did nothing would be
                // indistinguishable from a dead button.
                None if pannable => response.interact_pointer_pos(),
                None => None,
            };
        }
        // Every other way out. A left- or right-click is how anyone leaves
        // this mode by reflex; taking hold of the picture or the map means
        // steering by hand from now on, and leaving the drift running
        // under a drag would have two things moving the view at once;
        // Escape is the way out when the first reflex was wrong; and an
        // image that no longer overflows has nothing left to scroll.
        //
        // A drag start cannot collide with the toggle above — egui reports
        // a gesture as a click or as a drag, never both — so the order of
        // these two blocks does not decide anything.
        if self.view.autoscroll.is_some()
            && (!pannable
                || response.drag_started()
                || response.clicked_by(egui::PointerButton::Primary)
                || response.clicked_by(egui::PointerButton::Secondary)
                || ui.input(|i| i.key_pressed(egui::Key::Escape)))
        {
            self.view.autoscroll = None;
        }
        if let Some(anchor) = self.view.autoscroll {
            // `latest_pos` rather than `hover_pos`: the pointer may be over
            // a panel or off the window entirely while the mode runs, and
            // "keep scrolling the way I left it pointing" is the behaviour
            // every platform's version has.
            if let Some(at) = ui.input(|i| i.pointer.latest_pos()) {
                let dt = ui.input(|i| i.stable_dt);
                // `stable_dt` rather than the raw frame time: this is
                // multiplied by a velocity every frame, so a single long
                // frame — a decode, a window resize — would otherwise land
                // as a jump of hundreds of points.
                let v = autoscroll::velocity(anchor, at);
                // Opposite sign to the drag below, and deliberately: a drag
                // moves the *picture* under the pointer, while this moves
                // the *view* towards it. Pointing below the anchor scrolls
                // down, which means showing a lower part of the image,
                // which means a larger pan.
                self.view.pan += v * dt;
                // The mode animates, so it has to ask for the frames. Only
                // while actually moving: parked in the dead zone it should
                // cost no more than any other idle window.
                if v != egui::Vec2::ZERO {
                    ui.ctx().request_repaint();
                }
            }
        }

        // Drag to pan when the image overflows the panel: any button but the
        // primary one while a tool has it, and not a pinch or a drag the
        // navigator has (see `Arbitration::pans`).
        let pans = tools.pans(
            response.dragged(),
            response.dragged_by(egui::PointerButton::Primary),
            pinching,
            on_map,
        );
        if pannable && pans {
            // Taking hold of the picture ends any glide: the hand on it now
            // decides where it goes.
            self.view.pan_glide.stop();
            self.view.pan -= response.drag_delta();
        }
        self.view.pan.x = self.view.pan.x.clamp(0.0, overflow.x);
        self.view.pan.y = self.view.pan.y.clamp(0.0, overflow.y);

        // Where the image's top-left *would* be if drawn full-size: scrolled
        // by `pan` on an overflowing axis, centered on an axis that fits.
        // (Cached for cursor-centered zoom; may lie outside the panel.)
        let origin = egui::pos2(
            if overflow.x > 0.0 {
                panel_rect.min.x - self.view.pan.x
            } else {
                panel_rect.min.x + (available.x - img_px.x) * 0.5
            },
            if overflow.y > 0.0 {
                panel_rect.min.y - self.view.pan.y
            } else {
                panel_rect.min.y + (available.y - img_px.y) * 0.5
            },
        );
        self.view.image_origin = origin;

        // The tool that has the primary button, now that the transform for
        // this frame is settled: this frame of the pointer, in the picture's
        // pixels. Not a gesture the navigator took: a drag that began on the
        // map is moving the view rather than measuring anything, and the
        // other buttons pan (see the rule above). Without either guard one
        // gesture would do two things.
        let xf = ImageXform {
            origin,
            zoom: self.view.zoom,
            size: (w, h),
        };
        let gesture = CanvasGesture {
            // A click on the navigator moves the view (a drag that began
            // there never reaches a tool: `on_map`), and one that ends
            // autoscroll only ends it.
            taken: CanvasGesture::is_taken(autoscroll_was_on, map, response.interact_pointer_pos()),
            ..CanvasGesture::read(&response, ui)
        };
        if let Some(slot) = tools.routed(on_map) {
            let event = self
                .tools
                .get_mut(slot)
                .and_then(|tool| tool.track(&gesture, &xf, frame));
            if let Some(event) = event {
                self.on_tool_event(slot, event);
            }
        }

        // Render into the on-screen *visible* rectangle only, and pan/zoom
        // via UVs. Drawing into an oversized rect doesn't work: the callback
        // viewport is clamped to the framebuffer, which would just squash the
        // whole image back to fit instead of zooming.
        let full_rect = egui::Rect::from_min_size(origin, img_px);
        let visible = full_rect.intersect(panel_rect);
        if visible.is_positive() {
            let inv = egui::vec2(1.0 / img_px.x.max(1.0), 1.0 / img_px.y.max(1.0));
            self.core.uv_offset = ((visible.min - origin) * inv).into();
            self.core.uv_scale = (visible.size() * inv).into();
            // How much room there is to draw it, which is what decides how
            // finely an over-large image needs to be sampled.
            let ppp = ui.ctx().pixels_per_point();
            self.core.viewport_px = (visible.size() * ppp).into();
            ui.painter()
                .with_clip_rect(panel_rect)
                .add(render::paint_callback(&self.render, visible));
        }

        // The tools' regions and marks, over the picture and clipped to the
        // panel like everything else here: one tool's regions, then the
        // counted points over them (see `canvas_tool::Layer`).
        for tool in tools.drawn.iter().filter_map(|&slot| self.tools.get(slot)) {
            tool.draw(&ui.painter().with_clip_rect(panel_rect), &xf, frame);
        }

        // Where the view sits in the frame, once that stops being obvious.
        // Drawn after the image so it is over it, and clipped to the panel
        // like everything else here.
        //
        // `map` is `Some` exactly when this should be drawn, and it was
        // decided above, before the drag handlers — see the comment there
        // for why it has to be, and why `steady_pannable` is the condition.
        if let Some(outer) = map {
            minimap::draw(
                &ui.painter().with_clip_rect(panel_rect),
                ui.visuals(),
                outer,
                full_rect,
                visible,
            );
        }

        // The autoscroll anchor, so the mode is visible and so its neutral
        // point is findable. Without a marker the only way to stop
        // scrolling is to hunt for the dead zone by trial and error, and
        // the only way to know the mode is on at all is that the picture
        // is moving — which is also what it looks like when something else
        // has gone wrong.
        if let Some(anchor) = self.view.autoscroll {
            let painter = ui.painter().with_clip_rect(panel_rect);
            let (ink, opposite) = minimap::contrasting_ink(ui.visuals());
            // A ring the size of the dead zone, so the thing on screen is
            // the thing the arithmetic uses: inside the ring is parked,
            // outside it moves. Haloed in the opposite ink for the same
            // reason the map has a backdrop — this sits over the picture,
            // which can be any brightness.
            for (r, w, c) in [
                (
                    autoscroll::DEAD_ZONE + 1.0,
                    3.0,
                    opposite.gamma_multiply(0.6),
                ),
                (autoscroll::DEAD_ZONE, 1.5_f32, ink),
            ] {
                painter.circle_stroke(anchor, r, egui::Stroke::new(w, c));
            }
            painter.circle_filled(anchor, 1.5_f32, ink);
        }

        // What the pointer is over decides the cursor: autoscroll's, the
        // navigator's, the tool's that has the primary button, or the
        // picture's (see `Arbitration::cursor`).
        let hover = gesture.hover;
        let over_map = map.zip(hover).is_some_and(|(outer, at)| outer.contains(at));
        response.on_hover_cursor(tools.cursor(
            self.view.autoscroll.is_some(),
            over_map || on_map,
            |slot| self.tools.get(slot)?.cursor(hover, &xf, frame),
            pannable,
        ));

        // Scrub frames by scrolling over the image (Ctrl+scroll is zoom, so
        // it's excluded). Two modes:
        //   • normal — discrete wheel *events*, so one mouse notch is exactly
        //     one frame (touchpad pixels accumulate to ~one notch);
        //   • Shift (fast-scroll) — ride the smoothed glide, advancing a
        //     ~10%-of-stack step at `FAST_SCROLL_GLIDE_RATE` per second (time-
        //     scaled, so single- and multi-channel stacks scroll the same),
        //     so one notch sums to ~10% while keeping the smooth glide feel.
        // egui remaps Shift+wheel to horizontal scrolling, so the smoothed
        // delta lands on `.x` with the same sign — `x + y` recovers it.
        let mut pan_swipe = egui::Vec2::ZERO;
        if ui.rect_contains_pointer(panel_rect) {
            let shift = ui.input(|i| i.modifiers.shift);
            if shift {
                let (glide, dt) = ui.input(|i| {
                    let s = i.smooth_scroll_delta;
                    (s.x + s.y, i.stable_dt)
                });
                if glide != 0.0 {
                    // ~10% of the stack per notch, spread across the glide.
                    let n_frames = self
                        .core
                        .document
                        .as_ref()
                        .map(|l| l.display.dims().frames)
                        .unwrap_or(1);
                    let fast_step = (n_frames as f64 * FAST_SCROLL_RATE).max(1.0);
                    // glide < 0 is scroll-down → advance frames. Advance at a
                    // fixed rate *per second* (scaled by the frame time), so
                    // the jump depends only on the glide's real-time duration
                    // — identical for single- and multi-channel stacks despite
                    // their different render speeds. Fractions accumulate so
                    // short stacks still move.
                    let dir = if glide < 0.0 { 1.0 } else { -1.0 };
                    self.view.scroll_accum +=
                        (dir * fast_step * FAST_SCROLL_GLIDE_RATE * dt as f64) as f32;
                    let steps = self.view.scroll_accum.trunc();
                    self.view.scroll_accum -= steps;
                    let steps = steps as i32;
                    if steps != 0 {
                        cmds.push(UiCommand::StepFrames(steps));
                    }
                }
            } else {
                // Which device sent this, and therefore what it means —
                // see `scroll`.
                let scroll::Wheel { swipe, notches } =
                    ui.input(|i| scroll::classify(&i.events, pannable));
                pan_swipe = swipe;
                // egui scroll is +y up; we scrub the next frame on scroll-down.
                self.view.scroll_accum -= notches;
                let steps = self.view.scroll_accum.trunc();
                self.view.scroll_accum -= steps;
                let steps = steps as i32;
                if steps != 0 {
                    cmds.push(UiCommand::StepFrames(steps));
                }
            }
        } else {
            self.view.scroll_accum = 0.0;
        }

        // Move the field of view by swipe, and keep moving for a moment
        // after the swipe stops. Deliberately outside the hover test above:
        // a flick that carries the pointer off the image should still
        // coast, and stopping dead at the edge of the panel would feel like
        // the picture had hit something.
        //
        // The sign matches dragging — the picture follows the fingers —
        // because they are the same gesture done two ways, and having them
        // disagree would be worse than either choice.
        let dt = ui.input(|i| i.stable_dt);
        if !pannable {
            self.view.pan_glide.stop();
        }
        let motion = if pan_swipe == egui::Vec2::ZERO {
            self.view.pan_glide.coast(dt)
        } else {
            self.view.pan_glide.push(pan_swipe, dt)
        };
        if motion != egui::Vec2::ZERO {
            let want = self.view.pan - motion;
            self.view.pan.x = want.x.clamp(0.0, overflow.x);
            self.view.pan.y = want.y.clamp(0.0, overflow.y);
            // Clamped means that axis ran into the edge of the image. Stop
            // it there: a glide left wound up against a bound would spring
            // forward again the moment the bound moved.
            self.view.pan_glide.stop_axes(
                (self.view.pan.x - want.x).abs() > 0.01,
                (self.view.pan.y - want.y).abs() > 0.01,
            );
        }
        // A coast is motion with no input behind it, so nothing else will
        // ask for the next frame.
        if self.view.pan_glide.is_moving() {
            ui.ctx().request_repaint();
        }
    }
}

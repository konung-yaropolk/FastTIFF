//! The tools that take the primary button on the 2D picture, and the one
//! place it is decided which of them has it.
//!
//! Three tools draw on the picture with the left button: the rectangle a
//! plugin asked for ([`region_pick`]), the point counter ([`point_counter`])
//! and a plot's regions ([`plot`]). Each is a [`CanvasTool`]: how strongly it
//! claims the button ([`Priority`]), what it draws ([`Layer`]), and what it
//! does with a frame of the pointer, the cursor, Ctrl/Cmd+Z, a change of axes,
//! and a frame's tick.
//!
//! The canvas asks [`Tools::arbitrate`] once a frame which tool has the button
//! and which are drawn, and routes the pointer, draws the overlay and sets the
//! cursor from that one answer. Who wins is [`owner`], a function of the
//! tools' claims and nothing else, so the rule is written once and tested
//! without a window rather than restated by each of the three.
//!
//! The pointer reaches a tool as a [`CanvasGesture`], read from egui's
//! response once a frame, and goes into the image through one [`ImageXform`],
//! which has both the whole-pixel and the continuous conversion.
//!
//! # A new tool
//!
//! A type implementing [`CanvasTool`], a field of [`Tools`] holding it while
//! it is in use, and a [`ToolSlot`] naming it. The compiler asks for the rest:
//! the slot's arms in [`Tools::get`], [`Tools::get_mut`] and [`Tools::close`],
//! and the field's place in [`Tools::slots`], which every dispatcher goes by.
//! Its [`Priority`] places it among the others; the canvas does not change.
//!
//! # When each hook runs
//!
//! | hook | when | dispatched by |
//! |---|---|---|
//! | [`undo`](CanvasTool::undo) | [`UiCommand::UndoTool`], at [`Phase::ToolWindows`] | [`Tools::undo`] |
//! | [`axes_changed`](CanvasTool::axes_changed) | [`UiCommand::SetDimensionOrder`], at [`Phase::Playback`] | [`Tools::axes_changed`] |
//! | [`tick`](CanvasTool::tick) | after [`Phase::Playback`], once the view is settled | [`ViewerApp::tick_tools`] |
//! | [`track`](CanvasTool::track), [`draw`](CanvasTool::draw), [`cursor`](CanvasTool::cursor) | the canvas, before [`Phase::Canvas`] | [`Tools::arbitrate`] |
//!
//! No tool outlives the document it was used on, on the desktop or in a
//! browser: a new one puts them all away ([`Tools::reset`]).
//!
//! A tool's window is not one of its hooks. The plot's and the counter's are
//! drawn with the other tool windows, after [`Phase::Toolbar`], and each acts
//! on the whole app while it is drawn — a plot runs its plugin again, the
//! counter writes its points to a file — which a hook holding only its tool
//! cannot do.
//!
//! [`ViewerApp::tick_tools`]: super::ViewerApp::tick_tools

use super::*;
use egui::{CursorIcon, Painter, Pos2, Rect, Vec2};
use fasttiff_plugin_api::{Params, Roi};

/// How strongly a tool claims the primary button. Of two tools that want it,
/// the one whose priority comes first in this list has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Priority {
    /// What the user was just asked to do: a rectangle a plugin asked for.
    Asked,
    /// A tool the user opened by hand: the point counter. Ahead of a plot,
    /// whose regions can be drawn again once the counter is closed.
    Opened,
    /// A selection offered with a result: the regions of a plot whose plugin
    /// asked for them.
    Offered,
}

/// What a tool draws over the picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layer {
    /// Regions, of which one tool's are drawn: the one with the highest
    /// priority. A rectangle being drawn for a plugin hides a plot's regions
    /// until it is done.
    Regions,
    /// Marks, drawn over any regions: they are what the pointer is acting on
    /// while their tool is open.
    Marks,
}

impl Layer {
    /// Every layer, in the order they are drawn.
    pub(super) const ALL: [Layer; 2] = [Layer::Regions, Layer::Marks];
}

/// What a tool on the canvas claims this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Claim {
    pub(super) priority: Priority,
    pub(super) layer: Layer,
}

/// Of these claims, the one that wins: the highest priority, and of equals
/// the first.
///
/// The one rule for who has the canvas. [`arbitrate`] asks it once for the
/// pointer, which is also whose cursor is shown, and once per [`Layer`] for
/// what is drawn.
pub(super) fn owner<K>(claims: impl IntoIterator<Item = (K, Claim)>) -> Option<K> {
    // `min_by_key` keeps the first of equal keys.
    claims
        .into_iter()
        .min_by_key(|(_, claim)| claim.priority)
        .map(|(key, _)| key)
}

/// Who has the canvas this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Arbitration<K> {
    /// The tool the primary button belongs to: its gesture, and its cursor.
    ///
    /// No tool has it during a pinch. egui synthesises a pointer from the
    /// first touch, so the two fingers would otherwise also be a press on
    /// the picture.
    pub(super) owner: Option<K>,
    /// The tools drawn over the picture, in the order they are drawn: each
    /// layer's own winner. A pinch leaves them drawn.
    pub(super) drawn: Vec<K>,
}

/// Who has the pointer and who is drawn, from what the tools claim and
/// whether a pinch is under way.
pub(super) fn arbitrate<K: Copy>(claims: &[(K, Claim)], pinching: bool) -> Arbitration<K> {
    Arbitration {
        owner: if pinching {
            None
        } else {
            owner(claims.iter().copied())
        },
        drawn: Layer::ALL
            .iter()
            .filter_map(|&layer| owner(claims.iter().copied().filter(|(_, c)| c.layer == layer)))
            .collect(),
    }
}

/// What the canvas does with the answer: the pointer, the pan and the cursor.
/// Apart from the canvas, so the rules are tested without a window.
impl<K: Copy> Arbitration<K> {
    /// The tool this frame's pointer goes to: the owner, unless the drag in
    /// flight began on the navigator (`on_map`). That drag is moving the view
    /// rather than measuring anything, and given to a tool as well, one
    /// gesture would do two things.
    pub(super) fn routed(&self, on_map: bool) -> Option<K> {
        self.owner.filter(|_| !on_map)
    }

    /// Whether this frame's drag pans the picture, where it overflows:
    /// `dragged` by any button, `by_primary` by the primary one.
    ///
    /// **Any button pans**, with one exception: the primary one belongs to the
    /// tool that has it. Spelling the rule out this way rather than leaving it
    /// to a drag at all is what gives the middle button a pan that survives the
    /// tools — and the middle button is the one this matters for, being what
    /// CAD and image editors use to pan and what the 3D view's own CAD mode
    /// already maps to pan (`camera.rs`). It is also why the tools act on the
    /// primary button specifically ([`CanvasTool::track`]): without that, a
    /// middle-drag would pan and draw a region at once.
    ///
    /// Not a drag the navigator has (`on_map`), which moves the view by the
    /// map instead; and not during a pinch: egui synthesises a pointer from the
    /// first touch, so a two-finger pan arrives as a drag as well and would be
    /// applied twice.
    pub(super) fn pans(
        &self,
        dragged: bool,
        by_primary: bool,
        pinching: bool,
        on_map: bool,
    ) -> bool {
        dragged && !(self.owner.is_some() && by_primary) && !pinching && !on_map
    }

    /// The cursor over the picture. Three answers, in the order of what has
    /// the gesture: the autoscroll mode owns the pointer wherever it is, the
    /// navigator owns it over the map and while it has the drag (`navigator`),
    /// and otherwise it is the picture's — or the owner's, when `tool_cursor`
    /// gives it one. The picture's is a hand where it pans (`pannable`), and
    /// cross-hairs where it does not.
    pub(super) fn cursor(
        &self,
        autoscroll: bool,
        navigator: bool,
        tool_cursor: impl FnOnce(K) -> Option<CursorIcon>,
        pannable: bool,
    ) -> CursorIcon {
        if autoscroll {
            CursorIcon::AllScroll
        } else if navigator {
            // The hand means "drag what is under the pointer", and over the
            // map what moves is the view somewhere else. The four-way arrow
            // is what every navigator uses for that.
            CursorIcon::Move
        } else if let Some(icon) = self.owner.and_then(tool_cursor) {
            icon
        } else if pannable {
            CursorIcon::Grab
        } else {
            CursorIcon::Crosshair
        }
    }
}

/// The picture's place on screen this frame: where the top-left corner of
/// its first pixel is, how many screen points one pixel is across, and how
/// many pixels it has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ImageXform {
    /// May lie outside the panel, when the picture is scrolled.
    pub(super) origin: Pos2,
    pub(super) zoom: f32,
    pub(super) size: (u32, u32),
}

impl ImageXform {
    /// The image position under a point on screen, continuous: pixel `(i, j)`
    /// covers `[i, i+1) x [j, j+1)`, so the middle of it is `(i + 0.5, j +
    /// 0.5)`. Where a point is placed: it is where it was clicked, and the
    /// ImageJ export puts it on the same pixel.
    pub(super) fn image_at(&self, pos: Pos2) -> (f32, f32) {
        (
            (pos.x - self.origin.x) / self.zoom,
            (pos.y - self.origin.y) / self.zoom,
        )
    }

    /// The image pixel under a point on screen: [`image_at`](Self::image_at),
    /// floored. Not rounded: the pixel under the cursor is the one it is
    /// *within*, and rounding would give the top-left quarter of a pixel to
    /// its neighbour. Not clamped either: a drag that began off the image
    /// starts where it began, not at the image's corner.
    pub(super) fn pixel_at(&self, pos: Pos2) -> (i64, i64) {
        let (x, y) = self.image_at(pos);
        (x.floor() as i64, y.floor() as i64)
    }

    /// The screen position of an image position: [`image_at`](Self::image_at)
    /// the other way.
    pub(super) fn screen_at(&self, (x, y): (f32, f32)) -> Pos2 {
        self.origin + Vec2::new(x * self.zoom, y * self.zoom)
    }

    /// Where a region's pixels are on screen.
    pub(super) fn roi_rect(&self, roi: &Roi) -> Rect {
        Rect::from_min_size(
            self.origin + Vec2::new(roi.x as f32 * self.zoom, roi.y as f32 * self.zoom),
            Vec2::new(roi.w as f32 * self.zoom, roi.h as f32 * self.zoom),
        )
    }
}

/// One frame of the pointer over the picture, as the tools see it. Read from
/// egui's response once, so every tool and the cursor see the same frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct CanvasGesture {
    /// Where the pointer is acting: over the picture, or holding it.
    pub(super) at: Option<Pos2>,
    /// Where the button went down. egui reports a drag only once the pointer
    /// has moved past its click threshold, so this — not `at` — is where the
    /// user aimed when the drag is first seen.
    pub(super) pressed_at: Option<Pos2>,
    /// Where the pointer rests over the picture, for the cursor.
    pub(super) hover: Option<Pos2>,
    /// A primary click: pressed and released without moving.
    pub(super) clicked: bool,
    /// A primary drag began, or went on, this frame.
    pub(super) drag_started: bool,
    pub(super) dragging: bool,
    /// A drag ended this frame, by any button: egui does not say which.
    pub(super) drag_stopped: bool,
    pub(super) alt: bool,
    pub(super) shift: bool,
    /// Something else had this click or press: the navigator, or leaving
    /// autoscroll. Set by the canvas, which knows about both, with
    /// [`is_taken`](Self::is_taken).
    ///
    /// Every tool honours it alike: a press that is taken starts nothing —
    /// no point, no region — since one gesture doing two things is what it
    /// is there to stop. A drag already in flight goes on wherever the
    /// pointer goes, over the navigator too.
    pub(super) taken: bool,
}

impl CanvasGesture {
    /// This frame of the pointer, from the picture's response. `taken` is
    /// left for the canvas to fill in.
    pub(super) fn read(response: &egui::Response, ui: &egui::Ui) -> Self {
        let primary = egui::PointerButton::Primary;
        let (pressed_at, alt, shift) =
            ui.input(|i| (i.pointer.press_origin(), i.modifiers.alt, i.modifiers.shift));
        CanvasGesture {
            at: response.interact_pointer_pos(),
            pressed_at,
            hover: response.hover_pos(),
            clicked: response.clicked_by(primary),
            drag_started: response.drag_started_by(primary),
            dragging: response.dragged_by(primary),
            drag_stopped: response.drag_stopped(),
            alt,
            shift,
            taken: false,
        }
    }

    /// Whether something else had this frame's click or press: one on the
    /// navigator (`map`, the pointer acting `at`) moves the view, and one
    /// while autoscroll was on only ends it. A drag that began on the
    /// navigator never reaches a tool at all ([`Arbitration::routed`]).
    pub(super) fn is_taken(autoscroll_was_on: bool, map: Option<Rect>, at: Option<Pos2>) -> bool {
        autoscroll_was_on || map.zip(at).is_some_and(|(outer, at)| outer.contains(at))
    }
}

/// What a tool asks of the app, which it cannot do holding only itself.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ToolEvent {
    /// Say this on the status line.
    Status(String),
    /// Say this on the status line, and put the tool away.
    Close(String),
    /// Say that something else is running. The tool stays, to be asked again
    /// once it is done.
    Busy,
    /// Put the tool away, and run the plugin at `plugin` with `params` and
    /// these regions as its selection.
    Run {
        plugin: usize,
        params: Params,
        rois: Vec<Roi>,
    },
}

/// Where each frame of the stack starts, taken just before its axes are
/// reassigned: what stays put about a frame when they are, and so what a tool
/// finds its marks' frames again by ([`CanvasTool::axes_changed`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FrameStarts(Vec<u32>);

impl FrameStarts {
    /// The stack's, as its axes are now.
    pub(super) fn of(stack: &Document) -> Self {
        FrameStarts(fast_tiff_viewer::plugins::builtin::point_counter::frame_starts(stack))
    }

    /// One per frame, in frame order: the ImageJ stack position each starts
    /// at.
    pub(super) fn positions(&self) -> &[u32] {
        &self.0
    }
}

/// Whether Ctrl/Cmd+Z, the tools' undo, was pressed this frame in the viewport
/// `ctx` is drawing: over the main window for [`Tools::undo`], or in a tool's
/// own window for that tool.
///
/// Skipped while text has the keyboard, for the reason [`copy_view::hotkey`]
/// is: in a text field (a point counter group's name) the chord is the
/// field's own undo, and taking back a tool's edit instead would read as the
/// field being broken. Any other widget with the focus — a button just
/// reached with Tab — has no undo of its own, and leaves it to the tools.
pub(super) fn undo_hotkey(ctx: &egui::Context) -> bool {
    if super::copy_view::text_has_the_keyboard(ctx) {
        return false;
    }
    ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z))
}

/// A tool that takes the primary button on the 2D picture while it is in use.
///
/// Every hook but [`priority`](Self::priority), [`layer`](Self::layer),
/// [`track`](Self::track) and [`draw`](Self::draw) has a default that does
/// nothing, so a tool says only what it does.
pub(super) trait CanvasTool {
    /// How strongly it claims the primary button.
    fn priority(&self) -> Priority;

    /// What it draws over the picture.
    fn layer(&self) -> Layer;

    /// Whether it wants the primary button this frame. One that does not is
    /// not on the canvas at all: it takes no gesture, sets no cursor and is
    /// not drawn.
    fn wants_primary(&self) -> bool {
        true
    }

    /// One frame of the pointer, while this tool has the primary button and
    /// the navigator does not have the drag. `frame` is the frame in view.
    ///
    /// The primary button only: the others pan the picture whichever tool has
    /// it, which is what `drag_started` and `dragging` say. Without that, a
    /// middle-drag would pan and draw at once. And not a press something else
    /// had ([`CanvasGesture::taken`]), for the same reason.
    fn track(
        &mut self,
        gesture: &CanvasGesture,
        xf: &ImageXform,
        frame: usize,
    ) -> Option<ToolEvent>;

    /// Draw it over the picture, on the frame in view.
    fn draw(&self, painter: &Painter, xf: &ImageXform, frame: usize);

    /// The cursor over the picture while it has the primary button, `hover`
    /// being where the pointer rests. `None` leaves it to the picture: a hand
    /// where it pans, cross-hairs where it does not.
    fn cursor(&self, _hover: Option<Pos2>, _xf: &ImageXform, _frame: usize) -> Option<CursorIcon> {
        None
    }

    /// The stack's axes were just reassigned. `before` is where its frames
    /// started as they were, and `stack` the stack as it is now.
    fn axes_changed(&mut self, _before: &FrameStarts, _stack: &Document) {}

    /// Whether it keeps an undo history, and so takes Ctrl/Cmd+Z over the main
    /// window while it is open.
    fn has_undo(&self) -> bool {
        false
    }

    /// Take back its last edit.
    fn undo(&mut self) {}

    /// Once a frame, before the canvas, with the view settled: `core` is what
    /// the canvas will draw, and `busy` whether something is running.
    fn tick(&mut self, _ctx: &egui::Context, _core: &Viewer, _busy: bool) -> Option<ToolEvent> {
        None
    }
}

/// Which of the [`Tools`] a tool is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolSlot {
    Plot,
    RegionPick,
    PointCounter,
}

/// The canvas tools, each while it is in use.
///
/// A field per tool rather than a list of `Box<dyn CanvasTool>`: the rest of
/// the app reaches each by its type — a plugin's answer refreshes its plot,
/// the counter's window saves its points — and a list would have to be
/// downcast for that. The canvas and the hooks see a tool only as `dyn
/// CanvasTool`, through [`get`](Self::get), and borrow nothing of the app but
/// `tools` while they do, so the view and the core stay free beside them.
#[derive(Default)]
pub(super) struct Tools {
    /// The plot a plugin returned, while one is on screen.
    ///
    /// It owns the canvas selection too: the regions belong to the plot they
    /// feed, so closing the window takes the tool away with it rather than
    /// leaving a selection behind that nothing is measuring.
    pub(super) plot: Option<plot::PlotWindow>,
    /// A rectangle being drawn because a plugin asked for one. See
    /// [`region_pick`].
    pub(super) region_pick: Option<region_pick::RegionPick>,
    /// The point counter, while its window is open. Like the plot's regions,
    /// the points belong to the window: closing it takes them, and their
    /// marks, away. See [`point_counter`].
    pub(super) point_counter: Option<point_counter::Session>,
}

impl Tools {
    /// The tool in `slot`, if it is in use.
    pub(super) fn get(&self, slot: ToolSlot) -> Option<&dyn CanvasTool> {
        match slot {
            ToolSlot::Plot => self.plot.as_ref().map(|t| t as &dyn CanvasTool),
            ToolSlot::RegionPick => self.region_pick.as_ref().map(|t| t as &dyn CanvasTool),
            ToolSlot::PointCounter => self.point_counter.as_ref().map(|t| t as &dyn CanvasTool),
        }
    }

    pub(super) fn get_mut(&mut self, slot: ToolSlot) -> Option<&mut dyn CanvasTool> {
        match slot {
            ToolSlot::Plot => self.plot.as_mut().map(|t| t as &mut dyn CanvasTool),
            ToolSlot::RegionPick => self.region_pick.as_mut().map(|t| t as &mut dyn CanvasTool),
            ToolSlot::PointCounter => self
                .point_counter
                .as_mut()
                .map(|t| t as &mut dyn CanvasTool),
        }
    }

    /// Put the tool in `slot` away.
    pub(super) fn close(&mut self, slot: ToolSlot) {
        match slot {
            ToolSlot::Plot => self.plot = None,
            ToolSlot::RegionPick => self.region_pick = None,
            ToolSlot::PointCounter => self.point_counter = None,
        }
    }

    /// Every tool, in use or not, with its slot: the list every dispatcher
    /// goes by, in the order the tools are asked. The order decides nothing
    /// but a tie between equal priorities, and there is none.
    ///
    /// Built from the fields, named without `..`: a tool added to `Tools`
    /// does not compile until it is named here, and once named, its binding
    /// is unused (a warning, which CI fails on) until it is listed with its
    /// slot. A tool left out would never be arbitrated, drawn, ticked or
    /// undone.
    pub(super) fn table(&self) -> [(ToolSlot, Option<&dyn CanvasTool>); 3] {
        let Tools {
            plot,
            region_pick,
            point_counter,
        } = self;
        [
            (ToolSlot::Plot, plot.as_ref().map(|t| t as &dyn CanvasTool)),
            (
                ToolSlot::RegionPick,
                region_pick.as_ref().map(|t| t as &dyn CanvasTool),
            ),
            (
                ToolSlot::PointCounter,
                point_counter.as_ref().map(|t| t as &dyn CanvasTool),
            ),
        ]
    }

    /// Every slot, in the order the tools are asked: [`table`](Self::table)'s.
    pub(super) fn slots(&self) -> [ToolSlot; 3] {
        self.table().map(|(slot, _)| slot)
    }

    /// The tools in use, in [`slots`](Self::slots)' order.
    fn each(&self) -> impl Iterator<Item = (ToolSlot, &dyn CanvasTool)> {
        self.table()
            .into_iter()
            .filter_map(|(slot, tool)| tool.map(|tool| (slot, tool)))
    }

    /// Whether no tool is in use.
    pub(super) fn is_empty(&self) -> bool {
        self.each().next().is_none()
    }

    /// What each tool on the canvas claims this frame.
    pub(super) fn claims(&self) -> Vec<(ToolSlot, Claim)> {
        self.each()
            .filter(|(_, tool)| tool.wants_primary())
            .map(|(slot, tool)| {
                let claim = Claim {
                    priority: tool.priority(),
                    layer: tool.layer(),
                };
                (slot, claim)
            })
            .collect()
    }

    /// Which tool has the primary button this frame, and which are drawn.
    pub(super) fn arbitrate(&self, pinching: bool) -> Arbitration<ToolSlot> {
        arbitrate(&self.claims(), pinching)
    }

    /// Whether Ctrl/Cmd+Z over the main window is the tools': whether one in
    /// use keeps an undo history.
    pub(super) fn takes_undo(&self) -> bool {
        self.each().any(|(_, tool)| tool.has_undo())
    }

    /// Ctrl/Cmd+Z: each tool that keeps an undo history takes back its last
    /// edit, whichever of them has the pointer. The point counter is the one
    /// that does.
    pub(super) fn undo(&mut self) {
        for slot in self.slots() {
            if let Some(tool) = self.get_mut(slot).filter(|t| t.has_undo()) {
                tool.undo();
            }
        }
    }

    /// The stack's axes were just reassigned; see
    /// [`CanvasTool::axes_changed`].
    pub(super) fn axes_changed(&mut self, before: &FrameStarts, stack: &Document) {
        for slot in self.slots() {
            if let Some(tool) = self.get_mut(slot) {
                tool.axes_changed(before, stack);
            }
        }
    }

    /// Put every tool away, now that another document has landed, on the
    /// desktop and in a browser alike.
    ///
    /// What each holds is in pixels of the picture it was used on: a plot's
    /// regions, the rectangle being drawn for a plugin, the counted points.
    /// Carried to a different image they would sit somewhere arbitrary — a
    /// plot would go on measuring there, and a counter would mix two images
    /// in one table — which is worse than losing them. Every field, so a tool
    /// added to `Tools` is put away too without being named here.
    pub(super) fn reset(&mut self) {
        *self = Tools::default();
    }
}

impl super::ViewerApp {
    /// Every tool's [`tick`](CanvasTool::tick), once a frame, with the view
    /// settled for the canvas.
    ///
    /// Each slot is looked up as its turn comes rather than all of them
    /// beforehand, since a tool's event changes which tools are in use.
    pub(super) fn tick_tools(&mut self, ctx: &egui::Context) {
        for slot in self.tools.slots() {
            let busy = self.job.is_some();
            let event = match self.tools.get_mut(slot) {
                Some(tool) => tool.tick(ctx, &self.core, busy),
                None => None,
            };
            if let Some(event) = event {
                self.on_tool_event(slot, event);
            }
        }
    }

    /// Do what the tool in `slot` asked.
    pub(super) fn on_tool_event(&mut self, slot: ToolSlot, event: ToolEvent) {
        match event {
            ToolEvent::Status(message) => self.core.status = Some(message),
            ToolEvent::Close(message) => {
                self.tools.close(slot);
                self.core.status = Some(message);
            }
            ToolEvent::Busy => self.core.status = Some(self.busy_message()),
            ToolEvent::Run {
                plugin,
                params,
                rois,
            } => {
                // Away first: in a browser the plugin runs before this
                // returns, and may ask for a rectangle of its own again.
                self.tools.close(slot);
                self.picked = Some((plugin, rois));
                self.run_plugin(plugin, params);
            }
        }
    }
}

#[cfg(test)]
#[path = "canvas_tool_tests.rs"]
mod tests;

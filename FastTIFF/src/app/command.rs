//! What the interface asks for, and the one order it takes effect in.
//!
//! egui lays a frame out through closures, and a closure drawing the toolbar
//! holds a borrow of the app that acting on a click would need a second time:
//! opening a file, starting a plugin, switching to 3D. So the widgets do not
//! act. They push a [`UiCommand`] onto the frame's [`Commands`], and
//! [`ViewerApp::apply`](super::ViewerApp::apply) carries it out once the
//! closure has returned.
//!
//! *When* a command takes effect is part of what it does: a 2D/3D click has to
//! reach the bottom bar on the frame it was made, a copy asked for by key must
//! wait for the windows that could have the keyboard to be drawn, an undo has
//! to land before the picture takes this frame's pointer. That is written down
//! once, here, as data: [`UiCommand::slot`] gives every command a [`Phase`] and
//! a place within it, and the frame applies the phases in [`Phase::ALL`]'s
//! order, each straight after the step it is named for. The frame's steps and
//! its phases are one list, [`FRAME`], which `ViewerApp::draw` takes in turn.

use std::cell::{Cell, RefCell};

use fast_tiff_viewer::roi::LargeImageMode;
use fast_tiff_viewer::{DecodeMode, ViewMode};

use super::windows::WindowId;

/// Which of the two ways of asking for a copy this was. They are one action,
/// taking effect at two points of the frame — see [`UiCommand::slot`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Via {
    /// The toolbar's button.
    Button,
    /// Ctrl/Cmd+C over the main window.
    Hotkey,
}

/// One thing the interface asked for this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum UiCommand {
    /// Show the platform's file picker.
    Open,
    /// Save (on the web, download) the open document.
    Save,
    /// Put the view on the clipboard.
    Copy(Via),
    /// Reveal the folder plugins are installed into. Desktop-only, like the
    /// menu entry that asks for it.
    #[cfg(not(target_arch = "wasm32"))]
    OpenPluginFolder,
    /// Begin the plugin at this registry index: its dialog, or the run.
    StartPlugin(usize),
    /// Open the point counter, or raise it.
    OpenPointCounter,
    /// Take back the last edit of the canvas tools that keep an undo history:
    /// the point counter's.
    UndoTool,
    /// Switch between the movie and the volume.
    SetViewMode(ViewMode),
    /// The playback rate, in frames per second.
    SetFps(f64),
    SetDecodeMode(DecodeMode),
    SetLargeImageMode(LargeImageMode),
    /// Open the pop-up window if it is closed, close it if it is open.
    ToggleWindow(WindowId),
    /// Ask the running job to stop.
    StopJob,
    /// Expand or collapse the channels panel.
    TogglePanel,
    /// Play or pause the movie.
    TogglePlay,
    SetPseudocolor(bool),
    /// The single-channel LUT selector's choice.
    SetGrayLut(usize),
    /// Reassign the axes: (channels, slices, frames).
    SetDimensionOrder(usize, usize, usize),
    /// Move this many frames on, back when negative: the wheel over the
    /// picture.
    StepFrames(i32),
}

/// A point in the frame at which queued commands are applied, named for the
/// step it comes straight after: `Toolbar` is applied once the toolbar has
/// been laid out, `Canvas` once the canvas has. In frame order, which is also
/// their `Ord` — so a command in a phase before `Canvas` takes effect before
/// the canvas is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Phase {
    /// After the toolbar, before the tool windows (a plugin's dialog, its
    /// plot, the point counter) are drawn: so a dialog the click opened, or
    /// the counter it raised, is on screen on the same frame.
    Toolbar,
    /// After the tool windows, and the chords read once they are drawn.
    ///
    /// The chords wait for the windows because whether the keyboard belongs
    /// to text is only known once every window that could hold the caret has
    /// been laid out (see `copy_view::hotkey`). They also land after the
    /// buttons, so an explicit click this frame is not also counted as a
    /// chord, and before the 3-D copy is collected, so a copy asked for by key
    /// is picked up on the same schedule as one asked for by button.
    ToolWindows,
    /// After a 3-D copy has been collected, before the bottom bar.
    ///
    /// The 2D/3D switch waits for the copies and the tool windows, which see
    /// the mode the frame began in, and lands before the bar, which lays its
    /// controls out for the mode just chosen.
    CopyPoll,
    /// After the bottom bar, before the pop-up windows: so a toggle opens or
    /// closes its window on the frame it was clicked.
    BottomBar,
    /// After the pop-up windows and the bar's height bookkeeping, before the
    /// playback clock ticks: so play/pause reaches the clock this frame.
    ///
    /// The bar's colour choices land here too, after the histogram has drawn
    /// its curves in the colours the frame began with, and in time for the
    /// image, which the sync at the end of the frame draws in the new ones.
    /// The panel toggle measures the bar, so it can only come after it.
    Popups,
    /// After the playback clock, before the view is settled for the canvas.
    ///
    /// A new dimension order can leave too few frames for a volume, and the
    /// settling that follows (back to 2D, the animation's step) has to see it
    /// before the canvas is drawn, so the click frame already renders 2D.
    Playback,
    /// After the canvas, before the window is sized and the GPU synced: what
    /// the picture asked for, which the sync then shows.
    Canvas,
}

impl Phase {
    /// Every phase, in the order a frame applies them.
    pub(super) const ALL: [Phase; 7] = [
        Phase::Toolbar,
        Phase::ToolWindows,
        Phase::CopyPoll,
        Phase::BottomBar,
        Phase::Popups,
        Phase::Playback,
        Phase::Canvas,
    ];
}

/// One step of a frame: a part of the interface laid out or brought up to
/// date, or the point at which a phase's commands are applied. What each does
/// is in `ViewerApp::draw`; when, is [`FRAME`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    /// Take in what arrived since the last frame: files dropped on the window,
    /// what the picker and the workers sent, a load that finished.
    Intake,
    /// The wheel, the zoom keys and a pinch, applied to the 2D view. Before any
    /// panel is laid out, so the picture is drawn at the new zoom on the frame
    /// it was asked for; it also says whether a pinch is under way, which the
    /// canvas needs.
    ZoomInput,
    /// Lay the toolbar out.
    Toolbar,
    /// The tool windows: a plugin's dialog, its plot, the point counter. Each
    /// acts on its own buttons as it is drawn.
    ToolWindows,
    /// Read the main window's chords: copy, and the tools' undo.
    Hotkeys,
    /// Collect a 3-D copy, once the GPU has rendered it.
    PollCopy,
    /// Lay the bottom bar out.
    ScrubBar,
    /// Draw the pop-up windows that are open.
    Popups,
    /// Note what the bar's info row drew, to grow the window when that changed
    /// the bar's height.
    NoteInfoRow,
    /// Run the playback clock.
    RunPlayback,
    /// Settle the view for the canvas: back to 2D without a volume, then the
    /// animation's step and the canvas tools' tick.
    SettleView,
    /// Draw the central panel: the picture, in 2D or 3D.
    Canvas,
    /// Size the window, then bring the GPU up to date with the frame.
    EndFrame,
    /// Apply what is due at this phase.
    Apply(Phase),
}

/// A frame, in order. `ViewerApp::draw` takes these steps one after another,
/// and nothing else, so this is the whole of when anything in a frame happens.
///
/// Each phase comes straight after the step it is named for, and its doc says
/// why it has to be there; the tests hold this list to those reasons. A step
/// also comes after the steps whose results it reads: the canvas after the
/// zoom input (a pinch), the clock after the bar (a held scrubber), the end of
/// the frame after both bars.
pub(super) const FRAME: [Step; 20] = [
    Step::Intake,
    Step::ZoomInput,
    Step::Toolbar,
    Step::Apply(Phase::Toolbar),
    Step::ToolWindows,
    Step::Hotkeys,
    Step::Apply(Phase::ToolWindows),
    Step::PollCopy,
    Step::Apply(Phase::CopyPoll),
    Step::ScrubBar,
    Step::Apply(Phase::BottomBar),
    Step::Popups,
    Step::NoteInfoRow,
    Step::Apply(Phase::Popups),
    Step::RunPlayback,
    Step::Apply(Phase::Playback),
    Step::SettleView,
    Step::Canvas,
    Step::Apply(Phase::Canvas),
    Step::EndFrame,
];

impl UiCommand {
    /// When this takes effect: its phase, and its place within it.
    ///
    /// This is the whole of the order, so a new command is placed here and
    /// nowhere else. Within a phase the order is the one the frame has always
    /// had; the commands that share a place are applied as they were pushed,
    /// so of two 2D/3D clicks the later wins.
    pub(super) fn slot(&self) -> (Phase, u8) {
        use Phase::*;
        match self {
            // The toolbar, left to right; at most one of these is clicked in a
            // frame, but the order is fixed rather than left to the layout.
            UiCommand::Open => (Toolbar, 0),
            UiCommand::Save => (Toolbar, 1),
            UiCommand::Copy(Via::Button) => (Toolbar, 2),
            #[cfg(not(target_arch = "wasm32"))]
            UiCommand::OpenPluginFolder => (Toolbar, 3),
            UiCommand::StartPlugin(_) => (Toolbar, 4),
            UiCommand::OpenPointCounter => (Toolbar, 5),
            // The chords, copy first, as the keys are read.
            UiCommand::Copy(Via::Hotkey) => (ToolWindows, 0),
            UiCommand::UndoTool => (ToolWindows, 1),
            UiCommand::SetViewMode(_) => (CopyPoll, 0),
            // The bar's settings, then the window toggles — among which the
            // order means nothing, each window having a flag of its own.
            UiCommand::SetFps(_) => (BottomBar, 0),
            UiCommand::SetDecodeMode(_) => (BottomBar, 1),
            UiCommand::SetLargeImageMode(_) => (BottomBar, 2),
            UiCommand::ToggleWindow(_) => (BottomBar, 3),
            UiCommand::StopJob => (Popups, 0),
            UiCommand::TogglePanel => (Popups, 1),
            UiCommand::TogglePlay => (Popups, 2),
            UiCommand::SetPseudocolor(_) => (Popups, 3),
            UiCommand::SetGrayLut(_) => (Popups, 4),
            UiCommand::SetDimensionOrder(..) => (Playback, 0),
            UiCommand::StepFrames(_) => (Canvas, 0),
        }
    }
}

/// The commands queued this frame.
///
/// Shared rather than `&mut`: a frame's widgets are nested closures, several
/// of them alive at once (the bottom bar's info row is called from inside the
/// bar), and each needs to push.
#[derive(Default)]
pub(super) struct Commands {
    queue: RefCell<Vec<UiCommand>>,
    /// How many phases have been applied, to hold the frame to applying each
    /// of [`Phase::ALL`] once, in its order.
    applied: Cell<usize>,
}

impl Commands {
    pub(super) fn push(&self, command: UiCommand) {
        self.queue.borrow_mut().push(command);
    }

    /// Take what is due once `phase` is reached, in the order to apply it.
    ///
    /// Due means of this phase or an earlier one. An earlier one's command is
    /// one pushed after its phase had already been applied — by a step that
    /// runs later in the frame than the point the command belongs to, which
    /// breaks the order as surely as applying a phase out of turn. A debug
    /// build says so; a release build carries it out here, ahead of this
    /// phase's own, rather than lose it.
    pub(super) fn due(&self, phase: Phase) -> Vec<UiCommand> {
        let n = self.applied.get();
        debug_assert_eq!(
            Phase::ALL.get(n),
            Some(&phase),
            "phase applied out of order: a frame applies each of Phase::ALL once, in order",
        );
        self.applied.set(n + 1);
        let mut queue = self.queue.borrow_mut();
        let (mut due, rest): (Vec<_>, Vec<_>) = queue.drain(..).partition(|c| c.slot().0 <= phase);
        *queue = rest;
        drop(queue);
        debug_assert!(
            due.iter().all(|c| c.slot().0 == phase),
            "command queued after its phase was applied, at {phase:?}: {due:?}",
        );
        // Stable, so commands sharing a place keep the order they came in.
        due.sort_by_key(UiCommand::slot);
        due
    }

    /// The view mode the frame's commands will leave the stack in, if one was
    /// asked for: the last 2D/3D click, which is the one applied last.
    ///
    /// For the toolbar, which lays the rest of itself out for the mode just
    /// clicked rather than one frame late.
    pub(super) fn view_mode(&self) -> Option<ViewMode> {
        self.queue.borrow().iter().rev().find_map(|c| match c {
            UiCommand::SetViewMode(mode) => Some(*mode),
            _ => None,
        })
    }

    /// The end of the frame. Every phase has been applied by now, and with the
    /// last of them every command; anything else is a frame that broke the
    /// order.
    pub(super) fn finish(self) {
        debug_assert_eq!(
            self.applied.get(),
            Phase::ALL.len(),
            "a phase was not applied"
        );
        debug_assert!(self.queue.borrow().is_empty());
    }
}

/// What applying a command may need from the frame it is applied in.
pub(super) struct Applying<'a> {
    pub(super) ctx: &'a egui::Context,
    /// The bottom bar's height as laid out this frame, once it has been.
    pub(super) bar_height: Option<f32>,
}

impl super::ViewerApp {
    /// Apply what is due once `phase` is reached.
    pub(super) fn apply_due(&mut self, phase: Phase, commands: &Commands, at: &Applying<'_>) {
        for command in commands.due(phase) {
            self.apply(command, at);
        }
    }

    /// Carry out one command.
    pub(super) fn apply(&mut self, command: UiCommand, at: &Applying<'_>) {
        match command {
            UiCommand::Open => self.show_open_dialog(at.ctx),
            UiCommand::Save => self.save_as(),
            UiCommand::Copy(_) => self.copy_view(at.ctx),
            #[cfg(not(target_arch = "wasm32"))]
            UiCommand::OpenPluginFolder => self.open_plugin_folder(),
            UiCommand::StartPlugin(index) => self.start_plugin(index),
            UiCommand::OpenPointCounter => self.open_point_counter(at.ctx),
            // Ctrl/Cmd+Z while the main window has the keyboard. The counter's
            // own window takes the chord when it has the keyboard instead — see
            // `point_counter::counter_window`.
            UiCommand::UndoTool => self.tools.undo(),
            UiCommand::SetViewMode(mode) => self.set_view_mode(mode),
            UiCommand::SetFps(fps) => self.core.playback.fps = fps,
            UiCommand::SetDecodeMode(mode) => self.core.decode_mode = mode,
            UiCommand::SetLargeImageMode(mode) => self.core.large_image_mode = mode,
            UiCommand::ToggleWindow(id) => self.windows.toggle(id),
            // The stop button only raises the flag the worker polls: it does
            // not take the job away. The worker still owns the registry and
            // still has to hand it back, and until it does the app must keep
            // showing that something is running — a readout that vanished on
            // the click would claim the work had stopped before it had.
            UiCommand::StopJob => {
                if let Some(job) = &self.job {
                    job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            UiCommand::TogglePanel => {
                self.panel.expanded = !self.panel.expanded;
                // Remember the panel's height *before* it expands/collapses;
                // the next frame (once it's redrawn in the new state) grows or
                // shrinks the window by the difference. This frame still shows
                // the old height, so the actual delta only becomes known next
                // frame. Always known here: the toggle is in the bar, and its
                // phase follows the bar.
                let height = at.bar_height.unwrap_or(self.panel.last_h);
                self.panel.arm_grow(height);
            }
            UiCommand::TogglePlay => {
                self.core.playback.playing = !self.core.playback.playing;
                // Start each play/pause from a clean clock, so the first tick
                // after resuming doesn't jump by however long we were paused
                // and the keeping-up estimate starts neutral.
                // (`decode_parallel` stays latched across a pause — if this
                // stack needed it, it still does.)
                self.core.playback.restart();
            }
            UiCommand::SetPseudocolor(on) => self.core.set_pseudocolor(on),
            UiCommand::SetGrayLut(sel) => self.core.set_gray_lut(sel),
            UiCommand::SetDimensionOrder(c, z, f) => self.set_dimension_order(c, z, f),
            UiCommand::StepFrames(steps) => {
                if let Some(loaded) = &mut self.core.document {
                    loaded.set_frame(loaded.frame().saturating_add_signed(steps as isize));
                }
            }
        }
    }

    /// The 2D/3D switch.
    fn set_view_mode(&mut self, mode: ViewMode) {
        self.core.view_mode = mode;
        // Entering 3D stops movie playback — unless the stack is 4D (a
        // separate time axis, `slices > 1`), where playing animates the volume
        // through time.
        if mode == ViewMode::Volume {
            let is_4d = self
                .core
                .document
                .as_ref()
                .is_some_and(|l| l.display.dims().slices > 1);
            if !is_4d {
                self.core.playback.playing = false;
                self.core.playback.last_time = None;
            }
        }
    }

    /// Reassign the axes. That also refreshes LUTs + status; the volume is
    /// rebuilt at the next sync, which finds the channel layout moved (the
    /// frame axis, i.e. the volume depth, just changed).
    fn set_dimension_order(&mut self, c: usize, z: usize, f: usize) {
        // The tools' marks are on frames of the axes as they were (the
        // counter's points are), so where the frames began is taken first —
        // while a tool is in use, since without one nothing asks.
        let before = self
            .core
            .document
            .as_ref()
            .filter(|_| !self.tools.is_empty())
            .map(super::canvas_tool::FrameStarts::of);
        self.core.set_dimension_order(c, z, f);
        if let (Some(before), Some(stack)) = (before, self.core.document.as_ref()) {
            self.tools.axes_changed(&before, stack);
        }
    }

    /// The Plugins menu's "Open plugin folder": say where it is, and show it.
    #[cfg(not(target_arch = "wasm32"))]
    fn open_plugin_folder(&mut self) {
        match fast_tiff_viewer::plugins::install_dir() {
            Some(dir) => {
                self.report_done(format!("Plugin folder: {}", dir.display()));
                // Best effort: showing the path in the status bar is the part
                // that must not fail, so a file manager that will not launch is
                // not an error.
                let _ = super::open_in_file_manager(&dir);
            }
            None => self.core.status = Some("Could not determine a writable plugin folder".into()),
        }
    }
}

#[cfg(test)]
#[path = "command_tests.rs"]
mod tests;

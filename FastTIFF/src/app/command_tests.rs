//! The order a frame applies what the interface asked for.
//!
//! Each rule here used to be a comment beside a hand-placed block in a
//! 1,800-line function, and moving a block broke it without anything failing.
//! The order is now data, and these hold it to the rules the comments stated:
//! which phase each command lands in ([`UiCommand::slot`]), and where among the
//! frame's own steps each phase is applied ([`FRAME`]). Together they make
//! "before the canvas" mean "in a phase applied before the canvas is drawn",
//! and so on.

use super::*;

/// Every command, one of each kind.
fn one_of_each() -> Vec<UiCommand> {
    vec![
        UiCommand::Open,
        UiCommand::Save,
        UiCommand::Copy(Via::Button),
        #[cfg(not(target_arch = "wasm32"))]
        UiCommand::OpenPluginFolder,
        UiCommand::StartPlugin(3),
        UiCommand::OpenPointCounter,
        UiCommand::Copy(Via::Hotkey),
        UiCommand::UndoTool,
        UiCommand::SetViewMode(ViewMode::Volume),
        UiCommand::SetFps(12.5),
        UiCommand::SetDecodeMode(DecodeMode::Threaded),
        UiCommand::SetLargeImageMode(LargeImageMode::Preload),
        UiCommand::ToggleWindow(WindowId::HISTOGRAM),
        UiCommand::StopJob,
        UiCommand::TogglePanel,
        UiCommand::TogglePlay,
        UiCommand::SetPseudocolor(true),
        UiCommand::SetGrayLut(2),
        UiCommand::SetDimensionOrder(2, 1, 50),
        UiCommand::StepFrames(-1),
    ]
}

/// Queue `commands` in the order given, then drain them as a frame does: one
/// phase at a time, in [`Phase::ALL`]'s order. What comes out, per phase.
fn frame(commands: &[UiCommand]) -> Vec<(Phase, Vec<UiCommand>)> {
    let queue = Commands::default();
    for c in commands {
        queue.push(*c);
    }
    let applied = Phase::ALL.iter().map(|p| (*p, queue.due(*p))).collect();
    queue.finish();
    applied
}

/// The phase `command` is applied in when queued with nothing else.
fn phase_of(command: UiCommand) -> Phase {
    frame(&[command])
        .into_iter()
        .find(|(_, applied)| !applied.is_empty())
        .map(|(p, _)| p)
        .expect("every command is applied in some phase")
}

/// The whole order, written out once: a frame that asked for everything,
/// queued back to front, applies it in this sequence and no other.
#[test]
fn a_frame_applies_its_commands_in_the_documented_order() {
    let mut reversed = one_of_each();
    reversed.reverse();
    let applied = frame(&reversed);
    use Phase::*;
    let expected = vec![
        (
            Toolbar,
            vec![
                UiCommand::Open,
                UiCommand::Save,
                UiCommand::Copy(Via::Button),
                #[cfg(not(target_arch = "wasm32"))]
                UiCommand::OpenPluginFolder,
                UiCommand::StartPlugin(3),
                UiCommand::OpenPointCounter,
            ],
        ),
        (
            ToolWindows,
            vec![UiCommand::Copy(Via::Hotkey), UiCommand::UndoTool],
        ),
        (CopyPoll, vec![UiCommand::SetViewMode(ViewMode::Volume)]),
        (
            BottomBar,
            vec![
                UiCommand::SetFps(12.5),
                UiCommand::SetDecodeMode(DecodeMode::Threaded),
                UiCommand::SetLargeImageMode(LargeImageMode::Preload),
                UiCommand::ToggleWindow(WindowId::HISTOGRAM),
            ],
        ),
        (
            Popups,
            vec![
                UiCommand::StopJob,
                UiCommand::TogglePanel,
                UiCommand::TogglePlay,
                UiCommand::SetPseudocolor(true),
                UiCommand::SetGrayLut(2),
            ],
        ),
        (Playback, vec![UiCommand::SetDimensionOrder(2, 1, 50)]),
        (Canvas, vec![UiCommand::StepFrames(-1)]),
    ];
    assert_eq!(applied, expected);
}

/// The phases are listed in the order they are declared, which is the order
/// `Ord` compares them in and the one `due` holds a frame to.
#[test]
fn the_phases_are_listed_in_frame_order() {
    assert!(Phase::ALL.windows(2).all(|w| w[0] < w[1]));
}

/// "Ctrl/Cmd+C, after the buttons so an explicit click this frame is not also
/// counted as a chord, and before the poll so a copy asked for by key is
/// picked up on the same schedule as one asked for by button."
#[test]
fn a_copy_by_key_follows_the_buttons_and_precedes_the_copy_poll() {
    let by_button = phase_of(UiCommand::Copy(Via::Button));
    let by_key = phase_of(UiCommand::Copy(Via::Hotkey));
    assert!(by_key > by_button, "after the buttons");
    // After the tool windows, which decide whether text has the keyboard.
    assert_eq!(by_key, Phase::ToolWindows);
    // The 3-D copy is collected between ToolWindows and CopyPoll.
    assert!(by_button < Phase::CopyPoll && by_key < Phase::CopyPoll);
}

/// Both copies in one frame are both carried out, the button's first.
#[test]
fn a_click_and_a_chord_in_one_frame_are_two_copies() {
    let applied: Vec<_> = frame(&[UiCommand::Copy(Via::Hotkey), UiCommand::Copy(Via::Button)])
        .into_iter()
        .flat_map(|(_, c)| c)
        .collect();
    assert_eq!(
        applied,
        [UiCommand::Copy(Via::Button), UiCommand::Copy(Via::Hotkey)]
    );
}

/// "Before the picture handles this frame's pointer, so an undo that ends a
/// drag in flight is not followed by one more step of it."
#[test]
fn the_undo_lands_before_the_canvas_takes_the_pointer() {
    assert!(phase_of(UiCommand::UndoTool) < Phase::Canvas);
    // And after the tool windows, where the counter's own window takes the
    // chord first when it has the keyboard.
    assert!(phase_of(UiCommand::UndoTool) > Phase::Toolbar);
}

/// A dialog the toolbar opens is drawn on the same frame: everything the
/// toolbar asks for, bar the 2D/3D switch and the settings toggle, lands
/// before the tool windows.
#[test]
fn the_toolbar_lands_before_the_tool_windows() {
    for c in [
        UiCommand::Open,
        UiCommand::Save,
        UiCommand::Copy(Via::Button),
        UiCommand::StartPlugin(0),
        UiCommand::OpenPointCounter,
    ] {
        assert_eq!(phase_of(c), Phase::Toolbar, "{c:?}");
    }
}

/// The 2D/3D switch waits for the tool windows and the copies, which see the
/// mode the frame began in, and lands before the bottom bar, which lays out
/// for the mode just clicked.
#[test]
fn the_view_mode_lands_after_the_copies_and_before_the_bar() {
    let mode = phase_of(UiCommand::SetViewMode(ViewMode::Volume));
    assert!(mode > phase_of(UiCommand::Copy(Via::Hotkey)));
    assert!(mode < Phase::BottomBar);
}

/// A pop-up toggled in the bar (or the toolbar) opens on the click frame: the
/// toggles land before the pop-ups are drawn, which is before `Popups`.
#[test]
fn the_window_toggles_land_before_the_windows_are_drawn() {
    for id in [
        WindowId::METADATA,
        WindowId::HISTOGRAM,
        WindowId::RENDER_SETTINGS,
    ] {
        assert_eq!(phase_of(UiCommand::ToggleWindow(id)), Phase::BottomBar);
    }
}

/// Play/pause reaches this frame's clock; a dimension order follows the clock
/// and lands before the canvas, so the click frame already draws its layout.
#[test]
fn play_reaches_the_clock_and_the_axes_reach_the_canvas() {
    assert!(phase_of(UiCommand::TogglePlay) < Phase::Playback);
    assert_eq!(
        phase_of(UiCommand::SetDimensionOrder(1, 1, 1)),
        Phase::Playback
    );
    assert!(phase_of(UiCommand::SetDimensionOrder(1, 1, 1)) < Phase::Canvas);
}

/// The panel toggle measures the bar, so it can only come once the bar has
/// been laid out — and its height noted, which happens before `Popups`.
#[test]
fn the_panel_toggle_lands_after_the_bar_is_measured() {
    assert!(phase_of(UiCommand::TogglePanel) > Phase::BottomBar);
}

/// Two clicks in a place: applied as they came, so the later wins — and the
/// toolbar, which lays out for the mode clicked, is told the same one.
#[test]
fn the_last_view_mode_asked_for_wins() {
    let queue = Commands::default();
    queue.push(UiCommand::SetViewMode(ViewMode::Volume));
    queue.push(UiCommand::SetViewMode(ViewMode::Movie));
    assert_eq!(queue.view_mode(), Some(ViewMode::Movie));
    let mut applied = Vec::new();
    for p in Phase::ALL {
        applied.extend(queue.due(p));
    }
    queue.finish();
    assert_eq!(
        applied,
        [
            UiCommand::SetViewMode(ViewMode::Volume),
            UiCommand::SetViewMode(ViewMode::Movie)
        ]
    );
}

#[test]
fn no_view_mode_was_asked_for_until_one_is() {
    let queue = Commands::default();
    queue.push(UiCommand::Open);
    assert_eq!(queue.view_mode(), None);
}

/// A command pushed after its phase has been applied — a step queueing
/// something meant for an earlier point — is a frame that broke the order,
/// and a debug build says so where it is found.
#[test]
#[should_panic(expected = "queued after its phase was applied")]
#[cfg(debug_assertions)]
fn a_command_queued_too_late_for_its_phase_is_caught() {
    let queue = Commands::default();
    queue.push(UiCommand::Copy(Via::Hotkey));
    assert!(queue.due(Phase::Toolbar).is_empty());
    queue.push(UiCommand::Open);
    let _ = queue.due(Phase::ToolWindows);
}

/// A release build carries such a command out at the next point, ahead of
/// that point's own, rather than drop it.
#[test]
#[cfg(not(debug_assertions))]
fn a_command_queued_too_late_for_its_phase_is_applied_at_the_next() {
    let queue = Commands::default();
    queue.push(UiCommand::Copy(Via::Hotkey));
    assert!(queue.due(Phase::Toolbar).is_empty());
    queue.push(UiCommand::Open);
    assert_eq!(
        queue.due(Phase::ToolWindows),
        [UiCommand::Open, UiCommand::Copy(Via::Hotkey)]
    );
}

/// What is not yet due stays queued until its point comes.
#[test]
fn a_later_phase_waits_for_its_point() {
    let queue = Commands::default();
    queue.push(UiCommand::StepFrames(1));
    queue.push(UiCommand::Save);
    for phase in Phase::ALL {
        let due = queue.due(phase);
        match phase {
            Phase::Toolbar => assert_eq!(due, [UiCommand::Save]),
            Phase::Canvas => assert_eq!(due, [UiCommand::StepFrames(1)]),
            _ => assert!(due.is_empty(), "{phase:?}: {due:?}"),
        }
    }
    queue.finish();
}

/// Applying the phases out of order is a frame broken, and says so.
#[test]
#[should_panic(expected = "applied out of order")]
#[cfg(debug_assertions)]
fn applying_a_phase_out_of_order_is_caught() {
    let queue = Commands::default();
    let _ = queue.due(Phase::Toolbar);
    let _ = queue.due(Phase::BottomBar);
}

/// So is a frame that ends with a phase never applied: whatever was queued
/// for it would have been dropped.
#[test]
#[should_panic(expected = "a phase was not applied")]
#[cfg(debug_assertions)]
fn a_frame_that_skips_its_last_phases_is_caught() {
    let queue = Commands::default();
    let _ = queue.due(Phase::Toolbar);
    queue.finish();
}

/// Where `step` is in the frame. Each step is taken once a frame.
fn at(step: Step) -> usize {
    let found: Vec<usize> = (0..FRAME.len()).filter(|&i| FRAME[i] == step).collect();
    assert_eq!(
        found.len(),
        1,
        "{step:?} is taken {} times a frame",
        found.len()
    );
    found[0]
}

/// `a` is taken before `b`.
fn assert_before(a: Step, b: Step) {
    assert!(at(a) < at(b), "{a:?} must come before {b:?}: {FRAME:?}");
}

/// `phase` is applied straight after `steps`, in that order, with nothing
/// between.
fn assert_applied_straight_after(phase: Phase, steps: &[Step]) {
    let i = at(Step::Apply(phase));
    assert!(i >= steps.len(), "{phase:?} must come after {steps:?}");
    assert_eq!(
        &FRAME[i - steps.len()..i],
        steps,
        "{phase:?} must come straight after {steps:?}"
    );
}

/// A frame takes each of its steps once, and applies the phases in
/// [`Phase::ALL`]'s order, which is the one `Commands::due` holds it to.
#[test]
fn a_frame_takes_each_step_once_and_applies_the_phases_in_order() {
    for step in FRAME {
        at(step);
    }
    let applied: Vec<Phase> = FRAME
        .iter()
        .filter_map(|s| match s {
            Step::Apply(phase) => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(applied, Phase::ALL);
}

/// The frame begins by taking delivery of what arrived (a document, among
/// others) before anything is laid out for it, and ends by bringing the GPU up
/// to date with all it did.
#[test]
fn a_frame_begins_with_its_intake_and_ends_with_the_sync() {
    assert_eq!(FRAME.first(), Some(&Step::Intake));
    assert_eq!(FRAME.last(), Some(&Step::EndFrame));
}

/// The zoom input is applied before any panel is laid out, so the picture is
/// drawn at the new zoom on the frame it was asked for, and it tells the canvas
/// whether a pinch is under way. The bar's results — its height, what its info
/// row drew, a held scrubber — are read by the steps after it, and the window
/// is sized from both bars.
#[test]
fn a_step_comes_after_the_steps_whose_results_it_reads() {
    assert_before(Step::ZoomInput, Step::Toolbar);
    assert_before(Step::ZoomInput, Step::ScrubBar);
    assert_before(Step::ZoomInput, Step::Canvas);
    assert_before(Step::ScrubBar, Step::NoteInfoRow);
    assert_before(Step::ScrubBar, Step::RunPlayback);
    assert_before(Step::Toolbar, Step::EndFrame);
    assert_before(Step::ScrubBar, Step::EndFrame);
}

/// What the toolbar asks for lands before the tool windows are drawn, so a
/// dialog the click opened, or the counter it raised, is on screen on the same
/// frame.
#[test]
fn a_toolbar_click_lands_before_the_tool_windows_are_drawn() {
    assert_applied_straight_after(Phase::Toolbar, &[Step::Toolbar]);
    assert_before(Step::Apply(Phase::Toolbar), Step::ToolWindows);
}

/// The chords are read once every window that could hold the caret has been
/// laid out, and applied before the 3-D copy is collected, so a copy asked for
/// by key is picked up on the same schedule as one asked for by button.
#[test]
fn the_chords_are_read_once_the_tool_windows_are_drawn() {
    assert_before(Step::ToolWindows, Step::Hotkeys);
    assert_applied_straight_after(Phase::ToolWindows, &[Step::ToolWindows, Step::Hotkeys]);
    assert_before(Step::Apply(Phase::ToolWindows), Step::PollCopy);
}

/// The 2D/3D switch waits for the copies, which see the mode the frame began
/// in, and lands before the bottom bar, which lays its controls out for the
/// mode just chosen.
#[test]
fn the_mode_switch_lands_after_the_copy_poll_and_before_the_bar() {
    assert_applied_straight_after(Phase::CopyPoll, &[Step::PollCopy]);
    assert_before(Step::Apply(Phase::CopyPoll), Step::ScrubBar);
}

/// A toggle in the bar opens or closes its window on the frame it was clicked.
#[test]
fn a_window_toggle_lands_before_the_pop_ups_are_drawn() {
    assert_applied_straight_after(Phase::BottomBar, &[Step::ScrubBar]);
    assert_before(Step::Apply(Phase::BottomBar), Step::Popups);
}

/// Play/pause reaches this frame's clock. The colour choices land after the
/// histogram has drawn its curves in the colours the frame began with, and the
/// panel toggle after the bar it measures has had its height noted.
#[test]
fn play_reaches_this_frames_clock() {
    assert_applied_straight_after(Phase::Popups, &[Step::Popups, Step::NoteInfoRow]);
    assert_before(Step::Apply(Phase::Popups), Step::RunPlayback);
}

/// A new dimension order is applied after the clock and seen by the settling
/// that follows, so the click frame already draws its layout — in 2D, if the
/// order left too few frames for a volume.
#[test]
fn a_new_axis_order_is_settled_before_the_canvas() {
    assert_applied_straight_after(Phase::Playback, &[Step::RunPlayback]);
    assert_before(Step::Apply(Phase::Playback), Step::SettleView);
    assert_before(Step::SettleView, Step::Canvas);
}

/// What the picture asked for is applied before the sync, which shows it.
#[test]
fn what_the_picture_asked_for_is_synced_this_frame() {
    assert_applied_straight_after(Phase::Canvas, &[Step::Canvas]);
    assert_before(Step::Apply(Phase::Canvas), Step::EndFrame);
}

//! The counter model, one operation at a time, and the undo history across
//! them — the part a user leans on without looking.

use super::*;
use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use std::io::Cursor;

/// A counter with points at `(x, x)` on the given frames, all in group 0,
/// showing only the current frame's.
fn with_points(frames: &[usize]) -> PointCounter {
    let mut c = PointCounter::new();
    c.per_frame = true;
    for (i, &f) in frames.iter().enumerate() {
        c.add(i as f32 + 0.5, i as f32 + 0.5, f);
    }
    c
}

// ------------------------------------------------------------------ groups

#[test]
fn a_new_counter_has_one_selected_group_in_the_first_colour() {
    let c = PointCounter::new();
    assert_eq!(
        c.groups,
        vec![Group {
            name: "Group 1".into(),
            color: [0xe4, 0x1a, 0x1c]
        }]
    );
    assert_eq!(c.selected, 0);
    assert!(c.points.is_empty());
    assert!(
        !c.per_frame,
        "every point on every frame until asked otherwise"
    );
    assert_eq!(PointCounter::default().groups, c.groups);
}

#[test]
fn set1_is_matplotlibs_in_order() {
    let hex: Vec<String> = SET1
        .iter()
        .map(|[r, g, b]| format!("#{r:02x}{g:02x}{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf",
            "#999999"
        ]
    );
}

#[test]
fn a_new_group_is_numbered_coloured_and_selected() {
    let mut c = PointCounter::new();
    assert_eq!(c.add_group(), 1);
    assert_eq!(c.groups[1].name, "Group 2");
    assert_eq!(c.groups[1].color, SET1[1]);
    assert_eq!(c.selected, 1);
    // The colours repeat after the ninth.
    for _ in 2..10 {
        c.add_group();
    }
    assert_eq!(c.groups.len(), 10);
    assert_eq!(c.groups[8].color, SET1[8]);
    assert_eq!(c.groups[9].name, "Group 10");
    assert_eq!(c.groups[9].color, SET1[0]);
    assert_eq!(c.selected, 9);
}

// -------------------------------------------------------------------- edits

#[test]
fn a_point_goes_to_the_selected_group_on_its_frame() {
    let mut c = PointCounter::new();
    assert_eq!(c.add(1.5, 2.5, 0), 0);
    c.add_group();
    assert_eq!(c.add(3.5, 4.5, 2), 1);
    c.selected = 0;
    assert_eq!(c.add(5.5, 6.5, 1), 2);
    assert_eq!(
        c.points,
        vec![
            CountedPoint {
                group: 0,
                x: 1.5,
                y: 2.5,
                frame: Some(0),
                plane: None,
            },
            CountedPoint {
                group: 1,
                x: 3.5,
                y: 4.5,
                frame: Some(2),
                plane: None,
            },
            CountedPoint {
                group: 0,
                x: 5.5,
                y: 6.5,
                frame: Some(1),
                plane: None,
            },
        ]
    );
}

/// The frame is recorded while every point is shown on every frame, so turning
/// "Show only current frame's tags" on later finds each where it was placed.
#[test]
fn the_frame_is_kept_when_every_point_is_shown() {
    let mut c = PointCounter::new();
    assert!(!c.per_frame);
    c.add(1.0, 1.0, 2);
    assert_eq!(c.points[0].frame, Some(2));
    assert!(c.shown_on(0, 0));
    c.per_frame = true;
    assert!(c.shown_on(0, 2));
    assert!(!c.shown_on(0, 0));
}

#[test]
fn a_selection_past_the_last_group_falls_back_to_the_last() {
    let mut c = PointCounter::new();
    c.add_group();
    c.selected = 7;
    c.add(1.0, 1.0, 0);
    assert_eq!(c.points[0].group, 1);
}

#[test]
fn a_drag_moves_the_point_and_is_one_undo_step() {
    let mut c = with_points(&[0, 0]);
    let from = (c.points[1].x, c.points[1].y);
    for step in 1..=5 {
        c.drag_to(1, 10.0 + step as f32, 20.0);
    }
    c.finish_drag(1, from);
    assert_eq!((c.points[1].x, c.points[1].y), (15.0, 20.0));

    assert!(c.undo(), "the whole drag comes back in one step");
    assert_eq!((c.points[1].x, c.points[1].y), from);
    assert_eq!(c.points.len(), 2, "and only the drag");
}

#[test]
fn a_drag_that_ends_where_it_began_records_nothing() {
    let mut c = with_points(&[0, 0]);
    let from = (c.points[0].x, c.points[0].y);
    c.drag_to(0, 50.0, 50.0);
    c.drag_to(0, from.0, from.1);
    c.finish_drag(0, from);
    assert!(c.undo());
    assert_eq!(c.points.len(), 1, "the undo took back the last add instead");
}

#[test]
fn a_drag_of_no_point_is_ignored() {
    let mut c = with_points(&[0]);
    c.drag_to(5, 1.0, 1.0);
    c.finish_drag(5, (0.0, 0.0));
    assert_eq!(c.points.len(), 1);
    assert!(c.undo());
    assert!(c.points.is_empty());
}

#[test]
fn a_deleted_point_comes_back_where_it_was_with_its_number() {
    let mut c = with_points(&[0, 0, 0, 0]);
    let before = c.points.clone();
    assert_eq!(c.number_in_group(3), 4);
    c.delete(1);
    assert_eq!(c.points.len(), 3);
    assert_eq!(c.number_in_group(2), 3, "the last point is now number 3");

    assert!(c.undo());
    assert_eq!(c.points, before);
    assert_eq!(c.number_in_group(1), 2);
    assert_eq!(c.number_in_group(3), 4);
}

#[test]
fn deleting_no_point_records_nothing() {
    let mut c = with_points(&[0]);
    c.delete(3);
    assert_eq!(c.points.len(), 1);
    assert!(c.undo());
    assert!(c.points.is_empty(), "the undo took back the add");
}

#[test]
fn undo_walks_back_adds_moves_and_deletes_in_order() {
    let mut c = PointCounter::new();
    c.add(1.0, 1.0, 0); // add 0
    c.add(2.0, 2.0, 0); // add 1
    c.drag_to(0, 9.0, 9.0);
    c.finish_drag(0, (1.0, 1.0)); // move 0
    c.add_group(); // not undoable
    c.groups[0].name = "Renamed".into(); // not undoable
    c.add(3.0, 3.0, 1); // add 2, in group 1
    c.delete(1); // delete the second point
    let states = [
        vec![(9.0, 9.0, 0), (3.0, 3.0, 1)],
        vec![(9.0, 9.0, 0), (2.0, 2.0, 0), (3.0, 3.0, 1)],
        vec![(9.0, 9.0, 0), (2.0, 2.0, 0)],
        vec![(1.0, 1.0, 0), (2.0, 2.0, 0)],
        vec![(1.0, 1.0, 0)],
        vec![],
    ];
    let now = |c: &PointCounter| -> Vec<(f32, f32, usize)> {
        c.points.iter().map(|p| (p.x, p.y, p.group)).collect()
    };
    assert_eq!(now(&c), states[0]);
    for want in &states[1..] {
        assert!(c.undo());
        assert_eq!(&now(&c), want);
    }
    assert!(!c.undo(), "nothing left to undo");
    // The group and the rename were never in the history.
    assert_eq!(c.groups.len(), 2);
    assert_eq!(c.groups[0].name, "Renamed");
}

/// `points` is public. An edit that no longer fits what is there is skipped
/// rather than applied to the wrong point, or panicking.
#[test]
fn an_edit_that_no_longer_fits_is_skipped() {
    let mut c = with_points(&[0, 0, 0]);
    c.points.truncate(1); // two adds now point past the end
    assert!(c.undo(), "the first add still fits");
    assert!(c.points.is_empty());
    assert!(!c.undo());
}

// --------------------------------------------------------------- the view

#[test]
fn a_point_is_shown_on_its_frame_or_on_every_frame() {
    let mut c = with_points(&[0, 2]);
    assert!(c.shown_on(0, 0));
    assert!(!c.shown_on(0, 2));
    assert!(c.shown_on(1, 2));
    c.per_frame = false;
    assert!(c.shown_on(0, 2));
    assert!(c.shown_on(1, 0));
    assert!(!c.shown_on(9, 0), "no such point");
}

#[test]
fn a_hit_is_the_nearest_shown_point_within_the_radius() {
    let mut c = PointCounter::new();
    c.per_frame = true;
    c.add(10.0, 10.0, 0);
    c.add(13.0, 10.0, 0);
    c.add(11.0, 10.0, 1); // nearest to (11, 10), but on frame 1
    assert_eq!(c.hit(11.0, 10.0, 0, 5.0), Some(0));
    assert_eq!(c.hit(12.0, 10.0, 0, 5.0), Some(1));
    assert_eq!(c.hit(11.0, 10.0, 1, 5.0), Some(2));
    assert_eq!(c.hit(20.0, 10.0, 0, 5.0), None, "too far");
    assert_eq!(
        c.hit(18.0, 10.0, 0, 5.0),
        Some(1),
        "the radius is inclusive"
    );
    // Every point shown on every frame: the one on frame 1 is on frame 0 too.
    c.per_frame = false;
    assert_eq!(c.hit(11.0, 10.0, 0, 5.0), Some(2));
}

#[test]
fn of_two_points_in_one_place_the_one_on_top_is_hit() {
    let mut c = PointCounter::new();
    c.add(5.0, 5.0, 0);
    c.add(5.0, 5.0, 0);
    assert_eq!(c.hit(5.0, 5.0, 0, 1.0), Some(1));
}

#[test]
fn points_are_numbered_within_their_group_over_all_frames() {
    let mut c = PointCounter::new();
    c.add(0.0, 0.0, 0);
    c.add_group();
    c.add(0.0, 0.0, 1);
    c.selected = 0;
    c.add(0.0, 0.0, 2);
    c.selected = 1;
    c.add(0.0, 0.0, 0);
    let numbers: Vec<usize> = (0..4).map(|i| c.number_in_group(i)).collect();
    assert_eq!(numbers, [1, 1, 2, 2]);
    assert_eq!(c.number_in_group(4), 0, "no such point");
}

#[test]
fn counts_are_over_all_frames() {
    let mut c = with_points(&[0, 1, 2, 2]);
    c.add_group();
    c.add(0.0, 0.0, 3);
    assert_eq!(c.count(0), 4);
    assert_eq!(c.count(1), 1);
    assert_eq!(c.count(2), 0);
}

// ---------------------------------------------------------------- exports

#[test]
fn the_table_is_tab_separated_with_a_header() {
    let mut c = with_points(&[0, 1]);
    c.add_group();
    c.add_group();
    c.add(1.0, 1.0, 0);
    assert_eq!(
        c.table_text(),
        "Group\tPoints\nGroup 1\t2\nGroup 2\t0\nGroup 3\t1\nTotal\t3\n"
    );
    // A name cannot break the table.
    c.groups[1].name = "a\tb\r\nc".into();
    assert_eq!(
        c.table_text(),
        "Group\tPoints\nGroup 1\t2\na b  c\t0\nGroup 3\t1\nTotal\t3\n"
    );
    assert_eq!(c.counts(), [2, 0, 1]);
}

#[test]
fn every_group_with_points_is_one_roi() {
    let mut c = PointCounter::new();
    c.add(1.5, 2.5, 0);
    c.add_group(); // Group 2: no points
    c.add_group();
    c.add(3.5, 4.5, 2);
    c.selected = 0;
    c.add(5.5, 6.5, 1);
    c.groups[0].name = "Neurons".into();

    let rois = c.to_rois(|f| f as u32 * 10 + 1);
    assert_eq!(rois.len(), 2, "the empty group is left out");
    assert_eq!(rois[0].name, "Neurons");
    assert_eq!(rois[0].color, SET1[0]);
    assert_eq!(
        rois[0].points,
        vec![
            RoiPoint {
                x: 1.5,
                y: 2.5,
                position: 1
            },
            RoiPoint {
                x: 5.5,
                y: 6.5,
                position: 11
            },
        ]
    );
    assert_eq!(rois[1].name, "Group 3");
    assert_eq!(rois[1].color, SET1[2]);
    assert_eq!(
        rois[1].points,
        vec![RoiPoint {
            x: 3.5,
            y: 4.5,
            position: 21
        }]
    );

    // What is shown does not change what is exported: each point stays on
    // its frame's slice.
    c.per_frame = !c.per_frame;
    assert_eq!(c.to_rois(|f| f as u32 * 10 + 1), rois);
    // A point on no frame of its own goes out on every slice.
    c.points[0].frame = None;
    assert_eq!(c.to_rois(|f| f as u32 * 10 + 1)[0].points[0].position, 0);
}

#[test]
fn no_points_no_rois() {
    assert!(PointCounter::new().to_rois(|f| f as u32).is_empty());
}

// ------------------------------------------------------ imagej_position

/// A U8 stack of `planes` 2x2 planes described as `channels` x `slices` x the
/// rest.
fn stack(channels: usize, slices: usize, planes: usize) -> Document {
    let opts =
        WriterOptions::new(2, 2, SampleType::U8).metadata(StackMetaWrite::new(channels, slices));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for p in 0..planes {
        w.write_frame_u8(&[p as u8; 4]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    Document::from_bytes(bytes, "probe.tif".into(), false).unwrap()
}

#[test]
fn a_single_image_has_no_position() {
    assert_eq!(imagej_position(&stack(1, 1, 1), 0), 0);
}

#[test]
fn a_time_series_frame_is_its_own_slice() {
    let s = stack(1, 1, 5);
    let positions: Vec<u32> = (0..5).map(|f| imagej_position(&s, f)).collect();
    assert_eq!(positions, [1, 2, 3, 4, 5]);
}

/// Channel-fastest order: frame t's first channel is IFD `t * channels`.
#[test]
fn a_hyperstack_frame_is_its_first_channels_slice() {
    let s = stack(2, 1, 6);
    assert_eq!(
        (s.display.dims().channels, s.display.dims().frames),
        (2, 3),
        "the probe is the shape it is meant to be"
    );
    let positions: Vec<u32> = (0..3).map(|f| imagej_position(&s, f)).collect();
    assert_eq!(positions, [1, 3, 5]);
}

#[test]
fn a_new_stack_has_no_overlay() {
    assert!(stack(1, 1, 2).overlay.is_empty());
}

// ------------------------------------------------------ frame_of_position

/// An RGB stack of `frames` 2x2 frames: one IFD a frame, its channels the
/// samples inside it.
fn rgb_stack(frames: usize) -> Document {
    let opts = WriterOptions::new(2, 2, SampleType::U8)
        .samples_per_pixel(3)
        .metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..frames {
        w.write_frame_u8(&[f as u8; 12]).unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    Document::from_bytes(bytes, "rgb.tif".into(), false).unwrap()
}

/// Every frame's position comes back to that frame, on every shape of stack
/// `imagej_position` is defined for.
#[test]
fn frame_of_position_inverts_imagej_position() {
    for s in [
        stack(1, 1, 5),
        stack(2, 1, 6),
        stack(1, 3, 6),
        stack(2, 3, 12),
        rgb_stack(4),
    ] {
        let frames = s.frame_count();
        assert!(frames > 1, "the probe has frames to tell apart");
        for f in 0..frames {
            assert_eq!(
                frame_of_position(&s, imagej_position(&s, f)),
                Some(f),
                "frame {f} of {:?}",
                s.display.dims()
            );
        }
    }
}

/// Each IFD of a frame — every channel, every slice — is that frame.
#[test]
fn every_plane_of_a_frame_is_that_frame() {
    // Two channels, three frames: IFDs 1-2, 3-4, 5-6.
    let s = stack(2, 1, 6);
    let frames: Vec<Option<usize>> = (1..=6).map(|p| frame_of_position(&s, p)).collect();
    assert_eq!(
        frames,
        [Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]
    );
    // Two channels, three slices, two frames: six IFDs a frame.
    let s = stack(2, 3, 12);
    assert_eq!(
        (
            s.display.dims().channels,
            s.display.dims().slices,
            s.display.dims().frames
        ),
        (2, 3, 2),
        "the probe is the shape it is meant to be"
    );
    let frames: Vec<Option<usize>> = (1..=12).map(|p| frame_of_position(&s, p)).collect();
    assert_eq!(frames, [[Some(0); 6], [Some(1); 6]].concat());
    // RGB: the channels are inside the IFD, so each IFD is a frame.
    let s = rgb_stack(3);
    assert!(s.display.rgb());
    let frames: Vec<Option<usize>> = (1..=3).map(|p| frame_of_position(&s, p)).collect();
    assert_eq!(frames, [Some(0), Some(1), Some(2)]);
}

/// 0 is every slice, not one; past the file is nowhere. A single image's one
/// IFD is its one frame.
#[test]
fn no_position_or_one_past_the_stack_is_no_frame() {
    let s = stack(1, 1, 5);
    assert_eq!(frame_of_position(&s, 0), None);
    assert_eq!(frame_of_position(&s, 6), None);
    assert_eq!(frame_of_position(&s, u32::MAX), None);
    let one = stack(1, 1, 1);
    assert_eq!(frame_of_position(&one, 1), Some(0));
    assert_eq!(frame_of_position(&one, 2), None);
    assert_eq!(frame_of_position(&one, 0), None);
}

// ---------------------------------------------------------------- from_rois

fn roi(name: &str, color: [u8; 3], points: &[(f32, f32, u32)]) -> PointRoi {
    PointRoi {
        hyperstack_position: [0; 3],
        name: name.into(),
        color,
        points: points
            .iter()
            .map(|&(x, y, position)| RoiPoint { x, y, position })
            .collect(),
    }
}

/// Position `p` is frame `p - 1`, for a stack of `frames` frames.
fn flat(frames: usize) -> impl Fn(u32) -> Option<usize> {
    move |p| (p as usize).checked_sub(1).filter(|&f| f < frames)
}

#[test]
fn a_loaded_counter_has_a_group_per_roi_holding_its_points() {
    let rois = [
        roi(
            "Neurons",
            [1, 2, 3],
            &[(1.5, 2.5, 1), (3.5, 4.5, 3), (5.5, 6.5, 1)],
        ),
        roi("", [4, 5, 6], &[(7.25, 8.75, 2)]),
        roi("Empty", [7, 8, 9], &[]),
    ];
    let mut c = PointCounter::from_rois(&rois, 4, flat(4));
    assert_eq!(
        c.groups,
        vec![
            Group {
                name: "Neurons".into(),
                color: [1, 2, 3]
            },
            Group {
                name: "Group 2".into(),
                color: [4, 5, 6]
            },
            Group {
                name: "Empty".into(),
                color: [7, 8, 9]
            },
        ],
        "a nameless ROI is named after its row; an empty one is still a group"
    );
    let points: Vec<(usize, f32, f32, Option<usize>)> = c
        .points
        .iter()
        .map(|p| (p.group, p.x, p.y, p.frame))
        .collect();
    assert_eq!(
        points,
        [
            (0, 1.5, 2.5, Some(0)),
            (0, 3.5, 4.5, Some(2)),
            (0, 5.5, 6.5, Some(0)),
            (1, 7.25, 8.75, Some(1))
        ]
    );
    assert_eq!(c.number_in_group(2), 3, "numbered in the ROI's order");
    assert!(
        !c.per_frame,
        "every point shown on every frame, as in a new one"
    );
    assert_eq!(c.selected, 0);
    assert!(!c.undo(), "nothing loaded is an edit to take back");
    assert_eq!(c.points.len(), 4);
}

/// Every slice (position 0) on every point: on no frame of their own, so on
/// every frame, as ImageJ showed them — even with only the current frame's
/// shown — and back out on every slice.
#[test]
fn an_overlay_on_every_slice_is_on_every_frame() {
    let rois = [roi("A", [0; 3], &[(1.0, 1.0, 0), (2.0, 2.0, 0)])];
    let mut c = PointCounter::from_rois(&rois, 5, flat(5));
    assert!(!c.per_frame);
    assert!(c.points.iter().all(|p| p.frame.is_none()));
    c.per_frame = true;
    assert!(c.shown_on(1, 4) && c.shown_on(0, 0), "on every frame");
    let out = c.to_rois(|_| panic!("on no frame, so no position to ask for"));
    assert!(out[0].points.iter().all(|p| p.position == 0));
}

/// A position that is no frame — past the stack, or one `frame_of` refuses,
/// or one it maps past `frames` — is no frame of its own: on every frame. One
/// that is a frame keeps it, even among others that are not.
#[test]
fn a_position_that_is_no_frame_is_on_every_frame() {
    let nowhere = [roi("A", [0; 3], &[(1.0, 1.0, 9), (2.0, 2.0, 0)])];
    let c = PointCounter::from_rois(&nowhere, 5, flat(5));
    assert_eq!(c.points[0].frame, None);
    // A `frame_of` that answers past the stack is not believed.
    let c = PointCounter::from_rois(&nowhere, 5, |_| Some(7));
    assert_eq!(c.points[0].frame, None);

    let mixed = [roi(
        "A",
        [0; 3],
        &[(1.0, 1.0, 9), (2.0, 2.0, 3), (3.0, 3.0, 0)],
    )];
    let c = PointCounter::from_rois(&mixed, 5, flat(5));
    let frames: Vec<Option<usize>> = c.points.iter().map(|p| p.frame).collect();
    assert_eq!(frames, [None, Some(2), None]);
}

/// A single-frame file's points are on its one frame.
#[test]
fn a_single_frame_holds_every_point() {
    let rois = [roi("A", [0; 3], &[(1.0, 1.0, 1)])];
    let c = PointCounter::from_rois(&rois, 1, flat(1));
    assert!(!c.per_frame);
    assert_eq!(c.points[0].frame, Some(0));
}

#[test]
fn no_rois_is_a_new_counter() {
    for frames in [1, 6] {
        let c = PointCounter::from_rois(&[], frames, |_| panic!("no position to map"));
        let new = PointCounter::new();
        assert_eq!(c.groups, new.groups);
        assert_eq!(c.per_frame, new.per_frame);
        assert!(c.points.is_empty());
    }
}

/// A group added after a load takes a Set1 colour no group shows yet, so it
/// cannot be mistaken for a loaded one; once all nine are taken they repeat by
/// the group's number.
#[test]
fn a_group_added_after_a_load_takes_a_colour_no_group_has() {
    let rois = [
        roi("a", SET1[0], &[(1.0, 1.0, 0)]),
        roi("b", SET1[2], &[(1.0, 1.0, 0)]),
        roi("c", [1, 2, 3], &[(1.0, 1.0, 0)]),
    ];
    let mut c = PointCounter::from_rois(&rois, 1, flat(1));
    assert_eq!(c.add_group(), 3);
    assert_eq!(c.groups[3].color, SET1[1]);
    assert_eq!(c.groups[3].name, "Group 4");
    assert_eq!(c.selected, 3);
    c.add_group();
    assert_eq!(c.groups[4].color, SET1[3], "SET1[2] is taken");
    // Five more take the last five free colours, 4 to 8.
    for _ in 0..5 {
        c.add_group();
    }
    let colours: Vec<[u8; 3]> = c.groups[5..].iter().map(|g| g.color).collect();
    assert_eq!(colours, SET1[4..].to_vec());
    // All nine taken: by number, so the eleventh group (index 10) is
    // SET1[10 % 9].
    assert_eq!(c.add_group(), 10);
    assert_eq!(c.groups[10].color, SET1[1]);
}

// ------------------------------------------------- the round trip, by file

/// One group as a round trip has to keep it: its name, its colour, and its
/// points' `(x, y, frame)` in order.
type GroupPoints = (String, [u8; 3], Vec<(f32, f32, Option<usize>)>);

/// A counter's groups in group order. (The order of `points` across groups
/// is not kept: an export lists each group's points together.)
fn by_group(c: &PointCounter) -> Vec<GroupPoints> {
    c.groups
        .iter()
        .enumerate()
        .map(|(g, group)| {
            let points = c
                .points
                .iter()
                .filter(|p| p.group == g)
                .map(|p| (p.x, p.y, p.frame))
                .collect();
            (group.name.clone(), group.color, points)
        })
        .collect()
}

fn assert_same_counter(back: &PointCounter, want: &PointCounter) {
    let (back, want) = (by_group(back), by_group(want));
    assert_eq!(back.len(), want.len(), "{back:?} vs {want:?}");
    for ((bn, bc, bp), (wn, wc, wp)) in back.iter().zip(&want) {
        assert_eq!((bn, bc), (wn, wc));
        assert_eq!(bp.len(), wp.len(), "{bn}: {bp:?} vs {wp:?}");
        for ((bx, by, bf), (wx, wy, wf)) in bp.iter().zip(wp) {
            assert_eq!(bf, wf, "{bn}: {bp:?} vs {wp:?}");
            assert!(
                (bx - wx).abs() < 1e-4 && (by - wy).abs() < 1e-4,
                "{bn}: {bp:?} vs {wp:?}"
            );
        }
    }
}

/// The whole way: a counter's ROIs as ImageJ `.roi` bytes in the document's
/// overlay, saved as a TIFF (its `IJMetadata` block), opened again — the
/// stack has the overlay — and loaded into a counter, which has the same
/// groups, points and frames.
fn round_trip(s: &mut Document, counter: &PointCounter) -> PointCounter {
    let rois = counter.to_rois(|f| imagej_position(s, f));
    s.overlay = rois;
    let bytes = crate::save::save_to_bytes(&crate::save::SaveSource::of(s), &mut |_| true)
        .expect("the save");
    let back =
        Document::from_bytes(bytes, "again.tif".into(), false).expect("the saved file opens");
    assert_eq!(back.display.dims(), s.display.dims(), "the same axes");
    assert_eq!(back.overlay.len(), s.overlay.len(), "the overlay came back");
    PointCounter::from_rois(&back.overlay, back.display.dims().frames, |p| {
        frame_of_position(&back, p)
    })
}

/// Points on every frame of a time series, in three groups, one renamed and
/// one placed off the pixel centres.
#[test]
fn a_counter_survives_a_save_and_a_reopen_on_a_time_series() {
    let mut s = stack(1, 1, 5);
    let mut c = PointCounter::new();
    c.add(0.5, 0.5, 0);
    c.add(1.5, 0.5, 4);
    c.add_group();
    c.add(1.3, 0.7, 2);
    c.add(0.1, 1.9, 2);
    c.add_group();
    c.add(1.5, 1.5, 3);
    c.selected = 0;
    c.add(0.5, 1.5, 1);
    c.groups[1].name = "Glia".into();

    let back = round_trip(&mut s, &c);
    assert_same_counter(&back, &c);
    assert_eq!(back.groups[1].name, "Glia");
}

/// A hyperstack — two channels, three slices, two frames — keeps each point
/// on its frame through the IFD it is written on.
#[test]
fn a_counter_survives_a_save_and_a_reopen_on_a_hyperstack() {
    let mut s = stack(2, 3, 12);
    assert_eq!(s.frame_count(), 2);
    let mut c = PointCounter::new();
    c.add(0.5, 0.5, 1);
    c.add(1.5, 1.5, 0);
    c.add_group();
    c.add(1.25, 0.75, 1);
    let back = round_trip(&mut s, &c);
    assert_same_counter(&back, &c);
}

/// Showing only the current frame's points or every one is the screen's
/// business: either way the points come back on their frames.
#[test]
fn what_is_shown_does_not_change_what_comes_back() {
    let mut s = stack(2, 1, 6);
    let mut c = PointCounter::new();
    c.add(0.5, 0.5, 2);
    c.add(1.5, 1.5, 1);
    for per_frame in [false, true] {
        c.per_frame = per_frame;
        let back = round_trip(&mut s, &c);
        assert!(!back.per_frame, "starts showing every point, as a new one");
        assert_same_counter(&back, &c);
    }
}

/// A point on no frame of its own — every slice in the file it came from —
/// is on every slice again after a save and a reopen.
#[test]
fn a_point_on_every_frame_comes_back_on_every_frame() {
    let mut s = stack(1, 1, 4);
    let mut c = PointCounter::new();
    c.add(0.5, 0.5, 3);
    c.add(1.5, 1.5, 1);
    c.points[1].frame = None;
    let back = round_trip(&mut s, &c);
    assert_same_counter(&back, &c);
    assert_eq!(back.points[1].frame, None);
}

// ------------------------------------------------------- what ImageJ needs

/// On a stack ImageJ opens as a hyperstack, a group whose points are all on
/// one frame is tied to that frame's plane — every channel, the first slice —
/// or ImageJ draws it on none. A group over several frames is drawn point by
/// point, and a plain time series needs nothing of the kind.
#[test]
fn a_group_on_one_frame_of_a_hyperstack_gets_its_plane() {
    // 2 channels x 3 frames.
    let s = stack(2, 1, 6);
    let mut c = PointCounter::new();
    c.add(0.5, 0.5, 1);
    c.add(1.5, 0.5, 1);
    c.add_group();
    c.add(0.5, 1.5, 0);
    c.add(1.5, 1.5, 2);
    c.add_group();
    c.add(1.5, 1.5, 2);
    c.points.push(CountedPoint {
        group: 2,
        x: 0.5,
        y: 0.5,
        frame: None,
        plane: None,
    });
    let rois = c.rois_for(&s);
    let placed: Vec<[u32; 3]> = rois.iter().map(|r| r.hyperstack_position).collect();
    assert_eq!(placed, [[0, 1, 2], [0; 3], [0; 3]]);
    let positions: Vec<Vec<u32>> = rois
        .iter()
        .map(|r| r.points.iter().map(|p| p.position).collect())
        .collect();
    assert_eq!(positions, [vec![3, 3], vec![1, 5], vec![5, 0]]);

    // A time series is no hyperstack.
    let s = stack(1, 1, 3);
    assert!(c
        .rois_for(&s)
        .iter()
        .all(|r| r.hyperstack_position == [0; 3]));
    // Slices and frames both make one.
    let s = stack(1, 3, 6);
    assert_eq!(c.rois_for(&s)[0].hyperstack_position, [0, 1, 2]);
}

/// A stack of `channels` x `slices` x the rest, with `rois` as its overlay.
fn with_overlay(channels: usize, slices: usize, planes: usize, rois: Vec<PointRoi>) -> Document {
    let opts = WriterOptions::new(2, 2, SampleType::U8)
        .metadata(StackMetaWrite::new(channels, slices).overlay(rois));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for p in 0..planes {
        w.write_frame_u8(&[p as u8; 4]).unwrap();
    }
    Document::from_bytes(
        w.finish().unwrap().into_inner(),
        "overlay.tif".into(),
        false,
    )
    .unwrap()
}

/// A ROI ImageJ tied to a plane by its hyperstack position alone is found on
/// the frame that plane is on, in the axes the file was written with; one
/// whose position covers planes of several frames — or none — is on every
/// frame.
#[test]
fn a_roi_on_a_hyperstack_plane_is_on_its_frame() {
    let placed = |hs: [u32; 3]| {
        let mut r = roi("H", [1, 2, 3], &[(1.0, 1.0, 0), (2.0, 2.0, 0)]);
        r.hyperstack_position = hs;
        r
    };
    // 2 channels x 4 frames: channel 2 of frame 3, or every channel of it.
    for hs in [[2, 1, 3], [0, 1, 3]] {
        let s = with_overlay(2, 1, 8, vec![placed(hs)]);
        let c = PointCounter::from_overlay(&s);
        assert!(c.points.iter().all(|p| p.frame == Some(2)), "{hs:?}");
    }
    // Every frame of channel 1 is not one frame; nor is a frame past the end.
    for hs in [[1, 1, 0], [0, 0, 9]] {
        let s = with_overlay(2, 1, 8, vec![placed(hs)]);
        let c = PointCounter::from_overlay(&s);
        assert!(c.points.iter().all(|p| p.frame.is_none()), "{hs:?}");
    }
}

/// Changing the axes and changing them back puts every point where it was:
/// each is pinned to its plane the first time, and stays pinned.
#[test]
fn reassigning_the_axes_and_back_loses_nothing() {
    // A 1-channel series of 6, read for a moment as 6 channels of 1 frame.
    let mut s = stack(1, 1, 6);
    let mut c = PointCounter::new();
    for f in [0, 2, 5] {
        c.add(0.5, 0.5, f);
    }
    let before = frame_starts(&s);
    crate::dimensions::apply_dimension_override(&mut s, 6, 1, 1);
    c.reanchor(|f| before.get(f).copied(), |p| frame_of_position(&s, p));
    let frames: Vec<Option<usize>> = c.points.iter().map(|p| p.frame).collect();
    assert_eq!(frames, [Some(0), Some(0), Some(0)], "one frame now");

    let before = frame_starts(&s);
    crate::dimensions::apply_dimension_override(&mut s, 1, 1, 6);
    c.reanchor(|f| before.get(f).copied(), |p| frame_of_position(&s, p));
    let frames: Vec<Option<usize>> = c.points.iter().map(|p| p.frame).collect();
    assert_eq!(
        frames,
        [Some(0), Some(2), Some(5)],
        "and back where they were"
    );
}

/// A new group's name is one no group has, after a load that skipped numbers.
#[test]
fn a_new_groups_name_is_not_taken() {
    let rois = [
        roi("Group 1", [0; 3], &[(1.0, 1.0, 0)]),
        roi("Group 3", [0; 3], &[(1.0, 1.0, 0)]),
    ];
    let mut c = PointCounter::from_rois(&rois, 1, flat(1));
    c.add_group();
    c.add_group();
    let names: Vec<&str> = c.groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["Group 1", "Group 3", "Group 4", "Group 5"]);
}

/// Exporting is one pass over the points, however many groups a file's
/// overlay brought: a hundred thousand single-point groups take no time.
#[test]
fn many_groups_export_in_one_pass() {
    let rois: Vec<PointRoi> = (0..100_000)
        .map(|i| roi(&format!("P{i}"), [0; 3], &[(1.0, 1.0, 0)]))
        .collect();
    let c = PointCounter::from_rois(&rois, 1, flat(1));
    let started = std::time::Instant::now();
    assert_eq!(c.to_rois(|_| 0).len(), 100_000);
    assert_eq!(c.counts().len(), 100_000);
    assert!(c.table_text().ends_with("Total\t100000\n"));
    assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
}

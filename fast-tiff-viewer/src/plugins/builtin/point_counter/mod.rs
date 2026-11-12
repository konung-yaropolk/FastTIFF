//! The point counter: points clicked onto the image, sorted into named,
//! coloured groups, counted, and exported as ImageJ point ROIs.
//!
//! # Not a registry plugin
//!
//! It is in the Plugins menu, under Analysis, and its code is here with the
//! other built-ins — but it is the one directory here not written against
//! `fasttiff_plugin_api` alone. A plugin is constructed afresh for every run
//! and keeps nothing between them, and what crosses the plugin boundary has no
//! point shape, while this is a session: a list of points that grows click by
//! click, an undo history, and marks drawn over the picture for as long as its
//! window is open. So it uses the viewer's own types ([`Document`]), has no entry
//! in [`register`](super::register), and has no `plugin-…` feature: the
//! applications refer to it directly, and it brings no dependency for a feature
//! to leave out.
//!
//! Where its parts are:
//!
//! * here — the model, [`PointCounter`] (groups, points, undo, the export as
//!   ImageJ ROIs and the start from them), and the gestures on the picture,
//!   [`session`]; both are tested without a window;
//! * `fast_tiff_lib::metadata::roi` — the ImageJ `.roi` format and the RoiSet
//!   `.zip`. The TIFF reader and writer need it for the overlay in a file's
//!   `IJMetadata`, and that crate cannot reach up into this one;
//! * [`Document::overlay`] and [`save`](crate::save) — the overlay a document
//!   carries, and how a save writes it;
//! * `FastTIFF/src/app/point_counter.rs` — the window, the marks drawn over the
//!   picture, and filling a [`Gesture`] from the pointer. They are egui, and
//!   this crate has no user-interface toolkit.
//!
//! # The model
//!
//! No drawing, no input, no window. A frontend turns clicks into
//! [`PointCounter::add`], drags into [`drag_to`](PointCounter::drag_to) and
//! [`finish_drag`](PointCounter::finish_drag), Alt+clicks into
//! [`delete`](PointCounter::delete) and Ctrl+Z into [`undo`](PointCounter::undo)
//! — [`Session::track`] does that much of it — and draws what
//! [`shown_on`](PointCounter::shown_on) says is on the frame in view.
//!
//! # Coordinates
//!
//! A point is in image coordinates where pixel `(i, j)` covers
//! `[i, i+1) x [j, j+1)` — continuous, so a click in the middle of pixel
//! `(10, 20)` is `(10.5, 20.5)`. They are the coordinates a
//! [`RoiPoint`](fast_tiff_lib::RoiPoint) takes, which converts them to
//! ImageJ's own on the way out.
//!
//! # Frames
//!
//! Every point remembers the frame it was placed on, and the export ties it to
//! that frame — its ImageJ stack position — whatever the screen shows. "Show
//! only current frame's tags" ([`PointCounter::per_frame`]) is about the screen
//! alone: on, a point is drawn, and can be taken hold of, only on its own
//! frame; off, the default, every point is on every frame. Switching it back
//! and forth changes nothing else. Counts are always over all frames.
//!
//! A point read from a file can belong to no frame: an ImageJ overlay can put a
//! point on every slice (position 0). It has no frame
//! ([`CountedPoint::frame`] is `None`), is on every frame whatever the setting,
//! and is exported on every slice again.
//!
//! # Starting from a file's points
//!
//! An export can come back: [`PointCounter::from_rois`] starts a counter from
//! point ROIs — a file's ImageJ overlay as it was read
//! ([`Document::overlay`](crate::Document::overlay)) — with [`frame_of_position`]
//! turning each point's stack position back into its frame. What
//! [`to_rois`](PointCounter::to_rois) writes, `from_rois` reads back as the
//! same groups — those that had points — with the same points, numbered the
//! same, on the same frames.

use crate::document::Document;
use fast_tiff_lib::{PointRoi, RoiPoint};

pub mod session;
pub use session::{Drag, Gesture, Session};

/// Matplotlib's `Set1`, in order — the colour of each new group, repeating
/// after the ninth.
pub const SET1: [[u8; 3]; 9] = [
    [0xe4, 0x1a, 0x1c],
    [0x37, 0x7e, 0xb8],
    [0x4d, 0xaf, 0x4a],
    [0x98, 0x4e, 0xa3],
    [0xff, 0x7f, 0x00],
    [0xff, 0xff, 0x33],
    [0xa6, 0x56, 0x28],
    [0xf7, 0x81, 0xbf],
    [0x99, 0x99, 0x99],
];

/// A named, coloured group of points — one row of the counter's table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    pub color: [u8; 3],
}

/// One counted point: its group (an index into [`PointCounter::groups`]),
/// where it is (see the module docs), and the frame it was placed on — `None`
/// for a point on every frame, which only a file's overlay makes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CountedPoint {
    pub group: usize,
    pub x: f32,
    pub y: f32,
    pub frame: Option<usize>,
    /// The stack position (ImageJ's: the IFD index plus one) the point is
    /// pinned to, once one is known — its file's, for a point read from one;
    /// its frame's first plane, for one placed here, from the first time the
    /// axes are reassigned. A plane does not move when the axes do and a
    /// frame does, so this is what the frame is found again from
    /// ([`Session::follow_axes`]), however many times the axes change and
    /// change back. `None` until then: the frame says it all.
    pub plane: Option<u32>,
}

/// One step [`PointCounter::undo`] can take back.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Edit {
    /// A point was added at this index.
    Add { index: usize },
    /// The point at `index` was dragged here from `from`.
    Move { index: usize, from: (f32, f32) },
    /// This point was deleted from `index`.
    Delete { index: usize, point: CountedPoint },
}

/// The counter's state: the groups, the points, which group new points go to,
/// and the undo history.
#[derive(Clone, Debug)]
pub struct PointCounter {
    pub groups: Vec<Group>,
    /// In the order they were placed (a delete undone goes back where it was),
    /// which is the order they are numbered in within their group.
    pub points: Vec<CountedPoint>,
    /// The group new points go to.
    pub selected: usize,
    /// "Show only current frame's tags": each point drawn only on the frame it
    /// was placed on. Off, the default, every point is drawn on every frame.
    /// The screen only: the export ties each point to its frame either way.
    pub per_frame: bool,
    /// Adds, moves and deletes, newest last. Renames and new groups are not
    /// in it: they are table edits, and Ctrl+Z inside a name is the text
    /// field's own.
    undo: Vec<Edit>,
}

impl Default for PointCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl PointCounter {
    /// A new counter: one group, "Group 1", in the first Set1 colour,
    /// selected, and every point shown on every frame.
    pub fn new() -> Self {
        PointCounter {
            groups: vec![Group {
                name: "Group 1".to_string(),
                color: SET1[0],
            }],
            points: Vec::new(),
            selected: 0,
            per_frame: false,
            undo: Vec::new(),
        }
    }

    /// A counter that starts from point ROIs, such as a file's overlay: one
    /// group per ROI, with its name and colour, holding its points in order —
    /// so the numbers within each group are the order the ROI lists them in.
    /// A ROI with no name makes a group called "Group N" after its row, the
    /// name a new one would have had.
    ///
    /// `frame_of` turns a point's stack position into its frame —
    /// [`frame_of_position`] on the stack the ROIs came from. A point at
    /// position 0 ("every slice"), or at one that is no frame of `frames`, is
    /// on no frame of its own: it stays on every frame, and goes back out on
    /// every slice. (A ROI ImageJ tied to a plane by its hyperstack position
    /// alone has points at 0; [`from_overlay`](Self::from_overlay) finds the
    /// plane first.) "Show only current frame's tags" starts off, as in
    /// [`new`](Self::new).
    ///
    /// No ROIs is [`new`](Self::new). Otherwise the first group is selected,
    /// and the undo history is empty: what was loaded is where the session
    /// starts, not an edit to take back.
    pub fn from_rois(
        rois: &[PointRoi],
        frames: usize,
        frame_of: impl Fn(u32) -> Option<usize>,
    ) -> Self {
        if rois.is_empty() {
            return Self::new();
        }
        let mut groups = Vec::with_capacity(rois.len());
        let mut points = Vec::new();
        for (g, roi) in rois.iter().enumerate() {
            groups.push(Group {
                name: if roi.name.is_empty() {
                    format!("Group {}", g + 1)
                } else {
                    roi.name.clone()
                },
                color: roi.color,
            });
            for p in &roi.points {
                let plane = Some(p.position).filter(|&position| position != 0);
                let frame = plane.and_then(&frame_of).filter(|&f| f < frames.max(1));
                points.push(CountedPoint {
                    group: g,
                    x: p.x,
                    y: p.y,
                    frame,
                    plane,
                });
            }
        }
        PointCounter {
            groups,
            points,
            selected: 0,
            per_frame: false,
            undo: Vec::new(),
        }
    }

    /// Add "Group N" (N one past the current count, or the next number no
    /// group's name has), select it, and return its index.
    ///
    /// A counter started from a file can already have a later "Group N": an
    /// export leaves out the groups that had no points, so the names it brings
    /// back skip numbers. Two rows of one name would be told apart by nothing.
    ///
    /// Its colour is the first of Set1's that no group has yet. While every
    /// group came from here that is simply the next one in order; after a
    /// load it skips the colours the loaded groups already show, so a new
    /// group never looks like an old one. Once all nine are taken, Set1's
    /// colours repeat by the group's number.
    pub fn add_group(&mut self) -> usize {
        let index = self.groups.len();
        let color = SET1
            .iter()
            .find(|c| self.groups.iter().all(|g| g.color != **c))
            .copied()
            .unwrap_or(SET1[index % SET1.len()]);
        let mut number = index + 1;
        while self
            .groups
            .iter()
            .any(|g| g.name == format!("Group {number}"))
        {
            number += 1;
        }
        self.groups.push(Group {
            name: format!("Group {number}"),
            color,
        });
        self.selected = index;
        index
    }

    /// Add a point to the selected group on `frame`, and return its index.
    /// Undoable.
    pub fn add(&mut self, x: f32, y: f32, frame: usize) -> usize {
        // `selected` and `groups` are public; a selection past the end (or no
        // groups at all) puts the point in the last group there is.
        let group = self.selected.min(self.groups.len().saturating_sub(1));
        let index = self.points.len();
        self.points.push(CountedPoint {
            group,
            x,
            y,
            frame: Some(frame),
            plane: None,
        });
        self.undo.push(Edit::Add { index });
        index
    }

    /// Move point `i` to `(x, y)` while it is being dragged. Records nothing:
    /// a drag is one edit, recorded by [`finish_drag`](Self::finish_drag) when
    /// it ends. An index with no point is ignored.
    pub fn drag_to(&mut self, i: usize, x: f32, y: f32) {
        if let Some(p) = self.points.get_mut(i) {
            p.x = x;
            p.y = y;
        }
    }

    /// End a drag of point `i` that started at `from`: one undo entry for the
    /// whole drag, or none when the point ended where it started.
    pub fn finish_drag(&mut self, i: usize, from: (f32, f32)) {
        if let Some(p) = self.points.get(i) {
            if (p.x, p.y) != from {
                self.undo.push(Edit::Move { index: i, from });
            }
        }
    }

    /// Delete point `i`. Undoable: the point goes back at the same index, so
    /// the numbers in its group come back as they were. An index with no point
    /// is ignored.
    pub fn delete(&mut self, i: usize) {
        if i < self.points.len() {
            let point = self.points.remove(i);
            self.undo.push(Edit::Delete { index: i, point });
        }
    }

    /// Take back the latest add, move or delete. Returns whether there was one.
    ///
    /// The history is a stack, so the points are always as the edit left them
    /// when it is undone. `points` is public, though, and an edit that no
    /// longer fits what is there is dropped rather than applied to the wrong
    /// point — the next one down is undone instead.
    pub fn undo(&mut self) -> bool {
        while let Some(edit) = self.undo.pop() {
            let len = self.points.len();
            match edit {
                Edit::Add { index } if index < len => {
                    self.points.remove(index);
                }
                Edit::Move { index, from } if index < len => {
                    let p = &mut self.points[index];
                    (p.x, p.y) = from;
                }
                Edit::Delete { index, point } if index <= len => {
                    self.points.insert(index, point);
                }
                _ => continue,
            }
            return true;
        }
        false
    }

    /// Whether point `i` is drawn on `frame`: always unless only the current
    /// frame's are shown, and then when it is that frame's — or on no frame of
    /// its own, and so on every one. No point, no.
    pub fn shown_on(&self, i: usize, frame: usize) -> bool {
        self.points
            .get(i)
            .is_some_and(|p| !self.per_frame || p.frame.is_none_or(|f| f == frame))
    }

    /// The point shown on `frame` nearest `(x, y)`, if one is within `radius`
    /// (all in image pixels). Of two at the same distance, the later — the one
    /// drawn on top.
    pub fn hit(&self, x: f32, y: f32, frame: usize, radius: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, p) in self.points.iter().enumerate() {
            if !self.shown_on(i, frame) {
                continue;
            }
            let d = (p.x - x).hypot(p.y - y);
            if d <= radius && best.is_none_or(|(_, b)| d <= b) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Point `i`'s number within its group, from 1, in the order of
    /// [`points`](Self::points) and over all frames. 0 for no point.
    pub fn number_in_group(&self, i: usize) -> usize {
        match self.points.get(i) {
            Some(p) => self.points[..=i]
                .iter()
                .filter(|q| q.group == p.group)
                .count(),
            None => 0,
        }
    }

    /// How many points `group` has, over all frames.
    pub fn count(&self, group: usize) -> usize {
        self.points.iter().filter(|p| p.group == group).count()
    }

    /// How many points each group has, over all frames, in group order: every
    /// [`count`](Self::count) in one pass over the points rather than one each.
    pub fn counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.groups.len()];
        for p in &self.points {
            if let Some(n) = counts.get_mut(p.group) {
                *n += 1;
            }
        }
        counts
    }

    /// The table as tab-separated text, for the clipboard: a `Group\tPoints`
    /// header, one row per group, and the total under them, as the window
    /// shows it. Lines end in `\n`; the frontend's clipboard path makes them
    /// whatever the platform wants. A tab or line break typed into a group's
    /// name becomes a space, so that it cannot split the row.
    pub fn table_text(&self) -> String {
        let counts = self.counts();
        let mut text = String::from("Group\tPoints\n");
        for (g, group) in self.groups.iter().enumerate() {
            let name: String = group
                .name
                .chars()
                .map(|c| {
                    if matches!(c, '\t' | '\n' | '\r') {
                        ' '
                    } else {
                        c
                    }
                })
                .collect();
            text += &format!("{name}\t{}\n", counts[g]);
        }
        text += &format!("Total\t{}\n", self.points.len());
        text
    }

    /// One [`PointRoi`] per group that has points, in group order: the group's
    /// name and colour, and its points in numbering order. Each point's
    /// position is `position_of(frame)` — see [`imagej_position`] for the
    /// usual one — or 0 ("every slice") for a point on no frame of its own.
    /// What is shown on screen does not change it. No ROI has a hyperstack
    /// position; [`rois_for`](Self::rois_for) gives them one where ImageJ
    /// needs it.
    pub fn to_rois(&self, position_of: impl Fn(usize) -> u32) -> Vec<PointRoi> {
        self.rois(position_of, |_| [0; 3])
    }

    /// [`to_rois`](Self::to_rois) for the stack the points were counted on —
    /// each point at [`imagej_position`] of its frame — and, when ImageJ will
    /// open that stack as a hyperstack, a group whose points are all on one
    /// frame tied to that frame's plane by its hyperstack position too:
    /// every channel, the first slice, that frame, which is what the 2D view
    /// shows. ImageJ draws such a group on no plane at all without it (see
    /// [`PointRoi::hyperstack_position`]); a group spread over frames is drawn
    /// point by point, and needs none.
    pub fn rois_for(&self, stack: &Document) -> Vec<PointRoi> {
        let hyperstack =
            crate::save::opens_as_imagej_hyperstack(stack.display.dims(), stack.display.rgb());
        self.rois(
            |frame| imagej_position(stack, frame),
            |frame| {
                if hyperstack {
                    let t = u32::try_from(frame + 1).unwrap_or(u32::MAX);
                    [0, 1, t]
                } else {
                    [0; 3]
                }
            },
        )
    }

    /// What both exports are: one ROI per group that has points, its points
    /// in numbering order at `position_of` their frames, and
    /// `hyperstack_of(frame)` as its hyperstack position when every point is
    /// on that one frame. One pass over the points, however many groups.
    fn rois(
        &self,
        position_of: impl Fn(usize) -> u32,
        hyperstack_of: impl Fn(usize) -> [u32; 3],
    ) -> Vec<PointRoi> {
        let mut by_group: Vec<Vec<RoiPoint>> = vec![Vec::new(); self.groups.len()];
        // Each group's one frame so far: `None` before its first point, and
        // `Some(None)` once two differ or one is on every frame.
        let mut shared: Vec<Option<Option<usize>>> = vec![None; self.groups.len()];
        for p in &self.points {
            let Some(points) = by_group.get_mut(p.group) else {
                continue;
            };
            points.push(RoiPoint {
                x: p.x,
                y: p.y,
                position: p.frame.map_or(0, &position_of),
            });
            let one = &mut shared[p.group];
            *one = match *one {
                None => Some(p.frame),
                Some(f) if f == p.frame => Some(f),
                Some(_) => Some(None),
            };
        }
        self.groups
            .iter()
            .zip(by_group)
            .zip(shared)
            .filter(|((_, points), _)| !points.is_empty())
            .map(|((group, points), shared)| PointRoi {
                name: group.name.clone(),
                color: group.color,
                points,
                hyperstack_position: shared.flatten().map_or([0; 3], &hyperstack_of),
            })
            .collect()
    }

    /// Find every point's frame again, after the stack's axes were
    /// reassigned: pin it to its plane ([`CountedPoint::plane`]) — or, for one
    /// that has none yet, to `start(frame)`, the plane its frame began at
    /// under the old axes — and put it on the frame `frame_of` says that plane
    /// is on now, or on every frame for none. A deleted point waiting in the
    /// undo history moves with them, so that undoing the delete puts it back
    /// among them.
    pub fn reanchor(
        &mut self,
        start: impl Fn(usize) -> Option<u32>,
        frame_of: impl Fn(u32) -> Option<usize>,
    ) {
        let follow = |p: &mut CountedPoint| {
            let plane = p.plane.or_else(|| p.frame.and_then(&start));
            if plane.is_some() {
                p.plane = plane;
                p.frame = plane.and_then(&frame_of);
            }
        };
        for p in &mut self.points {
            follow(p);
        }
        for edit in &mut self.undo {
            if let Edit::Delete { point, .. } = edit {
                follow(point);
            }
        }
    }

    /// A counter that starts from `stack`'s overlay ([`Document::overlay`]):
    /// [`from_rois`](Self::from_rois), each point's stack position mapped to
    /// its frame by [`frame_of_position`].
    ///
    /// A ROI ImageJ tied to a plane by its hyperstack position alone has its
    /// points at position 0, and that position is in the axes the file was
    /// written with, which need not be the ones it is being looked at in. So
    /// it is found as a plane of the file first — and its points put on the
    /// frame that plane is on, when every plane the position covers (an axis
    /// at 0 covers all of it) is on the one frame. When they are not, the
    /// points are on every frame.
    pub fn from_overlay(stack: &Document) -> Self {
        let meta = &stack.tiff.meta;
        let written = crate::display::Dims {
            channels: meta.channels,
            slices: meta.slices,
            frames: meta.frames,
        };
        let rgb = stack.display.rgb();
        let rois: Vec<PointRoi> = stack
            .overlay
            .iter()
            .map(|roi| {
                let mut roi = roi.clone();
                let hs = roi.hyperstack_position;
                if hs == [0; 3] {
                    return roi;
                }
                let plane = |last| crate::save::hyperstack_to_flat(hs, written, rgb, last);
                let (Some(first), Some(last)) = (plane(false), plane(true)) else {
                    return roi;
                };
                let one_frame = frame_of_position(stack, first)
                    .is_some_and(|f| frame_of_position(stack, last) == Some(f));
                if one_frame {
                    for p in roi.points.iter_mut().filter(|p| p.position == 0) {
                        p.position = first;
                    }
                }
                roi
            })
            .collect();
        Self::from_rois(&rois, stack.display.dims().frames, |position| {
            frame_of_position(stack, position)
        })
    }
}

/// Where each frame of `stack` starts, as ImageJ stack positions
/// ([`imagej_position`]) — what stays put about a point's frame when the axes
/// are reassigned, and so how [`Session::follow_axes`] finds it again.
pub fn frame_starts(stack: &Document) -> Vec<u32> {
    (0..stack.frame_count())
        .map(|frame| imagej_position(stack, frame))
        .collect()
}

/// The ImageJ stack position of `frame` in `stack`: the IFD that holds
/// channel 0, slice 0 of that frame, plus one — ImageJ counts slices from 1 in
/// file order, and a save writes the IFDs in the order they are in. 0 for a
/// single-image file, where ImageJ has no slices to tell apart.
///
/// The 2D view composites every channel at Z = 0, so channel 0 at Z = 0 is the
/// plane that stands for what is on screen, and the one ImageJ shows the mark
/// on. The position follows the axes as they are being looked at
/// ([`Display::dims`](crate::display::Display::dims)), which are also the ones
/// a save writes into the file's description.
pub fn imagej_position(stack: &Document, frame: usize) -> u32 {
    if stack.tiff.frames.len() <= 1 {
        return 0;
    }
    let (ifd, _) =
        crate::dimensions::plane_index(stack.display.dims(), stack.display.rgb(), 0, 0, frame);
    u32::try_from(ifd + 1).unwrap_or(u32::MAX)
}

/// The frame of `stack` that ImageJ stack position `position` is on: the
/// inverse of [`imagej_position`]. `position` is the flat IFD index plus one,
/// and the frame is the time point that IFD belongs to under the axes as they
/// are being looked at ([`Display::dims`](crate::display::Display::dims)).
///
/// Every channel and slice of a frame is that frame: the 2D view shows the
/// channels together at Z = 0, so a point ImageJ placed on channel 2, or on
/// slice 3 of a 4D stack, is on that frame here. `None` for 0, which is
/// "every slice" rather than one, and for a position past the file's IFDs or
/// past the frames the axes describe.
pub fn frame_of_position(stack: &Document, position: u32) -> Option<usize> {
    let ifd = usize::try_from(position.checked_sub(1)?).ok()?;
    if ifd >= stack.tiff.frames.len() {
        return None;
    }
    // The IFDs between one frame's first plane and the next's.
    let (stride, _) =
        crate::dimensions::plane_index(stack.display.dims(), stack.display.rgb(), 0, 0, 1);
    let frame = ifd / stride.max(1);
    (frame < stack.frame_count()).then_some(frame)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

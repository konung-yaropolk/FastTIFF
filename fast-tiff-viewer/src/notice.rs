//! [`Notice`]: what the viewer has to tell the user, held one per kind.
//!
//! Four things in the core are worth a line on a status bar: the document's
//! shape (its metadata disagrees with its planes, its Z is frozen in 2D, it
//! has more channels than can be shown), an open that failed, a frame that
//! could not be decoded, and a 3D volume that could not be shown as built.
//! They used to share one `Option<String>`, the last writer winning, so a
//! decode failure wiped the note about mislabelled channels and a volume error
//! could be hidden by the rule that hid the frozen-Z note in 3D.
//!
//! Held by kind instead ([`NoticeKind`]): a notice is replaced only by
//! another of its own kind, and cleared when it stops applying. When that is
//! differs by kind, and is written on each variant of [`Notice`]. The rules
//! that do not need a document or a GPU to state are methods here, so they
//! are tested as such.

use crate::display::Dims;
use crate::viewer::ViewMode;
use scivis_render::MAX_CHANNELS;
use std::fmt;

/// Which kind a [`Notice`] is. Also the order a frontend shows them in: the
/// report of the last action first, the standing note about the document's
/// shape, the longest of them, last.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoticeKind {
    OpenFailed,
    DecodeFailed,
    Volume,
    Shape,
}

impl NoticeKind {
    /// Every kind, in the order a frontend shows them.
    pub const ALL: [NoticeKind; 4] = [
        NoticeKind::OpenFailed,
        NoticeKind::DecodeFailed,
        NoticeKind::Volume,
        NoticeKind::Shape,
    ];

    fn slot(self) -> usize {
        self as usize
    }
}

/// Something the viewer has to tell the user. Its [`Display`](fmt::Display)
/// is the line to show.
#[derive(Clone, Debug, PartialEq)]
pub enum Notice {
    /// The last open failed, for the reason given (see
    /// [`Viewer::open`](crate::Viewer::open)); the document on show, if any,
    /// is the one before it. A report of an attempt rather than a state, so
    /// it lasts until the next open starts or succeeds, or until the frontend
    /// moves on to other work and [dismisses](crate::Viewer::dismiss) it.
    OpenFailed(String),
    /// The frame on screen could not be decoded, for the reason given. Until a
    /// frame is decoded for the screen, or another document is opened.
    DecodeFailed(String),
    /// The 3D view could not show the volume as it was built. See
    /// [`VolumeNote`] for how long each lasts; shown only over the 3D view,
    /// which is all it is about.
    Volume(VolumeNote),
    /// The document's shape. Set when a document is installed or its axes
    /// are reassigned, for as long as that interpretation stands; shown only
    /// in the views it applies to ([`ShapeNote::applies_to`]).
    Shape(ShapeNote),
}

impl Notice {
    pub fn kind(&self) -> NoticeKind {
        match self {
            Notice::OpenFailed(_) => NoticeKind::OpenFailed,
            Notice::DecodeFailed(_) => NoticeKind::DecodeFailed,
            Notice::Volume(_) => NoticeKind::Volume,
            Notice::Shape(_) => NoticeKind::Shape,
        }
    }

    /// Whether this is worth showing over `mode`'s view.
    ///
    /// A note about the 3D view says nothing about the 2D picture, and held
    /// for as long as it stands it would otherwise follow the user back to
    /// 2D and stay there. The frozen-Z note is the other way round (see
    /// [`ShapeNote::applies_to`]). The rest are about the document, or the
    /// last attempt to open one, whichever view is on screen.
    pub fn applies_to(&self, mode: ViewMode) -> bool {
        match self {
            Notice::Shape(s) => s.applies_to(mode),
            Notice::Volume(_) => mode == ViewMode::Volume,
            Notice::OpenFailed(_) | Notice::DecodeFailed(_) => true,
        }
    }
}

impl fmt::Display for Notice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Notice::OpenFailed(why) => write!(f, "Failed to open file: {why}"),
            Notice::DecodeFailed(why) => write!(f, "Failed to decode frame: {why}"),
            Notice::Volume(v) => v.fmt(f),
            Notice::Shape(s) => s.fmt(f),
        }
    }
}

/// Why the 3D view is not showing the volume as built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VolumeNote {
    /// The volume did not fit in GPU memory and is being rebuilt at no more
    /// than half the size: what is shown is a reduced copy. Until the axes
    /// are reassigned or another document opened, which build another volume
    /// — a volume uploaded since is the reduced one.
    Reduced,
    /// Not even the smallest volume fits, for the reason given: nothing is
    /// shown. Until a volume is uploaded, or another is to be built.
    NoRoom(String),
    /// The upload failed another way, for the reason given. Likewise.
    Failed(String),
}

impl fmt::Display for VolumeNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VolumeNote::Reduced => f.write_str(
                "3D: the volume did not fit in GPU memory; showing it at a lower resolution",
            ),
            VolumeNote::NoRoom(why) => {
                write!(
                    f,
                    "3D: not enough GPU memory even for a small volume ({why})"
                )
            }
            VolumeNote::Failed(why) => write!(f, "3D: {why}"),
        }
    }
}

/// What is worth knowing about a document's shape (see
/// [`crate::dimensions::shape_note`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeNote {
    /// The metadata describes `declared` planes but the file holds
    /// `available`: what is shown is `dims`, which is what fits.
    PlaneMismatch {
        declared: usize,
        available: usize,
        dims: Dims,
    },
    /// Channels, Z-slices and time frames all at once: the 2D view shows the
    /// first Z-slice of each, and only the 3D view the rest.
    FrozenZ(Dims),
    /// More channels than can be shown at once; the first
    /// [`MAX_CHANNELS`] are.
    TooManyChannels(usize),
}

impl ShapeNote {
    /// Whether this is worth showing over `mode`'s view. The frozen-Z note
    /// says that 2D shows only the first Z-slice, and the 3D view uses every
    /// one (as the volume's depth): over the volume it would be wrong.
    pub fn applies_to(self, mode: ViewMode) -> bool {
        !(matches!(self, ShapeNote::FrozenZ(_)) && mode == ViewMode::Volume)
    }
}

impl fmt::Display for ShapeNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ShapeNote::PlaneMismatch {
                declared,
                available,
                dims,
            } => write!(
                f,
                "Warning: this file's metadata describes {declared} image plane(s) but the file \
                 contains {available}. Showing {} channel(s) × {} Z-slice(s) × {} frame(s), \
                 which is what fits — the rest may live in a companion file (a multi-file \
                 OME set gives every file the whole dataset's dimensions), or the file may \
                 be truncated.",
                dims.channels, dims.slices, dims.frames
            ),
            ShapeNote::FrozenZ(dims) => write!(
                f,
                "Warning: this file has channels, Z-slices, and time frames all present at once \
                 ({} channel(s) × {} Z-slice(s) × {} frame(s)). Z isn't shown as a separate axis here \
                 — only the first Z-slice is used; to see the whole picture, use 3D view.",
                dims.channels, dims.slices, dims.frames
            ),
            ShapeNote::TooManyChannels(n) => write!(
                f,
                "Note: stack has {n} channels; showing the first {MAX_CHANNELS}."
            ),
        }
    }
}

/// The notices held, one per kind.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Notices {
    slots: [Option<Notice>; NoticeKind::ALL.len()],
}

impl Notices {
    /// Replace the notice of `notice`'s kind, and only that one.
    pub(crate) fn set(&mut self, notice: Notice) {
        let slot = notice.kind().slot();
        self.slots[slot] = Some(notice);
    }

    /// Drop the notice of `kind`, if there is one.
    pub(crate) fn clear(&mut self, kind: NoticeKind) {
        self.slots[kind.slot()] = None;
    }

    pub(crate) fn get(&self, kind: NoticeKind) -> Option<&Notice> {
        self.slots[kind.slot()].as_ref()
    }

    /// Every notice held, in [`NoticeKind`] order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Notice> {
        self.slots.iter().flatten()
    }

    /// The document's shape note now stands as `note`: replaced, or cleared
    /// when there is nothing to say.
    pub(crate) fn shape(&mut self, note: Option<ShapeNote>) {
        match note {
            Some(n) => self.set(Notice::Shape(n)),
            None => self.clear(NoticeKind::Shape),
        }
    }

    /// Another document is on show: nothing said of the last one's frame or
    /// volume applies, and the open that brought it succeeded. Its shape is
    /// [`shape`](Self::shape)'s to say.
    pub(crate) fn new_document(&mut self) {
        self.clear(NoticeKind::OpenFailed);
        self.clear(NoticeKind::DecodeFailed);
        self.clear(NoticeKind::Volume);
    }

    /// The frame on screen could not be decoded: replaces what was said of
    /// the last that could not.
    pub(crate) fn decode_failed(&mut self, error: &anyhow::Error) {
        self.set(Notice::DecodeFailed(format!("{error:#}")));
    }

    /// A frame reached the screen: a failure said of an earlier one no
    /// longer applies.
    pub(crate) fn frame_shown(&mut self) {
        self.clear(NoticeKind::DecodeFailed);
    }

    /// A volume reached the GPU: a note that none is shown no longer applies.
    /// A reduced one still does — what reached it is the reduced copy.
    pub(crate) fn volume_uploaded(&mut self) {
        if !matches!(
            self.get(NoticeKind::Volume),
            Some(Notice::Volume(VolumeNote::Reduced))
        ) {
            self.clear(NoticeKind::Volume);
        }
    }

    /// Another volume is to be built (the axes were reassigned): nothing said
    /// of the last one applies.
    pub(crate) fn volume_replaced(&mut self) {
        self.clear(NoticeKind::Volume);
    }
}

#[cfg(test)]
#[path = "notice_tests.rs"]
mod tests;

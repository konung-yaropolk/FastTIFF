//! The notices' rules: one per kind, replaced only by their own kind, cleared
//! when they stop applying — and the words each one shows, which are the
//! words the single status string used to carry.

use super::*;

fn dims(channels: usize, slices: usize, frames: usize) -> Dims {
    Dims {
        channels,
        slices,
        frames,
    }
}

fn held(n: &Notices) -> Vec<NoticeKind> {
    n.iter().map(Notice::kind).collect()
}

/// The bug the kinds exist for: a decode failure no longer wipes the note
/// about the document's shape, and the note coming back does not wipe the
/// failure.
#[test]
fn one_kind_does_not_erase_another() {
    let mut n = Notices::default();
    n.shape(Some(ShapeNote::TooManyChannels(9)));
    n.decode_failed(&anyhow::anyhow!("truncated strip"));
    assert_eq!(held(&n), [NoticeKind::DecodeFailed, NoticeKind::Shape]);
    n.shape(Some(ShapeNote::TooManyChannels(12)));
    assert_eq!(held(&n), [NoticeKind::DecodeFailed, NoticeKind::Shape]);
    n.set(Notice::Volume(VolumeNote::Reduced));
    n.set(Notice::OpenFailed("file not supported".into()));
    assert_eq!(held(&n), NoticeKind::ALL);
}

#[test]
fn a_notice_is_replaced_by_its_own_kind() {
    let mut n = Notices::default();
    n.decode_failed(&anyhow::anyhow!("first"));
    n.decode_failed(&anyhow::anyhow!("second"));
    assert_eq!(
        n.get(NoticeKind::DecodeFailed),
        Some(&Notice::DecodeFailed("second".into()))
    );
    assert_eq!(n.iter().count(), 1);
}

#[test]
fn a_frame_shown_clears_a_decode_failure_and_nothing_else() {
    let mut n = Notices::default();
    n.shape(Some(ShapeNote::TooManyChannels(9)));
    n.decode_failed(&anyhow::anyhow!("bad"));
    n.frame_shown();
    assert_eq!(held(&n), [NoticeKind::Shape]);
}

#[test]
fn a_shape_that_stops_applying_is_cleared() {
    let mut n = Notices::default();
    n.shape(Some(ShapeNote::FrozenZ(dims(2, 3, 4))));
    n.set(Notice::OpenFailed("x".into()));
    n.shape(None);
    assert_eq!(held(&n), [NoticeKind::OpenFailed]);
}

/// A new document: what was said of the last one's frame and volume goes, as
/// does the open failure, the open having now succeeded. The shape note is
/// the new document's to set.
#[test]
fn a_new_document_leaves_only_its_shape_to_say() {
    let mut n = Notices::default();
    for notice in [
        Notice::OpenFailed("x".into()),
        Notice::DecodeFailed("y".into()),
        Notice::Volume(VolumeNote::Reduced),
        Notice::Shape(ShapeNote::TooManyChannels(9)),
    ] {
        n.set(notice);
    }
    n.new_document();
    assert_eq!(held(&n), [NoticeKind::Shape]);
}

/// A volume on the GPU answers "nothing is shown", but not "what is shown is
/// reduced": the volume that reached it is the reduced one.
#[test]
fn an_upload_clears_what_said_no_volume_is_shown() {
    for note in [
        VolumeNote::NoRoom("oom".into()),
        VolumeNote::Failed("lost".into()),
    ] {
        let mut n = Notices::default();
        n.set(Notice::Volume(note));
        n.volume_uploaded();
        assert_eq!(n.get(NoticeKind::Volume), None);
    }
    let mut n = Notices::default();
    n.set(Notice::Volume(VolumeNote::Reduced));
    n.volume_uploaded();
    assert_eq!(
        n.get(NoticeKind::Volume),
        Some(&Notice::Volume(VolumeNote::Reduced))
    );
    n.volume_replaced();
    assert_eq!(
        n.get(NoticeKind::Volume),
        None,
        "another volume is another story"
    );
}

/// The frozen-Z note stands down in 3D, which shows every slice; a note about
/// the 3D view stands down in 2D, which it says nothing about. The rest apply
/// to both — a mismatch between the metadata and the file above all, which
/// used to be hidden in 3D along with the frozen-Z note whenever both were
/// true, as was everything else on the status line, a volume error included.
#[test]
fn each_notice_shows_over_the_views_it_is_about() {
    let d = dims(2, 3, 4);
    let frozen = Notice::Shape(ShapeNote::FrozenZ(d));
    assert!(frozen.applies_to(ViewMode::Movie));
    assert!(!frozen.applies_to(ViewMode::Volume));
    for note in [
        VolumeNote::Reduced,
        VolumeNote::NoRoom("oom".into()),
        VolumeNote::Failed("lost".into()),
    ] {
        let n = Notice::Volume(note);
        assert!(!n.applies_to(ViewMode::Movie), "{n:?}");
        assert!(n.applies_to(ViewMode::Volume), "{n:?}");
    }
    for n in [
        Notice::Shape(ShapeNote::PlaneMismatch {
            declared: 40,
            available: 20,
            dims: d,
        }),
        Notice::Shape(ShapeNote::TooManyChannels(9)),
        Notice::OpenFailed("x".into()),
        Notice::DecodeFailed("y".into()),
    ] {
        assert!(
            n.applies_to(ViewMode::Movie) && n.applies_to(ViewMode::Volume),
            "{n:?}"
        );
    }
}

/// The words, exactly as the status string carried them.
#[test]
fn each_notice_reads_as_it_did() {
    assert_eq!(
        Notice::OpenFailed("file not supported".into()).to_string(),
        "Failed to open file: file not supported"
    );
    assert_eq!(
        Notice::DecodeFailed("LZW: bad code".into()).to_string(),
        "Failed to decode frame: LZW: bad code"
    );
    assert_eq!(
        Notice::Volume(VolumeNote::Reduced).to_string(),
        "3D: the volume did not fit in GPU memory; showing it at a lower resolution"
    );
    assert_eq!(
        Notice::Volume(VolumeNote::NoRoom("out of memory".into())).to_string(),
        "3D: not enough GPU memory even for a small volume (out of memory)"
    );
    assert_eq!(
        Notice::Volume(VolumeNote::Failed("device lost".into())).to_string(),
        "3D: device lost"
    );
    assert_eq!(
        Notice::Shape(ShapeNote::TooManyChannels(9)).to_string(),
        format!("Note: stack has 9 channels; showing the first {MAX_CHANNELS}.")
    );
    assert_eq!(
        Notice::Shape(ShapeNote::FrozenZ(dims(2, 3, 4))).to_string(),
        "Warning: this file has channels, Z-slices, and time frames all present at once \
         (2 channel(s) × 3 Z-slice(s) × 4 frame(s)). Z isn't shown as a separate axis here — \
         only the first Z-slice is used; to see the whole picture, use 3D view."
    );
    assert!(Notice::Shape(ShapeNote::PlaneMismatch {
        declared: 40,
        available: 20,
        dims: dims(2, 1, 10)
    })
    .to_string()
    .starts_with("Warning: this file's metadata describes 40 image plane(s) but the file contains 20. Showing 2 channel(s) × 1 Z-slice(s) × 10 frame(s), which is what fits"));
}

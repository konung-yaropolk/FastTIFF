//! The retained coarse overview: what gets kept, and when it may be used.
//!
//! A frame too big for one GPU texture is shown through a window. Zoom out and
//! the resident window — fine, and covering only part of the frame — cannot be
//! stretched over the wider view, so there is nothing to draw while a new one is
//! cut and the interface stops for as long as that takes (0.75 s, measured, on a
//! 40000 x 12788 mosaic). Keeping the whole frame at a coarse sampling in RAM
//! removes that: the fallback is already decoded, so it costs an upload.
//!
//! Everything here is about the two ways that goes wrong.
//!
//! **It is silently never kept.** The retention test has to be
//! [`spans_whole_frame`](fast_tiff_viewer::window::spans_whole_frame), not
//! `covers_whole_frame` — the latter also demands `stride == 1`, which a coarse
//! overview never has. Get that wrong and nothing is ever retained, the stall
//! stays exactly as it was, and every other test in the suite still passes.
//!
//! **It is used when it no longer describes the stack.** The planes are bytes
//! cut for one frame, one channel set, one set of texture formats. Uploading
//! them under a different frame is another frame's picture; under a different
//! channel mapping it is another plane's pixels; under a different *format* it
//! is a buffer that does not fit the texture, which is a validation failure that
//! takes the process with it rather than merely looking wrong. So the key has to
//! carry all of it: the frame index, and the display
//! [`Revision`](fast_tiff_viewer::Revision) — whose `layout` moves with the
//! channel mapping and the formats, and whose `enabled` moves with the channel
//! set. Each part is pinned separately below.

use fast_tiff_viewer::prefetch::Decoded;
use fast_tiff_viewer::roi::Roi;
use fast_tiff_viewer::window::{
    covers_whole_frame, spans_whole_frame, Built, Overview, OVERVIEW_MAX_BYTES,
};
use fast_tiff_viewer::Revision;

// The mosaic this exists for.
const W: u32 = 40_000;
const H: u32 = 12_788;

/// The display revision the overviews below are captured under.
const REV: Revision = Revision {
    layout: 1,
    enabled: 1,
};

fn roi(x: u32, y: u32, w: u32, h: u32, stride: u32) -> Roi {
    Roi { x, y, w, h, stride }
}

/// A build of `roi` holding `bytes` bytes across one 8-bit channel.
fn built(r: Roi, bytes: usize) -> Built {
    Built {
        frame_index: 0,
        roi: r,
        channels: vec![0],
        planes: vec![Decoded::U8(vec![0; bytes])],
    }
}

fn capture(r: Roi, bytes: usize) -> Option<Overview> {
    Overview::capture(built(r, bytes), W, H, REV, OVERVIEW_MAX_BYTES)
}

// ---------------------------------------------------------------------------
// What is worth keeping
// ---------------------------------------------------------------------------

/// The retention test must ignore the sampling. A fit view of a gigapixel mosaic
/// spans the frame at stride 16 and is exactly what we want to keep, yet it is
/// *not* `covers_whole_frame` — that asks a different question ("is any
/// windowing needed at all?") and answers no here.
///
/// This is the mistake that would leave the feature inert with a green suite.
#[test]
fn a_coarse_whole_frame_window_is_retained() {
    let coarse = roi(0, 0, W, H, 16);
    assert!(spans_whole_frame(&coarse, W, H), "it does span the frame");
    assert!(
        !covers_whole_frame(&coarse, W, H),
        "...but not at full resolution, which is why the two predicates are different"
    );
    assert!(
        capture(coarse, 4096).is_some(),
        "a coarse fit view is exactly what is worth keeping"
    );
}

/// A window of part of the frame cannot stand in for a view of another part, so
/// it is not kept at all.
#[test]
fn a_window_short_of_the_frame_is_not_retained() {
    for short in [
        roi(0, 0, W / 2, H, 16), // narrow
        roi(0, 0, W, H / 2, 16), // short
        roi(100, 0, W, H, 16),   // offset, so it misses the left edge
        roi(0, 100, W, H, 16),   // offset, so it misses the top
    ] {
        assert!(
            !spans_whole_frame(&short, W, H),
            "{short:?} does not span the frame"
        );
        assert!(
            capture(short, 4096).is_none(),
            "{short:?} should not be retained"
        );
    }
}

/// The overview is held for the life of the file, so it needs a ceiling of its
/// own: the planner's own bound is `MAX_ROI_BYTES`, half a gigabyte, which is
/// fine for a texture that comes and goes and far too much to keep.
#[test]
fn an_overview_over_the_cap_is_not_retained() {
    let whole = roi(0, 0, W, H, 16);
    assert!(
        capture(whole, OVERVIEW_MAX_BYTES).is_some(),
        "exactly the cap should fit"
    );
    assert!(
        capture(whole, OVERVIEW_MAX_BYTES + 1).is_none(),
        "one byte over should not"
    );
}

/// The cap counts bytes, not samples. A float plane is four bytes per sample, so
/// budgeting against the sample count would let four times the memory through.
#[test]
fn the_cap_counts_bytes_rather_than_samples() {
    let whole = roi(0, 0, W, H, 16);
    let samples = OVERVIEW_MAX_BYTES / 4 + 1; // under the cap as a count, over it as bytes
    let float = Built {
        frame_index: 0,
        roi: whole,
        channels: vec![0],
        planes: vec![Decoded::F32(vec![0.0; samples])],
    };
    assert!(
        Overview::capture(float, W, H, REV, OVERVIEW_MAX_BYTES).is_none(),
        "{samples} float samples is {} bytes, past the cap",
        samples * 4
    );
}

/// The reported size is the memory actually held, which is what the cap is
/// checked against and what anyone measuring the process will see.
#[test]
fn the_reported_size_is_the_memory_held() {
    let o = capture(roi(0, 0, W, H, 16), 1024).expect("should be retained");
    assert_eq!(o.bytes(), 1024);
}

// ---------------------------------------------------------------------------
// When it may be used
// ---------------------------------------------------------------------------

fn fresh() -> Overview {
    capture(roi(0, 0, W, H, 16), 4096).expect("should be retained")
}

/// The baseline: unchanged stack, so it still describes it.
#[test]
fn an_unchanged_stack_can_use_its_overview() {
    let o = fresh();
    assert!(o.matches(0, REV));
    assert!(o.can_upload(&roi(0, 0, W, H, 16), 0, REV));
}

/// These are one frame's pixels. Shown for another they are simply the wrong
/// picture, and it would persist — nothing else would repaint it.
#[test]
fn an_overview_for_another_frame_is_not_used() {
    let o = fresh();
    assert!(!o.matches(1, REV));
    assert!(!o.can_upload(&roi(0, 0, W, H, 16), 1, REV));
}

/// A channel toggled on has no pixels in an overview decoded without it, and
/// would render whatever the freshly allocated texture happened to hold; one
/// toggled off has planes the upload has no use for.
///
/// The toggle moves the display's `enabled` revision at the moment it is made,
/// before the next sync plans residency, so the revision alone is the check.
/// (It used to be bumped inside the sync, after planning, which is why the
/// overview once had to carry the enabled flags as a second key.)
#[test]
fn an_overview_under_another_enabled_set_is_not_used() {
    let o = fresh();
    let toggled = Revision {
        enabled: REV.enabled + 1,
        ..REV
    };
    assert!(!o.matches(0, toggled), "a channel was switched since");
    assert!(!o.can_upload(&roi(0, 0, W, H, 16), 0, toggled));
}

/// Reassigning the axes changes which IFD each display channel reads, without
/// changing the channel *list* — and rebuilding the channels for it can change
/// their texture formats, the case that is worse than a wrong picture: a 16-bit
/// plane uploaded to a channel that has become 8-bit is a buffer of the wrong
/// length for the texture, which fails validation and takes the process with
/// it. The display's `layout` revision records both, so it has to be in the
/// key — otherwise a reassignment landing back on frame 0 splices in planes
/// decoded under the old interpretation.
#[test]
fn an_overview_from_another_channel_layout_is_not_used() {
    let o = fresh();
    let reassigned = Revision {
        layout: REV.layout + 1,
        ..REV
    };
    assert!(!o.matches(0, reassigned));
    assert!(!o.can_upload(&roi(0, 0, W, H, 16), 0, reassigned));
}

/// The planes were cut to one window's texture size, so they are the right bytes
/// for that window and no other — a differing stride alone changes the texture
/// dimensions. (Which channels they fill is fixed by the revision.)
#[test]
fn the_overview_is_only_uploaded_for_the_window_asked_for() {
    let o = fresh();
    let right = roi(0, 0, W, H, 16);

    assert!(o.can_upload(&right, 0, REV), "the window it was cut for");
    assert!(
        !o.can_upload(&roi(0, 0, W, H, 8), 0, REV),
        "a different stride is a different texture size"
    );
    assert!(
        !o.can_upload(&roi(100, 0, W, H, 16), 0, REV),
        "a different origin is different pixels"
    );
}

/// `can_upload` is strictly stronger than `matches`: the planner asks the weaker
/// question (is this overview still valid for the stack?) and the upload site
/// the stronger one (are these the exact bytes for this window?). Collapsing
/// them either way is a bug — one direction uploads a mismatched buffer, the
/// other stops the planner ever offering the overview.
#[test]
fn upload_is_stricter_than_validity() {
    let o = fresh();
    // Valid for the stack, but not the window that is being asked for.
    assert!(o.matches(0, REV));
    assert!(!o.can_upload(&roi(0, 0, W, H, 4), 0, REV));
    // And nothing can_upload without also matching.
    assert!(!o.can_upload(&roi(0, 0, W, H, 16), 9, REV));
}

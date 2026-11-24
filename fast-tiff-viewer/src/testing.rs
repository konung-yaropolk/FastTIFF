//! Internals for tests outside this crate, which see only what is `pub`: the
//! frontend's oracle tests, which check this crate against the code it
//! replaced, the built-in plugins' tests, which set up a document the way no
//! frontend can, and this crate's counting-allocator suite (`tests/alloc`),
//! which has to wait for a worker no frontend waits for. Not API — hidden
//! from the docs, and nothing a frontend is meant to call. The crate's own
//! integration suites (`tests/it`) need nothing from here: they test what a
//! frontend can do, through the public API, and the internals are tested
//! beside them in the crate.
//!
//! A hidden module rather than a `testing` cargo feature, because a feature
//! would have to be turned on for every test run that needs it. Asked for
//! with `required-features`, `cargo test --workspace` and every CI step
//! naming its own features would skip those tests without a word unless
//! each one said `--features testing`; turned on by a dev-dependency, it
//! unifies into the production build of anything compiled in the same
//! `cargo test`, so it would not keep production code off it where that
//! matters either. A module that is always there costs nothing — these are
//! re-exports — and its name says what it is.

/// The zoom ladder [`View2d`](crate::view2d::View2d) steps along, which the
/// frontend's oracle checks rung for rung against the ladder it replaced.
pub use crate::view2d::zoom::ZoomLadder;

/// Whether the read-ahead worker of `viewer`'s document holds a frame decoded
/// ahead and not yet taken. The counting-allocator suite (`tests/alloc`)
/// waits on it, so that the frame it plays next is the one read ahead rather
/// than whichever of the worker and the sync got there first.
pub fn read_ahead_ready(viewer: &crate::Viewer) -> bool {
    viewer
        .document
        .as_ref()
        .and_then(|d| d.services.prefetch.as_ref())
        .is_some_and(crate::prefetch::Prefetcher::ready)
}

/// Give the channels of `doc` these LUTs, in order, as a reinterpretation of
/// its channels does. A frontend picks a LUT from the presets; the PNG
/// exporter's tests need one of their own — pure green, say — to show that
/// each channel goes through the LUT it is displayed with.
pub fn set_luts(doc: &mut crate::Document, luts: Vec<crate::Lut>) {
    doc.display.set_luts(luts);
}

//! [`Residency`]: what one renderer holds of the document it shows.
//!
//! The document ([`Stack`](crate::Stack)) is what is shown: the file, the
//! display model, the frame on screen, and the services every view of it
//! shares — the background workers and the caches that hold for any view. This
//! is one view's side of it: which frame its textures hold, the window of the
//! frame they are cut to, the LUTs it has uploaded, the overview it kept for
//! zooming out, the window it is waiting for, and the 3D volume.
//!
//! Kept apart from the document for two reasons. A second view of the same
//! document holds a residency of its own, rather than both fighting over one
//! record of what is on "the" GPU. And a new document starts from a fresh one:
//! the [`Viewer`](crate::Viewer) replaces its residency whenever its document
//! changes, so nothing one file left behind can be taken for another's.
//!
//! # Reconciled, not invalidated
//!
//! Nothing that changes the display model touches this. A channel toggle, a
//! dimension override, an RGB or CMYK setup, a LUT pick: each only edits the
//! [`Display`], which moves a counter as it does (see [`crate::display`]).
//! Once per sync, before anything is read or uploaded, [`Residency::reconcile`]
//! compares what was recorded here against what the display now publishes and
//! the frame on screen, and drops whatever no longer matches:
//!
//! | What moved | What goes |
//! |---|---|
//! | the frame | the frame's pixels, the overview and the window asked for (each of one frame) |
//! | which channels are on ([`Revision::enabled`]) | the same three: each is made of the channels that were on |
//! | the channel layout ([`Revision::layout`]) | everything: the pixels and the window they were cut to, the LUTs, the overview, the window asked for, and the volume, whose GPU memory goes back |
//! | the LUTs | the LUTs |
//!
//! A contrast window moves nothing: it goes to the shader with every frame.
//! A 4D timepoint is not here either: the previous one's volume stays on
//! screen until the next lands, so it is replaced rather than dropped.
//!
//! What this replaced was a flag at every site that could change the model,
//! each resetting its own subset; a site that forgot showed a stale texture
//! with nothing to say so. Something added later records the key it was made
//! under and gets a rule here instead.
//!
//! The sync reaches the reconcile only through [`Residency::begin_sync`],
//! whose [`Pending`] says what is to be sent to the renderer before anything
//! else, and asks [`Residency::needs_upload`] whether the frame's textures are
//! to be filled again; it acts on nothing it reads here by itself. So what a
//! change asks of the renderer is answered without one, and tested so.

// Reconciled and filled by the per-frame GPU path (`crate::sync`), which
// needs a backend; without one a viewer holds a residency nothing uses.
#![cfg_attr(
    not(any(feature = "backend-wgpu", feature = "backend-glow")),
    allow(dead_code)
)]

use crate::display::{Display, Revision};
use crate::roi::Roi;
use crate::window::Overview;
use std::sync::atomic::{AtomicU64, Ordering};

/// What one renderer holds of the document on show, and what it is waiting
/// for. See the [module docs](self).
pub(crate) struct Residency {
    /// The channel layout ([`Revision::layout`]) everything below was made
    /// under. `None` until the first reconcile; a different one drops it all.
    layout: Option<u64>,
    /// Subsampling applied on the way to the GPU: `1` when what is on screen
    /// is the file's own resolution, higher when the view had to be coarsened
    /// to fit a texture. Worked out on every sync.
    pub(crate) gpu_stride: u32,
    /// Whether the frame on screen is past the device's per-axis texture
    /// limit, and so can only be shown through a window at all. Worked out on
    /// every sync, since it depends on the device.
    pub(crate) over_texture_limit: bool,
    /// The window of the frame the 2D textures are cut to. `None` until a
    /// frame has been uploaded under this layout.
    ///
    /// Kept across a frame change or a channel toggle, unlike the pixels: the
    /// textures are still that shape, and what the view should show next is
    /// planned from what they cover.
    pub(crate) roi: Option<Roi>,
    /// The frame whose pixels the textures hold, and the display revision they
    /// were decoded under. A disabled channel is skipped during upload (the
    /// shader multiplies it out anyway), so a channel switched on has nothing
    /// in its texture yet: its `enabled` moving is what forces a re-upload of
    /// the frame on screen, as the frame moving is for any other.
    pub(crate) frame: Option<(usize, Revision)>,
    /// The LUT revision ([`Display::lut_revision`]) whose LUTs the renderer
    /// holds.
    pub(crate) luts: Option<u64>,
    /// The whole frame at a coarse sampling, decoded and held in RAM — what a
    /// view leaving the resident window falls back to, so it can show correct
    /// pixels at once instead of stopping while a new window is cut. See
    /// [`Overview`].
    ///
    /// `None` until a frame-spanning window has been built, which the opening
    /// fit-to-window view of a frame past the texture limit does anyway.
    pub(crate) overview: Option<Overview>,
    /// The window the background worker is currently building, if any, with
    /// the frame and the display [`Revision`] it was asked for under.
    ///
    /// Only so the same window is not asked for again on the next frame. A
    /// pan or zoom re-plans every frame, so the rectangle wanted drifts
    /// continuously even while the worker is part-way through one that would
    /// serve every one of those frames.
    pub(crate) pending_window: Option<(usize, Roi, Revision)>,
    /// The 3D volume.
    pub(crate) volume: VolumeResidency,
}

impl Default for Residency {
    fn default() -> Self {
        Residency {
            layout: None,
            // Corrected on the first sync, once a renderer is on hand to say
            // what this device can hold.
            gpu_stride: 1,
            over_texture_limit: false,
            roi: None,
            frame: None,
            luts: None,
            overview: None,
            pending_window: None,
            volume: VolumeResidency::default(),
        }
    }
}

/// What the renderer is to be sent before anything else this sync, as the
/// reconcile at its top leaves the residency. See [`Residency::begin_sync`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pending {
    /// Give back the renderer's 3D volume: it is another document's, or of
    /// the axes before an override.
    pub(crate) release_volume: bool,
    /// Upload every channel's LUT, then record it with
    /// [`Residency::luts_uploaded`]: the renderer holds none of this channel
    /// layout, or holds them as they were before a pick.
    pub(crate) upload_luts: bool,
}

impl Residency {
    /// A sync's first step: [`reconcile`](Self::reconcile) with `display` and
    /// `frame` on screen, and say what that leaves the renderer to be sent
    /// first. A release asked for is taken by this call, so the sync gives the
    /// volume back on the strength of what it returns, once.
    ///
    /// The 2D frame is not in it: whether its textures are to be filled again
    /// also depends on the window the view is planned to, known only once the
    /// renderer has said what it can hold. That is
    /// [`needs_upload`](Self::needs_upload).
    pub(crate) fn begin_sync(&mut self, display: &Display, frame: usize) -> Pending {
        self.reconcile(display, frame);
        Pending {
            release_volume: self.volume.take_release(),
            upload_luts: self.luts.is_none(),
        }
    }

    /// Whether the 2D textures are to be decoded into again to show `roi` of
    /// the frame on screen: the reconcile left no record of them holding it as
    /// the display now stands (another frame, a channel switched on, another
    /// layout), or they are cut to another window, which a view leaving a
    /// window smaller than the frame asks for.
    pub(crate) fn needs_upload(&self, roi: &Roi) -> bool {
        self.frame.is_none() || self.roi != Some(*roi)
    }

    /// Record that the renderer holds `display`'s LUTs, as of now.
    pub(crate) fn luts_uploaded(&mut self, display: &Display) {
        self.luts = Some(display.lut_revision());
    }

    /// Drop whatever no longer describes `display` with `frame` on screen; the
    /// rules are in the [module docs](self). Run once at the top of every sync,
    /// by [`begin_sync`](Self::begin_sync), before anything is read or
    /// uploaded, so nothing below it has to ask whether what it finds here is
    /// still current.
    fn reconcile(&mut self, display: &Display, frame: usize) {
        let now = display.revision();
        if self.layout != Some(now.layout) {
            // Every display channel may now read another IFD, or have another
            // texture format: whatever was decoded or uploaded for the old
            // layout is the wrong planes, and the textures may be the wrong
            // shape. A new document arrives the same way, its revisions never
            // having been seen.
            self.layout = Some(now.layout);
            self.roi = None;
            self.frame = None;
            self.luts = None;
            self.overview = None;
            self.pending_window = None;
        }
        // Of one frame, decoded for the channels that were on.
        if self.frame.is_some_and(|key| key != (frame, now)) {
            self.frame = None;
        }
        if self
            .overview
            .as_ref()
            .is_some_and(|o| !o.matches(frame, now))
        {
            // Dropped rather than kept for later: the revision only moves
            // forward, and a frame left is as likely never to come back as to
            // be scrubbed past again, so the memory is better returned.
            self.overview = None;
        }
        if self
            .pending_window
            .is_some_and(|(f, _, asked_under)| (f, asked_under) != (frame, now))
        {
            self.pending_window = None;
        }
        if self.luts.is_some_and(|l| l != display.lut_revision()) {
            self.luts = None;
        }
        self.volume.reconcile(now.layout);
    }

    /// Record an upload of `frame`'s pixels, decoded under `revision` and cut
    /// to `roi`. Every path that uploads a frame records it here, which is
    /// what keeps the window the shader is mapped through the one the
    /// textures actually hold.
    pub(crate) fn uploaded(&mut self, roi: Roi, frame: usize, revision: Revision) {
        self.roi = Some(roi);
        self.frame = Some((frame, revision));
    }
}

/// The next value in the sequence every volume generation is drawn from, so
/// a build asked for by one residency can never be taken as another's.
fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// What one renderer holds of the document's 3D volume, and what it has
/// asked the document's volume builder for.
#[derive(Debug)]
pub(crate) struct VolumeResidency {
    /// The channel layout ([`Revision::layout`]) the volume uploaded, and the
    /// one being built, are of. `None` until the first reconcile.
    layout: Option<u64>,
    /// Which build of the volume is wanted: moved whenever it has to be built
    /// again — another layout, or a smaller rebuild after the GPU ran out of
    /// memory — so an in-flight build for an older one is recognized as stale
    /// and ignored, and cancelled by the next request, which carries the new
    /// one.
    pub(crate) generation: u64,
    /// Which timepoint the uploaded volume holds (`Some(frame_index)` in the
    /// 4D case, else `Some(0)`); `None` means no volume is uploaded yet, so a
    /// frontend shows a loading state until the first build lands. A mismatch
    /// with the current timepoint queues a rebuild — this is what makes 4D
    /// playback advance the volume through time.
    pub(crate) built_frame: Option<usize>,
    /// The [`generation`](Self::generation) of the volume last uploaded, to
    /// tell whether an error the renderer reports late is about the volume
    /// still wanted.
    pub(crate) uploaded_generation: Option<u64>,
    /// The `(generation, time)` currently queued on the background builder, so
    /// a frontend can tell that a build is under way.
    pub(crate) requested: Option<(u64, usize)>,
    /// The GPU memory budget worked out for the current generation, so every
    /// timepoint of a 4D stack is built at the same size.
    pub(crate) planned_budget: Option<usize>,
    /// The most bytes the volume may take after an upload ran out of GPU
    /// memory: half the volume that failed, so the next one is smaller
    /// whatever limited the last one's size. Kept across a layout change —
    /// it is this renderer's memory that ran short — and gone with the
    /// residency when the document changes: what did not fit may have been
    /// the last stack's problem.
    pub(crate) budget_cap: Option<usize>,
    /// The renderer may hold a volume nothing here describes, whose memory
    /// should go back on the next sync, the first call that has the renderer.
    ///
    /// Set from the start: a fresh residency knows of nothing the renderer
    /// holds, and whatever volume is there is another document's. Without it
    /// the textures stayed resident until the next volume arrived, which for a
    /// session that never returned to 3D was never.
    release: bool,
}

impl Default for VolumeResidency {
    fn default() -> Self {
        VolumeResidency {
            layout: None,
            generation: next_generation(),
            built_frame: None,
            uploaded_generation: None,
            requested: None,
            planned_budget: None,
            budget_cap: None,
            release: true,
        }
    }
}

impl VolumeResidency {
    /// Make this the residency of a volume of channel layout `layout`: when
    /// it was another one's, build again, and give back what was uploaded.
    fn reconcile(&mut self, layout: u64) {
        if self.layout == Some(layout) {
            return;
        }
        self.layout = Some(layout);
        // The old arrangement of the planes, or another document's: no use to
        // this one, and not drawn in the meantime (a frontend shows its
        // loading state until the new one lands).
        self.release |= self.uploaded_generation.take().is_some();
        self.rebuild();
    }

    /// Build the same volume again, from a fresh plan (a smaller budget, say),
    /// dropping any build in flight.
    pub(crate) fn rebuild(&mut self) {
        self.built_frame = None;
        self.generation = next_generation();
        self.requested = None;
        self.planned_budget = None;
    }

    /// Whether a volume of channel layout `layout` has been uploaded, of some
    /// timepoint.
    pub(crate) fn built_for(&self, layout: u64) -> bool {
        self.layout == Some(layout) && self.built_frame.is_some()
    }

    /// Whether the renderer's volume is to be given back now, clearing the
    /// request.
    fn take_release(&mut self) -> bool {
        std::mem::take(&mut self.release)
    }
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod tests;

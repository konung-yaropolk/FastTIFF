//! Internals for tests outside this crate, which see only what is `pub`: the
//! frontend's oracle tests, which check this crate against the code it
//! replaced, the built-in plugins' tests, which set up a document the way no
//! frontend can, and this crate's counting-allocator suite (`tests/alloc`),
//! which has to wait for a worker no frontend waits for, and runs a volume
//! build and a save where it can count what they allocate. Not API — hidden
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

/// The 3D volume the view builds of `doc` on a renderer with `caps`, at
/// `budget` bytes, built where this is called. The counting-allocator suite
/// (`tests/alloc`) measures what a build allocates; through the view it
/// would be built on a worker, at a time of the worker's choosing.
pub fn build_volume(
    doc: &crate::Document,
    caps: scivis_render::VolumeCaps,
    budget: usize,
) -> Option<scivis_render::pack::PackedVolume> {
    let plan = crate::sync::plan_volume(
        doc,
        caps,
        0,
        budget,
        &crate::VolumeSettings::default(),
        &crate::VolumeInputs::default(),
    );
    crate::volume::build_volume(&doc.tiff, &plan, &crate::worker::CancelToken::never()).ok()?
}

/// A save of `source` into nothing: the file a save writes, thrown away as it
/// is written, handing the writer `batch_bytes` of frames at a time rather
/// than the save's own 64 MiB. Returns how many bytes it wrote. The
/// counting-allocator suite measures what a save allocates, which a sink
/// growing a buffer of its own would add to, and needs several batches of a
/// stack small enough to encode in a debug build.
pub fn save_discarding(
    source: &crate::save::SaveSource,
    batch_bytes: usize,
    on_progress: &mut dyn FnMut(f32) -> bool,
) -> anyhow::Result<u64> {
    /// Takes every byte, keeps none, and knows where it is: the writer seeks
    /// back to its header at the end.
    struct Discard {
        at: u64,
        len: u64,
    }
    impl std::io::Write for Discard {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.at += bytes.len() as u64;
            self.len = self.len.max(self.at);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl std::io::Seek for Discard {
        fn seek(&mut self, to: std::io::SeekFrom) -> std::io::Result<u64> {
            self.at = match to {
                std::io::SeekFrom::Start(at) => at,
                std::io::SeekFrom::End(by) => self.len.saturating_add_signed(by),
                std::io::SeekFrom::Current(by) => self.at.saturating_add_signed(by),
            };
            Ok(self.at)
        }
    }
    let sink = Discard { at: 0, len: 0 };
    Ok(crate::save::write_all(source, sink, on_progress, batch_bytes)?.len)
}

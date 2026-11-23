//! Turns a frame's bytes into pixels: the decode core behind [`Frame`].
//!
//! One core, four routes (see [`core`]): a plane that already is what was
//! asked for is borrowed straight from the file — the memory map, for a
//! stack opened from disk — with no decoding work at all; one that differs only
//! by a byte swap or a sign flip is decompressed straight into the
//! destination; anything that changes representation is converted a strip or
//! tile at a time, while the piece is still in cache; and only a wide frame
//! auto-ranged to `u16` is assembled whole first, because its range has to be
//! known before any sample can be converted.
//!
//! The process-wide and per-thread parallelism hints live here too: one switch
//! for all CPU-heavy pixel work, the writer's included.
//!
//! [`Frame`]: crate::Frame

use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) mod cmyk;
mod codecs;
mod convert;
pub(crate) mod core;
mod legacy;
mod pieces;
mod tiles;

pub use legacy::{
    frame_float_minmax, preload_frames_f32, preload_frames_u16, preload_frames_u8,
    read_frame_bytes_into, read_frame_f32, read_frame_f32_into, read_frame_u16,
    read_frame_u16_into, read_frame_u8, read_frame_u8_into, read_plane_f32, read_plane_f32_into,
    read_plane_rgb_u16, read_plane_rgb_u8, read_plane_u16, read_plane_u16_into, read_plane_u8,
    read_plane_u8_into, read_planes_f32, read_planes_f32_into, read_planes_rgb_u16,
    read_planes_rgb_u8, read_planes_u16, read_planes_u16_into, read_planes_u8, read_planes_u8_into,
};

/// Floor on frame size (pixels) below which a decode is never split across
/// cores: too small for rayon's fork-join to pay off, and such frames decode
/// fast enough that they're never the playback bottleneck.
///
/// Public for a caller that decodes a large frame a band at a time: a band is
/// a cropped [`FrameInfo`] and so a frame of its own here, and one of fewer
/// pixels than this is decoded on one core however the hint is set.
pub const PARALLEL_MIN_PIXELS: usize = 1024 * 1024;

/// Process-wide hint: split large decodes across cores when `true`, run them
/// serially when `false` (the default). Parallel decode spreads load across
/// cores but uses *more total CPU* (rayon overhead), so it's only worth it when
/// a single core can't keep up. The caller (the viewer) turns it on only while
/// real-time playback is falling behind — see `set_parallel_decode`.
static PARALLEL_DECODE: AtomicBool = AtomicBool::new(false);

/// Enable/disable parallel decoding *and* encoding for the whole process. A
/// performance hint, not a correctness knob: the pixels are identical either
/// way. The same hint governs the writer's per-strip compression (see
/// `encode`), so a host application has one switch for all CPU-heavy pixel
/// work. Off by default.
///
/// The default for every thread that has not chosen for itself
/// ([`set_thread_parallel_decode`], [`parallel_decode_scope`]). A host with
/// more than one kind of work running — a view decoding frames, a save
/// encoding in the background — should choose per thread instead, so that one
/// does not decide for the other.
pub fn set_parallel_decode(enabled: bool) {
    PARALLEL_DECODE.store(enabled, Ordering::Relaxed);
}

thread_local! {
    /// This thread's own choice, when it has made one. It wins over the
    /// process-wide hint.
    static PARALLEL_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Choose parallel decoding and encoding for work done on this thread from
/// now on, whatever [`set_parallel_decode`] says for the process; `None` goes
/// back to following it.
///
/// What a viewer's interface thread wants: it decides for its own decodes —
/// threads while playback falls behind, serial while it keeps up — without
/// that decision reaching a save or a volume build running elsewhere, which
/// a process-wide switch flipped sixty times a second did.
pub fn set_thread_parallel_decode(enabled: Option<bool>) {
    PARALLEL_HERE.with(|p| p.set(enabled));
}

/// [`set_thread_parallel_decode`] for a scope: the choice this thread had
/// before comes back when the guard is dropped.
pub fn parallel_decode_scope(enabled: bool) -> ParallelDecodeScope {
    let previous = PARALLEL_HERE.with(|p| p.replace(Some(enabled)));
    ParallelDecodeScope {
        previous,
        _this_thread: std::marker::PhantomData,
    }
}

/// The guard [`parallel_decode_scope`] returns. Not `Send`: it restores the
/// thread it was made on, and only that one.
#[must_use = "the choice lasts only as long as the guard"]
pub struct ParallelDecodeScope {
    previous: Option<bool>,
    _this_thread: std::marker::PhantomData<*const ()>,
}

impl Drop for ParallelDecodeScope {
    fn drop(&mut self) {
        PARALLEL_HERE.with(|p| p.set(self.previous));
    }
}

/// Whether a large decode or encode on this thread would be split across
/// cores: this thread's choice, or the process-wide hint when it has made
/// none. What a thread handing work to another passes along with it, so the
/// work is done as its requester would have done it.
pub fn parallel_decode_enabled() -> bool {
    PARALLEL_HERE
        .with(|p| p.get())
        .unwrap_or_else(|| PARALLEL_DECODE.load(Ordering::Relaxed))
}

/// Whether a CPU-heavy pass should use rayon: only when the caller asked for it
/// *and* the frame is big enough for the speedup to beat the fork-join cost.
/// Shared with `encode` so both directions honor the one hint.
pub(crate) fn should_parallelize(n_pixels: usize) -> bool {
    // wasm32 has no threads to fork onto: rayon compiles there, but spawning
    // would panic at runtime. Making every parallel path unreachable keeps one
    // code path for both targets, with no `#[cfg]` scattered through the
    // decoders.
    if cfg!(target_arch = "wasm32") {
        return false;
    }
    parallel_decode_enabled() && n_pixels >= PARALLEL_MIN_PIXELS
}

// What the unit tests of the old single-file decoder drive directly.
#[cfg(test)]
use crate::ifd::ByteOrder;
#[cfg(test)]
use crate::index::{Compression, FrameInfo, SampleFormat};
#[cfg(test)]
use codecs::{packbits_decode, undo_predictor};
#[cfg(test)]
use convert::{deinterleave_u8, minmax_f32};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "oracle_tests.rs"]
mod oracle_tests;

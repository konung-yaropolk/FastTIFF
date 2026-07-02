//! Background decode-ahead for playback. While the UI thread uploads + renders
//! the current frame, a worker thread decodes the *next* frame into ready
//! buffers, so when playback reaches it the UI thread only has to upload — a
//! steady second-core pipeline with none of the per-frame fork/join overhead of
//! intra-frame parallel decode.
//!
//! It only pays off when decoding is non-trivial *and* the inline decode isn't
//! already zero-copy (otherwise prefetch would just add a copy), so `app.rs`
//! gates it to **compressed** stacks during real-time playback that's keeping up
//! (the serial-decode regime). When playback falls behind, the adaptive parallel
//! decode takes over instead.
//!
//! The worker is self-contained: it opens its **own** memory map of the same
//! file (a second mmap shares the OS page cache — no duplicate RAM), so it never
//! touches the app's `TiffStack` and needs no shared/locked state beyond a
//! request channel and a result slot. Correctness is defensive: a prefetched
//! result is used only when its `(generation, frame_index)` and channel layout
//! exactly match what's wanted; any mismatch falls back to inline decode, so a
//! stale prefetch can cost a little work but can never show the wrong frame.

use crate::render::ChannelKind;
use fast_tiff_lib::{ByteOrder, FrameInfo, TiffStack};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// One channel's decoded pixels, in the format its GPU texture expects.
pub enum Decoded {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

/// Decode one channel of a frame into owned pixels.
/// `plane`/`rgb` select the RGB-plane deinterleave; otherwise the whole image.
fn decode_channel(
    mmap: &[u8],
    frame: &FrameInfo,
    order: ByteOrder,
    kind: ChannelKind,
    plane: usize,
    rgb: bool,
) -> anyhow::Result<Decoded> {
    Ok(match kind {
        ChannelKind::Int8 => {
            if rgb {
                Decoded::U8(fast_tiff_lib::read_plane_u8(mmap, frame, order, plane)?)
            } else {
                Decoded::U8(fast_tiff_lib::read_frame_u8(mmap, frame, order)?.into_owned())
            }
        }
        ChannelKind::Float => Decoded::F32(fast_tiff_lib::read_frame_f32(mmap, frame, order)?.into_owned()),
        ChannelKind::Int16 => {
            if rgb {
                Decoded::U16(fast_tiff_lib::read_plane_u16(mmap, frame, order, None, plane)?)
            } else {
                Decoded::U16(fast_tiff_lib::read_frame_u16(mmap, frame, order, None)?.into_owned())
            }
        }
    })
}

/// Decode all of one frame-request's channels, returning results in `jobs`
/// order. Shared by the inline path (in `app.rs`) and the prefetch worker so
/// both produce byte-identical results.
///
/// RGB channels are sample planes of the *same* IFD, so they're decoded with a
/// single decompression pass (`read_planes_*`) instead of one full decode per
/// channel — ~3x cheaper on compressed RGB. Non-RGB (and float) jobs decode
/// per-channel as before.
pub fn decode_jobs(
    mmap: &[u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
) -> anyhow::Result<Vec<Decoded>> {
    // Batched RGB path: every job is a plane of one IFD with one integer kind.
    if jobs.len() > 1
        && jobs
            .iter()
            .all(|j| j.rgb && j.ifd_idx == jobs[0].ifd_idx && j.kind == jobs[0].kind)
        && matches!(jobs[0].kind, ChannelKind::Int8 | ChannelKind::Int16)
    {
        let frame = frames
            .get(jobs[0].ifd_idx)
            .ok_or_else(|| anyhow::anyhow!("frame {} out of range", jobs[0].ifd_idx))?;
        return match jobs[0].kind {
            ChannelKind::Int8 => {
                let mut planes = fast_tiff_lib::read_planes_u8(mmap, frame, order)?;
                jobs.iter().map(|j| Ok(Decoded::U8(take_plane(&mut planes, j.plane)?))).collect()
            }
            _ => {
                let mut planes = fast_tiff_lib::read_planes_u16(mmap, frame, order, None)?;
                jobs.iter().map(|j| Ok(Decoded::U16(take_plane(&mut planes, j.plane)?))).collect()
            }
        };
    }

    jobs.iter()
        .map(|job| {
            let frame = frames
                .get(job.ifd_idx)
                .ok_or_else(|| anyhow::anyhow!("frame {} out of range", job.ifd_idx))?;
            decode_channel(mmap, frame, order, job.kind, job.plane, job.rgb)
        })
        .collect()
}

/// Move one plane out of a `read_planes_*` result (each plane is taken once —
/// display channels map to distinct planes by construction).
fn take_plane<T>(planes: &mut [Vec<T>], plane: usize) -> anyhow::Result<Vec<T>> {
    if plane >= planes.len() {
        anyhow::bail!("sample plane {plane} out of range ({} planes)", planes.len());
    }
    Ok(std::mem::take(&mut planes[plane]))
}

/// How to decode one channel of a requested frame (the app computes these from
/// the current metadata + per-channel settings and sends them to the worker).
#[derive(Clone)]
pub struct ChannelJob {
    pub channel: usize, // display channel index (upload target)
    pub ifd_idx: usize, // which IFD/plane in the file
    pub plane: usize,   // sample plane within the IFD (RGB)
    pub kind: ChannelKind,
    pub rgb: bool,
    pub width: u32,
    pub height: u32,
}

/// One decoded channel of a completed prefetch.
pub struct DecodedChannel {
    pub channel: usize,
    pub width: u32,
    pub height: u32,
    pub kind: ChannelKind,
    pub data: Decoded,
}

/// A fully-decoded frame produced by the worker, tagged so the app can confirm
/// it still matches what's wanted before using it.
pub struct PrefetchResult {
    pub generation: u64,
    pub frame_index: usize,
    pub channels: Vec<DecodedChannel>,
}

struct Request {
    generation: u64,
    frame_index: usize,
    jobs: Vec<ChannelJob>,
}

/// Owns the worker thread + the latest result. Dropping it closes the request
/// channel, which ends the worker (it finishes any in-flight decode first).
pub struct Prefetcher {
    tx: Sender<Request>,
    result: Arc<Mutex<Option<PrefetchResult>>>,
    _handle: JoinHandle<()>,
}

impl Prefetcher {
    /// Spawn a worker that opens its own map of `path`. Returns `None` if the
    /// thread or the worker's file open fails — callers then just decode inline.
    pub fn new(path: PathBuf) -> Option<Self> {
        let (tx, rx) = channel::<Request>();
        let result = Arc::new(Mutex::new(None));
        let result_worker = Arc::clone(&result);
        let handle = std::thread::Builder::new()
            .name("fasttiff-prefetch".to_owned())
            .spawn(move || {
                // Second mmap of the same file: shares the OS page cache, so no
                // duplicate pixel RAM; the IFD walk is a one-time cost.
                match TiffStack::open(&path) {
                    Ok(stack) => worker_loop(stack, rx, result_worker),
                    Err(e) => log::warn!("prefetch worker: can't open {}: {e:#}", path.display()),
                }
            })
            .ok()?;
        Some(Self { tx, result, _handle: handle })
    }

    /// Ask the worker to decode `frame_index`'s `jobs`. Fire-and-forget; the
    /// worker drains to the most recent request, so superseded predictions are
    /// skipped.
    pub fn request(&self, generation: u64, frame_index: usize, jobs: Vec<ChannelJob>) {
        let _ = self.tx.send(Request { generation, frame_index, jobs });
    }

    /// Take the prefetched result iff it matches `(generation, frame_index)`;
    /// otherwise leave the slot untouched and return `None`. The caller still
    /// verifies the channel layout before using it.
    pub fn take_matching(&self, generation: u64, frame_index: usize) -> Option<PrefetchResult> {
        let mut slot = self.result.lock().ok()?;
        let matches = slot
            .as_ref()
            .is_some_and(|r| r.generation == generation && r.frame_index == frame_index);
        if matches {
            slot.take()
        } else {
            None
        }
    }
}

fn worker_loop(stack: TiffStack, rx: Receiver<Request>, result: Arc<Mutex<Option<PrefetchResult>>>) {
    // Block for a request; channel closed (Prefetcher dropped) -> exit.
    while let Ok(mut req) = rx.recv() {
        // Skip superseded predictions: only the most recent request matters.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let mut channels = Vec::with_capacity(req.jobs.len());
        let ok = match decode_jobs(&stack.mmap, &stack.frames, stack.byte_order, &req.jobs) {
            Ok(decoded) => {
                for (job, data) in req.jobs.iter().zip(decoded) {
                    channels.push(DecodedChannel {
                        channel: job.channel,
                        width: job.width,
                        height: job.height,
                        kind: job.kind,
                        data,
                    });
                }
                true
            }
            Err(_) => false,
        };
        if ok {
            if let Ok(mut slot) = result.lock() {
                *slot = Some(PrefetchResult {
                    generation: req.generation,
                    frame_index: req.frame_index,
                    channels,
                });
            }
        }
    }
}

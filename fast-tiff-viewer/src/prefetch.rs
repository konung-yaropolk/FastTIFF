//! Background read-ahead for playback, in one of two modes. While the UI
//! thread uploads + renders the current frame, a worker thread prepares the
//! *next* one:
//!
//! - **Decode-ahead** (compressed stacks): the worker decodes the next frame
//!   into ready buffers, so reaching it is just an upload — a steady
//!   second-core pipeline with none of the per-frame fork/join overhead of
//!   intra-frame parallel decode.
//! - **Page-touch** (uncompressed stacks): decoding is a zero-copy borrow, so
//!   decoding ahead would only add a copy — but the *first* access to each
//!   memory-mapped page still soft-faults, which costs real time on Windows.
//!   The worker touches the next frame's pages (`TiffStack::prefetch_frame`),
//!   absorbing those faults off the UI thread; the UI thread then decodes
//!   inline, fault-free.
//!
//! Both modes run only during real-time playback that's keeping up (the
//! serial-decode regime). When playback falls behind, the adaptive parallel
//! decode takes over instead.
//!
//! The worker is self-contained: it opens its **own** memory map of the same
//! file (a second mmap shares the OS page cache — no duplicate RAM), so it never
//! touches the app's `TiffStack` and needs no shared/locked state beyond a
//! request channel and a result slot. Correctness is defensive: a prefetched
//! result is used only when its `(generation, frame_index)` and channel layout
//! exactly match what's wanted; any mismatch falls back to inline decode, so a
//! stale prefetch can cost a little work but can never show the wrong frame.

use crate::stack::Stack;
use scivis_render::ChannelKind;
use fast_tiff_lib::{ByteOrder, FrameInfo};
use std::path::PathBuf;
#[cfg(feature = "threads")]
use fast_tiff_lib::TiffStack;
#[cfg(feature = "threads")]
use std::sync::mpsc::{channel, Receiver, Sender};
#[cfg(feature = "threads")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "threads")]
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
    // CMYK frames read through the converting reader, which turns the four ink
    // planes into an R/G/B triple. `plane` indexes the *converted* triple here,
    // not the file's samples.
    let cmyk = frame.is_cmyk();
    Ok(match kind {
        ChannelKind::Int8 => {
            if cmyk {
                Decoded::U8(fast_tiff_lib::read_plane_rgb_u8(mmap, frame, order, plane)?)
            } else if rgb {
                Decoded::U8(fast_tiff_lib::read_plane_u8(mmap, frame, order, plane)?)
            } else {
                Decoded::U8(fast_tiff_lib::read_frame_u8(mmap, frame, order)?.into_owned())
            }
        }
        ChannelKind::Float => Decoded::F32(fast_tiff_lib::read_frame_f32(mmap, frame, order)?.into_owned()),
        ChannelKind::Int16 => {
            if cmyk {
                Decoded::U16(fast_tiff_lib::read_plane_rgb_u16(mmap, frame, order, plane)?)
            } else if rgb {
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
        // CMYK shares this batching: one decompression pass yields all four ink
        // planes, which convert to the R/G/B triple together. Doing it per
        // channel would decompress the same strips three times *and* redo the
        // conversion each time.
        let cmyk = frame.is_cmyk();
        return match jobs[0].kind {
            ChannelKind::Int8 => {
                let mut planes = if cmyk {
                    fast_tiff_lib::read_planes_rgb_u8(mmap, frame, order)?
                } else {
                    fast_tiff_lib::read_planes_u8(mmap, frame, order)?
                };
                jobs.iter().map(|j| Ok(Decoded::U8(take_plane(&mut planes, j.plane)?))).collect()
            }
            _ => {
                let mut planes = if cmyk {
                    fast_tiff_lib::read_planes_rgb_u16(mmap, frame, order)?
                } else {
                    fast_tiff_lib::read_planes_u16(mmap, frame, order, None)?
                };
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

/// The per-channel decode jobs for `frame_index`'s enabled channels, used to
/// decode inline, to ask the prefetch worker for the next frame, and to
/// histogram what is on screen.
///
/// This is the single definition of *which plane is which channel*, which is
/// why it lives next to `ChannelJob` rather than in the GPU-gated `sync`
/// module: every reader of pixel data must agree on the addressing, including
/// the ones that never touch a GPU. Maps each
/// display channel to its IFD/plane: for RGB, all channels are sample planes of
/// one IFD per frame; otherwise each channel is its own IFD in ImageJ's default
/// `xyczt` plane order (channel fastest, then Z — frozen at slice 0 — then time).
pub fn build_jobs(loaded: &Stack, frame_index: usize, enabled: &[bool], kinds: &[ChannelKind]) -> Vec<ChannelJob> {
    let Some((width, height)) = loaded.dimensions() else { return Vec::new() };
    // The *resolved* interpretation, never `tiff.meta`. Resolving is precisely
    // the act of reclassifying a mislabeled axis — a file claiming
    // `channels = 100` that is really a 100-frame movie — so the raw metadata
    // gives the wrong stride and addresses planes past the end of the chain.
    let dims = &loaded.display.dims;
    (0..loaded.display.settings.len())
        .filter(|&c| enabled.get(c).copied().unwrap_or(false))
        .map(|c| {
            let (ifd_idx, plane) = if loaded.display.rgb {
                (frame_index * dims.slices, c)
            } else {
                (frame_index * dims.slices * dims.channels + c, 0)
            };
            ChannelJob { channel: c, ifd_idx, plane, kind: kinds[c], rgb: loaded.display.rgb, width, height }
        })
        .collect()
}

/// Decode only the strips covering `rows`, for every job.
///
/// The reason this exists rather than decoding the frame and slicing it: on a
/// frame too large to upload whole, the viewer only ever shows a window, and
/// decoding the rest is the dominant cost of moving that window. Cropping first
/// makes the cost proportional to what is on screen — on a 40000 x 12788
/// mosaic, tenths of a second instead of seconds, and no need to hold the
/// decoded frame in memory at all.
///
/// Returns the decoded planes in `jobs` order together with the rows they
/// actually cover, which is snapped outward to strip boundaries because a strip
/// is the smallest thing that can be decompressed. Callers must index the
/// planes against *that* range, not the one they asked for.
pub fn decode_jobs_rows(
    mmap: &[u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
    rows: std::ops::Range<u32>,
) -> anyhow::Result<(Vec<Decoded>, std::ops::Range<u32>)> {
    // Crop each referenced IFD once and renumber the jobs onto the cropped
    // set. Jobs that shared an IFD still share one, which is what keeps RGB
    // planes on a single decompression pass.
    let mut band_frames: Vec<FrameInfo> = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut band_jobs: Vec<ChannelJob> = Vec::with_capacity(jobs.len());
    let mut covered: Option<std::ops::Range<u32>> = None;

    for job in jobs {
        let idx = match seen.iter().find(|(orig, _)| *orig == job.ifd_idx) {
            Some((_, mapped)) => *mapped,
            None => {
                let frame = frames
                    .get(job.ifd_idx)
                    .ok_or_else(|| anyhow::anyhow!("frame {} out of range", job.ifd_idx))?;
                let band = frame.crop_rows(rows.clone())?;
                // Every frame in a stack shares frame 0's geometry, so the
                // bands line up; take the first and hold the rest to it.
                match &covered {
                    Some(r) if *r != band.rows => anyhow::bail!(
                        "frame {} cropped to {:?}, expected {:?} — frames disagree on strip layout",
                        job.ifd_idx,
                        band.rows,
                        r
                    ),
                    Some(_) => {}
                    None => covered = Some(band.rows.clone()),
                }
                band_frames.push(band.frame);
                seen.push((job.ifd_idx, band_frames.len() - 1));
                band_frames.len() - 1
            }
        };
        let mut mapped = job.clone();
        mapped.ifd_idx = idx;
        band_jobs.push(mapped);
    }

    let covered = covered.ok_or_else(|| anyhow::anyhow!("nothing to decode"))?;
    Ok((decode_jobs(mmap, &band_frames, order, &band_jobs)?, covered))
}

/// Decode only the pieces covering `cols` x `rows`, for every job.
///
/// The two-axis sibling of [`decode_jobs_rows`]. On a **tiled** frame the
/// columns narrow too, so reading a window costs the window; on a stripped one
/// a strip spans the full width and cannot be split, so the columns are ignored
/// and this behaves exactly like the row version. Callers get the region back
/// and index against that rather than what they asked for.
pub fn decode_jobs_region(
    mmap: &[u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
    cols: std::ops::Range<u32>,
    rows: std::ops::Range<u32>,
) -> anyhow::Result<(Vec<Decoded>, std::ops::Range<u32>, std::ops::Range<u32>)> {
    let mut band_frames: Vec<FrameInfo> = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut band_jobs: Vec<ChannelJob> = Vec::with_capacity(jobs.len());
    let mut covered: Option<(std::ops::Range<u32>, std::ops::Range<u32>)> = None;

    for job in jobs {
        let idx = match seen.iter().find(|(orig, _)| *orig == job.ifd_idx) {
            Some((_, mapped)) => *mapped,
            None => {
                let frame = frames
                    .get(job.ifd_idx)
                    .ok_or_else(|| anyhow::anyhow!("frame {} out of range", job.ifd_idx))?;
                let region = frame.crop(cols.clone(), rows.clone())?;
                let got = (region.cols.clone(), region.rows.clone());
                match &covered {
                    Some(r) if *r != got => anyhow::bail!(
                        "frame {} cropped to {:?}, expected {:?} — frames disagree on layout",
                        job.ifd_idx,
                        got,
                        r
                    ),
                    Some(_) => {}
                    None => covered = Some(got),
                }
                band_frames.push(region.frame);
                seen.push((job.ifd_idx, band_frames.len() - 1));
                band_frames.len() - 1
            }
        };
        let mut mapped = job.clone();
        mapped.ifd_idx = idx;
        band_jobs.push(mapped);
    }

    let (cols, rows) = covered.ok_or_else(|| anyhow::anyhow!("nothing to decode"))?;
    Ok((decode_jobs(mmap, &band_frames, order, &band_jobs)?, cols, rows))
}

/// Decode only the pieces a coarse view actually samples.
///
/// At stride `step * rows_per_piece` a contiguous crop decompresses every piece
/// it spans, and at a coarse zoom most of them hold nothing that will be drawn.
/// This takes every `step`-th piece instead, so the work follows what is
/// sampled. Returns the planes and the band description, which says where each
/// decoded row came from.
///
/// Stripped frames only — a tiled frame narrows by cropping columns instead.
pub fn decode_jobs_stepped(
    mmap: &[u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
    rows: std::ops::Range<u32>,
    step: u32,
) -> anyhow::Result<(Vec<Decoded>, fast_tiff_lib::SampledBand)> {
    let mut band_frames: Vec<FrameInfo> = Vec::new();
    let mut seen: Vec<(usize, usize)> = Vec::new();
    let mut band_jobs: Vec<ChannelJob> = Vec::with_capacity(jobs.len());
    let mut covered: Option<fast_tiff_lib::SampledBand> = None;

    for job in jobs {
        let idx = match seen.iter().find(|(orig, _)| *orig == job.ifd_idx) {
            Some((_, mapped)) => *mapped,
            None => {
                let frame = frames
                    .get(job.ifd_idx)
                    .ok_or_else(|| anyhow::anyhow!("frame {} out of range", job.ifd_idx))?;
                let band = frame.crop_rows_step(rows.clone(), step)?;
                if let Some(r) = &covered {
                    if r.first_row != band.first_row || r.pieces != band.pieces {
                        anyhow::bail!("frames disagree on strip layout");
                    }
                }
                band_frames.push(band.frame.clone());
                covered.get_or_insert(band);
                seen.push((job.ifd_idx, band_frames.len() - 1));
                band_frames.len() - 1
            }
        };
        let mut mapped = job.clone();
        mapped.ifd_idx = idx;
        band_jobs.push(mapped);
    }

    let band = covered.ok_or_else(|| anyhow::anyhow!("nothing to decode"))?;
    Ok((decode_jobs(mmap, &band_frames, order, &band_jobs)?, band))
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

#[cfg(feature = "threads")]
struct Request {
    generation: u64,
    frame_index: usize,
    jobs: Vec<ChannelJob>,
}

/// Read-ahead is a pure optimization, so a host without threads (notably
/// `wasm32-unknown-unknown`) gets a stub that never produces a result. Callers
/// already handle `Prefetcher::new` returning `None` — that's the same path a
/// failed thread spawn takes — so nothing downstream changes.
#[cfg(not(feature = "threads"))]
pub struct Prefetcher(std::convert::Infallible);

#[cfg(not(feature = "threads"))]
impl Prefetcher {
    pub fn new(_path: PathBuf, _touch_only: bool) -> Option<Self> {
        None
    }

    pub fn request(&self, _generation: u64, _frame_index: usize, _jobs: Vec<ChannelJob>) {
        match self.0 {}
    }

    pub fn take_matching(&self, _generation: u64, _frame_index: usize) -> Option<PrefetchResult> {
        match self.0 {}
    }
}

/// Owns the worker thread + the latest result. Dropping it closes the request
/// channel, which ends the worker (it finishes any in-flight decode first).
#[cfg(feature = "threads")]
pub struct Prefetcher {
    tx: Sender<Request>,
    result: Arc<Mutex<Option<PrefetchResult>>>,
    _handle: JoinHandle<()>,
}

#[cfg(feature = "threads")]
impl Prefetcher {
    /// Spawn a worker that opens its own map of `path`. `touch_only` selects
    /// the page-touch mode (uncompressed stacks — see module docs); otherwise
    /// the worker decodes ahead. Returns `None` if the thread or the worker's
    /// file open fails — callers then just decode inline.
    pub fn new(path: PathBuf, touch_only: bool) -> Option<Self> {
        let (tx, rx) = channel::<Request>();
        let result = Arc::new(Mutex::new(None));
        let result_worker = Arc::clone(&result);
        let handle = std::thread::Builder::new()
            .name("fasttiff-prefetch".to_owned())
            .spawn(move || {
                // Second mmap of the same file: shares the OS page cache, so no
                // duplicate pixel RAM; the IFD walk is a one-time cost.
                match TiffStack::open(&path) {
                    Ok(stack) => worker_loop(stack, rx, result_worker, touch_only),
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

#[cfg(feature = "threads")]
fn worker_loop(
    stack: TiffStack,
    rx: Receiver<Request>,
    result: Arc<Mutex<Option<PrefetchResult>>>,
    touch_only: bool,
) {
    // Block for a request; channel closed (Prefetcher dropped) -> exit.
    while let Ok(mut req) = rx.recv() {
        // Skip superseded predictions: only the most recent request matters.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        // Page-touch mode: absorb the next frame's soft page faults here and
        // store no result — the UI thread's inline decode is then fault-free
        // (and stays zero-copy, which decoding ahead would have forfeited).
        if touch_only {
            for job in &req.jobs {
                if let Some(frame) = stack.frames.get(job.ifd_idx) {
                    stack.prefetch_frame(frame);
                }
            }
            continue;
        }
        let mut channels = Vec::with_capacity(req.jobs.len());
        let ok = match decode_jobs(&stack.data, &stack.frames, stack.byte_order, &req.jobs) {
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

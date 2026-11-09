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
//! The worker is a `LatestWorker` ([`crate::worker`]): asking for a different
//! frame replaces the request and cancels the read-ahead under way, which is
//! stale by then. It reads the window's own `TiffStack` through a shared `Arc`.
//! Nothing mutates a stack once it is indexed, so that needs no lock. Sharing
//! it also means the pages the worker touches are the ones the window then
//! reads, and that a stack held in memory, with no file behind it, gets a
//! worker too. Correctness is defensive: a prefetched result is used only when
//! its `(revision, frame_index)` — the display [`Revision`] it was asked for
//! under — and its channel layout exactly match what's wanted; any mismatch
//! falls back to inline decode, so a stale prefetch can cost a little work but
//! can never show the wrong frame.

use crate::display::Revision;
use crate::stack::Stack;
use crate::worker::{CancelToken, LatestWorker, Unclaimed};
use fast_tiff_lib::TiffStack;
use fast_tiff_lib::{ByteOrder, FrameInfo};
use scivis_render::ChannelKind;
use std::borrow::Cow;
use std::sync::Arc;

/// One channel's decoded pixels, in the format its GPU texture expects.
pub enum Decoded {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Decoded {
    /// Bytes these samples occupy — what they cost to hold, and what they cost
    /// to upload. Not the sample *count*: an `F32` plane is four times the
    /// bytes of a `U8` plane of the same length, and anything budgeting memory
    /// against a count under-counts it fourfold.
    pub fn bytes(&self) -> usize {
        match self {
            Decoded::U8(v) => v.len(),
            Decoded::U16(v) => v.len() * 2,
            Decoded::F32(v) => v.len() * 4,
        }
    }

    /// The samples, borrowed.
    pub fn view(&self) -> Samples<'_> {
        match self {
            Decoded::U8(v) => Samples::U8(v),
            Decoded::U16(v) => Samples::U16(v),
            Decoded::F32(v) => Samples::F32(v),
        }
    }
}

/// One channel's pixels, borrowed: what an upload, a cut or a histogram reads,
/// whoever owns them — a [`Decoded`] plane, or the file's own mapping.
#[derive(Clone, Copy)]
pub enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

/// One channel's decoded pixels that may still be the file's own bytes.
///
/// An uncompressed, single-strip frame in the host's byte order is already the
/// plane the GPU wants, and the decoder hands it back as a borrow of the
/// mapping. Keeping that borrow until the upload is what makes the display of
/// such a frame one copy (into the texture) instead of two: owning it first
/// meant an allocation, a memcpy out of the mapping and a free per channel per
/// frame — 8 MiB each way for a 2048 x 2048 16-bit plane, and on Windows about
/// two thousand fresh pages to fault in. Only data that has to outlive the
/// frame — a prefetched result, a cache entry — is made owned
/// ([`into_owned`](Self::into_owned)).
pub enum DecodedRef<'a> {
    U8(Cow<'a, [u8]>),
    U16(Cow<'a, [u16]>),
    F32(Cow<'a, [f32]>),
}

impl DecodedRef<'_> {
    /// The samples, borrowed.
    pub fn view(&self) -> Samples<'_> {
        match self {
            DecodedRef::U8(v) => Samples::U8(v),
            DecodedRef::U16(v) => Samples::U16(v),
            DecodedRef::F32(v) => Samples::F32(v),
        }
    }

    /// Owned pixels, copying out of the mapping only if they still borrow it.
    pub fn into_owned(self) -> Decoded {
        match self {
            DecodedRef::U8(v) => Decoded::U8(v.into_owned()),
            DecodedRef::U16(v) => Decoded::U16(v.into_owned()),
            DecodedRef::F32(v) => Decoded::F32(v.into_owned()),
        }
    }
}

impl From<Decoded> for DecodedRef<'_> {
    fn from(d: Decoded) -> Self {
        match d {
            Decoded::U8(v) => DecodedRef::U8(Cow::Owned(v)),
            Decoded::U16(v) => DecodedRef::U16(Cow::Owned(v)),
            Decoded::F32(v) => DecodedRef::F32(Cow::Owned(v)),
        }
    }
}

/// Decode one channel of a frame, borrowing the file's bytes where the decoder
/// can (see [`DecodedRef`]). `plane`/`rgb` select the RGB-plane deinterleave;
/// otherwise the whole image.
fn decode_channel<'a>(
    mmap: &'a [u8],
    frame: &FrameInfo,
    order: ByteOrder,
    kind: ChannelKind,
    plane: usize,
    rgb: bool,
) -> anyhow::Result<DecodedRef<'a>> {
    // CMYK frames read through the converting reader, which turns the four ink
    // planes into an R/G/B triple. `plane` indexes the *converted* triple here,
    // not the file's samples.
    let cmyk = frame.is_cmyk();
    Ok(match kind {
        ChannelKind::Int8 => DecodedRef::U8(if cmyk {
            Cow::Owned(fast_tiff_lib::read_plane_rgb_u8(mmap, frame, order, plane)?)
        } else if rgb {
            Cow::Owned(fast_tiff_lib::read_plane_u8(mmap, frame, order, plane)?)
        } else {
            fast_tiff_lib::read_frame_u8(mmap, frame, order)?
        }),
        ChannelKind::Float => DecodedRef::F32(fast_tiff_lib::read_frame_f32(mmap, frame, order)?),
        ChannelKind::Int16 => DecodedRef::U16(if cmyk {
            Cow::Owned(fast_tiff_lib::read_plane_rgb_u16(
                mmap, frame, order, plane,
            )?)
        } else if rgb {
            Cow::Owned(fast_tiff_lib::read_plane_u16(
                mmap, frame, order, None, plane,
            )?)
        } else {
            fast_tiff_lib::read_frame_u16(mmap, frame, order, None)?
        }),
    })
}

/// Decode all of one frame-request's channels into owned pixels, in `jobs`
/// order: [`decode_jobs_borrowed`], for a result that outlives the frame —
/// the prefetch worker's, a cache entry, a histogram source.
pub fn decode_jobs(
    mmap: &[u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
) -> anyhow::Result<Vec<Decoded>> {
    Ok(decode_jobs_borrowed(mmap, frames, order, jobs)?
        .into_iter()
        .map(DecodedRef::into_owned)
        .collect())
}

/// Decode all of one frame-request's channels, in `jobs` order, borrowing the
/// file's bytes wherever they already are the plane (see [`DecodedRef`]).
/// Shared by the inline upload and, through [`decode_jobs`], the prefetch
/// worker, so both produce byte-identical results.
///
/// RGB channels are sample planes of the *same* IFD, so they're decoded with a
/// single decompression pass (`read_planes_*`) instead of one full decode per
/// channel — ~3x cheaper on compressed RGB. Non-RGB (and float) jobs decode
/// per-channel as before.
pub fn decode_jobs_borrowed<'a>(
    mmap: &'a [u8],
    frames: &[FrameInfo],
    order: ByteOrder,
    jobs: &[ChannelJob],
) -> anyhow::Result<Vec<DecodedRef<'a>>> {
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
                jobs.iter()
                    .map(|j| {
                        Ok(DecodedRef::U8(Cow::Owned(take_plane(
                            &mut planes,
                            j.plane,
                        )?)))
                    })
                    .collect()
            }
            _ => {
                let mut planes = if cmyk {
                    fast_tiff_lib::read_planes_rgb_u16(mmap, frame, order)?
                } else {
                    fast_tiff_lib::read_planes_u16(mmap, frame, order, None)?
                };
                jobs.iter()
                    .map(|j| {
                        Ok(DecodedRef::U16(Cow::Owned(take_plane(
                            &mut planes,
                            j.plane,
                        )?)))
                    })
                    .collect()
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
        anyhow::bail!(
            "sample plane {plane} out of range ({} planes)",
            planes.len()
        );
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

/// The per-channel decode jobs for `frame_index`'s channels that `enabled`
/// selects (one flag per display channel), used to decode inline, to ask the
/// prefetch worker for the next frame, and to histogram what is on screen.
/// Each channel decodes to its own texture format, read from the display.
///
/// Addressing itself lives in [`crate::dimensions::plane_index`], the single
/// definition every reader of pixel data shares — including the ones that never
/// touch a GPU. This function decides only *which* channels to fetch for the 2D
/// view and at which timepoint; it asks for slice 0, because the 2D scrub axis
/// is time and Z is a stride there.
pub fn build_jobs(loaded: &Stack, frame_index: usize, enabled: &[bool]) -> Vec<ChannelJob> {
    let Some((width, height)) = loaded.dimensions() else {
        return Vec::new();
    };
    // The *resolved* interpretation, never `tiff.meta`. Resolving is precisely
    // the act of reclassifying a mislabeled axis — a file claiming
    // `channels = 100` that is really a 100-frame movie — so the raw metadata
    // gives the wrong stride and addresses planes past the end of the chain.
    let dims = loaded.display.dims();
    let rgb = loaded.display.rgb();
    loaded
        .display
        .settings()
        .iter()
        .enumerate()
        .filter(|&(c, _)| enabled.get(c).copied().unwrap_or(false))
        .map(|(c, s)| {
            // z = 0 explicitly: the 2D view's scrub axis is time, and Z acts
            // only as a stride (see `planes_addressed`). Writing the zero here
            // rather than folding it into the arithmetic is what keeps this
            // from reading like a general plane lookup, which it is not.
            let (ifd_idx, plane) = crate::dimensions::plane_index(dims, rgb, c, 0, frame_index);
            ChannelJob {
                channel: c,
                ifd_idx,
                plane,
                kind: s.kind(),
                rgb,
                width,
                height,
            }
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
    Ok((
        decode_jobs(mmap, &band_frames, order, &band_jobs)?,
        cols,
        rows,
    ))
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
    /// The display revision the frame was asked for under: its channel layout
    /// and which channels were on.
    pub revision: Revision,
    pub frame_index: usize,
    pub channels: Vec<DecodedChannel>,
}

/// What a read-ahead is asked for by: the display revision and the frame.
type Key = (Revision, usize);

/// What the read-ahead worker is asked for, besides the [`Key`] its answer is
/// keyed by.
struct Request {
    jobs: Vec<ChannelJob>,
    /// Whether the requester decodes in parallel: read-ahead decodes the way
    /// the view would have.
    parallel: bool,
}

/// The read-ahead worker: a `LatestWorker` ([`crate::worker`]) keyed by
/// `(revision, frame_index)`, so asking for the next frame again on every
/// repaint is free and asking for a different one cancels the old.
///
/// The whole [`Revision`], not only its layout: a frame is read ahead for the
/// channels that are on, so one decoded before a channel was switched on lacks
/// it, and one decoded before an override reads another mapping's planes.
/// Either way a new request is a new key, which cancels the stale decode, and
/// the stale answer is never taken.
///
/// `new` returns `None` when the thread cannot start, and always without the
/// `threads` feature (notably `wasm32-unknown-unknown`). Read-ahead is a pure
/// optimization, so callers then decode inline and nothing downstream changes.
pub struct Prefetcher(LatestWorker<Key, Request, PrefetchResult>);

impl Prefetcher {
    /// Spawn a worker over `stack` — the window's own, shared: no second map,
    /// no second walk of the IFD chain, and the pages it touches are the ones
    /// the window then reads. `touch_only` selects the page-touch mode
    /// (uncompressed stacks — see module docs); otherwise the worker decodes
    /// ahead. Returns `None` if the thread fails to spawn — callers then just
    /// decode inline.
    pub fn new(stack: Arc<TiffStack>, touch_only: bool) -> Option<Self> {
        let job = move |&(revision, frame_index): &Key, req: Request, cancel: &CancelToken| {
            fast_tiff_lib::set_thread_parallel_decode(Some(req.parallel));
            // Page-touch mode: absorb the next frame's soft page faults here
            // and publish nothing — the UI thread's inline decode is then
            // fault-free (and stays zero-copy, which decoding ahead would have
            // forfeited). A frame's channels are planes of their own, so a
            // newer request stops it between them.
            if touch_only {
                for job in &req.jobs {
                    if cancel.is_cancelled() {
                        break;
                    }
                    if let Some(frame) = stack.frames.get(job.ifd_idx) {
                        stack.prefetch_frame(frame);
                    }
                }
                return None;
            }
            // The step is the frame: a decode overtaken while it runs finishes,
            // and the worker then drops it unpublished.
            let decoded =
                decode_jobs(&stack.data, &stack.frames, stack.byte_order, &req.jobs).ok()?;
            let channels = req
                .jobs
                .iter()
                .zip(decoded)
                .map(|(job, data)| DecodedChannel {
                    channel: job.channel,
                    width: job.width,
                    height: job.height,
                    kind: job.kind,
                    data,
                })
                .collect();
            Some(PrefetchResult {
                revision,
                frame_index,
                channels,
            })
        };
        LatestWorker::spawn("fasttiff-prefetch", Unclaimed::Keep, job).map(Prefetcher)
    }

    /// Ask the worker to decode `frame_index`'s `jobs`, built under display
    /// `revision`. Fire-and-forget. A different frame (or revision) replaces
    /// whatever was asked before and cancels it; the same one again is ignored
    /// while it is still pending.
    pub fn request(&self, revision: Revision, frame_index: usize, jobs: Vec<ChannelJob>) {
        self.0.request(
            (revision, frame_index),
            Request {
                jobs,
                parallel: fast_tiff_lib::parallel_decode_enabled(),
            },
        );
    }

    /// Take the prefetched result iff it matches `(revision, frame_index)`;
    /// otherwise leave the slot untouched and return `None`. The caller still
    /// verifies the channel layout before using it.
    pub fn take_matching(&self, revision: Revision, frame_index: usize) -> Option<PrefetchResult> {
        self.0.take_if(|key| *key == (revision, frame_index))
    }
}

#[cfg(test)]
#[path = "prefetch_tests.rs"]
mod tests;

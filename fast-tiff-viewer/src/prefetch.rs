//! Decoding a frame's channels for the GPU, and reading the next frame ahead
//! for playback.
//!
//! # One decode
//!
//! Every reader of the pixels a channel shows — the upload, the read-ahead,
//! the histograms, a window of a frame too large for one texture, the copy —
//! goes through [`decode_jobs`]: one request's channels ([`ChannelJob`]s),
//! read through fast-tiff-lib's [`Frame`] in the units the channel's texture
//! holds, of the whole frame or of a [`Crop`] of it. A channel whose samples
//! the file already holds as the texture wants them (an uncompressed plane in
//! the host's byte order) is borrowed straight from the stack; any other is
//! decoded into that channel's buffer in a [`ChannelBufs`] the caller keeps.
//! A caller that keeps its buffers and its [`ReadScratch`] from one frame to
//! the next decodes a steady stream of frames with no frame-sized allocation
//! and no zero-fill: the interface thread's are its residency's
//! ([`InlineDecode`](crate::residency::InlineDecode)), the read-ahead's its
//! worker's.
//!
//! # Read-ahead
//!
//! While the UI thread uploads and renders the current frame, a worker thread
//! prepares the *next* one, in one of two modes, chosen per request from the
//! library's zero-copy predicate ([`Frame::borrows`]):
//!
//! - **Decode-ahead**, when any channel of the frame has to be decoded: the
//!   worker decodes the next frame into buffers handed back after the last
//!   one was shown ([`Prefetcher::recycle`]), so reaching it is just an
//!   upload — a steady second-core pipeline with none of the per-frame
//!   fork/join overhead of intra-frame parallel decode, and no allocation.
//! - **Page-touch**, when every channel is read straight from the file:
//!   decoding ahead would only add a copy, but the *first* access to each
//!   memory-mapped page still soft-faults, which costs real time on Windows.
//!   The worker touches the next frame's pages ([`Frame::touch`]), absorbing
//!   those faults off the UI thread; the UI thread then borrows the frame
//!   inline, fault-free.
//!
//! Which it is follows the file, not the compression tag: an uncompressed
//! big-endian frame has to be byte-swapped, so it is decoded ahead, and an
//! uncompressed frame in many strips laid end to end is one run, so it is
//! touched.
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
use crate::document::Document;
use crate::histogram::{Binning, Histogram};
use crate::worker::{CancelToken, LatestWorker, Unclaimed};
use fast_tiff_lib::units::{CmykRgb16, CmykRgb8, Display, File, Stored, Units};
use fast_tiff_lib::{ByteOrder, Coverage, Crop, DecodeScratch, Frame, Pixels, TiffStack};
use scivis_render::ChannelKind;
use std::sync::{Arc, Mutex};

/// One channel's decoded pixels, in the format its GPU texture expects.
pub(crate) enum Decoded {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Decoded {
    /// A buffer holding nothing, and no allocation.
    pub(crate) const fn empty() -> Self {
        Decoded::U8(Vec::new())
    }

    /// Bytes these samples occupy — what they cost to hold, and what they cost
    /// to upload. Not the sample *count*: an `F32` plane is four times the
    /// bytes of a `U8` plane of the same length, and anything budgeting memory
    /// against a count under-counts it fourfold.
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Decoded::U8(v) => v.len(),
            Decoded::U16(v) => v.len() * 2,
            Decoded::F32(v) => v.len() * 4,
        }
    }

    /// The samples, borrowed.
    pub(crate) fn view(&self) -> Samples<'_> {
        match self {
            Decoded::U8(v) => Samples::U8(v),
            Decoded::U16(v) => Samples::U16(v),
            Decoded::F32(v) => Samples::F32(v),
        }
    }

    /// Make this a copy of `samples`, in its allocation when that is of the
    /// same sample type and large enough. Nothing is zero-filled on the way.
    fn copy_from(&mut self, samples: Samples<'_>) {
        fn copy<T: Lane>(d: &mut Decoded, s: &[T]) {
            let v = T::buffer(d);
            v.clear();
            v.extend_from_slice(s);
        }
        match samples {
            Samples::U8(s) => copy(self, s),
            Samples::U16(s) => copy(self, s),
            Samples::F32(s) => copy(self, s),
        }
    }
}

/// One channel's pixels, borrowed: what an upload, a cut or a histogram reads,
/// whoever owns them — a [`Decoded`] plane, or the file's own mapping.
#[derive(Clone, Copy)]
pub(crate) enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

/// A sample type a channel's texture holds, as a [`Decoded`] buffer and as
/// borrowed [`Samples`].
trait Lane: bytemuck::Pod + Default + Send + Sync {
    fn samples(s: &[Self]) -> Samples<'_>;
    /// `d` as a buffer of this type, first made one, empty, if it was a
    /// buffer of another.
    fn buffer(d: &mut Decoded) -> &mut Vec<Self>;
}

macro_rules! lane {
    ($t:ty, $variant:ident) => {
        impl Lane for $t {
            fn samples(s: &[Self]) -> Samples<'_> {
                Samples::$variant(s)
            }
            fn buffer(d: &mut Decoded) -> &mut Vec<Self> {
                if !matches!(d, Decoded::$variant(_)) {
                    *d = Decoded::$variant(Vec::new());
                }
                match d {
                    Decoded::$variant(v) => v,
                    _ => unreachable!("made one just above"),
                }
            }
        }
    };
}
lane!(u8, U8);
lane!(u16, U16);
lane!(f32, F32);

/// The buffers one owner decodes channels into, one per display channel and
/// kept from one frame to the next: a buffer already the plane's size is
/// decoded into as it is, with no allocation and no zero-fill.
///
/// Indexed by display channel rather than by a request's job, so the channels
/// a sync shows and the switched-off ones the histograms decode next to them
/// each keep a buffer of their own.
#[derive(Default)]
pub(crate) struct ChannelBufs(Vec<Decoded>);

impl ChannelBufs {
    /// Channel `channel`'s buffer, made (empty) if there is none yet.
    fn slot(&mut self, channel: usize) -> &mut Decoded {
        if self.0.len() <= channel {
            self.0.resize_with(channel + 1, Decoded::empty);
        }
        &mut self.0[channel]
    }

    /// What channel `channel`'s buffer holds; nothing if it has none.
    fn view(&self, channel: usize) -> Samples<'_> {
        self.get(channel).view()
    }

    /// Channel `channel`'s buffer; an empty one if it has none.
    pub(crate) fn get(&self, channel: usize) -> &Decoded {
        const EMPTY: &Decoded = &Decoded::empty();
        self.0.get(channel).unwrap_or(EMPTY)
    }

    /// Take channel `channel`'s buffer, leaving an empty one.
    pub(crate) fn take(&mut self, channel: usize) -> Decoded {
        std::mem::replace(self.slot(channel), Decoded::empty())
    }

    /// Give channel `channel` `buffer` to decode into next.
    pub(crate) fn put(&mut self, channel: usize, buffer: Decoded) {
        *self.slot(channel) = buffer;
    }
}

/// What [`decode_jobs`] needs besides the channels' own buffers, kept by the
/// same owner: the library's [`DecodeScratch`], and a plane for each plane a
/// batched read has to decode that no channel shown wants — the green of an
/// RGB frame whose green channel is switched off, which one pass over the
/// frame decodes with the red and the blue.
#[derive(Default)]
pub(crate) struct ReadScratch {
    pub(crate) decode: DecodeScratch,
    unwanted: Vec<Decoded>,
}

impl ReadScratch {
    /// Give back everything it holds: after a one-off read whose buffers are
    /// not worth keeping (a window of a frame too large for one texture).
    pub(crate) fn release(&mut self) {
        self.decode.release();
        self.unwanted = Vec::new();
    }
}

/// What [`decode_jobs`] produced, per job in `jobs` order: the file's own
/// samples where a channel borrowed them, and otherwise that channel's
/// buffer, which the read filled.
///
/// Borrows the stack, never the buffers, so it can be kept beside them — the
/// copy keeps one band of each of several IFDs at once.
pub(crate) struct Planes<'a> {
    /// Each job's display channel, and its samples when they are the file's.
    got: Vec<(usize, Option<Samples<'a>>)>,
    /// What part of the frame a crop covers; `None` for a whole frame.
    pub(crate) coverage: Option<Coverage>,
    /// How many rows each plane holds: the frame's height, or the crop's.
    pub(crate) height: u32,
}

impl Planes<'_> {
    /// Job `i`'s samples, with `bufs` the buffers the decode was given.
    pub(crate) fn view<'s>(&'s self, i: usize, bufs: &'s ChannelBufs) -> Samples<'s> {
        let (channel, file) = self.got[i];
        file.unwrap_or_else(|| bufs.view(channel))
    }

    /// Every job's samples, in `jobs` order.
    pub(crate) fn views<'s>(
        &'s self,
        bufs: &'s ChannelBufs,
    ) -> impl Iterator<Item = Samples<'s>> + 's {
        (0..self.got.len()).map(move |i| self.view(i, bufs))
    }

    /// Whether job `i`'s samples are the file's own, borrowed.
    #[cfg(test)]
    pub(crate) fn is_file(&self, i: usize) -> bool {
        self.got[i].1.is_some()
    }

    /// Make every job's buffer hold its samples, copying the ones borrowed
    /// from the file into theirs: for a result that outlives the frame — a
    /// read-ahead, a cached band.
    pub(crate) fn own(&self, bufs: &mut ChannelBufs) {
        for &(channel, file) in &self.got {
            if let Some(samples) = file {
                bufs.slot(channel).copy_from(samples);
            }
        }
    }
}

/// What reading one channel comes to, in fast-tiff-lib's units.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// An `R8Uint` channel: the file's own bytes.
    Stored(ByteOrder),
    /// An `R16Uint` channel: the decoder's 16-bit display scale, a wide
    /// plane ranged to its own values.
    Display,
    /// An `R32F` channel: the file's own values.
    File,
    /// The red, green or blue of a CMYK frame, as bytes or on the 16-bit
    /// scale.
    CmykRgb8,
    CmykRgb16,
}

/// Run `$body` with `$u` bound to the units marker `$reading` names: one
/// copy of the body for each, since the units are types.
macro_rules! with_units {
    ($reading:expr, $u:ident => $body:expr) => {
        match $reading {
            Reading::Stored(order) => {
                let $u = Stored(order);
                $body
            }
            Reading::Display => {
                let $u = Display { range: None };
                $body
            }
            Reading::File => {
                let $u = File;
                $body
            }
            Reading::CmykRgb8 => {
                let $u = CmykRgb8;
                $body
            }
            Reading::CmykRgb16 => {
                let $u = CmykRgb16;
                $body
            }
        }
    };
}

/// How `job` reads `frame`: its units, and which of the read's planes.
///
/// What each texture format has always been decoded as. A float channel is
/// the file's own values of the frame's first plane. An 8- or 16-bit channel
/// of a CMYK frame is one of the red, green and blue its inks convert to;
/// one of an RGB frame is its sample plane, as bytes or on the display scale;
/// and any other is the frame's first plane, likewise. An 8-bit RGB channel
/// is refused for a frame that is not 8-bit, as the reader it replaced
/// refused it: its bytes would be another depth's.
fn reading(job: &ChannelJob, frame: &Frame<'_>) -> anyhow::Result<(Reading, usize)> {
    let info = frame.info();
    let cmyk = info.is_cmyk();
    Ok(match job.kind {
        ChannelKind::Float => (Reading::File, 0),
        ChannelKind::Int8 if cmyk => (Reading::CmykRgb8, job.plane),
        ChannelKind::Int8 if job.rgb => {
            anyhow::ensure!(
                info.bits_per_sample == 8,
                "an 8-bit RGB channel needs 8-bit samples, got {}",
                info.bits_per_sample
            );
            (Reading::Stored(frame.byte_order()), job.plane)
        }
        ChannelKind::Int8 => (Reading::Stored(frame.byte_order()), 0),
        ChannelKind::Int16 if cmyk => (Reading::CmykRgb16, job.plane),
        ChannelKind::Int16 if job.rgb => (Reading::Display, job.plane),
        ChannelKind::Int16 => (Reading::Display, 0),
    })
}

/// Plane `plane` of `frame` in `units`, for one channel: the file's own
/// samples when they already are the answer, and otherwise decoded into
/// `buf`.
fn read_one<'a, U>(
    frame: &Frame<'a>,
    units: U,
    plane: usize,
    buf: &mut Decoded,
    scratch: &mut DecodeScratch,
) -> fast_tiff_lib::Result<Option<Samples<'a>>>
where
    U: Units,
    U::Sample: Lane,
{
    if let Some(Pixels::Borrowed(file)) = frame.borrowed(units, plane) {
        return Ok(Some(U::Sample::samples(file)));
    }
    frame.read(units, plane, U::Sample::buffer(buf), scratch)?;
    Ok(None)
}

/// Planes of one frame for several channels at once, from as few passes over
/// the frame as its layout allows: `wanted` pairs each job (an index into
/// `got`) with its plane, the planes distinct and ascending.
///
/// A plane borrowed from the file is borrowed. The rest are read together:
/// a chunky frame (and any CMYK one, whose every plane needs all four inks)
/// in one pass over the span of planes asked for, with a buffer from
/// `unwanted` for a plane in between that nobody asked for; a planar frame,
/// whose planes are strips of their own, one pass per run of consecutive
/// planes, so no plane is decoded that is not shown.
#[allow(clippy::too_many_arguments)]
fn read_planes<'a, U>(
    frame: &Frame<'a>,
    units: U,
    wanted: &[(usize, usize)],
    channels: &[usize],
    bufs: &mut ChannelBufs,
    got: &mut [(usize, Option<Samples<'a>>)],
    scratch: &mut ReadScratch,
) -> anyhow::Result<()>
where
    U: Units,
    U::Sample: Lane,
{
    let planes = frame.planes(units);
    if let Some(&(_, plane)) = wanted.iter().find(|&&(_, p)| p >= planes) {
        anyhow::bail!("sample plane {plane} out of range ({planes} planes)");
    }
    // At most one plane per channel, and fewer than MAX_CHANNELS of them, so
    // these lists stay small.
    let mut rest: Vec<(usize, usize)> = Vec::with_capacity(wanted.len());
    for &(job, plane) in wanted {
        match frame.borrowed(units, plane) {
            Some(Pixels::Borrowed(file)) => got[job].1 = Some(U::Sample::samples(file)),
            _ => rest.push((job, plane)),
        }
    }
    if rest.is_empty() {
        return Ok(());
    }
    let n = frame.plane_len(units)?;
    let one_pass = !frame.info().is_planar() || matches!(planes, 3 if frame.info().is_cmyk());
    let mut at = 0;
    while at < rest.len() {
        // The planes this pass reads: every one left, or a planar frame's
        // next run of consecutive ones.
        let mut end = at + 1;
        while end < rest.len() && (one_pass || rest[end].1 == rest[end - 1].1 + 1) {
            end += 1;
        }
        let pass = &rest[at..end];
        let (first, last) = (pass[0].1, pass[pass.len() - 1].1);
        // A destination per plane of the span: the channel asking for it, or
        // one nobody shows.
        let gaps = (last - first + 1) - pass.len();
        if scratch.unwanted.len() < gaps {
            scratch.unwanted.resize_with(gaps, Decoded::empty);
        }
        let mut shown: Vec<(usize, &mut Vec<U::Sample>)> = bufs
            .0
            .iter_mut()
            .enumerate()
            .filter_map(|(channel, d)| {
                let job = pass.iter().find(|&&(j, _)| channels[j] == channel)?;
                Some((job.1, U::Sample::buffer(d)))
            })
            .collect();
        shown.sort_by_key(|&(plane, _)| plane);
        let mut spare = scratch.unwanted.iter_mut().map(U::Sample::buffer);
        let mut shown = shown.into_iter().peekable();
        let mut dests: Vec<&mut [U::Sample]> = Vec::with_capacity(last - first + 1);
        for plane in first..=last {
            let v = match shown.next_if(|&(p, _)| p == plane) {
                Some((_, v)) => v,
                None => spare.next().expect("a spare for every gap"),
            };
            if v.len() != n {
                v.resize(n, U::Sample::default());
            }
            dests.push(&mut v[..]);
        }
        frame.read_into(units, first, &mut dests, &mut scratch.decode)?;
        at = end;
    }
    Ok(())
}

/// Decode `jobs`, one display channel each, of the whole frame or of `crop`:
/// each channel borrowed from `tiff` where the file already holds its
/// samples as its texture wants them, and otherwise decoded into its buffer
/// in `bufs`, sized to the plane only when it is not already.
///
/// The one decode behind every reader of the shown pixels, so they all agree
/// byte for byte: the upload and the read-ahead, the histograms, the windows
/// of a frame too large for one texture, and the copy.
///
/// An RGB or CMYK frame's channels are planes of one IFD, so they are read
/// together, in one decompression pass ([`read_planes`]) instead of one full
/// decode per channel — about three times cheaper on compressed RGB. Any
/// other channel is read on its own.
///
/// A crop is snapped outward to whole pieces ([`Frame::crop`]); every IFD the
/// jobs read must snap to the same rows (and columns), or it is an error, and
/// the result's [`Planes::coverage`] says what they cover — what a caller
/// indexes against, rather than what it asked for.
pub(crate) fn decode_jobs<'a>(
    tiff: &'a TiffStack,
    jobs: &[ChannelJob],
    crop: Option<&Crop>,
    bufs: &mut ChannelBufs,
    scratch: &mut ReadScratch,
) -> anyhow::Result<Planes<'a>> {
    if crop.is_some() && jobs.is_empty() {
        anyhow::bail!("nothing to decode");
    }
    // Each IFD the jobs read, cropped once: jobs that share one (an RGB
    // frame's planes) share its frame.
    let mut frames: Vec<(usize, Frame<'a>)> = Vec::with_capacity(jobs.len());
    let mut coverage: Option<Coverage> = None;
    for job in jobs {
        if frames.iter().any(|(idx, _)| *idx == job.ifd_idx) {
            continue;
        }
        let frame = tiff.frame(job.ifd_idx)?;
        let frame = match crop {
            None => frame,
            Some(crop) => {
                let (band, got) = frame.crop(crop.clone())?;
                // Every frame in a stack shares frame 0's geometry, so the
                // bands line up; take the first and hold the rest to it.
                match &coverage {
                    Some(c) if !same_cover(crop, c, &got) => anyhow::bail!(
                        "frame {} cropped to {:?}, expected {:?} — frames disagree on strip \
                         layout",
                        job.ifd_idx,
                        (&got.cols, &got.rows),
                        (&c.cols, &c.rows)
                    ),
                    Some(_) => {}
                    None => coverage = Some(got),
                }
                band
            }
        };
        frames.push((job.ifd_idx, frame));
    }
    let frame_of = |job: &ChannelJob| {
        &frames
            .iter()
            .find(|(idx, _)| *idx == job.ifd_idx)
            .expect("cropped above")
            .1
    };
    let height = jobs.first().map_or(0, |j| frame_of(j).info().height);
    let mut got: Vec<(usize, Option<Samples<'a>>)> =
        jobs.iter().map(|j| (j.channel, None)).collect();
    let channels: Vec<usize> = jobs.iter().map(|j| j.channel).collect();

    // Batched: every job is a plane of one IFD with one integer kind.
    let batched = jobs.len() > 1
        && jobs
            .iter()
            .all(|j| j.rgb && j.ifd_idx == jobs[0].ifd_idx && j.kind == jobs[0].kind)
        && matches!(jobs[0].kind, ChannelKind::Int8 | ChannelKind::Int16);
    if batched {
        let frame = frame_of(&jobs[0]);
        let (reading, _) = reading(&jobs[0], frame)?;
        let mut wanted: Vec<(usize, usize)> =
            jobs.iter().enumerate().map(|(i, j)| (i, j.plane)).collect();
        wanted.sort_by_key(|&(_, plane)| plane);
        // Display channels map to distinct planes by construction; two jobs
        // on one plane would need one destination twice, so they read one by
        // one instead.
        if wanted.windows(2).all(|w| w[0].1 != w[1].1) {
            with_units!(reading, units => read_planes(
                frame, units, &wanted, &channels, bufs, &mut got, scratch
            ))?;
            return Ok(Planes {
                got,
                coverage,
                height,
            });
        }
    }

    for (i, job) in jobs.iter().enumerate() {
        let frame = frame_of(job);
        let (reading, plane) = reading(job, frame)?;
        let buf = bufs.slot(job.channel);
        got[i].1 = with_units!(reading, units => read_one(
            frame, units, plane, buf, &mut scratch.decode
        ))?;
    }
    Ok(Planes {
        got,
        coverage,
        height,
    })
}

/// Whether two IFDs cropped alike, by what each kind of crop's caller
/// indexes against: the rows of a band, the rows and columns of a region,
/// and where a sampled band starts and how many pieces it takes.
fn same_cover(crop: &Crop, a: &Coverage, b: &Coverage) -> bool {
    match crop {
        Crop::Rows(_) => a.rows == b.rows,
        Crop::Region { .. } => (&a.cols, &a.rows) == (&b.cols, &b.rows),
        Crop::Sampled { .. } => a.first_row == b.first_row && a.pieces == b.pieces,
    }
}

/// Whether every channel of `jobs` is read straight out of `tiff`'s bytes,
/// with nothing decoded: what makes touching a frame's pages ahead the
/// read-ahead it wants, rather than decoding it ahead (see the
/// [module docs](self)). The library's zero-copy predicate, asked of the
/// frames and units the jobs read; `false` for anything it cannot read.
pub(crate) fn touch_suffices(tiff: &TiffStack, jobs: &[ChannelJob]) -> bool {
    !jobs.is_empty()
        && jobs.iter().all(|job| {
            let Ok(frame) = tiff.frame(job.ifd_idx) else {
                return false;
            };
            let Ok((reading, plane)) = reading(job, &frame) else {
                return false;
            };
            with_units!(reading, units => frame.borrows(units, plane))
        })
}

/// [`decode_jobs`] of the whole frame into fresh buffers, owned: for tests
/// that look at what a decode produced.
#[cfg(test)]
pub(crate) fn decode_owned(tiff: &TiffStack, jobs: &[ChannelJob]) -> anyhow::Result<Vec<Decoded>> {
    let mut bufs = ChannelBufs::default();
    let planes = decode_jobs(tiff, jobs, None, &mut bufs, &mut ReadScratch::default())?;
    planes.own(&mut bufs);
    Ok(jobs.iter().map(|j| bufs.take(j.channel)).collect())
}

/// How to decode one channel of a requested frame (the app computes these from
/// the current metadata + per-channel settings and sends them to the worker).
#[derive(Clone)]
pub(crate) struct ChannelJob {
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
/// view, using the selected slice at the requested timepoint.
pub(crate) fn build_jobs(
    loaded: &Document,
    frame_index: usize,
    enabled: &[bool],
) -> Vec<ChannelJob> {
    let Some((width, height)) = loaded.dimensions() else {
        return Vec::new();
    };
    // The *resolved* interpretation, never `tiff.meta`. Resolving is precisely
    // the act of reclassifying a mislabeled axis — a file claiming
    // `channels = 100` that is really a 100-frame movie — so the raw metadata
    // gives the wrong stride and addresses planes past the end of the chain.
    let dims = loaded.display.dims();
    let rgb = loaded.display.layout().shown_as_rgb();
    let z = loaded.slice_at(frame_index);
    loaded
        .display
        .settings()
        .iter()
        .enumerate()
        .filter(|&(c, _)| enabled.get(c).copied().unwrap_or(false))
        .map(|(c, s)| {
            let (ifd_idx, plane) = crate::dimensions::plane_index(dims, rgb, c, z, frame_index);
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

/// One decoded channel of a completed prefetch.
pub(crate) struct DecodedChannel {
    pub channel: usize,
    pub width: u32,
    pub height: u32,
    pub kind: ChannelKind,
    pub data: Decoded,
}

/// A fully-decoded frame produced by the worker, tagged with what it was asked
/// for. The viewer takes one by the worker's key
/// ([`take_matching`](Prefetcher::take_matching)); the tags are what the tests
/// check the worker answered with.
pub(crate) struct PrefetchResult {
    /// The display revision the frame was asked for under: its channel layout
    /// and which channels were on.
    #[cfg_attr(not(all(test, feature = "threads")), allow(dead_code))]
    pub revision: Revision,
    #[cfg_attr(not(all(test, feature = "threads")), allow(dead_code))]
    pub frame_index: usize,
    pub channels: Vec<DecodedChannel>,
    /// The decoded channels' histograms, binned by the worker while it held
    /// them, when the request asked for them; empty when it did not. Binned
    /// exactly as the interface thread would have binned the same planes
    /// (see [`crate::histogram`]).
    pub histograms: Vec<Histogram>,
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
    /// How to bin the frame once decoded, when its histograms are wanted.
    binning: Option<Binning>,
}

/// How many frames' buffers [`Prefetcher::recycle`] keeps for the worker:
/// one being decoded into while one is on its way back is all a steady
/// stream needs, and any more would be memory held for frames that never
/// come.
const FREE_FRAMES: usize = 2;

/// Frames' channel buffers handed back after they were shown, for the worker
/// to decode the next ones into.
type FreeList = Arc<Mutex<Vec<ChannelBufs>>>;

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
pub(crate) struct Prefetcher {
    worker: LatestWorker<Key, Request, PrefetchResult>,
    /// The buffers of frames taken and shown, handed back for the worker's
    /// next decodes ([`recycle`](Self::recycle)).
    free: FreeList,
}

impl Prefetcher {
    /// Spawn a worker over `stack` — the window's own, shared: no second map,
    /// no second walk of the IFD chain, and the pages it touches are the ones
    /// the window then reads. Each request is read ahead in the mode its
    /// frame asks for (see the [module docs](self)). Returns `None` if the
    /// thread fails to spawn — callers then just decode inline.
    pub(crate) fn new(stack: Arc<TiffStack>) -> Option<Self> {
        let free = FreeList::default();
        let returned = Arc::clone(&free);
        // The worker's own, across every frame it reads ahead.
        let mut scratch = ReadScratch::default();
        let job = move |&(revision, frame_index): &Key, req: Request, cancel: &CancelToken| {
            fast_tiff_lib::set_thread_parallel_decode(Some(req.parallel));
            // Page-touch mode: absorb the next frame's soft page faults here
            // and publish nothing — the UI thread's inline read is then
            // fault-free (and stays zero-copy, which decoding ahead would have
            // forfeited). A frame's channels are planes of their own, so a
            // newer request stops it between them.
            if touch_suffices(&stack, &req.jobs) {
                for job in &req.jobs {
                    if cancel.is_cancelled() {
                        break;
                    }
                    if let Ok(frame) = stack.frame(job.ifd_idx) {
                        frame.touch();
                    }
                }
                return None;
            }
            // Into the buffers of a frame already shown, when one has come
            // back: the same allocations, frame after frame.
            let mut bufs = returned
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop()
                .unwrap_or_default();
            // The step is the frame: a decode overtaken while it runs finishes,
            // and the worker then drops it unpublished.
            let planes = decode_jobs(&stack, &req.jobs, None, &mut bufs, &mut scratch).ok()?;
            // A channel the file holds as it is (beside one that needed
            // decoding) is copied: the result outlives this frame's borrow.
            planes.own(&mut bufs);
            drop(planes);
            // Overtaken while it decoded: the worker would drop it unpublished,
            // so there is nothing to bin it for.
            if cancel.is_cancelled() {
                return None;
            }
            // Binned here, while the planes are in hand, rather than by the
            // interface thread after it takes them. Not free: about 4 ms a
            // channel of a million samples or more, which this frame is then
            // published later by, so it is asked for once a throttle interval
            // rather than every frame (see `histogram::PLAYBACK_INTERVAL`).
            let histograms = match &req.binning {
                Some(binning) => req
                    .jobs
                    .iter()
                    .filter_map(|job| binning.bin(job.channel, bufs.view(job.channel)))
                    .collect(),
                None => Vec::new(),
            };
            let channels = req
                .jobs
                .iter()
                .map(|job| DecodedChannel {
                    channel: job.channel,
                    width: job.width,
                    height: job.height,
                    kind: job.kind,
                    data: bufs.take(job.channel),
                })
                .collect();
            Some(PrefetchResult {
                revision,
                frame_index,
                channels,
                histograms,
            })
        };
        let worker = LatestWorker::spawn("fasttiff-prefetch", Unclaimed::Keep, job)?;
        Some(Prefetcher { worker, free })
    }

    /// Ask the worker to decode `frame_index`'s `jobs`, built under display
    /// `revision`, and with `binning` to bin them into
    /// [`PrefetchResult::histograms`]. Fire-and-forget. A different frame (or
    /// revision) replaces whatever was asked before and cancels it; the same
    /// one again is ignored while it is still pending, `binning` included.
    ///
    /// Page-touch mode decodes nothing ahead, so it bins nothing either: the
    /// interface thread bins what it reads inline.
    pub(crate) fn request(
        &self,
        revision: Revision,
        frame_index: usize,
        jobs: Vec<ChannelJob>,
        binning: Option<Binning>,
    ) {
        self.worker.request(
            (revision, frame_index),
            Request {
                jobs,
                parallel: fast_tiff_lib::parallel_decode_enabled(),
                binning,
            },
        );
    }

    /// Take the prefetched result iff it matches `(revision, frame_index)`;
    /// otherwise leave the slot untouched and return `None`. The caller still
    /// verifies the channel layout before using it.
    pub(crate) fn take_matching(
        &self,
        revision: Revision,
        frame_index: usize,
    ) -> Option<PrefetchResult> {
        self.worker.take_if(|key| *key == (revision, frame_index))
    }

    /// Hand back a result taken from [`take_matching`](Self::take_matching)
    /// once its planes have been uploaded and binned, or found not to match,
    /// so the worker decodes a later frame into the same buffers rather than
    /// into fresh ones. Kept for at most [`FREE_FRAMES`] frames; any more are
    /// let go.
    pub(crate) fn recycle(&self, result: PrefetchResult) {
        let mut bufs = ChannelBufs::default();
        for ch in result.channels {
            bufs.put(ch.channel, ch.data);
        }
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if free.len() < FREE_FRAMES {
            free.push(bufs);
        }
    }
}

#[cfg(test)]
#[path = "prefetch_tests.rs"]
mod tests;

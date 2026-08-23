//! Assembling the window of an over-large frame that is resident on the GPU.
//!
//! A frame past the device's texture limit is shown through a window (see
//! [`crate::roi`]). Building one means deciding which pieces of the file to
//! decompress, decompressing them, and cutting the sampled rows and columns out
//! into a texture-sized buffer. That is all this module does; uploading the
//! result is [`crate::sync`]'s business.
//!
//! Separated because the work is substantial — on a gigapixel mosaic it is
//! tenths of a second at best — and because it therefore wants to happen
//! somewhere other than the thread drawing the interface.

use crate::bandcache::{self, BandCache};
use crate::prefetch::{decode_jobs_region, decode_jobs_stepped, ChannelJob, Decoded};
use crate::roi::{self, Roi};
use fast_tiff_lib::{FrameInfo, TiffStack};
use scivis_render::ChannelKind;

/// Whether a window is the entire frame at full resolution, in which case no
/// view can leave it and nothing needs re-cutting.
///
/// Note the `stride == 1`: this is "the whole picture, every pixel", the test
/// for whether any windowing is needed at all. For "spans the whole frame at
/// whatever sampling", which is a different and weaker question, see
/// [`spans_whole_frame`].
pub fn covers_whole_frame(roi: &Roi, width: u32, height: u32) -> bool {
    roi.x == 0 && roi.y == 0 && roi.w >= width && roi.h >= height && roi.stride == 1
}

/// Whether a window spans the whole frame at *any* sampling.
///
/// Deliberately not [`covers_whole_frame`], which also demands `stride == 1`. A
/// fit-to-window view of a gigapixel mosaic spans the frame at a stride of 16
/// or more and fails that test, so using it to decide what is worth retaining
/// as an overview would retain nothing at all — and would do so silently, since
/// every other test would still pass.
pub fn spans_whole_frame(roi: &Roi, width: u32, height: u32) -> bool {
    roi.x == 0 && roi.y == 0 && roi.w >= width && roi.h >= height
}

/// How many strips to step over when sampling at `stride`, or `None` when
/// stepping does not apply and the band must be read contiguously.
///
/// Only when the stride is a whole number of strips *and* more than one: the
/// sampled rows then land at a fixed offset in every `step`-th strip, so the
/// decoded rows come out evenly spaced and the ordinary cut still works. A
/// stride that does not divide the strip height would sample a different offset
/// in each strip, which a uniform cut cannot express.
///
/// Never for a tiled frame: a tile row is several consecutive table entries, so
/// stepping would skip columns as well as rows. Tiled frames narrow by cropping,
/// which is better anyway.
fn step_for(frame: &FrameInfo, stride: u32) -> Option<u32> {
    if frame.is_tiled() {
        return None;
    }
    let per = frame.rows_per_strip.max(1);
    if stride <= per || !stride.is_multiple_of(per) {
        return None;
    }
    Some(stride / per)
}

/// Cut a window out of a decoded plane, whatever its format.
pub fn cut(data: &Decoded, width: u32, height: u32, roi: &Roi) -> Decoded {
    match data {
        Decoded::U8(v) => Decoded::U8(roi::extract(v, width, height, roi)),
        Decoded::U16(v) => Decoded::U16(roi::extract(v, width, height, roi)),
        Decoded::F32(v) => Decoded::F32(roi::extract(v, width, height, roi)),
    }
}

/// [`cut`], for a buffer whose rows are spaced differently from its columns.
/// See [`roi::Sampling`] for why that happens.
fn cut_sampled(data: &Decoded, width: u32, height: u32, s: &roi::Sampling) -> Decoded {
    match data {
        Decoded::U8(v) => Decoded::U8(roi::extract_sampled(v, width, height, s)),
        Decoded::U16(v) => Decoded::U16(roi::extract_sampled(v, width, height, s)),
        Decoded::F32(v) => Decoded::F32(roi::extract_sampled(v, width, height, s)),
    }
}

/// An all-zero plane of `n` samples in `kind`'s format.
fn empty_plane(kind: ChannelKind, n: usize) -> Decoded {
    match kind {
        ChannelKind::Int8 => Decoded::U8(vec![0; n]),
        ChannelKind::Int16 => Decoded::U16(vec![0; n]),
        ChannelKind::Float => Decoded::F32(vec![0.0; n]),
    }
}

/// Copy `src` into `dst` starting at sample `at`. Mismatched formats cannot
/// happen — both come from the same channel — so a mismatch is ignored rather
/// than given an error path nothing can reach.
fn blit(dst: &mut Decoded, src: &Decoded, at: usize) {
    match (dst, src) {
        (Decoded::U8(d), Decoded::U8(s)) => copy_into(d, s, at),
        (Decoded::U16(d), Decoded::U16(s)) => copy_into(d, s, at),
        (Decoded::F32(d), Decoded::F32(s)) => copy_into(d, s, at),
        _ => {}
    }
}

fn copy_into<T: Copy>(dst: &mut [T], src: &[T], at: usize) {
    let end = (at + src.len()).min(dst.len());
    if at < end {
        dst[at..end].copy_from_slice(&src[..end - at]);
    }
}

/// Trim decoded planes covering `actual` down to `want`, which it contains.
///
/// By value, so the usual case — the pieces happening to line up with the grid —
/// hands the planes straight back rather than copying a band to prove it did not
/// need copying.
fn trim_to(
    decoded: Vec<Decoded>,
    width: u32,
    actual: &std::ops::Range<u32>,
    want: &std::ops::Range<u32>,
) -> Vec<Decoded> {
    if actual == want {
        return decoded;
    }
    let keep = Roi { x: 0, y: want.start - actual.start, w: width, h: want.end - want.start, stride: 1 };
    let h = actual.end - actual.start;
    decoded.iter().map(|d| cut(d, width, h, &keep)).collect()
}

/// Cut texel rows `t0..t1` of the window out of one contiguously decoded band.
#[expect(clippy::too_many_arguments, reason = "all of it is one geometric mapping")]
fn blit_band(
    planes: &mut [Decoded],
    band: &[Decoded],
    roi: &Roi,
    band_w: u32,
    col0: u32,
    rows: &std::ops::Range<u32>,
    t0: u32,
    t1: u32,
    tw: u32,
    stride: u32,
) {
    let first = roi.y + t0 * stride;
    // The band holds columns from `col0`, so the window's x is relative to it.
    let sub = Roi {
        x: roi.x.saturating_sub(col0),
        y: first - rows.start,
        w: roi.w,
        h: (t1 - t0 - 1) * stride + 1,
        stride,
    };
    let band_h = rows.end - rows.start;
    for (plane, data) in planes.iter_mut().zip(band) {
        blit(plane, &cut(data, band_w, band_h, &sub), (t0 * tw) as usize);
    }
}

/// Build the texture-sized planes for `roi`, one per job, in `jobs` order.
///
/// Bands come off a grid fixed to the frame rather than to the window (see
/// [`crate::bandcache`]), which is what lets a moving view re-use what it has
/// already decompressed. Peak memory is a band, not the frame.
pub fn assemble(
    tiff: &TiffStack,
    bands: &mut BandCache,
    jobs: &[ChannelJob],
    kinds: &[ChannelKind],
    roi: &Roi,
    frame_index: usize,
) -> anyhow::Result<Vec<Decoded>> {
    let (frame_w, frame_h) = tiff.frames.first().map_or((0, 0), |f| (f.width, f.height));
    let (tw, th) = roi.texture_size();
    let stride = roi.stride.max(1);
    let channels: Vec<usize> = jobs.iter().map(|j| j.channel).collect();

    let mut planes: Vec<Decoded> = jobs
        .iter()
        .map(|j| {
            let kind = kinds.get(j.channel).copied().unwrap_or(ChannelKind::Int16);
            empty_plane(kind, tw as usize * th as usize)
        })
        .collect();

    let per_sample: usize = jobs
        .iter()
        .map(|j| match kinds.get(j.channel).copied().unwrap_or(ChannelKind::Int16) {
            ChannelKind::Int8 => 1usize,
            ChannelKind::Int16 => 2,
            ChannelKind::Float => 4,
        })
        .sum();
    let rows_per_band = bandcache::band_rows(frame_w, per_sample.max(1));
    let grid = bandcache::bands_covering(roi.y..roi.y + roi.h, rows_per_band);

    // Caching only pays when a later view can hit it. A window spanning more
    // bands than the cache holds — the zoomed-out overview, which covers every
    // row and never moves — would evict each band before it could be re-used.
    let span_bytes = (grid.end - grid.start) as usize
        * rows_per_band as usize
        * frame_w as usize
        * per_sample.max(1);
    let worth_caching = span_bytes <= bandcache::MAX_CACHE_BYTES;

    // The columns the window wants. On a tiled frame these narrow the decode to
    // the intersecting tile columns; on a stripped one a strip cannot be split,
    // so the crop hands back the full width and this has no effect.
    let want_cols = roi.x..roi.x + roi.w;
    let probe = tiff.frames.get(jobs.first().map_or(0, |j| j.ifd_idx));

    for band in grid {
        let rows = bandcache::band_range(band, rows_per_band, frame_h);
        if rows.is_empty() {
            continue;
        }
        // Texel rows of the window this band supplies. `roi.y` is aligned to the
        // stride and the grid is not, so round up to the first texel at or after
        // the band start.
        let t0 = rows.start.saturating_sub(roi.y).div_ceil(stride);
        let t1 = rows.end.saturating_sub(roi.y).div_ceil(stride).min(th);
        if t0 >= t1 {
            continue;
        }

        // What the file will actually hand over — asked without decoding,
        // because the answer is the cache key. Keying on the *request* would
        // miss every time the window moved by a pixel.
        let region = match probe {
            Some(f) => f.crop(want_cols.clone(), rows.clone())?,
            None => continue,
        };
        let cols = (region.cols.start, region.cols.end);
        let cut_w = region.frame.width;

        // At a coarse stride most strips in the band hold nothing that will be
        // drawn. When the stride is a whole number of strips, take only the ones
        // that are: the same picture for a fraction of the decompression.
        //
        // Nothing is cached on this path, and nothing needs to be. A stride that
        // coarse only arises when the view is zoomed far enough out that the
        // window covers the whole frame — at which point it stops moving, so
        // there is never a second read to serve.
        let first_sampled = roi.y + t0 * stride;
        if let Some(step) = step_for(&region.frame, stride) {
            let (band_planes, sampled) = decode_jobs_stepped(
                &tiff.data,
                &tiff.frames,
                tiff.byte_order,
                jobs,
                first_sampled..rows.end,
                step,
            )?;
            // The two axes are now sampled at different rates, and this is the
            // only place in the viewer where that is true. Skipping strips
            // removed rows from the buffer, so what survives sits
            // `rows_per_piece` apart rather than `stride` apart — but nothing
            // touched the columns, which are still every pixel and must be
            // sampled at `stride` on the way into the texture.
            //
            // Using one spacing for both is not an error anyone would see as
            // one: it builds a picture out of the right pixels taken from the
            // wrong columns, which looks like the image with horizontal
            // banding. Hence [`roi::Sampling`], which will not let the two be
            // written as the same number by accident.
            let per = sampled.rows_per_piece.max(1);
            let off = first_sampled.saturating_sub(sampled.first_row);
            let sub = roi::Sampling {
                x: roi.x.saturating_sub(region.cols.start),
                y: off,
                cols: tw,
                rows: t1 - t0,
                x_step: stride,
                y_step: per,
            };
            for (plane, data) in planes.iter_mut().zip(&band_planes) {
                blit(
                    plane,
                    &cut_sampled(data, cut_w, sampled.frame.height, &sub),
                    (t0 * tw) as usize,
                );
            }
            continue;
        }

        if bands.get(frame_index, &channels, band, cols).is_none() {
            let (decoded, got_cols, got_rows) = decode_jobs_region(
                &tiff.data,
                &tiff.frames,
                tiff.byte_order,
                jobs,
                want_cols.clone(),
                rows.clone(),
            )?;
            debug_assert_eq!((got_cols.start, got_cols.end), cols, "probe and decode disagree");
            // A band is keyed by its grid slot, so what is stored has to be the
            // slot's rows. The crop snaps outward to piece boundaries, so trim
            // back to the grid, or the next hit would be offset by up to a whole
            // piece.
            let trimmed = trim_to(decoded, cut_w, &got_rows, &rows);
            if worth_caching {
                bands.put(frame_index, channels.clone(), band, cols, trimmed);
            } else {
                blit_band(&mut planes, &trimmed, roi, cut_w, region.cols.start, &rows, t0, t1, tw, stride);
                continue;
            }
        }
        let Some(band_planes) = bands.get(frame_index, &channels, band, cols) else { continue };
        blit_band(&mut planes, band_planes, roi, cut_w, region.cols.start, &rows, t0, t1, tw, stride);
    }

    Ok(planes)
}

// ---------------------------------------------------------------------------
// Building windows off the interface thread
// ---------------------------------------------------------------------------

/// A finished window, ready to upload.
pub struct Built {
    pub frame_index: usize,
    pub roi: Roi,
    /// Which display channels it holds, in order — the same list decides
    /// whether it is still the right answer when it arrives.
    pub channels: Vec<usize>,
    pub planes: Vec<Decoded>,
}

/// The most RAM a retained overview may occupy.
///
/// An overview is a planned resident window, so [`roi::MAX_ROI_BYTES`] already
/// bounds it — at 512 MB, which is far too much to hold for the life of a file.
/// This is sized instead at ten times the ~6 MB a fit view of a gigapixel
/// mosaic costs on an ordinary panel, which leaves room for a HiDPI one and
/// refuses the combinations (4K, several float channels) where holding it would
/// cost more than the stall it saves.
pub const OVERVIEW_MAX_BYTES: usize = 64 << 20;

/// The retention budget for `mode` — see [`Overview::capture`].
pub fn overview_budget(mode: roi::LargeImageMode) -> usize {
    match mode {
        roi::LargeImageMode::Preload => roi::MAX_PRELOAD_BYTES,
        roi::LargeImageMode::Tiled => OVERVIEW_MAX_BYTES,
    }
}

/// A frame-spanning window kept in RAM, with everything needed to tell whether
/// it still describes the stack as it is now.
///
/// The point of retaining one is that a view leaving the resident window has
/// *correct* pixels to show at once — coarse, but the right picture — instead of
/// the interface stopping while a new window is cut. Zooming out is the case
/// that needs it: a fine window covers too little ground to be stretched over a
/// wider view, so without a fallback there is nothing to draw and the rebuild
/// has to be waited for.
///
/// Validated where it is *used*, never invalidated where its inputs are written
/// — the same contract as [`WindowWorker::take_matching`], and for the same
/// reason: which frame is showing, which channels are on, and what format each
/// one is in are written from a dozen places across two crates, and a scheme
/// that has to remember them all is a scheme that will eventually forget one.
/// Forgetting here means showing pixels from another frame or another channel
/// mapping, so it is not allowed to depend on remembering.
pub struct Overview {
    /// The window itself.
    pub built: Built,
    /// `Stack::prefetch_gen` when this was captured — the decode-plan
    /// generation, which moves whenever the channel-to-IFD mapping does.
    pub gen: u64,
    /// The per-channel enabled flags at capture.
    ///
    /// Kept separately from `gen` rather than folded into it, because the
    /// generation bump for a channel toggle happens *after* residency is
    /// planned: at planning time the two disagree, and the stale one is the
    /// generation.
    pub enabled: Vec<bool>,
    /// The texture format of each display channel at capture. A `Decoded::U16`
    /// plane uploaded to a channel that has since become `Int8` is not a wrong
    /// picture but a failed validation, which takes the process with it.
    pub kinds: Vec<ChannelKind>,
}

impl Overview {
    /// Retain `built` as the overview, or decline it.
    ///
    /// Declined when it does not span the frame — a window of part of the frame
    /// cannot serve a view of another part — or when its planes are past
    /// `budget`.
    ///
    /// The budget is the mode's, not a constant, and the difference is the
    /// difference between the two modes. Under
    /// [`LargeImageMode::Tiled`](roi::LargeImageMode::Tiled) an overview is a
    /// lucky by-product of a fit-to-window view, worth keeping only while it is
    /// small ([`OVERVIEW_MAX_BYTES`]). Under
    /// [`Preload`](roi::LargeImageMode::Preload) it *is* the mode, and refusing
    /// it would leave the planner asking every frame for a level nothing holds.
    pub fn capture(
        built: Built,
        width: u32,
        height: u32,
        gen: u64,
        enabled: &[bool],
        kinds: &[ChannelKind],
        budget: usize,
    ) -> Option<Overview> {
        if !spans_whole_frame(&built.roi, width, height) {
            return None;
        }
        let bytes: usize = built.planes.iter().map(Decoded::bytes).sum();
        if bytes > budget {
            return None;
        }
        Some(Overview { built, gen, enabled: enabled.to_vec(), kinds: kinds.to_vec() })
    }

    /// Bytes the retained planes occupy.
    pub fn bytes(&self) -> usize {
        self.built.planes.iter().map(Decoded::bytes).sum()
    }

    /// Whether this still describes the stack as it is now — the same frame,
    /// the same decode plan, the same channels on, in the same formats.
    pub fn matches(
        &self,
        frame_index: usize,
        gen: u64,
        enabled: &[bool],
        kinds: &[ChannelKind],
    ) -> bool {
        self.built.frame_index == frame_index
            && self.gen == gen
            && self.enabled == enabled
            && self.kinds == kinds
    }

    /// Whether these planes can be uploaded as they stand for `roi`.
    ///
    /// Stricter than [`matches`](Self::matches): the planes were cut to one
    /// window's texture size, so they are the right bytes only for that exact
    /// window and that exact channel list. A mismatch is a buffer that does not
    /// fit the texture it is going into.
    pub fn can_upload(
        &self,
        roi: &Roi,
        frame_index: usize,
        gen: u64,
        enabled: &[bool],
        kinds: &[ChannelKind],
        channels: &[usize],
    ) -> bool {
        self.matches(frame_index, gen, enabled, kinds)
            && self.built.roi == *roi
            && self.built.channels == channels
    }
}

#[cfg(feature = "threads")]
struct Job {
    frame_index: usize,
    roi: Roi,
    jobs: Vec<ChannelJob>,
    kinds: Vec<ChannelKind>,
}

/// Builds windows on a thread of its own, so the interface keeps drawing while
/// a new one is prepared.
///
/// Assembling a window is tenths of a second at best on a gigapixel mosaic, and
/// seconds when the view is coarse enough to span most of the file. Doing it
/// inline means the whole interface stops for that long every time the zoom
/// crosses a sampling level — which is exactly when the user is moving fastest.
/// Off-thread, the previous window keeps being drawn, scaled, until the new one
/// is ready: the picture goes momentarily soft instead of the program going
/// momentarily dead.
///
/// Only ever used when there *is* something to keep showing. The first view of a
/// file has nothing to fall back on, so that one is built inline and waited for.
#[cfg(feature = "threads")]
pub struct WindowWorker {
    tx: std::sync::mpsc::Sender<Job>,
    result: std::sync::Arc<std::sync::Mutex<Option<Built>>>,
    _handle: std::thread::JoinHandle<()>,
}

#[cfg(feature = "threads")]
impl WindowWorker {
    /// Spawn a worker with its own map of `path`. `None` if the thread or the
    /// open fails, in which case the caller builds windows inline as before.
    pub fn new(path: std::path::PathBuf) -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let slot = std::sync::Arc::clone(&result);
        let handle = std::thread::Builder::new()
            .name("fasttiff-window".to_owned())
            .spawn(move || {
                // Its own map of the same file: shares the OS page cache, so no
                // duplicate pixel memory, and its own band cache so the two
                // threads never contend over one.
                let Ok(tiff) = TiffStack::open(&path) else {
                    log::warn!("window worker: can't open {}", path.display());
                    return;
                };
                let mut bands = BandCache::default();
                while let Ok(job) = rx.recv() {
                    // Drain to the newest request: while one window was being
                    // built the view has probably moved again, and the older
                    // answers are already stale.
                    let mut job = job;
                    while let Ok(newer) = rx.try_recv() {
                        job = newer;
                    }
                    match assemble(&tiff, &mut bands, &job.jobs, &job.kinds, &job.roi, job.frame_index) {
                        Ok(planes) => {
                            let built = Built {
                                frame_index: job.frame_index,
                                roi: job.roi,
                                channels: job.jobs.iter().map(|j| j.channel).collect(),
                                planes,
                            };
                            if let Ok(mut slot) = slot.lock() {
                                *slot = Some(built);
                            }
                        }
                        Err(e) => log::warn!("window worker: {e:#}"),
                    }
                }
            })
            .ok()?;
        Some(Self { tx, result, _handle: handle })
    }

    /// Ask for a window. Fire and forget; superseded requests are skipped.
    pub fn request(&self, frame_index: usize, roi: Roi, jobs: Vec<ChannelJob>, kinds: Vec<ChannelKind>) {
        let _ = self.tx.send(Job { frame_index, roi, jobs, kinds });
    }

    /// Take a finished window if it is the one now wanted, leaving anything
    /// else in place — a result for a view already moved on from is not worth
    /// uploading, but nor is it worth discarding a result that has not been
    /// asked about yet.
    pub fn take_matching(&self, frame_index: usize, roi: &Roi, channels: &[usize]) -> Option<Built> {
        let mut slot = self.result.lock().ok()?;
        let matches = slot
            .as_ref()
            .is_some_and(|b| b.frame_index == frame_index && b.roi == *roi && b.channels == channels);
        if matches {
            slot.take()
        } else {
            None
        }
    }
}

/// Without threads there is no worker; every window is built inline.
#[cfg(not(feature = "threads"))]
pub struct WindowWorker(std::convert::Infallible);

#[cfg(not(feature = "threads"))]
impl WindowWorker {
    pub fn new(_path: std::path::PathBuf) -> Option<Self> {
        None
    }
    pub fn request(&self, _frame_index: usize, _roi: Roi, _jobs: Vec<ChannelJob>, _kinds: Vec<ChannelKind>) {
        match self.0 {}
    }
    pub fn take_matching(&self, _frame_index: usize, _roi: &Roi, _channels: &[usize]) -> Option<Built> {
        match self.0 {}
    }
}

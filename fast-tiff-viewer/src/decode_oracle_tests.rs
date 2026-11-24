//! The viewer's decode on fast-tiff-lib's [`Frame`], checked against the
//! decode it replaced, sample for sample.
//!
//! [`old`] is that decode as it was at 3492148 — `decode_channel`,
//! `decode_jobs_borrowed` and its owned, row, region and stepped siblings in
//! `prefetch.rs`, and the auto-contrast scans in `channels.rs` — kept verbatim
//! as scaffolding while the library's old readers, which it calls, still
//! exist. Every file here (every fixture, fuzz seed and fuzz regression, and a
//! synthetic matrix of depths, sample counts, interleavings, strip layouts and
//! codecs) is read through every job list a view can send — each texture
//! format, one plane or a batch of an RGB frame's planes in any order, several
//! IFDs at once, a plane past the last — whole and cropped to rows, to a
//! region and to sampled strips, through [`decode_jobs`] into fresh buffers and
//! again into buffers left dirty, of every type and the wrong length, by
//! earlier reads. The answers must be the old ones: equal samples in the same
//! texture format, the same coverage, and an error where the old one failed.
//! And where the old decode borrowed the file's bytes, so must the new one.
//!
//! Two differences are by design, and are held to an answer derived from the
//! old one rather than to the old one:
//!
//! - An 8-bit channel of a frame of several samples not shown as RGB (a
//!   grey-and-alpha frame, say) was the whole frame's bytes, every sample of
//!   every pixel, uploaded into a texture one sample per pixel — the first
//!   third of the picture, three times over, for a three-sample frame. Every
//!   other depth showed the first sample, and so does 8-bit now: the old
//!   answer's first sample of each pixel.
//! - A plane past the last is an error, as it already was for the planes of
//!   an RGB frame read together; a channel read on its own was clamped to the
//!   last plane, and so showed another channel's pixels as its own.
//!
//! When the library's old readers go (item 25e), [`old`] goes with them and
//! [`DIGESTS`] stays: a digest of what the old decode gave, case by case, for
//! every file here, which the new decode is then held to instead. The case
//! list must not change after that — a digest is only as good as the code it
//! was taken from.

use crate::prefetch::{decode_jobs, ChannelBufs, ChannelJob, Decoded, ReadScratch, Samples};
use fast_tiff_lib::{
    Compression, Coverage, Crop, SampleType, TiffStack, TiffWriter, WriterOptions,
};
use scivis_render::ChannelKind;
use std::path::{Path, PathBuf};

/// The code as it was at 3492148, unchanged but for being gathered here.
mod old {
    use crate::prefetch::{ChannelJob, Decoded};
    use fast_tiff_lib::{ByteOrder, DisplayMap, FrameInfo, TiffStack};
    use scivis_render::ChannelKind;
    use std::borrow::Cow;

    pub(super) enum DecodedRef<'a> {
        U8(Cow<'a, [u8]>),
        U16(Cow<'a, [u16]>),
        F32(Cow<'a, [f32]>),
    }

    impl DecodedRef<'_> {
        pub(super) fn into_owned(self) -> Decoded {
            match self {
                DecodedRef::U8(v) => Decoded::U8(v.into_owned()),
                DecodedRef::U16(v) => Decoded::U16(v.into_owned()),
                DecodedRef::F32(v) => Decoded::F32(v.into_owned()),
            }
        }
    }

    fn decode_channel<'a>(
        mmap: &'a [u8],
        frame: &FrameInfo,
        order: ByteOrder,
        kind: ChannelKind,
        plane: usize,
        rgb: bool,
    ) -> anyhow::Result<DecodedRef<'a>> {
        let cmyk = frame.is_cmyk();
        Ok(match kind {
            ChannelKind::Int8 => DecodedRef::U8(if cmyk {
                Cow::Owned(fast_tiff_lib::read_plane_rgb_u8(mmap, frame, order, plane)?)
            } else if rgb {
                Cow::Owned(fast_tiff_lib::read_plane_u8(mmap, frame, order, plane)?)
            } else {
                fast_tiff_lib::read_frame_u8(mmap, frame, order)?
            }),
            ChannelKind::Float => {
                DecodedRef::F32(fast_tiff_lib::read_frame_f32(mmap, frame, order)?)
            }
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

    pub(super) fn decode_jobs(
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

    pub(super) fn decode_jobs_borrowed<'a>(
        mmap: &'a [u8],
        frames: &[FrameInfo],
        order: ByteOrder,
        jobs: &[ChannelJob],
    ) -> anyhow::Result<Vec<DecodedRef<'a>>> {
        if jobs.len() > 1
            && jobs
                .iter()
                .all(|j| j.rgb && j.ifd_idx == jobs[0].ifd_idx && j.kind == jobs[0].kind)
            && matches!(jobs[0].kind, ChannelKind::Int8 | ChannelKind::Int16)
        {
            let frame = frames
                .get(jobs[0].ifd_idx)
                .ok_or_else(|| anyhow::anyhow!("frame {} out of range", jobs[0].ifd_idx))?;
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

    fn take_plane<T>(planes: &mut [Vec<T>], plane: usize) -> anyhow::Result<Vec<T>> {
        if plane >= planes.len() {
            anyhow::bail!(
                "sample plane {plane} out of range ({} planes)",
                planes.len()
            );
        }
        Ok(std::mem::take(&mut planes[plane]))
    }

    pub(super) fn decode_jobs_rows<'a>(
        mmap: &'a [u8],
        frames: &[FrameInfo],
        order: ByteOrder,
        jobs: &[ChannelJob],
        rows: std::ops::Range<u32>,
    ) -> anyhow::Result<(Vec<DecodedRef<'a>>, std::ops::Range<u32>)> {
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
                    match &covered {
                        Some(r) if *r != band.rows => anyhow::bail!(
                            "frame {} cropped to {:?}, expected {:?} — frames disagree on strip \
                             layout",
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
        Ok((
            decode_jobs_borrowed(mmap, &band_frames, order, &band_jobs)?,
            covered,
        ))
    }

    pub(super) fn decode_jobs_region(
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

    pub(super) fn decode_jobs_stepped(
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

    // channels.rs

    const MINMAX_SAMPLE_BUDGET: usize = 1 << 18;

    pub(super) fn channel_minmax(
        tiff: &TiffStack,
        channel: usize,
        step: usize,
    ) -> Option<((f32, f32), DisplayMap)> {
        let one = |idx: usize| -> Option<((f32, f32), DisplayMap)> {
            let frame = tiff.frames.get(idx)?;
            let mut owned = Vec::new();
            let (pixels, map) = match DisplayMap::of(frame, None) {
                Some(map) => (
                    fast_tiff_lib::read_frame_u16(&tiff.data, frame, tiff.byte_order, None).ok()?,
                    map,
                ),
                None => {
                    let map = fast_tiff_lib::read_frame_u16_into(
                        &tiff.data,
                        frame,
                        tiff.byte_order,
                        None,
                        &mut owned,
                    )
                    .ok()?;
                    (Cow::Borrowed(&owned[..]), map)
                }
            };
            let (mut lo, mut hi) = (u16::MAX, 0u16);
            let stride = (pixels.len() / MINMAX_SAMPLE_BUDGET).max(1);
            for &p in pixels.iter().step_by(stride) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
            Some(((lo as f32, hi as f32), map))
        };
        // The window is put into file units from these by the caller's code,
        // which is unchanged; the scan's answer is what is compared.
        scan_for_contrast(tiff, channel, step, one)
    }

    pub(super) fn channel_float_minmax(
        tiff: &TiffStack,
        channel: usize,
        step: usize,
    ) -> Option<(f32, f32)> {
        let measure = |idx: usize| {
            let frame = tiff.frames.get(idx)?;
            let range =
                fast_tiff_lib::frame_float_minmax(&tiff.data, frame, tiff.byte_order).ok()??;
            Some((range, ()))
        };
        scan_for_contrast(tiff, channel, step, measure).map(|(range, ())| range)
    }

    const PROBES: [(usize, usize); 8] = [
        (0, 1),
        (1, 2),
        (1, 4),
        (3, 4),
        (1, 8),
        (3, 8),
        (5, 8),
        (7, 8),
    ];

    fn scan_for_contrast<T: Copy>(
        tiff: &TiffStack,
        channel: usize,
        step: usize,
        measure: impl Fn(usize) -> Option<((f32, f32), T)>,
    ) -> Option<((f32, f32), T)> {
        let step = step.max(1);
        let total = tiff.frames.len();
        if total == 0 {
            return None;
        }
        let start = channel.min(total - 1);
        let planes = (total - start).div_ceil(step).max(1);

        let mut first = None;
        let mut tried = [usize::MAX; PROBES.len()];
        for (slot, &(num, den)) in PROBES.iter().enumerate() {
            let n = (planes.saturating_mul(num) / den).min(planes - 1);
            if tried[..slot].contains(&n) {
                continue;
            }
            tried[slot] = n;
            let Some(((lo, hi), with)) = measure(start + n * step) else {
                continue;
            };
            if hi > lo {
                return Some(((lo, hi), with));
            }
            first = first.or(Some(((lo, hi), with)));
        }
        first.map(|((lo, hi), with)| ((lo, if hi > lo { hi } else { lo + 1.0 }), with))
    }
}

// ------------------------------------------------------------------ files

fn lib_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fast-tiff-lib")
        .join(sub)
}

/// Every file in `dir` that opens, by name.
fn files_in(dir: &Path) -> Vec<(String, TiffStack)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tif" || x == "tiff"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| {
            let name = format!(
                "{}/{}",
                dir.file_name()?.to_string_lossy(),
                p.file_name()?.to_string_lossy()
            );
            let stack = TiffStack::from_bytes(std::fs::read(&p).ok()?).ok()?;
            Some((name, stack))
        })
        .collect()
}

/// A two-frame stack written by this crate's writer.
fn written(
    st: SampleType,
    spp: u16,
    planar: bool,
    compression: Compression,
    predictor: bool,
    rows_per_strip: u32,
) -> TiffStack {
    let (w, h) = (13u32, 10u32);
    let opts = WriterOptions::new(w, h, st)
        .samples_per_pixel(spp)
        .planar(planar)
        .compression(compression)
        .predictor(predictor)
        .rows_per_strip(rows_per_strip);
    let mut wr = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let n = (w * h) as usize * spp as usize * st.bytes();
    for f in 0..2u8 {
        // Every byte varies, so a sample read from the wrong place, the
        // wrong plane or in the wrong order shows.
        let bytes: Vec<u8> = (0..n)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(f * 101) ^ (i >> 8) as u8)
            .map(|b| if st == SampleType::F32 { b & 0x3f } else { b })
            .collect();
        wr.write_frame_bytes(&bytes).unwrap();
    }
    TiffStack::from_bytes(wr.finish().unwrap().into_inner()).unwrap()
}

/// The synthetic matrix: depths and signs, one sample or several (chunky and
/// planar), one strip or several, uncompressed, LZW, and LZW with the
/// predictor.
fn synthetic() -> Vec<(String, TiffStack)> {
    let mut out = Vec::new();
    let types = [
        SampleType::U8,
        SampleType::U16,
        SampleType::I16,
        SampleType::U32,
        SampleType::F32,
    ];
    let layouts: [(u16, bool); 5] = [(1, false), (2, false), (3, false), (3, true), (4, true)];
    let codecs = [
        (Compression::None, false),
        (Compression::Lzw, false),
        (Compression::Lzw, true),
    ];
    for st in types {
        for (spp, planar) in layouts {
            for (compression, predictor) in codecs {
                for rps in [10, 3] {
                    let name = format!(
                        "synthetic/{st:?} spp{spp}{} {compression:?}{} rps{rps}",
                        if planar { " planar" } else { "" },
                        if predictor { "+pred" } else { "" }
                    );
                    out.push((name, written(st, spp, planar, compression, predictor, rps)));
                }
            }
        }
    }
    out
}

fn every_file() -> Vec<(String, TiffStack)> {
    let mut all = files_in(&lib_dir("tests/fixtures"));
    all.extend(files_in(&lib_dir("tests/fuzz-regressions")));
    all.extend(files_in(&lib_dir("fuzz/seeds")));
    all.extend(synthetic());
    all
}

// ------------------------------------------------------------------ cases

fn job(
    tiff: &TiffStack,
    channel: usize,
    ifd: usize,
    plane: usize,
    kind: ChannelKind,
    rgb: bool,
) -> ChannelJob {
    let (width, height) = tiff.frames.first().map_or((0, 0), |f| (f.width, f.height));
    ChannelJob {
        channel,
        ifd_idx: ifd,
        plane,
        kind,
        rgb,
        width,
        height,
    }
}

/// Every job list a view could send for `tiff`, and a few it could not
/// (a plane past the last, an IFD past the end), in a fixed order.
fn job_lists(tiff: &TiffStack) -> Vec<Vec<ChannelJob>> {
    use ChannelKind::{Float, Int16, Int8};
    let mut lists = Vec::new();
    // The first two IFDs: a stack's frames share a layout, and the job lists
    // over several IFDs below reach the second.
    let ifds = tiff.frames.len().min(2);
    for ifd in 0..ifds {
        let spp = (tiff.frames[ifd].samples_per_pixel as usize).max(1);
        for kind in [Int8, Int16, Float] {
            for rgb in [false, true] {
                for plane in 0..=spp.min(4) {
                    lists.push(vec![job(tiff, plane, ifd, plane, kind, rgb)]);
                }
            }
        }
        for kind in [Int8, Int16] {
            let all: Vec<usize> = (0..spp.min(6)).collect();
            let mut sets: Vec<Vec<usize>> = vec![all.clone()];
            if spp >= 3 {
                sets.push(vec![0, 2]);
                sets.push(vec![2, 1]);
                sets.push(vec![1, 2]);
            }
            if spp >= 2 {
                sets.push(vec![1, 0]);
                sets.push(vec![spp - 1, 0]);
            }
            sets.push(vec![0, spp]);
            for set in sets {
                lists.push(
                    set.iter()
                        .map(|&p| job(tiff, p, ifd, p, kind, true))
                        .collect(),
                );
                // The same planes on other display channels.
                lists.push(
                    set.iter()
                        .map(|&p| job(tiff, 7 - p.min(7), ifd, p, kind, true))
                        .collect(),
                );
            }
        }
    }
    // Several IFDs, one channel each, as a multi-channel stack sends them.
    if tiff.frames.len() >= 2 {
        for kind in [Int8, Int16, Float] {
            lists.push(vec![
                job(tiff, 0, 0, 0, kind, false),
                job(tiff, 1, 1, 0, kind, false),
            ]);
        }
        lists.push(vec![
            job(tiff, 0, 1, 0, Int16, false),
            job(tiff, 1, 0, 0, Float, false),
            job(tiff, 2, 0, 0, Int8, false),
        ]);
    }
    lists.push(vec![job(tiff, 0, tiff.frames.len(), 0, Int16, false)]);
    lists
}

/// The crops a view asks for: none, a band of rows, a region, and strips
/// sampled at two steps.
fn crops(tiff: &TiffStack) -> Vec<Option<Crop>> {
    let (w, h) = tiff.frames.first().map_or((1, 1), |f| (f.width, f.height));
    vec![
        None,
        Some(Crop::Rows(h / 3..(h / 3 + 2).min(h))),
        Some(Crop::Region {
            cols: w / 4..(w / 2 + 1).min(w),
            rows: 1.min(h)..h.saturating_sub(1).max(1),
        }),
        Some(Crop::Sampled {
            rows: 0..h,
            step: 2,
        }),
        Some(Crop::Sampled {
            rows: 1.min(h)..h,
            step: 3,
        }),
    ]
}

// ------------------------------------------------------------- comparison

/// One channel's samples, as bytes, with their texture format.
#[derive(Clone, PartialEq, Debug)]
enum Got {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<u32>),
}

impl Got {
    fn of(s: Samples<'_>) -> Self {
        match s {
            Samples::U8(v) => Got::U8(v.to_vec()),
            Samples::U16(v) => Got::U16(v.to_vec()),
            Samples::F32(v) => Got::F32(v.iter().map(|x| x.to_bits()).collect()),
        }
    }

    fn of_decoded(d: &Decoded) -> Self {
        Got::of(d.view())
    }

    fn of_ref(d: &old::DecodedRef<'_>) -> (Self, bool) {
        use std::borrow::Cow::Borrowed;
        match d {
            old::DecodedRef::U8(v) => (Got::U8(v.to_vec()), matches!(v, Borrowed(_))),
            old::DecodedRef::U16(v) => (Got::U16(v.to_vec()), matches!(v, Borrowed(_))),
            old::DecodedRef::F32(v) => (
                Got::F32(v.iter().map(|x| x.to_bits()).collect()),
                matches!(v, Borrowed(_)),
            ),
        }
    }

    fn bytes(&self) -> (u8, &[u8]) {
        match self {
            Got::U8(v) => (1, v),
            Got::U16(v) => (2, bytemuck::cast_slice(v)),
            Got::F32(v) => (4, bytemuck::cast_slice(v)),
        }
    }
}

/// What a decode came to: an error, or each job's samples, whether each was
/// borrowed from the file, and the coverage numbers a caller indexes against.
#[derive(Debug)]
enum Outcome {
    Failed,
    Read {
        planes: Vec<Got>,
        borrowed: Vec<bool>,
        cover: Vec<u32>,
    },
}

impl Outcome {
    /// Everything but the borrows, which the new decode does more of.
    fn same(&self, other: &Outcome) -> bool {
        match (self, other) {
            (Outcome::Failed, Outcome::Failed) => true,
            (
                Outcome::Read { planes, cover, .. },
                Outcome::Read {
                    planes: p2,
                    cover: c2,
                    ..
                },
            ) => planes == p2 && cover == c2,
            _ => false,
        }
    }

    fn digest(&self, h: &mut Fnv) {
        match self {
            Outcome::Failed => h.byte(0),
            Outcome::Read { planes, cover, .. } => {
                h.byte(1);
                for &c in cover {
                    h.bytes(&c.to_le_bytes());
                }
                for p in planes {
                    let (tag, bytes) = p.bytes();
                    h.byte(tag);
                    h.bytes(&(bytes.len() as u64).to_le_bytes());
                    h.bytes(bytes);
                }
            }
        }
    }
}

/// 64-bit FNV-1a.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    fn bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.byte(b);
        }
    }
}

fn rows_cover(r: &std::ops::Range<u32>) -> Vec<u32> {
    vec![r.start, r.end]
}

/// The old decode of `jobs`, cropped as `crop` says.
fn old_decode(tiff: &TiffStack, jobs: &[ChannelJob], crop: &Option<Crop>) -> Outcome {
    let (d, f, o) = (&tiff.data[..], &tiff.frames[..], tiff.byte_order);
    let read = |refs: Vec<old::DecodedRef<'_>>, cover: Vec<u32>| {
        let (planes, borrowed) = refs.iter().map(Got::of_ref).unzip();
        Outcome::Read {
            planes,
            borrowed,
            cover,
        }
    };
    let owned = |decoded: Vec<Decoded>, cover: Vec<u32>| Outcome::Read {
        borrowed: vec![false; decoded.len()],
        planes: decoded.iter().map(Got::of_decoded).collect(),
        cover,
    };
    let result = match crop {
        None => old::decode_jobs_borrowed(d, f, o, jobs).map(|r| read(r, Vec::new())),
        Some(Crop::Rows(rows)) => old::decode_jobs_rows(d, f, o, jobs, rows.clone())
            .map(|(r, rows)| read(r, rows_cover(&rows))),
        Some(Crop::Region { cols, rows }) => {
            old::decode_jobs_region(d, f, o, jobs, cols.clone(), rows.clone())
                .map(|(r, cols, rows)| owned(r, vec![cols.start, cols.end, rows.start, rows.end]))
        }
        Some(Crop::Sampled { rows, step }) => {
            old::decode_jobs_stepped(d, f, o, jobs, rows.clone(), *step).map(|(r, band)| {
                let cover = vec![
                    band.first_row,
                    band.rows_per_piece,
                    band.pieces,
                    band.step,
                    band.frame.height,
                ];
                owned(r, cover)
            })
        }
    };
    result.unwrap_or(Outcome::Failed)
}

/// The new decode of `jobs` into `bufs`, with `scratch`, cropped as `crop`
/// says, in the old decode's terms.
fn new_decode(
    tiff: &TiffStack,
    jobs: &[ChannelJob],
    crop: &Option<Crop>,
    bufs: &mut ChannelBufs,
    scratch: &mut ReadScratch,
) -> Outcome {
    let Ok(planes) = decode_jobs(tiff, jobs, crop.as_ref(), bufs, scratch) else {
        return Outcome::Failed;
    };
    let cover = match (crop, &planes.coverage) {
        (None, None) => Vec::new(),
        (Some(Crop::Rows(_)), Some(c)) => rows_cover(&c.rows),
        (Some(Crop::Region { .. }), Some(c)) => {
            vec![c.cols.start, c.cols.end, c.rows.start, c.rows.end]
        }
        (
            Some(Crop::Sampled { .. }),
            Some(Coverage {
                first_row,
                rows_per_piece,
                pieces,
                step,
                ..
            }),
        ) => vec![*first_row, *rows_per_piece, *pieces, *step, planes.height],
        (crop, cover) => panic!("a {crop:?} read came back covering {cover:?}"),
    };
    Outcome::Read {
        planes: planes.views(bufs).map(Got::of).collect(),
        borrowed: (0..jobs.len()).map(|i| planes.is_file(i)).collect(),
        cover,
    }
}

/// The differences by design (see the module docs): a plane past the last is
/// an error; and an 8-bit channel of a frame of several samples, not shown as
/// RGB, was the whole frame's samples and is now its first, held to the old
/// answer's first sample of each pixel.
fn as_designed(tiff: &TiffStack, jobs: &[ChannelJob], old: Outcome) -> Outcome {
    let past_the_last = jobs.iter().any(|job| {
        let Some(info) = tiff.frames.get(job.ifd_idx) else {
            return false;
        };
        // The planes the job's units have, and the one it reads.
        let (planes, plane) = match job.kind {
            ChannelKind::Float => (1, 0),
            _ if info.is_cmyk() => (3, job.plane),
            _ if job.rgb => ((info.samples_per_pixel as usize).max(1), job.plane),
            _ => (1, 0),
        };
        plane >= planes
    });
    if past_the_last {
        return Outcome::Failed;
    }
    let Outcome::Read {
        mut planes,
        mut borrowed,
        cover,
    } = old
    else {
        return old;
    };
    for ((job, plane), borrowed) in jobs.iter().zip(&mut planes).zip(&mut borrowed) {
        let Some(info) = tiff.frames.get(job.ifd_idx) else {
            continue;
        };
        let spp = (info.samples_per_pixel as usize).max(1);
        if job.kind != ChannelKind::Int8 || job.rgb || info.is_cmyk() || spp == 1 {
            continue;
        }
        let Got::U8(all) = plane else {
            continue;
        };
        // The first sample of each pixel in the old answer: the first of
        // every `spp` in a chunky frame (each sample `bytes` wide, a 4-bit
        // index one byte), the first plane's worth of a planar one.
        let bytes = (info.bits_per_sample as usize).div_ceil(8).max(1);
        let first: Vec<u8> = if info.is_planar() {
            all[..all.len() / spp].to_vec()
        } else {
            all.chunks(bytes * spp)
                .flat_map(|px| px[..bytes].to_vec())
                .collect()
        };
        *plane = Got::U8(first);
        // The whole frame could be borrowed; one sample of each pixel of a
        // chunky frame cannot.
        *borrowed = false;
    }
    Outcome::Read {
        planes,
        borrowed,
        cover,
    }
}

/// Fill every channel's buffer with junk of some type and length: what an
/// earlier read of another file, another layout or another crop leaves.
fn dirty(bufs: &mut ChannelBufs, seed: usize) {
    for c in 0..8 {
        let n = 17 + (seed * 7 + c * 13) % 300;
        let junk = match (seed + c) % 3 {
            0 => Decoded::U8(vec![0xA5; n]),
            1 => Decoded::U16(vec![0xA5A5; n]),
            _ => Decoded::F32(vec![f32::from_bits(0x7fc0_dead); n]),
        };
        bufs.put(c, junk);
    }
}

/// Per file, what the old decode gave for every case: see the module docs.
/// Printed by `FASTTIFF_PRINT_DIGESTS=1 cargo test -p fast-tiff-viewer
/// decode_oracle -- --nocapture`.
const DIGESTS: &[(&str, u64)] = &[
    ("fixtures/ij_u16_spp1_p6_hyperstack.tif", 0x68707d3deae8a86b),
    ("fixtures/pil_u16_spp1_p1_lzw.tif", 0x88bb5ae56148f9f1),
    ("fixtures/pil_u8_spp1_p1_lzw.tif", 0x66cce5e5f2efb0f3),
    ("fixtures/pil_u8_spp1_p1_pb.tif", 0x66cce5e5f2efb0f3),
    ("fixtures/pil_u8_spp3_p1_zip.tif", 0x5713995c2549ebf0),
    ("fixtures/pil_u8_spp4_p1_cmyk-lzw.tif", 0xe2e797c7e9029951),
    (
        "fixtures/tff_f32_spp1_p1_tiled-zip-pred3.tif",
        0xd1127bf582974e41,
    ),
    ("fixtures/tff_f32_spp1_p2_lzw-pred3.tif", 0x8f94881d8ffabb12),
    ("fixtures/tff_f32_spp1_p2_none-le.tif", 0x8f94881d8ffabb12),
    (
        "fixtures/tff_f32_spp1_p2_zip-pred3-be.tif",
        0xd2845ae284f099b8,
    ),
    ("fixtures/tff_f32_spp1_p2_zip-pred3.tif", 0x8f94881d8ffabb12),
    ("fixtures/tff_i16_spp1_p2_lzw-pred2.tif", 0x4ca328e94d7471d0),
    ("fixtures/tff_i16_spp1_p2_none-be.tif", 0xeece60d1c021e24c),
    ("fixtures/tff_i16_spp1_p2_none-le.tif", 0x4ca328e94d7471d0),
    ("fixtures/tff_i32_spp1_p2_none-le.tif", 0xfa4aea51c25a00ef),
    ("fixtures/tff_i8_spp1_p2_none-le.tif", 0x52ac9c800e42347d),
    ("fixtures/tff_u16_spp1_p1_tiled.tif", 0x7d7d54e7057a19e0),
    (
        "fixtures/tff_u16_spp1_p2_lzw-pred2-be.tif",
        0xc50e4717294705bb,
    ),
    (
        "fixtures/tff_u16_spp1_p2_lzw-pred2-rps4.tif",
        0xb6c12e11210d6ffd,
    ),
    ("fixtures/tff_u16_spp1_p2_lzw-rps4.tif", 0xb6c12e11210d6ffd),
    ("fixtures/tff_u16_spp1_p2_none-be.tif", 0xc50e4717294705bb),
    ("fixtures/tff_u16_spp1_p2_none-le.tif", 0x68707d3deae8a86b),
    ("fixtures/tff_u16_spp1_p2_pb-rps4.tif", 0xb6c12e11210d6ffd),
    ("fixtures/tff_u16_spp1_p2_zip-pred2.tif", 0x68707d3deae8a86b),
    ("fixtures/tff_u16_spp1_p2_zip-rps4.tif", 0xb6c12e11210d6ffd),
    ("fixtures/tff_u16_spp3_p2_zip-pred2.tif", 0xec6078e1b2cad8f8),
    (
        "fixtures/tff_u16_spp4_p1_cmyk-zip-pred2.tif",
        0xedd20e8e6f3a6217,
    ),
    ("fixtures/tff_u16_spp4_p1_cmyk.tif", 0xedd20e8e6f3a6217),
    ("fixtures/tff_u32_spp1_p2_none-le.tif", 0x7554a8c7cdcc608d),
    ("fixtures/tff_u32_spp1_p2_zip-pred2.tif", 0x7554a8c7cdcc608d),
    ("fixtures/tff_u8_spp1_p2_none-le.tif", 0xcb9f6d0f8d1ad2aa),
    ("fixtures/tff_u8_spp1_p2_zip-pred2.tif", 0xcb9f6d0f8d1ad2aa),
    (
        "fixtures/tff_u8_spp3_p1_tiled-lzw-pred2.tif",
        0x96aed77af220fc29,
    ),
    ("fixtures/tff_u8_spp3_p2_lzw-pred2.tif", 0x5f77ea311bc00d50),
    ("fixtures/tff_u8_spp3_p2_none.tif", 0x5f77ea311bc00d50),
    ("fixtures/tff_u8_spp4_p1_cmyk-lzw.tif", 0xe2e797c7e9029951),
    ("fixtures/tff_u8_spp4_p1_cmyk.tif", 0xe2e797c7e9029951),
    (
        "fixtures/tff_u8_spp4_p2_cmyk-planar.tif",
        0x6550a691cefeecad,
    ),
    ("fixtures/tld_u16_spp1_p1_grid-lzw.tif", 0x78a9e2688535a025),
    ("fixtures/tld_u16_spp1_p1_grid.tif", 0x78a9e2688535a025),
    (
        "fixtures/tld_u8_spp3_p1_grid-planar.tif",
        0x195f3f4db6974836,
    ),
    ("fixtures/tld_u8_spp3_p1_grid-pred2.tif", 0x39de3cb850c6f306),
    (
        "fuzz-regressions/overflow-crop-rows-height-u32max.tif",
        0xff0377e1732b483f,
    ),
    (
        "fuzz-regressions/overflow-strip-band-h4g-rps2g.tif",
        0xff0377e1732b483f,
    ),
    (
        "fuzz-regressions/overflow-tiled-crop-h4g-tile2g.tif",
        0xff0377e1732b483f,
    ),
    (
        "fuzz-regressions/overflow-w2g-h2g-spp4.tif",
        0xaeaa33db06881a27,
    ),
    ("fuzz-regressions/slow-tall-tile.tif", 0xb0524d62fe0f1e45),
    ("fuzz-regressions/slow-tile-grid.tif", 0x9d983d98a33e25bf),
    ("seeds/bigtiff.tif", 0x74122662855bc4f1),
    ("seeds/cmyk16_lzw.tif", 0x3486f37e0a990479),
    ("seeds/cmyk_chunky.tif", 0x8202adf5209d13d5),
    ("seeds/cmyk_planar.tif", 0x678f4eecb5f4d771),
    ("seeds/float32.tif", 0x298d90e061040315),
    ("seeds/float32_zlib_pred.tif", 0x298d90e061040315),
    ("seeds/gray16.tif", 0x74122662855bc4f1),
    ("seeds/gray16_be.tif", 0x538f2431b6fd9e21),
    ("seeds/gray16_lzw.tif", 0x74122662855bc4f1),
    ("seeds/gray16_packbits.tif", 0x74122662855bc4f1),
    ("seeds/gray16_strips.tif", 0xaff5f69df32cb4df),
    ("seeds/gray16_zlib.tif", 0x74122662855bc4f1),
    ("seeds/gray16_zlib_pred.tif", 0x74122662855bc4f1),
    ("seeds/gray8.tif", 0xdde0bde661dd2f7c),
    ("seeds/imagej_hyperstack.tif", 0xb605965cdee8596c),
    ("seeds/imagej_lut.tif", 0x74122662855bc4f1),
    ("seeds/ome.tif", 0xb605965cdee8596c),
    ("seeds/palette4.tif", 0x3305f08ac2f3e3fc),
    ("seeds/palette8.tif", 0xdde0bde661dd2f7c),
    ("seeds/regression-oom-huge-width.tif", 0xff0377e1732b483f),
    ("seeds/rgb_chunky.tif", 0x814b063952bb4664),
    ("seeds/rgb_planar.tif", 0x814b063952bb4664),
    ("seeds/stack3.tif", 0x2b53aa428c0ab230),
    ("seeds/tiled16.tif", 0xe2c429d8896375d0),
    ("seeds/tiled16_lzw.tif", 0xe2c429d8896375d0),
    ("seeds/tiled_planar.tif", 0xa0637fef2c711b5c),
    ("seeds/tiled_rgb_pred.tif", 0x4398e62bbb0a66dc),
    ("synthetic/U8 spp1 None rps10", 0xa51220b1a6a2a502),
    ("synthetic/U8 spp1 None rps3", 0x547e100af3b77c11),
    ("synthetic/U8 spp1 Lzw rps10", 0xa51220b1a6a2a502),
    ("synthetic/U8 spp1 Lzw rps3", 0x547e100af3b77c11),
    ("synthetic/U8 spp1 Lzw+pred rps10", 0xa51220b1a6a2a502),
    ("synthetic/U8 spp1 Lzw+pred rps3", 0x547e100af3b77c11),
    ("synthetic/U8 spp2 None rps10", 0xab201740d060917d),
    ("synthetic/U8 spp2 None rps3", 0xe496e8a311526e0d),
    ("synthetic/U8 spp2 Lzw rps10", 0xab201740d060917d),
    ("synthetic/U8 spp2 Lzw rps3", 0xe496e8a311526e0d),
    ("synthetic/U8 spp2 Lzw+pred rps10", 0xab201740d060917d),
    ("synthetic/U8 spp2 Lzw+pred rps3", 0xe496e8a311526e0d),
    ("synthetic/U8 spp3 None rps10", 0x2290f4d306d24ceb),
    ("synthetic/U8 spp3 None rps3", 0x41a5ffb161c884fd),
    ("synthetic/U8 spp3 Lzw rps10", 0x2290f4d306d24ceb),
    ("synthetic/U8 spp3 Lzw rps3", 0x41a5ffb161c884fd),
    ("synthetic/U8 spp3 Lzw+pred rps10", 0x2290f4d306d24ceb),
    ("synthetic/U8 spp3 Lzw+pred rps3", 0x41a5ffb161c884fd),
    ("synthetic/U8 spp3 planar None rps10", 0xb0b21ceb14fb69ae),
    ("synthetic/U8 spp3 planar None rps3", 0x5c20a211ff5aca21),
    ("synthetic/U8 spp3 planar Lzw rps10", 0xb0b21ceb14fb69ae),
    ("synthetic/U8 spp3 planar Lzw rps3", 0x5c20a211ff5aca21),
    (
        "synthetic/U8 spp3 planar Lzw+pred rps10",
        0xb0b21ceb14fb69ae,
    ),
    ("synthetic/U8 spp3 planar Lzw+pred rps3", 0x5c20a211ff5aca21),
    ("synthetic/U8 spp4 planar None rps10", 0x4b0c266b37019ec6),
    ("synthetic/U8 spp4 planar None rps3", 0x69be335d43eb79f5),
    ("synthetic/U8 spp4 planar Lzw rps10", 0x4b0c266b37019ec6),
    ("synthetic/U8 spp4 planar Lzw rps3", 0x69be335d43eb79f5),
    (
        "synthetic/U8 spp4 planar Lzw+pred rps10",
        0x4b0c266b37019ec6,
    ),
    ("synthetic/U8 spp4 planar Lzw+pred rps3", 0x69be335d43eb79f5),
    ("synthetic/U16 spp1 None rps10", 0xbe7aae89d7c205cf),
    ("synthetic/U16 spp1 None rps3", 0x4c253e347f144ef7),
    ("synthetic/U16 spp1 Lzw rps10", 0xbe7aae89d7c205cf),
    ("synthetic/U16 spp1 Lzw rps3", 0x4c253e347f144ef7),
    ("synthetic/U16 spp1 Lzw+pred rps10", 0xbe7aae89d7c205cf),
    ("synthetic/U16 spp1 Lzw+pred rps3", 0x4c253e347f144ef7),
    ("synthetic/U16 spp2 None rps10", 0xc94cbd071aca45d8),
    ("synthetic/U16 spp2 None rps3", 0x85ba09ed6782bbb7),
    ("synthetic/U16 spp2 Lzw rps10", 0xc94cbd071aca45d8),
    ("synthetic/U16 spp2 Lzw rps3", 0x85ba09ed6782bbb7),
    ("synthetic/U16 spp2 Lzw+pred rps10", 0xc94cbd071aca45d8),
    ("synthetic/U16 spp2 Lzw+pred rps3", 0x85ba09ed6782bbb7),
    ("synthetic/U16 spp3 None rps10", 0x36ec752a8d19266f),
    ("synthetic/U16 spp3 None rps3", 0x2a6bf2199f0e446e),
    ("synthetic/U16 spp3 Lzw rps10", 0x36ec752a8d19266f),
    ("synthetic/U16 spp3 Lzw rps3", 0x2a6bf2199f0e446e),
    ("synthetic/U16 spp3 Lzw+pred rps10", 0x36ec752a8d19266f),
    ("synthetic/U16 spp3 Lzw+pred rps3", 0x2a6bf2199f0e446e),
    ("synthetic/U16 spp3 planar None rps10", 0x9e980f903d4a00fb),
    ("synthetic/U16 spp3 planar None rps3", 0x3f88c9ac60537d8f),
    ("synthetic/U16 spp3 planar Lzw rps10", 0x9e980f903d4a00fb),
    ("synthetic/U16 spp3 planar Lzw rps3", 0x3f88c9ac60537d8f),
    (
        "synthetic/U16 spp3 planar Lzw+pred rps10",
        0x9e980f903d4a00fb,
    ),
    (
        "synthetic/U16 spp3 planar Lzw+pred rps3",
        0x3f88c9ac60537d8f,
    ),
    ("synthetic/U16 spp4 planar None rps10", 0x9207f42e6a1dda13),
    ("synthetic/U16 spp4 planar None rps3", 0x683e9c72a511ff97),
    ("synthetic/U16 spp4 planar Lzw rps10", 0x9207f42e6a1dda13),
    ("synthetic/U16 spp4 planar Lzw rps3", 0x683e9c72a511ff97),
    (
        "synthetic/U16 spp4 planar Lzw+pred rps10",
        0x9207f42e6a1dda13,
    ),
    (
        "synthetic/U16 spp4 planar Lzw+pred rps3",
        0x683e9c72a511ff97,
    ),
    ("synthetic/I16 spp1 None rps10", 0x71f13345a28a461e),
    ("synthetic/I16 spp1 None rps3", 0x3f7e9c3d5f7be799),
    ("synthetic/I16 spp1 Lzw rps10", 0x71f13345a28a461e),
    ("synthetic/I16 spp1 Lzw rps3", 0x3f7e9c3d5f7be799),
    ("synthetic/I16 spp1 Lzw+pred rps10", 0x71f13345a28a461e),
    ("synthetic/I16 spp1 Lzw+pred rps3", 0x3f7e9c3d5f7be799),
    ("synthetic/I16 spp2 None rps10", 0xcec24225aa32f5cc),
    ("synthetic/I16 spp2 None rps3", 0xbc84564928ba73c9),
    ("synthetic/I16 spp2 Lzw rps10", 0xcec24225aa32f5cc),
    ("synthetic/I16 spp2 Lzw rps3", 0xbc84564928ba73c9),
    ("synthetic/I16 spp2 Lzw+pred rps10", 0xcec24225aa32f5cc),
    ("synthetic/I16 spp2 Lzw+pred rps3", 0xbc84564928ba73c9),
    ("synthetic/I16 spp3 None rps10", 0xd6cf70c6afac94d2),
    ("synthetic/I16 spp3 None rps3", 0x70dc61173050f656),
    ("synthetic/I16 spp3 Lzw rps10", 0xd6cf70c6afac94d2),
    ("synthetic/I16 spp3 Lzw rps3", 0x70dc61173050f656),
    ("synthetic/I16 spp3 Lzw+pred rps10", 0xd6cf70c6afac94d2),
    ("synthetic/I16 spp3 Lzw+pred rps3", 0x70dc61173050f656),
    ("synthetic/I16 spp3 planar None rps10", 0xa356c461480a8072),
    ("synthetic/I16 spp3 planar None rps3", 0xd2b3f8639e227761),
    ("synthetic/I16 spp3 planar Lzw rps10", 0xa356c461480a8072),
    ("synthetic/I16 spp3 planar Lzw rps3", 0xd2b3f8639e227761),
    (
        "synthetic/I16 spp3 planar Lzw+pred rps10",
        0xa356c461480a8072,
    ),
    (
        "synthetic/I16 spp3 planar Lzw+pred rps3",
        0xd2b3f8639e227761,
    ),
    ("synthetic/I16 spp4 planar None rps10", 0xcc4555641df36b96),
    ("synthetic/I16 spp4 planar None rps3", 0x0814d238a5fd5be1),
    ("synthetic/I16 spp4 planar Lzw rps10", 0xcc4555641df36b96),
    ("synthetic/I16 spp4 planar Lzw rps3", 0x0814d238a5fd5be1),
    (
        "synthetic/I16 spp4 planar Lzw+pred rps10",
        0xcc4555641df36b96,
    ),
    (
        "synthetic/I16 spp4 planar Lzw+pred rps3",
        0x0814d238a5fd5be1,
    ),
    ("synthetic/U32 spp1 None rps10", 0xacaa523cae91bc5d),
    ("synthetic/U32 spp1 None rps3", 0xf5e15709828e4bce),
    ("synthetic/U32 spp1 Lzw rps10", 0xacaa523cae91bc5d),
    ("synthetic/U32 spp1 Lzw rps3", 0xf5e15709828e4bce),
    ("synthetic/U32 spp1 Lzw+pred rps10", 0xacaa523cae91bc5d),
    ("synthetic/U32 spp1 Lzw+pred rps3", 0xf5e15709828e4bce),
    ("synthetic/U32 spp2 None rps10", 0x2d60a020a3171bbc),
    ("synthetic/U32 spp2 None rps3", 0x0b78fa62e9963b91),
    ("synthetic/U32 spp2 Lzw rps10", 0x2d60a020a3171bbc),
    ("synthetic/U32 spp2 Lzw rps3", 0x0b78fa62e9963b91),
    ("synthetic/U32 spp2 Lzw+pred rps10", 0x2d60a020a3171bbc),
    ("synthetic/U32 spp2 Lzw+pred rps3", 0x0b78fa62e9963b91),
    ("synthetic/U32 spp3 None rps10", 0xd3b0b71861325d02),
    ("synthetic/U32 spp3 None rps3", 0x17895d318e8fc079),
    ("synthetic/U32 spp3 Lzw rps10", 0xd3b0b71861325d02),
    ("synthetic/U32 spp3 Lzw rps3", 0x17895d318e8fc079),
    ("synthetic/U32 spp3 Lzw+pred rps10", 0xd3b0b71861325d02),
    ("synthetic/U32 spp3 Lzw+pred rps3", 0x17895d318e8fc079),
    ("synthetic/U32 spp3 planar None rps10", 0x2d8ba0f6880d4551),
    ("synthetic/U32 spp3 planar None rps3", 0xc43ba7278e77a0b2),
    ("synthetic/U32 spp3 planar Lzw rps10", 0x2d8ba0f6880d4551),
    ("synthetic/U32 spp3 planar Lzw rps3", 0xc43ba7278e77a0b2),
    (
        "synthetic/U32 spp3 planar Lzw+pred rps10",
        0x2d8ba0f6880d4551,
    ),
    (
        "synthetic/U32 spp3 planar Lzw+pred rps3",
        0xc43ba7278e77a0b2,
    ),
    ("synthetic/U32 spp4 planar None rps10", 0x06ffb78fe8377d14),
    ("synthetic/U32 spp4 planar None rps3", 0x47a8220e2abac582),
    ("synthetic/U32 spp4 planar Lzw rps10", 0x06ffb78fe8377d14),
    ("synthetic/U32 spp4 planar Lzw rps3", 0x47a8220e2abac582),
    (
        "synthetic/U32 spp4 planar Lzw+pred rps10",
        0x06ffb78fe8377d14,
    ),
    (
        "synthetic/U32 spp4 planar Lzw+pred rps3",
        0x47a8220e2abac582,
    ),
    ("synthetic/F32 spp1 None rps10", 0xcd23fe2223b9fd9d),
    ("synthetic/F32 spp1 None rps3", 0x095d2e268c2e26bf),
    ("synthetic/F32 spp1 Lzw rps10", 0xcd23fe2223b9fd9d),
    ("synthetic/F32 spp1 Lzw rps3", 0x095d2e268c2e26bf),
    ("synthetic/F32 spp1 Lzw+pred rps10", 0xcd23fe2223b9fd9d),
    ("synthetic/F32 spp1 Lzw+pred rps3", 0x095d2e268c2e26bf),
    ("synthetic/F32 spp2 None rps10", 0xdd275221b7f53f85),
    ("synthetic/F32 spp2 None rps3", 0xfbc70cd7a7703ce3),
    ("synthetic/F32 spp2 Lzw rps10", 0xdd275221b7f53f85),
    ("synthetic/F32 spp2 Lzw rps3", 0xfbc70cd7a7703ce3),
    ("synthetic/F32 spp2 Lzw+pred rps10", 0xdd275221b7f53f85),
    ("synthetic/F32 spp2 Lzw+pred rps3", 0xfbc70cd7a7703ce3),
    ("synthetic/F32 spp3 None rps10", 0x2f5233b0f65d2a35),
    ("synthetic/F32 spp3 None rps3", 0x4cc737ec0f607af2),
    ("synthetic/F32 spp3 Lzw rps10", 0x2f5233b0f65d2a35),
    ("synthetic/F32 spp3 Lzw rps3", 0x4cc737ec0f607af2),
    ("synthetic/F32 spp3 Lzw+pred rps10", 0x2f5233b0f65d2a35),
    ("synthetic/F32 spp3 Lzw+pred rps3", 0x4cc737ec0f607af2),
    ("synthetic/F32 spp3 planar None rps10", 0xc274929e7c133c69),
    ("synthetic/F32 spp3 planar None rps3", 0x29e36c0d935c0c32),
    ("synthetic/F32 spp3 planar Lzw rps10", 0xc274929e7c133c69),
    ("synthetic/F32 spp3 planar Lzw rps3", 0x29e36c0d935c0c32),
    (
        "synthetic/F32 spp3 planar Lzw+pred rps10",
        0xc274929e7c133c69,
    ),
    (
        "synthetic/F32 spp3 planar Lzw+pred rps3",
        0x29e36c0d935c0c32,
    ),
    ("synthetic/F32 spp4 planar None rps10", 0xa9be0a48f79a3716),
    ("synthetic/F32 spp4 planar None rps3", 0x4bbfe86be5cb7c7d),
    ("synthetic/F32 spp4 planar Lzw rps10", 0xa9be0a48f79a3716),
    ("synthetic/F32 spp4 planar Lzw rps3", 0x4bbfe86be5cb7c7d),
    (
        "synthetic/F32 spp4 planar Lzw+pred rps10",
        0xa9be0a48f79a3716,
    ),
    (
        "synthetic/F32 spp4 planar Lzw+pred rps3",
        0x4bbfe86be5cb7c7d,
    ),
];

/// Every case of one file: the old decode, the new one into fresh buffers,
/// and the new one into `warm`, a set left dirty by the cases before. Returns
/// the digest of the old answers and how many cases there were.
fn check_file(name: &str, tiff: &TiffStack, warm: &mut (ChannelBufs, ReadScratch)) -> (u64, usize) {
    let mut h = Fnv::new();
    let mut cases = 0;
    for (l, jobs) in job_lists(tiff).iter().enumerate() {
        for crop in crops(tiff) {
            let label = || {
                let jobs: Vec<_> = jobs
                    .iter()
                    .map(|j| (j.channel, j.ifd_idx, j.plane, j.kind, j.rgb))
                    .collect();
                format!("{name}: list {l} {jobs:?}, crop {crop:?}")
            };
            let old = as_designed(tiff, jobs, old_decode(tiff, jobs, &crop));
            let fresh = new_decode(
                tiff,
                jobs,
                &crop,
                &mut ChannelBufs::default(),
                &mut ReadScratch::default(),
            );
            assert!(
                old.same(&fresh),
                "{}\n old {old:?}\n new {fresh:?}",
                label()
            );
            if let (Outcome::Read { borrowed: o, .. }, Outcome::Read { borrowed: n, .. }) =
                (&old, &fresh)
            {
                for (i, (&o, &n)) in o.iter().zip(n).enumerate() {
                    assert!(
                        !o || n,
                        "{}: job {i} was borrowed and is now copied",
                        label()
                    );
                }
            }
            dirty(&mut warm.0, l * 5 + cases);
            let again = new_decode(tiff, jobs, &crop, &mut warm.0, &mut warm.1);
            assert!(
                old.same(&again),
                "{}, into dirty buffers\n old {old:?}\n new {again:?}",
                label()
            );
            old.digest(&mut h);
            cases += 1;
        }
    }
    (h.0, cases)
}

#[test]
fn the_viewer_decode_equals_the_one_it_replaced() {
    let files = every_file();
    assert!(
        files.len() > 100,
        "only {} files: the fixtures are missing",
        files.len()
    );
    // Split across threads: a debug build's LZW makes the run seconds long
    // on one. Each keeps one warm, dirty set for all its files, as a view
    // keeps one for every frame.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let chunk = files.len().div_ceil(threads);
    let results: Vec<(String, u64, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut warm = (ChannelBufs::default(), ReadScratch::default());
                    part.iter()
                        .map(|(name, tiff)| {
                            let (digest, cases) = check_file(name, tiff, &mut warm);
                            (name.clone(), digest, cases)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    if std::env::var_os("FASTTIFF_PRINT_DIGESTS").is_some() {
        let rows: Vec<String> = results
            .iter()
            .map(|(name, digest, _)| format!("    ({name:?}, {digest:#018x}),"))
            .collect();
        println!(
            "const DIGESTS: &[(&str, u64)] = &[\n{}\n];",
            rows.join("\n")
        );
    }
    for (name, digest, _) in &results {
        let want = DIGESTS.iter().find(|(n, _)| n == name).map(|&(_, d)| d);
        assert_eq!(want, Some(*digest), "{name}: not what the old decode gave");
    }
    // And every stored answer was checked, not only those of the files that
    // opened: once the old decode is gone these are the oracle, and a file
    // that stopped opening (which `files_in` passes over) must not quietly
    // take its answer out of the comparison.
    let unchecked: Vec<&str> = DIGESTS
        .iter()
        .map(|&(name, _)| name)
        .filter(|name| !results.iter().any(|(n, _, _)| n == name))
        .collect();
    assert!(
        unchecked.is_empty(),
        "stored answers no file was checked against (missing, or no longer opening): \
         {unchecked:?}"
    );
    let cases: usize = results.iter().map(|r| r.2).sum();
    assert!(cases > 80_000, "only {cases} cases");
}

/// The auto-contrast scans an open runs: the same plane chosen, the same
/// range and the same map, for every channel stride.
#[test]
fn the_contrast_scans_equal_the_ones_they_replaced() {
    for (name, tiff) in every_file() {
        let n = tiff.frames.len();
        for step in 1..=3usize {
            for channel in 0..step.min(n) {
                let old = old::channel_minmax(&tiff, channel, step);
                let new = crate::channels::channel_minmax(&tiff, channel, step);
                let old_window = old.map(|((lo, hi), map)| {
                    let file = |v: f32| map.to_file(v as f64) as f32;
                    (crate::units::Window::new(file(lo), file(hi)), map)
                });
                assert_eq!(
                    format!("{old_window:?}"),
                    format!("{new:?}"),
                    "{name}: channel {channel} of {step}"
                );
                let old = old::channel_float_minmax(&tiff, channel, step);
                let new = crate::channels::channel_float_minmax(&tiff, channel, step);
                assert_eq!(
                    format!("{:?}", old.map(crate::units::Window::from)),
                    format!("{new:?}"),
                    "{name}: float channel {channel} of {step}"
                );
            }
        }
    }
}

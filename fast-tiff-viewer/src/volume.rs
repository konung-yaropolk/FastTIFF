//! 3D volume assembly for the volume view: decoding a stack (or, for a 4D
//! stack, one timepoint's Z range) into a volume packed the way the GPU holds
//! it ([`scivis_render::pack`]) — channels sharing textures, a mip chain, and
//! per-brick maxima for skipping empty space — sized to fit the GPU's
//! 3D-texture limit, a memory budget and the screen; plus a background builder
//! thread so neither the initial build nor a 4D timepoint rebuild blocks the UI.
//!
//! The build is parallel **across output slices** (rayon): each task decodes
//! the source planes behind one output slice (RGB planes share a single
//! decompression), averages them down, and writes its slab of every packed
//! texture, which are disjoint `chunks_mut` of the final buffers — no locking,
//! no reassembly pass.
//!
//! # How a volume is made to fit
//!
//! A stack too large for the budget is reduced by a whole number of voxels per
//! axis, the **average** of each box rather than one sample of it: point
//! sampling keeps one voxel in nine at a stride of three and drops the rest,
//! which shimmers as the view turns and loses whatever fell between samples.
//! The axis reduced first is the one whose voxels are *finest* in physical
//! units, so a stack of thin optical sections — fine in x and y, coarse in z —
//! gives up in-plane detail it has to spare before depth it has little of.
//!
//! The builder is a `LatestWorker` ([`crate::worker`]) over the window's own
//! stack, shared. A request for another `(generation, time)` replaces the one
//! waiting and cancels the build under way at its next plane, so scrubbing a 4D
//! stack skips timepoints instead of building each one, and the wanted
//! timepoint waits for a plane of the stale build rather than all of it. The
//! finished volume is posted into a result slot the UI thread polls, and used
//! only on an exact `(generation, time)` match, so a stale build can cost work
//! but never show wrong data.

use crate::prefetch::{decode_jobs, ChannelJob, Decoded};
use crate::units::SampleUnits;
use crate::worker::{CancelToken, Cancelled, LatestWorker, Unclaimed};
use fast_tiff_lib::TiffStack;
#[cfg(feature = "threads")]
use rayon::prelude::*;
use scivis_render::pack::{self, PackedVolume, TexelFormat, VolumeLayout};
use scivis_render::{ChannelKind, MAX_CHANNELS};
use std::sync::Arc;

/// The GPU memory a volume may take when the frontend could not find out how
/// much there is. Every level of every packed texture counts against it.
pub const DEFAULT_VOLUME_BUDGET: usize = 768 << 20;

/// The least a budget is cut to by retries after the GPU ran out of memory:
/// below this there is no volume worth showing.
pub const MIN_VOLUME_BUDGET: usize = 32 << 20;

/// The most a budget may be, whatever the GPU has: the volume is built in RAM
/// before it is uploaded, and a frontend with a large card should not find its
/// host out of memory instead. 4 GB on a 64-bit host; 1 GB where addresses are
/// 32 bits (a browser's wasm32), whose whole address space is 4 GB — and where
/// `4 << 30` would not even fit in a `usize`.
pub const MAX_VOLUME_BUDGET: usize = if usize::BITS >= 64 {
    (4u64 << 30) as usize
} else {
    1 << 30
};

/// A built volume, packed for upload.
pub type BuiltVolume = PackedVolume;

/// The most voxels a volume keeps along any axis for a screen whose longest
/// side is `screen_px` pixels (0: unknown, no cap): twice the screen, so a
/// view zoomed in twofold is still sharp, and never under 2048.
pub fn detail_cap(screen_px: u32) -> u32 {
    if screen_px == 0 {
        u32::MAX
    } else {
        screen_px.saturating_mul(2).max(2048)
    }
}

/// Everything the builder needs besides the file itself. Dimensions come from
/// the app's (possibly manually overridden) metadata, not the worker's own
/// parse, so a channels/frames swap in the UI is honored by background builds.
#[derive(Clone)]
pub struct VolumePlan {
    /// Each channel's units, which say both how it decodes (its kind) and how
    /// its samples are normalized into the volume
    /// ([`SampleUnits::volume_affine`]).
    pub units: Vec<SampleUnits>,
    pub rgb: bool,
    pub channels: usize,
    pub slices: usize,
    pub frames: usize,
    /// Timepoint to build in the 4D (`slices > 1`) case; ignored otherwise.
    pub time: usize,
    /// The GPU's per-axis 3D-texture size limit.
    pub max_dim: u32,
    /// How the renderer stores the volume's samples.
    pub format: TexelFormat,
    /// Bytes the volume may take on the GPU, every level of every texture.
    pub budget: usize,
    /// The physical size of a voxel along each axis (any unit, only the ratios
    /// matter), which decides the axis reduced first.
    pub scale: [f32; 3],
    /// The most voxels any axis keeps, however much budget is left: about
    /// what the screen can show, with room to zoom.
    pub detail_cap: u32,
}

/// How many source voxels per output voxel along each axis, for a volume of
/// `dims` voxels with `channels` channels stored as `format`.
///
/// Starts from the smallest strides the per-axis texture limit and the detail
/// cap allow, then grows one stride at a time until the packed volume, its mip
/// chain included, fits `budget`: always the stride of the axis whose output
/// voxels are physically finest (`stride × scale`), the larger output winning a
/// tie, and never one already down to a single voxel.
pub fn choose_strides(
    dims: (u32, u32, u32),
    channels: usize,
    format: TexelFormat,
    budget: usize,
    scale: [f32; 3],
    max_dim: u32,
    detail_cap: u32,
) -> (u32, u32, u32) {
    let limit = max_dim.max(64).min(detail_cap.max(64));
    let dims = [dims.0.max(1), dims.1.max(1), dims.2.max(1)];
    let mut stride = dims.map(|d| d.div_ceil(limit).max(1));
    let per_voxel = pack::bytes_per_voxel(channels, format);
    let scale = scale.map(|s| if s.is_finite() && s > 0.0 { s } else { 1.0 });
    loop {
        let out = [0, 1, 2].map(|i| dims[i].div_ceil(stride[i]));
        let bytes = out.iter().map(|&o| o as f64).product::<f64>() * per_voxel;
        if bytes <= budget as f64 {
            break;
        }
        let pick = (0..3).filter(|&i| out[i] > 1).min_by(|&a, &b| {
            let (va, vb) = (stride[a] as f32 * scale[a], stride[b] as f32 * scale[b]);
            va.total_cmp(&vb).then(out[b].cmp(&out[a]))
        });
        match pick {
            Some(i) => stride[i] += 1,
            // One voxel on every axis: nothing left to give.
            None => break,
        }
    }
    (stride[0], stride[1], stride[2])
}

/// Build the 3D volume for `plan`, packed for upload. The depth axis is the
/// whole frame axis in the ordinary case, or (4D, `slices > 1`) the Z axis at
/// timepoint `plan.time`. All channels share the same reduction, chosen by
/// [`choose_strides`]. `Ok(None)` when there's no volume to build (or a decode
/// failed).
///
/// `cancel` is asked before each source plane is read — once per output slice,
/// and per plane of the several a reduced slice averages — and before the mip
/// chain is built. Once it says stop, every task still running finishes the
/// plane it is on, the half-built volume is dropped, and the result is
/// [`Cancelled`]. Inline callers pass [`CancelToken::never`].
pub fn build_volume(
    tiff: &TiffStack,
    plan: &VolumePlan,
    cancel: &CancelToken,
) -> Result<Option<BuiltVolume>, Cancelled> {
    let Some(f0) = tiff.frames.first() else {
        return Ok(None);
    };
    let (w, h) = (f0.width, f0.height);
    let slices = plan.slices.max(1);
    let channels = plan.channels.max(1);
    let is_4d = slices > 1;
    // Depth axis: Z at the current timepoint (4D) or the whole frame axis.
    let depth = if is_4d {
        slices as u32
    } else {
        plan.frames.max(1) as u32
    };
    let time = if is_4d { plan.time } else { 0 };
    let n = plan.units.len().min(MAX_CHANNELS);
    if w == 0 || h == 0 || depth < 2 || n == 0 {
        return Ok(None);
    }
    let ckinds: Vec<ChannelKind> = plan.units[..n].iter().map(|u| u.kind()).collect();
    let affines: Vec<(f32, f32)> = plan.units[..n].iter().map(|u| u.volume_affine()).collect();

    let (sx, sy, sz) = choose_strides(
        (w, h, depth),
        n,
        plan.format,
        plan.budget,
        plan.scale,
        plan.max_dim,
        plan.detail_cap,
    );
    let out = |n: u32, s: u32| n.div_ceil(s);
    let (ow, oh, od) = (out(w, sx), out(h, sy), out(depth, sz));
    let layout = VolumeLayout::new((ow, oh, od), n, plan.format);
    let mut packed = PackedVolume::new(layout.clone());

    // Parallel across output slices: hand each task its slab of every packed
    // texture — disjoint `chunks_mut` of level 0.
    let mut slab_iters: Vec<_> = packed
        .level0_mut()
        .into_iter()
        .enumerate()
        .map(|(t, level)| level.chunks_mut(layout.slice_bytes(t)))
        .collect();
    let mut tasks: Vec<(u32, Vec<&mut [u8]>)> = Vec::with_capacity(od as usize);
    for oz in 0..od {
        tasks.push((
            oz,
            slab_iters
                .iter_mut()
                .map(|it| it.next().expect("slab per slice"))
                .collect(),
        ));
    }
    drop(slab_iters);

    #[cfg(feature = "threads")]
    let task_iter = tasks.into_par_iter();
    // Without rayon the same closure runs sequentially — identical output, just
    // one core. (`try_for_each` exists on both iterator kinds with the same
    // signature, so only the iterator construction differs.)
    #[cfg(not(feature = "threads"))]
    let mut task_iter = tasks.into_iter();
    let result = task_iter.try_for_each(|(oz, mut slabs)| -> anyhow::Result<()> {
        // Every source slice this output slice averages over.
        let z_from = oz * sz;
        let z_to = (z_from + sz).min(depth);
        // Nothing to average at full size: samples go straight into the slab,
        // without a slice of sums per channel per thread.
        let direct = (sx, sy, sz) == (1, 1, 1);
        let mut boxes = SliceBoxes::new(if direct { 0 } else { n }, (w, h), (sx, sy), (ow, oh));
        for k in z_from..z_to {
            // The step a superseded build still costs: one plane's decode per
            // task running. `try_for_each` stops handing out slices at the
            // first `Err`.
            cancel.check()?;
            // Depth index → (time, z): in 4D the depth is Z at the fixed
            // timepoint; otherwise the depth index *is* the frame (time = k,
            // z = 0). The IFD layout is time-major, then Z, then channel.
            let (t, z) = if is_4d {
                (time, k as usize)
            } else {
                (k as usize, 0)
            };
            let jobs: Vec<ChannelJob> = (0..n)
                .map(|c| {
                    let (ifd_idx, plane) = crate::dimensions::plane_index(
                        crate::display::Dims {
                            channels,
                            slices,
                            frames: plan.frames.max(1),
                        },
                        plan.rgb,
                        c,
                        z,
                        t,
                    );
                    ChannelJob {
                        channel: c,
                        ifd_idx,
                        plane,
                        kind: ckinds[c],
                        rgb: plan.rgb,
                        width: w,
                        height: h,
                    }
                })
                .collect();
            let decoded = decode_jobs(&tiff.data, &tiff.frames, tiff.byte_order, &jobs)?;
            for (c, dec) in decoded.iter().enumerate() {
                if direct {
                    encode_slice(&layout, &mut slabs, c, dec, affines[c]);
                } else {
                    boxes.add(c, dec);
                }
            }
        }
        if !direct {
            boxes.write(&layout, &mut slabs, z_to - z_from, &affines);
        }
        Ok(())
    });
    if let Err(e) = result {
        if e.is::<Cancelled>() {
            return Err(Cancelled);
        }
        log::warn!("volume build failed: {e:#}");
        return Ok(None);
    }
    // The mip chain and brick maxima are a pass over the whole volume.
    cancel.check()?;
    packed.finish();
    Ok(Some(packed))
}

/// Store one decoded slice of channel `c`, at full size, into its slab of the
/// packed texture that holds it. A short decode leaves the rest zero.
///
/// Samples go through the channel's `(scale, offset)`
/// ([`SampleUnits::volume_affine`]), the map
/// [`SampleUnits::to_unorm`](crate::units::SampleUnits::to_unorm) puts the
/// channel's window on the same scale with.
fn encode_slice(
    layout: &VolumeLayout,
    slabs: &mut [&mut [u8]],
    c: usize,
    dec: &Decoded,
    (scale, offset): (f32, f32),
) {
    let (t, k) = layout.place(c);
    let texel = layout.texel_bytes(t);
    let fb = layout.format.bytes();
    let slab = &mut slabs[t];
    let voxels = slab.len() / texel;
    let mut put = |i: usize, v: f32| {
        let at = i * texel + k * fb;
        layout.format.encode(v, &mut slab[at..at + fb]);
    };
    match dec {
        Decoded::U8(v) => v
            .iter()
            .take(voxels)
            .enumerate()
            .for_each(|(i, &x)| put(i, x as f32 * scale + offset)),
        Decoded::U16(v) => v
            .iter()
            .take(voxels)
            .enumerate()
            .for_each(|(i, &x)| put(i, x as f32 * scale + offset)),
        Decoded::F32(v) => v
            .iter()
            .take(voxels)
            .enumerate()
            .for_each(|(i, &x)| put(i, x * scale + offset)),
    }
}

/// The sums an output slice averages, per channel: one per output pixel, over
/// the `stride` box of source pixels behind it in every source slice added.
struct SliceBoxes {
    sums: Vec<Vec<f32>>,
    /// Per float channel, how many samples behind each sum were finite, and
    /// which infinities were seen (bit 0 `+inf`, bit 1 `-inf`). NaN and the
    /// infinities are left out of the average rather than making all of it
    /// NaN or infinite: a stack masked with NaN, or the log of one with
    /// zeros, would otherwise lose a box-wide rim around everything it kept.
    /// A box with no finite sample is `+inf` if it saw one, else `-inf` if it
    /// saw that, else NaN.
    counts: Vec<Option<(Vec<u32>, Vec<u8>)>>,
    src: (u32, u32),
    stride: (u32, u32),
    out: (u32, u32),
}

impl SliceBoxes {
    fn new(channels: usize, src: (u32, u32), stride: (u32, u32), out: (u32, u32)) -> Self {
        SliceBoxes {
            sums: vec![vec![0.0; out.0 as usize * out.1 as usize]; channels],
            counts: vec![None; channels],
            src,
            stride,
            out,
        }
    }

    /// Add one decoded source slice of channel `c`, in texture units. The
    /// volume's normalization is applied to the average, in
    /// [`write`](Self::write): it is affine, so it commutes with averaging.
    fn add(&mut self, c: usize, dec: &Decoded) {
        match dec {
            Decoded::U8(v) => self.add_samples(c, v, |x| x as f32),
            Decoded::U16(v) => self.add_samples(c, v, |x| x as f32),
            Decoded::F32(v) => self.add_floats(c, v),
        }
    }

    /// [`add_samples`](Self::add_samples) for a float slice: the finite
    /// samples summed and counted, the rest noted (see `counts`).
    fn add_floats(&mut self, c: usize, src: &[f32]) {
        let (w, h) = self.src;
        let (sx, sy) = self.stride;
        let (ow, _) = self.out;
        let sums = &mut self.sums[c];
        let (counts, seen) =
            self.counts[c].get_or_insert_with(|| (vec![0; sums.len()], vec![0; sums.len()]));
        let rows = (src.len() / w.max(1) as usize).min(h as usize);
        for y in 0..rows {
            let o = (y / sy as usize) * ow as usize;
            for (x, &v) in src[y * w as usize..(y + 1) * w as usize].iter().enumerate() {
                let i = o + x / sx as usize;
                if v.is_finite() {
                    sums[i] += v;
                    counts[i] += 1;
                } else if v == f32::INFINITY {
                    seen[i] |= 1;
                } else if v == f32::NEG_INFINITY {
                    seen[i] |= 2;
                }
            }
        }
    }

    fn add_samples<T: Copy>(&mut self, c: usize, src: &[T], unit: impl Fn(T) -> f32) {
        let (w, h) = self.src;
        let (sx, sy) = self.stride;
        let (ow, _) = self.out;
        let sums = &mut self.sums[c];
        // A short decode leaves the rest of the slice as zeros, as before.
        let rows = (src.len() / w.max(1) as usize).min(h as usize);
        for y in 0..rows {
            let oy = y / sy as usize;
            let row = &src[y * w as usize..(y + 1) * w as usize];
            let out_row = &mut sums[oy * ow as usize..(oy + 1) * ow as usize];
            if sx == 1 {
                for (o, &v) in out_row.iter_mut().zip(row) {
                    *o += unit(v);
                }
            } else {
                for (x, &v) in row.iter().enumerate() {
                    out_row[x / sx as usize] += unit(v);
                }
            }
        }
    }

    /// Divide every sum by the number of source voxels behind it (`slices`
    /// source slices deep, the box clipped at the far edges), normalize it by
    /// its channel's `affines` entry, and store it into this output slice's
    /// slab of every packed texture.
    fn write(
        &self,
        layout: &VolumeLayout,
        slabs: &mut [&mut [u8]],
        slices: u32,
        affines: &[(f32, f32)],
    ) {
        let (w, h) = self.src;
        let (sx, sy) = self.stride;
        let (ow, oh) = self.out;
        let fb = layout.format.bytes();
        let count = |o: u32, s: u32, n: u32| (n - o * s).min(s);
        for (c, sums) in self.sums.iter().enumerate() {
            let (scale, offset) = affines[c];
            let (t, k) = layout.place(c);
            let texel = layout.texel_bytes(t);
            let slab = &mut slabs[t];
            let numbers = self.counts[c].as_ref();
            for oy in 0..oh {
                let ny = count(oy, sy, h);
                for ox in 0..ow {
                    let nx = count(ox, sx, w);
                    let i = oy as usize * ow as usize + ox as usize;
                    let v = match numbers {
                        Some((n, _)) if n[i] > 0 => sums[i] / n[i] as f32,
                        Some((_, seen)) if seen[i] & 1 != 0 => f32::INFINITY,
                        Some((_, seen)) if seen[i] & 2 != 0 => f32::NEG_INFINITY,
                        Some(_) => f32::NAN,
                        None => sums[i] / (nx * ny * slices.max(1)) as f32,
                    };
                    let at = i * texel + k * fb;
                    layout
                        .format
                        .encode(v * scale + offset, &mut slab[at..at + fb]);
                }
            }
        }
    }
}

/// The background volume builder: a `LatestWorker` ([`crate::worker`]) keyed
/// by `(generation, time)`.
///
/// A newer request cancels the build in progress at its next plane (see
/// [`build_volume`]), and dropping the builder — closing the file, opening
/// another — does the same, so the thread lets go of the stack and its
/// half-built volume one plane after it is dropped rather than when the build
/// would have finished. A finished volume nobody took is dropped before the
/// next build starts (`Unclaimed::Discard`): one builder never holds two.
///
/// `new` returns `None` when the thread cannot start, and always without the
/// `threads` feature; every build then runs synchronously through
/// [`build_volume`].
pub struct VolumeBuilder(LatestWorker<(u64, usize), VolumePlan, Option<BuiltVolume>>);

impl VolumeBuilder {
    /// Spawn a worker over `stack` — the window's own, shared, so a stack held
    /// in memory (a plugin's result, an import, every file in a browser) gets
    /// one as well as a mapped file does. Returns `None` if the thread fails
    /// to spawn — callers then build synchronously instead.
    pub fn new(stack: Arc<TiffStack>) -> Option<Self> {
        // A failed build is published as `None` (the app marks the timepoint
        // built anyway, so it doesn't retry every frame); a cancelled one is
        // not published at all.
        let build = move |_: &(u64, usize), plan: VolumePlan, cancel: &CancelToken| {
            build_volume(&stack, &plan, cancel).ok()
        };
        LatestWorker::spawn("fasttiff-volume", Unclaimed::Discard, build).map(VolumeBuilder)
    }

    /// Queue a build. Asking again for the `(generation, time)` already queued
    /// or building does nothing; asking for another cancels it. Returns
    /// `false` when the worker is gone (a build panicked) — the caller should
    /// build synchronously instead.
    pub fn request(&self, generation: u64, plan: VolumePlan) -> bool {
        self.0.request((generation, plan.time), plan)
    }

    /// Take the finished volume iff it matches `(generation, time)`; otherwise
    /// leave the slot untouched (a newer build will replace it) and return
    /// `None`. The outer `Option` is "is there a matching reply"; the inner one
    /// is the build's own success.
    #[allow(clippy::option_option)]
    pub fn take_matching(&self, generation: u64, time: usize) -> Option<Option<BuiltVolume>> {
        self.0.take_if(|key| *key == (generation, time))
    }
}

#[cfg(test)]
#[path = "volume_tests.rs"]
mod tests;

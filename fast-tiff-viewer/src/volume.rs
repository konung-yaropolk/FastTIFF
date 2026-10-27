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
//! The worker mirrors `prefetch::Prefetcher`: it opens its **own** memory map
//! of the file (sharing the OS page cache), takes the latest queued request
//! (draining stale ones, so scrubbing a 4D stack skips timepoints instead of
//! queueing them all), and posts the finished volume into a result slot the UI
//! thread polls. Results are tagged with `(generation, time)` and used only on
//! an exact match, so a stale build can cost work but never show wrong data.

use crate::prefetch::{decode_jobs, ChannelJob, Decoded};
use fast_tiff_lib::TiffStack;
#[cfg(feature = "threads")]
use rayon::prelude::*;
use scivis_render::pack::{self, PackedVolume, TexelFormat, VolumeLayout};
use scivis_render::{ChannelKind, MAX_CHANNELS};
use std::path::PathBuf;
#[cfg(feature = "threads")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "threads")]
use std::sync::mpsc::{channel, Receiver, Sender};
#[cfg(feature = "threads")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "threads")]
use std::thread::JoinHandle;

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
    pub kinds: Vec<ChannelKind>,
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
/// [`choose_strides`]. Returns `None` when there's no volume to build (or a
/// decode failed).
pub fn build_volume(tiff: &TiffStack, plan: &VolumePlan) -> Option<BuiltVolume> {
    let f0 = tiff.frames.first()?;
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
    let n = plan.kinds.len().min(MAX_CHANNELS);
    if w == 0 || h == 0 || depth < 2 || n == 0 {
        return None;
    }
    let ckinds = &plan.kinds[..n];

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
                    encode_slice(&layout, &mut slabs, c, dec);
                } else {
                    boxes.add(c, dec);
                }
            }
        }
        if !direct {
            boxes.write(&layout, &mut slabs, z_to - z_from);
        }
        Ok(())
    });
    if let Err(e) = result {
        log::warn!("volume build failed: {e:#}");
        return None;
    }
    packed.finish();
    Some(packed)
}

/// Store one decoded slice of channel `c`, at full size, into its slab of the
/// packed texture that holds it. A short decode leaves the rest zero.
fn encode_slice(layout: &VolumeLayout, slabs: &mut [&mut [u8]], c: usize, dec: &Decoded) {
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
            .for_each(|(i, &x)| put(i, x as f32 / 255.0)),
        Decoded::U16(v) => v
            .iter()
            .take(voxels)
            .enumerate()
            .for_each(|(i, &x)| put(i, x as f32 / 65535.0)),
        Decoded::F32(v) => v
            .iter()
            .take(voxels)
            .enumerate()
            .for_each(|(i, &x)| put(i, x)),
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

    /// Add one decoded source slice of channel `c`, in texture units.
    fn add(&mut self, c: usize, dec: &Decoded) {
        match dec {
            Decoded::U8(v) => self.add_samples(c, v, |x| x as f32 / 255.0),
            Decoded::U16(v) => self.add_samples(c, v, |x| x as f32 / 65535.0),
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
    /// source slices deep, the box clipped at the far edges) and store it into
    /// this output slice's slab of every packed texture.
    fn write(&self, layout: &VolumeLayout, slabs: &mut [&mut [u8]], slices: u32) {
        let (w, h) = self.src;
        let (sx, sy) = self.stride;
        let (ow, oh) = self.out;
        let fb = layout.format.bytes();
        let count = |o: u32, s: u32, n: u32| (n - o * s).min(s);
        for (c, sums) in self.sums.iter().enumerate() {
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
                    layout.format.encode(v, &mut slab[at..at + fb]);
                }
            }
        }
    }
}

#[cfg(feature = "threads")]
struct Request {
    generation: u64,
    plan: VolumePlan,
}

/// A finished background build, tagged so the app can confirm it still matches
/// what's wanted. `volume: None` means the build failed (the app marks the
/// timepoint built anyway, so it doesn't retry every frame).
#[cfg(feature = "threads")]
struct BuiltReply {
    generation: u64,
    time: usize,
    volume: Option<BuiltVolume>,
}

/// Owns the volume-builder worker thread + the latest result. Dropping it
/// closes the request channel, which ends the worker (it finishes any in-flight
/// build first).
#[cfg(feature = "threads")]
pub struct VolumeBuilder {
    tx: Sender<Request>,
    result: Arc<Mutex<Option<BuiltReply>>>,
    /// Set by the worker when it could not open the file and gave up.
    ///
    /// A dead worker cannot be detected by *asking* it: `request` only fails
    /// once the receiver has been dropped, which is a frame or two after the
    /// open failed, and by then a request has already been queued and is being
    /// polled for a reply that will never come. This is the flag that says so
    /// on every frame after, rather than only on the one that happened to send.
    failed: Arc<AtomicBool>,
    _handle: JoinHandle<()>,
}

#[cfg(feature = "threads")]
impl VolumeBuilder {
    /// Spawn a worker that opens its own map of `path` (shares the OS page
    /// cache — no duplicate pixel RAM). Returns `None` if the thread fails to
    /// spawn; if the worker's file open fails the thread exits and `request`
    /// starts returning `false` — callers then build synchronously instead.
    pub fn new(path: PathBuf) -> Option<Self> {
        let (tx, rx) = channel::<Request>();
        let result = Arc::new(Mutex::new(None));
        let result_worker = Arc::clone(&result);
        let failed = Arc::new(AtomicBool::new(false));
        let failed_worker = Arc::clone(&failed);
        let handle = std::thread::Builder::new()
            .name("fasttiff-volume".to_owned())
            .spawn(move || match TiffStack::open(&path) {
                Ok(stack) => worker_loop(stack, rx, result_worker),
                Err(e) => {
                    // Not always a problem worth reporting: a stack a plugin
                    // imported has no file on disk at all, and its `path` is
                    // the name it is shown under. The caller builds volumes
                    // here on the main thread instead.
                    log::debug!("volume builder: can't open {}: {e:#}", path.display());
                    failed_worker.store(true, Ordering::Release);
                }
            })
            .ok()?;
        Some(Self {
            tx,
            result,
            failed,
            _handle: handle,
        })
    }

    /// Whether the worker gave up before it started: its file could not be
    /// opened, and it will never answer anything.
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Queue a build. Returns `false` when the worker is gone (its file open
    /// failed) — the caller should build synchronously instead.
    pub fn request(&self, generation: u64, plan: VolumePlan) -> bool {
        self.tx.send(Request { generation, plan }).is_ok()
    }

    /// Take the finished volume iff it matches `(generation, time)`; otherwise
    /// leave the slot untouched (a newer build will overwrite it) and return
    /// `None`. The outer `Option` is "is there a matching reply"; the inner one
    /// is the build's own success.
    #[allow(clippy::option_option)]
    pub fn take_matching(&self, generation: u64, time: usize) -> Option<Option<BuiltVolume>> {
        let mut slot = self.result.lock().ok()?;
        let matches = slot
            .as_ref()
            .is_some_and(|r| r.generation == generation && r.time == time);
        if matches {
            slot.take().map(|r| r.volume)
        } else {
            None
        }
    }
}

#[cfg(feature = "threads")]
fn worker_loop(stack: TiffStack, rx: Receiver<Request>, result: Arc<Mutex<Option<BuiltReply>>>) {
    // Block for a request; channel closed (VolumeBuilder dropped) -> exit.
    while let Ok(mut req) = rx.recv() {
        // Only the most recent request matters: a 4D scrub queues many
        // timepoints, and building stale ones would just delay the wanted one.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        // A reply still in the slot is for an earlier request, which nobody
        // waits for now (the frontend asks once per build it wants): let it
        // go before building, so two volumes are never held at once.
        if let Ok(mut slot) = result.lock() {
            *slot = None;
        }
        let volume = build_volume(&stack, &req.plan);
        if let Ok(mut slot) = result.lock() {
            *slot = Some(BuiltReply {
                generation: req.generation,
                time: req.plan.time,
                volume,
            });
        }
    }
}

/// Without threads there's no background builder; `VolumeBuilder::new` returns
/// `None` and every build runs synchronously through [`build_volume`] — the
/// same path a failed thread spawn already takes.
#[cfg(not(feature = "threads"))]
pub struct VolumeBuilder(std::convert::Infallible);

#[cfg(not(feature = "threads"))]
impl VolumeBuilder {
    pub fn new(_path: PathBuf) -> Option<Self> {
        None
    }

    pub fn request(&self, _generation: u64, _plan: VolumePlan) -> bool {
        match self.0 {}
    }

    pub fn failed(&self) -> bool {
        match self.0 {}
    }

    #[allow(clippy::option_option)]
    pub fn take_matching(&self, _generation: u64, _time: usize) -> Option<Option<BuiltVolume>> {
        match self.0 {}
    }
}

#[cfg(test)]
#[cfg(feature = "threads")]
#[path = "volume_tests.rs"]
mod tests;

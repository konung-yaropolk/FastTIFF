//! 3D volume assembly for the volume view: decoding a stack (or, for a 4D
//! stack, one timepoint's Z range) into per-channel scalar volumes sized to fit
//! the GPU's 3D-texture limit and a memory budget — plus a background builder
//! thread so neither the initial build nor a 4D timepoint rebuild blocks the UI.
//!
//! The build is parallel **across output slices** (rayon): each task decodes
//! one source plane set (RGB planes share a single decompression) and writes
//! its per-channel destination slabs, which are disjoint `chunks_mut` of the
//! final buffers — no locking, no reassembly pass.
//!
//! The worker mirrors `prefetch::Prefetcher`: it opens its **own** memory map
//! of the file (sharing the OS page cache), takes the latest queued request
//! (draining stale ones, so scrubbing a 4D stack skips timepoints instead of
//! queueing them all), and posts the finished volume into a result slot the UI
//! thread polls. Results are tagged with `(generation, time)` and used only on
//! an exact match, so a stale build can cost work but never show wrong data.

use crate::prefetch::{decode_jobs, ChannelJob, Decoded};
use crate::render::{self, ChannelKind, MAX_CHANNELS};
use fast_tiff_lib::TiffStack;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Rough upper bound on the volume's footprint. The builder subsamples each
/// axis until the volume fits under this before uploading, so a huge stack
/// degrades to a coarser volume rather than exhausting RAM/VRAM. Each channel
/// is budgeted at the *larger* of its CPU and GPU bytes-per-sample (the wgpu
/// backend stores every sample as 16 bits, so an 8-bit source costs 2 bytes of
/// VRAM; glow matches the CPU size exactly).
const MAX_VOLUME_BYTES: usize = 768 << 20;

/// A built volume: `(width, height, depth)` plus one `(kind, native-endian
/// bytes)` entry per channel — what `build_volume` hands to `upload_volumes`.
pub type BuiltVolume = (u32, u32, u32, Vec<(render::VolumeKind, Vec<u8>)>);

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
}

/// Build the 3D volume for `plan` — one scalar buffer per channel. The depth
/// axis is the whole frame axis in the ordinary case, or (4D, `slices > 1`)
/// the Z axis at timepoint `plan.time`. All channels share the same
/// subsampling, chosen so the *combined* footprint fits `plan.max_dim` and the
/// `MAX_VOLUME_BYTES` budget. Returns `None` when there's no volume to build
/// (or a decode failed).
pub fn build_volume(tiff: &TiffStack, plan: &VolumePlan) -> Option<BuiltVolume> {
    let f0 = tiff.frames.first()?;
    let (w, h) = (f0.width, f0.height);
    let slices = plan.slices.max(1);
    let channels = plan.channels.max(1);
    let is_4d = slices > 1;
    // Depth axis: Z at the current timepoint (4D) or the whole frame axis.
    let depth = if is_4d { slices as u32 } else { plan.frames.max(1) as u32 };
    let time = if is_4d { plan.time } else { 0 };
    let n = plan.kinds.len().min(MAX_CHANNELS);
    if w == 0 || h == 0 || depth < 2 || n == 0 {
        return None;
    }

    // Per-channel source kind + GPU volume kind + CPU bytes-per-sample.
    let ckinds = &plan.kinds[..n];
    let vkinds: Vec<(render::VolumeKind, usize)> = ckinds
        .iter()
        .map(|k| match k {
            ChannelKind::Int8 => (render::VolumeKind::U8, 1usize),
            ChannelKind::Int16 => (render::VolumeKind::U16, 2),
            ChannelKind::Float => (render::VolumeKind::F32, 4),
        })
        .collect();
    let sum_bps: usize = vkinds.iter().map(|(k, bps)| (*bps).max(render::volume_gpu_bps(*k))).sum();

    let max_dim = plan.max_dim.max(64);
    let out = |n: u32, s: u32| n.div_ceil(s);
    // Smallest stride that fits the texture-size limit per axis, then coarsen the
    // largest output axis until the *combined* byte budget is met.
    let mut sx = w.div_ceil(max_dim).max(1);
    let mut sy = h.div_ceil(max_dim).max(1);
    let mut sz = depth.div_ceil(max_dim).max(1);
    loop {
        let (ox, oy, oz) = (out(w, sx), out(h, sy), out(depth, sz));
        if (ox as usize) * (oy as usize) * (oz as usize) * sum_bps <= MAX_VOLUME_BYTES {
            break;
        }
        if ox >= oy && ox >= oz {
            sx += 1;
        } else if oy >= oz {
            sy += 1;
        } else {
            sz += 1;
        }
    }
    let (ow, oh, od) = (out(w, sx), out(h, sy), out(depth, sz));

    let mut bufs: Vec<Vec<u8>> = vkinds
        .iter()
        .map(|(_, bps)| vec![0u8; ow as usize * oh as usize * od as usize * bps])
        .collect();

    // Parallel across output slices: hand each task its per-channel destination
    // slabs — disjoint `chunks_mut` of the final buffers — decode once (RGB
    // planes share the decompression), downsample into place.
    let mut chunk_iters: Vec<_> = bufs
        .iter_mut()
        .zip(&vkinds)
        .map(|(b, (_, bps))| b.chunks_mut(ow as usize * oh as usize * bps))
        .collect();
    let mut tasks: Vec<(u32, Vec<&mut [u8]>)> = Vec::with_capacity(od as usize);
    for oz in 0..od {
        tasks.push((oz, chunk_iters.iter_mut().map(|it| it.next().expect("slab per slice")).collect()));
    }
    drop(chunk_iters);

    let result = tasks.into_par_iter().try_for_each(|(oz, mut dsts)| -> anyhow::Result<()> {
        // Depth index → (time, z): in 4D the depth is Z at the fixed timepoint;
        // otherwise the depth index *is* the frame (time = k, z = 0). The IFD
        // layout is time-major, then Z, then channel.
        let k = (oz * sz) as usize;
        let (t, z) = if is_4d { (time, k) } else { (k, 0) };
        let jobs: Vec<ChannelJob> = (0..n)
            .map(|c| {
                let (ifd_idx, plane) = if plan.rgb {
                    (t * slices + z, c)
                } else {
                    (t * slices * channels + z * channels + c, 0)
                };
                ChannelJob { channel: c, ifd_idx, plane, kind: ckinds[c], rgb: plan.rgb, width: w, height: h }
            })
            .collect();
        let decoded = decode_jobs(&tiff.mmap, &tiff.frames, tiff.byte_order, &jobs)?;
        for (dec, dst) in decoded.iter().zip(dsts.iter_mut()) {
            let dst: &mut [u8] = dst;
            match dec {
                Decoded::U8(v) => downsample_slice(v, (w, h), (sx, sy), (ow, oh), dst),
                Decoded::U16(v) => downsample_slice(v, (w, h), (sx, sy), (ow, oh), dst),
                Decoded::F32(v) => downsample_slice(v, (w, h), (sx, sy), (ow, oh), dst),
            }
        }
        Ok(())
    });
    if let Err(e) = result {
        log::warn!("volume build failed: {e:#}");
        return None;
    }

    let channels_out: Vec<(render::VolumeKind, Vec<u8>)> = vkinds.iter().map(|(k, _)| *k).zip(bufs).collect();
    Some((ow, oh, od, channels_out))
}

/// Point-sample one decoded slice into `dst` at `stride` (sx, sy), writing each
/// sample's native-endian bytes (matches the GPU upload's pixel type). `dims` is
/// the source (w, h); `out` is the destination (ow, oh).
fn downsample_slice<T: bytemuck::Pod>(src: &[T], dims: (u32, u32), stride: (u32, u32), out: (u32, u32), dst: &mut [u8]) {
    let (w, h) = dims;
    let (sx, sy) = stride;
    let (ow, oh) = out;
    let bps = std::mem::size_of::<T>();

    // Fast path — no x/y subsampling (the common case: only the frame axis was
    // strided, or nothing at all): the whole slice is one contiguous memcpy.
    // (`min` guards a short decode; the remainder stays zero, as below.)
    if sx == 1 && sy == 1 && ow == w && oh == h {
        let n = (w as usize * h as usize).min(src.len());
        dst[..n * bps].copy_from_slice(bytemuck::cast_slice(&src[..n]));
        return;
    }
    // Row fast path — y-only subsampling: each output row is one memcpy.
    if sx == 1 && ow == w {
        for oy in 0..oh {
            let s = (oy * sy).min(h - 1) as usize * w as usize;
            let row = &src[s.min(src.len())..(s + w as usize).min(src.len())];
            let di = oy as usize * ow as usize * bps;
            dst[di..di + std::mem::size_of_val(row)].copy_from_slice(bytemuck::cast_slice(row));
        }
        return;
    }
    // General point-sampled path.
    for oy in 0..oh {
        let src_row = (oy * sy).min(h - 1) as usize * w as usize;
        let dst_row = oy as usize * ow as usize;
        for ox in 0..ow {
            let sx_i = (ox * sx).min(w - 1) as usize;
            if let Some(sample) = src.get(src_row + sx_i) {
                let di = (dst_row + ox as usize) * bps;
                dst[di..di + bps].copy_from_slice(bytemuck::bytes_of(sample));
            }
        }
    }
}

struct Request {
    generation: u64,
    plan: VolumePlan,
}

/// A finished background build, tagged so the app can confirm it still matches
/// what's wanted. `volume: None` means the build failed (the app marks the
/// timepoint built anyway, so it doesn't retry every frame).
struct BuiltReply {
    generation: u64,
    time: usize,
    volume: Option<BuiltVolume>,
}

/// Owns the volume-builder worker thread + the latest result. Dropping it
/// closes the request channel, which ends the worker (it finishes any in-flight
/// build first).
pub struct VolumeBuilder {
    tx: Sender<Request>,
    result: Arc<Mutex<Option<BuiltReply>>>,
    _handle: JoinHandle<()>,
}

impl VolumeBuilder {
    /// Spawn a worker that opens its own map of `path` (shares the OS page
    /// cache — no duplicate pixel RAM). Returns `None` if the thread fails to
    /// spawn; if the worker's file open fails the thread exits and `request`
    /// starts returning `false` — callers then build synchronously instead.
    pub fn new(path: PathBuf) -> Option<Self> {
        let (tx, rx) = channel::<Request>();
        let result = Arc::new(Mutex::new(None));
        let result_worker = Arc::clone(&result);
        let handle = std::thread::Builder::new()
            .name("fasttiff-volume".to_owned())
            .spawn(move || match TiffStack::open(&path) {
                Ok(stack) => worker_loop(stack, rx, result_worker),
                Err(e) => log::warn!("volume builder: can't open {}: {e:#}", path.display()),
            })
            .ok()?;
        Some(Self { tx, result, _handle: handle })
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

fn worker_loop(stack: TiffStack, rx: Receiver<Request>, result: Arc<Mutex<Option<BuiltReply>>>) {
    // Block for a request; channel closed (VolumeBuilder dropped) -> exit.
    while let Ok(mut req) = rx.recv() {
        // Only the most recent request matters: a 4D scrub queues many
        // timepoints, and building stale ones would just delay the wanted one.
        while let Ok(newer) = rx.try_recv() {
            req = newer;
        }
        let volume = build_volume(&stack, &req.plan);
        if let Ok(mut slot) = result.lock() {
            *slot = Some(BuiltReply { generation: req.generation, time: req.plan.time, volume });
        }
    }
}

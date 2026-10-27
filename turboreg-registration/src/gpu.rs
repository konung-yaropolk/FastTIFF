// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! The criterion on the graphics card.
//!
//! # What runs here, and what does not
//!
//! The search spends nearly all its time in one place: scoring a candidate
//! transformation, which means interpolating the moving image and both its
//! gradient images at every pixel and summing the squared residual, the
//! gradient and the curvature. That sum is what this module puts on the device.
//!
//! Everything else stays on the CPU, multi-threaded: building the pyramids, the
//! spline prefilters (a recursive filter, which is a poor fit for a GPU), the
//! accept/reject logic, and resampling the result. So does every pyramid level
//! smaller than [`MIN_PIXELS`] — the coarse levels have the most iterations and
//! the fewest pixels, and there a round trip to the device costs more than the
//! arithmetic it would save.
//!
//! # Why frames go in batches
//!
//! The search cannot run ahead of the device: whether a step is accepted
//! decides where the next one goes, so every step needs its score back before
//! the next is taken. One frame at a time that is a round trip per step, and a
//! round trip is far longer than the step. So frames are scored as *lanes* of
//! one dispatch: every frame of the batch takes its next step together, one
//! submission scores all of them, and the host then updates each one. The
//! round trips are shared across the batch.
//!
//! That needs frames that do not depend on each other, and in every reference
//! mode they do not. Each lane carries its own reference, so the previous-frame
//! mode batches like the others: lane `t` is fitted against the raw frame
//! `t - 1`, which does not wait on any other lane's answer. Only composing the
//! answers into matrices relative to frame 0 is sequential, and that happens on
//! the host afterwards. (It used to run one lane per round trip, on the false
//! premise that the fits themselves formed a chain, and was slower on the
//! device than on the CPU.)
//!
//! # How it agrees with the CPU
//!
//! **To a tolerance, not bit for bit.** WGSL has no `f64`, so the per-pixel
//! arithmetic is `f32` where the CPU's is `f64`, and the accept/reject branch
//! can resolve a near-tie the other way, after which the two searches take
//! slightly different paths to the same minimum. The answers agree to well
//! under a hundredth of a pixel on the test data; `tests/gpu_agrees.rs` holds
//! them to that.
//!
//! What is preserved is determinism: which pixel each invocation takes and the
//! shape of every reduction are fixed by the image, never by scheduling, and
//! the host adds the per-workgroup sums in a fixed order in `f64`. The same
//! input gives the same answer on the same device, run after run.
//!
//! # Refused, not substituted
//!
//! Without a usable adapter — and a software adapter does not count, since it
//! would run this "GPU" path on the processor, slower than the processor path —
//! the backend reports itself unavailable rather than quietly running on the
//! CPU. In a browser it always does: the device would have to be awaited, and
//! there is nothing here that could wait for it without freezing the page.
//!
//! # Whose device
//!
//! A host that draws with wgpu should hand this crate its own device with
//! [`use_device`] before the first run, and the registration then runs on it.
//! Left alone, the crate opens a device of its own — and opening one beside a
//! renderer's, on a card whose memory the two of them have overrun, can
//! deadlock both: measured on a Quadro P620 (2 GB, Vulkan), with a 1.2 GB
//! volume resident on the renderer's device, opening this crate's never
//! returned, and the renderer's next frame never completed either. In an
//! application that ends with the system resetting the GPU, and every device
//! in the process lost.
//!
//! Sharing a device puts the registration's memory next to the renderer's on
//! the same card, and on a small card the two can overrun it. So nothing here
//! fails the way wgpu fails by default, which is to panic: every allocation
//! and every round runs inside error scopes, and a level the device has no
//! room for fails its batch — its frames are reported as not fitted, and
//! [`take_trouble`] says why.
//!
//! What a scope cannot do is keep the device. wgpu (29) survives running out
//! of memory for a buffer or a texture, which is where nearly all of this
//! crate's memory goes; out of memory for its own staging memory, a pipeline
//! or a submission it treats as the device lost, and a shared device is the
//! renderer's. The planes go up a slab at a time so that the staging an upload
//! asks for stays small at any frame size, and on Vulkan that memory falls
//! back to the host's when the card's is gone.

use crate::optimize::{Prepared, Sums};
use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
use std::time::Duration;

/// Invocations per workgroup. Must match `WG` in `gpu.wgsl`.
const WG: usize = 256;

/// The most workgroups one lane's pixels are split into. Each one sends a row
/// of partial sums back, so this bounds the read-back; past it, invocations
/// simply take more pixels each.
const MOST_GROUPS_PER_LANE: usize = 64;

/// The most frames scored in one dispatch. Past this the round trips are
/// already shared thinly enough that more lanes buy little, and the batch's
/// coefficients have to fit next to the viewer's own textures.
pub const MOST_LANES: usize = 32;

/// How much device memory one batch's coefficient planes may take.
const BATCH_MEMORY: u64 = 256 << 20;

/// The most bytes of planes in one write: the staging memory an upload needs is
/// two of these, whatever the frame size (see [`upload_planes`]).
const UPLOAD_SLAB: usize = 32 << 20;

/// Pyramid levels with fewer pixels than this are fitted on the CPU.
///
/// The coarse levels are where the search takes most of its steps — the cap is
/// ten times two to the octave — on the fewest pixels, so on the device they
/// would be all round trip and no arithmetic. Fixed rather than measured at
/// run time, so which levels run where depends only on the image size and the
/// result is reproducible.
pub const MIN_PIXELS: usize = 128 * 128;

/// How long one round may take on the device before it is given up for lost.
///
/// A round is milliseconds. This is only for a device that has stopped
/// answering — a driver reset, a card pulled from under a laptop — where
/// waiting without a limit would hang the plugin's worker for good.
const ROUND_TIMEOUT: Duration = Duration::from_secs(60);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Info {
    w: u32,
    h: u32,
    pixels: u32,
    groups: u32,
}

/// `Lane` in `gpu.wgsl`: 304 bytes, f32 arrays at stride 4.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LaneGpu {
    slot: u32,
    derivatives: u32,
    pad0: u32,
    pad1: u32,
    m: [f32; 8],
    dm: [f32; 64],
}

/// The device this crate runs on, and its adapter's name.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct Shared {
    device: wgpu::Device,
    queue: wgpu::Queue,
    name: String,
}

/// The one device this crate uses: the host's, given with [`use_device`], or
/// else one of its own, opened the first time it is needed.
///
/// One for the process rather than one per batch. Several devices opened side
/// by side in one process, each driven from its own thread, hang on the drivers
/// this workspace was measured against — `suite2p-registration` found that out
/// first — and a registration only ever needs one anyway.
///
/// `None`, remembered, when there is no usable adapter.
///
/// Not in a browser build, where there is never a device to keep (see
/// [`shared_device`]) and wgpu's browser objects cannot sit in a `static`.
#[cfg(not(target_arch = "wasm32"))]
static DEVICE: OnceLock<Option<Shared>> = OnceLock::new();

/// Run on `device` instead of opening a device of this crate's own: the
/// host's, typically the one its renderer draws with, which then carries the
/// registration's work between its own frames. See the module docs for why a
/// host that has one should.
///
/// Takes effect only before the first GPU run (or [`available`] call), and
/// says whether it did. A software adapter counts as no device, as it would
/// for one this crate opened. The device should have the adapter's own limits
/// (`required_limits: adapter.limits()`): the batch size is derived from them.
#[cfg(not(target_arch = "wasm32"))]
pub fn use_device(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::AdapterInfo) -> bool {
    let shared = (adapter.device_type != wgpu::DeviceType::Cpu).then(|| Shared {
        device,
        queue,
        name: adapter.name.clone(),
    });
    DEVICE.set(shared).is_ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn shared() -> Option<Shared> {
    DEVICE
        .get_or_init(|| {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .ok()?;
            let info = adapter.get_info();
            if info.device_type == wgpu::DeviceType::Cpu {
                return None;
            }
            // Whatever the adapter can do: the batch size is derived from
            // these, and the defaults would cap it well below the card.
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("turboreg-registration"),
                    required_limits: adapter.limits(),
                    ..Default::default()
                }))
                .ok()?;
            Some(Shared {
                device,
                queue,
                name: info.name,
            })
        })
        .clone()
}

fn shared_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    // Requesting an adapter is asynchronous, and in a browser it resolves
    // through the page's event loop — which `block_on` would be blocking. So
    // there is no device to offer there; the backend says so instead.
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        shared().map(|s| (s.device, s.queue))
    }
}

/// Whether this machine has a device the GPU backend can use.
pub fn available() -> bool {
    shared_device().is_some()
}

/// The device's name, for a log line saying where a run went.
pub fn adapter_name() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Kept from when the device was opened or handed over, rather than
        // asked of a new instance: enumerating adapters again is opening
        // drivers again, which is the step that can deadlock (module docs).
        shared().map(|s| s.name)
    }
}

/// How many frames of `w * h` fit in one batch on this device. Zero when not
/// even one does, or there is no device.
pub fn lanes_for(w: usize, h: usize) -> usize {
    let Some((device, _)) = shared_device() else {
        return 0;
    };
    let limits = device.limits();
    let plane = (w * h * 4) as u64;
    if plane == 0 {
        return 0;
    }
    let budget = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
        .min(BATCH_MEMORY);
    // Four planes per lane: its reference, and its image's three coefficient
    // planes.
    ((budget / (4 * plane)) as usize).min(MOST_LANES)
}

/// One pipeline per parameter count, built the first time each is needed.
///
/// The count is compiled in rather than passed at run time so each invocation's
/// running sums are fixed-size arrays the compiler can keep in registers.
///
/// Native only, like [`DEVICE`]: a browser build never has a device to build a
/// pipeline for.
#[cfg(not(target_arch = "wasm32"))]
static PIPELINES: Mutex<Vec<(usize, wgpu::ComputePipeline)>> = Mutex::new(Vec::new());

fn pipeline(device: &wgpu::Device, n: usize) -> Option<wgpu::ComputePipeline> {
    #[cfg(not(target_arch = "wasm32"))]
    let mut cache = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(target_arch = "wasm32")]
    let mut cache: Vec<(usize, wgpu::ComputePipeline)> = Vec::new();
    if let Some((_, p)) = cache.iter().find(|(k, _)| *k == n) {
        return Some(p.clone());
    }
    let h = n * (n + 1) / 2;
    let source = include_str!("gpu.wgsl")
        .replace("__N__", &n.to_string())
        .replace("__H__", &h.to_string())
        .replace("__K__", &(1 + n + h).to_string());
    // A driver that rejects the shader should make the backend unavailable,
    // not take the application down with an uncaptured validation error.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("turboreg"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let built = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("turboreg evaluate"),
        layout: None,
        module: &module,
        entry_point: Some("evaluate"),
        compilation_options: Default::default(),
        cache: None,
    });
    if pollster::block_on(scope.pop()).is_some() {
        return None;
    }
    cache.push((n, built.clone()));
    Some(built)
}

/// One submission in flight at a time. Batches from different threads queue
/// here rather than interleaving on the device, which keeps memory use bounded
/// and the device's behaviour predictable.
static ROUND: Mutex<()> = Mutex::new(());

/// Why the device last failed a batch, until [`take_trouble`] takes it.
static TROUBLE: Mutex<Option<String>> = Mutex::new(None);

fn note_trouble(why: String) {
    *TROUBLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
}

/// Why the device failed a batch since this was last asked, if it did: no
/// room on it for a level's planes, or a device that stopped answering. The
/// batch's frames are reported as not fitted (`Registered::failed`); this is
/// the sentence that says why. Ask after a run, and once before it to forget
/// an earlier one's.
pub fn take_trouble() -> Option<String> {
    TROUBLE.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// Run `work` with the device's out-of-memory and validation errors caught,
/// and say which was raised, if one was.
///
/// Without a scope wgpu hands an error to the device's handler, which by
/// default panics — on a device shared with a renderer, taking the
/// application with it for want of memory a registration could do without.
/// Scopes are per thread in wgpu, so these catch this thread's errors only,
/// never the renderer's.
fn scoped<T>(device: &wgpu::Device, work: impl FnOnce() -> T) -> Result<T, String> {
    let no_room = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let done = work();
    // Innermost first.
    let invalid = pollster::block_on(invalid.pop());
    let no_room = pollster::block_on(no_room.pop());
    match no_room.or(invalid) {
        None => Ok(done),
        Some(e) => Err(e.to_string()),
    }
}

/// Copy each slot's four planes into `coeffs`, [`UPLOAD_SLAB`] bytes a
/// write, each write submitted and the one before it waited for.
///
/// A write is staged in memory of the queue's own until the submission that
/// carries it is done. All of a 4096 x 4096 level at once is 256 MB of that,
/// asked for at the moment a card short of memory can least spare it, and a
/// staging allocation that fails can lose the device. Two slabs in flight
/// keep it to 64 MB at any size, and the waits are short: a slab is a copy.
/// A buffer of `contents`, for inside a [`scoped`] call: made unmapped and
/// filled by the queue. `create_buffer_init` maps the buffer at creation
/// instead, and wgpu treats mapping a buffer that could not be made as fatal,
/// past any scope; this way a buffer there was no room for is an error the
/// scope returns.
fn filled(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = (contents.len() as u64)
        .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(4);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut padded = contents.to_vec();
    padded.resize(size as usize, 0);
    queue.write_buffer(&buffer, 0, &padded);
    buffer
}

fn upload_planes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    coeffs: &wgpu::Buffer,
    planes: &[[&[f32]; 4]],
    pixels: usize,
    slab: usize,
) -> Result<(), String> {
    let per = (slab / 4).max(1);
    let mut before: Option<wgpu::SubmissionIndex> = None;
    for (slot, four) in planes.iter().enumerate() {
        for (p, plane) in four.iter().enumerate() {
            let at = ((slot * 4 + p) * pixels * 4) as u64;
            for (k, slab) in plane[..pixels].chunks(per).enumerate() {
                let submitted = scoped(device, || {
                    queue.write_buffer(
                        coeffs,
                        at + (k * per * 4) as u64,
                        bytemuck::cast_slice(slab),
                    );
                    queue.submit([])
                })?;
                if let Some(previous) = before.replace(submitted) {
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(previous),
                            timeout: Some(ROUND_TIMEOUT),
                        })
                        .map_err(|e| format!("the GPU stopped answering ({e})"))?;
                }
            }
        }
    }
    Ok(())
}

/// One pyramid level of one batch, on the device.
///
/// Built once per level: the reference samples, and each lane's three
/// coefficient planes, uploaded together. After that each round sends only the
/// lanes' candidate transformations — a few hundred bytes a frame — and reads
/// back one row of partial sums per workgroup.
pub(crate) struct GpuLevel {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    lanes: wgpu::Buffer,
    partials: wgpu::Buffer,
    readback: wgpu::Buffer,
    n: usize,
    k: usize,
    pixels: usize,
    groups: usize,
    slots: usize,
}

impl GpuLevel {
    /// Upload a level. `planes[s]` is slot `s`'s reference samples, then its
    /// image's, x-gradient's and y-gradient's spline coefficients, each `w * h`.
    ///
    /// The reference is per slot because lanes need not share one: under the
    /// previous-frame reference each frame is aligned to the one before it.
    ///
    /// `None` when there is no device, the shader will not build, or the batch
    /// does not fit.
    pub fn new(n: usize, w: usize, h: usize, planes: &[[&[f32]; 4]]) -> Option<GpuLevel> {
        Self::build(n, w, h, planes, UPLOAD_SLAB)
    }

    /// [`new`](Self::new), the planes uploaded `slab` bytes a write.
    fn build(
        n: usize,
        w: usize,
        h: usize,
        planes: &[[&[f32]; 4]],
        slab: usize,
    ) -> Option<GpuLevel> {
        let (device, queue) = shared_device()?;
        let Some(pipeline) = pipeline(&device, n) else {
            note_trouble("the GPU rejected the registration shader".to_string());
            return None;
        };
        let pixels = w * h;
        let slots = planes.len();
        if pixels == 0 || slots == 0 {
            return None;
        }
        let limits = device.limits();
        let coeff_bytes = (slots * 4 * pixels * 4) as u64;
        if coeff_bytes > limits.max_storage_buffer_binding_size
            || coeff_bytes > limits.max_buffer_size
        {
            note_trouble(format!(
                "a {w} x {h} level of {slots} frame(s) ({} MB) is more than one GPU buffer holds",
                coeff_bytes >> 20
            ));
            return None;
        }
        let groups = pixels.div_ceil(WG).clamp(1, MOST_GROUPS_PER_LANE);
        let k = 1 + n + n * (n + 1) / 2;

        if planes.iter().flatten().any(|plane| plane.len() < pixels) {
            return None;
        }

        // Rounded up to 8 bytes, which a mapped range has to be.
        let out_bytes = ((slots * groups * k * 4) as u64).next_multiple_of(8);
        let made = scoped(&device, || {
            let info = filled(
                &device,
                &queue,
                "turboreg info",
                bytemuck::bytes_of(&Info {
                    w: w as u32,
                    h: h as u32,
                    pixels: pixels as u32,
                    groups: groups as u32,
                }),
                wgpu::BufferUsages::UNIFORM,
            );
            let coeffs = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("turboreg planes"),
                size: coeff_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let lanes = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("turboreg lanes"),
                size: (slots * std::mem::size_of::<LaneGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let partials = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("turboreg partials"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("turboreg readback"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("turboreg level"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: info.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: coeffs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lanes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: partials.as_entire_binding(),
                    },
                ],
            });
            (coeffs, lanes, partials, readback, bind)
        });
        // Nothing is written to a buffer that came back invalid: that would be
        // an error of its own, outside any scope.
        let (coeffs, lanes, partials, readback, bind) = match made {
            Ok(made) => made,
            Err(why) => {
                note_trouble(format!(
                    "the GPU had no room for a {w} x {h} level of {slots} frame(s) \
                     ({} MB): {why}",
                    coeff_bytes >> 20
                ));
                return None;
            }
        };
        if let Err(why) = upload_planes(&device, &queue, &coeffs, planes, pixels, slab) {
            note_trouble(format!("uploading a {w} x {h} level failed: {why}"));
            return None;
        }
        Some(GpuLevel {
            device,
            queue,
            pipeline,
            bind,
            lanes,
            partials,
            readback,
            n,
            k,
            pixels,
            groups,
            slots,
        })
    }

    /// Score one candidate for each entry: `(slot, candidate)`.
    ///
    /// One submission and one wait for all of them. `None` if the device did
    /// not answer, in which case this level is finished with.
    pub fn evaluate(&self, batch: &[(usize, &Prepared)]) -> Option<Vec<Sums>> {
        if batch.is_empty() {
            return Some(Vec::new());
        }
        // More lanes than the partials buffer holds, or a slot with no planes
        // behind it, would have the kernel read or write past a buffer.
        if batch.len() > self.slots || batch.iter().any(|&(s, _)| s >= self.slots) {
            return None;
        }
        let n = self.n;
        let lanes: Vec<LaneGpu> = batch
            .iter()
            .map(|&(slot, prep)| {
                let mut lane = LaneGpu {
                    slot: slot as u32,
                    derivatives: u32::from(prep.derivatives),
                    pad0: 0,
                    pad1: 0,
                    m: [0.0; 8],
                    dm: [0.0; 64],
                };
                for (o, &c) in lane.m.iter_mut().zip(prep.m.c.iter()) {
                    *o = c as f32;
                }
                for (k, d) in prep.dm.iter().enumerate().take(n) {
                    for (o, &c) in lane.dm[k * 8..k * 8 + 8].iter_mut().zip(d.c.iter()) {
                        *o = c as f32;
                    }
                }
                lane
            })
            .collect();

        let _one_at_a_time = ROUND.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = ((batch.len() * self.groups * self.k * 4) as u64).next_multiple_of(8);
        let submitted = scoped(&self.device, || {
            self.queue
                .write_buffer(&self.lanes, 0, bytemuck::cast_slice(&lanes));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("turboreg round"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("turboreg evaluate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind, &[]);
                pass.dispatch_workgroups(self.groups as u32, batch.len() as u32, 1);
            }
            encoder.copy_buffer_to_buffer(&self.partials, 0, &self.readback, 0, bytes);
            self.queue.submit(Some(encoder.finish()))
        });
        let submitted = match submitted {
            Ok(submitted) => submitted,
            Err(why) => {
                note_trouble(format!("a round failed on the GPU: {why}"));
                return None;
            }
        };

        let slice = self.readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let finished = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(ROUND_TIMEOUT),
        });
        let mapped =
            finished.is_ok() && matches!(rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
        if !mapped {
            note_trouble("the GPU stopped answering".to_string());
            return None;
        }
        let view = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&view);
        let (k, groups) = (self.k, self.groups);
        let out = (0..batch.len())
            .map(|i| {
                // The workgroups' rows, in workgroup order, in f64.
                let mut row = vec![0.0f64; k];
                for g in 0..groups {
                    let at = (i * groups + g) * k;
                    for (acc, &v) in row.iter_mut().zip(&values[at..at + k]) {
                        *acc += v as f64;
                    }
                }
                // Unpack: the criterion, the gradient, then the curvature's
                // upper triangle row by row into an n-by-n matrix.
                let mut hessian = vec![0.0f64; n * n];
                let mut t = 1 + n;
                for a in 0..n {
                    for b in a..n {
                        hessian[a * n + b] = row[t];
                        t += 1;
                    }
                }
                Sums {
                    sum_sq: row[0],
                    counted: self.pixels,
                    gradient: row[1..=n].to_vec(),
                    hessian,
                }
            })
            .collect();
        drop(view);
        self.readback.unmap();
        Some(out)
    }
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;

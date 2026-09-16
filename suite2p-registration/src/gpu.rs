// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.
//
// Ported from suite2p (https://github.com/MouseLand/suite2p), Copyright © 2023
// Howard Hughes Medical Institute, authored by Carsen Stringer and Marius
// Pachitariu, and licensed GPL-3.

//! Phase correlation on the graphics card.
//!
//! suite2p gets this from PyTorch, which has a CUDA FFT behind it. There is no
//! equivalent in Rust, so the transform is written here in WGSL — a radix-2
//! Stockham FFT.
//!
//! # What it can and cannot do
//!
//! **Power-of-two frame sizes only**, up to 1024 on a side. Radix-2 is exactly
//! the algorithm that needs powers of two, and the general case wants
//! Bluestein's — a chirp convolution, i.e. three more FFTs. The side limit is
//! the scratch each line is transformed in, which has to fit in the workgroup
//! memory every WebGPU device is required to offer. 512x512 and 256x256, which
//! is most two-photon, are fine; 1024x768 and 2048x2048 are not.
//!
//! A size this cannot take is **refused, not silently run on the CPU**. Being
//! handed the processor after asking for the device is indistinguishable from
//! the device being slow, and there would be no way to tell.
//!
//! # Why it is shaped like this
//!
//! The first version ran one frame at a time, one dispatch per butterfly stage,
//! with a fresh uniform buffer and bind group for every stage and a blocking
//! wait for every frame. It was slower than a single CPU thread. Measured, the
//! time went roughly half to overhead — seventy-odd driver allocations and a
//! device stall per frame, plus rebuilding the whole device for every batch —
//! and most of the rest to memory traffic: each stage wrote the whole plane to
//! fresh storage, so one frame streamed through the device's main memory
//! thirty-six times. The arithmetic itself was a few percent.
//!
//! So, now:
//!
//! * **one device and one context for the process**, built the first time they
//!   are asked for;
//! * **a batch of frames per submission**, sized from the device's own limits,
//!   with one wait per batch;
//! * **every buffer, uniform and bind group built once**;
//! * **a whole FFT axis per dispatch**: each workgroup is one line, whose
//!   invocations read it into shared workgroup memory together, run each
//!   stage's butterflies side by side there, and write it back once (see
//!   `fft_line` in `gpu.wgsl`);
//! * **the taper applied on the device** and **only the search window read
//!   back** — 42 kilobytes a frame at the defaults, not two megabytes.
//!
//! The peak is still taken on the host, from the window, by the same
//! [`crate::rigid::peak_of`] the CPU path uses. Tie-breaking and the correlation value come
//! from one definition, and `smooth_sigma_time` works on this path too.

use crate::masks::RefFilters;
use crate::rigid::{lcorr_for, peak_of, Shift};
use bytemuck::{Pod, Zeroable};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use wgpu::util::DeviceExt;

/// The longest side the device path takes. Two lines of that length fit the
/// 16 KiB of workgroup storage every WebGPU device must provide.
const LONGEST_SIDE: usize = 1024;

/// The most frames in one submission. Past this the batch buffers stop fitting
/// comfortably next to the viewer's own textures on a small card, and the win
/// from batching has long since flattened.
const MOST_FRAMES_PER_SUBMIT: usize = 32;

/// How much device memory the batch buffers may take between them.
const BATCH_MEMORY: u64 = 256 << 20;

/// How long one batch may take on the device before it is given up for lost.
///
/// A batch is tens of milliseconds. This is only for a device that has stopped
/// answering — a driver reset, a card pulled from under a laptop — where
/// waiting without a limit would hang the plugin's worker for good, with no way
/// for the user's Stop to reach it.
const BATCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether a frame of this size can run on the device.
pub fn size_supported(ly: usize, lx: usize) -> bool {
    ly.is_power_of_two() && lx.is_power_of_two() && ly >= 2 && lx >= 2 && ly.max(lx) <= LONGEST_SIDE
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExpandParams {
    pixels: u32,
    clip: u32,
    lo: f32,
    hi: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LineParams {
    n: u32,
    lines: u32,
    stride: u32,
    line_stride: u32,
    conjugate: f32,
    pixels: u32,
    first: u32,
    all: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PhaseParams {
    pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CropParams {
    ly: u32,
    lx: u32,
    lcorr: u32,
    width: u32,
    pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// One FFT axis pass: its pipeline input, ready to dispatch.
struct Pass {
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    lines: u32,
}

/// The output side of a batch, which depends on the search window's size and
/// so is rebuilt only when that changes.
struct Window {
    lcorr: usize,
    width: usize,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Only ever reached through `bind`; held so it lives as long as that does.
    _params: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// A device, its pipelines, and every buffer a batch of one frame size needs.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ly: usize,
    lx: usize,
    frames_per_submit: usize,

    expand: wgpu::ComputePipeline,
    line: wgpu::ComputePipeline,
    phase: wgpu::ComputePipeline,
    crop: wgpu::ComputePipeline,

    raw: wgpu::Buffer,
    data: wgpu::Buffer,
    refs: wgpu::Buffer,
    expand_params: wgpu::Buffer,

    expand_bind: wgpu::BindGroup,
    phase_bind: wgpu::BindGroup,
    /// Rows forward, columns forward, rows inverse, columns inverse.
    passes: [Pass; 4],
    window: Option<Window>,
    /// Set when a batch never came back. The context is not used again.
    lost: bool,
}

/// The one device the registration uses, opened the first time it is needed.
///
/// One for the process, shared by every context, rather than one per context.
/// Several devices opened side by side in one process, each driven from its own
/// thread, hang on the drivers this was measured against: the suite that checks
/// the device against the processor locked up every time it ran eight at once,
/// and never with one or two. A registration only ever needs one anyway.
///
/// `None`, remembered, when there is no usable adapter. A software adapter
/// counts as none: it would run the device path on the processor, slower than
/// the processor path, while the user believes they chose the card.
static DEVICE: OnceLock<Option<(wgpu::Device, wgpu::Queue)>> = OnceLock::new();

fn shared_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    DEVICE
        .get_or_init(|| {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .ok()?;
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                return None;
            }
            // Whatever the adapter can do: the batch size is derived from these,
            // and the defaults would cap it at a fraction of what the card holds.
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("suite2p-registration"),
                required_limits: adapter.limits(),
                ..Default::default()
            }))
            .ok()
        })
        .clone()
}

impl GpuContext {
    /// Open a device and build everything for `ly * lx` frames.
    ///
    /// `None` when there is no adapter, or the size is not one this can take —
    /// the caller reports that rather than falling back.
    pub fn new(ly: usize, lx: usize) -> Option<GpuContext> {
        Self::with_submit_limit(ly, lx, MOST_FRAMES_PER_SUBMIT)
    }

    /// [`new`](Self::new), with at most `limit` frames per submission. For
    /// tests, which need a batch to span several submissions without a
    /// recording long enough to make that happen on its own.
    pub fn with_submit_limit(ly: usize, lx: usize, limit: usize) -> Option<GpuContext> {
        if !size_supported(ly, lx) {
            return None;
        }
        let (device, queue) = shared_device()?;
        let limits = device.limits();
        let longest = ly.max(lx);
        if (longest as u64) * 16 > limits.max_compute_workgroup_storage_size as u64 {
            return None;
        }

        let pixels = ly * lx;
        let plane = (pixels * 8) as u64;
        let frames_per_submit = (limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            .min(BATCH_MEMORY)
            / plane)
            .clamp(1, limit.max(1) as u64) as usize;

        // One workgroup is one line; its lanes share the line's butterflies.
        // As many lanes as a stage has butterflies, up to what every device
        // must allow in a workgroup, and the butterflies split evenly past that.
        let most_lanes = (limits.max_compute_invocations_per_workgroup as usize).clamp(1, 256);
        let lanes = (longest / 2).clamp(1, most_lanes);
        let per = (longest / 2).div_ceil(lanes).max(1);
        let source = include_str!("gpu.wgsl")
            .replace("__TWICE_LONGEST_LINE__", &(2 * longest).to_string())
            .replace("__LONGEST_LINE__", &longest.to_string())
            .replace("__LANES__", &lanes.to_string())
            .replace("__PER__", &per.to_string());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("registration"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let expand = pipeline("expand");
        let line = pipeline("fft_line");
        let phase = pipeline("phase");
        let crop = pipeline("crop");

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let batch = frames_per_submit as u64;
        let raw = buffer("raw", batch * pixels as u64 * 4, storage);
        let data = buffer("data", batch * plane, wgpu::BufferUsages::STORAGE);
        let refs = buffer("refs", pixels as u64 * 16, storage);

        // exp(-i*pi*j/s) for every stage half-size s < longest and j < s, stage
        // s starting at s - 1. The same for rows and columns: a twiddle depends
        // on the stage, not on the length of the line.
        let mut twiddles: Vec<[f32; 2]> = Vec::with_capacity(longest);
        let mut span = 1usize;
        while span < longest {
            for j in 0..span {
                let angle = std::f64::consts::PI * j as f64 / span as f64;
                twiddles.push([angle.cos() as f32, -angle.sin() as f32]);
            }
            span *= 2;
        }
        let twiddles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddles"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniform = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        fn entries<'a>(pairs: &[(u32, &'a wgpu::Buffer)]) -> Vec<wgpu::BindGroupEntry<'a>> {
            pairs
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect()
        }

        let expand_params = uniform(
            "expand",
            bytemuck::bytes_of(&ExpandParams {
                pixels: pixels as u32,
                clip: 0,
                lo: 0.0,
                hi: 0.0,
            }),
        );
        let expand_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand"),
            layout: &expand.get_bind_group_layout(0),
            entries: &entries(&[(0, &data), (1, &refs), (2, &raw), (3, &expand_params)]),
        });

        let phase_params = uniform(
            "phase",
            bytemuck::bytes_of(&PhaseParams {
                pixels: pixels as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );
        let phase_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase"),
            layout: &phase.get_bind_group_layout(0),
            entries: &entries(&[(0, &data), (1, &refs), (6, &phase_params)]),
        });

        // Rows are lx long, ly of them, a sample apart; columns ly long, lx of
        // them, a row apart. Every pass starts out covering every line; the last
        // is narrowed to the search window by `window_for`.
        let pass = |rows: bool, inverse: bool| -> Pass {
            let (n, lines, stride, line_stride) = if rows {
                (lx, ly, 1, lx)
            } else {
                (ly, lx, lx, 1)
            };
            let params = uniform(
                "line",
                bytemuck::bytes_of(&LineParams {
                    n: n as u32,
                    lines: lines as u32,
                    stride: stride as u32,
                    line_stride: line_stride as u32,
                    conjugate: if inverse { -1.0 } else { 1.0 },
                    pixels: pixels as u32,
                    first: 0,
                    all: lines as u32,
                }),
            );
            Pass {
                bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("line"),
                    layout: &line.get_bind_group_layout(0),
                    entries: &entries(&[(0, &data), (4, &params), (5, &twiddles)]),
                }),
                params,
                lines: lines as u32,
            }
        };
        // The order the CPU path runs them in, so the two accumulate their
        // rounding the same way.
        let passes = [
            pass(true, false),
            pass(false, false),
            pass(true, true),
            pass(false, true),
        ];

        Some(GpuContext {
            device,
            queue,
            ly,
            lx,
            frames_per_submit,
            expand,
            line,
            phase,
            crop,
            raw,
            data,
            refs,
            expand_params,
            expand_bind,
            phase_bind,
            passes,
            window: None,
            lost: false,
        })
    }

    /// The frame size this context was built for.
    pub fn size(&self) -> (usize, usize) {
        (self.ly, self.lx)
    }

    /// How many frames go to the device in one submission.
    pub fn frames_per_submit(&self) -> usize {
        self.frames_per_submit
    }

    /// Upload a reference: its prepared spectrum, the taper, and the clip range.
    ///
    /// Once per reference, not per frame — about four megabytes at 512x512.
    pub fn set_reference(&mut self, filters: &RefFilters) {
        let pixels: Vec<[f32; 4]> = filters
            .cf_ref
            .iter()
            .zip(&filters.mask_mul)
            .zip(&filters.mask_offset)
            .map(|((c, &m), &o)| [c.re, c.im, m, o])
            .collect();
        self.queue
            .write_buffer(&self.refs, 0, bytemuck::cast_slice(&pixels));
        let (clip, lo, hi) = match filters.clip {
            Some((lo, hi)) => (1, lo, hi),
            None => (0, 0.0, 0.0),
        };
        self.queue.write_buffer(
            &self.expand_params,
            0,
            bytemuck::bytes_of(&ExpandParams {
                pixels: (self.ly * self.lx) as u32,
                clip,
                lo,
                hi,
            }),
        );
    }

    /// The output buffers for a window of half-width `lcorr`, built on first
    /// use and whenever the window grows.
    fn window_for(&mut self, lcorr: usize) -> &Window {
        if self.window.as_ref().is_none_or(|w| w.lcorr != lcorr) {
            let width = 2 * lcorr + 1;
            let bytes = (self.frames_per_submit * width * width * 4) as u64;
            let out = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("windows"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("windows readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("crop"),
                    contents: bytemuck::bytes_of(&CropParams {
                        ly: self.ly as u32,
                        lx: self.lx as u32,
                        lcorr: lcorr as u32,
                        width: width as u32,
                        pixels: (self.ly * self.lx) as u32,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crop"),
                layout: &self.crop.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: out.as_entire_binding(),
                    },
                ],
            });
            // The crop reads `width` columns, wrapping round from `lx - lcorr`,
            // and the columns' inverse is the last transform before it — so
            // that pass need only touch those. Every other column's inverse
            // would be computed and never read: four fifths of a pass at the
            // default window, and it is the strided pass, the dearer of the two.
            let columns = width.min(self.lx);
            let last = &mut self.passes[3];
            last.lines = columns as u32;
            self.queue.write_buffer(
                &last.params,
                0,
                bytemuck::bytes_of(&LineParams {
                    n: self.ly as u32,
                    lines: columns as u32,
                    stride: self.lx as u32,
                    line_stride: 1,
                    conjugate: -1.0,
                    pixels: (self.ly * self.lx) as u32,
                    first: ((self.lx - lcorr % self.lx) % self.lx) as u32,
                    all: self.lx as u32,
                }),
            );
            self.window = Some(Window {
                lcorr,
                width,
                out,
                readback,
                _params: params,
                bind,
            });
        }
        self.window.as_ref().expect("built just above")
    }

    /// The correlation window of every frame against the uploaded reference.
    ///
    /// The same `(2*lcorr+1)^2` row-major windows [`crate::rigid::correlation_map`]
    /// returns, one per frame, in order. Call [`set_reference`](Self::set_reference)
    /// first.
    pub fn maps(&mut self, frames: &[Vec<f32>], max_shift: f64) -> Vec<Vec<f32>> {
        let (ly, lx) = (self.ly, self.lx);
        let pixels = ly * lx;
        let lcorr = lcorr_for(ly, lx, max_shift);
        let per_submit = self.frames_per_submit;
        let width = self.window_for(lcorr).width;
        let area = width * width;

        let mut out = Vec::with_capacity(frames.len());
        for chunk in frames.chunks(per_submit) {
            let count = chunk.len();
            if self.lost {
                // Flat maps peak at no-shift with zero correlation, which the
                // bad-frame pass flags: a registration that finishes and says
                // what went wrong, rather than one that never finishes.
                out.extend((0..count).map(|_| vec![0.0; area]));
                continue;
            }
            // Straight into the queue's staging memory: one copy per frame, and
            // no allocation of our own.
            if let Some(size) = wgpu::BufferSize::new((count * pixels * 4) as u64) {
                if let Some(mut view) = self.queue.write_buffer_with(&self.raw, 0, size) {
                    for (i, frame) in chunk.iter().enumerate() {
                        view.slice(i * pixels * 4..(i + 1) * pixels * 4)
                            .copy_from_slice(bytemuck::cast_slice(frame));
                    }
                }
            }

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let window = self.window.as_ref().expect("built above");
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("correlate"),
                    timestamp_writes: None,
                });
                let across = (pixels as u32).div_ceil(256);
                pass.set_pipeline(&self.expand);
                pass.set_bind_group(0, &self.expand_bind, &[]);
                pass.dispatch_workgroups(across, count as u32, 1);

                pass.set_pipeline(&self.line);
                for axis in &self.passes[..2] {
                    pass.set_bind_group(0, &axis.bind, &[]);
                    pass.dispatch_workgroups(axis.lines, count as u32, 1);
                }

                pass.set_pipeline(&self.phase);
                pass.set_bind_group(0, &self.phase_bind, &[]);
                pass.dispatch_workgroups(across, count as u32, 1);

                pass.set_pipeline(&self.line);
                for axis in &self.passes[2..] {
                    pass.set_bind_group(0, &axis.bind, &[]);
                    pass.dispatch_workgroups(axis.lines, count as u32, 1);
                }

                pass.set_pipeline(&self.crop);
                pass.set_bind_group(0, &window.bind, &[]);
                pass.dispatch_workgroups((area as u32).div_ceil(256), count as u32, 1);
            }
            let bytes = (count * area * 4) as u64;
            encoder.copy_buffer_to_buffer(&window.out, 0, &window.readback, 0, bytes);
            let submitted = self.queue.submit(Some(encoder.finish()));

            let slice = window.readback.slice(..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            // One wait per batch, for this batch's submission.
            let finished = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submitted),
                timeout: Some(BATCH_TIMEOUT),
            });
            let mapped =
                finished.is_ok() && matches!(rx.recv_timeout(Duration::from_secs(1)), Ok(Ok(())));
            if mapped {
                let view = slice.get_mapped_range();
                let values: &[f32] = bytemuck::cast_slice(&view);
                out.extend(values.chunks_exact(area).map(<[f32]>::to_vec));
                drop(view);
                window.readback.unmap();
            } else {
                // The batch never came back. Its buffer may still be waiting on
                // a mapping that will not arrive, so this context is finished;
                // `maps_of` builds a new one next time.
                self.lost = true;
                out.extend((0..count).map(|_| vec![0.0; area]));
            }
        }
        out
    }

    /// One frame's shift, as the CPU path reports it. Uploads `filters` first.
    pub fn shift_of(&mut self, frame: &[f32], filters: &RefFilters, max_shift: f64) -> Shift {
        self.set_reference(filters);
        let lcorr = lcorr_for(self.ly, self.lx, max_shift);
        let maps = self.maps(std::slice::from_ref(&frame.to_vec()), max_shift);
        peak_of(&maps[0], lcorr)
    }
}

/// The process's context, kept between batches.
///
/// Building one is a pipeline compile and a few hundred milliseconds of driver
/// work, and a registration asks for correlations dozens of times — once per
/// reference pass and once per batch of the recording. Rebuilding it every time
/// cost more than the correlations did.
static RESIDENT: Mutex<Option<GpuContext>> = Mutex::new(None);

/// Correlation windows for `frames` against `filters`, on the resident device.
///
/// `None` when no device can be opened for this size, which the caller treats
/// as the adapter having gone away between the availability check and the run.
pub fn maps_of(
    ly: usize,
    lx: usize,
    filters: &RefFilters,
    frames: &[Vec<f32>],
    max_shift: f64,
) -> Option<Vec<Vec<f32>>> {
    let mut slot = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().is_none_or(|c| c.size() != (ly, lx) || c.lost) {
        // The old context goes first, so two sets of batch buffers are never
        // alive at once.
        *slot = None;
        *slot = GpuContext::new(ly, lx);
    }
    let context = slot.as_mut()?;
    context.set_reference(filters);
    Some(context.maps(frames, max_shift))
}

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
//! Stockham FFT, run as one compute dispatch per butterfly stage.
//!
//! # What it can and cannot do
//!
//! **Power-of-two frame sizes only.** Radix-2 is exactly the algorithm that
//! needs them, and the general case wants Bluestein's — which is a chirp
//! convolution, i.e. three more FFTs, and a substantial piece of work on its
//! own. 512x512 and 256x256, which is most two-photon, are fine; 1024x768 is
//! not.
//!
//! A size this cannot take is **refused, not silently run on the CPU**. Being
//! handed the processor after asking for the device is indistinguishable from
//! the device being slow, and there would be no way to tell.
//!
//! # Why Stockham
//!
//! Cooley-Tukey works in place and then needs a bit-reversal permutation.
//! In-place across a compute dispatch means workgroups racing each other for
//! the same elements, and there is no barrier that spans a dispatch. Stockham
//! writes each stage to a second buffer instead — two buffers ping-ponged, no
//! permutation, and no hazard.

use crate::masks::RefFilters;
use crate::rigid::{lcorr_for, peak_of, Shift};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Whether a frame of this size can run on the device.
pub fn size_supported(ly: usize, lx: usize) -> bool {
    ly.is_power_of_two() && lx.is_power_of_two() && ly >= 2 && lx >= 2
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    batch: u32,
    span: u32,
    inverse: u32,
    stride: u32,
    batch_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CorrParams {
    len: u32,
    scale: u32,
    _pad0: u32,
    _pad1: u32,
}

/// A device, its pipelines, and the buffers for one frame size.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fft: wgpu::ComputePipeline,
    phase: wgpu::ComputePipeline,
    fft_layout: wgpu::BindGroupLayout,
    phase_layout: wgpu::BindGroupLayout,
    ly: usize,
    lx: usize,
    /// Ping-pong pair for the transform stages.
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    reference: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl GpuContext {
    /// Open a device and build the pipelines for `ly * lx` frames.
    ///
    /// `None` when there is no adapter, or the size is not one radix-2 can
    /// take — the caller reports that rather than falling back.
    pub fn new(ly: usize, lx: usize) -> Option<GpuContext> {
        if !size_supported(ly, lx) {
            return None;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("suite2p-registration"),
            ..Default::default()
        }))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("registration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });

        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |b: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };

        let fft_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(false)),
                entry(2, uniform),
            ],
        });
        let phase_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phase"),
            entries: &[
                entry(0, storage(false)),
                entry(1, storage(true)),
                entry(2, uniform),
            ],
        });

        let pipeline = |name: &str, layout: &wgpu::BindGroupLayout, entry_point: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fft = pipeline("fft_stage", &fft_layout, "fft_stage");
        let phase = pipeline("phase_multiply", &phase_layout, "phase_multiply");

        let bytes = (ly * lx * 8) as u64;
        let buf = |label: &str, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let st = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        Some(GpuContext {
            a: buf("a", st),
            b: buf("b", st),
            reference: buf("reference", st),
            readback: buf(
                "readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            device,
            queue,
            fft,
            phase,
            fft_layout,
            phase_layout,
            ly,
            lx,
        })
    }

    /// Upload the reference's prepared spectrum. Done once per reference, not
    /// once per frame.
    pub fn set_reference(&self, filters: &RefFilters) {
        let flat: Vec<[f32; 2]> = filters.cf_ref.iter().map(|c| [c.re, c.im]).collect();
        self.queue
            .write_buffer(&self.reference, 0, bytemuck::cast_slice(&flat));
    }

    /// Run one axis of the transform over the whole plane.
    ///
    /// `stride`/`batch_stride` pick rows or columns out of the same buffer, so
    /// the column pass needs no transpose.
    #[allow(clippy::too_many_arguments)]
    fn transform_axis(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        batch: usize,
        stride: usize,
        batch_stride: usize,
        inverse: bool,
        src_is_a: &mut bool,
    ) {
        let mut span = 1usize;
        while span < n {
            let params = Params {
                n: n as u32,
                batch: batch as u32,
                span: span as u32,
                inverse: u32::from(inverse),
                stride: stride as u32,
                batch_stride: batch_stride as u32,
                _pad0: 0,
                _pad1: 0,
            };
            let ub = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let (src, dst) = if *src_is_a {
                (&self.a, &self.b)
            } else {
                (&self.b, &self.a)
            };
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft"),
                layout: &self.fft_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ub.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fft"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.fft);
                pass.set_bind_group(0, &bg, &[]);
                let threads = (n / 2 * batch) as u32;
                pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
            }
            *src_is_a = !*src_is_a;
            span *= 2;
        }
    }

    /// Correlate one frame against the loaded reference and read the plane back.
    pub fn correlate(&self, frame: &[f32], filters: &RefFilters) -> Vec<f32> {
        let (ly, lx) = (self.ly, self.lx);
        let n = ly * lx;

        // Masked and offset on the CPU — it is one pass over the plane and
        // would need its own kernel and upload either way.
        let staged: Vec<[f32; 2]> = frame
            .iter()
            .zip(&filters.mask_mul)
            .zip(&filters.mask_offset)
            .map(|((&v, &m), &o)| {
                let v = match filters.clip {
                    Some((lo, hi)) => v.clamp(lo, hi),
                    None => v,
                };
                [v * m + o, 0.0]
            })
            .collect();
        self.queue
            .write_buffer(&self.a, 0, bytemuck::cast_slice(&staged));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut src_is_a = true;
        // Rows, then columns — the same separable transform the CPU path does.
        self.transform_axis(&mut encoder, lx, ly, 1, lx, false, &mut src_is_a);
        self.transform_axis(&mut encoder, ly, lx, lx, 1, false, &mut src_is_a);

        // The phase multiply reads and writes whichever buffer the stages left
        // the data in.
        let current = if src_is_a { &self.a } else { &self.b };
        let cp = CorrParams {
            len: n as u32,
            scale: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let ub = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("corr"),
                contents: bytemuck::bytes_of(&cp),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase"),
            layout: &self.phase_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.reference.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ub.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }

        self.transform_axis(&mut encoder, lx, ly, 1, lx, true, &mut src_is_a);
        self.transform_axis(&mut encoder, ly, lx, lx, 1, true, &mut src_is_a);

        let final_buf = if src_is_a { &self.a } else { &self.b };
        encoder.copy_buffer_to_buffer(final_buf, 0, &self.readback, 0, (n * 8) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Block until the queue has drained and the readback is mapped.
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let complex: &[[f32; 2]] = bytemuck::cast_slice(&data);
        // The inverse's `1/N`, applied here rather than in a fourth dispatch.
        let scale = 1.0 / n as f32;
        let out: Vec<f32> = complex.iter().map(|c| c[0] * scale).collect();
        drop(data);
        self.readback.unmap();
        out
    }

    /// The correlation window and its peak, as the CPU path reports them.
    pub fn shift_of(&self, frame: &[f32], filters: &RefFilters, max_shift: f64) -> Shift {
        let (ly, lx) = (self.ly, self.lx);
        let plane = self.correlate(frame, filters);
        let lcorr = lcorr_for(ly, lx, max_shift);
        let n = 2 * lcorr + 1;
        let mut cc = vec![0.0f32; n * n];
        for iy in 0..n {
            let y = (iy + ly - lcorr) % ly;
            for ix in 0..n {
                let x = (ix + lx - lcorr) % lx;
                cc[iy * n + ix] = plane[y * lx + x];
            }
        }
        peak_of(&cc, lcorr)
    }
}

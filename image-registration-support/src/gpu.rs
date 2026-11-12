// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! One GPU device for every registration in the process, and the habits
//! that keep a shared card alive.
//!
//! Both registration crates used to keep a device slot of their own. A host
//! that handed its renderer's device to one and not the other, or a library
//! user calling both, got a second device beside the first — and two devices
//! side by side in one process, each driven from its own thread, hang on the
//! drivers this workspace was measured against; on a card whose memory the two
//! have overrun (a Quadro P620, 2 GB, Vulkan) both deadlocked and the system
//! reset the GPU. So the slot is here, once: [`use_device`] given the
//! renderer's device sets it for every registration, and without one the first
//! that needs a device opens it for all of them.
//!
//! The rest is how work goes onto a device that is not the registration's
//! alone. Every allocation and submission runs inside error scopes
//! ([`scoped`]), so a card with no room left fails a batch instead of panicking
//! the host; buffers are made unmapped and filled by the queue ([`filled`]);
//! uploads go a slab at a time ([`UPLOAD_SLAB`]) so wgpu's own staging stays
//! small, since running out there loses the device.

use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

/// What one batch of a registration may take on the device: enough for its
/// buffers to be worth the round trip, small enough to leave the renderer
/// sharing the card its textures.
pub const BATCH_MEMORY: u64 = 256 << 20;

/// The most bytes an upload stages before it submits and waits: a write is
/// held in memory of the queue's own until the submission that carries it is
/// done, and a staging allocation that fails can lose the device.
pub const UPLOAD_SLAB: usize = 32 << 20;

/// The device every registration runs on, with its queue and the adapter's
/// name.
#[derive(Clone, Debug)]
pub struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
}

/// The host's device, given with [`use_device`], or one opened the first time
/// one is needed. `None`, remembered, when there is no usable adapter: a
/// software adapter counts as none, since it would run the device path on the
/// processor, slower than the processor path, while the user believes they
/// chose the card.
///
/// Not in a browser build, where there is never a device to keep (see
/// [`device`]) and wgpu's browser objects cannot sit in a `static`.
#[cfg(not(target_arch = "wasm32"))]
static DEVICE: OnceLock<Option<Device>> = OnceLock::new();

/// Run every registration on `device` instead of on one of their own: the
/// host's, typically the one its renderer draws with, which then carries the
/// registrations' work between its own frames.
///
/// A host that draws with wgpu should call this before the first GPU run (or
/// availability check): see the module docs for what a second device costs.
/// Takes effect only before a device is first asked for, and says whether it
/// did. A software adapter counts as no device. The device should have the
/// adapter's own limits (`required_limits: adapter.limits()`): batch sizes are
/// derived from them.
#[cfg(not(target_arch = "wasm32"))]
pub fn use_device(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::AdapterInfo) -> bool {
    let shared = (adapter.device_type != wgpu::DeviceType::Cpu).then(|| Device {
        device,
        queue,
        name: adapter.name.clone(),
    });
    DEVICE.set(shared).is_ok()
}

/// The device registrations run on, opened the first time it is asked for
/// when no host gave one; `None` when there is none to use.
///
/// Always `None` in a browser: requesting an adapter is asynchronous there and
/// resolves through the page's event loop, which a blocking wait would block.
/// The registration crates' backends say so before a run gets this far.
pub fn device() -> Option<Device> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
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
                // Whatever the adapter can do: batch sizes are derived from
                // these, and the defaults would cap them well below the card.
                let (device, queue) =
                    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                        label: Some("image-registration"),
                        required_limits: adapter.limits(),
                        ..Default::default()
                    }))
                    .ok()?;
                Some(Device {
                    device,
                    queue,
                    name: info.name,
                })
            })
            .clone()
    }
}

/// Whether this machine has a device the GPU backends can use. Opens it the
/// first time it is asked, which takes a moment; the answer is remembered, and
/// the device is the one a run then uses.
pub fn available() -> bool {
    device().is_some()
}

/// Run `work` with the device's out-of-memory and validation errors caught,
/// and say which was raised, if one was.
///
/// Without a scope wgpu hands an error to the device's handler, which by
/// default panics — on a device shared with a renderer, taking the host with
/// it. Scopes are per thread in wgpu, so these catch this thread's errors only.
pub fn scoped<T>(device: &wgpu::Device, work: impl FnOnce() -> T) -> Result<T, String> {
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

/// A buffer of `contents`, for inside a [`scoped`] call: made unmapped and
/// filled by the queue. `create_buffer_init` maps the buffer at creation
/// instead, and wgpu treats mapping a buffer that could not be made as fatal,
/// past any scope; this way a buffer there was no room for is an error the
/// scope returns.
pub fn filled(
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

/// Why a registration's device work last failed, until it is taken: no room
/// for a batch's buffers, or a batch that never came back. One per
/// registration crate, so one crate's report is not taken by the other's.
pub struct Trouble(Mutex<Option<String>>);

impl Trouble {
    pub const fn new() -> Self {
        Trouble(Mutex::new(None))
    }

    /// Record why.
    pub fn note(&self, why: String) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
    }

    /// What was recorded since this was last asked, if anything.
    pub fn take(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

impl Default for Trouble {
    fn default() -> Self {
        Self::new()
    }
}

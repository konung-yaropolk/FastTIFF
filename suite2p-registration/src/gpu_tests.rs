//! The device path's own plumbing. Whether it registers like the processor
//! path is `tests/gpu_agrees.rs`.

use super::*;

/// What would be a panic in the device's error handler comes back from a
/// scope as a value: on a device shared with a renderer, the difference
/// between a failed batch and a failed application.
#[test]
fn an_error_in_a_scope_is_returned_not_raised() {
    let Some((device, _)) = shared_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let too_big = device.limits().max_buffer_size.saturating_add(4);
    let made = scoped(&device, || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("too big"),
            size: too_big,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    });
    assert!(made.is_err());
}

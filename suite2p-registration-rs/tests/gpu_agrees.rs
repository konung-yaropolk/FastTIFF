//! The GPU path must agree with the CPU path, or the backend selector is a
//! correctness setting rather than a speed one.
//!
//! Skips rather than fails when there is no adapter: whether a machine has a
//! usable GPU is a property of the machine, and a red test on a headless CI box
//! would say nothing about the code.

#![cfg(feature = "gpu")]

use suite2p_registration::fft::Fft2;
use suite2p_registration::gpu::{size_supported, GpuContext};
use suite2p_registration::masks::reference_filters;
use suite2p_registration::rigid::{correlation_map, lcorr_for, peak_of, phase_correlate};

fn blobs(ly: usize, lx: usize) -> Vec<f32> {
    let mut f = vec![10.0f32; ly * lx];
    for (cy, cx, amp) in [
        (20usize, 24usize, 200.0f32),
        (40, 44, 150.0),
        (28, 12, 120.0),
    ] {
        for y in 0..ly {
            for x in 0..lx {
                let d2 = ((y as f32 - cy as f32).powi(2) + (x as f32 - cx as f32).powi(2)) / 8.0;
                f[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    f
}

fn shift(f: &[f32], ly: usize, lx: usize, dy: i32, dx: i32) -> Vec<f32> {
    suite2p_registration::shift_frame(f, ly, lx, dy, dx)
}

#[test]
fn only_power_of_two_sizes_are_accepted() {
    assert!(size_supported(512, 512));
    assert!(size_supported(256, 128));
    // Radix-2 cannot take these, and the caller is told rather than quietly
    // given the CPU.
    assert!(!size_supported(1024, 768));
    assert!(!size_supported(300, 256));
    assert!(!size_supported(1, 1));
}

#[test]
fn the_gpu_finds_the_same_shifts_as_the_cpu() {
    let (ly, lx) = (64, 64);
    let Some(gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);
    gpu.set_reference(&filters);

    for (dy, dx) in [(0i32, 0i32), (3, 0), (0, -4), (-5, 2), (2, 6)] {
        let frame = shift(&reference, ly, lx, -dy, -dx);
        let cpu = phase_correlate(&mut fft, &filters, &frame, 0.2);
        let dev = gpu.shift_of(&frame, &filters, 0.2);
        assert_eq!(
            (cpu.dy, cpu.dx),
            (dev.dy, dev.dx),
            "GPU and CPU disagreed for a frame displaced by ({dy},{dx})"
        );
        // And it is the right answer, not merely the same wrong one.
        assert_eq!((dev.dy, dev.dx), (dy, dx));
    }
}

/// Not just the peak — the whole correlation plane has to match, or the two
/// agree by luck on these frames and diverge on real data.
#[test]
fn the_correlation_planes_match() {
    let (ly, lx) = (64, 64);
    let Some(gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);
    gpu.set_reference(&filters);

    let frame = shift(&reference, ly, lx, -3, 4);
    let max_shift = 0.2;
    let lcorr = lcorr_for(ly, lx, max_shift);
    let cpu = correlation_map(&mut fft, &filters, &frame, max_shift, None);

    let plane = gpu.correlate(&frame, &filters);
    let n = 2 * lcorr + 1;
    let mut dev = vec![0.0f32; n * n];
    for iy in 0..n {
        let y = (iy + ly - lcorr) % ly;
        for ix in 0..n {
            let x = (ix + lx - lcorr) % lx;
            dev[iy * n + ix] = plane[y * lx + x];
        }
    }

    // f32 on two different pieces of hardware will not be bit-identical; what
    // matters is that the surface is the same shape.
    let scale = cpu.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    for (i, (a, b)) in cpu.iter().zip(&dev).enumerate() {
        assert!(
            (a - b).abs() < scale * 1e-3,
            "position {i}: cpu {a}, gpu {b} (peak {scale})"
        );
    }
    assert_eq!(peak_of(&cpu, lcorr).dy, peak_of(&dev, lcorr).dy);
}

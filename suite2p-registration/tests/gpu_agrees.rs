//! The GPU path must agree with the CPU path, or the backend selector is a
//! correctness setting rather than a speed one.
//!
//! Skips rather than fails when there is no adapter: whether a machine has a
//! usable GPU is a property of the machine, and a red test on a headless CI box
//! would say nothing about the code.

#![cfg(feature = "gpu")]

use suite2p_registration::fft::Fft2;
use suite2p_registration::gpu::{size_supported, GpuContext};
use suite2p_registration::masks::{reference_filters, reference_filters_normed, RefFilters};
use suite2p_registration::pipeline::measure_batch;
use suite2p_registration::rigid::{correlation_map, lcorr_for, peak_of, phase_correlate};
use suite2p_registration::{Backend, Settings};

/// Blobs scattered over a frame of any size, so there is structure to lock onto
/// in both axes whatever the aspect ratio.
fn blobs(ly: usize, lx: usize) -> Vec<f32> {
    let mut f = vec![10.0f32; ly * lx];
    let mut state = 0x9e37_79b9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..(ly * lx / 256).clamp(4, 60) {
        let cy = (next() as usize % ly) as f32;
        let cx = (next() as usize % lx) as f32;
        let amp = 100.0 + (next() % 200) as f32;
        for y in 0..ly {
            for x in 0..lx {
                let d2 = ((y as f32 - cy).powi(2) + (x as f32 - cx).powi(2)) / 8.0;
                if d2 < 40.0 {
                    f[y * lx + x] += amp * (-d2).exp();
                }
            }
        }
    }
    f
}

fn shift(f: &[f32], ly: usize, lx: usize, dy: i32, dx: i32) -> Vec<f32> {
    suite2p_registration::shift_frame(f, ly, lx, dy, dx)
}

/// Assert two correlation windows are the same surface.
///
/// f32 on two different pieces of hardware is not bit-identical; what matters
/// is that every position agrees to a small fraction of the peak, and that the
/// peak lands in the same place.
fn assert_same_surface(cpu: &[f32], dev: &[f32], lcorr: usize, what: &str) {
    assert_eq!(cpu.len(), dev.len(), "{what}: window sizes differ");
    let scale = cpu.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(scale > 0.0, "{what}: the CPU window is empty");
    for (i, (a, b)) in cpu.iter().zip(dev).enumerate() {
        assert!(
            (a - b).abs() < scale * 1e-3,
            "{what}: position {i}: cpu {a}, gpu {b} (peak {scale})"
        );
    }
    let (c, d) = (peak_of(cpu, lcorr), peak_of(dev, lcorr));
    assert_eq!((c.dy, c.dx), (d.dy, d.dx), "{what}: peaks differ");
}

/// CPU windows for `frames`, the reference implementation.
fn cpu_maps(
    ly: usize,
    lx: usize,
    filters: &RefFilters,
    frames: &[Vec<f32>],
    max_shift: f64,
) -> Vec<Vec<f32>> {
    let mut fft = Fft2::new(ly, lx);
    frames
        .iter()
        .map(|f| correlation_map(&mut fft, filters, f, max_shift, filters.clip))
        .collect()
}

#[test]
fn only_supported_sizes_are_accepted() {
    assert!(size_supported(512, 512));
    assert!(size_supported(256, 128));
    assert!(size_supported(1024, 1024));
    // Radix-2 cannot take these, and the caller is told rather than quietly
    // given the CPU.
    assert!(!size_supported(1024, 768));
    assert!(!size_supported(300, 256));
    assert!(!size_supported(1, 1));
    // Two lines of 2048 do not fit the workgroup memory every device offers.
    assert!(!size_supported(2048, 2048));
}

#[test]
fn the_gpu_finds_the_same_shifts_as_the_cpu() {
    let (ly, lx) = (64, 64);
    let Some(mut gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);

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

/// Not just the peak — the whole window has to match, or the two agree by luck
/// on these frames and diverge on real data.
#[test]
fn the_correlation_windows_match() {
    let (ly, lx) = (64, 64);
    let Some(mut gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);
    let frames = vec![shift(&reference, ly, lx, -3, 4)];
    let max_shift = 0.2;

    gpu.set_reference(&filters);
    let dev = gpu.maps(&frames, max_shift);
    let cpu = cpu_maps(ly, lx, &filters, &frames, max_shift);
    assert_same_surface(&cpu[0], &dev[0], lcorr_for(ly, lx, max_shift), "64x64");
}

/// Frames that are not square.
///
/// The case a mix-up between the axes hides in. Give the column pass a line
/// length of `lx` and a line count of `ly` — the wrong way round — and every
/// square test still passes, because on a square frame the two are equal. This
/// is the one that fails. Both orientations, so neither axis is the easy one.
#[test]
fn non_square_frames_match() {
    for (ly, lx) in [(64usize, 128usize), (128, 32)] {
        let Some(mut gpu) = GpuContext::new(ly, lx) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let reference = blobs(ly, lx);
        let mut fft = Fft2::new(ly, lx);
        let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);
        let frames: Vec<Vec<f32>> = [(2i32, -3i32), (-4, 1), (0, 5)]
            .iter()
            .map(|&(dy, dx)| shift(&reference, ly, lx, -dy, -dx))
            .collect();
        let max_shift = 0.2;

        gpu.set_reference(&filters);
        let dev = gpu.maps(&frames, max_shift);
        let cpu = cpu_maps(ly, lx, &filters, &frames, max_shift);
        for (i, (c, d)) in cpu.iter().zip(&dev).enumerate() {
            assert_same_surface(
                c,
                d,
                lcorr_for(ly, lx, max_shift),
                &format!("{ly}x{lx} frame {i}"),
            );
        }
    }
}

/// A batch larger than one submission comes back whole, in order, and right.
///
/// Five-and-a-bit submissions of three: the last is a partial one, which is the
/// one that would read stale windows or drop frames off the end.
#[test]
fn a_batch_spanning_several_submissions_comes_back_in_order() {
    let (ly, lx) = (64, 64);
    let Some(mut gpu) = GpuContext::with_submit_limit(ly, lx, 3) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert_eq!(gpu.frames_per_submit(), 3);
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters(&mut fft, &reference, ly, lx, 5.0, 1.15);
    let path: Vec<(i32, i32)> = (0..16).map(|t| ((t % 7) - 3, (t % 5) - 2)).collect();
    let frames: Vec<Vec<f32>> = path
        .iter()
        .map(|&(dy, dx)| shift(&reference, ly, lx, -dy, -dx))
        .collect();
    let max_shift = 0.2;
    let lcorr = lcorr_for(ly, lx, max_shift);

    gpu.set_reference(&filters);
    let dev = gpu.maps(&frames, max_shift);
    assert_eq!(dev.len(), frames.len(), "frames were lost or invented");
    let cpu = cpu_maps(ly, lx, &filters, &frames, max_shift);
    for (t, ((c, d), &(dy, dx))) in cpu.iter().zip(&dev).zip(&path).enumerate() {
        assert_same_surface(c, d, lcorr, &format!("frame {t}"));
        let found = peak_of(d, lcorr);
        assert_eq!(
            (found.dy, found.dx),
            (dy, dx),
            "frame {t} out of order or wrong"
        );
    }
}

/// `norm_frames` clips on the device exactly as it does on the processor.
#[test]
fn clipping_matches() {
    let (ly, lx) = (64, 64);
    let Some(mut gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters_normed(&mut fft, &reference, ly, lx, 5.0, 1.15, true);
    assert!(
        filters.clip.is_some(),
        "the fixture should exercise clipping"
    );
    // A hot pixel far outside the reference's range, which clipping must tame
    // identically on both.
    let mut frame = shift(&reference, ly, lx, -2, 3);
    frame[10 * lx + 50] = 1.0e6;
    let frames = vec![frame];
    let max_shift = 0.2;

    gpu.set_reference(&filters);
    let dev = gpu.maps(&frames, max_shift);
    let cpu = cpu_maps(ly, lx, &filters, &frames, max_shift);
    assert_same_surface(&cpu[0], &dev[0], lcorr_for(ly, lx, max_shift), "clipped");
}

/// The size two-photon recordings actually are, not just a toy one.
#[test]
fn a_full_size_frame_matches() {
    let (ly, lx) = (512, 512);
    let Some(mut gpu) = GpuContext::new(ly, lx) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters_normed(&mut fft, &reference, ly, lx, 50.0, 1.15, true);
    let frames = vec![
        shift(&reference, ly, lx, -7, 11),
        shift(&reference, ly, lx, 20, -5),
    ];
    let max_shift = 0.1;

    gpu.set_reference(&filters);
    let dev = gpu.maps(&frames, max_shift);
    let cpu = cpu_maps(ly, lx, &filters, &frames, max_shift);
    for (i, (c, d)) in cpu.iter().zip(&dev).enumerate() {
        assert_same_surface(
            c,
            d,
            lcorr_for(ly, lx, max_shift),
            &format!("512x512 frame {i}"),
        );
    }
}

/// Through `measure_batch`, the backends find the same shifts — with temporal
/// smoothing on, which the device path used to skip.
#[test]
fn the_backends_agree_through_measure_batch_with_smoothing() {
    let (ly, lx) = (64, 64);
    if GpuContext::new(ly, lx).is_none() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let reference = blobs(ly, lx);
    let mut fft = Fft2::new(ly, lx);
    let filters = reference_filters_normed(&mut fft, &reference, ly, lx, 5.0, 1.15, true);
    let frames: Vec<Vec<f32>> = (0..40)
        .map(|t: i32| shift(&reference, ly, lx, -(t / 10), t % 4 - 2))
        .collect();

    let run = |backend| {
        measure_batch(
            ly,
            lx,
            &filters,
            &frames,
            &Settings {
                backend,
                smooth_sigma_time: 1.5,
                maxregshift: 0.2,
                ..Settings::default()
            },
        )
    };
    let cpu = run(Backend::MultiThread);
    // Twice, so the second goes through the device kept from the first.
    for pass in 0..2 {
        let dev = run(Backend::Gpu);
        for (t, (c, d)) in cpu.iter().zip(&dev).enumerate() {
            assert_eq!(
                (c.dy, c.dx),
                (d.dy, d.dx),
                "pass {pass}, frame {t}: the backends disagreed"
            );
            assert!(
                (c.corr - d.corr).abs() <= c.corr.abs() * 1e-3 + 1e-6,
                "pass {pass}, frame {t}: correlation {} against {}",
                c.corr,
                d.corr
            );
        }
    }
}

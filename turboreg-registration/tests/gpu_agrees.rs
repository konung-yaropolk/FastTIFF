//! The GPU backend must land where the CPU backends land, or the backend
//! selector would be a correctness setting rather than a speed one.
//!
//! It is held to agreement *in pixels*, not bit for bit. The device works in
//! `f32` where the CPU works in `f64`, so a near-tie in the accept/reject step
//! can resolve the other way and the two searches take slightly different
//! paths to the same minimum. What matters is that the transformation they
//! report moves every pixel to the same place, and these tests measure that
//! directly: the largest distance between the two answers over the frame.
//!
//! Skips rather than fails without a usable adapter. Whether a machine has one
//! is a property of the machine; a red test on a headless CI box would say
//! nothing about the code.

#![cfg(feature = "gpu")]

use turboreg_registration::gpu::{available, MIN_PIXELS};
use turboreg_registration::{
    register_pair, register_stack, transform, Backend, Frames, Matrix, Point, Reference, Settings,
    Transformation,
};

/// Large enough that the full-resolution level goes to the device.
const W: usize = 192;
const H: usize = 160;

fn textured(w: usize, h: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            400.0
                + ((x * 0.031).sin() * (y * 0.027).cos()) * 120.0
                + (x * 0.11 + y * 0.07).sin() * 40.0
                + ((x * 0.41).sin() + (y * 0.37).cos()) * 10.0
        })
        .collect()
}

fn rigid(angle: f64, dx: f64, dy: f64, w: usize, h: usize) -> Matrix {
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (s, c) = (angle.sin(), angle.cos());
    Matrix {
        c: [
            dx + cx - c * cx + s * cy,
            c,
            -s,
            0.0,
            dy + cy - s * cx - c * cy,
            s,
            c,
            0.0,
        ],
    }
}

/// The largest distance, over a grid spanning the frame, between where two
/// transformations send the same point.
fn disagreement(a: &Matrix, b: &Matrix, w: usize, h: usize) -> f64 {
    let mut worst = 0.0f64;
    for gy in 0..=4 {
        for gx in 0..=4 {
            let p = Point::new(
                gx as f64 * (w - 1) as f64 / 4.0,
                gy as f64 * (h - 1) as f64 / 4.0,
            );
            let (pa, pb) = (a.apply(p), b.apply(p));
            worst = worst.max(((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt());
        }
    }
    worst
}

fn on(backend: Backend, model: Transformation, reference: Reference) -> Settings {
    Settings {
        transformation: model,
        reference,
        backend,
        ..Settings::default()
    }
}

/// The test's premise: the full-resolution level really is large enough to
/// run on the device, so these tests exercise the kernel and not only the CPU
/// fallback for small levels. Checked when the tests compile, since both sides
/// are constants.
const _: () = assert!(
    W * H >= MIN_PIXELS,
    "the test frames are fitted entirely on the CPU"
);

/// One pair, every model: the GPU's answer moves every pixel to within a
/// hundredth of a pixel of where the CPU's does.
#[test]
fn a_pair_lands_where_the_cpu_lands() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let reference = textured(W, H);
    let moving = transform::warp(&reference, W, H, &rigid(0.02, 2.5, -1.75, W, H));
    for model in Transformation::all() {
        let cpu = register_pair(
            &reference,
            &moving,
            W,
            H,
            &on(Backend::MultiThread, model, Reference::First),
        )
        .expect("cpu");
        let gpu = register_pair(
            &reference,
            &moving,
            W,
            H,
            &on(Backend::Gpu, model, Reference::First),
        )
        .expect("gpu");
        let d = disagreement(&cpu.matrix, &gpu.matrix, W, H);
        assert!(d < 0.01, "{model:?}: GPU and CPU disagree by {d} px");
        // And the criterion agrees to f32 precision.
        let rel = (gpu.criterion - cpu.criterion).abs() / cpu.criterion.max(1e-9);
        assert!(
            rel < 1e-3,
            "{model:?}: criterion {} vs {}",
            gpu.criterion,
            cpu.criterion
        );
    }
}

/// A whole stack in every reference mode: each frame's transformation agrees.
/// `First` and `Mean` give every lane of the batch the same reference;
/// `Previous` gives each lane its own, the frame before it, and composes the
/// answers afterwards — both are covered.
#[test]
fn a_stack_lands_where_the_cpu_lands_in_every_mode() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let base = textured(W, H);
    let frames: Vec<Vec<f32>> = (0..6)
        .map(|t| {
            transform::warp(
                &base,
                W,
                H,
                &rigid(0.006 * t as f64, 0.8 * t as f64, -0.5 * t as f64, W, H),
            )
        })
        .collect();
    let movie = Frames {
        ly: H,
        lx: W,
        frames: &frames,
    };
    for reference in Reference::all() {
        let s = |backend| on(backend, Transformation::RigidBody, reference);
        let cpu = register_stack(&movie, &s(Backend::MultiThread), &mut |_| true).expect("cpu");
        let gpu = register_stack(&movie, &s(Backend::Gpu), &mut |_| true).expect("gpu");
        assert!(
            gpu.failed.is_empty(),
            "{reference:?}: GPU failed {:?}",
            gpu.failed
        );
        for (t, (a, b)) in cpu.matrices.iter().zip(&gpu.matrices).enumerate() {
            let d = disagreement(a, b, W, H);
            assert!(d < 0.02, "{reference:?} frame {t}: disagree by {d} px");
        }
    }
}

/// A batch larger than one dispatch's lanes is split into several and still
/// comes back whole and in order.
#[test]
fn a_stack_longer_than_one_batch_comes_back_in_order() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let base = textured(W, H);
    let n = turboreg_registration::gpu::MOST_LANES + 5;
    let shifts: Vec<f64> = (0..n).map(|t| (t % 7) as f64 * 0.6 - 1.8).collect();
    let frames: Vec<Vec<f32>> = shifts
        .iter()
        .map(|&d| transform::warp(&base, W, H, &rigid(0.0, d, -d * 0.5, W, H)))
        .collect();
    let movie = Frames {
        ly: H,
        lx: W,
        frames: &frames,
    };
    let s = Settings {
        transformation: Transformation::Translation,
        reference: Reference::Mean,
        backend: Backend::Gpu,
        ..Settings::default()
    };
    let out = register_stack(&movie, &s, &mut |_| true).expect("gpu");
    assert_eq!(out.matrices.len(), n);
    assert!(out.failed.is_empty(), "{:?}", out.failed);
    // Each frame's correction tracks its own shift: frames with equal shifts
    // got equal corrections, so nothing was reordered across batches.
    for i in 0..n {
        for j in (i + 1)..n {
            if shifts[i] == shifts[j] {
                let d = disagreement(&out.matrices[i], &out.matrices[j], W, H);
                assert!(
                    d < 0.02,
                    "frames {i} and {j} share a shift but got {d} px apart"
                );
            }
        }
    }
}

/// The same stack on the GPU twice gives the same answer — the device path is
/// deterministic, even though it is not bit-identical to the CPU.
#[test]
fn the_gpu_path_is_reproducible() {
    if !available() {
        eprintln!("no GPU adapter; skipping");
        return;
    }
    let base = textured(W, H);
    let frames: Vec<Vec<f32>> = (0..4)
        .map(|t| {
            transform::warp(
                &base,
                W,
                H,
                &rigid(0.004 * t as f64, 0.7 * t as f64, 0.3, W, H),
            )
        })
        .collect();
    let movie = Frames {
        ly: H,
        lx: W,
        frames: &frames,
    };
    let s = on(Backend::Gpu, Transformation::Affine, Reference::First);
    let a = register_stack(&movie, &s, &mut |_| true).expect("first");
    let b = register_stack(&movie, &s, &mut |_| true).expect("second");
    assert_eq!(a.matrices, b.matrices);
}

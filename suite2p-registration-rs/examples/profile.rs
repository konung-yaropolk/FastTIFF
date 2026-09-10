// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! Where the time in a registration actually goes.
//!
//! Run it release — a debug build is ten to fifty times slower through
//! `rustfft` and the pixel loops, which changes *which* stage dominates and so
//! answers a different question than the one being asked:
//!
//! ```text
//! cargo run --release -p suite2p-registration --example profile
//! cargo run --release -p suite2p-registration --example profile --features gpu
//! ```
//!
//! Written for one question in particular: choosing single-thread, multi-thread
//! or GPU in the viewer made no measurable difference to a run, which meant the
//! backend was not where the time was. Each stage is timed separately, and the
//! backend-sensitive ones are timed once per backend, so that the answer is a
//! table rather than an opinion.

use std::time::Instant;
use suite2p_registration::masks::reference_filters_normed;
use suite2p_registration::pipeline::{apply_batch, measure_batch};
use suite2p_registration::{compute_reference, fft::Fft2, shift_frame, Backend, Frames, Settings};

const LY: usize = 512;
const LX: usize = 512;

/// A frame of blobs, displaced by `(dy, dx)`.
///
/// Blobs rather than noise: phase correlation locks onto structure, and a field
/// of pure noise would correlate with itself everywhere and measure nothing.
fn frame(dy: f32, dx: f32, seed: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; LY * LX];
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..60 {
        let cy = (next() % LY as u32) as f32 + dy;
        let cx = (next() % LX as u32) as f32 + dx;
        let amp = 200.0 + (next() % 800) as f32;
        let sigma = 3.0 + (next() % 5) as f32;
        let r = (3.0 * sigma) as isize;
        for y in -r..=r {
            for x in -r..=r {
                let (py, px) = (cy as isize + y, cx as isize + x);
                if py < 0 || px < 0 || py >= LY as isize || px >= LX as isize {
                    continue;
                }
                let d2 = (y * y + x * x) as f32;
                out[py as usize * LX + px as usize] += amp * (-d2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    out
}

fn ms(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(300);
    println!("{n} frames of {LX}x{LY}\n");

    let built = Instant::now();
    // A slow drift plus a per-frame jitter, which is what a recording looks
    // like: the animal settles, and the heartbeat moves it a pixel or two.
    let movie: Vec<Vec<f32>> = (0..n)
        .map(|t| {
            let phase = t as f32 / n as f32;
            frame(
                8.0 * phase + (t % 3) as f32,
                -6.0 * phase + (t % 2) as f32,
                7,
            )
        })
        .collect();
    println!("  building the fixture      {:8.1} ms", ms(built));

    let settings = Settings::default();
    let frames = Frames {
        ly: LY,
        lx: LX,
        frames: &movie,
    };

    // ---- the largest single piece of work in the run ---------------------
    println!(
        "
  pick_initial_reference (O(nimg_init^2) — every frame against every other):"
    );
    let mut picked: Option<Vec<f32>> = None;
    for backend in [Backend::SingleThread, Backend::MultiThread] {
        let settings = Settings {
            backend,
            ..Settings::default()
        };
        let at = Instant::now();
        let r = suite2p_registration::pick_initial_reference(&movie, LY, LX, &settings);
        println!("    {:<18} {:8.1} ms", backend.label(), ms(at));
        // The two must agree, or "single-thread" would be a different answer
        // rather than the same one taken slowly.
        if let Some(prev) = &picked {
            assert_eq!(prev, &r, "the backends disagreed about the reference");
        }
        picked = Some(r);
    }
    assert_eq!(picked.as_ref().map(|r| r.len()), Some(LY * LX));

    // ---- the reference, per backend --------------------------------------
    println!("\n  compute_reference, by backend (this is where `backend` is read):");
    let mut reference = Vec::new();
    for backend in Backend::all().iter() {
        if let Some(why) = backend.unavailable_reason(LY, LX) {
            println!("    {:<18} skipped: {why}", backend.label());
            continue;
        }
        let settings = Settings {
            backend: *backend,
            ..Settings::default()
        };
        let at = Instant::now();
        let r = compute_reference(&frames, &settings, &mut |_| true).expect("not cancelled");
        println!("    {:<18} {:8.1} ms", backend.label(), ms(at));
        reference = r;
    }

    let mut fft = Fft2::new(LY, LX);
    let filters = reference_filters_normed(
        &mut fft,
        &reference,
        LY,
        LX,
        settings.spatial_taper,
        settings.smooth_sigma,
        settings.norm_frames,
    );

    // ---- measuring the recording, per backend ----------------------------
    println!("\n  measure_batch over the whole recording, by backend:");
    for backend in Backend::all().iter() {
        if backend.unavailable_reason(LY, LX).is_some() {
            continue;
        }
        let settings = Settings {
            backend: *backend,
            ..Settings::default()
        };
        let at = Instant::now();
        let mut shifts = Vec::with_capacity(n);
        for chunk in movie.chunks(settings.batch_size.max(1)) {
            shifts.extend(measure_batch(LY, LX, &filters, chunk, &settings));
        }
        println!("    {:<18} {:8.1} ms", backend.label(), ms(at));
    }

    // ---- applying the shifts, one thread against all of them -------------
    let shifts = {
        let mut v = Vec::with_capacity(n);
        for chunk in movie.chunks(settings.batch_size.max(1)) {
            v.extend(measure_batch(LY, LX, &filters, chunk, &settings));
        }
        v
    };
    let frame_of: Vec<usize> = (0..n).collect();

    let mut serial = movie.clone();
    let at = Instant::now();
    for (plane, s) in serial.iter_mut().zip(&shifts) {
        *plane = shift_frame(plane, LY, LX, s.dy, s.dx);
    }
    let one_thread = ms(at);

    let mut parallel = movie.clone();
    let at = Instant::now();
    apply_batch(&mut parallel, LY, LX, &frame_of, &shifts, None, &settings);
    let many = ms(at);

    println!("\n  applying the shifts to every plane:");
    println!("    one thread         {one_thread:8.1} ms");
    println!(
        "    apply_batch        {many:8.1} ms   ({:.1}x)",
        one_thread / many.max(0.001)
    );
    assert_eq!(
        serial, parallel,
        "the parallel apply must give the same answer"
    );

    // ---- the non-rigid measurement, which is now the default ------------
    if settings.nonrigid {
        let blocks =
            suite2p_registration::nonrigid::make_blocks(LY, LX, settings.block_size, settings.subpixel);
        let search = suite2p_registration::nonrigid::BlockSearch {
            maxregshift_nr: settings.maxregshift_nr,
            snr_thresh: settings.snr_thresh,
            subpixel: settings.subpixel,
            clip: filters.clip,
        };
        println!(
            "
  non-rigid block measurement over {} blocks:",
            blocks.blocks.len()
        );
        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let mut first: Option<usize> = None;
        for workers in [1, cores] {
            let at = Instant::now();
            let mut sets = suite2p_registration::nonrigid::filter_sets(
                &reference,
                LX,
                &blocks,
                settings.spatial_taper,
                settings.smooth_sigma,
                workers,
            );
            let built = ms(at);
            let at = Instant::now();
            let fields = suite2p_registration::nonrigid::measure_blocks_batch(
                &mut sets, LY, LX, &blocks, &movie, &shifts, &search,
            );
            println!(
                "    {workers:>3} worker(s)  {:8.1} ms   (+{built:.1} ms building filters)",
                ms(at)
            );
            // Splitting the work must not lose a frame.
            assert_eq!(fields.len(), movie.len());
            if let Some(prev) = first {
                assert_eq!(prev, fields.len());
            }
            first = Some(fields.len());
            if cores == 1 {
                break;
            }
        }
    }

    println!(
        "\n  measured shifts: {} of {n} frames moved, largest {} px",
        shifts.iter().filter(|s| s.dy != 0 || s.dx != 0).count(),
        shifts
            .iter()
            .map(|s| s.dy.abs().max(s.dx.abs()))
            .max()
            .unwrap_or(0)
    );
}

//! The pyramid: that it halves, that it preserves level, and that it stops.

use super::*;

/// Unit gain. A flat image stays at its own value all the way down, which
/// matters because the criterion is compared in intensity units — a filter with
/// gain 1.02 would make every level's errors mean something slightly different.
#[test]
fn the_reduction_preserves_a_constant() {
    let (w, h) = (64, 48);
    let img = vec![37.5f32; w * h];
    let (mut cur, mut cw, mut ch) = (img, w, h);
    for _ in 0..3 {
        let r = reduce_2d(&cur, cw, ch);
        cur = r.0;
        cw = r.1;
        ch = r.2;
        for (i, &v) in cur.iter().enumerate() {
            assert!(
                (v - 37.5).abs() < 1e-3,
                "level {cw}x{ch} pixel {i} drifted to {v}"
            );
        }
    }
}

/// The kernel's coefficients sum to one, which is the same claim stated on the
/// filter rather than on an image.
#[test]
fn the_kernel_sums_to_one() {
    let h = [6.0f64 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let total = h[0] + 2.0 * h[1] + 2.0 * h[2];
    assert!((total - 1.0).abs() < 1e-15, "kernel gain is {total}");
}

/// Each level is half the size, truncating.
#[test]
fn each_level_halves() {
    let (w, h) = (101, 60);
    let img = vec![1.0f32; w * h];
    let (_, rw, rh) = reduce_2d(&img, w, h);
    assert_eq!(
        (rw, rh),
        (50, 30),
        "odd sizes truncate rather than round up"
    );
}

/// Depth counts the full-resolution level and stops before a side would fall
/// below the minimum. Pinned by worked example: this is the number that decides
/// how large a displacement the search can find.
#[test]
fn the_depth_stops_at_the_minimum_size() {
    // 99 -> 49 -> 24 -> 12, and 6 would be below MIN_SIZE.
    assert_eq!(depth_for(99, 99, 0), 4);
    assert_eq!(MIN_SIZE, 12);
    // A small image gets one level: itself.
    assert_eq!(depth_for(20, 20, 0), 1);
    assert_eq!(depth_for(1, 1, 0), 1);
    // The shorter side decides.
    assert_eq!(depth_for(4096, 20, 0), 1);
    assert!(depth_for(512, 512, 0) > depth_for(64, 64, 0));
}

/// A cap truncates the depth without changing where it would otherwise stop.
#[test]
fn a_cap_limits_the_depth() {
    let uncapped = depth_for(512, 512, 0);
    assert!(uncapped > 2, "the test needs room to cap");
    assert_eq!(depth_for(512, 512, 2), 2);
    assert_eq!(depth_for(512, 512, 1), 1);
    // A cap larger than the image allows changes nothing.
    assert_eq!(depth_for(512, 512, 99), uncapped);
}

/// `build` returns the levels finest-first with the octaves labelled, and the
/// first level is the image untouched.
#[test]
fn the_pyramid_is_built_finest_first() {
    let (w, h) = (64, 64);
    let img: Vec<f32> = (0..w * h).map(|i| (i % 17) as f32).collect();
    let levels = build(&img, w, h, 3);
    assert_eq!(levels.len(), 3);
    assert_eq!((levels[0].w, levels[0].h), (64, 64));
    assert_eq!(levels[0].img, img, "level 0 is the image as given");
    assert_eq!((levels[1].w, levels[1].h), (32, 32));
    assert_eq!((levels[2].w, levels[2].h), (16, 16));
    for (i, l) in levels.iter().enumerate() {
        assert_eq!(l.octave, i);
        assert_eq!(l.scale(), (1 << i) as f64);
        assert_eq!(l.img.len(), l.w * l.h);
    }
}

/// A depth of zero or one is one level, not none — the caller always has
/// something to register on.
#[test]
fn a_degenerate_depth_still_yields_a_level() {
    let img = vec![5.0f32; 16];
    for depth in [0, 1] {
        let levels = build(&img, 4, 4, depth);
        assert_eq!(levels.len(), 1, "depth {depth}");
    }
}

/// The reduction smooths: a single bright pixel spreads rather than either
/// vanishing or staying a spike. A plain decimation (take every second sample)
/// would do one or the other depending on parity, and would alias.
#[test]
fn the_reduction_smooths_rather_than_decimating() {
    let (w, h) = (32, 32);
    let mut img = vec![0.0f32; w * h];
    img[16 * w + 16] = 1600.0;
    let (small, sw, sh) = reduce_2d(&img, w, h);
    let total: f64 = small.iter().map(|&v| v as f64).sum();
    assert!(
        total > 300.0,
        "energy should survive the reduction: {total}"
    );
    let nonzero = small.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        nonzero > 1,
        "a spike should spread over several samples, not stay a spike"
    );
    assert_eq!((sw, sh), (16, 16));
}

/// Gradients are the derivative of the interpolant, so on a linear ramp they
/// are the ramp's slope — everywhere, including at the edges where the mirror
/// takes over and a naive finite difference would report nonsense.
#[test]
fn gradients_recover_a_known_slope() {
    let (w, h) = (24, 20);
    // 3 per column across, -2 per row down.
    let img: Vec<f32> = (0..w * h)
        .map(|i| 100.0 + 3.0 * (i % w) as f32 - 2.0 * (i / w) as f32)
        .collect();
    let level = Level {
        w,
        h,
        img,
        octave: 0,
    };
    let (gx, gy) = gradients(&level);
    // Six pixels in. The border is not wrong, it is a different function: the
    // mirror turns a ramp into a V, and the prefilter is recursive, so that
    // influence reaches inward — decaying by a factor of 1/|pole| ~= 3.7 per
    // pixel, which is exactly what it measures (0.22 at one pixel in, 0.016 at
    // three, 3e-4 at six). Testing at the very edge would be testing the
    // boundary model, which the spline tests already do.
    for y in 6..h - 6 {
        for x in 6..w - 6 {
            let (a, b) = (gx[y * w + x], gy[y * w + x]);
            assert!((a - 3.0).abs() < 1e-3, "dx at ({x},{y}) is {a}, want 3");
            assert!((b + 2.0).abs() < 1e-3, "dy at ({x},{y}) is {b}, want -2");
        }
    }
}

/// A flat image has no gradient anywhere, border included.
#[test]
fn a_flat_image_has_no_gradient() {
    let level = Level {
        w: 16,
        h: 16,
        img: vec![9.0f32; 256],
        octave: 0,
    };
    let (gx, gy) = gradients(&level);
    for (i, (&a, &b)) in gx.iter().zip(gy.iter()).enumerate() {
        assert!(
            a.abs() < 1e-3 && b.abs() < 1e-3,
            "pixel {i}: gradient ({a}, {b}) on a flat image"
        );
    }
}

/// Gradient images are the same on every core as on one: each output pixel
/// reads only the coefficients, so rows cannot affect each other.
#[test]
fn gradients_do_not_depend_on_threads() {
    let (w, h) = (41usize, 29usize);
    let level = Level {
        w,
        h,
        img: (0..w * h).map(|i| ((i * 37) % 97) as f32).collect(),
        octave: 0,
    };
    let coeff = coefficients(&level);
    let seq = gradients_from(&coeff, w, h, false);
    let par = gradients_from(&coeff, w, h, true);
    assert_eq!(seq, par, "threaded gradients differ");
    assert_eq!(
        gradients(&level),
        seq,
        "the plain entry point is sequential"
    );
    assert_eq!(coefficients_with(&level, true), coeff);
}

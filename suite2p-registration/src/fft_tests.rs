//! The transform, against values that can be worked out by hand.

use super::*;

fn plane(ly: usize, lx: usize, f: impl Fn(usize, usize) -> f32) -> Vec<Complex32> {
    (0..ly * lx)
        .map(|i| Complex32::new(f(i / lx, i % lx), 0.0))
        .collect()
}

/// A constant plane transforms to a single spike at DC of `sum`, and nothing
/// else. Anything wrong with the row/column split shows up here immediately.
#[test]
fn a_constant_plane_is_a_single_dc_spike() {
    let (ly, lx) = (6, 8);
    let mut f = Fft2::new(ly, lx);
    let mut data = plane(ly, lx, |_, _| 2.0);
    f.forward(&mut data);
    assert!(
        (data[0].re - 2.0 * (ly * lx) as f32).abs() < 1e-3,
        "{:?}",
        data[0]
    );
    for c in &data[1..] {
        assert!(c.norm() < 1e-3, "{c:?}");
    }
}

/// Forward then inverse is the identity, on a non-square, non-power-of-two
/// plane — the case a hand-rolled radix-2 would get wrong.
#[test]
fn a_round_trip_returns_the_original() {
    let (ly, lx) = (6, 10);
    let mut f = Fft2::new(ly, lx);
    let original = plane(ly, lx, |y, x| (y * 3 + x * 7) as f32 % 11.0);
    let mut data = original.clone();
    f.forward(&mut data);
    f.inverse(&mut data);
    for (a, b) in data.iter().zip(&original) {
        assert!((a.re - b.re).abs() < 1e-3, "{a:?} vs {b:?}");
        assert!(a.im.abs() < 1e-3, "{a:?} grew an imaginary part");
    }
}

/// A non-power-of-two size in both axes must work; suite2p is run on 1024x768
/// and worse, and Bluestein is what `rustfft` falls back to.
#[test]
fn an_awkward_size_still_round_trips() {
    let (ly, lx) = (7, 13);
    let mut f = Fft2::new(ly, lx);
    let original = plane(ly, lx, |y, x| (y as f32) - (x as f32) * 0.5);
    let mut data = original.clone();
    f.forward(&mut data);
    f.inverse(&mut data);
    for (a, b) in data.iter().zip(&original) {
        assert!((a.re - b.re).abs() < 1e-3);
    }
}

/// `fftshift` moves the corner to the centre, and `ifftshift` puts it back —
/// for odd sizes too, where the two are not the same operation.
#[test]
fn the_shifts_are_inverses_including_for_odd_sizes() {
    for (ly, lx) in [(4usize, 4usize), (5, 4), (4, 5), (5, 7)] {
        let data: Vec<f32> = (0..ly * lx).map(|i| i as f32).collect();
        let there = fftshift(&data, ly, lx);
        let back = ifftshift(&there, ly, lx);
        assert_eq!(back, data, "{ly}x{lx}");
    }
}

#[test]
fn fftshift_puts_the_origin_in_the_middle() {
    // A spike at the corner should land at (ly/2, lx/2).
    let (ly, lx) = (4, 6);
    let mut data = vec![0.0f32; ly * lx];
    data[0] = 1.0;
    let shifted = fftshift(&data, ly, lx);
    assert_eq!(shifted[(ly / 2) * lx + lx / 2], 1.0);
    assert_eq!(shifted.iter().filter(|v| **v != 0.0).count(), 1);
}

/// The transform by its definition, one output sample at a time.
fn naive_dft(data: &[Complex32], ly: usize, lx: usize) -> Vec<Complex32> {
    let tau = std::f64::consts::TAU;
    (0..ly * lx)
        .map(|k| {
            let (ky, kx) = (k / lx, k % lx);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, v) in data.iter().enumerate() {
                let (y, x) = (i / lx, i % lx);
                let a = -tau * ((ky * y) as f64 / ly as f64 + (kx * x) as f64 / lx as f64);
                re += v.re as f64 * a.cos() - v.im as f64 * a.sin();
                im += v.re as f64 * a.sin() + v.im as f64 * a.cos();
            }
            Complex32::new(re as f32, im as f32)
        })
        .collect()
}

/// The forward transform is the DFT, sample for sample, on sizes whose width is
/// not a multiple of the column tile (so the last tile is short) and on one
/// narrower than a tile. A gather that put a column back in the wrong place
/// would still round-trip; it would not match this.
#[test]
fn the_forward_transform_is_the_dft() {
    for (ly, lx) in [(5, 11), (4, 8), (3, 19), (6, 3), (1, 9), (9, 1)] {
        let original = plane(ly, lx, |y, x| ((y * 5 + x * 3) % 7) as f32 - 2.5);
        let want = naive_dft(&original, ly, lx);
        let mut got = original.clone();
        Fft2::new(ly, lx).forward(&mut got);
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).norm() < 1e-3, "{ly}x{lx} at {i}: {a:?} vs {b:?}");
        }
    }
}

/// Workers sharing one set of plans, each with its own scratch, compute what a
/// transform with plans of its own does — on threads at once.
#[test]
fn shared_plans_transform_like_private_ones() {
    let (ly, lx) = (12, 20);
    let plans = FftPlans2::new(ly, lx);
    let original = plane(ly, lx, |y, x| (y * x % 13) as f32);
    let mut want = original.clone();
    Fft2::new(ly, lx).forward(&mut want);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let mut fft = Fft2::with_plans(&plans);
                for _ in 0..3 {
                    let mut got = original.clone();
                    fft.forward(&mut got);
                    assert_eq!(got, want);
                    fft.inverse(&mut got);
                    for (a, b) in got.iter().zip(&original) {
                        assert!((a - b).norm() < 1e-3);
                    }
                }
            });
        }
    });
}

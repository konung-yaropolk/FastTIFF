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

//! The masks, against their stated shape.

use super::*;

#[test]
fn the_taper_is_one_in_the_middle_and_fades_at_the_edge() {
    let (ly, lx) = (64, 64);
    // Slope 3 on a 64-pixel frame. The taper's half-way point sits `2*sig`
    // inside the edge, so a slope anywhere near the frame's own size fades the
    // middle too — which is why `spatial_taper` defaults to 40 for a 512-pixel
    // frame and would be wrong here.
    let m = spatial_taper(3.0, ly, lx);
    let centre = m[(ly / 2) * lx + lx / 2];
    assert!(centre > 0.99, "centre should be untouched: {centre}");
    assert!(m[0] < 0.1, "the corner should be faded: {}", m[0]);
    assert!(
        m[(ly / 2) * lx] < m[(ly / 2) * lx + lx / 2],
        "an edge must be fainter than the middle"
    );
}

/// A wider slope fades more of the frame, which is the whole knob.
#[test]
fn a_wider_slope_fades_more() {
    let (ly, lx) = (64, 64);
    let narrow = spatial_taper(4.0, ly, lx);
    let wide = spatial_taper(16.0, ly, lx);
    let sum = |m: &Vec<f32>| m.iter().map(|&v| v as f64).sum::<f64>();
    assert!(
        sum(&wide) < sum(&narrow),
        "a wider taper keeps less of the frame"
    );
}

#[test]
fn the_taper_is_symmetric() {
    let (ly, lx) = (32, 40);
    let m = spatial_taper(8.0, ly, lx);
    for y in 0..ly {
        for x in 0..lx {
            let mirrored = m[(ly - 1 - y) * lx + (lx - 1 - x)];
            assert!(
                (m[y * lx + x] - mirrored).abs() < 1e-6,
                "({y},{x}) is not mirrored"
            );
        }
    }
}

#[test]
fn the_gaussian_kernel_sums_to_one_and_peaks_in_the_middle() {
    let (ly, lx) = (32, 32);
    let k = gaussian_kernel(2.0, 2.0, ly, lx);
    let sum: f64 = k.iter().map(|&v| v as f64).sum();
    assert!((sum - 1.0).abs() < 1e-5, "sums to {sum}");
    let peak = k.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // The centre of an even-sized plane sits between samples, so the peak is
    // one of the four middle pixels.
    assert_eq!(k[(ly / 2) * lx + lx / 2], peak);
}

/// The frequency-domain Gaussian must be real and positive at DC — if the
/// `ifftshift` were skipped it would alternate in sign, and every correlation
/// peak would land half a frame away.
#[test]
fn the_frequency_gaussian_is_centred_not_alternating() {
    let (ly, lx) = (32, 32);
    let mut fft = Fft2::new(ly, lx);
    let g = gaussian_fft(&mut fft, 2.0, ly, lx);
    assert!(
        (g[0] - 1.0).abs() < 1e-3,
        "DC should be the kernel's sum: {}",
        g[0]
    );
    // A Gaussian's transform is a Gaussian: positive everywhere, falling away
    // from DC. Alternating signs would mean the shift was missed.
    assert!(g.iter().all(|&v| v > -1e-4), "the transform changed sign");
    assert!(g[1] < g[0]);
}

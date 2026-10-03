//! Padding, PSF placement, and the blur operator.
//!
//! The two things checked hardest here are the two that produce a believable
//! wrong answer: a PSF placed at the middle of the array instead of the
//! origin (which translates the result by half the grid), and a `correlate`
//! that is not actually the adjoint of `blur` (which makes every iterative
//! method converge to the wrong image, slowly and smoothly).

use super::*;

fn ramp(d: Dims) -> Vec<f32> {
    (0..d.len()).map(|i| i as f32 + 1.0).collect()
}

// --------------------------------------------------------------- the edges

#[test]
fn each_edge_rule_extends_the_way_it_says() {
    // Source of width 3: samples 0, 1, 2.
    let cases = [
        (Edge::Replicate, [0usize, 0, 0, 1, 2, 2, 2]),
        (Edge::Wrap, [1, 2, 0, 1, 2, 0, 1]),
        (Edge::Mirror, [1, 0, 0, 1, 2, 2, 1]),
    ];
    for (edge, want) in cases {
        let got: Vec<usize> = (-2..5)
            .map(|p| edge.source(p, 3).expect("these rules always have a source"))
            .collect();
        assert_eq!(got, want.to_vec(), "{edge:?}");
    }
    assert_eq!(Edge::Zero.source(-1, 3), None);
    assert_eq!(Edge::Zero.source(3, 3), None);
    assert_eq!(Edge::Zero.source(1, 3), Some(1));
}

/// A halo wider than the image folds more than once. Reachable: a 32-slice
/// PSF applied to a 4-slice stack asks for exactly this, and a single fold
/// would index out of range.
#[test]
fn mirroring_folds_as_many_times_as_it_has_to() {
    for p in -40..40 {
        let s = Edge::Mirror.source(p, 3).expect("mirror always resolves");
        assert!(s < 3, "{p} mapped to {s}");
    }
    // The pattern has period 2n, so -6 is two whole periods out and lands
    // back on sample 0, where -4 reflects to 2.
    assert_eq!(Edge::Mirror.source(-6, 3), Some(0));
    assert_eq!(Edge::Mirror.source(-4, 3), Some(2));
    assert_eq!(Edge::Mirror.source(8, 3), Some(2));
}

// ------------------------------------------------------------ embed and crop

#[test]
fn cropping_undoes_embedding() {
    let src = Dims::new(5, 4, 3);
    let img = ramp(src);
    for edge in Edge::ALL {
        let grid = Grid::plan(src, Dims::new(3, 3, 3), edge);
        let mut padded = Vec::new();
        let mut back = Vec::new();
        grid.embed(&img, &mut padded);
        assert_eq!(padded.len(), grid.padded().len());
        grid.crop(&padded, &mut back);
        assert_eq!(back, img, "{edge:?} did not survive the round trip");
    }
}

#[test]
fn the_halo_is_filled_the_way_the_edge_rule_says() {
    let src = Dims::new(3, 1, 1);
    let grid = Grid::plan(src, Dims::new(3, 1, 1), Edge::Replicate);
    let mut padded = Vec::new();
    grid.embed(&[7.0, 8.0, 9.0], &mut padded);
    // Padded to 3 + 2 * 1 = 5, which is already a fast length.
    assert_eq!(padded, vec![7.0, 7.0, 8.0, 9.0, 9.0]);

    let grid = Grid::plan(src, Dims::new(3, 1, 1), Edge::Zero);
    grid.embed(&[7.0, 8.0, 9.0], &mut padded);
    assert_eq!(padded, vec![0.0, 7.0, 8.0, 9.0, 0.0]);
}

/// Wrapping asks for the circular convolution on purpose, so it gets no halo
/// — only whatever rounding up to a fast length adds.
#[test]
fn wrapping_adds_no_halo() {
    let src = Dims::new(8, 8, 4);
    let grid = Grid::plan(src, Dims::new(5, 5, 3), Edge::Wrap);
    assert_eq!(grid.padded(), Dims::new(8, 8, 4));
}

/// A single-slice image stays single-slice however deep the PSF is: growing a
/// third axis would turn a 2-D problem into a 3-D one the user did not ask
/// for and cannot afford.
#[test]
fn a_flat_image_is_not_given_depth() {
    let grid = Grid::plan(Dims::new(16, 16, 1), Dims::new(5, 5, 1), Edge::Mirror);
    assert_eq!(grid.padded().z, 1);
}

/// But a flat image with a deep PSF does get depth, because that convolution
/// genuinely needs it.
#[test]
fn a_deep_psf_on_a_flat_image_still_gets_a_z_axis() {
    let grid = Grid::plan(Dims::new(16, 16, 1), Dims::new(5, 5, 7), Edge::Mirror);
    assert!(grid.padded().z >= 7, "{:?}", grid.padded());
}

// ------------------------------------------------------------ PSF placement

/// The peak lands at index 0 and the rest wraps. Getting this wrong
/// translates every deconvolved stack by half the grid, which looks like the
/// algorithm failed rather than like the kernel was misplaced.
#[test]
fn the_psf_peak_lands_at_the_origin() {
    let pd = Dims::new(3, 3, 1);
    // A kernel whose peak is deliberately not at its own centre.
    let mut psf = vec![0.0f32; 9];
    psf[pd.at(2, 0, 0)] = 1.0;
    psf[pd.at(1, 0, 0)] = 0.5;

    let grid = Dims::new(8, 8, 1);
    let placed = place_psf(grid, &psf, pd, brightest(&psf, pd));
    assert_eq!(placed[0], 1.0, "the peak belongs at index 0");
    // The sample one to the peak's left belongs at the far end of the x axis.
    assert_eq!(placed[grid.at(7, 0, 0)], 0.5);
    assert_eq!(placed.iter().filter(|v| **v != 0.0).count(), 2);
}

#[test]
fn the_brightest_voxel_is_found_in_three_dimensions() {
    let d = Dims::new(3, 4, 2);
    let mut v = vec![0.0f32; d.len()];
    v[d.at(2, 3, 1)] = 5.0;
    assert_eq!(brightest(&v, d), (2, 3, 1));
    assert_eq!(centre(d), (1, 2, 1));
}

#[test]
fn normalising_makes_the_sum_one_and_refuses_what_it_cannot() {
    let mut v = vec![1.0f32, 2.0, 1.0];
    assert!(normalise(&mut v));
    assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);

    assert!(
        !normalise(&mut [0.0, 0.0]),
        "an empty PSF cannot be normalised"
    );
    assert!(
        !normalise(&mut [1.0, -2.0]),
        "nor one that sums to less than nothing"
    );
}

/// A PSF wider than the grid folds onto itself rather than reading out of
/// bounds. Degraded, and the alternative is a panic in the middle of a run.
#[test]
fn an_oversized_psf_wraps_instead_of_panicking() {
    let pd = Dims::new(7, 1, 1);
    let psf = vec![1.0f32; 7];
    let grid = Dims::new(4, 1, 1);
    let placed = place_psf(grid, &psf, pd, (3, 0, 0));
    assert_eq!(placed.len(), 4);
    assert!(
        (placed.iter().sum::<f32>() - 7.0).abs() < 1e-6,
        "energy is kept"
    );
}

// ------------------------------------------------------------- the operator

fn operator(grid: Dims, psf: &[f32], psf_dims: Dims, origin: (usize, usize, usize)) -> Operator {
    let placed = place_psf(grid, psf, psf_dims, origin);
    Operator::new(Transform::new(grid), &placed)
}

#[test]
fn blurring_by_a_delta_is_the_identity() {
    let grid = Dims::new(8, 6, 2);
    let mut op = operator(grid, &[1.0], Dims::new(1, 1, 1), (0, 0, 0));
    let src = ramp(grid);
    let mut dst = vec![0.0; grid.len()];
    op.blur(&src, &mut dst);
    for (a, b) in dst.iter().zip(&src) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}

/// A blur by a normalised kernel preserves the total — which is the whole
/// reason `normalise` exists, stated as a property of the operator.
#[test]
fn blurring_preserves_the_total() {
    let grid = Dims::new(16, 16, 1);
    let pd = Dims::new(3, 3, 1);
    let mut psf = vec![1.0f32; 9];
    normalise(&mut psf);
    let mut op = operator(grid, &psf, pd, centre(pd));

    let src = ramp(grid);
    let mut dst = vec![0.0; grid.len()];
    op.blur(&src, &mut dst);
    let (a, b) = (
        src.iter().map(|&v| v as f64).sum::<f64>(),
        dst.iter().map(|&v| v as f64).sum::<f64>(),
    );
    assert!((a - b).abs() < 1e-2 * a, "{a} vs {b}");
}

/// `correlate` is the adjoint of `blur`: `<Hx, y>` must equal `<x, HTy>` for
/// every `x` and `y`.
///
/// The one test that would catch a missing conjugate, and it needs an
/// *asymmetric* PSF to do it — with a symmetric one the operator is
/// self-adjoint and the identity holds whether or not the code is right.
/// Every iterative method here is built on this holding.
#[test]
fn correlate_is_the_adjoint_of_blur() {
    let grid = Dims::new(9, 7, 3);
    let pd = Dims::new(3, 2, 2);
    // Deliberately lopsided in all three axes.
    let psf: Vec<f32> = (0..pd.len()).map(|i| (i * i % 7) as f32 + 0.5).collect();
    let mut op = operator(grid, &psf, pd, (0, 0, 0));

    let x: Vec<f32> = (0..grid.len())
        .map(|i| ((i * 37) % 11) as f32 - 5.0)
        .collect();
    let y: Vec<f32> = (0..grid.len())
        .map(|i| ((i * 53) % 13) as f32 - 6.0)
        .collect();

    let mut hx = vec![0.0; grid.len()];
    let mut hty = vec![0.0; grid.len()];
    op.blur(&x, &mut hx);
    op.correlate(&y, &mut hty);

    let left: f64 = hx
        .iter()
        .zip(&y)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let right: f64 = x
        .iter()
        .zip(&hty)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    assert!(
        (left - right).abs() < 1e-3 * left.abs().max(1.0),
        "<Hx,y> = {left} but <x,HTy> = {right}"
    );
}

/// The spectrum of an arbitrary kernel, which the regularised methods need,
/// is the same one the operator would have computed for it as a PSF.
#[test]
fn a_second_kernel_transforms_on_the_same_plan() {
    let grid = Dims::new(6, 4, 1);
    let mut op = operator(grid, &[1.0], Dims::new(1, 1, 1), (0, 0, 0));
    let mut kernel = vec![0.0f32; grid.len()];
    kernel[0] = 1.0;
    let spec = op.spectrum_of(&kernel);
    // A delta's spectrum is 1 everywhere.
    assert!(spec
        .iter()
        .all(|c| (c.re - 1.0).abs() < 1e-5 && c.im.abs() < 1e-5));
}

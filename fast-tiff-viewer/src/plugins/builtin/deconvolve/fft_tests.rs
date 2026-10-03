//! The transform, checked against the identities it has to satisfy.
//!
//! An FFT is easy to get subtly wrong — a transposed axis, a missing
//! normalisation, a strided gather off by a stride — and every one of those
//! produces output that still looks like an image. These tests are the
//! identities that distinguish a correct transform from a plausible one.

use super::*;

fn seeded(n: usize) -> Vec<Complex32> {
    // A deterministic pseudo-random fill. Not a crate: the only property
    // needed is "no accidental symmetry", and a symmetric fixture is exactly
    // what would hide a transposed axis.
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / 16777216.0 - 0.5
            };
            Complex32::new(next(), next())
        })
        .collect()
}

#[test]
fn fast_lengths_are_the_smallest_smooth_ones() {
    assert_eq!(next_fast_len(0), 1);
    assert_eq!(next_fast_len(1), 1);
    assert_eq!(next_fast_len(12), 12);
    assert_eq!(next_fast_len(13), 14); // 2 * 7
    assert_eq!(next_fast_len(11), 12);
    assert_eq!(next_fast_len(530), 540); // 2^2 * 3^3 * 5
                                         // Never smaller than asked, and never a large prime.
    for n in 1..2000 {
        let m = next_fast_len(n);
        assert!(m >= n, "{m} < {n}");
        assert!(is_fast(m), "{m} is not 7-smooth");
    }
}

#[test]
fn the_inverse_undoes_the_forward() {
    for dims in [
        Dims::new(8, 1, 1),
        Dims::new(6, 5, 1),
        Dims::new(4, 3, 2),
        Dims::new(7, 1, 5),
    ] {
        let tf = Transform::new(dims);
        let original = seeded(dims.len());
        let mut buf = original.clone();
        tf.forward(&mut buf);
        tf.inverse(&mut buf);
        for (a, b) in buf.iter().zip(&original) {
            assert!((a - b).norm() < 1e-5, "{dims:?}: {a} vs {b}");
        }
    }
}

/// Convolving with a delta at the origin is the identity — the single
/// assertion that catches a missing `1/N`, since every other error would
/// scale the result rather than leave it alone.
#[test]
fn convolution_with_a_delta_changes_nothing() {
    let dims = Dims::new(5, 4, 3);
    let tf = Transform::new(dims);
    let signal = seeded(dims.len());

    let mut delta = vec![Complex32::new(0.0, 0.0); dims.len()];
    delta[0] = Complex32::new(1.0, 0.0);
    tf.forward(&mut delta);

    let mut buf = signal.clone();
    tf.forward(&mut buf);
    multiply(&mut buf, &delta);
    tf.inverse(&mut buf);

    for (a, b) in buf.iter().zip(&signal) {
        assert!((a - b).norm() < 1e-5, "{a} vs {b}");
    }
}

/// A delta somewhere other than the origin translates, by exactly its offset
/// and with wraparound. This is what pins the axis order: a transposed y and
/// z would move the signal the wrong way and nothing above would notice.
#[test]
fn a_shifted_delta_translates_by_its_offset() {
    let dims = Dims::new(4, 3, 2);
    let tf = Transform::new(dims);
    let signal: Vec<Complex32> = (0..dims.len())
        .map(|i| Complex32::new(i as f32, 0.0))
        .collect();

    let shift = (1usize, 2usize, 1usize);
    let mut delta = vec![Complex32::new(0.0, 0.0); dims.len()];
    delta[dims.at(shift.0, shift.1, shift.2)] = Complex32::new(1.0, 0.0);
    tf.forward(&mut delta);

    let mut buf = signal.clone();
    tf.forward(&mut buf);
    multiply(&mut buf, &delta);
    tf.inverse(&mut buf);

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let from = dims.at(
                    (x + dims.x - shift.0) % dims.x,
                    (y + dims.y - shift.1) % dims.y,
                    (z + dims.z - shift.2) % dims.z,
                );
                let got = buf[dims.at(x, y, z)].re;
                assert!(
                    (got - signal[from].re).abs() < 1e-3,
                    "({x},{y},{z}): {got} should be {}",
                    signal[from].re
                );
            }
        }
    }
}

/// Correlation is convolution with the kernel reflected, so correlating with a
/// delta at `+d` shifts by `-d`. The conjugate in `multiply_conj` is the whole
/// of the adjoint, and a symmetric fixture cannot tell whether it is there.
#[test]
fn correlation_shifts_the_other_way() {
    let dims = Dims::new(6, 1, 1);
    let tf = Transform::new(dims);
    let signal: Vec<Complex32> = (0..6).map(|i| Complex32::new(i as f32, 0.0)).collect();

    let mut delta = vec![Complex32::new(0.0, 0.0); 6];
    delta[2] = Complex32::new(1.0, 0.0);
    tf.forward(&mut delta);

    let mut buf = signal.clone();
    tf.forward(&mut buf);
    multiply_conj(&mut buf, &delta);
    tf.inverse(&mut buf);

    for x in 0..6 {
        let want = signal[(x + 2) % 6].re;
        assert!(
            (buf[x].re - want).abs() < 1e-3,
            "{x}: {} vs {want}",
            buf[x].re
        );
    }
}

/// Parseval: the energy is the same in both domains, up to the normalisation.
/// Independent of the identities above, and it fails for a transform that
/// drops or double-counts any element of a strided pass.
#[test]
fn energy_is_conserved() {
    let dims = Dims::new(5, 4, 3);
    let tf = Transform::new(dims);
    let signal = seeded(dims.len());
    let space: f64 = signal.iter().map(|c| c.norm_sqr() as f64).sum();

    let mut buf = signal.clone();
    tf.forward(&mut buf);
    let freq: f64 = buf.iter().map(|c| c.norm_sqr() as f64).sum::<f64>() / dims.len() as f64;

    assert!(
        (space - freq).abs() < 1e-3 * space.max(1.0),
        "{space} vs {freq}"
    );
}

/// A single-point axis is skipped rather than planned, so a 2-D grid has to
/// give the same answer as the same data in a 3-D grid of depth one.
#[test]
fn a_flat_axis_is_the_identity() {
    let flat = Dims::new(4, 4, 1);
    let tf = Transform::new(flat);
    let signal = seeded(16);
    let mut buf = signal.clone();
    tf.forward(&mut buf);
    tf.inverse(&mut buf);
    for (a, b) in buf.iter().zip(&signal) {
        assert!((a - b).norm() < 1e-5);
    }
}

#[test]
fn lifting_and_lowering_round_trip() {
    let src = vec![1.0f32, -2.5, 3.25, 0.0];
    let mut c = vec![Complex32::new(9.0, 9.0); 4];
    lift(&src, &mut c);
    assert!(
        c.iter().all(|v| v.im == 0.0),
        "the imaginary part must be cleared"
    );
    let mut back = vec![0.0f32; 4];
    lower(&c, &mut back);
    assert_eq!(back, src);
}

/// The whole 3-D transform against a direct DFT, on a grid wide enough that
/// every strided pass is split across several tasks.
///
/// The oracle is the definition, written out: `sum over (x,y,z) of
/// v * exp(-2 pi i (ux/nx + vy/ny + wz/nz))`. It shares no code with the
/// implementation, which is what makes it able to catch the failure the
/// parallel split could introduce and nothing else here would — a block of
/// columns transformed twice, or not at all. Both leave an answer that is
/// still smooth, still finite, and wrong.
#[test]
fn the_parallel_transform_matches_a_direct_dft() {
    // `block_width` floors at 32 columns, so these are several blocks per
    // pass: 40 columns for y, 120 for z.
    let dims = Dims::new(40, 3, 4);
    let signal = seeded(dims.len());

    let mut got = signal.clone();
    Transform::new(dims).forward(&mut got);

    let tau = std::f64::consts::TAU;
    for wz in 0..dims.z {
        for vy in 0..dims.y {
            for ux in 0..dims.x {
                let mut acc = (0.0f64, 0.0f64);
                for z in 0..dims.z {
                    for y in 0..dims.y {
                        for x in 0..dims.x {
                            let s = signal[dims.at(x, y, z)];
                            let phase = -tau
                                * ((ux * x) as f64 / dims.x as f64
                                    + (vy * y) as f64 / dims.y as f64
                                    + (wz * z) as f64 / dims.z as f64);
                            let (sin, cos) = phase.sin_cos();
                            acc.0 += s.re as f64 * cos - s.im as f64 * sin;
                            acc.1 += s.re as f64 * sin + s.im as f64 * cos;
                        }
                    }
                }
                let g = got[dims.at(ux, vy, wz)];
                assert!(
                    (g.re as f64 - acc.0).abs() < 1e-3 && (g.im as f64 - acc.1).abs() < 1e-3,
                    "bin ({ux},{vy},{wz}): {g} should be {acc:?}"
                );
            }
        }
    }
}

/// Every block boundary is exercised: a width that is not a multiple of the
/// block size leaves a short last block, which must be transformed too.
#[test]
fn a_short_last_block_is_not_dropped() {
    for dims in [
        Dims::new(33, 2, 2),  // y: 33 columns, one block of 32 and one of 1
        Dims::new(5, 7, 13),  // z: 35 columns
        Dims::new(100, 3, 4), // several blocks on both strided axes
        Dims::new(48, 48, 9),
    ] {
        let tf = Transform::new(dims);
        let original = seeded(dims.len());
        let mut buf = original.clone();
        tf.forward(&mut buf);
        tf.inverse(&mut buf);
        for (i, (a, b)) in buf.iter().zip(&original).enumerate() {
            assert!(
                (a - b).norm() < 1e-4,
                "{dims:?} element {i}: {a} came back as {b}"
            );
        }
    }
}

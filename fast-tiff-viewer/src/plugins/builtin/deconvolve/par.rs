//! Elementwise work over a whole grid, across cores.
//!
//! An iterative deconvolution is two things: transforms, and passes over the
//! grid. The passes look negligible next to an FFT and are not — one
//! Richardson-Lucy iteration on a 1080x1080x120 grid walks half a gigabyte
//! five or six times, which at the memory bandwidth of one core is seconds.
//! Parallelising the transforms and leaving these serial moved a real run from
//! 34 seconds to 17; this module is what took it the rest of the way.
//!
//! Every operation here is a read, an arithmetic operation and a write. None
//! of them is compute-bound, so the speedup comes from using more than one
//! core's share of the memory controller rather than from more arithmetic.
//!
//! # Reductions are chunked so they stay reproducible
//!
//! Floating-point addition is not associative, so a sum's value depends on the
//! order it was added in. `rayon`'s own `sum` splits according to how work
//! happened to be stolen, which varies between runs of the same program — and
//! two runs of a deconvolution that differ in the last bits of their
//! convergence test would be a thing that is nearly impossible to debug later.
//! So [`sum_by`] and [`zip_sum`] reduce over *fixed-size* chunks in index
//! order: the tree is decided by the length alone, and the answer is the same
//! every time. (It is not the same as the serial answer, and does not need to
//! be; it is better, being a shallower tree.)

/// Below this many elements, spreading a pass over cores costs more than it
/// saves.
pub(crate) const FLOOR: usize = 1 << 16;

/// How much one task takes, and the unit a reduction is chunked by.
const CHUNK: usize = 1 << 16;

/// Apply `f` to every element.
pub(crate) fn each<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn(&mut T) + Send + Sync,
{
    #[cfg(feature = "threads")]
    if v.len() >= FLOOR {
        use rayon::prelude::*;
        v.par_chunks_mut(CHUNK)
            .for_each(|c| c.iter_mut().for_each(&f));
        return;
    }
    v.iter_mut().for_each(f);
}

/// Apply `f` to every `(a, b)` pair. `a` and `b` must be the same length.
pub(crate) fn zip<A, B, F>(a: &mut [A], b: &[B], f: F)
where
    A: Send,
    B: Sync,
    F: Fn(&mut A, &B) + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "threads")]
    if a.len() >= FLOOR {
        use rayon::prelude::*;
        a.par_chunks_mut(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .for_each(|(ac, bc)| ac.iter_mut().zip(bc).for_each(|(x, y)| f(x, y)));
        return;
    }
    a.iter_mut().zip(b).for_each(|(x, y)| f(x, y));
}

/// Apply `f` to every `(a, b, c)` triple. All three must be the same length.
pub(crate) fn zip3<A, B, C, F>(a: &mut [A], b: &[B], c: &[C], f: F)
where
    A: Send,
    B: Sync,
    C: Sync,
    F: Fn(&mut A, &B, &C) + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    #[cfg(feature = "threads")]
    if a.len() >= FLOOR {
        use rayon::prelude::*;
        a.par_chunks_mut(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .zip(c.par_chunks(CHUNK))
            .for_each(|((ac, bc), cc)| {
                ac.iter_mut()
                    .zip(bc)
                    .zip(cc)
                    .for_each(|((x, y), z)| f(x, y, z))
            });
        return;
    }
    a.iter_mut()
        .zip(b)
        .zip(c)
        .for_each(|((x, y), z)| f(x, y, z));
}

/// Sum `f` over every element, in `f64`.
pub(crate) fn sum_by<T, F>(v: &[T], f: F) -> f64
where
    T: Sync,
    F: Fn(&T) -> f64 + Send + Sync,
{
    #[cfg(feature = "threads")]
    if v.len() >= FLOOR {
        use rayon::prelude::*;
        // `collect` on an indexed parallel iterator keeps index order, so the
        // final sum is over a sequence decided by the length and nothing else.
        let parts: Vec<f64> = v
            .par_chunks(CHUNK)
            .map(|c| c.iter().map(&f).sum::<f64>())
            .collect();
        return parts.iter().sum();
    }
    v.iter().map(&f).sum()
}

/// Sum `f` over every `(a, b)` pair, in `f64`.
///
/// The read-only sibling of [`zip_sum`], and chunked in index order for the
/// same reason.
pub(crate) fn zip_sum_by<A, B, F>(a: &[A], b: &[B], f: F) -> f64
where
    A: Sync,
    B: Sync,
    F: Fn(&A, &B) -> f64 + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "threads")]
    if a.len() >= FLOOR {
        use rayon::prelude::*;
        let parts: Vec<f64> = a
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(ac, bc)| ac.iter().zip(bc).map(|(x, y)| f(x, y)).sum::<f64>())
            .collect();
        return parts.iter().sum();
    }
    a.iter().zip(b).map(|(x, y)| f(x, y)).sum()
}

/// Mutate every `(a, b)` pair and sum what `f` returns.
///
/// For the steps that have to measure something while they change it —
/// MRNSD's search direction and its scaling are one pass over the grid, not
/// two.
pub(crate) fn zip_sum<A, B, F>(a: &mut [A], b: &[B], f: F) -> f64
where
    A: Send,
    B: Sync,
    F: Fn(&mut A, &B) -> f64 + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "threads")]
    if a.len() >= FLOOR {
        use rayon::prelude::*;
        let parts: Vec<f64> = a
            .par_chunks_mut(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(ac, bc)| ac.iter_mut().zip(bc).map(|(x, y)| f(x, y)).sum::<f64>())
            .collect();
        return parts.iter().sum();
    }
    a.iter_mut().zip(b).map(|(x, y)| f(x, y)).sum()
}

/// The smallest `f` over every `(a, b)` pair, starting from `init`.
///
/// No chunking needed: `min` is associative and exact, so the answer does not
/// depend on the order.
pub(crate) fn zip_min<A, B, F>(a: &[A], b: &[B], init: f32, f: F) -> f32
where
    A: Sync,
    B: Sync,
    F: Fn(&A, &B) -> f32 + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "threads")]
    if a.len() >= FLOOR {
        use rayon::prelude::*;
        return a
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(ac, bc)| {
                ac.iter()
                    .zip(bc)
                    .fold(f32::INFINITY, |m, (x, y)| m.min(f(x, y)))
            })
            .fold(|| init, f32::min)
            .reduce(|| init, f32::min);
    }
    a.iter().zip(b).fold(init, |m, (x, y)| m.min(f(x, y)))
}

#[cfg(test)]
#[path = "par_tests.rs"]
mod tests;

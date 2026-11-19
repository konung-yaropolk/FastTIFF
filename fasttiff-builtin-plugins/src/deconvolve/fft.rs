//! The separable 3-D FFT everything in this directory convolves through.
//!
//! Deconvolution is convolution's inverse, and every method of doing it —
//! inverse filtering, Wiener, Richardson-Lucy, Landweber — is built from two
//! operations: blur an estimate by the PSF, and correlate a residual with it.
//! Done directly, one blur of a 512x512x64 stack by a 64x64x32 PSF is 2e14
//! multiplies. Through a pair of transforms it is a few hundred million, which
//! is the difference between a tool and a demonstration.
//!
//! # Why separable rather than a 3-D plan
//!
//! `rustfft` plans one dimension. A 3-D transform is three passes of 1-D
//! transforms — along x, then y, then z — and that is not an approximation but
//! the definition: the DFT of a product grid is the Kronecker product of the
//! per-axis transforms. The cost is the two strided passes, which walk down a
//! column and so touch a new cache line per element. Gathering each line into
//! a contiguous buffer first is what the strided passes below do, and at these
//! sizes it pays for itself several times over.
//!
//! # Across cores, without a transpose and without `unsafe`
//!
//! Each pass is many independent 1-D transforms, so the work parallelises
//! perfectly — except that the lines along y and z are *strided*, and safe
//! Rust hands out `&mut` to contiguous ranges only. The usual answers are to
//! transpose the grid (a second buffer the size of the first: another
//! gigabyte, on the stacks this is for) or to scatter raw pointers across
//! threads and argue about it.
//!
//! Neither is needed. A strided pass is "FFT every column of a row-major
//! matrix", and a block of columns is reached by splitting *every row* at the
//! same two offsets — which `split_at_mut` does, and which the compiler
//! therefore checks. Blocks of columns are disjoint by construction, each goes
//! to its own task, and nothing is copied. See [`columns`].
//!
//! # Axes of length one are skipped, not planned
//!
//! A 1-point DFT is the identity, so planning one would be correct and
//! pointless. Skipping it is what makes a 2-D problem — a single-slice image,
//! or the slice-by-slice mode — cost nothing for the axis it does not have,
//! without a separate 2-D code path to keep in step with this one.

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// A grid's extent, in voxels.
///
/// `x` is the fastest-varying axis, matching the plane layout everywhere else
/// in this crate: index `x + width * (y + height * z)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Dims {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Dims {
    pub(crate) fn new(x: usize, y: usize, z: usize) -> Self {
        Dims { x, y, z }
    }

    /// Voxels in the whole grid.
    pub(crate) fn len(self) -> usize {
        self.x * self.y * self.z
    }

    /// The flat index of `(x, y, z)`.
    pub(crate) fn at(self, x: usize, y: usize, z: usize) -> usize {
        x + self.x * (y + self.y * z)
    }
}

/// The smallest length at least `n` that factors into 2, 3, 5 and 7.
///
/// `rustfft` transforms any length, falling back to Bluestein's algorithm for
/// one with a large prime factor — correct, and several times slower. Rounding
/// up is free here: the extra samples land in the halo the padding adds anyway
/// (see [`super::grid`]), so 530 becoming 540 buys a mixed-radix transform at
/// the price of nothing.
pub(crate) fn next_fast_len(n: usize) -> usize {
    let mut m = n.max(1);
    while !is_fast(m) {
        m += 1;
    }
    m
}

fn is_fast(mut n: usize) -> bool {
    for f in [2usize, 3, 5, 7] {
        while n.is_multiple_of(f) {
            n /= f;
        }
    }
    n == 1
}

/// A planned transform for one grid size, reusable for as many buffers as the
/// caller has.
///
/// Planning is the expensive part — `rustfft` builds and caches twiddle
/// factors — and an iterative deconvolution runs hundreds of transforms of the
/// same shape, so the plan is made once and the algorithms borrow it.
pub(crate) struct Transform {
    dims: Dims,
    /// Per axis, `None` when that axis is a single point.
    fwd: [Option<Arc<dyn Fft<f32>>>; 3],
    inv: [Option<Arc<dyn Fft<f32>>>; 3],
    /// How much working space the widest of the plans asks for. Each task
    /// allocates its own: a buffer shared between them is what would make
    /// these methods need `&mut self`, and so make the whole pass
    /// single-threaded.
    scratch_len: usize,
}

impl Transform {
    pub(crate) fn new(dims: Dims) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let mut plan = |n: usize, inverse: bool| -> Option<Arc<dyn Fft<f32>>> {
            if n <= 1 {
                None
            } else if inverse {
                Some(planner.plan_fft_inverse(n))
            } else {
                Some(planner.plan_fft_forward(n))
            }
        };
        let fwd = [
            plan(dims.x, false),
            plan(dims.y, false),
            plan(dims.z, false),
        ];
        let inv = [plan(dims.x, true), plan(dims.y, true), plan(dims.z, true)];
        let scratch_len = fwd
            .iter()
            .chain(inv.iter())
            .flatten()
            .map(|f| f.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Transform {
            dims,
            fwd,
            inv,
            scratch_len,
        }
    }

    pub(crate) fn dims(&self) -> Dims {
        self.dims
    }

    /// Transform in place, unnormalised.
    pub(crate) fn forward(&self, buf: &mut [Complex32]) {
        self.run(buf, false);
    }

    /// Transform in place and divide by the voxel count, so that
    /// `inverse(forward(x))` is `x`.
    ///
    /// `rustfft` normalises neither direction — the factor is the caller's to
    /// place, and both conventions are in use. Putting the whole `1/N` here
    /// means a spectrum is always "what forward produced", which is what a
    /// regularisation parameter has to be comparable against.
    pub(crate) fn inverse(&self, buf: &mut [Complex32]) {
        self.run(buf, true);
        let scale = 1.0 / self.dims.len().max(1) as f32;
        #[cfg(feature = "threads")]
        if buf.len() >= super::par::FLOOR {
            use rayon::prelude::*;
            buf.par_iter_mut().for_each(|v| *v *= scale);
            return;
        }
        for v in buf.iter_mut() {
            *v *= scale;
        }
    }

    fn run(&self, buf: &mut [Complex32], inverse: bool) {
        debug_assert_eq!(buf.len(), self.dims.len());
        let plans = if inverse { &self.inv } else { &self.fwd };
        let (nx, ny, nz) = (self.dims.x, self.dims.y, self.dims.z);

        // x is contiguous, so each line is already a slice.
        if let Some(p) = &plans[0] {
            rows(p, buf, nx, self.scratch_len);
        }
        // y runs down the columns of each z slab, taken one slab at a time
        // because a y line does not cross from one slab into the next.
        if let Some(p) = &plans[1] {
            for slab in buf.chunks_exact_mut(nx * ny) {
                columns(p, slab, ny, nx, self.scratch_len);
            }
        }
        // z runs down the columns of the whole grid, each slab being one row.
        if let Some(p) = &plans[2] {
            columns(p, buf, nz, nx * ny, self.scratch_len);
        }
    }
}

/// Transform every contiguous row of length `n`.
fn rows(plan: &Arc<dyn Fft<f32>>, buf: &mut [Complex32], n: usize, scratch_len: usize) {
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(n).for_each_init(
            || vec![Complex32::new(0.0, 0.0); scratch_len],
            |scratch, row| plan.process_with_scratch(row, scratch),
        );
    }
    #[cfg(not(feature = "threads"))]
    {
        let mut scratch = vec![Complex32::new(0.0, 0.0); scratch_len];
        for row in buf.chunks_exact_mut(n) {
            plan.process_with_scratch(row, &mut scratch);
        }
    }
}

/// Transform every column of the row-major `n` x `w` matrix in `mat`.
///
/// This is what a strided pass is, stated so that it can be split up: the
/// lines run down the columns, and a *block of columns* is obtained by cutting
/// every row at the same two offsets. `split_at_mut` does that cut, so the
/// blocks are disjoint `&mut` slices and the compiler knows it — which is the
/// whole reason no raw pointers appear here. The grid is never copied; each
/// task walks the rows it was handed.
fn columns(
    plan: &Arc<dyn Fft<f32>>,
    mat: &mut [Complex32],
    n: usize,
    w: usize,
    scratch_len: usize,
) {
    if n <= 1 || w == 0 {
        return;
    }
    debug_assert_eq!(mat.len(), n * w);

    #[cfg(feature = "threads")]
    rayon::scope(|scope| {
        let block = block_width(w);
        // `rest` is the columns not yet handed out, as one slice per row.
        let mut rest: Vec<&mut [Complex32]> = mat.chunks_exact_mut(w).collect();
        while rest.first().is_some_and(|r| !r.is_empty()) {
            let mut heads: Vec<&mut [Complex32]> = Vec::with_capacity(rest.len());
            for row in rest.iter_mut() {
                // Take the row out so it can be split, then put the remainder
                // back. `mem::take` of a `&mut [T]` leaves an empty slice,
                // which is why this does not need an `Option`.
                let whole = std::mem::take(row);
                let (head, tail) = whole.split_at_mut(block.min(whole.len()));
                heads.push(head);
                *row = tail;
            }
            scope.spawn(move |_| transform_block(plan, &mut heads, scratch_len));
        }
    });

    #[cfg(not(feature = "threads"))]
    {
        let mut rows: Vec<&mut [Complex32]> = mat.chunks_exact_mut(w).collect();
        transform_block(plan, &mut rows, scratch_len);
    }
}

/// How wide a block of columns one task should get.
///
/// `cfg`: there are no tasks in a single-threaded build.
#[cfg(feature = "threads")]
///
/// Enough tasks to keep every core fed and not so many that the bookkeeping
/// costs more than the transform: a few per core. The floor matters more than
/// the ceiling — a task holding one column spends its life gathering.
fn block_width(w: usize) -> usize {
    #[cfg(feature = "threads")]
    let tasks = rayon::current_num_threads().max(1) * 4;
    #[cfg(not(feature = "threads"))]
    let tasks = 1usize;
    w.div_ceil(tasks.max(1)).clamp(32, 8192)
}

/// Gather each column of `rows`, transform it, and write it back.
///
/// `rows` is one block's worth: `rows.len()` is the transform length and every
/// entry is the same number of columns long.
///
/// # Why this copies through a tile
///
/// The obvious version takes one column at a time: read `rows[r][c]` for every
/// `r`, transform, write back. For the z pass of a 1080x1080x120 grid the rows
/// are nine megabytes apart, so each of those reads is its own cache line
/// *and* its own page — a hundred and twenty TLB misses per column, a million
/// columns. It is the slowest way to touch memory that still looks like a
/// loop.
///
/// Copying a tile of columns in first makes every access to the grid
/// contiguous: one `copy_from_slice` per row on the way in and one on the way
/// out, with all the strided work confined to a tile small enough to sit in
/// L2. The arithmetic is identical; only the order of the memory traffic
/// changes, and on the stacks this is for that is most of the run time.
fn transform_block(plan: &Arc<dyn Fft<f32>>, rows: &mut [&mut [Complex32]], scratch_len: usize) {
    let n = rows.len();
    let w = rows.first().map(|r| r.len()).unwrap_or(0);
    if n <= 1 || w == 0 {
        return;
    }
    let tile_w = tile_width(n).min(w);
    let mut tile = vec![Complex32::new(0.0, 0.0); n * tile_w];
    let mut line = vec![Complex32::new(0.0, 0.0); n];
    let mut scratch = vec![Complex32::new(0.0, 0.0); scratch_len];

    let mut c0 = 0;
    while c0 < w {
        let span = tile_w.min(w - c0);
        for (r, row) in rows.iter().enumerate() {
            tile[r * tile_w..r * tile_w + span].copy_from_slice(&row[c0..c0 + span]);
        }
        for j in 0..span {
            for (r, slot) in line.iter_mut().enumerate() {
                *slot = tile[r * tile_w + j];
            }
            plan.process_with_scratch(&mut line, &mut scratch);
            for (r, v) in line.iter().enumerate() {
                tile[r * tile_w + j] = *v;
            }
        }
        for (r, row) in rows.iter_mut().enumerate() {
            row[c0..c0 + span].copy_from_slice(&tile[r * tile_w..r * tile_w + span]);
        }
        c0 += span;
    }
}

/// How many columns to copy in at once, so the tile stays in cache.
///
/// Sized for about 128 KB — comfortably inside a modern L2 and well clear of
/// what the line and scratch buffers also want.
fn tile_width(n: usize) -> usize {
    (8192 / n.max(1)).clamp(8, 256)
}

/// Apply `f` to each `(a, b)` pair, across cores when the slice is long enough
/// to pay for it.
///
/// The threshold is [`super::par::FLOOR`] — one definition for the whole
/// directory, since the reasoning behind it is the same here as there.
pub(crate) fn zip_each<F>(a: &mut [Complex32], b: &[Complex32], f: F)
where
    F: Fn(&mut Complex32, &Complex32) + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "threads")]
    if a.len() >= super::par::FLOOR {
        use rayon::prelude::*;
        a.par_iter_mut()
            .zip(b.par_iter())
            .for_each(|(x, y)| f(x, y));
        return;
    }
    for (x, y) in a.iter_mut().zip(b) {
        f(x, y);
    }
}

/// `a *= b`, elementwise — convolution, in the spectral domain.
pub(crate) fn multiply(a: &mut [Complex32], b: &[Complex32]) {
    zip_each(a, b, |x, y| *x *= *y);
}

/// `a *= conj(b)`, elementwise — correlation, which is what the adjoint of a
/// convolution is.
///
/// Every iterative method here needs both: the forward model blurs, and
/// pushing a residual back through it correlates. They differ only by this
/// conjugate — which is also why a test built on a symmetric PSF cannot tell a
/// correct adjoint from a missing one, and why `algorithms_tests` uses one
/// that is not symmetric.
pub(crate) fn multiply_conj(a: &mut [Complex32], b: &[Complex32]) {
    zip_each(a, b, |x, y| *x *= y.conj());
}

/// Copy real samples into a complex buffer.
pub(crate) fn lift(src: &[f32], dst: &mut [Complex32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(feature = "threads")]
    if src.len() >= super::par::FLOOR {
        use rayon::prelude::*;
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(d, s)| *d = Complex32::new(*s, 0.0));
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = Complex32::new(*s, 0.0);
    }
}

/// Take the real part back out.
pub(crate) fn lower(src: &[Complex32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(feature = "threads")]
    if src.len() >= super::par::FLOOR {
        use rayon::prelude::*;
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(d, s)| *d = s.re);
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.re;
    }
}

#[cfg(test)]
#[path = "fft_tests.rs"]
mod tests;

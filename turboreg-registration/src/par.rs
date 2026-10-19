// Copyright (C) 2026 SciWare LLC
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See the LICENSE file at the root of this crate.

//! Parallelism, or the lack of it.
//!
//! Every parallel loop in this crate goes through `use crate::par::*` rather
//! than `use rayon::prelude::*`. With the `threads` feature on — which is the
//! default, and what every desktop build uses — that import *is* rayon's
//! prelude and nothing has changed. With it off, the same names resolve to
//! sequential look-alikes defined here, and rayon is not compiled at all.
//!
//! # Why the crate can be built without threads
//!
//! `wasm32-unknown-unknown` has no threads. rayon does build there — it
//! documents a fallback that runs a `par_iter` sequentially on the calling
//! thread rather than failing — so this is not a crate that *could not* be
//! compiled for a browser. It is a crate that should not carry rayon there,
//! for two reasons.
//!
//! The first is honesty. Under the fallback, [`Backend::MultiThread`] is a
//! label with nothing behind it: the run is single-threaded and says it is not.
//! That is the same complaint [`Backend::Gpu`] already answers by reporting
//! itself unavailable, and a slow run indistinguishable from a correct one is
//! precisely what that mechanism exists to prevent. With the feature off,
//! `MultiThread` says so instead.
//!
//! The second is weight. A thread pool, a work-stealing deque and a sleep/wake
//! state machine are a large thing to link into a bundle shipped over the
//! network when none of it can ever run.
//!
//! [`Backend::Gpu`]: crate::Backend::Gpu
//! [`Backend::MultiThread`]: crate::Backend::MultiThread
//!
//! Turning threads off does not change what the crate *computes*.
//! Every call site here already had a sequential path — [`Backend::SingleThread`]
//! is a backend a user can pick, and it exists precisely so a result can be
//! reproduced without scheduling in the picture. This module only makes that
//! path the one the compiler can see, so that the parallel one need not exist.
//!
//! [`Backend::SingleThread`]: crate::Backend::SingleThread
//!
//! # What the sequential shapes guarantee
//!
//! Nothing here reorders anything. `par_iter` becomes `iter`, `par_chunks`
//! becomes `chunks`, and each produces its items in index order — which rayon's
//! *indexed* iterators also promise, and which is why `zip` against a second
//! sequence is sound in both. The results are identical, not merely equivalent,
//! and the reason is worth stating because it is a rule rather than luck.
//!
//! Most reductions here accumulate inside one item of a parallel loop and
//! never across items. The one that does cross items — the optimiser's sum of
//! squared differences, taken over chunks of rows — splits its work into
//! pieces fixed by the *image*, never by the thread count, collects the partial
//! sums in piece order with [`map_in_order`], and adds them in that order on
//! the calling thread. The single-threaded backend runs the same pieces in the
//! same order. So there is no sum whose association changes with the number of
//! workers, and "Single-thread CPU" and "Multi-thread CPU" give bit-identical
//! answers rather than answers that agree to a tolerance.

/// With threads, this is rayon.
#[cfg(feature = "threads")]
pub use rayon::prelude::*;

/// How many workers a parallel loop should plan for.
///
/// A scheduling hint, not a correctness one: it decides how work is divided,
/// never what the division computes. Without threads there is one worker, and
/// the chunked loops collapse to a single chunk.
#[cfg(feature = "threads")]
pub fn workers() -> usize {
    rayon::current_num_threads().max(1)
}

/// See the threaded one above.
#[cfg(not(feature = "threads"))]
pub fn workers() -> usize {
    1
}

/// `par_iter` and `par_chunks` over a slice, done on this thread.
///
/// Deliberately *not* generic over "anything parallel-iterable": the point is
/// to type-check the handful of call sites this crate has, with std's own
/// iterators as the return types, rather than to reimplement rayon.
#[cfg(not(feature = "threads"))]
pub trait ParallelSlice<T> {
    /// Each element in turn, in index order.
    fn par_iter(&self) -> std::slice::Iter<'_, T>;
    /// Runs of `size` elements, the last one short.
    fn par_chunks(&self, size: usize) -> std::slice::Chunks<'_, T>;
}

#[cfg(not(feature = "threads"))]
impl<T> ParallelSlice<T> for [T] {
    fn par_iter(&self) -> std::slice::Iter<'_, T> {
        self.iter()
    }
    fn par_chunks(&self, size: usize) -> std::slice::Chunks<'_, T> {
        self.chunks(size)
    }
}

/// The `&mut` half of [`ParallelSlice`].
#[cfg(not(feature = "threads"))]
pub trait ParallelSliceMut<T> {
    /// Each element in turn, mutably, in index order.
    fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, T>;
}

#[cfg(not(feature = "threads"))]
impl<T> ParallelSliceMut<T> for [T] {
    fn par_iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.iter_mut()
    }
}

/// rayon's `flat_map_iter`, which is `flat_map` once there is one thread.
///
/// The distinction rayon draws — that the *inner* iterator stays sequential —
/// is the only shape there is here, so the two coincide.
#[cfg(not(feature = "threads"))]
pub trait FlatMapIter: Iterator + Sized {
    /// Map each item to a sequence and concatenate them, in order.
    fn flat_map_iter<U, F>(self, f: F) -> std::iter::FlatMap<Self, U, F>
    where
        U: IntoIterator,
        F: FnMut(Self::Item) -> U,
    {
        self.flat_map(f)
    }
}

#[cfg(not(feature = "threads"))]
impl<I: Iterator> FlatMapIter for I {}

/// `f` over each of `items`, on every core or on this one, results in input
/// order either way.
///
/// The switch is a runtime flag rather than the feature alone because the
/// backend is the *user's* choice: a threaded build still has to honour
/// "Single-thread CPU" when it is picked. With the feature off there is only
/// the sequential path and the flag is ignored.
///
/// Order is what makes this safe for reductions. A caller that splits its work
/// into a fixed set of pieces — fixed by the data, never by the thread count —
/// gets the partial results back in piece order and combines them in that
/// order, so the floating-point sum is the same sum on one thread or sixteen.
pub fn map_in_order<T, R, F>(items: &[T], parallel: bool, f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    #[cfg(feature = "threads")]
    {
        if parallel {
            return items.par_iter().map(&f).collect();
        }
    }
    let _ = parallel;
    items.iter().map(f).collect()
}

/// `f(index, chunk)` over consecutive `size`-long chunks of `buf`, on every
/// core or on this one.
///
/// For work whose outputs do not interact — a row of a resampled image, a run
/// of pixels of a mean — so there is no combining step and nothing whose order
/// could matter. `size` of zero is treated as one rather than panicking.
pub fn for_each_chunk_mut<T, F>(buf: &mut [T], size: usize, parallel: bool, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let size = size.max(1);
    #[cfg(feature = "threads")]
    {
        if parallel {
            buf.par_chunks_mut(size)
                .enumerate()
                .for_each(|(i, c)| f(i, c));
            return;
        }
    }
    let _ = parallel;
    buf.chunks_mut(size).enumerate().for_each(|(i, c)| f(i, c));
}

/// [`for_each_chunk_mut`] over two buffers in step: chunk `i` of `a` and chunk
/// `i` of `b` go to the same call. For an output that comes in pairs, like the
/// two gradient images, computed together because they share their inputs.
pub fn for_each_chunk_pair_mut<T, F>(a: &mut [T], b: &mut [T], size: usize, parallel: bool, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T], &mut [T]) + Sync + Send,
{
    let size = size.max(1);
    #[cfg(feature = "threads")]
    {
        if parallel {
            a.par_chunks_mut(size)
                .zip(b.par_chunks_mut(size))
                .enumerate()
                .for_each(|(i, (ca, cb))| f(i, ca, cb));
            return;
        }
    }
    let _ = parallel;
    a.chunks_mut(size)
        .zip(b.chunks_mut(size))
        .enumerate()
        .for_each(|(i, (ca, cb))| f(i, ca, cb));
}

#[cfg(test)]
#[path = "par_tests.rs"]
mod tests;

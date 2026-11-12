//! The one kind of background worker the viewer has: the newest request wins,
//! and the work it overtook stops.
//!
//! Three things prepare data off the interface thread — the read-ahead
//! ([`crate::prefetch::Prefetcher`]), the window of an over-large frame
//! ([`crate::window::WindowWorker`]) and the 3D volume
//! ([`crate::volume::VolumeBuilder`]) — and all three want the same thing from
//! a thread. The interface asks once a frame, the question moves faster than
//! the answers arrive, and only the answer to the latest question is worth
//! having. Each used to carry its own copy of that loop: a channel drained to
//! its newest message, a result slot behind a lock, a thread never joined, and
//! a stub for builds without threads. None of them could stop work that had
//! been overtaken. During a 4D scrub the wanted timepoint waited for the stale
//! build ahead of it to finish, and a closed file's volume went on being built,
//! holding the file's mapping, after the window had let go of both.
//! `LatestWorker`, private to the crate, is that loop written once, with the
//! stopping added. What this module makes public is what its jobs see: the
//! [`CancelToken`] they are handed and the [`Cancelled`] they stop with.
//!
//! # Latest wins
//!
//! A worker holds at most one request that has not started. `request` puts its
//! request there, replacing whatever was waiting, so a request overtaken before
//! it started is never started at all. It also cancels the job that is
//! running. A finished result waits in a slot until `take_if` is handed a key
//! it is the answer to; asking with any other key leaves it there.
//!
//! Asking again for the key asked for last does nothing while that request is
//! still being answered (waiting, running, or finished and not yet taken). The
//! interface asks every frame, and an identical request must not cancel the
//! very job it asks for.
//!
//! While the question changes faster than a job takes — 4D playback faster
//! than a volume builds — no job finishes until it settles. That loses
//! nothing: each caller takes only an answer to what it is asking *now* (the
//! exact `(generation, time)` for a volume; for a window, one that serves the
//! current view, and a new one is asked for only when the one in progress
//! would not), so the overtaken jobs, run to the end as they used to be,
//! produced answers nobody would take. Stopping them is what lets the settled
//! question start at once.
//!
//! # The epoch
//!
//! Cancelling is one counter that a worker and its jobs share. Every new
//! request moves it on, and so does dropping the worker. A job's
//! [`CancelToken`] remembers the value it was issued at and is cancelled once
//! the two differ, so a check is one atomic load. That is cheap enough for a
//! job to make at each natural step — a slice of a volume, a band of a window,
//! a frame of read-ahead — and it should check nowhere finer: no per-pixel or
//! per-sample atomics. The work between two checks is what a superseded job
//! still costs. A job that sees the cancel returns early (with [`Cancelled`],
//! or nothing). Whatever a cancelled job produced is dropped on the worker
//! thread and never published, so an overtaken answer cannot be taken by
//! mistake, and a volume of gigabytes is never freed on the interface thread.
//!
//! # Dropping a worker
//!
//! Drop cancels the running job, discards the waiting request and lets the
//! thread go. It does not join it. The thread stops at the job's next check,
//! but one step can be long: a slice of a volume over a gigapixel mosaic is a
//! whole frame's decode, seconds of it, and the interface thread closing a
//! file must not wait that long. Nobody wants anything the thread still holds.
//! It drops the job, and with it the job's `Arc<TiffStack>` (the file's
//! mapping), together with any result it was keeping, as it exits, one step
//! after the drop.
//!
//! # Without threads
//!
//! With the `threads` feature off (a browser), `LatestWorker::spawn` returns
//! `None`. That is also what it returns when a spawn fails, and every caller
//! already does the work inline then. The type has no values in that build, so
//! the workers built on it need no stubs of their own. [`CancelToken::never`]
//! is the token for inline work, which nothing can overtake.

/// A job's way of asking whether its answer is still wanted.
///
/// Cheap to clone and to ask: one atomic load. Ask once per step of the job
/// (see the module docs), never per sample.
#[derive(Clone, Debug)]
pub struct CancelToken(Source);

#[derive(Clone, Debug)]
enum Source {
    /// Nothing cancels it: work run inline, which nothing can overtake.
    Never,
    /// Cancelled once the worker's epoch has moved past the value this was
    /// issued at.
    #[cfg(feature = "threads")]
    Epoch {
        epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
        issued: u64,
    },
    /// Tests only: says "go on" `allow` times and "stop" after that, counting
    /// every time it is asked. This is how a test cancels a job at a known
    /// step without a thread or a sleep.
    #[cfg(test)]
    After {
        allow: usize,
        asked: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    },
}

impl CancelToken {
    /// A token that is never cancelled, for running a job inline.
    pub fn never() -> Self {
        CancelToken(Source::Never)
    }

    #[cfg(feature = "threads")]
    fn issued(epoch: &std::sync::Arc<std::sync::atomic::AtomicU64>, issued: u64) -> Self {
        CancelToken(Source::Epoch {
            epoch: std::sync::Arc::clone(epoch),
            issued,
        })
    }

    /// Whether the job has been overtaken by a newer request or its worker
    /// dropped, so its answer will not be used.
    pub fn is_cancelled(&self) -> bool {
        match &self.0 {
            Source::Never => false,
            #[cfg(feature = "threads")]
            Source::Epoch { epoch, issued } => {
                epoch.load(std::sync::atomic::Ordering::Relaxed) != *issued
            }
            #[cfg(test)]
            Source::After { allow, asked } => {
                asked.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= *allow
            }
        }
    }

    /// [`is_cancelled`](Self::is_cancelled) as a `Result`, for a job's `?`.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// A token that lets a job take `allow` steps and cancels it at the next,
    /// with the count of times it was asked.
    #[cfg(test)]
    pub(crate) fn after(allow: usize) -> (Self, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            CancelToken(Source::After {
                allow,
                asked: std::sync::Arc::clone(&asked),
            }),
            asked,
        )
    }
}

/// What a job returns when it stopped because its [`CancelToken`] was
/// cancelled. Not a failure: nobody wanted the answer any more, so there is
/// nothing to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled: a newer request overtook it")
    }
}

impl std::error::Error for Cancelled {}

/// What becomes of a finished result nobody has taken when the worker starts
/// on a newer request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Unclaimed {
    /// Kept until it is taken or a newer result replaces it: it may still be
    /// what the caller wants. A window built for a view the user has left can
    /// serve the view they come back to.
    Keep,
    /// Dropped before the newer job starts, so the worker never holds two
    /// results at once. Right for a volume, which can be gigabytes, and which
    /// the caller has stopped wanting by the time it asks for another.
    Discard,
}

#[cfg(feature = "threads")]
pub(crate) use threaded::LatestWorker;

#[cfg(feature = "threads")]
mod threaded {
    use super::{CancelToken, Unclaimed};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

    /// A background thread that answers the newest request and drops the
    /// rest. See the [module docs](super).
    ///
    /// `K` says what a request is for: it is compared against the last one to
    /// skip a repeat, kept with the result, and shown to
    /// [`take_if`](Self::take_if) to decide whether the result is the answer
    /// wanted now. `Req` is what the job needs to do its work; `Out` is what it
    /// produces.
    pub(crate) struct LatestWorker<K, Req, Out> {
        shared: Arc<Shared<K, Req, Out>>,
    }

    struct Shared<K, Req, Out> {
        state: Mutex<State<K, Req, Out>>,
        /// Signalled when a request arrives or the worker is dropped.
        wake: Condvar,
        /// Moved on by every request and by drop; see [`CancelToken`].
        epoch: Arc<AtomicU64>,
        unclaimed: Unclaimed,
    }

    struct State<K, Req, Out> {
        /// The newest request not yet started, with the epoch it moved to.
        queued: Option<(u64, K, Req)>,
        /// The key of the newest request, until its result is taken. What a
        /// repeat is recognised by.
        last: Option<K>,
        /// The newest finished result not yet taken, with its key.
        done: Option<(K, Out)>,
        /// Set by drop: the thread exits at its next look.
        closed: bool,
        /// Cleared when the thread ends, whether it was dropped or a job
        /// panicked.
        alive: bool,
    }

    impl<K, Req, Out> Shared<K, Req, Out> {
        /// The state, whatever a panic elsewhere left in the lock. Nothing
        /// runs a job while holding it, so a poisoned lock still holds
        /// consistent state.
        fn lock(&self) -> MutexGuard<'_, State<K, Req, Out>> {
            self.state.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    impl<K, Req, Out> LatestWorker<K, Req, Out>
    where
        K: PartialEq + Clone + Send + 'static,
        Req: Send + 'static,
        Out: Send + 'static,
    {
        /// Start a thread named `name` that runs `job` on each request it
        /// takes: `job(key, request, cancel)` returns the result to publish,
        /// or `None` for nothing (a failure it has reported itself, or a
        /// cancel). `job` owns whatever it needs across requests, such as the
        /// stack and a cache, and the thread drops it when it exits.
        ///
        /// `None` when the thread could not be started; the caller then does
        /// the work inline.
        pub fn spawn<F>(name: &str, unclaimed: Unclaimed, job: F) -> Option<Self>
        where
            F: FnMut(&K, Req, &CancelToken) -> Option<Out> + Send + 'static,
        {
            let shared = Arc::new(Shared {
                state: Mutex::new(State {
                    queued: None,
                    last: None,
                    done: None,
                    closed: false,
                    alive: true,
                }),
                wake: Condvar::new(),
                epoch: Arc::new(AtomicU64::new(0)),
                unclaimed,
            });
            let worker = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(name.to_owned())
                .spawn(move || run(&worker, job))
                .ok()?;
            Some(LatestWorker { shared })
        }

        /// Ask for `key`, computed from `req`. Replaces any request still
        /// waiting and cancels the job running.
        ///
        /// Does nothing when `key` is what was asked for last and its result
        /// has not been taken: the request is waiting, running, or finished.
        /// A job that finished with nothing counts as finished too, so a
        /// failure is not retried until something else has been asked for.
        ///
        /// `false` when the thread is gone (a job panicked); the caller should
        /// do the work inline.
        pub fn request(&self, key: K, req: Req) -> bool {
            let mut s = self.shared.lock();
            if !s.alive {
                return false;
            }
            if s.last.as_ref() == Some(&key) {
                return true;
            }
            // Under the lock, so the worker's publish check sees either the
            // old epoch and no newer request, or this one.
            let issued = self.shared.epoch.fetch_add(1, Ordering::Relaxed) + 1;
            s.last = Some(key.clone());
            let replaced = s.queued.replace((issued, key, req));
            drop(s);
            self.shared.wake.notify_one();
            drop(replaced);
            true
        }

        /// Take the finished result if `wanted` says its key is the answer to
        /// what is wanted now; otherwise leave it in place and return `None`.
        /// `wanted` runs under the worker's lock, so keep it a comparison.
        pub fn take_if(&self, wanted: impl FnOnce(&K) -> bool) -> Option<Out> {
            let mut s = self.shared.lock();
            if !s.done.as_ref().is_some_and(|(key, _)| wanted(key)) {
                return None;
            }
            let (key, out) = s.done.take()?;
            // Answered: asking for it again is a new request, not a repeat.
            if s.last.as_ref() == Some(&key) {
                s.last = None;
            }
            Some(out)
        }
    }

    impl<K, Req, Out> Drop for LatestWorker<K, Req, Out> {
        /// Cancel the running job, discard the waiting request and let the
        /// thread go without joining it; see the module docs for why.
        fn drop(&mut self) {
            let queued = {
                let mut s = self.shared.lock();
                s.closed = true;
                self.shared.epoch.fetch_add(1, Ordering::Relaxed);
                s.queued.take()
            };
            self.shared.wake.notify_one();
            drop(queued);
        }
    }

    /// The thread: wait for a request, run it, publish what it made unless it
    /// was cancelled meanwhile, and repeat until dropped.
    fn run<K, Req, Out>(
        shared: &Shared<K, Req, Out>,
        mut job: impl FnMut(&K, Req, &CancelToken) -> Option<Out>,
    ) {
        let _exit = Exit(shared);
        loop {
            let (key, req, token, stale) = {
                let mut s = shared.lock();
                let (issued, key, req) = loop {
                    if s.closed {
                        drop(s);
                        return;
                    }
                    if let Some(next) = s.queued.take() {
                        break next;
                    }
                    s = shared.wake.wait(s).unwrap_or_else(PoisonError::into_inner);
                };
                let stale = match shared.unclaimed {
                    Unclaimed::Discard => s.done.take(),
                    Unclaimed::Keep => None,
                };
                (key, req, CancelToken::issued(&shared.epoch, issued), stale)
            };
            // Off the lock: a volume's worth of memory takes a moment to free.
            drop(stale);

            let out = if token.is_cancelled() {
                None
            } else {
                job(&key, req, &token)
            };

            let mut s = shared.lock();
            // Decided under the lock that `request` moves the epoch under, so
            // a newer request either came first (and this is dropped) or comes
            // after it is published.
            let unwanted = match out {
                Some(out) if !token.is_cancelled() => s.done.replace((key, out)).map(|(_, o)| o),
                cancelled => cancelled,
            };
            drop(s);
            drop(unwanted);
        }
    }

    /// Marks the thread gone however it ends, a panic in a job included, and
    /// lets go of what it held there and then rather than when the last handle
    /// does.
    struct Exit<'a, K, Req, Out>(&'a Shared<K, Req, Out>);

    impl<K, Req, Out> Drop for Exit<'_, K, Req, Out> {
        fn drop(&mut self) {
            let mut s = self.0.lock();
            s.alive = false;
            let leftovers = (s.queued.take(), s.done.take());
            drop(s);
            drop(leftovers);
        }
    }
}

/// Without threads there is no worker: [`spawn`](Self::spawn) returns `None`,
/// exactly as for a thread that failed to start, and the callers do the work
/// inline. The type has no values, so the methods below can never run.
#[cfg(not(feature = "threads"))]
pub(crate) struct LatestWorker<K, Req, Out> {
    never: std::convert::Infallible,
    _types: std::marker::PhantomData<(K, Req, Out)>,
}

#[cfg(not(feature = "threads"))]
impl<K, Req, Out> LatestWorker<K, Req, Out>
where
    K: PartialEq + Clone + Send + 'static,
    Req: Send + 'static,
    Out: Send + 'static,
{
    /// Always `None`: there is no thread to start.
    pub fn spawn<F>(_name: &str, _unclaimed: Unclaimed, _job: F) -> Option<Self>
    where
        F: FnMut(&K, Req, &CancelToken) -> Option<Out> + Send + 'static,
    {
        None
    }

    pub fn request(&self, _key: K, _req: Req) -> bool {
        match self.never {}
    }

    pub fn take_if(&self, _wanted: impl FnOnce(&K) -> bool) -> Option<Out> {
        match self.never {}
    }
}

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;

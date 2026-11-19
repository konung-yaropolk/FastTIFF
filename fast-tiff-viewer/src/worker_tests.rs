//! The latest-wins worker: which requests are answered, which are stopped, and
//! what a dropped worker lets go of.
//!
//! The jobs here are synthetic, so every interleaving that matters is forced
//! with channels: a job announces the step it reached and waits to be let go,
//! and the test acts in between. The only waiting by polling is for a result
//! to be published (or a thread to exit) after the job has already returned,
//! and that is bounded so a broken worker fails the suite rather than hanging
//! it.

use super::*;
use std::sync::atomic::Ordering;

#[test]
fn a_token_for_inline_work_is_never_cancelled() {
    let token = CancelToken::never();
    assert!(!token.is_cancelled());
    assert_eq!(token.check(), Ok(()));
}

#[test]
fn a_test_token_allows_the_steps_it_was_given() {
    let (token, asked) = CancelToken::after(2);
    assert!(!token.is_cancelled());
    assert!(!token.is_cancelled());
    assert!(token.is_cancelled());
    assert_eq!(token.check(), Err(Cancelled));
    assert_eq!(asked.load(Ordering::Relaxed), 4);
}

#[cfg(feature = "threads")]
mod threaded {
    use super::super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};

    /// Poll `f` until it gives something, for up to five seconds.
    fn eventually<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(5) {
            if let Some(v) = f() {
                return v;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("{what} did not happen within five seconds");
    }

    /// Whether a finished result is waiting, without taking it.
    fn published<K, Req, Out>(worker: &LatestWorker<K, Req, Out>) -> bool
    where
        K: PartialEq + Clone + Send + 'static,
        Req: Send + 'static,
        Out: Send + 'static,
    {
        let mut there = false;
        let taken = worker.take_if(|_| {
            there = true;
            false
        });
        assert!(taken.is_none());
        there
    }

    /// Requests made while the worker is busy replace one another: only the
    /// newest starts, and only its answer can be taken.
    #[test]
    fn of_three_requests_in_a_row_only_the_last_is_answered() {
        let (started_tx, started) = mpsc::channel::<u32>();
        let (go, wait) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn(
            "test-latest",
            Unclaimed::Keep,
            move |&key: &u32, (): (), cancel: &CancelToken| {
                started_tx.send(key).unwrap();
                if key == 0 {
                    // Keep the thread busy while the test queues the others.
                    wait.recv().unwrap();
                }
                (!cancel.is_cancelled()).then_some(key * 10)
            },
        )
        .expect("the thread should spawn");

        assert!(worker.request(0, ()));
        assert_eq!(started.recv().unwrap(), 0);
        for key in 1..=3 {
            assert!(worker.request(key, ()));
        }
        go.send(()).unwrap();

        let got = eventually("the last request's answer", || worker.take_if(|&k| k == 3));
        assert_eq!(got, 30);
        // 1 and 2 were replaced before they started; 0 was overtaken while it
        // ran, so it published nothing.
        assert_eq!(started.try_iter().collect::<Vec<_>>(), vec![3]);
        assert!(!published(&worker), "nothing else was published");
    }

    /// A job overtaken while it runs stops at its next check, not at its end.
    #[test]
    fn a_newer_request_stops_the_job_it_overtakes() {
        const STEPS: usize = 1000;
        let steps = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&steps);
        let (first_tx, first) = mpsc::channel::<()>();
        let (go, wait) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn(
            "test-overtaken",
            Unclaimed::Keep,
            move |&key: &u32, (): (), cancel: &CancelToken| {
                for step in 0..STEPS {
                    if cancel.is_cancelled() {
                        return None;
                    }
                    if key == 1 {
                        counted.fetch_add(1, Ordering::Relaxed);
                        if step == 0 {
                            first_tx.send(()).unwrap();
                            wait.recv().unwrap();
                        }
                    }
                }
                Some(key)
            },
        )
        .expect("the thread should spawn");

        worker.request(1, ());
        first.recv().unwrap();
        // Overtake it while it is part-way through, then let it go on.
        worker.request(2, ());
        go.send(()).unwrap();

        assert_eq!(
            eventually("the newer answer", || worker.take_if(|&k| k == 2)),
            2
        );
        assert_eq!(
            steps.load(Ordering::Relaxed),
            1,
            "the overtaken job took one of its {STEPS} steps"
        );
        assert!(!published(&worker), "and published nothing");
    }

    /// A job that never looks at its token, as read-ahead's decode does not,
    /// still cannot publish an answer it was overtaken while computing: the
    /// worker drops it. With [`Unclaimed::Keep`] a stale answer, once
    /// published, would sit in the slot until the newer one replaced it, so the
    /// slot is looked at while the newer job is held part-way.
    #[test]
    fn an_answer_overtaken_while_computed_is_never_published() {
        let (started_tx, started) = mpsc::channel::<u32>();
        let (go, wait) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn(
            "test-unpublished",
            Unclaimed::Keep,
            move |&key: &u32, (): (), _: &CancelToken| {
                started_tx.send(key).unwrap();
                wait.recv().unwrap();
                Some(key)
            },
        )
        .expect("the thread should spawn");

        worker.request(1, ());
        assert_eq!(started.recv().unwrap(), 1);
        worker.request(2, ());
        // Job 1 returns its answer, overtaken; the thread then starts job 2,
        // and by then whatever it did with job 1's answer is done.
        go.send(()).unwrap();
        assert_eq!(started.recv().unwrap(), 2);
        let mut seen = Vec::new();
        assert!(worker
            .take_if(|&k| {
                seen.push(k);
                false
            })
            .is_none());
        assert_eq!(seen, Vec::<u32>::new(), "job 1's answer was dropped");

        go.send(()).unwrap();
        assert_eq!(
            eventually("the newer answer", || worker.take_if(|&k| k == 2)),
            2
        );
        assert!(!published(&worker), "and nothing else was published");
    }

    /// The interface asks every frame. Asking again for what is being built
    /// must not cancel it, and once the answer is taken asking again is a new
    /// request.
    #[test]
    fn asking_again_for_the_same_thing_does_not_restart_it() {
        let runs = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&runs);
        let (first_tx, first) = mpsc::channel::<()>();
        let (go, wait) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn(
            "test-repeat",
            Unclaimed::Keep,
            move |&key: &u32, (): (), cancel: &CancelToken| {
                if counted.fetch_add(1, Ordering::Relaxed) == 0 {
                    first_tx.send(()).unwrap();
                    wait.recv().unwrap();
                }
                (!cancel.is_cancelled()).then_some(key)
            },
        )
        .expect("the thread should spawn");

        worker.request(7, ());
        first.recv().unwrap();
        for _ in 0..5 {
            assert!(worker.request(7, ()));
        }
        go.send(()).unwrap();
        assert_eq!(eventually("the answer", || worker.take_if(|&k| k == 7)), 7);
        assert_eq!(runs.load(Ordering::Relaxed), 1, "built once, not restarted");

        // Answered and taken: the same key again is built again.
        worker.request(7, ());
        assert_eq!(
            eventually("the second answer", || worker.take_if(|&k| k == 7)),
            7
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    /// A result is handed back only for the key it answers, and a refusal
    /// leaves it where it is.
    #[test]
    fn a_result_waits_for_its_own_key() {
        let worker = LatestWorker::spawn(
            "test-key",
            Unclaimed::Keep,
            |&key: &u32, (): (), _: &CancelToken| Some(key),
        )
        .expect("the thread should spawn");
        worker.request(4, ());
        eventually("the result", || published(&worker).then_some(()));
        assert!(worker.take_if(|&k| k == 5).is_none());
        assert_eq!(worker.take_if(|&k| k == 4), Some(4));
        assert!(worker.take_if(|&k| k == 4).is_none(), "taken once");
    }

    /// Dropping a worker mid-job does not wait for the job, and its thread
    /// lets go of what the job holds — the stack, in the viewer — at the job's
    /// next check.
    #[test]
    fn dropping_a_worker_mid_job_lets_go_of_what_it_holds() {
        const STEPS: usize = 1000;
        let held = Arc::new(vec![0u8; 64]);
        let weak = Arc::downgrade(&held);
        let steps = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&steps);
        let (first_tx, first) = mpsc::channel::<()>();
        let (go, wait) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn(
            "test-drop",
            Unclaimed::Keep,
            move |_: &u32, (): (), cancel: &CancelToken| {
                for step in 0..STEPS {
                    if cancel.is_cancelled() {
                        return None;
                    }
                    counted.fetch_add(1, Ordering::Relaxed);
                    if step == 0 {
                        first_tx.send(()).unwrap();
                        wait.recv().unwrap();
                    }
                }
                Some(held.len())
            },
        )
        .expect("the thread should spawn");

        worker.request(0, ());
        first.recv().unwrap();
        // The job is blocked on `go` right now, so a drop that joined the
        // thread would never return.
        drop(worker);
        go.send(()).unwrap();

        eventually("the thread letting go of what the job held", || {
            weak.upgrade().is_none().then_some(())
        });
        assert_eq!(
            steps.load(Ordering::Relaxed),
            1,
            "it stopped at its next check, {STEPS} steps short"
        );
    }

    /// A finished result nobody took goes with the worker, not with the
    /// process.
    #[test]
    fn dropping_a_worker_lets_go_of_a_result_nobody_took() {
        let worker = LatestWorker::spawn(
            "test-unclaimed",
            Unclaimed::Keep,
            |_: &u32, out: Arc<Vec<u8>>, _: &CancelToken| Some(out),
        )
        .expect("the thread should spawn");
        let out = Arc::new(vec![0u8; 64]);
        let weak = Arc::downgrade(&out);
        worker.request(0, out);
        eventually("the result", || published(&worker).then_some(()));
        drop(worker);
        eventually("the result being dropped", || {
            weak.upgrade().is_none().then_some(())
        });
    }

    /// [`Unclaimed::Discard`] drops a result nobody took before the next job
    /// starts, so the worker never holds two.
    #[test]
    fn a_discarding_worker_never_holds_two_results() {
        type Req = (Arc<Vec<u8>>, Option<std::sync::Weak<Vec<u8>>>);
        let (report, reported) = mpsc::channel::<bool>();
        let worker = LatestWorker::spawn(
            "test-discard",
            Unclaimed::Discard,
            move |_: &u32, (out, previous): Req, _: &CancelToken| {
                if let Some(previous) = previous {
                    report.send(previous.upgrade().is_none()).unwrap();
                }
                Some(out)
            },
        )
        .expect("the thread should spawn");
        let first = Arc::new(vec![1u8; 64]);
        let first_weak = Arc::downgrade(&first);
        worker.request(0, (first, None));
        eventually("the first result", || published(&worker).then_some(()));
        worker.request(1, (Arc::new(vec![2u8; 64]), Some(first_weak)));
        assert!(
            reported.recv().unwrap(),
            "the first result was gone before the second job started"
        );
    }

    /// A worker whose job panicked refuses requests, so the caller knows to do
    /// the work itself.
    #[test]
    fn a_worker_whose_job_panicked_says_so() {
        let worker = LatestWorker::spawn(
            "test-panic",
            Unclaimed::Keep,
            |_: &u32, (): (), _: &CancelToken| -> Option<()> {
                panic!("a job that panics, on purpose");
            },
        )
        .expect("the thread should spawn");
        assert!(worker.request(0, ()));
        let mut key = 0;
        eventually("a refused request", || {
            key += 1;
            (!worker.request(key, ())).then_some(())
        });
    }

    #[test]
    fn an_idle_worker_exits_when_dropped() {
        // Nothing asked: the thread is waiting, and the drop wakes it to exit.
        let held = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&held);
        let worker = LatestWorker::spawn(
            "test-idle",
            Unclaimed::Keep,
            move |_: &u32, (): (), _: &CancelToken| Some(held.load(Ordering::Relaxed)),
        )
        .expect("the thread should spawn");
        drop(worker);
        eventually("an idle thread exiting", || {
            weak.upgrade().is_none().then_some(())
        });
    }
}

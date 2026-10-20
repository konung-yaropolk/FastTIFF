//! The knobs, and the two places where a wrong default is a wrong result.

use super::*;

/// The defaults are the ones a caller gets by writing nothing, so they are the
/// settings most runs will actually use. Pinned: a drift here changes every
/// result produced by the plugin above without anybody choosing it.
#[test]
fn the_defaults_are_the_documented_ones() {
    let d = Settings::default();
    assert_eq!(d.transformation, Transformation::RigidBody);
    assert_eq!(d.reference, Reference::First);
    assert_eq!(d.n_frames, 1);
    assert_eq!(d.moving_average, 1, "no temporal smoothing unless asked");
    assert_eq!(
        d.max_pyramid_depth, 0,
        "0 means as deep as the image allows"
    );
}

/// The transformation discriminants are pystackreg's constants, so a value
/// copied from a Python script means the same thing here.
#[test]
fn the_transformation_constants_are_pystackregs() {
    assert_eq!(Transformation::Translation as usize, 2);
    assert_eq!(Transformation::RigidBody as usize, 3);
    assert_eq!(Transformation::ScaledRotation as usize, 4);
    assert_eq!(Transformation::Affine as usize, 6);
    assert_eq!(Transformation::Bilinear as usize, 8);
}

/// Degrees of freedom and landmark counts are not the same number, and the one
/// place that is true is the model most people use. A rigid body carries three
/// landmarks for three degrees of freedom; dividing by two would give one.
#[test]
fn rigid_body_has_more_landmarks_than_half_its_freedom() {
    assert_eq!(Transformation::RigidBody.dof(), 3);
    assert_eq!(Transformation::RigidBody.landmarks(), 3);
    // The others are the halves they look like.
    for m in [
        Transformation::Translation,
        Transformation::ScaledRotation,
        Transformation::Affine,
        Transformation::Bilinear,
    ] {
        assert_eq!(
            m.landmarks(),
            m.dof() / 2,
            "{m:?} should carry dof/2 landmarks"
        );
    }
}

/// Every model and every reference mode is offered by `all()`, so a selector
/// built from it cannot silently omit one.
#[test]
fn every_variant_is_offered() {
    assert_eq!(Transformation::all().len(), 5);
    assert_eq!(Reference::all().len(), 3);
    for m in Transformation::all() {
        assert!(!m.label().is_empty());
    }
    for r in Reference::all() {
        assert!(!r.label().is_empty());
    }
}

/// The single-thread backend is always available; the parallel one depends on
/// the build and says so rather than quietly running on one thread.
#[test]
fn a_backend_reports_whether_it_can_run() {
    assert!(Backend::SingleThread.available());
    assert_eq!(
        Backend::MultiThread.available(),
        cfg!(feature = "threads"),
        "the parallel backend exists exactly when the feature does"
    );
    if !cfg!(feature = "threads") {
        assert!(Backend::MultiThread
            .unavailable_reason(64, 64)
            .is_some_and(|r| r.contains("threads")));
    }
}

/// The GPU backend is offered in every build and refused, with a reason, where
/// it cannot run — never silently replaced by the processor.
#[test]
fn the_gpu_backend_is_refused_with_a_reason_where_it_cannot_run() {
    assert_eq!(Backend::all().len(), 3);
    assert!(Backend::all().contains(&Backend::Gpu));
    assert_eq!(Backend::Gpu.label(), "GPU");
    match Backend::Gpu.unavailable_reason(64, 64) {
        // Available: the crate has the feature, this is not a browser, and the
        // machine has a real adapter.
        None => {
            #[cfg(not(feature = "gpu"))]
            panic!("available without the feature?");
            #[cfg(feature = "gpu")]
            assert!(Backend::Gpu.available());
        }
        Some(why) => {
            assert!(!Backend::Gpu.available());
            if !cfg!(feature = "gpu") {
                assert!(why.contains("`gpu` feature"), "{why}");
            }
            assert!(
                why.contains("CPU backend"),
                "the reason should say what to do: {why}"
            );
        }
    }
    // It does the rest of its work on the CPU threads, like the multi-thread
    // backend, and is not the default.
    assert_eq!(Backend::Gpu.threaded(), cfg!(feature = "threads"));
    assert_ne!(Backend::default(), Backend::Gpu);
}

/// Every reference mode runs separate frames at the same time on a threaded
/// backend — the previous-frame one included.
///
/// This test used to assert the opposite for `Reference::Previous`, on the
/// grounds that each frame needs the last. It does not: frame `t` is fitted
/// against the raw frame `t - 1`, so the fits are independent and only their
/// composition is sequential. The old answer kept that mode to one frame at a
/// time, which on the GPU meant one lane per round trip.
#[test]
fn every_reference_mode_runs_frames_side_by_side() {
    for r in Reference::all() {
        for b in Backend::all() {
            assert_eq!(b.parallel_for(r), b.threaded(), "{b:?} under {r:?}");
        }
    }
    assert_eq!(Backend::MultiThread.threaded(), cfg!(feature = "threads"));
    assert!(!Backend::SingleThread.threaded());
    assert!(!Backend::SingleThread.parallel_for(Reference::Previous));
}

//! The C interface against the Rust one it wraps: the same answers, and every
//! refusal a status with a reason rather than a crash.

use super::*;
use std::ffi::CStr;

/// A field with three blobs, moved along a known path.
fn movie(ly: usize, lx: usize, path: &[(i32, i32)]) -> Vec<f32> {
    let mut base = vec![10.0f32; ly * lx];
    for (cy, cx, amp) in [(20.0f32, 24.0f32, 200.0f32), (40.0, 44.0, 150.0), (28.0, 12.0, 120.0)] {
        for y in 0..ly {
            for x in 0..lx {
                let d2 = ((y as f32 - cy).powi(2) + (x as f32 - cx).powi(2)) / 8.0;
                base[y * lx + x] += amp * (-d2).exp();
            }
        }
    }
    path.iter()
        .flat_map(|&(dy, dx)| crate::shift_frame(&base, ly, lx, -dy, -dx))
        .collect()
}

fn settings() -> S2pSettings {
    let mut s = std::mem::MaybeUninit::<S2pSettings>::uninit();
    assert_eq!(unsafe { s2p_settings_default(s.as_mut_ptr()) }, S2P_OK);
    let mut s = unsafe { s.assume_init() };
    s.nonrigid = false;
    s.spatial_taper = 5.0;
    s.backend = S2P_BACKEND_SINGLE_THREAD;
    s
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(s2p_last_error()) }
        .to_string_lossy()
        .into_owned()
}

const PATH: [(i32, i32); 6] = [(0, 0), (2, -1), (-3, 2), (1, 4), (0, -2), (3, 3)];

#[test]
fn the_defaults_are_the_crates() {
    let s = settings();
    let back = S2pSettings {
        nonrigid: true,
        spatial_taper: Settings::default().spatial_taper,
        backend: S2pSettings::from(&Settings::default()).backend,
        ..s
    }
    .to_settings()
    .expect("valid");
    assert_eq!(back, Settings::default());
}

/// Measured through C, the shifts and the moved frames are the Rust API's.
#[test]
fn a_registration_through_c_matches_the_rust_one() {
    let (ly, lx) = (64, 64);
    let pixels = movie(ly, lx, &PATH);
    let s = settings();
    let mut r = std::ptr::null_mut();
    let status = unsafe {
        s2p_register(pixels.as_ptr(), PATH.len(), ly, lx, &s, None, std::ptr::null_mut(), &mut r)
    };
    assert_eq!(status, S2P_OK, "{}", last_error());
    assert!(!r.is_null());

    let frames: Vec<Vec<f32>> = pixels.chunks(ly * lx).map(<[f32]>::to_vec).collect();
    let direct = register(&Frames { ly, lx, frames: &frames }, &s.to_settings().unwrap(), &mut |_| true)
        .expect("not cancelled");

    assert_eq!(unsafe { s2p_registered_frames(r) }, PATH.len());
    for (t, want) in direct.shifts.iter().enumerate() {
        let (mut dy, mut dx, mut corr) = (0, 0, 0.0);
        assert_eq!(unsafe { s2p_registered_shift(r, t, &mut dy, &mut dx, &mut corr) }, S2P_OK);
        assert_eq!((dy, dx, corr), (want.dy, want.dx, want.corr));
        let mut bad = true;
        assert_eq!(unsafe { s2p_registered_is_bad(r, t, &mut bad) }, S2P_OK);
        assert_eq!(bad, direct.bad_frames[t]);

        // In place, which the header promises works.
        let mut frame = frames[t].clone();
        let p = frame.as_mut_ptr();
        assert_eq!(unsafe { s2p_registered_apply(r, p, t, p) }, S2P_OK);
        assert_eq!(frame, direct.apply(&frames[t], ly, lx, t));
    }
    let mut reference = vec![0.0f32; ly * lx];
    assert_eq!(
        unsafe { s2p_registered_reference(r, reference.as_mut_ptr(), reference.len()) },
        S2P_OK
    );
    assert_eq!(reference, direct.reference);
    assert!(!unsafe { s2p_registered_is_nonrigid(r) });

    // The border can be changed after the run, and apply follows it.
    assert_eq!(unsafe { s2p_registered_set_border(r, S2P_BORDER_CONSTANT) }, S2P_OK);
    let mut moved = vec![0.0f32; ly * lx];
    assert_eq!(
        unsafe { s2p_registered_apply(r, frames[5].as_ptr(), 5, moved.as_mut_ptr()) },
        S2P_OK
    );
    let (dy, dx) = (direct.shifts[5].dy, direct.shifts[5].dx);
    assert_eq!(
        moved,
        crate::shift_frame_bordered(&frames[5], ly, lx, dy, dx, Border::Constant)
    );

    unsafe { s2p_registered_free(r) };
}

/// A bidirectional offset the run corrected is corrected again when a frame
/// is applied: the shifts were measured on corrected frames.
#[test]
fn apply_includes_the_bidiphase_correction() {
    let (ly, lx) = (64, 64);
    let pixels = movie(ly, lx, &PATH);
    let s = S2pSettings {
        bidiphase: 2,
        ..settings()
    };
    let mut r = std::ptr::null_mut();
    let status = unsafe {
        s2p_register(pixels.as_ptr(), PATH.len(), ly, lx, &s, None, std::ptr::null_mut(), &mut r)
    };
    assert_eq!(status, S2P_OK, "{}", last_error());
    assert_eq!(unsafe { s2p_registered_bidiphase(r) }, 2);

    let frame = pixels[ly * lx..2 * ly * lx].to_vec();
    let mut combed = frame.clone();
    bidiphase::shift(&mut combed, ly, lx, 2);
    let want = unsafe { &*r }.inner.apply(&combed, ly, lx, 1);
    let mut got = vec![0.0f32; ly * lx];
    assert_eq!(unsafe { s2p_registered_apply(r, frame.as_ptr(), 1, got.as_mut_ptr()) }, S2P_OK);
    assert_eq!(got, want);
    unsafe { s2p_registered_free(r) };
}

unsafe extern "C" fn stop_at_once(_fraction: f32, user: *mut c_void) -> c_int {
    *user.cast::<usize>() += 1;
    0
}

#[test]
fn the_callback_can_stop_a_run() {
    let (ly, lx) = (64, 64);
    let pixels = movie(ly, lx, &PATH);
    let s = settings();
    let mut calls = 0usize;
    let mut r = 1 as *mut S2pRegistered;
    let status = unsafe {
        s2p_register(
            pixels.as_ptr(),
            PATH.len(),
            ly,
            lx,
            &s,
            Some(stop_at_once),
            (&mut calls as *mut usize).cast(),
            &mut r,
        )
    };
    assert_eq!(status, S2P_CANCELLED);
    assert!(r.is_null(), "a stopped run hands back nothing");
    assert!(calls >= 1);
}

/// What C can say and Rust cannot comes back as a status and a reason.
#[test]
fn bad_arguments_are_refused_with_a_reason() {
    let (ly, lx) = (64, 64);
    let pixels = movie(ly, lx, &PATH);
    let mut r = std::ptr::null_mut();
    let run = |s: &S2pSettings, frames: *const f32, n: usize, r: &mut *mut S2pRegistered| unsafe {
        s2p_register(frames, n, ly, lx, s, None, std::ptr::null_mut(), r)
    };

    for broken in [
        S2pSettings { backend: 7, ..settings() },
        S2pSettings { border: -1, ..settings() },
        S2pSettings { subpixel: 0, ..settings() },
        S2pSettings { maxregshift: f64::NAN, ..settings() },
    ] {
        assert_eq!(run(&broken, pixels.as_ptr(), PATH.len(), &mut r), S2P_INVALID_ARGUMENT);
        assert!(!last_error().is_empty());
        assert!(r.is_null());
    }
    assert_eq!(run(&settings(), std::ptr::null(), PATH.len(), &mut r), S2P_INVALID_ARGUMENT);
    assert!(last_error().contains("NULL"), "{}", last_error());
    assert_eq!(run(&settings(), pixels.as_ptr(), 0, &mut r), S2P_INVALID_ARGUMENT);
    let huge = unsafe {
        s2p_register(pixels.as_ptr(), usize::MAX, 2, 2, std::ptr::null(), None, std::ptr::null_mut(), &mut r)
    };
    assert_eq!(huge, S2P_INVALID_ARGUMENT);
    assert_eq!(unsafe { s2p_settings_default(std::ptr::null_mut()) }, S2P_INVALID_ARGUMENT);
    assert_eq!(s2p_backend_available(42, ly, lx), S2P_INVALID_ARGUMENT);
    assert_eq!(s2p_backend_available(S2P_BACKEND_SINGLE_THREAD, ly, lx), S2P_OK);

    // A handle's accessors refuse NULL and an index past the end.
    assert_eq!(unsafe { s2p_registered_frames(std::ptr::null()) }, 0);
    assert_eq!(
        unsafe { s2p_registered_shift(std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) },
        S2P_INVALID_ARGUMENT
    );
    assert_eq!(run(&settings(), pixels.as_ptr(), PATH.len(), &mut r), S2P_OK);
    let mut out = vec![0.0f32; ly * lx];
    assert_eq!(
        unsafe { s2p_registered_apply(r, pixels.as_ptr(), PATH.len(), out.as_mut_ptr()) },
        S2P_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { s2p_registered_reference(r, out.as_mut_ptr(), out.len() - 1) },
        S2P_INVALID_ARGUMENT
    );
    assert_eq!(unsafe { s2p_registered_set_border(r, 3) }, S2P_INVALID_ARGUMENT);
    unsafe { s2p_registered_free(r) };
    unsafe { s2p_registered_free(std::ptr::null_mut()) };
}

/// A GPU run in a build or on a machine that cannot do it is refused up front.
#[test]
fn an_unavailable_backend_is_refused_before_running() {
    let (ly, lx) = (48, 48);
    if Backend::Gpu.unavailable_reason(ly, lx).is_none() {
        return;
    }
    let pixels = movie(ly, lx, &PATH);
    let s = S2pSettings {
        backend: S2P_BACKEND_GPU,
        ..settings()
    };
    let mut r = std::ptr::null_mut();
    let status = unsafe {
        s2p_register(pixels.as_ptr(), PATH.len(), ly, lx, &s, None, std::ptr::null_mut(), &mut r)
    };
    assert_eq!(status, S2P_UNAVAILABLE);
    assert!(!last_error().is_empty());
}

#[test]
fn the_version_is_the_crates() {
    let v = unsafe { CStr::from_ptr(s2p_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

/// The header is written by hand, so it is read here and held against this
/// module: every constant's value, the settings structure's fields in order,
/// and the functions it declares. The struct literal in this test lists every
/// field without `..`, so a field added to the Rust structure stops this from
/// compiling until the list — and the header — are brought along.
#[test]
fn the_header_matches_this_module() {
    let header = include_str!("../include/suite2p_registration.h");

    // Constants.
    let defines: Vec<(&str, i64)> = header
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define S2P_"))
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap();
            (name, it.next().and_then(|v| v.parse().ok()).expect(name))
        })
        .collect();
    let want: &[(&str, c_int)] = &[("OK", S2P_OK), ("CANCELLED", S2P_CANCELLED), ("INVALID_ARGUMENT", S2P_INVALID_ARGUMENT), ("UNAVAILABLE", S2P_UNAVAILABLE), ("FAILED", S2P_FAILED), ("BACKEND_SINGLE_THREAD", S2P_BACKEND_SINGLE_THREAD), ("BACKEND_MULTI_THREAD", S2P_BACKEND_MULTI_THREAD), ("BACKEND_GPU", S2P_BACKEND_GPU), ("BORDER_NATIVE", S2P_BORDER_NATIVE), ("BORDER_EDGE", S2P_BORDER_EDGE), ("BORDER_CONSTANT", S2P_BORDER_CONSTANT)];
    assert_eq!(defines.len(), want.len(), "{defines:?}");
    for (name, value) in want {
        let found = defines.iter().find(|(n, _)| n == name);
        assert_eq!(found.map(|(_, v)| *v), Some(*value as i64), "S2P_{name}");
    }

    // The settings structure: names, in order.
    let body = header
        .split("typedef struct S2pSettings {")
        .nth(1)
        .and_then(|rest| rest.split("} S2pSettings;").next())
        .expect("the settings struct");
    let fields: Vec<&str> = body
        .lines()
        .filter_map(|l| l.split(';').next())
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("/*"))
        .map(|l| {
            let decl = l.split_whitespace().last().unwrap();
            decl.split('[').next().unwrap()
        })
        .collect();
    let s = S2pSettings {
        nimg_init: Default::default(),
        reference_iterations: Default::default(),
        batch_size: Default::default(),
        block_size: Default::default(),
        subpixel: Default::default(),
        maxregshift: Default::default(),
        maxregshift_nr: Default::default(),
        smooth_sigma: Default::default(),
        smooth_sigma_time: Default::default(),
        spatial_taper: Default::default(),
        snr_thresh: Default::default(),
        th_badframes: Default::default(),
        bidiphase: Default::default(),
        backend: Default::default(),
        border: Default::default(),
        norm_frames: Default::default(),
        two_step_registration: Default::default(),
        do_bidiphase: Default::default(),
        nonrigid: Default::default(),
    };
    let _ = s;
    assert_eq!(fields, ["nimg_init", "reference_iterations", "batch_size", "block_size", "subpixel", "maxregshift", "maxregshift_nr", "smooth_sigma", "smooth_sigma_time", "spatial_taper", "snr_thresh", "th_badframes", "bidiphase", "backend", "border", "norm_frames", "two_step_registration", "do_bidiphase", "nonrigid"]);

    // Functions: what the header declares is what this module exports.
    let mut declared: Vec<&str> = header
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.starts_with("s2p_") && header.contains(&format!("{w}(")))
        .collect();
    declared.sort_unstable();
    declared.dedup();
    let mut exported = Vec::from(["s2p_last_error", "s2p_version", "s2p_settings_default", "s2p_backend_available", "s2p_register", "s2p_registered_free", "s2p_registered_frames", "s2p_registered_shift", "s2p_registered_is_bad", "s2p_registered_bidiphase", "s2p_registered_is_nonrigid", "s2p_registered_reference", "s2p_registered_set_border", "s2p_registered_apply"]);
    exported.sort_unstable();
    assert_eq!(declared, exported);
}

//! The C interface against the Rust one it wraps: the same answers, and every
//! refusal a status with a reason rather than a crash.

use super::*;
use crate::{transform, Matrix};
use std::ffi::CStr;

const W: usize = 48;
const H: usize = 40;
const N: usize = 5;

/// A drifting stack, each frame translated a little further than the last.
fn stack() -> Vec<f32> {
    let base: Vec<f32> = (0..W * H)
        .map(|i| {
            let (x, y) = ((i % W) as f32, (i / W) as f32);
            128.0 + (x * 0.23).sin() * 50.0 + (y * 0.19).cos() * 40.0 + (x * 0.7 + y * 0.5).sin() * 15.0
        })
        .collect();
    (0..N)
        .flat_map(|t| {
            let d = 0.8 * t as f64;
            transform::warp(&base, W, H, &Matrix { c: [d, 1.0, 0.0, 0.0, d * 0.5, 0.0, 1.0, 0.0] })
        })
        .collect()
}

fn settings() -> TrSettings {
    let mut s = std::mem::MaybeUninit::<TrSettings>::uninit();
    assert_eq!(unsafe { tr_settings_default(s.as_mut_ptr()) }, TR_OK);
    let mut s = unsafe { s.assume_init() };
    s.transformation = TR_TRANSFORMATION_TRANSLATION;
    s.backend = TR_BACKEND_SINGLE_THREAD;
    s
}

fn last_error() -> String {
    unsafe { CStr::from_ptr(tr_last_error()) }
        .to_string_lossy()
        .into_owned()
}

fn run(s: &TrSettings, pixels: &[f32], r: &mut *mut TrRegistered) -> c_int {
    unsafe { tr_register(pixels.as_ptr(), N, H, W, s, None, std::ptr::null_mut(), r) }
}

#[test]
fn the_defaults_are_the_crates() {
    let mut s = std::mem::MaybeUninit::<TrSettings>::uninit();
    assert_eq!(unsafe { tr_settings_default(s.as_mut_ptr()) }, TR_OK);
    let s = unsafe { s.assume_init() };
    assert_eq!(s.to_settings(), Ok((Settings::default(), Border::default())));
    // And the numbering is the header's.
    assert_eq!(s.transformation, TR_TRANSFORMATION_RIGID_BODY);
    assert_eq!(s.reference, TR_REFERENCE_FIRST);
    assert_eq!(s.border, TR_BORDER_NATIVE);
}

/// Measured through C, the matrices and the moved frames are the Rust API's.
#[test]
fn a_registration_through_c_matches_the_rust_one() {
    let pixels = stack();
    for reference in [TR_REFERENCE_PREVIOUS, TR_REFERENCE_FIRST, TR_REFERENCE_MEAN] {
        let s = TrSettings {
            reference,
            ..settings()
        };
        let mut r = std::ptr::null_mut();
        assert_eq!(run(&s, &pixels, &mut r), TR_OK, "{}", last_error());

        let frames: Vec<Vec<f32>> = pixels.chunks(W * H).map(<[f32]>::to_vec).collect();
        let (rs, _) = s.to_settings().unwrap();
        let direct = register_stack(&Frames { ly: H, lx: W, frames: &frames }, &rs, &mut |_| true)
            .expect("not cancelled");

        assert_eq!(unsafe { tr_registered_frames(r) }, N);
        for (t, frame) in frames.iter().enumerate() {
            let mut c = [0.0f64; 8];
            assert_eq!(unsafe { tr_registered_matrix(r, t, c.as_mut_ptr()) }, TR_OK);
            assert_eq!(c, direct.matrices[t].c);
            let mut failed = true;
            assert_eq!(unsafe { tr_registered_failed(r, t, &mut failed) }, TR_OK);
            assert_eq!(failed, direct.failed.contains(&t));

            // In place, which the header promises works.
            let mut moved = frame.clone();
            let p = moved.as_mut_ptr();
            assert_eq!(unsafe { tr_registered_apply(r, p, t, p) }, TR_OK);
            assert_eq!(moved, direct.apply(frame, H, W, t).unwrap());
        }
        let mut out = vec![0.0f32; W * H];
        let has = unsafe { tr_registered_has_reference(r) };
        assert_eq!(has, direct.reference.is_some());
        let status = unsafe { tr_registered_reference(r, out.as_mut_ptr(), out.len()) };
        match &direct.reference {
            Some(want) => {
                assert_eq!(status, TR_OK);
                assert_eq!(&out, want);
            }
            None => assert_eq!(status, TR_INVALID_ARGUMENT),
        }
        unsafe { tr_registered_free(r) };
    }
}

/// The border comes from the settings and can be changed after the run.
#[test]
fn the_border_reaches_apply() {
    let pixels = stack();
    let s = TrSettings {
        border: TR_BORDER_CONSTANT,
        ..settings()
    };
    let mut r = std::ptr::null_mut();
    assert_eq!(run(&s, &pixels, &mut r), TR_OK, "{}", last_error());
    let frame = &pixels[(N - 1) * W * H..];
    let mut c = [0.0f64; 8];
    assert_eq!(unsafe { tr_registered_matrix(r, N - 1, c.as_mut_ptr()) }, TR_OK);
    let m = Matrix { c };
    let mut out = vec![0.0f32; W * H];
    for (border, b) in [(TR_BORDER_CONSTANT, Border::Constant), (TR_BORDER_EDGE, Border::Edge)] {
        assert_eq!(unsafe { tr_registered_set_border(r, border) }, TR_OK);
        assert_eq!(
            unsafe { tr_registered_apply(r, frame.as_ptr(), N - 1, out.as_mut_ptr()) },
            TR_OK
        );
        assert_eq!(out, transform::warp_bordered(frame, W, H, &m, b, false));
    }
    unsafe { tr_registered_free(r) };
}

unsafe extern "C" fn stop_at_once(_fraction: f32, user: *mut c_void) -> c_int {
    *user.cast::<usize>() += 1;
    0
}

#[test]
fn the_callback_can_stop_a_run() {
    let pixels = stack();
    let s = settings();
    let mut calls = 0usize;
    let mut r = 1 as *mut TrRegistered;
    let status = unsafe {
        tr_register(
            pixels.as_ptr(),
            N,
            H,
            W,
            &s,
            Some(stop_at_once),
            (&mut calls as *mut usize).cast(),
            &mut r,
        )
    };
    assert_eq!(status, TR_CANCELLED);
    assert!(r.is_null(), "a stopped run hands back nothing");
    assert!(calls >= 1);
}

/// What C can say and Rust cannot comes back as a status and a reason.
#[test]
fn bad_arguments_are_refused_with_a_reason() {
    let pixels = stack();
    let mut r = std::ptr::null_mut();
    for broken in [
        TrSettings { transformation: 5, ..settings() },
        TrSettings { reference: -1, ..settings() },
        TrSettings { backend: 3, ..settings() },
        TrSettings { border: 9, ..settings() },
        TrSettings { n_frames: 0, ..settings() },
        TrSettings {
            transformation: TR_TRANSFORMATION_BILINEAR,
            reference: TR_REFERENCE_PREVIOUS,
            ..settings()
        },
    ] {
        assert_eq!(run(&broken, &pixels, &mut r), TR_INVALID_ARGUMENT, "{broken:?}");
        assert!(!last_error().is_empty());
        assert!(r.is_null());
    }
    assert!(last_error().contains("bilinear"), "{}", last_error());
    let null = unsafe { tr_register(std::ptr::null(), N, H, W, std::ptr::null(), None, std::ptr::null_mut(), &mut r) };
    assert_eq!(null, TR_INVALID_ARGUMENT);
    let huge = unsafe {
        tr_register(pixels.as_ptr(), usize::MAX, 2, 2, std::ptr::null(), None, std::ptr::null_mut(), &mut r)
    };
    assert_eq!(huge, TR_INVALID_ARGUMENT);
    assert_eq!(unsafe { tr_settings_default(std::ptr::null_mut()) }, TR_INVALID_ARGUMENT);
    assert_eq!(tr_backend_available(-3, H, W), TR_INVALID_ARGUMENT);
    assert_eq!(tr_backend_available(TR_BACKEND_SINGLE_THREAD, H, W), TR_OK);

    assert_eq!(unsafe { tr_registered_frames(std::ptr::null()) }, 0);
    assert_eq!(run(&settings(), &pixels, &mut r), TR_OK);
    let mut out = vec![0.0f32; W * H];
    assert_eq!(
        unsafe { tr_registered_apply(r, pixels.as_ptr(), N, out.as_mut_ptr()) },
        TR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { tr_registered_matrix(r, 0, std::ptr::null_mut()) },
        TR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { tr_registered_reference(r, out.as_mut_ptr(), 3) },
        TR_INVALID_ARGUMENT
    );
    // 3 is TR_BORDER_CROP; 4 is nothing.
    assert_eq!(unsafe { tr_registered_set_border(r, 4) }, TR_INVALID_ARGUMENT);
    unsafe { tr_registered_free(r) };
    unsafe { tr_registered_free(std::ptr::null_mut()) };
}

#[test]
fn the_version_is_the_crates() {
    let v = unsafe { CStr::from_ptr(tr_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

/// The header is written by hand, so it is read here and held against this
/// module: every constant's value, the settings structure's fields in order,
/// and the functions it declares. The struct literal in this test lists every
/// field without `..`, so a field added to the Rust structure stops this from
/// compiling until the list — and the header — are brought along.
#[test]
fn the_header_matches_this_module() {
    let header = include_str!("../include/turboreg_registration.h");

    // Constants.
    let defines: Vec<(&str, i64)> = header
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#define TR_"))
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap();
            (name, it.next().and_then(|v| v.parse().ok()).expect(name))
        })
        .collect();
    let want: &[(&str, c_int)] = &[("OK", TR_OK), ("CANCELLED", TR_CANCELLED), ("INVALID_ARGUMENT", TR_INVALID_ARGUMENT), ("UNAVAILABLE", TR_UNAVAILABLE), ("FAILED", TR_FAILED), ("NO_COMMON_AREA", TR_NO_COMMON_AREA), ("TRANSFORMATION_TRANSLATION", TR_TRANSFORMATION_TRANSLATION), ("TRANSFORMATION_RIGID_BODY", TR_TRANSFORMATION_RIGID_BODY), ("TRANSFORMATION_SCALED_ROTATION", TR_TRANSFORMATION_SCALED_ROTATION), ("TRANSFORMATION_AFFINE", TR_TRANSFORMATION_AFFINE), ("TRANSFORMATION_BILINEAR", TR_TRANSFORMATION_BILINEAR), ("REFERENCE_PREVIOUS", TR_REFERENCE_PREVIOUS), ("REFERENCE_FIRST", TR_REFERENCE_FIRST), ("REFERENCE_MEAN", TR_REFERENCE_MEAN), ("BACKEND_SINGLE_THREAD", TR_BACKEND_SINGLE_THREAD), ("BACKEND_MULTI_THREAD", TR_BACKEND_MULTI_THREAD), ("BACKEND_GPU", TR_BACKEND_GPU), ("BORDER_NATIVE", TR_BORDER_NATIVE), ("BORDER_EDGE", TR_BORDER_EDGE), ("BORDER_CONSTANT", TR_BORDER_CONSTANT), ("BORDER_CROP", TR_BORDER_CROP)];
    assert_eq!(defines.len(), want.len(), "{defines:?}");
    for (name, value) in want {
        let found = defines.iter().find(|(n, _)| n == name);
        assert_eq!(found.map(|(_, v)| *v), Some(*value as i64), "TR_{name}");
    }

    // The settings structure: names, in order.
    let body = header
        .split("typedef struct TrSettings {")
        .nth(1)
        .and_then(|rest| rest.split("} TrSettings;").next())
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
    let s = TrSettings {
        n_frames: Default::default(),
        moving_average: Default::default(),
        max_pyramid_depth: Default::default(),
        transformation: Default::default(),
        reference: Default::default(),
        backend: Default::default(),
        border: Default::default(),
    };
    let _ = s;
    assert_eq!(fields, ["n_frames", "moving_average", "max_pyramid_depth", "transformation", "reference", "backend", "border"]);

    // Functions: what the header declares is what this module exports.
    let mut declared: Vec<&str> = header
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| w.starts_with("tr_") && header.contains(&format!("{w}(")))
        .collect();
    declared.sort_unstable();
    declared.dedup();
    let mut exported = Vec::from(["tr_last_error", "tr_version", "tr_settings_default", "tr_backend_available", "tr_register", "tr_registered_free", "tr_registered_frames", "tr_registered_matrix", "tr_registered_failed", "tr_registered_has_reference", "tr_registered_reference", "tr_registered_set_border", "tr_registered_output_size", "tr_registered_common_area", "tr_registered_apply"]);
    exported.sort_unstable();
    assert_eq!(declared, exported);
}

/// `TR_BORDER_CROP` through C: the output size is the common area's, frames
/// come back that size, and frames with nothing in common are
/// `TR_NO_COMMON_AREA` with a reason.
#[test]
fn a_crop_through_c_is_the_rust_one() {
    let pixels = stack();
    let s = TrSettings {
        border: TR_BORDER_CROP,
        ..settings()
    };
    let mut r = std::ptr::null_mut();
    assert_eq!(run(&s, &pixels, &mut r), TR_OK, "{}", last_error());
    let inner = unsafe { &*r }.inner.common_area(H, W).expect("an area");
    assert!(inner.height < H || inner.width < W, "the stack drifts: {inner:?}");

    let (mut oly, mut olx) = (0, 0);
    assert_eq!(unsafe { tr_registered_output_size(r, &mut oly, &mut olx) }, TR_OK);
    assert_eq!((oly, olx), (inner.height, inner.width));
    let (mut y, mut x, mut h, mut w, mut angle) = (9.0, 9.0, 9, 9, 9.0);
    assert_eq!(
        unsafe { tr_registered_common_area(r, &mut y, &mut x, &mut h, &mut w, &mut angle) },
        TR_OK
    );
    assert_eq!(
        (y, x, h, w, angle),
        (inner.y, inner.x, inner.height, inner.width, inner.angle)
    );
    let last = &pixels[(N - 1) * W * H..];
    let want = unsafe { &*r }.inner.apply(last, H, W, N - 1).unwrap();
    assert_eq!(want.len(), oly * olx);
    let mut frame = last.to_vec();
    let p = frame.as_mut_ptr();
    assert_eq!(unsafe { tr_registered_apply(r, p, N - 1, p) }, TR_OK);
    assert_eq!(&frame[..oly * olx], want.as_slice());
    unsafe { tr_registered_free(r) };

    // Frames that share nothing: every crop call says so.
    let mut r = std::ptr::null_mut();
    assert_eq!(run(&s, &pixels, &mut r), TR_OK);
    unsafe { &mut *r }.inner.matrices[0].c[0] = 1000.0;
    let mut out = vec![0.0f32; W * H];
    for status in [
        unsafe { tr_registered_output_size(r, &mut oly, &mut olx) },
        unsafe { tr_registered_common_area(r, &mut y, &mut x, &mut h, &mut w, &mut angle) },
        unsafe { tr_registered_apply(r, pixels.as_ptr(), 0, out.as_mut_ptr()) },
    ] {
        assert_eq!(status, TR_NO_COMMON_AREA);
        assert!(last_error().contains("no common area"), "{}", last_error());
    }
    assert_eq!(unsafe { tr_registered_set_border(r, TR_BORDER_CONSTANT) }, TR_OK);
    assert_eq!(
        unsafe { tr_registered_apply(r, pixels.as_ptr(), 0, out.as_mut_ptr()) },
        TR_OK
    );
    unsafe { tr_registered_free(r) };
}

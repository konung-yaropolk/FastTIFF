//! Regression tests for mapping a display frame to an IFD in the file.
//!
//! The failure these guard against: the stack's *resolved* interpretation
//! (`Stack::display.dims`) and the file's *raw* metadata (`tiff.meta`) are
//! deliberately allowed to disagree — resolving is exactly the act of
//! reclassifying a mislabeled axis, e.g. a file claiming `channels=10` that is
//! really a 10-frame movie. Anything that addresses planes must use the
//! resolved view; reading the raw one computes indices off the wrong stride and
//! walks past the end of the IFD chain.

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::{Stack, Viewer};
use std::io::Cursor;

/// A stack of `n` single-channel planes whose ImageJ metadata *claims*
/// `channels = n` — the mislabeling `resolve_dimensions` exists to correct.
/// With `n` above the channel-size cutoff it resolves to `1 channel x n frames`.
fn mislabeled_stack(n: usize, w: u32, h: u32) -> Vec<u8> {
    let opts = WriterOptions::new(w, h, SampleType::U16).metadata(StackMetaWrite::new(n, 1));
    let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..n {
        let px: Vec<u16> = (0..w * h)
            .map(|i| (i as u16).wrapping_add(f as u16 * 101))
            .collect();
        let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write_frame_bytes(&bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn load(bytes: Vec<u8>) -> Stack {
    Stack::from_bytes(bytes, "mislabeled.tif".into(), false).expect("should open")
}

#[test]
fn resolved_and_raw_dimensions_really_do_diverge() {
    // The premise of every test below: if these ever agreed, the rest would
    // pass for the wrong reason.
    let stack = load(mislabeled_stack(10, 4, 3));
    assert_eq!(stack.tiff.frames.len(), 10, "ten planes in the file");
    assert_eq!(
        (stack.tiff.meta.channels, stack.tiff.meta.frames),
        (10, 1),
        "raw metadata keeps the file's own (mislabeled) claim"
    );
    assert_eq!(
        (stack.display.dims.channels, stack.display.dims.frames),
        (1, 10),
        "the resolved view reclassifies it as a movie"
    );
}

#[test]
fn every_frame_maps_to_an_ifd_that_exists() {
    let stack = load(mislabeled_stack(10, 4, 3));
    let n = stack.frame_count();
    assert_eq!(n, 10);

    let enabled = vec![true; stack.display.settings.len()];
    let kinds: Vec<_> = stack.display.settings.iter().map(|s| s.kind).collect();

    // Scrubbing the whole stack must only ever address planes that are there.
    // Computing the stride from the raw metadata gives `frame * 10`, which is
    // in range only for frame 0 — the "stuck on the first frame" symptom.
    for frame in 0..n {
        for job in fast_tiff_viewer::prefetch::build_jobs(&stack, frame, &enabled, &kinds) {
            assert!(
                job.ifd_idx < stack.tiff.frames.len(),
                "frame {frame} -> IFD {} but the file has only {} planes",
                job.ifd_idx,
                stack.tiff.frames.len()
            );
        }
    }
}

#[test]
fn every_frame_actually_decodes() {
    // The end-to-end version: the index being in range is necessary but not
    // sufficient — each plane must also decode, and to *different* pixels.
    let stack = load(mislabeled_stack(8, 4, 3));
    let enabled = vec![true; stack.display.settings.len()];
    let kinds: Vec<_> = stack.display.settings.iter().map(|s| s.kind).collect();

    let mut first_pixels = Vec::new();
    for frame in 0..stack.frame_count() {
        let jobs = fast_tiff_viewer::prefetch::build_jobs(&stack, frame, &enabled, &kinds);
        let decoded = fast_tiff_viewer::prefetch::decode_jobs(
            &stack.tiff.data,
            &stack.tiff.frames,
            stack.tiff.byte_order,
            &jobs,
        )
        .unwrap_or_else(|e| panic!("frame {frame} failed to decode: {e:#}"));
        match &decoded[0] {
            fast_tiff_viewer::prefetch::Decoded::U16(v) => first_pixels.push(v[0]),
            other => panic!(
                "unexpected decode kind for frame {frame}: {:?}",
                std::mem::discriminant(other)
            ),
        }
    }
    // Each written frame was offset by 101, so no two frames share a first pixel.
    let mut seen = first_pixels.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        first_pixels.len(),
        "frames decoded to duplicate pixels: {first_pixels:?}"
    );
}

#[test]
fn volume_and_playback_gates_follow_the_resolved_view() {
    // A 10-plane movie has depth to ray-march and no separate time axis. Read
    // off the raw metadata it looks like a single frame with ten channels,
    // which would disable the 3D toggle outright.
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(mislabeled_stack(10, 4, 3), "mislabeled.tif".into())
        .unwrap();
    assert!(
        viewer.can_show_volume(),
        "ten planes are enough to build a volume"
    );
    assert!(!viewer.is_4d(), "there is no separate time axis here");
}

// ---- one definition of (c, z, t) -> (IFD, plane) ----
//
// The formula used to be written twice: once in `prefetch::build_jobs` for the
// 2D view and once in `volume` for the ray-marched one. They agreed only
// because the 2D copy was the `z == 0` case of the same arithmetic — but its
// doc comment called itself "the single definition of which plane is which
// channel", so anything addressing planes generally (a plugin API, say) would
// reasonably route through it and silently receive slice 0 for every z.

use fast_tiff_viewer::{plane_index, planes_addressed, Dims};

fn dims(channels: usize, slices: usize, frames: usize) -> Dims {
    Dims {
        channels,
        slices,
        frames,
    }
}

/// `xyczt`: channel fastest, then Z, then time — and every plane distinct.
#[test]
fn plane_index_walks_xyczt_order_without_collisions() {
    let d = dims(3, 4, 5);
    let mut seen = std::collections::HashSet::new();
    let mut expected = 0usize;
    for t in 0..d.frames {
        for z in 0..d.slices {
            for c in 0..d.channels {
                let (ifd, plane) = plane_index(d, false, c, z, t);
                assert_eq!(plane, 0, "non-RGB puts every channel in its own IFD");
                assert_eq!(ifd, expected, "c{c} z{z} t{t} is not in xyczt order");
                assert!(seen.insert((ifd, plane)), "c{c} z{z} t{t} collides");
                expected += 1;
            }
        }
    }
    assert_eq!(seen.len(), d.channels * d.slices * d.frames);
}

/// Chunky RGB keeps all channels in one IFD, selecting the sample plane.
#[test]
fn plane_index_treats_rgb_channels_as_sample_planes() {
    let d = dims(3, 4, 5);
    for t in 0..d.frames {
        for z in 0..d.slices {
            let ifds: Vec<usize> = (0..d.channels)
                .map(|c| {
                    let (ifd, plane) = plane_index(d, true, c, z, t);
                    assert_eq!(plane, c, "RGB channel {c} must select sample plane {c}");
                    ifd
                })
                .collect();
            assert!(
                ifds.windows(2).all(|w| w[0] == w[1]),
                "one IFD per (z, t) for RGB"
            );
            assert_eq!(ifds[0], t * d.slices + z);
        }
    }
}

/// The 2D view is the `z == 0` slice of the same function — that equivalence is
/// what let the duplicate formula go unnoticed, so pin it rather than rely on it.
#[test]
fn the_2d_views_addressing_is_the_z_zero_case() {
    for &(c, z, f) in &[(1usize, 1usize, 1usize), (3, 4, 5), (2, 1, 9), (1, 7, 3)] {
        let d = dims(c, z, f);
        for rgb in [false, true] {
            for t in 0..d.frames {
                for ch in 0..d.channels {
                    let general = plane_index(d, rgb, ch, 0, t);
                    let two_d = if rgb {
                        (t * d.slices, ch)
                    } else {
                        (t * d.slices * d.channels + ch, 0)
                    };
                    assert_eq!(general, two_d, "rgb={rgb} c{ch} t{t} of {d:?}");
                }
            }
        }
    }
}

/// `planes_addressed` must be exactly one past the highest IFD the 2D view can
/// ask for — it exists to stop that view requesting a plane the file lacks.
#[test]
fn planes_addressed_bounds_what_the_2d_view_reaches() {
    for &(c, z, f) in &[
        (1usize, 1usize, 1usize),
        (3, 4, 5),
        (2, 1, 9),
        (1, 7, 3),
        (6, 2, 11),
    ] {
        let d = dims(c, z, f);
        for rgb in [false, true] {
            let highest = (0..d.frames)
                .flat_map(|t| (0..d.channels).map(move |ch| (ch, t)))
                .map(|(ch, t)| plane_index(d, rgb, ch, 0, t).0)
                .max()
                .unwrap();
            assert_eq!(
                planes_addressed(d, rgb),
                highest + 1,
                "planes_addressed disagrees with plane_index for rgb={rgb} on {d:?}"
            );
        }
    }
}

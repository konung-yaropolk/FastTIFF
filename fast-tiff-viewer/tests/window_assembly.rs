//! Assembling a window: do the pixels land where they belong?
//!
//! Everything else about windowing is about *cost* — which strips to touch, what
//! to keep, what to build off-thread. This is about correctness, and it is the
//! one property none of that is worth anything without: the texture handed to
//! the GPU must hold the same samples the frame does, at the sampling asked for.
//!
//! The hazard is the coarse path. At a stride that is a whole number of strips,
//! `assemble` stops decoding the strips it does not need and asks the library
//! for a *sampled* band instead — rows that were spaced `stride` apart in the
//! file arrive spaced `rows_per_strip` apart in the buffer, starting wherever
//! the first sampled row fell inside its strip. Getting that remapping wrong
//! does not fail, error or panic: it draws a picture made of the right pixels in
//! the wrong rows, which at a glance looks like a plausible image with
//! horizontal banding.
//!
//! So each case here decodes the whole frame, subsamples it in the most obvious
//! way possible, and demands the window match.

#![cfg(feature = "mmap")]

use fast_tiff_lib::{SampleType, TiffStack, TiffWriter, WriterOptions};
use fast_tiff_viewer::bandcache::BandCache;
use fast_tiff_viewer::prefetch::{ChannelJob, Decoded};
use fast_tiff_viewer::roi::Roi;
use fast_tiff_viewer::window;
use scivis_render::ChannelKind;
use std::io::Cursor;

const W: u32 = 61; // deliberately not a multiple of any stride used here
const H: u32 = 96;

/// A single-channel 16-bit frame whose every sample is a function of its
/// position, so a sample landing in the wrong row is visible as a mismatch
/// rather than as plausible noise.
fn write(rows_per_strip: u32) -> Vec<u8> {
    let opts = WriterOptions::new(W, H, SampleType::U16)
        .samples_per_pixel(1)
        .rows_per_strip(rows_per_strip);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    let px: Vec<u16> = (0..(W * H) as usize).map(|i| (i * 131 + 17) as u16).collect();
    w.write_frame_u16(&px).unwrap();
    w.finish().unwrap().into_inner()
}

fn job(stack: &TiffStack) -> Vec<ChannelJob> {
    let f = &stack.frames[0];
    vec![ChannelJob {
        channel: 0,
        ifd_idx: 0,
        plane: 0,
        kind: ChannelKind::Int16,
        rgb: false,
        width: f.width,
        height: f.height,
    }]
}

fn u16s(d: &Decoded) -> &[u16] {
    match d {
        Decoded::U16(v) => v,
        _ => panic!("expected a u16 plane"),
    }
}

/// What the window *should* hold: the whole frame, sampled every `stride` from
/// the window's own origin. Written the slow, obvious way on purpose — it is the
/// oracle, so it must not share any arithmetic with the code under test.
fn expected(full: &[u16], roi: &Roi) -> Vec<u16> {
    let (tw, th) = roi.texture_size();
    let mut out = Vec::with_capacity((tw * th) as usize);
    for ty in 0..th {
        for tx in 0..tw {
            let sx = roi.x + tx * roi.stride;
            let sy = roi.y + ty * roi.stride;
            out.push(if sx < W && sy < H { full[(sy * W + sx) as usize] } else { 0 });
        }
    }
    out
}

fn check(rows_per_strip: u32, roi: Roi) {
    let bytes = write(rows_per_strip);
    let stack = TiffStack::from_bytes(bytes).unwrap();
    let full = fast_tiff_lib::read_planes_u16(&stack.data, &stack.frames[0], stack.byte_order, None)
        .unwrap();
    let jobs = job(&stack);
    let mut bands = BandCache::default();
    let got =
        window::assemble(&stack, &mut bands, &jobs, &[ChannelKind::Int16], &roi, 0).unwrap();

    let want = expected(&full[0], &roi);
    let got = u16s(&got[0]);
    assert_eq!(got.len(), want.len(), "rows_per_strip {rows_per_strip}, {roi:?}: texture size");

    if got != want.as_slice() {
        let (tw, _) = roi.texture_size();
        let bad = (0..got.len()).find(|&i| got[i] != want[i]).unwrap();
        panic!(
            "rows_per_strip {rows_per_strip}, {roi:?}: first mismatch at texel row {}, col {} \
             (index {bad}): got {}, expected {} — a row of the window is reading the wrong \
             source row",
            bad / tw as usize,
            bad % tw as usize,
            got[bad],
            want[bad]
        );
    }
}

/// Stride 1: no sampling at all, the case every zoomed-in view takes.
#[test]
fn a_full_resolution_window_holds_the_frame_it_covers() {
    for rps in [1, 2, 4, 8] {
        check(rps, Roi { x: 0, y: 0, w: W, h: H, stride: 1 });
        check(rps, Roi { x: 8, y: 12, w: 32, h: 40, stride: 1 });
    }
}

/// A stride that is *not* a whole number of strips, so the band is read
/// contiguously and sampled on the way into the texture.
#[test]
fn a_window_sampled_within_its_strips_holds_the_right_rows() {
    // rows_per_strip 8 with stride 2 or 4: `step_for` declines, ordinary path.
    for stride in [2, 4] {
        check(8, Roi { x: 0, y: 0, w: W, h: H, stride });
        check(8, Roi { x: 4, y: 8, w: 48, h: 64, stride });
    }
}

/// The coarse path: the stride is a whole number of strips, so whole strips are
/// skipped rather than decoded and thrown away. This is the one that draws
/// banding when the remapping is wrong, and it is what a fit-to-window view of a
/// frame too large for a texture actually takes.
#[test]
fn a_window_sampled_across_whole_strips_holds_the_right_rows() {
    // rows_per_strip 2, strides 4/8/16 -> step 2/4/8.
    for stride in [4, 8, 16] {
        check(2, Roi { x: 0, y: 0, w: W, h: H, stride });
    }
    // rows_per_strip 4, strides 8/16 -> step 2/4.
    for stride in [8, 16] {
        check(4, Roi { x: 0, y: 0, w: W, h: H, stride });
    }
    // And the same, off the origin, so the first sampled row does not fall at
    // the start of its strip.
    check(2, Roi { x: 5, y: 6, w: 48, h: 80, stride: 4 });
    check(4, Roi { x: 5, y: 10, w: 48, h: 80, stride: 8 });
}

/// The two paths must agree with each other, not merely each with the oracle —
/// the same window at the same stride is the same picture however it was read.
#[test]
fn the_stepped_and_contiguous_paths_agree() {
    let roi = Roi { x: 0, y: 0, w: W, h: H, stride: 8 };

    // rows_per_strip 8 makes `step_for` decline (stride == per); rows_per_strip
    // 2 makes it take the stepped path. Same file content either way.
    let mut planes = Vec::new();
    for rps in [8u32, 2] {
        let stack = TiffStack::from_bytes(write(rps)).unwrap();
        let jobs = job(&stack);
        let mut bands = BandCache::default();
        let got =
            window::assemble(&stack, &mut bands, &jobs, &[ChannelKind::Int16], &roi, 0).unwrap();
        planes.push(u16s(&got[0]).to_vec());
    }
    assert_eq!(planes[0], planes[1], "the contiguous and stepped reads disagree");
}

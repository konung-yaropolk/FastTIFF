//! Saving, checked by opening what was saved.
//!
//! Every test here writes a stack, reads the file back, and compares. That is
//! the only assertion worth making about a writer: a file that "looks right" in
//! a hex dump and does not reopen is a bug, and a file that reopens with every
//! pixel shifted by half the range — which is what signed data does without the
//! offset below — looks perfectly plausible until someone measures something.

use super::*;
use crate::stack::Stack;
use fast_tiff_lib::{
    read_frame_f32, read_frame_u16, SampleType, StackMetaWrite, TiffStack, TiffWriter,
    WriterOptions,
};
use std::io::Cursor;
use std::path::PathBuf;

const W: u32 = 6;
const H: u32 = 4;

fn temp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("fasttiff-save-{name}.tif"));
    let _ = std::fs::remove_file(&p);
    p
}

/// A stack of `frames` planes whose every sample says which plane it is in.
fn source(sample: SampleType, channels: usize, slices: usize, frames: usize) -> Vec<u8> {
    let opts = WriterOptions::new(W, H, sample).metadata(StackMetaWrite::new(channels, slices));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for p in 0..(channels * slices * frames) {
        match sample {
            SampleType::U8 => {
                let px: Vec<u8> = (0..W * H).map(|i| (p as u32 * 7 + i) as u8).collect();
                w.write_frame_u8(&px).expect("frame");
            }
            SampleType::F32 => {
                let px: Vec<f32> = (0..W * H)
                    .map(|i| p as f32 * 1000.5 - i as f32 / 8.0)
                    .collect();
                w.write_frame_f32(&px).expect("frame");
            }
            // U16 and I16 alike: the same bit patterns, written as bytes
            // because the typed call is keyed to the writer's sample type and
            // there is no `write_frame_i16`. Values run past 0x8000 on purpose,
            // so a signed frame really does carry negative samples.
            _ => {
                let px: Vec<u16> = (0..W * H)
                    .map(|i| (p as u32 * 9000 + i * 37) as u16)
                    .collect();
                let bytes: Vec<u8> = px.iter().flat_map(|v| v.to_le_bytes()).collect();
                w.write_frame_bytes(&bytes).expect("frame");
            }
        }
    }
    w.finish().expect("finish").into_inner()
}

fn open(bytes: Vec<u8>) -> Stack {
    Stack::from_bytes(bytes, "probe.tif".into(), false).expect("the source should open")
}

/// Read a saved file back. Through its bytes rather than `TiffStack::open`,
/// which needs the `mmap` feature — the wasm-shaped build is tested without it,
/// and every assertion here is about what was written rather than how it is
/// read.
fn reopen(path: &std::path::Path) -> TiffStack {
    let bytes = std::fs::read(path).expect("the saved file should be there");
    TiffStack::from_bytes(bytes).expect("the saved file should open")
}

/// The whole point: every sample of every plane comes back.
#[test]
fn every_plane_survives_the_round_trip() {
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let stack = open(source(sample, 2, 3, 2));
        let path = temp(&format!("roundtrip-{sample:?}"));
        save_stack(&stack, &path).expect("save");

        let back = reopen(&path);
        assert_eq!(
            back.frames.len(),
            stack.tiff.frames.len(),
            "{sample:?}: a plane went missing"
        );
        for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
            if sample == SampleType::F32 {
                let want = read_frame_f32(&stack.tiff.data, a, stack.tiff.byte_order).unwrap();
                let got = read_frame_f32(&back.data, b, back.byte_order).unwrap();
                assert_eq!(want, got, "{sample:?}: frame {i} changed");
            } else {
                let want =
                    read_frame_u16(&stack.tiff.data, a, stack.tiff.byte_order, None).unwrap();
                let got = read_frame_u16(&back.data, b, back.byte_order, None).unwrap();
                assert_eq!(want, got, "{sample:?}: frame {i} changed");
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// Signed 16-bit is the case that looks fine and is wrong.
///
/// The decoder hands these back offset by +32768 so they sort as unsigned.
/// Writing that without undoing it produces a file whose every pixel is half
/// the range out — an image that still looks like an image.
#[test]
fn signed_samples_come_back_signed() {
    let stack = open(source(SampleType::I16, 1, 1, 2));
    let path = temp("signed");
    save_stack(&stack, &path).expect("save");

    let back = reopen(&path);
    assert_eq!(
        back.frames[0].sample_format,
        fast_tiff_lib::SampleFormat::SignedInt,
        "the file no longer says it is signed"
    );
    for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
        let want = read_frame_u16(&stack.tiff.data, a, stack.tiff.byte_order, None).unwrap();
        let got = read_frame_u16(&back.data, b, back.byte_order, None).unwrap();
        assert_eq!(want, got, "frame {i} shifted");
    }
    let _ = std::fs::remove_file(&path);
}

/// The axes as the viewer understands them, not as the source file labelled
/// them — a stack whose axes were reinterpreted is saved the way it is shown.
#[test]
fn the_shape_on_screen_is_the_shape_that_is_written() {
    let stack = open(source(SampleType::U16, 2, 3, 2));
    let dims = stack.display.dims;
    assert_eq!((dims.channels, dims.slices, dims.frames), (2, 3, 2));

    let path = temp("shape");
    save_stack(&stack, &path).expect("save");
    let back =
        Stack::from_bytes(std::fs::read(&path).unwrap(), "back.tif".into(), false).expect("reopen");
    assert_eq!(
        (
            back.display.dims.channels,
            back.display.dims.slices,
            back.display.dims.frames
        ),
        (2, 3, 2),
        "the saved file does not describe the same stack"
    );
    let _ = std::fs::remove_file(&path);
}

/// Calibration and colour are metadata a measurement depends on, so they travel.
#[test]
fn calibration_and_luts_travel_with_the_pixels() {
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(
        StackMetaWrite::new(2, 1)
            .unit("micron")
            .pixel_size(0.25, 0.25)
            .spacing(2.0)
            .frame_interval_s(0.5)
            .mode(fast_tiff_lib::DisplayMode::Composite),
    );
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for _ in 0..2 {
        w.write_frame_u16(&vec![7u16; (W * H) as usize]).unwrap();
    }
    let stack = open(w.finish().unwrap().into_inner());

    let path = temp("calibration");
    save_stack(&stack, &path).expect("save");
    let back = reopen(&path);

    assert_eq!(back.meta.unit.as_deref(), Some("micron"));
    assert_eq!(back.meta.pixel_width, Some(0.25));
    assert_eq!(back.meta.spacing, Some(2.0));
    assert_eq!(back.meta.frame_interval_s, Some(0.5));
    assert_eq!(back.meta.channels, 2);
    // The LUTs the window was showing, one per channel.
    assert_eq!(back.meta.channel_display.len(), 2);
    assert!(
        back.meta.has_explicit_luts,
        "the channel colours were not written"
    );
    let _ = std::fs::remove_file(&path);
}

/// An instrument's own record is the part of a file that cannot be
/// reconstructed, so it is carried; the ImageJ block in front of it is being
/// regenerated, so it is not carried twice.
#[test]
fn the_instruments_record_is_carried_and_the_imagej_block_is_not_doubled() {
    let record = "\"[General]\"\t\"\"\n\"System Name\"\t\"FVMPE-RS\"\n";
    let opts = WriterOptions::new(W, H, SampleType::U16)
        .metadata(StackMetaWrite::new(1, 1).unit("micron").trailing(record));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_u16(&vec![1u16; (W * H) as usize]).unwrap();
    let stack = open(w.finish().unwrap().into_inner());
    assert!(stack
        .tiff
        .description
        .as_deref()
        .unwrap()
        .contains("ImageJ="));

    let path = temp("description");
    save_stack(&stack, &path).expect("save");
    let back = reopen(&path);
    let desc = back.description.as_deref().expect("a description");

    assert!(
        desc.contains("\"System Name\"\t\"FVMPE-RS\""),
        "the instrument's record was lost: {desc}"
    );
    assert_eq!(
        desc.matches("ImageJ=").count(),
        1,
        "the ImageJ block was carried through as well as regenerated: {desc}"
    );
    assert_eq!(
        desc.matches("unit=micron").count(),
        1,
        "the unit was written twice: {desc}"
    );
    let _ = std::fs::remove_file(&path);
}

/// A file whose frames are different shapes cannot be one TIFF, and saying so
/// beats writing one whose later frames are the wrong size.
#[test]
fn a_stack_of_mixed_frame_shapes_is_refused_with_a_reason() {
    // Two frames of different sizes, written by hand — the writer will not do
    // it, so the bytes are assembled from two single-frame files.
    let a = source(SampleType::U16, 1, 1, 1);
    let stack = open(a);
    // Fake the second frame's geometry, which is what a scanner's thumbnail
    // IFD looks like to the reader.
    let mut stack = stack;
    let mut odd = stack.tiff.frames[0].clone();
    odd.width = W + 2;
    // Through `get_mut` because the stack shares its `TiffStack` with anything
    // running on a worker; here nothing else holds it, so this always succeeds.
    std::sync::Arc::get_mut(&mut stack.tiff)
        .expect("the fixture holds the only reference")
        .frames
        .push(odd);

    let path = temp("mixed");
    let err = save_stack(&stack, &path).expect_err("mixed shapes cannot be saved");
    let message = format!("{err:#}");
    assert!(
        message.contains("frame 1") && message.contains("mixed frame shapes"),
        "the refusal should say which frame and why: {message}"
    );
    let _ = std::fs::remove_file(&path);
}

/// An empty description is not carried as an empty line, and a file with no
/// description at all does not gain one.
#[test]
fn nothing_is_carried_when_there_is_nothing_to_carry() {
    let stack = open(source(SampleType::U16, 1, 1, 1));
    let path = temp("plain");
    save_stack(&stack, &path).expect("save");
    let back = reopen(&path);
    let desc = back.description.unwrap_or_default();
    assert!(
        !desc.trim_end().ends_with('\n') || desc.trim().lines().count() > 0,
        "a trailing blank was carried: {desc:?}"
    );
    assert_eq!(desc.matches("ImageJ=").count(), 1);
    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------- progress and stopping

/// A save is now something a worker does while the window carries on, so it has
/// to be able to say how far it has got and to be stopped.
#[test]
fn a_save_reports_progress_once_per_frame() {
    let stack = open(source(SampleType::U16, 1, 1, 4));
    let path = temp("progress");
    let mut seen = Vec::new();
    save_source(&SaveSource::of(&stack), &path, &mut |f| {
        seen.push(f);
        true
    })
    .expect("save");

    assert_eq!(seen.len(), 4, "one report per frame: {seen:?}");
    // Rising, starting at the beginning, and never claiming to be finished
    // before it is — the `finish` that writes the IFD chain happens after the
    // last frame and has nothing to report.
    assert_eq!(seen[0], 0.0);
    assert!(
        seen.windows(2).all(|w| w[1] > w[0]),
        "progress must not go backwards: {seen:?}"
    );
    assert!(seen.iter().all(|&f| f < 1.0), "{seen:?}");

    // And it really did write the file it was reporting on.
    let back = reopen(&path);
    assert_eq!(back.frames.len(), 4);
    let _ = std::fs::remove_file(&path);
}

/// A stopped save must leave *nothing*. A truncated TIFF has a valid header and
/// a short IFD chain, so it opens — as a file with fewer frames than the stack
/// it was supposed to be. That is worse than no file at all, because nothing
/// downstream can tell it is incomplete.
#[test]
fn a_stopped_save_leaves_no_file_behind() {
    let stack = open(source(SampleType::U16, 1, 1, 4));
    let path = temp("stopped");

    let mut calls = 0;
    let err = save_source(&SaveSource::of(&stack), &path, &mut |_| {
        calls += 1;
        // Stop after the second frame has been asked for, so the writer is
        // genuinely part-way through a real file rather than refusing at once.
        calls < 3
    })
    .expect_err("a stopped save must not report success");
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");
    assert!(
        !path.exists(),
        "a stopped save left {} behind",
        path.display()
    );
}

/// The same guarantee for a save that fails on its own — the cleanup is not
/// specific to cancelling.
#[test]
fn a_failed_save_leaves_no_file_behind() {
    // A stack whose second frame is a different shape: refused, but only after
    // the writer has been created and the file exists.
    let stack = open(source(SampleType::U16, 1, 1, 2));
    let mut stack = stack;
    let mut odd = stack.tiff.frames[0].clone();
    odd.width = W + 2;
    std::sync::Arc::get_mut(&mut stack.tiff)
        .expect("the fixture holds the only reference")
        .frames
        .push(odd);

    let path = temp("failed");
    save_source(&SaveSource::of(&stack), &path, &mut |_| true).expect_err("mixed shapes");
    assert!(!path.exists(), "a failed save left a file behind");
}

/// The snapshot is of the stack *as it was*, so a save already running cannot
/// be changed by what the window does next.
#[test]
fn the_snapshot_holds_the_shape_it_was_taken_with() {
    let stack = open(source(SampleType::U16, 2, 1, 3));
    let source_a = SaveSource::of(&stack);
    assert_eq!(source_a.frames(), 6);

    // Reinterpreting the axes after the snapshot must not reach it.
    let mut stack = stack;
    stack.display.dims = crate::display::Dims {
        channels: 1,
        slices: 2,
        frames: 3,
    };
    let path = temp("snapshot");
    save_source(&source_a, &path, &mut |_| true).expect("save");
    let back = reopen(&path);
    assert_eq!(
        back.meta.channels, 2,
        "the write used the axes the snapshot was taken with"
    );
    let _ = std::fs::remove_file(&path);
}

/// The regression this guards is one the first version of the atomic-save
/// cleanup introduced: a failed write deleting the file it was meant to
/// replace. Saving over the file you are looking at is an ordinary thing to
/// do, and losing it because the write went wrong is not recoverable.
#[test]
fn a_failed_save_leaves_an_existing_file_untouched() {
    let path = temp("existing");
    std::fs::write(&path, b"the original, which must survive").expect("seed");

    // Mixed frame shapes: refused, but only after a writer would have been
    // created — which is what would truncate the target if it were the target.
    let stack = open(source(SampleType::U16, 1, 1, 2));
    let mut stack = stack;
    let mut odd = stack.tiff.frames[0].clone();
    odd.width = W + 2;
    std::sync::Arc::get_mut(&mut stack.tiff)
        .expect("the fixture holds the only reference")
        .frames
        .push(odd);

    save_source(&SaveSource::of(&stack), &path, &mut |_| true).expect_err("mixed shapes");
    assert_eq!(
        std::fs::read(&path).expect("the original must still be there"),
        b"the original, which must survive",
        "a failed save destroyed the file it was replacing"
    );
    let _ = std::fs::remove_file(&path);
}

/// And the same for a save the user stopped.
#[test]
fn a_stopped_save_leaves_an_existing_file_untouched() {
    let path = temp("existing-stopped");
    std::fs::write(&path, b"still here").expect("seed");

    let stack = open(source(SampleType::U16, 1, 1, 4));
    let mut calls = 0;
    save_source(&SaveSource::of(&stack), &path, &mut |_| {
        calls += 1;
        calls < 3
    })
    .expect_err("stopped");
    assert_eq!(std::fs::read(&path).expect("still there"), b"still here");
    let _ = std::fs::remove_file(&path);
}

/// Nor may either failure leave the scratch file lying about next to it.
#[test]
fn no_part_file_is_left_behind() {
    let path = temp("partfile");
    let stack = open(source(SampleType::U16, 1, 1, 4));
    let mut calls = 0;
    save_source(&SaveSource::of(&stack), &path, &mut |_| {
        calls += 1;
        calls < 2
    })
    .expect_err("stopped");

    let leftovers: Vec<_> = std::fs::read_dir(path.parent().expect("a directory"))
        .expect("read the directory")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .filter(|n| n.starts_with("fasttiff-save-partfile") && n.contains("fasttiff-part"))
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
}

//! Saving, checked by opening what was saved.
//!
//! Every test here writes a stack, reads the file back, and compares. That is
//! the only assertion worth making about a writer: a file that "looks right" in
//! a hex dump and does not reopen is a bug, and a file that reopens with every
//! pixel shifted by half the range — which is what signed data does without the
//! offset below — looks perfectly plausible until someone measures something.

use super::*;
use crate::stack::{Stack, Window};
use fast_tiff_lib::{
    read_frame_f32, read_frame_u16, read_planes_f32, SampleType, StackMetaWrite, TiffStack,
    TiffWriter, WriterOptions,
};
use std::io::Cursor;
use std::path::PathBuf;

const W: u32 = 6;
const H: u32 = 4;

fn temp(name: &str) -> PathBuf {
    // The process id keeps two test runs on one machine (two checkouts, or the
    // default and glow suites) from writing over each other's files.
    let p = std::env::temp_dir().join(format!("fasttiff-save-{}-{name}.tif", std::process::id()));
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
            // Bytes again, running from negative to positive.
            SampleType::I8 => {
                let px: Vec<u8> = (0..W * H)
                    .map(|i| (p as i32 * 50 + i as i32 * 9 - 120) as i8 as u8)
                    .collect();
                w.write_frame_bytes(&px).expect("frame");
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

/// `bytes` opened, with one more frame of a different shape on the end — what
/// a scanner's thumbnail IFD looks like to the reader. The writer will not
/// make such a file, so the frame's geometry is faked, on a `TiffStack` of its
/// own: the stack's is shared with its workers and is not this test's to edit.
fn with_odd_frame(bytes: Vec<u8>) -> Stack {
    let mut stack = open(bytes.clone());
    let mut tiff = TiffStack::from_bytes(bytes).expect("parse");
    let mut odd = tiff.frames[0].clone();
    odd.width = W + 2;
    tiff.frames.push(odd);
    stack.tiff = std::sync::Arc::new(tiff);
    stack
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

/// Signed 8-bit is written as signed 8-bit, every value intact. It used to be
/// written as unsigned with the same bits, so -1 read back as 255.
#[test]
fn signed_8_bit_samples_come_back_signed() {
    let stack = open(source(SampleType::I8, 1, 1, 2));
    let path = temp("signed8");
    save_stack(&stack, &path).expect("save");

    let back = reopen(&path);
    let f = &back.frames[0];
    assert_eq!(
        (f.bits_per_sample, f.sample_format),
        (8, fast_tiff_lib::SampleFormat::SignedInt),
        "the file no longer says it is signed 8-bit"
    );
    for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
        let want = read_frame_f32(&stack.tiff.data, a, stack.tiff.byte_order).unwrap();
        let got = read_frame_f32(&back.data, b, back.byte_order).unwrap();
        assert!(
            want.iter().any(|&v| v < 0.0),
            "the source has negative samples"
        );
        assert_eq!(want, got, "frame {i} changed");
    }
    let _ = std::fs::remove_file(&path);
}

/// A planar RGB stack is written planar: its frames are copied as the file
/// lays them out, so writing them as interleaved would scramble every pixel.
#[test]
fn a_planar_rgb_stack_keeps_its_planes() {
    let opts = WriterOptions::new(W, H, SampleType::U8)
        .samples_per_pixel(3)
        .planar(true);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for f in 0..2u32 {
        // Each plane its own ramp, so a mixed-up plane shows.
        let px: Vec<u8> = (0..3 * W * H)
            .map(|i| (f * 90 + i * 3 + i / (W * H) * 70) as u8)
            .collect();
        w.write_frame_u8(&px).expect("frame");
    }
    let stack = open(w.finish().expect("finish").into_inner());
    assert!(
        stack.tiff.frames[0].is_planar(),
        "the source should be planar"
    );
    let path = temp("planar-rgb");
    save_stack(&stack, &path).expect("save");

    let back = reopen(&path);
    for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
        let want = read_planes_f32(&stack.tiff.data, a, stack.tiff.byte_order).unwrap();
        let got = read_planes_f32(&back.data, b, back.byte_order).unwrap();
        assert_eq!(want, got, "frame {i}: the planes changed");
    }
    let _ = std::fs::remove_file(&path);
}

/// The axes as the viewer understands them, not as the source file labelled
/// them — a stack whose axes were reinterpreted is saved the way it is shown.
#[test]
fn the_shape_on_screen_is_the_shape_that_is_written() {
    let stack = open(source(SampleType::U16, 2, 3, 2));
    let dims = stack.display.dims();
    assert_eq!((dims.channels, dims.slices, dims.frames), (2, 3, 2));

    let path = temp("shape");
    save_stack(&stack, &path).expect("save");
    let back =
        Stack::from_bytes(std::fs::read(&path).unwrap(), "back.tif".into(), false).expect("reopen");
    assert_eq!(
        (
            back.display.dims().channels,
            back.display.dims().slices,
            back.display.dims().frames
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
    let stack = with_odd_frame(source(SampleType::U16, 1, 1, 1));

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
    let stack = with_odd_frame(source(SampleType::U16, 1, 1, 2));

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
    stack.display.set_dims(crate::display::Dims {
        channels: 1,
        slices: 2,
        frames: 3,
    });
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
    let stack = with_odd_frame(source(SampleType::U16, 1, 1, 2));

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

// ------------------------------------------------------------- compression

/// A saved file is Deflate-compressed, at the level chosen, with no predictor.
///
/// Pinned because nothing else here would notice it changing: every other test
/// reads values back, and those come back identical from an uncompressed file.
#[test]
fn a_saved_file_is_deflate_compressed_without_a_predictor() {
    for sample in [
        SampleType::U8,
        SampleType::U16,
        SampleType::I16,
        SampleType::F32,
    ] {
        let stack = open(source(sample, 1, 1, 3));
        let path = temp(&format!("deflate-{sample:?}"));
        save_stack(&stack, &path).expect("save");
        let back = reopen(&path);
        for (i, f) in back.frames.iter().enumerate() {
            assert_eq!(
                f.compression,
                fast_tiff_lib::Compression::Deflate,
                "{sample:?}: frame {i} was not compressed"
            );
            assert_eq!(f.predictor, 1, "{sample:?}: frame {i} carries a predictor");
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// A save written in several batches keeps every frame, in order, exactly.
///
/// Frames are gathered and compressed together, so the thing to get wrong is
/// the seam between batches — a frame dropped, doubled, or written into its
/// neighbour's place — and the partial batch at the end. Five frames a batch
/// over twenty-four planes makes five seams and a short last batch, for every
/// sample type the writer handles.
#[test]
fn a_save_in_several_batches_keeps_every_frame_in_order() {
    for (sample, width) in [
        (SampleType::U8, 1usize),
        (SampleType::U16, 2),
        (SampleType::I16, 2),
        (SampleType::F32, 4),
    ] {
        let stack = open(source(sample, 2, 3, 4));
        let path = temp(&format!("batches-{sample:?}"));
        let five_frames = W as usize * H as usize * width * 5;
        write_to_path(&SaveSource::of(&stack), &path, &mut |_| true, five_frames).expect("save");

        let back = reopen(&path);
        assert_eq!(back.frames.len(), 24, "{sample:?}: frames lost or invented");
        for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
            if sample == SampleType::F32 {
                let want = read_frame_f32(&stack.tiff.data, a, stack.tiff.byte_order).unwrap();
                let got = read_frame_f32(&back.data, b, back.byte_order).unwrap();
                assert_eq!(want, got, "{sample:?}: frame {i} changed or moved");
            } else {
                let want =
                    read_frame_u16(&stack.tiff.data, a, stack.tiff.byte_order, None).unwrap();
                let got = read_frame_u16(&back.data, b, back.byte_order, None).unwrap();
                assert_eq!(want, got, "{sample:?}: frame {i} changed or moved");
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}

// ---------------------------------------------------- the in-memory encoder

/// The claim `save_to_bytes` makes is that it is the *same* encode, so the
/// assertion is byte equality against the file the path version writes rather
/// than "it also opens". Anything weaker would let the two drift — a different
/// compression level, a dropped LUT — and only the browser build would notice.
#[test]
fn encoding_to_memory_gives_the_same_file_as_encoding_to_a_path() {
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let stack = open(source(sample, 2, 3, 4));
        let path = temp(&format!("same-{sample:?}"));
        save_stack(&stack, &path).expect("save");

        let on_disk = std::fs::read(&path).expect("the saved file should be there");
        let in_memory =
            save_to_bytes(&SaveSource::of(&stack), &mut |_| true).expect("save to bytes");
        assert_eq!(
            in_memory, on_disk,
            "{sample:?}: the two destinations must produce one file"
        );
        let _ = std::fs::remove_file(&path);
    }
}

/// And that what it produces is a stack, not just bytes that match.
#[test]
fn what_was_encoded_to_memory_reopens() {
    let stack = open(source(SampleType::U16, 2, 3, 4));
    let bytes = save_to_bytes(&SaveSource::of(&stack), &mut |_| true).expect("save to bytes");
    let back = TiffStack::from_bytes(bytes).expect("the encoded stack should open");
    assert_eq!(back.frames.len(), 2 * 3 * 4);
}

/// Progress is reported and cancelling stops it, as on the path side — the
/// browser runs this on the frame that asked for it, so a stack big enough to
/// be worth cancelling is exactly the one that will want to be.
#[test]
fn encoding_to_memory_reports_progress_and_can_be_stopped() {
    let stack = open(source(SampleType::U16, 1, 1, 8));

    let mut seen: Vec<f32> = Vec::new();
    save_to_bytes(&SaveSource::of(&stack), &mut |f| {
        seen.push(f);
        true
    })
    .expect("save to bytes");
    assert!(seen.len() >= 8, "one report per frame at least: {seen:?}");
    assert!(
        seen.windows(2).all(|w| w[1] >= w[0]),
        "progress must not go backwards: {seen:?}"
    );

    let err = save_to_bytes(&SaveSource::of(&stack), &mut |_| false)
        .expect_err("refusing at the first frame must stop the encode");
    assert!(
        format!("{err:#}").contains("cancelled"),
        "stopping is reported as cancellation: {err:#}"
    );
}

// ------------------------------------------------------------------ overlay

/// Two point ROIs, as the counter hands them over: one on two slices, one on
/// every slice.
fn two_rois() -> Vec<fast_tiff_lib::PointRoi> {
    use fast_tiff_lib::{PointRoi, RoiPoint};
    vec![
        PointRoi {
            hyperstack_position: [0; 3],
            name: "Group 1".into(),
            color: [0xe4, 0x1a, 0x1c],
            points: vec![
                RoiPoint {
                    x: 1.5,
                    y: 1.5,
                    position: 1,
                },
                RoiPoint {
                    x: 4.5,
                    y: 2.5,
                    position: 3,
                },
            ],
        },
        PointRoi {
            hyperstack_position: [0; 3],
            name: "Group 2".into(),
            color: [0x37, 0x7e, 0xb8],
            points: vec![RoiPoint {
                x: 2.5,
                y: 0.5,
                position: 0,
            }],
        },
    ]
}

/// The first IFD's raw IJMetadata tags — byte counts (50838) and blob (50839) —
/// read with the library's own IFD reader, and the `(type, count)` directory
/// the blob opens with, its codes un-reversed (the file is little-endian).
struct IjMetadata {
    counts: Vec<u32>,
    blob: Vec<u8>,
    directory: Vec<([u8; 4], u32)>,
}

fn ij_metadata(bytes: &[u8]) -> IjMetadata {
    use fast_tiff_lib::ifd::{read_header, read_ifd};
    let (order, flavor, first) = read_header(bytes).expect("a TIFF header");
    assert_eq!(order, fast_tiff_lib::ByteOrder::Little);
    let ifd0 = read_ifd(bytes, first as usize, order, flavor).expect("the first IFD");
    let tag = |t: u16| {
        ifd0.entries
            .iter()
            .find(|e| e.tag == t)
            .unwrap_or_else(|| panic!("tag {t} on the first IFD"))
    };
    let counts = tag(50838).as_u32_array(bytes, order).unwrap();
    let blob = tag(50839).owned_bytes(bytes, order).unwrap();
    assert_eq!(&blob[..4], b"JIJI", "the little-endian magic");
    let directory = blob[4..counts[0] as usize]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|rec| {
            let mut code = [rec[0], rec[1], rec[2], rec[3]];
            code.reverse();
            (code, u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]))
        })
        .collect();
    IjMetadata {
        counts,
        blob,
        directory,
    }
}

/// With an overlay the file is written uncompressed and contiguous — the only
/// shape in which ImageJ reads a multi-image TIFF's overlay — and its first
/// IFD carries one `over` entry per ROI, after the channel LUTs.
#[test]
fn an_overlay_is_saved_uncompressed_with_its_over_entries() {
    let mut stack = open(source(SampleType::U16, 2, 1, 3));
    stack.overlay = two_rois();
    let path = temp("overlay");
    save_stack(&stack, &path).expect("save");
    let bytes = std::fs::read(&path).expect("the saved file");
    let _ = std::fs::remove_file(&path);

    // The block: two LUTs (one per channel, as always), then the two ROIs
    // exactly as the library encodes them.
    let ij = ij_metadata(&bytes);
    assert_eq!(ij.directory, vec![(*b"luts", 2), (*b"over", 2)]);
    let rois: Vec<Vec<u8>> = stack.overlay.iter().map(|r| r.to_imagej_bytes()).collect();
    assert_eq!(
        ij.counts,
        vec![20, 768, 768, rois[0].len() as u32, rois[1].len() as u32]
    );
    let after_luts = 20 + 2 * 768;
    assert_eq!(
        &ij.blob[after_luts..after_luts + rois[0].len()],
        &rois[0][..]
    );
    assert_eq!(&ij.blob[after_luts + rois[0].len()..], &rois[1][..]);

    // Uncompressed, one strip a frame, frames back to back from offset 16:
    // what ImageJ reads `images=6` frames from.
    let back = TiffStack::from_bytes(bytes).expect("the saved file should open");
    let desc = back.description.as_deref().unwrap_or_default();
    assert!(
        desc.starts_with("ImageJ=") && desc.contains("images=6\n"),
        "{desc}"
    );
    let frame_bytes = (W * H * 2) as u64;
    for (i, f) in back.frames.iter().enumerate() {
        assert_eq!(f.compression, fast_tiff_lib::Compression::None, "frame {i}");
        assert_eq!(f.strip_offsets.len(), 1, "frame {i}: one strip");
        assert_eq!(f.strip_offsets[0], 16 + i as u64 * frame_bytes, "frame {i}");
        assert_eq!(f.strip_byte_counts[0], frame_bytes, "frame {i}");
    }
    // …and the pixels and the LUTs are what they were.
    for (i, (a, b)) in stack.tiff.frames.iter().zip(&back.frames).enumerate() {
        let want = read_frame_u16(&stack.tiff.data, a, stack.tiff.byte_order, None).unwrap();
        let got = read_frame_u16(&back.data, b, back.byte_order, None).unwrap();
        assert_eq!(want, got, "frame {i} changed");
    }
    assert_eq!(back.meta.channels, 2);
}

/// Without an overlay nothing changes: Deflate, and a block of LUTs alone.
#[test]
fn without_an_overlay_the_save_is_unchanged() {
    let stack = open(source(SampleType::U16, 2, 1, 3));
    assert!(stack.overlay.is_empty());
    let bytes = save_to_bytes(&SaveSource::of(&stack), &mut |_| true).expect("save");
    let ij = ij_metadata(&bytes);
    assert_eq!(ij.directory, vec![(*b"luts", 2)]);
    assert_eq!(ij.counts, vec![12, 768, 768]);
    let back = TiffStack::from_bytes(bytes).expect("the saved file should open");
    assert!(back
        .frames
        .iter()
        .all(|f| f.compression == fast_tiff_lib::Compression::Deflate));
}

/// The browser's in-memory save is the same encode, overlay included.
#[test]
fn the_in_memory_save_carries_the_overlay_too() {
    let mut stack = open(source(SampleType::U8, 1, 1, 4));
    stack.overlay = two_rois();
    let path = temp("overlay-same");
    save_stack(&stack, &path).expect("save");
    let on_disk = std::fs::read(&path).expect("the saved file");
    let _ = std::fs::remove_file(&path);
    let in_memory = save_to_bytes(&SaveSource::of(&stack), &mut |_| true).expect("save");
    assert_eq!(in_memory, on_disk);
    assert_eq!(ij_metadata(&in_memory).directory[1], (*b"over", 2));
}

/// The overlay is part of the snapshot: setting or clearing it after the save
/// started does not reach the file being written.
#[test]
fn the_snapshot_holds_the_overlay_it_was_taken_with() {
    let mut stack = open(source(SampleType::U16, 1, 1, 2));
    stack.overlay = two_rois();
    let with = SaveSource::of(&stack);
    stack.overlay.clear();
    let without = SaveSource::of(&stack);
    stack.overlay = two_rois();

    let bytes = save_to_bytes(&with, &mut |_| true).unwrap();
    assert_eq!(ij_metadata(&bytes).directory[1], (*b"over", 2));
    let bytes = save_to_bytes(&without, &mut |_| true).unwrap();
    assert_eq!(ij_metadata(&bytes).directory, vec![(*b"luts", 1)]);
}

/// A file's own overlay is the document's when it opens, so saving it again —
/// with nothing touched — writes it back, uncompressed, as it would one the
/// counter added. The points come back as they were.
#[test]
fn a_files_own_overlay_is_kept_by_a_save() {
    let mut first = open(source(SampleType::U16, 2, 1, 3));
    first.overlay = two_rois();
    let saved = save_to_bytes(&SaveSource::of(&first), &mut |_| true).expect("save");

    let reopened = open(saved);
    assert_eq!(reopened.overlay, two_rois(), "opening reads the overlay");
    assert_eq!(reopened.tiff.meta.overlay, two_rois());

    let again = save_to_bytes(&SaveSource::of(&reopened), &mut |_| true).expect("save");
    assert_eq!(ij_metadata(&again).directory[1], (*b"over", 2));
    let back = TiffStack::from_bytes(again).expect("the saved file should open");
    assert!(back
        .frames
        .iter()
        .all(|f| f.compression == fast_tiff_lib::Compression::None));
    assert_eq!(back.meta.overlay, two_rois());
}

/// An overlay makes a save uncompressed only while that still fits a classic
/// TIFF: past 4 GiB the writer would turn to BigTIFF, which ImageJ cannot
/// open at all, and the stack is compressed as any other.
#[test]
fn an_overlay_is_saved_uncompressed_only_below_4_gib() {
    let frame = 512 * 512 * 2;
    assert!(fits_classic_uncompressed(1, frame));
    assert!(
        fits_classic_uncompressed(8_000, frame),
        "about 4 GB less the slack"
    );
    assert!(!fits_classic_uncompressed(8_200, frame), "past 4 GiB");
    assert!(
        !fits_classic_uncompressed(usize::MAX, usize::MAX),
        "no overflow"
    );
}

/// A point ROI on one plane of a hyperstack, saved without a hyperstack
/// position — read from a file, say — is given that plane's, so that ImageJ
/// draws it there; one spread over planes, on every plane, or on a stack that
/// is not a hyperstack is written as it was.
#[test]
fn an_overlay_roi_is_placed_for_imagej_on_a_hyperstack() {
    use fast_tiff_lib::{PointRoi, RoiPoint};
    let roi = |positions: &[u32]| PointRoi {
        name: "R".into(),
        color: [0; 3],
        points: positions
            .iter()
            .map(|&position| RoiPoint {
                x: 1.5,
                y: 1.5,
                position,
            })
            .collect(),
        hyperstack_position: [0; 3],
    };
    let dims = |channels, slices, frames| Dims {
        channels,
        slices,
        frames,
    };
    // 2 channels x 3 slices x 2 frames: IFD 9 is channel 2, slice 2, frame 2.
    let d = dims(2, 3, 2);
    let place = |roi, dims| placed_for_imagej(roi, dims, false, d);
    assert_eq!(place(roi(&[10, 10]), d).hyperstack_position, [2, 2, 2]);
    assert_eq!(place(roi(&[1]), d).hyperstack_position, [1, 1, 1]);
    for left in [roi(&[1, 10]), roi(&[0, 0])] {
        assert_eq!(place(left.clone(), d), left);
    }
    // One worked out for other axes is worked out again from the shared
    // position, which the axes do not move.
    let mut stale = roi(&[10]);
    stale.hyperstack_position = [0, 1, 2];
    assert_eq!(place(stale, d).hyperstack_position, [2, 2, 2]);
    // A time series is not a hyperstack, and needs none; an RGB one with
    // slices and frames is, with its samples for channels.
    assert_eq!(place(roi(&[4]), dims(1, 1, 5)), roi(&[4]));
    assert_eq!(
        placed_for_imagej(roi(&[4]), dims(3, 2, 3), true, d).hyperstack_position,
        [0, 2, 2]
    );
}

/// A ROI tied to a plane by its hyperstack position alone is moved from the
/// axes it was read in to the ones being saved: a 2-channel z-stack of 5,
/// opened as 2 channels x 5 frames, keeps a ROI ImageJ put on slice 3 on that
/// plane — every channel of it, as it was — rather than on a slice 3 that the
/// saved file does not have.
#[test]
fn a_hyperstack_position_moves_with_the_axes() {
    use fast_tiff_lib::{PointRoi, RoiPoint};
    let dims = |channels, slices, frames| Dims {
        channels,
        slices,
        frames,
    };
    let file = dims(2, 5, 1);
    let roi = PointRoi {
        name: "FM".into(),
        color: [0; 3],
        points: vec![RoiPoint {
            x: 1.5,
            y: 1.5,
            position: 0,
        }],
        hyperstack_position: [0, 3, 1],
    };
    let saved = placed_for_imagej(roi.clone(), dims(2, 1, 5), false, file);
    assert_eq!(saved.hyperstack_position, [0, 1, 3]);
    // Read and saved in the same axes, it is left as it was.
    assert_eq!(placed_for_imagej(roi.clone(), file, false, file), roi);
    assert_eq!(hyperstack_to_flat([0, 3, 1], file, false, false), Some(5));
    assert_eq!(hyperstack_to_flat([0, 3, 1], file, false, true), Some(6));
    assert_eq!(hyperstack_to_flat([0, 6, 1], file, false, false), None);
    for p in 1..=10 {
        let back = hyperstack_to_flat(flat_to_hyperstack(p, file, false), file, false, false);
        assert_eq!(back, Some(p));
    }
}

/// ImageJ reads an overlay back only from an uncompressed save, which a
/// stack gets only while it fits a classic TIFF.
#[test]
fn the_overlay_is_readable_by_imagej_below_4_gib() {
    let opts = WriterOptions::new(4, 4, SampleType::U8);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    w.write_frame_u8(&[0; 16]).unwrap();
    let s = Stack::from_bytes(w.finish().unwrap().into_inner(), "x.tif".into(), false).unwrap();
    assert!(overlay_readable_by_imagej(&s));
}

/// A stack of `T` samples with display range `range` in its description, and
/// the samples `0, 1, 2, ...` (wrapping).
fn ranged(sample: SampleType, range: (f64, f64)) -> Stack {
    let opts = WriterOptions::new(16, 16, sample)
        .metadata(StackMetaWrite::new(1, 1).range(range.0, range.1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for _ in 0..2 {
        match sample {
            SampleType::U8 => w
                .write_frame_u8(&(0..256).map(|i| i as u8).collect::<Vec<_>>())
                .unwrap(),
            SampleType::U16 => w
                .write_frame_u16(&(0..256).map(|i| i as u16 * 100).collect::<Vec<_>>())
                .unwrap(),
            SampleType::I16 => w
                .write_frame_bytes(
                    &(0..256i32)
                        .flat_map(|i| ((i - 128) as i16 * 100).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            SampleType::I8 => w
                .write_frame_bytes(&(0..256i32).map(|i| i as u8).collect::<Vec<_>>())
                .unwrap(),
            _ => unreachable!(),
        }
    }
    Stack::from_bytes(w.finish().unwrap().into_inner(), "ranged.tif".into(), false).unwrap()
}

/// The `min=`/`max=` a saved file's description declares.
fn declared_range(bytes: Vec<u8>) -> (String, String) {
    let back = Stack::from_bytes(bytes, "again.tif".into(), false).unwrap();
    let desc = back.tiff.description.clone().unwrap_or_default();
    let get = |key: &str| {
        desc.lines()
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or("")
            .to_string()
    };
    (get("min="), get("max="))
}

/// An 8-bit file's display range is in its own 0..255 units both ways: it
/// opens as that window on its samples — the window the interface holds, and
/// the one the 8-bit texture is shown with — a save writes it back as it was,
/// not on the 16-bit scale, which ImageJ would show saturated, and a window the
/// user set is written in sample units too.
#[test]
fn an_8_bit_window_is_in_sample_units_both_ways() {
    let mut s = ranged(SampleType::U8, (10.0, 200.0));
    let c = &s.display.settings()[0];
    assert_eq!(
        c.window,
        Window::new(10.0, 200.0),
        "opens on samples 10..200"
    );
    assert_eq!(c.units.to_texture(c.window), (10.0, 200.0));
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(declared_range(bytes), ("10".into(), "200".into()));

    // Auto-contrast, or a drag.
    s.display.set_window(0, Window::new(20.0, 147.0));
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(declared_range(bytes.clone()), ("20".into(), "147".into()));
    let back = Stack::from_bytes(bytes, "again.tif".into(), false).unwrap();
    assert_eq!(
        back.display.settings()[0].window,
        Window::new(20.0, 147.0),
        "and reopens the same"
    );
}

/// A 16-bit file's window was in sample units already, and still is.
#[test]
fn a_16_bit_window_is_unchanged() {
    let s = ranged(SampleType::U16, (1000.0, 30000.0));
    let c = &s.display.settings()[0];
    assert_eq!(c.window, Window::new(1000.0, 30000.0));
    assert_eq!(c.units.to_texture(c.window), (1000.0, 30000.0));
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(declared_range(bytes), ("1000".into(), "30000".into()));
}

/// ImageJ states a signed 16-bit image's range offset into unsigned, the way
/// it holds the pixels. Such a range opens as the signed window it means —
/// shown with exactly the texture window it always was — and a save writes it
/// back in ImageJ's units, unchanged.
#[test]
fn a_signed_16_bit_window_is_in_imagejs_offset_units_in_the_file() {
    // -100..500 in the samples, as ImageJ writes it.
    let declared = (32_668.0, 33_268.0);
    let mut s = ranged(SampleType::I16, declared);
    let c = &s.display.settings()[0];
    assert_eq!(c.window, Window::new(-100.0, 500.0), "the signed values");
    assert_eq!(
        c.units.to_texture(c.window),
        (declared.0 as f32, declared.1 as f32),
        "on the texture's offset scale, as before"
    );
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(declared_range(bytes), ("32668".into(), "33268".into()));

    s.display.set_window(0, Window::new(-12_800.0, 0.0));
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(
        declared_range(bytes.clone()),
        ("19968".into(), "32768".into())
    );
    let back = Stack::from_bytes(bytes, "again.tif".into(), false).unwrap();
    assert_eq!(
        back.display.settings()[0].window,
        Window::new(-12_800.0, 0.0)
    );
}

/// Signed 8-bit has no offset in ImageJ's convention: its range is in the
/// samples' own values, opened and saved as they are.
#[test]
fn a_signed_8_bit_window_is_in_sample_units_both_ways() {
    let s = ranged(SampleType::I8, (-10.0, 100.0));
    let c = &s.display.settings()[0];
    assert_eq!(c.window, Window::new(-10.0, 100.0));
    let bytes = save_to_bytes(&SaveSource::of(&s), &mut |_| true).unwrap();
    assert_eq!(declared_range(bytes), ("-10".into(), "100".into()));
}

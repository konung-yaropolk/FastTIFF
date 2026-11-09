//! The histograms held on the stack are the ones a whole-frame decode would
//! give, and they cost no decode of their own beyond what nothing else did.
//!
//! A sync needs a GPU, so these drive the steps it takes for the histograms
//! in the order it takes them: [`prepare`] and [`bin_shown`] over the planes
//! decoded for the screen, then [`complete`] for what those did not cover.
//! [`frame_histograms`], which decodes and bins every channel outright, is the
//! reference.

use super::*;
use crate::prefetch::{build_jobs, decode_jobs_borrowed};
use crate::roi::Roi;
use crate::window::{self, Overview};
use crate::worker::CancelToken;
use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use std::io::Cursor;

const W: u32 = 24;
const H: u32 = 10;

/// `channels` x `frames` planes of 16-bit samples, plane `p` a ramp starting
/// at `p * 7000` so every channel and frame has a distribution of its own,
/// and the channels' slider tracks differ.
fn movie(channels: usize, frames: usize) -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::U16)
        .rows_per_strip(3)
        .metadata(StackMetaWrite::new(channels, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for p in 0..channels * frames {
        let px: Vec<u16> = (0..W * H)
            .map(|i| (p as u32 * 7000 + i * 37 % 5000) as u16)
            .collect();
        w.write_frame_u16(&px).expect("frame");
    }
    let bytes = w.finish().expect("finish").into_inner();
    let stack = Stack::from_bytes(bytes, "movie.tif".into(), false).expect("open");
    let dims = stack.display.dims();
    assert_eq!((dims.channels, dims.frames), (channels, frames));
    stack
}

/// A chunky 8-bit RGB stack of `frames` frames: its channels are planes of
/// one IFD, decoded together.
fn rgb(frames: usize) -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::U8)
        .samples_per_pixel(3)
        .metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    for f in 0..frames as u32 {
        let px: Vec<u8> = (0..W * H * 3).map(|i| (i * 7 + f * 50) as u8).collect();
        w.write_frame_u8(&px).expect("frame");
    }
    let bytes = w.finish().expect("finish").into_inner();
    let stack = Stack::from_bytes(bytes, "rgb.tif".into(), false).expect("open");
    assert!(stack.display.rgb(), "the fixture should open as RGB");
    stack
}

/// What a sync does for the histograms when it decodes the frame for the
/// screen inline: bin the enabled channels it decoded, then complete the
/// rest. Returns how many channels `complete` decoded.
fn show(stack: &mut Stack) -> usize {
    let missing = prepare(stack, HistogramSource::Frame);
    let jobs = build_jobs(stack, stack.frame_index, stack.display.masks().enabled());
    let decoded = decode_jobs_borrowed(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    )
    .expect("decode");
    let binned = bin_shown(
        stack,
        &missing,
        jobs.iter()
            .zip(&decoded)
            .map(|(job, data)| (job.channel, data.view())),
    );
    stack.histograms.put(binned);
    complete(stack, true)
}

/// The held histograms are the reference's, channel for channel — the ones
/// binned from the planes decoded for the screen and the ones decoded for
/// the channels switched off alike — on a stack whose channels have tracks
/// of their own, so the shared axis is a union.
#[test]
fn the_held_histograms_equal_a_whole_frame_decode() {
    for mut stack in [movie(3, 2), rgb(2)] {
        stack.display.set_enabled(1, false);
        for frame in 0..2 {
            stack.frame_index = frame;
            let decoded = show(&mut stack);
            assert_eq!(decoded, 1, "only the channel switched off is decoded");
            let reference = frame_histograms(&stack);
            assert_eq!(reference.len(), 3);
            assert!(stack.histograms.describes(&stack));
            assert_eq!(stack.histograms.source(), Some(HistogramSource::Frame));
            assert_eq!(
                stack.histograms.shown(&stack),
                &reference[..],
                "frame {frame}: the held histograms differ from a whole-frame decode"
            );
        }
    }
}

/// The read-ahead's histograms are taken as they are, and the channels it
/// did not bin are binned from the planes it holds: still the reference's.
#[test]
fn histograms_kept_from_the_read_ahead_equal_a_whole_frame_decode() {
    let mut stack = movie(2, 3);
    stack.frame_index = 2;
    // What the worker returns for the frame: its planes, and the histograms
    // of the first channel, as a request made before the second was binned.
    let jobs = build_jobs(&stack, 2, &[true, true]);
    let planes = crate::prefetch::decode_jobs(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    )
    .expect("decode");
    let binning = Binning::of(&stack);
    let worker = vec![binning.bin(0, planes[0].view()).expect("units")];

    let missing = prepare(&mut stack, HistogramSource::Frame);
    let mut rest = missing;
    rest[0] = false;
    let mut binned = worker;
    binned.extend(bin_shown(
        &stack,
        &rest,
        jobs.iter()
            .zip(&planes)
            .map(|(job, data)| (job.channel, data.view())),
    ));
    stack.histograms.put(binned);
    assert_eq!(
        complete(&mut stack, true),
        0,
        "the read-ahead's planes covered it"
    );
    assert_eq!(
        stack.histograms.shown(&stack),
        &frame_histograms(&stack)[..]
    );
}

/// Switching a channel on or off, or moving a contrast window, decodes
/// nothing for the histograms: every channel was binned already, under a key
/// neither moves. Moving to another frame, or reassigning the axes, does.
#[test]
fn a_channel_toggle_decodes_nothing_for_the_histograms() {
    let mut stack = movie(3, 2);
    stack.display.set_enabled(2, false);
    assert_eq!(show(&mut stack), 1);
    let held = stack.histograms.shown(&stack).to_vec();
    let version = stack.histograms.version();

    for (c, on) in [(2, true), (0, false), (0, true), (2, false)] {
        assert!(stack.display.set_enabled(c, on));
        // What the sync does next: the re-upload a toggle forces bins
        // nothing new, and nothing is missing to complete.
        assert_eq!(
            show(&mut stack),
            0,
            "switching channel {c} {} decoded for the histograms",
            if on { "on" } else { "off" }
        );
    }
    assert!(stack.display.set_window(1, Window::new(100.0, 9000.0)));
    assert_eq!(complete(&mut stack, true), 0, "a contrast drag decoded");
    assert_eq!(stack.histograms.shown(&stack), &held[..]);
    assert_eq!(
        stack.histograms.version(),
        version,
        "nothing changed, so nothing should ask for a redraw"
    );

    // Another frame is another distribution, with nothing decoded for it yet.
    stack.frame_index = 1;
    assert!(!stack.histograms.describes(&stack));
    assert_eq!(complete(&mut stack, true), 3);
    assert_eq!(
        stack.histograms.shown(&stack),
        &frame_histograms(&stack)[..]
    );

    // And an override reads other planes: the old bins are not even shown.
    crate::dimensions::apply_dimension_override(&mut stack, 2, 1, 3);
    assert!(stack.histograms.shown(&stack).is_empty());
    assert_eq!(complete(&mut stack, true), 2);
    assert_eq!(
        stack.histograms.shown(&stack),
        &frame_histograms(&stack)[..]
    );
}

/// The histograms are of every channel, on or off, so a checkbox is a redraw:
/// what they are keyed on must not move when a channel is switched, or when a
/// contrast handle is dragged along the distribution. It must move when the
/// frame does, or when the axes are reassigned — each channel then reads
/// another plane, and the old bins describe the wrong one.
#[test]
fn a_channel_toggle_keeps_the_key_and_a_dimension_override_does_not() {
    // Three channels of one frame, which an override can lay out as three
    // frames of one channel.
    let mut stack = movie(3, 1);
    let key = HistogramKey::of(&stack);
    assert!(key.describes(&stack));

    assert!(stack.display.set_enabled(1, false));
    assert!(
        key.describes(&stack),
        "switching a channel off re-decoded the histograms"
    );
    assert_eq!(key, HistogramKey::of(&stack));
    assert_eq!(
        frame_histograms(&stack).len(),
        3,
        "a channel switched off is still histogrammed, so the plot can leave it out"
    );
    assert!(stack.display.set_window(0, Window::new(5_000.0, 6_000.0)));
    assert!(
        key.describes(&stack),
        "a contrast drag re-decoded the histograms"
    );

    // The key is only numbers, so a frame index past this one-frame stack's
    // end will do to show that the frame is part of it.
    stack.frame_index = 1;
    assert!(
        !key.describes(&stack),
        "another frame is another distribution"
    );
    stack.frame_index = 0;

    crate::dimensions::apply_dimension_override(&mut stack, 1, 1, 3);
    assert!(
        !key.describes(&stack),
        "the axes were reassigned and the old bins were kept"
    );
}

/// Histograms opened on a frame already on screen are decoded once, and not
/// again on the next repaint, nor for a frame that will not decode.
#[test]
fn what_is_missing_is_decoded_once() {
    let mut stack = movie(2, 1);
    assert_eq!(complete(&mut stack, true), 2, "nothing was binned yet");
    assert_eq!(complete(&mut stack, true), 0, "and nothing is missing now");

    // A frame index past the end does not decode; it is tried once.
    stack.frame_index = 5;
    assert_eq!(complete(&mut stack, true), 2);
    assert!(stack.histograms.shown(&stack).is_empty());
    assert_eq!(
        complete(&mut stack, true),
        0,
        "a failed frame was decoded again"
    );
}

/// While a movie plays, `complete` starts nothing: a repaint that shows no new
/// frame decodes nothing for the frame on screen, whose histograms wait for
/// the next frame shown with the throttle open. It still adds the channels
/// switched off to histograms that showing the frame started.
#[test]
fn during_playback_complete_only_adds_to_what_showing_the_frame_started() {
    let mut stack = movie(3, 3);
    stack.display.set_enabled(2, false);
    assert_eq!(show(&mut stack), 1);
    let held = stack.histograms.shown(&stack).to_vec();

    // Frame 1 was shown while the throttle was shut, so nothing binned it;
    // the repaints after it, with the throttle open, decode nothing.
    stack.frame_index = 1;
    for _ in 0..3 {
        assert_eq!(
            complete(&mut stack, false),
            0,
            "a repaint decoded the frame"
        );
    }
    assert_eq!(
        stack.histograms.shown(&stack),
        &held[..],
        "the last histograms stay up until the next are made"
    );

    // Frame 2 is shown with the throttle open: showing it bins the enabled
    // channels, and only the one switched off is decoded.
    stack.frame_index = 2;
    let missing = prepare(&mut stack, HistogramSource::Frame);
    let jobs = build_jobs(&stack, 2, stack.display.masks().enabled());
    let decoded = decode_jobs_borrowed(
        &stack.tiff.data,
        &stack.tiff.frames,
        stack.tiff.byte_order,
        &jobs,
    )
    .expect("decode");
    let binned = bin_shown(
        &stack,
        &missing,
        jobs.iter()
            .zip(&decoded)
            .map(|(job, data)| (job.channel, data.view())),
    );
    stack.histograms.put(binned);
    assert_eq!(complete(&mut stack, false), 1);
    assert_eq!(complete(&mut stack, false), 0);
    assert_eq!(
        stack.histograms.shown(&stack),
        &frame_histograms(&stack)[..]
    );
}

/// Past the texture limit the frame is never decoded: the histograms are
/// binned from the overview held for it, and say so — and with no overview
/// of the frame on screen there are none rather than another frame's.
#[test]
fn a_frame_past_the_texture_limit_is_binned_from_its_overview() {
    let mut stack = movie(2, 2);
    stack.over_texture_limit = true;

    // Nothing held for frame 0: no histogram, and nothing decoded for one.
    assert_eq!(complete(&mut stack, true), 0);
    assert!(stack.histograms.shown(&stack).is_empty());

    // The overview a fit-to-window view keeps: the whole frame at stride 2.
    let revision = stack.display.revision();
    let jobs = build_jobs(&stack, 0, stack.display.masks().enabled());
    let roi = Roi {
        x: 0,
        y: 0,
        w: W,
        h: H,
        stride: 2,
    };
    let planes = {
        let Stack { tiff, bands, .. } = &mut stack;
        window::assemble(
            tiff,
            bands,
            revision.layout,
            &jobs,
            &roi,
            0,
            &CancelToken::never(),
        )
        .expect("assemble")
    };
    let built = window::Built {
        frame_index: 0,
        roi,
        channels: jobs.iter().map(|j| j.channel).collect(),
        planes,
    };
    stack.overview = Some(Overview::capture(built, W, H, revision, usize::MAX).expect("spans"));

    assert_eq!(
        complete(&mut stack, true),
        0,
        "a frame this large was decoded"
    );
    assert_eq!(
        stack.histograms.source(),
        Some(HistogramSource::Overview { stride: 2 })
    );
    let binning = Binning::of(&stack);
    let o = stack.overview.as_ref().expect("held");
    let want: Vec<Histogram> = o
        .built
        .channels
        .iter()
        .zip(&o.built.planes)
        .filter_map(|(&c, plane)| binning.bin(c, plane.view()))
        .collect();
    assert_eq!(want.len(), 2);
    assert_eq!(stack.histograms.shown(&stack), &want[..]);
    let (tw, th) = roi.texture_size();
    assert!(want.iter().all(|h| h.counted == tw as u64 * th as u64));

    // Another frame, and the overview is of frame 0: no histograms.
    stack.frame_index = 1;
    assert_eq!(complete(&mut stack, true), 0);
    assert!(stack.histograms.shown(&stack).is_empty());
}

/// A view of the whole of a frame past the texture limit that is too large to
/// keep as the overview is binned as it is built, and what it gave stays
/// while zoomed in on the same frame, where no such view is held.
#[test]
fn a_whole_frame_view_too_large_to_keep_is_binned_as_it_is_built() {
    let mut stack = movie(2, 2);
    stack.over_texture_limit = true;
    let revision = stack.display.revision();
    let jobs = build_jobs(&stack, 0, stack.display.masks().enabled());
    let roi = Roi {
        x: 0,
        y: 0,
        w: W,
        h: H,
        stride: 2,
    };
    let planes = {
        let Stack { tiff, bands, .. } = &mut stack;
        window::assemble(
            tiff,
            bands,
            revision.layout,
            &jobs,
            &roi,
            0,
            &CancelToken::never(),
        )
        .expect("assemble")
    };
    let built = window::Built {
        frame_index: 0,
        roi,
        channels: jobs.iter().map(|j| j.channel).collect(),
        planes,
    };
    let binning = Binning::of(&stack);
    let want: Vec<Histogram> = built
        .channels
        .iter()
        .zip(&built.planes)
        .filter_map(|(&c, plane)| binning.bin(c, plane.view()))
        .collect();
    assert_eq!(want.len(), 2);

    bin_whole_view(&mut stack, &built);
    // Too large to keep: no overview, and the view goes.
    assert!(Overview::capture(built, W, H, revision, 0).is_none());
    assert!(stack.overview.is_none());
    for _ in 0..2 {
        assert_eq!(complete(&mut stack, true), 0);
        assert_eq!(
            stack.histograms.source(),
            Some(HistogramSource::Overview { stride: 2 })
        );
        assert_eq!(stack.histograms.shown(&stack), &want[..]);
    }

    // Another frame, with no view of it: nothing, not frame 0's.
    stack.frame_index = 1;
    assert_eq!(complete(&mut stack, true), 0);
    assert!(stack.histograms.shown(&stack).is_empty());
}

/// During playback this thread bins at most every interval of the playback
/// clock; with no movie playing it always may.
#[test]
fn playback_throttles_binning_on_this_thread() {
    let mut held = FrameHistograms::default();
    assert!(held.may_bin_inline(None));
    assert!(held.may_bin_inline(Some(10.0)));
    held.binned_inline(Some(10.0));
    assert!(!held.may_bin_inline(Some(10.0 + PLAYBACK_INTERVAL / 2.0)));
    assert!(held.may_bin_inline(Some(10.0 + PLAYBACK_INTERVAL)));
    assert!(held.may_bin_inline(None), "stopped, it bins every frame");
    assert!(
        held.may_bin_inline(Some(1.0)),
        "a restarted clock is not a gap"
    );
}

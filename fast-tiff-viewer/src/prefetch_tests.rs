//! The inline decode keeps the file's bytes when they already are the plane,
//! and the read-ahead answers only the frame and display revision asked for.

use super::*;
use fast_tiff_lib::{Compression, SampleType, TiffWriter, WriterOptions};

const W: u32 = 6;
const H: u32 = 4;

/// Any one revision, for a worker driven without a display.
#[cfg(feature = "threads")]
const REV: Revision = Revision {
    layout: 1,
    enabled: 1,
};

fn stack(compression: Compression) -> TiffStack {
    let opts = WriterOptions::new(W, H, SampleType::U16).compression(compression);
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let px: Vec<u16> = (0..(W * H) as u16).map(|i| i * 300).collect();
    w.write_frame_u16(&px).unwrap();
    TiffStack::from_bytes(w.finish().unwrap().into_inner()).unwrap()
}

fn job() -> ChannelJob {
    ChannelJob {
        channel: 0,
        ifd_idx: 0,
        plane: 0,
        kind: ChannelKind::Int16,
        rgb: false,
        width: W,
        height: H,
    }
}

/// An uncompressed plane comes back as a borrow of the file, so the upload is
/// the only copy; a compressed one is decoded into a buffer of its own. Both
/// hold the same samples, and the owned decode agrees with them.
#[test]
fn an_uncompressed_plane_is_borrowed_from_the_file() {
    let want: Vec<u16> = (0..(W * H) as u16).map(|i| i * 300).collect();
    for (compression, borrowed) in [(Compression::None, true), (Compression::Deflate, false)] {
        let tiff = stack(compression);
        let decoded =
            decode_jobs_borrowed(&tiff.data, &tiff.frames, tiff.byte_order, &[job()]).unwrap();
        match &decoded[0] {
            DecodedRef::U16(Cow::Borrowed(v)) => {
                assert!(borrowed, "{compression:?} should have been decoded");
                assert_eq!(*v, &want[..]);
            }
            DecodedRef::U16(Cow::Owned(v)) => {
                assert!(!borrowed, "{compression:?} should have been borrowed");
                assert_eq!(v, &want);
            }
            _ => panic!("16-bit samples should decode as U16"),
        }
        let owned = decode_jobs(&tiff.data, &tiff.frames, tiff.byte_order, &[job()]).unwrap();
        assert!(matches!(&owned[0], Decoded::U16(v) if *v == want));
    }
}

/// The read-ahead answers the frame asked for last, and nothing for the frames
/// asked for before it is left to take.
///
/// Which of those were replaced before they started and which were decoded
/// and then dropped is up to the scheduler, so this proves only which answer
/// wins. A decode never looks at its token; that the worker drops an answer
/// overtaken while it was computed is forced in the worker's tests
/// (`an_answer_overtaken_while_computed_is_never_published`).
#[cfg(feature = "threads")]
#[test]
fn the_read_ahead_answers_the_frame_asked_for_last() {
    use std::time::{Duration, Instant};
    let tiff = Arc::new(stack(Compression::Deflate));
    let p = Prefetcher::new(tiff, false).expect("the thread should spawn");
    for frame in [3, 4, 5] {
        p.request(REV, frame, vec![job()]);
    }
    let start = Instant::now();
    let got = loop {
        if let Some(r) = p.take_matching(REV, 5) {
            break r;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "the read-ahead never answered"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!((got.revision, got.frame_index), (REV, 5));
    let want: Vec<u16> = (0..(W * H) as u16).map(|i| i * 300).collect();
    assert!(matches!(&got.channels[..], [c] if matches!(&c.data, Decoded::U16(v) if *v == want)));
    assert!(p.take_matching(REV, 3).is_none());
    assert!(p.take_matching(REV, 4).is_none());
}

/// Two channels by three timepoints, deflated (so the stack reads ahead by
/// decoding), plane `i` filled with `i * 1000`.
#[cfg(feature = "threads")]
fn two_channel_movie() -> Stack {
    use fast_tiff_lib::StackMetaWrite;
    let opts = WriterOptions::new(W, H, SampleType::U16)
        .compression(Compression::Deflate)
        .metadata(StackMetaWrite::new(2, 1));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    for plane in 0..6u16 {
        w.write_frame_u16(&vec![plane * 1000; (W * H) as usize])
            .unwrap();
    }
    let bytes = w.finish().unwrap().into_inner();
    Stack::from_bytes(bytes, "movie.tif".into(), false).unwrap()
}

/// A frame read ahead under one channel layout is not taken under the next.
///
/// Reassigning the axes moves which IFD every display channel reads, so a
/// frame decoded just before is the old mapping's planes: frame 1's channel 0
/// was IFD 2, and after the swap it is IFD 3. It must be refused rather than
/// uploaded — and left where it is rather than lost, which is what shows it
/// was there to be refused.
#[cfg(feature = "threads")]
#[test]
fn a_frame_read_ahead_under_an_old_layout_is_refused() {
    use std::time::{Duration, Instant};
    let mut stack = two_channel_movie();
    let dims = stack.display.dims();
    assert_eq!((dims.channels, dims.slices, dims.frames), (2, 1, 3));
    let before = stack.display.revision();
    let jobs = build_jobs(&stack, 1, &[true, true]);
    let ifds: Vec<usize> = jobs.iter().map(|j| j.ifd_idx).collect();
    assert_eq!(ifds, [2, 3]);
    let p = stack
        .prefetch
        .as_ref()
        .expect("a compressed stack reads ahead");
    p.request(before, 1, jobs);

    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    let after = stack.display.revision();
    assert_ne!(after.layout, before.layout, "an override moves the layout");
    let moved = build_jobs(&stack, 1, &[true; 3]);
    assert_eq!(
        moved[0].ifd_idx, 3,
        "channel 0 of frame 1 reads another IFD"
    );

    let p = stack.prefetch.as_ref().unwrap();
    let start = Instant::now();
    loop {
        assert!(
            p.take_matching(after, 1).is_none(),
            "a frame read ahead under the old layout was taken under the new"
        );
        if let Some(r) = p.take_matching(before, 1) {
            assert_eq!(r.revision, before);
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "the read-ahead never answered"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

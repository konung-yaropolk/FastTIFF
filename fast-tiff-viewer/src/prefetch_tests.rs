//! The inline decode keeps the file's bytes when they already are the plane,
//! and the read-ahead answers only the frame and display revision asked for.

use super::*;
use fast_tiff_lib::{Compression, SampleType, TiffWriter, WriterOptions};

const W: u32 = 6;
const H: u32 = 4;

/// Any one revision, for a worker driven without a display.
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
/// the only copy; a compressed one is decoded into the channel's buffer. Both
/// hold the same samples, and the owned decode agrees with them.
#[test]
fn an_uncompressed_plane_is_borrowed_from_the_file() {
    let want: Vec<u16> = (0..(W * H) as u16).map(|i| i * 300).collect();
    for (compression, borrowed) in [(Compression::None, true), (Compression::Deflate, false)] {
        let tiff = stack(compression);
        let mut bufs = ChannelBufs::default();
        let mut scratch = ReadScratch::default();
        let planes = decode_jobs(&tiff, &[job()], None, &mut bufs, &mut scratch).unwrap();
        assert_eq!(planes.is_file(0), borrowed, "{compression:?}");
        assert!(
            matches!(planes.view(0, &bufs), Samples::U16(v) if v == &want[..]),
            "{compression:?}"
        );
        if borrowed {
            assert!(
                matches!(bufs.get(0), Decoded::U8(v) if v.is_empty()),
                "a borrowed plane leaves its buffer alone"
            );
        }
        let owned = decode_owned(&tiff, &[job()]).unwrap();
        assert!(matches!(&owned[0], Decoded::U16(v) if *v == want));
    }
}

/// A frame decoded again into buffers the last decode left is decoded into
/// the same allocations: what keeps the interface thread's decode of a
/// compressed movie free of frame-sized allocations.
#[test]
fn a_warm_decode_fills_the_same_buffers() {
    let tiff = stack(Compression::Deflate);
    let mut bufs = ChannelBufs::default();
    let mut scratch = ReadScratch::default();
    decode_jobs(&tiff, &[job()], None, &mut bufs, &mut scratch).unwrap();
    let Decoded::U16(first) = bufs.get(0) else {
        panic!("16-bit samples should decode as U16");
    };
    let first = first.as_ptr();
    decode_jobs(&tiff, &[job()], None, &mut bufs, &mut scratch).unwrap();
    let Decoded::U16(again) = bufs.get(0) else {
        panic!("16-bit samples should decode as U16");
    };
    assert_eq!(again.as_ptr(), first, "decoded into a fresh buffer");
}

/// A 16 x 12 frame of 7s in strips of 4 rows, `compression`ed.
fn strips(compression: Compression) -> TiffStack {
    let opts = WriterOptions::new(16, 12, SampleType::U16)
        .compression(compression)
        .rows_per_strip(4);
    let mut wr = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    wr.write_frame_u16(&[7; 16 * 12]).unwrap();
    TiffStack::from_bytes(wr.finish().unwrap().into_inner()).unwrap()
}

/// [`job`] for a frame of [`strips`].
fn strips_job() -> ChannelJob {
    ChannelJob {
        width: 16,
        height: 12,
        ..job()
    }
}

/// One of fast-tiff-lib's fixtures, when the checkout has them.
fn fixture(name: &str) -> Option<TiffStack> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("fast-tiff-lib/tests/fixtures")
        .join(name);
    TiffStack::from_bytes(std::fs::read(p).ok()?).ok()
}

/// The two uncompressed 16-bit fixtures, big- and little-endian, each with
/// its byte order and a job for its first frame.
fn byte_orders() -> Option<[(TiffStack, ByteOrder, ChannelJob); 2]> {
    let with_job = |tiff: TiffStack, order| {
        let f = &tiff.frames[0];
        assert_eq!(f.compression, Compression::None);
        let job = ChannelJob {
            width: f.width,
            height: f.height,
            ..job()
        };
        (tiff, order, job)
    };
    Some([
        with_job(fixture("tff_u16_spp1_p2_none-be.tif")?, ByteOrder::Big),
        with_job(fixture("tff_u16_spp1_p2_none-le.tif")?, ByteOrder::Little),
    ])
}

/// Whether two buffers hold the same samples, bit for bit.
fn same(a: &Decoded, b: &Decoded) -> bool {
    match (a.view(), b.view()) {
        (Samples::U8(a), Samples::U8(b)) => a == b,
        (Samples::U16(a), Samples::U16(b)) => a == b,
        (Samples::F32(a), Samples::F32(b)) => a
            .iter()
            .map(|v| v.to_bits())
            .eq(b.iter().map(|v| v.to_bits())),
        _ => false,
    }
}

/// The read-ahead touches a frame's pages only when every channel it is
/// asked for is read straight from the file, and decodes it ahead otherwise
/// — whatever the compression tag says. An uncompressed frame in many strips
/// laid end to end is one run of the plane, and borrowed; an uncompressed
/// big-endian one has to be byte-swapped on a little-endian host, so it is
/// decoded, as a compressed one is. (Before, the mode was the compression
/// tag's: the big-endian stack's frames were touched ahead and then swapped
/// on the interface thread.)
#[test]
fn the_read_ahead_touches_only_frames_read_straight_from_the_file() {
    let contiguous = strips(Compression::None);
    assert_eq!(
        contiguous.frames[0].strip_offsets.len(),
        3,
        "several strips"
    );
    assert!(
        touch_suffices(&contiguous, &[strips_job()]),
        "contiguous uncompressed strips are borrowed, so touched ahead"
    );
    assert!(
        !touch_suffices(&strips(Compression::Deflate), &[strips_job()]),
        "a compressed frame is decoded ahead"
    );
    assert!(!touch_suffices(&contiguous, &[]), "nothing to read");

    let Some(orders) = byte_orders() else {
        eprintln!("skipping: the fast-tiff-lib fixtures are not here");
        return;
    };
    for (tiff, order, job) in orders {
        assert_eq!(
            touch_suffices(&tiff, &[job]),
            order == ByteOrder::host(),
            "an uncompressed {order:?} frame on a {:?} host",
            ByteOrder::host()
        );
    }
}

/// And the worker goes by it, request by request, run here as its thread
/// runs it: a frame read straight from the file is touched, publishes
/// nothing (so the interface thread's read of it stays a borrow) and leaves
/// the buffers handed back for a frame that needs them; any other is decoded
/// ahead, an uncompressed frame in the other byte order included, into those
/// buffers and to what the inline decode gives. (Driven directly rather than
/// through a [`Prefetcher`], whose thread shows no sign of a request it
/// touched rather than skipped.)
#[test]
fn the_read_ahead_decodes_ahead_only_what_the_file_does_not_hold() {
    let free = FreeList::default();
    let mut scratch = ReadScratch::default();
    let mut ask = |tiff: &TiffStack, job: &ChannelJob| {
        let req = Request {
            jobs: vec![job.clone()],
            parallel: false,
            binning: None,
        };
        read_ahead(
            tiff,
            (REV, 0),
            &req,
            &CancelToken::never(),
            &mut scratch,
            &free,
        )
    };
    let handed_back = || free.lock().unwrap().len();
    let decoded_as_inline = |tiff: &TiffStack, job: &ChannelJob, got: PrefetchResult| {
        let want = decode_owned(tiff, std::slice::from_ref(job)).unwrap();
        assert!(matches!(&got.channels[..], [c] if same(&c.data, &want[0])));
    };

    give_back(&free, ChannelBufs::default());
    assert!(
        ask(&strips(Compression::None), &strips_job()).is_none(),
        "contiguous uncompressed strips were decoded ahead"
    );
    assert_eq!(handed_back(), 1, "a frame only touched took a frame's buffers");
    let deflated = strips(Compression::Deflate);
    let got = ask(&deflated, &strips_job()).expect("a compressed frame is decoded ahead");
    assert_eq!(handed_back(), 0, "not decoded into the buffers handed back");
    decoded_as_inline(&deflated, &strips_job(), got);

    let Some(orders) = byte_orders() else {
        eprintln!("skipping: the fast-tiff-lib fixtures are not here");
        return;
    };
    for (tiff, order, job) in orders {
        match ask(&tiff, &job) {
            None => assert_eq!(order, ByteOrder::host(), "{order:?} was not decoded ahead"),
            Some(got) => {
                assert_ne!(order, ByteOrder::host(), "{order:?} was decoded ahead");
                decoded_as_inline(&tiff, &job, got);
            }
        }
    }
}

/// Through the worker's thread, the answer that matters: an uncompressed
/// frame in the other byte order is published decoded, where the mode chosen
/// once from the compression tag only touched it and left the byte swap to
/// the interface thread.
#[cfg(feature = "threads")]
#[test]
fn an_uncompressed_frame_in_the_other_byte_order_is_published_ahead() {
    use std::time::{Duration, Instant};
    let Some(orders) = byte_orders() else {
        eprintln!("skipping: the fast-tiff-lib fixtures are not here");
        return;
    };
    let (tiff, _, job) = orders
        .into_iter()
        .find(|(_, order, _)| *order != ByteOrder::host())
        .expect("one of the two is in the other byte order");
    let job = ChannelJob { ifd_idx: 1, ..job };
    let want = decode_owned(&tiff, std::slice::from_ref(&job)).unwrap();
    let p = Prefetcher::new(Arc::new(tiff)).expect("the thread should spawn");
    p.request(REV, 1, vec![job], None);
    let start = Instant::now();
    let got = loop {
        if let Some(r) = p.take_matching(REV, 1) {
            break r;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "the frame was not read ahead"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert!(matches!(&got.channels[..], [c] if same(&c.data, &want[0])));
}

/// A planar frame's planes are strips of their own, so a channel switched off
/// is not decoded at all: with the green plane's strips made garbage, red and
/// blue still read, and as the clean file has them, while asking for green
/// too fails. (One pass over the span of planes asked for would decode green
/// into a spare, and fail the lot.)
#[test]
fn a_planar_plane_nobody_shows_is_not_decoded() {
    let (w, h) = (64u32, 64u32);
    let n = (w * h) as usize;
    for (st, kind) in [
        (SampleType::U8, ChannelKind::Int8),
        (SampleType::U16, ChannelKind::Int16),
    ] {
        let opts = WriterOptions::new(w, h, st)
            .samples_per_pixel(3)
            .planar(true)
            .compression(Compression::Lzw)
            .rows_per_strip(16);
        let mut wr = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
        if st == SampleType::U8 {
            let px: Vec<u8> = (0..3 * n).map(|i| (i * 7) as u8).collect();
            wr.write_frame_u8(&px).unwrap();
        } else {
            let px: Vec<u16> = (0..3 * n).map(|i| (i * 37) as u16).collect();
            wr.write_frame_u16(&px).unwrap();
        }
        let mut bytes = wr.finish().unwrap().into_inner();
        let clean = TiffStack::from_bytes(bytes.clone()).unwrap();
        let f = &clean.frames[0];
        let per_plane = f.strip_offsets.len() / 3;
        assert_eq!(per_plane, 4, "{st:?}: four strips a plane");
        for s in per_plane..2 * per_plane {
            let at = f.strip_offsets[s] as usize;
            bytes[at..at + f.strip_byte_counts[s] as usize].fill(0xFF);
        }
        let green_garbled = TiffStack::from_bytes(bytes).unwrap();
        let rgb = |channel: usize| ChannelJob {
            channel,
            ifd_idx: 0,
            plane: channel,
            kind,
            rgb: true,
            width: w,
            height: h,
        };
        let red_and_blue = [rgb(0), rgb(2)];
        let got = decode_owned(&green_garbled, &red_and_blue)
            .unwrap_or_else(|e| panic!("{st:?}: green, switched off, was decoded: {e}"));
        let want = decode_owned(&clean, &red_and_blue).unwrap();
        assert!(
            got.iter().zip(&want).all(|(a, b)| same(a, b)),
            "{st:?}: red and blue are not the clean file's"
        );
        let all = [rgb(0), rgb(1), rgb(2)];
        assert!(decode_owned(&clean, &all).is_ok(), "{st:?}");
        assert!(
            decode_owned(&green_garbled, &all).is_err(),
            "{st:?}: garbage in green should not decode"
        );
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
    let p = Prefetcher::new(tiff).expect("the thread should spawn");
    for frame in [3, 4, 5] {
        p.request(REV, frame, vec![job()], None);
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
fn two_channel_movie() -> Document {
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
    Document::from_bytes(bytes, "movie.tif".into(), false).unwrap()
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
    // Started as a sync starts it, on the first frame read ahead: a stack
    // runs no worker of its own accord.
    assert!(stack.services.prefetch.is_none());
    let p = stack
        .services
        .start_prefetch(&stack.tiff)
        .expect("a compressed stack reads ahead");
    p.request(before, 1, jobs, None);

    crate::dimensions::apply_dimension_override(&mut stack, 3, 1, 2);
    let after = stack.display.revision();
    assert_ne!(after.layout, before.layout, "an override moves the layout");
    let moved = build_jobs(&stack, 1, &[true; 3]);
    assert_eq!(
        moved[0].ifd_idx, 3,
        "channel 0 of frame 1 reads another IFD"
    );

    let p = stack.services.prefetch.as_ref().unwrap();
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

/// A frame read ahead with its histograms asked for comes back binned exactly
/// as a whole-frame decode bins it, and one read ahead without carries none.
#[cfg(feature = "threads")]
#[test]
fn the_read_ahead_bins_the_frame_it_decodes() {
    use crate::histogram::{frame_histograms, Binning};
    use std::time::{Duration, Instant};
    let mut stack = two_channel_movie();
    stack.services.start_prefetch(&stack.tiff);
    let revision = stack.display.revision();
    let take = |stack: &Document, frame: usize| {
        let p = stack
            .services
            .prefetch
            .as_ref()
            .expect("a compressed stack reads ahead");
        let start = Instant::now();
        loop {
            if let Some(r) = p.take_matching(revision, frame) {
                break r;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the read-ahead never answered"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };

    let jobs = build_jobs(&stack, 2, &[true, true]);
    let binning = Binning::of(&stack);
    let p = stack.services.prefetch.as_ref().unwrap();
    p.request(revision, 2, jobs, Some(binning));
    let got = take(&stack, 2);
    stack.set_frame(2);
    assert_eq!(got.histograms, frame_histograms(&stack));
    assert_eq!(got.histograms.len(), 2);

    let jobs = build_jobs(&stack, 1, &[true, true]);
    stack
        .services
        .prefetch
        .as_ref()
        .unwrap()
        .request(revision, 1, jobs, None);
    assert!(take(&stack, 1).histograms.is_empty());
}

/// A result handed back after it was shown is what the worker decodes the
/// next frame into: the same allocation, frame after frame. Before, every
/// frame read ahead was a fresh buffer, allocated and zero-filled.
#[cfg(feature = "threads")]
#[test]
fn the_read_ahead_decodes_into_the_buffers_handed_back() {
    use std::time::{Duration, Instant};
    let p = Prefetcher::new(Arc::new(stack(Compression::Deflate))).expect("spawn");
    let take = |frame: usize| {
        let start = Instant::now();
        loop {
            if let Some(r) = p.take_matching(REV, frame) {
                break r;
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "the read-ahead never answered"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let ptr = |r: &PrefetchResult| match &r.channels[..] {
        [c] => match &c.data {
            Decoded::U16(v) => v.as_ptr(),
            _ => panic!("16-bit samples should decode as U16"),
        },
        _ => panic!("one channel asked for"),
    };
    p.request(REV, 0, vec![job()], None);
    let first = take(0);
    let at = ptr(&first);
    p.recycle(first);
    for frame in 1..4 {
        p.request(REV, frame, vec![job()], None);
        let next = take(frame);
        assert_eq!(
            ptr(&next),
            at,
            "frame {frame} was decoded into a fresh buffer"
        );
        p.recycle(next);
    }
    // Not handed back, the next frame cannot be decoded into it.
    p.request(REV, 9, vec![job()], None);
    let kept = take(9);
    p.request(REV, 10, vec![job()], None);
    assert_ne!(ptr(&take(10)), ptr(&kept));
}

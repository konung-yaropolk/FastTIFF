//! Cancelling a window part-way through its bands.

use super::*;
use std::sync::atomic::Ordering;

/// A tall, narrow 8-bit frame: at four bytes a row, a band is the most rows
/// one may hold (4096), so the frame is eight bands of a few kilobytes each.
const W: u32 = 4;
const H: u32 = 8 * 4096;

fn tall_stack() -> (TiffStack, Vec<u8>) {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    let opts = WriterOptions::new(W, H, SampleType::U8).rows_per_strip(512);
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let px: Vec<u8> = (0..W * H).map(|i| (i % 251) as u8).collect();
    w.write_frame_bytes(&px).unwrap();
    let tiff = TiffStack::from_bytes(w.finish().unwrap().into_inner()).unwrap();
    (tiff, px)
}

fn job() -> ChannelJob {
    ChannelJob {
        channel: 0,
        ifd_idx: 0,
        plane: 0,
        kind: ChannelKind::Int8,
        rgb: false,
        width: W,
        height: H,
    }
}

/// Told to stop after its first band, a build asks before the second, stops
/// there with `Cancelled`, and decodes nothing more.
#[test]
fn a_cancelled_window_stops_at_the_next_band() {
    let (tiff, px) = tall_stack();
    let roi = Roi::full(W, H);
    let bands_in_frame = bandcache::bands_covering(0..H, bandcache::band_rows(W, 1)).len();
    assert_eq!(bands_in_frame, 8, "the test needs several bands");

    let mut bands = BandCache::default();
    let (token, asked) = CancelToken::after(1);
    match assemble(&tiff, &mut bands, 1, &[job()], &roi, 0, &token) {
        Err(e) => assert!(e.is::<Cancelled>(), "{e:#}"),
        Ok(_) => panic!("a cancelled window came back built"),
    }
    assert_eq!(
        asked.load(Ordering::Relaxed),
        2,
        "asked before each band: one decoded, stopped before the second"
    );

    // Never cancelled, the same window is the whole frame.
    let planes = assemble(
        &tiff,
        &mut bands,
        1,
        &[job()],
        &roi,
        0,
        &CancelToken::never(),
    )
    .expect("not cancelled");
    match &planes[0] {
        Decoded::U8(got) => assert!(got == &px, "the frame, sample for sample"),
        _ => panic!("8-bit samples should come back as U8"),
    }
}

//! The inline decode keeps the file's bytes when they already are the plane.

use super::*;
use fast_tiff_lib::{Compression, SampleType, TiffWriter, WriterOptions};

const W: u32 = 6;
const H: u32 = 4;

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

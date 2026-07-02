//! Encoder unit tests: everything stays in memory (Cursor), and pixel checks
//! round-trip through the *public decoder* — the strongest guarantee that what
//! the writer emits is exactly what the reader consumes.

use super::*;
use crate::decode::{read_frame_f32, read_frame_u16, read_frame_u8, read_plane_u8};
use crate::ifd::{self, ByteOrder};
use crate::index::{FrameInfo, SampleFormat};
use std::io::Cursor;

/// Write `frames` through a fully-configured writer into an in-memory buffer.
fn write_stack(options: WriterOptions, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), options).unwrap();
    for f in frames {
        w.write_frame_bytes(f).unwrap();
    }
    w.finish().unwrap().into_inner()
}

/// Minimal reader-side parse of one IFD into a `FrameInfo` (the same defaults
/// `index::frame_info_from_entries` uses), so unit tests can decode written
/// bytes without touching the filesystem.
fn parse_frames(bytes: &[u8]) -> (Vec<FrameInfo>, ByteOrder) {
    let (order, first) = ifd::read_header(bytes).unwrap();
    let mut frames = Vec::new();
    let mut offset = first as usize;
    while offset != 0 {
        let parsed = ifd::read_ifd(bytes, offset, order).unwrap();
        // The spec requires ascending tag order; verify on every IFD we parse.
        let tags: Vec<u16> = parsed.entries.iter().map(|e| e.tag).collect();
        let mut sorted = tags.clone();
        sorted.sort_unstable();
        assert_eq!(tags, sorted, "IFD entries must be in ascending tag order");

        let mut width = 0;
        let mut height = 0;
        let mut bits = 16u16;
        let mut spp = 1u16;
        let mut format = 1u16;
        let mut compression = 1u16;
        let mut predictor = 1u16;
        let mut photometric = 1u16;
        let mut rows_per_strip = u32::MAX;
        let mut strip_offsets = Vec::new();
        let mut strip_byte_counts = Vec::new();
        for e in &parsed.entries {
            match e.tag {
                256 => width = e.as_u32(bytes, order).unwrap(),
                257 => height = e.as_u32(bytes, order).unwrap(),
                258 => bits = e.as_u32(bytes, order).unwrap() as u16,
                259 => compression = e.as_u32(bytes, order).unwrap() as u16,
                262 => photometric = e.as_u32(bytes, order).unwrap() as u16,
                273 => strip_offsets = e.as_u32_array(bytes, order).unwrap(),
                277 => spp = e.as_u32(bytes, order).unwrap() as u16,
                278 => rows_per_strip = e.as_u32(bytes, order).unwrap(),
                279 => strip_byte_counts = e.as_u32_array(bytes, order).unwrap(),
                317 => predictor = e.as_u32(bytes, order).unwrap() as u16,
                339 => format = e.as_u32(bytes, order).unwrap() as u16,
                _ => {}
            }
        }
        if rows_per_strip == u32::MAX {
            rows_per_strip = height;
        }
        frames.push(FrameInfo {
            width,
            height,
            bits_per_sample: bits,
            samples_per_pixel: spp,
            sample_format: match format {
                2 => SampleFormat::SignedInt,
                3 => SampleFormat::Float,
                _ => SampleFormat::UnsignedInt,
            },
            compression: match compression {
                1 => Compression::None,
                5 => Compression::Lzw,
                32773 => Compression::PackBits,
                8 | 32946 => Compression::Deflate,
                other => Compression::Other(other),
            },
            predictor,
            photometric,
            planar_config: 1,
            strip_offsets: strip_offsets.into_iter().map(u64::from).collect(),
            strip_byte_counts: strip_byte_counts.into_iter().map(u64::from).collect(),
            rows_per_strip,
        });
        offset = parsed.next_offset as usize;
    }
    (frames, order)
}

fn le_bytes_u16(vals: &[u16]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn header_and_chain_structure() {
    let opts = WriterOptions::new(2, 2, SampleType::U8);
    let bytes = write_stack(opts, &[vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);

    assert_eq!(&bytes[0..2], b"II", "always little-endian");
    let (frames, order) = parse_frames(&bytes);
    assert_eq!(order, ByteOrder::Little);
    assert_eq!(frames.len(), 2);
    assert_eq!((frames[0].width, frames[0].height), (2, 2));
    assert_eq!(frames[0].bits_per_sample, 8);
    assert_eq!(frames[0].compression, Compression::None);
    // Uncompressed default: the whole frame as one strip.
    assert_eq!(frames[0].strip_offsets.len(), 1);
    assert_eq!(frames[1].strip_byte_counts, vec![4]);
}

#[test]
fn uncompressed_u16_is_the_readers_zero_copy_layout() {
    let pixels: Vec<u16> = (0..12).map(|v| v * 1000).collect();
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), WriterOptions::new(4, 3, SampleType::U16)).unwrap();
    w.write_frame_u16(&pixels).unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let (frames, order) = parse_frames(&bytes);
    let decoded = read_frame_u16(&bytes, &frames[0], order, None).unwrap();
    assert_eq!(decoded.as_ref(), &pixels[..]);
    // Single uncompressed native-order strip: the reader borrows, no decode.
    #[cfg(target_endian = "little")]
    assert!(
        matches!(decoded, std::borrow::Cow::Borrowed(_)),
        "uncompressed single-strip output must hit the reader's zero-copy path"
    );
}

#[test]
fn all_codecs_roundtrip_with_and_without_predictor() {
    // Gradient + repetition: exercises literal and run PackBits records and
    // gives the predictor something to shrink.
    let pixels: Vec<u16> = (0..6 * 5).map(|i| 500 + (i as u16 % 7) * 3).collect();
    let data = le_bytes_u16(&pixels);

    for compression in [Compression::None, Compression::Lzw, Compression::Deflate, Compression::PackBits] {
        for predictor in [false, true] {
            let opts = WriterOptions::new(6, 5, SampleType::U16)
                .compression(compression)
                .predictor(predictor)
                // 2 rows per strip over height 5: multi-strip with a short
                // last strip — the layout that once broke naive readers.
                .rows_per_strip(2);
            let bytes = write_stack(opts, &[data.clone()]);
            let (frames, order) = parse_frames(&bytes);
            assert_eq!(frames[0].strip_offsets.len(), 3, "{compression:?}");
            let decoded = read_frame_u16(&bytes, &frames[0], order, None).unwrap();
            assert_eq!(
                decoded.as_ref(),
                &pixels[..],
                "roundtrip failed for {compression:?}, predictor={predictor}"
            );
        }
    }
}

#[test]
fn packbits_handles_long_runs_and_literals() {
    // One row of 300 bytes: a >128-byte run (needs record splitting), then
    // varied literals, then a short run.
    let mut row = vec![7u8; 150];
    row.extend((0..140).map(|i| (i * 13 % 251) as u8));
    row.extend([9u8; 10]);
    let opts = WriterOptions::new(300, 1, SampleType::U8).compression(Compression::PackBits);
    let bytes = write_stack(opts, &[row.clone()]);
    let (frames, order) = parse_frames(&bytes);
    let decoded = read_frame_u8(&bytes, &frames[0], order).unwrap();
    assert_eq!(decoded.as_ref(), &row[..]);
}

#[test]
fn chunky_rgb8_reads_back_per_plane() {
    // 2 pixels: (10,20,30), (40,50,60) — chunky RGB.
    let opts = WriterOptions::new(2, 1, SampleType::U8).samples_per_pixel(3);
    let bytes = write_stack(opts, &[vec![10, 20, 30, 40, 50, 60]]);
    let (frames, order) = parse_frames(&bytes);
    assert!(frames[0].is_rgb(), "spp=3 chunky must be tagged photometric=RGB");
    assert_eq!(read_plane_u8(&bytes, &frames[0], order, 0).unwrap(), vec![10, 40]);
    assert_eq!(read_plane_u8(&bytes, &frames[0], order, 2).unwrap(), vec![30, 60]);
}

#[test]
fn f32_frames_roundtrip_bit_exact() {
    let pixels: Vec<f32> = vec![-0.5, 0.0, 1.25, 1e-7, 3.4e38, -2.5];
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), WriterOptions::new(3, 2, SampleType::F32)).unwrap();
    w.write_frame_f32(&pixels).unwrap();
    let bytes = w.finish().unwrap().into_inner();
    let (frames, order) = parse_frames(&bytes);
    assert_eq!(frames[0].sample_format, SampleFormat::Float);
    let decoded = read_frame_f32(&bytes, &frames[0], order).unwrap();
    assert_eq!(decoded.as_ref(), &pixels[..]);
}

#[test]
fn signed_i16_maps_into_display_space_like_imagej() {
    // Written via the raw-bytes escape hatch; the reader offsets signed data
    // into unsigned display space (-72 -> 32696), matching ImageJ.
    let values: [i16; 4] = [-72, 0, 878, -32768];
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let opts = WriterOptions::new(2, 2, SampleType::I16);
    let bytes = write_stack(opts, &[data]);
    let (frames, order) = parse_frames(&bytes);
    assert_eq!(frames[0].sample_format, SampleFormat::SignedInt);
    let decoded = read_frame_u16(&bytes, &frames[0], order, None).unwrap();
    assert_eq!(decoded.as_ref(), &[32696, 32768, 33646, 0]);
}

#[test]
fn imagej_description_lands_on_first_ifd_only() {
    let opts = WriterOptions::new(2, 1, SampleType::U16).imagej(
        ImageJOptions::new(2, 1)
            .mode(DisplayMode::Composite)
            .fps(12.5)
            .unit("um")
            .range(10.0, 200.0)
            .calibration(2.0, 0.5),
    );
    let frame = le_bytes_u16(&[1, 2]);
    let bytes = write_stack(opts, &[frame.clone(), frame.clone(), frame.clone(), frame]);

    let (order, first) = ifd::read_header(&bytes).unwrap();
    let ifd0 = ifd::read_ifd(&bytes, first as usize, order).unwrap();
    let desc_entry = ifd0.entries.iter().find(|e| e.tag == 270).expect("first IFD carries tag 270");
    let desc = desc_entry.as_ascii(&bytes, order).unwrap();
    for expected in [
        "ImageJ=", "images=4", "channels=2", "frames=2", "mode=composite", "unit=um",
        "fps=12.5", "min=10", "max=200", "cf=0", "c0=2", "c1=0.5",
    ] {
        assert!(desc.contains(expected), "description missing {expected:?}:\n{desc}");
    }
    let ifd1 = ifd::read_ifd(&bytes, ifd0.next_offset as usize, order).unwrap();
    assert!(ifd1.entries.iter().all(|e| e.tag != 270), "later IFDs must not repeat the description");
}

#[test]
fn predictor_covers_all_sample_widths_and_floats() {
    // 32-bit integers: Predictor 2 with 4-byte differencing (libtiff parity).
    let ints: [i32; 6] = [-1_000_000, -999_800, 0, 7, 2_000_000_000, 1_999_999_500];
    let data: Vec<u8> = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
    let opts = WriterOptions::new(3, 2, SampleType::I32)
        .compression(Compression::Deflate)
        .predictor(true);
    let bytes = write_stack(opts, &[data]);
    let (frames, order) = parse_frames(&bytes);
    assert_eq!(frames[0].predictor, 2, "integer data gets predictor 2");
    let decoded = read_frame_f32(&bytes, &frames[0], order).unwrap();
    let expected: Vec<f32> = ints.iter().map(|&v| v as f32).collect();
    assert_eq!(decoded.as_ref(), &expected[..]);

    // f32: Predictor 3 (TechNote 3 floating-point differencing), compressed
    // and uncompressed — uncompressed also proves the reader's zero-copy fast
    // path correctly steps aside for predictor-differenced data.
    let floats: Vec<f32> = (0..8).map(|i| -2.5 + i as f32 * 0.75).collect();
    let data: Vec<u8> = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
    for compression in [Compression::None, Compression::Deflate] {
        let opts = WriterOptions::new(4, 2, SampleType::F32)
            .compression(compression)
            .predictor(true);
        let bytes = write_stack(opts, &[data.clone()]);
        let (frames, order) = parse_frames(&bytes);
        assert_eq!(frames[0].predictor, 3, "float data gets predictor 3");
        let decoded = read_frame_f32(&bytes, &frames[0], order).unwrap();
        assert_eq!(decoded.as_ref(), &floats[..], "{compression:?}");
    }

    // Predictor without compression on u16: the read_frame_u16 zero-copy fast
    // path must also step aside and undo the differencing.
    let pixels: Vec<u16> = vec![100, 105, 90, 200, 1000, 999];
    let opts = WriterOptions::new(3, 2, SampleType::U16).predictor(true);
    let bytes = write_stack(opts, &[le_bytes_u16(&pixels)]);
    let (frames, order) = parse_frames(&bytes);
    let decoded = read_frame_u16(&bytes, &frames[0], order, None).unwrap();
    assert_eq!(decoded.as_ref(), &pixels[..]);

    // Chunky RGB16 + predictor + LZW: differencing must stride by
    // samples-per-pixel so each color plane differences independently.
    let rgb: Vec<u16> = vec![1000, 2000, 3000, 1010, 1990, 3020, 990, 2020, 2980];
    let opts = WriterOptions::new(3, 1, SampleType::U16)
        .samples_per_pixel(3)
        .compression(Compression::Lzw)
        .predictor(true);
    let bytes = write_stack(opts, &[le_bytes_u16(&rgb)]);
    let (frames, order) = parse_frames(&bytes);
    let red = crate::decode::read_plane_u16(&bytes, &frames[0], order, None, 0).unwrap();
    assert_eq!(red, vec![1000, 1010, 990]);
    let blue = crate::decode::read_plane_u16(&bytes, &frames[0], order, None, 2).unwrap();
    assert_eq!(blue, vec![3000, 3020, 2980]);
}

#[test]
fn unknown_predictor_errors_instead_of_garbage() {
    // A frame claiming predictor 9: the reader must refuse, not decode wrong.
    let pixels: Vec<u16> = vec![1, 2, 3, 4];
    let bytes = write_stack(WriterOptions::new(2, 2, SampleType::U16), &[le_bytes_u16(&pixels)]);
    let (mut frames, order) = parse_frames(&bytes);
    frames[0].predictor = 9;
    assert!(read_frame_u16(&bytes, &frames[0], order, None).is_err());
}

#[test]
fn description_carries_spacing_loop_and_extra_keys() {
    let opts = WriterOptions::new(1, 1, SampleType::U8).imagej(
        ImageJOptions::new(1, 2)
            .spacing(1.5)
            .loop_playback(true)
            .extra("vunit", "V")
            .extra("tunit", "s"),
    );
    let bytes = write_stack(opts, &[vec![0], vec![1]]);
    let (order, first) = ifd::read_header(&bytes).unwrap();
    let ifd0 = ifd::read_ifd(&bytes, first as usize, order).unwrap();
    let desc = ifd0.entries.iter().find(|e| e.tag == 270).unwrap().as_ascii(&bytes, order).unwrap();
    for expected in ["spacing=1.5", "loop=true", "vunit=V", "tunit=s", "slices=2"] {
        assert!(desc.contains(expected), "description missing {expected:?}:\n{desc}");
    }
}

#[test]
fn rejects_invalid_configurations_and_data() {
    // Wrong frame length.
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), WriterOptions::new(4, 4, SampleType::U16)).unwrap();
    assert!(w.write_frame_bytes(&[0u8; 3]).is_err());
    // Typed method on the wrong sample type.
    assert!(w.write_frame_u8(&[0u8; 32]).is_err());
    // Unknown compression code.
    assert!(TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(2, 2, SampleType::U8).compression(Compression::Other(7))
    )
    .is_err());
    // imagej + description both set.
    assert!(TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(2, 2, SampleType::U8)
            .imagej(ImageJOptions::new(1, 1))
            .description("x")
    )
    .is_err());
    // Zero frames.
    let w = TiffWriter::new(Cursor::new(Vec::new()), WriterOptions::new(2, 2, SampleType::U8)).unwrap();
    assert!(w.finish().is_err());
    // Plane count not divisible by channels x slices.
    let mut w = TiffWriter::new(
        Cursor::new(Vec::new()),
        WriterOptions::new(1, 1, SampleType::U8).imagej(ImageJOptions::new(2, 1)),
    )
    .unwrap();
    w.write_frame_u8(&[1]).unwrap();
    w.write_frame_u8(&[2]).unwrap();
    w.write_frame_u8(&[3]).unwrap();
    assert!(w.finish().is_err(), "3 planes into 2 channels must fail");
}

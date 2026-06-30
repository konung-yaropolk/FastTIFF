use super::*;

#[test]
fn packbits_literal_run() {
    // n=2 (literal run of 3 bytes) then n=-2 (repeat next byte 3 times)
    let input = [2u8, 0xAA, 0xBB, 0xCC, (-2i8) as u8, 0xFF];
    let out = packbits_decode(&input, 6);
    assert_eq!(out, vec![0xAA, 0xBB, 0xCC, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn packbits_noop_byte_is_skipped() {
    let input = [0u8, 0x42, (-128i8) as u8, 0u8, 0x99];
    let out = packbits_decode(&input, 3);
    // first record: literal run of 1 byte (0x42)
    // second record: -128 is a documented no-op, skipped entirely
    // third record: literal run of 1 byte (0x99)
    assert_eq!(out, vec![0x42, 0x99]);
}

fn make_frame(width: u32, height: u32, predictor: u16) -> FrameInfo {
    FrameInfo {
        width,
        height,
        bits_per_sample: 16,
        samples_per_pixel: 1,
        sample_format: crate::index::SampleFormat::UnsignedInt,
        compression: Compression::None,
        predictor,
        photometric: 1,
        planar_config: 1,
        strip_offsets: vec![0],
        strip_byte_counts: vec![(width * height * 2) as u64],
        rows_per_strip: height,
    }
}

#[test]
fn predictor_undo_le_roundtrip() {
    // Two rows of 4 pixels each. Original values:
    let original: [u16; 8] = [100, 105, 90, 200, 1000, 999, 999, 1005];
    let frame = make_frame(4, 2, 2);

    // Encode (horizontal differencing) to build the "compressed" input.
    let mut differenced = [0u16; 8];
    for row in 0..2 {
        let base = row * 4;
        differenced[base] = original[base];
        for i in 1..4 {
            differenced[base + i] = original[base + i].wrapping_sub(original[base + i - 1]);
        }
    }
    let mut bytes = Vec::new();
    for v in differenced {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let restored = undo_predictor(bytes, &frame, 2, ByteOrder::Little).unwrap();
    let mut out = [0u16; 8];
    for (i, chunk) in restored.chunks_exact(2).enumerate() {
        out[i] = ByteOrder::Little.u16(chunk);
    }
    assert_eq!(out, original);
}

#[test]
fn predictor_undo_be_roundtrip() {
    // Same as above but file byte order is big-endian — this is exactly
    // the bug that was caught and fixed: undo must read/write in the
    // file's declared order, not native order.
    let original: [u16; 4] = [5000, 5010, 4990, 5200];
    let frame = make_frame(4, 1, 2);

    let mut differenced = [0u16; 4];
    differenced[0] = original[0];
    for i in 1..4 {
        differenced[i] = original[i].wrapping_sub(original[i - 1]);
    }
    let mut bytes = Vec::new();
    for v in differenced {
        bytes.extend_from_slice(&v.to_be_bytes());
    }

    let restored = undo_predictor(bytes, &frame, 2, ByteOrder::Big).unwrap();
    let mut out = [0u16; 4];
    for (i, chunk) in restored.chunks_exact(2).enumerate() {
        out[i] = ByteOrder::Big.u16(chunk);
    }
    assert_eq!(out, original);
}

#[test]
fn minmax_f32_handles_normal_nan_and_constant_data() {
    assert_eq!(minmax_f32(&[1.5, -2.0, 3.25, f32::NAN]), (-2.0, 3.25));
    assert_eq!(minmax_f32(&[]), (0.0, 1.0));
    assert_eq!(minmax_f32(&[f32::NAN, f32::NAN]), (0.0, 1.0));
    assert_eq!(minmax_f32(&[4.0, 4.0, 4.0]), (0.0, 1.0)); // constant -- no usable span
}

fn float_frame(width: u32, height: u32) -> FrameInfo {
    FrameInfo {
        width,
        height,
        bits_per_sample: 32,
        samples_per_pixel: 1,
        sample_format: SampleFormat::Float,
        compression: Compression::None,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        strip_offsets: vec![0],
        strip_byte_counts: vec![(width * height * 4) as u64],
        rows_per_strip: height,
    }
}

#[test]
fn float_data_rescales_like_imagej_auto_contrast() {
    // Not 0..65535-shaped at all -- a realistic ratiometric range.
    let values: [f32; 4] = [-0.5, 0.0, 1.0, 2.5];
    let mut file = Vec::new();
    for v in values {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let frame = float_frame(2, 2);

    // float_range = None -> auto-ranges to this frame's own min/max,
    // same as ImageJ auto-contrasting a float image to its own data.
    let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, None).unwrap();
    assert_eq!(pixels[0], 0, "min value should map to 0");
    assert_eq!(pixels[3], 65535, "max value should map to 65535");
    // 0.0 is 1/6 of the way from -0.5 to 2.5.
    let expected_mid = (65535.0_f64 * (1.0 / 6.0)).round() as u16;
    assert!(
        pixels[1].abs_diff(expected_mid) <= 1,
        "got {}, expected close to {expected_mid}",
        pixels[1]
    );
}

#[test]
fn float_data_respects_an_explicit_range() {
    // A fixed display range, e.g. one established from the first frame
    // and reused for every subsequent frame in the same channel so
    // contrast doesn't jump around as you scrub.
    let values: [f32; 2] = [0.0, 10.0];
    let mut file = Vec::new();
    for v in values {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let frame = float_frame(2, 1);

    let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, Some((0.0, 100.0))).unwrap();
    assert_eq!(pixels[0], 0);
    // 10.0 out of a 0..100 range is exactly 10%.
    let expected = (65535.0_f64 * 0.10).round() as u16;
    assert!(pixels[1].abs_diff(expected) <= 1);
}

#[test]
fn rgb8_deinterleaves_into_color_planes() {
    // Two pixels, chunky RGB8: pixel0 = (10,20,30), pixel1 = (40,50,60).
    let mut frame = make_frame(2, 1, 1);
    frame.bits_per_sample = 8;
    frame.samples_per_pixel = 3;
    frame.photometric = 2;
    frame.strip_byte_counts = vec![6];
    let file: Vec<u8> = vec![10, 20, 30, 40, 50, 60];

    let up = |b: u8| ((b as u16) << 8) | b as u16;
    let red = read_plane_u16(&file, &frame, ByteOrder::Little, None, 0).unwrap();
    let green = read_plane_u16(&file, &frame, ByteOrder::Little, None, 1).unwrap();
    let blue = read_plane_u16(&file, &frame, ByteOrder::Little, None, 2).unwrap();
    assert_eq!(red, vec![up(10), up(40)]);
    assert_eq!(green, vec![up(20), up(50)]);
    assert_eq!(blue, vec![up(30), up(60)]);
}

#[test]
fn rgb8_plane_u8_keeps_raw_bytes() {
    // Same chunky RGB8 as `rgb8_deinterleaves_into_color_planes`, but the u8
    // plane reader returns the bytes un-widened — the R8Uint upload path.
    let mut frame = make_frame(2, 1, 1);
    frame.bits_per_sample = 8;
    frame.samples_per_pixel = 3;
    frame.photometric = 2;
    frame.strip_byte_counts = vec![6];
    let file: Vec<u8> = vec![10, 20, 30, 40, 50, 60];

    let red = read_plane_u8(&file, &frame, ByteOrder::Little, 0).unwrap();
    let green = read_plane_u8(&file, &frame, ByteOrder::Little, 1).unwrap();
    let blue = read_plane_u8(&file, &frame, ByteOrder::Little, 2).unwrap();
    assert_eq!(red, vec![10, 40]);
    assert_eq!(green, vec![20, 50]);
    assert_eq!(blue, vec![30, 60]);
}

#[test]
fn unsigned_int32_rescales_into_texture_range() {
    // 32-bit unsigned integers must be rescaled (not reinterpreted as
    // float). With an explicit range the mapping is linear into 0..65535.
    let mut frame = make_frame(2, 1, 1);
    frame.bits_per_sample = 32;
    frame.sample_format = SampleFormat::UnsignedInt;
    frame.strip_byte_counts = vec![8];
    let values: [u32; 2] = [0, 1000];
    let mut file = Vec::new();
    for v in values {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let pixels = read_plane_u16(&file, &frame, ByteOrder::Little, Some((0.0, 1000.0)), 0).unwrap();
    assert_eq!(pixels[0], 0);
    assert_eq!(pixels[1], 65535);
}

#[test]
fn signed_int16_is_offset_into_unsigned_display_space() {
    // ImageJ stores signed-16 images and windows them in unsigned+32768
    // space. A signed -72 must decode to 32696, signed 0 to 32768, etc. —
    // so a signed file lines up with the equivalent unsigned+calibration
    // file (and the same min=/max= window applies to both).
    let mut frame = make_frame(2, 2, 1);
    frame.sample_format = SampleFormat::SignedInt;
    let values: [i16; 4] = [-72, 0, 878, -32768];
    let mut file = Vec::new();
    for v in values {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let pixels = read_frame_u16(&file, &frame, ByteOrder::Little, None).unwrap();
    assert_eq!(&*pixels, &[32696u16, 32768, 33646, 0]);
}

#[test]
fn frame_float_minmax_matches_actual_data() {
    let values: [f32; 3] = [-3.0, 0.5, 7.0];
    let mut file = Vec::new();
    for v in values {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let frame = float_frame(3, 1);
    let range = frame_float_minmax(&file, &frame, ByteOrder::Little).unwrap();
    assert_eq!(range, Some((-3.0, 7.0)));
}

#[test]
fn frame_float_minmax_is_none_for_integer_frames() {
    let frame = make_frame(2, 2, 1);
    let file = vec![0u8; 8];
    assert_eq!(frame_float_minmax(&file, &frame, ByteOrder::Little).unwrap(), None);
}

/// The actual bug: two strips, each independently LZW-compressed (a
/// fresh `Encoder` per strip, exactly like a real multi-strip TIFF
/// writer would produce). Concatenating the *compressed* bytes and
/// decompressing once — the old behavior — silently stops at roughly
/// strip 1's data. This proves the per-strip fix actually decodes both.
#[test]
fn multi_strip_lzw_decodes_past_the_first_strip() {
    let width = 4usize;
    let height = 4usize; // two 2-row strips
    let rows_per_strip = 2usize;

    let top: Vec<u16> = (0..(width * rows_per_strip) as u16).collect(); // 0..8
    let bottom: Vec<u16> = (100..100 + (width * rows_per_strip) as u16).collect(); // 100..108

    let encode_strip = |pixels: &[u16]| -> Vec<u8> {
        let mut raw_bytes = Vec::new();
        for &v in pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        weezl::encode::Encoder::new(weezl::BitOrder::Msb, 8)
            .encode(&raw_bytes)
            .expect("LZW encode failed in test setup")
    };

    let top_compressed = encode_strip(&top);
    let bottom_compressed = encode_strip(&bottom);

    // Lay the two compressed strips out in a fake "file" buffer.
    let mut fake_file = Vec::new();
    fake_file.extend_from_slice(&top_compressed);
    let bottom_offset = fake_file.len() as u64;
    fake_file.extend_from_slice(&bottom_compressed);

    let frame = FrameInfo {
        width: width as u32,
        height: height as u32,
        bits_per_sample: 16,
        samples_per_pixel: 1,
        sample_format: SampleFormat::UnsignedInt,
        compression: Compression::Lzw,
        predictor: 1,
        photometric: 1,
        planar_config: 1,
        strip_offsets: vec![0, bottom_offset],
        strip_byte_counts: vec![top_compressed.len() as u64, bottom_compressed.len() as u64],
        rows_per_strip: rows_per_strip as u32,
    };

    let pixels = read_frame_u16(&fake_file, &frame, ByteOrder::Little, None).expect("decode failed");
    let mut expected = top.clone();
    expected.extend_from_slice(&bottom);
    assert_eq!(&*pixels, expected.as_slice(), "bottom strip's pixels are missing or wrong");
}

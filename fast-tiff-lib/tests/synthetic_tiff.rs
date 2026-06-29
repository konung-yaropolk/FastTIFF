//! Builds a tiny, real, byte-for-byte valid multi-IFD TIFF in memory —
//! 3 frames, ImageDescription metadata, and a synthetic IJMetadata blob —
//! then runs it through `TiffStack::open` and checks every value against
//! what we put in. This is the test that actually exercises offset
//! arithmetic, not just type-checking.

use std::io::Write;
use fast_tiff_lib::TiffStack;

/// Minimal stand-in for `tempfile::tempdir()` — avoids pulling in a
/// dev-dependency whose transitive deps require a newer toolchain than
/// what's available in some environments. Each call gets a unique path
/// under the OS temp dir; the file is left behind (harmless for tests).
fn unique_temp_path(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("FastTIFF_test_{nanos}_{name}"))
}


const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_IMAGE_DESCRIPTION: u16 = 270;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIG: u16 = 284;
const TAG_IJ_METADATA_BYTE_COUNTS: u16 = 50838;
const TAG_IJ_METADATA: u16 = 50839;

type IfdEntrySpec = (u16, u16, u32, [u8; 4]); // tag, type, count, inline-or-offset value

fn short_val(v: u16) -> [u8; 4] {
    let mut b = [0u8; 4];
    b[0..2].copy_from_slice(&v.to_le_bytes());
    b
}
fn long_val(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

/// Builds: header, N raw pixel planes, then (description, ij_counts,
/// ij_metadata) "extra" blocks, then N IFDs chained together. Frame 0 gets
/// the ImageDescription + IJMetadata tags; frames 1..N don't (matching how
/// ImageJ writes hyperstacks — that metadata lives only on the first IFD).
fn build_synthetic_tiff(
    width: u32,
    height: u32,
    frame_pixels: &[Vec<u16>],
    description: &str,
    ij_metadata: Option<&[u8]>,
    ij_byte_counts: Option<&[u32]>,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // first-IFD offset patched later

    // --- pixel planes ---
    let mut strip_offsets = Vec::new();
    for pixels in frame_pixels {
        strip_offsets.push(buf.len() as u32);
        for &p in pixels {
            buf.extend_from_slice(&p.to_le_bytes());
        }
    }

    // --- extra data (description / IJ metadata) ---
    let desc_offset = buf.len() as u32;
    buf.extend_from_slice(description.as_bytes());
    buf.push(0); // null terminator
    let desc_len = description.len() as u32 + 1;

    let mut ij_counts_offset = 0u32;
    let mut ij_counts_len = 0u32;
    if let Some(counts) = ij_byte_counts {
        ij_counts_offset = buf.len() as u32;
        for c in counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        ij_counts_len = counts.len() as u32;
    }

    let mut ij_meta_offset = 0u32;
    let mut ij_meta_len = 0u32;
    if let Some(data) = ij_metadata {
        ij_meta_offset = buf.len() as u32;
        buf.write_all(data).unwrap();
        ij_meta_len = data.len() as u32;
    }

    // --- IFDs ---
    let n = frame_pixels.len();
    let mut ifd_sizes = Vec::with_capacity(n);
    for i in 0..n {
        let mut entries = 9; // base tags
        if i == 0 {
            entries += 1; // ImageDescription
            if ij_metadata.is_some() {
                entries += 2; // IJMetadataByteCounts + IJMetadata
            }
        }
        ifd_sizes.push(2 + entries * 12 + 4);
    }
    let ifd_start: Vec<u32> = {
        let mut starts = Vec::with_capacity(n);
        let mut cursor = buf.len() as u32;
        for &sz in &ifd_sizes {
            starts.push(cursor);
            cursor += sz as u32;
        }
        starts
    };

    // patch first-IFD offset in header
    buf[4..8].copy_from_slice(&ifd_start[0].to_le_bytes());

    for i in 0..n {
        let mut entries: Vec<IfdEntrySpec> = vec![
            (TAG_IMAGE_WIDTH, 4, 1, long_val(width)),
            (TAG_IMAGE_LENGTH, 4, 1, long_val(height)),
            (TAG_BITS_PER_SAMPLE, 3, 1, short_val(16)),
            (TAG_COMPRESSION, 3, 1, short_val(1)),
            (TAG_PHOTOMETRIC, 3, 1, short_val(1)),
            (TAG_STRIP_OFFSETS, 4, 1, long_val(strip_offsets[i])),
            (TAG_SAMPLES_PER_PIXEL, 3, 1, short_val(1)),
            (TAG_ROWS_PER_STRIP, 4, 1, long_val(height)),
            (
                TAG_STRIP_BYTE_COUNTS,
                4,
                1,
                long_val(width * height * 2),
            ),
        ];
        if i == 0 {
            entries.push((TAG_IMAGE_DESCRIPTION, 2, desc_len, long_val(desc_offset)));
            if ij_metadata.is_some() {
                entries.push((
                    TAG_IJ_METADATA_BYTE_COUNTS,
                    4,
                    ij_counts_len,
                    long_val(ij_counts_offset),
                ));
                entries.push((TAG_IJ_METADATA, 1, ij_meta_len, long_val(ij_meta_offset)));
            }
        }
        // TIFF spec requires entries sorted by ascending tag.
        entries.sort_by_key(|e| e.0);

        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, ftype, count, val) in &entries {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&ftype.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(val);
        }
        let next = if i + 1 < n { ifd_start[i + 1] } else { 0 };
        buf.extend_from_slice(&next.to_le_bytes());
    }

    buf
}

/// Build a synthetic IJMetadata blob matching our best-effort assumed
/// format: an 8-byte-record big-endian directory, then a `rang` block
/// (2 channels x (min,max) f64 pairs), then 2 `luts` blocks (768 bytes each).
fn build_ij_metadata_blob(ranges: &[(f64, f64)], luts: &[[[u8; 3]; 256]]) -> (Vec<u8>, Vec<u32>) {
    let mut header = Vec::new();
    let mut byte_counts = Vec::new();

    let mut body = Vec::new();

    // directory: one "rang" record covering all channels, one "luts" record per channel
    header.extend_from_slice(b"rang");
    header.extend_from_slice(&1u32.to_be_bytes());
    header.extend_from_slice(b"luts");
    header.extend_from_slice(&(luts.len() as u32).to_be_bytes());

    byte_counts.push(header.len() as u32); // byte_counts[0] = header size

    // rang block
    let mut rang_block = Vec::new();
    for &(lo, hi) in ranges {
        rang_block.extend_from_slice(&lo.to_be_bytes());
        rang_block.extend_from_slice(&hi.to_be_bytes());
    }
    byte_counts.push(rang_block.len() as u32);
    body.extend_from_slice(&rang_block);

    // luts blocks (planar R,G,B x256)
    for lut in luts {
        let mut block = vec![0u8; 768];
        for i in 0..256 {
            block[i] = lut[i][0];
            block[256 + i] = lut[i][1];
            block[512 + i] = lut[i][2];
        }
        byte_counts.push(block.len() as u32);
        body.extend_from_slice(&block);
    }

    let mut full = header;
    full.extend_from_slice(&body);
    (full, byte_counts)
}

#[test]
fn opens_multi_frame_stack_and_reads_correct_pixels() {
    let width = 4;
    let height = 3;
    let frame0: Vec<u16> = (0..12).collect();
    let frame1: Vec<u16> = (100..112).collect();
    let frame2: Vec<u16> = (1000..1012).collect();

    let bytes = build_synthetic_tiff(
        width,
        height,
        &[frame0.clone(), frame1.clone(), frame2.clone()],
        "ImageJ=1.54f\nimages=3\nchannels=1\nslices=1\nframes=3\nmode=grayscale\nmin=0.0\nmax=4095.0\nunit=micron\nfinterval=0.25\n",
        None,
        None,
    );

    let path = unique_temp_path("synthetic.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).expect("should parse synthetic TIFF");

    assert_eq!(stack.frames.len(), 3, "should walk all 3 IFDs in the chain");
    assert_eq!(stack.meta.channels, 1);
    assert_eq!(stack.meta.slices, 1);
    assert_eq!(stack.meta.frames, 3);
    assert_eq!(stack.meta.unit.as_deref(), Some("micron"));
    assert_eq!(stack.meta.frame_interval_s, Some(0.25));
    assert_eq!(stack.meta.channel_display[0].range, Some((0.0, 4095.0)));

    for (i, expected) in [&frame0, &frame1, &frame2].into_iter().enumerate() {
        let frame = &stack.frames[i];
        assert_eq!(frame.width, width);
        assert_eq!(frame.height, height);
        let pixels = fast_tiff_lib::read_frame_u16(&stack.mmap, frame, stack.byte_order, None).unwrap();
        assert_eq!(&*pixels, expected.as_slice(), "frame {i} pixel mismatch");
    }
}

#[test]
fn infers_frame_count_when_dimension_tags_absent() {
    // A bare N-page TIFF with no ImageJ ImageDescription at all should
    // still scrub correctly — frame count falls back to the IFD count.
    let width = 2;
    let height = 2;
    let frames: Vec<Vec<u16>> = (0..5).map(|f| vec![f as u16; 4]).collect();
    let bytes = build_synthetic_tiff(width, height, &frames, "", None, None);

    let path = unique_temp_path("plain.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 5);
    assert_eq!(stack.meta.frames, 5);
    assert_eq!(stack.meta.channels, 1);
}

#[test]
fn ignores_non_imagej_description_text() {
    // Some other software's free-form ImageDescription that happens to
    // contain text shaped like "key=value" — including a line that would,
    // if taken at face value, falsely claim 999 channels. Since it lacks
    // ImageJ's own "ImageJ=" signature, none of it should be trusted; the
    // frame count must still fall back to the real IFD count, exactly as
    // if there were no description at all.
    let width = 2;
    let height = 2;
    let frames: Vec<Vec<u16>> = (0..6).map(|f| vec![f as u16; 4]).collect();
    let description = "Acquired with SomeOtherScope v3.2\nchannels=999\nmin=12345\nexposure=100ms";
    let bytes = build_synthetic_tiff(width, height, &frames, description, None, None);

    let path = unique_temp_path("non_imagej.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.frames.len(), 6);
    assert_eq!(stack.meta.channels, 1, "should not have picked up the fake channels=999");
    assert_eq!(stack.meta.frames, 6, "should fall back to the real IFD count");
    assert_eq!(
        stack.meta.channel_display[0].range, None,
        "should not have picked up the fake min=12345 with no matching max="
    );
}

#[test]
fn supplements_missing_range_and_luts_from_ij_metadata() {
    // The IJMetadata block (tags 50838/50839) is a supplementary fallback: it
    // fills in display info ImageDescription didn't provide. Here the
    // description declares 2 composite channels but no min=/max= and (as
    // always) no explicit LUTs, so the per-channel ranges and the custom white
    // LUTs from the binary block should both be picked up.
    let width = 2;
    let height = 2;
    let frame0 = vec![10u16, 20, 30, 40];

    // Distinctive constant-white LUTs so they can't coincide with any of the
    // default composite channel colors (channel 0's default is a red ramp).
    let white_lut = [[255u8; 3]; 256];
    let (ij_bytes, ij_counts) = build_ij_metadata_blob(
        &[(50.0, 500.0), (60.0, 600.0)],
        &[white_lut, white_lut],
    );

    let bytes = build_synthetic_tiff(
        width,
        height,
        &[frame0.clone(), frame0.clone()],
        "ImageJ=1.54f\nimages=2\nchannels=2\nslices=1\nframes=1\nmode=composite\n",
        Some(&ij_bytes),
        Some(&ij_counts),
    );

    let path = unique_temp_path("composite.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    // ImageDescription had no min=/max=, so the binary ranges fill in.
    assert_eq!(stack.meta.channel_display[0].range, Some((50.0, 500.0)));
    assert_eq!(stack.meta.channel_display[1].range, Some((60.0, 600.0)));
    // ImageDescription carries no LUTs, so the custom white LUTs are applied.
    assert_eq!(stack.meta.channel_display[0].lut[128], [255, 255, 255]);
    assert_eq!(stack.meta.channel_display[1].lut[200], [255, 255, 255]);
}

#[test]
fn ij_metadata_does_not_override_an_explicit_range() {
    // When ImageDescription *does* specify min=/max=, the binary block must not
    // override it — IJMetadata only fills genuinely-missing values.
    let width = 2;
    let height = 2;
    let frame0 = vec![10u16, 20, 30, 40];

    let white_lut = [[255u8; 3]; 256];
    let (ij_bytes, ij_counts) =
        build_ij_metadata_blob(&[(50.0, 500.0), (60.0, 600.0)], &[white_lut, white_lut]);

    let bytes = build_synthetic_tiff(
        width,
        height,
        &[frame0.clone(), frame0.clone()],
        "ImageJ=1.54f\nimages=2\nchannels=2\nslices=1\nframes=1\nmode=composite\nmin=100.0\nmax=200.0\n",
        Some(&ij_bytes),
        Some(&ij_counts),
    );

    let path = unique_temp_path("composite_explicit_range.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    // The explicit 270 window wins over the block's 50..500 / 60..600.
    assert_eq!(stack.meta.channel_display[0].range, Some((100.0, 200.0)));
    assert_eq!(stack.meta.channel_display[1].range, Some((100.0, 200.0)));
}

#[test]
fn rejects_stale_lut_block_with_mismatched_channel_count() {
    // The actual reported bug: a genuinely single-channel grayscale file
    // (channels=1 in ImageDescription) whose binary IJMetadata block still
    // contains LUT/range entries for 2 channels — e.g. left over from
    // before the file was reduced from a 2-channel acquisition down to 1.
    // Applying the first of those stale entries to channel 0 would silently
    // replace its correct grayscale LUT with red. The count mismatch
    // (2 LUTs found, but only 1 channel) must be detected and the whole
    // binary block ignored, falling back to the default grayscale LUT.
    let width = 2;
    let height = 2;
    let frame0 = vec![10u16, 20, 30, 40];

    let mut stale_red_lut = [[0u8; 3]; 256];
    let mut stale_green_lut = [[0u8; 3]; 256];
    for i in 0..256 {
        stale_red_lut[i] = [i as u8, 0, 0];
        stale_green_lut[i] = [0, i as u8, 0];
    }
    let (ij_bytes, ij_counts) =
        build_ij_metadata_blob(&[(50.0, 500.0), (60.0, 600.0)], &[stale_red_lut, stale_green_lut]);

    let bytes = build_synthetic_tiff(
        width,
        height,
        &[frame0.clone()],
        "ImageJ=1.54f\nimages=1\nchannels=1\nslices=1\nframes=1\nmode=grayscale\n",
        Some(&ij_bytes),
        Some(&ij_counts),
    );

    let path = unique_temp_path("stale_lut.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).unwrap();
    assert_eq!(stack.meta.channels, 1);
    // Default grayscale LUT: identity ramp, not the stale red.
    assert_eq!(stack.meta.channel_display[0].lut[128], [128, 128, 128]);
    assert_eq!(stack.meta.channel_display[0].range, None);
}

/// Builds a minimal single-IFD chunky RGB8 TIFF (photometric=2, spp=3).
fn build_rgb8_tiff(width: u32, height: u32, pixels: &[(u8, u8, u8)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // first-IFD offset, patched below

    let strip_offset = buf.len() as u32;
    for &(r, g, b) in pixels {
        buf.extend_from_slice(&[r, g, b]);
    }
    let strip_len = (pixels.len() * 3) as u32;

    // BitsPerSample is a 3-element array (8,8,8) — too big for the 4-byte
    // inline value field, so it lives at its own offset.
    let bits_offset = buf.len() as u32;
    for _ in 0..3 {
        buf.extend_from_slice(&8u16.to_le_bytes());
    }

    let ifd_offset = buf.len() as u32;
    buf[4..8].copy_from_slice(&ifd_offset.to_le_bytes());

    let mut entries: Vec<IfdEntrySpec> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, long_val(width)),
        (TAG_IMAGE_LENGTH, 4, 1, long_val(height)),
        (TAG_BITS_PER_SAMPLE, 3, 3, long_val(bits_offset)),
        (TAG_COMPRESSION, 3, 1, short_val(1)),
        (TAG_PHOTOMETRIC, 3, 1, short_val(2)), // RGB
        (TAG_STRIP_OFFSETS, 4, 1, long_val(strip_offset)),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, short_val(3)),
        (TAG_ROWS_PER_STRIP, 4, 1, long_val(height)),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, long_val(strip_len)),
        (TAG_PLANAR_CONFIG, 3, 1, short_val(1)), // chunky
    ];
    entries.sort_by_key(|e| e.0);

    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ftype, count, val) in &entries {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&ftype.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(val);
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    buf
}

#[test]
fn opens_rgb8_tiff_and_deinterleaves_planes() {
    let pixels = [(10u8, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)];
    let bytes = build_rgb8_tiff(2, 2, &pixels);
    let path = unique_temp_path("rgb8.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).expect("should parse RGB8 TIFF");
    let frame = &stack.frames[0];
    assert_eq!(frame.samples_per_pixel, 3);
    assert_eq!(frame.photometric, 2);
    assert_eq!(frame.bits_per_sample, 8);
    assert!(frame.is_rgb(), "frame should be detected as chunky RGB");

    let up = |b: u8| ((b as u16) << 8) | b as u16;
    let red = fast_tiff_lib::read_plane_u16(&stack.mmap, frame, stack.byte_order, None, 0).unwrap();
    let blue = fast_tiff_lib::read_plane_u16(&stack.mmap, frame, stack.byte_order, None, 2).unwrap();
    assert_eq!(red, vec![up(10), up(40), up(70), up(100)]);
    assert_eq!(blue, vec![up(30), up(60), up(90), up(120)]);
}

/// Builds a chained multi-IFD 16-bit grayscale TIFF with one frame per `(w, h)`
/// in `dims` — used to exercise the uniform-size validation with deliberately
/// mismatched frames (e.g. a pyramidal stack). Pixel data is just zeros.
fn build_multi_size_tiff(dims: &[(u32, u32)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // first-IFD offset, patched below

    let mut strip_offsets = Vec::new();
    for &(w, h) in dims {
        strip_offsets.push(buf.len() as u32);
        buf.extend(std::iter::repeat(0u8).take((w * h * 2) as usize));
    }

    let n = dims.len();
    let ifd_size = 2 + 9 * 12 + 4; // entry count + 9 entries + next-offset
    let ifd_start: Vec<u32> = {
        let mut starts = Vec::new();
        let mut cursor = buf.len() as u32;
        for _ in 0..n {
            starts.push(cursor);
            cursor += ifd_size as u32;
        }
        starts
    };
    buf[4..8].copy_from_slice(&ifd_start[0].to_le_bytes());

    for i in 0..n {
        let (w, h) = dims[i];
        let mut entries: Vec<IfdEntrySpec> = vec![
            (TAG_IMAGE_WIDTH, 4, 1, long_val(w)),
            (TAG_IMAGE_LENGTH, 4, 1, long_val(h)),
            (TAG_BITS_PER_SAMPLE, 3, 1, short_val(16)),
            (TAG_COMPRESSION, 3, 1, short_val(1)),
            (TAG_PHOTOMETRIC, 3, 1, short_val(1)),
            (TAG_STRIP_OFFSETS, 4, 1, long_val(strip_offsets[i])),
            (TAG_SAMPLES_PER_PIXEL, 3, 1, short_val(1)),
            (TAG_ROWS_PER_STRIP, 4, 1, long_val(h)),
            (TAG_STRIP_BYTE_COUNTS, 4, 1, long_val(w * h * 2)),
        ];
        entries.sort_by_key(|e| e.0);
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, ftype, count, val) in &entries {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&ftype.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(val);
        }
        let next = if i + 1 < n { ifd_start[i + 1] } else { 0 };
        buf.extend_from_slice(&next.to_le_bytes());
    }
    buf
}

#[test]
fn rejects_non_uniform_frame_sizes() {
    // Frame 0 is 2x2, frame 1 is 3x3 — a pyramidal / mixed-size stack that
    // can't go into one fixed-size texture. Must fail to open with a clear
    // error rather than silently mis-rendering the odd frame.
    let bytes = build_multi_size_tiff(&[(2, 2), (3, 3)]);
    let path = unique_temp_path("mismatched.tif");
    std::fs::write(&path, &bytes).unwrap();

    match TiffStack::open(&path) {
        Ok(_) => panic!("mismatched frame sizes should fail to open"),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains("not uniform"), "unexpected error: {msg}");
        }
    }
}

#[test]
fn accepts_uniform_frame_sizes() {
    // The same builder with consistent dimensions opens fine (guards against
    // the uniform-size check being too strict).
    let bytes = build_multi_size_tiff(&[(4, 3), (4, 3), (4, 3)]);
    let path = unique_temp_path("uniform.tif");
    std::fs::write(&path, &bytes).unwrap();

    let stack = TiffStack::open(&path).expect("uniform frames should open");
    assert_eq!(stack.frames.len(), 3);
}
//! Robustness against malformed / hostile TIFFs.
//!
//! `TiffStack::from_bytes` and the `read_*` decoders take completely untrusted
//! bytes, so for *any* input the only acceptable outcomes are `Ok` or `Err`.
//! Three things are specifically not acceptable:
//!
//! - a **panic** (unwinds through a library boundary; a DoS for any embedder),
//! - an **abort** — what an oversized `vec![0u8; n]` produces, because Rust's
//!   allocation-failure path calls `abort()` and is *not* catchable, so it kills
//!   the whole process (this test binary included, very visibly),
//! - **unbounded memory** — a few hundred bytes must not be able to demand
//!   gigabytes.
//!
//! This is the always-on counterpart to the `cargo fuzz` targets in `fuzz/`:
//! those explore, this one pins the cases we already know about so a regression
//! is caught in ordinary CI on every platform, with no nightly toolchain.

use fast_tiff_lib::TiffStack;

// ---- a minimal valid TIFF to mutate ----

/// Little-endian classic TIFF: `w`x`h`, `bits`-deep, `spp` samples/px,
/// `compression`, with `pixel_bytes` as its single strip.
fn build_tiff(w: u32, h: u32, bits: u16, spp: u16, compression: u16, pixel_bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // first-IFD offset, patched below

    let strip_off = buf.len() as u32;
    buf.extend_from_slice(pixel_bytes);
    let strip_len = pixel_bytes.len() as u32;

    let ifd_off = buf.len() as u32;
    buf[4..8].copy_from_slice(&ifd_off.to_le_bytes());

    // (tag, type, count, value) — SHORTs are stored inline in the low 2 bytes.
    let entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, w),              // ImageWidth
        (257, 4, 1, h),              // ImageLength
        (258, 3, 1, bits as u32),    // BitsPerSample
        (259, 3, 1, compression as u32),
        (262, 3, 1, 1),              // Photometric = BlackIsZero
        (273, 4, 1, strip_off),      // StripOffsets
        (277, 3, 1, spp as u32),     // SamplesPerPixel
        (278, 4, 1, h),              // RowsPerStrip
        (279, 4, 1, strip_len),      // StripByteCounts
    ];
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in &entries {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&ty.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        if *ty == 3 {
            buf.extend_from_slice(&(*val as u16).to_le_bytes());
            buf.extend_from_slice(&[0, 0]);
        } else {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    buf
}

/// A well-formed 4x4 16-bit grayscale TIFF — the seed every mutation starts from.
fn valid_tiff() -> Vec<u8> {
    let pixels: Vec<u8> = (0..16u16).flat_map(|v| (v * 4000).to_le_bytes()).collect();
    build_tiff(4, 4, 16, 1, 1, &pixels)
}

/// A valid 4x4 16-bit TIFF carrying `desc` as its ImageDescription (tag 270) —
/// the vector for metadata-driven attacks, since that tag is free-form text the
/// ImageJ and OME parsers read numbers out of.
fn build_tiff_with_description(desc: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());

    let strip_off = buf.len() as u32;
    let pixels: Vec<u8> = (0..16u16).flat_map(|v| (v * 4000).to_le_bytes()).collect();
    buf.extend_from_slice(&pixels);
    let strip_len = pixels.len() as u32;

    let desc_off = buf.len() as u32;
    buf.extend_from_slice(desc.as_bytes());
    buf.push(0); // ASCII fields are NUL-terminated
    let desc_len = desc.len() as u32 + 1;

    let ifd_off = buf.len() as u32;
    buf[4..8].copy_from_slice(&ifd_off.to_le_bytes());

    let entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, 4),
        (257, 4, 1, 4),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (270, 2, desc_len, desc_off), // ImageDescription
        (273, 4, 1, strip_off),
        (277, 3, 1, 1),
        (278, 4, 1, 4),
        (279, 4, 1, strip_len),
    ];
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in &entries {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&ty.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        if *ty == 3 {
            buf.extend_from_slice(&(*val as u16).to_le_bytes());
            buf.extend_from_slice(&[0, 0]);
        } else {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf
}

/// Push `bytes` through the whole public surface. Returns without asserting —
/// every call is expected to return `Result`; the *test* is that we get here at
/// all (no panic, no abort).
fn exercise(bytes: &[u8]) {
    let Ok(stack) = TiffStack::from_bytes(bytes.to_vec()) else {
        return;
    };
    let Some(frame) = stack.frames.first() else {
        return;
    };
    let (data, order) = (&stack.data, stack.byte_order);
    let _ = fast_tiff_lib::read_frame_u16(data, frame, order, None);
    let _ = fast_tiff_lib::read_frame_u8(data, frame, order);
    let _ = fast_tiff_lib::read_frame_f32(data, frame, order);
    let _ = fast_tiff_lib::frame_float_minmax(data, frame, order);
    for plane in [0usize, 1, 5] {
        let _ = fast_tiff_lib::read_plane_u16(data, frame, order, None, plane);
        let _ = fast_tiff_lib::read_plane_u8(data, frame, order, plane);
        let _ = fast_tiff_lib::read_plane_f32(data, frame, order, plane);
    }
    let _ = fast_tiff_lib::read_planes_u16(data, frame, order, None);
    let _ = fast_tiff_lib::read_planes_u8(data, frame, order);
    let _ = fast_tiff_lib::read_planes_f32(data, frame, order);
}

/// Run `exercise` with unwinding caught, so one panicking case reports its input
/// instead of taking the whole suite down. (An *abort* can't be caught — if the
/// size guards regress, this binary dies outright, which is the loud signal.)
fn exercise_catching(bytes: &[u8]) -> Result<(), ()> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exercise(bytes))).map_err(|_| ())
}

// ---- regression cases: inputs that used to abort or over-allocate ----

/// A tiny file must not be able to demand a huge buffer. Both of these are
/// ~130 bytes; before the size guards, the first allocated 128 MB and the second
/// overflowed `usize` into a 512 TiB request that aborted the process.
#[test]
fn tiny_file_declaring_huge_dimensions_is_rejected() {
    let cases: [(&str, u32, u32, u16, u16, u16); 5] = [
        // label,                     w,      h,     bits, spp, compression
        ("8000x8000 16-bit LZW", 8_000, 8_000, 16, 1, 5),
        ("u32::MAX wide", u32::MAX, 4, 16, 1, 5),
        ("u32::MAX tall", 4, u32::MAX, 16, 1, 5),
        // width*height*spp*bytes == 2^64 exactly -> wraps to 0 without checked math
        ("overflow to zero", 1 << 24, 1 << 24, 64, 1 << 13, 1),
        ("overflow, compressed", 1 << 24, 1 << 24, 64, 1 << 13, 5),
    ];
    for (label, w, h, bits, spp, compression) in cases {
        let bytes = build_tiff(w, h, bits, spp, compression, &[0u8; 8]);
        assert!(
            bytes.len() < 512,
            "{label}: the probe file itself should be tiny, got {} bytes",
            bytes.len()
        );
        // Must not panic. Must not abort (an abort kills this process outright).
        assert!(exercise_catching(&bytes).is_ok(), "{label}: panicked on malformed input");

        // And whatever the file claims, no decode may succeed — there is nowhere
        // near enough data behind it.
        if let Ok(stack) = TiffStack::from_bytes(bytes.clone()) {
            if let Some(frame) = stack.frames.first() {
                assert!(
                    fast_tiff_lib::read_frame_u16(&stack.data, frame, stack.byte_order, None).is_err(),
                    "{label}: decoded pixels that cannot exist in a {}-byte file",
                    bytes.len()
                );
            }
        }
    }
}

/// A *compressed* frame claiming an enormous height, with `RowsPerStrip` set
/// high enough that its single strip nominally "covers" the whole image.
///
/// This is its own class of attack, and the reason the size guard needs an
/// input-supply bound and not only a structural one: `RowsPerStrip` is declared
/// by the file, so "the strips can cover this image" is trivially satisfiable at
/// any height. When only the structural bound was in place these inputs were
/// *accepted* and the allocation slowly serviced — the robustness suite went
/// from 0.01s to 38 minutes, which is a denial of service in its own right.
/// Hence the deadline: rejection must be immediate, not merely eventual.
#[test]
fn compressed_frame_with_inflated_rows_per_strip_is_rejected_fast() {
    let start = std::time::Instant::now();
    for (w, h, compression) in [
        (4u32, 0x3FFF_FFFFu32, 5u16),   // LZW
        (4, 0x3FFF_FFFF, 8),            // Deflate
        (4, 0x3FFF_FFFF, 32773),        // PackBits
        (0x0FFF_FFFF, 64, 5),           // wide rather than tall
    ] {
        // RowsPerStrip == height, so `strips x rows x row_bytes` == the full
        // frame and the structural check alone would wave it through.
        let mut bytes = build_tiff(w, h, 16, 1, compression, &[0u8; 8]);
        // Patch RowsPerStrip (tag 278) from `h` to u32::MAX for good measure.
        patch_long_tag(&mut bytes, 278, u32::MAX);

        assert!(exercise_catching(&bytes).is_ok(), "panicked on {w}x{h} compression={compression}");
        let stack = TiffStack::from_bytes(bytes).expect("header itself is well-formed");
        let err = fast_tiff_lib::read_frame_u16(&stack.data, &stack.frames[0], stack.byte_order, None)
            .expect_err("a 130-byte file cannot supply a multi-gigabyte frame");
        let msg = err.to_string();
        assert!(
            msg.contains("refusing to allocate") || msg.contains("beyond any real codec"),
            "expected a size-guard rejection, got: {msg}"
        );
    }
    // Generous, but far below the minutes a serviced allocation would take.
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "size guards took {:?} — they must reject up front, not allocate first",
        start.elapsed()
    );
}

/// Overwrite the value of a LONG-typed IFD entry in a file built by `build_tiff`.
fn patch_long_tag(buf: &mut [u8], tag: u16, value: u32) {
    let ifd_off = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(buf[ifd_off..ifd_off + 2].try_into().unwrap()) as usize;
    for i in 0..count {
        let e = ifd_off + 2 + i * 12;
        if u16::from_le_bytes(buf[e..e + 2].try_into().unwrap()) == tag {
            buf[e + 8..e + 12].copy_from_slice(&value.to_le_bytes());
            return;
        }
    }
    panic!("tag {tag} not present in the built file");
}

/// The `ImageDescription` tag is free-form text that the ImageJ and OME parsers
/// read dimensions out of, and the channel count sizes a `Vec<ChannelDisplay>`
/// (a 256-entry LUT apiece). A one-line description must not be able to demand
/// hundreds of gigabytes, and the derived frame-count division must not divide
/// by zero when the factors overflow.
#[test]
fn hostile_metadata_dimensions_are_clamped() {
    let huge = usize::MAX;
    let cases = [
        format!("ImageJ=1.54f\nchannels={huge}\n"),
        format!("ImageJ=1.54f\nslices={huge}\n"),
        format!("ImageJ=1.54f\nchannels={huge}\nslices={huge}\n"),
        // channels * slices overflows usize to exactly 0 -> divide-by-zero if unguarded
        format!("ImageJ=1.54f\nchannels={}\nslices={}\n", 1u64 << 32, 1u64 << 32),
        "ImageJ=1.54f\nchannels=4000000000\nmode=composite\n".to_string(),
        format!(
            "<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">\
             <Image><Pixels Type=\"uint16\" SizeX=\"4\" SizeY=\"4\" SizeC=\"{huge}\" SizeZ=\"{huge}\">\
             <TiffData/></Pixels></Image></OME>"
        ),
        format!(
            "<?xml version=\"1.0\"?><OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">\
             <Image><Pixels Type=\"uint16\" SizeX=\"4\" SizeY=\"4\" SizeC=\"{}\" SizeZ=\"{}\">\
             <TiffData/></Pixels></Image></OME>",
            1u64 << 32,
            1u64 << 32
        ),
    ];
    for desc in cases {
        let bytes = build_tiff_with_description(&desc);
        assert!(
            exercise_catching(&bytes).is_ok(),
            "panicked on a hostile description: {}",
            &desc[..desc.len().min(80)]
        );
        // The file has exactly one plane, so no honest reading of it has more
        // than one channel — and `channel_display` must stay in step with the
        // reported count, since callers index it by channel.
        let stack = TiffStack::from_bytes(bytes).expect("a valid image with odd metadata still opens");
        assert!(
            stack.meta.channels <= stack.frames.len(),
            "channels ({}) exceeds the {} plane(s) in the file",
            stack.meta.channels,
            stack.frames.len()
        );
        assert_eq!(
            stack.meta.channel_display.len(),
            stack.meta.channels,
            "channel_display must match the channel count — callers index it by channel"
        );
    }
}

/// Truncation at every prefix length: a file cut off mid-header, mid-IFD, or
/// mid-strip must error, never panic.
#[test]
fn every_truncation_errors_cleanly() {
    let full = valid_tiff();
    for cut in 0..full.len() {
        assert!(
            exercise_catching(&full[..cut]).is_ok(),
            "panicked on a {cut}-byte truncation of a valid TIFF"
        );
    }
}

/// Single-byte corruption at every offset. Catches panics reachable by flipping
/// one field (a tag type, a count, an offset) without changing the file's shape.
#[test]
fn single_byte_corruption_never_panics() {
    let full = valid_tiff();
    for pos in 0..full.len() {
        for patch in [0x00u8, 0x01, 0x7f, 0x80, 0xff] {
            let mut m = full.clone();
            m[pos] = patch;
            assert!(
                exercise_catching(&m).is_ok(),
                "panicked with byte {pos} set to {patch:#04x}"
            );
        }
    }
}

/// Deterministic multi-byte mutation sweep — a cheap always-on stand-in for the
/// `cargo fuzz` targets, seeded so a failure is exactly reproducible.
#[test]
fn deterministic_mutation_sweep_never_panics() {
    let full = valid_tiff();
    // xorshift64*: no dev-dependency, identical sequence on every platform.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for round in 0..2_000 {
        let mut m = full.clone();
        let edits = 1 + (next() % 6) as usize;
        for _ in 0..edits {
            let pos = (next() as usize) % m.len();
            m[pos] = (next() & 0xff) as u8;
        }
        assert!(
            exercise_catching(&m).is_ok(),
            "panicked in mutation round {round} (seed 0x2545F4914F6CDD1D)"
        );
    }
}

/// The valid seed must still decode correctly — proof the hardening did not
/// break the happy path these tests mutate away from.
#[test]
fn the_valid_seed_still_decodes() {
    let stack = TiffStack::from_bytes(valid_tiff()).expect("seed must open");
    let frame = &stack.frames[0];
    assert_eq!((frame.width, frame.height), (4, 4));
    let px = fast_tiff_lib::read_frame_u16(&stack.data, frame, stack.byte_order, None).expect("seed must decode");
    assert_eq!(px.len(), 16);
    assert_eq!(px[0], 0);
    assert_eq!(px[1], 4000);
    assert_eq!(px[15], 15 * 4000);
}

// ---- IFD chain cycles ----
//
// A looping chain has to be rejected rather than walked forever. The check is
// Brent's cycle detection over the same forward walk that builds the index —
// it replaced a `HashSet` of every offset seen, which cost a hash insert per
// frame and tens of megabytes on a large stack for a check that only ever
// fires on a malformed file. Different algorithm, same guarantee, so it is
// worth pinning all three shapes: a self-loop, a cycle through the head, and a
// cycle the head is *not* part of (the case that needs the tortoise to walk
// into the loop before it can meet the hare).

/// Offset of the first IFD, and of its next-IFD pointer, in a `build_tiff` file.
fn ifd_offsets(buf: &[u8]) -> (u32, usize) {
    let ifd_off = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    (ifd_off, buf.len() - 4)
}

/// Append `n` more copies of the file's single IFD, then link the chain
/// according to `next`: entry `i` of `next` is the index of the IFD that IFD
/// `i` points at. Index 0 is the original.
fn chain_ifds(mut buf: Vec<u8>, copies: usize, next: &[usize]) -> Vec<u8> {
    let (ifd_off, _) = ifd_offsets(&buf);
    let block = buf[ifd_off as usize..].to_vec();
    let mut offsets = vec![ifd_off];
    for _ in 0..copies {
        offsets.push(buf.len() as u32);
        buf.extend_from_slice(&block);
    }
    for (i, &target) in next.iter().enumerate() {
        // The next-IFD pointer is the last 4 bytes of each IFD block.
        let ptr = offsets[i] as usize + block.len() - 4;
        buf[ptr..ptr + 4].copy_from_slice(&offsets[target].to_le_bytes());
    }
    buf
}

#[test]
fn self_referential_ifd_chain_is_rejected() {
    let mut buf = valid_tiff();
    let (ifd_off, next_ptr) = ifd_offsets(&buf);
    buf[next_ptr..next_ptr + 4].copy_from_slice(&ifd_off.to_le_bytes());
    let err = match TiffStack::from_bytes(buf) {
        Ok(_) => panic!("a looping IFD chain was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("loops back"), "expected a loop error, got: {err}");
}

#[test]
fn ifd_chain_cycle_through_the_first_directory_is_rejected() {
    // 0 -> 1 -> 0
    let buf = chain_ifds(valid_tiff(), 1, &[1, 0]);
    let err = match TiffStack::from_bytes(buf) {
        Ok(_) => panic!("a looping IFD chain was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("loops back"), "expected a loop error, got: {err}");
}

#[test]
fn ifd_chain_cycle_the_head_is_not_part_of_is_rejected() {
    // 0 -> 1 -> 2 -> 1: the first directory is outside the loop.
    let buf = chain_ifds(valid_tiff(), 2, &[1, 2, 1]);
    let err = match TiffStack::from_bytes(buf) {
        Ok(_) => panic!("a looping IFD chain was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("loops back"), "expected a loop error, got: {err}");
}

#[test]
fn a_long_acyclic_chain_is_not_mistaken_for_a_loop() {
    // The counterpart the cycle check must not break: 40 distinct directories
    // ending in 0. Brent's resets its reference offset at powers of two, so a
    // chain long enough to cross several of those is the shape to check.
    let n = 40usize;
    let next: Vec<usize> = (1..=n).collect(); // 0->1, 1->2, ..., last->0 (end)
    let buf = chain_ifds(valid_tiff(), n, &next[..n]);
    let stack = TiffStack::from_bytes(buf).expect("a chain of distinct IFDs must open");
    assert_eq!(stack.frames.len(), n + 1);
}

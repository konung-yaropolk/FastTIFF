//! CMYK (`PhotometricInterpretation = 5`, "Separated") decoding.
//!
//! Separated files store *ink coverage*: how much of each plate the press lays
//! down. Ink absorbs light, so more ink means a darker pixel — the opposite of
//! the RGB channels the rest of the pipeline assumes. These tests pin both the
//! conversion and, just as importantly, the fact that adding it changed nothing
//! about how the raw per-sample readers behave.

use fast_tiff_lib::TiffStack;

const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_PHOTOMETRIC: u16 = 262;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_SAMPLES_PER_PIXEL: u16 = 277;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_PLANAR_CONFIG: u16 = 284;
const TAG_INK_SET: u16 = 332;
const TAG_EXTRA_SAMPLES: u16 = 338;
const TAG_SAMPLE_FORMAT: u16 = 339;

const UNSIGNED: u16 = 1;
const SIGNED: u16 = 2;
const FLOAT: u16 = 3;

type IfdEntrySpec = (u16, u16, u32, [u8; 4]); // tag, type, count, inline-or-offset value

fn short_val(v: u16) -> [u8; 4] {
    let mut b = [0u8; 4];
    b[0..2].copy_from_slice(&v.to_le_bytes());
    b
}

fn long_val(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

/// Writes `payload` into an IFD value field, spilling to the end of `buf` and
/// returning the offset when it does not fit in the inline four bytes. Spilled
/// values are word-aligned, as the spec requires.
fn place(buf: &mut Vec<u8>, payload: &[u8]) -> [u8; 4] {
    if payload.len() <= 4 {
        let mut v = [0u8; 4];
        v[..payload.len()].copy_from_slice(payload);
        return v;
    }
    if buf.len() % 2 == 1 {
        buf.push(0);
    }
    let off = buf.len() as u32;
    buf.extend_from_slice(payload);
    long_val(off)
}

/// How a Separated test file is laid out. Defaults to the plain case — 8-bit
/// unsigned, chunky, `InkSet = 1` — so each test names only what it varies.
#[derive(Clone, Copy)]
struct Layout {
    bits: u16,
    planar: u16,
    ink_set: u16,
    sample_format: u16,
}

impl Default for Layout {
    fn default() -> Self {
        Layout { bits: 8, planar: 1, ink_set: 1, sample_format: UNSIGNED }
    }
}

impl Layout {
    fn bits(self, bits: u16) -> Self {
        Layout { bits, ..self }
    }
    fn planar(self) -> Self {
        Layout { planar: 2, ..self }
    }
    fn ink_set(self, ink_set: u16) -> Self {
        Layout { ink_set, ..self }
    }
    fn format(self, sample_format: u16) -> Self {
        Layout { sample_format, ..self }
    }
}

/// Builds a single-IFD Separated TIFF, one row wide enough to hold `inks`.
/// `inks` is one Vec of sample values per plate — four for plain CMYK, more to
/// exercise extra samples. Values are given as `u16` whatever the target depth;
/// the tests that vary depth do not check pixels, only whether the frame is
/// recognised as CMYK.
fn build_cmyk_tiff(inks: &[Vec<u16>], layout: Layout) -> Vec<u8> {
    let Layout { bits, planar, ink_set, sample_format } = layout;
    let spp = inks.len() as u16;
    let px = inks[0].len();
    assert!(inks.iter().all(|p| p.len() == px), "every ink plane must be the same length");

    let mut buf = Vec::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // first-IFD offset, patched below

    let emit = |b: &mut Vec<u8>, v: u16| match bits {
        8 => b.push(v as u8),
        16 => b.extend_from_slice(&v.to_le_bytes()),
        _ => b.extend_from_slice(&(v as u32).to_le_bytes()),
    };

    // Chunky keeps all samples of a pixel together in one strip; planar gives
    // each plate its own strip. Both are ordinary TIFF layouts, and the reader
    // is expected to gather either into the same per-plane result.
    let (offsets, counts) = if planar == 1 {
        let start = buf.len() as u32;
        for i in 0..px {
            for ink in inks {
                emit(&mut buf, ink[i]);
            }
        }
        (vec![start], vec![buf.len() as u32 - start])
    } else {
        let (mut o, mut c) = (Vec::new(), Vec::new());
        for ink in inks {
            let start = buf.len() as u32;
            for &v in ink {
                emit(&mut buf, v);
            }
            o.push(start);
            c.push(buf.len() as u32 - start);
        }
        (o, c)
    };

    let bps: Vec<u8> = (0..spp).flat_map(|_| bits.to_le_bytes()).collect();
    let fmts: Vec<u8> = (0..spp).flat_map(|_| sample_format.to_le_bytes()).collect();
    let off_bytes: Vec<u8> = offsets.iter().flat_map(|v| v.to_le_bytes()).collect();
    let cnt_bytes: Vec<u8> = counts.iter().flat_map(|v| v.to_le_bytes()).collect();

    let bps_val = place(&mut buf, &bps);
    let fmt_val = place(&mut buf, &fmts);
    let off_val = place(&mut buf, &off_bytes);
    let cnt_val = place(&mut buf, &cnt_bytes);

    let mut entries: Vec<IfdEntrySpec> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, long_val(px as u32)),
        (TAG_IMAGE_LENGTH, 4, 1, long_val(1)),
        (TAG_BITS_PER_SAMPLE, 3, spp as u32, bps_val),
        (TAG_COMPRESSION, 3, 1, short_val(1)),
        (TAG_PHOTOMETRIC, 3, 1, short_val(5)), // Separated
        (TAG_STRIP_OFFSETS, 4, offsets.len() as u32, off_val),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, short_val(spp)),
        (TAG_ROWS_PER_STRIP, 4, 1, long_val(1)),
        (TAG_STRIP_BYTE_COUNTS, 4, counts.len() as u32, cnt_val),
        (TAG_INK_SET, 3, 1, short_val(ink_set)),
        (TAG_PLANAR_CONFIG, 3, 1, short_val(planar)),
        (TAG_SAMPLE_FORMAT, 3, spp as u32, fmt_val),
    ];
    if spp > 4 {
        // Anything past the four inks is unspecified extra data.
        let extra: Vec<u8> = (4..spp).flat_map(|_| 0u16.to_le_bytes()).collect();
        let extra_val = place(&mut buf, &extra);
        entries.push((TAG_EXTRA_SAMPLES, 3, (spp - 4) as u32, extra_val));
    }
    entries.sort_by_key(|e| e.0);

    if buf.len() % 2 == 1 {
        buf.push(0);
    }
    let ifd_offset = buf.len() as u32;
    buf[4..8].copy_from_slice(&ifd_offset.to_le_bytes());
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

/// CMYK swatches and the RGB each must produce. The conversion is the
/// multiplicative one, `component = (1 - ink) * (1 - black)`, checked against
/// Pillow on real Pillow- and tifffile-written files.
///
/// The last two rows are the ones that carry the weight. Where only one plate
/// is inked, the multiplicative form and the naive additive one
/// (`255 - ink - black`, clipped) agree exactly — so a suite of pure cyan,
/// pure black and paper white passes under either formula and proves nothing.
/// Mixing a partial ink with a partial black is what separates them: for
/// `(128, 64, 32, 64)` the additive form yields `(63, 127, 159)`, nowhere near
/// the correct `(95, 143, 167)`.
const SWATCHES: [([u16; 4], [u8; 3]); 10] = [
    ([0, 0, 0, 0], [255, 255, 255]),        // no ink -> paper white
    ([255, 0, 0, 0], [0, 255, 255]),        // cyan
    ([0, 255, 0, 0], [255, 0, 255]),        // magenta
    ([0, 0, 255, 0], [255, 255, 0]),        // yellow
    ([0, 0, 0, 255], [0, 0, 0]),            // full black plate
    ([255, 255, 0, 0], [0, 0, 255]),        // cyan + magenta -> blue
    ([0, 0, 0, 128], [127, 127, 127]),      // half black -> mid gray
    ([128, 0, 0, 0], [127, 255, 255]),      // half cyan
    ([128, 64, 32, 64], [95, 143, 167]),    // mixed inks over partial black
    ([64, 128, 192, 32], [167, 111, 55]),   // ...and again, different mix
];

/// Transposes [`SWATCHES`] into four ink planes, scaled to `bits` width.
fn swatch_planes(bits: u16) -> Vec<Vec<u16>> {
    let scale = |v: u16| if bits == 8 { v } else { v * 257 }; // 255 -> 65535
    (0..4)
        .map(|i| SWATCHES.iter().map(|(cmyk, _)| scale(cmyk[i])).collect())
        .collect()
}

fn assert_swatches(planes: &[Vec<u8>], label: &str) {
    assert_eq!(planes.len(), 3, "{label}: four inks must convert to three components");
    for (x, (cmyk, want)) in SWATCHES.iter().enumerate() {
        let got = [planes[0][x], planes[1][x], planes[2][x]];
        let close = got.iter().zip(want).all(|(a, b)| (*a as i16 - *b as i16).abs() <= 1);
        assert!(close, "{label}: CMYK{cmyk:?}: got {got:?}, want {want:?}");
    }
}

#[test]
fn cmyk_chunky_8bit_converts_to_rgb() {
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default())).unwrap();
    let frame = &stack.frames[0];
    assert!(frame.is_cmyk());
    assert!(!frame.is_rgb(), "CMYK must not masquerade as photometric=2 RGB");
    let planes = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order).unwrap();
    assert_swatches(&planes, "chunky 8-bit");
}

/// Planar CMYK: identical pixels, one strip per plate. The conversion runs
/// *after* the plane gather, so by the time it applies the layout is already
/// normalised — this test is what proves that, rather than the conversion
/// re-deriving the layout itself and needing a second implementation.
#[test]
fn cmyk_planar_matches_chunky() {
    let chunky = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default())).unwrap();
    let planar = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default().planar())).unwrap();
    assert!(planar.frames[0].is_planar());
    let a = fast_tiff_lib::read_planes_rgb_u8(&chunky.data, &chunky.frames[0], chunky.byte_order).unwrap();
    let b = fast_tiff_lib::read_planes_rgb_u8(&planar.data, &planar.frames[0], planar.byte_order).unwrap();
    assert_eq!(a, b, "planar and chunky CMYK must decode identically");
    assert_swatches(&b, "planar 8-bit");
}

#[test]
fn cmyk_16bit_converts_to_rgb() {
    for (label, layout) in [
        ("16-bit chunky", Layout::default().bits(16)),
        ("16-bit planar", Layout::default().bits(16).planar()),
    ] {
        let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(16), layout)).unwrap();
        let frame = &stack.frames[0];
        assert!(frame.is_cmyk(), "{label}");
        let planes = fast_tiff_lib::read_planes_rgb_u16(&stack.data, frame, stack.byte_order).unwrap();
        let narrowed: Vec<Vec<u8>> = planes.iter().map(|p| p.iter().map(|v| (v >> 8) as u8).collect()).collect();
        assert_swatches(&narrowed, label);
    }
}

/// Single-plane addressing must agree with the batched call — the viewer uses
/// the batched path when several channels are visible and the per-plane one
/// when only a single channel is ticked, and the two must not disagree.
#[test]
fn single_plane_reads_match_the_batched_read() {
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default())).unwrap();
    let frame = &stack.frames[0];
    let batched = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order).unwrap();
    for (c, want) in batched.iter().enumerate() {
        let one = fast_tiff_lib::read_plane_rgb_u8(&stack.data, frame, stack.byte_order, c).unwrap();
        assert_eq!(&one, want, "component {c} differs between the single and batched reads");
    }

    let wide = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(16), Layout::default().bits(16))).unwrap();
    let wf = &wide.frames[0];
    let batched16 = fast_tiff_lib::read_planes_rgb_u16(&wide.data, wf, wide.byte_order).unwrap();
    for (c, want) in batched16.iter().enumerate() {
        let one = fast_tiff_lib::read_plane_rgb_u16(&wide.data, wf, wide.byte_order, c).unwrap();
        assert_eq!(&one, want, "16-bit component {c} differs between the single and batched reads");
    }
}

/// A fifth sample (tagged ExtraSamples) rides alongside the four inks. It is
/// ignored by the conversion, which must still find C, M, Y and K in the first
/// four planes rather than refusing the frame for having "too many" samples.
#[test]
fn cmyk_with_extra_sample_still_converts() {
    let mut planes = swatch_planes(8);
    planes.push(vec![42; SWATCHES.len()]); // an extra sample of no defined meaning
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&planes, Layout::default())).unwrap();
    let frame = &stack.frames[0];
    assert_eq!(frame.samples_per_pixel, 5);
    assert!(frame.is_cmyk());
    let rgb = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order).unwrap();
    assert_swatches(&rgb, "5-sample");
}

/// `InkSet = 2` means "not CMYK" — some other set of inks, in some other order
/// (hi-fi printing adds orange/green/violet plates). Reading those four numbers
/// as C, M, Y, K would produce confidently wrong colours, so such a frame must
/// fail the CMYK test and fall through to the raw per-ink path.
#[test]
fn separated_with_non_cmyk_inkset_is_not_converted() {
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default().ink_set(2))).unwrap();
    let frame = &stack.frames[0];
    assert_eq!(frame.photometric, 5);
    assert_eq!(frame.ink_set, 2);
    assert!(!frame.is_cmyk(), "InkSet=2 is not CMYK");

    let err = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires a CMYK frame"), "unhelpful error: {err}");

    // ...but the inks themselves still read, so nothing is lost.
    assert_eq!(
        fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap().len(),
        4
    );
}

/// The conversion is defined only for 8- and 16-bit unsigned inks, and the gate
/// has to hold for a reason that is easy to forget: the wide-sample readers
/// auto-range each plane independently when normalising to `u16`. On RGB that
/// is a display convenience; on CMYK it would rescale the four plates against
/// four *different* ranges before they are combined, so the arithmetic would be
/// wrong rather than merely differently-scaled. Anything outside the gate must
/// be left to the raw per-ink path.
#[test]
fn cmyk_gate_rejects_depths_and_formats_the_conversion_cannot_handle() {
    let cases = [
        ("32-bit unsigned", Layout::default().bits(32)),
        ("32-bit float", Layout::default().bits(32).format(FLOAT)),
        ("16-bit signed", Layout::default().bits(16).format(SIGNED)),
        ("8-bit signed", Layout::default().format(SIGNED)),
    ];
    for (label, layout) in cases {
        let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), layout)).unwrap();
        let frame = &stack.frames[0];
        assert_eq!(frame.photometric, 5, "{label}: still a Separated frame");
        assert!(!frame.is_cmyk(), "{label} must not pass the CMYK gate");
        let err = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order)
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires a CMYK frame"), "{label}: unhelpful error: {err}");
    }
}

/// The compatibility guarantee behind this whole feature. `read_planes_u8` is
/// documented as "one plane per sample" and is relied on by the viewer's plane
/// addressing and by the fuzz target, so on a CMYK frame it must still hand
/// back the four untouched ink plates. If this test ever fails, the conversion
/// leaked into the raw path and every existing caller changed behaviour.
#[test]
fn raw_readers_still_return_untouched_ink_planes() {
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&swatch_planes(8), Layout::default())).unwrap();
    let frame = &stack.frames[0];
    assert!(frame.is_cmyk());

    let raw = fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap();
    assert_eq!(raw.len(), 4, "raw reader must still be one plane per sample");
    for (i, plane) in raw.iter().enumerate() {
        let want: Vec<u8> = SWATCHES.iter().map(|(cmyk, _)| cmyk[i] as u8).collect();
        assert_eq!(plane, &want, "ink plane {i} was modified");
    }

    // Single-plane addressing is unchanged too — including plane 3, which the
    // converting reader has no equivalent of.
    assert_eq!(
        fast_tiff_lib::read_plane_u8(&stack.data, frame, stack.byte_order, 3).unwrap(),
        raw[3]
    );
}

/// Non-CMYK frames are refused rather than silently misread. The converting
/// readers are public API; someone will eventually point one at a grayscale or
/// RGB frame, and the answer should be a clear error naming what was wrong.
#[test]
fn converting_readers_refuse_frames_with_too_few_inks() {
    let mut planes = swatch_planes(8);
    planes.truncate(3); // three inks: too few to be CMYK
    let stack = TiffStack::from_bytes(build_cmyk_tiff(&planes, Layout::default())).unwrap();
    let frame = &stack.frames[0];
    assert!(!frame.is_cmyk());
    let err = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires a CMYK frame") && err.contains("3 sample"), "unhelpful error: {err}");
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Writer -> reader round-trip. Without `cmyk()`, four samples would be tagged
/// RGB-plus-alpha, so a file this crate wrote could not be read back as the
/// CMYK it actually holds.
#[test]
fn written_cmyk_reads_back_as_cmyk() {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    use std::io::Cursor;

    for planar in [false, true] {
        // Chunky wants C,M,Y,K per pixel; planar wants whole plates in turn.
        // The writer stores the buffer as given either way, so the test has to
        // hand over the layout it asked for.
        let planes = swatch_planes(8);
        let n = planes[0].len();
        let mut data = Vec::with_capacity(n * 4);
        if planar {
            for p in &planes {
                data.extend(p.iter().map(|&v| v as u8));
            }
        } else {
            for i in 0..n {
                data.extend(planes.iter().map(|p| p[i] as u8));
            }
        }

        let opts = WriterOptions::new(n as u32, 1, SampleType::U8)
            .samples_per_pixel(4)
            .planar(planar)
            .cmyk(true);
        let mut buf = Cursor::new(Vec::new());
        let mut w = TiffWriter::new(&mut buf, opts).unwrap();
        w.write_frame_u8(&data).unwrap();
        w.finish().unwrap();

        let bytes = buf.into_inner();
        // InkSet must be physically present, not merely defaulted on read: an
        // outside reader that does not apply the TIFF6 default would otherwise
        // have nothing to go on.
        let (count, value) = ifd_tag(&bytes, TAG_INK_SET).expect("InkSet should be written");
        assert_eq!((count, value & 0xffff), (1, 1), "InkSet should be a single SHORT of value 1");

        let stack = TiffStack::from_bytes(bytes).unwrap();
        let frame = &stack.frames[0];
        let label = if planar { "planar" } else { "chunky" };
        assert_eq!(frame.photometric, 5, "{label}: must be tagged Separated");
        assert_eq!(frame.ink_set, 1, "{label}");
        assert_eq!(frame.is_planar(), planar, "{label}");
        assert!(frame.is_cmyk(), "{label}: what we wrote must satisfy the reader gate");

        // The plates come back byte-identical...
        let raw = fast_tiff_lib::read_planes_u8(&stack.data, frame, stack.byte_order).unwrap();
        for (i, plane) in raw.iter().enumerate() {
            let want: Vec<u8> = planes[i].iter().map(|&v| v as u8).collect();
            assert_eq!(plane, &want, "{label}: ink plane {i}");
        }
        // ...and convert to the colours the swatches call for.
        let rgb = fast_tiff_lib::read_planes_rgb_u8(&stack.data, frame, stack.byte_order).unwrap();
        assert_swatches(&rgb, label);
    }
}

/// A fifth sample must be declared in ExtraSamples relative to the *four* ink
/// plates, not the three an RGB file would have. Getting that count wrong makes
/// the file self-contradictory: five samples, a photometric that accounts for
/// four, and two declared as extra.
#[test]
fn written_cmyk_declares_extra_samples_against_four_inks() {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    use std::io::Cursor;

    let n = 4usize;
    let data: Vec<u8> = (0..n * 5).map(|i| (i * 7) as u8).collect();
    let opts = WriterOptions::new(n as u32, 1, SampleType::U8)
        .samples_per_pixel(5)
        .cmyk(true);
    let mut buf = Cursor::new(Vec::new());
    let mut w = TiffWriter::new(&mut buf, opts).unwrap();
    w.write_frame_u8(&data).unwrap();
    w.finish().unwrap();

    let bytes = buf.into_inner();
    let stack = TiffStack::from_bytes(bytes.clone()).unwrap();
    let frame = &stack.frames[0];
    assert_eq!(frame.samples_per_pixel, 5);
    assert!(frame.is_cmyk());

    // Read tag 338 straight out of the file: exactly one extra sample.
    let (count, _) = ifd_tag(&bytes, TAG_EXTRA_SAMPLES).expect("ExtraSamples tag should be present");
    assert_eq!(count, 1, "5 samples - 4 inks = 1 extra, not 2");
}

/// Find one tag in the first IFD of a little-endian classic TIFF and return
/// `(count, inline value)`, so assertions can be about the bytes on disk rather
/// than about what the reader chose to believe. That distinction matters for
/// tags the reader supplies a default for: InkSet reads back as 1 whether or
/// not it was written, so only the file itself can say it is really there.
fn ifd_tag(bytes: &[u8], tag: u16) -> Option<(u32, u32)> {
    let ifd = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let n = u16::from_le_bytes(bytes[ifd..ifd + 2].try_into().ok()?) as usize;
    (0..n).find_map(|i| {
        let e = ifd + 2 + i * 12;
        (u16::from_le_bytes(bytes[e..e + 2].try_into().ok()?) == tag).then(|| {
            (
                u32::from_le_bytes(bytes[e + 4..e + 8].try_into().unwrap()),
                u32::from_le_bytes(bytes[e + 8..e + 12].try_into().unwrap()),
            )
        })
    })
}

/// The writer refuses combinations the reader would not accept back. Writing a
/// file that opens but then fails to convert would be a worse outcome than
/// failing at the call that asked for it.
#[test]
fn cmyk_writer_rejects_what_the_reader_would_refuse() {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    use std::io::Cursor;

    let cases: [(&str, WriterOptions, &str); 3] = [
        (
            "too few samples",
            WriterOptions::new(4, 1, SampleType::U8).samples_per_pixel(3).cmyk(true),
            "at least 4 samples",
        ),
        (
            "float samples",
            WriterOptions::new(4, 1, SampleType::F32).samples_per_pixel(4).cmyk(true),
            "unsigned 8- or 16-bit",
        ),
        (
            "signed samples",
            WriterOptions::new(4, 1, SampleType::I16).samples_per_pixel(4).cmyk(true),
            "unsigned 8- or 16-bit",
        ),
    ];
    for (label, opts, expected) in cases {
        let err = TiffWriter::new(Cursor::new(Vec::new()), opts)
            .err()
            .unwrap_or_else(|| panic!("{label}: should have been rejected"))
            .to_string();
        assert!(err.contains(expected), "{label}: unhelpful error: {err}");
    }
}

/// Without `cmyk()`, nothing changes: four samples stay RGB-plus-extra, exactly
/// as before this feature existed.
#[test]
fn four_samples_without_the_flag_are_still_rgb() {
    use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
    use std::io::Cursor;

    let data: Vec<u8> = (0..4 * 4).map(|i| (i * 7) as u8).collect();
    let opts = WriterOptions::new(4, 1, SampleType::U8).samples_per_pixel(4);
    let mut buf = Cursor::new(Vec::new());
    let mut w = TiffWriter::new(&mut buf, opts).unwrap();
    w.write_frame_u8(&data).unwrap();
    w.finish().unwrap();

    let stack = TiffStack::from_bytes(buf.into_inner()).unwrap();
    let frame = &stack.frames[0];
    assert_eq!(frame.photometric, 2, "unflagged 4-sample output stays RGB");
    assert!(!frame.is_cmyk());
    assert!(frame.is_rgb());
}

//! What actually comes out of a PNG.
//!
//! Every case here is a PNG built by the `png` crate's own encoder and read
//! back through the real [`Importer`] — so what is checked is the file, not a
//! helper called with the arguments it expects.
//!
//! The cases worth having are the ones where a wrong reader still produces a
//! picture: a palette read as samples, a 16-bit file read little-endian, a
//! colour file de-interleaved at the wrong offset, a grey file that happens to
//! carry transparency and so has two samples per pixel instead of one. None of
//! those fail loudly. Each one is pinned.

use super::*;
use fasttiff_plugin_api::Params;

/// A host that remembers what it was told, so the lines the reader promises to
/// log can be checked rather than assumed.
#[derive(Default)]
struct Notes {
    lines: Vec<String>,
    progress: Vec<f32>,
}

impl ImportHost for Notes {
    fn progress(&mut self, fraction: f32) -> bool {
        self.progress.push(fraction);
        true
    }
    fn log(&mut self, message: &str) {
        self.lines.push(message.to_string());
    }
}

impl Notes {
    fn said(&self, needle: &str) -> bool {
        self.lines.iter().any(|l| l.contains(needle))
    }
}

/// A PNG built to order.
fn encode(
    width: u32,
    height: u32,
    color: ::png::ColorType,
    depth: ::png::BitDepth,
    data: &[u8],
    setup: impl FnOnce(&mut ::png::Encoder<'_, &mut Vec<u8>>),
) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = ::png::Encoder::new(&mut out, width, height);
        enc.set_color(color);
        enc.set_depth(depth);
        setup(&mut enc);
        let mut w = enc.write_header().expect("header");
        w.write_image_data(data).expect("image data");
        w.finish().expect("finish");
    }
    out
}

fn plain(
    width: u32,
    height: u32,
    color: ::png::ColorType,
    depth: ::png::BitDepth,
    data: &[u8],
) -> Vec<u8> {
    encode(width, height, color, depth, data, |_| {})
}

/// A host that stops the import the first time it is asked.
struct Stops;

impl ImportHost for Stops {
    fn progress(&mut self, _fraction: f32) -> bool {
        false
    }
    fn log(&mut self, _message: &str) {}
}

/// A PNG whose extra chunks are written *after* the image data, which is where
/// a streaming writer has to put them and where `next_frame` stops short of.
fn encode_trailing(
    width: u32,
    height: u32,
    data: &[u8],
    after: impl FnOnce(&mut ::png::Writer<&mut Vec<u8>>),
) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = ::png::Encoder::new(&mut out, width, height);
        enc.set_color(::png::ColorType::Grayscale);
        enc.set_depth(::png::BitDepth::Eight);
        let mut w = enc.write_header().expect("header");
        w.write_image_data(data).expect("image data");
        after(&mut w);
        w.finish().expect("finish");
    }
    out
}

/// `png`'s bytes for one chunk of type `kind`, and where it sits.
///
/// Chunks are length-prefixed and self-delimiting, so finding one is a walk
/// from the signature rather than a search.
fn find_chunk(png: &[u8], kind: &[u8; 4]) -> Option<std::ops::Range<usize>> {
    let mut at = MAGIC.len();
    while at + 8 <= png.len() {
        let len = u32::from_be_bytes(png[at..at + 4].try_into().ok()?) as usize;
        let end = at + 12 + len;
        if end > png.len() {
            return None;
        }
        if &png[at + 4..at + 8] == kind {
            return Some(at..end);
        }
        at = end;
    }
    None
}

/// A PNG carrying `n` copies of its own compressed-text chunk.
///
/// Spliced rather than encoded `n` times: compressing 64 KiB of text costs far
/// more than decompressing it, so building this the obvious way would make the
/// *fixture* the slow part and the measurement meaningless. The copies are
/// byte-identical, so every CRC is still the one `png` computed. PNG allows a
/// text chunk to appear as often as it likes.
fn with_many_ztxt(n: usize) -> Vec<u8> {
    let one = encode(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[7],
        |e| {
            e.add_ztxt_chunk("Comment".into(), "z".repeat(64 * 1024))
                .unwrap()
        },
    );
    let at = find_chunk(&one, b"zTXt").expect("the fixture must carry a zTXt");
    let chunk = one[at.clone()].to_vec();
    // Compressible enough that the file stays small while the text it stands
    // for does not.
    assert!(
        chunk.len() < 1024,
        "the fixture chunk is too big: {}",
        chunk.len()
    );

    let mut out = Vec::with_capacity(one.len() + chunk.len() * n);
    out.extend_from_slice(&one[..at.start]);
    for _ in 0..n {
        out.extend_from_slice(&chunk);
    }
    out.extend_from_slice(&one[at.end..]);
    out
}

fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("fasttiff_png_import_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn read(name: &str, bytes: &[u8]) -> (ImportResult, Notes) {
    let path = write(name, bytes);
    let mut host = Notes::default();
    let r = PngImport
        .import(
            &ImportRequest {
                path,
                params: Params::new(),
            },
            &mut host,
        )
        .expect("should import");
    r.image
        .validate()
        .expect("the result must describe itself correctly");
    (r, host)
}

fn u8s(plane: &PlaneData) -> Vec<u8> {
    match plane {
        PlaneData::U8(v) => v.clone(),
        other => panic!("expected U8, got {:?}", other.pixel_type()),
    }
}

fn u16s(plane: &PlaneData) -> Vec<u16> {
    match plane {
        PlaneData::U16(v) => v.clone(),
        other => panic!("expected U16, got {:?}", other.pixel_type()),
    }
}

// ---------------------------------------------------------------- the samples

/// An 8-bit greyscale PNG arrives sample for sample.
#[test]
fn an_8_bit_greyscale_png_arrives_unchanged() {
    let px: Vec<u8> = vec![0, 1, 17, 128, 254, 255];
    let bytes = plain(
        3,
        2,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &px,
    );
    let (r, _) = read("gray8.png", &bytes);

    assert_eq!((r.image.width, r.image.height), (3, 2));
    assert_eq!(r.image.channels, 1);
    assert_eq!(r.image.frames, 1);
    assert_eq!(r.image.slices, 1);
    assert_eq!(r.image.pixel_type, PixelType::U8);
    assert_eq!(u8s(&r.image.planes[0]), px);
}

/// A 16-bit PNG stays 16-bit, and its samples are read big-endian.
///
/// This is most of the reason the importer exists rather than the file being
/// opened as a picture: narrowed to eight bits a 16-bit measurement loses eight
/// of them, and read little-endian it keeps all of them in the wrong order —
/// which looks like noise, not like a bug.
#[test]
fn a_16_bit_png_keeps_its_width_and_its_byte_order() {
    let vals: [u16; 4] = [0, 1, 256, 65535];
    let mut px = Vec::new();
    for v in vals {
        px.extend_from_slice(&v.to_be_bytes());
    }
    let bytes = plain(
        2,
        2,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Sixteen,
        &px,
    );
    let (r, _) = read("gray16.png", &bytes);

    assert_eq!(r.image.pixel_type, PixelType::U16);
    // Read the other way round these would be 0, 256, 1, 65535 — all plausible.
    assert_eq!(u16s(&r.image.planes[0]), vals);
}

/// Colour arrives as three separate channels, each holding its own samples.
///
/// The values are chosen so that any confusion between channels — a swap, an
/// off-by-one in the interleave — lands on a different number rather than on
/// the same one.
#[test]
fn a_colour_png_is_split_into_three_channels() {
    // 2x2; in each pixel red counts from 10, green from 100, blue from 200.
    let mut px = Vec::new();
    for i in 0..4u8 {
        px.extend_from_slice(&[10 + i, 100 + i, 200 + i]);
    }
    let bytes = plain(2, 2, ::png::ColorType::Rgb, ::png::BitDepth::Eight, &px);
    let (r, _) = read("rgb8.png", &bytes);

    assert_eq!(r.image.channels, 3);
    assert_eq!(u8s(&r.image.planes[0]), vec![10, 11, 12, 13]);
    assert_eq!(u8s(&r.image.planes[1]), vec![100, 101, 102, 103]);
    assert_eq!(u8s(&r.image.planes[2]), vec![200, 201, 202, 203]);
    // Colour, so the viewer composites it rather than showing one plane grey.
    assert_eq!(r.info.as_ref().unwrap().mode, DisplayMode::Composite);
}

/// A 16-bit colour PNG: both the width and the de-interleaving at once.
#[test]
fn a_16_bit_colour_png_is_split_correctly() {
    let pixels: [[u16; 3]; 2] = [[1, 40000, 65535], [2, 40001, 65534]];
    let mut px = Vec::new();
    for p in pixels {
        for v in p {
            px.extend_from_slice(&v.to_be_bytes());
        }
    }
    let bytes = plain(2, 1, ::png::ColorType::Rgb, ::png::BitDepth::Sixteen, &px);
    let (r, _) = read("rgb16.png", &bytes);

    assert_eq!(r.image.pixel_type, PixelType::U16);
    assert_eq!(u16s(&r.image.planes[0]), vec![1, 2]);
    assert_eq!(u16s(&r.image.planes[1]), vec![40000, 40001]);
    assert_eq!(u16s(&r.image.planes[2]), vec![65535, 65534]);
}

/// A palette image arrives as the colours it stands for, not as its indices.
///
/// The one case that would be wrong *everywhere* and still look like an image:
/// carried through as samples, a palette index is a small number, so a
/// three-colour picture would open as an almost-black one, and a palette that
/// is not a ramp would open as a picture of something else entirely.
#[test]
fn a_palette_png_arrives_as_colours_not_indices() {
    let palette = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
    let indices = [0u8, 1, 2, 1];
    let bytes = encode(
        4,
        1,
        ::png::ColorType::Indexed,
        ::png::BitDepth::Eight,
        &indices,
        |e| e.set_palette(palette.clone()),
    );
    let (r, _) = read("palette.png", &bytes);

    assert_eq!(r.image.channels, 3, "a palette stands for colours");
    assert_eq!(u8s(&r.image.planes[0]), vec![255, 0, 0, 0]);
    assert_eq!(u8s(&r.image.planes[1]), vec![0, 255, 0, 255]);
    assert_eq!(u8s(&r.image.planes[2]), vec![0, 0, 255, 0]);
}

/// A narrow bit depth is widened to the whole of 8 bits, not left as 0 and 1.
///
/// A 1-bit mask read as a 0/1 image is black. It is the same picture, in the
/// sense that the contrast slider could recover it — and nobody would think to
/// reach for the slider, because the file looks empty.
#[test]
fn a_1_bit_png_is_expanded_to_the_full_range() {
    // Eight pixels in one byte: 1, 0, 1, 0, 1, 0, 1, 0.
    let bytes = plain(
        8,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::One,
        &[0b1010_1010],
    );
    let (r, _) = read("bilevel.png", &bytes);

    assert_eq!(r.image.pixel_type, PixelType::U8);
    assert_eq!(
        u8s(&r.image.planes[0]),
        vec![255, 0, 255, 0, 255, 0, 255, 0]
    );
}

/// And a 4-bit one, whose top value is 15 rather than 1.
#[test]
fn a_4_bit_png_is_expanded_to_the_full_range() {
    // Two pixels per byte: 0, 15, 8, 1.
    let bytes = plain(
        4,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Four,
        &[0x0F, 0x81],
    );
    let (r, _) = read("gray4.png", &bytes);

    // 15 * 17 = 255, 8 * 17 = 136, 1 * 17 = 17.
    assert_eq!(u8s(&r.image.planes[0]), vec![0, 255, 136, 17]);
}

// ------------------------------------------------------------------ the alpha

/// Alpha is dropped, the colours under it are left alone, and the log says so.
///
/// Kept as a fourth channel it would be composited as if it were light; and
/// dropped in silence it would leave a viewer showing three of a file's four
/// channels with nothing to say why.
#[test]
fn alpha_is_dropped_and_the_colours_are_untouched() {
    // A fully transparent bright pixel, and an opaque dark one.
    let px: Vec<u8> = vec![200, 100, 50, 0, 10, 20, 30, 255];
    let bytes = plain(2, 1, ::png::ColorType::Rgba, ::png::BitDepth::Eight, &px);
    let (r, host) = read("rgba8.png", &bytes);

    assert_eq!(r.image.channels, 3);
    assert_eq!(r.image.planes.len(), 3);
    // Not premultiplied, not zeroed: the sample is what the file stored.
    assert_eq!(u8s(&r.image.planes[0]), vec![200, 10]);
    assert_eq!(u8s(&r.image.planes[1]), vec![100, 20]);
    assert_eq!(u8s(&r.image.planes[2]), vec![50, 30]);
    assert!(host.said("alpha dropped"), "{:?}", host.lines);
}

/// Grey with alpha is still one channel — and its samples are still its own.
///
/// Two samples per pixel in a file the reader presents as one channel: get the
/// stride wrong and every second sample read is an opacity, which for an opaque
/// image is 255 — a picture of white stripes.
#[test]
fn greyscale_with_alpha_is_one_channel_read_at_the_right_stride() {
    let px: Vec<u8> = vec![10, 255, 20, 255, 30, 128, 40, 0];
    let bytes = plain(
        4,
        1,
        ::png::ColorType::GrayscaleAlpha,
        ::png::BitDepth::Eight,
        &px,
    );
    let (r, host) = read("graya8.png", &bytes);

    assert_eq!(r.image.channels, 1);
    assert_eq!(u8s(&r.image.planes[0]), vec![10, 20, 30, 40]);
    assert!(host.said("alpha dropped"), "{:?}", host.lines);
    assert_eq!(r.info.as_ref().unwrap().mode, DisplayMode::Grayscale);
}

/// A plain grey PNG that merely *mentions* transparency grows a second sample
/// per pixel on the way out of the decoder. Its samples are still its own.
///
/// This is the trap in `EXPAND`: the file is `Grayscale`, the decoder's output
/// is `GrayscaleAlpha`, and a reader that trusted the file's own colour type
/// would read every second byte as a sample.
#[test]
fn a_transparency_chunk_does_not_shift_the_samples() {
    let px: Vec<u8> = vec![10, 20, 30, 40];
    let bytes = encode(
        4,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &px,
        // Sample value 30 is the transparent one, as a big-endian u16.
        |e| e.set_trns(vec![0, 30]),
    );
    let (r, _) = read("gray_trns.png", &bytes);

    assert_eq!(r.image.channels, 1);
    assert_eq!(u8s(&r.image.planes[0]), px);
}

// ------------------------------------------------------------ what comes with

/// A file that states its pixel size is calibrated in microns.
#[test]
fn a_png_that_states_its_pixel_size_is_calibrated() {
    let bytes = encode(
        2,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[1, 2],
        |e| {
            e.set_pixel_dims(Some(::png::PixelDimensions {
                // Two and four million pixels per metre: half a micron, then a
                // quarter.
                xppu: 2_000_000,
                yppu: 4_000_000,
                unit: ::png::Unit::Meter,
            }))
        },
    );
    let (r, host) = read("phys.png", &bytes);

    let info = r.info.as_ref().unwrap();
    assert_eq!(info.unit.as_deref(), Some("micron"));
    assert_eq!(info.spacing.x, Some(0.5));
    assert_eq!(info.spacing.y, Some(0.25));
    assert_eq!(info.spacing.z, None, "one frame has no depth");
    assert!(host.said("0.5000 micron/pixel"), "{:?}", host.lines);
}

/// A pixel size in no particular unit is an aspect ratio, and is not a scale.
///
/// `pHYs` with `unit: Unspecified` says only that pixels are *this* much wider
/// than tall. Taken as metres it would calibrate the image to a number nobody
/// wrote down, and every measurement made on it would be confidently wrong.
#[test]
fn an_uncalibrated_pixel_size_is_not_treated_as_a_scale() {
    let bytes = encode(
        2,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[1, 2],
        |e| {
            e.set_pixel_dims(Some(::png::PixelDimensions {
                xppu: 1,
                yppu: 2,
                unit: ::png::Unit::Unspecified,
            }))
        },
    );
    let (r, _) = read("aspect.png", &bytes);

    let info = r.info.as_ref().unwrap();
    assert_eq!(info.unit, None);
    assert_eq!(info.spacing.x, None);
    assert_eq!(info.spacing.y, None);
}

/// A file with no `pHYs` at all is uncalibrated, and says nothing about microns.
#[test]
fn a_png_without_a_pixel_size_is_uncalibrated() {
    let bytes = plain(
        2,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[1, 2],
    );
    let (r, host) = read("bare.png", &bytes);

    let info = r.info.as_ref().unwrap();
    assert_eq!(info.unit, None);
    assert_eq!(info.spacing.x, None);
    assert!(!host.said("micron"), "{:?}", host.lines);
}

/// Every flavour of text chunk reaches the description, keyword and all.
#[test]
fn the_files_text_becomes_the_description() {
    let bytes = encode(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[7],
        |e| {
            e.add_text_chunk("Software".into(), "FastTIFF".into())
                .unwrap();
            e.add_ztxt_chunk("Comment".into(), "packed away".into())
                .unwrap();
            e.add_itxt_chunk("Title".into(), "a title".into()).unwrap();
        },
    );
    let (r, _) = read("text.png", &bytes);

    let d = r
        .info
        .as_ref()
        .unwrap()
        .description
        .clone()
        .expect("description");
    assert!(d.contains("Software: FastTIFF"), "{d}");
    // Compressed, and still readable.
    assert!(d.contains("Comment: packed away"), "{d}");
    assert!(d.contains("Title: a title"), "{d}");
}

/// A file with no text at all has no description, rather than an empty one.
#[test]
fn a_png_without_text_has_no_description() {
    let bytes = plain(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[7],
    );
    let (r, _) = read("notext.png", &bytes);
    assert_eq!(r.info.as_ref().unwrap().description, None);
}

/// An animated PNG opens at its first frame, and the log says the rest are
/// there — a file that silently lost frames would be worse than one that says.
#[test]
fn an_animated_png_reads_its_first_frame_and_says_so() {
    let mut out = Vec::new();
    {
        let mut enc = ::png::Encoder::new(&mut out, 2, 1);
        enc.set_color(::png::ColorType::Grayscale);
        enc.set_depth(::png::BitDepth::Eight);
        enc.set_animated(2, 0).unwrap();
        let mut w = enc.write_header().expect("header");
        w.write_image_data(&[1, 2]).expect("frame 1");
        w.write_image_data(&[3, 4]).expect("frame 2");
        w.finish().expect("finish");
    }
    let (r, host) = read("apng.png", &out);

    assert_eq!(r.image.frames, 1);
    assert_eq!(u8s(&r.image.planes[0]), vec![1, 2]);
    assert!(host.said("animated"), "{:?}", host.lines);
}

/// The stack is named after the file it came from, and remembers where it was.
#[test]
fn the_stack_is_named_after_the_file() {
    let bytes = plain(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[1],
    );
    let (r, _) = read("a_named_file.png", &bytes);

    assert_eq!(r.image.name, "a_named_file.png");
    let info = r.info.as_ref().unwrap();
    assert_eq!(info.name, "a_named_file.png");
    assert!(info.path.as_deref().unwrap().ends_with("a_named_file.png"));
}

/// Progress runs forwards and finishes.
#[test]
fn progress_reaches_the_end() {
    let bytes = plain(
        4,
        4,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[0; 16],
    );
    let (_, host) = read("progress.png", &bytes);

    assert_eq!(host.progress.last().copied(), Some(1.0));
    assert!(
        host.progress.windows(2).all(|w| w[1] >= w[0]),
        "{:?}",
        host.progress
    );
}

// ------------------------------------------------------------------ the probe

#[test]
fn the_signature_decides_and_the_extension_only_breaks_ties() {
    let png = plain(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[1],
    );

    // The signature wins wherever it is found.
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.png"), &png),
        Confidence::Certain
    );
    assert_eq!(
        PngImport.probe(std::path::Path::new("misnamed.dat"), &png),
        Confidence::Certain
    );

    // A `.png` that is plainly something else is not this importer's.
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.png"), b"\xff\xd8\xff\xe0JFIF"),
        Confidence::No
    );
    // Nor is a file that is neither.
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.tif"), b"II\x2a\x00"),
        Confidence::No
    );
    // An unreadable head is no evidence either way; the extension still offers.
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.PNG"), b""),
        Confidence::Maybe
    );
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.oir"), b""),
        Confidence::No
    );

    // And a head too short to decide on. The host reads it with a single
    // `read`, which may return fewer bytes than it asked for, so a real PNG
    // can arrive with three bytes of signature — answering `No` to that would
    // leave the file with no importer at all.
    for n in 1..MAGIC.len() {
        assert_eq!(
            PngImport.probe(std::path::Path::new("x.png"), &MAGIC[..n]),
            Confidence::Maybe,
            "{n} bytes of the signature"
        );
    }
    // A short head that is *not* a prefix of the signature is decisive.
    assert_eq!(
        PngImport.probe(std::path::Path::new("x.png"), b"\x89PNQ"),
        Confidence::No
    );
}

// --------------------------------------------------------------- the refusals

/// A file that is not a PNG is refused, rather than opened as something.
#[test]
fn a_file_that_is_not_a_png_is_refused() {
    let path = write("garbage.png", b"this is not a PNG at all, not even close");
    let err = PngImport
        .import(
            &ImportRequest {
                path,
                params: Params::new(),
            },
            &mut Notes::default(),
        )
        .expect_err("should refuse");
    assert!(matches!(err, PluginError::Unsupported(_)), "{err:?}");
}

/// And a file that is not there at all.
#[test]
fn a_missing_file_is_reported_rather_than_panicking() {
    let path = std::env::temp_dir().join("fasttiff_png_import_tests/no_such_file.png");
    let _ = std::fs::remove_file(&path);
    assert!(PngImport
        .import(
            &ImportRequest {
                path,
                params: Params::new(),
            },
            &mut Notes::default(),
        )
        .is_err());
}

// ------------------------------------------------- the text, and what bounds it

/// Text written after the image data is still read.
///
/// The spec puts tEXt/zTXt/iTXt in the "anywhere" class, and a writer that only
/// learns the text once the pixels are out has nowhere else to put it — it is
/// the only layout the `png` crate's own writer can produce. Decoding stops at
/// the end of the image data, so without reading on to the end of the file the
/// description is silently empty for every one of those files.
#[test]
fn text_written_after_the_image_is_still_read() {
    let bytes = encode_trailing(1, 1, &[7], |w| {
        w.write_text_chunk(&::png::text_metadata::TEXtChunk::new(
            "Software",
            "written after IDAT",
        ))
        .unwrap();
    });
    let (r, _) = read("trailing_text.png", &bytes);

    let d = r
        .info
        .as_ref()
        .unwrap()
        .description
        .clone()
        .expect("a description");
    assert!(d.contains("Software: written after IDAT"), "{d}");
}

/// One chunk larger than the budget is clipped, not thrown away.
///
/// The budget exists for a file using text chunks as storage. Dropping the
/// chunk whole loses everything in exactly the case it was written for.
#[test]
fn a_text_chunk_larger_than_the_budget_is_clipped_not_dropped() {
    let huge = "y".repeat(70_000);
    let bytes = encode(
        1,
        1,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[7],
        |e| e.add_text_chunk("Comment".into(), huge.clone()).unwrap(),
    );
    let (r, _) = read("huge_text.png", &bytes);

    let d = r
        .info
        .as_ref()
        .unwrap()
        .description
        .clone()
        .expect("a description");
    assert!(
        d.starts_with("Comment: yyy"),
        "the record was lost: {}",
        &d[..40.min(d.len())]
    );
    // Kept up to the budget, and stopped there. The slack is the keyword and
    // its separator, which are appended without being counted.
    assert!(d.len() >= 64 * 1024, "clipped too hard: {}", d.len());
    assert!(d.len() < 64 * 1024 + 128, "not clipped at all: {}", d.len());
}

/// A file stuffed with compressed text costs a bounded amount of work.
///
/// Every zTXt here inflates to 64 KiB from under a hundred bytes. Inflating
/// them all and keeping the first — which is what a per-chunk bound does — is
/// most of a gigabyte of decompression to produce 64 KiB of description, with
/// the window frozen at 90% for the duration and no way to stop it.
///
/// The length assertion holds whatever the implementation does; the time is
/// what separates stopping from carrying on, and it is the only thing that
/// can. The bound is loose on purpose: the right answer inflates two chunks
/// and takes milliseconds even in a debug build, the wrong one inflates ten
/// thousand and takes minutes, so anything in between is a wide, quiet gap.
#[test]
fn a_file_stuffed_with_compressed_text_stops_at_the_budget() {
    let bytes = with_many_ztxt(10_000);

    let started = std::time::Instant::now();
    let (r, _) = read("many_ztxt.png", &bytes);
    let took = started.elapsed();

    let d = r
        .info
        .as_ref()
        .unwrap()
        .description
        .clone()
        .expect("a description");
    assert!(
        d.len() < 64 * 1024 + 128,
        "the budget did not hold: {}",
        d.len()
    );
    assert!(
        took < std::time::Duration::from_secs(5),
        "inflating every chunk to keep the first: {took:?}"
    );
}

// ------------------------------------------------------------ stopping, and APNG

/// A host that says stop is obeyed, rather than handed a finished stack.
#[test]
fn an_import_the_user_stopped_does_not_produce_a_document() {
    let bytes = plain(
        8,
        8,
        ::png::ColorType::Grayscale,
        ::png::BitDepth::Eight,
        &[0; 64],
    );
    let path = write("cancelled.png", &bytes);
    let err = PngImport
        .import(
            &ImportRequest {
                path,
                params: Params::new(),
            },
            &mut Stops,
        )
        .expect_err("a stopped import must not return a document");
    assert!(err.to_string().contains("cancelled"), "{err}");
}

/// An animated PNG whose image data is a separate still says so.
///
/// APNG allows the image data to sit outside the animation: with no `fcTL`
/// before it, it is the default image and animation frame 1 is a different
/// picture entirely. Showing the default image is right; calling it "the first
/// frame" is not, and would send someone looking for a frame they are not
/// being shown.
#[test]
fn an_animated_png_with_a_separate_default_image_says_which_one_it_read() {
    let mut out = Vec::new();
    {
        let mut enc = ::png::Encoder::new(&mut out, 2, 1);
        enc.set_color(::png::ColorType::Grayscale);
        enc.set_depth(::png::BitDepth::Eight);
        enc.set_animated(2, 0).unwrap();
        enc.set_sep_def_img(true).unwrap();
        let mut w = enc.write_header().expect("header");
        w.write_image_data(&[9, 9]).expect("the default image");
        w.write_image_data(&[1, 2]).expect("frame 1");
        w.write_image_data(&[3, 4]).expect("frame 2");
        w.finish().expect("finish");
    }
    let (r, host) = read("apng_default.png", &out);

    // The still, which is the picture a viewer that does not animate wants.
    assert_eq!(u8s(&r.image.planes[0]), vec![9, 9]);
    assert!(
        host.said("separate default image"),
        "it called the default image a frame: {:?}",
        host.lines
    );
    assert!(!host.said("only its first frame"), "{:?}", host.lines);
}

//! The PNG exporter, checked by decoding the PNG it wrote.
//!
//! The assertion that matters is not "a file appeared" but "the file is the
//! picture". The contrast window is applied differently depending on whether
//! the samples are wide or not (see the module docs), and getting the units of
//! either wrong produces a uniformly black image, a uniformly white one, or —
//! worst — one that merely looks like a different contrast setting. Only
//! reading the pixels back catches that.

use super::*;
use crate::plugins::{describe_view, StackHost};
use crate::document::{Document, Window};
use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fasttiff_plugin_api::{Params, VolumeMode, VolumeView};
use std::io::Cursor;

const W: u32 = 4;
const H: u32 = 2;

fn view() -> VolumeView {
    VolumeView {
        mode: VolumeMode::Mip,
        density: 1.0,
        iso: 0.5,
        eye: [0.0; 3],
        forward: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fasttiff-png-{}-{name}.png", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

/// A one-channel 16-bit stack whose pixels run 0, 8192, 16384, … so the
/// contrast window has something to bite on.
fn ramp_stack() -> Document {
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    let px: Vec<u16> = (0..W * H).map(|i| (i * 8192).min(65535) as u16).collect();
    w.write_frame_u16(&px).expect("frame");
    Document::from_bytes(
        w.finish().expect("finish").into_inner(),
        "ramp.tif".into(),
        false,
    )
    .expect("open")
}

/// Decode a written PNG back to `(width, height, rgb)`.
fn read_png(path: &std::path::Path) -> (u32, u32, Vec<u8>) {
    let file = std::io::BufReader::new(std::fs::File::open(path).expect("the PNG should be there"));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("PNG header");
    let mut buf = vec![0; reader.output_buffer_size().expect("buffer size")];
    let info = reader.next_frame(&mut buf).expect("PNG data");
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

fn export(stack: &Document, path: &std::path::Path) {
    let v = describe_view(stack, 0, false, view());
    let mut host = StackHost::new(stack, v);
    Png.export(
        &ExportRequest {
            path: path.to_path_buf(),
            params: Params::new(),
        },
        &mut host,
    )
    .expect("export");
}

/// An integer channel's window is in the file's own units, the units
/// `read_plane_f32` gives the samples in, and has to be applied here — the
/// decoder does not do it.
#[test]
fn the_contrast_window_on_screen_is_the_contrast_in_the_file() {
    let mut stack = ramp_stack();
    // A window over the lower half of the range: everything at or above 32768
    // should come out white, and the bottom of the ramp black.
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(0.0, 32768.0));
    }
    let path = temp("window");
    export(&stack, &path);

    let (w, h, rgb) = read_png(&path);
    assert_eq!((w, h), (W, H));
    assert_eq!(rgb.len(), (W * H * 3) as usize);
    // Pixel 0 is sample 0 -> black; pixel 4 is 32768 -> the top of the window.
    assert_eq!(&rgb[0..3], &[0, 0, 0], "the low end is not black");
    assert_eq!(
        &rgb[12..15],
        &[255, 255, 255],
        "the top of the window is not white"
    );
    // And the window really is doing something: a pixel above it clamps rather
    // than wrapping round to dark.
    assert_eq!(
        &rgb[15..18],
        &[255, 255, 255],
        "a value above the window wrapped"
    );
    let _ = std::fs::remove_file(&path);
}

/// Widening the window darkens the picture. The point is that the export
/// *tracks* the window rather than baking in one reading of the data.
#[test]
fn a_wider_window_writes_a_darker_image() {
    let path = temp("narrow");
    let mut stack = ramp_stack();
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(0.0, 16384.0));
    }
    export(&stack, &path);
    let (_, _, narrow) = read_png(&path);

    let path2 = temp("wide");
    let mut stack = ramp_stack();
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(0.0, 65535.0));
    }
    export(&stack, &path2);
    let (_, _, wide) = read_png(&path2);

    let sum = |v: &[u8]| v.iter().map(|&b| b as u32).sum::<u32>();
    assert!(
        sum(&wide) < sum(&narrow),
        "the window was ignored: {} vs {}",
        sum(&wide),
        sum(&narrow)
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&path2);
}

/// A channel's LUT is what makes the export a picture rather than a grey plate,
/// and it has to be the LUT the window is showing.
#[test]
fn each_channel_goes_through_the_lut_it_is_displayed_with() {
    let mut stack = ramp_stack();
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(0.0, 65535.0));
    }
    // Pure green: entry i is (0, i, 0).
    let mut green = [[0u8; 3]; 256];
    for (i, e) in green.iter_mut().enumerate() {
        *e = [0, i as u8, 0];
    }
    stack.display.set_luts(vec![green]);

    let path = temp("lut");
    export(&stack, &path);
    let (_, _, rgb) = read_png(&path);
    for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        assert_eq!(px[0], 0, "pixel {i} has red in a green-only LUT");
        assert_eq!(px[2], 0, "pixel {i} has blue in a green-only LUT");
    }
    assert!(
        rgb.as_chunks::<3>().0.iter().any(|px| px[1] > 200),
        "nothing is bright, so the LUT was not reached at all"
    );
    let _ = std::fs::remove_file(&path);
}

/// A disabled channel is not on screen, so it is not in the file either.
#[test]
fn a_channel_that_is_switched_off_is_not_composited() {
    let mut stack = ramp_stack();
    for c in 0..stack.display.channel_count() {
        stack.display.set_window(c, Window::new(0.0, 65535.0));
        stack.display.set_enabled(c, false);
    }
    let path = temp("disabled");
    export(&stack, &path);
    let (_, _, rgb) = read_png(&path);
    assert!(
        rgb.iter().all(|&b| b == 0),
        "a channel that is not shown reached the file"
    );
    let _ = std::fs::remove_file(&path);
}

/// A one-channel stack of `T` samples `0, 1, 2, …` (one per pixel) with
/// `window` set on it, exported and read back: the grey of each pixel.
fn export_ramp_of(sample: SampleType, window: Window) -> Vec<u8> {
    let opts = WriterOptions::new(W, H, sample).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    let n = (W * H) as usize;
    match sample {
        SampleType::U8 => w.write_frame_u8(&(0..n as u8).collect::<Vec<_>>()),
        SampleType::I16 => w.write_frame_bytes(
            &(0..n as i16)
                .flat_map(|v| (v - 4).to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    }
    .expect("frame");
    let bytes = w.finish().expect("finish").into_inner();
    let mut stack = Document::from_bytes(bytes, "ramp.tif".into(), false).expect("open");
    stack.display.set_window(0, window);
    stack.display.set_luts(vec![grayscale_ramp()]);
    let path = temp(&format!("{sample:?}"));
    export(&stack, &path);
    let (_, _, rgb) = read_png(&path);
    let _ = std::fs::remove_file(&path);
    rgb.as_chunks::<3>().0.iter().map(|px| px[0]).collect()
}

/// An 8-bit channel's window reaches a plugin as the 8-bit values it is, and
/// the export applies it to the 8-bit samples: a window of 0..4 makes sample 2
/// mid-grey and 4 and up white. Handed the window on the 16-bit scale, as the
/// view once did, every pixel here would be all but black.
#[test]
fn an_8_bit_window_is_applied_in_8_bit_units() {
    let grey = export_ramp_of(SampleType::U8, Window::new(0.0, 4.0));
    assert_eq!(grey, [0, 64, 128, 191, 255, 255, 255, 255]);
}

/// A signed 16-bit window is in the signed values, not offset into unsigned:
/// samples run -4..=3 here and a window of -4..0 puts -2 at mid-grey.
#[test]
fn a_signed_16_bit_window_is_applied_in_signed_units() {
    let grey = export_ramp_of(SampleType::I16, Window::new(-4.0, 0.0));
    assert_eq!(grey, [0, 64, 128, 191, 255, 255, 255, 255]);
}

/// The format it offers is what the Save-as dialog will show.
#[test]
fn it_offers_png() {
    let types = Png.file_types();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].extensions, vec!["png".to_string()]);
    assert!(types[0].matches(std::path::Path::new("figure.PNG")), "case");
    assert!(!types[0].matches(std::path::Path::new("figure.tif")));
}

//! The PNG exporter, checked by decoding the PNG it wrote.
//!
//! The assertion that matters is not "a file appeared" but "the file is the
//! picture". The contrast window is applied in different units depending on
//! whether the samples are float or integer (see the module docs), and getting
//! that backwards produces a uniformly black image, a uniformly white one, or —
//! worst — one that merely looks like a different contrast setting. Only
//! reading the pixels back catches that.

use super::*;
use crate::plugins::{describe_view, StackHost};
use crate::stack::Stack;
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
    let p = std::env::temp_dir().join(format!("fasttiff-png-{name}.png"));
    let _ = std::fs::remove_file(&p);
    p
}

/// A one-channel 16-bit stack whose pixels run 0, 8192, 16384, … so the
/// contrast window has something to bite on.
fn ramp_stack() -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(StackMetaWrite::new(1, 1));
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).expect("writer");
    let px: Vec<u16> = (0..W * H).map(|i| (i * 8192).min(65535) as u16).collect();
    w.write_frame_u16(&px).expect("frame");
    Stack::from_bytes(
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

fn export(stack: &Stack, path: &std::path::Path) {
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

/// An integer channel's window is in the same `0..65535` space the samples come
/// back in, and has to be applied here — the decoder does not do it.
#[test]
fn the_contrast_window_on_screen_is_the_contrast_in_the_file() {
    let mut stack = ramp_stack();
    // A window over the lower half of the range: everything at or above 32768
    // should come out white, and the bottom of the ramp black.
    for s in &mut stack.display.settings {
        s.min = 0.0;
        s.max = 32768.0;
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
    for s in &mut stack.display.settings {
        s.min = 0.0;
        s.max = 16384.0;
    }
    export(&stack, &path);
    let (_, _, narrow) = read_png(&path);

    let path2 = temp("wide");
    let mut stack = ramp_stack();
    for s in &mut stack.display.settings {
        s.min = 0.0;
        s.max = 65535.0;
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
    for s in &mut stack.display.settings {
        s.min = 0.0;
        s.max = 65535.0;
    }
    // Pure green: entry i is (0, i, 0).
    let mut green = [[0u8; 3]; 256];
    for (i, e) in green.iter_mut().enumerate() {
        *e = [0, i as u8, 0];
    }
    stack.display.luts = vec![green];

    let path = temp("lut");
    export(&stack, &path);
    let (_, _, rgb) = read_png(&path);
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        assert_eq!(px[0], 0, "pixel {i} has red in a green-only LUT");
        assert_eq!(px[2], 0, "pixel {i} has blue in a green-only LUT");
    }
    assert!(
        rgb.chunks_exact(3).any(|px| px[1] > 200),
        "nothing is bright, so the LUT was not reached at all"
    );
    let _ = std::fs::remove_file(&path);
}

/// A disabled channel is not on screen, so it is not in the file either.
#[test]
fn a_channel_that_is_switched_off_is_not_composited() {
    let mut stack = ramp_stack();
    for s in &mut stack.display.settings {
        s.min = 0.0;
        s.max = 65535.0;
        s.enabled = false;
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

/// The format it offers is what the Save-as dialog will show.
#[test]
fn it_offers_png() {
    let types = Png.file_types();
    assert_eq!(types.len(), 1);
    assert_eq!(types[0].extensions, vec!["png".to_string()]);
    assert!(types[0].matches(std::path::Path::new("figure.PNG")), "case");
    assert!(!types[0].matches(std::path::Path::new("figure.tif")));
}

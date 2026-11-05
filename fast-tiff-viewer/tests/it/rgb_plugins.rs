//! A plugin run on an RGB stack sees, and gives back, all three colours.
//!
//! The viewer counts chunky RGB as one channel, its colours being samples of
//! one IFD. That count used to reach plugins too, so Slice Order Invert (and
//! every tool built the same way) read red alone and returned a one-channel
//! grayscale stack.
#![cfg(feature = "plugin-stack-tools")]

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::plugins::{builtin, describe_view, StackHost};
use fast_tiff_viewer::Stack;
use fasttiff_plugin_api::{
    DisplayMode, HostContext, Outcome, Params, Plane, PlaneData, Plugin, VolumeMode, VolumeView,
};
use std::io::Cursor;

const W: u32 = 8;
const H: u32 = 6;
const FRAMES: u8 = 5;

/// Red, green and blue each their own ramp, and a different offset per frame.
fn sample(frame: u8, i: usize, s: usize) -> u8 {
    (i as u8)
        .wrapping_mul(3)
        .wrapping_add(60 * s as u8)
        .wrapping_add(7 * frame)
}

fn rgb_stack() -> Stack {
    let opts = WriterOptions::new(W, H, SampleType::U8).samples_per_pixel(3);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..FRAMES {
        let px: Vec<u8> = (0..(W * H) as usize)
            .flat_map(|i| (0..3).map(move |s| sample(f, i, s)))
            .collect();
        w.write_frame_bytes(&px).unwrap();
    }
    Stack::from_bytes(w.finish().unwrap().into_inner(), "rgb.tif".into(), false).unwrap()
}

fn host(s: &Stack) -> StackHost {
    let v = VolumeView {
        mode: VolumeMode::Mip,
        density: 1.0,
        iso: 0.5,
        eye: [0.0; 3],
        forward: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
    };
    StackHost::new(s, describe_view(s, 0, false, v))
}

#[test]
fn a_plugin_reads_every_colour_of_an_rgb_stack() {
    let s = rgb_stack();
    assert!(s.display.rgb, "the fixture is chunky RGB");
    let mut h = host(&s);
    assert_eq!(h.image().channels, 3);
    assert_eq!(h.stack_info().mode, DisplayMode::Composite);
    let mut buf = Vec::new();
    for c in 0..3 {
        h.read_plane_f32(Plane::new(c, 0, 2), &mut buf).unwrap();
        assert_eq!(buf[5], sample(2, 5, c) as f32, "channel {c}");
    }
}

#[test]
fn slice_order_invert_keeps_the_colours_and_their_tables() {
    let s = rgb_stack();
    let luts = s.display.luts.clone();
    let mut h = host(&s);
    let Outcome::NewDocument(img) = builtin::SliceOrderInvert
        .run(&mut h, &Params::new())
        .unwrap()
    else {
        panic!("no image");
    };
    assert_eq!((img.channels, img.frames), (3, FRAMES as usize));
    // The first frame out is the last frame in, every colour of it.
    for c in 0..3 {
        let PlaneData::U8(px) = &img.planes[c] else {
            panic!("plane {c} changed type");
        };
        assert_eq!(px[5], sample(FRAMES - 1, 5, c), "channel {c}");
    }

    // And it opens as three coloured channels, as the source shows.
    let bytes =
        fast_tiff_viewer::plugins::to_tiff_bytes_reporting(&img, None, &luts, &mut |_| true)
            .unwrap()
            .unwrap();
    let out = Stack::from_bytes(bytes, "out.tif".into(), false).unwrap();
    assert_eq!(out.display.dims.channels, 3);
    assert_eq!(out.display.mode, fast_tiff_lib::DisplayMode::Composite);
    assert_eq!(out.display.luts, luts);
}

/// A multi-channel stack was already right; it stays right.
#[test]
fn a_multichannel_stack_still_crosses_whole() {
    let meta = StackMetaWrite::new(2, 1)
        .mode(fast_tiff_lib::DisplayMode::Composite)
        .channel_lut(fast_tiff_lib::color_ramp_lut([255, 0, 255]))
        .channel_lut(fast_tiff_lib::color_ramp_lut([0, 255, 0]));
    let opts = WriterOptions::new(W, H, SampleType::U16).metadata(meta);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for i in 0..10u16 {
        let px: Vec<u8> = (0..(W * H) as u16)
            .flat_map(|v| (v + 100 * i).to_le_bytes())
            .collect();
        w.write_frame_bytes(&px).unwrap();
    }
    let s = Stack::from_bytes(w.finish().unwrap().into_inner(), "c2.tif".into(), false).unwrap();
    let luts = s.display.luts.clone();
    let Outcome::NewDocument(img) = builtin::SliceOrderInvert
        .run(&mut host(&s), &Params::new())
        .unwrap()
    else {
        panic!("no image");
    };
    assert_eq!(img.channels, 2);
    let bytes =
        fast_tiff_viewer::plugins::to_tiff_bytes_reporting(&img, None, &luts, &mut |_| true)
            .unwrap()
            .unwrap();
    let out = Stack::from_bytes(bytes, "out.tif".into(), false).unwrap();
    assert_eq!(out.display.dims.channels, 2);
    assert_eq!(out.display.luts, luts);
}

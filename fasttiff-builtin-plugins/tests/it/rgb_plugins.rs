//! A plugin run on an RGB stack sees, and gives back, all three colours.
//!
//! The viewer counts chunky RGB as one channel, its colours being samples of
//! one IFD. That count used to reach plugins too, so Slice Order Invert (and
//! every tool built the same way) read red alone and returned a one-channel
//! grayscale stack.
#![cfg(feature = "plugin-image-tools")]

use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use fast_tiff_viewer::plugins::{describe_view, Registry, ResultDest, StackHost};
use fast_tiff_viewer::Document;
use fasttiff_builtin_plugins as builtin;
use fasttiff_plugin_api::{
    DisplayMode, HostContext, ImageResult, Outcome, ParamKind, ParamValue, Params, PixelType,
    Plane, PlaneData, Plugin, PluginError, StreamTarget, VolumeMode, VolumeView,
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

fn rgb_stack() -> Document {
    let opts = WriterOptions::new(W, H, SampleType::U8).samples_per_pixel(3);
    let mut w = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
    for f in 0..FRAMES {
        let px: Vec<u8> = (0..(W * H) as usize)
            .flat_map(|i| (0..3).map(move |s| sample(f, i, s)))
            .collect();
        w.write_frame_bytes(&px).unwrap();
    }
    Document::from_bytes(w.finish().unwrap().into_inner(), "rgb.tif".into(), false).unwrap()
}

fn host(s: &Document) -> StackHost {
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
    assert!(
        s.display().layout().shown_as_rgb(),
        "the fixture is chunky RGB"
    );
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
    let luts = s.display().luts().to_vec();
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
    let out = Document::from_bytes(bytes, "out.tif".into(), false).unwrap();
    assert_eq!(out.display().dims().channels, 3);
    assert_eq!(out.display().mode(), fast_tiff_lib::DisplayMode::Composite);
    assert_eq!(out.display().luts(), &luts[..]);
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
    let s = Document::from_bytes(w.finish().unwrap().into_inner(), "c2.tif".into(), false).unwrap();
    let luts = s.display().luts().to_vec();
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
    let out = Document::from_bytes(bytes, "out.tif".into(), false).unwrap();
    assert_eq!(out.display().dims().channels, 2);
    assert_eq!(out.display().luts(), &luts[..]);
}

#[test]
fn split_channels_makes_one_coloured_stack_per_source_channel() {
    let s = rgb_stack();
    let luts = s.display().luts().to_vec();
    let Outcome::NewDocuments(documents) = builtin::SplitChannels
        .run(&mut host(&s), &Params::new())
        .unwrap()
    else {
        panic!("expected separate documents")
    };
    assert_eq!(documents.len(), 3);
    for (c, document) in documents.iter().enumerate() {
        let image = &document.image;
        assert_eq!(image.name, format!("rgb_ch{}", c + 1));
        assert_eq!((image.channels, image.frames), (1, FRAMES as usize));
        assert_eq!(document.luts, vec![luts[c]], "channel {c}'s LUT");
        assert_eq!(image.metadata.as_ref().unwrap().mode, DisplayMode::Color);
        let PlaneData::U8(px) = &image.planes[2] else {
            panic!("channel {c} changed sample type")
        };
        assert_eq!(px[5], sample(2, 5, c), "channel {c}'s frame 3");

        // The explicit table is not merely carried in memory: it reaches the
        // TIFF this document opens from.
        let bytes = fast_tiff_viewer::plugins::to_tiff_bytes_reporting(
            image,
            None,
            &document.luts,
            &mut |_| true,
        )
        .unwrap()
        .unwrap();
        let reopened = Document::from_bytes(bytes, image.name.clone().into(), false).unwrap();
        assert_eq!(reopened.display().luts(), &document.luts[..]);
    }
}

#[test]
fn split_channels_refuses_an_image_without_a_channel_dimension() {
    let image = ImageResult {
        width: W,
        height: H,
        channels: 1,
        slices: 1,
        frames: 1,
        pixel_type: PixelType::U8,
        planes: vec![PlaneData::U8(vec![0; (W * H) as usize])],
        channel_colors: Vec::new(),
        metadata: None,
        name: "gray".into(),
    };
    let s = fast_tiff_viewer::plugins::to_stack(&image, None, false).unwrap();
    assert_eq!(
        builtin::SplitChannels.run(&mut host(&s), &Params::new()),
        Err(PluginError::unsupported(
            "this image has no channel dimension to split"
        ))
    );
}

/// Exercise the registry, both result TIFFs, and the source's original Z/T
/// declarations (a Z-only stack is navigated through T in the viewer).
#[test]
fn split_stack_preserves_planes_axes_and_luts_in_both_files() {
    for (channels, nz, nt, split_z, first) in [
        (2, 1, 5, false, 2),
        (2, 5, 1, true, 4),
        (2, 4, 5, true, 3),
        (7, 4, 5, false, 3),
    ] {
        let luts: Vec<_> = (0..channels)
            .map(|c| std::array::from_fn(|i| [i as u8, (255 - i) as u8, (c + i / 3) as u8]))
            .collect();
        let mut meta = StackMetaWrite::new(channels, nz)
            .mode(fast_tiff_lib::DisplayMode::Composite)
            .unit("micron")
            .spacing(2.5)
            .pixel_size(0.25, 0.5)
            .frame_interval_s(0.75)
            .calibration(1.5, 2.0)
            .trailing("Acquisition=blue".to_owned());
        for lut in &luts {
            meta = meta.channel_lut(*lut);
        }
        let opts = WriterOptions::new(2, 1, SampleType::U16).metadata(meta);
        let mut writer = TiffWriter::new(Cursor::new(Vec::new()), opts).unwrap();
        for i in 0..(channels * nz * nt) as u16 {
            writer.write_frame_u16(&[i, 1000 + i]).unwrap();
        }
        let source = Document::from_bytes(
            writer.finish().unwrap().into_inner(),
            "cells.tif".into(),
            false,
        )
        .unwrap();
        let mut registry = Registry::new(builtin::register);
        let index = registry
            .find("dev.fasttiff.stacktools.split-stack")
            .unwrap();
        let plugin = &mut registry.get_mut(index).unwrap().plugin;
        let mut params = Params::new();
        let both = nz > 1 && nt > 1;
        if both {
            params.set("axis", ParamValue::Choice(usize::from(split_z)));
        }
        let dialog = plugin
            .dialog(&host(&source).with_pending_params(params.clone()))
            .unwrap();
        assert_eq!(dialog.iter().any(|d| d.key == "axis"), both);
        let position = dialog.iter().find(|d| d.key == "first").unwrap();
        let n = if split_z { nz } else { nt } as i64;
        assert_eq!(
            position.label,
            if split_z {
                "First slice in second stack"
            } else {
                "First frame in second stack"
            }
        );
        assert_eq!(
            position.kind,
            ParamKind::Int {
                min: 2,
                max: n - 1,
                default: n / 2 + 1
            }
        );
        params.set("first", ParamValue::Int(first as i64));
        let mut streamed = host(&source).streaming_to(ResultDest::Memory);
        assert_eq!(
            plugin.run(&mut streamed, &params).unwrap(),
            Outcome::Streamed {
                target: StreamTarget::NewDocuments
            }
        );
        let results = streamed.take_documents();
        assert_eq!(results.len(), 2);
        for (part, result) in results.iter().enumerate() {
            assert_eq!(result.name(), format!("cells-part{}", part + 1));
            let range = if part == 0 {
                0..first - 1
            } else {
                first - 1..n as usize
            };
            let (times, slices) = if split_z {
                (0..nt, range)
            } else {
                (range, 0..nz)
            };
            let out =
                Document::from_bytes(result.bytes().unwrap().to_vec(), "out.tif".into(), false)
                    .unwrap();
            let m = &out.tiff().meta;
            assert_eq!(
                (m.channels, m.slices, m.frames),
                (channels, slices.len(), times.len())
            );
            assert_eq!(m.spacing, Some(2.5));
            assert_eq!(m.pixel_width, Some(0.25));
            assert_eq!(m.pixel_height, Some(0.5));
            assert_eq!(m.frame_interval_s, Some(0.75));
            assert_eq!(m.calibration, Some((1.5, 2.0)));
            assert_eq!(m.unit.as_deref(), Some("micron"));
            assert!(out
                .tiff()
                .description
                .as_ref()
                .unwrap()
                .contains("Acquisition=blue"));
            assert_eq!(
                m.channel_display.iter().map(|d| d.lut).collect::<Vec<_>>(),
                luts
            );
            let image = result.to_image().unwrap();
            let mut planes = image.planes.iter();
            for t in times {
                for z in slices.clone() {
                    for c in 0..channels {
                        let i = ((t * nz + z) * channels + c) as u16;
                        assert_eq!(planes.next(), Some(&PlaneData::U16(vec![i, 1000 + i])));
                    }
                }
            }
            assert!(planes.next().is_none());
        }
    }
}

#[test]
fn split_stack_refuses_cuts_outside_the_dialog_range() {
    let source = rgb_stack();
    let mut h = host(&source);
    for first in [1, i64::from(FRAMES)] {
        let mut params = Params::new();
        params.set("first", ParamValue::Int(first));
        assert!(builtin::SplitStack.run(&mut h, &params).is_err());
    }
}

/// The real app uses the ABI registry and a streaming host. This makes sure a
/// multi-document result closes every TIFF before the next channel begins,
/// rather than leaving all but the final channel in the sink.
#[test]
fn split_channels_streams_every_coloured_document_through_the_boundary() {
    let s = rgb_stack();
    let luts = s.display().luts().to_vec();
    let mut registry = Registry::new(builtin::register);
    let index = registry
        .find("dev.fasttiff.stacktools.split-channels")
        .expect("Split Channels is installed");
    let mut streamed = host(&s).streaming_to(ResultDest::Memory);
    let outcome = registry
        .get_mut(index)
        .unwrap()
        .plugin
        .run(&mut streamed, &Params::new())
        .unwrap();
    assert_eq!(
        outcome,
        Outcome::Streamed {
            target: StreamTarget::NewDocuments
        }
    );

    let results = streamed.take_documents();
    assert_eq!(results.len(), 3);
    for (c, result) in results.iter().enumerate() {
        assert_eq!(result.name(), format!("rgb_ch{}", c + 1));
        let opened = Document::from_bytes(
            result.bytes().unwrap().to_vec(),
            format!("out_ch{}.tif", c + 1).into(),
            false,
        )
        .unwrap();
        assert_eq!(opened.display().luts(), &[luts[c]], "channel {c}'s LUT");
        let image = result.to_image().unwrap();
        let PlaneData::U8(px) = &image.planes[2] else {
            panic!("channel {c} changed sample type")
        };
        assert_eq!(px[5], sample(2, 5, c));
    }
}

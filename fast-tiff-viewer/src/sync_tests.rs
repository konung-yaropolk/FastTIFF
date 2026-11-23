//! [`Viewer::sync`] against a [`Recorder`], with no GPU: what reaches the
//! renderer, when, and in what order.
//!
//! The sync used to take the one backend the crate was built with, so it
//! could only run with a GPU and had no tests. It is generic over
//! scivis-render's renderer traits now, and the recorder stands in for a
//! backend. These pin what it did then: they were written against the sync
//! as it was, made generic and nothing more.
//!
//! The stacks are made so that a plane's first sample says which plane of the
//! file it is (`plane * 1000 + 7` in a 16-bit one; [`value`] for the other
//! kinds), and so which frame and channel an upload, or a volume's first
//! slice, carried. Most are 16-bit, two channels by three timepoints; the
//! 8-bit, signed and float stacks, and one through Z and time, are for what
//! the sync does differently for them.

use super::*;
use crate::notice::NoticeKind;
use crate::recorder::{Call, Recorder};
use crate::roi::Roi;
use crate::units::Window;
use crate::{FrameInputs, Viewer};
use fast_tiff_lib::{SampleType, StackMetaWrite, TiffWriter, WriterOptions};
use scivis_render::pack::{PackedVolume, TexelFormat, VolumeLayout};
use scivis_render::uniforms::{ChannelParams, CompositeParams};
use scivis_render::{VolumeCaps, VolumeInterp, VolumeParams, VolumeUploadError};
use std::time::{Duration, Instant};

const W: u32 = 32;
const H: u32 = 16;
const MB: usize = 1 << 20;

/// The first sample of the file's plane `p` of a 16-bit stack, and every
/// other one of it.
fn sample(p: u32) -> f64 {
    value(SampleType::U16, p)
}

/// The first sample of plane `p` of a stack of `ty`, and of a plane of the
/// integer types every other one too: different for every plane, within the
/// type, and negative for the first few planes of a signed one.
fn value(ty: SampleType, p: u32) -> f64 {
    match ty {
        SampleType::U8 => (p * 20 + 7) as f64,
        SampleType::I16 => p as f64 * 1000.0 - 2993.0,
        SampleType::F32 => p as f64 * 1000.0 + 7.25,
        SampleType::U16 => (p * 1000 + 7) as f64,
        _ => unimplemented!("{ty:?}"),
    }
}

/// How much a float plane's last sample is above the rest of it. A float
/// channel's 3D volume is normalized over the range of the first plane of it
/// that has any contrast, so this gives the channel a span the tests know:
/// its first plane's value to that plus this.
const FLOAT_RISE: f64 = 1000.0;

/// `channels` channels by `slices` by `frames` of `ty` samples, `W` x `H`,
/// in file order as ImageJ writes them: channels fastest, then slices.
fn stack_of(ty: SampleType, channels: u32, slices: u32, frames: u32) -> Vec<u8> {
    let opts = WriterOptions::new(W, H, ty)
        .metadata(StackMetaWrite::new(channels as usize, slices as usize));
    let mut w = TiffWriter::new(std::io::Cursor::new(Vec::new()), opts).unwrap();
    let n = (W * H) as usize;
    for p in 0..channels * slices * frames {
        let v = value(ty, p);
        match ty {
            SampleType::U8 => w.write_frame_u8(&vec![v as u8; n]),
            SampleType::U16 => w.write_frame_u16(&vec![v as u16; n]),
            SampleType::I16 => {
                let bytes: Vec<u8> = (0..n).flat_map(|_| (v as i16).to_le_bytes()).collect();
                w.write_frame_bytes(&bytes)
            }
            SampleType::F32 => {
                let mut plane = vec![v as f32; n];
                plane[n - 1] = (v + FLOAT_RISE) as f32;
                w.write_frame_f32(&plane)
            }
            _ => unimplemented!("{ty:?}"),
        }
        .unwrap();
    }
    w.finish().unwrap().into_inner()
}

/// A viewer with a two-channel, three-timepoint 16-bit stack open.
fn opened() -> Viewer {
    opened_as(SampleType::U16, 2, 1, 3)
}

/// A viewer with a stack of [`stack_of`] open.
fn opened_as(ty: SampleType, channels: u32, slices: u32, frames: u32) -> Viewer {
    let mut viewer = Viewer::new();
    viewer
        .load_bytes(stack_of(ty, channels, slices, frames), "stack.tif".into())
        .unwrap();
    let dims = viewer.document().unwrap().display.dims();
    assert_eq!(
        (dims.channels, dims.slices, dims.frames),
        (channels as usize, slices as usize, frames as usize)
    );
    viewer
}

/// What each call was, in order, for a test about the order.
fn kinds(calls: &[Call]) -> Vec<&'static str> {
    calls
        .iter()
        .map(|c| match c {
            Call::EnsureSize { .. } => "size",
            Call::Upload { .. } => "upload",
            Call::Lut { .. } => "lut",
            Call::Params(_) => "params",
            Call::ReleaseVolume => "release volume",
            Call::UploadVolume { .. } => "upload volume",
            Call::VolumeInterp(_) => "volume interp",
            Call::VolumeTarget { .. } => "volume target",
            Call::VolumeParams(_) => "volume params",
        })
        .collect()
}

/// The planes uploaded, as `(channel, first sample)`.
fn uploads(calls: &[Call]) -> Vec<(usize, f64)> {
    calls
        .iter()
        .filter_map(|c| match c {
            Call::Upload { channel, first, .. } => Some((*channel, *first)),
            _ => None,
        })
        .collect()
}

/// The last thing the composite's shader was given.
fn params(calls: &[Call]) -> CompositeParams {
    calls
        .iter()
        .rev()
        .find_map(|c| match c {
            Call::Params(p) => Some(*p),
            _ => None,
        })
        .expect("the shader was given its params")
}

/// The last thing the ray-march was given.
fn volume_params(calls: &[Call]) -> VolumeParams {
    calls
        .iter()
        .rev()
        .find_map(|c| match c {
            Call::VolumeParams(p) => Some(*p),
            _ => None,
        })
        .expect("the ray-march was given its params")
}

/// The shape, format and origin voxels of the one volume uploaded.
fn the_volume(calls: &[Call]) -> ((u32, u32, u32), usize, TexelFormat, Vec<f32>) {
    let mut volumes = calls.iter().filter_map(|c| match c {
        Call::UploadVolume {
            dims,
            channels,
            format,
            corner,
            ..
        } => Some((*dims, *channels, *format, corner.clone())),
        _ => None,
    });
    let volume = volumes.next().expect("a volume was uploaded");
    assert!(volumes.next().is_none(), "one volume, not more");
    volume
}

/// That `got` is `want` to within a unorm16's rounding.
fn assert_near(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} for {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "{what}: {got:?} for {want:?}");
    }
}

fn channel(min: f32, max: f32, on: bool) -> ChannelParams {
    ChannelParams {
        min_max: [min, max],
        enabled: if on { 1.0 } else { 0.0 },
        is_float: 0.0,
    }
}

/// Sync until `done` says so, and every call made on the way: a volume is
/// built on a thread of its own where there are threads, and lands a few
/// frames later. Without them it is built in the first sync.
fn sync_until(
    viewer: &mut Viewer,
    r: &mut Recorder,
    inputs: &FrameInputs,
    mut done: impl FnMut(&Viewer, &[Call]) -> bool,
) -> Vec<Call> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut calls = Vec::new();
    loop {
        viewer.sync(r, inputs);
        calls.extend(r.take());
        if done(viewer, &calls) {
            return calls;
        }
        assert!(Instant::now() < deadline, "never came: {:?}", kinds(&calls));
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn volume_uploads(calls: &[Call]) -> Vec<(usize, TexelFormat)> {
    calls
        .iter()
        .filter_map(|c| match c {
            Call::UploadVolume { bytes, format, .. } => Some((*bytes, *format)),
            _ => None,
        })
        .collect()
}

fn uploaded_a_volume(_: &Viewer, calls: &[Call]) -> bool {
    !volume_uploads(calls).is_empty()
}

/// The first sync of a document gives back whatever volume the renderer
/// held, sizes the textures to the frame, uploads every LUT and then the
/// frame's planes, and last gives the shader the windows and the view.
/// The double refuses a texture past the 2D limit it reports, as both
/// backends do, so a sync that sized one would fail every test here even with
/// no upload after it.
#[test]
#[should_panic(expected = "at most 64 x 64")]
fn the_recorder_refuses_a_texture_past_the_limit_it_reports() {
    use scivis_render::ImageRenderer as _;
    let mut gpu = Recorder {
        max_2d: 64,
        ..Recorder::default()
    };
    gpu.ensure_size(64, 64, &[ChannelKind::Int16]);
    gpu.ensure_size(65, 64, &[ChannelKind::Int16]);
}

#[test]
fn changing_depth_uploads_and_bins_the_selected_slice_at_each_timepoint() {
    let mut viewer = opened_as(SampleType::U16, 2, 3, 2);
    let mut recorder = Recorder::default();
    let inputs = FrameInputs {
        histograms: true,
        ..Default::default()
    };
    let mut version = 0;
    for (t, z, first_plane) in [(0, 0, 0), (0, 2, 4), (1, 2, 10), (1, 0, 6)] {
        let mut doc = viewer.document_mut().unwrap();
        doc.set_frame(t);
        doc.set_slice(z);
        viewer.sync(&mut recorder, &inputs);
        assert_eq!(
            uploads(&recorder.take()),
            [(0, sample(first_plane)), (1, sample(first_plane + 1))]
        );
        let new_version = viewer.document().unwrap().histograms().version();
        assert!(new_version > version);
        version = new_version;
    }
}

#[test]
fn a_first_sync_sizes_colours_and_fills_the_textures_then_sets_the_shader() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    let outcome = viewer.sync(&mut r, &FrameInputs::default());
    let calls = r.take();
    assert_eq!(
        kinds(&calls),
        [
            "release volume",
            "size",
            "lut",
            "lut",
            "upload",
            "upload",
            "params"
        ]
    );
    let Call::EnsureSize {
        width,
        height,
        kinds: sized,
    } = &calls[1]
    else {
        unreachable!()
    };
    assert_eq!(
        (*width, *height, sized.as_slice()),
        (W, H, [ChannelKind::Int16, ChannelKind::Int16].as_slice())
    );
    for call in &calls {
        if let Call::Upload {
            kind,
            width,
            height,
            ..
        } = call
        {
            assert_eq!((*kind, *width, *height), (ChannelKind::Int16, W, H));
        }
    }
    assert_eq!(uploads(&calls), [(0, sample(0)), (1, sample(1))]);
    let p = params(&calls);
    assert_eq!(
        (p.num_channels, p.uv_offset, p.uv_scale),
        (2, [0.0, 0.0], [1.0, 1.0])
    );
    // Each window as the file's own: the stack's samples are one value a
    // plane, so the seeded windows are the ones the display chose.
    let display = &viewer.document().unwrap().display;
    for (c, s) in display.settings().iter().enumerate() {
        assert_eq!(
            p.channels[c],
            channel(s.window.lo, s.window.hi, true),
            "channel {c}"
        );
    }
    assert!(!outcome.needs_repaint);
    assert_eq!(viewer.gpu_stride(), 1);
    assert!(!viewer.over_texture_limit());
    assert_eq!(viewer.resident_window(), Some(Roi::full(W, H)));
}

/// With nothing changed the textures are left as they are: the size is
/// asked for again (a no-op in a backend) and the shader given its params,
/// and nothing is uploaded.
#[test]
fn a_sync_with_nothing_changed_uploads_nothing() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    viewer.sync(&mut r, &FrameInputs::default());
    r.take();
    viewer.sync(&mut r, &FrameInputs::default());
    assert_eq!(kinds(&r.take()), ["size", "params"]);
}

/// Another frame is its planes and nothing else: the LUTs stay.
#[test]
fn another_frame_is_its_planes_alone() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    viewer.sync(&mut r, &FrameInputs::default());
    r.take();
    viewer.document_mut().unwrap().set_frame(2);
    viewer.sync(&mut r, &FrameInputs::default());
    let calls = r.take();
    assert_eq!(kinds(&calls), ["size", "upload", "upload", "params"]);
    assert_eq!(uploads(&calls), [(0, sample(4)), (1, sample(5))]);
}

/// A channel switched off is not uploaded, and the shader is told it is
/// off; switched on again, it is uploaded with the rest of the frame, since
/// its texture holds nothing of it.
#[test]
fn a_channel_switched_off_is_left_out_until_it_is_on_again() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    viewer.sync(&mut r, &FrameInputs::default());
    r.take();

    let mut doc = viewer.document_mut().unwrap();
    assert!(doc.display_mut().set_enabled(1, false));
    viewer.sync(&mut r, &FrameInputs::default());
    let calls = r.take();
    assert_eq!(uploads(&calls), [(0, sample(0))]);
    let p = params(&calls);
    assert_eq!((p.channels[0].enabled, p.channels[1].enabled), (1.0, 0.0));
    assert_eq!(p.num_channels, 2);

    // Another frame while it is off: still only channel 0.
    viewer.document_mut().unwrap().set_frame(1);
    viewer.sync(&mut r, &FrameInputs::default());
    assert_eq!(uploads(&r.take()), [(0, sample(2))]);

    let mut doc = viewer.document_mut().unwrap();
    assert!(doc.display_mut().set_enabled(1, true));
    viewer.sync(&mut r, &FrameInputs::default());
    let calls = r.take();
    assert_eq!(uploads(&calls), [(0, sample(2)), (1, sample(3))]);
    assert_eq!(params(&calls).channels[1].enabled, 1.0);
}

/// New LUTs are uploaded, every channel's, and nothing else is.
#[test]
fn new_luts_are_uploaded_without_the_frame() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    viewer.sync(&mut r, &FrameInputs::default());
    let first: Vec<Call> = r.take();
    let tops = |calls: &[Call]| -> Vec<(usize, [u8; 3])> {
        calls
            .iter()
            .filter_map(|c| match c {
                Call::Lut { channel, top } => Some((*channel, *top)),
                _ => None,
            })
            .collect()
    };
    assert_eq!(tops(&first), [(0, [255; 3]), (1, [255; 3])], "grays");

    viewer.set_pseudocolor(true);
    viewer.sync(&mut r, &FrameInputs::default());
    let calls = r.take();
    assert_eq!(kinds(&calls), ["size", "lut", "lut", "params"]);
    assert_eq!(
        tops(&calls),
        [(0, [255, 0, 0]), (1, [0, 255, 0])],
        "the standard channel palette"
    );
}

/// A contrast window and the part of the image on screen reach the shader
/// on the next sync, with nothing uploaded for either: a 16-bit channel's
/// window in the file's own units, which its R16Uint texture holds, and the
/// view's UV window as it was asked for, the whole frame being resident.
#[test]
fn a_window_and_the_view_reach_the_shader_alone() {
    let mut viewer = opened();
    let mut r = Recorder::default();
    viewer.sync(&mut r, &FrameInputs::default());
    r.take();

    let mut doc = viewer.document_mut().unwrap();
    assert!(doc.display_mut().set_window(0, Window::new(100.0, 5000.0)));
    let mut inputs = FrameInputs::default();
    inputs.image.uv_offset = [0.25, 0.125];
    inputs.image.uv_scale = [0.5, 0.25];
    viewer.sync(&mut r, &inputs);
    let calls = r.take();
    assert_eq!(kinds(&calls), ["size", "params"]);
    let p = params(&calls);
    assert_eq!(p.channels[0], channel(100.0, 5000.0, true));
    assert_eq!((p.uv_offset, p.uv_scale), ([0.25, 0.125], [0.5, 0.25]));
}

/// A frame wider than the device's 2D textures is not sent whole: the
/// textures are sized to a window of it that fits, every plane uploaded is
/// that size, and the shader is told where the part of the image on screen
/// lies in that window. Zoomed in on the right half here, which fits at full
/// resolution once the window for it lands. On every sync the view is mapped
/// onto the window resident when it ends.
#[test]
fn a_frame_past_the_texture_limit_is_sent_as_a_window_that_fits() {
    let mut viewer = opened();
    let mut r = Recorder {
        max_2d: W / 2,
        ..Recorder::default()
    };
    let mut inputs = FrameInputs::default();
    inputs.image.uv_offset = [0.5, 0.0];
    inputs.image.uv_scale = [0.5, 1.0];
    inputs.image.viewport = crate::PhysicalPx::new((W / 2) as f32, H as f32);
    let (offset, scale) = (inputs.image.uv_offset, inputs.image.uv_scale);
    let mut syncs = 0;
    let calls = sync_until(&mut viewer, &mut r, &inputs, |v, calls| {
        syncs += 1;
        let resident = v.resident_window().expect("a window is resident");
        let p = params(calls);
        assert_eq!(
            (p.uv_offset, p.uv_scale),
            resident.map_uv(W, H, offset, scale),
            "sync {syncs}: the view on {resident:?}"
        );
        resident.stride == 1
    });
    assert!(viewer.over_texture_limit());
    let roi = viewer.resident_window().unwrap();
    assert!(!roi.covers_whole_frame(W, H), "{roi:?}");
    let (tw, th) = roi.texture_size();
    assert!(tw <= W / 2 && th <= W / 2, "{roi:?}");
    let sizes: Vec<(u32, u32)> = calls
        .iter()
        .filter_map(|c| match c {
            Call::EnsureSize { width, height, .. } => Some((*width, *height)),
            _ => None,
        })
        .collect();
    assert_eq!(sizes.last(), Some(&(tw, th)));
    for call in &calls {
        if let Call::Upload { width, height, .. } = call {
            assert!(*width <= W / 2 && *height <= W / 2);
        }
    }
    // Planes of frame 0, of both channels.
    let planes = uploads(&calls);
    assert!(planes.contains(&(0, sample(0))) && planes.contains(&(1, sample(1))));
    assert_eq!(viewer.gpu_stride(), 1);
    // The view asked for, mapped onto the window resident: not what was
    // asked, which is in the whole image's terms.
    let p = params(&calls);
    let mapped = roi.map_uv(W, H, inputs.image.uv_offset, inputs.image.uv_scale);
    assert_eq!((p.uv_offset, p.uv_scale), mapped);
    assert_ne!(mapped, (inputs.image.uv_offset, inputs.image.uv_scale));
}

/// A zoom into a frame past the texture limit keeps the coarse view of all of
/// it on screen while the window for the zoom is built, and on every sync the
/// shader is told where the view lies on the window resident as that sync
/// ends. The finer window lands during a sync that planned against the coarse
/// one (from the worker, or built inline where there are no threads), and
/// mapping from the window planned would draw that frame with the wrong part
/// of the image.
#[test]
fn a_zoom_past_the_texture_limit_maps_every_frame_onto_the_window_held() {
    let mut viewer = opened();
    let mut r = Recorder {
        max_2d: W / 2,
        ..Recorder::default()
    };
    // All of the frame first, at the stride that fits.
    let mut inputs = FrameInputs::default();
    inputs.image.viewport = crate::PhysicalPx::new((W / 2) as f32, (H / 2) as f32);
    sync_until(&mut viewer, &mut r, &inputs, |v, _| {
        v.resident_window().is_some()
    });
    let coarse = viewer.resident_window().unwrap();
    assert!(
        coarse.spans_whole_frame(W, H) && coarse.stride > 1,
        "{coarse:?}"
    );

    // Then the right half, which fits at full resolution.
    inputs.image.uv_offset = [0.5, 0.0];
    inputs.image.uv_scale = [0.5, 1.0];
    inputs.image.viewport = crate::PhysicalPx::new((W / 2) as f32, H as f32);
    let (offset, scale) = (inputs.image.uv_offset, inputs.image.uv_scale);
    let mut held = Vec::new();
    sync_until(&mut viewer, &mut r, &inputs, |v, calls| {
        let resident = v.resident_window().expect("a window is resident");
        held.push(resident);
        let p = params(calls);
        assert_eq!(
            (p.uv_offset, p.uv_scale),
            resident.map_uv(W, H, offset, scale),
            "sync {} of the zoom: the view on {resident:?}",
            held.len()
        );
        resident.stride == 1
    });
    let fine = *held.last().unwrap();
    assert!(
        !fine.spans_whole_frame(W, H) && fine.x >= W / 2 - 1,
        "{fine:?}"
    );
    // Mapped onto the coarse window, the view would have been another.
    assert_ne!(
        coarse.map_uv(W, H, offset, scale),
        fine.map_uv(W, H, offset, scale)
    );
}

/// In the 3D view: no planes, the volume built to half the free memory the
/// renderer reports (the default budget when it cannot say), uploaded once
/// as the device stores it, and the ray-march set every frame, at the
/// reduction the frontend asks for. The free memory is asked for once, for
/// the build: it can cost the driver a system call or two, so not on each of
/// the frames the build takes, nor after.
#[test]
fn the_volume_is_built_to_half_the_free_memory_and_set_every_frame() {
    for (free, budget) in [
        (Some(1u64 << 30), 512 * MB),
        (None, crate::volume::DEFAULT_VOLUME_BUDGET),
        (Some(10 << 20), crate::volume::MIN_VOLUME_BUDGET),
    ] {
        let mut viewer = opened();
        viewer.view_mode = ViewMode::Volume;
        let mut r = Recorder {
            free,
            ..Recorder::default()
        };
        let mut inputs = FrameInputs::default();
        inputs.volume.downscale = 2;
        let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
        assert_eq!(viewer.residency.volume.planned_budget, Some(budget));
        assert!(uploads(&calls).is_empty(), "no 2D planes in 3D");
        assert_eq!(
            calls
                .iter()
                .filter(|c| matches!(c, Call::ReleaseVolume))
                .count(),
            1
        );
        let Some(Call::UploadVolume {
            dims,
            channels,
            format,
            ..
        }) = calls
            .iter()
            .find(|c| matches!(c, Call::UploadVolume { .. }))
        else {
            unreachable!()
        };
        // Three frames deep, two channels, 16-bit as this device stores it.
        assert_eq!(
            (*dims, *channels, *format),
            ((W, H, 3), 2, TexelFormat::Unorm16)
        );
        assert!(viewer.volume_built());

        // Every frame after: the ray-march's settings, and no upload.
        let outcome = viewer.sync(&mut r, &inputs);
        let calls = r.take();
        assert!(!outcome.needs_repaint);
        assert_eq!(
            kinds(&calls),
            ["size", "volume interp", "volume target", "volume params"]
        );
        let [vw, vh] = inputs.volume.viewport.whole();
        assert!(matches!(calls[1], Call::VolumeInterp(VolumeInterp::Linear)));
        assert!(matches!(
            calls[2],
            Call::VolumeTarget { width, height, downscale: 2 } if (width, height) == (vw, vh)
        ));
        let Call::VolumeParams(p) = &calls[3] else {
            unreachable!()
        };
        assert_eq!(p.num_channels, 2);
        assert_eq!(
            p.max_steps,
            inputs.volume.max_steps(2, VolumeInterp::Linear)
        );
        assert_eq!(r.memory_asks.get(), 1, "the free memory, asked once");
        assert_eq!(p.enabled[..2], [1.0, 1.0]);
        // The windows in the volume's units: a 16-bit channel's divided by
        // 65535, as its unorm texels are.
        let s = &viewer.document().unwrap().display.settings()[0];
        for (got, want) in p.windows[..2].iter().zip([s.window.lo, s.window.hi]) {
            assert!((got - want / 65535.0).abs() < 1e-6, "{got} for {want}");
        }
    }
}

/// How the device stores a volume is the renderer's to say: a device that
/// filters no 16-bit unorm (wgpu without the feature) gets the same stack
/// as half floats.
#[test]
fn the_volume_is_packed_as_the_device_stores_it() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder {
        caps: VolumeCaps {
            max_3d_size: 2048,
            unorm16: false,
            float32: false,
        },
        ..Recorder::default()
    };
    let calls = sync_until(
        &mut viewer,
        &mut r,
        &FrameInputs::default(),
        uploaded_a_volume,
    );
    assert_eq!(volume_uploads(&calls)[0].1, TexelFormat::Float16);
}

/// A volume the GPU refuses after its upload returned (WebGPU says so a
/// frame or more later) is rebuilt at no more than half its size: the
/// budget of the next build is capped there, and what is shown is said to
/// be reduced, also once the smaller one is up.
#[test]
fn a_volume_refused_late_is_rebuilt_at_half_its_size() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder {
        free: Some(4 << 30),
        ..Recorder::default()
    };
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    let identity = viewer.volume_identity();
    let generation = viewer.residency.volume.generation;
    assert_eq!(
        viewer.residency.volume.planned_budget,
        Some(crate::volume::MAX_VOLUME_BUDGET.min(2 << 30))
    );

    r.late = Some((VolumeUploadError::OutOfMemory("no room".into()), 512 * MB));
    let outcome = viewer.sync(&mut r, &inputs);
    assert!(outcome.needs_repaint, "a rebuild needs frames of its own");
    let res = &viewer.residency.volume;
    assert_ne!(res.generation, generation);
    assert_eq!(res.budget_cap, Some(256 * MB));
    assert_eq!(
        viewer.volume_identity(),
        identity,
        "the same volume, smaller"
    );
    assert_eq!(
        viewer.notices.get(NoticeKind::Volume),
        Some(&Notice::Volume(VolumeNote::Reduced))
    );

    let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    assert_eq!(viewer.residency.volume.planned_budget, Some(256 * MB));
    assert_eq!(r.memory_asks.get(), 2, "asked again for the rebuild, once");
    assert_eq!(volume_uploads(&calls).len(), 1);
    assert!(viewer.volume_built());
    assert_eq!(
        viewer.notices.get(NoticeKind::Volume),
        Some(&Notice::Volume(VolumeNote::Reduced)),
        "still a reduced copy"
    );
}

/// A late refusal of a volume no longer wanted — the residency moved on to
/// another build since — is the old volume's business, and changes nothing.
#[test]
fn a_late_refusal_of_a_volume_no_longer_wanted_is_ignored() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder::default();
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    viewer.residency.volume.rebuild();
    let generation = viewer.residency.volume.generation;
    r.late = Some((VolumeUploadError::OutOfMemory("no room".into()), 512 * MB));
    viewer.sync(&mut r, &inputs);
    assert_eq!(viewer.residency.volume.generation, generation);
    assert_eq!(viewer.residency.volume.budget_cap, None);
    assert_eq!(viewer.notices.get(NoticeKind::Volume), None);
}

/// A finished one-channel 8-bit volume of `dims` voxels, all zero, as
/// `PackedVolume::finish` would leave it: every level and brick maximum the
/// size its layout says, which is what an upload checks. Allocated zeroed
/// and never written, so even a large one costs no time and touches no
/// memory.
fn zeroed(dims: (u32, u32, u32)) -> PackedVolume {
    let layout = VolumeLayout::new(dims, 1, TexelFormat::Unorm8);
    let mut volume = PackedVolume::new(layout.clone());
    volume.levels = (0..layout.textures.len())
        .map(|t| {
            (0..layout.levels)
                .map(|level| vec![0; layout.level_bytes(t, level)])
                .collect()
        })
        .collect();
    let (bx, by, bz) = volume.bricks.dims;
    volume.bricks.max = vec![vec![0.0; (bx * by * bz) as usize]; layout.channels];
    volume.check().expect("as finish() leaves it");
    volume
}

/// An upload refused for want of memory: rebuilt at no more than half the
/// size that failed, unless that is below the smallest budget there is,
/// which is said instead, with nothing rebuilt. Another failure is said as
/// it is. A volume that fits clears what was said of one that did not.
#[test]
fn an_upload_out_of_memory_is_retried_at_half_or_given_up() {
    let oom = || Err(VolumeUploadError::OutOfMemory("no room".into()));
    // Big enough that half of it is a budget: 1024 x 1024 x 64 bytes and its
    // mip chain.
    let big = zeroed((1024, 1024, 64));
    let small = zeroed((16, 16, 16));
    let big_bytes = big.layout.total_bytes();
    assert!(big_bytes / 2 >= crate::volume::MIN_VOLUME_BUDGET);

    let mut r = Recorder::default();
    let mut res = VolumeResidency::default();
    res.built_frame = Some(0);
    res.planned_budget = Some(1 << 30);
    let generation = res.generation;
    let mut notices = Notices::default();
    r.upload_results.push_back(oom());
    upload_volume(&mut r, &big, &mut res, &mut notices);
    assert_ne!(res.generation, generation, "rebuilt");
    assert_eq!(
        (res.budget_cap, res.built_frame, res.planned_budget),
        (Some(big_bytes / 2), None, None)
    );
    assert_eq!(
        notices.get(NoticeKind::Volume),
        Some(&Notice::Volume(VolumeNote::Reduced))
    );
    assert_eq!(res.uploaded_generation, None);

    let mut res = VolumeResidency::default();
    let generation = res.generation;
    let mut notices = Notices::default();
    r.upload_results.push_back(oom());
    upload_volume(&mut r, &small, &mut res, &mut notices);
    assert_eq!(res.generation, generation, "not rebuilt");
    assert_eq!(res.budget_cap, None);
    assert!(matches!(
        notices.get(NoticeKind::Volume),
        Some(Notice::Volume(VolumeNote::NoRoom(_)))
    ));

    r.upload_results
        .push_back(Err(VolumeUploadError::Unsupported("too odd".into())));
    upload_volume(&mut r, &small, &mut res, &mut notices);
    assert_eq!(
        notices.get(NoticeKind::Volume),
        Some(&Notice::Volume(VolumeNote::Failed("too odd".into())))
    );

    upload_volume(&mut r, &small, &mut res, &mut notices);
    assert_eq!(res.uploaded_generation, Some(res.generation));
    assert_eq!(notices.get(NoticeKind::Volume), None);
}

/// A volume the GPU had no room for even at the smallest budget is marked
/// built anyway, so it is not retried on every frame, and said so.
#[test]
fn a_volume_that_never_fits_is_not_retried_every_frame() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder::default();
    r.upload_results
        .push_back(Err(VolumeUploadError::OutOfMemory("no room".into())));
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    assert!(matches!(
        viewer.notices.get(NoticeKind::Volume),
        Some(Notice::Volume(VolumeNote::NoRoom(_)))
    ));
    for _ in 0..3 {
        viewer.sync(&mut r, &inputs);
        assert!(volume_uploads(&r.take()).is_empty());
    }
}

/// While the renderer has not heard whether there was room for the volume,
/// the frontend is asked for frame after frame: nothing else would ask for
/// the one that draws it.
#[test]
fn a_volume_waiting_to_hear_keeps_the_frames_coming() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder::default();
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    assert!(!viewer.sync(&mut r, &inputs).needs_repaint);
    r.pending = true;
    assert!(viewer.sync(&mut r, &inputs).needs_repaint);
    r.pending = false;
    assert!(!viewer.sync(&mut r, &inputs).needs_repaint);
}

/// Reassigning the axes makes the volume another one: the renderer gives the
/// old one's memory back before anything else, and a new one is built.
#[test]
fn an_override_gives_the_volume_back_first() {
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder::default();
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    viewer.set_dimension_order(3, 1, 2);
    let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    assert!(matches!(calls[0], Call::ReleaseVolume));
    let (_, format) = volume_uploads(&calls)[0];
    assert_eq!(format, TexelFormat::Unorm16);
    let Some(Call::UploadVolume { dims, channels, .. }) = calls
        .iter()
        .find(|c| matches!(c, Call::UploadVolume { .. }))
    else {
        unreachable!()
    };
    assert_eq!((*dims, *channels), ((W, H, 2), 3));
}

/// With no document there is nothing to send.
#[test]
fn with_no_document_nothing_is_sent() {
    let mut r = Recorder::default();
    let outcome = Viewer::new().sync(&mut r, &FrameInputs::default());
    assert!(r.take().is_empty());
    assert!(!outcome.needs_repaint);
}

/// The budget, worked out from the free memory the renderer reports and
/// nothing else of it: half of it, within the smallest and largest budgets
/// there are, the default when the renderer cannot say, and never more than
/// the cap an upload out of memory left.
#[test]
fn the_budget_is_half_the_free_memory_within_bounds_and_the_cap() {
    use crate::volume::{DEFAULT_VOLUME_BUDGET, MAX_VOLUME_BUDGET, MIN_VOLUME_BUDGET};
    let gb = 1u64 << 30;
    assert_eq!(volume_budget(None, None), DEFAULT_VOLUME_BUDGET);
    assert_eq!(volume_budget(Some(gb), None), 512 * MB);
    assert_eq!(volume_budget(Some(3 * gb / 2), None), 768 * MB);
    assert_eq!(volume_budget(Some(10 << 20), None), MIN_VOLUME_BUDGET);
    assert_eq!(volume_budget(Some(0), None), MIN_VOLUME_BUDGET);
    assert_eq!(volume_budget(Some(64 * gb), None), MAX_VOLUME_BUDGET);
    assert_eq!(volume_budget(Some(gb), Some(100 * MB)), 100 * MB);
    assert_eq!(volume_budget(None, Some(100 * MB)), 100 * MB);
    // A cap is a ceiling, not a floor, and below the smallest budget too:
    // the retry says it is giving up before it gets there.
    assert_eq!(volume_budget(Some(gb), Some(gb as usize)), 512 * MB);
    assert_eq!(volume_budget(None, Some(MB)), MB);
}

/// The plan takes from the renderer only how it stores a volume and its 3D
/// limit; the rest is the document's, the view's and the screen's.
#[test]
fn the_plan_takes_the_format_and_the_limit_from_the_caps() {
    let viewer = opened();
    let doc = viewer.document().unwrap();
    let view = crate::VolumeSettings::default();
    let inputs = crate::VolumeInputs::default();
    let glow = VolumeCaps {
        max_3d_size: 2048,
        unorm16: true,
        float32: true,
    };
    let plan = plan_volume(doc, glow, 1, 64 * MB, &view, &inputs);
    assert_eq!(
        (plan.format, plan.max_dim, plan.budget, plan.time),
        (TexelFormat::Unorm16, 2048, 64 * MB, 1)
    );
    assert_eq!(
        (plan.channels, plan.slices, plan.frames, plan.rgb),
        (2, 1, 3, false)
    );
    let small = VolumeCaps {
        max_3d_size: 256,
        unorm16: false,
        float32: false,
    };
    let plan = plan_volume(doc, small, 0, 64 * MB, &view, &inputs);
    assert_eq!((plan.format, plan.max_dim), (TexelFormat::Float16, 256));
}

/// Each kind of sample goes to its own texture, in that texture's units: an
/// 8-bit stack raw into an R8Uint, a signed 16-bit one moved onto the
/// unsigned scale of the R16Uint the decoder fills (`+ 32768`), and a float
/// one raw into an R32F, which the shader is told to sample instead of the
/// integer texture. The window, held in the file's units, reaches the shader
/// in the texture's. (The recorder holds every plane to the kind and size its
/// channel was given.)
#[test]
fn each_kind_of_sample_reaches_its_own_texture_in_its_units() {
    use ChannelKind::{Float, Int16, Int8};
    // The type, its texture, what the texture holds of a file value (the
    // value plus this), the shader's float flag, and a window in file units.
    for (ty, kind, offset, is_float, window) in [
        (SampleType::U8, Int8, 0.0, 0.0, Window::new(10.0, 200.0)),
        (
            SampleType::I16,
            Int16,
            32768.0,
            0.0,
            Window::new(-1500.0, 2500.0),
        ),
        (SampleType::F32, Float, 0.0, 1.0, Window::new(-0.5, 4000.25)),
    ] {
        let what = format!("{ty:?}");
        let texture = |p| value(ty, p) + offset as f64;
        let mut viewer = opened_as(ty, 2, 1, 3);
        let mut r = Recorder::default();
        viewer.sync(&mut r, &FrameInputs::default());
        let calls = r.take();
        let sized: Vec<&[ChannelKind]> = calls
            .iter()
            .filter_map(|c| match c {
                Call::EnsureSize { kinds, .. } => Some(kinds.as_slice()),
                _ => None,
            })
            .collect();
        assert_eq!(sized, [[kind, kind].as_slice()], "{what}");
        assert_eq!(
            uploads(&calls),
            [(0, texture(0)), (1, texture(1))],
            "{what}"
        );
        let p = params(&calls);
        assert_eq!(
            [p.channels[0].is_float, p.channels[1].is_float],
            [is_float; 2],
            "{what}"
        );

        let mut doc = viewer.document_mut().unwrap();
        assert!(doc.display_mut().set_window(0, window), "{what}");
        doc.set_frame(1);
        viewer.sync(&mut r, &FrameInputs::default());
        let calls = r.take();
        assert_eq!(
            uploads(&calls),
            [(0, texture(2)), (1, texture(3))],
            "{what}"
        );
        let p = params(&calls).channels[0];
        assert_eq!(
            (p.min_max, p.is_float),
            ([window.lo + offset, window.hi + offset], is_float),
            "{what}"
        );
    }
}

/// In the 3D view each kind is stored, flagged and windowed as the volume
/// builder normalizes it: an integer channel as a unorm of its texture's
/// scale (an 8-bit one over 255, a signed 16-bit one moved onto the
/// unsigned scale and over 65535), a float one over the span of its first
/// plane with contrast, which the shader is told to read as a float. Each
/// window below is 0.2 to 0.8 of the voxels' scale.
#[test]
fn each_kind_of_volume_is_stored_flagged_and_windowed_as_it_is_normalized() {
    let float_lo = value(SampleType::F32, 0);
    for (ty, format, is_float, window, corner) in [
        (
            SampleType::U8,
            TexelFormat::Unorm8,
            0.0,
            Window::new(51.0, 204.0),
            [0, 1].map(|p| value(SampleType::U8, p) / 255.0),
        ),
        (
            SampleType::I16,
            TexelFormat::Unorm16,
            0.0,
            Window::new(-19661.0, 19660.0),
            [0, 1].map(|p| (value(SampleType::I16, p) + 32768.0) / 65535.0),
        ),
        (
            // Channel 0's span is plane 0's value to that plus FLOAT_RISE.
            SampleType::F32,
            TexelFormat::Float32,
            1.0,
            Window::new(
                (float_lo + 0.2 * FLOAT_RISE) as f32,
                (float_lo + 0.8 * FLOAT_RISE) as f32,
            ),
            // Each channel's first plane is the bottom of its span.
            [0.0, 0.0],
        ),
    ] {
        let what = format!("{ty:?}");
        let mut viewer = opened_as(ty, 2, 1, 3);
        viewer.view_mode = ViewMode::Volume;
        let mut doc = viewer.document_mut().unwrap();
        assert!(doc.display_mut().set_window(0, window), "{what}");
        let mut r = Recorder::default();
        let inputs = FrameInputs::default();
        let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
        let (dims, channels, stored, origin) = the_volume(&calls);
        assert_eq!((dims, channels, stored), ((W, H, 3), 2, format), "{what}");
        assert_near(&origin, &corner.map(|v| v as f32), &what);

        viewer.sync(&mut r, &inputs);
        let p = volume_params(&r.take());
        assert_eq!(p.is_float[..2], [is_float; 2], "{what}");
        assert_near(&p.windows[..2], &[0.2, 0.8], &what);
    }
}

/// A stack through time in 3D: the volume is the slices of the timepoint on
/// screen, and a step to another timepoint rebuilds it from that one's
/// planes, once, in place of the last (nothing is given back first, as it is
/// for another volume: the textures are refilled).
#[test]
fn a_volume_through_time_is_rebuilt_from_each_timepoint() {
    // Two channels, three slices, two timepoints: the planes of timepoint
    // `t` start at `t * 6`.
    let mut viewer = opened_as(SampleType::U16, 2, 3, 2);
    viewer.view_mode = ViewMode::Volume;
    let mut r = Recorder::default();
    let inputs = FrameInputs::default();
    let unorm = |p| (sample(p) / 65535.0) as f32;
    let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    let (dims, channels, _, origin) = the_volume(&calls);
    assert_eq!((dims, channels), ((W, H, 3), 2), "the slices are the depth");
    assert_near(&origin, &[unorm(0), unorm(1)], "timepoint 0");

    viewer.document_mut().unwrap().set_frame(1);
    let calls = sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    assert!(!calls.iter().any(|c| matches!(c, Call::ReleaseVolume)));
    let (dims, _, _, origin) = the_volume(&calls);
    assert_eq!(dims, (W, H, 3));
    assert_near(&origin, &[unorm(6), unorm(7)], "timepoint 1");
    for _ in 0..3 {
        viewer.sync(&mut r, &inputs);
        assert!(volume_uploads(&r.take()).is_empty(), "built once");
    }
}

/// What the ray-march is given that the sync works out from the display and
/// the view, every frame: each channel's isosurface colour, from the
/// brightest point of its LUT rather than its top entry (a stretched palette
/// ends black there), and the gain the mode takes: the MIP's own, the alpha
/// blend's own or its autolevel, and none for the surface, which shades.
#[test]
fn the_ray_march_takes_its_colours_from_the_luts_and_its_gain_from_the_mode() {
    use scivis_render::VolumeRender::{Alpha, Mip, Surface};
    let mut viewer = opened();
    viewer.view_mode = ViewMode::Volume;
    // Channel 0 brightest halfway along, channel 1 a gray ramp.
    let peaked: scivis_render::Lut = std::array::from_fn(|i| {
        let from_middle = (i as i32 - 128).unsigned_abs();
        [255u32.saturating_sub(2 * from_middle) as u8; 3]
    });
    let ramp: scivis_render::Lut = std::array::from_fn(|i| [i as u8; 3]);
    let doc = viewer.document.as_mut().unwrap();
    doc.display.set_luts(vec![peaked, ramp]);
    let mut r = Recorder::default();
    let inputs = FrameInputs::default();
    sync_until(&mut viewer, &mut r, &inputs, uploaded_a_volume);
    let autolevel = crate::brightness::autolevel(viewer.volume.density);
    for (render, auto, gain) in [
        (Mip, false, 2.5),
        (Alpha, false, 1.75),
        (Alpha, true, autolevel),
        (Surface, false, 1.0),
    ] {
        viewer.volume.render = render;
        viewer.volume.mip_brightness = 2.5;
        viewer.volume.volume_brightness = 1.75;
        viewer.volume.volume_autolevel = auto;
        viewer.sync(&mut r, &inputs);
        let p = volume_params(&r.take());
        assert_eq!(p.brightness, gain, "{render:?}, autolevel {auto}");
        assert_eq!(p.albedo_t[..2], [128.0 / 255.0, 1.0], "{render:?}");
    }
}

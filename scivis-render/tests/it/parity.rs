//! The two backends' pictures of one scene, compared pixel by pixel.
//!
//! Every shader is written twice, in WGSL for wgpu and in GLSL for glow, and
//! the other tests check each copy on its own: that it compiles, that its
//! uniforms are laid out as the Rust side fills them, that its per-channel
//! code covers every channel. None of that sees the two copies come to mean
//! different things — a window applied differently, a LUT row read from the
//! wrong place, a mode code swapped — which is a picture the glow build (the
//! Linux release) draws differently from the wgpu one. This draws a fixed
//! scene through both, off screen, and compares the pictures.
//!
//! The scene (`scene.rs`): a 2D image of `MAX_CHANNELS` channels of every
//! kind (8-bit, 16-bit, float with negative values), each through a LUT of
//! its own and a window of its own, seen magnified, minified (the box
//! filter) and panned, each view with another channel switched off; and a
//! small volume of `MAX_CHANNELS` channels, so that both packed textures are
//! read, in each render mode and each sampling mode, and with each packed
//! texture on its own. Every channel, because each copy writes its code out
//! once per channel, and a channel left out of the scene is a line of each
//! copy that nothing compares. Needs both a GL 3.1 context and a hardware
//! wgpu adapter (see `headless.rs`); skipped, with a message, without either.
//!
//! The tolerances (per 8-bit colour component, at `IMAGE`, `VOLUME` and
//! `VOLUME_NEAREST` below) are a few times what was measured on a Quadro
//! P620 (Windows, GL 3.3 core against Vulkan), the same over three runs: the
//! 2D pictures identical, the volumes at most 3 apart and 0.007 on average,
//! but for nearest sampling, which the two do differently by design. They
//! are what separates rounding from a meaning that drifted. Measured on this
//! scene, each of these in one copy fails here: channel 5 of the composite
//! windowed with channel 4's window (24.8 apart on average); channel 4 of
//! the volume through channel 3's LUT (2.7), or channel 5 read from channel
//! 4's component (2.0); the alpha density 5% higher (0.067); a window 2%
//! wider (0.21, with 1.25% of components more than 3 apart); the
//! composite's LUT row read a twentieth of a row off (2.7); and a MIP gain
//! 3% higher in the WGSL copy (0.48). A 3% change to one weight of the cubic
//! kernel fails too, but on a single pixel at a surface's edge (12 apart,
//! the mean 0.006): luck, not a margin, so take a drift in the cubic kernel
//! as one this does not see.
//!
//! The ray-march jitters each ray's start by a hash of its pixel's position.
//! GL counted that from the bottom of the picture and wgpu from the top, so
//! the same pixel was sampled at other points along its ray in the two, and
//! the volumes were sampling noise apart (on the two-channel scene of the
//! time, a nearest-sampled MIP 0.2 on average, 0.9% of components more than
//! 8 apart, and a surface pixel 82 apart). Both count from the top
//! now, which is what lets the tolerances above be tight. The scene keeps
//! clear of the box's faces, where the shaders still differ by design: wgpu
//! fades the edge voxel in the shader, GL lets a zero border do it.

#![cfg(not(target_arch = "wasm32"))]

use crate::headless::{self, Gl, Wgpu};
use crate::scene::{self, show_image, show_volume, Channel, View, ALL, VIEWS};
use glow::HasContext as _;
use scivis_render::pack::{TexelFormat, VolumeLayout};
use scivis_render::{
    glow_backend, wgpu_backend, VolumeImage, VolumeInterp, VolumeRender, VolumeRenderer as _,
    MAX_CHANNELS,
};

/// The format both backends render into here.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// An 8-bit RGBA picture, rows top to bottom.
struct Picture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl From<VolumeImage> for Picture {
    fn from(image: VolumeImage) -> Picture {
        Picture {
            width: image.width,
            height: image.height,
            rgba: image.rgba,
        }
    }
}

/// How far apart two pictures may be, per colour component: no more than
/// `max` anywhere, more than `near` in at most a share `outliers` of them,
/// and `mean` on average.
struct Tolerance {
    max: u8,
    near: u8,
    outliers: f64,
    mean: f64,
}

/// The 2D composite: measured identical; 1 apart is the rounding of a LUT's
/// linear filter, which GPUs do in fixed point, so two drivers may differ.
const IMAGE: Tolerance = Tolerance {
    max: 1,
    near: 1,
    outliers: 0.0,
    mean: 0.05,
};

/// The volume, linear or cubic: measured at most 3 apart and 0.007 on
/// average, none more than 3 apart, which is float rounding along a ray of a
/// hundred-odd samples. The largest differences are where a ray crosses a
/// bump's edge, so they move with the scene: with smaller bumps, the far MIP
/// (filtering a coarser level, where the hardware's fixed-point weights are
/// coarser in voxels too) was 6 apart and 0.17% more than 3 apart.
const VOLUME: Tolerance = Tolerance {
    max: 10,
    near: 3,
    outliers: 0.003,
    mean: 0.05,
};

/// The volume sampled nearest, which the backends do differently by design:
/// wgpu, with one linear sampler for every mode, snaps the coordinate to the
/// texel centre in the shader; GL has its sampler pick the texel, in fixed
/// point. A sample within a fraction of a texel of a texel's edge can land
/// on either side, and the pixel takes the voxel next door: measured 0.27% of
/// components more than 8 apart, and 0.059 on average.
const VOLUME_NEAREST: Tolerance = Tolerance {
    max: 255,
    near: 8,
    outliers: 0.01,
    mean: 0.25,
};

/// `glow` matches `wgpu` within `tol`, and neither is all but blank (two
/// blank pictures would match). On a mismatch both are written out as PPM
/// files to look at.
fn assert_close(what: &str, wgpu: &Picture, glow: &Picture, tol: &Tolerance) {
    assert_eq!(
        (wgpu.width, wgpu.height),
        (glow.width, glow.height),
        "{what}: sizes"
    );
    let mut worst = (0u8, 0usize);
    let mut far = 0usize;
    let mut sum = 0u64;
    let mut components = 0usize;
    for (i, (a, b)) in wgpu.rgba.iter().zip(&glow.rgba).enumerate() {
        if i % 4 == 3 {
            continue; // alpha: 255 in both
        }
        let d = a.abs_diff(*b);
        components += 1;
        sum += d as u64;
        far += (d > tol.near) as usize;
        if d > worst.0 {
            worst = (d, i / 4);
        }
    }
    let mean = sum as f64 / components as f64;
    let share = far as f64 / components as f64;
    let (lit_wgpu, lit_glow) = (scene::lit(&wgpu.rgba), scene::lit(&glow.rgba));
    let report = format!(
        "{what}: largest difference {} at ({}, {}), mean {mean:.3}, {:.2}% more than {} apart; \
         {:.0}% / {:.0}% of pixels lit (wgpu / glow)",
        worst.0,
        worst.1 % wgpu.width as usize,
        worst.1 / wgpu.width as usize,
        share * 100.0,
        tol.near,
        lit_wgpu * 100.0,
        lit_glow * 100.0,
    );
    eprintln!("{report}");
    let ok = worst.0 <= tol.max
        && share <= tol.outliers
        && mean <= tol.mean
        && lit_wgpu > 0.02
        && lit_glow > 0.02;
    if !ok {
        let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
        let name = what.replace(|c: char| !c.is_ascii_alphanumeric(), "-");
        for (backend, p) in [("wgpu", wgpu), ("glow", glow)] {
            let mut ppm = format!("P6 {} {} 255\n", p.width, p.height).into_bytes();
            ppm.extend(p.rgba.chunks(4).flat_map(|px| px[..3].to_vec()));
            let _ = std::fs::write(dir.join(format!("parity-{name}-{backend}.ppm")), ppm);
        }
        panic!(
            "{report}\nthe backends draw this differently (both pictures are in {})",
            dir.display()
        );
    }
}

/// The wgpu device and a GL context, or `None` (having said why) without
/// either. wgpu first: see `headless::wgpu`.
fn both() -> Option<(&'static Wgpu, Gl)> {
    let gpu = headless::wgpu()?;
    let gl = headless::gl()?;
    Some((gpu, gl))
}

// --- 2D composite --------------------------------------------------------

fn wgpu_image(gpu: &Wgpu, channels: &[Channel], view: &View) -> Picture {
    let mut r =
        wgpu_backend::ImageRenderResources::new(gpu.device.clone(), gpu.queue.clone(), FORMAT);
    show_image(&mut r, channels, view);
    let (width, height) = view.size;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("parity target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let attachment = target.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("parity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        r.paint(&mut pass);
    }
    let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity readback"),
        size: (row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        target.size(),
    );
    gpu.queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.expect("readback"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finished");
    let rgba = buffer
        .slice(..)
        .get_mapped_range()
        .chunks(row as usize)
        .flat_map(|r| r[..width as usize * 4].to_vec())
        .collect();
    Picture {
        width,
        height,
        rgba,
    }
}

fn glow_image(gl: &Gl, channels: &[Channel], view: &View) -> Picture {
    let mut r = glow_backend::ImageRenderResources::new(gl.gl.clone());
    show_image(&mut r, channels, view);
    let rgba = headless::draw_offscreen(&gl.gl, view.size, || r.paint(&gl.gl));
    headless::assert_no_error(&gl.gl, view.name);
    Picture {
        width: view.size.0,
        height: view.size.1,
        rgba,
    }
}

#[test]
fn the_backends_composite_an_image_alike() {
    let _gpu = headless::exclusive();
    let Some((gpu, gl)) = both() else { return };
    let channels = scene::image();
    for view in &VIEWS {
        let wgpu = wgpu_image(gpu, &channels, view);
        let glow = glow_image(&gl, &channels, view);
        let what = format!("image, {}, channel {} off", view.name, view.off);
        assert_close(&what, &wgpu, &glow, &IMAGE);
    }
}

// --- 3D volume -----------------------------------------------------------

/// Where the camera is, and the size of the picture it takes.
#[derive(Clone, Copy, Debug)]
struct View3d {
    distance: f32,
    size: (u32, u32),
}

/// At `NEAR` a voxel is about a pixel, and a MIP reads level 0; at `FAR` it
/// is about a third of one, and level 1 everywhere (`floor(log2)` of 2.75 to
/// 3.45 pixel footprints a voxel, clear of the boundaries at 2 and 4, where a
/// pixel could read either level).
const NEAR: View3d = View3d {
    distance: 1.5,
    size: (64, 48),
};

const FAR: View3d = View3d {
    distance: 2.2,
    size: (32, 24),
};

/// The channels the volume packer puts in packed texture `t` (0 or 1), on.
fn in_texture(t: usize) -> [bool; MAX_CHANNELS] {
    let layout = VolumeLayout::new((1, 1, 1), MAX_CHANNELS, TexelFormat::Unorm8);
    std::array::from_fn(|c| layout.place(c).0 == t)
}

/// Each render mode with each sampling mode, every channel on; the far view,
/// in which a MIP reads a coarser mip level, with each sampling mode, as
/// each sizes its grid by the level it reads (cubic's taps, or nearest's
/// snap to a texel centre, sized by level 0 in one copy pass at `NEAR`, and
/// fail here 0.135 and 0.958 apart on average); and the two packed textures
/// on their own, where the shaders skip the read of the other (`need_a`,
/// `need_b`).
fn renders() -> Vec<(VolumeRender, VolumeInterp, View3d, [bool; MAX_CHANNELS])> {
    use VolumeInterp::{Cubic, Linear, Nearest};
    use VolumeRender::{Alpha, Mip, Surface};
    vec![
        (Mip, Linear, NEAR, ALL),
        (Alpha, Linear, NEAR, ALL),
        (Surface, Linear, NEAR, ALL),
        (Mip, Cubic, NEAR, ALL),
        (Alpha, Cubic, NEAR, ALL),
        (Surface, Cubic, NEAR, ALL),
        (Mip, Linear, FAR, ALL),
        (Mip, Cubic, FAR, ALL),
        (Mip, Nearest, FAR, ALL),
        (Mip, Nearest, NEAR, ALL),
        (Alpha, Linear, NEAR, in_texture(0)),
        (Mip, Linear, NEAR, in_texture(1)),
    ]
}

fn wgpu_volume(
    gpu: &Wgpu,
    mode: VolumeRender,
    interp: VolumeInterp,
    view: View3d,
    on: [bool; MAX_CHANNELS],
) -> Picture {
    let mut r =
        wgpu_backend::ImageRenderResources::new(gpu.device.clone(), gpu.queue.clone(), FORMAT);
    show_volume(&mut r, mode, interp, view.distance, on);
    r.capture_volume(view.size.0, view.size.1)
        .expect("a capture starts");
    loop {
        if let Some(image) = r.take_volume_capture() {
            return image.expect("the capture is read back").into();
        }
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the GPU finished");
    }
}

fn glow_volume(
    gl: &Gl,
    mode: VolumeRender,
    interp: VolumeInterp,
    view: View3d,
    on: [bool; MAX_CHANNELS],
) -> Picture {
    let mut r = glow_backend::ImageRenderResources::new(gl.gl.clone());
    show_volume(&mut r, mode, interp, view.distance, on);
    // SAFETY: the context is current on this thread.
    unsafe { gl.gl.disable(glow::DITHER) };
    r.capture_volume(view.size.0, view.size.1)
        .expect("a capture starts");
    let image = r.take_volume_capture().expect("GL renders at once");
    headless::assert_no_error(&gl.gl, "the volume capture");
    image.expect("the capture is read back").into()
}

#[test]
fn the_backends_render_a_volume_alike() {
    let _gpu = headless::exclusive();
    let Some((gpu, gl)) = both() else { return };
    for (mode, interp, view, on) in renders() {
        let wgpu = wgpu_volume(gpu, mode, interp, view, on);
        let glow = glow_volume(&gl, mode, interp, view, on);
        let tolerance = match interp {
            VolumeInterp::Nearest => &VOLUME_NEAREST,
            VolumeInterp::Linear | VolumeInterp::Cubic => &VOLUME,
        };
        let channels: Vec<usize> = (0..MAX_CHANNELS).filter(|&c| on[c]).collect();
        let what = format!(
            "volume, {mode:?}, {interp:?}, {} away, channels {channels:?}",
            view.distance
        );
        assert_close(&what, &wgpu, &glow, tolerance);
    }
}

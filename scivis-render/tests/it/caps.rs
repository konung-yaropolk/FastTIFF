//! What each real backend says it stores a volume as, and that it stores
//! each kind so.
//!
//! The texel policy is [`VolumeCaps`] as a function of the kind, and
//! `src/renderer_tests.rs` pins that function for caps written out by hand.
//! What each backend answers is set in its own `new`, from the device it is
//! given, and is pinned here, on a real device: glow keeps every kind at its
//! own width, wgpu a 16-bit volume as a unorm where the device filters one
//! and a float volume always as a half float. A backend that answered
//! otherwise would still draw (a 16-bit volume as half floats looks much the
//! same), and the other GPU suites upload only 8-bit volumes, so nothing else
//! would see it: glow saying `float32: false` silently halves its float
//! volumes' precision, and wgpu saying `float32: true` asks for a filtered
//! R32Float that a device without `FLOAT32_FILTERABLE` refuses, the first
//! time a float stack is shown in 3D.
//!
//! Then each kind is drawn: the 8-bit scene's volume as 16-bit and as float
//! samples standing for the same values, through the textures each backend
//! makes for them, and held to the 8-bit picture. That is the one place the
//! 16-bit and float volume textures of either backend are drawn with.
//!
//! Skipped, with a message, without a GL context or a hardware wgpu adapter
//! (see `headless.rs`).

#![cfg(not(target_arch = "wasm32"))]

use crate::headless;
use crate::scene::{self, show_volume_as, ALL};
use scivis_render::{
    VolumeCaps, VolumeImage, VolumeInterp, VolumeKind, VolumeRender, VolumeRenderer,
};

const KINDS: [VolumeKind; 3] = [VolumeKind::U8, VolumeKind::U16, VolumeKind::F32];

/// The size of every capture here: small, for a software rasterizer.
const SIZE: (u32, u32) = (48, 36);

/// That `kind`'s picture is the 8-bit one's, but for rounding: no component
/// more than `max` apart, `mean` apart on average, and lit.
fn assert_as_8_bit(what: &str, kind: VolumeKind, eight: &VolumeImage, image: &VolumeImage) {
    assert_eq!((image.width, image.height), (eight.width, eight.height));
    let (mut worst, mut sum) = (0u8, 0u64);
    for (a, b) in eight.rgba.iter().zip(&image.rgba) {
        let d = a.abs_diff(*b);
        worst = worst.max(d);
        sum += d as u64;
    }
    let mean = sum as f64 / eight.rgba.len() as f64;
    let lit = scene::lit(&image.rgba);
    headless::say(format_args!(
        "{what}, {kind:?}: largest difference from 8-bit {worst}, mean {mean:.3}, {:.0}% lit",
        lit * 100.0
    ));
    assert!(lit > 0.02, "{what}, {kind:?}: nothing was drawn");
    assert!(
        worst <= MAX && mean <= MEAN,
        "{what}, {kind:?}: largest difference from the 8-bit picture {worst}, mean {mean:.3}"
    );
}

/// How far a kind's picture may be from the 8-bit one's. The same values,
/// stored finer (16-bit, float) or as half floats, filtered and marched:
/// rounding alone, measured on a Quadro P620 (GL 3.3 core and Vulkan) at most
/// 1 apart and 0.002 on average, in both modes on both backends. Looser than
/// that by far, because the glow half also runs on the Linux CI runner's
/// software rasterizer (Mesa's llvmpipe), where nothing was measured and which
/// may filter an 8-bit texture in fixed point and a float one in float. What
/// this is for is a kind stored or read as another format, which moves the
/// picture by far more than a rounding.
const MAX: u8 = 16;
const MEAN: f64 = 0.25;

/// Every kind's capture, in the order of [`KINDS`], of the scene in `mode`.
fn captures(
    r: &mut impl VolumeRenderer,
    mode: VolumeRender,
    capture: impl Fn(&mut dyn FnMut() -> Option<Result<VolumeImage, String>>) -> VolumeImage,
) -> Vec<VolumeImage> {
    KINDS
        .iter()
        .map(|&kind| {
            show_volume_as(r, kind, mode, VolumeInterp::Linear, 1.5, ALL);
            r.capture_volume(SIZE.0, SIZE.1).expect("a capture starts");
            capture(&mut || r.take_volume_capture())
        })
        .collect()
}

#[cfg(feature = "backend-glow")]
#[test]
fn the_glow_backend_stores_every_kind_at_its_own_width() {
    use glow::HasContext as _;
    use scivis_render::glow_backend::ImageRenderResources;
    let _gpu = headless::exclusive();
    let Some(gl) = headless::gl() else { return };
    let mut r = ImageRenderResources::new(std::sync::Arc::clone(&gl.gl));
    // SAFETY: the context is current on this thread.
    let max_3d = unsafe { gl.gl.get_parameter_i32(glow::MAX_3D_TEXTURE_SIZE) } as u32;
    assert_eq!(
        r.volume_caps(),
        VolumeCaps {
            max_3d_size: max_3d,
            unorm16: true,
            float32: true,
        }
    );
    for mode in [VolumeRender::Mip, VolumeRender::Alpha] {
        let pictures = captures(&mut r, mode, |take| {
            let image = take().expect("GL renders at once");
            headless::assert_no_error(&gl.gl, "a capture");
            image.expect("the capture is read back")
        });
        for (kind, picture) in KINDS.iter().zip(&pictures).skip(1) {
            assert_as_8_bit(&format!("glow, {mode:?}"), *kind, &pictures[0], picture);
        }
    }
}

#[cfg(feature = "backend-wgpu")]
#[test]
fn the_wgpu_backend_stores_float_as_half_and_16_bit_as_the_device_filters() {
    use scivis_render::wgpu_backend::ImageRenderResources;
    let _gpu = headless::exclusive();
    let Some(gpu) = headless::wgpu() else { return };
    let mut r = ImageRenderResources::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        wgpu::TextureFormat::Rgba8Unorm,
    );
    assert_eq!(
        r.volume_caps(),
        VolumeCaps {
            max_3d_size: gpu.device.limits().max_texture_dimension_3d,
            unorm16: gpu
                .device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM),
            float32: false,
        }
    );
    for mode in [VolumeRender::Mip, VolumeRender::Alpha] {
        let pictures = captures(&mut r, mode, |take| loop {
            if let Some(image) = take() {
                return image.expect("the capture is read back");
            }
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("the GPU finished");
        });
        for (kind, picture) in KINDS.iter().zip(&pictures).skip(1) {
            assert_as_8_bit(&format!("wgpu, {mode:?}"), *kind, &pictures[0], picture);
        }
    }
}

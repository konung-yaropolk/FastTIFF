//! What both backends hand their shaders, packed once: each test pins what
//! the two backends' own copies of it did.

use super::*;
use std::mem::{offset_of, size_of};

fn channel(min: f32, max: f32, enabled: bool, is_float: bool) -> ChannelUniform {
    ChannelUniform {
        min,
        max,
        enabled,
        is_float,
    }
}

/// Each channel given goes in its slot, its flags as 1.0 and 0.0; a slot
/// with no channel is off and integer, as the wgpu backend sent it. (The
/// glow backend sent such a slot a window of zero: never applied either way,
/// since it is off and past `num_channels`.)
#[test]
fn the_composite_block_holds_each_channel_given() {
    let block = CompositeParams::new(
        &[
            channel(10.0, 200.0, true, false),
            channel(-1.5, 2.5, false, true),
        ],
        2,
        [0.25, 0.5],
        [0.5, 0.25],
    );
    assert_eq!(
        block.channels[0],
        ChannelParams {
            min_max: [10.0, 200.0],
            enabled: 1.0,
            is_float: 0.0,
        }
    );
    assert_eq!(
        block.channels[1],
        ChannelParams {
            min_max: [-1.5, 2.5],
            enabled: 0.0,
            is_float: 1.0,
        }
    );
    for slot in &block.channels[2..] {
        assert_eq!(*slot, NO_CHANNEL);
    }
    assert_eq!(
        (block.num_channels, block.uv_offset, block.uv_scale),
        (2, [0.25, 0.5], [0.5, 0.25])
    );
    assert_eq!((block._pad0, block._pad1, block._pad2), (0, 0, 0));
}

/// Channels past [`MAX_CHANNELS`] have no slot, and are dropped; the count is
/// passed on as given, for the shader to bound.
#[test]
fn channels_past_the_last_slot_are_dropped() {
    let many: Vec<ChannelUniform> = (0..MAX_CHANNELS + 2)
        .map(|c| channel(c as f32, 100.0 + c as f32, true, false))
        .collect();
    let block = CompositeParams::new(&many, many.len() as u32, [0.0; 2], [1.0; 2]);
    for (c, slot) in block.channels.iter().enumerate() {
        assert_eq!(slot.min_max, [c as f32, 100.0 + c as f32]);
    }
    assert_eq!(block.num_channels as usize, MAX_CHANNELS + 2);
    assert_eq!(CompositeParams::default().num_channels, 0);
}

/// A channel is one `vec4` — window min, max, on, float, in that order —
/// which is how `composite.frag.glsl` declares `ch_params`, and the channels
/// come first, one after another, which is what lets the glow backend set
/// them all from the block in one call. (The WGSL side of the layout is held
/// to naga's in `wgpu_backend_tests.rs`.)
#[test]
fn a_channel_is_the_vec4_the_glsl_reads() {
    assert_eq!(size_of::<ChannelParams>(), 16);
    assert_eq!(offset_of!(ChannelParams, min_max), 0);
    assert_eq!(offset_of!(ChannelParams, enabled), 8);
    assert_eq!(offset_of!(ChannelParams, is_float), 12);
    assert_eq!(offset_of!(CompositeParams, channels), 0);
    assert_eq!(offset_of!(CompositeParams, uv_offset), 16 * MAX_CHANNELS);
    let glsl = include_str!("shaders/composite.frag.glsl");
    assert!(
        glsl.contains("uniform vec4 ch_params[MAX_CHANNELS];"),
        "composite.frag.glsl declares ch_params otherwise"
    );
    // .x and .y the window, .z on, .w float, as `apply_channel` and the
    // per-channel lines read them.
    assert!(glsl.contains("float span = max(p.y - p.x, 1.0);"));
    assert!(glsl.contains("return color * p.z;"));
    assert!(glsl.contains("ch_params[0].w > 0.5 ? sample_flt("));
}

/// The LUT row is each entry's colour, opaque, in order.
#[test]
fn a_lut_row_is_its_entries_opaque() {
    let lut: Lut = std::array::from_fn(|i| [i as u8, (255 - i) as u8, (i / 2) as u8]);
    let rgba = lut_rgba(&lut);
    assert_eq!(rgba.len(), LUT_WIDTH * 4);
    for (i, texel) in rgba.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(*texel, [lut[i][0], lut[i][1], lut[i][2], 255], "entry {i}");
    }
}

/// The acceleration uniform as both backends worked it out: the size of a
/// pixel at unit distance, the coarsest level a MIP may read (none when the
/// mip levels are off, or the height is not known), the brick size, and the
/// skip flag.
#[test]
fn the_acceleration_uniform_is_as_both_backends_had_it() {
    let p = VolumeParams {
        num_channels: 1,
        windows: [0.0; MAX_CHANNELS * 2],
        enabled: [1.0; MAX_CHANNELS],
        is_float: [0.0; MAX_CHANNELS],
        albedo_t: [1.0; MAX_CHANNELS],
        render_mode: crate::VolumeRender::Mip,
        density: 1.0,
        brightness: 1.0,
        iso: 0.5,
        eye: [0.0, 0.0, 2.0],
        forward: [0.0, 0.0, -1.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        tan_half_fov: 0.5,
        aspect: 1.0,
        box_he: [0.5; 3],
        max_steps: 0,
    };
    let brick = crate::pack::BRICK as f32;
    assert_eq!(
        volume_accel(&p, 500, 4, true, true),
        [0.002, 3.0, brick, 1.0]
    );
    assert_eq!(
        volume_accel(&p, 500, 4, false, false),
        [0.002, 0.0, brick, 0.0]
    );
    // A height not known yet: a pixel of the whole view, and no coarser
    // level.
    assert_eq!(volume_accel(&p, 0, 4, true, true), [1.0, 0.0, brick, 1.0]);
    // One level is the finest only.
    assert_eq!(volume_accel(&p, 10, 1, true, false)[1], 0.0);
}

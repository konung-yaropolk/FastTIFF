//! What both backends keep on the CPU side, held to what each backend's own
//! copy of it did.

use super::*;
use crate::pack::{PackedVolume, TexelFormat};
use crate::{VolumeKind, VolumeRender};

/// The textures each kind of channel gets: its data's texture at the image's
/// size, the other a 1x1 dummy, and both dummies for a channel not there,
/// the integer one 16-bit.
#[test]
fn each_kind_gets_its_textures() {
    let size = (640, 480);
    let of = |kind| {
        let t = ChannelTextures::of(kind, size);
        (t.int_8bit, t.int_size, t.float_size)
    };
    assert_eq!(of(Some(ChannelKind::Int8)), (true, size, (1, 1)));
    assert_eq!(of(Some(ChannelKind::Int16)), (false, size, (1, 1)));
    assert_eq!(of(Some(ChannelKind::Float)), (false, (1, 1), size));
    assert_eq!(of(None), (false, (1, 1), (1, 1)));
}

/// The textures are remade only when the size or a channel's kind changed:
/// what a backend is made with (dummies) is no change, the same image again
/// is none, and the kinds past the last channel there is room for are not
/// looked at.
#[test]
fn the_textures_are_remade_only_on_a_change() {
    let mut a = ChannelAllocation::default();
    assert!(!a.update(1, 1, &[]), "as made");
    let kinds = [ChannelKind::Int16, ChannelKind::Float];
    assert!(a.update(64, 32, &kinds), "a new image");
    let made = a.textures();
    assert_eq!(
        made[0],
        ChannelTextures::of(Some(ChannelKind::Int16), (64, 32))
    );
    assert_eq!(
        made[1],
        ChannelTextures::of(Some(ChannelKind::Float), (64, 32))
    );
    assert!(made[2..]
        .iter()
        .all(|t| *t == ChannelTextures::of(None, (64, 32))));
    assert!(!a.update(64, 32, &kinds), "the same again");
    assert!(a.update(64, 33, &kinds), "another size");
    assert!(
        a.update(64, 33, &[ChannelKind::Int8, ChannelKind::Float]),
        "another kind"
    );
    let mut many = [ChannelKind::Int16; MAX_CHANNELS + 1];
    a.update(8, 8, &many);
    many[MAX_CHANNELS] = ChannelKind::Float;
    assert!(!a.update(8, 8, &many), "past the last slot");
}

fn params(window_max: f32) -> VolumeParams {
    let mut windows = [0.0; MAX_CHANNELS * 2];
    windows[1] = window_max;
    VolumeParams {
        num_channels: 1,
        windows,
        enabled: [1.0; MAX_CHANNELS],
        is_float: [0.0; MAX_CHANNELS],
        albedo_t: [1.0; MAX_CHANNELS],
        render_mode: VolumeRender::Mip,
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
    }
}

/// A volume of two by two by one bricks, one 8-bit channel.
fn uploaded() -> VolumeState {
    let dims = (32, 32, 16);
    let bytes = vec![200u8; 32 * 32 * 16];
    let volume =
        PackedVolume::from_channels(dims, &[(VolumeKind::U8, &bytes)], TexelFormat::Unorm8);
    VolumeState {
        layout: Some(volume.layout),
        bricks: Some(volume.bricks),
        ..VolumeState::default()
    }
}

/// The occupancy is worked out when there is a volume and params to see it
/// with, and again only when what can be seen changed.
#[test]
fn the_occupancy_is_stale_until_worked_out_for_the_params() {
    let v = VolumeState {
        params: Some(params(1.0)),
        ..VolumeState::default()
    };
    assert!(v.stale_occupancy().is_none(), "no volume");
    let mut v = uploaded();
    assert!(v.stale_occupancy().is_none(), "no params");
    v.params = Some(params(1.0));
    let occ = v.stale_occupancy().expect("never worked out");
    assert_eq!((occ.bytes.len(), occ.dims), (4, (2, 2, 1)));
    v.occupancy_for = Some(occ.vis);
    assert!(v.stale_occupancy().is_none(), "worked out for these");
    v.params = Some(params(0.5));
    assert!(v.stale_occupancy().is_some(), "another window");
}

/// Empty space is skipped only where the occupancy is the params' own, and
/// only while skipping is on; the coarsest level is the volume's, while mip
/// levels are on.
#[test]
fn the_acceleration_follows_the_volume_and_the_switches() {
    let mut v = uploaded();
    let levels = v.layout.as_ref().unwrap().levels;
    assert!(levels > 1);
    let (p, other) = (params(1.0), params(0.5));
    v.occupancy_for = Some(crate::pack::Visibility::of(&p));
    assert_eq!(v.accel(&p, 100)[3], 1.0);
    assert_eq!(v.accel(&other, 100)[3], 0.0, "a capture's own params");
    assert_eq!(v.accel(&p, 100)[1], (levels - 1) as f32);
    v.skip_empty = false;
    v.mip_levels = false;
    assert_eq!(v.accel(&p, 100)[1..], [0.0, crate::pack::BRICK as f32, 0.0]);
    v.forget();
    assert_eq!((v.resident_bytes(), v.accel(&p, 100)[1]), (0, 0.0));
}

/// A capture starts only with params to render, at a size the device can
/// render a tile of, and with no other capture still to be taken: the checks
/// and the messages both backends had, each written out for itself.
#[test]
fn a_capture_starts_only_when_it_can() {
    let mut v = VolumeState::default();
    assert_eq!(
        v.capture_params(64, 48, 4096, false).err(),
        Some("there is no volume to render".into())
    );
    v.params = Some(params(0.75));
    let beyond = |w: u32, h: u32| {
        Some(format!(
            "a {w} x {h} picture is beyond what this GPU can render (up to 4096 a side)"
        ))
    };
    for (w, h) in [(0, 48), (64, 0), (4097, 48), (64, 4097)] {
        assert_eq!(v.capture_params(w, h, 4096, false).err(), beyond(w, h));
    }
    assert_eq!(
        v.capture_params(64, 48, 4096, true).err(),
        Some("the last capture has not been taken yet".into())
    );
    let got = v
        .capture_params(4096, 4096, 4096, false)
        .expect("at the limit");
    assert_eq!(got.windows[1], 0.75, "the params last set");
}
